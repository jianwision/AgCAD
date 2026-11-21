// Deterministic primitives. Each kernel mirrors agcad_kernels::reference exactly
// (same association order) so results are bit-identical (AG-06-006).
#include "ag_common.cuh"

#define SCAN_TILE 1024
#define SORT_TILE 1024
#define RADIX_BINS 256
#define REDUCE_THREADS 256

struct ScanParams { u64 input, output, block_sums; u32 n, _pad; };
struct ScanAddParams { u64 output, block_offsets; u32 n, _pad; };
struct CompactParams { u64 flags, offsets, src, dst; u32 n, elem_bytes; };
struct RadixParams { u64 keys_in, vals_in, keys_out, vals_out, hist; u32 n, shift, tiles, _pad; };
struct SegReduceParams { u64 values, offsets, out; u32 segments, op; };
struct OrderKeysParams { u64 scores, keys, vals; u32 n, descending; };

// 256 threads x 4 items = one 1024-element tile per block.
AG_KERNEL void ag_scan_blocks(const ScanParams* p) {
    __shared__ u32 totals[256];
    const u32* in = ag_ptr<const u32>(p->input);
    u32* out = ag_ptr<u32>(p->output);
    u32 base = blockIdx.x * SCAN_TILE + threadIdx.x * 4;
    u32 v[4];
    u32 sum = 0;
    for (int k = 0; k < 4; ++k) {
        u32 i = base + k;
        v[k] = (i < p->n) ? in[i] : 0u;
        sum += v[k];
    }
    totals[threadIdx.x] = sum;
    __syncthreads();
    // Hillis-Steele inclusive scan of thread totals (integer: order-independent result).
    for (u32 off = 1; off < 256; off <<= 1) {
        u32 t = (threadIdx.x >= off) ? totals[threadIdx.x - off] : 0u;
        __syncthreads();
        totals[threadIdx.x] += t;
        __syncthreads();
    }
    u32 run = (threadIdx.x == 0) ? 0u : totals[threadIdx.x - 1];
    for (int k = 0; k < 4; ++k) {
        u32 i = base + k;
        if (i < p->n) out[i] = run;
        run += v[k];
    }
    if (threadIdx.x == 255) ag_ptr<u32>(p->block_sums)[blockIdx.x] = totals[255];
}

AG_KERNEL void ag_scan_add(const ScanAddParams* p) {
    u32* out = ag_ptr<u32>(p->output);
    u32 add = ag_ptr<const u32>(p->block_offsets)[blockIdx.x];
    for (int k = 0; k < 4; ++k) {
        u32 i = blockIdx.x * SCAN_TILE + threadIdx.x * 4 + k;
        if (i < p->n) out[i] += add;
    }
}

AG_KERNEL void ag_compact_scatter(const CompactParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    if (ag_ptr<const u32>(p->flags)[i] == 0u) return;
    const unsigned char* src = ag_ptr<const unsigned char>(p->src) + (u64)i * p->elem_bytes;
    unsigned char* dst = ag_ptr<unsigned char>(p->dst) + (u64)ag_ptr<const u32>(p->offsets)[i] * p->elem_bytes;
    for (u32 b = 0; b < p->elem_bytes; ++b) dst[b] = src[b];
}

// One thread per digit; tile staged in dynamic shared memory (keys 8 KB + vals 4 KB).
AG_KERNEL void ag_radix_histogram(const RadixParams* p) {
    extern __shared__ unsigned char smem[];
    u64* keys = reinterpret_cast<u64*>(smem);
    u32 t = blockIdx.x;
    u32 begin = t * SORT_TILE;
    u32 count = min((u32)SORT_TILE, p->n - begin);
    for (u32 i = threadIdx.x; i < count; i += blockDim.x) keys[i] = ag_ptr<const u64>(p->keys_in)[begin + i];
    __syncthreads();
    u32 d = threadIdx.x;
    u32 c = 0;
    for (u32 i = 0; i < count; ++i) c += (u32)((keys[i] >> p->shift) & (RADIX_BINS - 1)) == d;
    ag_ptr<u32>(p->hist)[d * p->tiles + t] = c;
}

AG_KERNEL void ag_radix_scatter(const RadixParams* p) {
    extern __shared__ unsigned char smem[];
    u64* keys = reinterpret_cast<u64*>(smem);
    u32* vals = reinterpret_cast<u32*>(smem + SORT_TILE * 8);
    u32 t = blockIdx.x;
    u32 begin = t * SORT_TILE;
    u32 count = min((u32)SORT_TILE, p->n - begin);
    for (u32 i = threadIdx.x; i < count; i += blockDim.x) {
        keys[i] = ag_ptr<const u64>(p->keys_in)[begin + i];
        vals[i] = ag_ptr<const u32>(p->vals_in)[begin + i];
    }
    __syncthreads();
    u32 d = threadIdx.x;
    u32 dst = ag_ptr<const u32>(p->hist)[d * p->tiles + t];
    u64* ko = ag_ptr<u64>(p->keys_out);
    u32* vo = ag_ptr<u32>(p->vals_out);
    for (u32 i = 0; i < count; ++i) {  // in-order walk: stable
        if ((u32)((keys[i] >> p->shift) & (RADIX_BINS - 1)) == d) {
            ko[dst] = keys[i];
            vo[dst] = vals[i];
            ++dst;
        }
    }
}

__device__ __forceinline__ double ag_combine(u32 op, double a, double b) {
    if (op == 1) return (b < a) ? b : a;
    if (op == 2) return (b > a) ? b : a;
    return a + b;
}
__device__ __forceinline__ double ag_identity(u32 op) {
    if (op == 1) return __longlong_as_double(0x7FF0000000000000LL);
    if (op == 2) return __longlong_as_double(0xFFF0000000000000LL);
    return 0.0;
}

AG_KERNEL void ag_segmented_reduce_f64(const SegReduceParams* p) {
    extern __shared__ unsigned char smem[];
    double* part = reinterpret_cast<double*>(smem);
    u32 seg = blockIdx.x;
    if (seg >= p->segments) return;
    const u32* offs = ag_ptr<const u32>(p->offsets);
    const double* v = ag_ptr<const double>(p->values);
    u32 b = offs[seg], e = offs[seg + 1];
    double acc = ag_identity(p->op);
    for (u32 j = b + threadIdx.x; j < e; j += REDUCE_THREADS) acc = ag_combine(p->op, acc, v[j]);
    part[threadIdx.x] = acc;
    __syncthreads();
    for (u32 stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) part[threadIdx.x] = ag_combine(p->op, part[threadIdx.x], part[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) ag_ptr<double>(p->out)[seg] = part[0];
}

AG_KERNEL void ag_order_keys_f64(const OrderKeysParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    double x = ag_ptr<const double>(p->scores)[i];
    if (x != x) x = __longlong_as_double(0x7FF0000000000000LL);  // NaN sorts last
    if (p->descending) x = -x;
    u64 b = (u64)__double_as_longlong(x);
    ag_ptr<u64>(p->keys)[i] = (b >> 63) ? ~b : (b | (1ull << 63));
    ag_ptr<u32>(p->vals)[i] = i;
}

struct MeshTermsParams { u64 positions; u64 tris; u64 out; u32 n_tris; u32 _pad; };

// Per-triangle mass-property terms; same operation order as the CPU reference (fmad off).
AG_KERNEL void ag_mesh_terms(const MeshTermsParams* p) {
    u32 i = AG_TID;
    u32 n = p->n_tris;
    if (i >= n) return;
    const u32* t = ag_ptr<const u32>(p->tris);
    const double* P = ag_ptr<const double>(p->positions);
    double a[3], b[3], d[3];
    for (int k = 0; k < 3; ++k) { a[k] = P[3 * t[3 * i] + k]; b[k] = P[3 * t[3 * i + 1] + k]; d[k] = P[3 * t[3 * i + 2] + k]; }
    double bxd0 = b[1] * d[2] - b[2] * d[1], bxd1 = b[2] * d[0] - b[0] * d[2], bxd2 = b[0] * d[1] - b[1] * d[0];
    double v6 = a[0] * bxd0 + a[1] * bxd1 + a[2] * bxd2;
    double e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { d[0] - a[0], d[1] - a[1], d[2] - a[2] };
    double c0 = e1[1] * e2[2] - e1[2] * e2[1], c1 = e1[2] * e2[0] - e1[0] * e2[2], c2 = e1[0] * e2[1] - e1[1] * e2[0];
    double* o = ag_ptr<double>(p->out);
    o[i] = v6;
    o[n + i] = 0.5 * sqrt(c0 * c0 + c1 * c1 + c2 * c2);
    for (int k = 0; k < 3; ++k) o[(2 + k) * n + i] = (a[k] + b[k] + d[k]) * v6;
}
