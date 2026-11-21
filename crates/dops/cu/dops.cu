// Device operation layer (Tier N). Line-by-line twin of crates/dops/src/{math,reference}.rs:
// same operation order, no FMA contraction (NVRTC fmad=false), no libm transcendentals, so
// every result is bit-identical to the CPU reference (AG-06-006, AG-08-005).
#include "ag_common.cuh"

typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __attribute__((cudart_builtin)) void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

#define NVALS 16
#define OBS 32
#define CS 20
#define C_BLO 16
#define C_BHI 17
#define TQ 24u
#define TK 32
#define C_INSIDE 0
#define C_BOUNDARY 1
#define C_FILL 2
#define C_THIN 3
#define C_M1 4
#define C_M2 7
#define C_UNDERCUT 10
#define C_GAP 11
#define C_MIN_WALL 12
#define C_MIN_GAP 13
#define C_AREA 14
#define C_V6 15
#define MARCH_STEPS 160u
#define MAX_PARAMS 8
#define MAX_PATTERN 64u
#define NO_PARAM 0xFFFFFFFFu
#define LAST 0xFFFFFFFFu
#define PROFILE_RECT 0u
#define PROFILE_CIRCLE 1u
#define PROFILE_POLYGON 2u
#define PROFILE_SHELL 3u
#define PROFILE_BODY 4u
#define PROFILE_COMBINE 5u
#define PROFILE_SKETCH 6u
#define PROFILE_MIRROR 7u
#define PROFILE_REVOLVE 8u
#define MAX_SEGS 48u
#define PSTRIDE (1u + 8u * MAX_SEGS)
#define MAX_BODIES 4
#define FOP_INTERSECT 2u
#define FOP_ADD 0u
#define FOP_REMOVE 1u
#define NF_ACTIVE 1u
#define NF_PATTERN 2u
#define NF_PATTERN_RECT 4u
#define NF_FILLET 8u
#define NF_CHAMFER 16u
#define NF_DRAFT 32u
#define NF_ROUND_Z 64u
#define NF_CHAMFER_Z 128u
#define NF_MODS (NF_FILLET | NF_CHAMFER | NF_DRAFT | NF_ROUND_Z | NF_CHAMFER_Z)
#define NF_ZMODS (NF_DRAFT | NF_ROUND_Z | NF_CHAMFER_Z)
#define PARAM_ADD 1u
#define FAR 1.0e30
#define TAU 6.283185307179586
#define PI 3.141592653589793

struct Node { u32 kind, op, flags, src; u32 pref[NVALS]; double val[NVALS]; };
struct FeatureArgs { u32 pref[NVALS]; double val[NVALS]; };
struct CNode { u32 kind, op, active, count, sides, mods, _pad[2]; float v[NVALS]; float apothem, cosd; float _pad2[6]; };
#define FAR32 1.0e30f
#define TABLE 640
#define POLY 256
struct Goal {
    double domain_lo[3], domain_hi[3], vol_lo, vol_hi, max_ext[3], min_wall;
    double place[4], hole_r, hole_z0, hole_z1, gain;
    double param_lo[MAX_PARAMS], param_hi[MAX_PARAMS], param_sign[MAX_PARAMS];
    u32 n_params, seed, machining_3ax, _pad1;
    double min_gap, undercut_tol, centroid_lo[3], centroid_hi[3];
    u32 objective, _pad2;
    double obj_scale;
};
struct TraceRow { u32 round, committed, accepted, rejected; double vol_lo, vol_hi; };
struct OpTable { u32 update, parameter, pad, pocket, hole, circular_pattern, deactivate, rect_pattern, fillet, chamfer, shell, body, add, remove, intersect, draft, mirror, shaft, groove, _pad[1]; };
struct DopsArgs {
    u64 goal, cur_nodes, cur_count, cur_params, tr_nodes, tr_count, tr_params, rev, budget, done,
        cmds, arena, status, counts, domain_hit, bbox, obs, score, trace, round, active, compiled, tables,
        wl0, wl1, cls, flag, offs, crank, start, lvl, term, toffs, tlist, tcount, opt, samp, tscore, pool, cond_handle;
    u32 configs, nodes, params, records, coarse, levels, eval_trial, max_rounds, level, capacity, cap, cfg_offset, tighten, policy, island, general_ok;
    OpTable ops;
};
#define POLICY_CMAES 1u
#define MAX_ISLAND 64u
struct OptState { double mean[MAX_PARAMS], diag[MAX_PARAMS], sigma, best; u32 restarts, generation; };
struct CommandRecord { unsigned short opcode, flags; u32 configuration, dst, args[6], expected_revision, configuration_mask_slot, reserved; double imm[2]; };
struct StatusRecord { unsigned short status, flags; u32 dst, budget_used, aux; };

// ---- deterministic math (twin of math.rs) ----
__device__ const double SIN_C[9] = {-8.22063524662433e-18, 2.8114572543455206e-15, -7.647163731819816e-13, 1.6059043836821613e-10, -2.505210838544172e-08, 2.7557319223985893e-06, -0.0001984126984126984, 0.008333333333333333, -0.16666666666666666};
__device__ const double COS_C[10] = {4.110317623312165e-19, -1.5619206968586225e-16, 4.779477332387385e-14, -1.1470745597729725e-11, 2.08767569878681e-09, -2.755731922398589e-07, 2.48015873015873e-05, -0.001388888888888889, 0.041666666666666664, -0.5};

__device__ void ag_sincos(double x, double* so, double* co) {
    double k = floor(x * 6.36619772367581382433e-01 + 0.5);
    double r = (x - k * 1.57079632673412561417e+00) - k * 6.07710050650619224932e-11;
    double r2 = r * r;
    double ps = SIN_C[0];
    for (int i = 1; i < 9; ++i) ps = ps * r2 + SIN_C[i];
    double s = r + r * r2 * ps;
    double pc = COS_C[0];
    for (int i = 1; i < 10; ++i) pc = pc * r2 + COS_C[i];
    double c = 1.0 + r2 * pc;
    long long ki = (long long)k;
    int q = (int)(((ki % 4) + 4) % 4);
    if (q == 0) { *so = s; *co = c; }
    else if (q == 1) { *so = c; *co = -s; }
    else if (q == 2) { *so = -s; *co = -c; }
    else { *so = -c; *co = s; }
}

__device__ __forceinline__ double dmax(double a, double b) { return (a > b) ? a : ((b > a) ? b : (a == a ? a : b)); }
__device__ __forceinline__ double dmin(double a, double b) { return (a < b) ? a : ((b < a) ? b : (a == a ? a : b)); }

__device__ double value_of(const Node& n, const double* params, int j) {
    u32 p = n.pref[j];
    return (p == NO_PARAM) ? n.val[j] : params[p] + n.val[j];
}

__device__ double extrude(double d2, double z, double z0, double z1) {
    double dz = dmax(z0 - z, z - z1);
    double a = dmax(d2, 0.0);
    double b = dmax(dz, 0.0);
    return sqrt(a * a + b * b) + dmin(dmax(d2, dz), 0.0);
}

__device__ double prism_sdf(u32 kind, const double* v, double x, double y, double z) {
    double z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return FAR;
    if (kind == PROFILE_RECT) {
        double x0 = dmin(v[0], v[2]), y0 = dmin(v[1], v[3]), x1 = dmax(v[0], v[2]), y1 = dmax(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return FAR;
        double qx = dmax(x0 - x, x - x1);
        double qy = dmax(y0 - y, y - y1);
        double qz = dmax(z0 - z, z - z1);
        double a = dmax(qx, 0.0);
        double b = dmax(qy, 0.0);
        double c = dmax(qz, 0.0);
        return sqrt(a * a + b * b + c * c) + dmin(dmax(qx, dmax(qy, qz)), 0.0);
    }
    if (kind == PROFILE_CIRCLE) {
        double r = v[2];
        if (!(r > 0.0)) return FAR;
        double dx = x - v[0];
        double dy = y - v[1];
        double d2 = sqrt(dx * dx + dy * dy) - r;
        return extrude(d2, z, z0, z1);
    }
    double r = v[2];
    double sides = floor(v[3]);
    if (!(r > 0.0) || !(sides >= 3.0) || sides > 64.0) return FAR;
    u32 k = (u32)sides;
    double sa, ca;
    ag_sincos(PI / sides, &sa, &ca);
    double apothem = r * ca;
    double dx = x - v[0];
    double dy = y - v[1];
    double d2 = -FAR;
    for (u32 i = 0; i < k; ++i) {
        double ang = TAU * ((double)i + 0.5) / sides;
        double s, c;
        ag_sincos(ang, &s, &c);
        d2 = dmax(d2, dx * c + dy * s - apothem);
    }
    return extrude(d2, z, z0, z1);
}

__device__ double tool_sdf(const Node& n, const double* params, double x, double y, double z) {
    double v[NVALS];
    for (int j = 0; j < NVALS; ++j) v[j] = value_of(n, params, j);
    if ((n.flags & NF_PATTERN) == 0u) return prism_sdf(n.kind, v, x, y, z);
    double cnt = floor(v[6]);
    u32 count = (cnt < 1.0) ? 1u : ((cnt > (double)MAX_PATTERN) ? MAX_PATTERN : (u32)cnt);
    double total = v[7];
    bool full = fabs(total - TAU) < 1e-9;
    double step = full ? total / (double)count : (count > 1u ? total / (double)(count - 1u) : 0.0);
    double f = FAR;
    for (u32 i = 0; i < count; ++i) {
        double s, c;
        ag_sincos(step * (double)i, &s, &c);
        double px = x * c + y * s;
        double py = y * c - x * s;
        f = dmin(f, prism_sdf(n.kind, v, px, py, z));
    }
    return f;
}

__device__ double tape_sdf(const Node* nodes, u32 count, const double* params, double x, double y, double z) {
    double f = FAR;
    for (u32 i = 0; i < count; ++i) {
        const Node& n = nodes[i];
        if ((n.flags & NF_ACTIVE) == 0u) continue;
        double t = tool_sdf(n, params, x, y, z);
        f = (n.op == FOP_ADD) ? dmin(f, t) : dmax(f, -t);
    }
    return f;
}

// ---- compiled tapes (twin of math.rs compile_node / prism_sdf_c / tape_sdf_c) ----
__device__ CNode compile_node(const Node& n, const double* params, const float* pool, u32 npool, float* table) {
    CNode c;
    double v[NVALS];
    for (int j = 0; j < NVALS; ++j) v[j] = value_of(n, params, j);
    for (int j = 0; j < NVALS; ++j) c.v[j] = (float)v[j];
    c.kind = n.kind; c.op = n.op; c.active = ((n.flags & NF_ACTIVE) != 0u) ? 1u : 0u; c.count = 1u; c.sides = 0u;
    c.mods = n.flags & NF_MODS;
    c._pad[0] = c._pad[1] = 0u; c.apothem = 0.0f; c.cosd = 1.0f;
    for (int j = 0; j < 6; ++j) c._pad2[j] = 0.0f;
    if ((n.flags & NF_DRAFT) != 0u) c.cosd = (float)(1.0 / sqrt(1.0 + v[12] * v[12]));
    table[0] = 1.0f;
    table[1] = 0.0f;
    table[2] = 0.0f;
    table[3] = 0.0f;
    if ((n.flags & NF_PATTERN) != 0u) {
        double cnt = floor(v[6]);
        u32 count = (cnt < 1.0) ? 1u : ((cnt > (double)MAX_PATTERN) ? MAX_PATTERN : (u32)cnt);
        double total = v[7];
        bool full = fabs(total - TAU) < 1e-9;
        double step = full ? total / (double)count : (count > 1u ? total / (double)(count - 1u) : 0.0);
        for (u32 i = 0; i < count; ++i) {
            double s, co;
            ag_sincos(step * (double)i, &s, &co);
            table[4 * i] = (float)co;
            table[4 * i + 1] = (float)s;
            table[4 * i + 2] = 0.0f;
            table[4 * i + 3] = 0.0f;
        }
        c.count = count;
    } else if ((n.flags & NF_PATTERN_RECT) != 0u) {
        double c1f = floor(v[6]);
        u32 c1 = (c1f < 1.0) ? 1u : ((c1f > (double)MAX_PATTERN) ? MAX_PATTERN : (u32)c1f);
        double c2f = floor(v[8]);
        u32 lim = MAX_PATTERN / c1;
        u32 c2 = (c2f < 1.0) ? 1u : ((c2f > (double)lim) ? lim : (u32)c2f);
        u32 k = 0;
        for (u32 j = 0; j < c2; ++j) {
            for (u32 i = 0; i < c1; ++i) {
                table[4 * k] = 1.0f;
                table[4 * k + 1] = 0.0f;
                table[4 * k + 2] = (float)((double)i * v[7]);
                table[4 * k + 3] = (float)((double)j * v[9]);
                k += 1u;
            }
        }
        c.count = c1 * c2;
    }
    if (n.kind == PROFILE_POLYGON) {
        double sides = floor(v[3]);
        if (v[2] > 0.0 && sides >= 3.0 && sides <= 64.0) {
            u32 k = (u32)sides;
            double sa, ca;
            ag_sincos(PI / sides, &sa, &ca);
            c.apothem = (float)(v[2] * ca);
            for (u32 i = 0; i < k; ++i) {
                double ang = TAU * ((double)i + 0.5) / sides;
                double s, co;
                ag_sincos(ang, &s, &co);
                table[POLY + 2 * i] = (float)co;
                table[POLY + 2 * i + 1] = (float)s;
            }
            c.sides = k;
        }
    }
    if (n.kind == PROFILE_SKETCH || n.kind == PROFILE_REVOLVE) {
        double idx = floor(v[15]);
        if (idx >= 0.0 && (u64)idx < (u64)npool) {
            const float* src = pool + (u64)idx * PSTRIDE;
            u32 ns = (u32)src[0]; if (ns > MAX_SEGS) ns = MAX_SEGS;
            for (u32 j = 0; j < 8u * ns; ++j) table[POLY + j] = src[1 + j];
            c.sides = ns;
        }
    }
    return c;
}

// f32 evaluation (all literals f-suffixed: no silent promotion to double)
__device__ __forceinline__ float fmaxx(float a, float b) { return (a > b) ? a : ((b > a) ? b : (a == a ? a : b)); }
__device__ __forceinline__ float fminn(float a, float b) { return (a < b) ? a : ((b < a) ? b : (a == a ? a : b)); }

__device__ float extrude32(float d2, float z, float z0, float z1) {
    float dz = fmaxx(z0 - z, z - z1);
    float a = fmaxx(d2, 0.0f);
    float b = fmaxx(dz, 0.0f);
    return sqrtf(a * a + b * b) + fminn(fmaxx(d2, dz), 0.0f);
}

__device__ __noinline__ float sketch2d(const float* t, u32 n, float x, float y) {
    float dmin = FAR32;
    bool inside = false;
    for (u32 s = 0; s < n; ++s) {
        const float* r = t + POLY + 8u * s;
        float ax = r[1], ay = r[2], bx = r[3], by = r[4];
        if ((ay > y) != (by > y)) {
            float xi = ax + (y - ay) * (bx - ax) / (by - ay);
            if (x < xi) inside = !inside;
        }
        float d;
        if (r[0] == 0.0f) {
            float dx = bx - ax, dy = by - ay;
            float l2 = dx * dx + dy * dy;
            float u = (l2 > 0.0f) ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0.0f;
            if (u < 0.0f) u = 0.0f;
            if (u > 1.0f) u = 1.0f;
            float px = x - (ax + u * dx), py = y - (ay + u * dy);
            d = sqrtf(px * px + py * py);
        } else {
            float cx = r[5], cy = r[6], rr = fabsf(r[7]);
            float sg = (r[7] > 0.0f) ? 1.0f : -1.0f;
            float qx = x - cx, qy = y - cy;
            float ux = ax - cx, uy = ay - cy, vx = bx - cx, vy = by - cy;
            bool span = sg * (ux * qy - uy * qx) >= 0.0f && sg * (qx * vy - qy * vx) >= 0.0f;
            float q2 = qx * qx + qy * qy;
            if (span) {
                d = fabsf(sqrtf(q2) - rr);
            } else {
                float pax = x - ax, pay = y - ay, pbx = x - bx, pby = y - by;
                d = fminn(sqrtf(pax * pax + pay * pay), sqrtf(pbx * pbx + pby * pby));
            }
            float side = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if (side * sg < 0.0f && q2 < rr * rr) inside = !inside;
        }
        if (d < dmin) dmin = d;
    }
    return inside ? -dmin : dmin;
}

__device__ float profile_d2(const CNode& c, const float* table, float x, float y) {
    const float* v = c.v;
    if (c.kind == PROFILE_RECT) {
        float x0 = fminn(v[0], v[2]), y0 = fminn(v[1], v[3]), x1 = fmaxx(v[0], v[2]), y1 = fmaxx(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return FAR32;
        float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);
        float hx = 0.5f * (x1 - x0), hy = 0.5f * (y1 - y0);
        float ax = fabsf(x - cx);
        float ay = fabsf(y - cy);
        float k = fminn(fmaxx(v[10], 0.0f), fminn(hx, hy));
        if ((c.mods & NF_FILLET) != 0u) {
            float qx = ax - (hx - k);
            float qy = ay - (hy - k);
            float a = fmaxx(qx, 0.0f);
            float b = fmaxx(qy, 0.0f);
            return sqrtf(a * a + b * b) + fminn(fmaxx(qx, qy), 0.0f) - k;
        }
        float qx = ax - hx;
        float qy = ay - hy;
        float a = fmaxx(qx, 0.0f);
        float b = fmaxx(qy, 0.0f);
        float dbox = sqrtf(a * a + b * b) + fminn(fmaxx(qx, qy), 0.0f);
        return ((c.mods & NF_CHAMFER) != 0u) ? fmaxx(dbox, (ax + ay - (hx + hy - k)) * 0.70710678f) : dbox;
    }
    if (c.kind == PROFILE_CIRCLE) {
        float r = v[2];
        if (!(r > 0.0f)) return FAR32;
        float dx = x - v[0];
        float dy = y - v[1];
        return sqrtf(dx * dx + dy * dy) - r;
    }
    if (c.kind == PROFILE_SKETCH) return (c.sides == 0u) ? FAR32 : sketch2d(table, c.sides, x - v[0], y - v[1]);
    if (c.sides == 0u) return FAR32;
    float dx = x - v[0];
    float dy = y - v[1];
    float d2 = -FAR32;
    for (u32 i = 0; i < c.sides; ++i) d2 = fmaxx(d2, dx * table[POLY + 2 * i] + dy * table[POLY + 2 * i + 1] - c.apothem);
    return d2;
}

__device__ __noinline__ float zmod(const CNode& c, float d2, float z) {
    const float* v = c.v;
    float z0 = v[4], z1 = v[5];
    float d = ((c.mods & NF_DRAFT) != 0u) ? (d2 + (z - z0) * v[12]) * c.cosd : d2;
    float dz = fmaxx(z0 - z, z - z1);
    float r = fminn(fmaxx(v[13], 0.0f), 0.5f * (z1 - z0));
    if ((c.mods & NF_ROUND_Z) != 0u) {
        float qx = d + r;
        float qy = dz + r;
        float a = fmaxx(qx, 0.0f);
        float b = fmaxx(qy, 0.0f);
        return fminn(fmaxx(qx, qy), 0.0f) + sqrtf(a * a + b * b) - r;
    }
    if ((c.mods & NF_CHAMFER_Z) != 0u) return fmaxx(extrude32(d, z, z0, z1), (d + dz + r) * 0.70710678f);
    return extrude32(d, z, z0, z1);
}

__device__ __noinline__ float revolve_sdf(const CNode& c, const float* table, float x, float y, float z) {
    if (c.sides == 0u) return FAR32;
    float dx = x - c.v[0];
    float dy = y - c.v[1];
    return sketch2d(table, c.sides, sqrtf(dx * dx + dy * dy), z);
}

// new profile kinds and z modifiers: out of line (the classic rect / circle / polygon path
// below is the original hot code)
__device__ __noinline__ float prism_sdf_rare(const CNode& c, const float* table, float x, float y, float z) {
    if (c.kind == PROFILE_REVOLVE) return revolve_sdf(c, table, x, y, z);
    const float* v = c.v;
    float z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return FAR32;
    float d2 = profile_d2(c, table, x, y);
    if (d2 == FAR32) return FAR32;
    return ((c.mods & NF_ZMODS) != 0u) ? zmod(c, d2, z) : extrude32(d2, z, z0, z1);
}

// classic rect / circle / polygon prisms (tapes without general features)
__device__ __forceinline__ float prism_sdf_plain(const CNode& c, const float* table, float x, float y, float z) {
    const float* v = c.v;
    float z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return FAR32;
    if (c.kind == PROFILE_RECT) {
        float x0 = fminn(v[0], v[2]), y0 = fminn(v[1], v[3]), x1 = fmaxx(v[0], v[2]), y1 = fmaxx(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return FAR32;
        if ((c.mods & (NF_FILLET | NF_CHAMFER)) != 0u) {
            float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);
            float hx = 0.5f * (x1 - x0), hy = 0.5f * (y1 - y0);
            float ax = fabsf(x - cx);
            float ay = fabsf(y - cy);
            float k = fminn(fmaxx(v[10], 0.0f), fminn(hx, hy));
            float d2;
            if ((c.mods & NF_FILLET) != 0u) {
                float qx = ax - (hx - k);
                float qy = ay - (hy - k);
                float a = fmaxx(qx, 0.0f);
                float b = fmaxx(qy, 0.0f);
                d2 = sqrtf(a * a + b * b) + fminn(fmaxx(qx, qy), 0.0f) - k;
            } else {
                float qx = ax - hx;
                float qy = ay - hy;
                float a = fmaxx(qx, 0.0f);
                float b = fmaxx(qy, 0.0f);
                float dbox = sqrtf(a * a + b * b) + fminn(fmaxx(qx, qy), 0.0f);
                d2 = fmaxx(dbox, (ax + ay - (hx + hy - k)) * 0.70710678f);
            }
            return extrude32(d2, z, z0, z1);
        }
        float qx = fmaxx(x0 - x, x - x1);
        float qy = fmaxx(y0 - y, y - y1);
        float qz = fmaxx(z0 - z, z - z1);
        float a = fmaxx(qx, 0.0f);
        float b = fmaxx(qy, 0.0f);
        float cc = fmaxx(qz, 0.0f);
        return sqrtf(a * a + b * b + cc * cc) + fminn(fmaxx(qx, fmaxx(qy, qz)), 0.0f);
    }
    if (c.kind == PROFILE_CIRCLE) {
        float r = v[2];
        if (!(r > 0.0f)) return FAR32;
        float dx = x - v[0];
        float dy = y - v[1];
        float d2 = sqrtf(dx * dx + dy * dy) - r;
        return extrude32(d2, z, z0, z1);
    }
    if (c.sides == 0u) return FAR32;
    float dx = x - v[0];
    float dy = y - v[1];
    float d2 = -FAR32;
    for (u32 i = 0; i < c.sides; ++i) d2 = fmaxx(d2, dx * table[POLY + 2 * i] + dy * table[POLY + 2 * i + 1] - c.apothem);
    return extrude32(d2, z, z0, z1);
}

__device__ float prism_sdf_c(const CNode& c, const float* table, float x, float y, float z) {
    if (c.kind >= PROFILE_SKETCH || (c.mods & NF_ZMODS) != 0u) return prism_sdf_rare(c, table, x, y, z);
    return prism_sdf_plain(c, table, x, y, z);
}
template <bool G> __device__ __forceinline__ float P_sdf(const CNode& c, const float* t, float x, float y, float z) { if (G) return prism_sdf_c(c, t, x, y, z); return prism_sdf_plain(c, t, x, y, z); }

template <bool G> __device__ float tape_sdf_c_core(const CNode* nodes, u32 count, const float* tables, float x, float y, float z) {
    float f = FAR32;
    float stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u || c.kind == PROFILE_MIRROR) continue;
        if (c.kind == PROFILE_SHELL) { f = fmaxx(f, -f - c.v[0]); continue; }
        if (c.kind == PROFILE_BODY) { if (depth < MAX_BODIES) { stack[depth] = f; depth += 1; f = FAR32; } continue; }
        if (c.kind == PROFILE_COMBINE) {
            if (depth > 0) {
                depth -= 1;
                float t = stack[depth];
                f = (c.op == FOP_ADD) ? fminn(t, f) : ((c.op == FOP_REMOVE) ? fmaxx(t, -f) : fmaxx(t, f));
            }
            continue;
        }
        const float* t = tables + (u64)i * TABLE;
        float tf = FAR32;
        for (u32 k = 0; k < c.count; ++k) {
            float co = t[4 * k], s = t[4 * k + 1], tx = t[4 * k + 2], ty = t[4 * k + 3];
            float qx = x - tx;
            float qy = y - ty;
            float px = qx * co + qy * s;
            float py = qy * co - qx * s;
            tf = fminn(tf, P_sdf<G>(c, t, px, py, z));
        }
        f = (c.op == FOP_ADD) ? fminn(f, tf) : fmaxx(f, -tf);
    }
    return f;
}
// tape flags: bits 0-1 mirror planes, bit 2 general features (sketch / revolve / z modifiers)
__device__ u32 scan_mirror(const CNode* nodes, u32 count) {
    u32 m = 0u;
    for (u32 i = 0; i < count; ++i) {
        if (nodes[i].active == 0u) continue;
        if (nodes[i].kind == PROFILE_MIRROR) {
            if (nodes[i].v[0] > 0.5f) m |= 1u;
            if (nodes[i].v[1] > 0.5f) m |= 2u;
        }
        if (nodes[i].kind == PROFILE_SKETCH || nodes[i].kind == PROFILE_REVOLVE || (nodes[i].mods & NF_ZMODS) != 0u) m |= 4u;
    }
    return m;
}
__device__ __forceinline__ bool tape_general(const CNode* nodes, u32 count) {
    if (count > 0u && (nodes[0]._pad[0] & 0x80000000u) != 0u) return (nodes[0]._pad[0] & 7u) != 0u;
    return scan_mirror(nodes, count) != 0u;
}
__device__ __forceinline__ u32 mirror_mask(const CNode* nodes, u32 count) {
    if (count > 0u && (nodes[0]._pad[0] & 0x80000000u) != 0u) return nodes[0]._pad[0] & 3u;
    return scan_mirror(nodes, count);
}
// mirrored tapes: out of line, so unmirrored evaluation stays a single inlined tape
__device__ __noinline__ float tape_sdf_mirror(const CNode* nodes, u32 count, const float* tables, u32 m, float x, float y, float z) {
    float f = FAR32;
    for (u32 k = 0; k < 4u; ++k) {
        if ((k & ~m) != 0u) continue;
        f = fminn(f, tape_sdf_c_core<true>(nodes, count, tables, (k & 1u) ? -x : x, (k & 2u) ? -y : y, z));
    }
    return f;
}
__device__ float tape_sdf_c(const CNode* nodes, u32 count, const float* tables, float x, float y, float z) {
    u32 m = mirror_mask(nodes, count);
    if (m == 0u) return tape_sdf_c_core<true>(nodes, count, tables, x, y, z);
    return tape_sdf_mirror(nodes, count, tables, m, x, y, z);
}

__device__ u32 hash3(u32 a, u32 b, u32 c) {
    u32 h = (a * 0x9E3779B1u) ^ (b * 0x85EBCA77u) ^ (c * 0xC2B2AE3Du);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}
__device__ __forceinline__ double unit01(u32 h) { return (double)(h >> 8) * (1.0 / 16777216.0); }

// ---- CMA-ES islands (twin of reference.rs cma_sample / ag_opt_*) ----
__device__ __forceinline__ bool param_free(const Goal& g, u32 j) { return g.param_hi[j] > g.param_lo[j]; }
__device__ double normal12(u32 seed, u32 gcfg, u32 idx) {
    double s = 0.0;
    for (u32 k = 0; k < 12u; ++k) s += unit01(hash3(seed ^ 0x51ED270Bu, gcfg, idx * 12u + k));
    return s - 6.0;
}

__device__ void cma_sample(const DopsArgs* a, const Goal& g, u32 cfg, u32 gcfg, u32 round, CommandRecord* cmds) {
    u32 kn = a->params, lam = a->island;
    const OptState st = ag_ptr<const OptState>(a->opt)[cfg / lam];
    double* samp = ag_ptr<double>(a->samp) + (u64)cfg * MAX_PARAMS;
    const double* pr = ag_ptr<const double>(a->cur_params) + (u64)cfg * kn;
    u32 np = g.n_params; if (np > kn) np = kn; if (np > MAX_PARAMS) np = MAX_PARAMS;
    bool idle = ag_ptr<const u32>(a->done)[cfg] != 0u || ag_ptr<const u32>(a->budget)[cfg] == 0u;
    if (idle) {
        for (u32 j = 0; j < MAX_PARAMS; ++j) samp[j] = (j < np && param_free(g, j)) ? (pr[j] - g.param_lo[j]) / (g.param_hi[j] - g.param_lo[j]) : 0.0;
        ag_ptr<double>(a->tscore)[cfg] = ag_ptr<const double>(a->score)[cfg];
        return;
    }
    u32 slot = 0;
    for (u32 j = 0; j < MAX_PARAMS; ++j) {
        double u = 0.0;
        if (j < np && param_free(g, j)) {
            double z = normal12(g.seed, gcfg, round * 16u + j);
            u = st.mean[j] + st.sigma * sqrt(st.diag[j]) * z;
            if (u < 0.0) u = 0.0;
            if (u > 1.0) u = 1.0;
            if (slot < a->records) {
                CommandRecord& c = cmds[slot];
                c.opcode = (unsigned short)a->ops.parameter; c.args[0] = j; c.args[1] = 0u;
                c.imm[0] = g.param_lo[j] + u * (g.param_hi[j] - g.param_lo[j]);
                slot += 1u;
            }
        }
        samp[j] = u;
    }
}

AG_KERNEL void ag_opt_init(const DopsArgs* a) {
    if (AG_TID != 0 || a->policy != POLICY_CMAES) return;
    u32 cn = a->configs, kn = a->params, lam = a->island;
    const Goal g = *ag_ptr<const Goal>(a->goal);
    const double* pr = ag_ptr<const double>(a->cur_params);
    OptState* opt = ag_ptr<OptState>(a->opt);
    u32 np = g.n_params; if (np > kn) np = kn; if (np > MAX_PARAMS) np = MAX_PARAMS;
    u32 ni = (cn + lam - 1u) / lam;
    for (u32 i = 0; i < ni; ++i) {
        u32 gi = a->cfg_offset / lam + i;
        OptState st;
        for (u32 j = 0; j < MAX_PARAMS; ++j) {
            st.mean[j] = 0.0; st.diag[j] = 1.0;
            if (j < np && param_free(g, j)) st.mean[j] = (gi == 0u) ? (pr[(u64)i * lam * kn + j] - g.param_lo[j]) / (g.param_hi[j] - g.param_lo[j]) : unit01(hash3(g.seed ^ 0x0CA15EEDu, gi, j));
        }
        st.sigma = 0.3; st.best = 1.0e30; st.restarts = 0u; st.generation = 0u;
        opt[i] = st;
    }
}

__device__ void island_update(OptState& st, const double* samp, const double* fs, u32 m, const Goal& g, u32 np, u32 gi, u32 round) {
    if (m == 0u) return;
    u32 order[MAX_ISLAND];
    for (u32 k = 0; k < m; ++k) {
        u32 p = k;
        while (p > 0u && (fs[order[p - 1u]] > fs[k] || (fs[order[p - 1u]] == fs[k] && order[p - 1u] > k))) { order[p] = order[p - 1u]; p -= 1u; }
        order[p] = k;
    }
    u32 mu = m / 2u; if (mu < 1u) mu = 1u;
    double wsum = (double)(mu * (mu + 1u) / 2u);
    u32 succ = 0;
    for (u32 k = 0; k < m; ++k) if (fs[k] < st.best) succ += 1u;
    if (fs[order[0]] < st.best) st.best = fs[order[0]];
    double old[MAX_PARAMS];
    for (u32 j = 0; j < MAX_PARAMS; ++j) old[j] = st.mean[j];
    for (u32 j = 0; j < np; ++j) {
        if (!param_free(g, j)) continue;
        double mj = 0.0, var = 0.0;
        for (u32 r = 0; r < mu; ++r) {
            double w = (double)(mu - r) / wsum;
            double x = samp[(u64)order[r] * MAX_PARAMS + j];
            mj += w * x;
            double d = (x - old[j]) / st.sigma;
            var += w * d * d;
        }
        st.mean[j] = mj;
        double dg = 0.8 * st.diag[j] + 0.2 * var;
        if (dg < 1e-8) dg = 1e-8;
        if (dg > 1e4) dg = 1e4;
        st.diag[j] = dg;
    }
    st.sigma *= (succ * 5u > m) ? 1.2 : 0.85;
    if (st.sigma < 1e-4) st.sigma = 1e-4;
    if (st.sigma > 1.0) st.sigma = 1.0;
    st.generation += 1u;
    if (st.sigma < 1e-3 && st.best >= 1000.0) {
        for (u32 j = 0; j < np; ++j) if (param_free(g, j)) { st.mean[j] = unit01(hash3(g.seed ^ 0x0CA15EEDu, gi, ((round + 1u) << 4) | j)); st.diag[j] = 1.0; }
        st.sigma = 0.3;
        st.restarts += 1u;
    }
}

AG_KERNEL void ag_opt_update(const DopsArgs* a) {
    if (a->policy != POLICY_CMAES) return;
    u32 cn = a->configs, lam = a->island;
    const Goal g = *ag_ptr<const Goal>(a->goal);
    u32 round = ag_ptr<const u32>(a->round)[0];
    const double* samp = ag_ptr<const double>(a->samp);
    const double* ts = ag_ptr<const double>(a->tscore);
    OptState* opt = ag_ptr<OptState>(a->opt);
    u32 np = g.n_params; if (np > a->params) np = a->params; if (np > MAX_PARAMS) np = MAX_PARAMS;
    u32 ni = (cn + lam - 1u) / lam;
    u32 i = AG_TID; // one thread per island (islands are independent)
    if (i >= ni) return;
    u32 m0 = i * lam;
    u32 m = (lam < cn - m0) ? lam : cn - m0;
    island_update(opt[i], samp + (u64)m0 * MAX_PARAMS, ts + m0, m, g, np, a->cfg_offset / lam + i, round);
}

// ---- policy (mock GPU model) ----
AG_KERNEL void ag_dop_policy(const DopsArgs* a) {
    u32 cfg = AG_TID;
    if (cfg >= a->configs) return;
    u32 rn = a->records, kn = a->params;
    const Goal g = *ag_ptr<const Goal>(a->goal);
    u32 rev = ag_ptr<const u32>(a->rev)[cfg];
    CommandRecord* cmds = ag_ptr<CommandRecord>(a->cmds);
    FeatureArgs* arena = ag_ptr<FeatureArgs>(a->arena);
    u32 base = cfg * rn;
    u32 gcfg = cfg + a->cfg_offset;
    for (u32 s = 0; s < rn; ++s) {
        CommandRecord c;
        c.opcode = (unsigned short)a->ops.update; c.flags = 0; c.configuration = cfg; c.dst = 0;
        for (int i = 0; i < 6; ++i) c.args[i] = 0;
        c.expected_revision = rev; c.configuration_mask_slot = 0; c.reserved = 0; c.imm[0] = 0.0; c.imm[1] = 0.0;
        cmds[base + s] = c;
        FeatureArgs f;
        for (int j = 0; j < NVALS; ++j) { f.pref[j] = 0u; f.val[j] = 0.0; }
        arena[base + s] = f;
    }
    if (a->policy == POLICY_CMAES) { cma_sample(a, g, cfg, gcfg, ag_ptr<const u32>(a->round)[0], cmds + base); return; }
    if (ag_ptr<const u32>(a->done)[cfg] != 0u || ag_ptr<const u32>(a->budget)[cfg] == 0u) return;
    u32 np = g.n_params; if (np > kn) np = kn; if (np > MAX_PARAMS) np = MAX_PARAMS;
    if (np == 0u) return;
    u32 round = ag_ptr<const u32>(a->round)[0];
    if (round == 0u) {
        if (gcfg != 0u) {
            u32 m = (np < rn) ? np : rn;
            for (u32 j = 0; j < m; ++j) {
                double v = g.param_lo[j] + unit01(hash3(g.seed, gcfg, j)) * (g.param_hi[j] - g.param_lo[j]);
                CommandRecord& c = cmds[base + j];
                c.opcode = (unsigned short)a->ops.parameter; c.args[0] = j; c.args[1] = 0u; c.imm[0] = v;
            }
        }
        return;
    }
    const double* o = ag_ptr<const double>(a->obs) + (u64)cfg * OBS;
    const double* pr = ag_ptr<const double>(a->cur_params) + (u64)cfg * kn;
    u32 count = ag_ptr<const u32>(a->cur_count)[cfg];
    double t = 0.5 * (g.vol_lo + g.vol_hi);
    double err = (g.vol_hi > g.vol_lo) ? (o[12] - t) / t : 0.0;
    u32 slot = 0;
    u32 acts = (np >= 2u) ? 2u : 1u;
    for (u32 act = 0; act < acts; ++act) {
        if (slot >= rn) break;
        u32 j = (gcfg + round + act) % np;
        double lo = g.param_lo[j], hi = g.param_hi[j];
        double cur = pr[j];
        double d = -err * g.param_sign[j] * g.gain * (hi - lo);
        double nv = cur + d;
        if (nv < lo) nv = lo;
        if (nv > hi) nv = hi;
        if (fabs(nv - cur) <= 1e-12 * (hi - lo)) {
            if (err > 0.0 && act == 0u && count < a->nodes && g.hole_r > 0.0) {
                double ux = unit01(hash3(g.seed ^ 0xA5A5A5A5u, gcfg, round));
                double uy = unit01(hash3(g.seed ^ 0x5A5A5A5Au, gcfg, round));
                double px = g.place[0] + ux * (g.place[2] - g.place[0]);
                double py = g.place[1] + uy * (g.place[3] - g.place[1]);
                bool rect = ((gcfg + round) % 2u == 1u) && (slot + 1u < rn);
                CommandRecord& c = cmds[base + slot];
                c.opcode = (unsigned short)a->ops.pocket;
                FeatureArgs& f = arena[base + slot];
                for (int jj = 0; jj < NVALS; ++jj) { f.pref[jj] = NO_PARAM; f.val[jj] = 0.0; }
                if (rect) {
                    c.args[0] = PROFILE_RECT;
                    double h = g.hole_r;
                    f.val[0] = px - h; f.val[1] = py - h; f.val[2] = px + h; f.val[3] = py + h; f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
                    slot += 1u;
                    CommandRecord& c2 = cmds[base + slot];
                    c2.opcode = (unsigned short)a->ops.fillet; c2.args[0] = LAST;
                    FeatureArgs& f2 = arena[base + slot];
                    for (int jj = 0; jj < NVALS; ++jj) { f2.pref[jj] = NO_PARAM; f2.val[jj] = 0.0; }
                    f2.val[10] = 0.4 * h;
                } else {
                    c.args[0] = PROFILE_CIRCLE;
                    f.val[0] = px; f.val[1] = py; f.val[2] = g.hole_r; f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
                }
                slot += 1u;
            }
            continue;
        }
        CommandRecord& c = cmds[base + slot];
        c.opcode = (unsigned short)a->ops.parameter; c.args[0] = j; c.args[1] = 0u; c.imm[0] = nv;
        slot += 1u;
    }
}

// ---- execute ----
AG_KERNEL void ag_dop_apply(const DopsArgs* a) {
    u32 npool = (u32)ag_ptr<const float>(a->pool)[0];
    u32 cfg = AG_TID;
    if (cfg >= a->configs) return;
    u32 nn = a->nodes, kn = a->params, rn = a->records;
    const Node* cur_n = ag_ptr<const Node>(a->cur_nodes) + (u64)cfg * nn;
    const double* cur_p = ag_ptr<const double>(a->cur_params) + (u64)cfg * kn;
    Node* tr_n = ag_ptr<Node>(a->tr_nodes) + (u64)cfg * nn;
    double* tr_p = ag_ptr<double>(a->tr_params) + (u64)cfg * kn;
    const CommandRecord* cmds = ag_ptr<const CommandRecord>(a->cmds) + (u64)cfg * rn;
    const FeatureArgs* arena = ag_ptr<const FeatureArgs>(a->arena) + (u64)cfg * rn;
    StatusRecord* status = ag_ptr<StatusRecord>(a->status) + (u64)cfg * rn;
    for (u32 i = 0; i < nn; ++i) tr_n[i] = cur_n[i];
    for (u32 k = 0; k < kn; ++k) tr_p[k] = cur_p[k];
    u32 count = ag_ptr<const u32>(a->cur_count)[cfg];
    for (u32 r = 0; r < rn; ++r) {
        if (status[r].status != AG_ACCEPTED) continue;
        const CommandRecord c = cmds[r];
        u32 op = (u32)c.opcode;
        unsigned short st = AG_ACCEPTED;
        if (op == a->ops.update) {
        } else if (op == a->ops.parameter) {
            u32 k = c.args[0];
            if (k >= kn) st = AG_INVALID_PARAMETER; else {
                double v = (c.args[1] == PARAM_ADD) ? tr_p[k] + c.imm[0] : c.imm[0];
                tr_p[k] = v;
            }
        } else if (op == a->ops.pad || op == a->ops.pocket || op == a->ops.hole || op == a->ops.shaft || op == a->ops.groove) {
            const FeatureArgs f = arena[r];
            bool bad_ref = false;
            for (int j = 0; j < NVALS; ++j) if (f.pref[j] != NO_PARAM && f.pref[j] >= kn) bad_ref = true;
            bool revolve = (op == a->ops.shaft || op == a->ops.groove);
            u32 kind = revolve ? PROFILE_REVOLVE : c.args[0];
            bool pooled = (kind == PROFILE_SKETCH || kind == PROFILE_REVOLVE);
            bool pool_ok = !pooled || (f.pref[15] == NO_PARAM && f.val[15] >= 0.0 && (u64)floor(f.val[15]) < (u64)npool);
            if ((!revolve && kind > PROFILE_POLYGON && kind != PROFILE_SKETCH) || bad_ref || !pool_ok) st = AG_INVALID_PARAMETER;
            else if (pooled && a->general_ok == 0u) st = AG_UNSUPPORTED_OP;
            else if (count >= nn) st = AG_CAPACITY_EXCEEDED;
            else {
                Node n;
                bool add = (op == a->ops.pad || op == a->ops.shaft);
                n.kind = kind; n.op = add ? FOP_ADD : FOP_REMOVE; n.flags = NF_ACTIVE; n.src = r;
                for (int j = 0; j < NVALS; ++j) { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                tr_n[count] = n;
                count += 1u;
            }
        } else if (op == a->ops.mirror) {
            const FeatureArgs f = arena[r];
            int depth = 0;
            for (u32 k = 0; k < count; ++k) { const Node& m = tr_n[k]; if ((m.flags & NF_ACTIVE) != 0u) { if (m.kind == PROFILE_BODY) depth += 1; else if (m.kind == PROFILE_COMBINE) depth -= 1; } }
            if (a->general_ok == 0u) st = AG_UNSUPPORTED_OP;
            else if (count >= nn) st = AG_CAPACITY_EXCEEDED;
            else if (depth != 0) st = AG_INVALID_PARAMETER;
            else {
                Node nd;
                nd.kind = PROFILE_MIRROR; nd.op = FOP_ADD; nd.flags = NF_ACTIVE; nd.src = r;
                for (int j = 0; j < NVALS; ++j) { nd.pref[j] = NO_PARAM; nd.val[j] = 0.0; }
                nd.val[0] = (f.val[0] > 0.5) ? 1.0 : 0.0;
                nd.val[1] = (f.val[1] > 0.5) ? 1.0 : 0.0;
                tr_n[count] = nd;
                count += 1u;
            }
        } else if (op == a->ops.shell || op == a->ops.body || op == a->ops.add || op == a->ops.remove || op == a->ops.intersect) {
            int depth = 0;
            for (u32 k = 0; k < count; ++k) {
                const Node& m = tr_n[k];
                if ((m.flags & NF_ACTIVE) == 0u) continue;
                if (m.kind == PROFILE_BODY) depth += 1; else if (m.kind == PROFILE_COMBINE) depth -= 1;
            }
            const FeatureArgs f = arena[r];
            bool is_bool = (op == a->ops.add || op == a->ops.remove || op == a->ops.intersect);
            if (count >= nn) st = AG_CAPACITY_EXCEEDED;
            else if (op == a->ops.body && depth >= MAX_BODIES) st = AG_CAPACITY_EXCEEDED;
            else if (is_bool && depth <= 0) st = AG_INVALID_PARAMETER;
            else if (op == a->ops.shell && f.pref[0] != NO_PARAM && f.pref[0] >= kn) st = AG_INVALID_PARAMETER;
            else {
                Node nd;
                nd.kind = (op == a->ops.shell) ? PROFILE_SHELL : ((op == a->ops.body) ? PROFILE_BODY : PROFILE_COMBINE);
                nd.op = (op == a->ops.remove) ? FOP_REMOVE : ((op == a->ops.intersect) ? FOP_INTERSECT : FOP_ADD);
                nd.flags = NF_ACTIVE; nd.src = r;
                for (int j = 0; j < NVALS; ++j) { nd.pref[j] = NO_PARAM; nd.val[j] = 0.0; }
                if (op == a->ops.shell) { nd.pref[0] = f.pref[0]; nd.val[0] = f.val[0]; }
                tr_n[count] = nd;
                count += 1u;
            }
        } else if (op == a->ops.circular_pattern || op == a->ops.rect_pattern || op == a->ops.fillet || op == a->ops.chamfer || op == a->ops.deactivate || op == a->ops.draft) {
            u32 fi = (c.args[0] == LAST) ? count - 1u : c.args[0];
            if (count == 0u || fi >= count) st = AG_INVALID_PARAMETER; else {
                Node& n = tr_n[fi];
                const FeatureArgs f = arena[r];
                bool edges_z = (op == a->ops.fillet || op == a->ops.chamfer) && c.args[1] == 1u;
                int lo, hi;
                if (op == a->ops.circular_pattern) { lo = 6; hi = 8; }
                else if (op == a->ops.rect_pattern) { lo = 6; hi = 10; }
                else if (op == a->ops.deactivate) { lo = 0; hi = 0; }
                else if (op == a->ops.draft) { lo = 12; hi = 13; }
                else if (edges_z) { lo = 13; hi = 14; }
                else { lo = 10; hi = 11; }
                bool bad_ref = false;
                for (int j = lo; j < hi; ++j) if (f.pref[j] != NO_PARAM && f.pref[j] >= kn) bad_ref = true;
                bool is_profile_op = (op == a->ops.fillet || op == a->ops.chamfer);
                bool extrusion = n.kind <= PROFILE_POLYGON || n.kind == PROFILE_SKETCH;
                bool unsupported = (op == a->ops.draft || edges_z) ? !extrusion : (is_profile_op ? n.kind != PROFILE_RECT : false);
                bool out_of_range = (op == a->ops.draft && f.pref[12] == NO_PARAM && !(fabs(f.val[12]) <= 0.57735)) || (edges_z && f.pref[13] == NO_PARAM && !(f.val[13] >= 0.0));
                if (bad_ref || unsupported || ((op == a->ops.draft || edges_z) && a->general_ok == 0u)) st = bad_ref ? AG_INVALID_PARAMETER : AG_UNSUPPORTED_OP;
                else if (out_of_range) st = AG_GEOMETRY_CHECK_FAILED;
                else if (op == a->ops.deactivate) { if (c.args[1] != 0u) n.flags |= NF_ACTIVE; else n.flags &= ~NF_ACTIVE; }
                else {
                    for (int j = lo; j < hi; ++j) { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                    if (op == a->ops.circular_pattern) n.flags = (n.flags & ~NF_PATTERN_RECT) | NF_PATTERN;
                    else if (op == a->ops.rect_pattern) n.flags = (n.flags & ~NF_PATTERN) | NF_PATTERN_RECT;
                    else if (op == a->ops.draft) n.flags |= NF_DRAFT;
                    else if (edges_z) n.flags = (n.flags & ~(NF_ROUND_Z | NF_CHAMFER_Z)) | ((op == a->ops.fillet) ? NF_ROUND_Z : NF_CHAMFER_Z);
                    else if (op == a->ops.fillet) n.flags = (n.flags & ~NF_CHAMFER) | NF_FILLET;
                    else n.flags = (n.flags & ~NF_FILLET) | NF_CHAMFER;
                }
            }
        } else {
            st = AG_UNSUPPORTED_OP;
        }
        status[r].status = st;
    }
    ag_ptr<u32>(a->tr_count)[cfg] = count;
}

// ---- compile ----
AG_KERNEL void ag_dop_compile(const DopsArgs* a) {
    u32 c = AG_TID;
    if (c >= a->configs) return;
    if (a->eval_trial != 0u && ag_ptr<const u32>(a->done)[c] != 0u) return;
    u32 nn = a->nodes, kn = a->params;
    const Node* nodes = ag_ptr<const Node>(a->eval_trial ? a->tr_nodes : a->cur_nodes) + (u64)c * nn;
    u32 cnt = ag_ptr<const u32>(a->eval_trial ? a->tr_count : a->cur_count)[c];
    const double* prm = ag_ptr<const double>(a->eval_trial ? a->tr_params : a->cur_params) + (u64)c * kn;
    CNode* out = ag_ptr<CNode>(a->compiled) + (u64)c * nn;
    float* tabs = ag_ptr<float>(a->tables) + (u64)c * nn * TABLE;
    const float* pool = ag_ptr<const float>(a->pool);
    u32 np = (u32)pool[0];
    for (u32 i = 0; i < cnt; ++i) out[i] = compile_node(nodes[i], prm, pool + 1, np, tabs + (u64)i * TABLE);
    if (cnt > 0u) out[0]._pad[0] = 0x80000000u | scan_mirror(out, cnt);
}

// ---- evaluate ----
AG_KERNEL void ag_dop_eval_clear(const DopsArgs* a) {
    u32 c = AG_TID;
    if (c >= a->configs) return;
    u64* counts = ag_ptr<u64>(a->counts);
    for (int j = 0; j < CS; ++j) counts[CS * c + j] = 0ull;
    counts[CS * c + C_MIN_WALL] = 0xFFFFFFFFFFFFFFFFull;
    counts[CS * c + C_MIN_GAP] = 0xFFFFFFFFFFFFFFFFull;
    ag_ptr<u32>(a->domain_hit)[c] = 0u;
    u32* bb = ag_ptr<u32>(a->bbox) + 12u * c;
    for (int k = 0; k < 3; ++k) { bb[k] = 0xFFFFFFFFu; bb[3 + k] = 0u; bb[6 + k] = 0xFFFFFFFFu; bb[9 + k] = 0u; }
}

// ---- interval arithmetic (twin of math.rs Iv) ----
struct Iv { float lo, hi; };
__device__ __forceinline__ Iv mkiv(float lo, float hi) { Iv r; r.lo = lo; r.hi = hi; return r; }
__device__ __forceinline__ Iv iv_sub_ci(float c, Iv x) { return mkiv(c - x.hi, c - x.lo); }
__device__ __forceinline__ Iv iv_sub_ic(Iv x, float c) { return mkiv(x.lo - c, x.hi - c); }
__device__ __forceinline__ Iv iv_sub(Iv a, Iv b) { return mkiv(a.lo - b.hi, a.hi - b.lo); }
__device__ __forceinline__ Iv iv_add(Iv a, Iv b) { return mkiv(a.lo + b.lo, a.hi + b.hi); }
__device__ __forceinline__ Iv iv_max(Iv a, Iv b) { return mkiv(fmaxx(a.lo, b.lo), fmaxx(a.hi, b.hi)); }
__device__ __forceinline__ Iv iv_min(Iv a, Iv b) { return mkiv(fminn(a.lo, b.lo), fminn(a.hi, b.hi)); }
__device__ __forceinline__ Iv iv_maxc(Iv a, float c) { return mkiv(fmaxx(a.lo, c), fmaxx(a.hi, c)); }
__device__ __forceinline__ Iv iv_minc(Iv a, float c) { return mkiv(fminn(a.lo, c), fminn(a.hi, c)); }
__device__ __forceinline__ Iv iv_neg(Iv a) { return mkiv(-a.hi, -a.lo); }
__device__ __forceinline__ Iv iv_scale(Iv a, float c) { return (c >= 0.0f) ? mkiv(a.lo * c, a.hi * c) : mkiv(a.hi * c, a.lo * c); }
__device__ __forceinline__ Iv iv_sq(Iv a) {
    if (a.lo >= 0.0f) return mkiv(a.lo * a.lo, a.hi * a.hi);
    if (a.hi <= 0.0f) return mkiv(a.hi * a.hi, a.lo * a.lo);
    return mkiv(0.0f, fmaxx(a.lo * a.lo, a.hi * a.hi));
}
__device__ __forceinline__ Iv iv_sqrt(Iv a) { return mkiv(sqrtf(fmaxx(a.lo, 0.0f)), sqrtf(fmaxx(a.hi, 0.0f))); }
__device__ __forceinline__ Iv iv_abs(Iv a) { if (a.lo >= 0.0f) return a; if (a.hi <= 0.0f) return mkiv(-a.hi, -a.lo); return mkiv(0.0f, fmaxx(-a.lo, a.hi)); }

__device__ Iv extrude_iv(Iv d2, Iv z, float z0, float z1) {
    Iv dz = iv_max(iv_sub_ci(z0, z), iv_sub_ic(z, z1));
    Iv a = iv_maxc(d2, 0.0f);
    Iv b = iv_maxc(dz, 0.0f);
    return iv_add(iv_sqrt(iv_add(iv_sq(a), iv_sq(b))), iv_minc(iv_max(d2, dz), 0.0f));
}

__device__ __forceinline__ float iv_mid(Iv a) { return 0.5f * (a.lo + a.hi); }
__device__ __forceinline__ float iv_wid(Iv a) { return a.hi - a.lo; }
__device__ __forceinline__ Iv center_iv(float f, float r) { return mkiv(f - r, f + r); }

// Lipschitz-centre intervals of the rare profiles (kept out of line: the hot rect / circle /
// polygon path stays small).
__device__ __noinline__ Iv prism_iv_rare(const CNode& c, const float* table, Iv x, Iv y, Iv z) {
    const float* v = c.v;
    Iv far = mkiv(FAR32, FAR32);
    if (c.kind == PROFILE_REVOLVE) {
        if (c.sides == 0u) return far;
        Iv rho = iv_sqrt(iv_add(iv_sq(iv_sub_ic(x, v[0])), iv_sq(iv_sub_ic(y, v[1]))));
        float wr = iv_wid(rho), wz = iv_wid(z);
        return center_iv(sketch2d(table, c.sides, iv_mid(rho), iv_mid(z)), 0.5f * sqrtf(wr * wr + wz * wz));
    }
    float z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return far;
    if ((c.mods & NF_ZMODS) != 0u) {
        float wx = iv_wid(x), wy = iv_wid(y), wz = iv_wid(z);
        float f = prism_sdf_c(c, table, iv_mid(x), iv_mid(y), iv_mid(z));
        if (f == FAR32) return far;
        return center_iv(f, 0.5f * sqrtf(wx * wx + wy * wy + wz * wz));
    }
    if (c.sides == 0u) return far;
    float wx = iv_wid(x), wy = iv_wid(y);
    float d = sketch2d(table, c.sides, iv_mid(x) - v[0], iv_mid(y) - v[1]);
    return extrude_iv(center_iv(d, 0.5f * sqrtf(wx * wx + wy * wy)), z, z0, z1);
}

__device__ __forceinline__ Iv prism_iv_plain(const CNode& c, const float* table, Iv x, Iv y, Iv z) {
    const float* v = c.v;
    Iv far = mkiv(FAR32, FAR32);
    float z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return far;
    if (c.kind == PROFILE_RECT) {
        float x0 = fminn(v[0], v[2]), y0 = fminn(v[1], v[3]), x1 = fmaxx(v[0], v[2]), y1 = fmaxx(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return far;
        if ((c.mods & (NF_FILLET | NF_CHAMFER)) != 0u) {
            float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);
            float hx = 0.5f * (x1 - x0), hy = 0.5f * (y1 - y0);
            Iv ax = iv_abs(iv_sub_ic(x, cx));
            Iv ay = iv_abs(iv_sub_ic(y, cy));
            float k = fminn(fmaxx(v[10], 0.0f), fminn(hx, hy));
            Iv d2;
            if ((c.mods & NF_FILLET) != 0u) {
                Iv qx = iv_sub_ic(ax, hx - k);
                Iv qy = iv_sub_ic(ay, hy - k);
                d2 = iv_sub_ic(iv_add(iv_sqrt(iv_add(iv_sq(iv_maxc(qx, 0.0f)), iv_sq(iv_maxc(qy, 0.0f)))), iv_minc(iv_max(qx, qy), 0.0f)), k);
            } else {
                Iv qx = iv_sub_ic(ax, hx);
                Iv qy = iv_sub_ic(ay, hy);
                Iv dbox = iv_add(iv_sqrt(iv_add(iv_sq(iv_maxc(qx, 0.0f)), iv_sq(iv_maxc(qy, 0.0f)))), iv_minc(iv_max(qx, qy), 0.0f));
                d2 = iv_max(dbox, iv_scale(iv_sub_ic(iv_add(ax, ay), hx + hy - k), 0.70710678f));
            }
            return extrude_iv(d2, z, z0, z1);
        }
        Iv qx = iv_max(iv_sub_ci(x0, x), iv_sub_ic(x, x1));
        Iv qy = iv_max(iv_sub_ci(y0, y), iv_sub_ic(y, y1));
        Iv qz = iv_max(iv_sub_ci(z0, z), iv_sub_ic(z, z1));
        Iv s = iv_sqrt(iv_add(iv_add(iv_sq(iv_maxc(qx, 0.0f)), iv_sq(iv_maxc(qy, 0.0f))), iv_sq(iv_maxc(qz, 0.0f))));
        return iv_add(s, iv_minc(iv_max(qx, iv_max(qy, qz)), 0.0f));
    }
    if (c.kind == PROFILE_CIRCLE) {
        float r = v[2];
        if (!(r > 0.0f)) return far;
        Iv dx = iv_sub_ic(x, v[0]);
        Iv dy = iv_sub_ic(y, v[1]);
        Iv d2 = iv_sub_ic(iv_sqrt(iv_add(iv_sq(dx), iv_sq(dy))), r);
        return extrude_iv(d2, z, z0, z1);
    }
    if (c.sides == 0u) return far;
    Iv dx = iv_sub_ic(x, v[0]);
    Iv dy = iv_sub_ic(y, v[1]);
    Iv d2 = mkiv(-FAR32, -FAR32);
    for (u32 i = 0; i < c.sides; ++i) {
        Iv lin = iv_sub_ic(iv_add(iv_scale(dx, table[POLY + 2 * i]), iv_scale(dy, table[POLY + 2 * i + 1])), c.apothem);
        d2 = iv_max(d2, lin);
    }
    return extrude_iv(d2, z, z0, z1);
}

__device__ Iv prism_iv(const CNode& c, const float* table, Iv x, Iv y, Iv z) {
    if (c.kind == PROFILE_REVOLVE || c.kind == PROFILE_SKETCH || (c.mods & NF_ZMODS) != 0u) return prism_iv_rare(c, table, x, y, z);
    return prism_iv_plain(c, table, x, y, z);
}
template <bool G> __device__ __forceinline__ Iv P_iv(const CNode& c, const float* t, Iv x, Iv y, Iv z) { if (G) return prism_iv(c, t, x, y, z); return prism_iv_plain(c, t, x, y, z); }

template <bool G> __device__ Iv tape_iv_core(const CNode* nodes, u32 count, const float* tables, Iv x, Iv y, Iv z) {
    Iv f = mkiv(FAR32, FAR32);
    Iv stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u || c.kind == PROFILE_MIRROR) continue;
        if (c.kind == PROFILE_SHELL) { f = iv_max(f, iv_sub_ic(iv_neg(f), c.v[0])); continue; }
        if (c.kind == PROFILE_BODY) { if (depth < MAX_BODIES) { stack[depth] = f; depth += 1; f = mkiv(FAR32, FAR32); } continue; }
        if (c.kind == PROFILE_COMBINE) {
            if (depth > 0) {
                depth -= 1;
                Iv t = stack[depth];
                f = (c.op == FOP_ADD) ? iv_min(t, f) : ((c.op == FOP_REMOVE) ? iv_max(t, iv_neg(f)) : iv_max(t, f));
            }
            continue;
        }
        const float* t = tables + (u64)i * TABLE;
        Iv tf = mkiv(FAR32, FAR32);
        for (u32 k = 0; k < c.count; ++k) {
            float co = t[4 * k], s = t[4 * k + 1], tx = t[4 * k + 2], ty = t[4 * k + 3];
            Iv qx = iv_sub_ic(x, tx);
            Iv qy = iv_sub_ic(y, ty);
            Iv px = iv_add(iv_scale(qx, co), iv_scale(qy, s));
            Iv py = iv_sub(iv_scale(qy, co), iv_scale(qx, s));
            tf = iv_min(tf, P_iv<G>(c, t, px, py, z));
        }
        f = (c.op == FOP_ADD) ? iv_min(f, tf) : iv_max(f, iv_neg(tf));
    }
    return f;
}
__device__ __noinline__ Iv tape_iv_mirror(const CNode* nodes, u32 count, const float* tables, u32 m, Iv x, Iv y, Iv z) {
    Iv f = mkiv(FAR32, FAR32);
    for (u32 k = 0; k < 4u; ++k) {
        if ((k & ~m) != 0u) continue;
        f = iv_min(f, tape_iv_core<true>(nodes, count, tables, (k & 1u) ? iv_neg(x) : x, (k & 2u) ? iv_neg(y) : y, z));
    }
    return f;
}
__device__ Iv tape_iv(const CNode* nodes, u32 count, const float* tables, Iv x, Iv y, Iv z) {
    u32 m = mirror_mask(nodes, count);
    if (m == 0u) return tape_iv_core<true>(nodes, count, tables, x, y, z);
    return tape_iv_mirror(nodes, count, tables, m, x, y, z);
}

// ---- per-cell tape pruning (twin of math.rs Prune) ----
#define PRUNE_NODES 32u
struct Prune { u32 skip; u64 inst[PRUNE_NODES]; };
__device__ __forceinline__ void prune_none(Prune& p) { p.skip = 0u; for (u32 i = 0; i < PRUNE_NODES; ++i) p.inst[i] = 0xFFFFFFFFFFFFFFFFull; }
__device__ __forceinline__ bool pr_skipped(const Prune& p, u32 i) { return i < PRUNE_NODES && ((p.skip >> i) & 1u) != 0u; }
__device__ __forceinline__ bool pr_keeps(const Prune& p, u32 i, u32 k) { return i >= PRUNE_NODES || k >= 64u || ((p.inst[i] >> k) & 1ull) != 0ull; }

template <bool G> __device__ __forceinline__ Iv inst_iv(const CNode& c, const float* t, u32 k, Iv x, Iv y, Iv z) {
    float co = t[4 * k], s = t[4 * k + 1], tx = t[4 * k + 2], ty = t[4 * k + 3];
    Iv qx = iv_sub_ic(x, tx);
    Iv qy = iv_sub_ic(y, ty);
    Iv px = iv_add(iv_scale(qx, co), iv_scale(qy, s));
    Iv py = iv_sub(iv_scale(qy, co), iv_scale(qx, s));
    return P_iv<G>(c, t, px, py, z);
}

template <bool G> __device__ Iv tape_iv_prune_core(const CNode* nodes, u32 count, const float* tables, Iv x, Iv y, Iv z, float m, Prune& pr) {
    prune_none(pr);
    Iv f = mkiv(FAR32, FAR32);
    Iv stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u || c.kind == PROFILE_MIRROR) continue;
        if (c.kind == PROFILE_SHELL) { f = iv_max(f, iv_sub_ic(iv_neg(f), c.v[0])); continue; }
        if (c.kind == PROFILE_BODY) { if (depth < MAX_BODIES) { stack[depth] = f; depth += 1; f = mkiv(FAR32, FAR32); } continue; }
        if (c.kind == PROFILE_COMBINE) {
            if (depth > 0) {
                depth -= 1;
                Iv t = stack[depth];
                f = (c.op == FOP_ADD) ? iv_min(t, f) : ((c.op == FOP_REMOVE) ? iv_max(t, iv_neg(f)) : iv_max(t, f));
            }
            continue;
        }
        const float* t = tables + (u64)i * TABLE;
        Iv tf = mkiv(FAR32, FAR32);
        for (u32 k = 0; k < c.count; ++k) tf = iv_min(tf, inst_iv<G>(c, t, k, x, y, z));
        if (i < PRUNE_NODES) {
            if (c.count > 1u && c.count <= 64u) {
                u64 mask = 0ull;
                for (u32 k = 0; k < c.count; ++k) { Iv r = inst_iv<G>(c, t, k, x, y, z); if (!(r.lo > tf.hi + m)) mask |= 1ull << k; }
                pr.inst[i] = mask;
            }
            bool dominated = (c.op == FOP_ADD) ? (tf.lo > f.hi + m) : (-tf.lo < f.lo - m);
            if (dominated) pr.skip |= 1u << i;
        }
        f = (c.op == FOP_ADD) ? iv_min(f, tf) : iv_max(f, iv_neg(tf));
    }
    return f;
}
__device__ Iv tape_iv_prune(const CNode* nodes, u32 count, const float* tables, Iv x, Iv y, Iv z, float m, Prune& pr) {
    u32 mm = mirror_mask(nodes, count);
    if (mm != 0u) { prune_none(pr); return tape_iv_mirror(nodes, count, tables, mm, x, y, z); }
    return tape_iv_prune_core<true>(nodes, count, tables, x, y, z, m, pr);
}

template <bool G> __device__ float tape_sdf_p_core(const CNode* nodes, u32 count, const float* tables, const Prune& pr, float x, float y, float z) {
    float f = FAR32;
    float stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u || c.kind == PROFILE_MIRROR) continue;
        if (c.kind == PROFILE_SHELL) { f = fmaxx(f, -f - c.v[0]); continue; }
        if (c.kind == PROFILE_BODY) { if (depth < MAX_BODIES) { stack[depth] = f; depth += 1; f = FAR32; } continue; }
        if (c.kind == PROFILE_COMBINE) {
            if (depth > 0) {
                depth -= 1;
                float t = stack[depth];
                f = (c.op == FOP_ADD) ? fminn(t, f) : ((c.op == FOP_REMOVE) ? fmaxx(t, -f) : fmaxx(t, f));
            }
            continue;
        }
        if (pr_skipped(pr, i)) continue;
        const float* t = tables + (u64)i * TABLE;
        float tf = FAR32;
        for (u32 k = 0; k < c.count; ++k) {
            if (!pr_keeps(pr, i, k)) continue;
            float co = t[4 * k], s = t[4 * k + 1], tx = t[4 * k + 2], ty = t[4 * k + 3];
            float qx = x - tx;
            float qy = y - ty;
            float px = qx * co + qy * s;
            float py = qy * co - qx * s;
            tf = fminn(tf, P_sdf<G>(c, t, px, py, z));
        }
        f = (c.op == FOP_ADD) ? fminn(f, tf) : fmaxx(f, -tf);
    }
    return f;
}
__device__ float tape_sdf_p(const CNode* nodes, u32 count, const float* tables, const Prune& pr, float x, float y, float z) {
    u32 mm = mirror_mask(nodes, count);
    if (mm != 0u) return tape_sdf_mirror(nodes, count, tables, mm, x, y, z);
    return tape_sdf_p_core<true>(nodes, count, tables, pr, x, y, z);
}

template <bool G> __device__ Iv tape_iv_p_core(const CNode* nodes, u32 count, const float* tables, const Prune& pr, Iv x, Iv y, Iv z) {
    Iv f = mkiv(FAR32, FAR32);
    Iv stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u || c.kind == PROFILE_MIRROR) continue;
        if (c.kind == PROFILE_SHELL) { f = iv_max(f, iv_sub_ic(iv_neg(f), c.v[0])); continue; }
        if (c.kind == PROFILE_BODY) { if (depth < MAX_BODIES) { stack[depth] = f; depth += 1; f = mkiv(FAR32, FAR32); } continue; }
        if (c.kind == PROFILE_COMBINE) {
            if (depth > 0) {
                depth -= 1;
                Iv t = stack[depth];
                f = (c.op == FOP_ADD) ? iv_min(t, f) : ((c.op == FOP_REMOVE) ? iv_max(t, iv_neg(f)) : iv_max(t, f));
            }
            continue;
        }
        if (pr_skipped(pr, i)) continue;
        const float* t = tables + (u64)i * TABLE;
        Iv tf = mkiv(FAR32, FAR32);
        for (u32 k = 0; k < c.count; ++k) { if (pr_keeps(pr, i, k)) tf = iv_min(tf, inst_iv<G>(c, t, k, x, y, z)); }
        f = (c.op == FOP_ADD) ? iv_min(f, tf) : iv_max(f, iv_neg(tf));
    }
    return f;
}
__device__ Iv tape_iv_p(const CNode* nodes, u32 count, const float* tables, const Prune& pr, Iv x, Iv y, Iv z) {
    u32 mm = mirror_mask(nodes, count);
    if (mm != 0u) return tape_iv_mirror(nodes, count, tables, mm, x, y, z);
    return tape_iv_p_core<true>(nodes, count, tables, pr, x, y, z);
}

// ---- adaptive evaluation (twin of reference.rs ag_ad_*) ----
struct Grid { double lo[3], size[3], fine[3], eps; u32 r[3], levels; };

__device__ Grid make_grid(const Goal& goal, u32 r0, u32 levels) {
    Grid gr;
    for (int k = 0; k < 3; ++k) gr.size[k] = goal.domain_hi[k] - goal.domain_lo[k];
    for (int k = 0; k < 3; ++k) gr.r[k] = r0;
    for (int k = 0; k < 3; ++k) gr.fine[k] = gr.size[k] / (double)(gr.r[k] << (levels - 1u));
    double reach = 0.0;
    for (int k = 0; k < 3; ++k) reach = dmax(reach, fabs(goal.domain_lo[k]));
    for (int k = 0; k < 3; ++k) reach = dmax(reach, fabs(goal.domain_hi[k]));
    reach = reach + sqrt(gr.size[0] * gr.size[0] + gr.size[1] * gr.size[1] + gr.size[2] * gr.size[2]);
    gr.eps = 4.0e-6 * reach + 1e-6;
    for (int k = 0; k < 3; ++k) gr.lo[k] = goal.domain_lo[k];
    gr.levels = levels;
    return gr;
}

__device__ void cell_box(const Grid& gr, u32 level, u32 ix, u32 iy, u32 iz, Iv* b) {
    u32 idx[3] = {ix, iy, iz};
    for (int k = 0; k < 3; ++k) {
        double w = gr.size[k] / (double)(gr.r[k] << level);
        b[k] = mkiv((float)(gr.lo[k] + (double)idx[k] * w), (float)(gr.lo[k] + (double)(idx[k] + 1u) * w));
    }
}

// Mirror dispatch hoisted out of the evaluators: M = false compiles the plain tape (the hot
// path carries no mirror code); M = true handles mirrored tapes.
template <bool M> __device__ __forceinline__ float T_sdf(const CNode* n, u32 c, const float* t, float x, float y, float z) { if (M) return tape_sdf_c(n, c, t, x, y, z); return tape_sdf_c_core<false>(n, c, t, x, y, z); }
template <bool M> __device__ __forceinline__ Iv T_iv(const CNode* n, u32 c, const float* t, Iv x, Iv y, Iv z) { if (M) return tape_iv(n, c, t, x, y, z); return tape_iv_core<false>(n, c, t, x, y, z); }
template <bool M> __device__ __forceinline__ Iv T_iv_prune(const CNode* n, u32 c, const float* t, Iv x, Iv y, Iv z, float m, Prune& pr) { if (M) return tape_iv_prune(n, c, t, x, y, z, m, pr); return tape_iv_prune_core<false>(n, c, t, x, y, z, m, pr); }
template <bool M> __device__ __forceinline__ float T_sdf_p(const CNode* n, u32 c, const float* t, const Prune& pr, float x, float y, float z) { if (M) return tape_sdf_p(n, c, t, pr, x, y, z); return tape_sdf_p_core<false>(n, c, t, pr, x, y, z); }
template <bool M> __device__ __forceinline__ Iv T_iv_p(const CNode* n, u32 c, const float* t, const Prune& pr, Iv x, Iv y, Iv z) { if (M) return tape_iv_p(n, c, t, pr, x, y, z); return tape_iv_p_core<false>(n, c, t, pr, x, y, z); }

struct Surface { float f, g[3], gl, w; };

template <bool M> __device__ Surface surface_p(const CNode* nodes, u32 cnt, const float* tabs, const Prune& pr, float x, float y, float z, const float* d) {
    Surface s;
    s.f = T_sdf_p<M>(nodes, cnt, tabs, pr, x, y, z);
    float h = 0.25f * fminn(fminn(d[0], d[1]), d[2]);
    float gx = T_sdf_p<M>(nodes, cnt, tabs, pr, x + h, y, z) - T_sdf_p<M>(nodes, cnt, tabs, pr, x - h, y, z);
    float gy = T_sdf_p<M>(nodes, cnt, tabs, pr, x, y + h, z) - T_sdf_p<M>(nodes, cnt, tabs, pr, x, y - h, z);
    float gz = T_sdf_p<M>(nodes, cnt, tabs, pr, x, y, z + h) - T_sdf_p<M>(nodes, cnt, tabs, pr, x, y, z - h);
    s.g[0] = gx; s.g[1] = gy; s.g[2] = gz;
    s.gl = sqrtf(gx * gx + gy * gy + gz * gz);
    s.w = (s.gl > 1e-30f) ? (fabsf(gx) * d[0] + fabsf(gy) * d[1] + fabsf(gz) * d[2]) / s.gl : 0.0f;
    return s;
}

__device__ Surface surface(const CNode* nodes, u32 cnt, const float* tabs, float x, float y, float z, const float* d) {
    Prune pr; prune_none(pr);
    return surface_p<true>(nodes, cnt, tabs, pr, x, y, z, d);
}

__device__ u64 fill_of(const Surface& s) {
    float frac = 0.5f;
    if (s.gl > 1e-30f) {
        frac = 0.5f - s.f / s.w;
        if (frac < 0.0f) frac = 0.0f;
        if (frac > 1.0f) frac = 1.0f;
    }
    return (u64)floor((double)frac * 1048576.0 + 0.5);
}

__device__ u64 fill_q(const CNode* nodes, u32 cnt, const float* tabs, float x, float y, float z, const float* d) { return fill_of(surface(nodes, cnt, tabs, x, y, z, d)); }

__device__ __forceinline__ u64 qarea(float a) { return (u64)floor((double)a * 1024.0 + 0.5); }

template <bool M> __device__ float march(const CNode* nodes, u32 cnt, const float* tabs, const float* p, const float* dir, bool inside, float t0, float tmax, float hmin) {
    float t = t0;
    for (u32 i = 0; i < MARCH_STEPS; ++i) {
        if (t >= tmax) return tmax;
        float f = T_sdf<M>(nodes, cnt, tabs, p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t);
        float s = inside ? -f : f;
        if (s <= 0.0f) return t;
        t = t + ((s > hmin) ? s : hmin);
    }
    return tmax;
}

struct Probe { float area, wall, gap; bool thin, narrow, undercut; };

template <bool M> __device__ Probe probe(const CNode* nodes, u32 cnt, const float* tabs, const Goal& goal, const Surface& s, const float* c, const float* d) {
    Probe pr; pr.area = 0.0f; pr.wall = FAR32; pr.gap = FAR32; pr.thin = false; pr.narrow = false; pr.undercut = false;
    if (!(s.gl > 1e-30f) || !(fabsf(s.f) < 0.5f * s.w)) return pr;
    pr.area = d[0] * d[1] * d[2] / s.w;
    float n[3] = {s.g[0] / s.gl, s.g[1] / s.gl, s.g[2] / s.gl};
    if (!(goal.min_wall > 0.0) && !(goal.min_gap > 0.0) && goal.machining_3ax == 0u) return pr;
    float p[3] = {c[0] - n[0] * s.f, c[1] - n[1] * s.f, c[2] - n[2] * s.f};
    float dm = fminn(fminn(d[0], d[1]), d[2]);
    float hmin = 0.1f * dm;
    if (!(fabsf(T_sdf<M>(nodes, cnt, tabs, p[0], p[1], p[2])) <= hmin)) return pr;
    float mh = -hmin;
    float fin = T_sdf<M>(nodes, cnt, tabs, p[0] + n[0] * mh, p[1] + n[1] * mh, p[2] + n[2] * mh);
    float fout = T_sdf<M>(nodes, cnt, tabs, p[0] + n[0] * hmin, p[1] + n[1] * hmin, p[2] + n[2] * hmin);
    if (!(fin < 0.0f && fout > 0.0f)) return pr;
    if (goal.min_wall > 0.0) {
        float tw = (float)goal.min_wall;
        float nd[3] = {-n[0], -n[1], -n[2]};
        pr.wall = march<M>(nodes, cnt, tabs, p, nd, true, hmin, tw, hmin);
        pr.thin = pr.wall < tw;
    }
    if (goal.min_gap > 0.0) {
        float tg = (float)goal.min_gap;
        pr.gap = march<M>(nodes, cnt, tabs, p, n, false, hmin, tg, hmin);
        pr.narrow = pr.gap < tg;
    }
    if (goal.machining_3ax != 0u) {
        float q[3] = {p[0] + n[0] * dm, p[1] + n[1] * dm, p[2] + n[2] * dm};
        bool up = n[2] >= -0.25f;
        float dir[3] = {0.0f, 0.0f, up ? 1.0f : -1.0f};
        float tmax = up ? (float)goal.domain_hi[2] - q[2] : q[2] - (float)goal.domain_lo[2];
        if (tmax > 0.0f && T_sdf<M>(nodes, cnt, tabs, q[0], q[1], q[2]) > 0.0f) pr.undercut = march<M>(nodes, cnt, tabs, q, dir, false, 0.0f, tmax, 0.5f * dm) < tmax;
    }
    return pr;
}

template <bool M> __device__ void tighten_cell(const CNode* nodes, u32 cnt, const float* tabs, const Prune& pr, const Iv* b, const float* g, float e, u32 budget, u64* olo, u64* ohi) {
    u64 full = 1ull << TQ;
    *olo = 0ull; *ohi = full;
    if (budget < 2u) return;
    double cd[3] = {(double)(b[0].hi - b[0].lo), (double)(b[1].hi - b[1].lo), (double)(b[2].hi - b[2].lo)};
    double cv = cd[0] * cd[1] * cd[2];
    if (!(cv > 0.0)) return;
    Iv bx[TK][3];
    double vol[TK];
    bx[0][0] = b[0]; bx[0][1] = b[1]; bx[0][2] = b[2];
    vol[0] = (double)full;
    u32 n = 1u;
    u64 lo = 0ull, hi = 0ull;
    u32 evals = 0u;
    while (n > 0u && evals + 2u <= budget && n < (u32)TK) {
        u32 k = 0u;
        for (u32 i = 1u; i < n; ++i) if (vol[i] > vol[k]) k = i;
        Iv pb[3] = {bx[k][0], bx[k][1], bx[k][2]};
        int ax = 0;
        float best = -1.0f;
        for (int j = 0; j < 3; ++j) { float v = fabsf(g[j]) * (pb[j].hi - pb[j].lo); if (v > best) { best = v; ax = j; } }
        float mid = 0.5f * (pb[ax].lo + pb[ax].hi);
        if (!(mid > pb[ax].lo && mid < pb[ax].hi) || (double)(pb[ax].hi - pb[ax].lo) < 1e-4 * cd[ax]) break;
        n -= 1u;
        bx[k][0] = bx[n][0]; bx[k][1] = bx[n][1]; bx[k][2] = bx[n][2];
        vol[k] = vol[n];
        for (int half = 0; half < 2; ++half) {
            Iv c[3] = {pb[0], pb[1], pb[2]};
            if (half == 0) c[ax].hi = mid; else c[ax].lo = mid;
            evals += 1u;
            Iv r = T_iv_p<M>(nodes, cnt, tabs, pr, c[0], c[1], c[2]);
            if (r.lo - e > 0.0f) continue;
            double v = (double)(c[0].hi - c[0].lo) * (double)(c[1].hi - c[1].lo) * (double)(c[2].hi - c[2].lo) / cv * (double)full;
            if (r.hi + e < 0.0f) { lo += (u64)floor(v); hi += (u64)ceil(v); continue; }
            bx[n][0] = c[0]; bx[n][1] = c[1]; bx[n][2] = c[2];
            vol[n] = v;
            n += 1u;
        }
    }
    for (u32 i = 0u; i < n; ++i) hi += (u64)ceil(vol[i]);
    *olo = lo; *ohi = (hi < full) ? hi : full;
}



__device__ __forceinline__ u64 enc(u32 cfg, u32 ix, u32 iy, u32 iz) { return ((u64)cfg << 32) | ((u64)ix << 20) | ((u64)iy << 10) | (u64)iz; }

template <bool M> __device__ bool border_hit(const Grid& gr, const CNode* nodes, u32 cnt, const float* tabs, u32 level, const u32* idx, const Iv* b, bool inside) {
    for (int ax = 0; ax < 3; ++ax) {
        u32 res = gr.r[ax] << level;
        for (u32 side = 0; side < 2u; ++side) {
            if ((side == 0u && idx[ax] != 0u) || (side == 1u && idx[ax] != res - 1u)) continue;
            if (inside) return true;
            Iv f[3] = {b[0], b[1], b[2]};
            float face = (side == 0u) ? b[ax].lo : b[ax].hi;
            f[ax] = mkiv(face, face);
            Iv r = T_iv<M>(nodes, cnt, tabs, f[0], f[1], f[2]);
            if (r.lo - (float)gr.eps <= 0.0f) return true;
        }
    }
    return false;
}


// Per-thread accumulator contributions, flushed with warp aggregation: when the full warp
// works on one configuration (worklists are sorted by configuration), integer butterfly
// reductions and one atomic per slot; otherwise per-lane atomics. Integer sums and min/max
// are associative, so the result is identical either way (and to the CPU reference).
struct Contrib { u64 v[CS]; u32 bb[12]; u32 dom; bool has; };
__device__ __forceinline__ void contrib_init(Contrib& c) {
    for (int j = 0; j < CS; ++j) c.v[j] = 0ull;
    c.v[C_MIN_WALL] = 0xFFFFFFFFFFFFFFFFull; c.v[C_MIN_GAP] = 0xFFFFFFFFFFFFFFFFull;
    for (int k = 0; k < 3; ++k) { c.bb[k] = 0xFFFFFFFFu; c.bb[3 + k] = 0u; c.bb[6 + k] = 0xFFFFFFFFu; c.bb[9 + k] = 0u; }
    c.dom = 0u;
    c.has = false;
}
__device__ __forceinline__ void contrib_bbox(Contrib& c, const u32* lo, u32 n, bool inside) {
    for (int k = 0; k < 3; ++k) {
        if (inside) { c.bb[k] = min(c.bb[k], lo[k]); c.bb[3 + k] = max(c.bb[3 + k], lo[k] + n); }
        c.bb[6 + k] = min(c.bb[6 + k], lo[k]); c.bb[9 + k] = max(c.bb[9 + k], lo[k] + n);
    }
}
__device__ __forceinline__ bool is_min_slot(int j) { return j == C_MIN_WALL || j == C_MIN_GAP; }
__device__ void contrib_flush(const DopsArgs* a, u32 cfg, Contrib& c) {
    u64* cc = ag_ptr<u64>(a->counts) + (u64)CS * cfg;
    u32* bb = ag_ptr<u32>(a->bbox) + 12u * cfg;
    unsigned act = __activemask();
    if (__ballot_sync(act, c.has) == 0u) return; // nothing to add in this warp
    unsigned grp = __match_any_sync(act, cfg);
    if (act == 0xFFFFFFFFu && grp == act) {
        // (lanes without contributions hold identities)
        for (int j = 0; j < CS; ++j) {
            u64 x = c.v[j];
            for (int o = 16; o > 0; o >>= 1) { u64 y = __shfl_xor_sync(0xFFFFFFFFu, x, o); x = is_min_slot(j) ? (y < x ? y : x) : x + y; }
            c.v[j] = x;
        }
        for (int k = 0; k < 12; ++k) {
            u32 x = c.bb[k];
            bool mx = (k % 6) >= 3;
            for (int o = 16; o > 0; o >>= 1) { u32 y = __shfl_xor_sync(0xFFFFFFFFu, x, o); x = mx ? max(x, y) : min(x, y); }
            c.bb[k] = x;
        }
        u32 d = __reduce_or_sync(0xFFFFFFFFu, c.dom);
        if ((threadIdx.x & 31u) != 0u) return;
        c.dom = d;
        c.has = true; // lane 0 carries the warp's totals
    }
    if (!c.has) return;
    for (int j = 0; j < CS; ++j) {
        if (is_min_slot(j)) { if (c.v[j] != 0xFFFFFFFFFFFFFFFFull) atomicMin((unsigned long long*)&cc[j], (unsigned long long)c.v[j]); }
        else if (c.v[j] != 0ull) atomicAdd((unsigned long long*)&cc[j], (unsigned long long)c.v[j]);
    }
    for (int k = 0; k < 12; ++k) {
        bool mx = (k % 6) >= 3;
        if (mx) { if (c.bb[k] != 0u) atomicMax(bb + k, c.bb[k]); }
        else if (c.bb[k] != 0xFFFFFFFFu) atomicMin(bb + k, c.bb[k]);
    }
    if (c.dom != 0u) atomicExch(ag_ptr<u32>(a->domain_hit) + cfg, 1u);
}

__device__ void contrib_moments(Contrib& ct, u32 ix, u32 iy, u32 iz, u32 sh, u64 q20) {
    u64 w = 1ull << (3u * sh);
    u64 q6 = q20 >> 14;
    u64 ss = 4ull << (2u * sh);
    u32 idx[3] = {ix, iy, iz};
    for (int k = 0; k < 3; ++k) {
        u64 c2 = ((u64)(2u * idx[k] + 1u)) << sh;
        ct.v[C_M1 + k] += q20 * w * c2;
        ct.v[C_M2 + k] += q6 * w * (12ull * c2 * c2 + ss);
    }
    ct.v[C_V6] += q6 * w;
}

__device__ void contrib_probe(Contrib& ct, const Probe& p) {
    u64 aq = qarea(p.area);
    ct.v[C_AREA] += aq;
    if (p.thin) ct.v[C_THIN] += aq;
    if (p.narrow) ct.v[C_GAP] += aq;
    if (p.undercut) ct.v[C_UNDERCUT] += aq;
    if (p.wall < FAR32) { u64 b = (u64)__float_as_uint(p.wall); if (b < ct.v[C_MIN_WALL]) ct.v[C_MIN_WALL] = b; }
    if (p.gap < FAR32) { u64 b = (u64)__float_as_uint(p.gap); if (b < ct.v[C_MIN_GAP]) ct.v[C_MIN_GAP] = b; }
}

#define GRID_STRIDE(i, n) for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (u64)(n); i += (u64)gridDim.x * blockDim.x)

AG_KERNEL void ag_ad_seed(const DopsArgs* a) {
    const Goal goal = *ag_ptr<const Goal>(a->goal);
    Grid gr = make_grid(goal, a->coarse, a->levels);
    u32 per = gr.r[0] * gr.r[1] * gr.r[2];
    u64 n = (u64)a->configs * per;
    u64* wl = ag_ptr<u64>(a->wl0);
    GRID_STRIDE(i, n) {
        u32 cfg = (u32)(i / per), cell = (u32)(i % per);
        wl[i] = enc(cfg, cell % gr.r[0], (cell / gr.r[0]) % gr.r[1], cell / (gr.r[0] * gr.r[1]));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ag_ptr<u32>(a->lvl)[0] = (u32)n;
}

AG_KERNEL __launch_bounds__(256, 3) void ag_ad_classify(const DopsArgs* a) {
    u32 nn = a->nodes;
    const u32* cnt = ag_ptr<const u32>(a->eval_trial ? a->tr_count : a->cur_count);
    const CNode* cnodes = ag_ptr<const CNode>(a->compiled);
    const float* tabs = ag_ptr<const float>(a->tables);
    const u32* done = ag_ptr<const u32>(a->done);
    const Goal goal = *ag_ptr<const Goal>(a->goal);
    Grid gr = make_grid(goal, a->coarse, a->levels);
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    u32* cls = ag_ptr<u32>(a->cls);
    u32 sh = a->levels - 1u - l;
    float e = (float)gr.eps;
    GRID_STRIDE(j, a->capacity) { if (j >= (u64)n) cls[j] = 0u; }
    GRID_STRIDE(i, n) {
        u64 en = wl[i];
        u32 cfg = (u32)(en >> 32), ix = (u32)((en >> 20) & 1023u), iy = (u32)((en >> 10) & 1023u), iz = (u32)(en & 1023u);
        cls[i] = 0u;
        Contrib ct;
        contrib_init(ct);
        if (!(a->eval_trial != 0u && done[cfg] != 0u)) {
            const CNode* tape = cnodes + (u64)cfg * nn;
            const float* tb = tabs + (u64)cfg * nn * TABLE;
            u32 c = cnt[cfg];
            Iv b[3];
            cell_box(gr, l, ix, iy, iz, b);
            bool mir = tape_general(tape, c);
            Iv r = mir ? T_iv<true>(tape, c, tb, b[0], b[1], b[2]) : T_iv<false>(tape, c, tb, b[0], b[1], b[2]);
            if (!(r.lo - e > 0.0f)) {
                if (r.hi + e < 0.0f) {
                    ct.has = true;
                    ct.v[C_INSIDE] = 1ull << (3u * sh);
                    contrib_moments(ct, ix, iy, iz, sh, 1ull << 20);
                    u32 lo[3] = {ix << sh, iy << sh, iz << sh};
                    contrib_bbox(ct, lo, 1u << sh, true);
                    u32 idx[3] = {ix, iy, iz};
                    if (mir ? border_hit<true>(gr, tape, c, tb, l, idx, b, true) : border_hit<false>(gr, tape, c, tb, l, idx, b, true)) ct.dom = 1u;
                } else {
                    cls[i] = 1u;
                }
            }
        }
        contrib_flush(a, cfg, ct);
    }
}

AG_KERNEL void ag_ad_resolve(const DopsArgs* a) {
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    const u32* cls = ag_ptr<const u32>(a->cls);
    u32* flag = ag_ptr<u32>(a->flag);
    u32* term = ag_ptr<u32>(a->term);
    const u32* crank = ag_ptr<const u32>(a->crank);
    const u32* start = ag_ptr<const u32>(a->start);
    GRID_STRIDE(i, a->capacity) {
        u32 f = 0u, t = 0u;
        if (i < (u64)n && cls[i] == 1u) {
            u32 cfg = (u32)(wl[i] >> 32);
            u32 rank = crank[i] - crank[start[cfg]];
            if (l + 1u < a->levels && ((u64)rank + 1ull) * 8ull <= (u64)a->cap) f = 1u; else t = 1u;
        }
        flag[i] = f;
        term[i] = t;
    }
}

AG_KERNEL void ag_ad_clip(const DopsArgs* a) {
    u32 cap = a->capacity;
    u32* flag = ag_ptr<u32>(a->flag);
    u32* term = ag_ptr<u32>(a->term);
    const u32* offs = ag_ptr<const u32>(a->offs);
    u32 keep = cap / 8u;
    GRID_STRIDE(i, cap) {
        u32 f = flag[i];
        if (i == (u64)cap - 1ull) { u32 total = offs[i] + f; ag_ptr<u32>(a->lvl)[a->level + 1u] = ((total < keep) ? total : keep) * 8u; }
        if (f != 0u && offs[i] >= keep) { flag[i] = 0u; term[i] = 1u; }
    }
}

AG_KERNEL void ag_ad_compact(const DopsArgs* a) {
    u32 cap = a->capacity;
    const u32* term = ag_ptr<const u32>(a->term);
    const u32* toffs = ag_ptr<const u32>(a->toffs);
    u32* tlist = ag_ptr<u32>(a->tlist);
    GRID_STRIDE(i, cap) {
        if (term[i] != 0u) tlist[toffs[i]] = (u32)i;
        if (i == (u64)cap - 1ull) ag_ptr<u32>(a->tcount)[0] = toffs[i] + term[i];
    }
}

// One terminal boundary cell (body of ag_ad_fill), compiled with (M) or without mirror support.
template <bool M> __device__ __forceinline__ void fill_cell(const DopsArgs* a, const Goal& goal, const Grid& gr, const CNode* tape, u32 c, const float* tb, u32 l, u32 sh, u32 ix, u32 iy, u32 iz, const Iv* b, u64 w, const float* d, const float* cxyz, Contrib& ct) {
    Prune pru;
    (void)T_iv_prune<M>(tape, c, tb, b[0], b[1], b[2], (float)gr.eps, pru);
    Surface sf = surface_p<M>(tape, c, tb, pru, cxyz[0], cxyz[1], cxyz[2], d);
    u64 tlo, thi;
    tighten_cell<M>(tape, c, tb, pru, b, sf.g, (float)gr.eps, a->tighten, &tlo, &thi);
    ct.v[C_BLO] = tlo * w;
    ct.v[C_BHI] = thi * w;
    u64 fq = fill_of(sf);
    if (fq < (tlo >> 4)) fq = tlo >> 4;
    if (fq > ((thi + 15ull) >> 4)) fq = (thi + 15ull) >> 4;
    ct.v[C_FILL] = fq * w;
    contrib_moments(ct, ix, iy, iz, sh, fq);
    Probe pr = probe<M>(tape, c, tb, goal, sf, cxyz, d);
    contrib_probe(ct, pr);
    u32 lo[3] = {ix << sh, iy << sh, iz << sh};
    contrib_bbox(ct, lo, 1u << sh, false);
    u32 idx[3] = {ix, iy, iz};
    if (border_hit<M>(gr, tape, c, tb, l, idx, b, false)) ct.dom = 1u;
}

template <bool MIR> __device__ __forceinline__ void fill_kernel(const DopsArgs* a) {
    u32 nn = a->nodes;
    const u32* cnt = ag_ptr<const u32>(a->eval_trial ? a->tr_count : a->cur_count);
    const CNode* cnodes = ag_ptr<const CNode>(a->compiled);
    const float* tabs = ag_ptr<const float>(a->tables);
    const Goal goal = *ag_ptr<const Goal>(a->goal);
    Grid gr = make_grid(goal, a->coarse, a->levels);
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    const u32* tlist = ag_ptr<const u32>(a->tlist);
    u32 tn = ag_ptr<const u32>(a->tcount)[0];
    u32 sh = a->levels - 1u - l;
    (void)n;
    GRID_STRIDE(j, tn) {
        u64 en = wl[tlist[j]];
        u32 cfg = (u32)(en >> 32), ix = (u32)((en >> 20) & 1023u), iy = (u32)((en >> 10) & 1023u), iz = (u32)(en & 1023u);
        Contrib ct;
        contrib_init(ct);
        if (tape_general(cnodes + (u64)cfg * nn, cnt[cfg]) != MIR) continue; // the other launch
        {
            {
                const CNode* tape = cnodes + (u64)cfg * nn;
                const float* tb = tabs + (u64)cfg * nn * TABLE;
                u32 c = cnt[cfg];
                Iv b[3];
                cell_box(gr, l, ix, iy, iz, b);
                u64 w = 1ull << (3u * sh);
                ct.has = true;
                ct.v[C_BOUNDARY] = w;
                float d[3] = {b[0].hi - b[0].lo, b[1].hi - b[1].lo, b[2].hi - b[2].lo};
                float cxyz[3] = {0.5f * (b[0].lo + b[0].hi), 0.5f * (b[1].lo + b[1].hi), 0.5f * (b[2].lo + b[2].hi)};
                fill_cell<MIR>(a, goal, gr, tape, c, tb, l, sh, ix, iy, iz, b, w, d, cxyz, ct);
            }
        }
        contrib_flush(a, cfg, ct);
    }
}

// Terminal boundary cells: plain tapes here, mirrored tapes in ag_ad_fill_mirror (separate
// launches keep mirror code out of the hot kernel; the CPU reference does both in one).
AG_KERNEL __launch_bounds__(256, 3) void ag_ad_fill(const DopsArgs* a) { fill_kernel<false>(a); }
AG_KERNEL __launch_bounds__(256, 3) void ag_ad_fill_mirror(const DopsArgs* a) { fill_kernel<true>(a); }

AG_KERNEL void ag_ad_start(const DopsArgs* a) {
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    u32* start = ag_ptr<u32>(a->start);
    GRID_STRIDE(i, n) {
        u32 c = (u32)(wl[i] >> 32);
        if (i == 0 || (u32)(wl[i - 1] >> 32) != c) start[c] = (u32)i;
    }
}

AG_KERNEL void ag_ad_emit(const DopsArgs* a) {
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    u64* out = ag_ptr<u64>((l % 2u == 0u) ? a->wl1 : a->wl0);
    const u32* flag = ag_ptr<const u32>(a->flag);
    const u32* offs = ag_ptr<const u32>(a->offs);
    u32 m = (n < a->capacity) ? n : a->capacity;
    GRID_STRIDE(i, m) {
        if (flag[i] == 0u) continue;
        u64 en = wl[i];
        u32 cfg = (u32)(en >> 32), ix = (u32)((en >> 20) & 1023u), iy = (u32)((en >> 10) & 1023u), iz = (u32)(en & 1023u);
        u64 base = (u64)offs[i] * 8ull;
        for (u32 k = 0; k < 8u; ++k) out[base + k] = enc(cfg, 2u * ix + (k & 1u), 2u * iy + ((k >> 1) & 1u), 2u * iz + (k >> 2));
    }
}

AG_KERNEL void ag_ad_total(const DopsArgs* a) {
    if (AG_TID != 0) return;
    u32 cap = a->capacity;
    ag_ptr<u32>(a->lvl)[a->level + 1u] = (ag_ptr<const u32>(a->offs)[cap - 1u] + ag_ptr<const u32>(a->flag)[cap - 1u]) * 8u;
}

// ---- engineering metrics (twin of reference.rs engineering) ----
#define OB_CENTROID 13
#define OB_INERTIA 16
#define OB_AREA 19
#define OB_THIN 20
#define OB_MIN_WALL 21
#define OB_GAP 22
#define OB_MIN_GAP 23
#define OB_UNDERCUT 24
#define OB_VIOLATION 25
#define OB_CENTROID_OK 26
struct Eng { double centroid[3], inertia[3], area, thin, min_wall, gap, min_gap, undercut, violation; bool centroid_ok; };

__device__ double min_bits(u64 v) { return (v == 0xFFFFFFFFFFFFFFFFull) ? -1.0 : (double)__uint_as_float((u32)v); }

__device__ Eng engineering(const Grid& gr, const Goal& goal, const u64* cc, double ve) {
    Eng e;
    double qa = 1.0 / 1024.0;
    double v20 = (double)cc[C_INSIDE] * 1048576.0 + (double)cc[C_FILL];
    double v6 = (double)cc[C_V6];
    double var[3];
    e.centroid_ok = true;
    for (int k = 0; k < 3; ++k) {
        double half = 0.5 * gr.fine[k];
        double m1 = (v20 > 0.0) ? (double)cc[C_M1 + k] / v20 : 0.0;
        double m2 = (v6 > 0.0) ? (double)cc[C_M2 + k] / (12.0 * v6) : 0.0;
        e.centroid[k] = gr.lo[k] + m1 * half;
        double vv = m2 - m1 * m1;
        var[k] = (vv > 0.0) ? vv * half * half : 0.0;
        if (goal.centroid_hi[k] > goal.centroid_lo[k] && !(e.centroid[k] >= goal.centroid_lo[k] && e.centroid[k] <= goal.centroid_hi[k])) e.centroid_ok = false;
    }
    e.inertia[0] = ve * (var[1] + var[2]);
    e.inertia[1] = ve * (var[0] + var[2]);
    e.inertia[2] = ve * (var[0] + var[1]);
    e.thin = (double)cc[C_THIN] * qa;
    e.gap = (double)cc[C_GAP] * qa;
    e.undercut = (double)cc[C_UNDERCUT] * qa;
    double over = e.undercut - goal.undercut_tol;
    e.violation = e.thin + e.gap + ((goal.machining_3ax != 0u && over > 0.0) ? over : 0.0);
    e.area = (double)cc[C_AREA] * qa;
    e.min_wall = (goal.min_wall > 0.0) ? min_bits(cc[C_MIN_WALL]) : -1.0;
    e.min_gap = (goal.min_gap > 0.0) ? min_bits(cc[C_MIN_GAP]) : -1.0;
    return e;
}

__device__ void eng_write(const Eng& e, double* o) {
    for (int k = 0; k < 3; ++k) { o[OB_CENTROID + k] = e.centroid[k]; o[OB_INERTIA + k] = e.inertia[k]; }
    o[OB_AREA] = e.area;
    o[OB_THIN] = e.thin;
    o[OB_MIN_WALL] = e.min_wall;
    o[OB_GAP] = e.gap;
    o[OB_MIN_GAP] = e.min_gap;
    o[OB_UNDERCUT] = e.undercut;
    o[OB_VIOLATION] = e.violation;
    o[OB_CENTROID_OK] = e.centroid_ok ? 1.0 : 0.0;
}

struct Judge { bool met; double err, sc; };
__device__ Judge judge(const Goal& goal, bool empty, bool dom, double ve, const double* eh, const Eng& en) {
    Judge j;
    bool has_iv = goal.vol_hi > goal.vol_lo;
    double t = 0.5 * (goal.vol_lo + goal.vol_hi);
    j.err = has_iv ? (ve - t) / t : 0.0;
    bool in_iv = !has_iv || (ve >= goal.vol_lo && ve <= goal.vol_hi);
    j.met = !empty && !dom && in_iv && eh[0] <= goal.max_ext[0] && eh[1] <= goal.max_ext[1] && eh[2] <= goal.max_ext[2] && en.violation == 0.0 && en.centroid_ok;
    double pen = ((en.area > 0.0) ? en.violation / en.area : 0.0) + (en.centroid_ok ? 0.0 : 1.0);
    double scale = (goal.obj_scale > 0.0) ? goal.obj_scale : 1.0;
    j.sc = !j.met ? 1000.0 + fabs(j.err) + pen : (goal.objective == 1u ? ve / scale : (goal.objective == 2u ? -(ve / scale) : fabs(j.err)));
    return j;
}

// ---- check + decide / observe ----
AG_KERNEL void ag_dop_decide(const DopsArgs* a) {
    u32 c = AG_TID;
    if (c >= a->configs) return;
    u32* done = ag_ptr<u32>(a->done);
    if (a->eval_trial != 0u && done[c] != 0u) return;
    u32 nn = a->nodes, kn = a->params, rn = a->records;
    const Goal goal = *ag_ptr<const Goal>(a->goal);
    Grid gr = make_grid(goal, a->coarse, a->levels);
    const u64* counts = ag_ptr<const u64>(a->counts) + (u64)CS * c;
    const u32* bb = ag_ptr<const u32>(a->bbox) + 12u * c;
    u32 dom = ag_ptr<const u32>(a->domain_hit)[c];
    double vc = gr.fine[0] * gr.fine[1] * gr.fine[2];
    double tq = 1.0 / 16777216.0;
    double vl = ((double)counts[0] + (double)counts[C_BLO] * tq) * vc;
    double vh = ((double)counts[0] + (double)counts[C_BHI] * tq) * vc;
    double ve = ((double)counts[0] + (double)counts[2] * (1.0 / 1048576.0)) * vc;
    double el[3], eh[3];
    for (int k = 0; k < 3; ++k) {
        el[k] = 0.0; eh[k] = 0.0;
        if (bb[3 + k] > bb[k]) el[k] = (double)(bb[3 + k] - bb[k]) * gr.fine[k];
        if (bb[9 + k] > bb[6 + k]) eh[k] = (double)(bb[9 + k] - bb[6 + k]) * gr.fine[k];
    }
    Eng en = engineering(gr, goal, counts, ve);
    bool cma = a->policy == POLICY_CMAES;
    bool empty = (counts[0] + counts[1]) == 0ull;
    bool hard = el[0] > goal.max_ext[0] || el[1] > goal.max_ext[1] || el[2] > goal.max_ext[2];
    double* obs = ag_ptr<double>(a->obs) + (u64)c * OBS;
    u32* cur_c = ag_ptr<u32>(a->cur_count);
    Judge jd = judge(goal, empty, dom != 0u, ve, eh, en);
    double* score = ag_ptr<double>(a->score);
    bool write = (a->eval_trial == 0u);
    if (a->eval_trial != 0u) {
        const CommandRecord* cmds = ag_ptr<const CommandRecord>(a->cmds) + (u64)c * rn;
        StatusRecord* status = ag_ptr<StatusRecord>(a->status) + (u64)c * rn;
        u32 acc = 0, rej = 0;
        for (u32 r = 0; r < rn; ++r) {
            if ((u32)cmds[r].opcode == a->ops.update) continue;
            if (status[r].status == AG_ACCEPTED) acc += 1u; else rej += 1u;
        }
        bool worse = !cma && en.violation > obs[OB_VIOLATION];
        bool geometry_ok = !empty && dom == 0u && !hard && !worse;
        if (cma) ag_ptr<double>(a->tscore)[c] = geometry_ok ? jd.sc : 1.0e30;
        bool commit = geometry_ok && acc > 0u && (!cma || jd.sc < score[c]);
        if (!geometry_ok && acc > 0u) {
            unsigned short code = empty ? AG_TOPOLOGY_FAILED : (dom != 0u ? AG_CAPACITY_EXCEEDED : AG_REQUIREMENT_FAILED);
            for (u32 r = 0; r < rn; ++r) if ((u32)cmds[r].opcode != a->ops.update && status[r].status == AG_ACCEPTED) status[r].status = code;
        }
        if (commit) {
            Node* cur_n = ag_ptr<Node>(a->cur_nodes) + (u64)c * nn;
            const Node* tr_n = ag_ptr<const Node>(a->tr_nodes) + (u64)c * nn;
            for (u32 i = 0; i < nn; ++i) cur_n[i] = tr_n[i];
            double* cur_p = ag_ptr<double>(a->cur_params) + (u64)c * kn;
            const double* tr_p = ag_ptr<const double>(a->tr_params) + (u64)c * kn;
            for (u32 k = 0; k < kn; ++k) cur_p[k] = tr_p[k];
            cur_c[c] = ag_ptr<const u32>(a->tr_count)[c];
            u32* rev = ag_ptr<u32>(a->rev);
            rev[c] = rev[c] + 1u;
            u32* budget = ag_ptr<u32>(a->budget);
            budget[c] = (budget[c] > acc) ? budget[c] - acc : 0u;
            write = true;
        }
        double ol = write ? vl : obs[0];
        double oh = write ? vh : obs[1];
        u32 round = ag_ptr<const u32>(a->round)[0];
        if (round < a->max_rounds) {
            TraceRow tr;
            tr.round = round; tr.committed = commit ? 1u : 0u; tr.accepted = commit ? acc : 0u; tr.rejected = commit ? rej : rej + acc;
            tr.vol_lo = ol; tr.vol_hi = oh;
            ag_ptr<TraceRow>(a->trace)[(u64)round * a->configs + c] = tr;
        }
    }
    if (write) {
        obs[0] = vl; obs[1] = vh;
        obs[2] = el[0]; obs[3] = el[1]; obs[4] = el[2];
        obs[5] = eh[0]; obs[6] = eh[1]; obs[7] = eh[2];
        obs[8] = (dom != 0u) ? 1.0 : 0.0;
        obs[9] = (double)cur_c[c];
        obs[10] = jd.met ? 1.0 : 0.0;
        obs[11] = jd.err;
        obs[12] = ve;
        eng_write(en, obs);
        for (int k = 0; k < 3; ++k) obs[27 + k] = (eh[k] > 0.0) ? gr.lo[k] + (double)bb[6 + k] * gr.fine[k] : 0.0;
        if (jd.met && goal.objective == 0u) done[c] = 1u;
        score[c] = jd.sc;
    }
}

// ---- loop control ----
AG_KERNEL void ag_dop_continue(const DopsArgs* a) {
    if (AG_TID != 0) return;
    const u32* done = ag_ptr<const u32>(a->done);
    const u32* budget = ag_ptr<const u32>(a->budget);
    u32 any = 0;
    for (u32 c = 0; c < a->configs; ++c) if (done[c] == 0u && budget[c] > 0u) any = 1u;
    u32* round = ag_ptr<u32>(a->round);
    u32 next = round[0] + 1u;
    round[0] = next;
    u32 flag = (any != 0u && next < a->max_rounds) ? 1u : 0u;
    ag_ptr<u32>(a->active)[0] = flag;
    if (a->cond_handle != 0ull) cudaGraphSetConditional(a->cond_handle, flag);
}

// ---- stage timestamps (not part of the deterministic state) ----
struct StampParams { u64 stamps, round; u32 stage, max_rounds; };
AG_KERNEL void ag_dop_stamp(const StampParams* p) {
    if (AG_TID != 0) return;
    u32 round = ag_ptr<const u32>(p->round)[0];
    if (round >= p->max_rounds) return;
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    ag_ptr<u64>(p->stamps)[(u64)round * 9u + p->stage] = t;
}

// =====================================================================================
// Stiffness proxy: voxel hex8 elasticity, matrix-free Jacobi-PCG (twin of src/fea.rs)
// =====================================================================================
#define DOT_BLOCKS 256u
#define DOT_T 256u
#define S_RZ 0
#define S_RZN 1
#define S_PAP 2
#define S_RR 3
#define S_BN 4
#define S_DONE 5
#define S_ITER 6
#define S_COMPLIANCE 7
#define S_CONVERGED 8
#define S_BB 9
#define N_USED 1u
#define N_FIXED 2u
#define N_LOADED 4u
struct FeaArgs {
    u64 tape, tables, solid, nflag, ke, x, r, z, p, ap, b, diag, scal, stat;
    double lo[3], h, force[3], tol;
    u32 count, dims[3], axis, _pad[3];
};
struct DotArgs { u64 a0, b0, a1, b1, part, scal, n; u32 slot0, slot1, ndots, always; };

__device__ __forceinline__ bool fea_active(const u32* nf, u64 dof) { u32 f = nf[dof / 3u]; return (f & N_USED) != 0u && (f & N_FIXED) == 0u; }

AG_KERNEL void ag_fea_voxelize(const FeaArgs* a) {
    u64 d0 = a->dims[0], d1 = a->dims[1], d2 = a->dims[2];
    const CNode* tape = ag_ptr<const CNode>(a->tape);
    const float* tb = ag_ptr<const float>(a->tables);
    u32* solid = ag_ptr<u32>(a->solid);
    u64* stat = ag_ptr<u64>(a->stat);
    GRID_STRIDE(v, d0 * d1 * d2) {
        u64 i = v % d0, j = (v / d0) % d1, k = v / (d0 * d1);
        float x = (float)(a->lo[0] + ((double)i + 0.5) * a->h);
        float y = (float)(a->lo[1] + ((double)j + 0.5) * a->h);
        float z = (float)(a->lo[2] + ((double)k + 0.5) * a->h);
        bool inside = a->count > 0u && tape_sdf_c(tape, a->count, tb, x, y, z) < 0.0f;
        solid[v] = inside ? 1u : 0u;
        if (inside) atomicAdd((unsigned long long*)&stat[2], 1ull);
    }
}

AG_KERNEL void ag_fea_nodes(const FeaArgs* a) {
    u64 d[3] = {a->dims[0], a->dims[1], a->dims[2]};
    const u32* solid = ag_ptr<const u32>(a->solid);
    u32* nf = ag_ptr<u32>(a->nflag);
    u64* stat = ag_ptr<u64>(a->stat);
    u64 nx = d[0] + 1, ny = d[1] + 1;
    GRID_STRIDE(n, nx * ny * (d[2] + 1)) {
        u64 c[3] = {n % nx, (n / nx) % ny, n / (nx * ny)};
        bool used = false;
        for (int dk = 0; dk < 2; ++dk) for (int dj = 0; dj < 2; ++dj) for (int di = 0; di < 2; ++di) {
            long long i = (long long)c[0] + di - 1, j = (long long)c[1] + dj - 1, k = (long long)c[2] + dk - 1;
            if (i >= 0 && j >= 0 && k >= 0 && (u64)i < d[0] && (u64)j < d[1] && (u64)k < d[2] && solid[((u64)k * d[1] + (u64)j) * d[0] + (u64)i] != 0u) used = true;
        }
        u32 ax = a->axis;
        u32 f = 0u;
        if (used) {
            f |= N_USED;
            if (c[ax] == 0ull) f |= N_FIXED;
            if (c[ax] == d[ax]) { f |= N_LOADED; atomicAdd((unsigned long long*)&stat[0], 1ull); }
            if (c[ax] != 0ull) atomicAdd((unsigned long long*)&stat[3], 3ull);
        }
        nf[n] = f;
    }
}

AG_KERNEL void ag_fea_init(const FeaArgs* a) {
    u64 nd = 3ull * (u64)(a->dims[0] + 1u) * (u64)(a->dims[1] + 1u) * (u64)(a->dims[2] + 1u);
    const u32* nf = ag_ptr<const u32>(a->nflag);
    const double* ke = ag_ptr<const double>(a->ke);
    u64 nload = ag_ptr<const u64>(a->stat)[0];
    double *x = ag_ptr<double>(a->x), *r = ag_ptr<double>(a->r), *z = ag_ptr<double>(a->z), *p = ag_ptr<double>(a->p), *b = ag_ptr<double>(a->b), *dg = ag_ptr<double>(a->diag);
    GRID_STRIDE(i, nd) {
        bool act = fea_active(nf, i);
        u32 c = (u32)(i % 3ull);
        b[i] = (act && (nf[i / 3u] & N_LOADED) != 0u) ? a->force[c] / (double)nload : 0.0;
        dg[i] = act ? ke[c * 25u] * 4.0 : 1.0;
        x[i] = 0.0;
        r[i] = b[i];
        z[i] = r[i] / dg[i];
        p[i] = z[i];
    }
}

AG_KERNEL void ag_fea_dot_part(const DotArgs* a) {
    __shared__ double s0[DOT_T];
    __shared__ double s1[DOT_T];
    const double* scal = ag_ptr<const double>(a->scal);
    if (a->always == 0u && scal[S_DONE] != 0.0) return;
    u32 t = threadIdx.x, blk = blockIdx.x;
    const double *a0 = ag_ptr<const double>(a->a0), *b0 = ag_ptr<const double>(a->b0), *a1 = ag_ptr<const double>(a->a1), *b1 = ag_ptr<const double>(a->b1);
    double v0 = 0.0, v1 = 0.0;
    for (u64 i = (u64)blk * DOT_T + t; i < a->n; i += (u64)DOT_BLOCKS * DOT_T) {
        v0 = v0 + a0[i] * b0[i];
        if (a->ndots > 1u) v1 = v1 + a1[i] * b1[i];
    }
    s0[t] = v0; s1[t] = v1;
    __syncthreads();
    for (u32 st = DOT_T / 2u; st > 0u; st >>= 1) {
        if (t < st) { s0[t] = s0[t] + s0[t + st]; s1[t] = s1[t] + s1[t + st]; }
        __syncthreads();
    }
    if (t == 0u) { double* part = ag_ptr<double>(a->part); part[blk] = s0[0]; part[DOT_BLOCKS + blk] = s1[0]; }
}

AG_KERNEL void ag_fea_dot_final(const DotArgs* a) {
    __shared__ double s0[DOT_T];
    __shared__ double s1[DOT_T];
    double* scal = ag_ptr<double>(a->scal);
    if (a->always == 0u && scal[S_DONE] != 0.0) return;
    u32 t = threadIdx.x;
    const double* part = ag_ptr<const double>(a->part);
    double v0 = 0.0, v1 = 0.0;
    for (u32 k = t; k < DOT_BLOCKS; k += DOT_T) { v0 = v0 + part[k]; v1 = v1 + part[DOT_BLOCKS + k]; }
    s0[t] = v0; s1[t] = v1;
    __syncthreads();
    for (u32 st = DOT_T / 2u; st > 0u; st >>= 1) {
        if (t < st) { s0[t] = s0[t] + s0[t + st]; s1[t] = s1[t] + s1[t + st]; }
        __syncthreads();
    }
    if (t == 0u) {
        scal[a->slot0] = s0[0];
        if (a->ndots > 1u) scal[a->slot1] = s1[0];
        if (a->always == 0u && a->slot0 == (u32)S_PAP && !(s0[0] > 0.0)) scal[S_DONE] = 1.0;
    }
}

AG_KERNEL void ag_fea_start(const FeaArgs* a) {
    if (AG_TID != 0) return;
    double* scal = ag_ptr<double>(a->scal);
    double bn = sqrt(scal[S_BB]);
    scal[S_BN] = (bn > 2.2250738585072014e-308) ? bn : 2.2250738585072014e-308;
    scal[S_DONE] = 0.0; scal[S_ITER] = 0.0; scal[S_CONVERGED] = 0.0;
}

AG_KERNEL void ag_fea_check(const FeaArgs* a) {
    if (AG_TID != 0) return;
    double* scal = ag_ptr<double>(a->scal);
    if (scal[S_DONE] != 0.0) return;
    if (sqrt(scal[S_RR]) / scal[S_BN] <= a->tol) { scal[S_DONE] = 1.0; scal[S_CONVERGED] = 1.0; }
}

__device__ const u32 FEA_CORNER[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};

AG_KERNEL void ag_fea_matvec(const FeaArgs* a) {
    if (ag_ptr<const double>(a->scal)[S_DONE] != 0.0) return;
    u64 d[3] = {a->dims[0], a->dims[1], a->dims[2]};
    u64 nx = d[0] + 1, ny = d[1] + 1;
    u64 nd = 3ull * nx * ny * (d[2] + 1);
    const u32* solid = ag_ptr<const u32>(a->solid);
    const u32* nf = ag_ptr<const u32>(a->nflag);
    const double* ke = ag_ptr<const double>(a->ke);
    const double* p = ag_ptr<const double>(a->p);
    double* yv = ag_ptr<double>(a->ap);
    GRID_STRIDE(dof, nd) {
        if (!fea_active(nf, dof)) { yv[dof] = 0.0; continue; }
        u64 n = dof / 3ull;
        u32 comp = (u32)(dof % 3ull);
        u64 c[3] = {n % nx, (n / nx) % ny, n / (nx * ny)};
        double y = 0.0;
        for (u32 dk = 0; dk < 2u; ++dk) for (u32 dj = 0; dj < 2u; ++dj) for (u32 di = 0; di < 2u; ++di) {
            if (c[0] + di < 1ull || c[1] + dj < 1ull || c[2] + dk < 1ull) continue;
            u64 i = c[0] + di - 1ull, j = c[1] + dj - 1ull, k = c[2] + dk - 1ull;
            if (i >= d[0] || j >= d[1] || k >= d[2] || solid[(k * d[1] + j) * d[0] + i] == 0u) continue;
            u32 l0 = 1u - di, l1 = 1u - dj, l2 = 1u - dk;
            u32 la = 0;
            for (u32 q = 0; q < 8u; ++q) if (FEA_CORNER[q][0] == l0 && FEA_CORNER[q][1] == l1 && FEA_CORNER[q][2] == l2) la = q;
            u32 row = 3u * la + comp;
            double s = 0.0;
            for (u32 q = 0; q < 8u; ++q) {
                u64 m = ((k + FEA_CORNER[q][2]) * ny + (j + FEA_CORNER[q][1])) * nx + (i + FEA_CORNER[q][0]);
                for (u32 e = 0; e < 3u; ++e) {
                    u64 dd = 3ull * m + e;
                    double u = fea_active(nf, dd) ? p[dd] : 0.0;
                    s = s + ke[row * 24u + 3u * q + e] * u;
                }
            }
            y = y + 1.0 * s;
        }
        yv[dof] = y;
    }
}

AG_KERNEL void ag_fea_update1(const FeaArgs* a) {
    const double* scal = ag_ptr<const double>(a->scal);
    if (scal[S_DONE] != 0.0) return;
    u64 nd = 3ull * (u64)(a->dims[0] + 1u) * (u64)(a->dims[1] + 1u) * (u64)(a->dims[2] + 1u);
    double alpha = scal[S_RZ] / scal[S_PAP];
    double *x = ag_ptr<double>(a->x), *r = ag_ptr<double>(a->r), *z = ag_ptr<double>(a->z);
    const double *p = ag_ptr<const double>(a->p), *ap = ag_ptr<const double>(a->ap), *dg = ag_ptr<const double>(a->diag);
    GRID_STRIDE(i, nd) {
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * ap[i];
        z[i] = r[i] / dg[i];
    }
}

AG_KERNEL void ag_fea_update2(const FeaArgs* a) {
    const double* scal = ag_ptr<const double>(a->scal);
    if (scal[S_DONE] != 0.0) return;
    u64 nd = 3ull * (u64)(a->dims[0] + 1u) * (u64)(a->dims[1] + 1u) * (u64)(a->dims[2] + 1u);
    double beta = scal[S_RZN] / scal[S_RZ];
    const double* z = ag_ptr<const double>(a->z);
    double* p = ag_ptr<double>(a->p);
    GRID_STRIDE(i, nd) { p[i] = z[i] + beta * p[i]; }
}

AG_KERNEL void ag_fea_step(const FeaArgs* a) {
    if (AG_TID != 0) return;
    double* scal = ag_ptr<double>(a->scal);
    if (scal[S_DONE] != 0.0) return;
    scal[S_RZ] = scal[S_RZN];
    scal[S_ITER] = scal[S_ITER] + 1.0;
}

AG_KERNEL void ag_fea_finish(const FeaArgs* a) {
    if (AG_TID != 0) return;
    double* scal = ag_ptr<double>(a->scal);
    if (sqrt(scal[S_RR]) / scal[S_BN] <= a->tol) scal[S_CONVERGED] = 1.0;
}

AG_KERNEL void ag_fea_maxdisp(const FeaArgs* a) {
    u64 nn = (u64)(a->dims[0] + 1u) * (u64)(a->dims[1] + 1u) * (u64)(a->dims[2] + 1u);
    const double* x = ag_ptr<const double>(a->x);
    u64* stat = ag_ptr<u64>(a->stat);
    GRID_STRIDE(n, nn) {
        double m = sqrt(x[3 * n] * x[3 * n] + x[3 * n + 1] * x[3 * n + 1] + x[3 * n + 2] * x[3 * n + 2]);
        atomicMax((unsigned long long*)&stat[1], (unsigned long long)__double_as_longlong(m));
    }
}

// ---- certified implicit mesh: SDF at grid nodes (twin of src/mesh.rs) ----
struct MeshArgs { u64 tape, tables, out, _pad0; double lo[3], h; u32 count, dims[3]; };
AG_KERNEL void ag_mesh_sample(const MeshArgs* a) {
    u64 nx = a->dims[0], ny = a->dims[1], nz = a->dims[2];
    const CNode* tape = ag_ptr<const CNode>(a->tape);
    const float* tb = ag_ptr<const float>(a->tables);
    float* out = ag_ptr<float>(a->out);
    GRID_STRIDE(n, nx * ny * nz) {
        u64 i = n % nx, j = (n / nx) % ny, k = n / (nx * ny);
        float x = (float)(a->lo[0] + (double)i * a->h);
        float y = (float)(a->lo[1] + (double)j * a->h);
        float z = (float)(a->lo[2] + (double)k * a->h);
        out[n] = (a->count == 0u) ? FAR32 : tape_sdf_c(tape, a->count, tb, x, y, z);
    }
}
