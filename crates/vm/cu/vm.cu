// Device VM: uniform interpretation of one expression program over [configuration, sample].
// Mirrors agcad_vm::ag_vm_eval_cpu exactly (AG-06-006). Requires ag_value_noise (prepended).
#include "ag_common.cuh"

#define MAX_VM_NODES 1024

struct VmInstr { unsigned short op, pad; unsigned int a, b, c; };
struct VmParams {
    u64 instrs, consts, inputs, params, out;
    u32 n_instrs, samples, configs, n_inputs, n_params, n_outputs, inputs_per_config, _pad;
};

__device__ __forceinline__ double vm_un(unsigned short o, double a) {
    switch (o) {
        case 0: return -a; case 1: return fabs(a); case 2: return sqrt(a); case 3: return sin(a);
        case 4: return cos(a); case 5: return tan(a); case 6: return asin(a); case 7: return acos(a);
        case 8: return atan(a); case 9: return exp(a); case 10: return log(a); case 11: return floor(a);
        case 12: return ceil(a); default: return a == 0.0 ? 1.0 : 0.0;
    }
}
__device__ __forceinline__ double vm_bin(unsigned short o, double a, double b) {
    switch (o) {
        case 0: return a + b; case 1: return a - b; case 2: return a * b; case 3: return a / b;
        case 4: return b < a ? b : a; case 5: return b > a ? b : a; case 6: return pow(a, b); case 7: return atan2(a, b);
        case 8: return a < b ? 1.0 : 0.0; case 9: return a <= b ? 1.0 : 0.0; case 10: return a > b ? 1.0 : 0.0;
        case 11: return a >= b ? 1.0 : 0.0; case 12: return a == b ? 1.0 : 0.0; case 13: return a != b ? 1.0 : 0.0;
        case 14: return (a != 0.0 && b != 0.0) ? 1.0 : 0.0; case 15: return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
        default: return a - b * floor(a / b);
    }
}

AG_KERNEL void ag_vm_eval(const VmParams* p) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 total = (u64)p->samples * p->configs;
    if (t >= total) return;
    u32 c = (u32)(t / p->samples), s = (u32)(t % p->samples);
    const VmInstr* ins = ag_ptr<const VmInstr>(p->instrs);
    const double* k = ag_ptr<const double>(p->consts);
    const double* in = ag_ptr<const double>(p->inputs);
    const double* prm = ag_ptr<const double>(p->params);
    double* out = ag_ptr<double>(p->out);
    u64 in_stride = (u64)(p->inputs_per_config ? p->configs : 1u) * p->samples;
    u64 in_base = (u64)(p->inputs_per_config ? c : 0u) * p->samples + s;
    double r[MAX_VM_NODES];
    u32 n = 0;
    for (u32 i = 0; i < p->n_instrs; ++i) {
        VmInstr q = ins[i];
        double v;
        if (q.op == 0xFF00) v = k[q.a];
        else if (q.op == 0xFF01) v = in[(u64)q.a * in_stride + in_base];
        else if (q.op == 0xFF02) v = prm[(u64)q.a * p->configs + c];
        else if (q.op == 0xFF80) v = r[q.a] != 0.0 ? r[q.b] : r[q.c];
        else if (q.op == 0xFF81) v = ag_value_noise(r[q.a], r[q.b], r[q.c]);
        else if (q.op == 0xFFF0) { out[((u64)q.b * p->configs + c) * p->samples + s] = r[q.a]; continue; }
        else if (q.op >= 0xFF40 && q.op < 0xFF40 + 17) v = vm_bin(q.op - 0xFF40, r[q.a], r[q.b]);
        else v = vm_un(q.op - 0xFF10, r[q.a]);
        r[n++] = v;
    }
}
