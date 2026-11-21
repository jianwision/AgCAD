// Batched MLP forward (CuModel inference inside the rollout graph). One thread per
// (configuration, output unit) per layer; ReLU hidden, linear output. Mirrors
// agcad_model::cumodel::ag_mlp_layer_cpu exactly (fmad off; fixed summation order).
#include "ag_common.cuh"

struct MlpLayerParams { u64 x, w, b, y; u32 configs, n_in, n_out, relu; };

AG_KERNEL void ag_mlp_layer(const MlpLayerParams* p) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (u64)p->configs * p->n_out) return;
    u32 c = (u32)(t / p->n_out), o = (u32)(t % p->n_out);
    const float* x = ag_ptr<const float>(p->x) + (u64)c * p->n_in;
    const float* w = ag_ptr<const float>(p->w) + (u64)o * p->n_in;
    float acc = ag_ptr<const float>(p->b)[o];
    for (u32 i = 0; i < p->n_in; ++i) acc += w[i] * x[i];
    if (p->relu && acc < 0.0f) acc = 0.0f;
    ag_ptr<float>(p->y)[(u64)c * p->n_out + o] = acc;
}
