struct LoftParams {
    const double *sections;
    double *poles;
    unsigned long long edges;
};

// Each section starts with origin and two basis vectors, followed by UV vertices.
extern "C" __global__ void ag_loft_poles(const LoftParams *params) {
    const unsigned long long edge = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (edge >= params->edges) return;
    for (int end = 0; end < 2; ++end) {
        const auto vertex = (edge + end) % params->edges;
        for (int section = 0; section < 2; ++section) {
            const double *frame = params->sections + section * (9 + 2 * params->edges);
            const double *uv = frame + 9 + 2 * vertex;
            double *p = params->poles + 12 * edge + 6 * end + 3 * section;
            for (int axis = 0; axis < 3; ++axis)
                p[axis] = frame[axis] + uv[0] * frame[3+axis] + uv[1] * frame[6+axis];
        }
    }
}

// Each row is a linear combination in homogeneous coordinates. The prefix holds
// offsets into the packed (coefficient, x, y, z, weight) terms.
extern "C" __global__ void ag_surface_rows(const LoftParams *params) {
    const auto i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= params->edges) return;
    const double *input = params->sections;
    const double *terms = input + params->edges + 1;
    double h[4] = {};
    for (unsigned long long j = (unsigned long long)input[i]; j < (unsigned long long)input[i+1]; ++j) {
        const double *p = terms + 5*j;
        const double w = p[0]*p[4];
        for (int k=0; k<3; ++k) h[k] += w*p[k+1];
        h[3] += w;
    }
    double *out = params->poles + 4*i;
    for (int k=0;k<3;++k) out[k] = h[k]/h[3];
    out[3] = h[3];
}

// Batched tensor-product rational NURBS evaluation: S(u,v) = (B_u * P_h * B_v^T) / W, plus
// first partial derivatives (S_u, S_v) in world coordinates.
// Packed input layout (all f64):
//   [nu, nv, su, sv, bu(su*nu), dbu(su*nu), bv(sv*nv), dbv(sv*nv), poles(nu*nv*4: x,y,z,w)]
// Output layout: su*sv*9 f64 (px,py,pz, dux,duy,duz, dvx,dvy,dvz).
extern "C" __global__ void ag_nurbs_eval_grid(const LoftParams *params) {
    const unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= params->edges) return;
    const double *in = params->sections;
    const unsigned long long nu = (unsigned long long)in[0];
    const unsigned long long nv = (unsigned long long)in[1];
    const unsigned long long su = (unsigned long long)in[2];
    const unsigned long long sv = (unsigned long long)in[3];
    const unsigned long long iu = idx / sv;
    const unsigned long long iv = idx % sv;
    const double *bu  = in + 4;
    const double *dbu = bu + su * nu;
    const double *bv  = dbu + su * nu;
    const double *dbv = bv + sv * nv;
    const double *poles = dbv + sv * nv;
    const double *bu_row  = bu  + iu * nu;
    const double *dbu_row = dbu + iu * nu;
    const double *bv_row  = bv  + iv * nv;
    const double *dbv_row = dbv + iv * nv;
    double h[4] = {}, hu[4] = {}, hv[4] = {};
    for (unsigned long long i = 0; i < nu; ++i) {
        const double bi = bu_row[i], dbi = dbu_row[i];
        if (bi == 0.0 && dbi == 0.0) continue;
        for (unsigned long long j = 0; j < nv; ++j) {
            const double bj = bv_row[j], dbj = dbv_row[j];
            if (bj == 0.0 && dbj == 0.0) continue;
            const double *pt = poles + 4 * (i * nv + j);
            const double w = pt[3];
            const double w0 = bi * bj * w;
            const double wu = dbi * bj * w;
            const double wv = bi * dbj * w;
            for (int k = 0; k < 3; ++k) {
                h[k]  += w0 * pt[k];
                hu[k] += wu * pt[k];
                hv[k] += wv * pt[k];
            }
            h[3]  += w0;
            hu[3] += wu;
            hv[3] += wv;
        }
    }
    double *out = params->poles + 9 * idx;
    const double invw = (h[3] != 0.0) ? (1.0 / h[3]) : 0.0;
    for (int k = 0; k < 3; ++k) {
        const double p = h[k] * invw;
        out[k]     = p;
        out[3 + k] = (hu[k] - hu[3] * p) * invw;
        out[6 + k] = (hv[k] - hv[3] * p) * invw;
    }
}
