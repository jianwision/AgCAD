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
