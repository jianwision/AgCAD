// GPU pixel-parallel offscreen rasterizer (spec 07 § Diagnostics).
// One thread per pixel (no atomic races -> bit-identical CUDA == CPU reference).
#include "ag_common.cuh"

struct RenderParams {
    u64 tris;     // float[n_tris * 12]: (ax,ay,az, bx,by,bz, cx,cy,cz, r,g,b) in screen space
    u64 out_rgb;  // u8[w * h * 3]
    u32 n_tris;
    u32 w;
    u32 h;
    u32 _pad;
};

AG_KERNEL void ag_rasterize(const RenderParams* p) {
    u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 total = (u64)p->w * (u64)p->h;
    if (idx >= total) return;
    u32 x = (u32)(idx % p->w);
    u32 y = (u32)(idx / p->w);
    double px = (double)x + 0.5;
    double py = (double)y + 0.5;
    const float* tris = ag_ptr<const float>(p->tris);
    unsigned char* img = ag_ptr<unsigned char>(p->out_rgb);
    double best_z = -1.0e300;
    unsigned char cr = 255u, cg = 255u, cb = 255u;
    for (u32 t = 0; t < p->n_tris; ++t) {
        const float* d = tris + (u64)t * 12ull;
        double ax = (double)d[0], ay = (double)d[1], az = (double)d[2];
        double bx = (double)d[3], by = (double)d[4], bz = (double)d[5];
        double cx = (double)d[6], cy = (double)d[7], cz = (double)d[8];
        double minx = ax < bx ? (ax < cx ? ax : cx) : (bx < cx ? bx : cx);
        double maxx = ax > bx ? (ax > cx ? ax : cx) : (bx > cx ? bx : cx);
        if (px < minx || px > maxx) continue;
        double miny = ay < by ? (ay < cy ? ay : cy) : (by < cy ? by : cy);
        double maxy = ay > by ? (ay > cy ? ay : cy) : (by > cy ? by : cy);
        if (py < miny || py > maxy) continue;
        double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (fabs(area) < 1e-12) continue;
        double w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
        double w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
        double w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) continue;
        double depth = -(w0 * az + w1 * bz + w2 * cz);
        if (depth > best_z) {
            best_z = depth;
            cr = (unsigned char)d[9];
            cg = (unsigned char)d[10];
            cb = (unsigned char)d[11];
        }
    }
    img[idx * 3ull + 0ull] = cr;
    img[idx * 3ull + 1ull] = cg;
    img[idx * 3ull + 2ull] = cb;
}
