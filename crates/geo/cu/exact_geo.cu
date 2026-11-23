// GPU-native exact B-rep kernel primitives (Phase 3):
// 1. Filtered Shewchuk predicates (orient2d, orient3d) with inconclusive mask for exact expansion compaction
// 2. Spatial AABB overlap pair culling (surface/face broadphase)
// 3. Batched minimal-norm Newton SSI refinement onto analytic surfaces (Plane/Cylinder/Sphere/Cone)
// 4. Parallel face-fragment classification (signed distance / parity against plane/box/analytic solids)
// 5. Parallel STEP AP242 prefix-scan line scatter into one contiguous device buffer
#include "ag_common.cuh"

// ---- 1. Filtered Shewchuk Predicates ----
struct Orient2dParams { u64 pts_a, pts_b, pts_c, out_sign; u32 n, _pad; };
struct Orient3dParams { u64 pts_a, pts_b, pts_c, pts_d, out_sign; u32 n, _pad; };

// Result encoding: -1, 0, +1 exact/filtered sign; 2 = inconclusive (needs exact expansion kernel)
AG_KERNEL void ag_geo_orient2d_batch(const Orient2dParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    const double* a = ag_ptr<const double>(p->pts_a) + (u64)i * 2u;
    const double* b = ag_ptr<const double>(p->pts_b) + (u64)i * 2u;
    const double* c = ag_ptr<const double>(p->pts_c) + (u64)i * 2u;
    double detl = (a[0] - c[0]) * (b[1] - c[1]);
    double detr = (a[1] - c[1]) * (b[0] - c[0]);
    double det = detl - detr;
    const double eps = 1.1102230246251565e-16;
    const double bound = (3.0 + 16.0 * eps) * eps * (fabs(detl) + fabs(detr));
    int res = 2;
    if (det > bound) res = 1;
    else if (-det > bound) res = -1;
    else if (bound == 0.0) res = 0;
    ag_ptr<int>(p->out_sign)[i] = res;
}

AG_KERNEL void ag_geo_orient3d_batch(const Orient3dParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    const double* a = ag_ptr<const double>(p->pts_a) + (u64)i * 3u;
    const double* b = ag_ptr<const double>(p->pts_b) + (u64)i * 3u;
    const double* c = ag_ptr<const double>(p->pts_c) + (u64)i * 3u;
    const double* d = ag_ptr<const double>(p->pts_d) + (u64)i * 3u;
    double adx = a[0] - d[0], ady = a[1] - d[1], adz = a[2] - d[2];
    double bdx = b[0] - d[0], bdy = b[1] - d[1], bdz = b[2] - d[2];
    double cdx = c[0] - d[0], cdy = c[1] - d[1], cdz = c[2] - d[2];
    double t1 = bdy * cdz - bdz * cdy;
    double t2 = bdz * cdx - bdx * cdz;
    double t3 = bdx * cdy - bdy * cdx;
    double det = adx * t1 + ady * t2 + adz * t3;
    double perm = fabs(adx) * (fabs(bdy) * fabs(cdz) + fabs(bdz) * fabs(cdy))
                + fabs(ady) * (fabs(bdz) * fabs(cdx) + fabs(bdx) * fabs(cdz))
                + fabs(adz) * (fabs(bdx) * fabs(cdy) + fabs(bdy) * fabs(cdx));
    const double eps = 1.1102230246251565e-16;
    const double bound = (7.0 + 56.0 * eps) * eps * perm;
    int res = 2;
    if (det > bound) res = 1;
    else if (-det > bound) res = -1;
    else if (bound == 0.0) res = 0;
    ag_ptr<int>(p->out_sign)[i] = res;
}

// ---- 2. Spatial AABB Overlap Pair Broadphase ----
// Each AABB is 6 doubles: [minx, miny, minz, maxx, maxy, maxz].
// Writes packed (ia << 32 | ib) pairs into out_pairs[0..max_pairs] using atomic counter out_count[0].
struct AabbPairParams { u64 boxes_a, boxes_b, out_pairs, out_count; double tol; u32 na, nb, max_pairs, _pad; };

AG_KERNEL void ag_geo_aabb_overlap_pairs(const AabbPairParams* p) {
    u64 idx = (u64)AG_TID;
    u64 total = (u64)p->na * (u64)p->nb;
    if (idx >= total) return;
    u32 ia = (u32)(idx / (u64)p->nb);
    u32 ib = (u32)(idx % (u64)p->nb);
    const double* a = ag_ptr<const double>(p->boxes_a) + (u64)ia * 6u;
    const double* b = ag_ptr<const double>(p->boxes_b) + (u64)ib * 6u;
    double t = p->tol;
    bool ov = (a[0] <= b[3] + t && a[3] >= b[0] - t)
           && (a[1] <= b[4] + t && a[4] >= b[1] - t)
           && (a[2] <= b[5] + t && a[5] >= b[2] - t);
    if (ov) {
        u32 slot = atomicAdd(ag_ptr<u32>(p->out_count), 1u);
        if (slot < p->max_pairs) {
            ag_ptr<u64>(p->out_pairs)[slot] = ((u64)ia << 32) | (u64)ib;
        }
    }
}

// ---- 3. Batched Analytic SSI Newton Refinement ----
// AnalyticSurface POD (16 doubles = 128 B):
// kind: 0=Plane, 1=Cylinder, 2=Sphere, 3=Cone
// o[3], x[3], y[3], z[3], r, angle, kind_as_f64, pad
struct AnalyticSurf {
    double ox, oy, oz;
    double xx, xy, xz;
    double yx, yy, yz;
    double zx, zy, zz;
    double r, angle;
    u32 kind, _pad;
};

__device__ void eval_analytic_jet(const AnalyticSurf& s, double u, double v, double p[3], double du[3], double dv[3]) {
    if (s.kind == 0u) { // Plane
        p[0] = s.ox + s.xx * u + s.yx * v;
        p[1] = s.oy + s.xy * u + s.yy * v;
        p[2] = s.oz + s.xz * u + s.yz * v;
        du[0] = s.xx; du[1] = s.xy; du[2] = s.xz;
        dv[0] = s.yx; dv[1] = s.yy; dv[2] = s.yz;
    } else if (s.kind == 1u) { // Cylinder
        double su = sin(u), cu = cos(u);
        double rx = s.xx * cu + s.yx * su, ry = s.xy * cu + s.yy * su, rz = s.xz * cu + s.yz * su;
        p[0] = s.ox + rx * s.r + s.zx * v;
        p[1] = s.oy + ry * s.r + s.zy * v;
        p[2] = s.oz + rz * s.r + s.zz * v;
        du[0] = (-s.xx * su + s.yx * cu) * s.r;
        du[1] = (-s.xy * su + s.yy * cu) * s.r;
        du[2] = (-s.xz * su + s.yz * cu) * s.r;
        dv[0] = s.zx; dv[1] = s.zy; dv[2] = s.zz;
    } else if (s.kind == 2u) { // Sphere
        double su = sin(u), cu = cos(u), sv = sin(v), cv = cos(v);
        double rx = s.xx * cu + s.yx * su, ry = s.xy * cu + s.yy * su, rz = s.xz * cu + s.yz * su;
        double tx = -s.xx * su + s.yx * cu, ty = -s.xy * su + s.yy * cu, tz = -s.xz * su + s.yz * cu;
        p[0] = s.ox + (rx * cv + s.zx * sv) * s.r;
        p[1] = s.oy + (ry * cv + s.zy * sv) * s.r;
        p[2] = s.oz + (rz * cv + s.zz * sv) * s.r;
        du[0] = tx * (cv * s.r); du[1] = ty * (cv * s.r); du[2] = tz * (cv * s.r);
        dv[0] = (-rx * sv + s.zx * cv) * s.r;
        dv[1] = (-ry * sv + s.zy * cv) * s.r;
        dv[2] = (-rz * sv + s.zz * cv) * s.r;
    } else { // Cone
        double su = sin(u), cu = cos(u), ta = tan(s.angle);
        double rr = s.r + v * ta;
        double rx = s.xx * cu + s.yx * su, ry = s.xy * cu + s.yy * su, rz = s.xz * cu + s.yz * su;
        p[0] = s.ox + rx * rr + s.zx * v;
        p[1] = s.oy + ry * rr + s.zy * v;
        p[2] = s.oz + rz * rr + s.zz * v;
        du[0] = (-s.xx * su + s.yx * cu) * rr;
        du[1] = (-s.xy * su + s.yy * cu) * rr;
        du[2] = (-s.xz * su + s.yz * cu) * rr;
        dv[0] = rx * ta + s.zx; dv[1] = ry * ta + s.zy; dv[2] = rz * ta + s.zz;
    }
}

struct SsiRefineParams { u64 surfs_a, surfs_b, seeds_q, out_q, out_ok; double tol; u32 n, max_iters; };

AG_KERNEL void ag_geo_ssi_refine_batch(const SsiRefineParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    AnalyticSurf sa = ag_ptr<const AnalyticSurf>(p->surfs_a)[i];
    AnalyticSurf sb = ag_ptr<const AnalyticSurf>(p->surfs_b)[i];
    const double* q0 = ag_ptr<const double>(p->seeds_q) + (u64)i * 4u;
    double q[4] = { q0[0], q0[1], q0[2], q0[3] };
    u32 ok = 0u;
    for (u32 it = 0u; it < p->max_iters; ++it) {
        double pa[3], dua[3], dva[3], pb[3], dub[3], dvb[3];
        eval_analytic_jet(sa, q[0], q[1], pa, dua, dva);
        eval_analytic_jet(sb, q[2], q[3], pb, dub, dvb);
        double r[3] = { pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2] };
        double rn = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (rn < p->tol) { ok = 1u; break; }
        double cols[4][3] = {
            { dua[0], dua[1], dua[2] },
            { dva[0], dva[1], dva[2] },
            { -dub[0], -dub[1], -dub[2] },
            { -dvb[0], -dvb[1], -dvb[2] }
        };
        double m[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    m[row][col] += cols[c][row] * cols[c][col];
        double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                   - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (fabs(det) < 1e-300) break;
        double rhs[3] = { -r[0], -r[1], -r[2] };
        double y[3];
        for (int col = 0; col < 3; ++col) {
            double k[3][3];
            for (int rr = 0; rr < 3; ++rr) for (int cc = 0; cc < 3; ++cc) k[rr][cc] = (cc == col) ? rhs[rr] : m[rr][cc];
            double dc = k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1])
                      - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0])
                      + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
            y[col] = dc / det;
        }
        bool finite = true;
        for (int c = 0; c < 4; ++c) {
            q[c] += cols[c][0] * y[0] + cols[c][1] * y[1] + cols[c][2] * y[2];
            if (!(q[c] == q[c]) || fabs(q[c]) > 1e100) finite = false;
        }
        if (!finite) break;
    }
    double* qo = ag_ptr<double>(p->out_q) + (u64)i * 4u;
    for (int k = 0; k < 4; ++k) qo[k] = q[k];
    ag_ptr<u32>(p->out_ok)[i] = ok;
}

// ---- 4. Parallel Face-Fragment Classification (Ray-Parity Against Triangle Mesh) ----
// Each ray starts at query point `pts[i]` in direction (+1, 0.3713906763541037, 0.2182178902359924) (skewed irrational-like to avoid coplanar hits).
// Out: 0 = outside, 1 = inside, 2 = on-boundary (within tol).
struct ClassifyParams { u64 pts, tris, out_cls; double tol; u32 n_pts, n_tris; };

AG_KERNEL void ag_geo_classify_faces(const ClassifyParams* p) {
    u32 i = AG_TID;
    if (i >= p->n_pts) return;
    const double* q = ag_ptr<const double>(p->pts) + (u64)i * 3u;
    const double* tris = ag_ptr<const double>(p->tris);
    double dx = 0.9162708326955029, dy = 0.3402944464297582, dz = 0.21081851067789195;
    u32 hits = 0u;
    bool on_surf = false;
    for (u32 t = 0u; t < p->n_tris; ++t) {
        const double* tr = tris + (u64)t * 9u;
        double e1x = tr[3] - tr[0], e1y = tr[4] - tr[1], e1z = tr[5] - tr[2];
        double e2x = tr[6] - tr[0], e2y = tr[7] - tr[1], e2z = tr[8] - tr[2];
        double px = dy * e2z - dz * e2y;
        double py = dz * e2x - dx * e2z;
        double pz = dx * e2y - dy * e2x;
        double det = e1x * px + e1y * py + e1z * pz;
        if (fabs(det) < 1e-15) continue;
        double inv = 1.0 / det;
        double tx = q[0] - tr[0], ty = q[1] - tr[1], tz = q[2] - tr[2];
        double u = (tx * px + ty * py + tz * pz) * inv;
        if (u < -1e-12 || u > 1.0 + 1e-12) continue;
        double qx = ty * e1z - tz * e1y;
        double qy = tz * e1x - tx * e1z;
        double qz = tx * e1y - ty * e1x;
        double v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < -1e-12 || u + v > 1.0 + 1e-12) continue;
        double dist = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (fabs(dist) <= p->tol) { on_surf = true; break; }
        if (dist > p->tol) hits ^= 1u;
    }
    ag_ptr<u32>(p->out_cls)[i] = on_surf ? 2u : hits;
}

// ---- 5. Parallel Prefix-Scan STEP Line Scatter ----
struct StepScatterParams { u64 src_bytes, src_offs, dst_offs, lengths, dst_bytes; u32 n_lines, _pad; };

AG_KERNEL void ag_geo_step_scatter(const StepScatterParams* p) {
    u32 i = AG_TID;
    if (i >= p->n_lines) return;
    const u8* src = ag_ptr<const u8>(p->src_bytes);
    u8* dst = ag_ptr<u8>(p->dst_bytes);
    u32 so = ag_ptr<const u32>(p->src_offs)[i];
    u32 doff = ag_ptr<const u32>(p->dst_offs)[i];
    u32 len = ag_ptr<const u32>(p->lengths)[i];
    for (u32 k = 0u; k < len; ++k) dst[doff + k] = src[so + k];
}
