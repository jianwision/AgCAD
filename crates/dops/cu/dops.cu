// Device operation layer (Tier N). Line-by-line twin of crates/dops/src/{math,reference}.rs:
// same operation order, no FMA contraction (NVRTC fmad=false), no libm transcendentals, so
// every result is bit-identical to the CPU reference (AG-06-006, AG-08-005).
#include "ag_common.cuh"

typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __attribute__((cudart_builtin)) void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

#define NVALS 12
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
#define MAX_BODIES 4
#define FOP_INTERSECT 2u
#define FOP_ADD 0u
#define FOP_REMOVE 1u
#define NF_ACTIVE 1u
#define NF_PATTERN 2u
#define NF_PATTERN_RECT 4u
#define NF_FILLET 8u
#define NF_CHAMFER 16u
#define PARAM_ADD 1u
#define FAR 1.0e30
#define TAU 6.283185307179586
#define PI 3.141592653589793

struct Node { u32 kind, op, flags, src; u32 pref[NVALS]; double val[NVALS]; };
struct FeatureArgs { u32 pref[NVALS]; double val[NVALS]; };
struct CNode { u32 kind, op, active, count, sides, mods, _pad[2]; float v[NVALS]; float apothem; float _pad2[11]; };
#define FAR32 1.0e30f
#define TABLE 384
#define POLY 256
struct Goal {
    double domain_lo[3], domain_hi[3], vol_lo, vol_hi, max_ext[3], min_wall;
    double place[4], hole_r, hole_z0, hole_z1, gain;
    double param_lo[MAX_PARAMS], param_hi[MAX_PARAMS], param_sign[MAX_PARAMS];
    u32 n_params, seed, machining_3ax, _pad1;
    double min_gap, undercut_tol, centroid_lo[3], centroid_hi[3];
};
struct TraceRow { u32 round, committed, accepted, rejected; double vol_lo, vol_hi; };
struct OpTable { u32 update, parameter, pad, pocket, hole, circular_pattern, deactivate, rect_pattern, fillet, chamfer, shell, body, add, remove, intersect, _pad; };
struct DopsArgs {
    u64 goal, cur_nodes, cur_count, cur_params, tr_nodes, tr_count, tr_params, rev, budget, done,
        cmds, arena, status, counts, domain_hit, bbox, obs, score, trace, round, active, compiled, tables,
        wl0, wl1, cls, flag, offs, crank, start, lvl, cond_handle;
    u32 configs, nodes, params, records, coarse, levels, eval_trial, max_rounds, level, capacity, cap, cfg_offset, tighten, _pad;
    OpTable ops;
};
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
__device__ CNode compile_node(const Node& n, const double* params, float* table) {
    CNode c;
    double v[NVALS];
    for (int j = 0; j < NVALS; ++j) v[j] = value_of(n, params, j);
    for (int j = 0; j < NVALS; ++j) c.v[j] = (float)v[j];
    c.kind = n.kind; c.op = n.op; c.active = ((n.flags & NF_ACTIVE) != 0u) ? 1u : 0u; c.count = 1u; c.sides = 0u;
    c.mods = n.flags & (NF_FILLET | NF_CHAMFER);
    c._pad[0] = c._pad[1] = 0u; c.apothem = 0.0f;
    for (int j = 0; j < 11; ++j) c._pad2[j] = 0.0f;
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

__device__ float prism_sdf_c(const CNode& c, const float* table, float x, float y, float z) {
    const float* v = c.v;
    float z0 = v[4], z1 = v[5];
    if (!(z1 > z0)) return FAR32;
    if (c.kind == PROFILE_RECT) {
        float x0 = fminn(v[0], v[2]), y0 = fminn(v[1], v[3]), x1 = fmaxx(v[0], v[2]), y1 = fmaxx(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return FAR32;
        if (c.mods != 0u) {
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

__device__ float tape_sdf_c(const CNode* nodes, u32 count, const float* tables, float x, float y, float z) {
    float f = FAR32;
    float stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u) continue;
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
            tf = fminn(tf, prism_sdf_c(c, t, px, py, z));
        }
        f = (c.op == FOP_ADD) ? fminn(f, tf) : fmaxx(f, -tf);
    }
    return f;
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
    double err = (o[12] - t) / t;
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
        } else if (op == a->ops.pad || op == a->ops.pocket || op == a->ops.hole) {
            const FeatureArgs f = arena[r];
            bool bad_ref = false;
            for (int j = 0; j < NVALS; ++j) if (f.pref[j] != NO_PARAM && f.pref[j] >= kn) bad_ref = true;
            if (c.args[0] > PROFILE_POLYGON || bad_ref) st = AG_INVALID_PARAMETER;
            else if (count >= nn) st = AG_CAPACITY_EXCEEDED;
            else {
                Node n;
                n.kind = c.args[0]; n.op = (op == a->ops.pad) ? FOP_ADD : FOP_REMOVE; n.flags = NF_ACTIVE; n.src = r;
                for (int j = 0; j < NVALS; ++j) { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                tr_n[count] = n;
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
        } else if (op == a->ops.circular_pattern || op == a->ops.rect_pattern || op == a->ops.fillet || op == a->ops.chamfer || op == a->ops.deactivate) {
            u32 fi = (c.args[0] == LAST) ? count - 1u : c.args[0];
            if (count == 0u || fi >= count) st = AG_INVALID_PARAMETER; else {
                Node& n = tr_n[fi];
                const FeatureArgs f = arena[r];
                int lo, hi;
                if (op == a->ops.circular_pattern) { lo = 6; hi = 8; }
                else if (op == a->ops.rect_pattern) { lo = 6; hi = 10; }
                else if (op == a->ops.deactivate) { lo = 0; hi = 0; }
                else { lo = 10; hi = 11; }
                bool bad_ref = false;
                for (int j = lo; j < hi; ++j) if (f.pref[j] != NO_PARAM && f.pref[j] >= kn) bad_ref = true;
                bool is_profile_op = (op == a->ops.fillet || op == a->ops.chamfer);
                if (bad_ref || (is_profile_op && n.kind != PROFILE_RECT)) st = bad_ref ? AG_INVALID_PARAMETER : AG_UNSUPPORTED_OP;
                else if (op == a->ops.deactivate) { if (c.args[1] != 0u) n.flags |= NF_ACTIVE; else n.flags &= ~NF_ACTIVE; }
                else {
                    for (int j = lo; j < hi; ++j) { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                    if (op == a->ops.circular_pattern) n.flags = (n.flags & ~NF_PATTERN_RECT) | NF_PATTERN;
                    else if (op == a->ops.rect_pattern) n.flags = (n.flags & ~NF_PATTERN) | NF_PATTERN_RECT;
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
    for (u32 i = 0; i < cnt; ++i) out[i] = compile_node(nodes[i], prm, tabs + (u64)i * TABLE);
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

__device__ Iv prism_iv(const CNode& c, const float* table, Iv x, Iv y, Iv z) {
    const float* v = c.v;
    float z0 = v[4], z1 = v[5];
    Iv far = mkiv(FAR32, FAR32);
    if (!(z1 > z0)) return far;
    if (c.kind == PROFILE_RECT) {
        float x0 = fminn(v[0], v[2]), y0 = fminn(v[1], v[3]), x1 = fmaxx(v[0], v[2]), y1 = fmaxx(v[1], v[3]);
        if (!(x1 > x0 && y1 > y0)) return far;
        if (c.mods != 0u) {
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

__device__ Iv tape_iv(const CNode* nodes, u32 count, const float* tables, Iv x, Iv y, Iv z) {
    Iv f = mkiv(FAR32, FAR32);
    Iv stack[MAX_BODIES];
    int depth = 0;
    for (u32 i = 0; i < count; ++i) {
        const CNode& c = nodes[i];
        if (c.active == 0u) continue;
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
            tf = iv_min(tf, prism_iv(c, t, px, py, z));
        }
        f = (c.op == FOP_ADD) ? iv_min(f, tf) : iv_max(f, iv_neg(tf));
    }
    return f;
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

struct Surface { float f, g[3], gl, w; };

__device__ Surface surface(const CNode* nodes, u32 cnt, const float* tabs, float x, float y, float z, const float* d) {
    Surface s;
    s.f = tape_sdf_c(nodes, cnt, tabs, x, y, z);
    float h = 0.25f * fminn(fminn(d[0], d[1]), d[2]);
    float gx = tape_sdf_c(nodes, cnt, tabs, x + h, y, z) - tape_sdf_c(nodes, cnt, tabs, x - h, y, z);
    float gy = tape_sdf_c(nodes, cnt, tabs, x, y + h, z) - tape_sdf_c(nodes, cnt, tabs, x, y - h, z);
    float gz = tape_sdf_c(nodes, cnt, tabs, x, y, z + h) - tape_sdf_c(nodes, cnt, tabs, x, y, z - h);
    s.g[0] = gx; s.g[1] = gy; s.g[2] = gz;
    s.gl = sqrtf(gx * gx + gy * gy + gz * gz);
    s.w = (s.gl > 1e-30f) ? (fabsf(gx) * d[0] + fabsf(gy) * d[1] + fabsf(gz) * d[2]) / s.gl : 0.0f;
    return s;
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

__device__ float march(const CNode* nodes, u32 cnt, const float* tabs, const float* p, const float* dir, bool inside, float t0, float tmax, float hmin) {
    float t = t0;
    for (u32 i = 0; i < MARCH_STEPS; ++i) {
        if (t >= tmax) return tmax;
        float f = tape_sdf_c(nodes, cnt, tabs, p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t);
        float s = inside ? -f : f;
        if (s <= 0.0f) return t;
        t = t + ((s > hmin) ? s : hmin);
    }
    return tmax;
}

struct Probe { float area, wall, gap; bool thin, narrow, undercut; };

__device__ Probe probe(const CNode* nodes, u32 cnt, const float* tabs, const Goal& goal, const Surface& s, const float* c, const float* d) {
    Probe pr; pr.area = 0.0f; pr.wall = FAR32; pr.gap = FAR32; pr.thin = false; pr.narrow = false; pr.undercut = false;
    if (!(s.gl > 1e-30f) || !(fabsf(s.f) < 0.5f * s.w)) return pr;
    pr.area = d[0] * d[1] * d[2] / s.w;
    float n[3] = {s.g[0] / s.gl, s.g[1] / s.gl, s.g[2] / s.gl};
    float p[3] = {c[0] - n[0] * s.f, c[1] - n[1] * s.f, c[2] - n[2] * s.f};
    float dm = fminn(fminn(d[0], d[1]), d[2]);
    float hmin = 0.1f * dm;
    if (!(fabsf(tape_sdf_c(nodes, cnt, tabs, p[0], p[1], p[2])) <= hmin)) return pr;
    float mh = -hmin;
    float fin = tape_sdf_c(nodes, cnt, tabs, p[0] + n[0] * mh, p[1] + n[1] * mh, p[2] + n[2] * mh);
    float fout = tape_sdf_c(nodes, cnt, tabs, p[0] + n[0] * hmin, p[1] + n[1] * hmin, p[2] + n[2] * hmin);
    if (!(fin < 0.0f && fout > 0.0f)) return pr;
    if (goal.min_wall > 0.0) {
        float tw = (float)goal.min_wall;
        float nd[3] = {-n[0], -n[1], -n[2]};
        pr.wall = march(nodes, cnt, tabs, p, nd, true, hmin, tw, hmin);
        pr.thin = pr.wall < tw;
    }
    if (goal.min_gap > 0.0) {
        float tg = (float)goal.min_gap;
        pr.gap = march(nodes, cnt, tabs, p, n, false, hmin, tg, hmin);
        pr.narrow = pr.gap < tg;
    }
    if (goal.machining_3ax != 0u) {
        float q[3] = {p[0] + n[0] * dm, p[1] + n[1] * dm, p[2] + n[2] * dm};
        bool up = n[2] >= -0.25f;
        float dir[3] = {0.0f, 0.0f, up ? 1.0f : -1.0f};
        float tmax = up ? (float)goal.domain_hi[2] - q[2] : q[2] - (float)goal.domain_lo[2];
        if (tmax > 0.0f && tape_sdf_c(nodes, cnt, tabs, q[0], q[1], q[2]) > 0.0f) pr.undercut = march(nodes, cnt, tabs, q, dir, false, 0.0f, tmax, 0.5f * dm) < tmax;
    }
    return pr;
}

__device__ void tighten_cell(const CNode* nodes, u32 cnt, const float* tabs, const Iv* b, const float* g, float e, u32 budget, u64* olo, u64* ohi) {
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
            Iv r = tape_iv(nodes, cnt, tabs, c[0], c[1], c[2]);
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

__device__ void acc_moments(u64* cc, u32 ix, u32 iy, u32 iz, u32 sh, u64 q20) {
    u64 w = 1ull << (3u * sh);
    u64 q6 = q20 >> 14;
    u64 ss = 4ull << (2u * sh);
    u32 idx[3] = {ix, iy, iz};
    for (int k = 0; k < 3; ++k) {
        u64 c2 = ((u64)(2u * idx[k] + 1u)) << sh;
        atomicAdd((unsigned long long*)&cc[C_M1 + k], (unsigned long long)(q20 * w * c2));
        atomicAdd((unsigned long long*)&cc[C_M2 + k], (unsigned long long)(q6 * w * (12ull * c2 * c2 + ss)));
    }
    atomicAdd((unsigned long long*)&cc[C_V6], (unsigned long long)(q6 * w));
}

__device__ void acc_probe(u64* cc, const Probe& p) {
    u64 aq = qarea(p.area);
    if (aq != 0ull) atomicAdd((unsigned long long*)&cc[C_AREA], (unsigned long long)aq);
    if (p.thin) atomicAdd((unsigned long long*)&cc[C_THIN], (unsigned long long)aq);
    if (p.narrow) atomicAdd((unsigned long long*)&cc[C_GAP], (unsigned long long)aq);
    if (p.undercut) atomicAdd((unsigned long long*)&cc[C_UNDERCUT], (unsigned long long)aq);
    if (p.wall < FAR32) atomicMin((unsigned long long*)&cc[C_MIN_WALL], (unsigned long long)__float_as_uint(p.wall));
    if (p.gap < FAR32) atomicMin((unsigned long long*)&cc[C_MIN_GAP], (unsigned long long)__float_as_uint(p.gap));
}

__device__ __forceinline__ u64 enc(u32 cfg, u32 ix, u32 iy, u32 iz) { return ((u64)cfg << 32) | ((u64)ix << 20) | ((u64)iy << 10) | (u64)iz; }

__device__ bool border_hit(const Grid& gr, const CNode* nodes, u32 cnt, const float* tabs, u32 level, const u32* idx, const Iv* b, bool inside) {
    for (int ax = 0; ax < 3; ++ax) {
        u32 res = gr.r[ax] << level;
        for (u32 side = 0; side < 2u; ++side) {
            if ((side == 0u && idx[ax] != 0u) || (side == 1u && idx[ax] != res - 1u)) continue;
            if (inside) return true;
            Iv f[3] = {b[0], b[1], b[2]};
            float face = (side == 0u) ? b[ax].lo : b[ax].hi;
            f[ax] = mkiv(face, face);
            Iv r = tape_iv(nodes, cnt, tabs, f[0], f[1], f[2]);
            if (r.lo - (float)gr.eps <= 0.0f) return true;
        }
    }
    return false;
}

__device__ void acc_bbox(u32* bb, const u32* lo, u32 n, bool inside) {
    for (int k = 0; k < 3; ++k) {
        if (inside) { atomicMin(bb + k, lo[k]); atomicMax(bb + 3 + k, lo[k] + n); }
        atomicMin(bb + 6 + k, lo[k]); atomicMax(bb + 9 + k, lo[k] + n);
    }
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

AG_KERNEL void ag_ad_classify(const DopsArgs* a) {
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
    u64* counts = ag_ptr<u64>(a->counts);
    u32 sh = a->levels - 1u - l;
    float e = (float)gr.eps;
    GRID_STRIDE(j, a->capacity) { if (j >= (u64)n) cls[j] = 0u; }
    GRID_STRIDE(i, n) {
        u64 en = wl[i];
        u32 cfg = (u32)(en >> 32), ix = (u32)((en >> 20) & 1023u), iy = (u32)((en >> 10) & 1023u), iz = (u32)(en & 1023u);
        cls[i] = 0u;
        if (a->eval_trial != 0u && done[cfg] != 0u) continue;
        const CNode* tape = cnodes + (u64)cfg * nn;
        const float* tb = tabs + (u64)cfg * nn * TABLE;
        u32 c = cnt[cfg];
        Iv b[3];
        cell_box(gr, l, ix, iy, iz, b);
        Iv r = tape_iv(tape, c, tb, b[0], b[1], b[2]);
        if (r.lo - e > 0.0f) continue;
        if (r.hi + e < 0.0f) {
            atomicAdd((unsigned long long*)&counts[CS * cfg], (unsigned long long)(1ull << (3u * sh)));
            acc_moments(counts + (u64)CS * cfg, ix, iy, iz, sh, 1ull << 20);
            u32 lo[3] = {ix << sh, iy << sh, iz << sh};
            acc_bbox(ag_ptr<u32>(a->bbox) + 12u * cfg, lo, 1u << sh, true);
            u32 idx[3] = {ix, iy, iz};
            if (border_hit(gr, tape, c, tb, l, idx, b, true)) atomicExch(ag_ptr<u32>(a->domain_hit) + cfg, 1u);
            continue;
        }
        cls[i] = 1u;
    }
}

AG_KERNEL void ag_ad_resolve(const DopsArgs* a) {
    u32 nn = a->nodes;
    const u32* cnt = ag_ptr<const u32>(a->eval_trial ? a->tr_count : a->cur_count);
    const CNode* cnodes = ag_ptr<const CNode>(a->compiled);
    const float* tabs = ag_ptr<const float>(a->tables);
    const Goal goal = *ag_ptr<const Goal>(a->goal);
    Grid gr = make_grid(goal, a->coarse, a->levels);
    u32 l = a->level;
    u32 n = ag_ptr<const u32>(a->lvl)[l];
    const u64* wl = ag_ptr<const u64>((l % 2u == 0u) ? a->wl0 : a->wl1);
    const u32* cls = ag_ptr<const u32>(a->cls);
    u32* flag = ag_ptr<u32>(a->flag);
    const u32* crank = ag_ptr<const u32>(a->crank);
    const u32* start = ag_ptr<const u32>(a->start);
    u64* counts = ag_ptr<u64>(a->counts);
    u32 sh = a->levels - 1u - l;
    GRID_STRIDE(i, a->capacity) {
        flag[i] = 0u;
        if (i >= (u64)n || cls[i] != 1u) continue;
        u64 en = wl[i];
        u32 cfg = (u32)(en >> 32), ix = (u32)((en >> 20) & 1023u), iy = (u32)((en >> 10) & 1023u), iz = (u32)(en & 1023u);
        u32 rank = crank[i] - crank[start[cfg]];
        if (l + 1u < a->levels && ((u64)rank + 1ull) * 8ull <= (u64)a->cap) { flag[i] = 1u; continue; }
        const CNode* tape = cnodes + (u64)cfg * nn;
        const float* tb = tabs + (u64)cfg * nn * TABLE;
        u32 c = cnt[cfg];
        Iv b[3];
        cell_box(gr, l, ix, iy, iz, b);
        u64 w = 1ull << (3u * sh);
        u64* cc = counts + (u64)CS * cfg;
        atomicAdd((unsigned long long*)&cc[C_BOUNDARY], (unsigned long long)w);
        float d[3] = {b[0].hi - b[0].lo, b[1].hi - b[1].lo, b[2].hi - b[2].lo};
        float cxyz[3] = {0.5f * (b[0].lo + b[0].hi), 0.5f * (b[1].lo + b[1].hi), 0.5f * (b[2].lo + b[2].hi)};
        Surface sf = surface(tape, c, tb, cxyz[0], cxyz[1], cxyz[2], d);
        u64 tlo, thi;
        tighten_cell(tape, c, tb, b, sf.g, (float)gr.eps, a->tighten, &tlo, &thi);
        atomicAdd((unsigned long long*)&cc[C_BLO], (unsigned long long)(tlo * w));
        atomicAdd((unsigned long long*)&cc[C_BHI], (unsigned long long)(thi * w));
        u64 fq = fill_of(sf);
        if (fq < (tlo >> 4)) fq = tlo >> 4;
        if (fq > ((thi + 15ull) >> 4)) fq = (thi + 15ull) >> 4;
        atomicAdd((unsigned long long*)&cc[C_FILL], (unsigned long long)(fq * w));
        acc_moments(cc, ix, iy, iz, sh, fq);
        Probe pr = probe(tape, c, tb, goal, sf, cxyz, d);
        acc_probe(cc, pr);
        u32 lo[3] = {ix << sh, iy << sh, iz << sh};
        acc_bbox(ag_ptr<u32>(a->bbox) + 12u * cfg, lo, 1u << sh, false);
        u32 idx[3] = {ix, iy, iz};
        if (border_hit(gr, tape, c, tb, l, idx, b, false)) atomicExch(ag_ptr<u32>(a->domain_hit) + cfg, 1u);
    }
}

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
    bool empty = (counts[0] + counts[1]) == 0ull;
    bool hard = el[0] > goal.max_ext[0] || el[1] > goal.max_ext[1] || el[2] > goal.max_ext[2];
    double* obs = ag_ptr<double>(a->obs) + (u64)c * OBS;
    u32* cur_c = ag_ptr<u32>(a->cur_count);
    bool write = (a->eval_trial == 0u);
    double t = 0.5 * (goal.vol_lo + goal.vol_hi);
    if (a->eval_trial != 0u) {
        const CommandRecord* cmds = ag_ptr<const CommandRecord>(a->cmds) + (u64)c * rn;
        StatusRecord* status = ag_ptr<StatusRecord>(a->status) + (u64)c * rn;
        u32 acc = 0, rej = 0;
        for (u32 r = 0; r < rn; ++r) {
            if ((u32)cmds[r].opcode == a->ops.update) continue;
            if (status[r].status == AG_ACCEPTED) acc += 1u; else rej += 1u;
        }
        bool worse = en.violation > obs[OB_VIOLATION];
        bool geometry_ok = !empty && dom == 0u && !hard && !worse;
        bool commit = geometry_ok && acc > 0u;
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
        bool met = !empty && dom == 0u && ve >= goal.vol_lo && ve <= goal.vol_hi && eh[0] <= goal.max_ext[0] && eh[1] <= goal.max_ext[1] && eh[2] <= goal.max_ext[2]
            && en.violation == 0.0 && en.centroid_ok;
        obs[10] = met ? 1.0 : 0.0;
        double err = (ve - t) / t;
        obs[11] = err;
        obs[12] = ve;
        eng_write(en, obs);
        for (int k = 0; k < 3; ++k) obs[27 + k] = (eh[k] > 0.0) ? gr.lo[k] + (double)bb[6 + k] * gr.fine[k] : 0.0;
        if (met) done[c] = 1u;
        double pen = ((en.area > 0.0) ? en.violation / en.area : 0.0) + (en.centroid_ok ? 0.0 : 1.0);
        ag_ptr<double>(a->score)[c] = met ? fabs(err) : 1000.0 + fabs(err) + pen;
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
    ag_ptr<u64>(p->stamps)[(u64)round * 8u + p->stage] = t;
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
