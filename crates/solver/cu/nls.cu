// Batched Levenberg–Marquardt, one thread per configuration. Mirrors
// agcad_solver::nls::lm_solve operation by operation (AG-06-006).
#include "ag_common.cuh"

#define NLS_MAX_VARS 32
#define NLS_MAX_RES 64
#define NLS_MAX_NODES 1024

struct VmInstr { unsigned short op, pad; unsigned int a, b, c; };
struct NlsParams {
    u64 r_instrs, r_consts, j_instrs, j_consts;
    u64 x;        // double[n][configs], in/out
    u64 data;     // double[n_data][configs]
    u64 status;   // u16[configs]
    u64 iters;    // u32[configs]
    u64 cost;     // double[configs]
    u32 r_n, j_n, n, m, n_data, configs, max_iter, _pad;
    double tol, xtol, lambda0;
};

__device__ double nls_un(unsigned short o, double a);
__device__ double nls_bin(unsigned short o, double a, double b);

__device__ __forceinline__ double nls_un(unsigned short o, double a) {
    switch (o) {
        case 0: return -a; case 1: return fabs(a); case 2: return sqrt(a); case 3: return sin(a);
        case 4: return cos(a); case 5: return tan(a); case 6: return asin(a); case 7: return acos(a);
        case 8: return atan(a); case 9: return exp(a); case 10: return log(a); case 11: return floor(a);
        case 12: return ceil(a); default: return a == 0.0 ? 1.0 : 0.0;
    }
}
__device__ __forceinline__ double nls_bin(unsigned short o, double a, double b) {
    switch (o) {
        case 0: return a + b; case 1: return a - b; case 2: return a * b; case 3: return a / b;
        case 4: return b < a ? b : a; case 5: return b > a ? b : a; case 6: return pow(a, b); case 7: return atan2(a, b);
        case 8: return a < b ? 1.0 : 0.0; case 9: return a <= b ? 1.0 : 0.0; case 10: return a > b ? 1.0 : 0.0;
        case 11: return a >= b ? 1.0 : 0.0; case 12: return a == b ? 1.0 : 0.0; case 13: return a != b ? 1.0 : 0.0;
        case 14: return (a != 0.0 && b != 0.0) ? 1.0 : 0.0; case 15: return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
        default: return a - b * floor(a / b);
    }
}

// Interpret a VM program; unknowns from x[], data from d[] (strided by configs).
__device__ void nls_exec(const VmInstr* ins, u32 nins, const double* k, const double* x, const double* d, u32 c, u32 configs, double* out, double* r) {
    u32 n = 0;
    for (u32 i = 0; i < nins; ++i) {
        VmInstr q = ins[i];
        double v;
        if (q.op == 0xFF00) v = k[q.a];
        else if (q.op == 0xFF01) v = x[q.a];
        else if (q.op == 0xFF02) v = d[(u64)q.a * configs + c];
        else if (q.op == 0xFF80) v = r[q.a] != 0.0 ? r[q.b] : r[q.c];
        else if (q.op == 0xFF81) v = ag_value_noise(r[q.a], r[q.b], r[q.c]);
        else if (q.op == 0xFFF0) { out[q.b] = r[q.a]; continue; }
        else if (q.op >= 0xFF40 && q.op < 0xFF40 + 17) v = nls_bin(q.op - 0xFF40, r[q.a], r[q.b]);
        else v = nls_un(q.op - 0xFF10, r[q.a]);
        r[n++] = v;
    }
}

__device__ bool nls_cholesky(double* a, u32 n) {
    for (u32 j = 0; j < n; ++j) {
        double dd = a[j * n + j];
        for (u32 k = 0; k < j; ++k) dd -= a[j * n + k] * a[j * n + k];
        if (!(dd > 0.0)) return false;
        dd = sqrt(dd);
        a[j * n + j] = dd;
        for (u32 i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (u32 k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s / dd;
        }
    }
    return true;
}

__device__ void nls_cholesky_solve(const double* l, u32 n, double* b) {
    for (u32 i = 0; i < n; ++i) { double s = b[i]; for (u32 k = 0; k < i; ++k) s -= l[i * n + k] * b[k]; b[i] = s / l[i * n + i]; }
    for (int i = (int)n - 1; i >= 0; --i) { double s = b[i]; for (u32 k = i + 1; k < n; ++k) s -= l[k * n + i] * b[k]; b[i] = s / l[i * n + i]; }
}

AG_KERNEL void ag_nls_lm(const NlsParams* p) {
    u32 c = AG_TID;
    if (c >= p->configs) return;
    const u32 n = p->n, m = p->m;
    double x[NLS_MAX_VARS], xn[NLS_MAX_VARS], delta[NLS_MAX_VARS], g[NLS_MAX_VARS];
    double r[NLS_MAX_RES], rn[NLS_MAX_RES];
    double J[NLS_MAX_RES * NLS_MAX_VARS];
    double A[NLS_MAX_VARS * NLS_MAX_VARS];
    double reg[NLS_MAX_NODES];
    double* X = ag_ptr<double>(p->x);
    const double* D = ag_ptr<const double>(p->data);
    const VmInstr* RI = ag_ptr<const VmInstr>(p->r_instrs);
    const VmInstr* JI = ag_ptr<const VmInstr>(p->j_instrs);
    const double* RK = ag_ptr<const double>(p->r_consts);
    const double* JK = ag_ptr<const double>(p->j_consts);
    for (u32 k = 0; k < n; ++k) x[k] = X[(u64)k * p->configs + c];
    double lambda = p->lambda0;
    nls_exec(RI, p->r_n, RK, x, D, c, p->configs, r, reg);
    double cost = 0.0;
    for (u32 i = 0; i < m; ++i) cost += r[i] * r[i];
    cost *= 0.5;
    unsigned short status = AG_SOLVER_DIVERGED;
    u32 it = 0;
    while (it < p->max_iter) {
        double rmax = 0.0;
        for (u32 i = 0; i < m; ++i) rmax = fmax(rmax, fabs(r[i]));
        if (!isfinite(rmax)) { status = AG_NONFINITE_PARAMETER; break; }
        if (rmax <= p->tol) { status = AG_ACCEPTED; break; }
        nls_exec(JI, p->j_n, JK, x, D, c, p->configs, J, reg);
        for (u32 a = 0; a < n; ++a) {
            for (u32 b = 0; b < n; ++b) { double s = 0.0; for (u32 i = 0; i < m; ++i) s += J[i * n + a] * J[i * n + b]; A[a * n + b] = s; }
            double s = 0.0; for (u32 i = 0; i < m; ++i) s += J[i * n + a] * r[i]; g[a] = s;
        }
        for (u32 a = 0; a < n; ++a) A[a * n + a] += lambda * A[a * n + a] + 1e-12;
        for (u32 a = 0; a < n; ++a) delta[a] = -g[a];
        ++it;
        if (!nls_cholesky(A, n)) { lambda *= 10.0; continue; }
        nls_cholesky_solve(A, n, delta);
        for (u32 a = 0; a < n; ++a) xn[a] = x[a] + delta[a];
        nls_exec(RI, p->r_n, RK, xn, D, c, p->configs, rn, reg);
        double cn = 0.0; for (u32 i = 0; i < m; ++i) cn += rn[i] * rn[i]; cn *= 0.5;
        double dn = 0.0, xnorm = 0.0;
        for (u32 a = 0; a < n; ++a) { dn += delta[a] * delta[a]; xnorm += x[a] * x[a]; }
        dn = sqrt(dn); xnorm = sqrt(xnorm);
        if (cn < cost) {
            for (u32 a = 0; a < n; ++a) x[a] = xn[a];
            for (u32 i = 0; i < m; ++i) r[i] = rn[i];
            cost = cn;
            lambda = fmax(lambda / 10.0, 1e-12);
            if (dn <= p->xtol * (xnorm + p->xtol)) {
                double rm = 0.0; for (u32 i = 0; i < m; ++i) rm = fmax(rm, fabs(r[i]));
                status = (rm <= p->tol) ? AG_ACCEPTED : AG_OVER_CONSTRAINED;
                break;
            }
        } else {
            lambda *= 10.0;
            if (lambda > 1e16) break;
        }
    }
    for (u32 k = 0; k < n; ++k) X[(u64)k * p->configs + c] = x[k];
    ag_ptr<unsigned short>(p->status)[c] = status;
    ag_ptr<u32>(p->iters)[c] = it;
    ag_ptr<double>(p->cost)[c] = cost;
}
