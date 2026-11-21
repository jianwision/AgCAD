//! Small dense linear algebra (row-major `n×n` / `m×n`), deterministic operation order.
//! Shared by host solvers and as CPU reference for batched GPU kernels (`cu/dense.cuh`
//! implements the same loops).

/// In-place Cholesky `A = L Lᵀ` (lower triangle). Returns `false` if not positive definite.
pub fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// Solve `L Lᵀ x = b` given the Cholesky factor in `l`.
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Householder QR of `m×n` `a` (m ≥ n) in place; returns `tau`. R in the upper triangle.
pub fn qr(a: &mut [f64], m: usize, n: usize) -> Vec<f64> {
    let mut tau = vec![0.0; n];
    for k in 0..n.min(m) {
        let mut norm = 0.0;
        for i in k..m {
            norm += a[i * n + k] * a[i * n + k];
        }
        let norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[k * n + k] > 0.0 { -norm } else { norm };
        let v0 = a[k * n + k] - alpha;
        a[k * n + k] = alpha;
        for i in k + 1..m {
            a[i * n + k] /= v0;
        }
        tau[k] = -v0 / alpha;
        for j in k + 1..n {
            let mut s = a[k * n + j];
            for i in k + 1..m {
                s += a[i * n + k] * a[i * n + j];
            }
            s *= tau[k];
            a[k * n + j] -= s;
            for i in k + 1..m {
                a[i * n + j] -= s * a[i * n + k];
            }
        }
    }
    tau
}

/// One-sided Jacobi SVD of `m×n` `a`: returns singular values (descending) and `V` (n×n,
/// row-major, columns = right singular vectors). Deterministic sweep order.
pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    alpha += u[i * n + p] * u[i * n + p];
                    beta += u[i * n + q] * u[i * n + q];
                    gamma += u[i * n + p] * u[i * n + q];
                }
                if gamma == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(f64::MIN_POSITIVE));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (u[i * n + p], u[i * n + q]);
                    u[i * n + p] = c * x - s * y;
                    u[i * n + q] = s * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * x - s * y;
                    v[i * n + q] = s * x + c * y;
                }
            }
        }
        if off < 1e-15 {
            break;
        }
    }
    let mut sv: Vec<(f64, usize)> = (0..n).map(|j| ((0..m).map(|i| u[i * n + j] * u[i * n + j]).sum::<f64>().sqrt(), j)).collect();
    sv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let mut vs = vec![0.0; n * n];
    for (newj, (_, oldj)) in sv.iter().enumerate() {
        for i in 0..n {
            vs[i * n + newj] = v[i * n + oldj];
        }
    }
    (sv.into_iter().map(|x| x.0).collect(), vs)
}

/// Numerical rank with relative tolerance (default `max(m,n)·eps·σ₀`).
pub fn rank(sv: &[f64], m: usize, n: usize, rtol: Option<f64>) -> usize {
    let s0 = sv.first().copied().unwrap_or(0.0);
    let tol = rtol.unwrap_or(m.max(n) as f64 * f64::EPSILON) * s0;
    sv.iter().filter(|s| **s > tol).count()
}

/// Orthonormal nullspace basis of `m×n` `a` (columns, row-major `n×k`).
pub fn nullspace(a: &[f64], m: usize, n: usize) -> (Vec<f64>, usize) {
    let (sv, v) = svd(a, m, n);
    let r = rank(&sv, m, n, None);
    let k = n - r;
    let mut z = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..k {
            z[i * k + j] = v[i * n + r + j];
        }
    }
    (z, k)
}

pub fn matvec(a: &[f64], m: usize, n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cholesky_solves_spd() {
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        assert!(cholesky(&mut a, 2));
        let mut b = vec![2.0, 1.0];
        cholesky_solve(&a, 2, &mut b);
        assert!((4.0 * b[0] + 2.0 * b[1] - 2.0).abs() < 1e-12 && (2.0 * b[0] + 3.0 * b[1] - 1.0).abs() < 1e-12);
    }
    #[test]
    fn svd_rank_and_nullspace() {
        let a = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        let (sv, _) = svd(&a, 2, 3);
        assert_eq!(rank(&sv, 2, 3, None), 1);
        let (z, k) = nullspace(&a, 2, 3);
        assert_eq!(k, 2);
        for j in 0..k {
            let r: f64 = (0..3).map(|i| a[i] * z[i * k + j]).sum();
            assert!(r.abs() < 1e-12);
        }
    }
}
