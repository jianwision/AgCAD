//! Linear solvers: matrix-free preconditioned conjugate gradient (the Demo's constraint
//! approach), and projection onto affine hard-constraint spaces via an orthonormal nullspace.

use crate::dense;

#[derive(Clone, Debug, PartialEq)]
pub struct PcgReport {
    pub iterations: u32,
    pub residual: f64,
    pub converged: bool,
}

/// Solve `A x = b` for SPD `A` given as a matvec closure, Jacobi preconditioner `diag`.
pub fn pcg(matvec: &dyn Fn(&[f64], &mut [f64]), diag: &[f64], b: &[f64], x: &mut [f64], tol: f64, max_iter: u32) -> PcgReport {
    let n = b.len();
    let mut r = vec![0.0; n];
    let mut ap = vec![0.0; n];
    matvec(x, &mut ap);
    for i in 0..n { r[i] = b[i] - ap[i]; }
    let mut z: Vec<f64> = (0..n).map(|i| r[i] / diag[i].max(f64::MIN_POSITIVE)).collect();
    let mut p = z.clone();
    let mut rz: f64 = (0..n).map(|i| r[i] * z[i]).sum();
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
    for it in 0..max_iter {
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn / bnorm <= tol { return PcgReport { iterations: it, residual: rn, converged: true }; }
        matvec(&p, &mut ap);
        let pap: f64 = (0..n).map(|i| p[i] * ap[i]).sum();
        if pap <= 0.0 { return PcgReport { iterations: it, residual: rn, converged: false }; }
        let alpha = rz / pap;
        for i in 0..n { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
        for i in 0..n { z[i] = r[i] / diag[i].max(f64::MIN_POSITIVE); }
        let rz2: f64 = (0..n).map(|i| r[i] * z[i]).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
    }
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    PcgReport { iterations: max_iter, residual: rn, converged: rn / bnorm <= tol }
}

/// Affine constraint space `{x | A x = b}` as `x = x_p + Z y`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineSpace {
    pub n: usize,
    pub k: usize,
    pub xp: Vec<f64>,
    /// `n×k` orthonormal basis (row-major).
    pub z: Vec<f64>,
    /// Max |A x_p - b| (hard-constraint feasibility evidence).
    pub residual: f64,
}

/// Build the feasible space of `m×n` hard equalities (minimum-norm particular solution).
pub fn affine_space(a: &[f64], m: usize, n: usize, b: &[f64]) -> AffineSpace {
    let (sv, v) = dense::svd(a, m, n);
    let r = dense::rank(&sv, m, n, None);
    // x_p = V_r Σ_r⁻¹ U_rᵀ b with U_r = A V_r Σ_r⁻¹.
    let mut xp = vec![0.0; n];
    for j in 0..r {
        let mut ucol = vec![0.0; m];
        for i in 0..m { ucol[i] = (0..n).map(|q| a[i * n + q] * v[q * n + j]).sum::<f64>() / sv[j]; }
        let coef = (0..m).map(|i| ucol[i] * b[i]).sum::<f64>() / sv[j];
        for q in 0..n { xp[q] += coef * v[q * n + j]; }
    }
    let k = n - r;
    let mut z = vec![0.0; n * k];
    for q in 0..n { for j in 0..k { z[q * k + j] = v[q * n + r + j]; } }
    let mut ax = vec![0.0; m];
    dense::matvec(a, m, n, &xp, &mut ax);
    let residual = ax.iter().zip(b).fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()));
    AffineSpace { n, k, xp, z, residual }
}

impl AffineSpace {
    /// Orthogonal projection of `x` onto the space.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = x.iter().zip(&self.xp).map(|(a, b)| a - b).collect();
        let mut out = self.xp.clone();
        for j in 0..self.k {
            let c: f64 = (0..self.n).map(|q| self.z[q * self.k + j] * d[q]).sum();
            for q in 0..self.n { out[q] += c * self.z[q * self.k + j]; }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pcg_and_projection() {
        let a = [4.0, 1.0, 1.0, 3.0];
        let mv = |x: &[f64], y: &mut [f64]| dense::matvec(&a, 2, 2, x, y);
        let mut x = vec![0.0, 0.0];
        let rep = pcg(&mv, &[4.0, 3.0], &[1.0, 2.0], &mut x, 1e-12, 50);
        assert!(rep.converged);
        let sp = affine_space(&[1.0, 1.0, 0.0], 1, 3, &[2.0]);
        let p = sp.project(&[5.0, 5.0, 7.0]);
        assert!((p[0] + p[1] - 2.0).abs() < 1e-12 && (p[2] - 7.0).abs() < 1e-12);
    }
}
