//! Batched nonlinear least squares: Levenberg–Marquardt (spec 14 § Constraint solving).
//!
//! Problem: residual graph `r(x; d)` where graph **inputs** are the unknowns `x` (per
//! configuration) and graph **params** are fixed data `d` (per configuration). The Jacobian
//! graph is derived symbolically (`agcad_rtc::autodiff::gradient_graph`). Both are encoded as
//! VM programs; the GPU kernel (`cu/nls.cu`, one thread per configuration, ≤ 32 unknowns,
//! ≤ 64 residuals) and [`lm_solve`] execute the identical algorithm (AG-06-006).

use crate::dense;
use agcad_driver::AgStatus;
use agcad_rtc::autodiff::gradient_graph;
use agcad_rtc::expr::{Dim, ExprGraph, Node};
use agcad_vm::{encode, exec, VmProgram};

pub const MAX_VARS: usize = 32;
pub const MAX_RES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmOptions {
    pub max_iter: u32,
    /// Converged when max |rᵢ| ≤ tol (internal units).
    pub tol: f64,
    /// Converged when |δ| ≤ xtol (|x| + xtol).
    pub xtol: f64,
    pub lambda0: f64,
}

impl Default for LmOptions {
    fn default() -> Self { LmOptions { max_iter: 100, tol: 1e-9, xtol: 1e-14, lambda0: 1e-3 } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LmReport {
    pub status: AgStatus,
    pub x: Vec<f64>,
    pub iterations: u32,
    pub cost: f64,
    pub max_residual: f64,
    /// Numerical rank of J at the solution; `dof = n - rank`.
    pub rank: usize,
    pub dof: usize,
}

/// Merge per-output gradient graphs into one Jacobian graph (row-major outputs `m×n`).
pub fn jacobian_graph(r: &ExprGraph) -> ExprGraph {
    let mut out = ExprGraph { input_names: r.input_names.clone(), input_dims: r.input_dims.clone(), param_names: r.param_names.clone(), param_dims: r.param_dims.clone(), ..Default::default() };
    for o in 0..r.outputs.len() {
        let g = gradient_graph(r, o);
        let base = out.nodes.len() as u32;
        for n in &g.nodes {
            let shifted = match n.clone() {
                Node::Un(op, a) => Node::Un(op, a + base),
                Node::Bin(op, a, b) => Node::Bin(op, a + base, b + base),
                Node::Select(c, a, b) => Node::Select(c + base, a + base, b + base),
                Node::Noise(x, y, z) => Node::Noise(x + base, y + base, z + base),
                other => other,
            };
            out.nodes.push(shifted);
            out.dims.push(Dim::NONE);
        }
        out.outputs.extend(g.outputs.iter().map(|x| x + base));
    }
    out
}

pub struct Compiled {
    pub n: usize,
    pub m: usize,
    pub residual: VmProgram,
    pub jacobian: VmProgram,
}

pub fn compile(r: &ExprGraph) -> Result<Compiled, AgStatus> {
    let n = r.input_names.len();
    let m = r.outputs.len();
    if n > MAX_VARS || m > MAX_RES { return Err(AgStatus::CapacityExceeded); }
    Ok(Compiled { n, m, residual: encode(r)?, jacobian: encode(&jacobian_graph(r))? })
}

fn eval(p: &VmProgram, x: &[f64], d: &[f64], out: &mut [f64]) {
    exec(&p.instrs, &p.consts, &|k| x[k as usize], &|k| d[k as usize], &mut |o, v| out[o as usize] = v);
}

/// Solve one configuration (CPU reference; identical to the device kernel).
pub fn lm_solve(c: &Compiled, x0: &[f64], data: &[f64], opt: LmOptions) -> LmReport {
    let (n, m) = (c.n, c.m);
    let mut x = x0.to_vec();
    let mut r = vec![0.0; m];
    let mut j = vec![0.0; m * n];
    let mut lambda = opt.lambda0;
    eval(&c.residual, &x, data, &mut r);
    let mut cost = 0.5 * r.iter().map(|v| v * v).sum::<f64>();
    let mut status = AgStatus::SolverDiverged;
    let mut it = 0u32;
    while it < opt.max_iter {
        let rmax = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if !rmax.is_finite() { status = AgStatus::NonfiniteParameter; break; }
        if rmax <= opt.tol { status = AgStatus::Accepted; break; }
        eval(&c.jacobian, &x, data, &mut j);
        // A = JᵀJ + λ diag(JᵀJ) + 1e-12 I ; g = Jᵀ r
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for p in 0..n {
            for q in 0..n {
                let mut s = 0.0;
                for i in 0..m { s += j[i * n + p] * j[i * n + q]; }
                a[p * n + q] = s;
            }
            let mut s = 0.0;
            for i in 0..m { s += j[i * n + p] * r[i]; }
            g[p] = s;
        }
        for p in 0..n { a[p * n + p] += lambda * a[p * n + p] + 1e-12; }
        let mut delta: Vec<f64> = g.iter().map(|v| -v).collect();
        it += 1;
        if !dense::cholesky(&mut a, n) { lambda *= 10.0; continue; }
        dense::cholesky_solve(&a, n, &mut delta);
        let xn: Vec<f64> = x.iter().zip(&delta).map(|(a, b)| a + b).collect();
        let mut rn = vec![0.0; m];
        eval(&c.residual, &xn, data, &mut rn);
        let cn = 0.5 * rn.iter().map(|v| v * v).sum::<f64>();
        let dnorm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let xnorm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if cn < cost {
            x = xn; r = rn; cost = cn;
            lambda = (lambda / 10.0).max(1e-12);
            if dnorm <= opt.xtol * (xnorm + opt.xtol) {
                let rmax = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                status = if rmax <= opt.tol { AgStatus::Accepted } else { AgStatus::OverConstrained };
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e16 { break; }
        }
    }
    eval(&c.jacobian, &x, data, &mut j);
    let (sv, _) = dense::svd(&j, m, n);
    let rank = dense::rank(&sv, m, n, Some(1e-10));
    let max_residual = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    LmReport { status, x, iterations: it, cost, max_residual, rank, dof: n - rank }
}

/// Batched host solve over configurations (SoA: `x0[var][cfg]`, `data[k][cfg]`).
pub fn lm_solve_batch(c: &Compiled, x0: &[f64], data: &[f64], n_data: usize, configs: usize, opt: LmOptions) -> Vec<LmReport> {
    (0..configs).map(|ci| {
        let x: Vec<f64> = (0..c.n).map(|k| x0[k * configs + ci]).collect();
        let d: Vec<f64> = (0..n_data).map(|k| data[k * configs + ci]).collect();
        lm_solve(c, &x, &d, opt)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_point_on_circle_and_line() {
        // unknowns (x, y): x² + y² - r² = 0 ; y - x = 0  → x = y = r/√2
        let src = "#[residual] fn r(x: f64, y: f64, rr: f64) -> f64 { x }";
        let _ = src;
        let mut g = ExprGraph::default();
        let x = g.input("x", Dim::NONE);
        let y = g.input("y", Dim::NONE);
        let r = g.param("r", Dim::NONE);
        let xx = g.bin(agcad_rtc::expr::Bin::Mul, x, x).unwrap();
        let yy = g.bin(agcad_rtc::expr::Bin::Mul, y, y).unwrap();
        let rr = g.bin(agcad_rtc::expr::Bin::Mul, r, r).unwrap();
        let s = g.bin(agcad_rtc::expr::Bin::Add, xx, yy).unwrap();
        let c1 = g.bin(agcad_rtc::expr::Bin::Sub, s, rr).unwrap();
        let c2 = g.bin(agcad_rtc::expr::Bin::Sub, y, x).unwrap();
        g.outputs = vec![c1, c2];
        let c = compile(&g).unwrap();
        let rep = lm_solve(&c, &[1.0, 0.2], &[10.0], LmOptions::default());
        assert_eq!(rep.status, AgStatus::Accepted, "{rep:?}");
        assert!((rep.x[0] - 10.0 / 2f64.sqrt()).abs() < 1e-8);
        assert_eq!(rep.dof, 0);
    }
}
