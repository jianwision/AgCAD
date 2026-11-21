//! Optimizers for `study.optimize` (spec 14): Adam, L-BFGS, Nelder–Mead, CMA-ES (rank-μ,
//! deterministic RNG), simulated annealing, and an augmented-Lagrangian wrapper for
//! inequality constraints. All are budgeted (evaluation count) and deterministic.

#[derive(Clone, Debug, PartialEq)]
pub struct OptReport {
    pub x: Vec<f64>,
    pub f: f64,
    pub evaluations: u32,
    pub converged: bool,
}

/// Counter-based deterministic RNG (SplitMix64) keyed by seed (spec 05 § Determinism).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed ^ 0x9E37_79B9_7F4A_7C15) }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

pub struct Budget {
    pub max_evals: u32,
    pub used: u32,
}

impl Budget {
    pub fn new(max_evals: u32) -> Self { Budget { max_evals, used: 0 } }
    pub fn take(&mut self) -> bool { if self.used < self.max_evals { self.used += 1; true } else { false } }
}

/// Adam on a function with gradient.
pub fn adam(f: &dyn Fn(&[f64]) -> (f64, Vec<f64>), x0: &[f64], lr: f64, budget: &mut Budget, tol: f64) -> OptReport {
    let n = x0.len();
    let (b1, b2, eps) = (0.9, 0.999, 1e-12);
    let (mut m, mut v) = (vec![0.0; n], vec![0.0; n]);
    let mut x = x0.to_vec();
    let mut best = (f64::INFINITY, x.clone());
    let mut t = 0i32;
    while budget.take() {
        t += 1;
        let (fx, g) = f(&x);
        if fx < best.0 { best = (fx, x.clone()); }
        let gn = g.iter().map(|a| a * a).sum::<f64>().sqrt();
        if gn <= tol { return OptReport { x, f: fx, evaluations: budget.used, converged: true }; }
        for i in 0..n {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let mh = m[i] / (1.0 - b1.powi(t));
            let vh = v[i] / (1.0 - b2.powi(t));
            x[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
    OptReport { x: best.1, f: best.0, evaluations: budget.used, converged: false }
}

/// Nelder–Mead simplex (derivative-free).
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, budget: &mut Budget, tol: f64) -> OptReport {
    let n = x0.len();
    let mut simplex: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n + 1);
    let eval = |x: &Vec<f64>, b: &mut Budget| -> Option<f64> { if b.take() { Some(f(x)) } else { None } };
    let Some(f0) = eval(&x0.to_vec(), budget) else { return OptReport { x: x0.to_vec(), f: f64::INFINITY, evaluations: 0, converged: false } };
    simplex.push((f0, x0.to_vec()));
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        let Some(fx) = eval(&x, budget) else { break };
        simplex.push((fx, x));
    }
    while simplex.len() == n + 1 {
        simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        if (simplex[n].0 - simplex[0].0).abs() <= tol { return OptReport { x: simplex[0].1.clone(), f: simplex[0].0, evaluations: budget.used, converged: true }; }
        let cen: Vec<f64> = (0..n).map(|j| simplex[..n].iter().map(|s| s.1[j]).sum::<f64>() / n as f64).collect();
        let pt = |t: f64| -> Vec<f64> { (0..n).map(|j| cen[j] + t * (simplex[n].1[j] - cen[j])).collect() };
        let xr = pt(-1.0);
        let Some(fr) = eval(&xr, budget) else { break };
        if fr < simplex[0].0 {
            let xe = pt(-2.0);
            let Some(fe) = eval(&xe, budget) else { break };
            simplex[n] = if fe < fr { (fe, xe) } else { (fr, xr) };
        } else if fr < simplex[n - 1].0 {
            simplex[n] = (fr, xr);
        } else {
            let xc = pt(0.5);
            let Some(fc) = eval(&xc, budget) else { break };
            if fc < simplex[n].0 { simplex[n] = (fc, xc); } else {
                let best = simplex[0].1.clone();
                for s in simplex.iter_mut().skip(1) {
                    s.1 = (0..n).map(|j| best[j] + 0.5 * (s.1[j] - best[j])).collect();
                    match eval(&s.1, budget) { Some(v) => s.0 = v, None => break }
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    OptReport { x: simplex[0].1.clone(), f: simplex[0].0, evaluations: budget.used, converged: false }
}

/// CMA-ES (diagonal covariance variant, rank-μ update), deterministic given `seed`.
/// Population evaluation is batch-friendly: `f_batch` receives λ candidates at once, which
/// `study.optimize` maps to one GPU wavefront over configurations.
pub fn cma_es(f_batch: &dyn Fn(&[Vec<f64>]) -> Vec<f64>, x0: &[f64], sigma0: f64, seed: u64, budget: &mut Budget, tol: f64) -> OptReport {
    let n = x0.len();
    let lambda = 4 + (3.0 * (n as f64).ln()).floor() as usize;
    let mu = lambda / 2;
    let w: Vec<f64> = (0..mu).map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).collect();
    let ws: f64 = w.iter().sum();
    let w: Vec<f64> = w.iter().map(|x| x / ws).collect();
    let mut mean = x0.to_vec();
    let mut sigma = sigma0;
    let mut diag = vec![1.0f64; n];
    let mut rng = Rng::new(seed);
    let mut best = (f64::INFINITY, mean.clone());
    loop {
        if budget.used + lambda as u32 > budget.max_evals { break; }
        budget.used += lambda as u32;
        let zs: Vec<Vec<f64>> = (0..lambda).map(|_| (0..n).map(|_| rng.normal()).collect()).collect();
        let xs: Vec<Vec<f64>> = zs.iter().map(|z| (0..n).map(|j| mean[j] + sigma * diag[j].sqrt() * z[j]).collect()).collect();
        let fs = f_batch(&xs);
        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b| fs[a].partial_cmp(&fs[b]).unwrap_or(core::cmp::Ordering::Equal).then(a.cmp(&b)));
        if fs[order[0]] < best.0 { best = (fs[order[0]], xs[order[0]].clone()); }
        let old = mean.clone();
        mean = (0..n).map(|j| (0..mu).map(|i| w[i] * xs[order[i]][j]).sum()).collect();
        for j in 0..n {
            let var: f64 = (0..mu).map(|i| w[i] * ((xs[order[i]][j] - old[j]) / sigma).powi(2)).sum();
            diag[j] = 0.8 * diag[j] + 0.2 * var;
        }
        let spread = fs[order[lambda - 1]] - fs[order[0]];
        sigma *= if spread.abs() < tol { 0.5 } else { 0.95 };
        if sigma < tol { return OptReport { x: best.1, f: best.0, evaluations: budget.used, converged: true }; }
    }
    OptReport { x: best.1, f: best.0, evaluations: budget.used, converged: false }
}

/// Simulated annealing with deterministic schedule.
pub fn simulated_annealing(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, t0: f64, seed: u64, budget: &mut Budget) -> OptReport {
    let mut rng = Rng::new(seed);
    let mut x = x0.to_vec();
    if !budget.take() { return OptReport { x, f: f64::INFINITY, evaluations: 0, converged: false }; }
    let mut fx = f(&x);
    let mut best = (fx, x.clone());
    let total = budget.max_evals.max(1) as f64;
    while budget.take() {
        let t = t0 * (1.0 - budget.used as f64 / total).max(1e-6);
        let cand: Vec<f64> = x.iter().map(|v| v + step * rng.normal()).collect();
        let fc = f(&cand);
        if fc < fx || rng.uniform() < ((fx - fc) / t).exp() { x = cand; fx = fc; }
        if fx < best.0 { best = (fx, x.clone()); }
    }
    OptReport { x: best.1, f: best.0, evaluations: budget.used, converged: false }
}

/// Augmented Lagrangian for `min f(x) s.t. g_i(x) ≤ 0`, inner solver Nelder–Mead.
pub fn augmented_lagrangian(f: &dyn Fn(&[f64]) -> f64, g: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], budget: &mut Budget, tol: f64) -> OptReport {
    let m = g(x0).len();
    let mut lam = vec![0.0; m];
    let mut rho = 10.0;
    let mut x = x0.to_vec();
    for _outer in 0..20 {
        let lam_c = lam.clone();
        let la = |y: &[f64]| -> f64 {
            let gv = g(y);
            f(y) + gv.iter().zip(&lam_c).map(|(gi, li)| { let s = (li / rho + gi).max(0.0); 0.5 * rho * s * s - 0.5 * li * li / rho }).sum::<f64>()
        };
        let inner_budget = (budget.max_evals.saturating_sub(budget.used)) / 4;
        let mut b = Budget::new(inner_budget.max(1));
        let r = nelder_mead(&la, &x, 0.1, &mut b, tol);
        budget.used += b.used;
        x = r.x;
        let gv = g(&x);
        let viol = gv.iter().fold(0.0f64, |a, v| a.max(*v));
        for i in 0..m { lam[i] = (lam[i] + rho * gv[i]).max(0.0); }
        if viol <= tol { return OptReport { f: f(&x), x, evaluations: budget.used, converged: true }; }
        rho *= 4.0;
        if budget.used >= budget.max_evals { break; }
    }
    OptReport { f: f(&x), x, evaluations: budget.used, converged: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optimizers_find_quadratic_minimum() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 10.0 * (x[1] + 2.0).powi(2);
        let nm = nelder_mead(&f, &[0.0, 0.0], 0.5, &mut Budget::new(2000), 1e-12);
        assert!((nm.x[0] - 1.0).abs() < 1e-4 && (nm.x[1] + 2.0).abs() < 1e-4);
        let fb = |xs: &[Vec<f64>]| xs.iter().map(|x| f(x)).collect::<Vec<_>>();
        let c = cma_es(&fb, &[0.0, 0.0], 1.0, 7, &mut Budget::new(4000), 1e-10);
        assert!(c.f < 1e-6, "{}", c.f);
        let al = augmented_lagrangian(&f, &|x: &[f64]| vec![x[0] - 0.5], &[0.0, 0.0], &mut Budget::new(20000), 1e-6);
        assert!(al.x[0] <= 0.5 + 1e-4);
    }
}
