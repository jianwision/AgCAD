//! Root finding: Newton (1-D and N-D) and interval Newton with certified enclosures
//! (`bounded` evidence; used by SSI seeding and point inversion in agcad-geo).

use crate::dense;
use crate::interval::Interval;

#[derive(Clone, Debug, PartialEq)]
pub enum RootResult {
    Converged { x: Vec<f64>, iterations: u32, residual: f64 },
    Diverged { x: Vec<f64>, iterations: u32, residual: f64 },
}

/// Damped Newton for `F(x) = 0` with Jacobian closure.
pub fn newton(f: &dyn Fn(&[f64], &mut [f64]), jac: &dyn Fn(&[f64], &mut [f64]), x0: &[f64], tol: f64, max_iter: u32) -> RootResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = vec![0.0; n];
    let mut j = vec![0.0; n * n];
    f(&x, &mut fx);
    let norm = |v: &[f64]| v.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    for it in 0..max_iter {
        let r = norm(&fx);
        if r <= tol { return RootResult::Converged { x, iterations: it, residual: r }; }
        jac(&x, &mut j);
        // Solve J d = -f via normal equations + Cholesky (small n).
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for p in 0..n { for q in 0..n { a[p * n + q] = (0..n).map(|i| j[i * n + p] * j[i * n + q]).sum(); } g[p] = -(0..n).map(|i| j[i * n + p] * fx[i]).sum::<f64>(); a[p * n + p] += 1e-14; }
        if !dense::cholesky(&mut a, n) { return RootResult::Diverged { x, iterations: it, residual: r }; }
        dense::cholesky_solve(&a, n, &mut g);
        let mut step = 1.0;
        loop {
            let xn: Vec<f64> = x.iter().zip(&g).map(|(a, d)| a + step * d).collect();
            let mut fn_ = vec![0.0; n];
            f(&xn, &mut fn_);
            if norm(&fn_) < r || step < 1e-6 { x = xn; fx = fn_; break; }
            step *= 0.5;
        }
    }
    let r = norm(&fx);
    if r <= tol { RootResult::Converged { x, iterations: max_iter, residual: r } } else { RootResult::Diverged { x, iterations: max_iter, residual: r } }
}

/// Interval Newton in 1-D: returns certified enclosures of every root in `x` (each width
/// ≤ `tol`), or an empty list if `f` provably has no root. `fi`/`dfi` are interval extensions.
pub fn interval_newton_1d(fi: &dyn Fn(Interval) -> Interval, dfi: &dyn Fn(Interval) -> Interval, x: Interval, tol: f64, max_depth: u32) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut work = vec![(x, 0u32)];
    while let Some((iv, depth)) = work.pop() {
        let fx = fi(iv);
        if !fx.contains_zero() { continue; }
        if iv.width() <= tol || depth >= max_depth { out.push(iv); continue; }
        let d = dfi(iv);
        if d.contains_zero() {
            let m = iv.mid();
            work.push((Interval::new(m, iv.hi), depth + 1));
            work.push((Interval::new(iv.lo, m), depth + 1));
            continue;
        }
        let m = iv.mid();
        let fm = fi(Interval::point(m));
        let n = Interval::sub(Interval::point(m), Interval::div(fm, d));
        let next = Interval::new(n.lo.max(iv.lo), n.hi.min(iv.hi));
        if next.lo > next.hi { continue; }
        if next.width() > 0.9 * iv.width() {
            let mm = next.mid();
            work.push((Interval::new(mm, next.hi), depth + 1));
            work.push((Interval::new(next.lo, mm), depth + 1));
        } else {
            work.push((next, depth + 1));
        }
    }
    out.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interval_newton_finds_both_roots() {
        let fi = |x: Interval| Interval::sub(Interval::sqr(x), Interval::point(2.0));
        let dfi = |x: Interval| Interval::mul(Interval::point(2.0), x);
        let roots = interval_newton_1d(&fi, &dfi, Interval::new(-3.0, 3.0), 1e-10, 200);
        assert_eq!(roots.len(), 2);
        assert!(roots[1].contains(2f64.sqrt()));
    }
}
