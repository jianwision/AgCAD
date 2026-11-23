//! B-spline / NURBS evaluation with derivatives (Piegl & Tiller A2.1–A2.3, A4.2/A4.4),
//! knot insertion (A5.1). Ported in spirit from the Demo jets (`cuda/surface_primitives.cuh`).

use crate::math::V3;
use serde::{Deserialize, Serialize};

/// Find knot span index for `u` (A2.1).
pub fn find_span(n: usize, p: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[n + 1] { return n; }
    if u <= knots[p] { return p; }
    let (mut lo, mut hi) = (p, n + 1);
    let mut mid = (lo + hi) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] { hi = mid; } else { lo = mid; }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Basis functions and derivatives up to order `nd` at span `i` (A2.3): `ders[k][j]`.
pub fn ders_basis(i: usize, u: f64, p: usize, nd: usize, knots: &[f64]) -> Vec<Vec<f64>> {
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[i + 1 - j];
        right[j] = knots[i + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let nd = nd.min(p);
    let mut ders = vec![vec![0.0; p + 1]; nd + 1];
    for j in 0..=p { ders[0][j] = ndu[j][p]; }
    let mut a = vec![vec![0.0; p + 1]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0usize, 1usize);
        a[0][0] = 1.0;
        for k in 1..=nd {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = p as isize - k as isize;
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk { k - 1 } else { p - r };
            for j in j1..=j2 {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                d += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }
            if r as isize <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                d += a[s2][k] * ndu[r][pk as usize];
            }
            ders[k][r] = d;
            core::mem::swap(&mut s1, &mut s2);
        }
    }
    let mut r = p as f64;
    for k in 1..=nd {
        for j in 0..=p { ders[k][j] *= r; }
        r *= (p - k) as f64;
    }
    ders
}

/// Rational B-spline curve (homogeneous control points `[x w, y w, z w, w]` stored as
/// unweighted xyz + weight, like the Demo records).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NurbsCurve {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub poles: Vec<V3>,
    pub weights: Vec<f64>,
}

impl NurbsCurve {
    pub fn bezier(poles: Vec<V3>) -> Self {
        let p = poles.len() - 1;
        let mut knots = vec![0.0; p + 1];
        knots.extend(vec![1.0; p + 1]);
        NurbsCurve { degree: p, knots, weights: vec![1.0; poles.len()], poles }
    }
    pub fn domain(&self) -> (f64, f64) { (self.knots[self.degree], self.knots[self.poles.len()]) }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.knots.len() != self.poles.len() + self.degree + 1 { return Err("knot count"); }
        if self.weights.len() != self.poles.len() { return Err("weight count"); }
        if self.weights.iter().any(|w| !(*w > 0.0)) { return Err("weights must be positive"); }
        if self.knots.windows(2).any(|w| w[1] < w[0]) { return Err("knots must be non-decreasing"); }
        Ok(())
    }
    /// Point and derivatives up to `nd` (A4.2 rational derivatives).
    pub fn derivs(&self, u: f64, nd: usize) -> Vec<V3> {
        let n = self.poles.len() - 1;
        let p = self.degree;
        let span = find_span(n, p, u, &self.knots);
        let b = ders_basis(span, u, p, nd, &self.knots);
        let mut aw = vec![(V3::ZERO, 0.0); nd + 1];
        for k in 0..=nd.min(p) {
            for j in 0..=p {
                let idx = span - p + j;
                let w = self.weights[idx];
                aw[k].0 = aw[k].0 + self.poles[idx] * (b[k][j] * w);
                aw[k].1 += b[k][j] * w;
            }
        }
        let mut ck = vec![V3::ZERO; nd + 1];
        for k in 0..=nd {
            let mut v = aw[k].0;
            for i in 1..=k {
                v = v - ck[k - i] * (binom(k, i) * aw[i].1);
            }
            ck[k] = v * (1.0 / aw[0].1);
        }
        ck
    }
    pub fn point(&self, u: f64) -> V3 { self.derivs(u, 0)[0] }

    /// Insert knot `u` once (A5.1); shape unchanged.
    pub fn insert_knot(&self, u: f64) -> NurbsCurve {
        let p = self.degree;
        let n = self.poles.len() - 1;
        let k = find_span(n, p, u, &self.knots);
        let hw: Vec<(V3, f64)> = self.poles.iter().zip(&self.weights).map(|(q, w)| (*q * *w, *w)).collect();
        let mut out = Vec::with_capacity(hw.len() + 1);
        for i in 0..=n + 1 {
            if i <= k - p { out.push(hw[i]); }
            else if i > k { out.push(hw[i - 1]); }
            else {
                let a = (u - self.knots[i]) / (self.knots[i + p] - self.knots[i]);
                out.push((hw[i - 1].0 * (1.0 - a) + hw[i].0 * a, hw[i - 1].1 * (1.0 - a) + hw[i].1 * a));
            }
        }
        let mut knots = self.knots.clone();
        knots.insert(k + 1, u);
        NurbsCurve { degree: p, knots, poles: out.iter().map(|(q, w)| *q * (1.0 / *w)).collect(), weights: out.iter().map(|x| x.1).collect() }
    }

    /// Exact rational circle arc (≤ 4 segments of ≤ 90°), used for revolve/STEP export.
    pub fn arc(center: V3, x: V3, y: V3, r: f64, a0: f64, a1: f64) -> NurbsCurve {
        let sweep = a1 - a0;
        let segs = ((sweep.abs() / core::f64::consts::FRAC_PI_2).ceil() as usize).clamp(1, 4);
        let d = sweep / segs as f64;
        let wm = (d / 2.0).cos();
        let mut poles = Vec::new();
        let mut weights = Vec::new();
        let pt = |a: f64| center + x * (r * a.cos()) + y * (r * a.sin());
        for s in 0..segs {
            let a = a0 + d * s as f64;
            if s == 0 { poles.push(pt(a)); weights.push(1.0); }
            let mid = a + d / 2.0;
            poles.push(center + (x * mid.cos() + y * mid.sin()) * (r / wm));
            weights.push(wm);
            poles.push(pt(a + d));
            weights.push(1.0);
        }
        let mut knots = vec![0.0; 3];
        for s in 1..segs { knots.push(s as f64 / segs as f64); knots.push(s as f64 / segs as f64); }
        knots.extend([1.0, 1.0, 1.0]);
        NurbsCurve { degree: 2, knots, poles, weights }
    }
}

/// Rational tensor-product surface. Poles row-major `[u][v]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NurbsSurface {
    pub du: usize,
    pub dv: usize,
    pub ku: Vec<f64>,
    pub kv: Vec<f64>,
    pub nu: usize,
    pub nv: usize,
    pub poles: Vec<V3>,
    pub weights: Vec<f64>,
}

/// Surface jet: point and first/second partials (S, Su, Sv, Suu, Suv, Svv).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Jet {
    pub p: V3,
    pub du: V3,
    pub dv: V3,
    pub duu: V3,
    pub duv: V3,
    pub dvv: V3,
}

impl Jet {
    pub fn normal(&self) -> V3 { self.du.cross(self.dv).normalized() }
    /// Gaussian and mean curvature (first/second fundamental forms).
    pub fn curvature(&self) -> (f64, f64) {
        let n = self.normal();
        let (e, f, g) = (self.du.dot(self.du), self.du.dot(self.dv), self.dv.dot(self.dv));
        let (l, m, nn) = (self.duu.dot(n), self.duv.dot(n), self.dvv.dot(n));
        let den = e * g - f * f;
        if den.abs() < 1e-300 { return (f64::NAN, f64::NAN); }
        ((l * nn - m * m) / den, (e * nn - 2.0 * f * m + g * l) / (2.0 * den))
    }
}

impl NurbsSurface {
    /// Evaluate position without computing the first and second derivatives.
    pub fn point(&self, u: f64, v: f64) -> V3 {
        let iu = find_span(self.nu - 1, self.du, u, &self.ku);
        let iv = find_span(self.nv - 1, self.dv, v, &self.kv);
        let bu = ders_basis(iu, u, self.du, 0, &self.ku);
        let bv = ders_basis(iv, v, self.dv, 0, &self.kv);
        let origin = self.poles[0];
        let mut p = V3::ZERO;
        let mut weight = 0.;
        for (i, &a) in bu[0].iter().enumerate() {
            for (j, &b) in bv[0].iter().enumerate() {
                let k = (iu - self.du + i) * self.nv + iv - self.dv + j;
                let w = a * b * self.weights[k];
                p = p + (self.poles[k] - origin) * w;
                weight += w;
            }
        }
        origin + p * (1. / weight)
    }
    pub fn domain(&self) -> ((f64, f64), (f64, f64)) { ((self.ku[self.du], self.ku[self.nu]), (self.kv[self.dv], self.kv[self.nv])) }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.ku.len() != self.nu + self.du + 1 || self.kv.len() != self.nv + self.dv + 1 { return Err("knot count"); }
        if self.poles.len() != self.nu * self.nv || self.weights.len() != self.poles.len() { return Err("pole count"); }
        if self.weights.iter().any(|w| !(*w > 0.0)) { return Err("weights must be positive"); }
        Ok(())
    }
    /// Jet at (u, v) (rational, A4.4 to second order).
    pub fn jet(&self, u: f64, v: f64) -> Jet {
        let (p, q) = (self.du, self.dv);
        let su = find_span(self.nu - 1, p, u, &self.ku);
        let sv = find_span(self.nv - 1, q, v, &self.kv);
        let bu = ders_basis(su, u, p, 2, &self.ku);
        let bv = ders_basis(sv, v, q, 2, &self.kv);
        let origin = self.poles[0];
        let mut aw = [[(V3::ZERO, 0.0); 3]; 3];
        for k in 0..=2.min(p) {
            for l in 0..=(2 - k).min(q) {
                let mut acc = (V3::ZERO, 0.0);
                for i in 0..=p {
                    for j in 0..=q {
                        let idx = (su - p + i) * self.nv + (sv - q + j);
                        let w = self.weights[idx];
                        let b = bu[k][i] * bv[l][j];
                        acc.0 = acc.0 + (self.poles[idx] - origin) * (b * w);
                        acc.1 += b * w;
                    }
                }
                aw[k][l] = acc;
            }
        }
        let mut s = [[V3::ZERO; 3]; 3];
        for k in 0..=2 {
            for l in 0..=(2 - k) {
                let mut v2 = aw[k][l].0;
                for j in 1..=l { v2 = v2 - s[k][l - j] * (binom(l, j) * aw[0][j].1); }
                for i in 1..=k {
                    v2 = v2 - s[k - i][l] * (binom(k, i) * aw[i][0].1);
                    let mut v3_ = V3::ZERO;
                    for j in 1..=l { v3_ = v3_ + s[k - i][l - j] * (binom(l, j) * aw[i][j].1); }
                    v2 = v2 - v3_ * binom(k, i);
                }
                s[k][l] = v2 * (1.0 / aw[0][0].1);
            }
        }
        Jet { p: origin + s[0][0], du: s[1][0], dv: s[0][1], duu: s[2][0], duv: s[1][1], dvv: s[0][2] }
    }
}

fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k { r = r * (n - i) as f64 / (i + 1) as f64; }
    r
}

/// Bilinear "plane" patch helper used by tests and builders.
pub fn bilinear(p00: V3, p10: V3, p01: V3, p11: V3) -> NurbsSurface {
    NurbsSurface { du: 1, dv: 1, ku: vec![0.0, 0.0, 1.0, 1.0], kv: vec![0.0, 0.0, 1.0, 1.0], nu: 2, nv: 2, poles: vec![p00, p01, p10, p11], weights: vec![1.0; 4] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v3;
    #[test]
    fn rational_arc_is_exact_circle() {
        let c = NurbsCurve::arc(V3::ZERO, V3::X, V3::Y, 5.0, 0.0, core::f64::consts::PI * 1.5);
        c.validate().unwrap();
        for i in 0..=20 {
            let p = c.point(i as f64 / 20.0);
            assert!((p.norm() - 5.0).abs() < 1e-12, "{p:?}");
        }
        let c2 = c.insert_knot(0.3);
        assert!(c2.point(0.7).dist(c.point(0.7)) < 1e-12);
    }
    #[test]
    fn bilinear_jet() {
        let s = bilinear(v3(0.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), v3(0.0, 3.0, 0.0), v3(2.0, 3.0, 0.0));
        let j = s.jet(0.5, 0.5);
        assert!((j.p.dist(v3(1.0, 1.5, 0.0))) < 1e-12);
        assert!((j.du.dist(v3(2.0, 0.0, 0.0))) < 1e-12);
        assert!((j.normal().z - 1.0).abs() < 1e-12);
    }
}
