//! Robust geometric predicates (spec 02 § Exact layer; AG-02-004): a fast floating-point
//! evaluation with a forward error bound ("filter"); when inconclusive, exact evaluation
//! with expansion arithmetic (Shewchuk two-sum / two-product, no FMA). Degenerate results
//! are exact zeros, which topology builders resolve by symbolic perturbation (index order).

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    (x, (a - av) + (b - bv))
}

#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = 134217729.0 * a; // 2^27 + 1
    let abig = c - a;
    let hi = c - abig;
    (hi, a - hi)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let err = x - ah * bh - al * bh - ah * bl;
    (x, al * bl - err)
}

/// Expansion: non-overlapping components, increasing magnitude.
#[derive(Clone, Debug, Default)]
pub struct Expansion(pub Vec<f64>);

impl Expansion {
    pub fn from(a: f64) -> Self { Expansion(vec![a]) }
    pub fn from_prod(a: f64, b: f64) -> Self { let (x, y) = two_prod(a, b); Expansion(vec![y, x]) }
    pub fn from_diff(a: f64, b: f64) -> Self { let (x, y) = two_sum(a, -b); Expansion(vec![y, x]) }
    /// Grow-expansion sum.
    pub fn add(&self, o: &Expansion) -> Expansion {
        let mut out = self.0.clone();
        for &b in &o.0 {
            let mut q = b;
            for h in out.iter_mut() {
                let (s, e) = two_sum(q, *h);
                *h = e;
                q = s;
            }
            out.push(q);
        }
        out.retain(|x| *x != 0.0);
        Expansion(out)
    }
    pub fn neg(&self) -> Expansion { Expansion(self.0.iter().map(|x| -x).collect()) }
    pub fn sub(&self, o: &Expansion) -> Expansion { self.add(&o.neg()) }
    /// Scale by a double (exact).
    pub fn scale(&self, b: f64) -> Expansion {
        let mut out = Expansion::default();
        for &a in &self.0 {
            let (x, y) = two_prod(a, b);
            out = out.add(&Expansion(vec![y, x]));
        }
        out
    }
    pub fn mul(&self, o: &Expansion) -> Expansion {
        let mut acc = Expansion::default();
        for &b in &o.0 { acc = acc.add(&self.scale(b)); }
        acc
    }
    /// Sign of the exact value.
    pub fn sign(&self) -> i32 {
        match self.0.iter().rev().find(|x| **x != 0.0) { Some(x) if *x > 0.0 => 1, Some(_) => -1, None => 0 }
    }
    pub fn estimate(&self) -> f64 { self.0.iter().sum() }
}

const EPS: f64 = f64::EPSILON * 0.5;
const ORIENT2D_BOUND: f64 = (3.0 + 16.0 * EPS) * EPS;
const ORIENT3D_BOUND: f64 = (7.0 + 56.0 * EPS) * EPS;

/// Sign of the area of triangle (a, b, c): +1 counter-clockwise, -1 clockwise, 0 collinear.
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i32 {
    let detl = (a[0] - c[0]) * (b[1] - c[1]);
    let detr = (a[1] - c[1]) * (b[0] - c[0]);
    let det = detl - detr;
    let bound = ORIENT2D_BOUND * (detl.abs() + detr.abs());
    if det > bound { return 1; }
    if -det > bound { return -1; }
    // exact
    let acx = Expansion::from_diff(a[0], c[0]);
    let bcy = Expansion::from_diff(b[1], c[1]);
    let acy = Expansion::from_diff(a[1], c[1]);
    let bcx = Expansion::from_diff(b[0], c[0]);
    acx.mul(&bcy).sub(&acy.mul(&bcx)).sign()
}

/// Sign of the signed volume of (a, b, c, d): +1 if d is below the plane abc (right-hand rule).
pub fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> i32 {
    let ad = [a[0] - d[0], a[1] - d[1], a[2] - d[2]];
    let bd = [b[0] - d[0], b[1] - d[1], b[2] - d[2]];
    let cd = [c[0] - d[0], c[1] - d[1], c[2] - d[2]];
    let t1 = bd[1] * cd[2] - bd[2] * cd[1];
    let t2 = bd[2] * cd[0] - bd[0] * cd[2];
    let t3 = bd[0] * cd[1] - bd[1] * cd[0];
    let det = ad[0] * t1 + ad[1] * t2 + ad[2] * t3;
    let perm = ad[0].abs() * (bd[1].abs() * cd[2].abs() + bd[2].abs() * cd[1].abs())
        + ad[1].abs() * (bd[2].abs() * cd[0].abs() + bd[0].abs() * cd[2].abs())
        + ad[2].abs() * (bd[0].abs() * cd[1].abs() + bd[1].abs() * cd[0].abs());
    let bound = ORIENT3D_BOUND * perm;
    if det > bound { return 1; }
    if -det > bound { return -1; }
    let e = |x: f64, y: f64| Expansion::from_diff(x, y);
    let (adx, ady, adz) = (e(a[0], d[0]), e(a[1], d[1]), e(a[2], d[2]));
    let (bdx, bdy, bdz) = (e(b[0], d[0]), e(b[1], d[1]), e(b[2], d[2]));
    let (cdx, cdy, cdz) = (e(c[0], d[0]), e(c[1], d[1]), e(c[2], d[2]));
    let m1 = bdy.mul(&cdz).sub(&bdz.mul(&cdy));
    let m2 = bdz.mul(&cdx).sub(&bdx.mul(&cdz));
    let m3 = bdx.mul(&cdy).sub(&bdy.mul(&cdx));
    adx.mul(&m1).add(&ady.mul(&m2)).add(&adz.mul(&m3)).sign()
}

/// Exact sign of `n · (p - o)` for a plane given by point `o` and normal `n`.
pub fn side_of_plane(o: [f64; 3], n: [f64; 3], p: [f64; 3]) -> i32 {
    let approx = n[0] * (p[0] - o[0]) + n[1] * (p[1] - o[1]) + n[2] * (p[2] - o[2]);
    let mag = n[0].abs() * (p[0] - o[0]).abs() + n[1].abs() * (p[1] - o[1]).abs() + n[2].abs() * (p[2] - o[2]).abs();
    let bound = 8.0 * EPS * mag;
    if approx > bound { return 1; }
    if -approx > bound { return -1; }
    let mut acc = Expansion::default();
    for i in 0..3 { acc = acc.add(&Expansion::from_diff(p[i], o[i]).scale(n[i])); }
    acc.sign()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_on_near_degenerate_input() {
        // Classic failure case for naive orientation: points nearly collinear at large offset.
        let a = [0.5, 0.5];
        let b = [12.0, 12.0];
        let c = [24.0, 24.0];
        assert_eq!(orient2d(a, b, c), 0);
        let c2 = [24.0, 24.000000000000004];
        assert_eq!(orient2d(a, b, c2), 1);
        assert_eq!(orient3d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 0.0]), 0);
        assert_eq!(orient3d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, -1e-300]), 1);
    }
}
