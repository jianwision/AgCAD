//! Interval and affine arithmetic for certified bounds (evidence `bounded`, spec 02/14).
//! Outward rounding by one ulp after every operation keeps enclosures valid without
//! changing the FPU rounding mode (portable to GPUs).

use agcad_rtc::expr::{Bin, ExprGraph, Node, Un};

fn down(x: f64) -> f64 { if x.is_finite() { next_toward(x, f64::NEG_INFINITY) } else { x } }
fn up(x: f64) -> f64 { if x.is_finite() { next_toward(x, f64::INFINITY) } else { x } }

fn next_toward(x: f64, dir: f64) -> f64 {
    if x.is_nan() { return x; }
    if x == 0.0 { return if dir > 0.0 { f64::from_bits(1) } else { -f64::from_bits(1) }; }
    let b = x.to_bits();
    let b = if (x < dir) == (x > 0.0) { b + 1 } else { b - 1 };
    f64::from_bits(b)
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const ENTIRE: Interval = Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
    pub fn new(lo: f64, hi: f64) -> Self { Interval { lo, hi } }
    pub fn point(x: f64) -> Self { Interval { lo: x, hi: x } }
    pub fn width(&self) -> f64 { self.hi - self.lo }
    pub fn mid(&self) -> f64 { 0.5 * (self.lo + self.hi) }
    pub fn contains(&self, x: f64) -> bool { self.lo <= x && x <= self.hi }
    pub fn contains_zero(&self) -> bool { self.lo <= 0.0 && 0.0 <= self.hi }
    pub fn hull(a: Self, b: Self) -> Self { Interval { lo: a.lo.min(b.lo), hi: a.hi.max(b.hi) } }

    pub fn add(a: Self, b: Self) -> Self { Interval { lo: down(a.lo + b.lo), hi: up(a.hi + b.hi) } }
    pub fn sub(a: Self, b: Self) -> Self { Interval { lo: down(a.lo - b.hi), hi: up(a.hi - b.lo) } }
    pub fn mul(a: Self, b: Self) -> Self {
        let c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Interval { lo: down(lo), hi: up(hi) }
    }
    pub fn div(a: Self, b: Self) -> Self {
        if b.contains_zero() { return Interval::ENTIRE; }
        Self::mul(a, Interval { lo: down(1.0 / b.hi), hi: up(1.0 / b.lo) })
    }
    pub fn sqr(a: Self) -> Self {
        if a.contains_zero() { Interval { lo: 0.0, hi: up(a.lo.abs().max(a.hi.abs()).powi(2)) } } else { Self::mul(a, a) }
    }
    pub fn sqrt(a: Self) -> Self { Interval { lo: down(a.lo.max(0.0).sqrt()), hi: up(a.hi.max(0.0).sqrt()) } }
    pub fn abs(a: Self) -> Self {
        if a.contains_zero() { Interval { lo: 0.0, hi: a.lo.abs().max(a.hi.abs()) } } else if a.hi < 0.0 { Interval { lo: -a.hi, hi: -a.lo } } else { a }
    }
    pub fn min(a: Self, b: Self) -> Self { Interval { lo: a.lo.min(b.lo), hi: a.hi.min(b.hi) } }
    pub fn max(a: Self, b: Self) -> Self { Interval { lo: a.lo.max(b.lo), hi: a.hi.max(b.hi) } }
    fn monotone(a: Self, f: fn(f64) -> f64) -> Self { Interval { lo: down(f(a.lo)), hi: up(f(a.hi)) } }
    /// Sine enclosure: exact extrema handling by checking whether ±π/2 + 2kπ lie inside.
    pub fn sin(a: Self) -> Self {
        if a.width() >= core::f64::consts::TAU { return Interval::new(-1.0, 1.0); }
        let mut lo = a.lo.sin().min(a.hi.sin());
        let mut hi = a.lo.sin().max(a.hi.sin());
        let k0 = ((a.lo - core::f64::consts::FRAC_PI_2) / core::f64::consts::TAU).ceil();
        if core::f64::consts::FRAC_PI_2 + k0 * core::f64::consts::TAU <= a.hi { hi = 1.0; }
        let k1 = ((a.lo + core::f64::consts::FRAC_PI_2) / core::f64::consts::TAU).ceil();
        if -core::f64::consts::FRAC_PI_2 + k1 * core::f64::consts::TAU <= a.hi { lo = -1.0; }
        Interval { lo: down(lo).max(-1.0), hi: up(hi).min(1.0) }
    }
    pub fn cos(a: Self) -> Self { Self::sin(Self::add(a, Interval::point(core::f64::consts::FRAC_PI_2))) }
}

/// Interval evaluation of an expression graph over an input box: a certified enclosure of
/// each output (used for `bounded` evidence: implicit-body inside/outside, thickness).
/// Unsupported ops widen to ENTIRE (still valid, just useless).
pub fn eval_interval(g: &ExprGraph, inputs: &[Interval], params: &[f64]) -> Vec<Interval> {
    let mut v: Vec<Interval> = Vec::with_capacity(g.nodes.len());
    for n in &g.nodes {
        let r = match *n {
            Node::Const(c) => Interval::point(c),
            Node::Input(k) => inputs[k as usize],
            Node::Param(k) => Interval::point(params[k as usize]),
            Node::Un(op, a) => {
                let x = v[a as usize];
                match op {
                    Un::Neg => Interval { lo: -x.hi, hi: -x.lo },
                    Un::Abs => Interval::abs(x),
                    Un::Sqrt => Interval::sqrt(x),
                    Un::Sin => Interval::sin(x),
                    Un::Cos => Interval::cos(x),
                    Un::Exp => Interval::monotone(x, f64::exp),
                    Un::Ln => if x.lo > 0.0 { Interval::monotone(x, f64::ln) } else { Interval::ENTIRE },
                    Un::Atan => Interval::monotone(x, f64::atan),
                    Un::Floor => Interval { lo: x.lo.floor(), hi: x.hi.floor() },
                    Un::Ceil => Interval { lo: x.lo.ceil(), hi: x.hi.ceil() },
                    _ => Interval::ENTIRE,
                }
            }
            Node::Bin(op, a, b) => {
                let (x, y) = (v[a as usize], v[b as usize]);
                match op {
                    Bin::Add => Interval::add(x, y),
                    Bin::Sub => Interval::sub(x, y),
                    Bin::Mul => if a == b { Interval::sqr(x) } else { Interval::mul(x, y) },
                    Bin::Div => Interval::div(x, y),
                    Bin::Min => Interval::min(x, y),
                    Bin::Max => Interval::max(x, y),
                    _ => Interval::ENTIRE,
                }
            }
            Node::Select(_, a, b) => Interval::hull(v[a as usize], v[b as usize]),
            Node::Noise(..) => Interval::new(-1.0, 1.0),
        };
        v.push(r);
    }
    g.outputs.iter().map(|&o| v[o as usize]).collect()
}

/// Affine form x₀ + Σ xᵢ εᵢ (εᵢ ∈ [-1,1]) for tighter enclosures of correlated expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Affine {
    pub c: f64,
    pub terms: Vec<(u32, f64)>,
    pub err: f64,
}

impl Affine {
    pub fn from_interval(i: Interval, sym: u32) -> Self { Affine { c: i.mid(), terms: vec![(sym, 0.5 * i.width())], err: 0.0 } }
    pub fn radius(&self) -> f64 { self.terms.iter().map(|t| t.1.abs()).sum::<f64>() + self.err }
    pub fn to_interval(&self) -> Interval { let r = self.radius(); Interval { lo: down(self.c - r), hi: up(self.c + r) } }
    fn merge(a: &Self, b: &Self, sb: f64) -> Vec<(u32, f64)> {
        let mut t = a.terms.clone();
        for &(s, x) in &b.terms {
            match t.iter_mut().find(|e| e.0 == s) { Some(e) => e.1 += sb * x, None => t.push((s, sb * x)) }
        }
        t
    }
    pub fn add(a: &Self, b: &Self) -> Self { Affine { c: a.c + b.c, terms: Self::merge(a, b, 1.0), err: a.err + b.err + f64::EPSILON * (a.c.abs() + b.c.abs()) } }
    pub fn sub(a: &Self, b: &Self) -> Self { Affine { c: a.c - b.c, terms: Self::merge(a, b, -1.0), err: a.err + b.err + f64::EPSILON * (a.c.abs() + b.c.abs()) } }
    pub fn mul(a: &Self, b: &Self) -> Self {
        let mut t: Vec<(u32, f64)> = a.terms.iter().map(|&(s, x)| (s, x * b.c)).collect();
        for &(s, y) in &b.terms {
            match t.iter_mut().find(|e| e.0 == s) { Some(e) => e.1 += a.c * y, None => t.push((s, a.c * y)) }
        }
        Affine { c: a.c * b.c, terms: t, err: a.radius() * b.radius() + a.err * b.c.abs() + b.err * a.c.abs() + f64::EPSILON * (a.c * b.c).abs() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enclosure_contains_samples() {
        let src = "#[implicit] fn s(p: Vec3) -> Length { length(p) - 5.0 * mm }";
        let m = agcad_rtc::compile_module(src, agcad_ir::Catalog::global()).unwrap();
        let g = m.get("s").unwrap().expr.as_ref().unwrap();
        let bx = [Interval::new(1.0, 2.0), Interval::new(-1.0, 0.5), Interval::new(0.0, 3.0)];
        let enc = eval_interval(g, &bx, &[])[0];
        for &x in &[1.0, 1.5, 2.0] { for &y in &[-1.0, 0.0, 0.5] { for &z in &[0.0, 1.5, 3.0] {
            let d = g.eval(&[x, y, z], &[])[0];
            assert!(enc.contains(d), "{d} not in {enc:?}");
        }}}
    }
}
