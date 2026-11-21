//! Double-float ("df64") arithmetic: an unevaluated sum of two f32 (≈48-bit mantissa).
//! Used by the `Df64Refined` precision policy on consumer GPUs and wgpu (AG-08-002).
//! Error-free transformations without FMA (Dekker split), so CUDA (`fmad=false`) and WGSL
//! produce identical bits.

use agcad_driver::Df64;

#[inline]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

#[inline]
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline]
fn split(a: f32) -> (f32, f32) {
    const C: f32 = 4097.0; // 2^12 + 1
    let t = C * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

#[inline]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

pub fn add(a: Df64, b: Df64) -> Df64 {
    let (s, e) = two_sum(a.hi, b.hi);
    let (t, f) = two_sum(a.lo, b.lo);
    let (s, e) = quick_two_sum(s, e + t);
    let (hi, lo) = quick_two_sum(s, e + f);
    Df64 { hi, lo }
}

pub fn neg(a: Df64) -> Df64 { Df64 { hi: -a.hi, lo: -a.lo } }
pub fn sub(a: Df64, b: Df64) -> Df64 { add(a, neg(b)) }

pub fn mul(a: Df64, b: Df64) -> Df64 {
    let (p, e) = two_prod(a.hi, b.hi);
    let e = e + (a.hi * b.lo + a.lo * b.hi);
    let (hi, lo) = quick_two_sum(p, e);
    Df64 { hi, lo }
}

pub fn div(a: Df64, b: Df64) -> Df64 {
    let q1 = a.hi / b.hi;
    let r = sub(a, mul(b, Df64 { hi: q1, lo: 0.0 }));
    let q2 = r.hi / b.hi;
    let (hi, lo) = quick_two_sum(q1, q2);
    Df64 { hi, lo }
}

pub fn sqrt(a: Df64) -> Df64 {
    if a.hi <= 0.0 { return Df64 { hi: 0.0, lo: 0.0 }; }
    let x = 1.0 / a.hi.sqrt();
    let ax = a.hi * x;
    let (p, e) = two_prod(ax, ax);
    let d = sub(a, Df64 { hi: p, lo: e });
    let (hi, lo) = two_sum(ax, d.hi * (x * 0.5));
    Df64 { hi, lo }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn df64_beats_f32() {
        let a = Df64::from_f64(1.0 / 3.0);
        let b = Df64::from_f64(1e-4);
        let p = mul(add(a, b), Df64::from_f64(3.0)).to_f64();
        let exact = (1.0 / 3.0 + 1e-4) * 3.0;
        assert!((p - exact).abs() < 1e-12, "{p} vs {exact}");
        assert!((sqrt(Df64::from_f64(2.0)).to_f64() - 2f64.sqrt()).abs() < 1e-13);
    }
}
