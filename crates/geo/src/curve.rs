//! Curves (spec 02): analytic types stay analytic; NURBS and procedural helix.

use crate::math::{Frame, V3};
use crate::nurbs::NurbsCurve;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Curve {
    /// p(t) = origin + dir * t
    Line { origin: V3, dir: V3 },
    /// p(t) = c + r (cos t x + sin t y), frame z = axis
    Circle { frame: Frame, r: f64 },
    Ellipse { frame: Frame, a: f64, b: f64 },
    Helix { frame: Frame, r: f64, pitch: f64 },
    Nurbs(NurbsCurve),
    /// Intersection curve: NURBS fit + error bound + parent surface ids (spec 02).
    Intersection { fit: NurbsCurve, max_error: f64, parents: [u32; 2] },
}

impl Curve {
    /// Point, first and second derivative.
    pub fn d2(&self, t: f64) -> (V3, V3, V3) {
        match self {
            Curve::Line { origin, dir } => (*origin + *dir * t, *dir, V3::ZERO),
            Curve::Circle { frame, r } => {
                let (s, c) = t.sin_cos();
                (frame.origin + (frame.x * c + frame.y * s) * *r, (frame.x * -s + frame.y * c) * *r, (frame.x * -c + frame.y * -s) * *r)
            }
            Curve::Ellipse { frame, a, b } => {
                let (s, c) = t.sin_cos();
                (frame.origin + frame.x * (a * c) + frame.y * (b * s), frame.x * (-a * s) + frame.y * (b * c), frame.x * (-a * c) + frame.y * (-b * s))
            }
            Curve::Helix { frame, r, pitch } => {
                let (s, c) = t.sin_cos();
                let k = pitch / core::f64::consts::TAU;
                (frame.origin + (frame.x * c + frame.y * s) * *r + frame.z * (k * t), (frame.x * -s + frame.y * c) * *r + frame.z * k, (frame.x * -c + frame.y * -s) * *r)
            }
            Curve::Nurbs(n) | Curve::Intersection { fit: n, .. } => {
                let d = n.derivs(t, 2);
                (d[0], d[1], d[2])
            }
        }
    }
    pub fn point(&self, t: f64) -> V3 { self.d2(t).0 }
    pub fn tangent(&self, t: f64) -> V3 { self.d2(t).1.normalized() }

    /// Curvature magnitude at t.
    pub fn curvature(&self, t: f64) -> f64 {
        let (_, d1, d2) = self.d2(t);
        let n = d1.norm();
        if n == 0.0 { return f64::NAN; }
        d1.cross(d2).norm() / (n * n * n)
    }

    /// Arc length over [t0, t1] by composite Gauss–Legendre (5-point × 16 panels).
    pub fn length(&self, t0: f64, t1: f64) -> f64 {
        const X: [f64; 5] = [0.0, -0.538_469_310_105_683_1, 0.538_469_310_105_683_1, -0.906_179_845_938_664, 0.906_179_845_938_664];
        const W: [f64; 5] = [0.568_888_888_888_888_9, 0.478_628_670_499_366_5, 0.478_628_670_499_366_5, 0.236_926_885_056_189_1, 0.236_926_885_056_189_1];
        if let Curve::Line { dir, .. } = self { return dir.norm() * (t1 - t0).abs(); }
        if let Curve::Circle { r, .. } = self { return r * (t1 - t0).abs(); }
        let panels = 16;
        let h = (t1 - t0) / panels as f64;
        let mut s = 0.0;
        for k in 0..panels {
            let a = t0 + h * k as f64;
            for i in 0..5 { s += W[i] * self.d2(a + h * 0.5 * (X[i] + 1.0)).1.norm(); }
        }
        s * h * 0.5
    }

    /// Closest parameter to `p` near `t_guess` (Newton on (C(t)-p)·C'(t), clamped).
    pub fn closest(&self, p: V3, t_guess: f64, range: (f64, f64)) -> f64 {
        match self {
            Curve::Line { origin, dir } => ((p - *origin).dot(*dir) / dir.dot(*dir)).clamp(range.0, range.1),
            Curve::Circle { frame, .. } => {
                let l = frame.to_local(p);
                let mut t = l.y.atan2(l.x);
                while t < range.0 { t += core::f64::consts::TAU; }
                t.clamp(range.0, range.1)
            }
            _ => {
                let mut t = t_guess;
                for _ in 0..30 {
                    let (c, d1, d2) = self.d2(t);
                    let f = (c - p).dot(d1);
                    let df = d1.dot(d1) + (c - p).dot(d2);
                    if df.abs() < 1e-300 { break; }
                    let step = f / df;
                    t = (t - step).clamp(range.0, range.1);
                    if step.abs() < 1e-14 { break; }
                }
                t
            }
        }
    }

    pub fn transformed(&self, x: &crate::math::Xform) -> Curve {
        let tf = |f: &Frame| Frame { origin: x.apply(f.origin), x: x.apply_dir(f.x).normalized(), y: x.apply_dir(f.y).normalized(), z: x.apply_dir(f.z).normalized() };
        match self {
            Curve::Line { origin, dir } => Curve::Line { origin: x.apply(*origin), dir: x.apply_dir(*dir) },
            Curve::Circle { frame, r } => Curve::Circle { frame: tf(frame), r: r * x.s },
            Curve::Ellipse { frame, a, b } => Curve::Ellipse { frame: tf(frame), a: a * x.s, b: b * x.s },
            Curve::Helix { frame, r, pitch } => Curve::Helix { frame: tf(frame), r: r * x.s, pitch: pitch * x.s },
            Curve::Nurbs(n) => Curve::Nurbs(NurbsCurve { poles: n.poles.iter().map(|p| x.apply(*p)).collect(), ..n.clone() }),
            Curve::Intersection { fit, max_error, parents } => Curve::Intersection { fit: NurbsCurve { poles: fit.poles.iter().map(|p| x.apply(*p)).collect(), ..fit.clone() }, max_error: max_error * x.s, parents: *parents },
        }
    }

    /// Exact NURBS form (for STEP B-spline export or ops requiring NURBS).
    pub fn to_nurbs(&self, range: (f64, f64)) -> NurbsCurve {
        match self {
            Curve::Line { origin, dir } => NurbsCurve::bezier(vec![*origin + *dir * range.0, *origin + *dir * range.1]),
            Curve::Circle { frame, r } => NurbsCurve::arc(frame.origin, frame.x, frame.y, *r, range.0, range.1),
            Curve::Nurbs(n) | Curve::Intersection { fit: n, .. } => n.clone(),
            other => {
                // Interpolating approximation with 64 spans (error checked by callers).
                let pts: Vec<V3> = (0..=64).map(|i| other.point(range.0 + (range.1 - range.0) * i as f64 / 64.0)).collect();
                let mut k = vec![0.0, 0.0];
                k.extend((0..=64).map(|i| i as f64 / 64.0));
                k.extend([1.0, 1.0]);
                let _ = k;
                NurbsCurve { degree: 1, knots: { let mut kk = vec![0.0]; kk.extend((0..=64).map(|i| i as f64 / 64.0)); kk.push(1.0); kk }, weights: vec![1.0; pts.len()], poles: pts }
            }
        }
    }
}
