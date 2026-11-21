//! Surfaces (spec 02): analytic types are kept analytic; NURBS and procedural types.

use crate::math::{v3, Frame, V3, Xform};
use crate::nurbs::{Jet, NurbsSurface};
use crate::curve::Curve;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Surface {
    /// S(u,v) = o + u x + v y
    Plane { frame: Frame },
    /// S(u,v) = o + r(cos u x + sin u y) + v z
    Cylinder { frame: Frame, r: f64 },
    /// Cone with apex half-angle `angle`, radius `r` at v = 0.
    Cone { frame: Frame, r: f64, angle: f64 },
    /// S(u,v) = o + r(cos v (cos u x + sin u y) + sin v z)
    Sphere { frame: Frame, r: f64 },
    /// Major radius R, minor r.
    Torus { frame: Frame, major: f64, minor: f64 },
    Nurbs(NurbsSurface),
    /// Linear extrusion of a curve along `dir`: S(u,v) = C(u) + v dir
    Extrusion { curve: Box<Curve>, dir: V3 },
    /// Revolution of a curve about the frame z axis: u = curve param, v = angle.
    Revolution { curve: Box<Curve>, axis: Frame },
    /// Offset of a base surface along its unit normal.
    Offset { base: Box<Surface>, d: f64 },
}

impl Surface {
    pub fn jet(&self, u: f64, v: f64) -> Jet {
        match self {
            Surface::Plane { frame } => Jet { p: frame.origin + frame.x * u + frame.y * v, du: frame.x, dv: frame.y, ..Jet::default() },
            Surface::Cylinder { frame, r } => {
                let (s, c) = u.sin_cos();
                let radial = frame.x * c + frame.y * s;
                Jet { p: frame.origin + radial * *r + frame.z * v, du: (frame.x * -s + frame.y * c) * *r, dv: frame.z, duu: radial * -*r, duv: V3::ZERO, dvv: V3::ZERO }
            }
            Surface::Cone { frame, r, angle } => {
                let (s, c) = u.sin_cos();
                let radial = frame.x * c + frame.y * s;
                let rr = r + v * angle.tan();
                Jet { p: frame.origin + radial * rr + frame.z * v, du: (frame.x * -s + frame.y * c) * rr, dv: radial * angle.tan() + frame.z, duu: radial * -rr, duv: (frame.x * -s + frame.y * c) * angle.tan(), dvv: V3::ZERO }
            }
            Surface::Sphere { frame, r } => {
                let (su, cu) = u.sin_cos();
                let (sv, cv) = v.sin_cos();
                let radial = frame.x * cu + frame.y * su;
                let tang = frame.x * -su + frame.y * cu;
                Jet {
                    p: frame.origin + (radial * cv + frame.z * sv) * *r,
                    du: tang * (cv * r),
                    dv: (radial * -sv + frame.z * cv) * *r,
                    duu: radial * (-cv * r),
                    duv: tang * (-sv * r),
                    dvv: (radial * -cv + frame.z * -sv) * *r,
                }
            }
            Surface::Torus { frame, major, minor } => {
                let (su, cu) = u.sin_cos();
                let (sv, cv) = v.sin_cos();
                let radial = frame.x * cu + frame.y * su;
                let tang = frame.x * -su + frame.y * cu;
                let rr = major + minor * cv;
                Jet {
                    p: frame.origin + radial * rr + frame.z * (minor * sv),
                    du: tang * rr,
                    dv: radial * (-minor * sv) + frame.z * (minor * cv),
                    duu: radial * -rr,
                    duv: tang * (-minor * sv),
                    dvv: radial * (-minor * cv) + frame.z * (-minor * sv),
                }
            }
            Surface::Nurbs(n) => n.jet(u, v),
            Surface::Extrusion { curve, dir } => {
                let (p, d1, d2) = curve.d2(u);
                Jet { p: p + *dir * v, du: d1, dv: *dir, duu: d2, duv: V3::ZERO, dvv: V3::ZERO }
            }
            Surface::Revolution { curve, axis } => {
                let (p, d1, d2) = curve.d2(u);
                let rot = |q: V3, a: f64| -> V3 {
                    let l = axis.to_local(q);
                    let (s, c) = a.sin_cos();
                    axis.to_world(v3(l.x * c - l.y * s, l.x * s + l.y * c, l.z))
                };
                let rotd = |q: V3, a: f64| -> V3 {
                    let l = v3(q.dot(axis.x), q.dot(axis.y), q.dot(axis.z));
                    let (s, c) = a.sin_cos();
                    axis.dir_to_world(v3(l.x * c - l.y * s, l.x * s + l.y * c, l.z))
                };
                let pr = rot(p, v);
                let l = axis.to_local(p);
                let (s, c) = v.sin_cos();
                let dv = axis.dir_to_world(v3(-l.x * s - l.y * c, l.x * c - l.y * s, 0.0));
                let dvv = axis.dir_to_world(v3(-l.x * c + l.y * s, -l.x * s - l.y * c, 0.0));
                let ld = v3(d1.dot(axis.x), d1.dot(axis.y), 0.0);
                let duv = axis.dir_to_world(v3(-ld.x * s - ld.y * c, ld.x * c - ld.y * s, 0.0));
                Jet { p: pr, du: rotd(d1, v), dv, duu: rotd(d2, v), duv, dvv }
            }
            Surface::Offset { base, d } => {
                // First-order offset jet (second derivatives of the base; adequate for
                // tessellation and SSI seeding; exact evaluation of position).
                let j = base.jet(u, v);
                let n = j.normal();
                Jet { p: j.p + n * *d, ..j }
            }
        }
    }

    pub fn point(&self, u: f64, v: f64) -> V3 { self.jet(u, v).p }
    pub fn normal(&self, u: f64, v: f64) -> V3 { self.jet(u, v).normal() }

    pub fn is_analytic(&self) -> bool { matches!(self, Surface::Plane { .. } | Surface::Cylinder { .. } | Surface::Cone { .. } | Surface::Sphere { .. } | Surface::Torus { .. }) }

    /// Parameters of the closest point (analytic closed forms; Newton otherwise).
    pub fn invert(&self, p: V3, guess: (f64, f64)) -> (f64, f64) {
        match self {
            Surface::Plane { frame } => { let l = frame.to_local(p); (l.x, l.y) }
            Surface::Cylinder { frame, .. } => { let l = frame.to_local(p); (l.y.atan2(l.x), l.z) }
            Surface::Sphere { frame, .. } => { let l = frame.to_local(p); (l.y.atan2(l.x), l.z.atan2((l.x * l.x + l.y * l.y).sqrt())) }
            Surface::Cone { frame, .. } => { let l = frame.to_local(p); (l.y.atan2(l.x), l.z) }
            Surface::Torus { frame, major, .. } => {
                let l = frame.to_local(p);
                let u = l.y.atan2(l.x);
                let rho = (l.x * l.x + l.y * l.y).sqrt();
                (u, l.z.atan2(rho - major))
            }
            _ => {
                let (mut u, mut v) = guess;
                for _ in 0..40 {
                    let j = self.jet(u, v);
                    let r = j.p - p;
                    let (a11, a12, a22) = (j.du.dot(j.du) + r.dot(j.duu), j.du.dot(j.dv) + r.dot(j.duv), j.dv.dot(j.dv) + r.dot(j.dvv));
                    let (b1, b2) = (-r.dot(j.du), -r.dot(j.dv));
                    let det = a11 * a22 - a12 * a12;
                    if det.abs() < 1e-300 { break; }
                    let du = (b1 * a22 - b2 * a12) / det;
                    let dv = (a11 * b2 - a12 * b1) / det;
                    u += du;
                    v += dv;
                    if du.abs() + dv.abs() < 1e-14 { break; }
                }
                (u, v)
            }
        }
    }

    /// Exact signed distance for closed analytic surfaces (positive outside), else unsigned
    /// distance to the closest point (callers decide sign by face orientation).
    pub fn distance(&self, p: V3) -> f64 {
        match self {
            Surface::Plane { frame } => (p - frame.origin).dot(frame.z),
            Surface::Cylinder { frame, r } => { let l = frame.to_local(p); (l.x * l.x + l.y * l.y).sqrt() - r }
            Surface::Sphere { frame, r } => (p - frame.origin).norm() - r,
            Surface::Torus { frame, major, minor } => { let l = frame.to_local(p); let q = ((l.x * l.x + l.y * l.y).sqrt() - major, l.z); (q.0 * q.0 + q.1 * q.1).sqrt() - minor }
            _ => { let (u, v) = self.invert(p, (0.5, 0.5)); self.point(u, v).dist(p) }
        }
    }

    pub fn transformed(&self, x: &Xform) -> Surface {
        let tf = |f: &Frame| Frame { origin: x.apply(f.origin), x: x.apply_dir(f.x).normalized(), y: x.apply_dir(f.y).normalized(), z: x.apply_dir(f.z).normalized() };
        match self {
            Surface::Plane { frame } => Surface::Plane { frame: tf(frame) },
            Surface::Cylinder { frame, r } => Surface::Cylinder { frame: tf(frame), r: r * x.s },
            Surface::Cone { frame, r, angle } => Surface::Cone { frame: tf(frame), r: r * x.s, angle: *angle },
            Surface::Sphere { frame, r } => Surface::Sphere { frame: tf(frame), r: r * x.s },
            Surface::Torus { frame, major, minor } => Surface::Torus { frame: tf(frame), major: major * x.s, minor: minor * x.s },
            Surface::Nurbs(n) => Surface::Nurbs(NurbsSurface { poles: n.poles.iter().map(|p| x.apply(*p)).collect(), ..n.clone() }),
            Surface::Extrusion { curve, dir } => Surface::Extrusion { curve: Box::new(curve.transformed(x)), dir: x.apply_dir(*dir) },
            Surface::Revolution { curve, axis } => Surface::Revolution { curve: Box::new(curve.transformed(x)), axis: tf(axis) },
            Surface::Offset { base, d } => Surface::Offset { base: Box::new(base.transformed(x)), d: d * x.s },
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Surface::Plane { .. } => "plane", Surface::Cylinder { .. } => "cylinder", Surface::Cone { .. } => "cone",
            Surface::Sphere { .. } => "sphere", Surface::Torus { .. } => "torus", Surface::Nurbs(_) => "nurbs",
            Surface::Extrusion { .. } => "extrusion", Surface::Revolution { .. } => "revolution", Surface::Offset { .. } => "offset",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_jets_consistent_with_finite_differences() {
        let surfaces = [
            Surface::Cylinder { frame: Frame::WORLD, r: 3.0 },
            Surface::Sphere { frame: Frame::WORLD, r: 2.0 },
            Surface::Torus { frame: Frame::WORLD, major: 5.0, minor: 1.0 },
            Surface::Cone { frame: Frame::WORLD, r: 2.0, angle: 0.3 },
        ];
        let h = 1e-6;
        for s in &surfaces {
            let j = s.jet(0.4, 0.3);
            let fu = (s.point(0.4 + h, 0.3) - s.point(0.4 - h, 0.3)) * (0.5 / h);
            let fv = (s.point(0.4, 0.3 + h) - s.point(0.4, 0.3 - h)) * (0.5 / h);
            assert!(fu.dist(j.du) < 1e-6, "{} du", s.kind_name());
            assert!(fv.dist(j.dv) < 1e-6, "{} dv", s.kind_name());
            let (u, v) = s.invert(j.p, (0.0, 0.0));
            assert!(s.point(u, v).dist(j.p) < 1e-9, "{} invert", s.kind_name());
        }
    }
}
