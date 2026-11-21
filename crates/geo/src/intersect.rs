//! Analytic-first surface intersection (spec 02 § Exact geometry): co-surface detection,
//! then a closed-form table for analytic pairs. General NURBS marching is the fallback
//! (`Intersection::NeedsMarching`, implemented with GPU seeding in pass P11).

use crate::curve::Curve;
use crate::math::{Frame, V3};
use crate::surface::Surface;

#[derive(Clone, Debug, PartialEq)]
pub enum Intersection {
    Empty,
    /// Same geometric carrier (coplanar, coaxial-equal, co-spherical): resolve by 2D boolean
    /// in the shared parameter domain, never numerically.
    Coincident,
    /// Tangential contact at a point.
    Point(V3),
    Curves(Vec<Curve>),
    /// Not in the closed-form table.
    NeedsMarching,
}

fn parallel(a: V3, b: V3, tol: f64) -> bool { a.normalized().cross(b.normalized()).norm() < tol }

/// Co-surface test within linear tolerance `tol` (angular tolerance derived).
pub fn co_surface(a: &Surface, b: &Surface, tol: f64) -> bool {
    match (a, b) {
        (Surface::Plane { frame: f1 }, Surface::Plane { frame: f2 }) => parallel(f1.z, f2.z, 1e-9) && (f2.origin - f1.origin).dot(f1.z).abs() < tol,
        (Surface::Cylinder { frame: f1, r: r1 }, Surface::Cylinder { frame: f2, r: r2 }) => coaxial(f1, f2, tol) && (r1 - r2).abs() < tol,
        (Surface::Sphere { frame: f1, r: r1 }, Surface::Sphere { frame: f2, r: r2 }) => f1.origin.dist(f2.origin) < tol && (r1 - r2).abs() < tol,
        (Surface::Cone { frame: f1, r: r1, angle: a1 }, Surface::Cone { frame: f2, r: r2, angle: a2 }) => coaxial(f1, f2, tol) && (a1 - a2).abs() < 1e-9 && ((r1 + (f2.origin - f1.origin).dot(f1.z) * a1.tan()) - r2).abs() < tol,
        (Surface::Torus { frame: f1, major: m1, minor: n1 }, Surface::Torus { frame: f2, major: m2, minor: n2 }) => f1.origin.dist(f2.origin) < tol && parallel(f1.z, f2.z, 1e-9) && (m1 - m2).abs() < tol && (n1 - n2).abs() < tol,
        _ => false,
    }
}

fn coaxial(f1: &Frame, f2: &Frame, tol: f64) -> bool {
    parallel(f1.z, f2.z, 1e-9) && { let d = f2.origin - f1.origin; (d - f1.z * d.dot(f1.z)).norm() < tol }
}

/// Closed-form intersection of two surfaces (untrimmed).
pub fn intersect(a: &Surface, b: &Surface, tol: f64) -> Intersection {
    if co_surface(a, b, tol) { return Intersection::Coincident; }
    use Surface::*;
    match (a, b) {
        (Plane { frame: p1 }, Plane { frame: p2 }) => {
            let dir = p1.z.cross(p2.z);
            if dir.norm() < 1e-12 { return Intersection::Empty; }
            // point on both planes: solve with the two plane equations + dir component = 0
            let d1 = p1.z.dot(p1.origin);
            let d2 = p2.z.dot(p2.origin);
            let n1n2 = p1.z.dot(p2.z);
            let det = 1.0 - n1n2 * n1n2;
            let c1 = (d1 - d2 * n1n2) / det;
            let c2 = (d2 - d1 * n1n2) / det;
            Intersection::Curves(vec![Curve::Line { origin: p1.z * c1 + p2.z * c2, dir: dir.normalized() }])
        }
        (Plane { frame: p }, Cylinder { frame: c, r }) | (Cylinder { frame: c, r }, Plane { frame: p }) => {
            let cosang = p.z.dot(c.z);
            if (cosang.abs() - 1.0).abs() < 1e-12 {
                // axis ∥ plane normal → circle
                let t = (p.origin - c.origin).dot(p.z) / cosang;
                Intersection::Curves(vec![Curve::Circle { frame: Frame { origin: c.origin + c.z * t, ..*c }, r: *r }])
            } else if cosang.abs() < 1e-12 {
                // axis in plane direction → 0, 1 (tangent) or 2 lines
                let h = (c.origin - p.origin).dot(p.z);
                if h.abs() > r + tol { return Intersection::Empty; }
                let foot = c.origin - p.z * h;
                let side = p.z.cross(c.z).normalized();
                let w = (r * r - h * h).max(0.0).sqrt();
                if w < tol { return Intersection::Curves(vec![Curve::Line { origin: foot, dir: c.z }]); }
                Intersection::Curves(vec![Curve::Line { origin: foot + side * w, dir: c.z }, Curve::Line { origin: foot - side * w, dir: c.z }])
            } else {
                // oblique → ellipse: center on axis, minor = r, major = r / |cos|
                let t = (p.origin - c.origin).dot(p.z) / cosang;
                let center = c.origin + c.z * t;
                let major_dir = (c.z - p.z * cosang).normalized();
                let minor_dir = p.z.cross(major_dir).normalized();
                Intersection::Curves(vec![Curve::Ellipse { frame: Frame { origin: center, x: major_dir, y: minor_dir, z: p.z }, a: r / cosang.abs(), b: *r }])
            }
        }
        (Plane { frame: p }, Sphere { frame: s, r }) | (Sphere { frame: s, r }, Plane { frame: p }) => {
            let h = (s.origin - p.origin).dot(p.z);
            if h.abs() > r + tol { return Intersection::Empty; }
            let center = s.origin - p.z * h;
            let rr = (r * r - h * h).max(0.0).sqrt();
            if rr < tol { return Intersection::Point(center); }
            Intersection::Curves(vec![Curve::Circle { frame: Frame::new(center, p.z, p.x), r: rr }])
        }
        (Plane { frame: p }, Cone { frame: c, r, angle }) | (Cone { frame: c, r, angle }, Plane { frame: p }) if parallel(p.z, c.z, 1e-12) => {
            let t = (p.origin - c.origin).dot(c.z);
            let rr = r + t * angle.tan();
            if rr < -tol { return Intersection::Empty; }
            if rr.abs() < tol { return Intersection::Point(c.origin + c.z * t); }
            Intersection::Curves(vec![Curve::Circle { frame: Frame { origin: c.origin + c.z * t, ..*c }, r: rr }])
        }
        (Plane { frame: p }, Torus { frame: t, major, minor }) | (Torus { frame: t, major, minor }, Plane { frame: p }) if parallel(p.z, t.z, 1e-12) => {
            let h = (p.origin - t.origin).dot(t.z);
            if h.abs() > minor + tol { return Intersection::Empty; }
            let w = (minor * minor - h * h).max(0.0).sqrt();
            let center = t.origin + t.z * h;
            let mut cs = vec![Curve::Circle { frame: Frame { origin: center, ..*t }, r: major + w }];
            if w > tol { cs.push(Curve::Circle { frame: Frame { origin: center, ..*t }, r: major - w }); }
            Intersection::Curves(cs)
        }
        (Cylinder { frame: c1, r: r1 }, Cylinder { frame: c2, r: r2 }) if parallel(c1.z, c2.z, 1e-12) => {
            // parallel axes → circle-circle intersection in the cross-section
            let d = c2.origin - c1.origin;
            let dperp = d - c1.z * d.dot(c1.z);
            let dist = dperp.norm();
            if dist < 1e-12 { return Intersection::Empty; } // coaxial, different radius
            if dist > r1 + r2 + tol || dist < (r1 - r2).abs() - tol { return Intersection::Empty; }
            let a = (r1 * r1 - r2 * r2 + dist * dist) / (2.0 * dist);
            let h = (r1 * r1 - a * a).max(0.0).sqrt();
            let ex = dperp.normalized();
            let ey = c1.z.cross(ex);
            let m = c1.origin + ex * a;
            if h < tol { return Intersection::Curves(vec![Curve::Line { origin: m, dir: c1.z }]); }
            Intersection::Curves(vec![Curve::Line { origin: m + ey * h, dir: c1.z }, Curve::Line { origin: m - ey * h, dir: c1.z }])
        }
        (Sphere { frame: s1, r: r1 }, Sphere { frame: s2, r: r2 }) => {
            let d = s2.origin - s1.origin;
            let dist = d.norm();
            if dist > r1 + r2 + tol || dist < (r1 - r2).abs() - tol || dist < 1e-12 { return Intersection::Empty; }
            let a = (r1 * r1 - r2 * r2 + dist * dist) / (2.0 * dist);
            let h = (r1 * r1 - a * a).max(0.0).sqrt();
            let n = d.normalized();
            if h < tol { return Intersection::Point(s1.origin + n * a); }
            Intersection::Curves(vec![Curve::Circle { frame: Frame::new(s1.origin + n * a, n, n.any_perp()), r: h }])
        }
        (Cylinder { frame: c, r: rc }, Sphere { frame: s, r: rs }) | (Sphere { frame: s, r: rs }, Cylinder { frame: c, r: rc }) if {
            let d = s.origin - c.origin; (d - c.z * d.dot(c.z)).norm() < tol
        } => {
            if *rc > *rs + tol { return Intersection::Empty; }
            let t0 = (s.origin - c.origin).dot(c.z);
            let h = (*rs * *rs - *rc * *rc).max(0.0).sqrt();
            if h < tol { return Intersection::Curves(vec![Curve::Circle { frame: Frame { origin: c.origin + c.z * t0, ..*c }, r: *rc }]); }
            Intersection::Curves(vec![
                Curve::Circle { frame: Frame { origin: c.origin + c.z * (t0 - h), ..*c }, r: *rc },
                Curve::Circle { frame: Frame { origin: c.origin + c.z * (t0 + h), ..*c }, r: *rc },
            ])
        }
        _ => Intersection::NeedsMarching,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v3;
    #[test]
    fn table_results_lie_on_both_surfaces() {
        let cases = [
            (Surface::Plane { frame: Frame::new(v3(0.0, 0.0, 1.0), v3(0.0, 0.3, 1.0), V3::X) }, Surface::Cylinder { frame: Frame::WORLD, r: 2.0 }),
            (Surface::Plane { frame: Frame::new(v3(0.0, 0.0, 0.5), V3::Z, V3::X) }, Surface::Sphere { frame: Frame::WORLD, r: 2.0 }),
            (Surface::Sphere { frame: Frame::WORLD, r: 2.0 }, Surface::Sphere { frame: Frame { origin: v3(1.0, 1.0, 0.0), ..Frame::WORLD }, r: 1.5 }),
            (Surface::Plane { frame: Frame::new(V3::ZERO, V3::X, V3::Y) }, Surface::Plane { frame: Frame::new(V3::ZERO, v3(0.0, 1.0, 1.0), V3::X) }),
        ];
        for (a, b) in &cases {
            match intersect(a, b, 1e-9) {
                Intersection::Curves(cs) => for c in cs { for i in 0..10 {
                    let p = c.point(i as f64 * 0.6);
                    assert!(a.distance(p).abs() < 1e-9 && b.distance(p).abs() < 1e-9, "{} x {}: {p:?}", a.kind_name(), b.kind_name());
                } },
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(intersect(&Surface::Cylinder { frame: Frame::WORLD, r: 2.0 }, &Surface::Cylinder { frame: Frame::WORLD, r: 2.0 }, 1e-9), Intersection::Coincident);
    }
}
