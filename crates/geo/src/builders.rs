//! Exact B-rep builders: prism (pad/pocket), revolve (shaft/groove) and primitives.
//! Profiles are 2D regions in a sketch frame (outer loop CCW, holes CW). Output faces carry
//! role names (`cap_start`, `cap_end`, `side.<k>`) that the runtime prefixes with the feature
//! reference (spec 02 § Stable naming).

use crate::brep::{BodyKind, Brep};
use crate::curve::Curve;
use crate::math::{v3, Frame, V3};
use crate::surface::Surface;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Seg2 {
    Line { a: [f64; 2], b: [f64; 2] },
    /// Arc around `c` from angle `a0` to `a1` (a1 > a0: counter-clockwise).
    Arc { c: [f64; 2], r: f64, a0: f64, a1: f64 },
    /// Full circle (counter-clockwise if `ccw`), must be alone in its loop.
    Circle { c: [f64; 2], r: f64, ccw: bool },
}

impl Seg2 {
    pub fn start(&self) -> [f64; 2] {
        match *self {
            Seg2::Line { a, .. } => a,
            Seg2::Arc { c, r, a0, .. } => [c[0] + r * a0.cos(), c[1] + r * a0.sin()],
            Seg2::Circle { c, r, .. } => [c[0] + r, c[1]],
        }
    }
    pub fn end(&self) -> [f64; 2] {
        match *self {
            Seg2::Line { b, .. } => b,
            Seg2::Arc { c, r, a1, .. } => [c[0] + r * a1.cos(), c[1] + r * a1.sin()],
            Seg2::Circle { c, r, .. } => [c[0] + r, c[1]],
        }
    }
    pub fn reversed(&self) -> Seg2 {
        match *self {
            Seg2::Line { a, b } => Seg2::Line { a: b, b: a },
            Seg2::Arc { c, r, a0, a1 } => Seg2::Arc { c, r, a0: a1, a1: a0 },
            Seg2::Circle { c, r, ccw } => Seg2::Circle { c, r, ccw: !ccw },
        }
    }
    /// Signed area contribution (shoelace / Green's theorem).
    pub fn area_term(&self) -> f64 {
        match *self {
            Seg2::Line { a, b } => 0.5 * (a[0] * b[1] - b[0] * a[1]),
            Seg2::Arc { c, r, a0, a1 } => 0.5 * (r * r * (a1 - a0) + c[0] * r * (a1.sin() - a0.sin()) - c[1] * r * (a1.cos() - a0.cos())),
            Seg2::Circle { c: _, r, ccw } => if ccw { PI * r * r } else { -PI * r * r },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// loops[0] = outer (CCW), others = holes (CW). Normalized by [`Profile::normalize`].
    pub loops: Vec<Vec<Seg2>>,
}

impl Profile {
    pub fn loop_area(lp: &[Seg2]) -> f64 { lp.iter().map(Seg2::area_term).sum() }
    pub fn area(&self) -> f64 { self.loops.iter().map(|l| Self::loop_area(l)).sum() }

    /// Orient outer CCW and holes CW; the loop with the largest |area| becomes outer.
    pub fn normalize(mut self) -> Self {
        if self.loops.is_empty() { return self; }
        let k = (0..self.loops.len()).max_by(|&a, &b| Self::loop_area(&self.loops[a]).abs().partial_cmp(&Self::loop_area(&self.loops[b]).abs()).unwrap()).unwrap();
        self.loops.swap(0, k);
        for (i, lp) in self.loops.iter_mut().enumerate() {
            let a = Self::loop_area(lp);
            if (i == 0 && a < 0.0) || (i > 0 && a > 0.0) {
                *lp = lp.iter().rev().map(Seg2::reversed).collect();
            }
        }
        self
    }

    pub fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Profile {
        Profile { loops: vec![vec![
            Seg2::Line { a: [x0, y0], b: [x1, y0] }, Seg2::Line { a: [x1, y0], b: [x1, y1] },
            Seg2::Line { a: [x1, y1], b: [x0, y1] }, Seg2::Line { a: [x0, y1], b: [x0, y0] },
        ]] }
    }
    pub fn circle(cx: f64, cy: f64, r: f64) -> Profile { Profile { loops: vec![vec![Seg2::Circle { c: [cx, cy], r, ccw: true }]] } }
    pub fn with_hole(mut self, hole: Profile) -> Profile {
        for l in hole.loops { self.loops.push(l); }
        self.normalize()
    }
}

/// Extrude `profile` (in `frame`'s xy plane) along +z by `h` (> 0): closed solid.
pub fn prism(profile: &Profile, frame: &Frame, h: f64) -> Brep {
    let profile = profile.clone().normalize();
    let mut b = Brep::new(BodyKind::Solid);
    let shell = b.add_shell(true);
    let w = |p: [f64; 2], z: f64| frame.to_world(v3(p[0], p[1], z));
    let bottom = b.add_surface(Surface::Plane { frame: *frame });
    let top = b.add_surface(Surface::Plane { frame: Frame { origin: frame.to_world(v3(0.0, 0.0, h)), ..*frame } });
    let mut bottom_loops: Vec<Vec<(u32, bool)>> = Vec::new();
    let mut top_loops: Vec<Vec<(u32, bool)>> = Vec::new();
    let mut side_idx = 0usize;
    for lp in &profile.loops {
        let n = lp.len();
        // vertices at each segment start, bottom and top
        let vb: Vec<u32> = lp.iter().map(|s| b.add_vertex(w(s.start(), 0.0))).collect();
        let vt: Vec<u32> = lp.iter().map(|s| b.add_vertex(w(s.start(), h))).collect();
        // vertical edges at segment starts
        let vert: Vec<u32> = (0..n).map(|k| {
            let c = b.add_curve(Curve::Line { origin: w(lp[k].start(), 0.0), dir: frame.z });
            b.add_edge(c, vb[k], vt[k], 0.0, h, format!("vertical.{}", side_idx + k))
        }).collect();
        let mut be = Vec::new();
        let mut te = Vec::new();
        for (k, s) in lp.iter().enumerate() {
            let k1 = (k + 1) % n;
            let (cb, ct, t0, t1) = match *s {
                Seg2::Line { a, b: e } => {
                    let d = w(e, 0.0) - w(a, 0.0);
                    (Curve::Line { origin: w(a, 0.0), dir: d }, Curve::Line { origin: w(a, h), dir: d }, 0.0, 1.0)
                }
                Seg2::Arc { c, r, a0, a1 } => {
                    let fb = Frame { origin: w(c, 0.0), ..*frame };
                    let ft = Frame { origin: w(c, h), ..*frame };
                    (Curve::Circle { frame: fb, r }, Curve::Circle { frame: ft, r }, a0, a1)
                }
                Seg2::Circle { c, r, ccw } => {
                    let fb = Frame { origin: w(c, 0.0), ..*frame };
                    let ft = Frame { origin: w(c, h), ..*frame };
                    let (t0, t1) = if ccw { (0.0, TAU) } else { (TAU, 0.0) };
                    (Curve::Circle { frame: fb, r }, Curve::Circle { frame: ft, r }, t0, t1)
                }
            };
            let cbi = b.add_curve(cb);
            let cti = b.add_curve(ct);
            be.push(b.add_edge(cbi, vb[k], vb[k1], t0, t1, format!("bottom.{}", side_idx + k)));
            te.push(b.add_edge(cti, vt[k], vt[k1], t0, t1, format!("top.{}", side_idx + k)));
        }
        // side faces
        for (k, s) in lp.iter().enumerate() {
            let k1 = (k + 1) % n;
            let name = format!("side.{}", side_idx + k);
            match *s {
                Seg2::Line { a, b: e } => {
                    let pa = w(a, 0.0);
                    let d = w(e, 0.0) - pa;
                    let len = d.norm();
                    let sf = b.add_surface(Surface::Plane { frame: Frame { origin: pa, x: d.normalized(), y: frame.z, z: d.normalized().cross(frame.z) } });
                    b.add_face(sf, true, shell, &[(be[k], true), (vert[k1], true), (te[k], false), (vert[k], false)], &[], name, [0.0, len, 0.0, h]);
                }
                Seg2::Arc { c, r, a0, a1 } => {
                    let sf = b.add_surface(Surface::Cylinder { frame: Frame { origin: w(c, 0.0), ..*frame }, r });
                    let ccw = a1 > a0;
                    b.add_face(sf, ccw, shell, &[(be[k], true), (vert[k1], true), (te[k], false), (vert[k], false)], &[], name, [a0.min(a1), a0.max(a1), 0.0, h]);
                }
                Seg2::Circle { c, r, ccw } => {
                    let sf = b.add_surface(Surface::Cylinder { frame: Frame { origin: w(c, 0.0), ..*frame }, r });
                    // single face with seam (vertical edge used twice, opposite senses)
                    b.add_face(sf, ccw, shell, &[(be[k], true), (vert[k], true), (te[k], false), (vert[k], false)], &[], name, [0.0, TAU, 0.0, h]);
                }
            }
        }
        side_idx += n;
        bottom_loops.push(be.iter().rev().map(|&e| (e, false)).collect());
        top_loops.push(te.iter().map(|&e| (e, true)).collect());
    }
    let ext = profile_extent(&profile);
    b.add_face(bottom, false, shell, &bottom_loops[0], &bottom_loops[1..], "cap_start", ext);
    b.add_face(top, true, shell, &top_loops[0], &top_loops[1..], "cap_end", ext);
    b
}

fn profile_extent(p: &Profile) -> [f64; 4] {
    let mut e = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for lp in &p.loops {
        for s in lp {
            let pts: Vec<[f64; 2]> = match *s {
                Seg2::Circle { c, r, .. } => vec![[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]],
                Seg2::Arc { c, r, .. } => vec![[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]],
                Seg2::Line { a, b } => vec![a, b],
            };
            for q in pts { e[0] = e[0].min(q[0]); e[1] = e[1].max(q[0]); e[2] = e[2].min(q[1]); e[3] = e[3].max(q[1]); }
        }
    }
    e
}

/// Box from min/max corners.
pub fn block(min: V3, max: V3) -> Brep {
    let f = Frame { origin: v3(0.0, 0.0, min.z), ..Frame::WORLD };
    prism(&Profile::rectangle(min.x, min.y, max.x, max.y), &f, max.z - min.z)
}

/// Cylinder of radius `r` and height `h` on `frame` (axis = frame z).
pub fn cylinder(frame: &Frame, r: f64, h: f64) -> Brep { prism(&Profile::circle(0.0, 0.0, r), frame, h) }

/// Sphere: one face with a seam meridian and two pole vertices.
pub fn sphere(frame: &Frame, r: f64) -> Brep {
    let mut b = Brep::new(BodyKind::Solid);
    let shell = b.add_shell(true);
    let s = b.add_surface(Surface::Sphere { frame: *frame, r });
    let south = b.add_vertex(frame.to_world(v3(0.0, 0.0, -r)));
    let north = b.add_vertex(frame.to_world(v3(0.0, 0.0, r)));
    let meridian = b.add_curve(Curve::Circle { frame: Frame { origin: frame.origin, x: frame.x, y: frame.z, z: frame.x.cross(frame.z) }, r });
    let seam = b.add_edge(meridian, south, north, -PI / 2.0, PI / 2.0, "seam");
    // loop: seam up, (degenerate pole), seam down — poles are vertices, not edges.
    b.add_face(s, true, shell, &[(seam, true), (seam, false)], &[], "sphere", [0.0, TAU, -PI / 2.0, PI / 2.0]);
    b
}

/// Torus about `frame` z: one face bounded by a u-seam and a v-seam through one vertex.
pub fn torus(frame: &Frame, major: f64, minor: f64) -> Brep {
    let mut b = Brep::new(BodyKind::Solid);
    let shell = b.add_shell(true);
    let s = b.add_surface(Surface::Torus { frame: *frame, major, minor });
    let v = b.add_vertex(frame.to_world(v3(major + minor, 0.0, 0.0)));
    let cu = b.add_curve(Curve::Circle { frame: *frame, r: major + minor });
    let cv = b.add_curve(Curve::Circle { frame: Frame { origin: frame.to_world(v3(major, 0.0, 0.0)), x: frame.x, y: frame.z, z: frame.x.cross(frame.z) }, r: minor });
    let eu = b.add_edge(cu, v, v, 0.0, TAU, "seam_u");
    let ev = b.add_edge(cv, v, v, 0.0, TAU, "seam_v");
    b.add_face(s, true, shell, &[(eu, true), (ev, true), (eu, false), (ev, false)], &[], "torus", [0.0, TAU, -PI, PI]);
    b
}

/// Cone frustum: bottom radius r0 at z=0, top radius r1 at z=h (r1 may be 0 → apex).
pub fn cone(frame: &Frame, r0: f64, r1: f64, h: f64) -> Brep {
    let mut b = Brep::new(BodyKind::Solid);
    let shell = b.add_shell(true);
    let angle = ((r1 - r0) / h).atan();
    let side = b.add_surface(Surface::Cone { frame: *frame, r: r0, angle });
    let bot = b.add_surface(Surface::Plane { frame: *frame });
    let vb = b.add_vertex(frame.to_world(v3(r0, 0.0, 0.0)));
    let vt = b.add_vertex(frame.to_world(v3(r1, 0.0, h)));
    let cbot = b.add_curve(Curve::Circle { frame: *frame, r: r0 });
    let ebot = b.add_edge(cbot, vb, vb, 0.0, TAU, "bottom.0");
    let gen_dir = frame.to_world(v3(r1, 0.0, h)) - frame.to_world(v3(r0, 0.0, 0.0));
    let cseam = b.add_curve(Curve::Line { origin: frame.to_world(v3(r0, 0.0, 0.0)), dir: gen_dir });
    let eseam = b.add_edge(cseam, vb, vt, 0.0, 1.0, "seam");
    if r1 > 0.0 {
        let top = b.add_surface(Surface::Plane { frame: Frame { origin: frame.to_world(v3(0.0, 0.0, h)), ..*frame } });
        let ctop = b.add_curve(Curve::Circle { frame: Frame { origin: frame.to_world(v3(0.0, 0.0, h)), ..*frame }, r: r1 });
        let etop = b.add_edge(ctop, vt, vt, 0.0, TAU, "top.0");
        b.add_face(side, true, shell, &[(ebot, true), (eseam, true), (etop, false), (eseam, false)], &[], "side.0", [0.0, TAU, 0.0, h]);
        b.add_face(top, true, shell, &[(etop, true)], &[], "cap_end", [-r1, r1, -r1, r1]);
    } else {
        b.add_face(side, true, shell, &[(ebot, true), (eseam, true), (eseam, false)], &[], "side.0", [0.0, TAU, 0.0, h]);
    }
    b.add_face(bot, false, shell, &[(ebot, false)], &[], "cap_start", [-r0, r0, -r0, r0]);
    b
}

/// Revolve a closed profile lying in the frame's xz half-plane (x ≥ 0 = radius, z = axial;
/// profile coordinates `[radius, z]`) a full turn about frame z. Each segment becomes one
/// face: line ∥ axis → cylinder, line ⟂ axis → plane annulus, other lines → cone, arcs →
/// exact procedural revolution surface. Segments on the axis are skipped (degenerate).
pub fn revolve(profile: &Profile, frame: &Frame) -> Brep {
    let profile = profile.clone().normalize();
    let mut b = Brep::new(BodyKind::Solid);
    let shell = b.add_shell(true);
    let lp = &profile.loops[0];
    let n = lp.len();
    let w = |p: [f64; 2]| frame.to_world(v3(p[0], 0.0, p[1]));
    let on_axis = |p: [f64; 2]| p[0].abs() < 1e-12;
    let verts: Vec<u32> = lp.iter().map(|s| b.add_vertex(w(s.start()))).collect();
    // circle edge at every off-axis vertex
    let circles: Vec<Option<u32>> = lp.iter().enumerate().map(|(k, s)| {
        let p = s.start();
        if on_axis(p) { return None; }
        let c = b.add_curve(Curve::Circle { frame: Frame { origin: frame.to_world(v3(0.0, 0.0, p[1])), ..*frame }, r: p[0] });
        Some(b.add_edge(c, verts[k], verts[k], 0.0, TAU, format!("parallel.{k}")))
    }).collect();
    // seam edge per segment (the profile segment itself at angle 0)
    let seams: Vec<Option<u32>> = lp.iter().enumerate().map(|(k, s)| {
        let k1 = (k + 1) % n;
        if on_axis(s.start()) && on_axis(s.end()) { return None; }
        let c = match *s {
            Seg2::Line { a, b: e } => Curve::Line { origin: w(a), dir: w(e) - w(a) },
            Seg2::Arc { c, r, .. } => Curve::Circle { frame: Frame { origin: w(c), x: frame.x, y: frame.z, z: frame.x.cross(frame.z) }, r },
            Seg2::Circle { .. } => Curve::Line { origin: w(s.start()), dir: V3::ZERO },
        };
        let (t0, t1) = match *s { Seg2::Arc { a0, a1, .. } => (a0, a1), _ => (0.0, 1.0) };
        let ci = b.add_curve(c);
        Some(b.add_edge(ci, verts[k], verts[k1], t0, t1, format!("seam.{k}")))
    }).collect();
    for (k, s) in lp.iter().enumerate() {
        let k1 = (k + 1) % n;
        let Some(seam) = seams[k] else { continue };
        let (a, e) = (s.start(), s.end());
        let surf = match *s {
            Seg2::Line { .. } if (a[0] - e[0]).abs() < 1e-12 => Surface::Cylinder { frame: *frame, r: a[0] },
            Seg2::Line { .. } if (a[1] - e[1]).abs() < 1e-12 => Surface::Plane { frame: Frame { origin: frame.to_world(v3(0.0, 0.0, a[1])), ..*frame } },
            Seg2::Line { .. } => {
                let angle = ((e[0] - a[0]) / (e[1] - a[1])).atan();
                let r0 = a[0] - a[1] * angle.tan();
                Surface::Cone { frame: *frame, r: r0, angle }
            }
            Seg2::Arc { .. } | Seg2::Circle { .. } => {
                let ci = b.edges[seam as usize].curve;
                Surface::Revolution { curve: Box::new(b.curves[ci as usize].clone()), axis: *frame }
            }
        };
        let si = b.add_surface(surf);
        let mut lpv: Vec<(u32, bool)> = Vec::new();
        if let Some(c0) = circles[k] { lpv.push((c0, false)); }
        lpv.push((seam, true));
        if let Some(c1) = circles[k1] { lpv.push((c1, true)); }
        lpv.push((seam, false));
        // Orientation: the profile is CCW in (r, z); outward normal of segment a→e is to its
        // right, which matches the revolution sense of the surface parameterization.
        b.add_face(si, true, shell, &lpv, &[], format!("side.{k}"), [0.0, TAU, a[1].min(e[1]), a[1].max(e[1])]);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prism_with_hole_is_valid() {
        let p = Profile::rectangle(-50.0, -50.0, 50.0, 50.0).with_hole(Profile::circle(0.0, 0.0, 10.0));
        let b = prism(&p, &Frame::WORLD, 12.0);
        let r = b.validate();
        assert!(r.valid, "{:?}", r.problems);
        assert_eq!(b.faces.len(), 4 + 1 + 2);
        assert!(b.face_by_name("cap_end").is_some());
    }
    #[test]
    fn primitives_are_valid() {
        for (name, b) in [("box", block(V3::ZERO, v3(1.0, 2.0, 3.0))), ("cyl", cylinder(&Frame::WORLD, 2.0, 5.0)), ("cone", cone(&Frame::WORLD, 3.0, 1.0, 4.0)), ("apex", cone(&Frame::WORLD, 3.0, 0.0, 4.0))] {
            let r = b.validate();
            assert!(r.valid, "{name}: {:?}", r.problems);
        }
    }
}
