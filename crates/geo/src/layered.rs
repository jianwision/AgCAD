//! Layered extrusion solids: a stack of z-slabs in one sketch frame, each with a 2D region.
//! Exact for any combination of pads, pockets (through or blind), holes and patterns that
//! share an extrusion direction, i.e. most prismatic machined parts. B-rep faces are built
//! directly with shared edges (watertight topology, exact analytic geometry).

use crate::brep::{BodyKind, Brep};
use crate::builders::{Profile, Seg2};
use crate::curve::Curve;
use crate::math::{v3, Frame, V3};
use crate::region2d::{self, BoolOp2};
use crate::surface::Surface;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub z0: f64,
    pub z1: f64,
    pub regions: Vec<Profile>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayeredSolid {
    pub frame: Frame,
    pub layers: Vec<Layer>,
}

fn region_op(a: &[Profile], b: &[Profile], op: BoolOp2) -> Vec<Profile> {
    match op {
        BoolOp2::Union => {
            let mut acc: Vec<Profile> = a.to_vec();
            for r in b {
                let mut next = Vec::new();
                let mut merged = r.clone();
                for x in acc.drain(..) {
                    let u = region2d::boolean(&x, &merged, BoolOp2::Union);
                    if u.len() == 1 { merged = u.into_iter().next().unwrap(); } else { next.push(x); }
                }
                next.push(merged);
                acc = next;
            }
            acc
        }
        BoolOp2::Difference => {
            let mut acc: Vec<Profile> = a.to_vec();
            for r in b { acc = acc.iter().flat_map(|x| region2d::boolean(x, r, BoolOp2::Difference)).collect(); }
            acc
        }
        BoolOp2::Intersection => a.iter().flat_map(|x| b.iter().flat_map(move |r| region2d::boolean(x, r, BoolOp2::Intersection))).collect(),
    }
}

impl LayeredSolid {
    pub fn extrusion(frame: Frame, profile: &Profile, z0: f64, z1: f64) -> Self {
        LayeredSolid { frame, layers: vec![Layer { z0: z0.min(z1), z1: z0.max(z1), regions: vec![profile.clone().normalize()] }] }
    }

    fn levels(&self, other: &LayeredSolid) -> Vec<f64> {
        let mut z: Vec<f64> = self.layers.iter().chain(&other.layers).flat_map(|l| [l.z0, l.z1]).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        z.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        z
    }

    fn regions_at(&self, zmid: f64) -> Vec<Profile> {
        self.layers.iter().find(|l| l.z0 <= zmid && zmid <= l.z1).map(|l| l.regions.clone()).unwrap_or_default()
    }

    /// Boolean with another layered solid in the same frame (callers check frame equality).
    pub fn boolean(&self, other: &LayeredSolid, op: BoolOp2) -> LayeredSolid {
        let z = self.levels(other);
        let mut layers: Vec<Layer> = Vec::new();
        for w in z.windows(2) {
            let zm = 0.5 * (w[0] + w[1]);
            let r = region_op(&self.regions_at(zm), &other.regions_at(zm), op);
            if r.is_empty() { continue; }
            // merge with previous layer if regions identical
            if let Some(last) = layers.last_mut() {
                if (last.z1 - w[0]).abs() < 1e-9 && same_regions(&last.regions, &r) { last.z1 = w[1]; continue; }
            }
            layers.push(Layer { z0: w[0], z1: w[1], regions: r });
        }
        LayeredSolid { frame: self.frame, layers }
    }

    /// Tangential contacts between the operands of a boolean (per z-slab, between the two
    /// solids' boundaries). The 2D region boolean is ill-conditioned exactly at tangency, so
    /// the kernel rejects these before computing the boolean.
    pub fn boolean_contacts(&self, other: &LayeredSolid) -> Vec<[f64; 3]> {
        const TOL: f64 = 1e-7;
        let mut out = Vec::new();
        for w in self.levels(other).windows(2) {
            let zm = 0.5 * (w[0] + w[1]);
            let a: Vec<Vec<Seg2>> = self.regions_at(zm).iter().flat_map(region2d::expand).collect();
            let b: Vec<Vec<Seg2>> = other.regions_at(zm).iter().flat_map(region2d::expand).collect();
            if a.is_empty() || b.is_empty() { continue; }
            let mut inner: Vec<[f64; 2]> = region2d::tangent_contacts(&a, TOL).into_iter().chain(region2d::tangent_contacts(&b, TOL)).collect();
            inner.extend(region2d::shared_junctions(&a, &b, TOL));
            let both: Vec<Vec<Seg2>> = a.iter().chain(&b).cloned().collect();
            for p in region2d::tangent_contacts(&both, TOL) {
                if !inner.iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < TOL) && !out.iter().any(|q: &[f64; 3]| (q[0] - p[0]).hypot(q[1] - p[1]) < TOL) { out.push([p[0], p[1], zm]); }
            }
        }
        out
    }

    /// Tangential point contacts (x, y, z in the sketch frame) inside any layer and between
    /// the boundaries meeting at each shared level. Non-empty means the result would be
    /// non-manifold; the kernel rejects such booleans with an explicit error.
    pub fn tangent_contacts(&self) -> Vec<[f64; 3]> {
        const TOL: f64 = 1e-7;
        let mut out = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            let own: Vec<Vec<Seg2>> = l.regions.iter().flat_map(region2d::expand).collect();
            for p in region2d::tangent_contacts(&own, TOL) { out.push([p[0], p[1], 0.5 * (l.z0 + l.z1)]); }
            if let Some(n) = self.layers.get(li + 1).filter(|n| (n.z0 - l.z1).abs() < 1e-9) {
                // contacts between this layer's and the next layer's boundaries at the shared level
                let up: Vec<Vec<Seg2>> = n.regions.iter().flat_map(region2d::expand).collect();
                let both: Vec<Vec<Seg2>> = own.iter().chain(&up).cloned().collect();
                let mut inner: Vec<[f64; 2]> = region2d::tangent_contacts(&own, TOL).into_iter().chain(region2d::tangent_contacts(&up, TOL)).collect();
                // a smooth junction of a wall present on both sides of the level is not a contact
                inner.extend(region2d::shared_junctions(&own, &up, TOL));
                for p in region2d::tangent_contacts(&both, TOL) {
                    if !inner.iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < TOL) { out.push([p[0], p[1], l.z1]); }
                }
            }
        }
        out
    }

    pub fn volume(&self) -> f64 { self.layers.iter().map(|l| (l.z1 - l.z0) * l.regions.iter().map(|r| r.area()).sum::<f64>()).sum() }

    /// Build the B-rep (single shell per connected component is not enforced here; faces are
    /// emitted into one shell and validity reports genus).
    pub fn to_brep(&self) -> Brep {
        let mut b = Brep::new(BodyKind::Solid);
        let shell = b.add_shell(true);
        let f = self.frame;
        let mut vmap: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let q = |x: f64| (x * 1e6).round() as i64;
        let mut vert = |b: &mut Brep, p2: [f64; 2], z: f64| -> u32 {
            let key = (q(p2[0]), q(p2[1]), q(z));
            *vmap.entry(key).or_insert_with(|| b.add_vertex(f.to_world(v3(p2[0], p2[1], z))))
        };
        // Horizontal edges keyed by (z, canonical segment key) → (edge id, stored direction start)
        let mut hedges: HashMap<(i64, String), (u32, [f64; 2])> = HashMap::new();
        let mut vedges: HashMap<(i64, i64, i64, i64), u32> = HashMap::new();
        // Closed circle edges (one per full circle and level, vertex at angle 0) → (edge, ccw).
        let mut cedges: HashMap<(i64, i64, i64, i64), (u32, bool)> = HashMap::new();
        let mut cedge = |b: &mut Brep, vert: &mut dyn FnMut(&mut Brep, [f64; 2], f64) -> u32, c: [f64; 2], r: f64, ccw: bool, z: f64| -> (u32, bool) {
            let key = (q(z), q(c[0]), q(c[1]), q(r));
            if let Some(&(e, stored)) = cedges.get(&key) { return (e, stored == ccw); }
            let v = vert(b, [c[0] + r, c[1]], z);
            let ci = b.add_curve(Curve::Circle { frame: Frame { origin: f.to_world(v3(c[0], c[1], z)), ..f }, r });
            let (t0, t1) = if ccw { (0.0, core::f64::consts::TAU) } else { (core::f64::consts::TAU, 0.0) };
            let e = b.add_edge(ci, v, v, t0, t1, format!("h.{}", b.edges.len()));
            cedges.insert(key, (e, ccw));
            (e, true)
        };
        let seg_key = |s: &Seg2| -> String {
            let (a, e) = (s.start(), s.end());
            let (p, r) = if (q(a[0]), q(a[1])) <= (q(e[0]), q(e[1])) { (a, e) } else { (e, a) };
            let m = region2d::point_at(s, 0.5);
            format!("{},{},{},{},{},{}", q(p[0]), q(p[1]), q(r[0]), q(r[1]), q(m[0]), q(m[1]))
        };
        let mut hedge = |b: &mut Brep, vert: &mut dyn FnMut(&mut Brep, [f64; 2], f64) -> u32, s: &Seg2, z: f64| -> (u32, bool) {
            let key = (q(z), seg_key(s));
            if let Some(&(e, start)) = hedges.get(&key) {
                let fwd = (q(start[0]), q(start[1])) == (q(s.start()[0]), q(s.start()[1]));
                return (e, fwd);
            }
            let (va, ve) = (vert(b, s.start(), z), vert(b, s.end(), z));
            let (c, t0, t1) = match *s {
                Seg2::Line { a, b: e2 } => (Curve::Line { origin: f.to_world(v3(a[0], a[1], z)), dir: f.to_world(v3(e2[0], e2[1], z)) - f.to_world(v3(a[0], a[1], z)) }, 0.0, 1.0),
                Seg2::Arc { c, r, a0, a1 } => (Curve::Circle { frame: Frame { origin: f.to_world(v3(c[0], c[1], z)), ..f }, r }, a0, a1),
                Seg2::Circle { .. } => unreachable!("circles expanded"),
            };
            let ci = b.add_curve(c);
            let e = b.add_edge(ci, va, ve, t0, t1, format!("h.{}", b.edges.len()));
            hedges.insert(key, (e, s.start()));
            (e, true)
        };
        let mut side_count = 0usize;
        let nl = self.layers.len();
        for (li, layer) in self.layers.iter().enumerate() {
            // Split this layer's boundary at points where the levels above/below touch it.
            let below: Vec<Profile> = if li > 0 && (self.layers[li - 1].z1 - layer.z0).abs() < 1e-9 { self.layers[li - 1].regions.clone() } else { vec![] };
            let above: Vec<Profile> = if li + 1 < nl && (self.layers[li + 1].z0 - layer.z1).abs() < 1e-9 { self.layers[li + 1].regions.clone() } else { vec![] };
            let below_b: Vec<Vec<Seg2>> = below.iter().flat_map(region2d::expand).collect();
            let above_b: Vec<Vec<Seg2>> = above.iter().flat_map(region2d::expand).collect();
            // Split points of `s` where another boundary touches it (bottom edges see only the
            // layer below, top edges only the layer above; matches the horizontal faces).
            let splits = |s: &Seg2, others: &[Vec<Seg2>]| -> Vec<f64> {
                let mut ts = Vec::new();
                for o in others.iter().flatten() {
                    for p in region2d::intersections(s, o) { if let Some(t) = region2d::param_of(s, p) { ts.push(t); } }
                    for p in [o.start(), o.end()] {
                        if let Some(t) = region2d::param_of(s, p) {
                            let pp = region2d::point_at(s, t);
                            if (pp[0] - p[0]).hypot(pp[1] - p[1]) < 1e-7 { ts.push(t); }
                        }
                    }
                }
                ts
            };
            let key_v = |p: [f64; 2]| (q(p[0]), q(p[1]), q(layer.z0), q(layer.z1));
            let mut vedge = |b: &mut Brep, vert: &mut dyn FnMut(&mut Brep, [f64; 2], f64) -> u32, p: [f64; 2]| -> u32 {
                *vedges.entry(key_v(p)).or_insert_with(|| {
                    let (v0, v1) = (vert(b, p, layer.z0), vert(b, p, layer.z1));
                    let c = b.add_curve(Curve::Line { origin: f.to_world(v3(p[0], p[1], layer.z0)), dir: f.z * (layer.z1 - layer.z0) });
                    b.add_edge(c, v0, v1, 0.0, 1.0, format!("v.{}", b.edges.len()))
                })
            };
            for reg in &layer.regions {
                for ol in &reg.loops {
                    let lp = region2d::expand(&Profile { loops: vec![ol.clone()] }).remove(0);
                    if let [Seg2::Circle { c, r, ccw }] = ol.as_slice() {
                        // Full circle: ONE cylindrical face with a seam at angle 0 (CATIA
                        // convention). Each level uses a closed circle edge unless another
                        // boundary touches the circle there, then the split pieces.
                        let (c, r, ccw) = (*c, *r, *ccw);
                        // the same per-level decision as the horizontal faces (see `circle_level`)
                        let own: Vec<Vec<Seg2>> = layer.regions.iter().flat_map(region2d::expand).collect();
                        let bot_level: Vec<Vec<Seg2>> = below_b.iter().chain(&own).cloned().collect();
                        let top_level: Vec<Vec<Seg2>> = own.iter().chain(&above_b).cloned().collect();
                        let p0 = [c[0] + r, c[1]];
                        let mut lpv: Vec<(u32, bool)> = Vec::new();
                        match circle_level(c, r, ccw, &bot_level) {
                            None => lpv.push(cedge(&mut b, &mut vert, c, r, ccw, layer.z0)),
                            Some(pieces) => for pc in &pieces { lpv.push(hedge(&mut b, &mut vert, pc, layer.z0)); },
                        }
                        let seam = vedge(&mut b, &mut vert, p0);
                        lpv.push((seam, true));
                        let top: Vec<(u32, bool)> = match circle_level(c, r, ccw, &top_level) {
                            None => vec![cedge(&mut b, &mut vert, c, r, ccw, layer.z1)],
                            Some(pieces) => pieces.iter().map(|pc| hedge(&mut b, &mut vert, pc, layer.z1)).collect(),
                        };
                        for (e, fw) in top.into_iter().rev() { lpv.push((e, !fw)); }
                        lpv.push((seam, false));
                        let si = b.add_surface(Surface::Cylinder { frame: Frame { origin: f.to_world(v3(c[0], c[1], layer.z0)), ..f }, r });
                        b.add_face(si, ccw, shell, &lpv, &[], format!("side.{side_count}"), [0.0, core::f64::consts::TAU, 0.0, layer.z1 - layer.z0]);
                        side_count += 1;
                        continue;
                    }
                    for s in &lp {
                        let pieces_bot = region2d::split(s, splits(s, &below_b));
                        let pieces_top = region2d::split(s, splits(s, &above_b));
                        // one side face per original segment, bottom/top edges = pieces
                        let mut lpv: Vec<(u32, bool)> = Vec::new();
                        for pc in &pieces_bot { lpv.push(hedge(&mut b, &mut vert, pc, layer.z0)); }
                        let (sa, se) = (s.start(), s.end());
                        let ve_end = vedge(&mut b, &mut vert, se);
                        lpv.push((ve_end, true));
                        for pc in pieces_top.iter().rev() { let (e, fw) = hedge(&mut b, &mut vert, pc, layer.z1); lpv.push((e, !fw)); }
                        let ve_start = vedge(&mut b, &mut vert, sa);
                        lpv.push((ve_start, false));
                        let (surf, sense, uv) = match *s {
                            Seg2::Line { a, b: e2 } => {
                                let pa = f.to_world(v3(a[0], a[1], layer.z0));
                                let d = f.to_world(v3(e2[0], e2[1], layer.z0)) - pa;
                                (Surface::Plane { frame: Frame { origin: pa, x: d.normalized(), y: f.z, z: d.normalized().cross(f.z) } }, true, [0.0, d.norm(), 0.0, layer.z1 - layer.z0])
                            }
                            Seg2::Arc { c, r, a0, a1 } => (Surface::Cylinder { frame: Frame { origin: f.to_world(v3(c[0], c[1], layer.z0)), ..f }, r }, a1 > a0, [a0.min(a1), a0.max(a1), 0.0, layer.z1 - layer.z0]),
                            Seg2::Circle { .. } => unreachable!(),
                        };
                        let si = b.add_surface(surf);
                        b.add_face(si, sense, shell, &lpv, &[], format!("side.{side_count}"), uv);
                        side_count += 1;
                    }
                }
            }
        }
        // Horizontal faces at every level: up = below \ above, down = above \ below.
        let mut levels: Vec<f64> = self.layers.iter().flat_map(|l| [l.z0, l.z1]).collect();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let zmin = levels.first().copied().unwrap_or(0.0);
        let zmax = levels.last().copied().unwrap_or(0.0);
        for &z in &levels {
            let below: Vec<Profile> = self.layers.iter().filter(|l| (l.z1 - z).abs() < 1e-9).flat_map(|l| l.regions.clone()).collect();
            let above: Vec<Profile> = self.layers.iter().filter(|l| (l.z0 - z).abs() < 1e-9).flat_map(|l| l.regions.clone()).collect();
            for (regs, up) in [(region_op(&below, &above, BoolOp2::Difference), true), (region_op(&above, &below, BoolOp2::Difference), false)] {
                for reg in regs {
                    let reg = reg.normalize();
                    let mut loops: Vec<Vec<(u32, bool)>> = Vec::new();
                    let all: Vec<Vec<Seg2>> = below.iter().chain(&above).flat_map(region2d::expand).collect();
                    for ol in &reg.loops {
                        let lp = region2d::expand(&Profile { loops: vec![ol.clone()] }).remove(0);
                        // split at touching points of every boundary at this level
                        let mut pieces_all: Vec<Seg2> = Vec::new();
                        let mut touched = false;
                        for s in &lp {
                            let mut ts = Vec::new();
                            for o in all.iter().flatten() {
                                for p in [o.start(), o.end()] { if let Some(t) = region2d::param_of(s, p) { let pp = region2d::point_at(s, t); if (pp[0]-p[0]).hypot(pp[1]-p[1]) < 1e-7 { ts.push(t); } } }
                                for p in region2d::intersections(s, o) { if let Some(t) = region2d::param_of(s, p) { ts.push(t); } }
                            }
                            touched |= interior(&ts);
                            pieces_all.extend(region2d::split(s, ts));
                        }
                        let _ = touched;
                        let mut lv: Vec<(u32, bool)> = match ol.as_slice() {
                            // same decision as the side faces: closed edge unless touched
                            [Seg2::Circle { c, r, ccw }] => match circle_level(*c, *r, *ccw, &all) {
                                None => vec![cedge(&mut b, &mut vert, *c, *r, *ccw, z)],
                                Some(pieces) => pieces.iter().map(|pc| hedge(&mut b, &mut vert, pc, z)).collect(),
                            },
                            _ => pieces_all.iter().map(|pc| hedge(&mut b, &mut vert, pc, z)).collect(),
                        };
                        if !up { lv = lv.into_iter().rev().map(|(e, fw)| (e, !fw)).collect(); }
                        loops.push(lv);
                    }
                    if loops.is_empty() { continue; }
                    let pl = b.add_surface(Surface::Plane { frame: Frame { origin: f.to_world(v3(0.0, 0.0, z)), ..f } });
                    let name = if (z - zmin).abs() < 1e-9 && !up { "cap_start".to_string() } else if (z - zmax).abs() < 1e-9 && up { "cap_end".to_string() } else { format!("{}.{}", if up { "floor" } else { "ceiling" }, q(z)) };
                    b.add_face(pl, up, shell, &loops[0], &loops[1..], name, [-1e9, 1e9, -1e9, 1e9]);
                }
            }
        }
        let _ = V3::ZERO;
        b.split_shells();
        b
    }
}

/// Does a split-parameter list contain a split strictly inside the segment?
fn interior(ts: &[f64]) -> bool { ts.iter().any(|t| *t > 1e-9 && *t < 1.0 - 1e-9) }

fn same_circle(s: &Seg2, c: [f64; 2], r: f64) -> bool {
    let near = |c2: [f64; 2], r2: f64| (c2[0] - c[0]).abs() < 1e-9 && (c2[1] - c[1]).abs() < 1e-9 && (r2 - r).abs() < 1e-9;
    match *s { Seg2::Arc { c: c2, r: r2, .. } | Seg2::Circle { c: c2, r: r2, .. } => near(c2, r2), Seg2::Line { .. } => false }
}

/// Parameters on `s` where another segment touches it (intersections and endpoints).
fn contact_params(s: &Seg2, others: &[Seg2]) -> Vec<f64> {
    let mut ts = Vec::new();
    for o in others {
        for p in region2d::intersections(s, o) { if let Some(t) = region2d::param_of(s, p) { ts.push(t); } }
        for p in [o.start(), o.end()] {
            if let Some(t) = region2d::param_of(s, p) {
                let pp = region2d::point_at(s, t);
                if (pp[0] - p[0]).hypot(pp[1] - p[1]) < 1e-7 { ts.push(t); }
            }
        }
    }
    ts
}

/// Edge decision for a full circle at one level, shared by side and horizontal faces so their
/// edges always match: `None` = one closed circle edge (vertex at angle 0) when no boundary at
/// that level other than the circle itself touches it, endpoints included; otherwise the two
/// half-arcs split at the interior contact parameters.
fn circle_level(c: [f64; 2], r: f64, ccw: bool, level: &[Vec<Seg2>]) -> Option<Vec<Seg2>> {
    let halves = region2d::expand(&Profile { loops: vec![vec![Seg2::Circle { c, r, ccw }]] }).remove(0);
    let others: Vec<Seg2> = level.iter().flatten().filter(|s| !same_circle(s, c, r)).copied().collect();
    // contact anywhere on the circle, including the half-arc endpoints (angles 0 and π)
    let on_circle = |p: [f64; 2]| ((p[0] - c[0]).hypot(p[1] - c[1]) - r).abs() < 1e-7;
    let touched = others.iter().any(|o| on_circle(o.start()) || on_circle(o.end()) || halves.iter().any(|h| !region2d::intersections(h, o).is_empty()));
    let mut pieces = Vec::new();
    for h in &halves {
        let ts = contact_params(h, &others);
        pieces.extend(region2d::split(h, ts.into_iter().filter(|t| *t > 1e-9 && *t < 1.0 - 1e-9).collect()));
    }
    if touched { Some(pieces) } else { None }
}

fn same_regions(a: &[Profile], b: &[Profile]) -> bool {
    a.len() == b.len() && (a.iter().map(|r| r.area()).sum::<f64>() - b.iter().map(|r| r.area()).sum::<f64>()).abs() < 1e-9 && region_op(a, b, BoolOp2::Difference).iter().map(|r| r.area()).sum::<f64>().abs() < 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessellate::{tessellate, TessOptions};

    #[test]
    fn plate_with_blind_pocket_and_through_hole() {
        let f = Frame::WORLD;
        let plate = LayeredSolid::extrusion(f, &Profile::rectangle(0.0, 0.0, 100.0, 60.0), 0.0, 20.0);
        let pocket = LayeredSolid::extrusion(f, &Profile::rectangle(20.0, 10.0, 60.0, 50.0), 12.0, 25.0);
        let hole = LayeredSolid::extrusion(f, &Profile::circle(80.0, 30.0, 5.0), -1.0, 21.0);
        let s = plate.boolean(&pocket, BoolOp2::Difference).boolean(&hole, BoolOp2::Difference);
        let exact = 100.0 * 60.0 * 20.0 - 40.0 * 40.0 * 8.0 - core::f64::consts::PI * 25.0 * 20.0;
        assert!((s.volume() - exact).abs() < 1e-6, "{}", s.volume());
        let b = s.to_brep();
        let r = b.validate();
        assert!(r.valid, "{:?}", r.problems);
        let m = tessellate(&b, &TessOptions { sag: 1e-4, ..Default::default() });
        assert!(m.is_watertight());
        let (vol, _, _) = m.mass_properties();
        assert!((vol - exact).abs() / exact < 1e-5, "{vol} vs {exact}");
    }
}

impl LayeredSolid {
    /// Fillet/chamfer vertical edges at 2D corner points (empty = all line–line corners).
    pub fn treat_vertical_edges(&self, at: &[[f64; 2]], c: crate::region2d::Corner) -> Result<LayeredSolid, (usize, [f64; 2], f64)> {
        // Treat each loop on its own (outer corners lose material, hole corners gain it), then
        // rebuild the region by exact booleans, so a treated corner that runs into another
        // loop (e.g. a chamfer cutting through a nearby hole) resolves to valid geometry.
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let mut regions = Vec::new();
            for r in &l.regions {
                let single = |lp: &Vec<Seg2>| Profile { loops: vec![lp.clone()] }.normalize();
                let mut acc = vec![crate::region2d::treat_corners(&single(&r.loops[0]), at, c)?.normalize()];
                for h in &r.loops[1..] {
                    let th = crate::region2d::treat_corners(&single(h), at, c)?.normalize();
                    acc = region_op(&acc, &[th], BoolOp2::Difference);
                }
                regions.extend(acc);
            }
            layers.push(Layer { z0: l.z0, z1: l.z1, regions });
        }
        Ok(LayeredSolid { frame: self.frame, layers })
    }
}

impl LayeredSolid {
    /// Closed shell (no faces removed) of a single-slab solid: the solid minus its erosion by
    /// a ball of radius `t`, i.e. minus the 2D inward offset of every region extruded over
    /// [z0 + t, z1 − t] (exact: the distance to a prism's boundary is the minimum of the 2D
    /// distance to its walls and the distances to its caps).
    pub fn shell_closed(&self, t: f64) -> Result<LayeredSolid, String> {
        if !(t > 0.0) { return Err("shell thickness must be > 0".into()); }
        let [layer] = self.layers.as_slice() else {
            return Err(format!("exact shell implemented for single-slab prismatic bodies (this body has {} slabs)", self.layers.len()));
        };
        if layer.z1 - layer.z0 <= 2.0 * t { return Ok(self.clone()); }
        let mut core = Vec::new();
        for r in &layer.regions { core.extend(region2d::offset_inward(r, t)?); }
        if core.is_empty() { return Ok(self.clone()); }
        let core = LayeredSolid { frame: self.frame, layers: vec![Layer { z0: layer.z0 + t, z1: layer.z1 - t, regions: core }] };
        Ok(self.boolean(&core, BoolOp2::Difference))
    }
}

impl LayeredSolid {
    /// Full revolution about the vertical axis through `axis` (world frame) of a meridian
    /// region given in (ρ, z) coordinates (ρ ≥ 0). Exact as a layered solid when every meridian
    /// segment is axis-aligned (cylinders and annular caps); otherwise `Err` (cones / tori need
    /// the general kernel).
    pub fn revolve_rectilinear(axis: [f64; 2], meridian: &Profile) -> Result<LayeredSolid, String> {
        let loops = region2d::expand(meridian);
        let (mut zs, mut rs): (Vec<f64>, Vec<(f64, f64, f64)>) = (Vec::new(), Vec::new());
        for lp in &loops {
            for s in lp {
                let Seg2::Line { a, b } = *s else { return Err("meridian has arcs (tori / spheres): not a layered solid".into()) };
                if a[0] < -1e-9 || b[0] < -1e-9 { return Err("meridian crosses the axis (ρ < 0)".into()); }
                if (a[0] - b[0]).abs() < 1e-12 { rs.push((a[0], a[1].min(b[1]), a[1].max(b[1]))); } else if (a[1] - b[1]).abs() >= 1e-12 { return Err("sloped meridian segment (cone): not a layered solid".into()); }
                zs.push(a[1]);
                zs.push(b[1]);
            }
        }
        zs.sort_by(|p, q| p.partial_cmp(q).unwrap());
        zs.dedup_by(|p, q| (*p - *q).abs() < 1e-12);
        let mut layers: Vec<Layer> = Vec::new();
        for w in zs.windows(2) {
            let zm = 0.5 * (w[0] + w[1]);
            // radii of the vertical walls crossing this slab, and the inside intervals between them
            let mut cut: Vec<f64> = rs.iter().filter(|(_, z0, z1)| *z0 < zm && zm < *z1).map(|(r, _, _)| *r).collect();
            cut.push(0.0);
            cut.sort_by(|p, q| p.partial_cmp(q).unwrap());
            cut.dedup_by(|p, q| (*p - *q).abs() < 1e-12);
            let mut regions: Vec<Profile> = Vec::new();
            for iv in cut.windows(2) {
                if region2d::winding(&loops, [0.5 * (iv[0] + iv[1]), zm]).round() as i64 <= 0 { continue; }
                let outer = Profile::circle(axis[0], axis[1], iv[1]);
                regions.push(if iv[0] > 1e-12 { outer.with_hole(Profile::circle(axis[0], axis[1], iv[0])) } else { outer.normalize() });
            }
            if regions.is_empty() { continue; }
            if let Some(last) = layers.last_mut() {
                if (last.z1 - w[0]).abs() < 1e-12 && same_regions(&last.regions, &regions) { last.z1 = w[1]; continue; }
            }
            layers.push(Layer { z0: w[0], z1: w[1], regions });
        }
        if layers.is_empty() { return Err("empty revolution".into()); }
        Ok(LayeredSolid { frame: Frame::WORLD, layers })
    }

    /// Reflection across the plane x = 0 (`axis` 0) or y = 0 (`axis` 1) of the frame.
    pub fn reflected(&self, axis: usize) -> LayeredSolid {
        let f = |p: [f64; 2]| if axis == 0 { [-p[0], p[1]] } else { [p[0], -p[1]] };
        let ang = |t: f64| if axis == 0 { std::f64::consts::PI - t } else { -t };
        let refl = |s: &Seg2| match *s {
            Seg2::Line { a, b } => Seg2::Line { a: f(a), b: f(b) },
            Seg2::Arc { c, r, a0, a1 } => Seg2::Arc { c: f(c), r, a0: ang(a0), a1: ang(a1) },
            Seg2::Circle { c, r, ccw } => Seg2::Circle { c: f(c), r, ccw: !ccw },
        };
        LayeredSolid {
            frame: self.frame,
            layers: self.layers.iter().map(|l| Layer { z0: l.z0, z1: l.z1, regions: l.regions.iter().map(|p| Profile { loops: p.loops.iter().map(|lp| lp.iter().map(refl).collect()).collect() }.normalize()).collect() }).collect(),
        }
    }
}
