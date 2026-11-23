//! Exact-geometry 2D region booleans on curved polygons (lines, circular arcs, circles):
//! split all boundary segments at mutual intersections, classify each piece by winding
//! number of the other region at its midpoint, keep pieces per operation, chain into loops.
//! Used by the layered-extrusion kernel (pad/pocket/hole stacks) and sketch regions.

use crate::builders::{Profile, Seg2};
use std::f64::consts::TAU;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoolOp2 {
    Union,
    Difference,
    Intersection,
}

const EPS: f64 = 1e-9;

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 { ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() }
fn norm_angle(a: f64) -> f64 { let mut x = a % TAU; if x < 0.0 { x += TAU; } x }

/// Expand full circles into two half arcs so every segment has distinct endpoints.
pub(crate) fn expand(p: &Profile) -> Vec<Vec<Seg2>> {
    p.loops.iter().map(|lp| lp.iter().flat_map(|s| match *s {
        Seg2::Circle { c, r, ccw } => if ccw {
            vec![Seg2::Arc { c, r, a0: 0.0, a1: TAU / 2.0 }, Seg2::Arc { c, r, a0: TAU / 2.0, a1: TAU }]
        } else {
            vec![Seg2::Arc { c, r, a0: TAU, a1: TAU / 2.0 }, Seg2::Arc { c, r, a0: TAU / 2.0, a1: 0.0 }]
        },
        other => vec![other],
    }).collect()).collect()
}

pub(crate) fn point_at(s: &Seg2, t: f64) -> [f64; 2] {
    match *s {
        Seg2::Line { a, b } => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t],
        Seg2::Arc { c, r, a0, a1 } => { let a = a0 + (a1 - a0) * t; [c[0] + r * a.cos(), c[1] + r * a.sin()] }
        Seg2::Circle { c, r, ccw } => { let a = if ccw { TAU * t } else { TAU * (1.0 - t) }; [c[0] + r * a.cos(), c[1] + r * a.sin()] }
    }
}

/// Parameter t ∈ (0,1) of point `p` on segment `s` (assumes p lies on the segment's carrier).
pub(crate) fn param_of(s: &Seg2, p: [f64; 2]) -> Option<f64> {
    match *s {
        Seg2::Line { a, b } => {
            let d = sub(b, a);
            let l2 = d[0] * d[0] + d[1] * d[1];
            let t = (sub(p, a)[0] * d[0] + sub(p, a)[1] * d[1]) / l2;
            if t > EPS && t < 1.0 - EPS { Some(t) } else { None }
        }
        Seg2::Arc { c, a0, a1, .. } => {
            let ang = (p[1] - c[1]).atan2(p[0] - c[0]);
            let sweep = a1 - a0;
            let rel = if sweep >= 0.0 { norm_angle(ang - a0) } else { norm_angle(a0 - ang) };
            let t = rel / sweep.abs();
            if t > EPS && t < 1.0 - EPS { Some(t) } else { None }
        }
        _ => None,
    }
}

/// Intersection points of two segments' carriers restricted to the segments.
pub(crate) fn intersections(s: &Seg2, o: &Seg2) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    match (*s, *o) {
        (Seg2::Line { a, b }, Seg2::Line { a: c, b: d }) => {
            let r = sub(b, a);
            let q = sub(d, c);
            let den = r[0] * q[1] - r[1] * q[0];
            // relative parallelism test: rounding noise in near-collinear directions must not
            // turn an overlap into a spurious crossing point
            if den.abs() > 1e-12 * (r[0].hypot(r[1]) * q[0].hypot(q[1])) {
                let t = (sub(c, a)[0] * q[1] - sub(c, a)[1] * q[0]) / den;
                let u = (sub(c, a)[0] * r[1] - sub(c, a)[1] * r[0]) / den;
                if (-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u) { pts.push([a[0] + r[0] * t, a[1] + r[1] * t]); }
            } else {
                // collinear overlap: endpoints of one inside the other
                for p in [c, d] { if param_of(s, p).is_some() && dist_to_line(p, a, b) < EPS { pts.push(p); } }
                for p in [a, b] { if param_of(o, p).is_some() && dist_to_line(p, c, d) < EPS { pts.push(p); } }
            }
        }
        (Seg2::Line { a, b }, Seg2::Arc { c, r, .. }) | (Seg2::Arc { c, r, .. }, Seg2::Line { a, b }) => {
            let d = sub(b, a);
            let f = sub(a, c);
            let (qa, qb, qc) = (d[0] * d[0] + d[1] * d[1], 2.0 * (f[0] * d[0] + f[1] * d[1]), f[0] * f[0] + f[1] * f[1] - r * r);
            let disc = qb * qb - 4.0 * qa * qc;
            if disc >= -1e-12 {
                let sq = disc.max(0.0).sqrt();
                for t in [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)] {
                    if (-EPS..=1.0 + EPS).contains(&t) { pts.push([a[0] + d[0] * t, a[1] + d[1] * t]); }
                }
            }
        }
        (Seg2::Arc { c: c1, r: r1, .. }, Seg2::Arc { c: c2, r: r2, .. }) => {
            let d = dist(c1, c2);
            if d > EPS && d <= r1 + r2 + EPS && d >= (r1 - r2).abs() - EPS {
                let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
                let h = (r1 * r1 - a * a).max(0.0).sqrt();
                let m = [c1[0] + a * (c2[0] - c1[0]) / d, c1[1] + a * (c2[1] - c1[1]) / d];
                pts.push([m[0] + h * (c2[1] - c1[1]) / d, m[1] - h * (c2[0] - c1[0]) / d]);
                if h > EPS { pts.push([m[0] - h * (c2[1] - c1[1]) / d, m[1] + h * (c2[0] - c1[0]) / d]); }
            }
        }
        _ => {}
    }
    // keep only points lying on both segments (arcs: within angular span)
    pts.retain(|p| on_segment(s, *p) && on_segment(o, *p));
    pts
}

fn dist_to_line(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = sub(b, a);
    ((p[0] - a[0]) * d[1] - (p[1] - a[1]) * d[0]).abs() / (d[0] * d[0] + d[1] * d[1]).sqrt()
}

/// Return the first pair of section edges that meet away from their shared endpoints.
pub fn crossing_edges(edges: &[Seg2], tolerance: f64) -> Option<(usize, usize)> {
    for (i, a) in edges.iter().enumerate() {
        for (j, b) in edges.iter().enumerate().skip(i + 1) {
            for p in intersections(a, b) {
                if j == i + 1 && dist(p, a.end()) <= tolerance { continue; }
                if i == 0 && j + 1 == edges.len() && dist(p, a.start()) <= tolerance { continue; }
                return Some((i, j));
            }
        }
    }
    None
}

fn on_segment(s: &Seg2, p: [f64; 2]) -> bool {
    match *s {
        Seg2::Line { a, b } => {
            let d = sub(b, a);
            let l2 = d[0] * d[0] + d[1] * d[1];
            let t = (sub(p, a)[0] * d[0] + sub(p, a)[1] * d[1]) / l2;
            (-EPS..=1.0 + EPS).contains(&t) && dist_to_line(p, a, b) < 1e-7
        }
        Seg2::Arc { c, r, a0, a1 } => {
            if (dist(p, c) - r).abs() > 1e-7 { return false; }
            let ang = (p[1] - c[1]).atan2(p[0] - c[0]);
            let sweep = a1 - a0;
            let rel = if sweep >= 0.0 { norm_angle(ang - a0) } else { norm_angle(a0 - ang) };
            rel <= sweep.abs() + 1e-9 || rel >= TAU - 1e-9
        }
        _ => false,
    }
}

/// Split segment `s` at parameters `ts` (sorted, inside (0,1)).
pub(crate) fn split(s: &Seg2, mut ts: Vec<f64>) -> Vec<Seg2> {
    // Split points within 1e-7 mm of the segment's own endpoints are the endpoints (tangent
    // contacts such as fillet arc ↔ line yield intersection points ~√ε off); splitting there
    // would create sliver pieces that the neighbouring faces do not have.
    let (s0, s1) = (s.start(), s.end());
    ts.retain(|&t| { let p = point_at(s, t); dist(p, s0) > 1e-7 && dist(p, s1) > 1e-7 });
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut out = Vec::new();
    let mut prev = 0.0;
    for t in ts.into_iter().chain(std::iter::once(1.0)) {
        out.push(match *s {
            Seg2::Line { .. } => Seg2::Line { a: point_at(s, prev), b: point_at(s, t) },
            Seg2::Arc { c, r, a0, a1 } => Seg2::Arc { c, r, a0: a0 + (a1 - a0) * prev, a1: a0 + (a1 - a0) * t },
            Seg2::Circle { .. } => unreachable!(),
        });
        prev = t;
    }
    out
}

/// Winding number of `p` w.r.t. a region (sum over loops; arcs integrated exactly by angle).
/// Exact winding angle of an arc about `p`: the chord A→B, plus ±2π when `p` lies in the
/// circular segment between chord and arc (arc + chord-back is a closed curve around it).
/// If `p` lies on the chord itself the chord angle is ±π-ambiguous, so the arc is split.
fn arc_winding(c: [f64; 2], r: f64, a0: f64, a1: f64, p: [f64; 2], depth: u32) -> f64 {
    let at = |a: f64| [c[0] + r * a.cos(), c[1] + r * a.sin()];
    let (a, b) = (at(a0), at(a1));
    let (ua, ub) = (sub(a, p), sub(b, p));
    let cross = ua[0] * ub[1] - ua[1] * ub[0];
    let dot = ua[0] * ub[0] + ua[1] * ub[1];
    let l2 = dist(a, b).powi(2);
    if depth < 8 && cross.abs() <= 1e-12 * l2.max(1e-300) && dot < 0.0 {
        let am = 0.5 * (a0 + a1);
        return arc_winding(c, r, a0, am, p, depth + 1) + arc_winding(c, r, am, a1, p, depth + 1);
    }
    let mut w = cross.atan2(dot);
    if dist(p, c) < r {
        let m = at(0.5 * (a0 + a1));
        let side = |q: [f64; 2]| { let (d, e) = (sub(b, a), sub(q, a)); d[0] * e[1] - d[1] * e[0] };
        if side(p) * side(m) > 0.0 { w += if a1 > a0 { TAU } else { -TAU }; }
    }
    w
}

pub fn winding(loops: &[Vec<Seg2>], p: [f64; 2]) -> f64 {
    let mut w = 0.0;
    for lp in loops {
        for s in lp {
            match *s {
                Seg2::Line { a, b } => {
                    let (ua, ub) = (sub(a, p), sub(b, p));
                    w += (ua[0] * ub[1] - ua[1] * ub[0]).atan2(ua[0] * ub[0] + ua[1] * ub[1]);
                }
                Seg2::Arc { c, r, a0, a1 } => w += arc_winding(c, r, a0, a1, p, 0),
                Seg2::Circle { c, r, ccw } => if dist(p, c) < r { w += if ccw { TAU } else { -TAU } },
            }
        }
    }
    w / TAU
}

fn inside(loops: &[Vec<Seg2>], p: [f64; 2]) -> bool { winding(loops, p).round() as i64 > 0 }

/// Boolean of two normalized profiles. Returns a normalized profile list (possibly several
/// disjoint regions, each as its own Profile with holes).
pub fn boolean(a: &Profile, b: &Profile, op: BoolOp2) -> Vec<Profile> {
    let la = expand(a);
    let lb = expand(b);
    let split_all = |own: &[Vec<Seg2>], other: &[Vec<Seg2>]| -> Vec<Seg2> {
        let mut out = Vec::new();
        for lp in own {
            for s in lp {
                let mut ts = Vec::new();
                for olp in other { for o in olp { for p in intersections(s, o) { if let Some(t) = param_of(s, p) { ts.push(t); } } } }
                out.extend(split(s, ts));
            }
        }
        out
    };
    let pa = split_all(&la, &lb);
    let pb = split_all(&lb, &la);
    let mut keep: Vec<Seg2> = Vec::new();
    let mid = |s: &Seg2| point_at(s, 0.5);
    for s in &pa {
        let ins = inside(&lb, mid(s));
        let on_b = lb.iter().flatten().any(|o| on_segment(o, mid(s)));
        match op {
            BoolOp2::Union => if !ins && !on_b { keep.push(*s) } else if on_b && same_dir_on(&lb, s) { keep.push(*s) },
            BoolOp2::Intersection => if ins && !on_b { keep.push(*s) } else if on_b && same_dir_on(&lb, s) { keep.push(*s) },
            BoolOp2::Difference => if !ins && !on_b { keep.push(*s) } else if on_b && !same_dir_on(&lb, s) { keep.push(*s) },
        }
    }
    for s in &pb {
        let ins = inside(&la, mid(s));
        let on_a = la.iter().flatten().any(|o| on_segment(o, mid(s)));
        if on_a { continue; } // coincident pieces handled from `a`'s side
        match op {
            BoolOp2::Union => if !ins { keep.push(*s) },
            BoolOp2::Intersection => if ins { keep.push(*s) },
            BoolOp2::Difference => if ins { keep.push(s.reversed()) },
        }
    }
    chain(keep)
}

/// Is the coincident boundary piece `s` oriented the same way as the matching piece of `other`?
fn same_dir_on(other: &[Vec<Seg2>], s: &Seg2) -> bool {
    let m = point_at(s, 0.5);
    let m2 = point_at(s, 0.51);
    let ds = sub(m2, m);
    for o in other.iter().flatten() {
        if on_segment(o, m) {
            if let Some(t) = param_of(o, m) {
                let d = sub(point_at(o, (t + 0.01).min(1.0)), point_at(o, t));
                return ds[0] * d[0] + ds[1] * d[1] > 0.0;
            }
        }
    }
    false
}

/// Chain kept pieces into closed loops; group loops into regions (outer + holes).
fn chain(mut segs: Vec<Seg2>) -> Vec<Profile> {
    let mut loops: Vec<Vec<Seg2>> = Vec::new();
    while let Some(first) = segs.pop() {
        let mut lp = vec![first];
        let start = first.start();
        let mut guard = 0;
        while dist(lp.last().unwrap().end(), start) > 1e-7 && guard < 100_000 {
            guard += 1;
            let end = lp.last().unwrap().end();
            let Some(k) = segs.iter().position(|s| dist(s.start(), end) < 1e-7) else { break };
            lp.push(segs.swap_remove(k));
        }
        if dist(lp.last().unwrap().end(), start) <= 1e-7 && lp.len() >= 1 { loops.push(merge_arcs(lp)); }
    }
    // Outer loops (CCW) own the holes (CW) they contain.
    let (outers, holes): (Vec<_>, Vec<_>) = loops.into_iter().partition(|l| Profile::loop_area(l) > 0.0);
    let mut regions: Vec<Profile> = outers.into_iter().map(|o| Profile { loops: vec![o] }).collect();
    for h in holes {
        let p = point_at(&h[0], 0.5);
        let probe = [p[0] + 1e-6, p[1]];
        if let Some(r) = regions.iter_mut().filter(|r| inside(&r.loops[..1], probe) || inside(&r.loops[..1], p)).min_by(|a, b| Profile::loop_area(&a.loops[0]).partial_cmp(&Profile::loop_area(&b.loops[0])).unwrap()) {
            r.loops.push(h);
        }
    }
    regions
}

/// Re-merge consecutive arcs of the same circle (and a full turn back into a Circle).
fn merge_arcs(lp: Vec<Seg2>) -> Vec<Seg2> { canonical_loop(lp) }

/// Merge two consecutive pieces that are one segment: collinear same-direction lines, or
/// contiguous same-sense arcs of one circle.
fn merge_pair(x: Seg2, y: Seg2) -> Option<Seg2> {
    match (x, y) {
        (Seg2::Line { a, b }, Seg2::Line { a: b2, b: c }) if dist(b, b2) < 1e-9 => {
            let (u, v) = (sub(b, a), sub(c, b));
            let cross = u[0] * v[1] - u[1] * v[0];
            let dot = u[0] * v[0] + u[1] * v[1];
            let scale = (u[0].hypot(u[1])) * (v[0].hypot(v[1]));
            (dot > 0.0 && cross.abs() <= 1e-10 * scale).then_some(Seg2::Line { a, b: c })
        }
        (Seg2::Arc { c, r, a0, a1 }, Seg2::Arc { c: c2, r: r2, a0: b0, a1: b1 })
            if dist(c, c2) < 1e-9 && (r - r2).abs() < 1e-9 && ((a1 - b0).abs() < 1e-9 || (norm_angle(a1) - norm_angle(b0)).abs() < 1e-9) && (a1 - a0).signum() == (b1 - b0).signum() =>
        {
            Some(Seg2::Arc { c, r, a0, a1: a1 + (b1 - b0) })
        }
        _ => None,
    }
}

fn start_dir(s: &Seg2) -> [f64; 2] {
    match *s {
        Seg2::Line { a, b } => sub(b, a),
        Seg2::Arc { a0, a1, .. } => { let g = (a1 - a0).signum(); [-a0.sin() * g, a0.cos() * g] }
        Seg2::Circle { ccw, .. } => if ccw { [0.0, 1.0] } else { [0.0, -1.0] },
    }
}
fn end_dir(s: &Seg2) -> [f64; 2] {
    match *s {
        Seg2::Line { a, b } => sub(b, a),
        Seg2::Arc { a0, a1, .. } => { let g = (a1 - a0).signum(); [-a1.sin() * g, a1.cos() * g] }
        Seg2::Circle { ccw, .. } => if ccw { [0.0, 1.0] } else { [0.0, -1.0] },
    }
}

/// Is `p` within `tol` (distance) of segment `s`? Tangent intersection points are
/// ill-conditioned (error ~ √ε·r), so contact tests use a distance tolerance, not angles.
fn near_segment(s: &Seg2, p: [f64; 2], tol: f64) -> bool {
    match *s {
        Seg2::Line { a, b } => {
            let d = sub(b, a);
            let l2 = d[0] * d[0] + d[1] * d[1];
            let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2.max(1e-300)).clamp(0.0, 1.0);
            dist(p, [a[0] + d[0] * t, a[1] + d[1] * t]) < tol
        }
        Seg2::Arc { c, r, a0, a1 } => {
            if (dist(p, c) - r).abs() >= tol { return false; }
            let ang = (p[1] - c[1]).atan2(p[0] - c[0]);
            let sweep = a1 - a0;
            let rel = if sweep >= 0.0 { norm_angle(ang - a0) } else { norm_angle(a0 - ang) };
            let slack = tol / r.max(1e-300);
            rel <= sweep.abs() + slack || rel >= TAU - slack
        }
        Seg2::Circle { c, r, .. } => (dist(p, c) - r).abs() < tol,
    }
}

/// Carrier of a segment (circle centre and radius, or a line's unit direction up to sign and
/// its offset): two pieces of the same wall share a carrier.
fn carrier(s: &Seg2) -> [f64; 4] {
    match *s {
        Seg2::Line { a, b } => {
            let d = sub(b, a);
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let (mut ux, mut uy) = (d[0] / l, d[1] / l);
            if ux < -1e-12 || (ux.abs() <= 1e-12 && uy < 0.0) { ux = -ux; uy = -uy; }
            [0.0, ux, uy, a[0] * uy - a[1] * ux]
        }
        Seg2::Arc { c, r, .. } | Seg2::Circle { c, r, .. } => [1.0 + r, c[0], c[1], r],
    }
}

/// Smooth (G1) junctions of consecutive segments: (point, carriers of the two pieces). A wall
/// that continues across a layer boundary meets itself tangentially there by construction.
pub fn smooth_junctions(loops: &[Vec<Seg2>]) -> Vec<([f64; 2], [[f64; 4]; 2])> {
    let mut out = Vec::new();
    for lp in loops {
        for i in 0..lp.len() {
            let (p, q) = (&lp[i], &lp[(i + 1) % lp.len()]);
            if matches!(p, Seg2::Line { .. }) && matches!(q, Seg2::Line { .. }) { continue; }
            let (u, v) = (end_dir(p), start_dir(q));
            let (nu, nv) = ((u[0] * u[0] + u[1] * u[1]).sqrt(), (v[0] * v[0] + v[1] * v[1]).sqrt());
            if nu > 0.0 && nv > 0.0 && (u[0] * v[1] - u[1] * v[0]).abs() < 1e-9 * nu * nv && u[0] * v[0] + u[1] * v[1] > 0.0 { out.push((p.end(), [carrier(p), carrier(q)])); }
        }
    }
    out
}

/// Junction points of `a` that are the same wall junction in `b` (same point, same carriers).
pub fn shared_junctions(a: &[Vec<Seg2>], b: &[Vec<Seg2>], tol: f64) -> Vec<[f64; 2]> {
    let jb = smooth_junctions(b);
    let same = |x: &[f64; 4], y: &[f64; 4]| (0..4).all(|k| (x[k] - y[k]).abs() < tol);
    smooth_junctions(a).into_iter().filter(|(p, ca)| jb.iter().any(|(q, cb)| dist(*p, *q) < tol && ca.iter().all(|x| cb.iter().any(|y| same(x, y))))).map(|(p, _)| p).collect()
}

/// Point where two segments touch tangentially (zero-angle contact), if any.
fn tangent_point(x: &Seg2, y: &Seg2, tol: f64) -> Option<[f64; 2]> {
    match (*x, *y) {
        (Seg2::Line { a, b }, Seg2::Arc { c, r, .. }) | (Seg2::Arc { c, r, .. }, Seg2::Line { a, b }) => {
            let d = sub(b, a);
            let l2 = d[0] * d[0] + d[1] * d[1];
            if l2 < 1e-24 { return None; }
            let t = ((c[0] - a[0]) * d[0] + (c[1] - a[1]) * d[1]) / l2;
            let foot = [a[0] + d[0] * t, a[1] + d[1] * t];
            let ok = (-1e-9..=1.0 + 1e-9).contains(&t) && (dist(foot, c) - r).abs() < tol;
            let arc = if let Seg2::Arc { .. } = x { x } else { y };
            (ok && near_segment(arc, foot, tol)).then_some(foot)
        }
        (Seg2::Arc { c: c1, r: r1, .. }, Seg2::Arc { c: c2, r: r2, .. }) => {
            let d = dist(c1, c2);
            if d < tol { return None; } // concentric: coincident or disjoint, not a point contact
            let u = [(c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d];
            let p = if (d - (r1 + r2)).abs() < tol {
                [c1[0] + u[0] * r1, c1[1] + u[1] * r1]
            } else if (d - (r1 - r2).abs()).abs() < tol {
                if r1 >= r2 { [c1[0] + u[0] * r1, c1[1] + u[1] * r1] } else { [c1[0] - u[0] * r1, c1[1] - u[1] * r1] }
            } else {
                return None;
            };
            (near_segment(x, p, tol) && near_segment(y, p, tol)).then_some(p)
        }
        _ => None,
    }
}

/// Tangential point contacts between boundary loops (`loops` already expanded: lines and
/// arcs only). Segments adjacent in the same loop are excluded (fillet arcs are tangent to
/// their neighbours by design). A contact pinches material to zero thickness — non-manifold.
pub fn tangent_contacts(loops: &[Vec<Seg2>], tol: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let segs: Vec<(usize, usize, &Seg2)> = loops.iter().enumerate().flat_map(|(li, l)| l.iter().enumerate().map(move |(si, s)| (li, si, s))).collect();
    for i in 0..segs.len() {
        for k in i + 1..segs.len() {
            let ((la, sa, x), (lb, sb, y)) = (segs[i], segs[k]);
            if la == lb {
                // consecutive segments meeting smoothly (fillet arc ↔ line) are tangent by design;
                // a direction reversal at the junction is a pinch and is checked
                let n = loops[la].len();
                let smooth = |p: &Seg2, q: &Seg2| { let (u, v) = (end_dir(p), start_dir(q)); u[0] * v[0] + u[1] * v[1] > 0.0 };
                if (sa + 1) % n == sb && smooth(x, y) { continue; }
                if (sb + 1) % n == sa && smooth(y, x) { continue; }
            }
            if let Some(p) = tangent_point(x, y, tol) {
                if !out.iter().any(|q: &[f64; 2]| dist(*q, p) < tol) { out.push(p); }
            }
        }
    }
    out
}

/// Canonical loop: pieces that are one segment are merged (also across the loop's wrap-around),
/// so a loop's segmentation does not depend on the history of booleans that produced it; a
/// single full-turn arc becomes a `Circle`.
pub(crate) fn canonical_loop(lp: Vec<Seg2>) -> Vec<Seg2> {
    let mut out: Vec<Seg2> = Vec::new();
    for s in lp {
        match out.last().copied().and_then(|l| merge_pair(l, s)) {
            Some(m) => { out.pop(); out.push(m); }
            None => out.push(s),
        }
    }
    while out.len() > 1 {
        let (l, f) = (out[out.len() - 1], out[0]);
        match merge_pair(l, f) {
            Some(m) => { out.pop(); out[0] = m; }
            None => break,
        }
    }
    if out.len() == 1 {
        if let Seg2::Arc { c, r, a0, a1 } = out[0] {
            if ((a1 - a0).abs() - TAU).abs() < 1e-9 { return vec![Seg2::Circle { c, r, ccw: a1 > a0 }]; }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn square_minus_circle_area() {
        let sq = Profile::rectangle(-10.0, -10.0, 10.0, 10.0).normalize();
        let c = Profile::circle(0.0, 0.0, 3.0).normalize();
        let r = boolean(&sq, &c, BoolOp2::Difference);
        assert_eq!(r.len(), 1);
        let area = r[0].area();
        assert!((area - (400.0 - core::f64::consts::PI * 9.0)).abs() < 1e-9, "{area}");
    }
    #[test]
    fn overlapping_squares_union_and_intersection() {
        let a = Profile::rectangle(0.0, 0.0, 2.0, 2.0).normalize();
        let b = Profile::rectangle(1.0, 1.0, 3.0, 3.0).normalize();
        let u: f64 = boolean(&a, &b, BoolOp2::Union).iter().map(|p| p.area()).sum();
        let i: f64 = boolean(&a, &b, BoolOp2::Intersection).iter().map(|p| p.area()).sum();
        assert!((u - 7.0).abs() < 1e-9 && (i - 1.0).abs() < 1e-9, "u={u} i={i}");
    }
    #[test]
    fn circle_crossing_edge() {
        let sq = Profile::rectangle(0.0, 0.0, 10.0, 10.0).normalize();
        let c = Profile::circle(10.0, 5.0, 2.0).normalize();
        let d: f64 = boolean(&sq, &c, BoolOp2::Difference).iter().map(|p| p.area()).sum();
        assert!((d - (100.0 - core::f64::consts::PI * 2.0)).abs() < 1e-8, "{d}");
    }
}

/// Corner treatment for vertical-edge fillets/chamfers on layered bodies.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Corner {
    Fillet(f64),
    Chamfer(f64),
}

/// Round/chamfer the corners of `p` located at `at` (within 1e-6), or all line–line corners
/// if `at` is empty. Returns the new profile or `Err(max_feasible)` for the first corner
/// where the treatment does not fit (causal failure report, spec 07).
pub fn treat_corners(p: &Profile, at: &[[f64; 2]], c: Corner) -> Result<Profile, (usize, [f64; 2], f64)> {
    let mut out = Profile { loops: Vec::new() };
    for lp in &p.loops {
        let n = lp.len();
        if n < 3 || lp.iter().any(|s| !matches!(s, Seg2::Line { .. })) { out.loops.push(lp.clone()); continue; }
        let mut segs: Vec<(f64, [f64; 2], [f64; 2])> = Vec::new(); // (unused, start, end)
        let pts: Vec<[f64; 2]> = lp.iter().map(|s| s.start()).collect();
        let mut new: Vec<Seg2> = Vec::new();
        let mut cursor: Option<[f64; 2]> = None;
        let mut first_start: Option<[f64; 2]> = None;
        for k in 0..n {
            let (prev, cur, next) = (pts[(k + n - 1) % n], pts[k], pts[(k + 1) % n]);
            let selected = at.is_empty() || at.iter().any(|q| (q[0] - cur[0]).hypot(q[1] - cur[1]) < 1e-6);
            let d1 = [prev[0] - cur[0], prev[1] - cur[1]];
            let d2 = [next[0] - cur[0], next[1] - cur[1]];
            let (l1, l2) = (d1[0].hypot(d1[1]), d2[0].hypot(d2[1]));
            let (u1, u2) = ([d1[0] / l1, d1[1] / l1], [d2[0] / l2, d2[1] / l2]);
            let cosang = (u1[0] * u2[0] + u1[1] * u2[1]).clamp(-1.0, 1.0);
            let half = cosang.acos() / 2.0;
            let straight = (half - core::f64::consts::FRAC_PI_2).abs() < 1e-9;
            let (a, b) = if selected && !straight {
                let setback = match c { Corner::Fillet(r) => r / half.tan(), Corner::Chamfer(d) => d };
                // half of each adjacent edge is available to this corner
                let avail = 0.5 * l1.min(l2);
                if setback > avail + 1e-12 {
                    let max = match c { Corner::Fillet(_) => avail * half.tan(), Corner::Chamfer(_) => avail };
                    return Err((k, cur, max));
                }
                ([cur[0] + u1[0] * setback, cur[1] + u1[1] * setback], [cur[0] + u2[0] * setback, cur[1] + u2[1] * setback])
            } else { (cur, cur) };
            if let Some(cs) = cursor { if (cs[0] - a[0]).hypot(cs[1] - a[1]) > 1e-12 { new.push(Seg2::Line { a: cs, b: a }); } } else { first_start = Some(a); }
            if a != b {
                match c {
                    Corner::Chamfer(_) => new.push(Seg2::Line { a, b }),
                    Corner::Fillet(r) => {
                        // arc center along the bisector at distance r / sin(half)
                        let bis = [u1[0] + u2[0], u1[1] + u2[1]];
                        let bl = bis[0].hypot(bis[1]);
                        let dist = r / half.sin();
                        let ctr = [cur[0] + bis[0] / bl * dist, cur[1] + bis[1] / bl * dist];
                        let a0 = (a[1] - ctr[1]).atan2(a[0] - ctr[0]);
                        let mut a1 = (b[1] - ctr[1]).atan2(b[0] - ctr[0]);
                        // direction: same turning sense as the loop corner
                        let cross = (cur[0] - prev[0]) * (next[1] - cur[1]) - (cur[1] - prev[1]) * (next[0] - cur[0]);
                        if cross > 0.0 { while a1 < a0 { a1 += core::f64::consts::TAU; } } else { while a1 > a0 { a1 -= core::f64::consts::TAU; } }
                        new.push(Seg2::Arc { c: ctr, r, a0, a1 });
                    }
                }
            }
            cursor = Some(b);
            let _ = &mut segs;
        }
        if let (Some(cs), Some(fs)) = (cursor, first_start) { if (cs[0] - fs[0]).hypot(cs[1] - fs[1]) > 1e-12 { new.push(Seg2::Line { a: cs, b: fs }); } }
        out.loops.push(new);
    }
    Ok(out)
}

#[cfg(test)]
mod corner_tests {
    use super::*;
    #[test]
    fn fillet_square_area_and_infeasible_radius() {
        let sq = Profile::rectangle(0.0, 0.0, 10.0, 10.0).normalize();
        let f = treat_corners(&sq, &[], Corner::Fillet(2.0)).unwrap();
        let expect = 100.0 - 4.0 * (4.0 - core::f64::consts::PI);
        assert!((f.area() - expect).abs() < 1e-9, "{}", f.area());
        let e = treat_corners(&sq, &[], Corner::Fillet(6.0)).unwrap_err();
        assert!((e.2 - 5.0).abs() < 1e-9, "max feasible radius {}", e.2);
        let ch = treat_corners(&sq, &[[10.0, 10.0]], Corner::Chamfer(1.0)).unwrap();
        assert!((ch.area() - 99.5).abs() < 1e-9);
    }
}

// ------------------------------------------------------------------------------------
// Inward offset (erosion by a disk): the exact boundary of {p ∈ R : dist(p, ∂R) ≥ t}
// ------------------------------------------------------------------------------------

fn seg_dir(s: &Seg2, at_end: bool) -> [f64; 2] {
    match *s {
        Seg2::Line { a, b } => { let d = sub(b, a); let l = (d[0] * d[0] + d[1] * d[1]).sqrt(); [d[0] / l, d[1] / l] }
        Seg2::Arc { a0, a1, .. } => {
            let th = if at_end { a1 } else { a0 };
            if a1 > a0 { [-th.sin(), th.cos()] } else { [th.sin(), -th.cos()] }
        }
        Seg2::Circle { .. } => [1.0, 0.0],
    }
}

fn left(d: [f64; 2]) -> [f64; 2] { [-d[1], d[0]] }

/// Offset one closed loop by `t` to its left (the region's interior side for normalized
/// profiles: outer loops CCW, holes CW). Lines shift, arcs change radius, convex corners
/// between lines are trimmed, reflex corners get an arc of radius `t`. `Ok(None)`: the loop
/// vanishes (a circle or arc collapses); `Err`: the thickness exceeds a feature, or a convex
/// line/arc corner (not supported exactly).
pub fn offset_loop(lp: &[Seg2], t: f64) -> Result<Option<Vec<Seg2>>, String> {
    if let [Seg2::Circle { c, r, ccw }] = lp {
        let r2 = if *ccw { r - t } else { r + t };
        return Ok(if r2 > EPS { Some(vec![Seg2::Circle { c: *c, r: r2, ccw: *ccw }]) } else { None });
    }
    let n = lp.len();
    let mut off: Vec<Seg2> = Vec::with_capacity(n);
    for s in lp {
        off.push(match *s {
            Seg2::Line { a, b } => { let nl = left(seg_dir(s, false)); Seg2::Line { a: [a[0] + t * nl[0], a[1] + t * nl[1]], b: [b[0] + t * nl[0], b[1] + t * nl[1]] } }
            Seg2::Arc { c, r, a0, a1 } => {
                let r2 = if a1 > a0 { r - t } else { r + t };
                if r2 <= EPS { return Ok(None); }
                Seg2::Arc { c, r: r2, a0, a1 }
            }
            Seg2::Circle { .. } => return Err("a full circle must be alone in its loop".into()),
        });
    }
    // corners: corner i joins segment i and i + 1
    let mut corner_arc: Vec<Option<Seg2>> = vec![None; n];
    for i in 0..n {
        let j = (i + 1) % n;
        let (d1, d2) = (seg_dir(&lp[i], true), seg_dir(&lp[j], false));
        let cr = d1[0] * d2[1] - d1[1] * d2[0];
        let dt = d1[0] * d2[0] + d1[1] * d2[1];
        if cr.abs() < 1e-9 && dt > 0.0 { continue; } // tangent: offset endpoints coincide
        let p = lp[i].end();
        if cr > 0.0 {
            // convex: trim both offset lines to their intersection
            let (Seg2::Line { a: a1, b: b1 }, Seg2::Line { a: a2, b: b2 }) = (off[i], off[j]) else {
                return Err(format!("convex corner between a line and an arc at ({:.4}, {:.4}): exact offset unsupported", p[0], p[1]));
            };
            let (u, v) = (sub(b1, a1), sub(b2, a2));
            let den = u[0] * v[1] - u[1] * v[0];
            let w = sub(a2, a1);
            let s = (w[0] * v[1] - w[1] * v[0]) / den;
            let x = [a1[0] + s * u[0], a1[1] + s * u[1]];
            off[i] = Seg2::Line { a: a1, b: x };
            off[j] = Seg2::Line { a: x, b: b2 };
        } else {
            // reflex (or a cusp): arc around the corner, turning clockwise
            let (n1, n2) = (left(d1), left(d2));
            let a0 = n1[1].atan2(n1[0]);
            let mut a1 = n2[1].atan2(n2[0]);
            while a1 >= a0 { a1 -= TAU; }
            while a0 - a1 > TAU { a1 += TAU; }
            corner_arc[i] = Some(Seg2::Arc { c: p, r: t, a0, a1 });
        }
    }
    // validity: offset lines keep their direction and a positive length
    for (s, o) in lp.iter().zip(&off) {
        if let (Seg2::Line { a, b }, Seg2::Line { a: a2, b: b2 }) = (s, o) {
            let (u, v) = (sub(*b, *a), sub(*b2, *a2));
            if u[0] * v[0] + u[1] * v[1] <= EPS * EPS { return Err(format!("offset {t} exceeds the feature at ({:.4}, {:.4})", a[0], a[1])); }
        }
    }
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        out.push(off[i]);
        if let Some(c) = corner_arc[i] { out.push(c); }
    }
    if (Profile::loop_area(&out) > 0.0) != (Profile::loop_area(lp) > 0.0) { return Err(format!("offset {t} inverts a loop")); }
    Ok(Some(out))
}

/// Erosion of a region by a disk of radius `t`: inward offset of the outer loop minus the
/// grown holes (exact lines and arcs; overlaps between grown holes and the outer loop are
/// resolved by the exact booleans).
pub fn offset_inward(p: &Profile, t: f64) -> Result<Vec<Profile>, String> {
    let p = p.clone().normalize();
    let Some(outer) = p.loops.first() else { return Ok(vec![]) };
    let Some(o) = offset_loop(outer, t)? else { return Ok(vec![]) };
    let mut acc = vec![Profile { loops: vec![o] }.normalize()];
    for h in &p.loops[1..] {
        let Some(g) = offset_loop(h, t)? else { continue };
        let hole = Profile { loops: vec![g] }.normalize();
        acc = acc.iter().flat_map(|x| boolean(x, &hole, BoolOp2::Difference)).collect();
    }
    Ok(acc)
}
