//! General surface–surface intersection (spec 02 § SSI; the fallback after the analytic
//! table): (1) seeding by recursive parameter-domain subdivision with bounding-box pruning
//! and Newton refinement onto both surfaces, (2) predictor–corrector marching along
//! T = n₁ × n₂ with adaptive step, (3) fitting a NURBS polyline and verifying the error
//! against both surfaces. CPU reference of the GPU seed/march kernels (one warp per branch).

use crate::curve::Curve;
use crate::math::{Aabb, V3};
use crate::nurbs::NurbsCurve;
use crate::surface::Surface;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain { pub u0: f64, pub u1: f64, pub v0: f64, pub v1: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct SsiOptions {
    pub tol: f64,
    pub max_depth: u32,
    pub step: f64,
    pub max_points: usize,
}

impl Default for SsiOptions {
    fn default() -> Self { SsiOptions { tol: 1e-7, max_depth: 7, step: 0.5, max_points: 20_000 } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    pub points: Vec<V3>,
    /// (u1, v1, u2, v2) per point.
    pub params: Vec<[f64; 4]>,
    pub closed: bool,
    /// Max distance of curve samples from either surface.
    pub max_error: f64,
}

fn patch_box(s: &Surface, d: &Domain, n: usize) -> Aabb {
    let mut b = Aabb::EMPTY;
    for i in 0..=n { for j in 0..=n {
        b.add(s.point(d.u0 + (d.u1 - d.u0) * i as f64 / n as f64, d.v0 + (d.v1 - d.v0) * j as f64 / n as f64));
    }}
    // inflate by an estimate of interior bulge (second-derivative bound from the samples)
    let e = b.extent().norm() * 0.15 / (n as f64);
    Aabb { min: b.min - V3 { x: e, y: e, z: e }, max: b.max + V3 { x: e, y: e, z: e } }
}

/// Newton refinement of (u1,v1,u2,v2) onto both surfaces, with the extra condition that the
/// step is minimal-norm (4 unknowns, 3 equations → pseudo-inverse via normal equations).
pub fn refine(a: &Surface, b: &Surface, mut q: [f64; 4], tol: f64) -> Option<[f64; 4]> {
    for _ in 0..30 {
        let ja = a.jet(q[0], q[1]);
        let jb = b.jet(q[2], q[3]);
        let r = ja.p - jb.p;
        if r.norm() < tol { return Some(q); }
        // J = [Su1, Sv1, -Su2, -Sv2] (3×4); solve J δ = -r minimal norm: δ = Jᵀ (J Jᵀ)⁻¹ (-r)
        let cols = [ja.du, ja.dv, -jb.du, -jb.dv];
        let mut m = [[0.0; 3]; 3];
        for c in &cols {
            let v = [c.x, c.y, c.z];
            for i in 0..3 { for j in 0..3 { m[i][j] += v[i] * v[j]; } }
        }
        let rhs = [-r.x, -r.y, -r.z];
        let y = solve3(m, rhs)?;
        let yv = V3 { x: y[0], y: y[1], z: y[2] };
        for k in 0..4 { q[k] += cols[k].dot(yv); }
        if q.iter().any(|x| !x.is_finite()) { return None; }
    }
    let r = a.point(q[0], q[1]) - b.point(q[2], q[3]);
    if r.norm() < tol * 10.0 { Some(q) } else { None }
}

fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-300 { return None; }
    let d = |c: usize| -> f64 {
        let mut k = m;
        for (i, row) in k.iter_mut().enumerate() { row[c] = b[i]; }
        k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1]) - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0]) + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0])
    };
    Some([d(0) / det, d(1) / det, d(2) / det])
}

/// Seeds: subdivide both domains while boxes overlap; refine the centers of overlapping leaf
/// pairs with Newton; de-duplicate by distance.
pub fn seeds(a: &Surface, da: Domain, b: &Surface, db: Domain, o: &SsiOptions) -> Vec<[f64; 4]> {
    let mut out: Vec<[f64; 4]> = Vec::new();
    let mut stack = vec![(da, db, 0u32)];
    while let Some((x, y, depth)) = stack.pop() {
        if !patch_box(a, &x, 3).overlaps(&patch_box(b, &y, 3), o.tol) { continue; }
        if depth >= o.max_depth {
            let q0 = [(x.u0 + x.u1) / 2.0, (x.v0 + x.v1) / 2.0, (y.u0 + y.u1) / 2.0, (y.v0 + y.v1) / 2.0];
            if let Some(q) = refine(a, b, q0, o.tol) {
                let p = a.point(q[0], q[1]);
                if !out.iter().any(|s| a.point(s[0], s[1]).dist(p) < o.step * 0.5) { out.push(q); }
            }
            continue;
        }
        let split = |d: Domain| -> [Domain; 4] {
            let (um, vm) = ((d.u0 + d.u1) / 2.0, (d.v0 + d.v1) / 2.0);
            [Domain { u1: um, v1: vm, ..d }, Domain { u0: um, v1: vm, ..d }, Domain { u1: um, v0: vm, ..d }, Domain { u0: um, v0: vm, ..d }]
        };
        for xs in split(x) { for ys in split(y) { stack.push((xs, ys, depth + 1)); } }
    }
    out
}

fn inside(d: &Domain, u: f64, v: f64) -> bool { u >= d.u0 - 1e-12 && u <= d.u1 + 1e-12 && v >= d.v0 - 1e-12 && v <= d.v1 + 1e-12 }

/// March one branch from a seed in both directions until it leaves a domain or closes.
pub fn march(a: &Surface, da: &Domain, b: &Surface, db: &Domain, seed: [f64; 4], o: &SsiOptions) -> Branch {
    let mut fwd: Vec<[f64; 4]> = vec![seed];
    let mut closed = false;
    let start_p = a.point(seed[0], seed[1]);
    let mut back: Vec<[f64; 4]> = Vec::new();
    for dir in [1.0, -1.0] {
        let mut q = seed;
        let mut h = o.step;
        let pts = if dir > 0.0 { &mut fwd } else { &mut back };
        for _ in 0..o.max_points {
            let ja = a.jet(q[0], q[1]);
            let jb = b.jet(q[2], q[3]);
            let t = ja.normal().cross(jb.normal());
            if t.norm() < 1e-9 { break; } // tangential contact: stop (singular point)
            let t = t.normalized() * dir;
            // predictor in parameter space via least-squares projection of h·t on each tangent plane
            let proj = |j: &crate::nurbs::Jet, v: V3| -> (f64, f64) {
                let (e, f, g) = (j.du.dot(j.du), j.du.dot(j.dv), j.dv.dot(j.dv));
                let (p, r) = (v.dot(j.du), v.dot(j.dv));
                let det = e * g - f * f;
                ((p * g - r * f) / det, (e * r - f * p) / det)
            };
            let (dua, dva) = proj(&ja, t * h);
            let (dub, dvb) = proj(&jb, t * h);
            let pred = [q[0] + dua, q[1] + dva, q[2] + dub, q[3] + dvb];
            match refine(a, b, pred, o.tol) {
                Some(next) => {
                    let p0 = a.point(q[0], q[1]);
                    let p1 = a.point(next[0], next[1]);
                    let chord = p1.dist(p0);
                    if chord > 1.5 * h || chord < 0.2 * h { h *= 0.5; if h < 1e-6 { break; } continue; }
                    if !inside(da, next[0], next[1]) || !inside(db, next[2], next[3]) { break; }
                    pts.push(next);
                    q = next;
                    if dir > 0.0 && pts.len() > 3 && p1.dist(start_p) < h * 0.9 { closed = true; break; }
                    h = (h * 1.2).min(o.step * 2.0);
                }
                None => { h *= 0.5; if h < 1e-6 { break; } }
            }
        }
        if closed { break; }
    }
    let mut params: Vec<[f64; 4]> = back.into_iter().rev().collect();
    params.extend(fwd);
    let points: Vec<V3> = params.iter().map(|q| a.point(q[0], q[1])).collect();
    // verification: midpoints of consecutive samples projected to both surfaces
    let mut max_error: f64 = 0.0;
    for w in params.windows(2) {
        let mid = [(w[0][0] + w[1][0]) / 2.0, (w[0][1] + w[1][1]) / 2.0, (w[0][2] + w[1][2]) / 2.0, (w[0][3] + w[1][3]) / 2.0];
        let pm = (a.point(w[0][0], w[0][1]) + a.point(w[1][0], w[1][1])) * 0.5;
        if let Some(r) = refine(a, b, mid, o.tol) { max_error = max_error.max(a.point(r[0], r[1]).dist(pm)); }
    }
    Branch { points, params, closed, max_error }
}

/// Full SSI: seeds → marched branches (seeds already covered by a branch are skipped) →
/// `Curve::Intersection` with a degree-1 fit and the measured error bound.
pub fn intersect_general(a: &Surface, da: Domain, b: &Surface, db: Domain, o: &SsiOptions, parents: [u32; 2]) -> Vec<Curve> {
    let mut branches: Vec<Branch> = Vec::new();
    for s in seeds(a, da, b, db, o) {
        let p = a.point(s[0], s[1]);
        if branches.iter().any(|br| br.points.iter().any(|q| q.dist(p) < o.step)) { continue; }
        let br = march(a, &da, b, &db, s, o);
        if br.points.len() >= 2 { branches.push(br); }
    }
    branches.into_iter().map(|br| {
        let mut pts = br.points.clone();
        if br.closed { pts.push(pts[0]); }
        let n = pts.len();
        let mut knots = vec![0.0];
        let mut acc = 0.0;
        let mut ts = vec![0.0];
        for w in pts.windows(2) { acc += w[0].dist(w[1]); ts.push(acc); }
        for t in &ts { knots.push(t / acc.max(1e-300)); }
        knots.push(1.0);
        Curve::Intersection { fit: NurbsCurve { degree: 1, knots, weights: vec![1.0; n], poles: pts }, max_error: br.max_error + o.tol, parents }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{v3, Frame};
    #[test]
    fn general_ssi_matches_analytic_plane_sphere() {
        let sphere = Surface::Sphere { frame: Frame::WORLD, r: 5.0 };
        let plane = Surface::Plane { frame: Frame { origin: v3(0.0, 0.0, 2.0), ..Frame::WORLD } };
        let ds = Domain { u0: -3.2, u1: 3.2, v0: -1.6, v1: 1.6 };
        let dp = Domain { u0: -6.0, u1: 6.0, v0: -6.0, v1: 6.0 };
        let cs = intersect_general(&sphere, ds, &plane, dp, &SsiOptions::default(), [0, 1]);
        assert!(!cs.is_empty());
        let r = (25.0f64 - 4.0).sqrt();
        for c in &cs {
            if let Curve::Intersection { fit, max_error, .. } = c {
                for p in &fit.poles { assert!(((p.x * p.x + p.y * p.y).sqrt() - r).abs() < 1e-6 && (p.z - 2.0).abs() < 1e-6, "{p:?}"); }
                assert!(*max_error < 0.05);
            }
        }
    }
}
