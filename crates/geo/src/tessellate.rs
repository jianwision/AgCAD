//! Watertight adaptive tessellation (`mesh.tessellate`; spec 02 lightweight representations).
//!
//! Edges are discretized once (chordal sag + angle) and every face reuses its edges' samples,
//! so adjacent faces share vertices exactly (watertight by construction). Faces whose loops
//! form a UV rectangle (prism sides, cylinders, cones, spheres, tori, revolutions) use a
//! structured UV grid; other faces are ear-clipped in UV (holes bridged) and refined by
//! interior centroid insertion until the sag tolerance holds.

use crate::brep::Brep;
use crate::curve::Curve;
use crate::math::V3;
use crate::predicates::orient2d;
use serde::{Deserialize, Serialize};
mod patch;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TessOptions {
    /// Max chordal deviation (mm).
    pub sag: f64,
    /// Max angle between adjacent segment directions (rad).
    pub angle: f64,
    pub max_segments_per_edge: usize,
}

impl Default for TessOptions {
    fn default() -> Self { TessOptions { sag: 0.01, angle: 0.2, max_segments_per_edge: 4096 } }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub positions: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
    /// Source face index per triangle (face-id renders, attribute transfer).
    pub face_of: Vec<u32>,
}

impl Mesh {
    /// Volume, area, centroid via the divergence theorem (closed meshes).
    pub fn mass_properties(&self) -> (f64, f64, V3) {
        let (mut vol, mut area) = (0.0, 0.0);
        let mut c = V3::ZERO;
        for t in &self.triangles {
            let (a, b, d) = (self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
            let v6 = a.dot(b.cross(d));
            vol += v6;
            c = c + (a + b + d) * v6;
            area += 0.5 * (b - a).cross(d - a).norm();
        }
        let vol = vol / 6.0;
        let c = if vol.abs() > 0.0 { c * (1.0 / (24.0 * vol)) } else { V3::ZERO };
        (vol, area, c)
    }

    /// Watertightness: every undirected edge used exactly twice with opposite orientation.
    pub fn is_watertight(&self) -> bool {
        let mut m = std::collections::HashMap::new();
        for t in &self.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *m.entry((a, b)).or_insert(0i32) += 1;
            }
        }
        m.iter().all(|(&(a, b), &n)| n == 1 && m.get(&(b, a)) == Some(&1))
    }

    pub fn append(&mut self, o: &Mesh, face_offset: u32) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&o.positions);
        self.triangles.extend(o.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        self.face_of.extend(o.face_of.iter().map(|f| f + face_offset));
    }
}

/// Number of segments for a curve span honoring sag and angle.
fn segments(c: &Curve, t0: f64, t1: f64, o: &TessOptions) -> usize {
    let n = match c {
        Curve::Line { .. } => 1,
        Curve::Circle { r, .. } => {
            let sweep = (t1 - t0).abs();
            let by_sag = if *r > o.sag { 2.0 * (1.0 - o.sag / r).acos() } else { core::f64::consts::PI };
            let step = by_sag.min(o.angle).max(1e-6);
            (sweep / step).ceil() as usize
        }
        _ => {
            // Curvature-sampled estimate.
            let mut n = 4usize;
            let probes = 32;
            let mut kmax: f64 = 0.0;
            for i in 0..=probes { kmax = kmax.max(c.curvature(t0 + (t1 - t0) * i as f64 / probes as f64)); }
            if kmax > 0.0 && kmax.is_finite() {
                let len = c.length(t0, t1);
                let r = 1.0 / kmax;
                let step = if r > o.sag { (2.0 * (1.0 - o.sag / r).acos()).min(o.angle) * r } else { len / 64.0 };
                n = n.max((len / step).ceil() as usize);
            }
            n
        }
    };
    n.clamp(1, o.max_segments_per_edge)
}

struct Builder<'a> {
    b: &'a Brep,
    mesh: Mesh,
    /// Global vertex index per B-rep vertex.
    vmap: Vec<u32>,
    /// Interior sample vertex indices per edge (excluding end vertices), in edge direction.
    esamples: Vec<Vec<(f64, u32)>>,
}

impl<'a> Builder<'a> {
    fn new(b: &'a Brep, grid: &patch::Grid) -> Self {
        let mut mesh = Mesh::default();
        let vmap: Vec<u32> = b.vertices.iter().map(|v| { mesh.positions.push(v.p); (mesh.positions.len() - 1) as u32 }).collect();
        let mut esamples = Vec::with_capacity(b.edges.len());
        for (e, ts) in b.edges.iter().zip(&grid.stations) {
            let c = &b.curves[e.curve as usize];
            let mut v = Vec::with_capacity(ts.len()-2);
            for &u in &ts[1..ts.len()-1] {
                let t = e.t0 + (e.t1 - e.t0) * u;
                mesh.positions.push(c.point(t));
                v.push((t, (mesh.positions.len() - 1) as u32));
            }
            esamples.push(v);
        }
        Builder { b, mesh, vmap, esamples }
    }

    /// Ordered vertex indices along a coedge, start vertex included, end excluded.
    fn coedge_points(&self, ce: u32) -> Vec<u32> {
        let c = &self.b.coedges[ce as usize];
        let e = &self.b.edges[c.edge as usize];
        let mut pts = vec![self.vmap[e.v0 as usize]];
        pts.extend(self.esamples[c.edge as usize].iter().map(|x| x.1));
        if !c.forward {
            let mut r = vec![self.vmap[e.v1 as usize]];
            r.extend(self.esamples[c.edge as usize].iter().rev().map(|x| x.1));
            return r;
        }
        pts
    }

    fn loop_points(&self, li: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for &ce in &self.b.loops[li as usize].coedges { out.extend(self.coedge_points(ce)); }
        // remove consecutive duplicates (degenerate pole loops)
        out.dedup();
        if out.len() > 1 && out.first() == out.last() { out.pop(); }
        out
    }
}

/// Tessellate a B-rep (closed bodies give watertight meshes).
pub fn tessellate(b: &Brep, o: &TessOptions) -> Mesh {
    let grid = patch::Grid::new(b,o);
    let mut bl = Builder::new(b, &grid);
    for (fi, f) in b.faces.iter().enumerate() {
        if grid.tessellate(fi,&mut bl) { continue; }
        let surf = &b.surfaces[f.surface as usize];
        if let Some(tris) = closed_periodic_grid(&mut bl, f, o) {
            for t in tris {
                let tri = if f.same_sense { t } else { [t[0], t[2], t[1]] };
                bl.mesh.triangles.push(tri);
                bl.mesh.face_of.push(fi as u32);
            }
            continue;
        }
        let loops: Vec<Vec<u32>> = f.loops.iter().map(|&l| bl.loop_points(l)).collect();
        // Map loop points to UV (seam-aware unwrap along each loop).
        let mut uv_loops: Vec<Vec<[f64; 2]>> = Vec::new();
        let period_u = matches!(surf, crate::surface::Surface::Cylinder { .. } | crate::surface::Surface::Cone { .. } | crate::surface::Surface::Sphere { .. } | crate::surface::Surface::Torus { .. } | crate::surface::Surface::Revolution { .. });
        for lp in &loops {
            let mut uv = Vec::with_capacity(lp.len());
            let mut prev: Option<[f64; 2]> = None;
            for &vi in lp {
                let p = bl.mesh.positions[vi as usize];
                let (mut u, v) = surf.invert(p, (0.5 * (f.uv[0] + f.uv[1]), 0.5 * (f.uv[2] + f.uv[3])));
                if period_u {
                    if let Some(pp) = prev {
                        while u - pp[0] > core::f64::consts::PI { u -= core::f64::consts::TAU; }
                        while pp[0] - u > core::f64::consts::PI { u += core::f64::consts::TAU; }
                    } else {
                        while u < f.uv[0] - 1e-9 { u += core::f64::consts::TAU; }
                    }
                }
                let q = [u, v];
                uv.push(q);
                prev = Some(q);
            }
            uv_loops.push(uv);
        }
        // UV metric at the face center so ear selection measures true lengths (u in radians
        // on a cylinder of radius r is r× longer than it looks in parameter space).
        let jc = surf.jet(0.5 * (f.uv[0] + f.uv[1]), 0.5 * (f.uv[2] + f.uv[3]));
        let metric = [jc.du.norm().max(1e-12), jc.dv.norm().max(1e-12)];
        let tris = triangulate_uv(&loops, &uv_loops, metric);
        let flip = !f.same_sense;
        // Orientation: triangles CCW in UV give surface-normal orientation.
        for t in tris {
            let tri = if flip { [t[0], t[2], t[1]] } else { t };
            bl.mesh.triangles.push(tri);
            bl.mesh.face_of.push(fi as u32);
        }
        let _ = o;
    }
    bl.mesh
}

/// Structured grid for faces covering a whole sphere or torus (seam/pole loops). Seam
/// columns/rows reuse the seam edges' sample vertices, so the result stays watertight.
fn closed_periodic_grid(bl: &mut Builder<'_>, f: &crate::brep::Face, o: &TessOptions) -> Option<Vec<[u32; 3]>> {
    use crate::surface::Surface;
    let surf = bl.b.surfaces[f.surface as usize].clone();
    let lp = &bl.b.loops[f.loops[0] as usize];
    let edge_of = |k: usize| bl.b.coedges[lp.coedges[k] as usize].edge as usize;
    match surf {
        Surface::Sphere { r, .. } => {
            let seam = edge_of(0);
            let e = bl.b.edges[seam].clone();
            let rows: Vec<(f64, u32)> = std::iter::once((e.t0, bl.vmap[e.v0 as usize]))
                .chain(bl.esamples[seam].iter().cloned())
                .chain(std::iter::once((e.t1, bl.vmap[e.v1 as usize]))).collect();
            let step = if r > o.sag { (2.0 * (1.0 - o.sag / r).acos()).min(o.angle) } else { 1.0 };
            let nu = ((core::f64::consts::TAU / step).ceil() as usize).max(3);
            // grid[j][i]: row j (latitude), column i (longitude); column 0 == column nu == seam.
            let mut grid = vec![vec![0u32; nu + 1]; rows.len()];
            for (j, &(v, seam_idx)) in rows.iter().enumerate() {
                grid[j][0] = seam_idx;
                grid[j][nu] = seam_idx;
                let pole = j == 0 || j == rows.len() - 1;
                for i in 1..nu {
                    grid[j][i] = if pole { seam_idx } else {
                        bl.mesh.positions.push(surf.point(core::f64::consts::TAU * i as f64 / nu as f64, v));
                        (bl.mesh.positions.len() - 1) as u32
                    };
                }
            }
            Some(grid_tris(&grid))
        }
        Surface::Torus { major, minor, .. } => {
            let (eu, ev) = (edge_of(0), edge_of(1));
            let cols: Vec<u32> = std::iter::once(bl.vmap[bl.b.edges[eu].v0 as usize]).chain(bl.esamples[eu].iter().map(|x| x.1)).collect();
            let rowsv: Vec<u32> = std::iter::once(bl.vmap[bl.b.edges[ev].v0 as usize]).chain(bl.esamples[ev].iter().map(|x| x.1)).collect();
            let (nu, nv) = (cols.len(), rowsv.len());
            let _ = (major, minor);
            let mut grid = vec![vec![0u32; nu + 1]; nv + 1];
            for j in 0..=nv {
                for i in 0..=nu {
                    let (ii, jj) = (i % nu, j % nv);
                    grid[j][i] = if jj == 0 { cols[ii] } else if ii == 0 { rowsv[jj] } else {
                        let u = core::f64::consts::TAU * ii as f64 / nu as f64;
                        let v = core::f64::consts::TAU * jj as f64 / nv as f64;
                        // v measured from the outer equator (seam_v starts at angle 0)
                        bl.mesh.positions.push(surf.point(u, v));
                        (bl.mesh.positions.len() - 1) as u32
                    };
                }
            }
            // Wrap: last row/col reuse the first.
            for i in 0..=nu { grid[nv][i] = grid[0][i % nu]; }
            for j in 0..=nv { grid[j][nu] = grid[j % nv][0]; }
            Some(grid_tris(&grid))
        }
        _ => None,
    }
}

fn grid_tris(g: &[Vec<u32>]) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for j in 0..g.len() - 1 {
        for i in 0..g[j].len() - 1 {
            let (a, b, c, d) = (g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]);
            if a != b && b != c && a != c { out.push([a, b, c]); }
            if a != c && c != d && a != d { out.push([a, c, d]); }
        }
    }
    out
}

/// Ear clipping in UV with hole bridging; returns triangles over global vertex indices.
fn triangulate_uv(loops: &[Vec<u32>], uv: &[Vec<[f64; 2]>], metric: [f64; 2]) -> Vec<[u32; 3]> {
    if loops.is_empty() || loops[0].len() < 3 && loops.len() == 1 { return vec![]; }
    // Seam faces: if the outer loop revisits vertices (seam edge used twice), split duplicates
    // by treating each occurrence as a distinct polygon vertex (indices stay global).
    let mut poly: Vec<(u32, [f64; 2])> = loops[0].iter().cloned().zip(uv[0].iter().cloned()).collect();
    if signed_area(&poly) < 0.0 { poly.reverse(); }
    // Bridge holes: rightmost hole vertex to nearest visible outer vertex.
    let mut holes: Vec<Vec<(u32, [f64; 2])>> = loops[1..].iter().zip(&uv[1..]).map(|(l, u)| {
        let mut h: Vec<(u32, [f64; 2])> = l.iter().cloned().zip(u.iter().cloned()).collect();
        if signed_area(&h) > 0.0 { h.reverse(); }
        h
    }).collect();
    holes.sort_by(|a, b| max_u(b).partial_cmp(&max_u(a)).unwrap());
    for k in 0..holes.len() {
        let h = holes[k].clone();
        if h.len() < 3 { continue; }
        let (hi, hp) = h.iter().enumerate().max_by(|a, b| a.1 .1[0].partial_cmp(&b.1 .1[0]).unwrap()).map(|(i, p)| (i, p.1)).unwrap();
        // Bridge to the nearest outer vertex whose segment crosses no boundary edge of the
        // current polygon, this hole or the holes still to be merged (visibility).
        let crosses = |a: [f64; 2], b: [f64; 2]| -> bool {
            let others = holes[k..].iter().flat_map(|l| loop_edges(l));
            loop_edges(&poly).chain(others).any(|(p, q)| {
                if p == a || p == b || q == a || q == b { return false; }
                let (o1, o2) = (orient2d(a, b, p), orient2d(a, b, q));
                let (o3, o4) = (orient2d(p, q, a), orient2d(p, q, b));
                o1 * o2 < 0 && o3 * o4 < 0
            })
        };
        // Vertices already used as a bridge end appear twice; bridging to them again makes the
        // merged polygon ambiguous at that vertex, so they are taken only as a last resort.
        let dup = |i: usize| poly.iter().filter(|q| q.1 == poly[i].1).count() > 1;
        let mut cand: Vec<(bool, f64, usize)> = poly.iter().enumerate().map(|(i, p)| (dup(i), dist2(p.1, hp), i)).collect();
        cand.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.partial_cmp(&y.1).unwrap()).then(x.2.cmp(&y.2)));
        let mut oi = cand.iter().find(|&&(_, _, i)| !crosses(hp, poly[i].1)).or(cand.first()).map(|&(_, _, i)| i).unwrap();
        // Among copies of that position (earlier bridges), attach to the copy whose interior
        // cone contains the hole point, otherwise the merged polygon overlaps itself.
        let np = poly.len();
        if let Some(j) = (0..np).filter(|&j| poly[j].1 == poly[oi].1).find(|&j| in_cone(poly[(j + np - 1) % np].1, poly[j].1, poly[(j + 1) % np].1, hp)) { oi = j; }
        let mut merged = Vec::with_capacity(poly.len() + h.len() + 2);
        merged.extend_from_slice(&poly[..=oi]);
        for k in 0..=h.len() { merged.push(h[(hi + k) % h.len()]); }
        merged.extend_from_slice(&poly[oi..]);
        poly = merged;
    }
    let mut out = Vec::new();
    // Ear clipping on a linked ring with a min-heap of ears keyed by metric diagonal length
    // (shortest first keeps triangles local, so bands on curved surfaces become strips, not
    // fans of chords). Clipping re-scores only the two neighbours; ears rejected because of
    // other vertices are rescanned once the heap runs dry. Ties break by vertex index.
    let m = poly.len();
    let mut prev: Vec<usize> = (0..m).map(|i| (i + m - 1) % m).collect();
    let mut next: Vec<usize> = (0..m).map(|i| (i + 1) % m).collect();
    let mut alive = vec![true; m];
    let mut ver = vec![0u32; m];
    let mut remaining = m;
    let mut head = 0usize;
    let dlen = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]) * metric[0]).powi(2) + ((p[1] - q[1]) * metric[1]).powi(2);
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize, u32)>> = std::collections::BinaryHeap::new();
    let push = |heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize, u32)>>, prev: &[usize], next: &[usize], ver: &[u32], i: usize| {
        let (a, c) = (prev[i], next[i]);
        if orient2d(poly[a].1, poly[i].1, poly[c].1) > 0 {
            heap.push(std::cmp::Reverse((dlen(poly[a].1, poly[c].1).to_bits(), i, ver[i])));
        }
    };
    for i in 0..m { push(&mut heap, &prev, &next, &ver, i); }
    let mut rescanned = false;
    while remaining > 3 {
        let Some(std::cmp::Reverse((_, b, v))) = heap.pop() else {
            if rescanned {
                // Degenerate remainder (collinear/seam): fan it to stay closed.
                let start = head;
                let mut k = next[start];
                while next[k] != start {
                    let (x, y, z) = (poly[start].0, poly[k].0, poly[next[k]].0);
                    if x != y && y != z && x != z { out.push([x, y, z]); }
                    k = next[k];
                }
                remaining = 0;
                break;
            }
            rescanned = true;
            let mut k = head;
            loop { push(&mut heap, &prev, &next, &ver, k); k = next[k]; if k == head { break; } }
            continue;
        };
        if !alive[b] || v != ver[b] { continue; }
        let (a, c) = (prev[b], next[b]);
        let (pa, pb, pc) = (poly[a].1, poly[b].1, poly[c].1);
        let (lo, hi) = ([pa[0].min(pb[0]).min(pc[0]), pa[1].min(pb[1]).min(pc[1])], [pa[0].max(pb[0]).max(pc[0]), pa[1].max(pb[1]).max(pc[1])]);
        let in_box = |p: [f64; 2]| p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1];
        let mut inside = false;
        let mut k = next[c];
        while k != a {
            let p = poly[k].1;
            if in_box(p) && p != pa && p != pb && p != pc && orient2d(pa, pb, p) >= 0 && orient2d(pb, pc, p) >= 0 && orient2d(pc, pa, p) >= 0 { inside = true; break; }
            k = next[k];
        }
        if inside { continue; }
        // The diagonal a–c must start into the polygon interior at both ends (matters at
        // duplicated bridge vertices) and must not cross any remaining polygon edge.
        if remaining > 4 && (!in_cone(poly[prev[a]].1, pa, pb, pc) || !in_cone(pb, pc, poly[next[c]].1, pa)) { continue; }
        let mut crosses = false;
        let mut j = c;
        loop {
            let (p, q) = (poly[j].1, poly[next[j]].1);
            // segment bbox vs diagonal bbox (diagonal a–c lies within the ear's box)
            let disjoint = p[0].max(q[0]) < lo[0] || p[0].min(q[0]) > hi[0] || p[1].max(q[1]) < lo[1] || p[1].min(q[1]) > hi[1];
            if !disjoint && !(p == pa || p == pc || q == pa || q == pc) && orient2d(pa, pc, p) * orient2d(pa, pc, q) < 0 && orient2d(p, q, pa) * orient2d(p, q, pc) < 0 { crosses = true; break; }
            j = next[j];
            if j == c { break; }
        }
        if crosses { continue; }
        if poly[a].0 != poly[b].0 && poly[b].0 != poly[c].0 && poly[a].0 != poly[c].0 {
            out.push([poly[a].0, poly[b].0, poly[c].0]);
        }
        alive[b] = false;
        next[a] = c;
        prev[c] = a;
        if head == b { head = c; }
        remaining -= 1;
        ver[a] += 1;
        ver[c] += 1;
        push(&mut heap, &prev, &next, &ver, a);
        push(&mut heap, &prev, &next, &ver, c);
        rescanned = false;
    }
    if remaining == 3 {
        let (x, y, z) = (poly[head].0, poly[next[head]].0, poly[next[next[head]]].0);
        if x != y && y != z && x != z { out.push([x, y, z]); }
    }
    out
}

/// Is `x` inside the interior cone at vertex `v` (prev `vp`, next `vn`) of a CCW polygon?
/// (O'Rourke, *Computational Geometry in C*, `InCone`.)
fn in_cone(vp: [f64; 2], v: [f64; 2], vn: [f64; 2], x: [f64; 2]) -> bool {
    if orient2d(vp, v, vn) >= 0 {
        orient2d(v, x, vp) > 0 && orient2d(x, v, vn) > 0
    } else {
        !(orient2d(v, x, vn) >= 0 && orient2d(x, v, vp) >= 0)
    }
}

fn loop_edges(l: &[(u32, [f64; 2])]) -> impl Iterator<Item = ([f64; 2], [f64; 2])> + '_ {
    (0..l.len()).map(move |i| (l[i].1, l[(i + 1) % l.len()].1))
}

fn signed_area(p: &[(u32, [f64; 2])]) -> f64 {
    let n = p.len();
    (0..n).map(|i| { let (a, b) = (p[i].1, p[(i + 1) % n].1); a[0] * b[1] - b[0] * a[1] }).sum::<f64>() * 0.5
}
fn max_u(p: &[(u32, [f64; 2])]) -> f64 { p.iter().map(|x| x.1[0]).fold(f64::NEG_INFINITY, f64::max) }
fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 { (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::*;
    use crate::math::{v3, Frame};

    #[test]
    fn box_mesh_is_watertight_with_exact_volume() {
        let b = block(V3::ZERO, v3(2.0, 3.0, 4.0));
        let m = tessellate(&b, &TessOptions::default());
        assert!(m.is_watertight());
        let (vol, area, c) = m.mass_properties();
        assert!((vol - 24.0).abs() < 1e-9, "vol {vol}");
        assert!((area - 52.0).abs() < 1e-9);
        assert!(c.dist(v3(1.0, 1.5, 2.0)) < 1e-9);
    }

    #[test]
    fn plate_with_hole_volume_converges() {
        let p = Profile::rectangle(-50.0, -50.0, 50.0, 50.0).with_hole(Profile::circle(0.0, 0.0, 10.0));
        let b = prism(&p, &Frame::WORLD, 10.0);
        let m = tessellate(&b, &TessOptions { sag: 1e-4, ..Default::default() });
        assert!(m.is_watertight());
        let exact = (100.0 * 100.0 - core::f64::consts::PI * 100.0) * 10.0;
        let (vol, _, _) = m.mass_properties();
        assert!((vol - exact).abs() / exact < 1e-5, "{vol} vs {exact}");
    }
}
