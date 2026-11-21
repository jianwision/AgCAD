//! Half-edge B-rep (spec 02 § Exact geometry): Body → Lump → Shell → Face → Loop → Coedge →
//! Edge → Vertex, stored as flat arrays (SoA-friendly, `u32` indices) so the same layout can be
//! mirrored into device arenas. Every face/edge carries a stable provenance name (spec 02 §
//! Stable naming) and edges/vertices carry tolerances (tolerant modeling).

use crate::curve::Curve;
use crate::math::{Aabb, V3, Xform};
use crate::surface::Surface;
use serde::{Deserialize, Serialize};

pub const DEFAULT_LINEAR_TOL: f64 = 1e-6;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub p: V3,
    pub tol: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub curve: u32,
    pub v0: u32,
    pub v1: u32,
    /// Curve parameter range [t0, t1] from v0 to v1.
    pub t0: f64,
    pub t1: f64,
    pub tol: f64,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coedge {
    pub edge: u32,
    /// true if the coedge runs v0 → v1 along the edge.
    pub forward: bool,
    pub lp: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Loop {
    pub face: u32,
    pub coedges: Vec<u32>,
    pub outer: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Face {
    pub surface: u32,
    /// true if the face normal equals the surface normal.
    pub same_sense: bool,
    pub loops: Vec<u32>,
    pub shell: u32,
    pub name: String,
    /// UV bounds of the face on its surface (used for tessellation of analytic faces).
    pub uv: [f64; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shell {
    pub faces: Vec<u32>,
    pub closed: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyKind {
    Solid,
    Sheet,
    Wire,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brep {
    pub kind: BodyKind,
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub coedges: Vec<Coedge>,
    pub loops: Vec<Loop>,
    pub faces: Vec<Face>,
    pub shells: Vec<Shell>,
    pub curves: Vec<Curve>,
    pub surfaces: Vec<Surface>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidityReport {
    pub valid: bool,
    pub euler: i64,
    pub closed: bool,
    pub problems: Vec<String>,
}

impl Brep {
    pub fn new(kind: BodyKind) -> Self {
        Brep { kind, vertices: vec![], edges: vec![], coedges: vec![], loops: vec![], faces: vec![], shells: vec![], curves: vec![], surfaces: vec![] }
    }

    pub fn add_vertex(&mut self, p: V3) -> u32 {
        self.vertices.push(Vertex { p, tol: DEFAULT_LINEAR_TOL });
        (self.vertices.len() - 1) as u32
    }
    pub fn add_curve(&mut self, c: Curve) -> u32 { self.curves.push(c); (self.curves.len() - 1) as u32 }
    pub fn add_surface(&mut self, s: Surface) -> u32 { self.surfaces.push(s); (self.surfaces.len() - 1) as u32 }

    pub fn add_edge(&mut self, curve: u32, v0: u32, v1: u32, t0: f64, t1: f64, name: impl Into<String>) -> u32 {
        self.edges.push(Edge { curve, v0, v1, t0, t1, tol: DEFAULT_LINEAR_TOL, name: name.into() });
        (self.edges.len() - 1) as u32
    }

    /// Add a face with one outer loop and optional inner loops; each loop is a list of
    /// `(edge, forward)` coedges in order.
    pub fn add_face(&mut self, surface: u32, same_sense: bool, shell: u32, outer: &[(u32, bool)], inner: &[Vec<(u32, bool)>], name: impl Into<String>, uv: [f64; 4]) -> u32 {
        let fi = self.faces.len() as u32;
        let mut loops = Vec::new();
        for (k, lp) in std::iter::once(outer.to_vec()).chain(inner.iter().cloned()).enumerate() {
            let li = self.loops.len() as u32;
            let mut ces = Vec::new();
            for (e, f) in lp {
                self.coedges.push(Coedge { edge: e, forward: f, lp: li });
                ces.push((self.coedges.len() - 1) as u32);
            }
            self.loops.push(Loop { face: fi, coedges: ces, outer: k == 0 });
            loops.push(li);
        }
        self.faces.push(Face { surface, same_sense, loops, shell, name: name.into(), uv });
        if let Some(s) = self.shells.get_mut(shell as usize) { s.faces.push(fi); }
        fi
    }

    pub fn add_shell(&mut self, closed: bool) -> u32 { self.shells.push(Shell { faces: vec![], closed }); (self.shells.len() - 1) as u32 }

    /// Re-partition faces into one shell per edge-connected component (deterministic: shells
    /// ordered by their lowest face index). Builders that emit all faces into one shell call
    /// this so disjoint lumps validate as separate shells.
    pub fn split_shells(&mut self) {
        let nf = self.faces.len();
        if nf == 0 { return; }
        let mut parent: Vec<usize> = (0..nf).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
        let mut first_face_of_edge: Vec<Option<usize>> = vec![None; self.edges.len()];
        for (fi, f) in self.faces.iter().enumerate() {
            for &l in &f.loops {
                for &ce in &self.loops[l as usize].coedges {
                    let e = self.coedges[ce as usize].edge as usize;
                    match first_face_of_edge[e] {
                        None => first_face_of_edge[e] = Some(fi),
                        Some(g) => { let (a, b) = (find(&mut parent, fi), find(&mut parent, g)); if a != b { parent[a.max(b)] = a.min(b); } }
                    }
                }
            }
        }
        let closed = self.shells.first().map(|s| s.closed).unwrap_or(true);
        let mut root_shell: std::collections::BTreeMap<usize, u32> = std::collections::BTreeMap::new();
        let mut shells: Vec<Shell> = Vec::new();
        for fi in 0..nf {
            let r = find(&mut parent, fi);
            let si = *root_shell.entry(r).or_insert_with(|| { shells.push(Shell { faces: vec![], closed }); (shells.len() - 1) as u32 });
            shells[si as usize].faces.push(fi as u32);
            self.faces[fi].shell = si;
        }
        self.shells = shells;
    }

    pub fn coedge_start(&self, ce: u32) -> u32 {
        let c = &self.coedges[ce as usize];
        let e = &self.edges[c.edge as usize];
        if c.forward { e.v0 } else { e.v1 }
    }
    pub fn coedge_end(&self, ce: u32) -> u32 {
        let c = &self.coedges[ce as usize];
        let e = &self.edges[c.edge as usize];
        if c.forward { e.v1 } else { e.v0 }
    }

    pub fn bbox(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        for v in &self.vertices { b.add(v.p); }
        // Curved faces can bulge beyond vertices: include tessellation samples of edges.
        for e in &self.edges {
            let c = &self.curves[e.curve as usize];
            for i in 0..=8 { b.add(c.point(e.t0 + (e.t1 - e.t0) * i as f64 / 8.0)); }
        }
        b
    }

    /// Topological validity (spec 02; `evaluate.check` exact layer): loop closure, each edge
    /// used by exactly two coedges with opposite sense in closed shells, vertices on curves,
    /// Euler–Poincaré V - E + F - (L - F) = 2(S - G) with genus inferred.
    pub fn validate(&self) -> ValidityReport {
        let mut problems = Vec::new();
        for (li, lp) in self.loops.iter().enumerate() {
            let n = lp.coedges.len();
            if n == 0 { problems.push(format!("loop {li} is empty")); continue; }
            for k in 0..n {
                let a = self.coedge_end(lp.coedges[k]);
                let b = self.coedge_start(lp.coedges[(k + 1) % n]);
                if a != b { problems.push(format!("loop {li} not closed between coedges {k} and {}", (k + 1) % n)); }
            }
        }
        let mut uses = vec![(0u32, 0u32); self.edges.len()];
        for c in &self.coedges {
            let u = &mut uses[c.edge as usize];
            if c.forward { u.0 += 1 } else { u.1 += 1 }
        }
        let closed = self.kind == BodyKind::Solid;
        if closed {
            for (ei, u) in uses.iter().enumerate() {
                if *u != (1, 1) { problems.push(format!("edge {ei} ({}) used {}+/{}- times (non-manifold or open)", self.edges[ei].name, u.0, u.1)); }
            }
        }
        for (ei, e) in self.edges.iter().enumerate() {
            let c = &self.curves[e.curve as usize];
            let (p0, p1) = (c.point(e.t0), c.point(e.t1));
            let (q0, q1) = (self.vertices[e.v0 as usize].p, self.vertices[e.v1 as usize].p);
            let tol = e.tol.max(self.vertices[e.v0 as usize].tol).max(1e-9) * 10.0;
            if p0.dist(q0) > tol || p1.dist(q1) > tol {
                problems.push(format!("edge {ei} ({}) curve endpoints off vertices by {:.3e}/{:.3e}", e.name, p0.dist(q0), p1.dist(q1)));
            }
        }
        let v = self.vertices.len() as i64;
        let e = self.edges.len() as i64;
        let f = self.faces.len() as i64;
        let l = self.loops.len() as i64;
        let euler = v - e + f - (l - f);
        if closed {
            let s = self.shells.len() as i64;
            // genus g = (2S - euler)/2 must be a non-negative integer
            if (2 * s - euler) < 0 || (2 * s - euler) % 2 != 0 {
                problems.push(format!("Euler characteristic {euler} inconsistent with {s} shell(s)"));
            }
        }
        ValidityReport { valid: problems.is_empty(), euler, closed, problems }
    }

    /// Apply a transform (mirror flips face senses so normals stay outward).
    pub fn transformed(&self, x: &Xform) -> Brep {
        let mut b = self.clone();
        for v in &mut b.vertices { v.p = x.apply(v.p); v.tol *= x.s.abs(); }
        for c in &mut b.curves { *c = c.transformed(x); }
        for s in &mut b.surfaces { *s = s.transformed(x); }
        if x.det() < 0.0 {
            for f in &mut b.faces { f.same_sense = !f.same_sense; }
            for lp in &mut b.loops { lp.coedges.reverse(); }
            for c in &mut b.coedges { c.forward = !c.forward; }
        }
        b
    }

    pub fn face_by_name(&self, name: &str) -> Option<u32> { self.faces.iter().position(|f| f.name == name).map(|i| i as u32) }

    /// Append another B-rep (disjoint lumps), returning index offsets.
    pub fn append(&mut self, o: &Brep) {
        let (vo, eo, co, lo, fo, so, cuo, suo) = (self.vertices.len() as u32, self.edges.len() as u32, self.coedges.len() as u32, self.loops.len() as u32, self.faces.len() as u32, self.shells.len() as u32, self.curves.len() as u32, self.surfaces.len() as u32);
        self.vertices.extend(o.vertices.iter().cloned());
        self.curves.extend(o.curves.iter().cloned());
        self.surfaces.extend(o.surfaces.iter().cloned());
        self.edges.extend(o.edges.iter().map(|e| Edge { curve: e.curve + cuo, v0: e.v0 + vo, v1: e.v1 + vo, ..e.clone() }));
        self.coedges.extend(o.coedges.iter().map(|c| Coedge { edge: c.edge + eo, forward: c.forward, lp: c.lp + lo }));
        self.loops.extend(o.loops.iter().map(|l| Loop { face: l.face + fo, coedges: l.coedges.iter().map(|c| c + co).collect(), outer: l.outer }));
        self.faces.extend(o.faces.iter().map(|f| Face { surface: f.surface + suo, loops: f.loops.iter().map(|l| l + lo).collect(), shell: f.shell + so, ..f.clone() }));
        self.shells.extend(o.shells.iter().map(|s| Shell { faces: s.faces.iter().map(|f| f + fo).collect(), closed: s.closed }));
    }
}
