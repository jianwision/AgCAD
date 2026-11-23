//! Tessellate untrimmed four-sided NURBS patches on a shared parameter grid.
use super::{Builder, TessOptions, segments};
use crate::brep::{Brep, DEFAULT_LINEAR_TOL, Face};
use crate::nurbs::NurbsSurface;
use crate::surface::Surface;

// Edges are ordered bottom, right, top, left. The flag follows increasing U or V.
type Sides = [(usize, bool); 4];

fn sides(b: &Brep, f: &Face, s: &NurbsSurface) -> Option<Sides> {
    let [lp] = f.loops.as_slice() else {
        return None;
    };
    let coedges = &b.loops[*lp as usize].coedges;
    if coedges.len() != 4 {
        return None;
    }
    let ((u0, u1), (v0, v1)) = s.domain();
    let mut result = [(usize::MAX, false); 4];
    for &co in coedges {
        let id = b.coedges[co as usize].edge as usize;
        let edge = &b.edges[id];
        let curve = &b.curves[edge.curve as usize];
        let (side, forward) = (0..4)
            .filter(|&i| result[i].0 == usize::MAX)
            .flat_map(|i| [true, false].map(|fwd| (i, fwd)))
            .find(|&(i, fwd)| {
                [0., 1., 0.5, 0.25, 0.75].iter().all(|&t| {
                    let u = u0 + (u1 - u0) * t;
                    let v = v0 + (v1 - v0) * t;
                    let uv = [(u, v0), (u1, v), (u, v1), (u0, v)][i];
                    let et = if fwd { t } else { 1. - t };
                    s.point(uv.0, uv.1)
                        .dist(curve.point(edge.t0 + (edge.t1 - edge.t0) * et))
                        <= edge.tol.max(DEFAULT_LINEAR_TOL)
                })
            })?;
        result[side] = (id, forward);
    }
    Some(result)
}

pub(super) struct Grid {
    faces: Vec<Option<Sides>>,
    pub stations: Vec<Vec<f64>>,
}

impl Grid {
    pub fn new(b: &Brep, o: &TessOptions) -> Self {
        let mut counts: Vec<_> = b
            .edges
            .iter()
            .map(|e| segments(&b.curves[e.curve as usize], e.t0, e.t1, o))
            .collect();
        let mut breaks = vec![Vec::new(); counts.len()];
        let mut linked = vec![Vec::new(); counts.len()];
        let faces = b
            .faces
            .iter()
            .map(|f| {
                let Surface::Nurbs(s) = &b.surfaces[f.surface as usize] else {
                    return None;
                };
                let sides = sides(b, f, s)?;
                let ((u0, u1), (v0, v1)) = s.domain();
                let (du, dv) = (u1 - u0, v1 - v0);
                let mut curvature = [0.0_f64; 2];
                for us in s.ku.windows(2).filter(|w| w[1] > w[0]) {
                    for vs in s.kv.windows(2).filter(|w| w[1] > w[0]) {
                        for a in [0.01, 0.5, 0.99] {
                            for e in [0.01, 0.5, 0.99] {
                                let jet =
                                    s.jet(us[0] + (us[1] - us[0]) * a, vs[0] + (vs[1] - vs[0]) * e);
                                let normal = jet.normal();
                                let twist = jet.duv.dot(normal).abs() * du * dv;
                                curvature[0] = curvature[0].max(jet.duu.norm() * du * du + twist);
                                curvature[1] = curvature[1].max(jet.dvv.norm() * dv * dv + twist);
                            }
                        }
                    }
                }
                for (axis, knots, lo, delta) in [(0, &s.ku, u0, du), (1, &s.kv, v0, dv)] {
                    let (a, z) = (sides[axis].0, sides[axis + 2].0);
                    linked[a].push(z);
                    linked[z].push(a);
                    // Include the cross derivative to resolve twisted bilinear faces.
                    let n = (curvature[axis] / (2. * o.sag.max(1e-12))).sqrt().ceil() as usize;
                    counts[a] = counts[a].max(n.min(o.max_segments_per_edge));
                    for &k in knots.iter().filter(|&&k| k > lo && k < lo + delta) {
                        let t = (k - lo) / delta;
                        breaks[a].extend([t, 1. - t]);
                    }
                }
                Some(sides)
            })
            .collect();
        let mut stations = vec![Vec::new(); counts.len()];
        for seed in 0..counts.len() {
            if !stations[seed].is_empty() {
                continue;
            }
            let mut group = vec![seed];
            stations[seed].push(0.);
            let mut i = 0;
            while i < group.len() {
                for &e in &linked[group[i]] {
                    if stations[e].is_empty() {
                        stations[e].push(0.);
                        group.push(e);
                    }
                }
                i += 1;
            }
            let count = group.iter().map(|&e| counts[e]).max().unwrap();
            let mut ts: Vec<_> = (0..=count).map(|i| i as f64 / count as f64).collect();
            for &e in &group {
                ts.extend_from_slice(&breaks[e]);
            }
            ts.sort_by(f64::total_cmp);
            ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            for e in group {
                stations[e].clone_from(&ts);
            }
        }
        Self { faces, stations }
    }

    pub fn tessellate(&self, face: usize, bl: &mut Builder<'_>) -> bool {
        let Some(sides) = self.faces[face] else {
            return false;
        };
        let f = &bl.b.faces[face];
        let Surface::Nurbs(s) = &bl.b.surfaces[f.surface as usize] else {
            return false;
        };
        let edges = sides.map(|(e, forward)| {
            let edge = &bl.b.edges[e];
            let mut ids = vec![bl.vmap[edge.v0 as usize]];
            ids.extend(bl.esamples[e].iter().map(|&(_, id)| id));
            ids.push(bl.vmap[edge.v1 as usize]);
            if !forward {
                ids.reverse();
            }
            ids
        });
        let us = &self.stations[sides[0].0];
        let vs = &self.stations[sides[1].0];
        let ((u0, u1), (v0, v1)) = s.domain();
        let mut prev = edges[0].clone();
        for (j, &v) in vs.iter().enumerate().skip(1) {
            let mut row = vec![edges[3][j]];
            for (i, &u) in us.iter().enumerate().skip(1).take(us.len() - 2) {
                let id = if j + 1 == vs.len() {
                    edges[2][i]
                } else {
                    let id = bl.mesh.positions.len() as u32;
                    bl.mesh
                        .positions
                        .push(s.point(u0 + (u1 - u0) * u, v0 + (v1 - v0) * v));
                    id
                };
                row.push(id);
            }
            row.push(edges[1][j]);
            for i in 1..row.len() {
                for mut t in [
                    [prev[i - 1], prev[i], row[i]],
                    [prev[i - 1], row[i], row[i - 1]],
                ] {
                    if !f.same_sense {
                        t.swap(1, 2);
                    }
                    bl.mesh.triangles.push(t);
                    bl.mesh.face_of.push(face as u32);
                }
            }
            prev = row;
        }
        true
    }
}
