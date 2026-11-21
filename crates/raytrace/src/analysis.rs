//! Design-for-manufacture analyses by ray casting a body's tessellation (outward-oriented,
//! watertight). One sample per triangle at its centroid, weighted by area. Evidence `sampled`.

use crate::{Ray, Tracer};
use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::Mesh;

/// Per-triangle surface samples: centroid, unit outward normal, area.
pub struct Samples { pub c: Vec<V3>, pub n: Vec<V3>, pub a: Vec<f64>, pub eps: f64, pub area: f64 }

pub fn samples(m: &Mesh) -> Samples {
    let (mut c, mut n, mut a) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lo, mut hi) = (v3(f64::INFINITY, f64::INFINITY, f64::INFINITY), v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY));
    for p in &m.positions { lo = v3(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)); hi = v3(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)); }
    for t in &m.triangles {
        let (p0, p1, p2) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
        let cr = (p1 - p0).cross(p2 - p0);
        let ar = 0.5 * cr.norm();
        if ar <= 0.0 { continue; }
        c.push((p0 + p1 + p2) * (1.0 / 3.0));
        n.push(cr * (1.0 / cr.norm()));
        a.push(ar);
    }
    let area = a.iter().sum();
    Samples { c, n, a, eps: 1e-5 * (hi - lo).norm().max(1.0), area }
}

fn ray(o: V3, d: V3) -> Ray { Ray { origin: [o.x as f32, o.y as f32, o.z as f32], tmin: 0.0, dir: [d.x as f32, d.y as f32, d.z as f32], tmax: 1e30 } }

/// Tool access: a surface sample is reachable from direction `d` (tool axis pointing from the
/// part toward the spindle) if it faces it and the ray from just above the surface along `d`
/// escapes. Returns per-direction reachable area, and the area reachable from none.
#[derive(Debug, Clone)]
pub struct Access { pub dirs: Vec<[f64; 3]>, pub reachable: Vec<f64>, pub unreachable: f64, pub area: f64 }

pub fn access(t: &dyn Tracer, m: &Mesh, dirs: &[V3]) -> Result<Access, String> {
    let s = samples(m);
    let mut rays = Vec::new();
    let mut who = Vec::new();
    for (i, (c, n)) in s.c.iter().zip(&s.n).enumerate() {
        for (k, d) in dirs.iter().enumerate() {
            // facing or perpendicular (a vertical wall is reached by the tool side)
            if n.dot(*d) > -1e-6 { rays.push(ray(*c + *n * s.eps, *d)); who.push((i, k)); }
        }
    }
    let hits = t.trace(&rays)?;
    let mut ok = vec![vec![false; dirs.len()]; s.c.len()];
    for (h, (i, k)) in hits.iter().zip(&who) { if h.t < 0.0 { ok[*i][*k] = true; } }
    let mut reachable = vec![0.0; dirs.len()];
    let mut unreachable = 0.0;
    for (i, row) in ok.iter().enumerate() {
        for (k, r) in row.iter().enumerate() { if *r { reachable[k] += s.a[i]; } }
        if !row.iter().any(|r| *r) { unreachable += s.a[i]; }
    }
    Ok(Access { dirs: dirs.iter().map(|d| [d.x, d.y, d.z]).collect(), reachable, unreachable, area: s.area })
}

/// Standard direction sets: "3ax" = +z; "3+2" = ±x, ±y, ±z; "5ax" = 26 directions (cube
/// faces, edges, corners).
pub fn directions(set: &str) -> Option<Vec<V3>> {
    Some(match set {
        "3ax" => vec![V3::Z],
        "3+2" => vec![V3::X, -V3::X, V3::Y, -V3::Y, V3::Z, -V3::Z],
        "5ax" => {
            let mut v = Vec::new();
            for i in -1..=1 { for j in -1..=1 { for k in -1..=1 { if (i, j, k) != (0, 0, 0) { v.push(v3(i as f64, j as f64, k as f64).normalized()); } } } }
            v
        }
        _ => return None,
    })
}

/// Distance along the normal to the next surface: inward (wall thickness) or outward
/// (clearance). Per-sample distances (∞ when nothing is hit) with areas.
pub struct Distances { pub d: Vec<f64>, pub a: Vec<f64> }

impl Distances {
    pub fn min(&self) -> f64 { self.d.iter().copied().fold(f64::INFINITY, f64::min) }
    /// Area of samples closer than `limit`.
    pub fn area_below(&self, limit: f64) -> f64 { self.d.iter().zip(&self.a).filter(|(d, _)| **d < limit).map(|(_, a)| a).sum() }
}

pub fn along_normal(t: &dyn Tracer, m: &Mesh, inward: bool) -> Result<Distances, String> {
    let s = samples(m);
    let sg = if inward { -1.0 } else { 1.0 };
    let rays: Vec<Ray> = s.c.iter().zip(&s.n).map(|(c, n)| ray(*c + *n * (sg * s.eps), *n * sg)).collect();
    let hits = t.trace(&rays)?;
    Ok(Distances { d: hits.iter().map(|h| if h.t < 0.0 { f64::INFINITY } else { h.t as f64 + s.eps }).collect(), a: s.a })
}

/// Ray-cast shaded render (iso view): Lambert key light with hard shadows plus ambient.
pub fn render(t: &dyn Tracer, m: &Mesh, w: usize, h: usize) -> Result<Vec<u8>, String> {
    let (mut lo, mut hi) = (v3(f64::INFINITY, f64::INFINITY, f64::INFINITY), v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY));
    for p in &m.positions { lo = v3(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)); hi = v3(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)); }
    let (ctr, rad) = ((lo + hi) * 0.5, 0.5 * (hi - lo).norm().max(1e-6));
    let view = v3(1.0, -1.0, 1.0).normalized();
    let right = V3::Z.cross(view).normalized();
    let up = view.cross(right);
    let eye = ctr + view * (4.0 * rad);
    let mut rays = Vec::with_capacity(w * h);
    for j in 0..h { for i in 0..w {
        let (x, y) = ((i as f64 + 0.5) / w as f64 * 2.0 - 1.0, 1.0 - (j as f64 + 0.5) / h as f64 * 2.0);
        // orthographic, framing the bounding sphere
        rays.push(ray(eye + right * (x * 1.1 * rad) + up * (y * 1.1 * rad), -view));
    }}
    let hits = t.trace(&rays)?;
    let light = v3(0.4, -0.3, 1.0).normalized();
    let mut shade = vec![0.0f64; w * h];
    let mut shadow_rays = Vec::new();
    let mut lit_idx = Vec::new();
    for (k, hit) in hits.iter().enumerate() {
        if hit.t < 0.0 { continue; }
        let tri = m.triangles[hit.prim as usize];
        let (p0, p1, p2) = (m.positions[tri[0] as usize], m.positions[tri[1] as usize], m.positions[tri[2] as usize]);
        let n = (p1 - p0).cross(p2 - p0).normalized();
        let p = p0 + (p1 - p0) * hit.u as f64 + (p2 - p0) * hit.v as f64;
        let ndl = n.dot(light);
        shade[k] = 0.25 + 0.15 * n.dot(view).max(0.0);
        if ndl > 0.0 { shadow_rays.push(ray(p + n * (1e-4 * rad), light)); lit_idx.push((k, ndl)); }
    }
    let sh = t.trace(&shadow_rays)?;
    for ((k, ndl), s) in lit_idx.iter().zip(&sh) { if s.t < 0.0 { shade[*k] += 0.6 * ndl; } }
    let mut rgb = vec![0u8; w * h * 3];
    for (k, hit) in hits.iter().enumerate() {
        let (r, g, b) = if hit.t < 0.0 { (245.0, 246.0, 248.0) } else { let s = shade[k].min(1.0); (90.0 + 150.0 * s, 110.0 + 135.0 * s, 140.0 + 110.0 * s) };
        rgb[3 * k] = r as u8;
        rgb[3 * k + 1] = g as u8;
        rgb[3 * k + 2] = b as u8;
    }
    Ok(agcad_harness::render::png(&rgb, w, h))
}
