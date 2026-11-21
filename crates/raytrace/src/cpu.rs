//! Exact CPU reference tracer: f64 Möller–Trumbore over a median-split BVH.

use crate::{Hit, Ray, Tracer};
use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::Mesh;

struct Node { lo: V3, hi: V3, left: u32, right: u32, first: u32, count: u32 }

pub struct CpuTracer { tri: Vec<[V3; 3]>, order: Vec<u32>, nodes: Vec<Node> }

fn vmin(a: V3, b: V3) -> V3 { v3(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)) }
fn vmax(a: V3, b: V3) -> V3 { v3(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)) }
fn axis(p: V3, k: usize) -> f64 { [p.x, p.y, p.z][k] }

impl CpuTracer {
    pub fn new(mesh: &Mesh) -> CpuTracer {
        let tri: Vec<[V3; 3]> = mesh.triangles.iter().map(|t| [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]]).collect();
        let mut t = CpuTracer { order: (0..tri.len() as u32).collect(), tri, nodes: Vec::new() };
        if !t.tri.is_empty() { let n = t.tri.len(); t.build(0, n); }
        t
    }

    fn build(&mut self, first: usize, count: usize) -> u32 {
        let (mut lo, mut hi) = (v3(f64::INFINITY, f64::INFINITY, f64::INFINITY), v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY));
        for &i in &self.order[first..first + count] { for p in self.tri[i as usize] { lo = vmin(lo, p); hi = vmax(hi, p); } }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { lo, hi, left: 0, right: 0, first: first as u32, count: count as u32 });
        if count <= 4 { return id; }
        let e = hi - lo;
        let k = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
        let tri = &self.tri;
        self.order[first..first + count].sort_by(|&a, &b| {
            let ca = axis(tri[a as usize][0] + tri[a as usize][1] + tri[a as usize][2], k);
            let cb = axis(tri[b as usize][0] + tri[b as usize][1] + tri[b as usize][2], k);
            ca.partial_cmp(&cb).unwrap().then(a.cmp(&b))
        });
        let half = count / 2;
        let l = self.build(first, half);
        let r = self.build(first + half, count - half);
        let n = &mut self.nodes[id as usize];
        n.left = l;
        n.right = r;
        n.count = 0;
        id
    }

    fn slab(n: &Node, o: V3, inv: V3, tmax: f64) -> bool {
        let (mut t0, mut t1) = (0.0f64, tmax);
        for k in 0..3 {
            let (a, b) = ((axis(n.lo, k) - axis(o, k)) * axis(inv, k), (axis(n.hi, k) - axis(o, k)) * axis(inv, k));
            let (a, b) = if a <= b { (a, b) } else { (b, a) };
            t0 = t0.max(a);
            t1 = t1.min(b);
            if t0 > t1 { return false; }
        }
        true
    }

    fn one(&self, r: &Ray) -> Hit {
        let o = v3(r.origin[0] as f64, r.origin[1] as f64, r.origin[2] as f64);
        let d = v3(r.dir[0] as f64, r.dir[1] as f64, r.dir[2] as f64);
        let inv = v3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best = Hit { t: -1.0, prim: u32::MAX, u: 0.0, v: 0.0, front: 0 };
        let mut tbest = r.tmax as f64;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !Self::slab(n, o, inv, tbest) { continue; }
            if n.count == 0 { stack.push(n.left); stack.push(n.right); continue; }
            for &ti in &self.order[n.first as usize..(n.first + n.count) as usize] {
                let [a, b, c] = self.tri[ti as usize];
                let (e1, e2) = (b - a, c - a);
                let p = d.cross(e2);
                let det = e1.dot(p);
                if det.abs() < 1e-18 { continue; }
                let s = o - a;
                let u = s.dot(p) / det;
                if !(0.0..=1.0).contains(&u) { continue; }
                let q = s.cross(e1);
                let v = d.dot(q) / det;
                if v < 0.0 || u + v > 1.0 { continue; }
                let t = e2.dot(q) / det;
                if t >= r.tmin as f64 && t < tbest {
                    tbest = t;
                    best = Hit { t: t as f32, prim: ti, u: u as f32, v: v as f32, front: (det > 0.0) as u32 };
                }
            }
        }
        best
    }
}

impl Tracer for CpuTracer {
    fn name(&self) -> &'static str { "cpu-reference" }
    fn trace(&self, rays: &[Ray]) -> Result<Vec<Hit>, String> {
        if self.tri.is_empty() { return Ok(rays.iter().map(|_| Hit { t: -1.0, prim: u32::MAX, u: 0.0, v: 0.0, front: 0 }).collect()); }
        // parallel over chunks (deterministic: each ray independent)
        let mut out = vec![Hit::default(); rays.len()];
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let per = rays.len().div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (rs, os) in rays.chunks(per).zip(out.chunks_mut(per)) {
                sc.spawn(move || for (r, o) in rs.iter().zip(os.iter_mut()) { *o = self.one(r); });
            }
        });
        Ok(out)
    }
}
