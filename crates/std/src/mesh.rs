//! Mesh features: `mesh.tessellate` and `convert.implicit_to_mesh` (marching tetrahedra with
//! six consistent tetrahedra per cube and edge-keyed vertex sharing → watertight, as in the
//! Demo's GPU extractor). CPU reference; the CUDA kernel mirrors it (P11).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::{Mesh, TessOptions};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, ImplicitBody, Product};
use std::collections::HashMap;

const TETS: [[usize; 4]; 6] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];
const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

/// Extract the zero level set of `f` over `body.bounds` with cell size `res` (mm).
pub fn implicit_to_mesh(body: &ImplicitBody, res: f64) -> Result<Mesh, FeatureError> {
    let b = body.bounds;
    let pad = res;
    let (min, max) = (b.min - v3(pad, pad, pad), b.max + v3(pad, pad, pad));
    let dims = [((max.x - min.x) / res).ceil() as usize, ((max.y - min.y) / res).ceil() as usize, ((max.z - min.z) / res).ceil() as usize];
    let cells = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
    if cells > 64_000_000 { return Err(FeatureError::new(AgStatus::CapacityExceeded, format!("{cells} cells exceed the reference extractor limit")).suggest("increase res or narrow bounds")); }
    let (nx, ny, nz) = (dims[0] + 1, dims[1] + 1, dims[2] + 1);
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let pos = |i: usize, j: usize, k: usize| v3(min.x + res * i as f64, min.y + res * j as f64, min.z + res * k as f64);
    let mut field = vec![0.0f64; nx * ny * nz];
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        let v = body.eval(pos(i, j, k));
        // Outside the padded domain boundary force positive so the surface closes.
        let boundary = i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
        field[idx(i, j, k)] = if boundary { v.abs().max(res * 1e-3) } else if v == 0.0 { 1e-12 } else { v };
    }}}
    let mut mesh = Mesh::default();
    let mut edge_vert: HashMap<(usize, usize), u32> = HashMap::new();
    let mut vertex = |mesh: &mut Mesh, a: usize, b: usize, pa: V3, pb: V3, fa: f64, fb: f64| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        *edge_vert.entry(key).or_insert_with(|| {
            // consistent interpolation order (smaller index first) → bit-identical shared vertices
            let (p0, p1, f0, f1) = if a < b { (pa, pb, fa, fb) } else { (pb, pa, fb, fa) };
            let t = f0 / (f0 - f1);
            mesh.positions.push(p0 + (p1 - p0) * t);
            (mesh.positions.len() - 1) as u32
        })
    };
    for k in 0..dims[2] { for j in 0..dims[1] { for i in 0..dims[0] {
        let ci: Vec<usize> = CORNERS.iter().map(|c| idx(i + c[0], j + c[1], k + c[2])).collect();
        let cp: Vec<V3> = CORNERS.iter().map(|c| pos(i + c[0], j + c[1], k + c[2])).collect();
        for t in TETS {
            let g = |q: usize| (ci[t[q]], cp[t[q]], field[ci[t[q]]]);
            let v: Vec<(usize, V3, f64)> = (0..4).map(g).collect();
            let inside: Vec<usize> = (0..4).filter(|&q| v[q].2 < 0.0).collect();
            let outside: Vec<usize> = (0..4).filter(|&q| v[q].2 >= 0.0).collect();
            let mk = |mesh: &mut Mesh, vertex: &mut dyn FnMut(&mut Mesh, usize, usize, V3, V3, f64, f64) -> u32, a: usize, b: usize| vertex(mesh, v[a].0, v[b].0, v[a].1, v[b].1, v[a].2, v[b].2);
            // orientation: outward = toward positive field
            let orient = |mesh: &Mesh, tri: [u32; 3], inner: V3| -> [u32; 3] {
                let (p0, p1, p2) = (mesh.positions[tri[0] as usize], mesh.positions[tri[1] as usize], mesh.positions[tri[2] as usize]);
                if (p1 - p0).cross(p2 - p0).dot(p0 - inner) < 0.0 { [tri[0], tri[2], tri[1]] } else { tri }
            };
            match inside.len() {
                1 => {
                    let a = inside[0];
                    let tri = [mk(&mut mesh, &mut vertex, a, outside[0]), mk(&mut mesh, &mut vertex, a, outside[1]), mk(&mut mesh, &mut vertex, a, outside[2])];
                    let tri = orient(&mesh, tri, v[a].1);
                    if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] { mesh.triangles.push(tri); mesh.face_of.push(0); }
                }
                3 => {
                    let o = outside[0];
                    let tri = [mk(&mut mesh, &mut vertex, inside[0], o), mk(&mut mesh, &mut vertex, inside[1], o), mk(&mut mesh, &mut vertex, inside[2], o)];
                    let inner = (v[inside[0]].1 + v[inside[1]].1 + v[inside[2]].1) * (1.0 / 3.0);
                    let tri = orient(&mesh, tri, inner);
                    if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] { mesh.triangles.push(tri); mesh.face_of.push(0); }
                }
                2 => {
                    let (a, b2) = (inside[0], inside[1]);
                    let (c, d) = (outside[0], outside[1]);
                    let q = [mk(&mut mesh, &mut vertex, a, c), mk(&mut mesh, &mut vertex, a, d), mk(&mut mesh, &mut vertex, b2, d), mk(&mut mesh, &mut vertex, b2, c)];
                    let inner = (v[a].1 + v[b2].1) * 0.5;
                    for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                        let tri = orient(&mesh, tri, inner);
                        if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] { mesh.triangles.push(tri); mesh.face_of.push(0); }
                    }
                }
                _ => {}
            }
        }
    }}}
    Ok(mesh)
}

pub struct MeshOp(pub &'static str);

impl FeatureImpl for MeshOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "mesh.tessellate" => {
                let item = list(req(n, "items")?).first().ok_or_else(|| bad("no items"))?;
                let o = TessOptions { sag: num_or(ctx, n, "sag", 0.01)?, angle: num_or(ctx, n, "angle", 0.2)?, ..Default::default() };
                let mesh = match ctx.product(item)? {
                    Product::Sheet(b) => agcad_geo::tessellate::try_tessellate(b, &o)?,
                    Product::Solid(s) => ctx.kernel.try_tessellate(s, &o)?,
                    _ => return Err(bad("tessellation requires a solid or sheet")),
                };
                Ok(Product::Mesh(mesh))
            }
            "convert.implicit_to_mesh" => {
                let body = match ctx.product(req(n, "body")?)? { Product::Implicit(i) => i.clone(), _ => return Err(bad("expected an implicit body")) };
                let res = num(ctx, n, "res")?;
                Ok(Product::Mesh(implicit_to_mesh(&body, res)?))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}
