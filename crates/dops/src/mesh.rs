//! Certified implicit deliverable (spec 14, evidence `implicit-certified`): a watertight mesh
//! of a device design for designs without an exact B-rep path (e.g. `part.shell`).
//!
//! The SDF is sampled on the device at the nodes of a uniform grid (`ag_mesh_sample`, CUDA ≡
//! CPU bitwise); the host extracts the zero set with marching tetrahedra on the Kuhn split of
//! every cube (all cubes share the main diagonal, so the triangulation is conforming and the
//! mesh is closed and consistently oriented by construction — orientation is combinatorial,
//! from positively oriented tetrahedra, never from degenerate geometry).
//!
//! Deviation: every vertex lies on a grid edge whose end signs differ, and the field's sign is
//! exact, so the true surface crosses that edge; every mesh point is therefore within one
//! tetrahedron diameter (h·√3) of the true surface. Features thinner than h may be missed:
//! the volume is reported against the certified (rigorous) bounds, which do not miss them.

use crate::abi::*;
use crate::math::compile_node;
use crate::studio::{dops_library, Design};
use agcad_driver::*;
use agcad_geo::math::v3;
use agcad_geo::tessellate::Mesh;
use agcad_kernels::params::upload_params;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshArgs {
    pub tape: u64,
    pub tables: u64,
    pub out: u64,
    pub _pad0: u64,
    pub lo: [f64; 3],
    pub h: f64,
    pub count: u32,
    /// Grid nodes per axis.
    pub dims: [u32; 3],
}

#[derive(Clone, Debug)]
pub struct ImplicitMesh {
    pub mesh: Mesh,
    /// Grid spacing (mm) and the rigorous mesh → surface distance bound h·√3.
    pub h: f64,
    pub deviation: f64,
    /// Mesh volume (divergence theorem) and area.
    pub volume: f64,
    pub area: f64,
}

/// Sample the design's SDF on a node grid covering [lo, hi] padded by two cells, then extract.
pub fn extract_mesh<D: KernelExt>(dev: Arc<D>, design: &Design, lo: [f64; 3], hi: [f64; 3], h: f64) -> DriverResult<ImplicitMesh> {
    let lo = [lo[0] - 2.0 * h, lo[1] - 2.0 * h, lo[2] - 2.0 * h];
    let dims = [0, 1, 2].map(|k| (((hi[k] + 2.0 * h - lo[k]) / h).ceil() as u32).max(1) + 1);
    let nn = dims.iter().map(|&n| n as usize).product::<usize>();
    if nn > 64_000_000 { return Err(EnqueueError::InvalidConfig(format!("mesh grid {dims:?} too large: increase the spacing")).into()); }
    let d = &*dev;
    let s = d.stream(Priority::Low)?;
    let lib = d.load_library(&dops_library())?;
    let mut cn = Vec::with_capacity(design.nodes.len());
    let mut tabs = vec![0.0f32; design.nodes.len().max(1) * TABLE];
    let prm: Vec<f64> = if design.params.is_empty() { vec![0.0] } else { design.params.clone() };
    for (i, n) in design.nodes.iter().enumerate() { cn.push(compile_node(n, &prm, &mut tabs[i * TABLE..(i + 1) * TABLE])); }
    if cn.is_empty() { cn.push(CNode::zeroed()); }
    let up = |bytes: &[u8]| -> DriverResult<D::Buffer> { let b = d.alloc_bytes(&s, bytes.len().max(16))?; d.upload_from_host_async(&s, bytes, &b, 0)?; Ok(b) };
    let tape = up(bytemuck::cast_slice(&cn))?;
    let tables = up(bytemuck::cast_slice(&tabs))?;
    let out = d.alloc_bytes(&s, nn * 4)?;
    let a = upload_params(d, &s, &MeshArgs { tape: tape.device_ptr(), tables: tables.device_ptr(), out: out.device_ptr(), _pad0: 0, lo, h, count: design.nodes.len() as u32, dims })?;
    d.launch(&s, lib, "ag_mesh_sample", &LaunchConfig::linear((nn as u64).min(8192 * 256), 256), &a.buffer)?;
    let mut pin = d.pinned(nn * 4)?;
    d.download_packed_async(&s, &[Region { buf: &out, offset: 0, bytes: nn * 4 }], &mut pin)?;
    d.sync(&s)?;
    let f: Vec<f32> = bytemuck::pod_collect_to_vec(&pin.as_bytes()[..nn * 4]);
    Ok(march_tets(&f, dims, lo, h))
}

/// Kuhn tetrahedra of the unit cube (corner bits x = 1, y = 2, z = 4), positively oriented.
fn kuhn() -> [[usize; 4]; 6] {
    let c = |i: usize| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64];
    let mut t = [[0usize; 4]; 6];
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for (q, p) in perms.iter().enumerate() {
        let a = 1usize << p[0];
        let b = a | (1 << p[1]);
        let mut tet = [0, a, b, 7];
        let (v0, v1, v2, v3) = (c(tet[0]), c(tet[1]), c(tet[2]), c(tet[3]));
        let e = |u: [f64; 3]| [u[0] - v0[0], u[1] - v0[1], u[2] - v0[2]];
        let (x, y, z) = (e(v1), e(v2), e(v3));
        let det = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0]);
        if det < 0.0 { tet.swap(2, 3); }
        t[q] = tet;
    }
    t
}

fn even(p: [usize; 4]) -> bool {
    let mut inv = 0;
    for i in 0..4 { for j in i + 1..4 { if p[i] > p[j] { inv += 1; } } }
    inv % 2 == 0
}

/// Marching tetrahedra over node values `f` (inside: f < 0) on a `dims` node grid.
pub fn march_tets(f: &[f32], dims: [u32; 3], lo: [f64; 3], h: f64) -> ImplicitMesh {
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let pos = |n: usize| v3(lo[0] + (n % nx) as f64 * h, lo[1] + ((n / nx) % ny) as f64 * h, lo[2] + (n / (nx * ny)) as f64 * h);
    let tets = kuhn();
    // outward faces opposite each vertex of a positively oriented tetrahedron
    const OPP: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    let mut mesh = Mesh { positions: Vec::new(), triangles: Vec::new(), face_of: Vec::new() };
    let mut verts: HashMap<(usize, usize), u32> = HashMap::new();
    let mut vid = |a: usize, b: usize, mesh: &mut Mesh| -> u32 {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        *verts.entry((a, b)).or_insert_with(|| {
            let (fa, fb) = (f[a] as f64, f[b] as f64);
            let t = (fa / (fa - fb)).clamp(0.0, 1.0);
            let (pa, pb) = (pos(a), pos(b));
            mesh.positions.push(pa + (pb - pa) * t);
            (mesh.positions.len() - 1) as u32
        })
    };
    for k in 0..nz.saturating_sub(1) { for j in 0..ny.saturating_sub(1) { for i in 0..nx.saturating_sub(1) {
        let corner = |c: usize| node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
        for tet in &tets {
            let g = [corner(tet[0]), corner(tet[1]), corner(tet[2]), corner(tet[3])];
            let ins: Vec<usize> = (0..4).filter(|&q| f[g[q]] < 0.0).collect();
            let mut tri = |a: (usize, usize), b: (usize, usize), c: (usize, usize), mesh: &mut Mesh| {
                let t = [vid(g[a.0], g[a.1], mesh), vid(g[b.0], g[b.1], mesh), vid(g[c.0], g[c.1], mesh)];
                mesh.triangles.push(t);
                mesh.face_of.push(0);
            };
            match ins.len() {
                1 => { let i0 = ins[0]; let o = OPP[i0]; tri((i0, o[0]), (i0, o[1]), (i0, o[2]), &mut mesh); }
                3 => { let o0 = (0..4).find(|q| !ins.contains(q)).unwrap(); let o = OPP[o0]; tri((o0, o[2]), (o0, o[1]), (o0, o[0]), &mut mesh); }
                2 => {
                    let out: Vec<usize> = (0..4).filter(|q| !ins.contains(q)).collect();
                    let (a, b) = (ins[0], ins[1]);
                    let (c, dd) = if even([a, b, out[0], out[1]]) { (out[0], out[1]) } else { (out[1], out[0]) };
                    tri((a, c), (a, dd), (b, dd), &mut mesh);
                    tri((a, c), (b, dd), (b, c), &mut mesh);
                }
                _ => {}
            }
        }
    }}}
    let (volume, area, _) = mesh.mass_properties();
    ImplicitMesh { mesh, h, deviation: h * 3f64.sqrt(), volume, area }
}

/// CPU reference twin of `ag_mesh_sample`: SDF at grid nodes.
pub unsafe fn ag_mesh_sample(params: *const u8, _: &LaunchConfig) {
    let a: MeshArgs = unsafe { core::ptr::read_unaligned(params as *const MeshArgs) };
    let (nx, ny, nz) = (a.dims[0] as usize, a.dims[1] as usize, a.dims[2] as usize);
    let tape = unsafe { core::slice::from_raw_parts(a.tape as *const CNode, a.count as usize) };
    let tb = unsafe { core::slice::from_raw_parts(a.tables as *const f32, (a.count as usize).max(1) * TABLE) };
    let out = unsafe { core::slice::from_raw_parts_mut(a.out as *mut f32, nx * ny * nz) };
    for n in 0..nx * ny * nz {
        let (i, j, k) = (n % nx, (n / nx) % ny, n / (nx * ny));
        let x = (a.lo[0] + i as f64 * a.h) as f32;
        let y = (a.lo[1] + j as f64 * a.h) as f32;
        let z = (a.lo[2] + k as f64 * a.h) as f32;
        out[n] = if tape.is_empty() { crate::math::FAR32 } else { crate::math::tape_sdf_c(tape, tb, x, y, z) };
    }
}
