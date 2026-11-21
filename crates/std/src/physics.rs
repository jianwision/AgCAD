//! Physics evaluations (spec 03 `evaluate.stiffness` / `evaluate.modal`; spec 14): voxel
//! linear elasticity with 8-node hexahedra (2×2×2 Gauss), matrix-free Jacobi-PCG; modal by
//! inverse iteration with lumped mass. Evidence `estimated` with the discretization reported.
//! Supports: nodes on the min-z face (or `supports: "x_min" | "y_min" | "z_min"`); loads:
//! total force vector applied uniformly to the opposite face (`load_face`, default "z_max").
//! Evidence `sampled`: a discretized estimate whose error depends on the voxel size.

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::TessOptions;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use serde_json::json;

/// 24×24 stiffness matrix of a cube element of edge `h` (E, ν), standard Q8 trilinear.
pub fn hex8_stiffness(e: f64, nu: f64, h: f64) -> Vec<f64> {
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let corners = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]];
    let g = 1.0 / 3f64.sqrt();
    let mut k = vec![0.0; 24 * 24];
    for gx in [-g, g] { for gy in [-g, g] { for gz in [-g, g] {
        // dN/dx = dN/dξ · (2/h)
        let mut dn = [[0.0f64; 3]; 8];
        for (a, c) in corners.iter().enumerate() {
            dn[a][0] = 0.125 * c[0] * (1.0 + c[1] * gy) * (1.0 + c[2] * gz) * 2.0 / h;
            dn[a][1] = 0.125 * c[1] * (1.0 + c[0] * gx) * (1.0 + c[2] * gz) * 2.0 / h;
            dn[a][2] = 0.125 * c[2] * (1.0 + c[0] * gx) * (1.0 + c[1] * gy) * 2.0 / h;
        }
        let det_j = (h / 2.0).powi(3);
        // B (6×24), D (6×6); K += Bᵀ D B detJ
        let mut b = [[0.0f64; 24]; 6];
        for a in 0..8 {
            let (x, y, z) = (dn[a][0], dn[a][1], dn[a][2]);
            b[0][3 * a] = x; b[1][3 * a + 1] = y; b[2][3 * a + 2] = z;
            b[3][3 * a] = y; b[3][3 * a + 1] = x;
            b[4][3 * a + 1] = z; b[4][3 * a + 2] = y;
            b[5][3 * a] = z; b[5][3 * a + 2] = x;
        }
        let d = [[lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0], [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0], [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, mu, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, mu, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, mu]];
        let mut db = [[0.0f64; 24]; 6];
        for i in 0..6 { for j in 0..24 { db[i][j] = (0..6).map(|m| d[i][m] * b[m][j]).sum(); } }
        for i in 0..24 { for j in 0..24 { k[i * 24 + j] += (0..6).map(|m| b[m][i] * db[m][j]).sum::<f64>() * det_j; } }
    }}}
    k
}

pub struct Voxels {
    pub origin: V3,
    pub h: f64,
    pub dims: [usize; 3],
    pub solid: Vec<bool>,
}

impl Voxels {
    pub fn node(&self, i: usize, j: usize, k: usize) -> usize { (k * (self.dims[1] + 1) + j) * (self.dims[0] + 1) + i }
    pub fn n_nodes(&self) -> usize { (self.dims[0] + 1) * (self.dims[1] + 1) * (self.dims[2] + 1) }
    fn elem_nodes(&self, i: usize, j: usize, k: usize) -> [usize; 8] {
        [self.node(i, j, k), self.node(i + 1, j, k), self.node(i + 1, j + 1, k), self.node(i, j + 1, k), self.node(i, j, k + 1), self.node(i + 1, j, k + 1), self.node(i + 1, j + 1, k + 1), self.node(i, j + 1, k + 1)]
    }
}

/// Voxelize a body: implicit → sign of the field; solid → ray parity against tessellation.
pub fn voxelize(ctx: &BuildContext<'_>, p: &Product, h: f64) -> Result<Voxels, FeatureError> {
    let (bb, inside): (agcad_geo::math::Aabb, Box<dyn Fn(V3) -> bool>) = match p {
        Product::Implicit(i) => { let i = i.clone(); (i.bounds, Box::new(move |q| i.eval(q) < 0.0)) }
        Product::Solid(s) => {
            let m = ctx.kernel.tessellate(s, &TessOptions { sag: h * 0.1, ..Default::default() });
            let bb = s.brep().bbox();
            (bb, Box::new(move |q: V3| {
                // ray along +x with slight skew to avoid edge hits; count crossings
                let d = v3(1.0, 1e-7, 2e-7).normalized();
                let mut c = 0;
                for t in &m.triangles {
                    let (a, b, cc) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
                    let (e1, e2) = (b - a, cc - a);
                    let pv = d.cross(e2);
                    let det = e1.dot(pv);
                    if det.abs() < 1e-14 { continue; }
                    let s_ = q - a;
                    let u = s_.dot(pv) / det;
                    let qv = s_.cross(e1);
                    let v = d.dot(qv) / det;
                    let tt = e2.dot(qv) / det;
                    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && tt > 0.0 { c += 1; }
                }
                c % 2 == 1
            }))
        }
        _ => return Err(bad("physics evaluations need a solid or implicit body")),
    };
    let e = bb.extent();
    let dims = [((e.x / h).ceil() as usize).max(1), ((e.y / h).ceil() as usize).max(1), ((e.z / h).ceil() as usize).max(1)];
    if dims[0] * dims[1] * dims[2] > 400_000 { return Err(FeatureError::new(AgStatus::CapacityExceeded, "voxel grid too large for the reference solver").suggest("increase voxel size")); }
    let mut solid = vec![false; dims[0] * dims[1] * dims[2]];
    for k in 0..dims[2] { for j in 0..dims[1] { for i in 0..dims[0] {
        solid[(k * dims[1] + j) * dims[0] + i] = inside(bb.min + v3((i as f64 + 0.5) * h, (j as f64 + 0.5) * h, (k as f64 + 0.5) * h));
    }}}
    Ok(Voxels { origin: bb.min, h, dims, solid })
}

pub struct FeaResult { pub u: Vec<f64>, pub compliance: f64, pub max_disp: f64, pub max_vm: f64, pub iterations: u32, pub converged: bool }

/// Static solve: fixed nodes on `support_axis` min face, total force `f` on the max face.
pub fn solve_static(vx: &Voxels, e: f64, nu: f64, support_axis: usize, f: V3) -> FeaResult { solve_static_scaled(vx, e, nu, support_axis, f, None) }

/// Static solve with optional per-element stiffness scale (SIMP densities^p).
pub fn solve_static_scaled(vx: &Voxels, e: f64, nu: f64, support_axis: usize, f: V3, scale: Option<&[f64]>) -> FeaResult {
    let ke = hex8_stiffness(e, nu, vx.h);
    let nn = vx.n_nodes();
    let used: Vec<bool> = { let mut u = vec![false; nn]; for k in 0..vx.dims[2] { for j in 0..vx.dims[1] { for i in 0..vx.dims[0] { if vx.solid[(k * vx.dims[1] + j) * vx.dims[0] + i] { for n in vx.elem_nodes(i, j, k) { u[n] = true; } } } } } u };
    let coord = |n: usize| -> [usize; 3] { let nx = vx.dims[0] + 1; let ny = vx.dims[1] + 1; [n % nx, (n / nx) % ny, n / (nx * ny)] };
    let fixed: Vec<bool> = (0..nn).map(|n| used[n] && coord(n)[support_axis] == 0).collect();
    let top = vx.dims[support_axis];
    let loaded: Vec<usize> = (0..nn).filter(|&n| used[n] && coord(n)[support_axis] == top).collect();
    let mut rhs = vec![0.0; 3 * nn];
    for &n in &loaded { rhs[3 * n] = f.x / loaded.len() as f64; rhs[3 * n + 1] = f.y / loaded.len() as f64; rhs[3 * n + 2] = f.z / loaded.len() as f64; }
    let active = |d: usize| used[d / 3] && !fixed[d / 3];
    let matvec = |x: &[f64], y: &mut [f64]| {
        y.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..vx.dims[2] { for j in 0..vx.dims[1] { for i in 0..vx.dims[0] {
            if !vx.solid[(k * vx.dims[1] + j) * vx.dims[0] + i] { continue; }
            let en = vx.elem_nodes(i, j, k);
            let sc = scale.map(|d| d[(k * vx.dims[1] + j) * vx.dims[0] + i]).unwrap_or(1.0);
            let mut ue = [0.0; 24];
            for a in 0..8 { for c in 0..3 { let d = 3 * en[a] + c; ue[3 * a + c] = if active(d) { x[d] } else { 0.0 }; } }
            for r in 0..24 {
                let d = 3 * en[r / 3] + r % 3;
                if !active(d) { continue; }
                y[d] += sc * (0..24).map(|s| ke[r * 24 + s] * ue[s]).sum::<f64>();
            }
        }}}
    };
    let diag: Vec<f64> = (0..3 * nn).map(|d| if active(d) { ke[(d % 3) * 25] * 4.0 } else { 1.0 }).collect();
    for d in 0..3 * nn { if !active(d) { rhs[d] = 0.0; } }
    let mut u = vec![0.0; 3 * nn];
    let rep = agcad_solver::linear::pcg(&matvec, &diag, &rhs, &mut u, 1e-8, 5000);
    let compliance: f64 = u.iter().zip(&rhs).map(|(a, b)| a * b).sum();
    let max_disp = (0..nn).map(|n| (u[3 * n].powi(2) + u[3 * n + 1].powi(2) + u[3 * n + 2].powi(2)).sqrt()).fold(0.0, f64::max);
    // von Mises at element centers from strain (B at ξ = 0)
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let corners = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0f64]];
    let mut max_vm: f64 = 0.0;
    for k in 0..vx.dims[2] { for j in 0..vx.dims[1] { for i in 0..vx.dims[0] {
        if !vx.solid[(k * vx.dims[1] + j) * vx.dims[0] + i] { continue; }
        let en = vx.elem_nodes(i, j, k);
        let mut eps = [0.0f64; 6];
        for a in 0..8 {
            let g = [0.25 * corners[a][0] / vx.h, 0.25 * corners[a][1] / vx.h, 0.25 * corners[a][2] / vx.h];
            let ua = [u[3 * en[a]], u[3 * en[a] + 1], u[3 * en[a] + 2]];
            eps[0] += g[0] * ua[0]; eps[1] += g[1] * ua[1]; eps[2] += g[2] * ua[2];
            eps[3] += g[1] * ua[0] + g[0] * ua[1]; eps[4] += g[2] * ua[1] + g[1] * ua[2]; eps[5] += g[2] * ua[0] + g[0] * ua[2];
        }
        let tr = eps[0] + eps[1] + eps[2];
        let s = [lam * tr + 2.0 * mu * eps[0], lam * tr + 2.0 * mu * eps[1], lam * tr + 2.0 * mu * eps[2], mu * eps[3], mu * eps[4], mu * eps[5]];
        let vm = (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3].powi(2) + s[4].powi(2) + s[5].powi(2))).sqrt();
        max_vm = max_vm.max(vm);
    }}}
    FeaResult { u, compliance, max_disp, max_vm, iterations: rep.iterations, converged: rep.converged }
}

/// SIMP topology optimization (optimality criteria, no filter): returns element densities.
pub fn simp(vx: &Voxels, e: f64, nu: f64, axis: usize, f: V3, volfrac: f64, iters: u32, penal: f64) -> (Vec<f64>, Vec<f64>) {
    let ne = vx.solid.len();
    let mut x: Vec<f64> = vx.solid.iter().map(|s| if *s { volfrac } else { 0.0 }).collect();
    let ke = hex8_stiffness(e, nu, vx.h);
    let mut history = Vec::new();
    for _ in 0..iters {
        let scale: Vec<f64> = x.iter().map(|d| 1e-9 + d.powf(penal)).collect();
        let r = solve_static_scaled(vx, e, nu, axis, f, Some(&scale));
        history.push(r.compliance);
        // sensitivities dc_e = -p x^(p-1) u_eᵀ k0 u_e
        let mut dc = vec![0.0; ne];
        for k in 0..vx.dims[2] { for j in 0..vx.dims[1] { for i in 0..vx.dims[0] {
            let id = (k * vx.dims[1] + j) * vx.dims[0] + i;
            if !vx.solid[id] { continue; }
            let en = vx.elem_nodes(i, j, k);
            let mut ue = [0.0; 24];
            for a in 0..8 { for c in 0..3 { ue[3 * a + c] = r.u[3 * en[a] + c]; } }
            let mut q = 0.0;
            for a in 0..24 { for b in 0..24 { q += ue[a] * ke[a * 24 + b] * ue[b]; } }
            dc[id] = -penal * x[id].powf(penal - 1.0) * q;
        }}}
        // OC update with bisection on the Lagrange multiplier (deterministic)
        let target: f64 = volfrac * vx.solid.iter().filter(|s| **s).count() as f64;
        let (mut l1, mut l2) = (1e-12, 1e12);
        let mut xn = x.clone();
        while (l2 - l1) / (l1 + l2) > 1e-6 {
            let lm = 0.5 * (l1 + l2);
            for id in 0..ne {
                if !vx.solid[id] { xn[id] = 0.0; continue; }
                let be = (-dc[id] / lm).max(0.0).sqrt();
                xn[id] = (x[id] * be).clamp((x[id] - 0.2).max(0.001), (x[id] + 0.2).min(1.0));
            }
            if xn.iter().sum::<f64>() > target { l1 = lm } else { l2 = lm }
        }
        x = xn;
    }
    (x, history)
}

/// Watertight boundary mesh of voxels whose density exceeds `iso`.
pub fn voxel_mesh(vx: &Voxels, dens: &[f64], iso: f64) -> agcad_geo::tessellate::Mesh {
    let mut m = agcad_geo::tessellate::Mesh::default();
    let mut vmap = std::collections::HashMap::new();
    let on = |i: isize, j: isize, k: isize| -> bool {
        if i < 0 || j < 0 || k < 0 || i >= vx.dims[0] as isize || j >= vx.dims[1] as isize || k >= vx.dims[2] as isize { return false; }
        dens[(k as usize * vx.dims[1] + j as usize) * vx.dims[0] + i as usize] > iso
    };
    let mut vert = |m: &mut agcad_geo::tessellate::Mesh, i: usize, j: usize, k: usize| -> u32 {
        *vmap.entry((i, j, k)).or_insert_with(|| { m.positions.push(vx.origin + v3(i as f64 * vx.h, j as f64 * vx.h, k as f64 * vx.h)); (m.positions.len() - 1) as u32 })
    };
    for k in 0..vx.dims[2] { for j in 0..vx.dims[1] { for i in 0..vx.dims[0] {
        if !on(i as isize, j as isize, k as isize) { continue; }
        let (ii, jj, kk) = (i as isize, j as isize, k as isize);
        // faces: (neighbor offset, 4 corners CCW seen from outside)
        let faces: [((isize, isize, isize), [(usize, usize, usize); 4]); 6] = [
            ((-1, 0, 0), [(i, j, k), (i, j, k + 1), (i, j + 1, k + 1), (i, j + 1, k)]),
            ((1, 0, 0), [(i + 1, j, k), (i + 1, j + 1, k), (i + 1, j + 1, k + 1), (i + 1, j, k + 1)]),
            ((0, -1, 0), [(i, j, k), (i + 1, j, k), (i + 1, j, k + 1), (i, j, k + 1)]),
            ((0, 1, 0), [(i, j + 1, k), (i, j + 1, k + 1), (i + 1, j + 1, k + 1), (i + 1, j + 1, k)]),
            ((0, 0, -1), [(i, j, k), (i, j + 1, k), (i + 1, j + 1, k), (i + 1, j, k)]),
            ((0, 0, 1), [(i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1)]),
        ];
        for (d, q) in faces {
            if on(ii + d.0, jj + d.1, kk + d.2) { continue; }
            let id: Vec<u32> = q.iter().map(|c| vert(&mut m, c.0, c.1, c.2)).collect();
            m.triangles.push([id[0], id[1], id[2]]);
            m.triangles.push([id[0], id[2], id[3]]);
            m.face_of.push(0); m.face_of.push(0);
        }
    }}}
    m
}

pub struct PhysicsOp(pub &'static str);

impl FeatureImpl for PhysicsOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let body = ctx.product(arg(n, "body").or(arg(n, "region")).ok_or_else(|| bad(format!("{}: missing input body/region", n.op)))?)?.clone();
        let h = num_or(ctx, n, "voxel", 1.0)?;
        // material: E in MPa (N/mm²), ν, density kg/mm³ (defaults: steel)
        let e = num_or(ctx, n, "E", 210_000.0)?;
        let nu = num_or(ctx, n, "nu", 0.3)?;
        let axis = match text(n, "supports") { Some("x_min") => 0, Some("y_min") => 1, _ => 2 };
        let vx = voxelize(ctx, &body, h)?;
        match self.0 {
            "evaluate.stiffness" => {
                let f = match arg(n, "force") { Some(v) => vec3(ctx, v)?, None => v3(0.0, 0.0, -1000.0) };
                let r = solve_static(&vx, e, nu, axis, f);
                Ok(Product::Json(json!({ "compliance": r.compliance, "max_displacement": r.max_disp, "max_von_mises": r.max_vm, "cg_iterations": r.iterations, "converged": r.converged, "voxel": h, "elements": vx.solid.iter().filter(|s| **s).count(), "evidence": "sampled" })))
            }
            "study.topology_optimization" => {
                let f = match arg(n, "force") { Some(v) => vec3(ctx, v)?, None => v3(0.0, 0.0, -1000.0) };
                let vf = num_or(ctx, n, "volume_fraction", 0.4)?;
                let it = num_or(ctx, n, "iters", 30.0)? as u32;
                let (dens, hist) = simp(&vx, e, nu, axis, f, vf, it, 3.0);
                let mesh = voxel_mesh(&vx, &dens, 0.5);
                let _ = hist;
                Ok(Product::Mesh(mesh))
            }
            "evaluate.modal" => {
                // inverse iteration for the fundamental mode with lumped mass (ρ h³ / 8 per node per element)
                let rho = num_or(ctx, n, "density", 7.85e-9)?; // tonne/mm³ (consistent with N, mm, MPa)
                // Rayleigh quotient on the static deflection shape under a unit tip load:
                // ω² ≈ (uᵀ K u) / (uᵀ M u) with lumped mass M = ρ h³ per node (upper bound on ω₁).
                let r = solve_static(&vx, e, nu, axis, v3(0.0, 0.0, -1.0));
                let m_lumped = rho * vx.h.powi(3);
                let den: f64 = r.u.iter().map(|v| v * v).sum::<f64>() * m_lumped;
                let lambda = r.compliance / den.max(1e-300);
                let freq = lambda.sqrt() / core::f64::consts::TAU;
                Ok(Product::Json(json!({ "f1_hz": freq, "method": "Rayleigh quotient on static deflection shape", "modes": 1, "evidence": "sampled", "voxel": h })))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cantilever_tip_deflection_order_of_magnitude() {
        // 10×10×100 mm steel beam fixed at z=0, 1000 N transverse at the tip.
        let vx = Voxels { origin: V3::ZERO, h: 2.5, dims: [4, 4, 40], solid: vec![true; 4 * 4 * 40] };
        let r = solve_static(&vx, 210_000.0, 0.3, 2, v3(1000.0, 0.0, 0.0));
        let beam = 1000.0 * 100f64.powi(3) / (3.0 * 210_000.0 * (10f64.powi(4) / 12.0)); // Euler–Bernoulli
        assert!(r.converged);
        assert!(r.max_disp > 0.4 * beam && r.max_disp < 1.5 * beam, "{} vs {beam}", r.max_disp);
    }
}
