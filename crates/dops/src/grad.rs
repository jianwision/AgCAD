//! Differentiable GPU CAD: Reynolds-transport shape gradients and gradient-guided polish (Phase 4).
//!
//! For any compiled feature tape `φ(x; p)`, the exact shape derivative of volume with respect to
//! parameter `p_j` is given by the Reynolds transport theorem:
//!
//! ```text
//! ∂V/∂p_j = -∫_{∂Ω} (∂φ/∂p_j) / ‖∇φ‖ dA
//! ```
//!
//! And similarly for the first moments `∫_Ω x dV` (giving `∂centroid/∂p_j`).
//! Evaluating `∂φ/∂p_j` on terminal boundary cells avoids volumetric remeshing and yields
//! smooth, low-variance shape sensitivities for every continuous parameter, enabling
//! projected-gradient / L-BFGS micro-steps inside hybrid `CuModel` + CMA-ES search.

use crate::abi::*;
use crate::math::{tape_sdf_c, Iv};
use crate::studio::Design;

/// Shape derivatives of a design with respect to each parameter `0..params.len()`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeGradient {
    /// `∂V/∂p_j` (mm³ per parameter unit).
    pub d_vol: Vec<f64>,
    /// `∂centroid/∂p_j` (mm per parameter unit) for x, y, z.
    pub d_centroid: Vec<[f64; 3]>,
    /// Sampled volume at the evaluation point (mm³).
    pub volume: f64,
    /// Sampled surface area (mm²).
    pub surface_area: f64,
    /// Number of active boundary cells contributing to the surface integral.
    pub boundary_cells: usize,
    /// True if any parameter exhibited a non-smooth / topological jump (`AG-04-005`).
    pub discontinuous: Vec<bool>,
}

/// Compute Reynolds-transport shape gradients `(∂V/∂p, ∂centroid/∂p)` of `design` over `goal`'s
/// domain on a uniform/narrow-band grid of resolution `res` per axis.
pub fn reynolds_shape_gradients(design: &Design, goal: &Goal, res: u32) -> ShapeGradient {
    let np = design.params.len();
    let res = res.clamp(8, 128) as usize;
    let lo = goal.domain_lo;
    let hi = goal.domain_hi;
    let h = [
        (hi[0] - lo[0]) / res as f64,
        (hi[1] - lo[1]) / res as f64,
        (hi[2] - lo[2]) / res as f64,
    ];
    let cell_vol = h[0] * h[1] * h[2];
    let cell_diag = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt() as f32;
    let (base_nodes, base_tabs) = design.compile();

    // Collect boundary cells & volumetric totals in one pass using interval culling per cell
    let mut bcells: Vec<([f32; 3], [f32; 3], f64)> = Vec::new();
    let mut inside_vol = 0.0f64;
    let mut centroid_num = [0.0f64; 3];
    let mut area_sum = 0.0f64;
    let eps_g = (h[0].min(h[1]).min(h[2]) * 0.25) as f32;

    for kz in 0..res {
        let z0 = (lo[2] + kz as f64 * h[2]) as f32;
        let z1 = (lo[2] + (kz + 1) as f64 * h[2]) as f32;
        let zc = 0.5 * (z0 + z1);
        for ky in 0..res {
            let y0 = (lo[1] + ky as f64 * h[1]) as f32;
            let y1 = (lo[1] + (ky + 1) as f64 * h[1]) as f32;
            let yc = 0.5 * (y0 + y1);
            for kx in 0..res {
                let x0 = (lo[0] + kx as f64 * h[0]) as f32;
                let x1 = (lo[0] + (kx + 1) as f64 * h[0]) as f32;
                let xc = 0.5 * (x0 + x1);
                let iv = crate::math::tape_iv(&base_nodes, &base_tabs, Iv { lo: x0, hi: x1 }, Iv { lo: y0, hi: y1 }, Iv { lo: z0, hi: z1 });
                if iv.lo > 0.0 { continue; }
                if iv.hi < 0.0 {
                    inside_vol += cell_vol;
                    centroid_num[0] += cell_vol * xc as f64;
                    centroid_num[1] += cell_vol * yc as f64;
                    centroid_num[2] += cell_vol * zc as f64;
                    continue;
                }
                let f0 = tape_sdf_c(&base_nodes, &base_tabs, xc, yc, zc);
                if f0.abs() > 0.6 * cell_diag {
                    if f0 < 0.0 {
                        inside_vol += cell_vol;
                        centroid_num[0] += cell_vol * xc as f64;
                        centroid_num[1] += cell_vol * yc as f64;
                        centroid_num[2] += cell_vol * zc as f64;
                    }
                    continue;
                }
                let gx = (tape_sdf_c(&base_nodes, &base_tabs, xc + eps_g, yc, zc) - tape_sdf_c(&base_nodes, &base_tabs, xc - eps_g, yc, zc)) / (2.0 * eps_g);
                let gy = (tape_sdf_c(&base_nodes, &base_tabs, xc, yc + eps_g, zc) - tape_sdf_c(&base_nodes, &base_tabs, xc, yc - eps_g, zc)) / (2.0 * eps_g);
                let gz = (tape_sdf_c(&base_nodes, &base_tabs, xc, yc, zc + eps_g) - tape_sdf_c(&base_nodes, &base_tabs, xc, yc, zc - eps_g)) / (2.0 * eps_g);
                let gl = (gx * gx + gy * gy + gz * gz).sqrt();
                if gl < 1e-5 { continue; }
                // Projected surface point and cell surface area element (slab approximation)
                let nx = gx / gl;
                let ny = gy / gl;
                let nz = gz / gl;
                let px = xc - nx * f0;
                let py = yc - ny * f0;
                let pz = zc - nz * f0;
                // Surface area element inside axis-aligned cell: V_cell / (|nx|hx + |ny|hy + |nz|hz)
                let denom = (nx.abs() as f64 * h[0] + ny.abs() as f64 * h[1] + nz.abs() as f64 * h[2]).max(1e-9);
                let da = cell_vol / denom;
                let fill = (0.5 - (f0 as f64) / denom).clamp(0.0, 1.0);
                inside_vol += fill * cell_vol;
                centroid_num[0] += fill * cell_vol * xc as f64;
                centroid_num[1] += fill * cell_vol * yc as f64;
                centroid_num[2] += fill * cell_vol * zc as f64;
                area_sum += da;
                bcells.push(([px, py, pz], [nx, ny, nz], da));
            }
        }
    }

    let vol = inside_vol.max(1e-12);
    let centroid = [centroid_num[0] / vol, centroid_num[1] / vol, centroid_num[2] / vol];
    let mut d_vol = vec![0.0f64; np];
    let mut d_centroid = vec![[0.0f64; 3]; np];
    let mut discontinuous = vec![false; np];

    for j in 0..np {
        let span = if j < MAX_PARAMS && goal.param_hi[j] > goal.param_lo[j] {
            (goal.param_hi[j] - goal.param_lo[j]) * 1e-3
        } else {
            design.params[j].abs().max(1.0) * 1e-3
        };
        let dp = span.max(1e-4);
        let mut d_plus = design.clone();
        let mut d_minus = design.clone();
        d_plus.params[j] += dp;
        d_minus.params[j] -= dp;
        let (np_nodes, np_tabs) = d_plus.compile();
        let (nm_nodes, nm_tabs) = d_minus.compile();

        let mut dv = 0.0f64;
        let mut dm = [0.0f64; 3];
        let mut max_jump = 0.0f64;
        for &(p, _, da) in &bcells {
            let fp = tape_sdf_c(&np_nodes, &np_tabs, p[0], p[1], p[2]) as f64;
            let fm = tape_sdf_c(&nm_nodes, &nm_tabs, p[0], p[1], p[2]) as f64;
            let dphi_dp = (fp - fm) / (2.0 * dp);
            let f_mid = tape_sdf_c(&base_nodes, &base_tabs, p[0], p[1], p[2]) as f64;
            let cur = (fp - 2.0 * f_mid + fm).abs() / (dp * dp);
            if cur > max_jump { max_jump = cur; }
            let vn = -dphi_dp; // outward normal boundary velocity
            dv += vn * da;
            dm[0] += vn * (p[0] as f64 - centroid[0]) * da;
            dm[1] += vn * (p[1] as f64 - centroid[1]) * da;
            dm[2] += vn * (p[2] as f64 - centroid[2]) * da;
        }
        d_vol[j] = dv;
        d_centroid[j] = [dm[0] / vol, dm[1] / vol, dm[2] / vol];
        discontinuous[j] = max_jump > 1e4;
    }

    ShapeGradient {
        d_vol,
        d_centroid,
        volume: inside_vol,
        surface_area: area_sum,
        boundary_cells: bcells.len(),
        discontinuous,
    }
}

/// Projected-gradient continuous polish step targeting `target_vol` using Reynolds-transport
/// sensitivities (`d_vol`). Returns updated `Design` clamped strictly to `[param_lo, param_hi]`.
pub fn gradient_polish_volume(design: &Design, goal: &Goal, target_vol: f64, steps: usize, res: u32) -> (Design, Vec<f64>) {
    let mut cur = design.clone();
    let mut history = Vec::with_capacity(steps);
    let np = cur.params.len().min(MAX_PARAMS);
    for _ in 0..steps {
        let g = reynolds_shape_gradients(&cur, goal, res);
        history.push(g.volume);
        let err = g.volume - target_vol;
        if err.abs() <= 1e-3 * target_vol.abs().max(1.0) { break; }
        // Minimum-norm Newton step Δp = -(err / ‖∇V‖²) ∇V over free parameters
        let mut norm2 = 0.0f64;
        for j in 0..np {
            if goal.param_hi[j] > goal.param_lo[j] {
                norm2 += g.d_vol[j] * g.d_vol[j];
            }
        }
        if norm2 < 1e-18 { break; }
        let alpha = (-0.8 * err / norm2).clamp(-1e6, 1e6);
        for j in 0..np {
            if goal.param_hi[j] > goal.param_lo[j] {
                let max_step = 0.25 * (goal.param_hi[j] - goal.param_lo[j]);
                let step = (alpha * g.d_vol[j]).clamp(-max_step, max_step);
                cur.params[j] = (cur.params[j] + step).clamp(goal.param_lo[j], goal.param_hi[j]);
            }
        }
    }
    (cur, history)
}
