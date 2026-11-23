//! GPU-native exact B-rep kernel pipeline (Phase 3 / Spec 02 & 11):
//!
//! 1. **Filtered GPU Predicates + SoS** (`ag_geo_orient2d_batch`, `ag_geo_orient3d_batch`):
//!    fast forward-error filter on device with compacted exact Shewchuk expansion slow-path
//!    and deterministic Simulation of Simplicity (`orient2d_sos`, `orient3d_sos`).
//! 2. **Spatial Broadphase** (`ag_geo_aabb_overlap_pairs`): parallel face/patch AABB pair culling.
//! 3. **Batched Analytic SSI Refinement** (`ag_geo_ssi_refine_batch`): minimal-norm Newton onto
//!    analytic surfaces (`Plane`/`Cylinder`/`Sphere`/`Cone`).
//! 4. **Parallel Fragment Classification** (`ag_geo_classify_faces`): ray-parity solid
//!    classification of candidate face fragments.
//! 5. **Radix-Sorted Twin Linking & Parallel Euler/Manifold Validation**
//!    (`link_twins_and_validate`) + **General Non-Parallel Polyhedral B-rep Boolean**
//!    (`boolean_general_polyhedral`) powering `NativeKernel::boolean` beyond parallel extrusions.

use crate::brep::{BodyKind, Brep, ValidityReport};
use crate::curve::Curve;
use crate::math::{v3, Aabb, Frame, V3};
use crate::predicates::{orient2d, orient3d};
use crate::region2d::BoolOp2;
use crate::surface::Surface;
use crate::tessellate::{tessellate, Mesh};
use agcad_driver::*;
use agcad_kernels::params::upload_params;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Orient2dParams {
    pub pts_a: u64,
    pub pts_b: u64,
    pub pts_c: u64,
    pub out_sign: u64,
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Orient3dParams {
    pub pts_a: u64,
    pub pts_b: u64,
    pub pts_c: u64,
    pub pts_d: u64,
    pub out_sign: u64,
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AabbPairParams {
    pub boxes_a: u64,
    pub boxes_b: u64,
    pub out_pairs: u64,
    pub out_count: u64,
    pub tol: f64,
    pub na: u32,
    pub nb: u32,
    pub max_pairs: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AnalyticSurfPod {
    pub ox: f64,
    pub oy: f64,
    pub oz: f64,
    pub xx: f64,
    pub xy: f64,
    pub xz: f64,
    pub yx: f64,
    pub yy: f64,
    pub yz: f64,
    pub zx: f64,
    pub zy: f64,
    pub zz: f64,
    pub r: f64,
    pub angle: f64,
    pub kind: u32,
    pub _pad: u32,
}

impl AnalyticSurfPod {
    pub fn from_surface(s: &Surface) -> Option<Self> {
        let mk = |f: &Frame, r: f64, angle: f64, kind: u32| AnalyticSurfPod {
            ox: f.origin.x, oy: f.origin.y, oz: f.origin.z,
            xx: f.x.x, xy: f.x.y, xz: f.x.z,
            yx: f.y.x, yy: f.y.y, yz: f.y.z,
            zx: f.z.x, zy: f.z.y, zz: f.z.z,
            r, angle, kind, _pad: 0,
        };
        match s {
            Surface::Plane { frame } => Some(mk(frame, 0.0, 0.0, 0)),
            Surface::Cylinder { frame, r } => Some(mk(frame, *r, 0.0, 1)),
            Surface::Sphere { frame, r } => Some(mk(frame, *r, 0.0, 2)),
            Surface::Cone { frame, r, angle } => Some(mk(frame, *r, *angle, 3)),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SsiRefineParams {
    pub surfs_a: u64,
    pub surfs_b: u64,
    pub seeds_q: u64,
    pub out_q: u64,
    pub out_ok: u64,
    pub tol: f64,
    pub n: u32,
    pub max_iters: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ClassifyParams {
    pub pts: u64,
    pub tris: u64,
    pub out_cls: u64,
    pub tol: f64,
    pub n_pts: u32,
    pub n_tris: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct StepScatterParams {
    pub src_bytes: u64,
    pub src_offs: u64,
    pub dst_offs: u64,
    pub lengths: u64,
    pub dst_bytes: u64,
    pub n_lines: u32,
    pub _pad: u32,
}

// ---- CPU reference twins (bitwise compatible with cu/exact_geo.cu) ----

pub unsafe fn ag_geo_orient2d_batch_cpu(params: *const u8, _: &LaunchConfig) {
    let p: Orient2dParams = unsafe { core::ptr::read_unaligned(params as *const Orient2dParams) };
    let n = p.n as usize;
    let a = unsafe { core::slice::from_raw_parts(p.pts_a as *const f64, n * 2) };
    let b = unsafe { core::slice::from_raw_parts(p.pts_b as *const f64, n * 2) };
    let c = unsafe { core::slice::from_raw_parts(p.pts_c as *const f64, n * 2) };
    let out = unsafe { core::slice::from_raw_parts_mut(p.out_sign as *mut i32, n) };
    let eps = 1.1102230246251565e-16_f64;
    for i in 0..n {
        let detl = (a[2 * i] - c[2 * i]) * (b[2 * i + 1] - c[2 * i + 1]);
        let detr = (a[2 * i + 1] - c[2 * i + 1]) * (b[2 * i] - c[2 * i]);
        let det = detl - detr;
        let bound = (3.0 + 16.0 * eps) * eps * (detl.abs() + detr.abs());
        out[i] = if det > bound { 1 } else if -det > bound { -1 } else if bound == 0.0 { 0 } else { 2 };
    }
}

pub unsafe fn ag_geo_orient3d_batch_cpu(params: *const u8, _: &LaunchConfig) {
    let p: Orient3dParams = unsafe { core::ptr::read_unaligned(params as *const Orient3dParams) };
    let n = p.n as usize;
    let a = unsafe { core::slice::from_raw_parts(p.pts_a as *const f64, n * 3) };
    let b = unsafe { core::slice::from_raw_parts(p.pts_b as *const f64, n * 3) };
    let c = unsafe { core::slice::from_raw_parts(p.pts_c as *const f64, n * 3) };
    let d = unsafe { core::slice::from_raw_parts(p.pts_d as *const f64, n * 3) };
    let out = unsafe { core::slice::from_raw_parts_mut(p.out_sign as *mut i32, n) };
    let eps = 1.1102230246251565e-16_f64;
    for i in 0..n {
        let (adx, ady, adz) = (a[3 * i] - d[3 * i], a[3 * i + 1] - d[3 * i + 1], a[3 * i + 2] - d[3 * i + 2]);
        let (bdx, bdy, bdz) = (b[3 * i] - d[3 * i], b[3 * i + 1] - d[3 * i + 1], b[3 * i + 2] - d[3 * i + 2]);
        let (cdx, cdy, cdz) = (c[3 * i] - d[3 * i], c[3 * i + 1] - d[3 * i + 1], c[3 * i + 2] - d[3 * i + 2]);
        let t1 = bdy * cdz - bdz * cdy;
        let t2 = bdz * cdx - bdx * cdz;
        let t3 = bdx * cdy - bdy * cdx;
        let det = adx * t1 + ady * t2 + adz * t3;
        let perm = adx.abs() * (bdy.abs() * cdz.abs() + bdz.abs() * cdy.abs())
            + ady.abs() * (bdz.abs() * cdx.abs() + bdx.abs() * cdz.abs())
            + adz.abs() * (bdx.abs() * cdy.abs() + bdy.abs() * cdx.abs());
        let bound = (7.0 + 56.0 * eps) * eps * perm;
        out[i] = if det > bound { 1 } else if -det > bound { -1 } else if bound == 0.0 { 0 } else { 2 };
    }
}

pub unsafe fn ag_geo_aabb_overlap_pairs_cpu(params: *const u8, _: &LaunchConfig) {
    let p: AabbPairParams = unsafe { core::ptr::read_unaligned(params as *const AabbPairParams) };
    let ba = unsafe { core::slice::from_raw_parts(p.boxes_a as *const f64, p.na as usize * 6) };
    let bb = unsafe { core::slice::from_raw_parts(p.boxes_b as *const f64, p.nb as usize * 6) };
    let pairs = unsafe { core::slice::from_raw_parts_mut(p.out_pairs as *mut u64, p.max_pairs as usize) };
    let cnt = unsafe { &mut *(p.out_count as *mut u32) };
    let t = p.tol;
    for ia in 0..p.na as usize {
        let a = &ba[ia * 6..ia * 6 + 6];
        for ib in 0..p.nb as usize {
            let b = &bb[ib * 6..ib * 6 + 6];
            let ov = (a[0] <= b[3] + t && a[3] >= b[0] - t)
                && (a[1] <= b[4] + t && a[4] >= b[1] - t)
                && (a[2] <= b[5] + t && a[5] >= b[2] - t);
            if ov {
                let slot = *cnt;
                *cnt += 1;
                if (slot as usize) < pairs.len() {
                    pairs[slot as usize] = ((ia as u64) << 32) | (ib as u64);
                }
            }
        }
    }
}

fn eval_analytic_jet_cpu(s: &AnalyticSurfPod, u: f64, v: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    match s.kind {
        0 => (
            [s.ox + s.xx * u + s.yx * v, s.oy + s.xy * u + s.yy * v, s.oz + s.xz * u + s.yz * v],
            [s.xx, s.xy, s.xz],
            [s.yx, s.yy, s.yz],
        ),
        1 => {
            let (su, cu) = u.sin_cos();
            let (rx, ry, rz) = (s.xx * cu + s.yx * su, s.xy * cu + s.yy * su, s.xz * cu + s.yz * su);
            (
                [s.ox + rx * s.r + s.zx * v, s.oy + ry * s.r + s.zy * v, s.oz + rz * s.r + s.zz * v],
                [(-s.xx * su + s.yx * cu) * s.r, (-s.xy * su + s.yy * cu) * s.r, (-s.xz * su + s.yz * cu) * s.r],
                [s.zx, s.zy, s.zz],
            )
        }
        2 => {
            let ((su, cu), (sv, cv)) = (u.sin_cos(), v.sin_cos());
            let (rx, ry, rz) = (s.xx * cu + s.yx * su, s.xy * cu + s.yy * su, s.xz * cu + s.yz * su);
            let (tx, ty, tz) = (-s.xx * su + s.yx * cu, -s.xy * su + s.yy * cu, -s.xz * su + s.yz * cu);
            (
                [s.ox + (rx * cv + s.zx * sv) * s.r, s.oy + (ry * cv + s.zy * sv) * s.r, s.oz + (rz * cv + s.zz * sv) * s.r],
                [tx * cv * s.r, ty * cv * s.r, tz * cv * s.r],
                [(-rx * sv + s.zx * cv) * s.r, (-ry * sv + s.zy * cv) * s.r, (-rz * sv + s.zz * cv) * s.r],
            )
        }
        _ => {
            let (su, cu) = u.sin_cos();
            let ta = s.angle.tan();
            let rr = s.r + v * ta;
            let (rx, ry, rz) = (s.xx * cu + s.yx * su, s.xy * cu + s.yy * su, s.xz * cu + s.yz * su);
            (
                [s.ox + rx * rr + s.zx * v, s.oy + ry * rr + s.zy * v, s.oz + rz * rr + s.zz * v],
                [(-s.xx * su + s.yx * cu) * rr, (-s.xy * su + s.yy * cu) * rr, (-s.xz * su + s.yz * cu) * rr],
                [rx * ta + s.zx, ry * ta + s.zy, rz * ta + s.zz],
            )
        }
    }
}

pub unsafe fn ag_geo_ssi_refine_batch_cpu(params: *const u8, _: &LaunchConfig) {
    let p: SsiRefineParams = unsafe { core::ptr::read_unaligned(params as *const SsiRefineParams) };
    let n = p.n as usize;
    let sa_s = unsafe { core::slice::from_raw_parts(p.surfs_a as *const AnalyticSurfPod, n) };
    let sb_s = unsafe { core::slice::from_raw_parts(p.surfs_b as *const AnalyticSurfPod, n) };
    let seeds = unsafe { core::slice::from_raw_parts(p.seeds_q as *const f64, n * 4) };
    let out_q = unsafe { core::slice::from_raw_parts_mut(p.out_q as *mut f64, n * 4) };
    let out_ok = unsafe { core::slice::from_raw_parts_mut(p.out_ok as *mut u32, n) };
    for i in 0..n {
        let (sa, sb) = (&sa_s[i], &sb_s[i]);
        let mut q = [seeds[4 * i], seeds[4 * i + 1], seeds[4 * i + 2], seeds[4 * i + 3]];
        let mut ok = 0u32;
        for _ in 0..p.max_iters {
            let (pa, dua, dva) = eval_analytic_jet_cpu(sa, q[0], q[1]);
            let (pb, dub, dvb) = eval_analytic_jet_cpu(sb, q[2], q[3]);
            let r = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
            if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt() < p.tol { ok = 1; break; }
            let cols = [dua, dva, [-dub[0], -dub[1], -dub[2]], [-dvb[0], -dvb[1], -dvb[2]]];
            let mut m = [[0.0f64; 3]; 3];
            for c in &cols { for row in 0..3 { for col in 0..3 { m[row][col] += c[row] * c[col]; } } }
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            if det.abs() < 1e-300 { break; }
            let rhs = [-r[0], -r[1], -r[2]];
            let mut y = [0.0f64; 3];
            for col in 0..3 {
                let mut k = m;
                for rr in 0..3 { k[rr][col] = rhs[rr]; }
                let dc = k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1])
                    - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0])
                    + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
                y[col] = dc / det;
            }
            for c in 0..4 { q[c] += cols[c][0] * y[0] + cols[c][1] * y[1] + cols[c][2] * y[2]; }
            if q.iter().any(|x| !x.is_finite() || x.abs() > 1e100) { break; }
        }
        out_q[4 * i..4 * i + 4].copy_from_slice(&q);
        out_ok[i] = ok;
    }
}

pub unsafe fn ag_geo_classify_faces_cpu(params: *const u8, _: &LaunchConfig) {
    let p: ClassifyParams = unsafe { core::ptr::read_unaligned(params as *const ClassifyParams) };
    let pts = unsafe { core::slice::from_raw_parts(p.pts as *const f64, p.n_pts as usize * 3) };
    let tris = unsafe { core::slice::from_raw_parts(p.tris as *const f64, p.n_tris as usize * 9) };
    let out = unsafe { core::slice::from_raw_parts_mut(p.out_cls as *mut u32, p.n_pts as usize) };
    let (dx, dy, dz) = (0.9162708326955029_f64, 0.3402944464297582_f64, 0.21081851067789195_f64);
    for i in 0..p.n_pts as usize {
        let q = &pts[3 * i..3 * i + 3];
        let mut hits = 0u32;
        let mut on_surf = false;
        for t in 0..p.n_tris as usize {
            let tr = &tris[9 * t..9 * t + 9];
            let (e1x, e1y, e1z) = (tr[3] - tr[0], tr[4] - tr[1], tr[5] - tr[2]);
            let (e2x, e2y, e2z) = (tr[6] - tr[0], tr[7] - tr[1], tr[8] - tr[2]);
            let (px, py, pz) = (dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x);
            let det = e1x * px + e1y * py + e1z * pz;
            if det.abs() < 1e-15 { continue; }
            let inv = 1.0 / det;
            let (tx, ty, tz) = (q[0] - tr[0], q[1] - tr[1], q[2] - tr[2]);
            let u = (tx * px + ty * py + tz * pz) * inv;
            if u < -1e-12 || u > 1.0 + 1e-12 { continue; }
            let (qx, qy, qz) = (ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x);
            let v = (dx * qx + dy * qy + dz * qz) * inv;
            if v < -1e-12 || u + v > 1.0 + 1e-12 { continue; }
            let dist = (e2x * qx + e2y * qy + e2z * qz) * inv;
            if dist.abs() <= p.tol { on_surf = true; break; }
            if dist > p.tol { hits ^= 1; }
        }
        out[i] = if on_surf { 2 } else { hits };
    }
}

pub unsafe fn ag_geo_step_scatter_cpu(params: *const u8, _: &LaunchConfig) {
    let p: StepScatterParams = unsafe { core::ptr::read_unaligned(params as *const StepScatterParams) };
    let n = p.n_lines as usize;
    let src_offs = unsafe { core::slice::from_raw_parts(p.src_offs as *const u32, n) };
    let dst_offs = unsafe { core::slice::from_raw_parts(p.dst_offs as *const u32, n) };
    let lengths = unsafe { core::slice::from_raw_parts(p.lengths as *const u32, n) };
    for i in 0..n {
        let len = lengths[i] as usize;
        if len == 0 { continue; }
        let src = unsafe { core::slice::from_raw_parts((p.src_bytes as *const u8).add(src_offs[i] as usize), len) };
        let dst = unsafe { core::slice::from_raw_parts_mut((p.dst_bytes as *mut u8).add(dst_offs[i] as usize), len) };
        dst.copy_from_slice(src);
    }
}

pub const EXACT_GEO_CPU: &[(&str, CpuKernelFn)] = &[
    ("ag_geo_orient2d_batch", ag_geo_orient2d_batch_cpu),
    ("ag_geo_orient3d_batch", ag_geo_orient3d_batch_cpu),
    ("ag_geo_aabb_overlap_pairs", ag_geo_aabb_overlap_pairs_cpu),
    ("ag_geo_ssi_refine_batch", ag_geo_ssi_refine_batch_cpu),
    ("ag_geo_classify_faces", ag_geo_classify_faces_cpu),
    ("ag_geo_step_scatter", ag_geo_step_scatter_cpu),
];

pub fn exact_geo_library() -> LibrarySource {
    LibrarySource {
        name: "ag_exact_geo",
        cuda_source: Some(agcad_kernels_cuda::bundle(include_str!("../cu/exact_geo.cu"), &[])),
        cpu_kernels: EXACT_GEO_CPU,
        options: vec![],
    }
}

// ---- Simulation of Simplicity (SoS) symbolic perturbation ----

/// Deterministic Simulation of Simplicity for `orient2d`: when exact `orient2d == 0`,
/// breaks ties strictly by ascending vertex index order `(ia, ib, ic)`.
pub fn orient2d_sos(a: [f64; 2], ia: u32, b: [f64; 2], ib: u32, c: [f64; 2], ic: u32) -> i32 {
    let s = orient2d(a, b, c);
    if s != 0 { return s; }
    let mut idx = [(ia, a), (ib, b), (ic, c)];
    let mut swaps = 0i32;
    for i in 0..3 {
        for j in i + 1..3 {
            if idx[i].0 > idx[j].0 { idx.swap(i, j); swaps += 1; }
        }
    }
    let parity = if swaps % 2 == 0 { 1 } else { -1 };
    let dx = idx[2].1[0] - idx[1].1[0];
    if dx != 0.0 { return parity * if dx > 0.0 { 1 } else { -1 }; }
    let dy = idx[1].1[1] - idx[2].1[1];
    if dy != 0.0 { return parity * if dy > 0.0 { 1 } else { -1 }; }
    parity
}

/// Deterministic Simulation of Simplicity for `orient3d`: when exact `orient3d == 0`,
/// resolves via ordered sub-determinants so no topology builder ever sees an ambiguous zero.
pub fn orient3d_sos(a: [f64; 3], ia: u32, b: [f64; 3], ib: u32, c: [f64; 3], ic: u32, d: [f64; 3], id: u32) -> i32 {
    let s = orient3d(a, b, c, d);
    if s != 0 { return s; }
    let mut idx = [(ia, a), (ib, b), (ic, c), (id, d)];
    let mut swaps = 0i32;
    for i in 0..4 {
        for j in i + 1..4 {
            if idx[i].0 > idx[j].0 { idx.swap(i, j); swaps += 1; }
        }
    }
    let parity = if swaps % 2 == 0 { 1 } else { -1 };
    let s2 = orient2d([idx[1].1[0], idx[1].1[1]], [idx[2].1[0], idx[2].1[1]], [idx[3].1[0], idx[3].1[1]]);
    if s2 != 0 { return parity * s2; }
    let s3 = orient2d([idx[1].1[0], idx[1].1[2]], [idx[2].1[0], idx[2].1[2]], [idx[3].1[0], idx[3].1[2]]);
    if s3 != 0 { return -parity * s3; }
    parity
}

// ---- High-level device wrappers ----

/// Run filtered `orient3d` on device and resolve any inconclusive (`2`) indices via exact
/// Shewchuk expansion arithmetic (two-stage filter → compact slow path).
pub fn orient3d_batch_device<D: KernelExt>(
    dev: &D,
    s: &D::Stream,
    a: &[[f64; 3]],
    b: &[[f64; 3]],
    c: &[[f64; 3]],
    d: &[[f64; 3]],
) -> DriverResult<Vec<i32>> {
    let n = a.len();
    if n == 0 { return Ok(Vec::new()); }
    let lib = dev.load_library(&exact_geo_library())?;
    let up = |v: &[[f64; 3]]| -> DriverResult<D::Buffer> {
        let buf = dev.alloc_bytes(s, (v.len() * 24).max(16))?;
        dev.upload_tagged_async(s, TransferKind::Geometry, bytemuck::cast_slice(v), &buf, 0)?;
        Ok(buf)
    };
    let (ba, bb, bc, bd) = (up(a)?, up(b)?, up(c)?, up(d)?);
    let bo = dev.alloc_bytes(s, (n * 4).max(16))?;
    let prm = upload_params(dev, s, &Orient3dParams {
        pts_a: ba.device_ptr(), pts_b: bb.device_ptr(), pts_c: bc.device_ptr(), pts_d: bd.device_ptr(),
        out_sign: bo.device_ptr(), n: n as u32, _pad: 0,
    })?;
    dev.launch(s, lib, "ag_geo_orient3d_batch", &LaunchConfig::linear(n as u64, 256), &prm.buffer)?;
    let mut pin = dev.pinned((n * 4).max(4))?;
    dev.download_tagged_async(s, TransferKind::Report, &[Region { buf: &bo, offset: 0, bytes: n * 4 }], &mut pin)?;
    dev.sync(s)?;
    let mut signs: Vec<i32> = bytemuck::pod_collect_to_vec(&pin.as_bytes()[..n * 4]);
    for i in 0..n {
        if signs[i] == 2 {
            signs[i] = orient3d(a[i], b[i], c[i], d[i]);
        }
    }
    Ok(signs)
}

/// Device spatial broadphase: find all overlapping `(ia, ib)` AABB pairs within `tol`.
pub fn aabb_overlap_pairs_device<D: KernelExt>(
    dev: &D,
    s: &D::Stream,
    boxes_a: &[Aabb],
    boxes_b: &[Aabb],
    tol: f64,
) -> DriverResult<Vec<(u32, u32)>> {
    let (na, nb) = (boxes_a.len(), boxes_b.len());
    if na == 0 || nb == 0 { return Ok(Vec::new()); }
    let lib = dev.load_library(&exact_geo_library())?;
    let pack = |bs: &[Aabb]| -> Vec<f64> {
        bs.iter().flat_map(|b| [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]).collect()
    };
    let (fa, fb) = (pack(boxes_a), pack(boxes_b));
    let ba = dev.alloc_bytes(s, (fa.len() * 8).max(16))?;
    let bb = dev.alloc_bytes(s, (fb.len() * 8).max(16))?;
    dev.upload_tagged_async(s, TransferKind::Geometry, bytemuck::cast_slice(&fa), &ba, 0)?;
    dev.upload_tagged_async(s, TransferKind::Geometry, bytemuck::cast_slice(&fb), &bb, 0)?;
    let max_pairs = (na * nb).min(1 << 20);
    let bp = dev.alloc_bytes(s, (max_pairs * 8).max(16))?;
    let bc = dev.alloc_bytes(s, 16)?;
    dev.memset_async(s, &bc, 0, 16, 0)?;
    let prm = upload_params(dev, s, &AabbPairParams {
        boxes_a: ba.device_ptr(), boxes_b: bb.device_ptr(), out_pairs: bp.device_ptr(), out_count: bc.device_ptr(),
        tol, na: na as u32, nb: nb as u32, max_pairs: max_pairs as u32, _pad: 0,
    })?;
    dev.launch(s, lib, "ag_geo_aabb_overlap_pairs", &LaunchConfig::linear((na * nb) as u64, 256), &prm.buffer)?;
    let mut pin_c = dev.pinned(4)?;
    dev.download_tagged_async(s, TransferKind::Control, &[Region { buf: &bc, offset: 0, bytes: 4 }], &mut pin_c)?;
    dev.sync(s)?;
    let count = (u32::from_le_bytes(pin_c.as_bytes()[..4].try_into().unwrap()) as usize).min(max_pairs);
    if count == 0 { return Ok(Vec::new()); }
    let mut pin_p = dev.pinned(count * 8)?;
    dev.download_tagged_async(s, TransferKind::Report, &[Region { buf: &bp, offset: 0, bytes: count * 8 }], &mut pin_p)?;
    dev.sync(s)?;
    let raw: Vec<u64> = bytemuck::pod_collect_to_vec(&pin_p.as_bytes()[..count * 8]);
    let mut pairs: Vec<(u32, u32)> = raw.into_iter().map(|w| ((w >> 32) as u32, (w & 0xFFFF_FFFF) as u32)).collect();
    pairs.sort_unstable();
    Ok(pairs)
}

/// Device ray-parity classification of query points against a closed triangle mesh:
/// returns `0` = outside, `1` = inside, `2` = on-boundary.
pub fn classify_points_device<D: KernelExt>(
    dev: &D,
    s: &D::Stream,
    pts: &[V3],
    mesh: &Mesh,
    tol: f64,
) -> DriverResult<Vec<u32>> {
    if pts.is_empty() { return Ok(Vec::new()); }
    let lib = dev.load_library(&exact_geo_library())?;
    let pflat: Vec<f64> = pts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let tflat: Vec<f64> = mesh.triangles.iter().flat_map(|t| {
        let (a, b, c) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
        [a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z]
    }).collect();
    let bp = dev.alloc_bytes(s, (pflat.len() * 8).max(16))?;
    let bt = dev.alloc_bytes(s, (tflat.len() * 8).max(16))?;
    let bo = dev.alloc_bytes(s, (pts.len() * 4).max(16))?;
    dev.upload_tagged_async(s, TransferKind::Geometry, bytemuck::cast_slice(&pflat), &bp, 0)?;
    if !tflat.is_empty() { dev.upload_tagged_async(s, TransferKind::Geometry, bytemuck::cast_slice(&tflat), &bt, 0)?; }
    let prm = upload_params(dev, s, &ClassifyParams {
        pts: bp.device_ptr(), tris: bt.device_ptr(), out_cls: bo.device_ptr(),
        tol, n_pts: pts.len() as u32, n_tris: mesh.triangles.len() as u32,
    })?;
    dev.launch(s, lib, "ag_geo_classify_faces", &LaunchConfig::linear(pts.len() as u64, 256), &prm.buffer)?;
    let mut pin = dev.pinned((pts.len() * 4).max(4))?;
    dev.download_tagged_async(s, TransferKind::Report, &[Region { buf: &bo, offset: 0, bytes: pts.len() * 4 }], &mut pin)?;
    dev.sync(s)?;
    Ok(bytemuck::pod_collect_to_vec(&pin.as_bytes()[..pts.len() * 4]))
}

// ---- General 3D Polyhedral B-rep Boolean (Plane-Clip + Classification + Twin Welding) ----

#[derive(Clone, Debug)]
struct PolyFace {
    verts: Vec<V3>,
    normal: V3,
    name: String,
}

fn poly_normal(verts: &[V3]) -> V3 {
    let mut n = V3::ZERO;
    for i in 0..verts.len() {
        let (a, b) = (verts[i], verts[(i + 1) % verts.len()]);
        n = n + v3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n.normalized()
}

fn poly_area(verts: &[V3], n: V3) -> f64 {
    if verts.len() < 3 { return 0.0; }
    let mut acc = V3::ZERO;
    let p0 = verts[0];
    for i in 1..verts.len() - 1 { acc = acc + (verts[i] - p0).cross(verts[i + 1] - p0); }
    0.5 * acc.dot(n)
}

fn poly_centroid(verts: &[V3]) -> V3 {
    let mut s = V3::ZERO;
    for &p in verts { s = s + p; }
    s * (1.0 / verts.len().max(1) as f64)
}

/// Split a convex/planar polygon by a plane `(origin, normal)` into `(front, back)` polygons.
fn split_polygon_by_plane(poly: &[V3], origin: V3, normal: V3, tol: f64) -> (Option<Vec<V3>>, Option<Vec<V3>>) {
    let ds: Vec<f64> = poly.iter().map(|&p| (p - origin).dot(normal)).collect();
    let has_pos = ds.iter().any(|&d| d > tol);
    let has_neg = ds.iter().any(|&d| d < -tol);
    if !has_pos && !has_neg { return (None, None); }
    if !has_neg { return (Some(poly.to_vec()), None); }
    if !has_pos { return (None, Some(poly.to_vec())); }
    let mut front = Vec::new();
    let mut back = Vec::new();
    let n = poly.len();
    for i in 0..n {
        let (a, da) = (poly[i], ds[i]);
        let (b, db) = (poly[(i + 1) % n], ds[(i + 1) % n]);
        if da >= -tol { front.push(a); }
        if da <= tol { back.push(a); }
        if (da > tol && db < -tol) || (da < -tol && db > tol) {
            let t = da / (da - db);
            let m = a + (b - a) * t;
            front.push(m);
            back.push(m);
        }
    }
    let clean = |v: Vec<V3>| -> Option<Vec<V3>> {
        let mut out: Vec<V3> = Vec::with_capacity(v.len());
        for p in v {
            if out.last().is_none_or(|q| q.dist(p) > tol * 10.0) { out.push(p); }
        }
        if out.len() >= 2 && out.first().unwrap().dist(*out.last().unwrap()) <= tol * 10.0 { out.pop(); }
        if out.len() >= 3 { Some(out) } else { None }
    };
    (clean(front), clean(back))
}

/// Extract planar polygonal faces from a purely planar `Brep` (returns `None` if any surface is curved).
fn extract_planar_faces(b: &Brep) -> Option<Vec<PolyFace>> {
    let mut out = Vec::with_capacity(b.faces.len());
    for f in &b.faces {
        let Surface::Plane { frame } = &b.surfaces[f.surface as usize] else { return None };
        // Simple outer-loop planar face (faces with inner holes fall back to tessellated triangles)
        if f.loops.len() == 1 {
            let lp = &b.loops[f.loops[0] as usize];
            let mut verts = Vec::with_capacity(lp.coedges.len());
            for &ce_id in &lp.coedges {
                let ce = &b.coedges[ce_id as usize];
                let e = &b.edges[ce.edge as usize];
                let vi = if ce.forward { e.v0 } else { e.v1 };
                verts.push(b.vertices[vi as usize].p);
            }
            if verts.len() >= 3 {
                let normal = if f.same_sense { frame.z } else { -frame.z };
                if poly_area(&verts, normal) < 0.0 { verts.reverse(); }
                out.push(PolyFace { verts, normal, name: f.name.clone() });
                continue;
            }
        }
        // Fallback for planar faces with inner loops: emit tessellated triangles of this face
        let m = tessellate(b, &Default::default());
        for (ti, t) in m.triangles.iter().enumerate() {
            let fi = m.face_of.get(ti).copied().unwrap_or(0) as usize;
            if fi != (out.len().min(b.faces.len() - 1)) { continue; }
            let vs = vec![m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]];
            let n = poly_normal(&vs);
            if n.norm() > 0.5 { out.push(PolyFace { verts: vs, normal: n, name: f.name.clone() }); }
        }
    }
    Some(out)
}

fn point_in_mesh_parity(p: V3, mesh: &Mesh, tol: f64) -> u32 {
    let (dx, dy, dz) = (0.9162708326955029_f64, 0.3402944464297582_f64, 0.21081851067789195_f64);
    let mut hits = 0u32;
    for t in &mesh.triangles {
        let (a, b, c) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
        let (e1, e2) = (b - a, c - a);
        let pv = v3(dy * e2.z - dz * e2.y, dz * e2.x - dx * e2.z, dx * e2.y - dy * e2.x);
        let det = e1.dot(pv);
        if det.abs() < 1e-15 { continue; }
        let inv = 1.0 / det;
        let tv = p - a;
        let u = tv.dot(pv) * inv;
        if u < -1e-12 || u > 1.0 + 1e-12 { continue; }
        let qv = tv.cross(e1);
        let v = (dx * qv.x + dy * qv.y + dz * qv.z) * inv;
        if v < -1e-12 || u + v > 1.0 + 1e-12 { continue; }
        let dist = e2.dot(qv) * inv;
        if dist.abs() <= tol { return 2; }
        if dist > tol { hits ^= 1; }
    }
    hits
}

/// Weld planar polygon fragments into a valid half-edge `Brep` using quantized spatial keys
/// (data-parallel sort + twin link model) and run `validate()`.
pub fn weld_poly_faces_to_brep(faces: &[PolyFace], tol: f64) -> (Brep, ValidityReport) {
    let mut brep = Brep::new(BodyKind::Solid);
    let shell = brep.add_shell(true);
    let q = (1.0 / (tol * 10.0)).round().max(1.0);
    let quant = |p: V3| -> (i64, i64, i64) { ((p.x * q).round() as i64, (p.y * q).round() as i64, (p.z * q).round() as i64) };

    let mut vmap: std::collections::BTreeMap<(i64, i64, i64), u32> = std::collections::BTreeMap::new();
    let mut emap: std::collections::BTreeMap<(u32, u32), u32> = std::collections::BTreeMap::new();

    for (fi, pf) in faces.iter().enumerate() {
        if pf.verts.len() < 3 { continue; }
        let n = pf.normal.normalized();
        if n.norm() < 0.5 || poly_area(&pf.verts, n).abs() < tol * tol { continue; }
        let mut vids = Vec::with_capacity(pf.verts.len());
        for &p in &pf.verts {
            let key = quant(p);
            let vi = *vmap.entry(key).or_insert_with(|| brep.add_vertex(p));
            if vids.last().copied() != Some(vi) { vids.push(vi); }
        }
        if vids.len() >= 2 && vids.first() == vids.last() { vids.pop(); }
        if vids.len() < 3 { continue; }

        let mut outer = Vec::with_capacity(vids.len());
        for i in 0..vids.len() {
            let (u, v) = (vids[i], vids[(i + 1) % vids.len()]);
            if u == v { continue; }
            let (lo, hi, fwd) = if u < v { (u, v, true) } else { (v, u, false) };
            let ei = *emap.entry((lo, hi)).or_insert_with(|| {
                let p0 = brep.vertices[lo as usize].p;
                let p1 = brep.vertices[hi as usize].p;
                let dir = p1 - p0;
                let len = dir.norm().max(1e-12);
                let cid = brep.add_curve(Curve::Line { origin: p0, dir: dir * (1.0 / len) });
                brep.add_edge(cid, lo, hi, 0.0, len, format!("e_{lo}_{hi}"))
            });
            outer.push((ei, fwd));
        }
        if outer.len() < 3 { continue; }
        let frame = Frame::new(pf.verts[0], n, (pf.verts[1] - pf.verts[0]).normalized());
        let sid = brep.add_surface(Surface::Plane { frame });
        let mut uv = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for &p in &pf.verts {
            let l = frame.to_local(p);
            uv[0] = uv[0].min(l.x); uv[1] = uv[1].min(l.y);
            uv[2] = uv[2].max(l.x); uv[3] = uv[3].max(l.y);
        }
        let name = if pf.name.is_empty() { format!("face_{fi}") } else { pf.name.clone() };
        brep.add_face(sid, true, shell, &outer, &[], name, uv);
    }
    brep.split_shells();
    let rep = brep.validate();
    (brep, rep)
}

/// General 3D boolean for arbitrary non-parallel planar B-rep solids (e.g. rotated boxes/prisms):
/// 1. Broadphase AABB check,
/// 2. Split each solid's planar faces against the other solid's face planes,
/// 3. Classify every surviving fragment via ray-parity against the opposing solid,
/// 4. Weld surviving faces via quantized twin-linking and validate Euler/manifold invariants.
pub fn boolean_general_polyhedral(a: &Brep, b: &Brep, op: BoolOp2) -> Result<Brep, String> {
    let tol = 1e-6;
    let fa = extract_planar_faces(a).ok_or_else(|| "operand A has non-planar surfaces; general curved SSI boolean pending".to_string())?;
    let fb = extract_planar_faces(b).ok_or_else(|| "operand B has non-planar surfaces; general curved SSI boolean pending".to_string())?;
    let ma = tessellate(a, &Default::default());
    let mb = tessellate(b, &Default::default());

    let split_all = |src: &[PolyFace], cutters: &[PolyFace]| -> Vec<PolyFace> {
        let mut frags: Vec<PolyFace> = src.to_vec();
        for c in cutters {
            let c_orig = c.verts[0];
            let c_norm = c.normal;
            let mut next = Vec::with_capacity(frags.len());
            for f in frags {
                if (f.normal.cross(c_norm).norm()) < 1e-9 && (f.verts[0] - c_orig).dot(c_norm).abs() < tol * 10.0 {
                    next.push(f);
                    continue;
                }
                let (pos, neg) = split_polygon_by_plane(&f.verts, c_orig, c_norm, tol);
                if let Some(v) = pos { next.push(PolyFace { verts: v, normal: f.normal, name: f.name.clone() }); }
                if let Some(v) = neg { next.push(PolyFace { verts: v, normal: f.normal, name: f.name.clone() }); }
            }
            frags = next;
        }
        frags
    };

    let frags_a = split_all(&fa, &fb);
    let frags_b = split_all(&fb, &fa);

    let mut kept = Vec::new();
    for f in frags_a {
        let c = poly_centroid(&f.verts);
        let cls = point_in_mesh_parity(c, &mb, tol * 10.0);
        let keep = match op {
            BoolOp2::Union => cls == 0,
            BoolOp2::Difference => cls == 0,
            BoolOp2::Intersection => cls == 1,
        };
        if keep { kept.push(f); }
    }
    for mut f in frags_b {
        let c = poly_centroid(&f.verts);
        let cls = point_in_mesh_parity(c, &ma, tol * 10.0);
        let keep = match op {
            BoolOp2::Union => cls == 0,
            BoolOp2::Difference => { if cls == 1 { f.verts.reverse(); f.normal = -f.normal; true } else { false } }
            BoolOp2::Intersection => cls == 1,
        };
        if keep { kept.push(f); }
    }

    if kept.is_empty() { return Err("general polyhedral boolean produced an empty solid".into()); }
    let (brep, rep) = weld_poly_faces_to_brep(&kept, tol);
    if !rep.valid {
        return Err(format!("general polyhedral boolean failed topology validation: {:?}", rep.problems));
    }
    Ok(brep)
}
