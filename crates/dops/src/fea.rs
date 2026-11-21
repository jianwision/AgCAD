//! Device stiffness proxy (spec 14 escalation, evidence `sampled`): linear elasticity on a
//! voxelization of a device design (8-node hexahedra, 2×2×2 Gauss), solved matrix-free with
//! Jacobi-preconditioned CG entirely on the device.
//!
//! Determinism: the matvec is a per-DOF gather over the adjacent elements in ascending element
//! order (the same summation order as `agcad_std::physics::solve_static`); dot products use a
//! fixed-shape reduction (`DOT_BLOCKS` × `DOT_T` strided partial sums, shared-memory trees) that
//! the CPU reference reproduces exactly, so CUDA ≡ CPU bitwise.
//!
//! Load case (as `std::physics`): nodes on the `support_axis` min face fixed, total force `f`
//! spread uniformly over the used nodes of the max face.

use crate::abi::*;
use crate::math::compile_node;
use crate::studio::{dops_library, Design};
use agcad_driver::*;
use agcad_kernels::params::upload_params;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

pub const DOT_BLOCKS: usize = 256;
pub const DOT_T: usize = 256;

/// Scalar slots of the solver state (f64).
pub mod sc {
    pub const RZ: usize = 0;
    pub const RZN: usize = 1;
    pub const PAP: usize = 2;
    pub const RR: usize = 3;
    pub const BN: usize = 4;
    pub const DONE: usize = 5;
    pub const ITER: usize = 6;
    pub const COMPLIANCE: usize = 7;
    pub const CONVERGED: usize = 8;
    pub const BB: usize = 9;
    pub const N: usize = 16;
}

/// Node flags.
pub const N_USED: u32 = 1;
pub const N_FIXED: u32 = 2;
pub const N_LOADED: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FeaArgs {
    pub tape: u64,
    pub tables: u64,
    pub solid: u64,
    pub nflag: u64,
    pub ke: u64,
    pub x: u64,
    pub r: u64,
    pub z: u64,
    pub p: u64,
    pub ap: u64,
    pub b: u64,
    pub diag: u64,
    pub scal: u64,
    pub stat: u64,
    pub lo: [f64; 3],
    pub h: f64,
    pub force: [f64; 3],
    pub tol: f64,
    pub count: u32,
    pub dims: [u32; 3],
    pub axis: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DotArgs {
    pub a0: u64,
    pub b0: u64,
    pub a1: u64,
    pub b1: u64,
    pub part: u64,
    pub scal: u64,
    pub n: u64,
    pub slot0: u32,
    pub slot1: u32,
    /// 1 or 2 dot products.
    pub ndots: u32,
    /// Run even when the solver is done (setup and final reductions).
    pub always: u32,
}

/// 24×24 stiffness matrix of a cube element of edge `h` (E, ν), standard Q8 trilinear
/// (identical to `agcad_std::physics::hex8_stiffness`).
pub fn hex8_stiffness(e: f64, nu: f64, h: f64) -> Vec<f64> {
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let corners = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]];
    let g = 1.0 / 3f64.sqrt();
    let mut k = vec![0.0; 24 * 24];
    for gx in [-g, g] { for gy in [-g, g] { for gz in [-g, g] {
        let mut dn = [[0.0f64; 3]; 8];
        for (a, c) in corners.iter().enumerate() {
            dn[a][0] = 0.125 * c[0] * (1.0 + c[1] * gy) * (1.0 + c[2] * gz) * 2.0 / h;
            dn[a][1] = 0.125 * c[1] * (1.0 + c[0] * gx) * (1.0 + c[2] * gz) * 2.0 / h;
            dn[a][2] = 0.125 * c[2] * (1.0 + c[0] * gx) * (1.0 + c[1] * gy) * 2.0 / h;
        }
        let det_j = (h / 2.0).powi(3);
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

/// Load case and discretization.
#[derive(Clone, Copy, Debug)]
pub struct LoadCase {
    /// Voxel box minimum corner and edge length (mm); `dims` voxels per axis.
    pub lo: [f64; 3],
    pub h: f64,
    pub dims: [u32; 3],
    pub support_axis: u32,
    pub force: [f64; 3],
    pub e: f64,
    pub nu: f64,
    pub tol: f64,
    pub max_iter: u32,
}

impl LoadCase {
    /// Voxel grid covering the box [lo, hi] with edge `h`.
    pub fn over(lo: [f64; 3], hi: [f64; 3], h: f64) -> LoadCase {
        let dims = [0, 1, 2].map(|k| (((hi[k] - lo[k]) / h).ceil() as u32).max(1));
        LoadCase { lo, h, dims, support_axis: 2, force: [0.0, 0.0, -1000.0], e: 210_000.0, nu: 0.3, tol: 1e-8, max_iter: 5000 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeaReport {
    /// fᵀu (N·mm with E in MPa, mm, N).
    pub compliance: f64,
    pub max_disp: f64,
    pub iterations: u32,
    pub converged: bool,
    pub voxels: u64,
    pub dofs: u64,
}

/// Stiffness of one design under `lc`, solved on `dev`.
pub fn stiffness<D: KernelExt>(dev: Arc<D>, design: &Design, lc: &LoadCase) -> DriverResult<FeaReport> {
    let d = &*dev;
    let s = d.stream(Priority::Low)?;
    let lib = d.load_library(&dops_library())?;
    let nv = lc.dims.iter().map(|&n| n as usize).product::<usize>();
    let nn = lc.dims.iter().map(|&n| n as usize + 1).product::<usize>();
    let nd = 3 * nn;
    if nv > 4_000_000 { return Err(EnqueueError::InvalidConfig(format!("FEA grid {:?} too large", lc.dims)).into()); }
    // tape compiled on the host (the CPU reference compile; the device compile is bit-identical)
    let mut cn = Vec::with_capacity(design.nodes.len());
    let mut tabs = vec![0.0f32; design.nodes.len().max(1) * TABLE];
    let prm: Vec<f64> = if design.params.is_empty() { vec![0.0] } else { design.params.clone() };
    for (i, n) in design.nodes.iter().enumerate() { cn.push(compile_node(n, &prm, &mut tabs[i * TABLE..(i + 1) * TABLE])); }
    if cn.is_empty() { cn.push(CNode::zeroed()); }
    let up = |bytes: &[u8]| -> DriverResult<D::Buffer> { let b = d.alloc_bytes(&s, bytes.len().max(16))?; d.upload_from_host_async(&s, bytes, &b, 0)?; Ok(b) };
    let zeros = |bytes: usize| -> DriverResult<D::Buffer> { let b = d.alloc_bytes(&s, bytes.max(16))?; d.memset_async(&s, &b, 0, bytes.max(16), 0)?; Ok(b) };
    let tape = up(bytemuck::cast_slice(&cn))?;
    let tables = up(bytemuck::cast_slice(&tabs))?;
    let ke = up(bytemuck::cast_slice(&hex8_stiffness(lc.e, lc.nu, lc.h)))?;
    let solid = zeros(nv * 4)?;
    let nflag = zeros(nn * 4)?;
    let vecs: Vec<D::Buffer> = (0..7).map(|_| zeros(nd * 8)).collect::<DriverResult<_>>()?;
    let part = zeros(2 * DOT_BLOCKS * 8)?;
    let scal = zeros(sc::N * 8)?;
    let stat = zeros(8 * 8)?;
    let [x, r, z, p, ap, b, diag] = [0, 1, 2, 3, 4, 5, 6].map(|i| vecs[i].device_ptr());
    let fa = FeaArgs {
        tape: tape.device_ptr(), tables: tables.device_ptr(), solid: solid.device_ptr(), nflag: nflag.device_ptr(), ke: ke.device_ptr(),
        x, r, z, p, ap, b, diag, scal: scal.device_ptr(), stat: stat.device_ptr(),
        lo: lc.lo, h: lc.h, force: lc.force, tol: lc.tol, count: design.nodes.len() as u32, dims: lc.dims, axis: lc.support_axis, _pad: [0; 3],
    };
    let a = upload_params(d, &s, &fa)?;
    let dot = |a0: u64, b0: u64, a1: u64, b1: u64, slot0: usize, slot1: usize, ndots: u32, always: u32| upload_params(d, &s, &DotArgs { a0, b0, a1, b1, part: part.device_ptr(), scal: scal.device_ptr(), n: nd as u64, slot0: slot0 as u32, slot1: slot1 as u32, ndots, always });
    let d_init_rz = dot(r, z, r, r, sc::RZ, sc::RR, 2, 1)?;
    let d_init_bb = dot(b, b, b, b, sc::BB, sc::BB, 1, 1)?;
    let d_pap = dot(p, ap, p, ap, sc::PAP, sc::PAP, 1, 0)?;
    let d_rz = dot(r, z, r, r, sc::RZN, sc::RR, 2, 0)?;
    let d_c = dot(x, b, x, b, sc::COMPLIANCE, sc::COMPLIANCE, 1, 1)?;
    let lin = |n: usize| LaunchConfig::linear((n as u64).min(4096 * 256), 256);
    let (pdot, fdot, one) = (LaunchConfig::linear((DOT_BLOCKS * DOT_T) as u64, DOT_T as u32), LaunchConfig::linear(DOT_T as u64, DOT_T as u32), LaunchConfig::linear(1, 32));
    let run_dot = |da: &agcad_kernels::params::ParamUpload<D>| -> DriverResult<()> {
        d.launch(&s, lib, "ag_fea_dot_part", &pdot, &da.buffer)?;
        d.launch(&s, lib, "ag_fea_dot_final", &fdot, &da.buffer)?;
        Ok(())
    };
    d.launch(&s, lib, "ag_fea_voxelize", &lin(nv), &a.buffer)?;
    d.launch(&s, lib, "ag_fea_nodes", &lin(nn), &a.buffer)?;
    d.launch(&s, lib, "ag_fea_init", &lin(nd), &a.buffer)?;
    run_dot(&d_init_rz)?;
    run_dot(&d_init_bb)?;
    d.launch(&s, lib, "ag_fea_start", &one, &a.buffer)?;
    let mut it = 0u32;
    while it < lc.max_iter {
        let chunk = 32.min(lc.max_iter - it);
        for _ in 0..chunk {
            d.launch(&s, lib, "ag_fea_check", &one, &a.buffer)?;
            d.launch(&s, lib, "ag_fea_matvec", &lin(nd), &a.buffer)?;
            run_dot(&d_pap)?;
            d.launch(&s, lib, "ag_fea_update1", &lin(nd), &a.buffer)?;
            run_dot(&d_rz)?;
            d.launch(&s, lib, "ag_fea_update2", &lin(nd), &a.buffer)?;
            d.launch(&s, lib, "ag_fea_step", &one, &a.buffer)?;
        }
        it += chunk;
        // escalation step: one 8-byte flag read per chunk (not part of a rollout)
        let mut pin = d.pinned(8)?;
        d.download_packed_async(&s, &[Region { buf: &scal, offset: sc::DONE * 8, bytes: 8 }], &mut pin)?;
        d.sync(&s)?;
        if bytemuck::pod_read_unaligned::<f64>(&pin.as_bytes()[..8]) != 0.0 { break; }
    }
    d.launch(&s, lib, "ag_fea_finish", &one, &a.buffer)?;
    run_dot(&d_c)?;
    d.launch(&s, lib, "ag_fea_maxdisp", &lin(nn), &a.buffer)?;
    let mut pin = d.pinned(sc::N * 8 + 64)?;
    d.download_packed_async(&s, &[Region { buf: &scal, offset: 0, bytes: sc::N * 8 }, Region { buf: &stat, offset: 0, bytes: 64 }], &mut pin)?;
    d.sync(&s)?;
    let sv: Vec<f64> = bytemuck::pod_collect_to_vec(&pin.as_bytes()[..sc::N * 8]);
    let st: Vec<u64> = bytemuck::pod_collect_to_vec(&pin.as_bytes()[sc::N * 8..sc::N * 8 + 64]);
    Ok(FeaReport { compliance: sv[sc::COMPLIANCE], max_disp: f64::from_bits(st[1]), iterations: sv[sc::ITER] as u32, converged: sv[sc::CONVERGED] != 0.0, voxels: st[2], dofs: st[3] })
}

// ------------------------------------------------------------------------------------
// CPU reference kernels (twins of the ag_fea_* kernels in cu/dops.cu)
// ------------------------------------------------------------------------------------

unsafe fn sl<'a, T>(a: u64, n: usize) -> &'a [T] { if n == 0 { &[] } else { unsafe { core::slice::from_raw_parts(a as *const T, n) } } }
#[allow(clippy::mut_from_ref)]
unsafe fn slm<'a, T>(a: u64, n: usize) -> &'a mut [T] { if n == 0 { &mut [] } else { unsafe { core::slice::from_raw_parts_mut(a as *mut T, n) } } }
unsafe fn fargs(p: *const u8) -> FeaArgs { unsafe { core::ptr::read_unaligned(p as *const FeaArgs) } }
unsafe fn dargs(p: *const u8) -> DotArgs { unsafe { core::ptr::read_unaligned(p as *const DotArgs) } }

fn sizes(a: &FeaArgs) -> (usize, usize, [usize; 3]) {
    let d = [a.dims[0] as usize, a.dims[1] as usize, a.dims[2] as usize];
    (d[0] * d[1] * d[2], (d[0] + 1) * (d[1] + 1) * (d[2] + 1), d)
}

/// Voxel occupancy: SDF sign at voxel centres.
pub unsafe fn ag_fea_voxelize(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let (nv, _, d) = sizes(&a);
    let tape = unsafe { sl::<CNode>(a.tape, a.count as usize) };
    let tb = unsafe { sl::<f32>(a.tables, a.count as usize * TABLE) };
    let solid = unsafe { slm::<u32>(a.solid, nv) };
    let stat = unsafe { slm::<u64>(a.stat, 8) };
    for v in 0..nv {
        let (i, j, k) = (v % d[0], (v / d[0]) % d[1], v / (d[0] * d[1]));
        let x = (a.lo[0] + (i as f64 + 0.5) * a.h) as f32;
        let y = (a.lo[1] + (j as f64 + 0.5) * a.h) as f32;
        let z = (a.lo[2] + (k as f64 + 0.5) * a.h) as f32;
        let inside = !tape.is_empty() && crate::math::tape_sdf_c(tape, tb, x, y, z) < 0.0;
        solid[v] = inside as u32;
        if inside { stat[2] += 1; }
    }
}

/// Node flags: used (touches a solid voxel), fixed (support face), loaded (opposite face).
pub unsafe fn ag_fea_nodes(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let (nv, nn, d) = sizes(&a);
    let solid = unsafe { sl::<u32>(a.solid, nv) };
    let nf = unsafe { slm::<u32>(a.nflag, nn) };
    let stat = unsafe { slm::<u64>(a.stat, 8) };
    for n in 0..nn {
        let c = [n % (d[0] + 1), (n / (d[0] + 1)) % (d[1] + 1), n / ((d[0] + 1) * (d[1] + 1))];
        let mut used = false;
        for dk in 0..2 { for dj in 0..2 { for di in 0..2 {
            let (i, j, k) = (c[0] as i64 + di - 1, c[1] as i64 + dj - 1, c[2] as i64 + dk - 1);
            if i >= 0 && j >= 0 && k >= 0 && (i as usize) < d[0] && (j as usize) < d[1] && (k as usize) < d[2] && solid[(k as usize * d[1] + j as usize) * d[0] + i as usize] != 0 { used = true; }
        }}}
        let ax = a.axis as usize;
        let mut f = 0u32;
        if used {
            f |= N_USED;
            if c[ax] == 0 { f |= N_FIXED; }
            if c[ax] == d[ax] { f |= N_LOADED; stat[0] += 1; }
            if c[ax] != 0 { stat[3] += 3; }
        }
        nf[n] = f;
    }
}

fn active(nf: &[u32], dof: usize) -> bool { let f = nf[dof / 3]; f & N_USED != 0 && f & N_FIXED == 0 }

pub unsafe fn ag_fea_init(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let (_, nn, _) = sizes(&a);
    let nd = 3 * nn;
    let nf = unsafe { sl::<u32>(a.nflag, nn) };
    let ke = unsafe { sl::<f64>(a.ke, 576) };
    let nload = unsafe { sl::<u64>(a.stat, 8)[0] };
    let (x, r, z, p, b, dg) = unsafe { (slm::<f64>(a.x, nd), slm::<f64>(a.r, nd), slm::<f64>(a.z, nd), slm::<f64>(a.p, nd), slm::<f64>(a.b, nd), slm::<f64>(a.diag, nd)) };
    for i in 0..nd {
        let act = active(nf, i);
        let c = i % 3;
        b[i] = if act && nf[i / 3] & N_LOADED != 0 { a.force[c] / nload as f64 } else { 0.0 };
        dg[i] = if act { ke[c * 25] * 4.0 } else { 1.0 };
        x[i] = 0.0;
        r[i] = b[i];
        z[i] = r[i] / dg[i];
        p[i] = z[i];
    }
}

/// Fixed-shape partial sums: block b, thread t accumulates i = b·T + t (stride BLOCKS·T), then
/// a shared-memory tree per block.
pub unsafe fn ag_fea_dot_part(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { dargs(params) };
    let scal = unsafe { sl::<f64>(a.scal, sc::N) };
    if a.always == 0 && scal[sc::DONE] != 0.0 { return; }
    let n = a.n as usize;
    let (a0, b0) = unsafe { (sl::<f64>(a.a0, n), sl::<f64>(a.b0, n)) };
    let (a1, b1) = unsafe { (sl::<f64>(a.a1, n), sl::<f64>(a.b1, n)) };
    let part = unsafe { slm::<f64>(a.part, 2 * DOT_BLOCKS) };
    let mut s0 = [0.0f64; DOT_T];
    let mut s1 = [0.0f64; DOT_T];
    for blk in 0..DOT_BLOCKS {
        for t in 0..DOT_T {
            let (mut v0, mut v1) = (0.0f64, 0.0f64);
            let mut i = blk * DOT_T + t;
            while i < n {
                v0 = v0 + a0[i] * b0[i];
                if a.ndots > 1 { v1 = v1 + a1[i] * b1[i]; }
                i += DOT_BLOCKS * DOT_T;
            }
            s0[t] = v0;
            s1[t] = v1;
        }
        tree(&mut s0, &mut s1);
        part[blk] = s0[0];
        part[DOT_BLOCKS + blk] = s1[0];
    }
}

fn tree(s0: &mut [f64; DOT_T], s1: &mut [f64; DOT_T]) {
    let mut st = DOT_T / 2;
    while st > 0 {
        for t in 0..st { s0[t] = s0[t] + s0[t + st]; s1[t] = s1[t] + s1[t + st]; }
        st >>= 1;
    }
}

pub unsafe fn ag_fea_dot_final(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { dargs(params) };
    let scal = unsafe { slm::<f64>(a.scal, sc::N) };
    if a.always == 0 && scal[sc::DONE] != 0.0 { return; }
    let part = unsafe { sl::<f64>(a.part, 2 * DOT_BLOCKS) };
    let mut s0 = [0.0f64; DOT_T];
    let mut s1 = [0.0f64; DOT_T];
    for t in 0..DOT_T {
        let (mut v0, mut v1) = (0.0f64, 0.0f64);
        let mut k = t;
        while k < DOT_BLOCKS { v0 = v0 + part[k]; v1 = v1 + part[DOT_BLOCKS + k]; k += DOT_T; }
        s0[t] = v0;
        s1[t] = v1;
    }
    tree(&mut s0, &mut s1);
    scal[a.slot0 as usize] = s0[0];
    if a.ndots > 1 { scal[a.slot1 as usize] = s1[0]; }
    if a.always == 0 && a.slot0 as usize == sc::PAP && !(s0[0] > 0.0) { scal[sc::DONE] = 1.0; }
}

/// ‖b‖ (floored like `agcad_solver::linear::pcg`).
pub unsafe fn ag_fea_start(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { slm::<f64>(a.scal, sc::N) };
    scal[sc::BN] = scal[sc::BB].sqrt().max(f64::MIN_POSITIVE);
    scal[sc::DONE] = 0.0;
    scal[sc::ITER] = 0.0;
    scal[sc::CONVERGED] = 0.0;
}

pub unsafe fn ag_fea_check(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { slm::<f64>(a.scal, sc::N) };
    if scal[sc::DONE] != 0.0 { return; }
    if scal[sc::RR].sqrt() / scal[sc::BN] <= a.tol { scal[sc::DONE] = 1.0; scal[sc::CONVERGED] = 1.0; }
}

/// y = K p: per active DOF, gather over the ≤ 8 adjacent solid elements (ascending order).
pub unsafe fn ag_fea_matvec(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    if unsafe { sl::<f64>(a.scal, sc::N)[sc::DONE] } != 0.0 { return; }
    let (nv, nn, d) = sizes(&a);
    let nd = 3 * nn;
    let solid = unsafe { sl::<u32>(a.solid, nv) };
    let nf = unsafe { sl::<u32>(a.nflag, nn) };
    let ke = unsafe { sl::<f64>(a.ke, 576) };
    let p = unsafe { sl::<f64>(a.p, nd) };
    let y = unsafe { slm::<f64>(a.ap, nd) };
    for dof in 0..nd { y[dof] = matvec_row(&d, solid, nf, ke, p, dof); }
}

const CORNER: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

fn matvec_row(d: &[usize; 3], solid: &[u32], nf: &[u32], ke: &[f64], p: &[f64], dof: usize) -> f64 {
    if !active(nf, dof) { return 0.0; }
    let n = dof / 3;
    let comp = dof % 3;
    let (nx, ny) = (d[0] + 1, d[1] + 1);
    let c = [n % nx, (n / nx) % ny, n / (nx * ny)];
    let mut y = 0.0f64;
    for dk in 0..2usize { for dj in 0..2usize { for di in 0..2usize {
        if c[0] + di < 1 || c[1] + dj < 1 || c[2] + dk < 1 { continue; }
        let (i, j, k) = (c[0] + di - 1, c[1] + dj - 1, c[2] + dk - 1);
        if i >= d[0] || j >= d[1] || k >= d[2] || solid[(k * d[1] + j) * d[0] + i] == 0 { continue; }
        // local corner of node n in element (i, j, k)
        let loc = [1 - di, 1 - dj, 1 - dk];
        let mut la = 0;
        for (q, cc) in CORNER.iter().enumerate() { if *cc == loc { la = q; } }
        let row = 3 * la + comp;
        let mut s = 0.0f64;
        for (q, cc) in CORNER.iter().enumerate() {
            let m = ((k + cc[2]) * ny + (j + cc[1])) * nx + (i + cc[0]);
            for e in 0..3 {
                let dd = 3 * m + e;
                let u = if active(nf, dd) { p[dd] } else { 0.0 };
                s = s + ke[row * 24 + 3 * q + e] * u;
            }
        }
        y = y + 1.0 * s;
    }}}
    y
}

pub unsafe fn ag_fea_update1(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { sl::<f64>(a.scal, sc::N) };
    if scal[sc::DONE] != 0.0 { return; }
    let (_, nn, _) = sizes(&a);
    let nd = 3 * nn;
    let alpha = scal[sc::RZ] / scal[sc::PAP];
    let (x, r, z, p, ap, dg) = unsafe { (slm::<f64>(a.x, nd), slm::<f64>(a.r, nd), slm::<f64>(a.z, nd), sl::<f64>(a.p, nd), sl::<f64>(a.ap, nd), sl::<f64>(a.diag, nd)) };
    for i in 0..nd {
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * ap[i];
        z[i] = r[i] / dg[i];
    }
}

pub unsafe fn ag_fea_update2(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { sl::<f64>(a.scal, sc::N) };
    if scal[sc::DONE] != 0.0 { return; }
    let (_, nn, _) = sizes(&a);
    let nd = 3 * nn;
    let beta = scal[sc::RZN] / scal[sc::RZ];
    let (z, p) = unsafe { (sl::<f64>(a.z, nd), slm::<f64>(a.p, nd)) };
    for i in 0..nd { p[i] = z[i] + beta * p[i]; }
}

pub unsafe fn ag_fea_step(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { slm::<f64>(a.scal, sc::N) };
    if scal[sc::DONE] != 0.0 { return; }
    scal[sc::RZ] = scal[sc::RZN];
    scal[sc::ITER] = scal[sc::ITER] + 1.0;
}

pub unsafe fn ag_fea_finish(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let scal = unsafe { slm::<f64>(a.scal, sc::N) };
    if scal[sc::RR].sqrt() / scal[sc::BN] <= a.tol { scal[sc::CONVERGED] = 1.0; }
}

/// max ‖u‖ over nodes (f64 bits; non-negative values order as integers).
pub unsafe fn ag_fea_maxdisp(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { fargs(params) };
    let (_, nn, _) = sizes(&a);
    let x = unsafe { sl::<f64>(a.x, 3 * nn) };
    let stat = unsafe { slm::<u64>(a.stat, 8) };
    for n in 0..nn {
        let m = (x[3 * n] * x[3 * n] + x[3 * n + 1] * x[3 * n + 1] + x[3 * n + 2] * x[3 * n + 2]).sqrt();
        stat[1] = stat[1].max(m.to_bits());
    }
}
