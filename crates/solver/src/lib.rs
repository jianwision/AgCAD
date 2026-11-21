//! # agcad-solver — batched numerical solvers (cf. cuSOLVER / cuDSS)
//!
//! Every solve in AgCAD goes through this crate (spec 01 § Solver library):
//! * [`nls`]: batched Levenberg–Marquardt (sketch, assembly, residual constraints, fitting).
//! * [`graph`]: DOF, structural rank, Dulmage–Mendelsohn, conflict sets, solve plans.
//! * [`dense`], [`linear`]: Cholesky/QR/SVD, PCG, affine constraint spaces.
//! * [`root`]: Newton and certified interval Newton.
//! * [`opt`]: Adam, Nelder–Mead, CMA-ES, simulated annealing, augmented Lagrangian.
//! * [`interval`], [`df64`]: certified bounds and double-float arithmetic.
//!
//! Depends only on kernels + driver (+ rtc/vm for expression residuals); never on documents.

pub mod dense;
pub mod df64;
pub mod graph;
pub mod interval;
pub mod linear;
pub mod nls;
pub mod opt;
pub mod root;

use agcad_driver::*;
use agcad_kernels::upload_params;
use bytemuck::{Pod, Zeroable};

/// Parameter block of `ag_nls_lm` (layout mirrored in `cu/nls.cu`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NlsParams {
    pub r_instrs: u64,
    pub r_consts: u64,
    pub j_instrs: u64,
    pub j_consts: u64,
    pub x: u64,
    pub data: u64,
    pub status: u64,
    pub iters: u64,
    pub cost: u64,
    pub r_n: u32,
    pub j_n: u32,
    pub n: u32,
    pub m: u32,
    pub n_data: u32,
    pub configs: u32,
    pub max_iter: u32,
    pub _pad: u32,
    pub tol: f64,
    pub xtol: f64,
    pub lambda0: f64,
}

/// CPU reference of `ag_nls_lm`.
///
/// # Safety
/// `params` points at an [`NlsParams`] with valid host addresses (CPU reference device).
pub unsafe fn ag_nls_lm_cpu(params: *const u8, _cfg: &LaunchConfig) {
    use agcad_vm::{VmInstr, VmProgram};
    let p: NlsParams = unsafe { core::ptr::read_unaligned(params as *const NlsParams) };
    let read_prog = |instrs: u64, consts: u64, n_instrs: u32| -> VmProgram {
        let ins = unsafe { core::slice::from_raw_parts(instrs as *const VmInstr, n_instrs as usize) }.to_vec();
        let nk = ins.iter().filter(|i| i.op == agcad_vm::op::CONST).map(|i| i.a as usize + 1).max().unwrap_or(0);
        let k = unsafe { core::slice::from_raw_parts(consts as *const f64, nk) }.to_vec();
        VmProgram { instrs: ins, consts: k, n_inputs: p.n, n_params: p.n_data, n_outputs: 0 }
    };
    let compiled = nls::Compiled { n: p.n as usize, m: p.m as usize, residual: read_prog(p.r_instrs, p.r_consts, p.r_n), jacobian: read_prog(p.j_instrs, p.j_consts, p.j_n) };
    let c = p.configs as usize;
    let x = unsafe { core::slice::from_raw_parts_mut(p.x as *mut f64, p.n as usize * c) };
    let data = unsafe { core::slice::from_raw_parts(p.data as *const f64, p.n_data as usize * c) };
    let status = unsafe { core::slice::from_raw_parts_mut(p.status as *mut u16, c) };
    let iters = unsafe { core::slice::from_raw_parts_mut(p.iters as *mut u32, c) };
    let cost = unsafe { core::slice::from_raw_parts_mut(p.cost as *mut f64, c) };
    let opt = nls::LmOptions { max_iter: p.max_iter, tol: p.tol, xtol: p.xtol, lambda0: p.lambda0 };
    for ci in 0..c {
        let x0: Vec<f64> = (0..p.n as usize).map(|k| x[k * c + ci]).collect();
        let d: Vec<f64> = (0..p.n_data as usize).map(|k| data[k * c + ci]).collect();
        let r = nls::lm_solve(&compiled, &x0, &d, opt);
        for k in 0..p.n as usize { x[k * c + ci] = r.x[k]; }
        status[ci] = r.status.value();
        iters[ci] = r.iterations;
        cost[ci] = r.cost;
    }
}

pub const SOLVER_CPU: &[(&str, CpuKernelFn)] = &[("ag_nls_lm", ag_nls_lm_cpu)];

pub fn solver_library() -> LibrarySource {
    let src = format!("{}\n{}", agcad_rtc::codegen::VALUE_NOISE_CUDA, include_str!("../cu/nls.cu"));
    LibrarySource { name: "ag_solver", cuda_source: Some(agcad_kernels_cuda::bundle(&src, &[])), cpu_kernels: SOLVER_CPU, options: vec![] }
}

/// Device buffers for a batched LM solve.
pub struct NlsBuffers<'a, B: RawBuffer> {
    pub x: &'a B,
    pub data: &'a B,
    pub status: &'a B,
    pub iters: &'a B,
    pub cost: &'a B,
}

/// Launch the batched LM solve: one thread per configuration (spec 14).
pub fn nls_solve_device<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, c: &nls::Compiled, n_data: u32, configs: u32, bufs: NlsBuffers<'_, D::Buffer>, opt: nls::LmOptions) -> DriverResult<()> {
    let up = |bytes: &[u8]| -> DriverResult<D::Buffer> {
        let b = dev.alloc_bytes(s, bytes.len().max(1))?;
        dev.upload_from_host_async(s, bytes, &b, 0)?;
        Ok(b)
    };
    let ri = up(bytemuck::cast_slice(&c.residual.instrs))?;
    let rk = up(bytemuck::cast_slice(&c.residual.consts))?;
    let ji = up(bytemuck::cast_slice(&c.jacobian.instrs))?;
    let jk = up(bytemuck::cast_slice(&c.jacobian.consts))?;
    let prm = NlsParams {
        r_instrs: ri.device_ptr(), r_consts: rk.device_ptr(), j_instrs: ji.device_ptr(), j_consts: jk.device_ptr(),
        x: bufs.x.device_ptr(), data: bufs.data.device_ptr(), status: bufs.status.device_ptr(), iters: bufs.iters.device_ptr(), cost: bufs.cost.device_ptr(),
        r_n: c.residual.instrs.len() as u32, j_n: c.jacobian.instrs.len() as u32, n: c.n as u32, m: c.m as u32, n_data, configs,
        max_iter: opt.max_iter, _pad: 0, tol: opt.tol, xtol: opt.xtol, lambda0: opt.lambda0,
    };
    let p = upload_params(dev, s, &prm)?;
    // Local-memory heavy kernel: small blocks.
    dev.launch(s, lib, "ag_nls_lm", &LaunchConfig::linear(configs as u64, 64), &p.buffer)?;
    Ok(())
}
