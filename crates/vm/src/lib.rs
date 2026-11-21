//! # agcad-vm — expression bytecode VM (spec 04 § Bytecode)
//!
//! Model-invented geometry (implicit functions, wrangles) arrives as expression bytecode in
//! the `0xFF00–0xFFFF` opcode range. The device interpreter (`cu/vm.cu`) evaluates one
//! program uniformly over `[configuration, sample]` (no divergence: every thread runs the
//! same instruction stream). The CPU reference ([`ag_vm_eval_cpu`]) is the conformance
//! oracle and has identical semantics to `agcad_rtc::ExprGraph::eval`.

use agcad_driver::{AgStatus, CpuKernelFn, LaunchConfig, LibrarySource};
use agcad_rtc::expr::{eval_bin, eval_un, value_noise, Bin, Dim, ExprGraph, Node, Un};
use bytemuck::{Pod, Zeroable};

/// Hard limit on nodes per program (register file in local memory on the device).
pub const MAX_VM_NODES: usize = 1024;
pub const MAX_VM_IO: usize = 32;

pub mod op {
    pub const CONST: u16 = 0xFF00;
    pub const INPUT: u16 = 0xFF01;
    pub const PARAM: u16 = 0xFF02;
    pub const UN_BASE: u16 = 0xFF10; // + Un index
    pub const BIN_BASE: u16 = 0xFF40; // + Bin index
    pub const SELECT: u16 = 0xFF80;
    pub const NOISE: u16 = 0xFF81;
    pub const OUTPUT: u16 = 0xFFF0;
}

const UNS: [Un; 14] = [Un::Neg, Un::Abs, Un::Sqrt, Un::Sin, Un::Cos, Un::Tan, Un::Asin, Un::Acos, Un::Atan, Un::Exp, Un::Ln, Un::Floor, Un::Ceil, Un::Not];
const BINS: [Bin; 17] = [Bin::Add, Bin::Sub, Bin::Mul, Bin::Div, Bin::Min, Bin::Max, Bin::Pow, Bin::Atan2, Bin::Lt, Bin::Le, Bin::Gt, Bin::Ge, Bin::Eq, Bin::Ne, Bin::And, Bin::Or, Bin::Mod];

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VmInstr {
    pub op: u16,
    pub _pad: u16,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VmProgram {
    /// Node instructions followed by OUTPUT instructions.
    pub instrs: Vec<VmInstr>,
    pub consts: Vec<f64>,
    pub n_inputs: u32,
    pub n_params: u32,
    pub n_outputs: u32,
}

pub fn encode(g: &ExprGraph) -> Result<VmProgram, AgStatus> {
    if g.nodes.len() > MAX_VM_NODES || g.outputs.len() > MAX_VM_IO || g.input_names.len() > MAX_VM_IO || g.param_names.len() > MAX_VM_IO {
        return Err(AgStatus::CapacityExceeded);
    }
    let mut p = VmProgram { n_inputs: g.input_names.len() as u32, n_params: g.param_names.len() as u32, n_outputs: g.outputs.len() as u32, ..Default::default() };
    for n in &g.nodes {
        let i = match *n {
            Node::Const(c) => { p.consts.push(c); VmInstr { op: op::CONST, _pad: 0, a: (p.consts.len() - 1) as u32, b: 0, c: 0 } }
            Node::Input(k) => VmInstr { op: op::INPUT, _pad: 0, a: k as u32, b: 0, c: 0 },
            Node::Param(k) => VmInstr { op: op::PARAM, _pad: 0, a: k as u32, b: 0, c: 0 },
            Node::Un(u, a) => VmInstr { op: op::UN_BASE + UNS.iter().position(|x| *x == u).unwrap() as u16, _pad: 0, a, b: 0, c: 0 },
            Node::Bin(b2, a, b) => VmInstr { op: op::BIN_BASE + BINS.iter().position(|x| *x == b2).unwrap() as u16, _pad: 0, a, b, c: 0 },
            Node::Select(c, a, b) => VmInstr { op: op::SELECT, _pad: 0, a: c, b: a, c: b },
            Node::Noise(x, y, z) => VmInstr { op: op::NOISE, _pad: 0, a: x, b: y, c: z },
        };
        p.instrs.push(i);
    }
    for (slot, o) in g.outputs.iter().enumerate() {
        p.instrs.push(VmInstr { op: op::OUTPUT, _pad: 0, a: *o, b: slot as u32, c: 0 });
    }
    Ok(p)
}

/// Decode + validate bytecode written by a device model. Operands must refer to earlier
/// nodes (SSA order), indices must be in range, constants finite (`NONFINITE_PARAMETER`).
pub fn decode(p: &VmProgram) -> Result<ExprGraph, AgStatus> {
    let mut g = ExprGraph::default();
    for k in 0..p.n_inputs { g.input_names.push(format!("in{k}")); g.input_dims.push(Dim::NONE); }
    for k in 0..p.n_params { g.param_names.push(format!("prm{k}")); g.param_dims.push(Dim::NONE); }
    let mut nodes = 0u32;
    for ins in &p.instrs {
        let earlier = |x: u32| if x < nodes { Ok(x) } else { Err(AgStatus::InvalidParameter) };
        let n = match ins.op {
            op::CONST => { let c = *p.consts.get(ins.a as usize).ok_or(AgStatus::InvalidParameter)?; if !c.is_finite() { return Err(AgStatus::NonfiniteParameter); } Node::Const(c) }
            op::INPUT => { if ins.a >= p.n_inputs { return Err(AgStatus::InvalidParameter); } Node::Input(ins.a as u16) }
            op::PARAM => { if ins.a >= p.n_params { return Err(AgStatus::InvalidParameter); } Node::Param(ins.a as u16) }
            o if (op::UN_BASE..op::UN_BASE + UNS.len() as u16).contains(&o) => Node::Un(UNS[(o - op::UN_BASE) as usize], earlier(ins.a)?),
            o if (op::BIN_BASE..op::BIN_BASE + BINS.len() as u16).contains(&o) => Node::Bin(BINS[(o - op::BIN_BASE) as usize], earlier(ins.a)?, earlier(ins.b)?),
            op::SELECT => Node::Select(earlier(ins.a)?, earlier(ins.b)?, earlier(ins.c)?),
            op::NOISE => Node::Noise(earlier(ins.a)?, earlier(ins.b)?, earlier(ins.c)?),
            op::OUTPUT => { if ins.b as usize != g.outputs.len() { return Err(AgStatus::InvalidParameter); } g.outputs.push(earlier(ins.a)?); continue; }
            _ => return Err(AgStatus::InvalidOpcode),
        };
        g.nodes.push(n);
        g.dims.push(Dim::NONE);
        nodes += 1;
        if nodes as usize > MAX_VM_NODES { return Err(AgStatus::CapacityExceeded); }
    }
    if g.outputs.len() != p.n_outputs as usize { return Err(AgStatus::InvalidParameter); }
    Ok(g)
}

/// Parameter block of the VM kernel (identical layout to `AgExprParams` + program pointers).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VmParams {
    pub instrs: u64,
    pub consts: u64,
    pub inputs: u64,
    pub params: u64,
    pub out: u64,
    pub n_instrs: u32,
    pub samples: u32,
    pub configs: u32,
    pub n_inputs: u32,
    pub n_params: u32,
    pub n_outputs: u32,
    pub inputs_per_config: u32,
    pub _pad: u32,
}

pub fn exec(instrs: &[VmInstr], consts: &[f64], input: &dyn Fn(u32) -> f64, param: &dyn Fn(u32) -> f64, out: &mut dyn FnMut(u32, f64)) {
    let mut r = [0.0f64; MAX_VM_NODES];
    let mut k = 0usize;
    for ins in instrs {
        let v = match ins.op {
            op::CONST => consts[ins.a as usize],
            op::INPUT => input(ins.a),
            op::PARAM => param(ins.a),
            op::SELECT => if r[ins.a as usize] != 0.0 { r[ins.b as usize] } else { r[ins.c as usize] },
            op::NOISE => value_noise(r[ins.a as usize], r[ins.b as usize], r[ins.c as usize]),
            op::OUTPUT => { out(ins.b, r[ins.a as usize]); continue; }
            o if o >= op::BIN_BASE && o < op::BIN_BASE + 17 => eval_bin(BINS[(o - op::BIN_BASE) as usize], r[ins.a as usize], r[ins.b as usize]),
            o => eval_un(UNS[(o - op::UN_BASE) as usize], r[ins.a as usize]),
        };
        r[k] = v;
        k += 1;
    }
}

/// CPU reference of `ag_vm_eval` (programs are validated by [`decode`] before upload).
///
/// # Safety
/// `params` points at a [`VmParams`] whose addresses are valid host buffers.
pub unsafe fn ag_vm_eval_cpu(params: *const u8, _cfg: &LaunchConfig) {
    let p: VmParams = unsafe { core::ptr::read_unaligned(params as *const VmParams) };
    let instrs = unsafe { core::slice::from_raw_parts(p.instrs as *const VmInstr, p.n_instrs as usize) };
    let nconst = instrs.iter().filter(|i| i.op == op::CONST).map(|i| i.a as usize + 1).max().unwrap_or(0);
    let consts = unsafe { core::slice::from_raw_parts(p.consts as *const f64, nconst) };
    let (s, c) = (p.samples as usize, p.configs as usize);
    let in_stride = if p.inputs_per_config != 0 { c * s } else { s };
    let inputs = unsafe { core::slice::from_raw_parts(p.inputs as *const f64, p.n_inputs as usize * in_stride) };
    let prm = unsafe { core::slice::from_raw_parts(p.params as *const f64, p.n_params as usize * c) };
    let out = unsafe { core::slice::from_raw_parts_mut(p.out as *mut f64, p.n_outputs as usize * c * s) };
    for ci in 0..c {
        for si in 0..s {
            let base = if p.inputs_per_config != 0 { ci * s + si } else { si };
            exec(instrs, consts, &|k| inputs[k as usize * in_stride + base], &|k| prm[k as usize * c + ci], &mut |o, v| out[(o as usize * c + ci) * s + si] = v);
        }
    }
}

pub const VM_CPU: &[(&str, CpuKernelFn)] = &[("ag_vm_eval", ag_vm_eval_cpu)];

pub fn vm_library() -> LibrarySource {
    let src = format!("{}\n{}", agcad_rtc::codegen::VALUE_NOISE_CUDA, include_str!("../cu/vm.cu"));
    LibrarySource { name: "ag_vm", cuda_source: Some(agcad_kernels_cuda::bundle(&src, &[])), cpu_kernels: VM_CPU, options: vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_reference_matches_graph_eval() {
        let src = "#[implicit] fn s(p: Vec3, #[param(default = 5.0*mm)] r: Length) -> Length { length(p) - r + noise3(p) * 0.1 * mm }";
        let m = agcad_rtc::compile_module(src, agcad_ir::Catalog::global()).unwrap();
        let g = m.get("s").unwrap().expr.clone().unwrap();
        let prog = encode(&g).unwrap();
        let back = decode(&prog).unwrap();
        let pts = [1.0, 2.0, 3.0];
        assert_eq!(g.eval(&pts, &[5.0])[0].to_bits(), back.eval(&pts, &[5.0])[0].to_bits());
        let mut bad = prog.clone();
        bad.instrs[0] = VmInstr { op: op::BIN_BASE, _pad: 0, a: 5, b: 6, c: 0 };
        assert_eq!(decode(&bad).unwrap_err(), AgStatus::InvalidParameter);
    }
}
