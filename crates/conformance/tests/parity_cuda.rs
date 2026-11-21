//! CUDA vs CPU-reference bit parity for every device library (AG-08-001, AG-08-005):
//! primitives, expression VM, batched LM solver, MLP inference and harness wavefront stages.
//! Each case runs the same generic host orchestration on both devices and compares bytes.
//! Requires an NVIDIA GPU: `cargo test -p agcad-conformance --test parity_cuda -- --ignored`.

use agcad_driver::*;
use agcad_kernels::params::upload_params;
use agcad_kernels::primitives::*;
use bytemuck::Zeroable;

fn up<D: AgDevice>(d: &D, s: &D::Stream, bytes: &[u8]) -> D::Buffer {
    let b = d.alloc_bytes(s, bytes.len().max(1)).unwrap();
    if !bytes.is_empty() {
        d.upload_from_host_async(s, bytes, &b, 0).unwrap();
    }
    b
}

fn down<D: AgDevice>(d: &D, s: &D::Stream, b: &D::Buffer, bytes: usize) -> Vec<u8> {
    let mut pin = d.pinned(bytes).unwrap();
    let _ev = d.download_packed_async(s, &[Region { buf: b, offset: 0, bytes }], &mut pin).unwrap();
    d.sync(s).unwrap();
    pin.as_bytes()[..bytes].to_vec()
}

/// Run `$f` on the CUDA device and on the CPU reference; assert identical bytes.
macro_rules! parity {
    ($lib:expr, $f:ident) => {{
        let cuda = agcad_driver_cuda::CudaDevice::open(0).expect("CUDA device");
        let cpu = agcad_driver_cpu::CpuDevice::new();
        let g = $f(&*cuda, $lib);
        let c = $f(&*cpu, $lib);
        assert_eq!(g.len(), c.len());
        let diff = g.iter().zip(&c).position(|(a, b)| a != b);
        assert!(diff.is_none(), "first differing byte at {:?} of {}", diff, g.len());
        g
    }};
}

fn prim_lib() -> LibrarySource {
    agcad_kernels_cuda::primitives_library()
}

fn scan_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let n = 300_001u32; // multi-block + recursive block sums
    let v: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2654435761u32) >> 28).collect();
    let (i, o) = (up(d, &s, bytemuck::cast_slice(&v)), d.alloc_bytes(&s, n as usize * 4).unwrap());
    exclusive_scan_u32(d, &s, l, &i, &o, n).unwrap();
    down(d, &s, &o, n as usize * 4)
}

fn compact_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let n = 70_000u32;
    let flags: Vec<u32> = (0..n).map(|i| ((i % 7) == 0 || (i % 11) == 3) as u32).collect();
    let vals: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let (fb, vb, ob) = (up(d, &s, bytemuck::cast_slice(&flags)), up(d, &s, bytemuck::cast_slice(&vals)), d.alloc_bytes(&s, n as usize * 8).unwrap());
    let kept = flags.iter().filter(|f| **f != 0).count();
    let _offs = compact(d, &s, l, &fb, &vb, &ob, n, 8).unwrap();
    down(d, &s, &ob, kept * 8)
}

fn sort_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let n = 100_003u32;
    let keys: Vec<u64> = (0..n as u64).map(|i| (i.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 20) % 5000).collect(); // many ties
    let vals: Vec<u32> = (0..n).collect();
    let (kb, vb) = (up(d, &s, bytemuck::cast_slice(&keys)), up(d, &s, bytemuck::cast_slice(&vals)));
    radix_sort_pairs(d, &s, l, &kb, &vb, n, 64).unwrap();
    let mut out = down(d, &s, &kb, n as usize * 8);
    out.extend(down(d, &s, &vb, n as usize * 4));
    out
}

fn topk_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let mut sc: Vec<f64> = (0..50_000).map(|i| ((i as f64) * 0.37).sin().round()).collect();
    sc[7] = f64::NAN;
    sc[8] = -0.0;
    sc[9] = 0.0;
    sc[10] = f64::NEG_INFINITY;
    let n = sc.len() as u32;
    let sb = up(d, &s, bytemuck::cast_slice(&sc));
    let (kb, vb) = (d.alloc_bytes(&s, n as usize * 8).unwrap(), d.alloc_bytes(&s, n as usize * 4).unwrap());
    stable_order_by_score(d, &s, l, &sb, &kb, &vb, n, true).unwrap();
    down(d, &s, &vb, n as usize * 4)
}

fn reduce_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let vals: Vec<f64> = (0..200_000).map(|i| ((i as f64) * 0.1).sin() * 1e3 + 1e-7 * i as f64).collect();
    let offs: Vec<u32> = vec![0, 0, 1, 1000, 1001, 150_000, 200_000];
    let (vb, ob) = (up(d, &s, bytemuck::cast_slice(&vals)), up(d, &s, bytemuck::cast_slice(&offs)));
    let mut all = Vec::new();
    for op in [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max] {
        let out = d.alloc_bytes(&s, 6 * 8).unwrap();
        segmented_reduce_f64(d, &s, l, &vb, &ob, &out, 6, op).unwrap();
        all.extend(down(d, &s, &out, 48));
    }
    all
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn primitives_scan() {
    parity!(prim_lib, scan_case);
}
#[test]
#[ignore = "requires NVIDIA GPU"]
fn primitives_compact() {
    parity!(prim_lib, compact_case);
}
#[test]
#[ignore = "requires NVIDIA GPU"]
fn primitives_radix_sort() {
    parity!(prim_lib, sort_case);
}
#[test]
#[ignore = "requires NVIDIA GPU"]
fn primitives_top_k() {
    parity!(prim_lib, topk_case);
}
#[test]
#[ignore = "requires NVIDIA GPU"]
fn primitives_segmented_reduce() {
    parity!(prim_lib, reduce_case);
}

// ---- expression VM: implicit field with noise over samples × configurations ----

fn vm_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let m = agcad_rtc::compile_module(
        "#[implicit] fn s(p: Vec3, #[param(default = 5.0*mm)] r: Length) -> Length { length(p) - r + noise3(p) * 0.5 * mm }",
        agcad_ir::Catalog::global(),
    )
    .unwrap();
    let prog = agcad_vm::encode(&m.get("s").unwrap().expr.clone().unwrap()).unwrap();
    let (samples, configs) = (4096u32, 8u32);
    let mut inputs = vec![0.0f64; 3 * samples as usize];
    for i in 0..samples as usize {
        for k in 0..3 {
            inputs[k * samples as usize + i] = ((i * (k + 3)) as f64 * 0.013).sin() * 9.0;
        }
    }
    let params: Vec<f64> = (0..configs).map(|c| 3.0 + c as f64 * 0.5).collect();
    let (ib, kb, inb, pb) = (up(d, &s, bytemuck::cast_slice(&prog.instrs)), up(d, &s, bytemuck::cast_slice(&prog.consts)), up(d, &s, bytemuck::cast_slice(&inputs)), up(d, &s, bytemuck::cast_slice(&params)));
    let nout = (prog.n_outputs * samples * configs) as usize;
    let ob = d.alloc_bytes(&s, nout * 8).unwrap();
    let p = upload_params(
        d,
        &s,
        &agcad_vm::VmParams {
            instrs: ib.device_ptr(),
            consts: kb.device_ptr(),
            inputs: inb.device_ptr(),
            params: pb.device_ptr(),
            out: ob.device_ptr(),
            n_instrs: prog.instrs.len() as u32,
            samples,
            configs,
            n_inputs: prog.n_inputs,
            n_params: prog.n_params,
            n_outputs: prog.n_outputs,
            inputs_per_config: 0,
            _pad: 0,
        },
    )
    .unwrap();
    d.launch(&s, l, "ag_vm_eval", &LaunchConfig::linear(samples as u64 * configs as u64, 128), &p.buffer).unwrap();
    down(d, &s, &ob, nout * 8)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn vm_implicit_with_noise() {
    parity!(agcad_vm::vm_library, vm_case);
}

// ---- batched LM: circle–line intersection, one problem per configuration ----

const NLS_CONFIGS: usize = 1000;

fn nls_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    use agcad_rtc::expr::{Bin, Dim, ExprGraph};
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let mut g = ExprGraph::default();
    let (x, y) = (g.input("x", Dim::NONE), g.input("y", Dim::NONE));
    let r = g.param("r", Dim::NONE);
    let (xx, yy, rr) = (g.bin(Bin::Mul, x, x).unwrap(), g.bin(Bin::Mul, y, y).unwrap(), g.bin(Bin::Mul, r, r).unwrap());
    let sum = g.bin(Bin::Add, xx, yy).unwrap();
    let c0 = g.bin(Bin::Sub, sum, rr).unwrap();
    let c1 = g.bin(Bin::Sub, y, x).unwrap();
    g.outputs = vec![c0, c1];
    let comp = agcad_solver::nls::compile(&g).unwrap();
    let configs = NLS_CONFIGS as u32;
    let mut x0 = vec![0.0f64; 2 * NLS_CONFIGS];
    for c in 0..NLS_CONFIGS {
        x0[c] = 1.0 + (c % 7) as f64;
        x0[NLS_CONFIGS + c] = 0.5;
    }
    let data: Vec<f64> = (0..configs).map(|c| 2.0 + c as f64 * 0.01).collect();
    let (xb, db) = (up(d, &s, bytemuck::cast_slice(&x0)), up(d, &s, bytemuck::cast_slice(&data)));
    let (st, it, co) = (d.alloc_bytes(&s, NLS_CONFIGS * 2).unwrap(), d.alloc_bytes(&s, NLS_CONFIGS * 4).unwrap(), d.alloc_bytes(&s, NLS_CONFIGS * 8).unwrap());
    agcad_solver::nls_solve_device(d, &s, l, &comp, 1, configs, agcad_solver::NlsBuffers { x: &xb, data: &db, status: &st, iters: &it, cost: &co }, agcad_solver::nls::LmOptions::default()).unwrap();
    let mut out = down(d, &s, &xb, 2 * NLS_CONFIGS * 8);
    out.extend(down(d, &s, &st, NLS_CONFIGS * 2));
    out.extend(down(d, &s, &it, NLS_CONFIGS * 4));
    out.extend(down(d, &s, &co, NLS_CONFIGS * 8));
    out
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn nls_batched_lm() {
    let bytes = parity!(agcad_solver::solver_library, nls_case);
    let x: Vec<f64> = bytemuck::pod_collect_to_vec(&bytes[..2 * NLS_CONFIGS * 8]);
    let r = 2.0 / 2f64.sqrt();
    assert!((x[0] - r).abs() < 1e-9 && (x[NLS_CONFIGS] - r).abs() < 1e-9, "{} {}", x[0], x[NLS_CONFIGS]);
}

// ---- MLP inference (CuModel) ----

fn mlp_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let dims = [(37usize, 64usize), (64, 64), (64, 5)];
    let layers = dims
        .iter()
        .enumerate()
        .map(|(li, &(ni, no))| {
            let w: Vec<f32> = (0..ni * no).map(|k| (((k * 31 + li * 7) % 97) as f32 / 97.0 - 0.5) * 0.3).collect();
            let b: Vec<f32> = (0..no).map(|k| (k as f32 * 0.01) - 0.1).collect();
            (ni, no, w, b)
        })
        .collect();
    let mlp = agcad_model::cumodel::Mlp { layers };
    let configs = 513u32;
    let obs: Vec<f32> = (0..37 * configs as usize).map(|k| ((k as f32) * 0.017).sin()).collect();
    let ob = up(d, &s, bytemuck::cast_slice(&obs));
    let y = agcad_model::cumodel::forward_device(d, &s, l, &mlp, &ob, configs).unwrap();
    down(d, &s, &y, configs as usize * 5 * 4)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn mlp_forward() {
    parity!(agcad_model::cumodel::model_library, mlp_case);
}

// ---- harness wavefront: validate + bin ----

fn harness_lib() -> LibrarySource {
    agcad_harness::wavefront::harness_library()
}

fn wavefront_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    use agcad_harness::wavefront::*;
    let s = d.stream(Priority::High).unwrap();
    let h = d.load_library(&src()).unwrap();
    let p = d.load_library(&prim_lib()).unwrap();
    let configs = 64u32;
    let u = |v: &[u32]| up(d, &s, bytemuck::cast_slice(v));
    let budget: Vec<u32> = (0..configs).map(|c| if c % 9 == 4 { 0 } else { 10 }).collect();
    let st = WavefrontState {
        configs,
        revisions: u(&vec![7; configs as usize]),
        budget: u(&budget),
        active: u(&vec![1; configs as usize]),
        current: u(&(0..configs).collect::<Vec<_>>()),
        trial: u(&(configs..2 * configs).collect::<Vec<_>>()),
        cfg_status: u(&vec![0; configs as usize]),
    };
    let ops = [0x0700u16, 0x0E02, 0x0301, 0x9999];
    let cmds: Vec<CommandRecord> = (0..2000u32)
        .map(|i| {
            let mut c: CommandRecord = Zeroable::zeroed();
            c.opcode = ops[(i * 7 % 4) as usize];
            c.configuration = i % configs;
            c.expected_revision = if i % 13 == 0 { 6 } else { 7 };
            c
        })
        .collect();
    let allow = allow_bitmap(&[0x0700, 0x0E02, 0x0301]);
    let r = validate_and_bin(d, &s, h, p, &cmds, &allow, &st).unwrap();
    let mut out: Vec<u8> = r.statuses.iter().flat_map(|x| x.status.to_le_bytes()).collect();
    for (a, b) in r.bins {
        out.extend(a.to_le_bytes());
        out.extend(b.to_le_bytes());
    }
    out
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn harness_validate_and_bin() {
    parity!(harness_lib, wavefront_case);
}

// ---- batched mass properties of tessellated configurations (evaluate.measure_inertia) ----

/// 64 flange configurations (r × t × n), tessellated and concatenated with triangle offsets.
fn flange_meshes() -> Vec<agcad_geo::tessellate::Mesh> {
    use agcad_geo::builders::{prism, Profile};
    use agcad_geo::math::Frame;
    let mut out = Vec::new();
    for &n in &[3usize, 4, 6, 8] { for &r in &[50.0, 60.0, 70.0, 80.0] { for &t in &[6.0, 10.0, 14.0, 20.0] {
        let mut p = Profile::circle(0.0, 0.0, r);
        for k in 0..n { let a = k as f64 * core::f64::consts::TAU / n as f64; p = p.with_hole(Profile::circle((r - 15.0) * a.cos(), (r - 15.0) * a.sin(), 4.5)); }
        out.push(agcad_geo::tessellate::tessellate(&prism(&p, &Frame::WORLD, t), &Default::default()));
    }}}
    out
}

fn mass_case<D: KernelExt>(d: &D, src: fn() -> LibrarySource) -> Vec<u8> {
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&src()).unwrap();
    let meshes = flange_meshes();
    let (mut pos, mut tris, mut offs) = (Vec::<f64>::new(), Vec::<u32>::new(), vec![0u32]);
    for m in &meshes {
        let base = (pos.len() / 3) as u32;
        for p in &m.positions { pos.extend([p.x, p.y, p.z]); }
        for t in &m.triangles { tris.extend(t.iter().map(|i| i + base)); }
        offs.push((tris.len() / 3) as u32);
    }
    let n_tris = (tris.len() / 3) as u32;
    let (pb, tb, ob) = (up(d, &s, bytemuck::cast_slice(&pos)), up(d, &s, bytemuck::cast_slice(&tris)), up(d, &s, bytemuck::cast_slice(&offs)));
    let sums = mesh_mass_sums(d, &s, l, &pb, &tb, n_tris, &ob, meshes.len() as u32).unwrap();
    down(d, &s, &sums, MESH_TERMS * meshes.len() * 8)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn mesh_mass_properties_batched() {
    let bytes = parity!(prim_lib, mass_case);
    let sums: Vec<f64> = bytemuck::pod_collect_to_vec(&bytes);
    let meshes = flange_meshes();
    for (m, mesh) in meshes.iter().enumerate() {
        let (v, a, _) = mass_properties_from_sums(&sums, meshes.len(), m);
        let (v0, a0, _) = mesh.mass_properties();
        assert!((v - v0).abs() <= 1e-12 * v0.abs() && (a - a0).abs() <= 1e-12 * a0, "mesh {m}: {v} vs {v0}");
    }
}
