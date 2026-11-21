//! Device rollout on the GPU: bit parity with the CPU reference, graph-mode (device-driven
//! WHILE) equivalence with zero host round trips, and a 4096-configuration throughput run.
//! Requires an NVIDIA GPU: `cargo test -p agcad-dops --test rollout_cuda -- --ignored --nocapture`.

#[path = "rollout_cpu.rs"]
#[allow(dead_code)]
mod cpu;

use agcad_dops::lower::to_layered;
use agcad_dops::{Studio, StudioConfig};
use std::time::Instant;

fn cfg(configs: u32, rounds: u32) -> StudioConfig { StudioConfig { configs, records: 4, max_rounds: rounds, coarse: 8, levels: 5, cap: 4096, nodes: 8, budget: 64, tighten: 0, max_cells: 0, policy: 0, island: 16 } }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn cuda_host_loop_matches_cpu_reference_bitwise() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let cpu_dev = agcad_driver_cpu::CpuDevice::new();
    // tightened bounds on (per-cell interval refinement is part of the bit-parity contract)
    let tc = StudioConfig { tighten: 8, ..cfg(48, 10) };
    let g = Studio::new(cuda, &base, &goal, tc).unwrap();
    let c = Studio::new(cpu_dev, &base, &goal, tc).unwrap();
    let (rg, rc) = (g.run_host_loop().unwrap(), c.run_host_loop().unwrap());
    assert_eq!(rg, rc);
    let (a, b) = (g.state_bytes().unwrap(), c.state_bytes().unwrap());
    assert_eq!(a.len(), b.len());
    let diff = a.iter().zip(&b).position(|(x, y)| x != y);
    assert!(diff.is_none(), "first differing byte {diff:?} of {}", a.len());
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn graph_while_rollout_equals_host_loop_without_round_trips() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let h = Studio::new(cuda.clone(), &base, &goal, cfg(256, 16)).unwrap();
    let gr = Studio::new(cuda, &base, &goal, cfg(256, 16)).unwrap();
    h.run_host_loop().unwrap();
    gr.run_graph().unwrap();
    let (a, b) = (h.state_bytes().unwrap(), gr.state_bytes().unwrap());
    assert!(a == b, "graph-mode rollout differs from the host loop");
    let rg = gr.results(4).unwrap();
    assert_eq!(rg.in_loop_readback_bytes, 0, "graph mode must not read back during the rollout");
    assert_eq!(rg.rounds, h.results(1).unwrap().rounds);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn rollout_4096_configurations_throughput_and_soundness() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let c = StudioConfig { configs: 4096, records: 4, max_rounds: 24, coarse: 8, levels: 6, cap: 8192, nodes: 8, budget: 64, tighten: 0, max_cells: 0, policy: 0, island: 16 };
    let st = Studio::new(cuda, &base, &goal, c).unwrap();
    let t0 = Instant::now();
    st.run_graph().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let res = st.results(8).unwrap();
    let cells = c.configs as f64 * res.rounds as f64 * (c.coarse as f64).powi(3); // root cells only
    eprintln!("4096 configurations × {} rounds in {:.3} s (one graph launch): {:.0} config-rounds/s, {:.2e} coarse cells/s; goal met by {}/4096",
        res.rounds, dt, c.configs as f64 * res.rounds as f64 / dt, cells / dt, res.done);
    assert!(res.done > 2048, "goal met by only {}", res.done);
    for sv in &res.top {
        let s = to_layered(&sv.design).unwrap();
        let v = agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0;
        assert!(sv.obs[0] <= v && v <= sv.obs[1], "cfg {}: exact {v} ∉ [{}, {}]", sv.configuration, sv.obs[0], sv.obs[1]);
        assert!(s.brep().validate().valid);
    }
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn new_ops_rect_pattern_fillet_chamfer_bitwise() {
    use agcad_dops::examples::{plate_goal, pocketed_plate};
    let (base, goal) = (pocketed_plate(100.0, 3.0), plate_goal());
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let cpu_dev = agcad_driver_cpu::CpuDevice::new();
    let g = Studio::new(cuda, &base, &goal, cfg(32, 8)).unwrap();
    let c = Studio::new(cpu_dev, &base, &goal, cfg(32, 8)).unwrap();
    assert_eq!(g.run_host_loop().unwrap(), c.run_host_loop().unwrap());
    let (a, b) = (g.state_bytes().unwrap(), c.state_bytes().unwrap());
    let diff = a.iter().zip(&b).position(|(x, y)| x != y);
    assert!(diff.is_none(), "first differing byte {diff:?} of {}", a.len());
    let res = g.results(4).unwrap();
    assert!(res.done >= 1, "goal met by {}", res.done);
    for sv in &res.top {
        let s = to_layered(&sv.design).unwrap();
        let v = agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0;
        assert!(sv.obs[0] <= v && v <= sv.obs[1], "exact {v} ∉ [{}, {}]", sv.obs[0], sv.obs[1]);
    }
}

/// One observe pass per design on both devices: compiled tapes, cell counts and observation
/// rows must be bit-identical (covers shell, bodies and booleans, which the mock policy does
/// not emit itself).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn structural_ops_observe_bitwise() {
    use agcad_dops::abi::fop;
    use agcad_dops::examples::{box_goal, shelled_box, two_bodies};
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let mut g2 = box_goal();
    g2.domain_hi = [100.0, 56.0, 16.0];
    let cases = [(shelled_box(2.0), box_goal()), (two_bodies(fop::ADD, 12.0), g2), (two_bodies(fop::REMOVE, 12.0), g2), (two_bodies(fop::INTERSECT, 9.0), g2)];
    for (d, g) in cases {
        let a = agcad_dops::studio::certify(cuda.clone(), &d, &g, 7, 1 << 16).unwrap();
        let b = agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), &d, &g, 7, 1 << 16).unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{a:?}\n{b:?}");
    }
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn cached_graph_rerun_equals_fresh_and_skips_capture() {
    let (a, b) = (cpu::flange(60.0, 12.0, 6.0), cpu::flange(70.0, 9.0, 8.0));
    let (ga, gb) = (cpu::goal(60_000.0, 66_000.0), cpu::goal(80_000.0, 90_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let t0 = Instant::now();
    let mut cached = agcad_dops::GraphRollout::new(Studio::new(cuda.clone(), &a, &ga, cfg(1024, 12)).unwrap()).unwrap();
    let build = t0.elapsed().as_secs_f64();
    cached.run().unwrap();
    cached.reset(&b, &gb).unwrap();
    let t1 = Instant::now();
    cached.run().unwrap();
    let rerun = t1.elapsed().as_secs_f64();
    let fresh = Studio::new(cuda, &b, &gb, cfg(1024, 12)).unwrap();
    let t2 = Instant::now();
    fresh.run_graph().unwrap();
    let fresh_t = t2.elapsed().as_secs_f64();
    eprintln!("graph build {:.1} ms | cached rerun {:.1} ms | fresh capture+run {:.1} ms", build * 1e3, rerun * 1e3, fresh_t * 1e3);
    assert!(cached.studio.state_bytes().unwrap() == fresh.state_bytes().unwrap(), "cached rerun differs from a fresh rollout");
    assert_eq!(cached.studio.results(1).unwrap().in_loop_readback_bytes, 0);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn two_graph_shards_equal_one_device() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let c = cfg(512, 12);
    let single = Studio::new(cuda.clone(), &base, &goal, c).unwrap();
    single.run_graph().unwrap();
    let one = single.results(8).unwrap();
    let s = agcad_dops::studio::rollout_sharded(&[cuda.clone(), cuda], &base, &goal, c, 8, &|st| st.run_graph()).unwrap();
    assert_eq!(s.done, one.done);
    assert_eq!(s.top.iter().map(|t| t.configuration).collect::<Vec<_>>(), one.top.iter().map(|t| t.configuration).collect::<Vec<_>>());
    for (a, b) in s.top.iter().zip(&one.top) { assert_eq!(a.score.to_bits(), b.score.to_bits()); }
}

/// Engineering probes (moments, area, wall, gap, 3-axis access) are bit-identical on CUDA.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn engineering_probes_bitwise() {
    use agcad_dops::examples::{box_goal, flange, goal, pocketed_plate, plate_goal, shelled_box};
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let on = |mut g: agcad_dops::Goal| { g.min_wall = 3.0; g.min_gap = 12.0; g.machining_3ax = 1; g };
    for (d, g) in [(shelled_box(2.0), on(box_goal())), (flange(60.0, 12.0, 6.0), on(goal(1.0, 2.0))), (pocketed_plate(100.0, 3.0), on(plate_goal()))] {
        let a = agcad_dops::studio::certify(cuda.clone(), &d, &g, 6, 1 << 16).unwrap();
        let b = agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), &d, &g, 6, 1 << 16).unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{a:?}\n{b:?}");
    }
}

/// Device stiffness proxy: CUDA ≡ CPU bitwise (fixed-shape reductions).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn fea_pcg_bitwise() {
    use agcad_dops::fea::{stiffness, LoadCase};
    let d = cpu::flange(60.0, 12.0, 6.0);
    let mut lc = LoadCase::over([-61.0, -61.0, 0.0], [61.0, 61.0, 12.0], 3.0);
    lc.force = [500.0, 0.0, 0.0];
    let a = stiffness(agcad_driver_cuda::CudaDevice::open(0).unwrap(), &d, &lc).unwrap();
    let b = stiffness(agcad_driver_cpu::CpuDevice::new(), &d, &lc).unwrap();
    eprintln!("flange FEA: {} voxels, {} dofs, {} iterations, compliance {}", a.voxels, a.dofs, a.iterations, a.compliance);
    assert!(a.converged);
    assert_eq!(a.compliance.to_bits(), b.compliance.to_bits());
    assert_eq!(a.max_disp.to_bits(), b.max_disp.to_bits());
    assert_eq!((a.iterations, a.voxels, a.dofs), (b.iterations, b.voxels, b.dofs));
}

/// Certified implicit mesh: device sampling is bit-identical, so the meshes are identical.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn implicit_mesh_bitwise() {
    use agcad_dops::examples::shelled_box;
    let d = shelled_box(2.0);
    let (lo, hi) = ([0.0, 0.0, 0.0], [100.0, 60.0, 10.0]);
    let a = agcad_dops::mesh::extract_mesh(agcad_driver_cuda::CudaDevice::open(0).unwrap(), &d, lo, hi, 0.5).unwrap();
    let b = agcad_dops::mesh::extract_mesh(agcad_driver_cpu::CpuDevice::new(), &d, lo, hi, 0.5).unwrap();
    assert!(a.mesh.is_watertight());
    assert_eq!(a.mesh.triangles, b.mesh.triangles);
    assert_eq!(a.volume.to_bits(), b.volume.to_bits());
}

/// 4096 configurations at cap 65 536 (configs × cap = 268 M cells, beyond VRAM) under a
/// 32 M-cell global budget: memory follows the budget, bounds stay sound.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn large_cap_under_global_budget() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let c = StudioConfig { configs: 4096, records: 4, max_rounds: 8, coarse: 8, levels: 7, cap: 65536, nodes: 8, budget: 64, tighten: 0, max_cells: 1 << 25, policy: 0, island: 16 };
    let st = Studio::new(cuda, &base, &goal, c).unwrap();
    let t0 = Instant::now();
    st.run_graph().unwrap();
    let res = st.results(4).unwrap();
    eprintln!("4096 × cap 65536 (budget 32M cells, 512³ finest): {} rounds in {:.2} s, goal met by {}", res.rounds, t0.elapsed().as_secs_f64(), res.done);
    assert!(res.done > 2048);
    for sv in &res.top {
        let v = agcad_geo::tessellate::tessellate(&to_layered(&sv.design).unwrap().brep(), &Default::default()).mass_properties().0;
        assert!(sv.obs[0] <= v && v <= sv.obs[1]);
    }
}

/// CMA-ES islands: CUDA host loop ≡ CPU bitwise, graph WHILE ≡ host loop, and a 4096-config
/// rollout with rounds-to-goal (mock vs cmaes, wrong sign prior).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn cmaes_bitwise_graph_and_scale() {
    use agcad_dops::abi::policy;
    let base = cpu::flange(60.0, 12.0, 6.0);
    let mut g = cpu::goal(80_000.0, 90_000.0);
    g.hole_r = 0.0;
    for j in 0..3 { g.param_sign[j] = -g.param_sign[j]; }
    let c = StudioConfig { policy: policy::CMAES, ..cfg(64, 10) };
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let (a, b) = (Studio::new(cuda.clone(), &base, &g, c).unwrap(), Studio::new(agcad_driver_cpu::CpuDevice::new(), &base, &g, c).unwrap());
    assert_eq!(a.run_host_loop().unwrap(), b.run_host_loop().unwrap());
    assert!(a.state_bytes().unwrap() == b.state_bytes().unwrap(), "cmaes CUDA ≠ CPU");
    let gr = Studio::new(cuda.clone(), &base, &g, c).unwrap();
    gr.run_graph().unwrap();
    assert!(gr.state_bytes().unwrap() == a.state_bytes().unwrap(), "graph ≠ host loop");
    for pol in [policy::MOCK, policy::CMAES] {
        let c = StudioConfig { configs: 4096, records: 4, max_rounds: 24, coarse: 8, levels: 6, cap: 8192, nodes: 8, budget: 96, tighten: 0, max_cells: 0, policy: pol, island: 16 };
        let st = Studio::new(cuda.clone(), &base, &g, c).unwrap();
        let t0 = Instant::now();
        st.run_graph().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let r = st.results(1).unwrap();
        eprintln!("policy {pol}: {} rounds, goal met by {}/4096, {:.2} s ({:.0} config-rounds/s)", r.rounds, r.done, dt, 4096.0 * r.rounds as f64 / dt);
        if pol == policy::CMAES { assert!(r.done > 3500, "{}", r.done); }
    }
}

/// New operations (sketch, slot, revolve, mirror, draft, rounded caps): certify and rollouts
/// bit-identical CUDA ≡ CPU.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn new_ops_bitwise() {
    use agcad_dops::abi::policy;
    use agcad_dops::examples::*;
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let g = ops_goal();
    for d in [slotted_plate(), l_bracket(), stepped_shaft(), coned_shaft(), mirrored_bracket(), drafted_boss(), rounded_block(), user_pattern_plate()] {
        let a = agcad_dops::studio::certify(cuda.clone(), &d, &g, 6, 1 << 17).unwrap();
        let b = agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), &d, &g, 6, 1 << 17).unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{a:?}\n{b:?}");
    }
    let mut gr = ops_goal();
    gr.vol_lo = 12_000.0;
    gr.vol_hi = 14_000.0;
    gr.n_params = 1;
    gr.param_lo[0] = 5.0;
    gr.param_hi[0] = 20.0;
    gr.param_sign[0] = 1.0;
    for pol in [policy::MOCK, policy::CMAES] {
        let c = StudioConfig { policy: pol, ..cfg(32, 8) };
        let (x, y) = (Studio::new(cuda.clone(), &l_bracket(), &gr, c).unwrap(), Studio::new(agcad_driver_cpu::CpuDevice::new(), &l_bracket(), &gr, c).unwrap());
        assert_eq!(x.run_host_loop().unwrap(), y.run_host_loop().unwrap());
        assert!(x.state_bytes().unwrap() == y.state_bytes().unwrap(), "policy {pol}: CUDA ≠ CPU");
        assert!(x.results(1).unwrap().done > 0, "policy {pol}");
    }
}

/// Structural vocabulary (hole, rounded pocket, slot, bolt circle, boss): CUDA ≡ CPU bitwise.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn structural_moves_bitwise() {
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    for (lo, hi, r, zmax) in [(108_000.0, 116_000.0, 9.0, 26.0), (140_000.0, 150_000.0, 13.0, 40.0)] {
        let mut g = cpu::goal(lo, hi);
        g.param_lo[..3].copy_from_slice(&[59.0, 11.5, 5.0]);
        g.param_hi[..3].copy_from_slice(&[60.0, 12.0, 6.0]);
        g.place = [-30.0, -30.0, 30.0, 30.0];
        g.hole_r = r;
        g.domain_hi[2] = zmax;
        g.max_ext[2] = zmax;
        let c = StudioConfig { nodes: 12, ..cfg(32, 16) };
        let base = cpu::flange(60.0, 12.0, 6.0);
        let (a, b) = (Studio::new(cuda.clone(), &base, &g, c).unwrap(), Studio::new(agcad_driver_cpu::CpuDevice::new(), &base, &g, c).unwrap());
        assert_eq!(a.run_host_loop().unwrap(), b.run_host_loop().unwrap());
        assert!(a.state_bytes().unwrap() == b.state_bytes().unwrap(), "structural rollout CUDA ≠ CPU (target {lo})");
    }
}

/// CMA-ES islands that change structure on stall: CUDA ≡ CPU bitwise, graph ≡ host loop.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn structural_cmaes_bitwise() {
    use agcad_dops::abi::policy;
    let base = cpu::flange(60.0, 12.0, 6.0);
    let mut g = cpu::goal(60_000.0, 66_000.0);
    g.param_lo[..3].copy_from_slice(&[50.0, 10.0, 3.0]);
    g.param_hi[..3].copy_from_slice(&[62.0, 12.0, 6.0]);
    g.place = [-28.0, -28.0, 28.0, 28.0];
    g.hole_r = 9.0;
    let c = StudioConfig { records: 6, nodes: 14, policy: policy::CMAES, ..cfg(64, 40) };
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let (x, y) = (Studio::new(cuda.clone(), &base, &g, c).unwrap(), Studio::new(agcad_driver_cpu::CpuDevice::new(), &base, &g, c).unwrap());
    assert_eq!(x.run_host_loop().unwrap(), y.run_host_loop().unwrap());
    assert!(x.state_bytes().unwrap() == y.state_bytes().unwrap(), "structural cmaes CUDA ≠ CPU");
    let gr = Studio::new(cuda, &base, &g, c).unwrap();
    gr.run_graph().unwrap();
    assert!(gr.state_bytes().unwrap() == x.state_bytes().unwrap(), "graph ≠ host loop");
    assert!(x.results(1).unwrap().done > 16);
}
