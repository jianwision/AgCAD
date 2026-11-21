//! Device rollout on the GPU: bit parity with the CPU reference, graph-mode (device-driven
//! WHILE) equivalence with zero host round trips, and a 4096-configuration throughput run.
//! Requires an NVIDIA GPU: `cargo test -p agcad-dops --test rollout_cuda -- --ignored --nocapture`.

#[path = "rollout_cpu.rs"]
#[allow(dead_code)]
mod cpu;

use agcad_dops::lower::to_layered;
use agcad_dops::{Studio, StudioConfig};
use std::time::Instant;

fn cfg(configs: u32, rounds: u32) -> StudioConfig { StudioConfig { configs, records: 4, max_rounds: rounds, coarse: 8, levels: 5, cap: 4096, nodes: 8, budget: 64 } }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn cuda_host_loop_matches_cpu_reference_bitwise() {
    let (base, goal) = (cpu::flange(60.0, 12.0, 6.0), cpu::goal(60_000.0, 66_000.0));
    let cuda = agcad_driver_cuda::CudaDevice::open(0).unwrap();
    let cpu_dev = agcad_driver_cpu::CpuDevice::new();
    let g = Studio::new(cuda, &base, &goal, cfg(48, 10)).unwrap();
    let c = Studio::new(cpu_dev, &base, &goal, cfg(48, 10)).unwrap();
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
    let c = StudioConfig { configs: 4096, records: 4, max_rounds: 24, coarse: 8, levels: 6, cap: 8192, nodes: 8, budget: 64 };
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
