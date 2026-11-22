//! Device stiffness proxy vs the reference voxel FEA in `agcad_std::physics`.

use agcad_dops::abi::{fop, profile, NO_PARAM};
use agcad_dops::examples::feature;
use agcad_dops::fea::{stiffness, LoadCase};
use agcad_dops::Design;
use agcad_driver_cpu::CpuDevice;

pub fn bar(l: f64, w: f64) -> Design {
    let b = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, w), (NO_PARAM, w), (NO_PARAM, 0.0), (NO_PARAM, l)]);
    Design { nodes: vec![b], params: vec![], names: vec![], profiles: vec![] }
}

#[test]
fn device_pcg_matches_reference_voxel_fea() {
    // 10 × 10 × 40 column, fixed at z = 0, axial load on top
    let d = bar(40.0, 10.0);
    let mut lc = LoadCase::over([0.0, 0.0, 0.0], [10.0, 10.0, 40.0], 2.5);
    lc.force = [0.0, 0.0, -1000.0];
    let r = stiffness(CpuDevice::new(), &d, &lc).unwrap();
    assert!(r.converged, "{r:?}");
    assert_eq!(r.voxels, 4 * 4 * 16);
    // reference: same voxels through agcad_std::physics
    let vx = agcad_std::physics::Voxels { origin: agcad_geo::math::v3(0.0, 0.0, 0.0), h: 2.5, dims: [4, 4, 16], solid: vec![true; 256] };
    let rf = agcad_std::physics::solve_static(&vx, lc.e, lc.nu, 2, agcad_geo::math::v3(0.0, 0.0, -1000.0));
    assert!((r.compliance - rf.compliance).abs() / rf.compliance < 1e-6, "device {} vs reference {}", r.compliance, rf.compliance);
    assert!((r.max_disp - rf.max_disp).abs() / rf.max_disp < 1e-6);
    // axial bar: δ ≈ F L / (E A) (Poisson constraint at the support stiffens slightly)
    let delta = 1000.0 * 40.0 / (lc.e * 100.0);
    assert!(r.max_disp > 0.9 * delta && r.max_disp < 1.3 * delta, "{} vs {delta}", r.max_disp);
}

#[test]
fn thicker_is_stiffer_and_empty_is_reported() {
    let lc = |_w: f64| { let mut l = LoadCase::over([0.0, 0.0, 0.0], [20.0, 20.0, 30.0], 2.5); l.force = [1000.0, 0.0, 0.0]; l };
    let thin = stiffness(CpuDevice::new(), &bar(30.0, 10.0), &lc(10.0)).unwrap();
    let thick = stiffness(CpuDevice::new(), &bar(30.0, 20.0), &lc(20.0)).unwrap();
    assert!(thin.converged && thick.converged);
    assert!(thick.compliance < 0.3 * thin.compliance, "bending stiffness ~ w⁴: {} vs {}", thick.compliance, thin.compliance);
    let none = stiffness(CpuDevice::new(), &Design::default(), &lc(0.0)).unwrap();
    assert_eq!(none.voxels, 0);
    assert_eq!(none.compliance, 0.0);
}

#[test]
fn multigrid_matches_jacobi_in_far_fewer_iterations() {
    use agcad_dops::fea::Precond;
    // slender cantilever (bending: Jacobi-PCG's hard case) and a holed flange
    let cases: Vec<(Design, LoadCase)> = vec![
        ({ bar(60.0, 6.0) }, { let mut l = LoadCase::over([0.0, 0.0, 0.0], [6.0, 6.0, 60.0], 1.0); l.force = [200.0, 0.0, 0.0]; l }),
        (agcad_dops::examples::flange(40.0, 10.0, 5.0), { let mut l = LoadCase::over([-41.0, -41.0, 0.0], [41.0, 41.0, 10.0], 2.0); l.force = [0.0, 0.0, -500.0]; l }),
    ];
    for (d, lc) in cases {
        let mg = stiffness(CpuDevice::new(), &d, &LoadCase { precond: Precond::Multigrid, ..lc }).unwrap();
        let jc = stiffness(CpuDevice::new(), &d, &LoadCase { precond: Precond::Jacobi, ..lc }).unwrap();
        eprintln!("{} dofs: multigrid {} it ({} levels), Jacobi {} it; compliance {} vs {}", mg.dofs, mg.iterations, mg.levels, jc.iterations, mg.compliance, jc.compliance);
        assert!(mg.converged && jc.converged);
        assert!(mg.levels >= 3);
        assert!((mg.compliance - jc.compliance).abs() / jc.compliance < 1e-6);
        assert!((mg.max_disp - jc.max_disp).abs() / jc.max_disp < 1e-6);
        assert!(mg.iterations * 4 < jc.iterations, "{} vs {}", mg.iterations, jc.iterations);
    }
}

/// Wall-clock: multigrid vs Jacobi at growing resolution (`--ignored --nocapture`).
#[test]
#[ignore = "benchmark"]
fn bench_multigrid_vs_jacobi() {
    use agcad_dops::fea::Precond;
    let dev = || -> std::sync::Arc<agcad_driver_cuda::CudaDevice> { agcad_driver_cuda::CudaDevice::open(0).unwrap() };
    let cuda = std::env::var("AGCAD_BENCH_CUDA").is_ok();
    for h in [1.0, 0.5, 0.3] {
        let d = bar(60.0, 6.0);
        let mut lc = LoadCase::over([0.0, 0.0, 0.0], [6.0, 6.0, 60.0], h);
        lc.force = [200.0, 0.0, 0.0];
        for (name, d, lc) in [("cantilever", d, lc), ("flange", agcad_dops::examples::flange(40.0, 10.0, 5.0), { let mut l = LoadCase::over([-41.0, -41.0, 0.0], [41.0, 41.0, 10.0], 2.0 * h); l.force = [0.0, 0.0, -500.0]; l })] {
            for pc in [Precond::Multigrid, Precond::Jacobi] {
                let lc = LoadCase { precond: pc, ..lc };
                let w = std::time::Instant::now();
                let r = if cuda { stiffness(dev(), &d, &lc).unwrap() } else { stiffness(CpuDevice::new(), &d, &lc).unwrap() };
                eprintln!("{name:10} h {h:4} {:8} dofs {pc:?}: {:5} it, {:8.1} ms, converged {}", r.dofs, r.iterations, w.elapsed().as_secs_f64() * 1e3, r.converged);
            }
        }
    }
}
