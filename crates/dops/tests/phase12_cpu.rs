//! Phase 1 & Phase 2 integration tests:
//! - Orientation-aware global-worklist anisotropic refinement (soundness & tighter bounds)
//! - Device-resident `CuModel` (`policy::CUMODEL` & `policy::HYBRID`) inside rollout loop
//! - Extended `MAX_PARAMS = 32` parameter capacity
//! - NanoVDB sparse narrow-band extraction + feature-preserving Dual Contouring

use agcad_dops::abi::{ob, policy, MAX_PARAMS, OBS};
use agcad_dops::examples::{box_goal, flange, goal, plate_goal, pocketed_plate, shelled_box};
use agcad_dops::lower::to_layered;
use agcad_dops::mesh::{extract_dual_contour, extract_nanovdb};
use agcad_dops::{Studio, StudioConfig};
use agcad_driver::AgDevice;
use agcad_driver_cpu::CpuDevice;
use agcad_model::cumodel::Mlp;

fn exact_vol(d: &agcad_dops::Design) -> f64 {
    agcad_geo::tessellate::tessellate(&to_layered(d).unwrap().brep(), &Default::default()).mass_properties().0
}

#[test]
fn anisotropic_global_worklist_is_sound_and_tighter_on_thin_plate() {
    let plate = pocketed_plate(100.0, 3.0);
    let g = plate_goal();
    let v = exact_vol(&plate);
    let eval = |aniso: bool| {
        let cfg = StudioConfig { configs: 1, coarse: 8, levels: 6, cap: 4096, max_rounds: 1, aniso, ..Default::default() };
        let st = Studio::new(CpuDevice::new(), &plate, &g, cfg).unwrap();
        st.observe().unwrap();
        let r = st.results(1).unwrap();
        (r.top[0].obs[0], r.top[0].obs[1])
    };
    let (iso_lo, iso_hi) = eval(false);
    let (ani_lo, ani_hi) = eval(true);
    let iso_w = (iso_hi - iso_lo) / v;
    let ani_w = (ani_hi - ani_lo) / v;
    eprintln!("thin plate (cap 4096): iso ±{:.2}% vs aniso ±{:.2}%", 50.0 * iso_w, 50.0 * ani_w);
    assert!(iso_lo <= v && v <= iso_hi, "isotropic unsound");
    assert!(ani_lo <= v && v <= ani_hi, "anisotropic unsound: exact {v} not in [{ani_lo}, {ani_hi}]");
    assert!(ani_w < iso_w, "anisotropic splitting should tighten thin-plate bounds ({ani_w} vs {iso_w})");
}

#[test]
fn resident_cumodel_and_hybrid_policies_drive_rollout_with_zero_in_loop_uploads() {
    let base = flange(60.0, 12.0, 6.0);
    let mut g = goal(80_000.0, 90_000.0);
    g.hole_r = 0.0;
    // Construct a 1-layer linear CuModel [OBS -> 3] that steers r & t down when err > 0
    // (obs_f32[0] = clamped relative volume error).
    let mut w = vec![0.0f32; 3 * OBS];
    w[0 * OBS + 0] = -1.0; // r decreases when volume > target
    w[1 * OBS + 0] = -1.0; // t decreases when volume > target
    w[2 * OBS + 0] = 0.5;  // n increases when volume > target
    let mlp = Mlp { layers: vec![(OBS, 3, w, vec![0.0; 3])] };

    let dev = CpuDevice::new();
    let cfg = StudioConfig { configs: 16, records: 4, max_rounds: 10, coarse: 8, levels: 5, cap: 4096, nodes: 8, budget: 64, policy: policy::CUMODEL, island: 16, aniso: true, ..Default::default() };
    let mut st = Studio::new(dev.clone(), &base, &g, cfg).unwrap();
    st.attach_mlp(&mlp).unwrap();
    // Verify zero host transfers inside a round after ResidentMlp is bound
    dev.reset_transfer_stats();
    st.run_host_loop().unwrap();
    assert_eq!(dev.transfer_stats().geometry_bytes, 0);
    let res = st.results(4).unwrap();
    assert!(res.done > 0, "CuModel policy should meet volume goal (done={})", res.done);
    assert_eq!(res.top[0].obs[ob::MET], 1.0);

    // Hybrid (ResidentMlp + CMA-ES islands)
    let hcfg = StudioConfig { policy: policy::HYBRID, ..cfg };
    let mut hst = Studio::new(CpuDevice::new(), &base, &g, hcfg).unwrap();
    hst.attach_mlp(&mlp).unwrap();
    hst.run_host_loop().unwrap();
    let hres = hst.results(4).unwrap();
    assert!(hres.done > 0, "Hybrid policy should meet volume goal");
}

#[test]
fn supports_more_than_8_parameters() {
    assert!(MAX_PARAMS >= 32);
    let mut base = flange(60.0, 12.0, 6.0);
    let mut g = goal(80_000.0, 90_000.0);
    g.hole_r = 0.0;
    for i in 3..12 {
        base.params.push(i as f64);
        base.names.push(format!("extra_{i}"));
        g.param_lo[i] = i as f64;
        g.param_hi[i] = i as f64;
    }
    g.n_params = 12;
    let cfg = StudioConfig { configs: 4, records: 14, max_rounds: 4, coarse: 8, levels: 4, cap: 2048, nodes: 8, budget: 64, policy: policy::MOCK, island: 4, aniso: true, ..Default::default() };
    let st = Studio::new(CpuDevice::new(), &base, &g, cfg).unwrap();
    st.run_host_loop().unwrap();
    let r = st.results(1).unwrap();
    assert_eq!(r.top[0].design.params.len(), 12);
}

#[test]
fn nanovdb_and_dual_contour_extract_shelled_box() {
    let d = shelled_box(2.0);
    let g = box_goal();
    let grid = extract_nanovdb(CpuDevice::new(), &d, g.domain_lo, g.domain_hi, 1.5, 2.5).unwrap();
    assert!(grid.leaf_count() > 0);
    let roundtrip = agcad_geo::vdb::NanoVdbGrid::from_bytes(&grid.to_bytes()).unwrap();
    assert_eq!(grid, roundtrip);
    let im = extract_dual_contour(CpuDevice::new(), &d, g.domain_lo, g.domain_hi, 1.5).unwrap();
    assert!(!im.mesh.triangles.is_empty());
    assert!(im.mesh.is_watertight(), "dual-contoured shelled box must be watertight");
    let exact = 100.0 * 60.0 * 10.0 - 96.0 * 56.0 * 6.0;
    assert!((im.volume - exact).abs() / exact < 0.08, "dual contour shell vol {} vs {exact}", im.volume);
}
