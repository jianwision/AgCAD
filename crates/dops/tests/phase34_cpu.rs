//! Phase 3 & Phase 4 integration tests (CPU reference device):
//! - GPU filtered predicates + SoS + AABB broadphase + general non-parallel polyhedral B-rep boolean
//! - GPU prefix-scan STEP encoder byte-identical parity with `write_step`
//! - 3D WORKPLANE frames & analytical TPMS IMPLICIT lattices (Gyroid / Schwarz-P / Diamond)
//! - Reynolds-transport shape gradients & projected-gradient continuous polish
//! - Multi-load FEA von Mises stress proxy & exact negative compliance sensitivity
//! - On-device MOEA/D Pareto rollout (`objective = 4`)
//! - CANTO feature-tape tokenizer + spatial boundary-cell encoder + DAgger BC training + AG-07-005 parity
//! - 2×2 multi-view contact sheet & heatmap rendering

use agcad_dops::abi::*;
use agcad_dops::examples::flange;
use agcad_dops::fea::{stiffness_multi_load, stiffness_with_stress, LoadCase, Precond};
use agcad_dops::grad::{gradient_polish_volume, reynolds_shape_gradients};
use agcad_dops::studio::{certify, Design, Studio, StudioConfig};
use agcad_driver::*;
use agcad_driver_cpu::CpuDevice;
use agcad_geo::builders::{prism, Profile};
use agcad_geo::gpu_exact::{aabb_overlap_pairs_device, classify_points_device, orient2d_sos, orient3d_sos, orient3d_batch_device};
use agcad_geo::kernel::{ExactKernel, NativeKernel, Solid};
use agcad_geo::math::{v3, Aabb, Frame, V3};
use agcad_geo::region2d::BoolOp2;
use agcad_geo::step_gpu::write_step_device;
use agcad_geo::step_write::{write_step, StepOptions};
use agcad_harness::render::{contact_sheet, contact_sheet_device, png, rasterize_heatmap, Mode};
use agcad_model::dagger::{train_mlp_bc, verify_kernel_parity, RolloutDataset, TrainConfig, TrajectorySample};
use agcad_model::encoder::{encode_boundary_cells, BoundaryCellSample};
use agcad_model::tokenizer::{pack_policy_input, tokenize_tape, FeatureTokenInput};
use bytemuck::Zeroable;

#[test]
fn phase3_gpu_predicates_broadphase_general_boolean_and_step_prefix_scan() {
    let dev = CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();

    // 1. Filtered orient3d batch + SoS tie-breaking
    let a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let b = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let c = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    let d = [[0.3, 0.3, -1.0], [0.3, 0.3, -1e-300]];
    let signs = orient3d_batch_device(&*dev, &s, &a, &b, &c, &d).unwrap();
    assert_eq!(signs, vec![1, 1]);
    assert_ne!(orient2d_sos([0.0, 0.0], 0, [1.0, 1.0], 1, [2.0, 2.0], 2), 0);
    assert_ne!(orient3d_sos([0.0, 0.0, 0.0], 0, [1.0, 0.0, 0.0], 1, [0.0, 1.0, 0.0], 2, [0.2, 0.2, 0.0], 3), 0);

    // 2. Spatial AABB pair broadphase
    let ba = [Aabb { min: v3(0.0, 0.0, 0.0), max: v3(2.0, 2.0, 2.0) }, Aabb { min: v3(10.0, 0.0, 0.0), max: v3(12.0, 2.0, 2.0) }];
    let bb = [Aabb { min: v3(1.5, 1.5, 1.5), max: v3(3.0, 3.0, 3.0) }];
    let pairs = aabb_overlap_pairs_device(&*dev, &s, &ba, &bb, 1e-6).unwrap();
    assert_eq!(pairs, vec![(0, 0)]);

    // 3. General non-parallel polyhedral boolean (Z-box minus X-box -> NativeKernel fallback!)
    let box_z = Solid::Brep(prism(&Profile::rectangle(0.0, 0.0, 20.0, 20.0), &Frame::WORLD, 20.0));
    let frame_x = Frame::new(v3(-5.0, 5.0, 5.0), V3::X, V3::Y);
    let box_x = Solid::Brep(prism(&Profile::rectangle(0.0, 0.0, 10.0, 10.0), &frame_x, 30.0));
    let diff = NativeKernel.boolean(&box_z, &box_x, BoolOp2::Difference).unwrap();
    let brep = diff.brep();
    assert!(brep.validate().valid);
    let mesh = agcad_geo::tessellate::tessellate(&brep, &Default::default());
    // 20*20*20 - 20*10*10 = 8000 - 2000 = 6000 mm^3
    assert!((mesh.volume() - 6000.0).abs() < 5.0, "volume={}", mesh.volume());

    // 4. Device ray-parity point classification against the resulting mesh
    let cls = classify_points_device(&*dev, &s, &[v3(2.0, 2.0, 2.0), v3(10.0, 10.0, 10.0), v3(50.0, 50.0, 50.0)], &mesh, 1e-6).unwrap();
    assert_eq!(cls, vec![1, 0, 0]);

    // 5. Prefix-scan GPU STEP encoder matches `write_step` byte-for-byte
    let opt = StepOptions::default();
    let (host_step, host_rep) = write_step(&brep, &opt);
    let (gpu_step, gpu_rep) = write_step_device(&*dev, &s, &brep, &opt).unwrap();
    assert_eq!(gpu_step, host_step.as_bytes());
    assert_eq!(gpu_rep.entities, host_rep.entities);
}

#[test]
fn phase3_workplane_frame_and_tpms_implicit_lattice_are_rigorous() {
    let dev = CpuDevice::new();
    // 1. Box on an arbitrary rotated 3D workplane (normal along X)
    let mut d = Design { nodes: vec![], params: vec![10.0], names: vec!["h".into()], profiles: vec![] };
    let fidx = d.add_workplane_frame([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let mut n = Node::zeroed();
    n.kind = profile::RECT;
    n.op = fop::ADD;
    n.flags = nflags::ACTIVE | nflags::WORKPLANE;
    n.pref = [NO_PARAM; NVALS];
    n.pref[5] = 0;
    n.val[0] = -5.0; n.val[1] = -5.0; n.val[2] = 5.0; n.val[3] = 5.0;
    n.val[4] = 0.0; n.val[5] = 0.0;
    n.val[11] = fidx as f64;
    d.nodes.push(n);

    let mut g = Goal::zeroed();
    g.domain_lo = [-15.0, -15.0, -15.0];
    g.domain_hi = [25.0, 15.0, 15.0];
    g.max_ext = [100.0; 3];
    let obs = certify(dev.clone(), &d, &g, 5, 1 << 14).unwrap();
    // Exact volume = 10 * 10 * 10 = 1000 mm^3
    assert!(obs[0] <= 1000.0 && 1000.0 <= obs[1], "bounds {:?} must bracket 1000", (obs[0], obs[1]));

    // 2. Analytical Gyroid TPMS lattice inside [0, 20]^3
    let mut lat = Design { nodes: vec![], params: vec![1.2], names: vec!["t".into()], profiles: vec![] };
    let mut ln = Node::zeroed();
    ln.kind = profile::IMPLICIT;
    ln.op = fop::ADD;
    ln.flags = nflags::ACTIVE;
    ln.pref = [NO_PARAM; NVALS];
    ln.pref[7] = 0;
    ln.val[0] = 0.0; ln.val[1] = 0.0; ln.val[2] = 20.0; ln.val[3] = 20.0;
    ln.val[4] = 0.0; ln.val[5] = 20.0;
    ln.val[6] = 10.0; // period 10 mm
    ln.val[7] = 0.0;  // half-thickness = params[0]
    ln.val[8] = 0.0;  // Gyroid
    lat.nodes.push(ln);
    let mut lg = Goal::zeroed();
    lg.domain_lo = [-2.0, -2.0, -2.0];
    lg.domain_hi = [22.0, 22.0, 22.0];
    lg.max_ext = [100.0; 3];
    let lobs = certify(dev, &lat, &lg, 5, 1 << 14).unwrap();
    assert!(lobs[0] > 0.0 && lobs[1] < 8000.0 && lobs[0] <= lobs[12] && lobs[12] <= lobs[1], "gyroid bounds={:?}", (lobs[0], lobs[12], lobs[1]));
}

#[test]
fn phase4_reynolds_shape_gradients_and_polish_converge() {
    let (d, g) = flange();
    let grad = reynolds_shape_gradients(&d, &g, 24);
    assert!(grad.boundary_cells > 50);
    // Flange params: [outer_r, thickness, bolt_count] -> dV/dr > 0, dV/dt > 0
    assert!(grad.d_vol[0] > 0.0 && grad.d_vol[1] > 0.0, "d_vol={:?}", grad.d_vol);
    let target = 95_000.0;
    let (polished, hist) = gradient_polish_volume(&d, &g, target, 6, 24);
    let final_g = reynolds_shape_gradients(&polished, &g, 24);
    assert!((final_g.volume - target).abs() < (hist[0] - target).abs(), "polish should move closer to target: start={} final={}", hist[0], final_g.volume);
}

#[test]
fn phase4_fea_von_mises_stress_sensitivity_and_multi_load() {
    let dev = CpuDevice::new();
    let mut n = Node::zeroed();
    n.kind = profile::RECT;
    n.op = fop::ADD;
    n.flags = nflags::ACTIVE;
    n.pref = [NO_PARAM; NVALS];
    n.val[0] = 0.0; n.val[1] = 0.0; n.val[2] = 20.0; n.val[3] = 10.0;
    n.val[4] = 0.0; n.val[5] = 10.0;
    let d = Design { nodes: vec![n], params: vec![], names: vec![], profiles: vec![] };
    let mut lc1 = LoadCase::over([0.0, 0.0, 0.0], [20.0, 10.0, 10.0], 5.0);
    lc1.support_axis = 0;
    lc1.force = [0.0, 0.0, -500.0];
    lc1.precond = Precond::Jacobi;
    let mut lc2 = lc1;
    lc2.force = [0.0, 500.0, 0.0];
    let (fea, stress) = stiffness_with_stress(dev.clone(), &d, &lc1).unwrap();
    assert!(fea.converged && fea.compliance > 0.0);
    assert!(stress.max_von_mises > 0.0 && stress.p99_von_mises <= stress.max_von_mises);
    // Exact compliance sensitivity ∂C/∂ρ_e = -u_eᵀ K_e u_e must be <= 0 everywhere
    assert!(stress.element_sensitivity.iter().all(|&s| s <= 1e-12));
    assert!(stress.element_sensitivity.iter().any(|&s| s < -1e-6));
    let ml = stiffness_multi_load(dev, &d, &[lc1, lc2]).unwrap();
    assert!(ml.all_converged && ml.per_case.len() == 2 && ml.worst_compliance >= fea.compliance);
}

#[test]
fn phase4_moead_pareto_canto_tokenizer_dagger_and_contact_sheet() {
    let dev = CpuDevice::new();
    let (d, mut g) = flange();
    g.objective = 4; // MOEA/D Pareto islands
    g.obj_scale = 120_000.0;
    g.vol_lo = 60_000.0;
    g.vol_hi = 160_000.0;
    let cfg = StudioConfig { configs: 16, nodes: 8, records: 6, max_rounds: 6, coarse: 8, levels: 3, cap: 512, budget: 64, tighten: 0, max_cells: 0, policy: policy::CMAES, island: 4, aniso: true };
    let st = Studio::new(dev.clone(), &d, &g, cfg).unwrap();
    st.run_host_loop().unwrap();
    let res = st.results(4).unwrap();
    assert!(!res.top.is_empty());

    // CANTO feature-tape tokenizer + spatial boundary-cell encoder + DAgger BC training + parity
    let feats: Vec<FeatureTokenInput> = d.nodes.iter().map(|n| FeatureTokenInput { kind: n.kind, op: n.op, flags: n.flags, values: n.val }).collect();
    let toks = tokenize_tape(&feats, 100.0);
    assert_eq!(toks.len(), d.nodes.len());
    let packed = pack_policy_input(&toks, &res.top[0].obs, &[1.0, 2.0, -0.5], OBS);
    assert_eq!(packed.len(), OBS);

    let spatial = encode_boundary_cells(&[
        BoundaryCellSample { pos: [0.0, 0.0, 0.0], normal: [0.0, 0.0, -1.0], area: 10.0, wall: 2.0, severity: 0.1 },
        BoundaryCellSample { pos: [10.0, 5.0, 4.0], normal: [1.0, 0.0, 0.0], area: 8.0, wall: 1.5, severity: 0.8 },
    ], [-10.0, -10.0, -10.0], [20.0, 20.0, 20.0]);
    assert!(spatial[29] > 0.0 && spatial[23] > 0.0);

    let mut ds = RolloutDataset::default();
    for i in 0..12 {
        let x0 = (i as f32 - 6.0) * 0.2;
        ds.push(TrajectorySample { obs: vec![x0, -x0, 0.5, 1.0], action: vec![0.5 * x0, -0.25 * x0], reward: 1.0, committed: true });
    }
    let tr = train_mlp_bc(&ds, TrainConfig { hidden: 16, epochs: 60, lr: 5e-2, weight_decay: 0.0, seed: 42 }).unwrap();
    assert!(tr.final_mse < tr.initial_mse, "DAgger BC must reduce MSE: {} -> {}", tr.initial_mse, tr.final_mse);
    let s = dev.stream(Priority::Normal).unwrap();
    let err = verify_kernel_parity(&*dev, &s, &tr.mlp, &[0.2, -0.2, 0.5, 1.0, -0.4, 0.4, 0.5, 1.0], 2, 1e-5).unwrap();
    assert!(err <= 1e-5);

    // Multi-view 2×2 contact sheet + heatmap render
    let brep = prism(&Profile::rectangle(0.0, 0.0, 20.0, 15.0), &Frame::WORLD, 10.0);
    let mesh = agcad_geo::tessellate::tessellate(&brep, &Default::default());
    let sheet_host = contact_sheet(&mesh, Mode::Heatmap, 32, 32);
    let sheet_gpu = contact_sheet_device(&*dev, &s, &mesh, Mode::Heatmap, 32, 32).unwrap();
    assert_eq!(sheet_host.len(), 64 * 64 * 3);
    assert_eq!(sheet_gpu.len(), 64 * 64 * 3);
    let hm = rasterize_heatmap(&mesh, agcad_harness::render::View::Iso, &vec![0.5f32; mesh.triangles.len()], 0.0, 1.0, 32, 32);
    assert!(png(&hm, 32, 32).starts_with(b"\x89PNG"));
}
