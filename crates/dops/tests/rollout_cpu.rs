//! Device rollout on the CPU reference device: soundness of the interval bounds against the
//! exact kernel, goal-seeking by the mock GPU model, escalation of survivors to exact
//! documents, and raise (document → device design).

use agcad_dops::lower::{raise, to_layered, to_program};
use agcad_dops::{Design, Studio, StudioConfig};
use agcad_driver_cpu::CpuDevice;

pub use agcad_dops::examples::{flange, goal};

fn exact_volume(d: &Design) -> f64 {
    let s = to_layered(d).unwrap();
    agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0
}

#[test]
fn bounds_contain_exact_volume() {
    let dev = CpuDevice::new();
    for (r, t, n) in [(60.0, 12.0, 6.0), (50.0, 6.0, 3.0), (80.0, 20.0, 12.0), (67.3, 9.1, 7.9)] {
        let d = flange(r, t, n);
        let cfg = StudioConfig { configs: 1, coarse: 8, levels: 6, cap: 16384, max_rounds: 1, ..Default::default() };
        let st = Studio::new(dev.clone(), &d, &goal(1.0, 2.0), cfg).unwrap();
        st.observe().unwrap();
        let s = st.results(1).unwrap();
        let o = s.top[0].obs;
        let v = exact_volume(&d);
        let analytic = core::f64::consts::PI * (r * r - n.floor() * 4.5 * 4.5) * t;
        assert!((v - analytic).abs() / analytic < 2e-3, "tessellated {v} vs analytic {analytic}");
        assert!(o[0] <= analytic && analytic <= o[1], "({r},{t},{n}): exact {analytic} ∉ [{}, {}]", o[0], o[1]);
        assert!((o[12] - analytic).abs() / analytic < 0.02, "estimate {} vs {analytic} (sampled evidence)", o[12]);
        // rigorous extent bounds: true extent 2r in x/y, t in z
        assert!(o[2] <= 2.0 * r && 2.0 * r <= o[5] && o[4] <= t && t <= o[7], "{o:?}");
    }
}

#[test]
fn mock_gpu_model_reaches_goal_and_escalates_exactly() {
    let dev = CpuDevice::new();
    let base = flange(60.0, 12.0, 6.0);
    let g = goal(60_000.0, 66_000.0);
    let cfg = StudioConfig { configs: 8, records: 4, max_rounds: 14, coarse: 8, levels: 5, cap: 4096, nodes: 8, budget: 64, tighten: 0 };
    let st = Studio::new(dev, &base, &g, cfg).unwrap();
    let rounds = st.run_host_loop().unwrap();
    let res = st.results(4).unwrap();
    assert!(res.done >= 4, "only {} of 8 configurations met the goal in {rounds} rounds", res.done);
    assert_eq!(res.in_loop_readback_bytes, 4 * rounds as u64, "host-loop mode reads only the 4-byte flag per round");
    let trace = st.trace().unwrap();
    assert!(trace.iter().any(|t| t.committed == 1), "no commits recorded in the trace ring");
    for sv in &res.top {
        assert_eq!(sv.obs[10], 1.0, "top survivors met the goal: {:?}", sv.obs);
        let v = exact_volume(&sv.design);
        assert!(sv.obs[0] <= v && v <= sv.obs[1], "exact {v} ∉ device bounds [{}, {}]", sv.obs[0], sv.obs[1]);
        assert!(v > 0.97 * g.vol_lo && v < 1.03 * g.vol_hi, "exact {v} far from goal");
        // escalate: the survivor as Part Design features in a real document
        let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut sess = agcad_harness::HarnessSession::new(req, "survivor");
        let prog = agcad_ir::json::decode_program(&to_program(&sv.design, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
        let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
        assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
        let body = sess.last_update.as_ref().unwrap().products.values().rev().find_map(|p| if let agcad_runtime::rebuild::Product::Solid(s) = p { Some(s.brep()) } else { None }).unwrap();
        assert!(body.validate().valid);
        let vb = agcad_geo::tessellate::tessellate(&body, &Default::default()).mass_properties().0;
        assert!((vb - v).abs() / v < 1e-9, "document rebuild {vb} vs device design {v}");
        // raise it back: same design
        let ws = sess.doc.workspace("main").unwrap();
        let (raised, warnings) = raise(&ws.state).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!((exact_volume(&raised) - v).abs() / v < 1e-9);
    }
}

#[test]
fn hard_limit_rejects_and_capacity_is_reported() {
    // max extent below the base design: every geometric edit is rejected, statuses say why
    let dev = CpuDevice::new();
    let base = flange(60.0, 12.0, 6.0);
    let mut g = goal(60_000.0, 66_000.0);
    g.max_ext = [80.0, 80.0, 25.0]; // r ≥ 50 ⇒ extent ≥ 100: every edit definitely violates
    let cfg = StudioConfig { configs: 3, records: 4, max_rounds: 3, coarse: 8, levels: 6, cap: 16384, nodes: 2, budget: 64, tighten: 0 };
    let st = Studio::new(dev, &base, &g, cfg).unwrap();
    st.run_host_loop().unwrap();
    let (cmds, status) = st.last_round().unwrap();
    let param_op = agcad_ir::Catalog::global().get("knowledge.parameter").unwrap().opcode;
    let failed: Vec<u16> = cmds.iter().zip(&status).filter(|(c, _)| c.opcode == param_op).map(|(_, s)| s.status).collect();
    assert!(!failed.is_empty() && failed.iter().all(|s| *s == agcad_driver::AgStatus::RequirementFailed.value()), "{failed:?}");
    let res = st.results(3).unwrap();
    assert_eq!(res.done, 0);
}

#[test]
fn certify_tightens_survivor_bounds() {
    let dev = CpuDevice::new();
    let d = flange(60.0, 12.0, 6.0);
    let g = goal(1.0, 2.0);
    let exact = core::f64::consts::PI * (60.0f64 * 60.0 - 6.0 * 4.5 * 4.5) * 12.0;
    let quick = agcad_dops::studio::certify(dev.clone(), &d, &g, 5, 4096).unwrap();
    let deep = agcad_dops::studio::certify(dev, &d, &g, 7, 1 << 18).unwrap();
    for o in [&quick, &deep] { assert!(o[0] <= exact && exact <= o[1], "{exact} ∉ [{}, {}]", o[0], o[1]); }
    let (wq, wd) = (quick[1] - quick[0], deep[1] - deep[0]);
    eprintln!("quick ±{:.1}% → certified ±{:.1}% (exact {exact:.0}, estimate {:.0})", 50.0 * wq / exact, 50.0 * wd / exact, deep[12]);
    assert!(wd < 0.5 * wq, "certification should at least halve the interval: {wq} → {wd}");
    assert!((deep[12] - exact).abs() / exact < 0.005);
}

#[test]
fn rect_pattern_fillet_chamfer_sound_and_round_trip() {
    use agcad_dops::examples::{plate_goal, pocketed_plate};
    let dev = CpuDevice::new();
    for (w, r) in [(100.0, 3.0), (112.0, 0.5), (120.0, 7.0)] {
        let d = pocketed_plate(w, r);
        let exact = exact_volume(&d);
        let o = agcad_dops::studio::certify(dev.clone(), &d, &plate_goal(), 7, 1 << 17).unwrap();
        assert!(o[0] <= exact && exact <= o[1], "(w {w}, r {r}): exact {exact} ∉ [{}, {}]", o[0], o[1]);
        assert!((o[12] - exact).abs() / exact < 0.01, "estimate {} vs {exact}", o[12]);
        // device → AgIR → document: same exact solid
        let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut sess = agcad_harness::HarnessSession::new(req, "plate");
        let prog = agcad_ir::json::decode_program(&to_program(&d, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
        let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
        assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
        let body = sess.last_update.as_ref().unwrap().products.values().rev().find_map(|p| if let agcad_runtime::rebuild::Product::Solid(s) = p { Some(s.brep()) } else { None }).unwrap();
        assert!(body.validate().valid);
        let vb = agcad_geo::tessellate::tessellate(&body, &Default::default()).mass_properties().0;
        assert!((vb - exact).abs() / exact < 1e-6, "document {vb} vs device design {exact}");
        let (raised, warnings) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!((exact_volume(&raised) - exact).abs() / exact < 1e-9);
    }
}

#[test]
fn body_booleans_sound_and_round_trip() {
    use agcad_dops::abi::fop;
    use agcad_dops::examples::{box_goal, two_bodies};
    let dev = CpuDevice::new();
    for op in [fop::ADD, fop::REMOVE, fop::INTERSECT] {
        let d = two_bodies(op, 12.0);
        let exact = exact_volume(&d);
        let (pi, r2) = (core::f64::consts::PI, 144.0);
        let half_disc_in_plate = 0.5 * pi * r2 * 10.0;
        let want = match op { fop::ADD => 80.0 * 50.0 * 10.0 + pi * r2 * 16.0 - half_disc_in_plate, fop::REMOVE => 80.0 * 50.0 * 10.0 - half_disc_in_plate, _ => half_disc_in_plate };
        assert!((exact - want).abs() / want < 2e-3, "op {op}: exact {exact} vs analytic {want}");
        let mut g = box_goal();
        g.domain_lo = [-6.0, -6.0, -4.0];
        g.domain_hi = [100.0, 56.0, 16.0];
        let o = agcad_dops::studio::certify(dev.clone(), &d, &g, 7, 1 << 17).unwrap();
        assert!(o[0] <= exact && exact <= o[1], "op {op}: exact {exact} ∉ [{}, {}]", o[0], o[1]);
        let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut sess = agcad_harness::HarnessSession::new(req, "bodies");
        let prog = agcad_ir::json::decode_program(&to_program(&d, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
        let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
        assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
        let u = sess.last_update.as_ref().unwrap();
        let last = u.outcomes.last().unwrap().feature;
        let agcad_runtime::rebuild::Product::Solid(s) = &u.products[&last] else { panic!("last feature is not a solid") };
        let vb = agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0;
        assert!((vb - exact).abs() / exact < 1e-6, "op {op}: document {vb} vs device design {exact}");
        let (raised, warnings) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!((exact_volume(&raised) - exact).abs() / exact < 1e-9);
    }
}

#[test]
fn shell_is_exact_sound_and_round_trips() {
    use agcad_dops::abi::{fop, nflags, profile, Node, NO_PARAM, NVALS};
    use agcad_dops::examples::{box_goal, feature, shelled_box};
    let dev = CpuDevice::new();
    for t in [1.0, 2.0, 3.5] {
        let d = shelled_box(t);
        let want = 100.0 * 60.0 * 10.0 - (100.0 - 2.0 * t) * (60.0 - 2.0 * t) * (10.0 - 2.0 * t);
        let o = agcad_dops::studio::certify(dev.clone(), &d, &box_goal(), 7, 1 << 17).unwrap();
        assert!(o[0] <= want && want <= o[1], "t {t}: analytic {want} ∉ [{}, {}]", o[0], o[1]);
        assert!((o[12] - want).abs() / want < 0.02, "t {t}: estimate {} vs {want}", o[12]);
        // exact B-rep path (2D inward offset): volume exact, inside the device bounds
        let v = exact_volume(&d);
        assert!((v - want).abs() < 1e-6, "t {t}: exact shell {v} vs {want}");
        // document round trip: part.shell in the standard library builds the same solid
        let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut sess = agcad_harness::HarnessSession::new(req, "shell");
        let prog = agcad_ir::json::decode_program(&to_program(&d, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
        let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
        assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
        let (raised, warnings) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!((exact_volume(&raised) - want).abs() < 1e-6);
    }
    // plate with a through hole: the hole wall is offset too (ring of radius r + t)
    let plate = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let hole = feature(profile::CIRCLE, fop::REMOVE, [(NO_PARAM, 50.0), (NO_PARAM, 30.0), (NO_PARAM, 8.0), (NO_PARAM, 0.0), (NO_PARAM, -1.0), (NO_PARAM, 11.0)]);
    let mut sh = Node { kind: profile::SHELL, op: fop::ADD, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    sh.val[0] = 1.5;
    let d = Design { nodes: vec![plate, hole, sh], params: vec![], names: vec![] };
    let pi = core::f64::consts::PI;
    let want = (6000.0 - pi * 64.0) * 10.0 - (97.0 * 57.0 - pi * 9.5 * 9.5) * 7.0;
    let v = exact_volume(&d);
    assert!((v - want).abs() / want < 1e-3, "{v} vs {want}");
    let o = agcad_dops::studio::certify(dev.clone(), &d, &box_goal(), 7, 1 << 17).unwrap();
    assert!(o[0] <= v && v <= o[1], "exact {v} ∉ device [{}, {}]", o[0], o[1]);
    // a union body: the device field is not an exact interior distance → honestly device-only
    let pad2 = feature(profile::RECT, fop::ADD, [(NO_PARAM, 50.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let u = Design { nodes: vec![plate, pad2, sh], params: vec![], names: vec![] };
    let e = to_layered(&u).unwrap_err();
    assert!(e.contains("device-only"), "{e}");
}

#[test]
fn reset_reuses_buffers_and_matches_a_fresh_studio() {
    let dev = CpuDevice::new();
    let cfg = StudioConfig { configs: 6, records: 4, max_rounds: 8, coarse: 8, levels: 4, cap: 4096, nodes: 8, budget: 64, tighten: 0 };
    let (a, b) = (flange(60.0, 12.0, 6.0), flange(70.0, 9.0, 8.0));
    let (ga, gb) = (goal(60_000.0, 66_000.0), goal(80_000.0, 90_000.0));
    let mut reused = Studio::new(dev.clone(), &a, &ga, cfg).unwrap();
    reused.run_host_loop().unwrap();
    reused.reset(&b, &gb).unwrap();
    reused.run_host_loop().unwrap();
    let fresh = Studio::new(dev, &b, &gb, cfg).unwrap();
    fresh.run_host_loop().unwrap();
    assert!(reused.state_bytes().unwrap() == fresh.state_bytes().unwrap(), "reset studio differs from a fresh one");
}

#[test]
fn sharded_rollout_equals_single_device() {
    let dev = CpuDevice::new();
    let cfg = StudioConfig { configs: 10, records: 4, max_rounds: 10, coarse: 8, levels: 4, cap: 4096, nodes: 8, budget: 64, tighten: 0 };
    let (base, g) = (flange(60.0, 12.0, 6.0), goal(60_000.0, 66_000.0));
    let single = Studio::new(dev.clone(), &base, &g, cfg).unwrap();
    single.run_host_loop().unwrap();
    let one = single.results(6).unwrap();
    for shards in [2usize, 3] {
        let devs = vec![dev.clone(); shards];
        let s = agcad_dops::studio::rollout_sharded(&devs, &base, &g, cfg, 6, &|st| st.run_host_loop().map(|_| ())).unwrap();
        assert_eq!(s.done, one.done, "{shards} shards");
        let ids: Vec<u32> = s.top.iter().map(|t| t.configuration).collect();
        let want: Vec<u32> = one.top.iter().map(|t| t.configuration).collect();
        assert_eq!(ids, want, "{shards} shards: top-k ids");
        for (a, b) in s.top.iter().zip(&one.top) {
            assert_eq!(a.score.to_bits(), b.score.to_bits());
            assert_eq!(a.design.params, b.design.params);
        }
    }
}
