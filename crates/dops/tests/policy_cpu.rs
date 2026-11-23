//! The mock GPU model's structural vocabulary: when parameters saturate it adds exact-path
//! features (hole, rounded pocket, slot, 4-hole bolt circle to remove material; boss to add).
//! Survivors stay sound and lower to real documents.

use agcad_dops::abi::{fop, nflags, policy, profile};
use agcad_dops::examples::{flange, goal};
use agcad_dops::lower::{raise, to_layered, to_program};
use agcad_dops::{Design, Studio, StudioConfig};
use agcad_driver_cpu::CpuDevice;

fn exact(d: &Design) -> f64 { agcad_geo::tessellate::tessellate(&to_layered(d).unwrap().brep(), &Default::default()).mass_properties().0 }

fn doc_round_trip(d: &Design) {
    let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut sess = agcad_harness::HarnessSession::new(req, "p");
    let prog = agcad_ir::json::decode_program(&to_program(d, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
    let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
    assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
    let (r, w) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
    assert!(w.is_empty(), "{w:?}");
    let (a, b) = (exact(d), exact(&r));
    assert!((a - b).abs() / a < 1e-9, "{a} vs {b}");
}

fn run(g: &agcad_dops::Goal, configs: u32, rounds: u32) -> agcad_dops::RolloutSummary {
    let base = flange(60.0, 12.0, 6.0);
    let c = StudioConfig { configs, records: 4, max_rounds: rounds, coarse: 8, levels: 5, cap: 4096, nodes: 12, budget: 64, tighten: 0, max_cells: 0, policy: policy::MOCK, island: 16, aniso: true };
    let st = Studio::new(CpuDevice::new(), &base, g, c).unwrap();
    st.run_host_loop().unwrap();
    st.results(configs as usize).unwrap()
}

#[test]
fn saturated_parameters_trigger_every_removal_feature() {
    // parameters nearly pinned: the volume target is reachable only by removing material
    let mut g = goal(108_000.0, 116_000.0);
    g.param_lo[..3].copy_from_slice(&[59.0, 11.5, 5.0]);
    g.param_hi[..3].copy_from_slice(&[60.0, 12.0, 6.0]);
    g.place = [-30.0, -30.0, 30.0, 30.0];
    g.hole_r = 9.0;
    let r = run(&g, 32, 24);
    let (mut hole, mut rounded, mut slot, mut bolt) = (0, 0, 0, 0);
    for sv in &r.top {
        for n in &sv.design.nodes[2..] {
            if n.op != fop::REMOVE { continue; }
            let square = ((n.val[2] - n.val[0]).abs() - (n.val[3] - n.val[1]).abs()).abs() < 1e-9;
            match (n.kind, n.flags & nflags::PATTERN != 0) {
                (profile::CIRCLE, true) => bolt += 1,
                (profile::CIRCLE, false) => hole += 1,
                (profile::RECT, _) if square => rounded += 1,
                (profile::RECT, _) => slot += 1,
                _ => {}
            }
        }
    }
    eprintln!("removal features in survivors: hole {hole}, rounded pocket {rounded}, slot {slot}, bolt circle {bolt}; goal met {}/32", r.done);
    assert!(hole > 0 && rounded > 0 && slot > 0 && bolt > 0, "every structural move is exercised");
    assert!(r.done > 0);
    for sv in r.top.iter().filter(|s| s.obs[10] == 1.0).take(4) {
        let v = exact(&sv.design);
        assert!(sv.obs[0] <= v && v <= sv.obs[1], "exact {v} ∉ [{}, {}]", sv.obs[0], sv.obs[1]);
        doc_round_trip(&sv.design);
    }
}

#[test]
fn saturated_parameters_add_bosses_when_volume_is_short() {
    let mut g = goal(140_000.0, 150_000.0);
    g.param_lo[..3].copy_from_slice(&[59.0, 11.5, 5.0]);
    g.param_hi[..3].copy_from_slice(&[60.0, 12.0, 6.0]);
    g.place = [-30.0, -30.0, 30.0, 30.0];
    g.hole_r = 13.0;
    g.domain_hi[2] = 40.0;
    g.max_ext[2] = 40.0;
    let r = run(&g, 32, 24);
    let bosses: usize = r.top.iter().map(|s| s.design.nodes[2..].iter().filter(|n| n.op == fop::ADD && n.kind == profile::CIRCLE).count()).sum();
    eprintln!("bosses in survivors: {bosses}; goal met {}/32", r.done);
    assert!(bosses > 0 && r.done > 0);
    for sv in r.top.iter().filter(|s| s.obs[10] == 1.0).take(3) {
        let v = exact(&sv.design);
        assert!(sv.obs[0] <= v && v <= sv.obs[1]);
        doc_round_trip(&sv.design);
    }
}
