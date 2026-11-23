//! On-device CMA-ES islands as the GPU model's policy (CPU reference device).

use agcad_dops::abi::{ob, policy, OptState, MAX_PARAMS};
use agcad_dops::examples::{flange, goal};
use agcad_dops::lower::to_layered;
use agcad_dops::reference::{island_update, normal12};
use agcad_dops::{Goal, Studio, StudioConfig};
use agcad_driver_cpu::CpuDevice;
use bytemuck::Zeroable;

fn cfg(configs: u32, rounds: u32, pol: u32) -> StudioConfig {
    StudioConfig { configs, records: 4, max_rounds: rounds, coarse: 8, levels: 5, cap: 4096, nodes: 8, budget: 4 * rounds, tighten: 0, max_cells: 0, policy: pol, island: 16, aniso: true }
}

fn exact(d: &agcad_dops::Design) -> f64 { agcad_geo::tessellate::tessellate(&to_layered(d).unwrap().brep(), &Default::default()).mass_properties().0 }

#[test]
fn island_update_converges_on_a_quadratic() {
    let mut g = Goal::zeroed();
    g.n_params = 3;
    for j in 0..3 { g.param_lo[j] = 0.0; g.param_hi[j] = 1.0; }
    let mut st = OptState::zeroed();
    st.mean[..3].copy_from_slice(&[0.9, 0.1, 0.5]);
    st.diag = [1.0; MAX_PARAMS];
    st.sigma = 0.3;
    st.best = 1e30;
    let lam = 16;
    for gi in 0..80u32 {
        let mut samp = vec![0.0; lam * MAX_PARAMS];
        let mut fs = vec![0.0; lam];
        for k in 0..lam {
            let mut f = 0.0;
            for j in 0..3 {
                let u = (st.mean[j] + st.sigma * st.diag[j].sqrt() * normal12(7, k as u32, gi * 16 + j as u32)).clamp(0.0, 1.0);
                samp[k * MAX_PARAMS + j] = u;
                f += (u - [0.3, 0.6, 0.45][j]).powi(2) * [1.0, 10.0, 100.0][j];
            }
            fs[k] = f;
        }
        island_update(&mut st, &samp, &fs, &[0.0; 16], &g, 3, 0, gi);
    }
    for (j, want) in [0.3, 0.6, 0.45].iter().enumerate() { assert!((st.mean[j] - want).abs() < 0.02, "mean {:?}", &st.mean[..3]); }
}

#[test]
fn cmaes_succeeds_where_the_sign_prior_is_wrong() {
    let mut g = goal(80_000.0, 90_000.0);
    g.hole_r = 0.0; // no structural moves: parameters only
    for j in 0..3 { g.param_sign[j] = -g.param_sign[j]; } // the LLM guessed every effect backwards
    let base = flange(60.0, 12.0, 6.0);
    let run = |pol: u32| {
        let st = Studio::new(CpuDevice::new(), &base, &g, cfg(48, 16, pol)).unwrap();
        st.run_host_loop().unwrap();
        st.results(4).unwrap()
    };
    let (mock, cma) = (run(policy::MOCK), run(policy::CMAES));
    eprintln!("wrong sign prior: mock met {}/48, cmaes met {}/48", mock.done, cma.done);
    assert!(cma.done >= 36, "cmaes met only {}", cma.done);
    assert!(cma.done > mock.done);
    for sv in &cma.top {
        assert_eq!(sv.obs[ob::MET], 1.0);
        let v = exact(&sv.design);
        assert!(sv.obs[0] <= v && v <= sv.obs[1] && (80_000.0..=90_000.0).contains(&sv.obs[ob::VOL_EST]));
    }
}

#[test]
fn min_volume_objective_under_a_wall_constraint() {
    let mut g = goal(0.0, 0.0); // no target interval: minimize
    g.hole_r = 0.0;
    g.objective = 1;
    g.obj_scale = 131_000.0;
    g.min_wall = 7.0;
    let base = flange(60.0, 12.0, 6.0);
    let st = Studio::new(CpuDevice::new(), &base, &g, cfg(64, 24, policy::CMAES)).unwrap();
    st.run_host_loop().unwrap();
    let res = st.results(4).unwrap();
    let best = &res.top[0];
    let (r, t, n) = (best.design.params[0], best.design.params[1], best.design.params[2]);
    eprintln!("min volume s.t. wall ≥ 7: r {r:.2} t {t:.2} n {n:.2}, volume ≈ {:.0} (min wall sampled {})", best.obs[ob::VOL_EST], best.obs[ob::MIN_WALL]);
    assert_eq!(best.obs[ob::MET], 1.0, "feasible");
    assert!(best.obs[ob::THIN_AREA] == 0.0 && t >= 6.9, "wall respected: t {t}");
    // optimum r = 50, t = 7, n = 12: π (2500 − 12·4.5²) · 7 ≈ 49 600 mm³
    assert!(best.obs[ob::VOL_EST] < 56_000.0, "{}", best.obs[ob::VOL_EST]);
    let v = exact(&best.design);
    assert!(best.obs[0] <= v && v <= best.obs[1]);
    assert!(res.done == 0, "objective mode never finishes early");
}

#[test]
fn cmaes_sharding_keeps_islands_whole() {
    let mut g = goal(80_000.0, 90_000.0);
    g.hole_r = 0.0;
    let base = flange(60.0, 12.0, 6.0);
    let c = cfg(48, 10, policy::CMAES);
    let single = Studio::new(CpuDevice::new(), &base, &g, c).unwrap();
    single.run_host_loop().unwrap();
    let one = single.results(6).unwrap();
    for shards in [2usize, 3] {
        let devs = vec![CpuDevice::new(); shards];
        let s = agcad_dops::studio::rollout_sharded(&devs, &base, &g, c, 6, &|st| st.run_host_loop().map(|_| ())).unwrap();
        assert_eq!(s.done, one.done, "{shards} shards");
        assert_eq!(s.top.iter().map(|t| (t.configuration, t.score.to_bits())).collect::<Vec<_>>(), one.top.iter().map(|t| (t.configuration, t.score.to_bits())).collect::<Vec<_>>());
    }
}

#[test]
fn cmaes_changes_structure_when_parameters_cannot_reach_the_goal() {
    let base = flange(60.0, 12.0, 6.0);
    // smallest parametric volume ≈ π(2500 − 6·20.25)·10 ≈ 74.7k: the target needs material removed
    let mk = |hole_r: f64| {
        let mut g = goal(60_000.0, 66_000.0);
        g.param_lo[..3].copy_from_slice(&[50.0, 10.0, 3.0]);
        g.param_hi[..3].copy_from_slice(&[62.0, 12.0, 6.0]);
        g.place = [-28.0, -28.0, 28.0, 28.0];
        g.hole_r = hole_r;
        g
    };
    let run = |g: &Goal| {
        let c = StudioConfig { records: 10, nodes: 14, ..cfg(64, 40, policy::CMAES) };
        let st = Studio::new(CpuDevice::new(), &base, g, c).unwrap();
        st.run_host_loop().unwrap();
        st.results(4).unwrap()
    };
    let (plain, structural) = (run(&mk(0.0)), run(&mk(9.0)));
    eprintln!("unreachable by parameters: cmaes {}/64, cmaes + structure {}/64", plain.done, structural.done);
    assert_eq!(plain.done, 0);
    assert!(structural.done >= 16, "{}", structural.done);
    let mut tuned = 0;
    for sv in structural.top.iter().filter(|s| s.obs[ob::MET] == 1.0) {
        assert!(sv.design.nodes.len() > 2, "structure was added");
        let v = exact(&sv.design);
        assert!(sv.obs[0] <= v && v <= sv.obs[1], "exact {v} ∉ [{}, {}]", sv.obs[0], sv.obs[1]);
        // added features reference structural parameters (s{k}_r / _x / _y) ...
        let refs: Vec<u32> = sv.design.nodes[2..].iter().flat_map(|n| n.pref.iter().copied()).filter(|&p| p != agcad_dops::abi::NO_PARAM && p >= 3).collect();
        assert!(!refs.is_empty(), "structural features are parametric");
        for &p in &refs { assert!(sv.design.names[p as usize].starts_with('s'), "{}", sv.design.names[p as usize]); }
        // ... that CMA-ES moved away from their nominal values
        if refs.iter().any(|&p| { let j = p as usize; let mid = 0.5 * (mk(9.0).param_lo[j] + mk(9.0).param_hi[j]); (sv.design.params[j] - mid).abs() > 1e-6 }) { tuned += 1; }
        // lowered documents keep them as parameters and round-trip
        let prog = agcad_dops::lower::to_program(&sv.design, 1, 0).to_string();
        for &p in &refs { let nm = &sv.design.names[p as usize]; assert!(prog.contains(&format!("${nm}")) || prog.contains(&format!("={nm}")), "{nm} stays a parameter in the document"); }
        let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut sess = agcad_harness::HarnessSession::new(req, "p");
        let out = sess.run_program("main", &agcad_ir::json::decode_program(&prog, agcad_ir::Catalog::global()).unwrap(), agcad_ir::Catalog::global()).unwrap();
        assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
        let (r, w) = agcad_dops::lower::raise(&sess.doc.workspace("main").unwrap().state).unwrap();
        assert!(w.is_empty(), "{w:?}");
        assert!((exact(&r) - v).abs() / v < 1e-9);
    }
    assert!(tuned > 0, "CMA-ES tuned the added structure");
}
