//! The LLM ⇄ GPU-model loop over MCP: the LLM builds a design, observes it (describe /
//! render), sets a goal; the GPU model (mock) plans and executes edits on many configurations
//! under the device harness; the best survivor returns as Part Design features on a new branch;
//! the LLM observes again. Runs on the CPU reference device (`device: "cpu"`).

use agcad_harness::Requirements;
use agcad_server::Server;
use serde_json::{json, Value};

fn tool(s: &mut Server, name: &str, args: Value) -> Result<Value, String> {
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})).unwrap();
    let res = &r["result"];
    let text = res["content"][0]["text"].as_str().unwrap_or("").to_string();
    if res["isError"] == true { return Err(text); }
    Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

#[test]
fn llm_sets_goal_gpu_model_executes_llm_observes() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-loop-{}", std::process::id()));
    // 1. LLM: base design with parameters
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "pn", "op": "knowledge.parameter", "args": {"name": "n", "value": 6}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}},
        {"id": "sk2", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "h", "op": "sketch.circle", "args": {"sketch": "%sk2", "mode": "center_radius", "pts": [["=r - 15 mm", "0 mm"]], "r": "4.5 mm"}},
        {"id": "rg2", "op": "sketch.regions", "args": {"sketch": "%h"}},
        {"id": "hole", "op": "part.pocket", "args": {"profile": "%rg2", "limits": {"type": "two_sided", "d": "=t + 1 mm", "d2": "1 mm"}}},
        {"id": "pat", "op": "transform.circular_pattern", "args": {"items": ["%hole"], "axis": "z", "count": "$n", "angle": "360 deg"}}
    ]});
    let out = tool(&mut s, "run_program", json!({"program": prog})).unwrap();
    assert_eq!(out["report"]["status"], "ACCEPTED", "{out}");
    // 2. LLM observes
    let d0 = tool(&mut s, "describe", json!({})).unwrap();
    let v0 = d0["volume"].as_f64().unwrap();
    assert_eq!(d0["holes"].as_array().unwrap().len(), 6, "{d0}");
    // 3. LLM sets the goal: ~45 % less material, same envelope
    let g = tool(&mut s, "set_goal", json!({
        "target_volume": [0.52 * v0, 0.58 * v0], "max_extent": [130, 130, 25],
        "parameters": {"r": {"min": 50, "max": 65, "volume_effect": "+"}, "t": {"min": 6, "max": 20, "volume_effect": "+"}, "n": {"min": 3, "max": 12, "volume_effect": "-"}}
    })).unwrap();
    assert_eq!(g["ok"], true, "{g}");
    assert!(g["raise_warnings"].as_array().unwrap().is_empty(), "{g}");
    // 4. GPU model plans + executes under the device harness
    let r = tool(&mut s, "device_rollout", json!({"configurations": 12, "rounds": 12, "resolution": 128, "device": "cpu"})).unwrap();
    assert!(r["configurations_meeting_goal"].as_u64().unwrap() >= 3, "{r:#}");
    let applied = &r["applied"];
    assert_eq!(applied["status"], "ACCEPTED", "{r:#}");
    assert_eq!(applied["exact_within_device_bounds"], true, "{r:#}");
    let ve = applied["exact_volume"].as_f64().unwrap();
    let cb = applied["certified_volume_bounds"].as_array().unwrap();
    let (cl, ch) = (cb[0].as_f64().unwrap(), cb[1].as_f64().unwrap());
    assert!(cl <= ve && ve <= ch && (ch - cl) / ve < 0.1, "certified [{cl}, {ch}] vs exact {ve}");
    assert!(ve > 0.50 * v0 && ve < 0.60 * v0, "exact {ve} vs goal around {}", 0.55 * v0);
    // 5. LLM observes the result on the new branch, and can still go back
    assert_eq!(s.branch, applied["branch"]);
    let d1 = tool(&mut s, "describe", json!({})).unwrap();
    assert!((d1["volume"].as_f64().unwrap() - ve).abs() / ve < 1e-6, "{d1}");
    assert!(d1["bounding_box"]["size"][0].as_f64().unwrap() <= 130.0 + 1e-6);
    let png = tool(&mut s, "render", json!({"view": "iso"})).unwrap();
    assert!(png["png_base64"].as_str().unwrap().starts_with("iVBORw0KGgo"));
    let tree = tool(&mut s, "specification_tree", json!({})).unwrap();
    assert!(tree["parameters"]["r"]["value"].as_f64().is_some(), "{tree}");
    let _ = std::fs::remove_dir_all(&s.session.out_dir);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn repeated_rollouts_reuse_the_cached_graph() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-loop-cuda-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    let goal = |lo: f64, hi: f64| json!({"target_volume": [lo, hi], "max_extent": [170, 170, 25], "parameters": {"r": {"min": 40, "max": 80, "volume_effect": "+"}, "t": {"min": 6, "max": 20, "volume_effect": "+"}}});
    let roll = json!({"configurations": 256, "rounds": 12, "device": "cuda", "apply_best": false});
    tool(&mut s, "set_goal", goal(90_000.0, 100_000.0)).unwrap();
    let a = tool(&mut s, "device_rollout", roll.clone()).unwrap();
    tool(&mut s, "set_goal", goal(150_000.0, 160_000.0)).unwrap();
    let b = tool(&mut s, "device_rollout", roll).unwrap();
    assert_eq!(a["device"], "cuda", "{a}");
    assert!(b["mode"].as_str().unwrap().contains("cached"), "{b}");
    assert!(b["configurations_meeting_goal"].as_u64().unwrap() > 0, "{b}");
    let v = b["top"][0]["volume_estimate"].as_f64().unwrap();
    assert!((150_000.0..=160_000.0).contains(&v), "second goal not met: {v}");
    eprintln!("first rollout {} ms ({}), cached rerun {} ms ({})", a["wall_ms"], a["mode"], b["wall_ms"], b["mode"]);
}

#[test]
fn engineering_constraints_flow_from_requirements_and_goal_to_survivors() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"},"process":"machining_3ax"}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-eng-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    let g = tool(&mut s, "set_goal", json!({"target_volume": [90_000, 100_000], "max_extent": [170, 170, 25], "min_wall": 4, "centroid": {"min": [-1, -1, 0], "max": [1, 1, 20]},
        "parameters": {"r": {"min": 40, "max": 80, "volume_effect": "+"}, "t": {"min": 6, "max": 20, "volume_effect": "+"}}})).unwrap();
    assert_eq!(g["goal"]["machining_3ax"], true, "process machining_3ax implies the 3-axis check: {g}");
    assert_eq!(g["goal"]["min_wall"], 4.0);
    let r = tool(&mut s, "device_rollout", json!({"configurations": 32, "rounds": 12, "device": "cpu", "resolution": 64, "apply_best": false})).unwrap();
    let best = &r["top"][0];
    assert_eq!(best["goal_met"], true, "{r}");
    let e = &best["engineering"];
    assert_eq!(e["violation_area"], 0.0, "{e}");
    assert_eq!(e["undercut_area_3ax"], 0.0, "{e}");
    assert_eq!(e["min_wall_sampled"], 4.0, "no wall below 4 mm: {e}");
    assert!(e["centroid"][2].as_f64().unwrap() > 2.0 && e["surface_area"].as_f64().unwrap() > 1e4, "{e}");
    assert!(tool(&mut s, "set_goal", json!({"target_volume": [1, 2], "parameters": {}, "machining": "5ax"})).is_err());
}

#[test]
fn stiffness_requirement_is_checked_on_survivors() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-stiff-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    let goal = |maxc: f64| json!({"target_volume": [90_000, 100_000], "max_extent": [170, 170, 25], "load": {"support": "z_min", "force": [1000, 0, 0], "voxel": 6},
        "max_compliance": maxc, "parameters": {"r": {"min": 40, "max": 80, "volume_effect": "+"}, "t": {"min": 6, "max": 20, "volume_effect": "+"}}});
    tool(&mut s, "set_goal", goal(1e9)).unwrap();
    let roll = json!({"configurations": 16, "rounds": 10, "device": "cpu", "resolution": 32, "top_k": 2, "apply_best": false});
    let r = tool(&mut s, "device_rollout", roll.clone()).unwrap();
    let st = &r["top"][0]["stiffness"];
    assert_eq!(st["converged"], true, "{r}");
    assert!(st["compliance"].as_f64().unwrap() > 0.0 && st["voxels"].as_u64().unwrap() > 100, "{st}");
    assert_eq!(r["top"][0]["goal_met"], true);
    // an unreachable stiffness requirement: survivors no longer meet the goal
    tool(&mut s, "set_goal", goal(1e-9)).unwrap();
    let r = tool(&mut s, "device_rollout", roll).unwrap();
    assert_eq!(r["top"][0]["stiffness"]["ok"], false);
    assert_eq!(r["top"][0]["goal_met"], false);
    assert!(tool(&mut s, "set_goal", json!({"target_volume": [1, 2], "parameters": {}, "max_compliance": 1})).is_err());
}

#[test]
fn device_only_survivor_is_delivered_as_certified_implicit() {
    use agcad_dops::examples::{box_goal, shelled_box};
    let d = shelled_box(2.0);
    let im = agcad_server::device::deliver_implicit(&d, &box_goal(), "part.shell (device-only)", false).unwrap();
    let s = im.summary();
    assert_eq!(s["evidence"], "implicit-certified");
    assert_eq!(s["mesh"]["watertight"], true, "{s}");
    let analytic = 100.0 * 60.0 * 10.0 - 96.0 * 56.0 * 6.0;
    let (lo, hi) = (s["certified_volume_bounds"][0].as_f64().unwrap(), s["certified_volume_bounds"][1].as_f64().unwrap());
    assert!(lo <= analytic && analytic <= hi && (hi - lo) / analytic < 0.01, "{s}");
    assert!((s["mesh"]["volume"].as_f64().unwrap() - analytic).abs() / analytic < 0.02, "{s}");
    assert_eq!(agcad_server::device::exact_part(&d).nodes.len(), 1, "document keeps the exact box");
    // export: 3MF / STL from the implicit body; STEP refuses
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut srv = Server::new(req, "box");
    srv.session.out_dir = std::env::temp_dir().join(format!("agcad-implicit-{}", std::process::id()));
    srv.implicit.insert(srv.branch.clone(), im);
    assert!(tool(&mut srv, "export", json!({"format": "3mf"})).unwrap()["bytes"].as_u64().unwrap() > 1000);
    assert!(tool(&mut srv, "export", json!({"format": "stl"})).is_ok());
    let e = tool(&mut srv, "export", json!({"format": "step_ap242"})).unwrap_err();
    assert!(e.contains("implicit"), "{e}");
}

#[test]
fn llm_asks_for_lightest_part_meeting_constraints() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-opt-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    let g = tool(&mut s, "set_goal", json!({"objective": "min_volume", "min_wall": 8, "max_extent": [170, 170, 25],
        "parameters": {"r": {"min": 40, "max": 80}, "t": {"min": 6, "max": 20}}})).unwrap();
    assert_eq!(g["goal"]["objective"], "min_volume", "{g}");
    let r = tool(&mut s, "device_rollout", json!({"configurations": 32, "rounds": 16, "device": "cpu", "resolution": 64, "apply_best": true})).unwrap();
    assert_eq!(r["policy"], "cmaes", "{r}");
    let best = &r["top"][0];
    assert_eq!(best["goal_met"], true, "{r}");
    let (rr, t) = (best["parameters"]["r"].as_f64().unwrap(), best["parameters"]["t"].as_f64().unwrap());
    assert!(rr < 45.0 && (7.9..9.5).contains(&t), "lightest disc with an 8 mm wall: r {rr} t {t}");
    assert_eq!(r["applied"]["status"], "ACCEPTED", "{}", r["applied"]);
    assert_eq!(r["applied"]["exact_within_device_bounds"], true);
}

#[test]
fn llm_sketches_slots_mirrors_and_shafts_reach_the_device() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "bracket");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-ops-{}", std::process::id()));
    // plate with a slot, an arc-cornered pocket, a boss revolved about z, mirrored across x = 0
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "10 mm"}},
        {"id": "s1", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "r1", "op": "sketch.rectangle", "args": {"sketch": "%s1", "mode": "corners", "pts": [["10 mm", "-30 mm"], ["70 mm", "30 mm"]]}},
        {"id": "g1", "op": "sketch.regions", "args": {"sketch": "%r1"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%g1", "limits": "$t"}},
        {"id": "s2", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "sl", "op": "sketch.elongated_hole", "args": {"sketch": "%s2", "mode": "center_to_center", "pts": [["25 mm", "-15 mm"], ["55 mm", "-15 mm"]], "width": "8 mm"}},
        {"id": "g2", "op": "sketch.regions", "args": {"sketch": "%sl"}},
        {"id": "slot", "op": "part.pocket", "args": {"profile": "%g2", "limits": {"type": "two_sided", "d": "11 mm", "d2": "1 mm"}}},
        {"id": "s3", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "l1", "op": "sketch.line", "args": {"sketch": "%s3", "a": ["30 mm", "5 mm"], "b": ["50 mm", "5 mm"]}},
        {"id": "a1", "op": "sketch.arc", "args": {"sketch": "%l1", "mode": "center_start_end", "pts": [["50 mm", "15 mm"], ["50 mm", "5 mm"], ["50 mm", "25 mm"]]}},
        {"id": "l2", "op": "sketch.line", "args": {"sketch": "%a1", "a": ["50 mm", "25 mm"], "b": ["30 mm", "25 mm"]}},
        {"id": "l3", "op": "sketch.line", "args": {"sketch": "%l2", "a": ["30 mm", "25 mm"], "b": ["30 mm", "5 mm"]}},
        {"id": "g3", "op": "sketch.regions", "args": {"sketch": "%l3"}},
        {"id": "pk", "op": "part.pocket", "args": {"profile": "%g3", "limits": {"type": "two_sided", "d": "11 mm", "d2": "-5 mm"}}},
        {"id": "s4", "op": "sketch.create", "args": {"support": "yz"}},
        {"id": "b1", "op": "sketch.profile", "args": {"sketch": "%s4", "pts": [["0 mm", "10 mm"], ["6 mm", "10 mm"], ["6 mm", "18 mm"], ["0 mm", "18 mm"]]}},
        {"id": "g4", "op": "sketch.regions", "args": {"sketch": "%b1"}},
        {"id": "boss", "op": "part.shaft", "args": {"profile": "%g4", "axis": "z", "angles": ["0 deg", "360 deg"]}},
        {"id": "mir", "op": "transform.mirror", "args": {"items": ["%boss"], "plane": [1, 0, 0]}}
    ]});
    let r = tool(&mut s, "run_program", json!({"program": prog})).unwrap();
    assert_eq!(r["report"]["status"], "ACCEPTED", "{r}");
    let g = tool(&mut s, "set_goal", json!({"target_volume": [60_000, 80_000], "parameters": {"t": {"min": 6, "max": 16}}})).unwrap();
    assert_eq!(g["raise_warnings"], json!([]), "every feature reaches the device: {g}");
    assert_eq!(g["device_features"], 5, "{g}");
    let out = tool(&mut s, "device_rollout", json!({"configurations": 16, "rounds": 10, "device": "cpu", "resolution": 64})).unwrap();
    assert_eq!(out["top"][0]["goal_met"], true, "{out}");
    assert_eq!(out["applied"]["status"], "ACCEPTED", "{}", out["applied"]);
    assert_eq!(out["applied"]["exact_within_device_bounds"], true, "{}", out["applied"]);
}

#[test]
fn llm_gets_a_pareto_front_with_renders_and_picks_one() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-pareto-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "pr", "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt", "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "sk", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c", "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "rg", "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%rg", "limits": "$t"}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    // light vs stiff: any volume in a wide band, bending load on the rim
    tool(&mut s, "set_goal", json!({"target_volume": [40_000, 200_000], "max_extent": [170, 170, 25], "load": {"support": "z_min", "force": [1000, 0, 0], "voxel": 6},
        "parameters": {"r": {"min": 40, "max": 80}, "t": {"min": 6, "max": 20}}})).unwrap();
    let r = tool(&mut s, "device_rollout", json!({"configurations": 64, "rounds": 2, "device": "cpu", "resolution": 32, "pareto": {"objectives": ["volume", "compliance"], "k": 4}})).unwrap();
    let p = &r["pareto"];
    let c = p["candidates"].as_array().unwrap();
    assert!(c.len() >= 2, "{p}");
    assert!(r.get("applied").is_none(), "with a Pareto set the LLM picks");
    // a real trade-off: sorted by volume, compliance must not also improve
    let mut v: Vec<(f64, f64)> = c.iter().map(|x| (x["objectives"]["volume"].as_f64().unwrap(), x["objectives"]["compliance"].as_f64().unwrap())).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for w in v.windows(2) { assert!(w[1].1 <= w[0].1 + 1e-12, "non-dominated: lighter must be more compliant {v:?}"); }
    for x in c {
        let path = x["render"]["path"].as_str().unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[1..4], b"PNG", "render {path}");
        assert!(x["render"]["png_base64"].as_str().unwrap().len() > 100);
    }
    let a = tool(&mut s, "device_apply", json!({"candidate": 1})).unwrap();
    assert_eq!(a["applied"]["status"], "ACCEPTED", "{a}");
    assert_eq!(a["applied"]["exact_within_device_bounds"], true, "{a}");
    assert!(tool(&mut s, "device_apply", json!({"candidate": 99})).is_err());
}

#[test]
fn llm_asks_for_a_dfm_report() {
    let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut s = Server::new(req, "plate");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-dfm-{}", std::process::id()));
    let prog = json!({"requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "s1", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "r1", "op": "sketch.rectangle", "args": {"sketch": "%s1", "mode": "corners", "pts": [["0 mm", "0 mm"], ["100 mm", "60 mm"]]}},
        {"id": "g1", "op": "sketch.regions", "args": {"sketch": "%r1"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%g1", "limits": "12 mm"}},
        {"id": "h", "op": "part.hole", "args": {"support": "%pad", "positions": [["50 mm", "30 mm"]], "spec": {"d": "9 mm", "through": true}}},
        {"id": "s2", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "r2", "op": "sketch.rectangle", "args": {"sketch": "%s2", "mode": "corners", "pts": [["10 mm", "10 mm"], ["30 mm", "25 mm"]]}},
        {"id": "g2", "op": "sketch.regions", "args": {"sketch": "%r2"}},
        {"id": "pk", "op": "part.pocket", "args": {"profile": "%g2", "limits": {"type": "two_sided", "d": "13 mm", "d2": "-8 mm"}}}
    ]});
    assert_eq!(tool(&mut s, "run_program", json!({"program": prog})).unwrap()["report"]["status"], "ACCEPTED");
    let r = tool(&mut s, "dfm", json!({"min_wall": 5, "min_clearance": 10})).unwrap();
    // open pocket from the top and a through hole: fully 3-axis machinable once the base is held
    assert!(r["access"]["3ax"]["unreachable_area"].as_f64().unwrap() < 1.0, "{r}");
    assert!(r["access"]["5ax"]["unreachable_area"].as_f64().unwrap() < 1.0, "{r}");
    // pocket floor over 8 mm of material; hole Ø9
    assert!((r["min_wall"].as_f64().unwrap() - 8.0).abs() < 0.05, "{r}");
    assert!((r["min_clearance"].as_f64().unwrap() - 9.0).abs() < 0.05, "{r}");
    assert!(r["narrow_gap_area"].as_f64().unwrap() > 0.0);
    let img = tool(&mut s, "render", json!({"mode": "raytraced"})).unwrap();
    assert!(img["png_base64"].as_str().unwrap().len() > 100, "{img}");
}
