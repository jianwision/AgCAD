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
