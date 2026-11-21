//! Golden trace (spec 15): the flange study through every layer that exists today, driven the
//! way an LLM drives AgCAD — MCP JSON-RPC over the server. Each `step!` is a numbered
//! assertion point of spec 15. Points that need the GPU-resident feature data plane (which is
//! still the CPU reference) are listed in `PENDING` and printed, never silently passed.
//!
//! Scale: spec 15 uses 16×16×16 = 4,096 configurations; the CPU reference path runs 4×4×4 = 64
//! with the same structure (grid → topology classes by hole count → rank → exact → STEP).

use agcad_harness::Requirements;
use agcad_server::Server;
use serde_json::{json, Value};

const PENDING: &[&str] = &[
    "1.2 udf.compile through NVRTC per program (features use the registered standard library)",
    "1.3 ExecGraph capture with 0 host nodes (graphs/conditionals verified separately in driver-cuda tests)",
    "2.x/4.x feature kernels on the GPU (currently the CPU reference rebuild)",
    "2.5 measure_inertia: batched GPU kernel exists and is parity-tested (primitives::mesh_mass_sums); the study engine still evaluates on the host",
    "5.1 PNG encode on the GPU (host rasterizer)",
    "5.3 host geometry bytes = 0 (the host rebuild touches geometry)",
    "6.3 GPUDirect Storage / single fd write of device bytes",
];

macro_rules! step {
    ($n:literal, $cond:expr, $($msg:tt)*) => {{
        assert!($cond, "golden trace step {}: {}", $n, format!($($msg)*));
        eprintln!("  ✓ {}", $n);
    }};
}

const REQUIREMENTS: &str = r#"{
  "id": "flange-r", "rev": 3, "units": "mm", "process": "machining_3ax",
  "deliver": { "body_type": "SolidBody", "formats": ["step_ap242"] },
  "hard": [ { "id": "R1", "measure": "evaluate.measure_inertia", "field": "volume", "le": 200000.0, "evidence": "bounded" } ],
  "phases": { "update": ["R1"], "save": ["R1"] },
  "budget": { "ops": 1000, "exact_rebuilds": 64 }
}"#;

fn program(expected_revision: u32, requirements_rev: u32) -> Value {
    json!({
      "agir": "0.2", "requirements_rev": requirements_rev, "expected_revision": expected_revision,
      "program": [
        {"id": "p",    "op": "document.edit", "args": {"branch": "main"}},
        {"id": "pr",   "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
        {"id": "pt",   "op": "knowledge.parameter", "args": {"name": "t", "value": "12 mm"}},
        {"id": "pn",   "op": "knowledge.parameter", "args": {"name": "n", "value": 6}},
        {"id": "sk",   "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c",    "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "$r"}},
        {"id": "reg",  "op": "sketch.regions", "args": {"sketch": "%c"}},
        {"id": "pad",  "op": "part.pad", "args": {"profile": "%reg", "limits": "$t", "role": "disc"}},
        {"id": "hole", "op": "part.hole", "args": {"support": "cap_end", "positions": [["=r - 15 mm", "0 mm"]], "spec": {"d": "9 mm", "through": true}}},
        {"id": "pat",  "op": "transform.circular_pattern", "args": {"items": ["%hole"], "axis": "z", "count": "$n", "angle": "360 deg"}},
        {"id": "cfg",  "op": "study.configurations", "args": {"table": {"grid": {"r": ["50 mm", "80 mm", 4], "t": ["6 mm", "20 mm", 4], "n": [3, 4, 6, 8]}}}},
        {"id": "upd",  "op": "study.update_configurations", "args": {"cfgs": "%cfg"}},
        {"id": "sc",   "op": "study.evaluate", "args": {"cfgs": "%cfg", "objectives": [{"minimize": {"op": "evaluate.measure_inertia", "of": "%pat", "field": "volume"}}]}},
        {"id": "top",  "op": "study.rank", "args": {"cfgs": "%cfg", "scores": "%sc", "k": 4}},
        {"id": "chk",  "op": "evaluate.check", "args": {"body": "%pat"}},
        {"id": "cp",   "op": "document.checkpoint", "args": {"label": "top4"}}
      ]
    })
}

fn rpc(s: &mut Server, id: u64, method: &str, params: Value) -> Value {
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).expect("response");
    assert!(r.get("error").is_none(), "{method}: {r}");
    r["result"].clone()
}

/// MCP tools/call; returns the tool's JSON payload (or Err with the tool error text).
fn tool(s: &mut Server, id: u64, name: &str, args: Value) -> Result<Value, String> {
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": args}})).expect("response");
    if let Some(e) = r.get("error") { return Err(e.to_string()); }
    let res = &r["result"];
    let text = res["content"][0]["text"].as_str().unwrap_or("");
    if res["isError"] == true { return Err(text.to_string()); }
    Ok(serde_json::from_str(text).unwrap_or(Value::String(text.into())))
}

/// Exact flange volume: π (r² − n·4.5²) t.
fn exact_volume(r: f64, t: f64, n: f64) -> f64 { core::f64::consts::PI * (r * r - n * 4.5 * 4.5) * t }

struct Outcome { report: Value, step_bytes: Vec<u8>, trace: String }

fn run_session(out_dir: &std::path::Path) -> Outcome {
    let req = Requirements::parse(REQUIREMENTS).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = out_dir.to_path_buf();

    // 0. session setup
    let init = rpc(&mut s, 1, "initialize", json!({}));
    step!("0.1", init["serverInfo"]["name"] == "agcad", "{init}");
    let r = tool(&mut s, 2, "get_requirements", json!({})).unwrap();
    step!("0.1", r["rev"] == 3 && r["process"] == "machining_3ax", "{r}");

    // 1.0 lint before running: no errors, size preview
    let lint = tool(&mut s, 30, "check_program", json!({"program": program(0, 3)})).unwrap();
    step!("1.0", lint["ok"] == true && lint["preview"]["configurations"] == 64, "{lint}");
    let mut broken = program(0, 3);
    broken["program"][7]["args"].as_object_mut().unwrap().remove("limits");
    broken["program"][5]["op"] = json!("sketch.circel");
    let bad = tool(&mut s, 31, "run_program", json!({"program": broken}));
    step!("1.0", bad.as_ref().is_err_and(|e| e.contains("2 error(s)") && e.contains("unknown op 'sketch.circel'") && e.contains("missing required input 'limits'")), "{bad:?}");

    // 1.1 decode + revision checks
    let stale = tool(&mut s, 3, "run_program", json!({"program": program(41, 3)}));
    step!("1.1", stale.as_ref().is_err_and(|e| e.contains("STALE_REVISION")), "{stale:?}");
    let stale_req = tool(&mut s, 4, "run_program", json!({"program": program(0, 2)}));
    step!("1.1", stale_req.as_ref().is_err_and(|e| e.contains("STALE_REQUIREMENTS")), "{stale_req:?}");

    // 2–4. batched update, ranking, exact survivors
    let out = tool(&mut s, 5, "run_program", json!({"program": program(0, 3)})).unwrap();
    let rep = &out["report"];
    step!("2.0", rep["status"] == "ACCEPTED", "{}", serde_json::to_string_pretty(rep).unwrap());
    let st = &rep["study"];
    step!("2.1", st["configurations"] == 64, "{st}");
    step!("2.4", rep["topology_classes"]["classes"] == 4, "one class per hole count: {}", rep["topology_classes"]);
    let top = st["top"].as_array().unwrap();
    step!("3.4", top.len() == 4, "{st}");
    // independent oracle: rank all 64 configurations by exact volume, stable by index
    let grid_r = [50.0, 60.0, 70.0, 80.0];
    let grid_t = [6.0, 6.0 + 14.0 / 3.0, 6.0 + 28.0 / 3.0, 20.0];
    let grid_n = [3.0, 4.0, 6.0, 8.0];
    let mut all: Vec<(f64, usize, f64)> = Vec::new();
    let mut i = 0;
    // configuration order = BTreeMap key order of the grid axes (n, r, t) with the last fastest
    for &n in &grid_n { for &r in &grid_r { for &t in &grid_t { let v = exact_volume(r, t, n); all.push((v, i, v)); i += 1; } } }
    let mut feasible: Vec<(f64, usize, f64)> = all.iter().copied().filter(|x| x.0 <= 200000.0).collect();
    feasible.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let want: Vec<usize> = feasible.iter().take(4).map(|x| x.1).collect();
    let got: Vec<usize> = top.iter().map(|x| x["configuration"].as_u64().unwrap() as usize).collect();
    step!("3.4", got == want, "rank {got:?} vs exact oracle {want:?}");
    for (t, &w) in top.iter().zip(&want) {
        let score = t["score"].as_f64().unwrap();
        step!("2.5", (score - all[w].2).abs() / all[w].2 < 1e-4, "volume {score} vs exact {}", all[w].2);
    }
    step!("4.3", st["fallbacks"].as_array().is_some_and(|f| f.is_empty()), "{}", st["fallbacks"]);
    let tau = st["proxy_fidelity"]["kendall_tau"].as_f64().unwrap_or(f64::NAN);
    step!("4.4", (tau - 1.0).abs() < 1e-12, "τ = {tau}: {}", st["proxy_fidelity"]);
    let chk = rep["checks"].as_array().unwrap();
    step!("3.2", chk.iter().all(|c| c["verdict"] == "pass"), "{chk:?}");

    // 5. renders + specification tree
    let png = tool(&mut s, 6, "render", json!({"view": "iso", "mode": "face_ids"})).unwrap();
    let b64 = png["png_base64"].as_str().unwrap();
    step!("5.1", b64.starts_with("iVBORw0KGgo"), "PNG signature");
    let tree = tool(&mut s, 7, "specification_tree", json!({})).unwrap();
    step!("5.4", tree["features"].as_array().is_some_and(|f| f.len() == 10), "{tree}");
    step!("5.4", tree["parameters"]["r"]["value"] == 60.0 && tree["parameters"]["n"]["value"] == 6.0, "{}", tree["parameters"]);

    // 5.5 structured text view: the 6 bolt holes are reported as holes
    let d = tool(&mut s, 11, "describe", json!({})).unwrap();
    let holes = d["holes"].as_array().unwrap();
    step!("5.5", holes.len() == 6 && holes.iter().all(|h| h["diameter"] == 9.0 && h["length"] == 12.0 && h["faces"] == 1), "{}", d["holes"]);
    step!("5.5", d["cylindrical_walls"][0]["diameter"] == 120.0 && d["parameters"]["n"] == 6.0, "{d}");

    // 6. save + export + reimport
    let rev = tree["revision"].as_u64().unwrap() as u32;
    let save = json!({"agir": "0.2", "requirements_rev": 3, "expected_revision": rev, "program": [{"id": "sv", "op": "document.save_version", "args": {"label": "chosen"}}]});
    tool(&mut s, 8, "run_program", json!({"program": save})).unwrap();
    let head = s.session.doc.branches["main"].head;
    step!("6.1", s.session.doc.versions.len() == 2 && head == 1, "versions {}, head {head}", s.session.doc.versions.len());
    // semantic diff against the version just saved: nothing changed
    let same = tool(&mut s, 12, "describe", json!({"against_version": 1})).unwrap();
    step!("6.1", same["diff"]["unchanged"] == true, "{}", same["diff"]);
    // modify a parameter in place, diff, then restore (exported body stays the saved design)
    let rev = s.session.doc.workspace("main").unwrap().revision;
    let edit = json!({"agir": "0.2", "requirements_rev": 3, "expected_revision": rev, "program": [{"id": "pn", "op": "knowledge.parameter", "args": {"name": "n", "value": 8}}]});
    tool(&mut s, 13, "run_program", json!({"program": edit})).unwrap();
    let d8 = tool(&mut s, 14, "describe", json!({"against_version": 1})).unwrap();
    // 6 → 8 holes on the bolt circle: the holes at 0° and 180° are shared → +6 / −4
    let (added, removed) = (d8["diff"]["holes"]["added"].as_array().map_or(0, Vec::len), d8["diff"]["holes"]["removed"].as_array().map_or(0, Vec::len));
    step!("6.1", d8["current"]["holes"].as_array().map_or(0, Vec::len) == 8 && added == 6 && removed == 4, "+{added} −{removed}: {}", d8["diff"]);
    step!("6.1", d8["diff"]["parameters"]["n"]["to"] == 8.0 && d8["diff"]["volume_delta"].as_f64().is_some_and(|v| v < 0.0), "{}", d8["diff"]);
    step!("6.1", s.session.doc.workspace("main").unwrap().state.features.len() == 10, "parameter edited in place, not duplicated");
    let rev = s.session.doc.workspace("main").unwrap().revision;
    let back = json!({"agir": "0.2", "requirements_rev": 3, "expected_revision": rev, "program": [{"id": "pn", "op": "knowledge.parameter", "args": {"name": "n", "value": 6}}]});
    tool(&mut s, 15, "run_program", json!({"program": back})).unwrap();
    let ex = tool(&mut s, 9, "export", json!({"format": "step_ap242"})).unwrap();
    let path = ex["path"].as_str().unwrap().to_string();
    let step_bytes = std::fs::read(&path).unwrap();
    let txt = String::from_utf8(step_bytes.clone()).unwrap();
    step!("6.2", txt.starts_with("ISO-10303-21;") && txt.to_lowercase().contains("ap242"), "header: {}", &txt[..txt.len().min(300)]);
    let (bodies, _) = agcad_geo::step_read::read_step(&txt, &Default::default()).unwrap();
    step!("6.4", bodies.len() == 1 && bodies[0].validate().valid, "reimport: {} bodies", bodies.len());
    let faces = bodies[0].faces.len();
    // CATIA convention: one seamed cylindrical face per full cylinder → 2 caps + rim + 6 holes.
    step!("6.4", faces == 2 + 1 + 6, "base design n = 6, got {faces} faces");
    if agcad_geo_occt::available() {
        let p = agcad_geo_occt::evaluate_step(std::path::Path::new(&path)).expect("OCCT oracle");
        step!("6.4", p["solids"] == 1, "OCCT: {p}");
    } else {
        eprintln!("  - 6.4 OCCT oracle not installed (python3 -c 'import OCP'); native reimport only");
    }

    let fin = tool(&mut s, 10, "finish", json!({})).unwrap();
    step!("6.5", fin["stopped"] == true && fin["execution_ok"] == true && fin["requirements_met"] == true && fin["delivery_verified"] == true, "{fin}");

    Outcome { report: rep.clone(), step_bytes, trace: s.session.trace.to_jsonl() }
}

/// Remove wall-clock fields (host_ms) so replays compare only deterministic content.
fn strip_time(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(k, _)| !k.ends_with("_ms")).map(|(k, x)| (k.clone(), strip_time(x))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(strip_time).collect()),
        other => other.clone(),
    }
}

#[test]
fn golden_trace_flange_study() {
    eprintln!("golden trace (spec 15):");
    let base = std::env::temp_dir().join(format!("agcad-golden-{}", std::process::id()));
    let a = run_session(&base.join("a"));
    // 7. replay: a fresh session on the same device reproduces outputs bit-identically
    let b = run_session(&base.join("b"));
    step!("7", strip_time(&a.report) == strip_time(&b.report), "reports differ");
    step!("7", a.step_bytes == b.step_bytes, "STEP bytes differ");
    step!("7", a.trace == b.trace, "trace JSONL differs");
    let _ = std::fs::remove_dir_all(&base);
    eprintln!("pending (GPU data plane):");
    for p in PENDING { eprintln!("  - {p}"); }
}

/// The failure mode that is easy to hit as an agent: a requirement demanding stronger evidence
/// than any evaluator provides. It must be flagged up front and explained in the study report.
#[test]
fn unreachable_evidence_is_explained() {
    let req = Requirements::parse(&REQUIREMENTS.replace("\"evidence\": \"bounded\"", "\"evidence\": \"exact\"")).unwrap();
    let mut s = Server::new(req, "flange");
    s.session.out_dir = std::env::temp_dir().join(format!("agcad-golden-ev-{}", std::process::id()));
    let r = tool(&mut s, 1, "get_requirements", json!({})).unwrap();
    let w = r["warnings"].as_array().unwrap();
    assert!(w.len() == 1 && w[0].as_str().unwrap().contains("provides at most 'bounded'"), "{r}");
    let mut p = program(0, 3);
    p["program"][10]["args"]["table"] = json!({"grid": {"r": ["50 mm", "60 mm", 2], "t": ["6 mm"], "n": [3, 4]}});
    let out = tool(&mut s, 2, "run_program", json!({"program": p})).unwrap();
    let st = &out["report"]["study"];
    assert_eq!(st["feasible"], 0, "{st}");
    let causes = st["infeasible_causes"].as_array().unwrap();
    assert_eq!(causes[0]["configurations"], 4, "{st}");
    assert!(causes[0]["cause"].as_str().unwrap().starts_with("R1 unknown: evidence 'bounded' < required 'exact'"), "{st}");
    let _ = std::fs::remove_dir_all(&s.session.out_dir);
}
