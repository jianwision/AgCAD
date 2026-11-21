//! # agcad-server — headless MCP server (spec 07 § LLM path)
//!
//! JSON-RPC 2.0 over stdio implementing MCP `initialize`, `tools/list`, `tools/call`.
//! Local by construction (stdio; AG-17-005). Tools: `get_feature_table`, `view_source`,
//! `get_requirements`, `run_program`, `specification_tree`, `query`, `render`, `export`,
//! `finish`. Reports quote untrusted strings as data (AG-17-003).

use agcad_harness::render::{png, rasterize, Mode, View};
use agcad_harness::{HarnessSession, Requirements};
use agcad_ir::Catalog;
use agcad_runtime::rebuild::Product;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub mod device;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

pub struct Server {
    pub session: HarnessSession,
    pub branch: String,
    /// Goal for device rollouts (set by `set_goal`).
    pub goal: Option<(agcad_dops::abi::Goal, Vec<String>)>,
    /// Stiffness requirement checked on survivors.
    pub load: Option<device::LoadSpec>,
    /// Rollouts run so far (names the result branches).
    pub rollouts: u32,
    /// Cached device rollout graphs.
    pub device_cache: device::DeviceCache,
}

fn tool(name: &str, description: &str, schema: Value) -> Value {
    json!({ "name": name, "description": description, "inputSchema": schema })
}

pub fn tools() -> Vec<Value> {
    vec![
        tool("get_feature_table", "Feature table (CATIA names + aliases, signatures, phases). Optional filter by family or substring.", json!({ "type": "object", "properties": { "filter": { "type": "string" } } })),
        tool("view_source", "AgLang source of a standard feature or UDF.", json!({ "type": "object", "properties": { "feature": { "type": "string" } }, "required": ["feature"] })),
        tool("get_requirements", "The immutable requirements of this session.", json!({ "type": "object" })),
        tool("check_program", "Lint an AgIR JSON program without running it: every problem at once with JSON path, signature and a concrete fix, plus a size preview. Formulas: \"=od/2 - 5 mm\"; parameter references: \"$od\"; statement references: \"%id\".", json!({ "type": "object", "properties": { "program": { "type": "object" } }, "required": ["program"] })),
        tool("run_program", "Run an AgIR JSON program on the active workspace. Lints first (same diagnostics as check_program); returns an UpdateReport (and DeliveryReport on document.finish).", json!({ "type": "object", "properties": { "program": { "type": "object" } }, "required": ["program"] })),
        tool("specification_tree", "Specification tree of the active workspace: features, parameters, revision.", json!({ "type": "object" })),
        tool("query", "Resolve element names / face queries on the current body.", json!({ "type": "object", "properties": { "role": { "type": "string" } } })),
        tool("describe", "Structured text view of the current body: bounding box, volume, holes and cylindrical walls by diameter/axis/position, planar faces by normal/offset, parameters. With against_version: semantic diff (holes added/removed, volume delta, parameter changes).", json!({ "type": "object", "properties": { "against_version": { "type": "integer" } } })),
        tool("render", "Offscreen render (PNG, base64) of the current body: views iso/top/front/right, mode shaded/face_ids.", json!({ "type": "object", "properties": { "view": { "type": "string" }, "mode": { "type": "string" } } })),
        tool("export", "Export the current body (step_ap242 | stl | 3mf); returns path written by the control plane.", json!({ "type": "object", "properties": { "format": { "type": "string" } } })),
        tool("set_goal", "Goal for the GPU model: target_volume [min, max] mm³, optional max_extent [x, y, z] mm, parameters {name: {min, max, volume_effect: \"+\"|\"-\"}} it may vary, optional new_holes {r, region [x0, y0, x1, y1]} it may add, optional domain {min, max}. Engineering constraints (sampled on the device; a survivor must satisfy them and edits may not worsen them): min_wall (mm), min_clearance (mm), machining \"3ax\" (tool access from +z; defaults from the requirements' process) with undercut_tolerance (mm²), centroid {min, max}. Stiffness: load {support: \"x_min\"|\"y_min\"|\"z_min\", force [N], voxel mm, E MPa, nu} with max_compliance (N·mm), checked on survivors by a voxel FEA solve on the device. The GPU model then plans and executes edits on many configurations (device_rollout).", json!({ "type": "object", "properties": { "target_volume": { "type": "array" }, "max_extent": { "type": "array" }, "parameters": { "type": "object" }, "new_holes": { "type": "object" }, "domain": { "type": "object" }, "min_wall": { "type": "number" }, "min_clearance": { "type": "number" }, "machining": { "type": "string", "enum": ["3ax", "none"] }, "undercut_tolerance": { "type": "number" }, "centroid": { "type": "object" }, "load": { "type": "object" }, "max_compliance": { "type": "number" } }, "required": ["target_volume", "parameters"] })),
        tool("device_rollout", "Run the GPU model on the current design: `configurations` variants × up to `rounds` rounds of plan → edit → evaluate → check → decide, entirely on the device (CUDA graph) or the CPU reference. Returns the top survivors; with apply_best (default true) the best becomes the active branch as Part Design features (then render / describe / export it).", json!({ "type": "object", "properties": { "configurations": { "type": "integer" }, "rounds": { "type": "integer" }, "top_k": { "type": "integer" }, "apply_best": { "type": "boolean" }, "resolution": { "type": "integer" }, "device": { "type": "string", "enum": ["cuda", "cpu"] } } })),
        tool("finish", "Run delivery checks and produce the DeliveryReport (stop/execution/requirements/delivery).", json!({ "type": "object" })),
    ]
}

/// Values of `knowledge.parameter` features as last rebuilt.
fn parameter_values(state: &agcad_runtime::document::PartState, u: &agcad_runtime::rebuild::UpdateResult) -> std::collections::BTreeMap<String, f64> {
    state.order.iter().filter_map(|id| {
        let f = &state.features[id];
        if f.op != "knowledge.parameter" { return None; }
        let Some(agcad_ir::Value::String(name)) = f.args.get("name") else { return None };
        match u.products.get(id) { Some(Product::Scalar(v)) => Some((name.clone(), *v)), _ => None }
    }).collect()
}

/// Minimal statement for an op: required inputs filled with typed example values.
fn example_statement(o: &agcad_ir::OpManifest) -> Value {
    let mut args = serde_json::Map::new();
    for inp in &o.inputs {
        let Some((name, ty)) = inp.split_once(':') else { continue };
        if ty.ends_with('?') { continue; }
        let ex = agcad_ir::lint::example_for(ty);
        args.insert(name.to_string(), serde_json::from_str(&ex).unwrap_or(Value::String(ex)));
    }
    json!({ "id": o.name.rsplit('.').next().unwrap_or("s"), "op": o.name, "args": args })
}

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 { if k <= c.len() { s.push(T[(n >> (18 - 6 * k)) as usize & 63] as char); } else { s.push('='); } }
    }
    s
}

impl Server {
    pub fn new(requirements: Requirements, part: &str) -> Self {
        Server { session: HarnessSession::new(requirements, part), branch: "main".into(), goal: None, load: None, rollouts: 0, device_cache: Default::default() }
    }

    fn current_mesh(&self) -> Option<agcad_geo::tessellate::Mesh> {
        let u = self.session.last_update.as_ref()?;
        u.products.values().rev().find_map(|p| match p {
            Product::Solid(s) => Some(self.session.kernel.tessellate(s, &Default::default())),
            Product::Mesh(m) => Some(m.clone()),
            _ => None,
        })
    }

    pub fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        let catalog = Catalog::global();
        Ok(match name {
            "get_feature_table" => {
                let f = args.get("filter").and_then(Value::as_str).unwrap_or("");
                let ops: Vec<Value> = catalog.ops.iter().filter(|o| f.is_empty() || o.family == f || o.name.contains(f) || o.aliases.iter().any(|a| a.contains(f)))
                    .map(|o| json!({ "name": o.name, "aliases": o.aliases, "summary": o.summary, "inputs": o.inputs, "outputs": o.outputs, "phase": format!("{:?}", o.phase), "implemented": self.session.registry.get(&o.name).is_some(), "example": example_statement(o) })).collect();
                json!({ "catalog_version": catalog.version, "ops": ops })
            }
            "view_source" => {
                let feat = args.get("feature").and_then(Value::as_str).unwrap_or("");
                json!({ "feature": feat, "source": null, "note": "standard feature sources are registered with their AgLang text as the standard library migrates to #[udf] items; UDF sources are stored in the document" })
            }
            "get_requirements" => {
                let mut v = serde_json::to_value(&self.session.requirements).map_err(|e| e.to_string())?;
                v["warnings"] = json!(self.session.requirements.satisfiability());
                v
            }
            "check_program" => {
                let prog = args.get("program").ok_or("missing program")?;
                serde_json::to_value(agcad_ir::lint::check_program(&prog.to_string(), catalog)).map_err(|e| e.to_string())?
            }
            "run_program" => {
                let prog = args.get("program").ok_or("missing program")?;
                let lint = agcad_ir::lint::check_program(&prog.to_string(), catalog);
                if !lint.ok {
                    let first = lint.diagnostics.first().map(|d| d.status.clone()).unwrap_or_default();
                    return Err(format!("{first}: program has {} error(s); nothing was run\n{}", lint.errors, serde_json::to_string_pretty(&lint).unwrap_or_default()));
                }
                let p = agcad_ir::json::decode_program(&prog.to_string(), catalog).map_err(|e| e.to_string())?;
                let out = self.session.run_program(&self.branch, &p, catalog).map_err(|(s, m)| format!("{}: {m}", s.name()))?;
                json!({ "report": serde_json::to_value(&out.report).map_err(|e| e.to_string())?, "delivery": out.delivery.map(|d| serde_json::to_value(&d).unwrap_or(Value::Null)) })
            }
            "specification_tree" => {
                let ws = self.session.doc.workspace(&self.branch).ok_or("no workspace")?;
                let feats: Vec<Value> = ws.state.order.iter().map(|id| { let f = &ws.state.features[id]; json!({ "feature": agcad_harness::report::untrusted(&f.feature_ref(), "specification_tree"), "active": f.active, "role": f.role }) }).collect();
                // Declared parameters plus knowledge.parameter features (value as last rebuilt).
                let mut params = serde_json::to_value(&ws.state.parameters).unwrap_or(json!({}));
                for id in &ws.state.order {
                    let f = &ws.state.features[id];
                    if f.op != "knowledge.parameter" { continue; }
                    let Some(agcad_ir::Value::String(name)) = f.args.get("name") else { continue };
                    let v = self.session.last_update.as_ref().and_then(|u| match u.products.get(id) { Some(Product::Scalar(v)) => Some(*v), _ => None });
                    if let Some(m) = params.as_object_mut() { m.insert(name.clone(), json!({ "value": v, "feature": f.feature_ref() })); }
                }
                json!({ "revision": ws.revision, "features": feats, "parameters": params })
            }
            "query" => {
                let role = args.get("role").and_then(Value::as_str).unwrap_or("");
                let u = self.session.last_update.as_ref().ok_or("no update yet")?;
                let names: Vec<Value> = u.products.values().rev().find_map(|p| if let Product::Solid(s) = p { Some(s.brep().faces.iter().filter(|f| role.is_empty() || f.name.contains(role)).map(|f| agcad_harness::report::untrusted(&f.name, "element_name")).collect()) } else { None }).unwrap_or_default();
                json!({ "elements": names })
            }
            "describe" => {
                let u = self.session.last_update.as_ref().ok_or("no update yet")?;
                let solid = u.products.values().rev().find_map(|p| if let Product::Solid(s) = p { Some(s.clone()) } else { None }).ok_or("no solid body")?;
                let ws = self.session.doc.workspace(&self.branch).ok_or("no workspace")?;
                let now = agcad_harness::describe::describe(&solid.brep(), &parameter_values(&ws.state, u));
                match args.get("against_version").and_then(Value::as_u64) {
                    None => now,
                    Some(vid) => {
                        let ver = self.session.doc.versions.get(vid as usize).ok_or(format!("no version {vid}"))?;
                        let old = agcad_runtime::rebuild::update(&ver.state, &self.session.registry, self.session.kernel.as_ref(), Default::default(), vec![]);
                        let before = match old.products.values().rev().find_map(|p| if let Product::Solid(s) = p { Some(s.clone()) } else { None }) {
                            Some(s) => agcad_harness::describe::describe(&s.brep(), &parameter_values(&ver.state, &old)),
                            None => json!({}),
                        };
                        json!({ "current": now, "against_version": vid, "diff": agcad_harness::describe::diff(&before, &now) })
                    }
                }
            }
            "render" => {
                let m = self.current_mesh().ok_or("nothing to render")?;
                let view = match args.get("view").and_then(Value::as_str) { Some("top") => View::Top, Some("front") => View::Front, Some("right") => View::Right, _ => View::Iso };
                let mode = if args.get("mode").and_then(Value::as_str) == Some("face_ids") { Mode::FaceIds } else { Mode::Shaded };
                let (w, h) = (512, 512);
                json!({ "png_base64": b64(&png(&rasterize(&m, view, mode, w, h), w, h)), "width": w, "height": h })
            }
            "export" => {
                let fmt = args.get("format").and_then(Value::as_str).unwrap_or("step_ap242");
                let m = self.current_mesh().ok_or("nothing to export")?;
                let _ = std::fs::create_dir_all(&self.session.out_dir);
                let path = self.session.out_dir.join(match fmt { "stl" => "part.stl", "3mf" => "part.3mf", _ => "part.stp" });
                let bytes = match fmt {
                    "stl" => agcad_std::file::stl_binary(&m, "part"),
                    "3mf" => agcad_geo::threemf::write_3mf(&[agcad_geo::threemf::ThreeMfObject { name: "part".into(), mesh: m, beams: vec![], material: None }]),
                    _ => {
                        let u = self.session.last_update.as_ref().ok_or("no update yet")?;
                        let s = u.products.values().rev().find_map(|p| if let Product::Solid(s) = p { Some(s.clone()) } else { None }).ok_or("no solid body")?;
                        agcad_geo::step_write::write_step(&s.brep(), &Default::default()).0.into_bytes()
                    }
                };
                std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
                json!({ "path": path.display().to_string(), "bytes": bytes.len() })
            }
            "set_goal" => {
                let ws = self.session.doc.workspace(&self.branch).ok_or("no workspace")?;
                let (design, warnings) = device::raise_design(&ws.state)?;
                let bbox = self.session.last_update.as_ref().and_then(|u| u.products.values().rev().find_map(|p| if let Product::Solid(s) = p { Some(s.brep().bbox()) } else { None })).map(|b| ([b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z]));
                let p3 = self.session.requirements.process == agcad_harness::requirements::Process::Machining3ax;
                let (mut g, notes) = device::parse_goal(args, &design.names, bbox, p3)?;
                device::freeze_fixed(&mut g, &design.params);
                let load = device::parse_load(args)?;
                self.goal = Some((g, design.names.clone()));
                self.load = load;
                json!({ "ok": true, "parameters": design.names, "device_features": design.nodes.len(), "notes": notes, "raise_warnings": warnings,
                        "goal": { "target_volume": [g.vol_lo, g.vol_hi], "max_extent": g.max_ext, "domain": { "min": g.domain_lo, "max": g.domain_hi },
                                  "min_wall": (g.min_wall > 0.0).then_some(g.min_wall), "min_clearance": (g.min_gap > 0.0).then_some(g.min_gap), "machining_3ax": g.machining_3ax == 1,
                                  "stiffness": load.map(|l| json!({ "support_axis": l.support_axis, "force": l.force, "max_compliance": l.max_compliance })),
                                  "centroid": (g.centroid_hi.iter().zip(&g.centroid_lo).any(|(h, l)| h > l)).then(|| json!({ "min": g.centroid_lo, "max": g.centroid_hi })) } })
            }
            "device_rollout" => {
                let (g, names) = self.goal.clone().ok_or("no goal: call set_goal first")?;
                let ws = self.session.doc.workspace(&self.branch).ok_or("no workspace")?;
                let (design, warnings) = device::raise_design(&ws.state)?;
                if design.names != names { return Err("the design's parameters changed since set_goal; call set_goal again".into()); }
                let configs = args.get("configurations").and_then(Value::as_u64).unwrap_or(256).clamp(1, 65536) as u32;
                let rounds = args.get("rounds").and_then(Value::as_u64).unwrap_or(16).clamp(1, 256) as u32;
                // finest cells per axis (power of two ≥ 8): root 8³, levels = 1 + log2(res / 8)
                let res = args.get("resolution").and_then(Value::as_u64).unwrap_or(128).clamp(8, 1024) as u32;
                let levels = 1 + (res / 8).max(1).ilog2();
                let top_k = args.get("top_k").and_then(Value::as_u64).unwrap_or(4).clamp(1, 64) as usize;
                let cfg = agcad_dops::StudioConfig { configs, records: 4, max_rounds: rounds, coarse: 8, levels, cap: 4096, nodes: (design.nodes.len() as u32 + 8).max(8), budget: 4 * rounds, tighten: 0 };
                let prefer_cuda = match args.get("device").and_then(Value::as_str) { Some("cpu") => false, Some(_) => true, None => std::env::var("AGCAD_DEVICE").map(|v| v != "cpu").unwrap_or(true) };
                let run = device::run(&mut self.device_cache, &design, &g, cfg, top_k, prefer_cuda)?;
                let mut out = device::summarize(&run, &names);
                out["raise_warnings"] = json!(warnings);
                // stiffness escalation on the survivors (voxel PCG on the device)
                let mut stiff_ok = vec![true; run.summary.top.len()];
                if let Some(spec) = self.load {
                    for (i, sv) in run.summary.top.iter().enumerate() {
                        let v = match device::stiffness(&sv.design, &sv.obs, &spec, prefer_cuda) {
                            Ok((r, h)) => {
                                stiff_ok[i] = r.converged && spec.max_compliance.is_none_or(|m| r.compliance <= m);
                                json!({ "compliance": r.compliance, "max_displacement": r.max_disp, "converged": r.converged, "iterations": r.iterations, "voxel": h, "voxels": r.voxels, "ok": stiff_ok[i], "evidence": "sampled" })
                            }
                            Err(e) => { stiff_ok[i] = false; json!({ "error": e }) }
                        };
                        out["top"][i]["stiffness"] = v;
                        if !stiff_ok[i] { out["top"][i]["goal_met"] = json!(false); }
                    }
                }
                if args.get("apply_best").and_then(Value::as_bool).unwrap_or(true) {
                    let pick = run.summary.top.iter().enumerate().find(|(i, s)| s.obs[10] == 1.0 && stiff_ok[*i]).map(|(_, s)| s);
                    if let Some(best) = pick.or(run.summary.top.first()) {
                        // escalation, overlapped: deep certification of the survivor runs on its
                        // own (low-priority) device stream while the host builds the exact solid
                        let (exact, cert) = std::thread::scope(|sc| {
                            let h = sc.spawn(|| agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), &best.design, &g, 7, 1 << 18).ok());
                            let exact = device::exact_volume(&best.design);
                            (exact, h.join().ok().flatten())
                        });
                        self.rollouts += 1;
                        let branch = format!("device-{}", self.rollouts);
                        self.session.doc.branch(0, std::slice::from_ref(&branch));
                        let prog = device::lower_program(&best.design, self.session.requirements.rev, 0);
                        let p = agcad_ir::json::decode_program(&prog.to_string(), catalog).map_err(|e| e.to_string())?;
                        let o = self.session.run_program(&branch, &p, catalog).map_err(|(s, m)| format!("{}: {m}", s.name()))?;
                        self.branch = branch.clone();
                        out["applied"] = json!({ "branch": branch, "configuration": best.configuration, "status": o.report.status, "exact_volume": exact.as_ref().ok(), "exact_error": exact.as_ref().err(),
                            "exact_within_device_bounds": exact.as_ref().map(|v| *v >= best.obs[0] && *v <= best.obs[1]).unwrap_or(false), "checks": o.report.checks,
                            "certified_volume_bounds": cert.map(|c| json!([c[0], c[1]])), "certified_evidence": "bounded" });
                    }
                }
                out
            }
            "finish" => serde_json::to_value(self.session.finish(&self.branch)).map_err(|e| e.to_string())?,
            other => return Err(format!("unknown tool '{other}'")),
        })
    }

    /// Handle one JSON-RPC message; returns the response (None for notifications).
    pub fn handle(&mut self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let result: Result<Value, (i64, String)> = match method {
            "initialize" => Ok(json!({ "protocolVersion": PROTOCOL_VERSION, "capabilities": { "tools": {} }, "serverInfo": { "name": "agcad", "version": env!("CARGO_PKG_VERSION") } })),
            "notifications/initialized" => return None,
            "tools/list" => Ok(json!({ "tools": tools() })),
            "tools/call" => {
                let p = msg.get("params").cloned().unwrap_or(Value::Null);
                let name = p.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let args = p.get("arguments").cloned().unwrap_or(json!({}));
                Ok(match self.call_tool(&name, &args) {
                    Ok(v) => json!({ "content": [{ "type": "text", "text": v.to_string() }], "isError": false }),
                    Err(e) => json!({ "content": [{ "type": "text", "text": e }], "isError": true }),
                })
            }
            "ping" => Ok(json!({})),
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        id.map(|id| match result {
            Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
            Err((code, m)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": m } }),
        })
    }

    /// Serve newline-delimited JSON-RPC on stdin/stdout until EOF.
    pub fn serve_stdio(&mut self) -> std::io::Result<()> {
        let stdin = std::io::stdin();
        let mut out = std::io::stdout().lock();
        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() { continue; }
            let resp = match serde_json::from_str::<Value>(&line) {
                Ok(m) => self.handle(&m),
                Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } })),
            };
            if let Some(r) = resp { writeln!(out, "{r}")?; out.flush()?; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialize_and_list_tools() {
        let req = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
        let mut s = Server::new(req, "p");
        let r = s.handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})).unwrap();
        assert_eq!(r["result"]["serverInfo"]["name"], "agcad");
        let t = s.handle(&json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).unwrap();
        assert!(t["result"]["tools"].as_array().unwrap().len() >= 9);
        assert_eq!(b64(b"Man"), "TWFu");
    }

    #[test]
    fn feature_table_examples_lint_clean() {
        // Every op's example statement must pass check_program except for its placeholder
        // statement references (which are undefined in isolation by construction).
        let c = Catalog::global();
        for o in &c.ops {
            let prog = json!({ "requirements_rev": 1, "expected_revision": 0, "program": [example_statement(o)] });
            let r = agcad_ir::lint::check_program(&prog.to_string(), c);
            let real: Vec<_> = r.diagnostics.iter().filter(|d| d.severity == agcad_ir::lint::Severity::Error && d.status != "STALE_REFERENCE").collect();
            assert!(real.is_empty(), "{}: {real:#?}", o.name);
        }
        assert_eq!(b64(b"Ma"), "TWE=");
    }
}
