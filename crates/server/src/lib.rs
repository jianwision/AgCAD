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
use agcad_dops::abi::ob;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub mod device;
pub mod pareto;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

pub struct Server {
    pub session: HarnessSession,
    pub branch: String,
    /// Goal for device rollouts (set by `set_goal`).
    pub goal: Option<(agcad_dops::abi::Goal, Vec<String>)>,
    /// Stiffness requirement checked on survivors.
    pub load: Option<device::LoadSpec>,
    /// Ray-cast DFM requirement checked on survivors.
    pub dfm: Option<device::DfmSpec>,
    /// Rollouts run so far (names the result branches).
    pub rollouts: u32,
    /// Certified implicit bodies of branches whose survivor has device-only ops (no B-rep).
    pub implicit: std::collections::HashMap<String, device::Implicit>,
    /// Cached device rollout graphs.
    pub device_cache: device::DeviceCache,
    /// Pareto candidates of the last rollout (`device_apply` picks one).
    pub candidates: Vec<agcad_dops::Survivor>,
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
        tool("set_goal", "Goal for the GPU model: target_volume [min, max] mm³, optional max_extent [x, y, z] mm, parameters {name: {min, max, volume_effect: \"+\"|\"-\"}} it may vary, optional new_holes {r, region [x0, y0, x1, y1]} it may add, optional domain {min, max}. objective \"satisfy\" (default: reach target_volume) | \"min_volume\" | \"max_volume\" (optimize subject to every constraint; target_volume optional). Engineering constraints (sampled on the device; a survivor must satisfy them and edits may not worsen them): min_wall (mm), min_clearance (mm), machining \"3ax\" (tool access from +z; defaults from the requirements' process) with undercut_tolerance (mm²), centroid {min, max}. Stiffness: load {support: \"x_min\"|\"y_min\"|\"z_min\", force [N], voxel mm, E MPa, nu} with max_compliance (N·mm), checked on survivors by a voxel FEA solve on the device. The GPU model then plans and executes edits on many configurations (device_rollout).", json!({ "type": "object", "properties": { "target_volume": { "type": "array" }, "max_extent": { "type": "array" }, "parameters": { "type": "object" }, "new_holes": { "type": "object" }, "domain": { "type": "object" }, "min_wall": { "type": "number" }, "min_clearance": { "type": "number" }, "machining": { "type": "string", "enum": ["3ax", "none"] }, "undercut_tolerance": { "type": "number" }, "centroid": { "type": "object" }, "load": { "type": "object" }, "max_compliance": { "type": "number" }, "objective": { "type": "string", "enum": ["satisfy", "min_volume", "max_volume"] } }, "required": ["parameters"] })),
        tool("dfm", "Design-for-manufacture ray analysis of the current body (RTX ray tracing when available, exact CPU reference otherwise; evidence sampled, one sample per tessellation triangle): tool access for direction sets 3ax (+z, the fixture face excluded), 3+2 (±x ±y ±z) and 5ax (26 directions) — reachable / unreachable area; minimum wall thickness and clearance along surface normals, with the area below optional limits min_wall / min_clearance (mm).", json!({ "type": "object", "properties": { "min_wall": { "type": "number" }, "min_clearance": { "type": "number" }, "sets": { "type": "array" } } })),
        tool("device_apply", "Apply Pareto candidate `candidate` (index into the last device_rollout's pareto list) as a new branch of Part Design features, with exact / certified evidence.", json!({ "type": "object", "properties": { "candidate": { "type": "integer" } }, "required": ["candidate"] })),
        tool("device_rollout", "Run the GPU model on the current design: `configurations` variants × up to `rounds` rounds of plan → edit → evaluate → check → decide, entirely on the device (CUDA graph) or the CPU reference. policy \"cmaes\" (default when parameters are free and no new_holes: islands of `island` configurations run CMA-ES on the device) or \"mock\" (sign-prior controller with structural moves). Returns the top survivors; with apply_best (default true) the best becomes the active branch as Part Design features (then render / describe / export it). With pareto {objectives: [\"volume\", \"compliance\", …] (\"-\" prefix maximizes; keys volume, surface_area, extent, undercut, thin_wall, centroid_z, compliance), k} it returns up to k feasible designs trading the objectives off (non-dominated, diverse), each with a render; pick one with device_apply.", json!({ "type": "object", "properties": { "configurations": { "type": "integer" }, "rounds": { "type": "integer" }, "top_k": { "type": "integer" }, "apply_best": { "type": "boolean" }, "pareto": { "type": "object" }, "resolution": { "type": "integer" }, "device": { "type": "string", "enum": ["cuda", "cpu"] }, "policy": { "type": "string", "enum": ["cmaes", "mock"] }, "island": { "type": "integer" } } })),
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

/// Ray-cast DFM numbers of a body's tessellation.
pub struct Dfm { pub tracer: &'static str, pub fallback: Option<String>, pub samples: usize, pub area: f64, pub unreachable: Vec<(String, f64, f64, usize)>, pub wall: agcad_raytrace::analysis::Distances, pub gap: agcad_raytrace::analysis::Distances, pub ms: f64 }

impl Dfm {
    pub fn unreachable(&self, set: &str) -> Option<f64> { self.unreachable.iter().find(|u| u.0 == set).map(|u| u.1) }
}

/// Tool access for `sets` (3ax excludes the fixture face), wall thickness and clearance.
pub fn dfm_analyse(m: &agcad_geo::tessellate::Mesh, sets: &[String]) -> Result<Dfm, String> {
    use agcad_raytrace::analysis::{access, along_normal, directions, samples};
    let (t, why) = agcad_raytrace::best(m);
    let t0 = std::time::Instant::now();
    // the fixture face: downward-facing samples on the part's lowest plane (held by the vise)
    let s = samples(m);
    let zmin = m.positions.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
    let base: f64 = s.c.iter().zip(&s.n).zip(&s.a).filter(|((c, n), _)| n.z < -1.0 + 1e-6 && (c.z - zmin).abs() < 1e-6).map(|(_, a)| a).sum();
    let mut unreachable = Vec::new();
    for set in sets {
        let dirs = directions(set).ok_or(format!("unknown direction set '{set}' (3ax, 3+2, 5ax)"))?;
        let a = access(&*t, m, &dirs)?;
        let fixture = if set == "3ax" { base } else { 0.0 };
        unreachable.push((set.clone(), (a.unreachable - fixture).max(0.0), fixture, a.dirs.len()));
    }
    let wall = along_normal(&*t, m, true)?;
    let gap = along_normal(&*t, m, false)?;
    Ok(Dfm { tracer: t.name(), fallback: why, samples: s.c.len(), area: s.area, unreachable, wall, gap, ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// JSON of a DFM result (limits add the areas below them).
pub fn dfm_json(d: &Dfm, min_wall: Option<f64>, min_clearance: Option<f64>) -> Value {
    let fin = |x: f64| if x.is_finite() { json!((x * 1e4).round() / 1e4) } else { Value::Null };
    let acc: serde_json::Map<String, Value> = d.unreachable.iter().map(|(set, u, fx, n)| (set.clone(), json!({ "unreachable_area": (u * 100.0).round() / 100.0, "unreachable_fraction": u / d.area, "directions": n, "fixture_area_excluded": fx }))).collect();
    let mut out = json!({ "tracer": d.tracer, "fallback_reason": d.fallback, "samples": d.samples, "surface_area": d.area, "access": acc,
        "min_wall": fin(d.wall.min()), "min_clearance": fin(d.gap.min()), "ms": (d.ms * 10.0).round() / 10.0, "evidence": "sampled" });
    if let Some(l) = min_wall { out["thin_wall_area"] = json!(d.wall.area_below(l)); }
    if let Some(l) = min_clearance { out["narrow_gap_area"] = json!(d.gap.area_below(l)); }
    out
}

/// DFM report of a body's tessellation (see the `dfm` tool).
pub fn dfm_report(m: &agcad_geo::tessellate::Mesh, args: &Value) -> Result<Value, String> {
    let sets: Vec<String> = match args.get("sets").and_then(Value::as_array) { Some(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(), None => vec!["3ax".into(), "3+2".into(), "5ax".into()] };
    let d = dfm_analyse(m, &sets)?;
    Ok(dfm_json(&d, args.get("min_wall").and_then(Value::as_f64), args.get("min_clearance").and_then(Value::as_f64)))
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
        Server { session: HarnessSession::new(requirements, part), branch: "main".into(), goal: None, load: None, dfm: None, rollouts: 0, implicit: Default::default(), candidates: Vec::new(), device_cache: Default::default() }
    }

    fn current_mesh(&self) -> Option<agcad_geo::tessellate::Mesh> {
        if let Some(im) = self.implicit.get(&self.branch) { return Some(im.mesh.mesh.clone()); }
        let u = self.session.last_update.as_ref()?;
        u.products.values().rev().find_map(|p| match p {
            Product::Solid(s) => Some(self.session.kernel.tessellate(s, &Default::default())),
            Product::Mesh(m) => Some(m.clone()),
            _ => None,
        })
    }

    /// Escalate a survivor to a new branch of Part Design features: deep certification on its
    /// own (low-priority) stream overlapped with the exact solid; device-only ops get the exact
    /// part in the document plus a certified implicit body.
    fn apply_survivor(&mut self, best: &agcad_dops::Survivor, g: &agcad_dops::abi::Goal, prefer_cuda: bool, catalog: &Catalog) -> Result<Value, String> {
        let (exact, cert) = std::thread::scope(|sc| {
            let h = sc.spawn(|| agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), &best.design, g, 7, 1 << 18).ok());
            let exact = device::exact_volume(&best.design);
            (exact, h.join().ok().flatten())
        });
        self.rollouts += 1;
        let branch = format!("device-{}", self.rollouts);
        self.session.doc.branch(0, std::slice::from_ref(&branch));
        let implicit = match &exact {
            Err(e) => Some(device::deliver_implicit(&best.design, g, e, prefer_cuda)?),
            Ok(_) => None,
        };
        let doc_design = if implicit.is_some() { device::exact_part(&best.design) } else { best.design.clone() };
        let prog = device::lower_program(&doc_design, self.session.requirements.rev, 0);
        let p = agcad_ir::json::decode_program(&prog.to_string(), catalog).map_err(|e| e.to_string())?;
        let o = self.session.run_program(&branch, &p, catalog).map_err(|(s, m)| format!("{}: {m}", s.name()))?;
        self.branch = branch.clone();
        let mut out = json!({ "branch": branch, "configuration": best.configuration, "status": o.report.status, "exact_volume": exact.as_ref().ok(), "exact_error": exact.as_ref().err(),
            "exact_within_device_bounds": exact.as_ref().map(|v| *v >= best.obs[0] && *v <= best.obs[1]).unwrap_or(false), "checks": o.report.checks,
            "certified_volume_bounds": cert.map(|c| json!([c[0], c[1]])), "certified_evidence": "bounded" });
        if let Some(im) = implicit {
            out["implicit"] = im.summary();
            self.implicit.insert(branch, im);
        }
        Ok(out)
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
            "dfm" => {
                let m = self.current_mesh().ok_or("nothing to analyse")?;
                json!(dfm_report(&m, args)?)
            }
            "render" if args.get("mode").and_then(Value::as_str) == Some("raytraced") => {
                let m = self.current_mesh().ok_or("nothing to render")?;
                let (t, why) = agcad_raytrace::best(&m);
                let (w, h) = (512, 512);
                let png = agcad_raytrace::analysis::render(&*t, &m, w, h)?;
                json!({ "png_base64": b64(&png), "width": w, "height": h, "tracer": t.name(), "fallback_reason": why })
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
                        if let Some(im) = self.implicit.get(&self.branch) {
                            return Err(format!("branch {} is a certified implicit body ({}): STEP needs an exact B-rep; export \"3mf\" or \"stl\"", self.branch, im.reason));
                        }
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
                if g.objective != 0 { g.obj_scale = device::exact_volume(&design).unwrap_or(1.0).max(1.0); }
                let load = device::parse_load(args)?;
                self.dfm = device::parse_dfm(args, &g);
                self.goal = Some((g, design.names.clone()));
                self.load = load;
                json!({ "ok": true, "parameters": design.names, "device_features": design.nodes.len(), "notes": notes, "raise_warnings": warnings,
                        "goal": { "objective": (["satisfy", "min_volume", "max_volume"][g.objective as usize]), "target_volume": (g.vol_hi > g.vol_lo).then_some([g.vol_lo, g.vol_hi]), "max_extent": g.max_ext, "domain": { "min": g.domain_lo, "max": g.domain_hi },
                                  "min_wall": (g.min_wall > 0.0).then_some(g.min_wall), "min_clearance": (g.min_gap > 0.0).then_some(g.min_gap), "machining_3ax": g.machining_3ax == 1,
                                  "stiffness": load.map(|l| json!({ "support_axis": l.support_axis, "force": l.force, "max_compliance": l.max_compliance })),
                                  "ray_check": self.dfm.as_ref().map(|d| json!({ "machining": d.set, "min_wall": d.min_wall, "min_clearance": d.min_clearance, "on": "survivors" })),
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
                let pareto = match args.get("pareto") { Some(v) => Some(pareto::parse(v)?), None => None };
                let wants_compliance = pareto.as_ref().is_some_and(|(o, _)| o.iter().any(|x| x.key == "compliance"));
                if wants_compliance && self.load.is_none() { return Err("pareto objective compliance needs a load in set_goal".into()); }
                // candidates read back: more than top_k when a trade-off set is requested
                let read_k = if pareto.is_some() { (if wants_compliance { 16 } else { 64 }).max(top_k) } else { top_k };
                // policy: CMA-ES islands over the free parameters unless structural moves are wanted
                let free = (0..g.n_params as usize).filter(|&j| g.param_hi[j] > g.param_lo[j]).count();
                let pol = match args.get("policy").and_then(Value::as_str) {
                    Some("mock") => 0,
                    Some("cmaes") => 1,
                    Some(o) => return Err(format!("policy: \"cmaes\" | \"mock\" (got {o})")),
                    None => if free > 0 && g.hole_r <= 0.0 { 1 } else { 0 },
                };
                let island = args.get("island").and_then(Value::as_u64).unwrap_or(16).clamp(2, 64) as u32;
                // parameters (+ structural slots) + up to 2 structural records
                let records = if pol == 1 && g.hole_r > 0.0 { 2 + agcad_dops::abi::MAX_PARAMS as u32 } else { (g.n_params + 2).max(4) };
                let cfg = agcad_dops::StudioConfig { configs, records, max_rounds: rounds, coarse: 8, levels, cap: 4096, nodes: (design.nodes.len() as u32 + 8).max(8), budget: records * rounds, tighten: 0, max_cells: 0, policy: pol, island };
                let prefer_cuda = match args.get("device").and_then(Value::as_str) { Some("cpu") => false, Some(_) => true, None => std::env::var("AGCAD_DEVICE").map(|v| v != "cpu").unwrap_or(true) };
                // a trade-off set needs a spread population: explore instead of stopping at the
                // first design that satisfies the goal
                let mut g = g;
                if pareto.is_some() && g.objective == 0 { g.objective = 3; }
                let run = device::run(&mut self.device_cache, &design, &g, cfg, read_k, prefer_cuda)?;
                let mut out = device::summarize(&device::RolloutRun { summary: agcad_dops::RolloutSummary { top: run.summary.top.iter().take(top_k).cloned().collect(), ..run.summary.clone() }, ..run }, &run.summary.top.first().map(|s| s.design.names.clone()).unwrap_or(names.clone()));
                out["policy"] = json!(if pol == 1 { "cmaes" } else { "mock" });
                out["objective"] = json!(["satisfy", "min_volume", "max_volume", "explore"][g.objective as usize]);
                out["raise_warnings"] = json!(warnings);
                // stiffness escalation on the survivors (voxel PCG on the device)
                let mut stiff_ok = vec![true; run.summary.top.len()];
                let mut compliance: Vec<Option<f64>> = vec![None; run.summary.top.len()];
                if let Some(spec) = self.load {
                    for (i, sv) in run.summary.top.iter().enumerate() {
                        if i >= top_k && !wants_compliance { break; }
                        let v = match device::stiffness(&sv.design, &sv.obs, &spec, prefer_cuda) {
                            Ok((r, h)) => {
                                stiff_ok[i] = r.converged && spec.max_compliance.is_none_or(|m| r.compliance <= m);
                                compliance[i] = r.converged.then_some(r.compliance);
                                json!({ "compliance": r.compliance, "max_displacement": r.max_disp, "converged": r.converged, "iterations": r.iterations, "solver": if r.levels > 1 { format!("multigrid-PCG ({} levels)", r.levels) } else { "Jacobi-PCG".to_string() }, "voxel": h, "voxels": r.voxels, "ok": stiff_ok[i], "evidence": "sampled" })
                            }
                            Err(e) => { stiff_ok[i] = false; json!({ "error": e }) }
                        };
                        if i < top_k {
                            out["top"][i]["stiffness"] = v;
                            if !stiff_ok[i] { out["top"][i]["goal_met"] = json!(false); }
                        }
                    }
                }
                // ray-cast DFM escalation on the survivors (OptiX on RTX GPUs, CPU reference otherwise)
                let mut dfm_ok = vec![self.dfm.is_none(); run.summary.top.len()];
                let mut ray_undercut: Vec<Option<f64>> = vec![None; run.summary.top.len()];
                if let Some(spec) = self.dfm.clone() {
                    let n_check = if pareto.is_some() { read_k.min(16) } else { top_k }.min(run.summary.top.len());
                    let sets: Vec<String> = spec.set.iter().cloned().collect();
                    for i in 0..n_check {
                        let sv = &run.summary.top[i];
                        if !(stiff_ok[i] && sv.obs[ob::MET] == 1.0) { continue; }
                        let r = device::survivor_mesh(&sv.design, &g, prefer_cuda).and_then(|m| dfm_analyse(&m, &sets));
                        let v = match r {
                            Ok(d) => {
                                // facets of curved walls: 0.01 mm tolerance on distances
                                let un = spec.set.as_deref().and_then(|s| d.unreachable(s)).unwrap_or(0.0);
                                let ok = un <= spec.undercut_tol + 1e-6 * d.area
                                    && (!(spec.min_wall > 0.0) || d.wall.min() >= spec.min_wall - 0.01)
                                    && (!(spec.min_clearance > 0.0) || d.gap.min() >= spec.min_clearance - 0.01);
                                dfm_ok[i] = ok;
                                ray_undercut[i] = Some(un);
                                let mut j = dfm_json(&d, (spec.min_wall > 0.0).then_some(spec.min_wall), (spec.min_clearance > 0.0).then_some(spec.min_clearance));
                                j["ok"] = json!(ok);
                                j
                            }
                            Err(e) => json!({ "error": e }),
                        };
                        if i < top_k {
                            out["top"][i]["dfm"] = v;
                            if !dfm_ok[i] { out["top"][i]["goal_met"] = json!(false); }
                        }
                    }
                    out["dfm_checked"] = json!(n_check);
                }
                if let Some((objs, k)) = &pareto {
                    // feasible survivors → objective vectors (minimization) → trade-off set
                    let val = |i: usize, key: &str| if key == "ray_undercut" { ray_undercut[i] } else { pareto::value(&run.summary.top[i], key, compliance[i]) };
                    let feas: Vec<usize> = (0..run.summary.top.len()).filter(|&i| run.summary.top[i].obs[ob::MET] == 1.0 && stiff_ok[i] && dfm_ok[i] && objs.iter().all(|o| val(i, &o.key).is_some())).collect();
                    let pts: Vec<Vec<f64>> = feas.iter().map(|&i| objs.iter().map(|o| o.sign * val(i, &o.key).unwrap()).collect()).collect();
                    let uniq = pareto::dedupe(&pts);
                    let upts: Vec<Vec<f64>> = uniq.iter().map(|&u| pts[u].clone()).collect();
                    let pick = pareto::select(&upts, *k);
                    self.candidates.clear();
                    self.rollouts += 1;
                    let mut list = Vec::new();
                    for (ci, &pi) in pick.iter().enumerate() {
                        let sv = &run.summary.top[feas[uniq[pi]]];
                        let render = device::render_design(&sv.design, &sv.obs, &g, prefer_cuda).map(|png| {
                            let _ = std::fs::create_dir_all(&self.session.out_dir);
                            let path = self.session.out_dir.join(format!("pareto-{}-{ci}.png", self.rollouts));
                            let _ = std::fs::write(&path, &png);
                            json!({ "path": path.display().to_string(), "png_base64": b64(&png), "view": "iso" })
                        });
                        let params: serde_json::Map<String, Value> = sv.design.names.iter().zip(&sv.design.params).map(|(n, v)| (n.clone(), json!((v * 1e6).round() / 1e6))).collect();
                        list.push(json!({ "candidate": ci, "configuration": sv.configuration, "objectives": pareto::summary_row(objs, &upts[pi]), "parameters": params,
                                          "features": sv.design.nodes.len(), "engineering": device::engineering(&sv.obs), "volume_bounds": [sv.obs[0], sv.obs[1]],
                                          "compliance": compliance[feas[uniq[pi]]], "render": render.unwrap_or_else(|e| json!({ "error": e })) }));
                        self.candidates.push(sv.clone());
                    }
                    out["pareto"] = json!({ "objectives": objs.iter().map(|o| if o.sign < 0.0 { format!("-{}", o.key) } else { o.key.clone() }).collect::<Vec<_>>(),
                                            "feasible": feas.len(), "distinct": uniq.len(), "candidates": list,
                                            "note": "non-dominated, diverse (crowding distance); apply one with device_apply {candidate}" });
                }
                if args.get("apply_best").and_then(Value::as_bool).unwrap_or(pareto.is_none()) {
                    let pick = run.summary.top.iter().enumerate().find(|(i, s)| s.obs[10] == 1.0 && stiff_ok[*i] && dfm_ok[*i]).map(|(_, s)| s);
                    if let Some(best) = pick.or(run.summary.top.first()) {
                        let best = best.clone();
                        out["applied"] = self.apply_survivor(&best, &g, prefer_cuda, catalog)?;
                    }
                }
                out
            }
            "device_apply" => {
                let i = args.get("candidate").and_then(Value::as_u64).ok_or("candidate: index into the last pareto list")? as usize;
                let sv = self.candidates.get(i).cloned().ok_or(format!("no candidate {i} (last rollout has {})", self.candidates.len()))?;
                let (g, _) = self.goal.clone().ok_or("no goal")?;
                let prefer_cuda = std::env::var("AGCAD_DEVICE").map(|v| v != "cpu").unwrap_or(true);
                json!({ "applied": self.apply_survivor(&sv, &g, prefer_cuda, catalog)? })
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
