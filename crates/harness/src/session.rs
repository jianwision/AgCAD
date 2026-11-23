//! Harness session (spec 06 § Transactions): applies AgIR programs to a workspace, updates the
//! specification tree, runs Design Checker rules at declared phases, saves versions only when
//! save-phase checks pass (failed saves keep the workspace, AG-06-002), meters budget (never
//! refunded, AG-06-007), records traces and emits reports.

use crate::checker::{check, CheckResult, Verdict};
use crate::report::{untrusted, DeliveryReport, FeatureLine, UpdateReport};
use crate::requirements::{Meter, Phase, Requirements};
use agcad_driver::AgStatus;
use agcad_geo::kernel::{ExactKernel, NativeKernel};
use agcad_ir::program::{Program, Value};
use agcad_ir::Catalog;
use agcad_runtime::document::{FeatureId, PartDocument};
use agcad_runtime::rebuild::{update, Product, Registry, UpdateResult};
use agcad_trace::{Trace, TraceEvent};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};

pub struct AcceptedGeometry {
    pub exact: Option<Product>,
    pub mesh: Option<agcad_geo::tessellate::Mesh>,
    pub parameters: BTreeMap<String, f64>,
}

pub fn parameter_values(state: &agcad_runtime::document::PartState, u: &agcad_runtime::rebuild::UpdateResult) -> std::collections::BTreeMap<String, f64> {
    state.order.iter().filter_map(|id| {
        let f = &state.features[id];
        if f.op != "knowledge.parameter" { return None; }
        let Some(agcad_ir::Value::String(name)) = f.args.get("name") else { return None };
        match u.products.get(id) { Some(Product::Scalar(v)) => Some((name.clone(), *v)), _ => None }
    }).collect()
}

pub struct HarnessSession {
    pub requirements: Requirements,
    pub doc: PartDocument,
    pub registry: Registry,
    pub kernel: Box<dyn ExactKernel>,
    pub meter: Meter,
    pub trace: Trace,
    /// AgIR statement id → feature id, per branch (programs may reference earlier statements).
    pub ids: HashMap<String, HashMap<String, FeatureId>>,
    pub last_update: Option<UpdateResult>,
    /// Display/export geometry survives a failed edit. Rebuild and save still see its diagnostics.
    pub accepted_bodies: HashMap<String, AcceptedGeometry>,
    last_attempt: Option<(String, u32, bool)>,
    /// Output directory for file features (control-plane IO).
    pub out_dir: std::path::PathBuf,
}

#[derive(Debug)]
pub struct ProgramOutcome {
    pub report: UpdateReport,
    /// Set when the program contained `document.checkpoint` / `document.finish`.
    pub delivery: Option<DeliveryReport>,
}

fn remap(v: &Value, ids: &HashMap<String, FeatureId>) -> Value {
    match v {
        Value::Ref { id, field } => match ids.get(id) { Some(f) => Value::Ref { id: format!("f{f}"), field: field.clone() }, None => v.clone() },
        Value::List(l) => Value::List(l.iter().map(|x| remap(x, ids)).collect()),
        Value::Map(m) => Value::Map(m.iter().map(|(k, x)| (k.clone(), remap(x, ids))).collect()),
        other => other.clone(),
    }
}

impl HarnessSession {
    pub fn new(requirements: Requirements, part_name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        let session = format!("{}-{stamp}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
        HarnessSession {
            requirements,
            doc: PartDocument::new(part_name),
            registry: { let mut r = agcad_std::standard_registry(); agcad_proc::register(&mut r); r },
            kernel: Box::new(NativeKernel),
            meter: Meter::default(),
            trace: Trace::default(),
            ids: HashMap::new(),
            last_update: None,
            accepted_bodies: HashMap::new(),
            last_attempt: None,
            out_dir: std::env::temp_dir().join("agcad-out").join(session),
        }
    }

    /// Run a program on `branch` (AG-06-001: requirements revision must match).
    pub fn run_program(&mut self, branch: &str, p: &Program, catalog: &Catalog) -> Result<ProgramOutcome, (AgStatus, String)> {
        Self::check_worker()?;
        let (out, events) = agcad_geo::accelerator::capture(|| self.run_program_inner(branch, p, catalog));
        self.trace_geometry(events);
        Self::check_worker()?;
        out
    }

    fn check_worker() -> Result<(), (AgStatus, String)> {
        agcad_geo::accelerator::ensure_worker_healthy().map_err(|e| (AgStatus::InvalidBaseline, e.to_string()))
    }

    pub fn trace_geometry(&mut self, events: Vec<agcad_geo::accelerator::Dispatch>) {
        for event in events {
            self.trace.push(TraceEvent::GeometryWork { operation: event.operation.into(), backend: event.backend.into(), work: event.work, fallback: event.error });
        }
    }

    fn run_program_inner(&mut self, branch: &str, p: &Program, catalog: &Catalog) -> Result<ProgramOutcome, (AgStatus, String)> {
        self.requirements.check_rev(p.requirements_rev).map_err(|s| (s, format!("program targets requirements rev {}, current is {}", p.requirements_rev, self.requirements.rev)))?;
        let diags = agcad_ir::verify::verify(p, catalog);
        if let Some(d) = diags.first() { return Err((d.status, format!("{}: {}", d.stmt, d.message))); }
        let ws_rev = self.doc.workspace(branch).ok_or((AgStatus::StaleReference, format!("unknown branch '{branch}'")))?.revision;
        if p.expected_revision != ws_rev { return Err((AgStatus::StaleRevision, format!("expected revision {}, workspace is at {ws_rev}", p.expected_revision))); }
        self.trace.push(TraceEvent::Program { requirements_rev: p.requirements_rev, expected_revision: p.expected_revision, agir_json: agcad_ir::json::encode_program(p) });
        self.meter.charge_ops(&self.requirements.budget, p.statements.len() as u64).map_err(|s| (s, "op budget exhausted".into()))?;
        let mut delivery = None;
        let mut want_save = false;
        let mut save_label = String::from("program");
        let ids = self.ids.entry(branch.to_string()).or_default().clone();
        let mut ids = ids;
        for s in &p.statements {
            match s.op.as_str() {
                "document.edit" | "document.update" | "document.checkpoint" => {}
                "document.save_version" => {
                    want_save = true;
                    if let Some(Value::String(l)) = s.args.get("label") { save_label = l.clone(); }
                }
                "document.finish" => { delivery = Some(()); }
                "document.discard_edit" => { self.doc.discard(branch); }
                op if op.starts_with("study.") || op == "udf.define" || op == "udf.compile" => {
                    // studies run through the study engine (pass P9); compile-time UDF ops on host
                }
                _ => {
                    // "$body": k places the feature in Body container k (multi-body parts)
                    let container = match s.args.get("$body") { Some(Value::Int(k)) if *k >= 0 => *k as usize, _ => 0 };
                    let args: BTreeMap<String, Value> = s.args.iter().filter(|(k, _)| k.as_str() != "$body").map(|(k, v)| (k.clone(), remap(v, &ids))).collect();
                    let version = catalog.get(&s.op).map(|m| m.version).unwrap_or(1);
                    let role = match s.args.get("role") { Some(Value::String(r)) => Some(r.clone()), _ => None };
                    let ws = self.doc.workspace(branch).unwrap();
                    let mut fid = 0;
                    // knowledge.parameter with an existing name modifies that parameter in place
                    // (CATIA semantics: parameters are unique by name), it does not add another.
                    let existing = if s.op == "knowledge.parameter" {
                        let name = args.get("name").cloned();
                        ws.state.order.iter().copied().find(|id| { let f = &ws.state.features[id]; f.op == "knowledge.parameter" && f.args.get("name") == name.as_ref() })
                    } else { None };
                    match existing {
                        Some(id) => { ws.edit(|st| { if let Some(f) = st.features.get_mut(&id) { f.args = args; } }); fid = id; }
                        None => ws.edit(|st| { fid = st.add_feature(&s.op, version, args, container, role); }),
                    }
                    ids.insert(s.id.clone(), fid);
                }
            }
        }
        self.ids.insert(branch.to_string(), ids);
        let mut report = self.update(branch, Phase::Update);
        if let Some(study) = self.run_study_statements(branch, p) {
            match study {
                Ok(v) => {
                    report.topology_classes = v.get("topology_classes").cloned();
                    report.proxy_fidelity = v.get("proxy_fidelity").cloned();
                    report.study = Some(v);
                }
                Err(e) => report.study = Some(json!({ "error": e })),
            }
        }
        Self::check_worker()?;
        if want_save {
            if let Err((st, msg)) = self.save_version(branch, &save_label) {
                self.trace.push(TraceEvent::Note { text: format!("save_version rejected: {st} {msg}") });
            }
        }
        let delivery = delivery.map(|_| self.finish(branch));
        Ok(ProgramOutcome { report, delivery })
    }

    /// Execute `study.configurations` / `study.evaluate` / `study.rank` statements.
    fn run_study_statements(&mut self, branch: &str, p: &Program) -> Option<Result<serde_json::Value, String>> {
        let cfg = p.statements.iter().find(|s| s.op == "study.configurations")?;
        let to_json = |v: &Value| agcad_ir::json::encode_value(v);
        let spec = cfg.args.get("table").map(to_json).unwrap_or(json!({}));
        let mut objective: Option<(String, String, bool)> = None;
        if let Some(ev) = p.statements.iter().find(|s| s.op == "study.evaluate") {
            if let Some(Value::List(objs)) = ev.args.get("objectives") {
                for o in objs {
                    let j = to_json(o);
                    for (key, maximize) in [("minimize", false), ("maximize", true)] {
                        if let Some(m) = j.get(key) {
                            let measure = m.get("op").and_then(|x| x.as_str()).unwrap_or("evaluate.measure_inertia").to_string();
                            let field = m.get("field").and_then(|x| x.as_str()).unwrap_or("mass").to_string();
                            objective = Some((measure, field, maximize));
                        }
                    }
                }
            }
        }
        let k = p.statements.iter().find(|s| s.op == "study.rank").and_then(|s| s.args.get("k")).and_then(|v| v.as_f64()).unwrap_or(4.0) as usize;
        let ws = self.doc.workspaces.get(branch)?;
        let state = ws.state.clone();
        let obj = objective.as_ref().map(|(a, b, c)| (a.as_str(), b.as_str(), *c));
        let r = crate::study::run(&state, &self.registry, self.kernel.as_ref(), &self.requirements, &spec, obj, k);
        if let Ok(v) = &r {
            let n = v["configurations"].as_u64().unwrap_or(0);
            let _ = self.meter.charge_exact(&self.requirements.budget, k.min(n as usize) as u64);
            self.trace.push(TraceEvent::Segment { index: 0, configurations: n as u32, gpu_ms: 0.0 });
        }
        Some(r)
    }

    pub fn selected_body<'a>(&self, update: &'a UpdateResult) -> Option<(&'a FeatureId, &'a Product)> {
        let kind = self.requirements.deliver.body_type.as_str();
        update.products.iter().rev().find(|(_, p)| match kind {
            "SurfaceBody" => matches!(p, Product::Sheet(_)),
            "SolidBody" => matches!(p, Product::Solid(_)),
            _ => matches!(p, Product::Solid(_) | Product::Sheet(_) | Product::Mesh(_) | Product::Implicit(_)),
        })
    }

    fn run_checks(&self, branch: &str, phase: Phase) -> Vec<CheckResult> {
        let Some(u) = self.last_update.as_ref() else { return vec![]; };
        let Some((&body, _)) = self.selected_body(u) else {
            let required = self.requirements.at_phase(phase);
            let incompatible = u.products.values().any(|p| matches!(p, Product::Solid(_) | Product::Sheet(_)));
            if phase == Phase::Finish || incompatible || !required.is_empty() {
                return vec![CheckResult {
                    requirement: "deliver.body_type".into(), verdict: Verdict::Fail,
                    value: serde_json::Value::Null, margin: None, evidence: crate::requirements::Evidence::Exact,
                    required_evidence: crate::requirements::Evidence::Exact, checker_version: crate::checker::CHECKER_VERSION,
                    escalation: None, message: Some(format!("workspace requires a {}", self.requirements.deliver.body_type)),
                }];
            }
            return vec![];
        };
        let ws = self.doc.workspaces.get(branch).expect("workspace exists");
        self.requirements.at_phase(phase).into_iter().map(|h| check(h, &ws.state, self.kernel.as_ref(), &self.registry, &u.products, body)).collect()
    }

    /// `document.update`: rebuild + update-phase checks → report.
    pub fn update(&mut self, branch: &str, phase: Phase) -> UpdateReport {
        let t0 = std::time::Instant::now();
        let ws = self.doc.workspace(branch).expect("workspace");
        let state = ws.state.clone();
        let revision = ws.revision;
        let res = update(&state, &self.registry, self.kernel.as_ref(), BTreeMap::new(), vec![]);
        let mut features = Vec::new();
        let mut unsupported = Vec::new();
        for o in &res.outcomes {
            if let Some(&(device, patches)) = res.dispatches.get(&o.feature) {
                self.trace.push(TraceEvent::GeometryDispatch { feature_ref: o.feature_ref.clone(), device: device.into(), patches });
            }
            self.trace.push(TraceEvent::Feature { feature_ref: o.feature_ref.clone(), status: o.status.value(), message: o.error.as_ref().map(|e| e.message.clone()) });
            if o.status == AgStatus::UnsupportedOp { unsupported.push(o.feature_ref.clone()); }
            features.push(FeatureLine {
                feature: untrusted(&o.feature_ref, "specification_tree"),
                status: o.status.name().to_string(),
                code: o.status.value(),
                message: o.error.as_ref().map(|e| untrusted(&e.message, "feature")),
                elements: o.error.as_ref().map(|e| e.elements.iter().map(|x| untrusted(x, "element_name")).collect()).unwrap_or_default(),
                suggestion: o.error.as_ref().and_then(|e| e.suggestion.clone()),
            });
        }
        let ok = res.ok();
        self.last_update = Some(res);
        let checks = self.run_checks(branch, phase);
        let passed = checks.iter().all(|c| c.verdict == Verdict::Pass);
        let healthy = Self::check_worker().is_ok();
        let accepted = healthy && ok && passed;
        self.last_attempt = Some((branch.to_owned(), revision, accepted));
        if accepted {
            let u = self.last_update.as_ref().unwrap();
            let selected = self.selected_body(u);
            let exact = selected.and_then(|(_,p)| match p {
                Product::Solid(_) | Product::Sheet(_) => Some(p.clone()),
                _ => None,
            });
            let mesh = u.products.iter().rev().find_map(|(id,p)| match p {
                Product::Mesh(m) if selected.is_none_or(|(body,_)| id >= body) => Some(m.clone()),
                _ => None,
            });
            self.accepted_bodies.insert(branch.into(), AcceptedGeometry { exact, mesh, parameters: parameter_values(&state,u) });
        }
        for c in &checks {
            self.trace.push(TraceEvent::Check { requirement: c.requirement.clone(), verdict: format!("{:?}", c.verdict), value: c.value.as_f64().unwrap_or(f64::NAN), bound: f64::NAN, evidence: format!("{:?}", c.evidence), checker_version: c.checker_version });
        }
        let _ = self.meter.charge_time(&self.requirements.budget, 0.0, t0.elapsed().as_secs_f64() * 1000.0);
        let status = if !healthy { AgStatus::InvalidBaseline.name() } else if !ok { "FEATURE_FAILED" } else if !passed { AgStatus::RequirementFailed.name() } else { AgStatus::Accepted.name() };
        UpdateReport {
            status: status.into(), revision, requirements_rev: self.requirements.rev, workspace_state: if accepted { "valid".into() } else { "invalid".into() },
            features, checks, budget: self.meter.remaining(&self.requirements.budget), unsupported, topology_classes: None, proxy_fidelity: None, study: None,
        }
    }

    fn current_attempt(&mut self, branch: &str) -> bool {
        let Some(ws) = self.doc.workspaces.get(branch) else { return false };
        if let Some((name, revision, accepted)) = &self.last_attempt {
            if name == branch && *revision == ws.revision { return *accepted; }
        }
        self.update(branch, Phase::Update).status == AgStatus::Accepted.name()
    }

    /// `document.save_version`: save-phase checks must pass; otherwise the workspace is kept.
    pub fn save_version(&mut self, branch: &str, label: &str) -> Result<u32, (AgStatus, String)> {
        Self::check_worker()?;
        if !self.current_attempt(branch) {
            let status = if self.last_update.as_ref().is_some_and(|u| !u.ok()) { AgStatus::TopologyFailed } else { AgStatus::RequirementFailed };
            return Err((status, "workspace has a rejected update".into()));
        }
        let checks = self.run_checks(branch, Phase::Save);
        Self::check_worker()?;
        if let Some(f) = checks.iter().find(|c| c.verdict != Verdict::Pass) {
            return Err((AgStatus::RequirementFailed, format!("{} {:?} (value {}); workspace kept for repair", f.requirement, f.verdict, f.value)));
        }
        if self.last_update.as_ref().is_some_and(|u| !u.ok()) {
            return Err((AgStatus::TopologyFailed, "workspace has failing features; workspace kept for repair".into()));
        }
        let v = self.doc.save_version(branch, self.requirements.rev, label).ok_or((AgStatus::StaleReference, "unknown branch".into()))?;
        self.trace.push(TraceEvent::Version { branch: branch.into(), version: v, label: label.into() });
        Ok(v)
    }

    /// `document.finish`: finish-phase checks, write files (control plane), delivery report.
    pub fn finish(&mut self, branch: &str) -> DeliveryReport {
        let accepted = self.current_attempt(branch);
        let checks = self.run_checks(branch, Phase::Finish);
        let passed = accepted && Self::check_worker().is_ok() && checks.iter().all(|c| c.verdict == Verdict::Pass);
        let mut files = Vec::new();
        let mut delivered = true;
        if let Some(u) = self.last_update.as_ref().filter(|_| passed) {
            for p in u.products.values() {
                if let Product::Other(o) = p {
                    if let Some(fb) = o.downcast_ref::<agcad_std::file::FileBytes>() {
                        let _ = std::fs::create_dir_all(&self.out_dir);
                        let path = self.out_dir.join(std::path::Path::new(&fb.path).file_name().unwrap_or_default());
                        let ok = std::fs::write(&path, &fb.bytes).is_ok();
                        delivered &= ok;
                        files.push(json!({ "path": path.display().to_string(), "bytes": fb.bytes.len(), "warnings": fb.warnings }));
                    }
                }
            }
        }
        let exec_ok = self.last_update.as_ref().is_some_and(|u| u.ok());
        let mut report = DeliveryReport::from_checks(checks, exec_ok, files, delivered && passed);
        report.requirements_met &= accepted;
        report
    }
}
