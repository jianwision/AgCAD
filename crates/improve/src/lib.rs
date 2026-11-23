//! # agcad-improve — recursive self-improvement (spec 11)
//!
//! * Archive: content-addressed candidates, evaluations and promotions; MAP-Elites niches.
//! * Gate: non-regression + pre-declared improvement + full cost vector + pinned evidence +
//!   proxy fidelity (anti-gaming, spec 11); held-out queries rate-limited and counted.
//! * Promotion is versioned, content-addressed and reversible (AG-11-002); counterexamples are
//!   permanent (AG-11-003); L5 changes need a human approver (AG-11-004).
//! * Candidates run in sandbox worker processes (`AGCAD_SANDBOX=1`, AG-17-001/006).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub mod benchmark;
pub mod sandbox;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level { L1, L2, L3, L4, L5 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub level: Level,
    /// What it replaces/extends: UDF name, kernel `library::kernel`, checkpoint name, …
    pub target: String,
    pub parent_ids: Vec<String>,
    pub proposer: String,
    pub artifact_hash: String,
    pub claims: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostVector {
    pub gpu_ms: f64,
    pub host_ms: f64,
    pub exact_rebuilds: f64,
    pub llm_tokens: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub candidate_id: String,
    pub suite_version: String,
    pub backend: String,
    /// task id → success (1/0) or score
    pub metrics: BTreeMap<String, f64>,
    pub cost: CostVector,
    /// Evidence level achieved per task (pinned per benchmark; anti-gaming).
    pub evidence: BTreeMap<String, String>,
    pub proxy_kendall_tau: Option<f64>,
    pub conformance_failed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatePolicy {
    /// Minimum mean improvement on the targeted metric (pre-declared effect size).
    pub min_improvement: f64,
    /// Allowed per-task regression (noise bound, pre-registered).
    pub regression_tolerance: f64,
    /// Maximum relative cost increase per cost dimension.
    pub max_cost_increase: f64,
    pub min_proxy_tau: f64,
    /// Required evidence per task family (pinned).
    pub pinned_evidence: BTreeMap<String, String>,
}

impl Default for GatePolicy {
    fn default() -> Self { GatePolicy { min_improvement: 0.02, regression_tolerance: 0.0, max_cost_increase: 0.10, min_proxy_tau: 0.7, pinned_evidence: BTreeMap::new() } }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    pub pass: bool,
    pub reasons: Vec<String>,
    pub improvement: f64,
}

fn rank_evidence(s: &str) -> u8 { match s { "exact" => 3, "bounded" => 2, "sampled" => 1, _ => 0 } }

/// Promotion gate: candidate vs incumbent on the same suite/backend (spec 11 § The loop).
pub fn gate(incumbent: &Evaluation, cand: &Evaluation, policy: &GatePolicy, level: Level, human_approved: bool) -> GateReport {
    let mut reasons = Vec::new();
    if cand.suite_version != incumbent.suite_version || cand.backend != incumbent.backend {
        reasons.push("suite/backend mismatch: promotion compares within architecture (review F8)".into());
    }
    if cand.conformance_failed > 0 { reasons.push(format!("{} conformance failures", cand.conformance_failed)); }
    let mut deltas = Vec::new();
    for (task, base) in &incumbent.metrics {
        let new = cand.metrics.get(task).copied().unwrap_or(f64::NEG_INFINITY);
        if new + policy.regression_tolerance < *base { reasons.push(format!("regression on {task}: {base} → {new}")); }
        deltas.push(new - base);
    }
    let improvement = if deltas.is_empty() { 0.0 } else { deltas.iter().sum::<f64>() / deltas.len() as f64 };
    if improvement < policy.min_improvement { reasons.push(format!("improvement {improvement:.4} below declared {:.4}", policy.min_improvement)); }
    let rel = |a: f64, b: f64| if a <= 0.0 { if b > 0.0 { f64::INFINITY } else { 0.0 } } else { (b - a) / a };
    for (name, a, b) in [("gpu_ms", incumbent.cost.gpu_ms, cand.cost.gpu_ms), ("host_ms", incumbent.cost.host_ms, cand.cost.host_ms), ("exact_rebuilds", incumbent.cost.exact_rebuilds, cand.cost.exact_rebuilds), ("llm_tokens", incumbent.cost.llm_tokens, cand.cost.llm_tokens)] {
        if rel(a, b) > policy.max_cost_increase { reasons.push(format!("cost shift on {name}: {a} → {b}")); }
    }
    for (task, req) in &policy.pinned_evidence {
        let got = cand.evidence.get(task).map(String::as_str).unwrap_or("unknown");
        if rank_evidence(got) < rank_evidence(req) { reasons.push(format!("evidence on {task} is {got}, pinned {req}")); }
    }
    if let Some(t) = cand.proxy_kendall_tau { if t < policy.min_proxy_tau { reasons.push(format!("proxy fidelity τ={t:.2} below {}", policy.min_proxy_tau)); } }
    if level == Level::L5 && !human_approved { reasons.push("L5 changes require human approval (AG-11-004)".into()); }
    GateReport { pass: reasons.is_empty(), reasons, improvement }
}

/// Held-out set accounting: queries are counted and rate-limited (reusable-holdout discipline).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeldOut {
    pub queries: u64,
    pub limit: u64,
    pub rotation: u32,
}

impl HeldOut {
    pub fn query(&mut self) -> Result<(), String> {
        if self.queries >= self.limit { return Err(format!("held-out query budget exhausted ({}); rotate the set (rotation {})", self.limit, self.rotation)); }
        self.queries += 1;
        Ok(())
    }
}

pub fn content_hash(bytes: &[u8]) -> String { format!("{:016x}", agcad_ir::catalog::fnv1a64(bytes)) }

/// Directory-backed archive: candidates/, evaluations/, promotions.json, counterexamples/.
pub struct Archive {
    pub root: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Promotions {
    /// target → stack of promoted candidate ids (last = active); rollback pops.
    pub active: BTreeMap<String, Vec<String>>,
    pub log: Vec<Value>,
}

impl Archive {
    pub fn open(root: &Path) -> std::io::Result<Archive> {
        for d in ["candidates", "evaluations", "counterexamples", "artifacts"] { std::fs::create_dir_all(root.join(d))?; }
        Ok(Archive { root: root.to_path_buf() })
    }
    fn write_json(&self, rel: &str, v: &impl Serialize) -> std::io::Result<()> { std::fs::write(self.root.join(rel), serde_json::to_vec_pretty(v).unwrap()) }
    fn read_json<T: for<'de> Deserialize<'de>>(&self, rel: &str) -> Option<T> { serde_json::from_slice(&std::fs::read(self.root.join(rel)).ok()?).ok() }

    pub fn propose(&self, level: Level, target: &str, proposer: &str, artifact: &[u8], claims: Value, parents: Vec<String>) -> std::io::Result<Candidate> {
        let hash = content_hash(artifact);
        std::fs::write(self.root.join("artifacts").join(&hash), artifact)?;
        let id = format!("{:?}-{}-{}", level, target.replace(['/', ':'], "_"), &hash[..8]).to_lowercase();
        let c = Candidate { id: id.clone(), level, target: target.into(), parent_ids: parents, proposer: proposer.into(), artifact_hash: hash, claims };
        self.write_json(&format!("candidates/{id}.json"), &c)?;
        Ok(c)
    }
    pub fn record_evaluation(&self, e: &Evaluation) -> std::io::Result<()> { self.write_json(&format!("evaluations/{}.json", e.candidate_id), e) }
    pub fn evaluation(&self, id: &str) -> Option<Evaluation> { self.read_json(&format!("evaluations/{id}.json")) }
    pub fn candidate(&self, id: &str) -> Option<Candidate> { self.read_json(&format!("candidates/{id}.json")) }
    pub fn promotions(&self) -> Promotions { self.read_json("promotions.json").unwrap_or_default() }

    pub fn promote(&self, c: &Candidate, report: &GateReport, approver: Option<&str>) -> Result<(), String> {
        if !report.pass { return Err(format!("gate failed: {:?}", report.reasons)); }
        let mut p = self.promotions();
        p.active.entry(c.target.clone()).or_default().push(c.id.clone());
        p.log.push(json!({ "action": "promote", "candidate": c.id, "target": c.target, "improvement": report.improvement, "approver": approver }));
        self.write_json("promotions.json", &p).map_err(|e| e.to_string())
    }
    pub fn rollback(&self, target: &str) -> Result<Option<String>, String> {
        let mut p = self.promotions();
        let popped = p.active.get_mut(target).and_then(|v| v.pop());
        p.log.push(json!({ "action": "rollback", "target": target, "removed": popped }));
        self.write_json("promotions.json", &p).map_err(|e| e.to_string())?;
        Ok(popped)
    }
    /// Permanent counterexample corpus (AG-11-003): never deleted by the archive API.
    pub fn add_counterexample(&self, name: &str, data: &Value) -> std::io::Result<()> {
        self.write_json(&format!("counterexamples/{}-{}.json", name, content_hash(data.to_string().as_bytes())), data)
    }
    /// MAP-Elites niche key: task family × body type × cost band × method.
    pub fn niche(family: &str, body_type: &str, cost_ms: f64, method: &str) -> String {
        let band = if cost_ms < 10.0 { "fast" } else if cost_ms < 1000.0 { "mid" } else { "slow" };
        format!("{family}/{body_type}/{band}/{method}")
    }
}

/// Run a candidate's conformance in a sandbox worker process (own GPU context; AG-17-001).
pub fn sandbox_conformance(exe: &Path, fuzz: usize) -> Result<Value, String> {
    let out = std::process::Command::new(exe).args(["conformance", "--fuzz", &fuzz.to_string()]).env("AGCAD_SANDBOX", "1").output().map_err(|e| e.to_string())?;
    serde_json::from_slice(&out.stdout).map_err(|e| format!("sandbox output: {e}; stderr: {}", String::from_utf8_lossy(&out.stderr)))
}

/// `agcad improve <archive> <cmd> ...`
pub fn cli(args: &[String]) -> i32 {
    let Some(root) = args.first() else { eprintln!("usage: agcad improve <archive-dir> <propose|evaluate|gate|promote|rollback|list> ..."); return 2 };
    let a = match Archive::open(Path::new(root)) { Ok(a) => a, Err(e) => { eprintln!("{e}"); return 2 } };
    let get = |i: usize| args.get(i).cloned().unwrap_or_default();
    match args.get(1).map(String::as_str) {
        Some("propose") => {
            let level = match get(2).as_str() { "l1" => Level::L1, "l2" => Level::L2, "l3" => Level::L3, "l4" => Level::L4, "l5" => Level::L5, _ => { eprintln!("level l1..l5"); return 2 } };
            let artifact = std::fs::read(get(4)).unwrap_or_default();
            match a.propose(level, &get(3), "cli", &artifact, json!({}), vec![]) { Ok(c) => { println!("{}", c.id); 0 } Err(e) => { eprintln!("{e}"); 1 } }
        }
        Some("gate") => {
            let (inc, cand) = (a.evaluation(&get(2)), a.evaluation(&get(3)));
            let (Some(inc), Some(cand)) = (inc, cand) else { eprintln!("missing evaluations"); return 2 };
            let level = a.candidate(&get(3)).map(|c| c.level).unwrap_or(Level::L2);
            let r = gate(&inc, &cand, &GatePolicy::default(), level, args.iter().any(|x| x == "--approved"));
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            if r.pass { 0 } else { 1 }
        }
        Some("promote") => {
            let (Some(inc), Some(cand), Some(c)) = (a.evaluation(&get(2)), a.evaluation(&get(3)), a.candidate(&get(3))) else { eprintln!("missing records"); return 2 };
            let approver = args.iter().position(|x| x == "--approver").and_then(|i| args.get(i + 1)).map(String::as_str);
            let r = gate(&inc, &cand, &GatePolicy::default(), c.level, approver.is_some());
            match a.promote(&c, &r, approver) { Ok(()) => { println!("promoted {}", c.id); 0 } Err(e) => { eprintln!("{e}"); 1 } }
        }
        Some("rollback") => match a.rollback(&get(2)) { Ok(x) => { println!("rolled back {x:?}"); 0 } Err(e) => { eprintln!("{e}"); 1 } },
        Some("list") => { println!("{}", serde_json::to_string_pretty(&a.promotions()).unwrap()); 0 }
        _ => { eprintln!("unknown improve command"); 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ev(id: &str, m: &[(&str, f64)], gpu: f64) -> Evaluation {
        Evaluation { candidate_id: id.into(), suite_version: "v1".into(), backend: "cuda-sm89".into(), metrics: m.iter().map(|(k, v)| (k.to_string(), *v)).collect(), cost: CostVector { gpu_ms: gpu, host_ms: 1.0, exact_rebuilds: 4.0, llm_tokens: 0.0 }, evidence: BTreeMap::new(), proxy_kendall_tau: Some(0.9), conformance_failed: 0 }
    }
    #[test]
    fn gate_blocks_regressions_and_cost_shifts() {
        let inc = ev("a", &[("t1", 0.5), ("t2", 0.8)], 10.0);
        let good = ev("b", &[("t1", 0.6), ("t2", 0.8)], 10.5);
        assert!(gate(&inc, &good, &GatePolicy::default(), Level::L2, false).pass);
        let regress = ev("c", &[("t1", 0.9), ("t2", 0.7)], 10.0);
        assert!(!gate(&inc, &regress, &GatePolicy::default(), Level::L2, false).pass);
        let costly = ev("d", &[("t1", 0.9), ("t2", 0.9)], 30.0);
        assert!(!gate(&inc, &costly, &GatePolicy::default(), Level::L4, false).pass);
        assert!(!gate(&inc, &good, &GatePolicy::default(), Level::L5, false).pass, "L5 needs approval");
        let mut h = HeldOut { limit: 1, ..Default::default() };
        assert!(h.query().is_ok() && h.query().is_err());
    }
}
