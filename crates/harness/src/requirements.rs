//! Requirements (spec 06 § Requirements; spec 14): immutable during a session, caller-owned
//! (AG-06-001, AG-17-004), with phases, required evidence and the manufacturing process.

use agcad_driver::AgStatus;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence { Unknown, Sampled, Bounded, Exact }

impl Evidence {
    /// Spec name (`unknown`, `sampled`, `bounded`, `exact`).
    pub fn name(self) -> &'static str {
        match self { Evidence::Unknown => "unknown", Evidence::Sampled => "sampled", Evidence::Bounded => "bounded", Evidence::Exact => "exact" }
    }
    pub fn parse(s: &str) -> Evidence {
        match s { "exact" => Evidence::Exact, "bounded" => Evidence::Bounded, "sampled" => Evidence::Sampled, _ => Evidence::Unknown }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase { Edit, Update, Save, Finish }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Process {
    None,
    Additive,
    #[serde(rename = "machining_3ax")]
    Machining3ax,
    #[serde(rename = "machining_5ax")]
    Machining5ax,
    Molding,
    SheetMetal,
}

impl Default for Process { fn default() -> Self { Process::None } }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardRequirement {
    pub id: String,
    /// Evaluation op, e.g. `evaluate.bounding_box`.
    pub measure: String,
    /// Field of the evaluation result, e.g. `max_extent`.
    pub field: String,
    #[serde(default)]
    pub le: Option<f64>,
    #[serde(default)]
    pub ge: Option<f64>,
    #[serde(default)]
    pub eq: Option<serde_json::Value>,
    #[serde(default = "default_evidence")]
    pub evidence: Evidence,
    /// Extra arguments for the evaluation (e.g. `{"min": 3.0}`).
    #[serde(default)]
    pub args: BTreeMap<String, serde_json::Value>,
}

fn default_evidence() -> Evidence { Evidence::Sampled }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub id: String,
    pub minimize: Option<String>,
    pub maximize: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    #[serde(default)] pub ops: Option<u64>,
    #[serde(default)] pub gpu_ms: Option<f64>,
    #[serde(default)] pub host_ms: Option<f64>,
    #[serde(default)] pub exact_rebuilds: Option<u64>,
    #[serde(default)] pub llm_calls: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Delivery {
    pub body_type: String,
    #[serde(default = "one")] pub count: u32,
    #[serde(default)] pub formats: Vec<String>,
}
fn one() -> u32 { 1 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Requirements {
    pub id: String,
    pub rev: u32,
    #[serde(default = "mm")] pub units: String,
    pub deliver: Delivery,
    #[serde(default)] pub hard: Vec<HardRequirement>,
    #[serde(default)] pub objectives: Vec<Objective>,
    #[serde(default)] pub budget: Budget,
    #[serde(default)] pub process: Process,
    /// phase → requirement ids checked at that phase.
    #[serde(default)] pub phases: BTreeMap<String, Vec<String>>,
}
fn mm() -> String { "mm".into() }

impl Requirements {
    pub fn parse(json: &str) -> Result<Requirements, String> {
        let r: Requirements = serde_json::from_str(json).map_err(|e| e.to_string())?;
        for h in &r.hard {
            if h.le.is_none() && h.ge.is_none() && h.eq.is_none() { return Err(format!("{}: needs le, ge or eq", h.id)); }
            if !h.measure.starts_with("evaluate.") { return Err(format!("{}: measure must be an evaluate.* op", h.id)); }
        }
        for ids in r.phases.values() {
            for id in ids { if !r.hard.iter().any(|h| &h.id == id) { return Err(format!("phase references unknown requirement {id}")); } }
        }
        Ok(r)
    }

    pub fn at_phase(&self, p: Phase) -> Vec<&HardRequirement> {
        let key = match p { Phase::Edit => "edit", Phase::Update => "update", Phase::Save => "save", Phase::Finish => "finish" };
        match self.phases.get(key) {
            Some(ids) => self.hard.iter().filter(|h| ids.contains(&h.id)).collect(),
            None if p == Phase::Finish => self.hard.iter().collect(),
            None => vec![],
        }
    }

    /// Requirements that can never pass with the implemented evaluators (spec 14): the measure
    /// is not implemented, or its strongest evidence is below the required level. Reported up
    /// front (in `get_requirements` and study reports) instead of as silent `unknown` verdicts.
    pub fn satisfiability(&self) -> Vec<String> {
        self.hard.iter().filter_map(|h| {
            match agcad_std::evaluate::MAX_EVIDENCE.iter().find(|(m, _)| *m == h.measure) {
                None => Some(format!("{}: no implemented evaluator for '{}' (verdict will be unknown)", h.id, h.measure)),
                Some((_, max)) if Evidence::parse(max) < h.evidence => Some(format!(
                    "{}: requires '{}' evidence but {} provides at most '{max}'; lower the requirement's evidence to '{max}' or use a stronger evaluator",
                    h.id, h.evidence.name(), h.measure)),
                _ => None,
            }
        }).collect()
    }

    /// Enforce immutability: callers present `rev`; anything else is STALE_REQUIREMENTS.
    pub fn check_rev(&self, presented: u32) -> Result<(), AgStatus> {
        if presented == self.rev { Ok(()) } else { Err(AgStatus::StaleRequirements) }
    }
}

/// Never-refunded budget meter (spec 06; AG-06-007).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Meter {
    pub ops: u64,
    pub gpu_ms: f64,
    pub host_ms: f64,
    pub exact_rebuilds: u64,
    pub llm_calls: u64,
}

impl Meter {
    pub fn charge_ops(&mut self, b: &Budget, n: u64) -> Result<(), AgStatus> {
        if let Some(max) = b.ops { if self.ops + n > max { return Err(AgStatus::BudgetExhausted); } }
        self.ops += n;
        Ok(())
    }
    pub fn charge_exact(&mut self, b: &Budget, n: u64) -> Result<(), AgStatus> {
        if let Some(max) = b.exact_rebuilds { if self.exact_rebuilds + n > max { return Err(AgStatus::BudgetExhausted); } }
        self.exact_rebuilds += n;
        Ok(())
    }
    pub fn charge_time(&mut self, b: &Budget, gpu_ms: f64, host_ms: f64) -> Result<(), AgStatus> {
        self.gpu_ms += gpu_ms;
        self.host_ms += host_ms;
        if b.gpu_ms.is_some_and(|m| self.gpu_ms > m) || b.host_ms.is_some_and(|m| self.host_ms > m) { return Err(AgStatus::BudgetExhausted); }
        Ok(())
    }
    pub fn remaining(&self, b: &Budget) -> serde_json::Value {
        serde_json::json!({
            "ops": b.ops.map(|m| m.saturating_sub(self.ops)),
            "gpu_ms": b.gpu_ms.map(|m| (m - self.gpu_ms).max(0.0)),
            "host_ms": b.host_ms.map(|m| (m - self.host_ms).max(0.0)),
            "exact_rebuilds": b.exact_rebuilds.map(|m| m.saturating_sub(self.exact_rebuilds)),
            "llm_calls": b.llm_calls.map(|m| m.saturating_sub(self.llm_calls)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub const EXAMPLE: &str = r#"{"id":"flange-req","rev":3,"deliver":{"body_type":"SolidBody","formats":["step_ap242"]},
      "hard":[{"id":"R1","measure":"evaluate.bounding_box","field":"max_extent","le":160},
              {"id":"R3","measure":"evaluate.check","field":"valid","eq":true,"evidence":"exact"}],
      "budget":{"ops":200000,"exact_rebuilds":64},"process":"machining_3ax",
      "phases":{"update":["R1"],"save":["R1","R3"],"finish":["R1","R3"]}}"#;
    #[test]
    fn parse_and_phases() {
        let r = Requirements::parse(EXAMPLE).unwrap();
        assert_eq!(r.at_phase(Phase::Save).len(), 2);
        assert_eq!(r.check_rev(2), Err(AgStatus::StaleRequirements));
        let mut m = Meter::default();
        assert!(m.charge_exact(&r.budget, 64).is_ok());
        assert_eq!(m.charge_exact(&r.budget, 1), Err(AgStatus::BudgetExhausted));
    }
}

#[cfg(test)]
mod satisfiability_tests {
    use super::*;
    #[test]
    fn flags_unreachable_evidence_and_missing_evaluators() {
        let r = Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"},"hard":[
            {"id":"R1","measure":"evaluate.measure_inertia","field":"volume","le":1.0,"evidence":"exact"},
            {"id":"R2","measure":"evaluate.measure_inertia","field":"volume","le":1.0,"evidence":"bounded"},
            {"id":"R3","measure":"evaluate.nonexistent","field":"x","le":1.0}]}"#).unwrap();
        let w = r.satisfiability();
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].starts_with("R1: requires 'exact' evidence but evaluate.measure_inertia provides at most 'bounded'"), "{w:?}");
        assert!(w[1].starts_with("R3: no implemented evaluator"), "{w:?}");
    }
}
