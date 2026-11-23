//! Design Checker (spec 06): requirements evaluated with the frozen standard evaluation
//! implementations (never model-authored ones, AG-14-005); evidence levels enforced
//! (AG-06-004); verdicts are data, never silent repairs (AG-06-003).

use crate::requirements::{Evidence, HardRequirement};
use agcad_geo::kernel::ExactKernel;
use agcad_ir::program::Value;
use agcad_runtime::document::{FeatureNode, PartState};
use agcad_runtime::rebuild::{BuildContext, Product, Registry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHECKER_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict { Pass, Fail, Unknown }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckResult {
    pub requirement: String,
    pub verdict: Verdict,
    pub value: serde_json::Value,
    /// Signed, tolerance-normalized margin (spec 14): ≥ 0 passes.
    pub margin: Option<f64>,
    pub evidence: Evidence,
    pub required_evidence: Evidence,
    pub checker_version: u32,
    pub escalation: Option<String>,
    pub message: Option<String>,
}

fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => n.as_i64().map(Value::Int).unwrap_or_else(|| Value::Float(n.as_f64().unwrap_or(f64::NAN))),
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(a) => Value::List(a.iter().map(json_to_value).collect()),
        serde_json::Value::Object(o) => Value::Map(o.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect()),
    }
}

/// Evaluate one requirement against the body produced by feature `body_feature`.
pub fn check(req: &HardRequirement, part: &PartState, kernel: &dyn ExactKernel, registry: &Registry, products: &BTreeMap<u32, Product>, body_feature: u32) -> CheckResult {
    let fail = |msg: String| CheckResult { requirement: req.id.clone(), verdict: Verdict::Unknown, value: serde_json::Value::Null, margin: None, evidence: Evidence::Unknown, required_evidence: req.evidence, checker_version: CHECKER_VERSION, escalation: None, message: Some(msg) };
    let Some(imp) = registry.get(&req.measure) else { return fail(format!("no frozen implementation for {}", req.measure)); };
    let mut args: BTreeMap<String, Value> = req.args.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect();
    let body_ref = Value::Ref { id: format!("f{body_feature}"), field: None };
    args.insert("body".into(), body_ref.clone());
    args.insert("items".into(), Value::List(vec![body_ref]));
    let node = FeatureNode { id: u32::MAX, op: req.measure.clone(), op_version: 1, args, container: 0, active: true, role: None };
    let mut ctx = BuildContext { part, kernel, dispatches: BTreeMap::new(), products: products.clone(), bodies: BTreeMap::new(), params: BTreeMap::new(), exprs: vec![], tools: BTreeMap::new(), names: BTreeMap::new() };
    let out = match imp.execute(&mut ctx, &node) {
        Ok(Product::Json(j)) => j,
        Ok(other) => return fail(format!("{} returned {other:?}, expected a report", req.measure)),
        Err(e) => return fail(format!("{}: {}", req.measure, e.message)),
    };
    let value = out.get(&req.field).cloned().unwrap_or(serde_json::Value::Null);
    let got_evidence = out.get("evidence").and_then(|e| e.as_str()).map(Evidence::parse).unwrap_or(Evidence::Sampled);
    let (verdict, margin) = if let Some(eq) = &req.eq {
        (if &value == eq { Verdict::Pass } else { Verdict::Fail }, None)
    } else {
        let Some(x) = value.as_f64().filter(|x| x.is_finite()) else {
            return fail(format!("{} did not produce a finite numeric field '{}'", req.measure, req.field));
        };
        let mut m = f64::INFINITY;
        if let Some(le) = req.le { m = m.min((le - x) / le.abs().max(1e-9)); }
        if let Some(ge) = req.ge { m = m.min((x - ge) / ge.abs().max(1e-9)); }
        if m.is_nan() { (Verdict::Unknown, None) } else { (if m >= 0.0 { Verdict::Pass } else { Verdict::Fail }, Some(m)) }
    };
    // Insufficient evidence never passes (AG-06-004); a fail with weaker evidence stays a fail.
    let (verdict, escalation) = if got_evidence < req.evidence && verdict == Verdict::Pass {
        (Verdict::Unknown, Some(format!("re-evaluate {} at {:?} evidence (escalation chain, spec 14)", req.measure, req.evidence)))
    } else { (verdict, None) };
    CheckResult { requirement: req.id.clone(), verdict, value, margin, evidence: got_evidence, required_evidence: req.evidence, checker_version: CHECKER_VERSION, escalation, message: None }
}
