//! Update / delivery reports (spec 06 § Update report). Strings originating from document
//! content (names, attributes, UDF text, imported metadata) are wrapped as quoted data with
//! provenance, never mixed into instruction text (AG-17-003).

use crate::checker::{CheckResult, Verdict};
use serde::Serialize;
use serde_json::{json, Value};

/// Wrap an untrusted string with its provenance (AG-17-003).
pub fn untrusted(text: &str, provenance: &str) -> Value {
    json!({ "untrusted_text": text, "provenance": provenance })
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureLine {
    pub feature: Value,
    pub status: String,
    pub code: u16,
    pub message: Option<Value>,
    pub elements: Vec<Value>,
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateReport {
    pub status: String,
    pub revision: u32,
    pub requirements_rev: u32,
    pub workspace_state: String,
    pub features: Vec<FeatureLine>,
    pub checks: Vec<CheckResult>,
    pub budget: Value,
    pub unsupported: Vec<String>,
    pub topology_classes: Option<Value>,
    pub proxy_fidelity: Option<Value>,
    /// Study results (ranked configurations etc.), if the program ran a study.
    pub study: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeliveryReport {
    /// The four separate outcomes (spec 06): stop / execution / requirements / delivery.
    pub stopped: bool,
    pub execution_ok: bool,
    pub requirements_met: bool,
    pub delivery_verified: bool,
    pub checks: Vec<CheckResult>,
    pub files: Vec<Value>,
}

impl DeliveryReport {
    pub fn from_checks(checks: Vec<CheckResult>, execution_ok: bool, files: Vec<Value>, delivery_verified: bool) -> Self {
        let requirements_met = checks.iter().all(|c| c.verdict == Verdict::Pass);
        DeliveryReport { stopped: true, execution_ok, requirements_met, delivery_verified, checks, files }
    }
}
