//! Review workflow (`document.comment` / `approve` / `diff`; spec 10 capability track).

use agcad_runtime::document::{PartDocument, VersionId};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comment {
    pub version: VersionId,
    pub elements: Vec<String>,
    pub text: String,
    pub author: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision { Approved, Rejected, ChangesRequested }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Approval {
    pub version: VersionId,
    pub decision: Decision,
    pub note: Option<String>,
    pub approver: String,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ReviewLog {
    pub comments: Vec<Comment>,
    pub approvals: Vec<Approval>,
}

impl ReviewLog {
    /// Delivery gating: approved iff the latest decision for the version is `Approved`.
    pub fn approved(&self, v: VersionId) -> bool {
        self.approvals.iter().rev().find(|a| a.version == v).is_some_and(|a| a.decision == Decision::Approved)
    }
}

/// Feature-level diff between two versions (added / removed / changed parameters).
pub fn diff(doc: &PartDocument, a: VersionId, b: VersionId) -> serde_json::Value {
    let (sa, sb) = (&doc.versions[a as usize].state, &doc.versions[b as usize].state);
    let added: Vec<_> = sb.features.keys().filter(|k| !sa.features.contains_key(k)).map(|k| sb.features[k].feature_ref()).collect();
    let removed: Vec<_> = sa.features.keys().filter(|k| !sb.features.contains_key(k)).map(|k| sa.features[k].feature_ref()).collect();
    let changed: Vec<_> = sa.features.iter().filter_map(|(k, f)| sb.features.get(k).filter(|g| g.args != f.args || g.active != f.active).map(|g| g.feature_ref())).collect();
    json!({ "from": a, "to": b, "added": added, "removed": removed, "changed": changed })
}
