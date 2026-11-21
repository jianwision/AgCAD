//! # agcad-trace — traces (spec 06 § Trace; spec 07 § Cross-path equivalence)
//!
//! Every program, record, verdict, checker version, kernel version and budget event is
//! recorded. Traces serialize to JSON Lines; replay comparison reports the first divergence
//! (bit-identical replay on the same device/binary, AG-05-001).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    Session { session: u32, device: String, driver_version: u32, catalog_hash: String, build: String },
    Program { requirements_rev: u32, expected_revision: u32, agir_json: String },
    Feature { feature_ref: String, status: u16, message: Option<String> },
    Check { requirement: String, verdict: String, value: f64, bound: f64, evidence: String, checker_version: u32 },
    Budget { resource: String, used: f64, remaining: f64 },
    Version { branch: String, version: u32, label: String },
    Segment { index: u32, configurations: u32, gpu_ms: f64 },
    Status { configuration: u32, opcode: u16, status: u16, budget_used: u32 },
    Note { text: String },
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
}

impl Trace {
    pub fn push(&mut self, e: TraceEvent) { self.events.push(e); }

    pub fn to_jsonl(&self) -> String {
        self.events.iter().map(|e| serde_json::to_string(e).expect("serializable")).collect::<Vec<_>>().join("\n") + "\n"
    }

    pub fn from_jsonl(s: &str) -> Result<Trace, String> {
        let events = s.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
        Ok(Trace { events })
    }

    /// First diverging event index (excluding timing-only fields), or `None` if identical.
    pub fn diverges_from(&self, other: &Trace) -> Option<usize> {
        let norm = |e: &TraceEvent| match e { TraceEvent::Segment { index, configurations, .. } => TraceEvent::Segment { index: *index, configurations: *configurations, gpu_ms: 0.0 }, x => x.clone() };
        let n = self.events.len().max(other.events.len());
        (0..n).find(|&i| self.events.get(i).map(norm) != other.events.get(i).map(norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn jsonl_roundtrip_and_divergence() {
        let mut t = Trace::default();
        t.push(TraceEvent::Note { text: "a".into() });
        t.push(TraceEvent::Segment { index: 0, configurations: 4096, gpu_ms: 3.2 });
        let back = Trace::from_jsonl(&t.to_jsonl()).unwrap();
        assert_eq!(back, t);
        let mut u = back.clone();
        if let TraceEvent::Segment { gpu_ms, .. } = &mut u.events[1] { *gpu_ms = 9.9; }
        assert_eq!(t.diverges_from(&u), None, "timing does not count as divergence");
        u.push(TraceEvent::Note { text: "b".into() });
        assert_eq!(t.diverges_from(&u), Some(2));
    }
}
