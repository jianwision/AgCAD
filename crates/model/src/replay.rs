//! `ReplayModel`: replays recorded programs from a trace (regression, cross-path checks).

use crate::adaptor::*;
use agcad_driver::AgStatus;
use agcad_trace::{Trace, TraceEvent};

pub struct ReplayModel {
    manifest: CapabilityManifest,
    programs: Vec<String>,
    next: usize,
}

impl ReplayModel {
    pub fn from_trace(t: &Trace) -> Self {
        let programs = t.events.iter().filter_map(|e| if let TraceEvent::Program { agir_json, .. } = e { Some(agir_json.clone()) } else { None }).collect();
        ReplayModel { manifest: CapabilityManifest { name: "replay".into(), ..Default::default() }, programs, next: 0 }
    }
}

impl ModelAdaptor for ReplayModel {
    fn capabilities(&self) -> &CapabilityManifest { &self.manifest }
    fn decide(&mut self, obs: &Observation) -> Result<Decision, (AgStatus, String)> {
        let Some(src) = self.programs.get(self.next) else { return Ok(Decision::Finish) };
        self.next += 1;
        let mut p = agcad_ir::json::decode_program(src, agcad_ir::Catalog::global()).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
        // Replays target the live revision; requirements revision must still match (AG-06-001).
        p.expected_revision = obs.revision;
        if p.requirements_rev != obs.requirements_rev { return Err((AgStatus::StaleRequirements, "recorded program targets another requirements revision".into())); }
        Ok(Decision::Program(p))
    }
}
