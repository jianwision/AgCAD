//! `ModelAdaptor` (spec 07; ported from Demo `scripts/model_adaptor.py`). Switching adaptors
//! never changes requirements, op semantics or report definitions (AG-07-001); missing
//! capabilities return `UNSUPPORTED_OP` (AG-07-002).

use agcad_driver::AgStatus;
use agcad_ir::Program;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub name: String,
    /// Allowed opcodes (empty = all catalog ops).
    pub opcodes: Vec<u16>,
    pub precision: String,
    pub max_context_tokens: Option<u64>,
    pub device_resident: bool,
}

/// What the model sees: requirements, last report, renders, observation tensor.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub requirements: serde_json::Value,
    pub report: serde_json::Value,
    pub renders_png_base64: Vec<String>,
    pub tensor: Vec<f32>,
    pub tensor_desc: serde_json::Value,
    pub revision: u32,
    pub requirements_rev: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Program(Program),
    /// Device-resident rollout: number of steps the CuModel runs on device.
    DeviceRollout { steps: u32 },
    Finish,
}

pub trait ModelAdaptor: Send {
    fn capabilities(&self) -> &CapabilityManifest;
    fn decide(&mut self, obs: &Observation) -> Result<Decision, (AgStatus, String)>;
}
