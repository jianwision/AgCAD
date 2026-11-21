//! # agcad-dops — device operations for GPU-model-driven CAD
//!
//! The device half of the loop "LLM observes renders → sets a goal → GPU model plans and
//! executes CAD operations under the GPU harness → survivors escalate to exact B-rep":
//!
//! - [`abi`]: feature tapes (CATIA Part Design features on the device), goal, trace rows.
//! - [`registry`]: device ops ↔ catalog ops; allow-list and dispatch table.
//! - [`math`] / [`reference`] / `cu/dops.cu`: deterministic SDF bounds, rigorous interval
//!   evaluation, mock policy, apply/decide/continue kernels (CPU reference ≡ CUDA, bitwise).
//! - [`studio`]: the device-resident rollout (graph WHILE on CUDA, host loop on CPU).
//! - [`lower`]: device designs ⇄ exact layered solids and AgIR programs (escalation, raise).

pub mod abi;
pub mod examples;
pub mod lower;
pub mod math;
pub mod reference;
pub mod registry;
pub mod studio;

pub use abi::{Goal, Node};
pub use studio::{Design, GraphRollout, RolloutSummary, Studio, StudioConfig, Survivor};
