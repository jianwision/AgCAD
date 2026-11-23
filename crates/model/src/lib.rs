//! # agcad-model — model adaptors (spec 07)
//!
//! * [`adaptor`]: `ModelAdaptor`, `Observation`, `Decision`, capability manifests.
//! * [`llm`]: localhost HTTP bridge adaptor. [`replay`]: trace replay.
//! * [`cumodel`]: device-resident MLP policy (safetensors; CUDA + CPU-reference kernels).
//! * [`encoder`]: geometry descriptor for observations and retrieval.
//! * `agcad-train` binary (`--features train`): Candle trainer in a separate process.

pub mod adaptor;
pub mod cumodel;
pub mod dagger;
pub mod encoder;
pub mod llm;
pub mod replay;
pub mod tokenizer;

pub use adaptor::{CapabilityManifest, Decision, ModelAdaptor, Observation};
