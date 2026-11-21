//! # agcad-ir — AgIR
//!
//! Typed SSA programs over the feature catalog (`docs/spec/op-catalog.json`), with the JSON
//! encoding used by LLMs and the bytecode encoding used by device models (spec 04).

pub mod bytecode;
pub mod catalog;
pub mod formula;
pub mod json;
pub mod lint;
pub mod program;
pub mod types;
pub mod units;
pub mod verify;

pub use catalog::{Catalog, OpManifest};
pub use program::{Program, Stmt, Value};
pub use units::{Dimension, Quantity};
