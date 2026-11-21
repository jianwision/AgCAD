//! # agcad-runtime — runtime API (cf. CUDA `cudart`)
//!
//! Implicit, convenient layer over `agcad-driver` (spec 01 § Project layout):
//! * [`session`]: default session per thread/device; runtime → driver handle interop.
//! * [`document`]: Part/Product, Bodies, Geometrical Sets, specification tree, Knowledgeware,
//!   versions/branches/workspaces (copy-on-write).
//! * [`rebuild`]: `document.update` engine + feature registry (implemented by agcad-std).
//! * [`naming`]: stable names, resolution with lineage, signature recovery.
//! * [`sched`]: ExecGraph lowering, topology classes, capacity planner, representation cache.
//! * [`file`]: `.agcad` save/open, migrations, snapshots.

pub mod document;
pub mod file;
pub mod naming;
pub mod rebuild;
pub mod sched;
pub mod session;

pub use document::{PartDocument, PartState, Workspace};
pub use rebuild::{BuildContext, FeatureError, FeatureImpl, Product, Registry};
pub use session::{with_default, Session};
