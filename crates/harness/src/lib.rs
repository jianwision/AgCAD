//! # agcad-harness — harness (spec 06, 14, 17)
//!
//! * [`requirements`]: immutable requirements, phases, evidence, process; budget meter.
//! * [`checker`]: Design Checker with frozen reference evaluations and evidence enforcement.
//! * [`session`]: transactions (workspace → version), program application, update, save,
//!   finish; traces; reports with untrusted-string quoting (AG-17-003).
//! * [`wavefront`]: GPU-resident wavefront stages (validate, bin, decide, meter, rank) with
//!   CUDA kernels and bit-identical CPU references.
//! * [`study`]: design studies (configurations, batched update, topology classes, rank with
//!   fallback, proxy fidelity, optimize, sensitivity, tolerance analysis, observe).
//! * [`render`]: offscreen rasterizer + PNG for vision feedback.
//! * [`review`]: comments, approvals and version diffs (spec 10 capability track).

pub mod checker;
pub mod render;
pub mod report;
pub mod requirements;
pub mod review;
pub mod session;
pub mod study;
pub mod wavefront;
pub mod describe;

pub use requirements::Requirements;
pub use session::HarnessSession;
