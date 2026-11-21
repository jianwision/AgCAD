//! # agcad-geo — exact geometry kernel
//!
//! Spec 02 / 18. Pass P5 contents (CPU reference implementations; GPU kernels for the
//! data-parallel stages — SSI seeding/marching, topology builders, tessellation, STEP
//! encode/decode — follow in P11 and must match these bit-for-bit or within declared bounds):
//! * [`math`], [`nurbs`], [`curve`], [`surface`]: analytic + NURBS geometry with jets.
//! * [`predicates`]: robust orientation/side predicates (AG-02-004).
//! * [`brep`]: half-edge B-rep with names, tolerances, validity (AG-02-001..003).
//! * [`builders`]: prism / revolve / primitives; [`region2d`] + [`layered`]: exact 2.5D
//!   booleans for same-direction pad/pocket/hole stacks.
//! * [`intersect`]: co-surface detection + closed-form analytic intersection table.
//! * [`tessellate`]: watertight adaptive tessellation + mass properties.
//! * [`step_write`], [`step_read`]: STEP AP242/AP214 with receiver profiles and validation
//!   properties; [`threemf`], [`zip`]: 3MF with materials + beam lattices.
//! * [`kernel`]: `ExactKernel` trait and the native implementation.

pub mod brep;
pub mod builders;
pub mod curve;
pub mod intersect;
pub mod kernel;
pub mod layered;
pub mod math;
pub mod nurbs;
pub mod predicates;
pub mod region2d;
pub mod ssi;
pub mod step_read;
pub mod step_write;
pub mod surface;
pub mod tessellate;
pub mod threemf;
pub mod zip;

pub use brep::Brep;
pub use kernel::{ExactKernel, NativeKernel, Solid};
pub use math::{v3, Frame, V3};
