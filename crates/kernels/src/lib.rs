//! # agcad-kernels — kernel registry, parameter blocks, CPU references, orchestration
//!
//! Every GPU kernel has (a) a CUDA C++ implementation in `agcad-kernels-cuda/cu/*.cu` and
//! (b) a CPU reference here with the **same name and parameter block** (AG-08-001/005).
//! Host-side orchestration (multi-launch algorithms) is generic over any
//! [`agcad_driver::KernelExt`] device, so CUDA and the CPU oracle run identical sequences.
//!
//! Kernel parameter blocks are `repr(C)` structs of device addresses (`u64`) and scalars,
//! uploaded to a device buffer and passed as the kernel's single argument (review A2-1).

pub mod params;
pub mod primitives;
pub mod reference;

pub use params::{upload_params, ParamUpload};

/// Library name of the primitives (scan/sort/compact/reduce/top-k).
pub const PRIMITIVES: &str = "ag_primitives";
