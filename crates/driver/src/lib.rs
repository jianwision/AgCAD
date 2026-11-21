//! # agcad-driver — AgCAD driver API
//!
//! The explicit, handle-based layer of AgCAD, analogous to CUDA's driver API (`libcuda`).
//! See `docs/spec/01-architecture.md` § Project layout and `docs/spec/08-hal-backends.md`.
//!
//! * [`device`]: [`AgDevice`] core trait and extension traits (`KernelExt`, `GraphExt`,
//!   `ConditionalExt`, `VmmExt`, `PartitionExt`, `RayTracingExt`, `LibraryExt`).
//! * [`status`]: [`AgStatus`], fixed append-only `u16` codes (spec 06).
//! * [`abi`]: bytecode block header / command / status records (spec 04).
//! * [`arena`], [`handle`]: generational slab arenas and handles.
//! * [`layout`]: [`BatchLayout`] (strided / segmented / grouped), alignment, df64.
//! * [`caps`]: device capabilities and the per-device precision policy (AG-08-002).
//! * [`error`]: enqueue errors vs asynchronous device faults (AG-08-003).
//!
//! The driver never depends on the runtime (AG-01-001). Only `agcad-driver-*` crates use
//! `cudarc` / wgpu / OptiX directly (AG-01-002).

pub mod abi;
pub mod arena;
pub mod caps;
pub mod device;
pub mod error;
pub mod handle;
pub mod layout;
pub mod status;

pub use abi::{BlockHeader, CommandRecord, StatusRecord, ABI_VERSION};
pub use arena::Arena;
pub use caps::{BackendKind, Caps, CondSupport, PrecisionPolicy, TensorTypes};
pub use device::*;
pub use error::{DeviceFault, DriverError, DriverResult, EnqueueError, WorkerState};
pub use handle::Handle;
pub use layout::{align_up, BatchLayout, Df64, Dim3, Shape, DEVICE_ALIGN};
pub use status::{AgStatus, StatusCategory};

/// Driver ABI version reported to runtimes (backward compatible within a MAJOR; AG-01-004).
pub const DRIVER_VERSION: u32 = 1;
