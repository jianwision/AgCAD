//! # agcad-driver-wgpu — wgpu/CubeCL driver (Tier P)
//!
//! Placeholder until the CubeCL pass (plan P12). The Tier P contract is identical to the
//! CUDA device's `AgDevice` + `KernelExt`; kernels will be CubeCL ports of the Tier N
//! kernels with the same names and parameter blocks. Until then `open()` reports
//! `Unsupported`, so callers fall back to CUDA or fail explicitly (never silently).

use agcad_driver::{DriverError, EnqueueError};

pub fn open(_adapter: usize) -> Result<(), DriverError> {
    Err(EnqueueError::Unsupported("wgpu/CubeCL backend lands in plan pass P12").into())
}
