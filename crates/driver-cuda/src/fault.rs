//! CUresult → AgCAD error model. Sticky errors poison the worker (AG-08-003, AG-17-002).

use agcad_driver::{DeviceFault, DriverError, EnqueueError};
use cudarc::driver::DriverError as CuError;

/// CUresult codes that leave the context unusable (CUDA programming guide: "sticky").
const STICKY: &[i32] = &[
    700, // CUDA_ERROR_ILLEGAL_ADDRESS
    710, // CUDA_ERROR_ASSERT
    714, // CUDA_ERROR_HARDWARE_STACK_ERROR
    715, // CUDA_ERROR_ILLEGAL_INSTRUCTION
    716, // CUDA_ERROR_MISALIGNED_ADDRESS
    717, // CUDA_ERROR_INVALID_ADDRESS_SPACE
    718, // CUDA_ERROR_INVALID_PC
    719, // CUDA_ERROR_LAUNCH_FAILED
];

pub fn code(e: &CuError) -> i32 {
    e.0 as i32
}

pub fn is_sticky(c: i32) -> bool {
    STICKY.contains(&c)
}

pub fn fault(e: CuError, segment: Option<u64>) -> DeviceFault {
    let c = code(&e);
    DeviceFault { sticky: is_sticky(c), code: c, segment }
}

pub fn enqueue(e: CuError) -> EnqueueError {
    match code(&e) {
        2 => EnqueueError::OutOfMemory { requested: 0 }, // CUDA_ERROR_OUT_OF_MEMORY
        900 | 901 | 904 => EnqueueError::NotPermittedDuringCapture, // STREAM_CAPTURE_* errors
        c if is_sticky(c) => EnqueueError::Poisoned,
        c => EnqueueError::Other(c),
    }
}

pub fn driver(e: CuError) -> DriverError {
    let c = code(&e);
    if is_sticky(c) { DriverError::Fault(fault(e, None)) } else { DriverError::Enqueue(enqueue(e)) }
}

/// Convert a raw `CUresult` from `sys::` calls.
pub fn check(r: cudarc::driver::sys::CUresult) -> Result<(), CuError> {
    if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(CuError(r)) }
}
