//! Error model (spec 08; review A1-2). Enqueue errors are returned when work is queued;
//! device faults surface asynchronously at `query`/`sync`. Command outcomes are data
//! (`AgStatus` in status records), never Rust errors.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EnqueueError {
    #[error("out of device memory (requested {requested} bytes)")]
    OutOfMemory { requested: u64 },
    #[error("invalid launch configuration: {0}")]
    InvalidConfig(String),
    #[error("operation not permitted during stream capture")]
    NotPermittedDuringCapture,
    #[error("worker is poisoned by an earlier sticky fault")]
    Poisoned,
    #[error("unsupported on this device: {0}")]
    Unsupported(&'static str),
    #[error("backend error {0}")]
    Other(i32),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("device fault code {code} (sticky={sticky}, segment={segment:?})")]
pub struct DeviceFault {
    /// Sticky faults poison the context: the worker must be restarted (AG-08-003, AG-17-002).
    pub sticky: bool,
    pub code: i32,
    pub segment: Option<u64>,
}

#[derive(Debug, Error)]
pub enum DriverError {
    #[error(transparent)]
    Enqueue(#[from] EnqueueError),
    #[error(transparent)]
    Fault(#[from] DeviceFault),
    #[error("compilation failed: {0}")]
    Compile(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("io: {0}")]
    Io(String),
}

impl DriverError {
    pub fn requires_worker_restart(&self) -> bool {
        matches!(self, Self::Enqueue(EnqueueError::Poisoned) | Self::Fault(DeviceFault { sticky: true, .. }))
    }
}

pub type DriverResult<T> = Result<T, DriverError>;

/// Worker health. A sticky fault moves a worker to `Poisoned`; the harness then kills the
/// worker process and restores the last snapshot (spec 16 § 5).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorkerState {
    Healthy,
    Poisoned,
}
