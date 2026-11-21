//! Sessions (runtime API, cf. `cudart` primary context): a default session per thread,
//! lazily created, owning documents and the device choice. The CPU reference device is
//! opt-in only (`AGCAD_ALLOW_CPU_REFERENCE=1`) because it is never a production path
//! (AG-08-005); without CUDA the session reports the error explicitly.

use crate::document::PartDocument;
use agcad_driver::{AgDevice, DriverError, EnqueueError};
use agcad_driver_cpu::CpuDevice;
use agcad_driver_cuda::CudaDevice;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Clone)]
pub enum DeviceChoice {
    Cuda(Arc<CudaDevice>),
    /// Conformance/testing only.
    CpuReference(Arc<CpuDevice>),
}

impl DeviceChoice {
    pub fn name(&self) -> String {
        match self { DeviceChoice::Cuda(d) => d.caps().name.clone(), DeviceChoice::CpuReference(d) => d.caps().name.clone() }
    }
}

pub struct Session {
    pub id: u32,
    pub device: DeviceChoice,
    pub parts: Vec<PartDocument>,
}

impl Session {
    pub fn open(ordinal: usize) -> Result<Session, DriverError> {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let device = match CudaDevice::open(ordinal) {
            Ok(d) => DeviceChoice::Cuda(d),
            Err(e) => {
                if std::env::var("AGCAD_ALLOW_CPU_REFERENCE").as_deref() == Ok("1") {
                    DeviceChoice::CpuReference(CpuDevice::new())
                } else {
                    return Err(match e {
                        DriverError::Enqueue(x) => DriverError::Enqueue(x),
                        other => DriverError::Enqueue(EnqueueError::Unsupported(Box::leak(format!("no CUDA device available ({other}); set AGCAD_ALLOW_CPU_REFERENCE=1 for reference/testing only").into_boxed_str()))),
                    });
                }
            }
        };
        Ok(Session { id, device, parts: Vec::new() })
    }

    pub fn new_part(&mut self, name: &str) -> usize {
        self.parts.push(PartDocument::new(name));
        self.parts.len() - 1
    }
}

thread_local! {
    static DEFAULT: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// Run `f` with the thread's default session (created on first use, device 0).
pub fn with_default<R>(f: impl FnOnce(&mut Session) -> R) -> Result<R, DriverError> {
    DEFAULT.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() { *slot = Some(Session::open(0)?); }
        Ok(f(slot.as_mut().unwrap()))
    })
}
