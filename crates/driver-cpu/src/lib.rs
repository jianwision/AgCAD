//! # agcad-driver-cpu — CPU reference device
//!
//! Conformance oracle only (AG-08-005): host memory stands in for device memory; streams are
//! synchronous; kernels are the CPU reference implementations registered in each
//! [`LibrarySource`]. Device addresses are host addresses, so parameter blocks of `u64`
//! pointers work unchanged on both devices.

use agcad_driver::*;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// 128-byte aligned host allocation acting as device memory.
pub struct CpuBuffer {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    fn new(bytes: usize) -> Result<Self, EnqueueError> {
        let layout = Layout::from_size_align(bytes.max(1), DEVICE_ALIGN).map_err(|_| EnqueueError::InvalidConfig("layout".into()))?;
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(EnqueueError::OutOfMemory { requested: bytes as u64 });
        }
        Ok(CpuBuffer { ptr, bytes })
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr valid for `bytes` for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }
    /// Mutable view. The reference device is single-threaded per stream; callers must not alias.
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated with the same layout in `new`.
        unsafe { dealloc(self.ptr, Layout::from_size_align_unchecked(self.bytes.max(1), DEVICE_ALIGN)) }
    }
}

impl RawBuffer for CpuBuffer {
    fn bytes(&self) -> usize { self.bytes }
    fn device_ptr(&self) -> u64 { self.ptr as u64 }
}

pub struct CpuPinned(Vec<u8>);

impl PinnedHost for CpuPinned {
    fn as_bytes(&self) -> &[u8] { &self.0 }
    fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.0 }
}

#[derive(Clone, Debug)]
pub struct CpuStream {
    pub priority: Priority,
}

pub struct CpuEvent(Instant);

pub struct CpuDevice {
    caps: Caps,
    libraries: Mutex<Vec<LibrarySource>>,
}

impl CpuDevice {
    pub fn new() -> Arc<Self> {
        Arc::new(CpuDevice {
            caps: Caps {
                name: "AgCAD CPU reference device".into(),
                backend: BackendKind::CpuReference,
                compute_capability: None,
                driver_version: DRIVER_VERSION,
                fp64_rate: 1.0,
                fp64_gemm_emulation: false,
                tensor_types: TensorTypes::default(),
                rt_cores: false,
                conditional_nodes: CondSupport::HostEmulated,
                green_contexts: false,
                subgroup_size: 1,
                atomics64: true,
                max_alloc: u64::MAX,
                total_memory: u64::MAX,
            },
            libraries: Mutex::new(Vec::new()),
        })
    }
}

fn copy(src: &[u8], src_off: usize, dst: &mut [u8], dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
    if src_off + bytes > src.len() || dst_off + bytes > dst.len() {
        return Err(EnqueueError::InvalidConfig(format!("copy out of range ({bytes} bytes)")));
    }
    dst[dst_off..dst_off + bytes].copy_from_slice(&src[src_off..src_off + bytes]);
    Ok(())
}

impl AgDevice for CpuDevice {
    type Stream = CpuStream;
    type Event = CpuEvent;
    type Buffer = CpuBuffer;
    type Pinned = CpuPinned;

    fn caps(&self) -> &Caps { &self.caps }
    fn state(&self) -> WorkerState { WorkerState::Healthy }
    fn stream(&self, priority: Priority) -> DriverResult<CpuStream> { Ok(CpuStream { priority }) }

    fn alloc_bytes(&self, _s: &CpuStream, bytes: usize) -> Result<CpuBuffer, EnqueueError> { CpuBuffer::new(bytes) }

    fn memset_zero_async(&self, _s: &CpuStream, buf: &CpuBuffer, offset: usize, bytes: usize) -> Result<(), EnqueueError> {
        let d = buf.as_mut_slice();
        if offset + bytes > d.len() { return Err(EnqueueError::InvalidConfig("memset out of range".into())); }
        d[offset..offset + bytes].fill(0);
        Ok(())
    }

    fn pinned(&self, bytes: usize) -> DriverResult<CpuPinned> { Ok(CpuPinned(vec![0; bytes])) }

    fn upload_from_host_async(&self, _s: &CpuStream, src: &[u8], dst: &CpuBuffer, dst_off: usize) -> Result<(), EnqueueError> {
        copy(src, 0, dst.as_mut_slice(), dst_off, src.len())
    }

    fn upload_async(&self, _s: &CpuStream, src: &CpuPinned, src_off: usize, dst: &CpuBuffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        copy(&src.0, src_off, dst.as_mut_slice(), dst_off, bytes)
    }

    fn download_async(&self, _s: &CpuStream, src: &CpuBuffer, src_off: usize, dst: &mut CpuPinned, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        copy(src.as_slice(), src_off, &mut dst.0, dst_off, bytes)
    }

    fn memset_async(&self, _s: &CpuStream, dst: &CpuBuffer, off: usize, bytes: usize, value: u8) -> Result<(), EnqueueError> {
        let d = dst.as_mut_slice();
        if off + bytes > d.len() { return Err(EnqueueError::InvalidConfig("memset out of range".into())); }
        d[off..off + bytes].fill(value);
        Ok(())
    }

    fn copy_device_async(&self, _s: &CpuStream, src: &CpuBuffer, src_off: usize, dst: &CpuBuffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        if src.ptr == dst.ptr {
            let d = dst.as_mut_slice();
            d.copy_within(src_off..src_off + bytes, dst_off);
            return Ok(());
        }
        copy(src.as_slice(), src_off, dst.as_mut_slice(), dst_off, bytes)
    }

    fn record(&self, _s: &CpuStream) -> Result<CpuEvent, EnqueueError> { Ok(CpuEvent(Instant::now())) }
    fn wait(&self, _s: &CpuStream, _e: &CpuEvent) -> Result<(), EnqueueError> { Ok(()) }
    fn query(&self, _e: &CpuEvent) -> Result<bool, DeviceFault> { Ok(true) }
    fn sync(&self, _s: &CpuStream) -> Result<(), DeviceFault> { Ok(()) }
    fn elapsed_ms(&self, a: &CpuEvent, b: &CpuEvent) -> Result<f32, DeviceFault> {
        Ok(b.0.saturating_duration_since(a.0).as_secs_f32() * 1000.0)
    }
}

impl KernelExt for CpuDevice {
    fn load_library(&self, src: &LibrarySource) -> DriverResult<LibraryId> {
        let mut libs = self.libraries.lock().expect("library registry poisoned");
        if let Some(i) = libs.iter().position(|l| l.name == src.name) {
            return Ok(LibraryId(i as u32));
        }
        libs.push(src.clone());
        Ok(LibraryId((libs.len() - 1) as u32))
    }

    fn launch(&self, _s: &CpuStream, lib: LibraryId, kernel: &str, cfg: &LaunchConfig, params: &CpuBuffer) -> Result<(), EnqueueError> {
        let f = {
            let libs = self.libraries.lock().expect("library registry poisoned");
            let l = libs.get(lib.0 as usize).ok_or(EnqueueError::InvalidConfig("unknown library".into()))?;
            l.cpu_kernels.iter().find(|(n, _)| *n == kernel).map(|(_, f)| *f).ok_or_else(|| EnqueueError::InvalidConfig(format!("no CPU reference for kernel '{kernel}'")))?
        };
        // SAFETY: the kernel's parameter block layout is the contract shared with its CUDA twin.
        unsafe { f(params.as_slice().as_ptr(), cfg) };
        Ok(())
    }
}
