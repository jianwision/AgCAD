//! Device API: core [`AgDevice`] + extension traits (spec 08 § Device API; review A1-1).
//!
//! Capability lives in extension traits implemented only where it exists, so absence is
//! a compile-time / dispatch-time fact. CUDA vocabulary stays below this line (A3-3).
//!
//! Deviation from spec 08 noted for the build pass: kernel loading/launch is split into
//! [`KernelExt`] (implemented by CUDA *and* the CPU reference device, so conformance can
//! run the same kernel names on both) and [`LibraryExt`] (CUDA-only NVIDIA library handles).

use crate::caps::Caps;
use crate::error::{DeviceFault, DriverResult, EnqueueError, WorkerState};
use crate::layout::Dim3;
use bytemuck::Pod;
use core::marker::PhantomData;

/// Stream priority (harness high, proxy mid, exact/rebuild low; spec 05 § Overlap).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    High,
    Normal,
    Low,
}

/// Raw device allocation. Dropping it must free stream-ordered (AG-08-004).
pub trait RawBuffer: Send + Sync {
    fn bytes(&self) -> usize;
    /// Device address as seen by kernels (host address on the CPU reference device).
    fn device_ptr(&self) -> u64;
}

/// Page-locked host staging buffer (A2-4).
pub trait PinnedHost: Send {
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

/// Typed, stream-bound device slice. Never exposed to model code (handles are).
pub struct DeviceSlice<T: Pod, B: RawBuffer> {
    raw: B,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: Pod, B: RawBuffer> DeviceSlice<T, B> {
    /// # Safety-free constructor: caller guarantees `raw.bytes() >= len * size_of::<T>()`.
    pub fn from_raw(raw: B, len: usize) -> Self {
        debug_assert!(raw.bytes() >= len * core::mem::size_of::<T>());
        Self { raw, len, _t: PhantomData }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn raw(&self) -> &B { &self.raw }
    pub fn into_raw(self) -> B { self.raw }
    pub fn device_ptr(&self) -> u64 { self.raw.device_ptr() }
    pub fn bytes(&self) -> usize { self.len * core::mem::size_of::<T>() }
}

/// A region of a device buffer for packed readback.
pub struct Region<'a, B: RawBuffer> {
    pub buf: &'a B,
    pub offset: usize,
    pub bytes: usize,
}

/// Core device trait: CUDA, wgpu and the CPU reference implement it.
pub trait AgDevice: Send + Sync + 'static {
    type Stream: Clone + Send + Sync;
    type Event: Send + Sync;
    type Buffer: RawBuffer;
    type Pinned: PinnedHost;

    fn caps(&self) -> &Caps;
    fn state(&self) -> WorkerState;
    fn stream(&self, priority: Priority) -> DriverResult<Self::Stream>;

    /// Stream-ordered allocation (pool-backed on CUDA: `cudaMallocAsync`).
    fn alloc_bytes(&self, s: &Self::Stream, bytes: usize) -> Result<Self::Buffer, EnqueueError>;
    fn memset_zero_async(&self, s: &Self::Stream, buf: &Self::Buffer, offset: usize, bytes: usize) -> Result<(), EnqueueError>;
    fn pinned(&self, bytes: usize) -> DriverResult<Self::Pinned>;

    /// Copy from ordinary (pageable) host memory. The source may be freed when this returns:
    /// CUDA stages pageable sources before returning. Used for small parameter blocks.
    fn upload_from_host_async(&self, s: &Self::Stream, src: &[u8], dst: &Self::Buffer, dst_off: usize) -> Result<(), EnqueueError>;
    fn upload_async(&self, s: &Self::Stream, src: &Self::Pinned, src_off: usize, dst: &Self::Buffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError>;
    fn download_async(&self, s: &Self::Stream, src: &Self::Buffer, src_off: usize, dst: &mut Self::Pinned, dst_off: usize, bytes: usize) -> Result<(), EnqueueError>;
    fn copy_device_async(&self, s: &Self::Stream, src: &Self::Buffer, src_off: usize, dst: &Self::Buffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError>;
    /// Fill `bytes` bytes of `dst` at `off` with `value`, stream-ordered (no host staging).
    fn memset_async(&self, s: &Self::Stream, dst: &Self::Buffer, off: usize, bytes: usize, value: u8) -> Result<(), EnqueueError>;

    fn record(&self, s: &Self::Stream) -> Result<Self::Event, EnqueueError>;
    fn wait(&self, s: &Self::Stream, e: &Self::Event) -> Result<(), EnqueueError>;
    /// Async faults surface here (A1-2).
    fn query(&self, e: &Self::Event) -> Result<bool, DeviceFault>;
    fn sync(&self, s: &Self::Stream) -> Result<(), DeviceFault>;
    fn elapsed_ms(&self, a: &Self::Event, b: &Self::Event) -> Result<f32, DeviceFault>;

    // ---- provided ----

    fn alloc<T: Pod>(&self, s: &Self::Stream, n: usize) -> Result<DeviceSlice<T, Self::Buffer>, EnqueueError> {
        let bytes = n.checked_mul(core::mem::size_of::<T>()).ok_or(EnqueueError::OutOfMemory { requested: u64::MAX })?;
        Ok(DeviceSlice::from_raw(self.alloc_bytes(s, bytes.max(1))?, n))
    }

    fn alloc_zeroed<T: Pod>(&self, s: &Self::Stream, n: usize) -> Result<DeviceSlice<T, Self::Buffer>, EnqueueError> {
        let d = self.alloc::<T>(s, n)?;
        self.memset_zero_async(s, d.raw(), 0, d.bytes())?;
        Ok(d)
    }

    /// One packed readback into pinned staging, then one event (Demo single-readback rule).
    fn download_packed_async(&self, s: &Self::Stream, regions: &[Region<'_, Self::Buffer>], dst: &mut Self::Pinned) -> Result<Self::Event, EnqueueError> {
        let mut off = 0usize;
        for r in regions {
            self.download_async(s, r.buf, r.offset, dst, off, r.bytes)?;
            off += r.bytes;
        }
        self.record(s)
    }
}

// ------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------

/// Launch configuration in CUDA's shape vocabulary (A2-2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
    pub cluster: Option<Dim3>,
}

impl LaunchConfig {
    /// 1D launch covering `n` threads with `block` threads per block.
    pub fn linear(n: u64, block: u32) -> Self {
        let blocks = n.div_ceil(block as u64).max(1);
        LaunchConfig { grid: Dim3::x(blocks.min(u32::MAX as u64) as u32), block: Dim3::x(block), shared_mem_bytes: 0, cluster: None }
    }
}

/// CPU reference implementation of a kernel. `params` points at the parameter block
/// (a `repr(C)` struct of device addresses and scalars); on the CPU device, device
/// addresses are host addresses. Called once per *grid* (the implementation loops).
pub type CpuKernelFn = unsafe fn(params: *const u8, cfg: &LaunchConfig);

/// A kernel library: CUDA C++ source (Tier N) plus CPU reference kernels with the same names.
#[derive(Clone)]
pub struct LibrarySource {
    pub name: &'static str,
    /// Full CUDA C++ translation unit (headers already concatenated), or `None` if CPU-only.
    pub cuda_source: Option<String>,
    pub cpu_kernels: &'static [(&'static str, CpuKernelFn)],
    /// NVRTC options beyond the defaults.
    pub options: Vec<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibraryId(pub u32);

/// Load and launch kernels by name. Implemented by CUDA and the CPU reference device.
pub trait KernelExt: AgDevice {
    fn load_library(&self, src: &LibrarySource) -> DriverResult<LibraryId>;
    /// Kernels read parameters from a device-resident parameter block (A2-1).
    fn launch(&self, s: &Self::Stream, lib: LibraryId, kernel: &str, cfg: &LaunchConfig, params: &Self::Buffer) -> Result<(), EnqueueError>;
}

// ------------------------------------------------------------------------------------
// CUDA-only extensions
// ------------------------------------------------------------------------------------

/// NVIDIA library handles: per (stream, thread), `Send + !Sync`, fixed workspace (A2-7).
pub trait LibraryExt: AgDevice {
    type Blas: Send;
    type Solver: Send;
    fn blas(&self, s: &Self::Stream) -> DriverResult<Self::Blas>;
    fn solver(&self, s: &Self::Stream) -> DriverResult<Self::Solver>;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    ThreadLocal,
    Global,
    Relaxed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UpdateResult {
    Updated,
    TopologyChanged,
}

pub trait GraphExt: AgDevice {
    type Graph: Send;
    type GraphExec: Send;
    fn begin_capture(&self, s: &Self::Stream, mode: CaptureMode) -> DriverResult<()>;
    fn end_capture(&self, s: &Self::Stream) -> DriverResult<Self::Graph>;
    fn instantiate(&self, g: &Self::Graph) -> DriverResult<Self::GraphExec>;
    fn update(&self, exec: &mut Self::GraphExec, g: &Self::Graph) -> DriverResult<UpdateResult>;
    fn launch_graph(&self, s: &Self::Stream, exec: &Self::GraphExec) -> Result<(), EnqueueError>;
    /// An empty graph (top level for explicitly built graphs with conditional nodes).
    fn new_graph(&self) -> DriverResult<Self::Graph>;
    /// Record the work `f` enqueues on `s` into the existing graph `g` (e.g. a conditional
    /// node body). `g` stays owned by its parent.
    fn capture_into(&self, s: &Self::Stream, g: &Self::Graph, f: &mut dyn FnMut() -> DriverResult<()>) -> DriverResult<()>;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CondKind {
    If,
    IfElse,
    While,
    Switch(u32),
}

/// Handle to a device-settable graph condition (`cudaGraphSetConditional`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CondHandle(pub u64);

pub trait ConditionalExt: GraphExt {
    /// Adds a conditional node; returns the handle and the body graph(s) to populate.
    fn conditional(&self, g: &mut Self::Graph, kind: CondKind, default: u32) -> DriverResult<(CondHandle, Vec<Self::Graph>)>;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    ReadWrite,
}

pub trait VmmExt: AgDevice {
    type VaRange: Send;
    type PhysPage: Send;
    fn granularity(&self) -> usize;
    fn reserve(&self, va_bytes: usize) -> DriverResult<Self::VaRange>;
    fn create_page(&self, bytes: usize) -> DriverResult<Self::PhysPage>;
    fn map(&self, va: &Self::VaRange, off: usize, page: &Self::PhysPage, access: Access) -> DriverResult<()>;
    #[cfg(unix)]
    fn export_fd(&self, page: &Self::PhysPage) -> DriverResult<std::os::fd::OwnedFd>;
    #[cfg(unix)]
    fn import_fd(&self, fd: std::os::fd::OwnedFd, bytes: usize) -> DriverResult<Self::PhysPage>;
}

pub trait PartitionExt: AgDevice {
    type Partition: Send + Sync;
    fn green_context(&self, sms: u32) -> DriverResult<Self::Partition>;
}

pub trait RayTracingExt: AgDevice {
    type Optix: Send + Sync;
    fn optix(&self) -> DriverResult<&Self::Optix>;
}
