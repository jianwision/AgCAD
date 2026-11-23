//! # agcad-driver-cuda — CUDA driver (Tier N)
//!
//! Built on cudarc 0.19.10's raw `result`/`sys` layer (the Demo-proven crate) so that AgCAD
//! controls stream priorities, stream-ordered frees, graphs, conditional nodes and VMM.
//! Graphs, conditional nodes (IF/SWITCH/device-driven WHILE) and VMM are exercised on
//! hardware by `tests/graphs_vmm_cuda.rs`.

mod fault;
pub mod graph;
pub mod nvrtc;
pub mod vmm;

use agcad_driver::*;
use cudarc::driver::{result, sys, CudaContext};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub use fault::is_sticky;

/// Shared device state.
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    caps: Caps,
    arch: String,
    poisoned: AtomicBool,
    libraries: Mutex<Vec<LoadedLibrary>>,
    transfers: TransferCounters,
}

struct LoadedLibrary {
    name: &'static str,
    module: sys::CUmodule,
    functions: Vec<(String, sys::CUfunction)>,
}

unsafe impl Send for LoadedLibrary {}

/// Raw stream with priority. Destroyed when the last clone drops.
#[derive(Clone)]
pub struct CuStream {
    inner: Arc<StreamInner>,
}

struct StreamInner {
    raw: sys::CUstream,
    ctx: Arc<CudaContext>,
    pub priority: Priority,
}

unsafe impl Send for StreamInner {}
unsafe impl Sync for StreamInner {}

impl Drop for StreamInner {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuStreamDestroy_v2(self.raw);
        }
    }
}

impl CuStream {
    pub fn raw(&self) -> sys::CUstream { self.inner.raw }
    pub fn priority(&self) -> Priority { self.inner.priority }
}

/// Device allocation freed stream-ordered on its owning stream (AG-08-004).
pub struct CuBuffer {
    ptr: sys::CUdeviceptr,
    bytes: usize,
    stream: CuStream,
}

unsafe impl Send for CuBuffer {}
unsafe impl Sync for CuBuffer {}

impl RawBuffer for CuBuffer {
    fn bytes(&self) -> usize { self.bytes }
    fn device_ptr(&self) -> u64 { self.ptr as u64 }
}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        let _ = self.stream.inner.ctx.bind_to_thread();
        unsafe {
            let _ = result::free_async(self.ptr, self.stream.raw());
        }
    }
}

/// Page-locked host staging.
pub struct CuPinned {
    ptr: *mut u8,
    bytes: usize,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CuPinned {}

impl PinnedHost for CuPinned {
    fn as_bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) } }
    fn as_bytes_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) } }
}

impl Drop for CuPinned {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = result::free_host(self.ptr as *mut c_void);
        }
    }
}

pub struct CuEvent {
    raw: sys::CUevent,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl Drop for CuEvent {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuEventDestroy_v2(self.raw);
        }
    }
}

fn attr(dev: sys::CUdevice, a: sys::CUdevice_attribute) -> i32 {
    unsafe { result::device::get_attribute(dev, a).unwrap_or(0) }
}

impl CudaDevice {
    /// Open device `ordinal`; fill [`Caps`] from device attributes (A3-2).
    /// Returns `Err` cleanly (never panics) if `libcuda` is absent on this machine.
    pub fn open(ordinal: usize) -> DriverResult<Arc<Self>> {
        let ctx = match std::panic::catch_unwind(|| CudaContext::new(ordinal)) {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(fault::driver(e)),
            Err(_) => return Err(DriverError::Enqueue(EnqueueError::Unsupported("CUDA driver library not found"))),
        };
        ctx.bind_to_thread().map_err(fault::driver)?;
        let name = ctx.name().unwrap_or_else(|_| "CUDA device".into());
        let dev = result::device::get(ordinal as i32).map_err(fault::driver)?;
        let major = attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) as u8;
        let minor = attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) as u8;
        let mut drv: i32 = 0;
        unsafe {
            let _ = sys::cuDriverGetVersion(&mut drv);
        }
        let total = unsafe { result::device::total_mem(dev).unwrap_or(0) } as u64;
        let cond = if drv >= 12080 { CondSupport::Full } else if drv >= 12030 { CondSupport::IfWhile } else { CondSupport::HostEmulated };
        let caps = Caps {
            name,
            backend: BackendKind::Cuda,
            compute_capability: Some((major, minor)),
            driver_version: drv as u32,
            fp64_rate: Caps::fp64_rate_for_cc((major, minor)),
            fp64_gemm_emulation: drv >= 13002,
            tensor_types: TensorTypes { fp16: major >= 7, bf16: major >= 8, tf32: major >= 8, fp8: major >= 9 || (major == 8 && minor >= 9), int8: major >= 7, fp64_mma: major >= 8 && minor == 0 || major >= 9 },
            rt_cores: major >= 7 && minor >= 5 || major >= 8,
            conditional_nodes: cond,
            green_contexts: drv >= 12040,
            subgroup_size: 32,
            atomics64: true,
            max_alloc: total,
            total_memory: total,
        };
        let arch = format!("sm_{major}{minor}");
        Ok(Arc::new(CudaDevice { ctx, caps, arch, poisoned: AtomicBool::new(false), libraries: Mutex::new(Vec::new()), transfers: TransferCounters::default() }))
    }

    pub fn arch(&self) -> &str { &self.arch }
    pub fn context(&self) -> &Arc<CudaContext> { &self.ctx }

    fn bind(&self) -> Result<(), EnqueueError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(EnqueueError::Poisoned);
        }
        self.ctx.bind_to_thread().map_err(fault::enqueue)
    }

    fn note_fault(&self, f: &DeviceFault) {
        if f.sticky {
            self.poisoned.store(true, Ordering::Release);
        }
    }

    fn enq<T>(&self, r: Result<T, cudarc::driver::DriverError>) -> Result<T, EnqueueError> {
        r.map_err(|e| {
            let c = fault::code(&e);
            if fault::is_sticky(c) {
                self.poisoned.store(true, Ordering::Release);
            }
            fault::enqueue(e)
        })
    }
}

impl AgDevice for CudaDevice {
    type Stream = CuStream;
    type Event = CuEvent;
    type Buffer = CuBuffer;
    type Pinned = CuPinned;

    fn caps(&self) -> &Caps { &self.caps }

    fn state(&self) -> WorkerState {
        if self.poisoned.load(Ordering::Acquire) { WorkerState::Poisoned } else { WorkerState::Healthy }
    }

    fn stream(&self, priority: Priority) -> DriverResult<CuStream> {
        self.bind()?;
        let (mut least, mut greatest) = (0i32, 0i32);
        unsafe {
            fault::check(sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest)).map_err(fault::driver)?;
        }
        // CUDA: lower number = higher priority.
        let prio = match priority { Priority::High => greatest, Priority::Normal => (least + greatest) / 2, Priority::Low => least };
        let mut raw: sys::CUstream = std::ptr::null_mut();
        unsafe {
            fault::check(sys::cuStreamCreateWithPriority(&mut raw, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32, prio)).map_err(fault::driver)?;
        }
        Ok(CuStream { inner: Arc::new(StreamInner { raw, ctx: self.ctx.clone(), priority }) })
    }

    fn alloc_bytes(&self, s: &CuStream, bytes: usize) -> Result<CuBuffer, EnqueueError> {
        self.bind()?;
        let n = align_up(bytes.max(1), DEVICE_ALIGN);
        let ptr = self.enq(unsafe { result::malloc_async(s.raw(), n) }).map_err(|e| match e {
            EnqueueError::OutOfMemory { .. } => EnqueueError::OutOfMemory { requested: n as u64 },
            o => o,
        })?;
        Ok(CuBuffer { ptr, bytes, stream: s.clone() })
    }

    fn memset_zero_async(&self, s: &CuStream, buf: &CuBuffer, offset: usize, bytes: usize) -> Result<(), EnqueueError> {
        self.bind()?;
        if offset + bytes > buf.bytes {
            return Err(EnqueueError::InvalidConfig("memset out of range".into()));
        }
        self.enq(unsafe { result::memset_d8_async(buf.ptr + offset as u64, 0, bytes, s.raw()) })
    }

    fn pinned(&self, bytes: usize) -> DriverResult<CuPinned> {
        self.bind()?;
        let ptr = unsafe { result::malloc_host(bytes.max(1), 0) }.map_err(fault::driver)? as *mut u8;
        Ok(CuPinned { ptr, bytes, ctx: self.ctx.clone() })
    }

    fn upload_from_host_async(&self, s: &CuStream, src: &[u8], dst: &CuBuffer, dst_off: usize) -> Result<(), EnqueueError> {
        self.bind()?;
        if dst_off + src.len() > dst.bytes {
            return Err(EnqueueError::InvalidConfig("upload out of range".into()));
        }
        // Pageable source: the driver stages it before returning, so `src` may be freed after.
        self.enq(unsafe { result::memcpy_htod_async(dst.ptr + dst_off as u64, src, s.raw()) })
    }

    fn upload_async(&self, s: &CuStream, src: &CuPinned, src_off: usize, dst: &CuBuffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        self.bind()?;
        if src_off + bytes > src.bytes || dst_off + bytes > dst.bytes {
            return Err(EnqueueError::InvalidConfig("upload out of range".into()));
        }
        let host = &src.as_bytes()[src_off..src_off + bytes];
        self.enq(unsafe { result::memcpy_htod_async(dst.ptr + dst_off as u64, host, s.raw()) })
    }

    fn download_async(&self, s: &CuStream, src: &CuBuffer, src_off: usize, dst: &mut CuPinned, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        self.bind()?;
        if src_off + bytes > src.bytes || dst_off + bytes > dst.bytes {
            return Err(EnqueueError::InvalidConfig("download out of range".into()));
        }
        let host = &mut dst.as_bytes_mut()[dst_off..dst_off + bytes];
        self.enq(unsafe { result::memcpy_dtoh_async(host, src.ptr + src_off as u64, s.raw()) })
    }

    fn copy_device_async(&self, s: &CuStream, src: &CuBuffer, src_off: usize, dst: &CuBuffer, dst_off: usize, bytes: usize) -> Result<(), EnqueueError> {
        self.bind()?;
        if src_off + bytes > src.bytes || dst_off + bytes > dst.bytes {
            return Err(EnqueueError::InvalidConfig("copy out of range".into()));
        }
        self.enq(unsafe { result::memcpy_dtod_async(dst.ptr + dst_off as u64, src.ptr + src_off as u64, bytes, s.raw()) })
    }

    fn memset_async(&self, s: &CuStream, dst: &CuBuffer, off: usize, bytes: usize, value: u8) -> Result<(), EnqueueError> {
        self.bind()?;
        if off + bytes > dst.bytes {
            return Err(EnqueueError::InvalidConfig("memset out of range".into()));
        }
        if bytes == 0 { return Ok(()); }
        unsafe { fault::check(sys::cuMemsetD8Async(dst.ptr + off as u64, value, bytes, s.raw())).map_err(fault::enqueue) }
    }

    fn record(&self, s: &CuStream) -> Result<CuEvent, EnqueueError> {
        self.bind()?;
        let raw = self.enq(result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        self.enq(unsafe { result::event::record(raw, s.raw()) })?;
        Ok(CuEvent { raw, ctx: self.ctx.clone() })
    }

    fn wait(&self, s: &CuStream, e: &CuEvent) -> Result<(), EnqueueError> {
        self.bind()?;
        self.enq(unsafe { result::stream::wait_event(s.raw(), e.raw, sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT) })
    }

    fn query(&self, e: &CuEvent) -> Result<bool, DeviceFault> {
        self.ctx.bind_to_thread().map_err(|x| fault::fault(x, None))?;
        let r = unsafe { sys::cuEventQuery(e.raw) };
        match r {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            other => {
                let f = fault::fault(cudarc::driver::DriverError(other), None);
                self.note_fault(&f);
                Err(f)
            }
        }
    }

    fn sync(&self, s: &CuStream) -> Result<(), DeviceFault> {
        self.ctx.bind_to_thread().map_err(|x| fault::fault(x, None))?;
        let r = unsafe { sys::cuStreamSynchronize(s.raw()) };
        fault::check(r).map_err(|e| {
            let f = fault::fault(e, None);
            self.note_fault(&f);
            f
        })
    }

    fn elapsed_ms(&self, a: &CuEvent, b: &CuEvent) -> Result<f32, DeviceFault> {
        self.ctx.bind_to_thread().map_err(|x| fault::fault(x, None))?;
        let mut ms = 0f32;
        fault::check(unsafe { sys::cuEventElapsedTime(&mut ms, a.raw, b.raw) }).map_err(|e| fault::fault(e, None))?;
        Ok(ms)
    }

    fn transfer_stats(&self) -> TransferStats { self.transfers.snapshot() }
    fn reset_transfer_stats(&self) { self.transfers.reset(); }
    fn record_transfer(&self, kind: TransferKind, bytes: usize) { self.transfers.record(kind, bytes); }
}

impl KernelExt for CudaDevice {
    fn load_library(&self, src: &LibrarySource) -> DriverResult<LibraryId> {
        let mut libs = self.libraries.lock().expect("library registry poisoned");
        if let Some(i) = libs.iter().position(|l| l.name == src.name) {
            return Ok(LibraryId(i as u32));
        }
        let source = src.cuda_source.as_ref().ok_or_else(|| DriverError::NotFound(format!("library {} has no CUDA source", src.name)))?;
        let ptx = nvrtc::compile(src.name, source, &self.arch, &src.options)?;
        self.bind()?;
        let image = CString::new(ptx).map_err(|_| DriverError::Compile("PTX contains NUL".into()))?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        unsafe {
            fault::check(sys::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void)).map_err(fault::driver)?;
        }
        let mut functions = Vec::new();
        for (name, _) in src.cpu_kernels {
            let cname = CString::new(*name).expect("kernel names are ASCII");
            let mut f: sys::CUfunction = std::ptr::null_mut();
            let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
            if r == sys::CUresult::CUDA_SUCCESS {
                functions.push((name.to_string(), f));
            } else {
                return Err(DriverError::NotFound(format!("kernel {name} missing in CUDA library {}", src.name)));
            }
        }
        libs.push(LoadedLibrary { name: src.name, module, functions });
        Ok(LibraryId((libs.len() - 1) as u32))
    }

    fn launch(&self, s: &CuStream, lib: LibraryId, kernel: &str, cfg: &LaunchConfig, params: &CuBuffer) -> Result<(), EnqueueError> {
        self.bind()?;
        let f = {
            let libs = self.libraries.lock().expect("library registry poisoned");
            let l = libs.get(lib.0 as usize).ok_or_else(|| EnqueueError::InvalidConfig("unknown library".into()))?;
            l.functions.iter().find(|(n, _)| n == kernel).map(|(_, f)| *f).ok_or_else(|| EnqueueError::InvalidConfig(format!("unknown kernel {kernel}")))?
        };
        if cfg.shared_mem_bytes > 48 * 1024 {
            unsafe {
                let _ = sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, cfg.shared_mem_bytes as i32);
            }
        }
        // Single kernel argument: device address of the parameter block (A2-1).
        let mut arg: u64 = params.device_ptr();
        let mut args: [*mut c_void; 1] = [&mut arg as *mut u64 as *mut c_void];
        let r = unsafe {
            sys::cuLaunchKernel(
                f,
                cfg.grid.x, cfg.grid.y, cfg.grid.z,
                cfg.block.x, cfg.block.y, cfg.block.z,
                cfg.shared_mem_bytes,
                s.raw(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.enq(fault::check(r))
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if let Ok(libs) = self.libraries.lock() {
            for l in libs.iter() {
                unsafe {
                    let _ = sys::cuModuleUnload(l.module);
                }
            }
        }
    }
}
