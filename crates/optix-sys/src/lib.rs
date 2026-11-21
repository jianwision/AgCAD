//! # agcad-optix-sys — OptiX shim bindings (review F4)
//!
//! A thin C++ shim (`shim/agcad_optix.cpp`, built only with feature `optix` and an OptiX SDK
//! via `OPTIX_ROOT`) exposes a C API for: building a triangle GAS from device buffers, and
//! launching batched ray queries (closest hit: t, primitive id, barycentrics). Without the
//! feature every entry point reports `Unsupported`, and callers fall back to the CubeCL/CUDA
//! BVH path explicitly (never silently).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptixError {
    Unsupported,
    Failed(i32),
}

/// Ray record shared with the shim (`origin`, `tmin`, `dir`, `tmax`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Ray { pub origin: [f32; 3], pub tmin: f32, pub dir: [f32; 3], pub tmax: f32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Hit { pub t: f32, pub prim: u32, pub u: f32, pub v: f32 }

#[cfg(feature = "optix")]
unsafe extern "C" {
    fn agcad_optix_create(cu_ctx: *mut core::ffi::c_void, out: *mut *mut core::ffi::c_void) -> i32;
    fn agcad_optix_build_gas(h: *mut core::ffi::c_void, verts: u64, n_verts: u32, idx: u64, n_tris: u32, out_gas: *mut u64) -> i32;
    fn agcad_optix_trace(h: *mut core::ffi::c_void, gas: u64, rays: u64, hits: u64, n: u32, stream: *mut core::ffi::c_void) -> i32;
    fn agcad_optix_destroy(h: *mut core::ffi::c_void);
}

pub struct Context {
    #[allow(dead_code)]
    raw: *mut core::ffi::c_void,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// # Safety: `cu_ctx` must be a live CUcontext.
    pub unsafe fn new(cu_ctx: *mut core::ffi::c_void) -> Result<Context, OptixError> {
        #[cfg(feature = "optix")]
        {
            let mut raw = core::ptr::null_mut();
            let r = unsafe { agcad_optix_create(cu_ctx, &mut raw) };
            return if r == 0 { Ok(Context { raw }) } else { Err(OptixError::Failed(r)) };
        }
        #[cfg(not(feature = "optix"))]
        { let _ = cu_ctx; Err(OptixError::Unsupported) }
    }

    /// Build a geometry acceleration structure from device vertex/index buffers.
    pub fn build_gas(&self, verts: u64, n_verts: u32, idx: u64, n_tris: u32) -> Result<u64, OptixError> {
        #[cfg(feature = "optix")]
        {
            let mut gas = 0u64;
            let r = unsafe { agcad_optix_build_gas(self.raw, verts, n_verts, idx, n_tris, &mut gas) };
            return if r == 0 { Ok(gas) } else { Err(OptixError::Failed(r)) };
        }
        #[cfg(not(feature = "optix"))]
        { let _ = (verts, n_verts, idx, n_tris); Err(OptixError::Unsupported) }
    }

    /// Trace `n` rays (device buffers of [`Ray`] / [`Hit`]) on `stream`.
    pub fn trace(&self, gas: u64, rays: u64, hits: u64, n: u32, stream: *mut core::ffi::c_void) -> Result<(), OptixError> {
        #[cfg(feature = "optix")]
        {
            let r = unsafe { agcad_optix_trace(self.raw, gas, rays, hits, n, stream) };
            return if r == 0 { Ok(()) } else { Err(OptixError::Failed(r)) };
        }
        #[cfg(not(feature = "optix"))]
        { let _ = (gas, rays, hits, n, stream); Err(OptixError::Unsupported) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(feature = "optix")]
        unsafe { agcad_optix_destroy(self.raw) }
    }
}
