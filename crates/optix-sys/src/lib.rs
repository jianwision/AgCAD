//! # agcad-optix-sys — OptiX shim bindings (review F4)
//!
//! A thin C++ shim (`shim/agcad_optix.cpp`, built only with feature `optix` against an OptiX
//! SDK from `OPTIX_ROOT` or its default install path) exposes a C API for: building a triangle
//! GAS from device buffers, and launching batched ray queries (closest hit: t, primitive id,
//! barycentrics, front face). OptiX itself is loaded from the display driver. Without the
//! feature every entry point reports `Unsupported`, and callers use the exact CPU reference
//! tracer explicitly (never silently).

/// Device programs (raygen / miss / closest hit) for the shim's pipeline; compile with NVRTC
/// using [`include_paths`].
pub const DEVICE_PROGRAMS: &str = r#"
#include <optix.h>
struct Ray { float3 o; float tmin; float3 d; float tmax; };
struct Hit { float t; unsigned prim; float u; float v; unsigned front; };
struct Params { OptixTraversableHandle gas; Ray* rays; Hit* hits; unsigned n; unsigned pad; };
extern "C" { __constant__ Params params; }
extern "C" __global__ void __raygen__rg() {
    unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    Ray r = params.rays[i];
    unsigned p0 = __float_as_uint(-1.0f), p1 = 0xFFFFFFFFu, p2 = 0u, p3 = 0u, p4 = 0u;
    optixTrace(params.gas, r.o, r.d, r.tmin, r.tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2, p3, p4);
    Hit h; h.t = __uint_as_float(p0); h.prim = p1; h.u = __uint_as_float(p2); h.v = __uint_as_float(p3); h.front = p4;
    params.hits[i] = h;
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(__float_as_uint(-1.0f)); }
extern "C" __global__ void __closesthit__ch() {
    float2 b = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(b.x));
    optixSetPayload_3(__float_as_uint(b.y));
    optixSetPayload_4(optixIsFrontFaceHit() ? 1u : 0u);
}
"#;

/// Include directories for compiling [`DEVICE_PROGRAMS`] (OptiX, CUDA); empty without the
/// feature.
pub fn include_paths() -> Vec<&'static str> {
    #[cfg(feature = "optix")]
    { vec![env!("AGCAD_OPTIX_INCLUDE"), env!("AGCAD_CUDA_INCLUDE")] }
    #[cfg(not(feature = "optix"))]
    { vec![] }
}

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
pub struct Hit { pub t: f32, pub prim: u32, pub u: f32, pub v: f32, pub front: u32 }

#[cfg(feature = "optix")]
unsafe extern "C" {
    fn agcad_optix_create(cu_ctx: *mut core::ffi::c_void, ptx: *const u8, ptx_len: usize, out: *mut *mut core::ffi::c_void) -> i32;
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
    /// # Safety: `cu_ctx` must be a live CUcontext; `ptx` the compiled [`DEVICE_PROGRAMS`].
    pub unsafe fn new(cu_ctx: *mut core::ffi::c_void, ptx: &str) -> Result<Context, OptixError> {
        #[cfg(feature = "optix")]
        {
            let mut raw = core::ptr::null_mut();
            let r = unsafe { agcad_optix_create(cu_ctx, ptx.as_ptr(), ptx.len(), &mut raw) };
            return if r == 0 { Ok(Context { raw }) } else { Err(OptixError::Failed(r)) };
        }
        #[cfg(not(feature = "optix"))]
        { let _ = (cu_ctx, ptx); Err(OptixError::Unsupported) }
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
