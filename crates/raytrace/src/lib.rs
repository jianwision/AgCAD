//! # agcad-raytrace — ray queries for CAD analyses
//!
//! One interface ([`Tracer`]: batched closest-hit queries against a triangle mesh) with two
//! implementations:
//!
//! - [`OptixTracer`]: RTX hardware ray tracing through the OptiX shim (feature `optix`);
//! - [`CpuTracer`]: an exact f64 Möller–Trumbore reference over a median-split BVH (parity
//!   tests, machines without OptiX).
//!
//! On top: design-for-manufacture analyses of a body's tessellation ([`analysis`]):
//! - tool access from any set of directions (3-axis, 5-axis proxy);
//! - wall thickness and clearance along surface normals;
//! - a ray-cast shaded render.
//!
//! Evidence: `sampled` (one sample per triangle; mesh-resolution dependent).

pub mod analysis;
mod cpu;

pub use agcad_optix_sys::{Hit, Ray};
pub use cpu::CpuTracer;

use agcad_geo::tessellate::Mesh;

/// Batched closest-hit ray queries (`Hit::t < 0` = miss).
pub trait Tracer {
    fn trace(&self, rays: &[Ray]) -> Result<Vec<Hit>, String>;
    fn name(&self) -> &'static str;
}

/// The best tracer available: OptiX on an RTX GPU when built with feature `optix`, else the
/// CPU reference. Returns the reason when it falls back.
pub fn best(mesh: &Mesh) -> (Box<dyn Tracer>, Option<String>) {
    match OptixTracer::new(mesh) {
        Ok(t) => (Box::new(t), None),
        Err(e) => (Box::new(CpuTracer::new(mesh)), Some(e)),
    }
}

/// OptiX tracer: the mesh's acceleration structure lives on the GPU; rays are traced in
/// batches on a CUDA stream.
pub struct OptixTracer {
    #[allow(dead_code)]
    dev: std::sync::Arc<agcad_driver_cuda::CudaDevice>,
    #[cfg(feature = "optix")]
    inner: optix::Inner,
}

#[cfg(feature = "optix")]
mod optix {
    use agcad_driver::*;
    use agcad_driver_cuda::CudaDevice;
    use std::sync::Arc;

    pub struct Inner {
        pub ctx: agcad_optix_sys::Context,
        pub gas: u64,
        pub s: <CudaDevice as AgDevice>::Stream,
        pub _verts: <CudaDevice as AgDevice>::Buffer,
        pub _idx: <CudaDevice as AgDevice>::Buffer,
    }

    pub fn build(dev: &Arc<CudaDevice>, mesh: &agcad_geo::tessellate::Mesh) -> Result<Inner, String> {
        let opts: Vec<String> = agcad_optix_sys::include_paths().iter().map(|p| format!("-I{p}")).collect();
        // PTX only (virtual arch): OptiX compiles the programs itself; ptxas cannot resolve its intrinsics
        let arch = dev.arch().replace("sm_", "compute_");
        let ptx = agcad_driver_cuda::nvrtc::compile("agcad_optix_programs", agcad_optix_sys::DEVICE_PROGRAMS, &arch, &opts).map_err(|e| e.to_string())?;
        dev.context().bind_to_thread().map_err(|e| format!("{e:?}"))?;
        let ctx = unsafe { agcad_optix_sys::Context::new(dev.context().cu_ctx() as *mut core::ffi::c_void, &ptx) }.map_err(|e| format!("OptiX context: {e:?}"))?;
        let s = dev.stream(Priority::Low).map_err(|e| e.to_string())?;
        let v: Vec<f32> = mesh.positions.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        let ix: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
        let up = |b: &[u8]| -> Result<_, String> { let buf = dev.alloc_bytes(&s, b.len().max(16)).map_err(|e| e.to_string())?; dev.upload_from_host_async(&s, b, &buf, 0).map_err(|e| e.to_string())?; Ok(buf) };
        let (vb, ib) = (up(as_bytes(&v))?, up(as_bytes(&ix))?);
        dev.sync(&s).map_err(|e| e.to_string())?;
        let gas = ctx.build_gas(vb.device_ptr(), mesh.positions.len() as u32, ib.device_ptr(), mesh.triangles.len() as u32).map_err(|e| format!("OptiX GAS: {e:?}"))?;
        Ok(Inner { ctx, gas, s, _verts: vb, _idx: ib })
    }

    pub fn as_bytes<T>(v: &[T]) -> &[u8] { unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) } }
}

impl OptixTracer {
    pub fn new(mesh: &Mesh) -> Result<OptixTracer, String> {
        #[cfg(feature = "optix")]
        {
            if mesh.triangles.is_empty() { return Err("empty mesh".into()); }
            let dev = agcad_driver_cuda::CudaDevice::open(0).map_err(|e| format!("no CUDA device: {e}"))?;
            let inner = optix::build(&dev, mesh)?;
            return Ok(OptixTracer { dev, inner });
        }
        #[cfg(not(feature = "optix"))]
        { let _ = mesh; Err("built without the `optix` feature".into()) }
    }
}

impl Tracer for OptixTracer {
    fn name(&self) -> &'static str { "optix" }
    fn trace(&self, rays: &[Ray]) -> Result<Vec<Hit>, String> {
        #[cfg(feature = "optix")]
        {
            use agcad_driver::*;
            let (d, i) = (&*self.dev, &self.inner);
            let mut out = Vec::with_capacity(rays.len());
            // batches bound device memory for very large queries
            for chunk in rays.chunks(1 << 20) {
                let rb = d.alloc_bytes(&i.s, std::mem::size_of_val(chunk)).map_err(|e| e.to_string())?;
                d.upload_from_host_async(&i.s, optix::as_bytes(chunk), &rb, 0).map_err(|e| e.to_string())?;
                let hb_bytes = chunk.len() * std::mem::size_of::<Hit>();
                let hb = d.alloc_bytes(&i.s, hb_bytes).map_err(|e| e.to_string())?;
                d.sync(&i.s).map_err(|e| e.to_string())?;
                i.ctx.trace(i.gas, rb.device_ptr(), hb.device_ptr(), chunk.len() as u32, i.s.raw() as *mut core::ffi::c_void).map_err(|e| format!("OptiX trace: {e:?}"))?;
                let mut pin = d.pinned(hb_bytes).map_err(|e| e.to_string())?;
                d.download_packed_async(&i.s, &[Region { buf: &hb, offset: 0, bytes: hb_bytes }], &mut pin).map_err(|e| e.to_string())?;
                d.sync(&i.s).map_err(|e| e.to_string())?;
                let bytes = &pin.as_bytes()[..hb_bytes];
                for k in 0..chunk.len() {
                    let f = |j: usize| u32::from_le_bytes(bytes[k * 20 + 4 * j..k * 20 + 4 * j + 4].try_into().unwrap());
                    out.push(Hit { t: f32::from_bits(f(0)), prim: f(1), u: f32::from_bits(f(2)), v: f32::from_bits(f(3)), front: f(4) });
                }
            }
            return Ok(out);
        }
        #[cfg(not(feature = "optix"))]
        { let _ = rays; Err("built without the `optix` feature".into()) }
    }
}
