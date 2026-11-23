//! Operation-scoped CUDA batches for geometry. The host algorithms also serve as
//! the reference path for small batches and machines without a CUDA driver.
use agcad_driver::*;
use agcad_driver_cuda::{CuBuffer, CuStream, CudaDevice};
use bytemuck::Pod;
use std::{cell::RefCell, sync::Arc};

pub struct Dispatch {
    pub operation: &'static str,
    pub backend: &'static str,
    pub work: usize,
    pub error: Option<String>,
}
thread_local! { static EVENTS: RefCell<Option<Vec<Dispatch>>> = const { RefCell::new(None) }; }

pub fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<Dispatch>) {
    struct Restore(Option<Vec<Dispatch>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            EVENTS.with(|e| e.replace(self.0.take()));
        }
    }
    let _restore = Restore(EVENTS.with(|e| e.replace(Some(Vec::new()))));
    let result = f();
    (result, EVENTS.with(|e| e.take().unwrap_or_default()))
}

fn record(operation: &'static str, backend: &'static str, work: usize, error: Option<String>) {
    EVENTS.with(|e| {
        if let Some(events) = e.borrow_mut().as_mut() {
            events.push(Dispatch {
                operation,
                backend,
                work,
                error,
            });
        }
    });
}

fn selected(work: usize, crossover: Option<usize>) -> bool {
    work > 0
        && match std::env::var("AGCAD_GEOMETRY_BACKEND").as_deref() {
            Ok("host") => false,
            Ok("cuda") => true,
            _ => crossover.is_some_and(|limit| work >= limit),
        }
}

pub(crate) struct Gpu {
    pub dev: Arc<CudaDevice>,
    pub stream: CuStream,
}

impl Gpu {
    pub fn open() -> DriverResult<Self> {
        let (dev, stream) = CudaDevice::geometry_stream()?;
        Ok(Self { dev, stream })
    }
    pub fn upload<T: Pod>(&self, v: &[T]) -> DriverResult<CuBuffer> {
        let bytes = bytemuck::cast_slice(v);
        let buf = self.dev.alloc_bytes(&self.stream, bytes.len().max(1))?;
        if !bytes.is_empty() {
            self.dev
                .upload_from_host_async(&self.stream, bytes, &buf, 0)?;
        }
        Ok(buf)
    }
    pub fn output<T>(&self, n: usize) -> DriverResult<CuBuffer> {
        Ok(self
            .dev
            .alloc_bytes(&self.stream, bytes_for::<T>(n)?.max(1))?)
    }
    pub fn read<T: Pod>(&self, b: &CuBuffer, n: usize) -> DriverResult<Vec<T>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = bytes_for::<T>(n)?;
        // Bound page-locked staging even when a fine tessellation has millions of vertices.
        let chunk = n.min((4 * 1024 * 1024 / size_of::<T>()).max(1));
        let mut host = self.dev.pinned(bytes_for::<T>(chunk)?)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(n)
            .map_err(|_| EnqueueError::OutOfMemory {
                requested: bytes as u64,
            })?;
        for start in (0..n).step_by(chunk) {
            let size = bytes_for::<T>((n - start).min(chunk))?;
            self.dev
                .download_async(&self.stream, b, start * size_of::<T>(), &mut host, 0, size)?;
            self.dev.sync(&self.stream)?;
            values.extend_from_slice(bytemuck::cast_slice(&host.as_bytes()[..size]));
        }
        Ok(values)
    }
    pub fn launch(&self, name: &str, cfg: LaunchConfig, args: &[u64]) -> DriverResult<()> {
        const SRC: &str = concat!(
            include_str!("../../../docs/Demo/cuda/surface_primitives.cuh"),
            "\n",
            include_str!("accelerator.cu")
        );
        let lib = self.dev.load_cuda_source(
            "agcad_geometry_v1",
            SRC,
            [
                "surface_points",
                "surface_curvature",
                "wall_hits",
                "wall_min",
                "rail_overlap",
                "rebase_triangles",
                "local_positions",
                "block_edge_corners",
            ]
            .into_iter(),
            &[],
        )?;
        let params = self.upload(args)?;
        Ok(self.dev.launch(&self.stream, lib, name, &cfg, &params)?)
    }
}
fn bytes_for<T>(n: usize) -> DriverResult<usize> {
    Ok(n.checked_mul(size_of::<T>())
        .ok_or(EnqueueError::OutOfMemory {
            requested: u64::MAX,
        })?)
}

pub fn ensure_worker_healthy() -> DriverResult<()> {
    if CudaDevice::geometry_worker_poisoned() {
        return Err(EnqueueError::Poisoned.into());
    }
    Ok(())
}

pub(crate) fn attempt<T>(
    operation: &'static str,
    work: usize,
    crossover: Option<usize>,
    f: impl FnOnce() -> DriverResult<T>,
) -> DriverResult<Option<T>> {
    ensure_worker_healthy()?;
    if selected(work, crossover) {
        match f() {
            Ok(value) => {
                record(operation, "cuda", work, None);
                return Ok(Some(value));
            }
            Err(e) if e.requires_worker_restart() => {
                CudaDevice::note_geometry_error(&e);
                record(operation, "cuda", work, Some(e.to_string()));
                return Err(e);
            }
            Err(e) => {
                record(operation, "host", work, Some(e.to_string()));
                return Ok(None);
            }
        }
    }
    if work != 0 {
        record(operation, "host", work, None);
    }
    Ok(None)
}

use crate::math::{V3, v3};
use crate::tessellate::{Mesh, PreparedMesh};

pub fn wall_minimum(mesh: &Mesh, stride: usize) -> DriverResult<f64> {
    if !mesh.indices_valid() {
        return Err(EnqueueError::InvalidConfig("invalid mesh triangle index".into()).into());
    }
    let stride = stride.max(1);
    Ok(wall_cuda(mesh, stride)?.unwrap_or_else(|| {
        let p = |i: u32| mesh.positions[i as usize];
        let mut tmin = f64::INFINITY;
        for t in mesh.triangles.iter().step_by(stride) {
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let nrm = (b - a).cross(c - a).normalized();
            let o = (a + b + c) * (1.0 / 3.0) - nrm * 1e-7;
            for u in &mesh.triangles {
                if let Some(d) = ray_tri(o, -nrm, p(u[0]), p(u[1]), p(u[2])) {
                    tmin = tmin.min(d);
                }
            }
        }
        tmin
    }))
}

/// Ray/triangle intersection (Möller–Trumbore); returns t > eps.
fn ray_tri(o: V3, d: V3, a: V3, b: V3, c: V3) -> Option<f64> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 1e-9 { Some(t) } else { None }
}

fn wall_cuda(mesh: &Mesh, stride: usize) -> DriverResult<Option<f64>> {
    let triangles = mesh.triangles.len();
    let rays = triangles.div_ceil(stride);
    attempt(
        "wall_thickness",
        rays.saturating_mul(triangles),
        Some(4_000_000),
        || {
            let gpu = Gpu::open()?;
            if rays > i32::MAX as usize {
                return Err(EnqueueError::InvalidConfig("invalid ray batch".into()).into());
            }
            let pos = gpu.upload(&mesh.positions)?;
            let tri = gpu.upload(&mesh.triangles)?;
            let hits = gpu.output::<f64>(rays)?;
            let minimum = gpu.output::<f64>(1)?;
            let cfg = LaunchConfig {
                grid: Dim3::x(rays as u32),
                block: Dim3::x(128),
                shared_mem_bytes: 0,
                cluster: None,
            };
            gpu.launch(
                "wall_hits",
                cfg,
                &[
                    pos.device_ptr(),
                    tri.device_ptr(),
                    hits.device_ptr(),
                    triangles as u64,
                    stride as u64,
                ],
            )?;
            gpu.launch(
                "wall_min",
                LaunchConfig::linear(1, 128),
                &[hits.device_ptr(), minimum.device_ptr(), rays as u64],
            )?;
            Ok(gpu.read::<f64>(&minimum, 1)?[0])
        },
    )
}

/// One conservative bit per (i,j,span), ordered exactly as the host contact check.
pub fn rail_candidates(
    rails: &[crate::nurbs::NurbsCurve],
    tol: f64,
) -> DriverResult<Option<Vec<u8>>> {
    let Some(first) = rails.first() else {
        return Ok(None);
    };
    let d = first.degree;
    if d == 0
        || first.poles.len() <= d
        || (first.poles.len() - 1) % d != 0
        || rails.iter().any(|r| {
            r.degree != d
                || r.poles.len() != first.poles.len()
                || r.weights.len() != r.poles.len()
                || r.weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        })
    {
        return Ok(None);
    }
    let spans = (first.poles.len() - 1) / d;
    let Some(count) = rails
        .len()
        .checked_mul(rails.len() - 1)
        .and_then(|n| (n / 2).checked_mul(spans))
    else {
        return Ok(None);
    };
    attempt("loft_rail_pairs", count, Some(256), || {
        let mut bounds = Vec::with_capacity(rails.len() * spans);
        for rail in rails {
            for i in 0..spans {
                let mut bb = crate::math::Aabb::EMPTY;
                for &point in &rail.poles[i * d..=i * d + d] {
                    bb.add(point);
                }
                // Outward padding also covers floating-point knot insertion error.
                let pad = 32. * f64::EPSILON * (1. + bb.min.max_abs().max(bb.max.max_abs()));
                bounds.push([
                    bb.min.x - pad,
                    bb.min.y - pad,
                    bb.min.z - pad,
                    bb.max.x + pad,
                    bb.max.y + pad,
                    bb.max.z + pad,
                ]);
            }
        }
        let gpu = Gpu::open()?;
        let input = gpu.upload(&bounds)?;
        let keep = gpu.output::<u8>(count)?;
        gpu.launch(
            "rail_overlap",
            LaunchConfig::linear(count as u64, 128),
            &[
                input.device_ptr(),
                keep.device_ptr(),
                rails.len() as u64,
                spans as u64,
                count as u64,
                tol.to_bits(),
            ],
        )?;
        gpu.read(&keep, count)
    })
}

/// Automatic dispatch currently measures mass on the host (see the timing example).
pub fn cuda_mass_enabled() -> bool {
    selected(1, None)
}

pub type MassProperties = (f64, f64, V3);
/// Concatenate meshes in configuration order. The reduction returns five FP64
/// sums per segment; empty configurations keep their segment and output slot.
pub fn mass_properties(meshes: &[PreparedMesh]) -> DriverResult<Vec<MassProperties>> {
    if meshes.iter().any(|m| !m.indices_valid) {
        return Err(EnqueueError::InvalidConfig("invalid mesh triangle index".into()).into());
    }
    let work: usize = meshes.iter().map(|m| m.mesh.triangles.len()).sum();
    // A 16-configuration study through 1.96 million triangles showed no
    // repeatable transfer-inclusive CUDA win on the RTX 2000 Ada. Keep the
    // segmented CUDA implementation selectable for device-specific testing.
    Ok(attempt("mesh_mass_properties", work, None, || {
        use agcad_kernels::primitives::{mass_properties_from_sums, mesh_mass_sums};
        let gpu = Gpu::open()?;
        let overflow = || {
            DriverError::Enqueue(EnqueueError::InvalidConfig(
                "mesh batch exceeds 32-bit indexing".into(),
            ))
        };
        let n = u32::try_from(work).map_err(|_| overflow())?;
        let nm = u32::try_from(meshes.len()).map_err(|_| overflow())?;
        let mut offsets = vec![0u32];
        let mut pos_count = 0usize;
        let mut bases = Vec::with_capacity(meshes.len());
        let tris = gpu.output::<[u32; 3]>(work)?;
        for prepared in meshes {
            let mesh = &prepared.mesh;
            bases.push(u32::try_from(pos_count).map_err(|_| overflow())?);
            pos_count = pos_count
                .checked_add(mesh.positions.len())
                .ok_or_else(overflow)?;
            u32::try_from(pos_count).map_err(|_| overflow())?;
            let start = *offsets.last().unwrap();
            if !mesh.triangles.is_empty() {
                gpu.dev.upload_from_host_async(
                    &gpu.stream,
                    bytemuck::cast_slice(&mesh.triangles),
                    &tris,
                    start as usize * size_of::<[u32; 3]>(),
                )?;
            }
            offsets.push(start + mesh.triangles.len() as u32);
        }
        let segments = gpu.upload(&offsets)?;
        let origins: Vec<_> = meshes.iter().map(|m| m.mesh.mass_origin()).collect();
        if meshes.len() > 1 {
            let bases = gpu.upload(&bases)?;
            gpu.launch(
                "rebase_triangles",
                LaunchConfig::linear(work as u64, 128),
                &[
                    tris.device_ptr(),
                    segments.device_ptr(),
                    bases.device_ptr(),
                    nm as u64,
                    n as u64,
                ],
            )?;
        }
        let positions = gpu.output::<V3>(pos_count)?;
        let mut offset = 0;
        for prepared in meshes {
            let points = &prepared.mesh.positions;
            let stride = size_of::<V3>();
            let mut start = 0;
            if let Some(points_on_device) = &prepared.resident {
                for range in &points_on_device.ranges {
                    let dst = range.mesh_start;
                    if dst > start {
                        gpu.dev.upload_from_host_async(
                            &gpu.stream,
                            bytemuck::cast_slice(&points[start..dst]),
                            &positions,
                            offset + start * stride,
                        )?;
                    }
                    gpu.dev.copy_device_async(
                        &gpu.stream,
                        &points_on_device.buffer,
                        range.device_start * stride,
                        &positions,
                        offset + dst * stride,
                        range.len * stride,
                    )?;
                    start = dst + range.len;
                }
            }
            if start < points.len() {
                gpu.dev.upload_from_host_async(
                    &gpu.stream,
                    bytemuck::cast_slice(&points[start..]),
                    &positions,
                    offset + start * stride,
                )?;
            }
            offset += points.len() * stride;
        }
        bases.push(pos_count as u32);
        let vertex_offsets = gpu.upload(&bases)?;
        let origins_device = gpu.upload(&origins)?;
        gpu.launch(
            "local_positions",
            LaunchConfig::linear(pos_count as u64, 128),
            &[
                positions.device_ptr(),
                vertex_offsets.device_ptr(),
                origins_device.device_ptr(),
                nm as u64,
                pos_count as u64,
            ],
        )?;
        static SRC: std::sync::OnceLock<LibrarySource> = std::sync::OnceLock::new();
        let lib = gpu
            .dev
            .load_library(SRC.get_or_init(agcad_kernels_cuda::primitives_library))?;
        let result = mesh_mass_sums(
            &*gpu.dev,
            &gpu.stream,
            lib,
            &positions,
            &tris,
            n,
            &segments,
            nm,
        )?;
        let sums = gpu.read::<f64>(&result, 5 * meshes.len())?;
        Ok((0..meshes.len())
            .map(|i| {
                let (v, a, [x, y, z]) = mass_properties_from_sums(&sums, meshes.len(), i);
                let center = if v == 0. {
                    V3::ZERO
                } else {
                    origins[i] + v3(x, y, z)
                };
                (v, a, center)
            })
            .collect())
    })?
    .unwrap_or_else(|| meshes.iter().map(|m| m.mesh.mass_properties()).collect()))
}
