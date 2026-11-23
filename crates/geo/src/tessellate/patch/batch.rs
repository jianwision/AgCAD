//! Device inputs live for one tessellation. Both passes share the same patch upload.
use super::{NurbsSurface, Sides};
use crate::{
    accelerator::{Gpu, attempt},
    brep::Brep,
    math::V3,
    surface::Surface,
};
use agcad_driver::{DriverResult, LaunchConfig, RawBuffer};
use agcad_driver_cuda::CuBuffer;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Descriptor {
    nu: u32,
    nv: u32,
    pu: u32,
    pv: u32,
    ku: u64,
    kv: u64,
    cp: u64,
    origin: V3,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Sample {
    patch: u64,
    u: f64,
    v: f64,
}

fn supported(s: &NurbsSurface) -> bool {
    let axis = |n: usize, p: usize, k: &[f64]| {
        n <= 32
            && p <= 5
            && n > p
            && k.len() == n + p + 1
            && k[..=p].iter().all(|&v| v == k[p])
            && k[n..].iter().all(|&v| v == k[n])
            && k[p] < k[n]
            && k.windows(2).all(|w| w[0].is_finite() && w[0] <= w[1])
            && k[n].is_finite()
    };
    axis(s.nu, s.du, &s.ku) && axis(s.nv, s.dv, &s.kv) && s.validate().is_ok()
}

pub(super) fn probes(s: &NurbsSurface, mut visit: impl FnMut(f64, f64)) {
    for us in s.ku.windows(2).filter(|w| w[1] > w[0]) {
        for vs in s.kv.windows(2).filter(|w| w[1] > w[0]) {
            for a in [0.01, 0.5, 0.99] {
                for e in [0.01, 0.5, 0.99] {
                    visit(us[0] + (us[1] - us[0]) * a, vs[0] + (vs[1] - vs[0]) * e);
                }
            }
        }
    }
}

struct Upload {
    gpu: Gpu,
    descriptors: CuBuffer,
    data: CuBuffer,
}
pub(super) struct Batch<'a> {
    patches: Vec<(usize, &'a NurbsSurface, Sides)>,
    uploaded: Option<Upload>,
}
pub(in crate::tessellate) struct Points {
    pub offsets: Vec<Option<usize>>,
    pub values: Vec<V3>,
    pub buffer: CuBuffer,
}

impl<'a> Batch<'a> {
    pub fn new(b: &'a Brep, faces: &[Option<Sides>]) -> Self {
        let patches = faces
            .iter()
            .enumerate()
            .filter_map(|(i, sides)| {
                let sides = (*sides)?;
                let Surface::Nurbs(s) = &b.surfaces[b.faces[i].surface as usize] else {
                    return None;
                };
                supported(s).then_some((i, s, sides))
            })
            .collect();
        Self {
            patches,
            uploaded: None,
        }
    }
    fn upload(&mut self) -> DriverResult<&Upload> {
        if self.uploaded.is_none() {
            let gpu = Gpu::open()?;
            let mut data = Vec::new();
            let mut descriptors = Vec::new();
            for &(_, s, _) in &self.patches {
                let ku = data.len() as u64;
                data.extend_from_slice(&s.ku);
                let kv = data.len() as u64;
                data.extend_from_slice(&s.kv);
                let cp = data.len() as u64;
                let origin = s.poles[0];
                for (point, &w) in s.poles.iter().zip(&s.weights) {
                    let point = *point - origin;
                    data.extend([point.x, point.y, point.z, w]);
                }
                descriptors.push(Descriptor {
                    nu: s.nu as u32,
                    nv: s.nv as u32,
                    pu: s.du as u32,
                    pv: s.dv as u32,
                    ku,
                    kv,
                    cp,
                    origin,
                });
            }
            self.uploaded = Some(Upload {
                descriptors: gpu.upload(&descriptors)?,
                data: gpu.upload(&data)?,
                gpu,
            });
        }
        Ok(self.uploaded.as_ref().unwrap())
    }
    fn curvature_samples(&mut self, queries: &[Sample]) -> DriverResult<Vec<[f64; 2]>> {
        let upload = self.upload()?;
        let gpu = &upload.gpu;
        let samples = gpu.upload(queries)?;
        let output = gpu.output::<[f64; 2]>(queries.len())?;
        gpu.launch(
            "surface_curvature",
            LaunchConfig::linear(queries.len() as u64, 128),
            &[
                upload.descriptors.device_ptr(),
                upload.data.device_ptr(),
                samples.device_ptr(),
                output.device_ptr(),
                queries.len() as u64,
            ],
        )?;
        gpu.read(&output, queries.len())
    }
    pub fn curvature(&mut self, faces: usize) -> DriverResult<Option<Vec<Option<[f64; 2]>>>> {
        let spans = |k: &[f64]| k.windows(2).filter(|w| w[1] > w[0]).count();
        let count = self
            .patches
            .iter()
            .map(|(_, s, _)| 9 * spans(&s.ku) * spans(&s.kv))
            .sum();
        attempt("nurbs_curvature", count, Some(256), || {
            let mut queries = Vec::with_capacity(count);
            for (patch, (_, s, _)) in self.patches.iter().enumerate() {
                probes(s, |u, v| {
                    queries.push(Sample {
                        patch: patch as u64,
                        u,
                        v,
                    })
                });
            }
            let values = self.curvature_samples(&queries)?;
            let mut out = vec![None; faces];
            for (q, v) in queries.iter().zip(values) {
                let maximum = out[self.patches[q.patch as usize].0].get_or_insert([0.0_f64; 2]);
                for k in 0..2 {
                    maximum[k] = maximum[k].max(v[k]);
                }
            }
            Ok(out)
        })
    }
    pub fn points(&mut self, faces: usize, stations: &[Vec<f64>]) -> DriverResult<Option<Points>> {
        let count = self
            .patches
            .iter()
            .map(|(_, _, s)| (stations[s[0].0].len() - 2) * (stations[s[1].0].len() - 2))
            .sum();
        attempt("nurbs_interior_points", count, Some(4096), || {
            // Upload the two station axes once per face. The kernel reconstructs
            // the tensor grid, so transfer size grows with rows + columns.
            let mut axes = Vec::new();
            let mut grids = Vec::new();
            let mut offsets = vec![None; faces];
            let mut start = 0;
            for (patch, &(face, surface, sides)) in self.patches.iter().enumerate() {
                let us = &stations[sides[0].0];
                let vs = &stations[sides[1].0];
                if us.len() == 2 || vs.len() == 2 {
                    continue;
                }
                let ((u0, u1), (v0, v1)) = surface.domain();
                let u_offset = axes.len() as u64;
                axes.extend(us[1..us.len() - 1].iter().map(|&u| u0 + (u1 - u0) * u));
                let v_offset = axes.len() as u64;
                axes.extend(vs[1..vs.len() - 1].iter().map(|&v| v0 + (v1 - v0) * v));
                grids.push([
                    patch as u64,
                    start as u64,
                    u_offset,
                    v_offset,
                    (us.len() - 2) as u64,
                ]);
                offsets[face] = Some(start);
                start += (us.len() - 2) * (vs.len() - 2);
            }
            let upload = self.upload()?;
            let gpu = &upload.gpu;
            let grid_buffer = gpu.upload(&grids)?;
            let axis_buffer = gpu.upload(&axes)?;
            let buffer = gpu.output::<V3>(count)?;
            gpu.launch(
                "surface_points",
                LaunchConfig::linear(count as u64, 128),
                &[
                    upload.descriptors.device_ptr(),
                    upload.data.device_ptr(),
                    grid_buffer.device_ptr(),
                    axis_buffer.device_ptr(),
                    buffer.device_ptr(),
                    grids.len() as u64,
                    count as u64,
                ],
            )?;
            let values = gpu.read(&buffer, count)?;
            Ok(Points {
                offsets,
                values,
                buffer,
            })
        })
    }
}
