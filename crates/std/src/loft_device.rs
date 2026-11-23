use agcad_driver::*;
use agcad_geo::math::{v3, Frame, V3};
use agcad_runtime::session::{with_default, DeviceChoice};

const NAME: &str = "ag_loft_poles";

// Same three 64-bit words as LoftParams in loft.cu. CPU device pointers are host addresses.
unsafe fn reference(params: *const u8, _: &LaunchConfig) {
    let [src, dst, count] = unsafe { params.cast::<[u64; 3]>().read_unaligned() };
    let count = count as usize;
    let src = unsafe { std::slice::from_raw_parts(src as *const f64, 2 * (9 + 2 * count)) };
    let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut f64, 12 * count) };
    for (i, patch) in dst.chunks_exact_mut(12).enumerate() {
        for (j, pole) in patch.chunks_exact_mut(3).enumerate() {
            let frame = &src[(j % 2) * (9 + 2 * count)..];
            let uv = &frame[9 + 2 * ((i + j / 2) % count)..];
            for axis in 0..3 {
                pole[axis] = frame[axis] + uv[0] * frame[3 + axis] + uv[1] * frame[6 + axis];
            }
        }
    }
}

unsafe fn rows_reference(params: *const u8, _: &LaunchConfig) {
    let [src, dst, count] = unsafe { params.cast::<[u64; 3]>().read_unaligned() };
    let n = count as usize;
    let offsets = unsafe { std::slice::from_raw_parts(src as *const f64, n + 1) };
    let terms = unsafe {
        std::slice::from_raw_parts((src as *const f64).add(n + 1), 5 * offsets[n] as usize)
    };
    let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut f64, n * 4) };
    for (i, p) in out.chunks_exact_mut(4).enumerate() {
        let mut h = [0.; 4];
        for term in terms[5 * offsets[i] as usize..5 * offsets[i + 1] as usize].chunks_exact(5) {
            let w = term[0] * term[4];
            for k in 0..3 {
                h[k] += w * term[k + 1];
            }
            h[3] += w;
        }
        p.copy_from_slice(&[h[0] / h[3], h[1] / h[3], h[2] / h[3], h[3]]);
    }
}

unsafe fn nurbs_grid_reference(params: *const u8, _: &LaunchConfig) {
    let [src, dst, count] = unsafe { params.cast::<[u64; 3]>().read_unaligned() };
    let total = count as usize;
    let hdr = unsafe { std::slice::from_raw_parts(src as *const f64, 4) };
    let (nu, nv, su, sv) = (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize, hdr[3] as usize);
    let len = 4 + 2 * su * nu + 2 * sv * nv + 4 * nu * nv;
    let input = unsafe { std::slice::from_raw_parts(src as *const f64, len) };
    let bu = &input[4..4 + su * nu];
    let dbu = &input[4 + su * nu..4 + 2 * su * nu];
    let bv = &input[4 + 2 * su * nu..4 + 2 * su * nu + sv * nv];
    let dbv = &input[4 + 2 * su * nu + sv * nv..4 + 2 * su * nu + 2 * sv * nv];
    let poles = &input[4 + 2 * su * nu + 2 * sv * nv..];
    let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut f64, total * 9) };
    for idx in 0..total {
        let (iu, iv) = (idx / sv, idx % sv);
        let (bu_r, dbu_r) = (&bu[iu * nu..(iu + 1) * nu], &dbu[iu * nu..(iu + 1) * nu]);
        let (bv_r, dbv_r) = (&bv[iv * nv..(iv + 1) * nv], &dbv[iv * nv..(iv + 1) * nv]);
        let (mut h, mut hu, mut hv) = ([0.0; 4], [0.0; 4], [0.0; 4]);
        for i in 0..nu {
            let (bi, dbi) = (bu_r[i], dbu_r[i]);
            if bi == 0.0 && dbi == 0.0 { continue; }
            for j in 0..nv {
                let (bj, dbj) = (bv_r[j], dbv_r[j]);
                if bj == 0.0 && dbj == 0.0 { continue; }
                let pt = &poles[4 * (i * nv + j)..4 * (i * nv + j) + 4];
                let w = pt[3];
                let (w0, wu, wv) = (bi * bj * w, dbi * bj * w, bi * dbj * w);
                for k in 0..3 { h[k] += w0 * pt[k]; hu[k] += wu * pt[k]; hv[k] += wv * pt[k]; }
                h[3] += w0; hu[3] += wu; hv[3] += wv;
            }
        }
        let invw = if h[3] != 0.0 { 1.0 / h[3] } else { 0.0 };
        let o = &mut out[9 * idx..9 * idx + 9];
        for k in 0..3 {
            let p = h[k] * invw;
            o[k] = p;
            o[3 + k] = (hu[k] - hu[3] * p) * invw;
            o[6 + k] = (hv[k] - hv[3] * p) * invw;
        }
    }
}

pub fn loft_library() -> LibrarySource {
    LibrarySource {
        name: NAME,
        cuda_source: Some(include_str!("loft.cu").into()),
        cpu_kernels: &[
            (NAME, reference),
            ("ag_surface_rows", rows_reference),
            ("ag_nurbs_eval_grid", nurbs_grid_reference),
        ],
        options: Vec::new(),
    }
}

/// Launch directly on already-resident device buffers (zero host geometry copies).
pub fn launch_device<D: AgDevice + KernelExt>(
    device: &D,
    stream: &D::Stream,
    lib: LibraryId,
    name: &'static str,
    src: &D::Buffer,
    out: &D::Buffer,
    count: usize,
) -> DriverResult<()> {
    let params = device.alloc_bytes(stream, 24)?;
    let addresses = [src.device_ptr(), out.device_ptr(), count as u64];
    device.upload_tagged_async(stream, TransferKind::Control, bytemuck::bytes_of(&addresses), &params, 0)?;
    device.launch(stream, lib, name, &LaunchConfig::linear(count as u64, 64), &params)?;
    Ok(())
}

fn launch<D: AgDevice + KernelExt>(
    device: &D,
    input: &[f64],
    count: usize,
    name: &'static str,
    stride: usize,
) -> DriverResult<Vec<f64>> {
    // load_library deduplicates by name on both CudaDevice and CpuDevice.
    let lib = device.load_library(&loft_library())?;
    let stream = device.stream(Priority::Normal)?;
    let input = bytemuck::cast_slice(input);
    let src = device.alloc_bytes(&stream, input.len())?;
    let bytes = count * stride * size_of::<f64>();
    let out = device.alloc_bytes(&stream, bytes)?;
    device.upload_from_host_async(&stream, input, &src, 0)?;
    launch_device(device, &stream, lib, name, &src, &out, count)?;
    let mut result = device.pinned(bytes)?;
    let _done = device.download_packed_async(
        &stream,
        &[Region {
            buf: &out,
            offset: 0,
            bytes,
        }],
        &mut result,
    )?;
    device.sync(&stream)?;
    Ok(bytemuck::cast_slice::<u8, f64>(result.as_bytes()).to_vec())
}

fn dispatch(
    input: &[f64],
    count: usize,
    name: &'static str,
    stride: usize,
) -> DriverResult<(Vec<f64>, &'static str)> {
    with_default(|s| match &s.device {
        DeviceChoice::Cuda(d) => launch(&**d, input, count, name, stride).map(|p| (p, "cuda")),
        DeviceChoice::CpuReference(d) => {
            launch(&**d, input, count, name, stride).map(|p| (p, "cpu-reference"))
        }
    })?
}

pub type Weighted = (V3, f64);
pub fn rows(rows: &[Vec<(f64, Weighted)>]) -> DriverResult<(Vec<Weighted>, &'static str)> {
    let mut data = Vec::with_capacity(rows.len() + 1);
    let mut offset = 0;
    data.push(0.);
    for row in rows {
        offset += row.len();
        data.push(offset as f64);
    }
    for row in rows {
        for &(c, (p, w)) in row {
            data.extend([c, p.x, p.y, p.z, w]);
        }
    }
    let (result, device) = dispatch(&data, rows.len(), "ag_surface_rows", 4)?;
    Ok((
        result
            .chunks_exact(4)
            .map(|p| (v3(p[0], p[1], p[2]), p[3]))
            .collect(),
        device,
    ))
}

pub fn patches(
    frames: &[Frame; 2],
    sections: &[Vec<[f64; 2]>; 2],
) -> DriverResult<(Vec<[V3; 4]>, &'static str)> {
    let count = sections[0].len();
    let mut input = Vec::with_capacity(18 + 4 * count);
    for (frame, points) in frames.iter().zip(sections) {
        for v in [frame.origin, frame.x, frame.y] {
            input.extend([v.x, v.y, v.z]);
        }
        for point in points {
            input.extend(point);
        }
    }
    let (out, device) = dispatch(&input, count, NAME, 12)?;
    Ok((
        out.chunks_exact(12)
            .map(|p| std::array::from_fn(|j| v3(p[j * 3], p[j * 3 + 1], p[j * 3 + 2])))
            .collect(),
        device,
    ))
}

/// Evaluate a rational NURBS surface over a tensor-product `(us × vs)` grid on the GPU
/// (returns `(point, du, dv)` per grid sample and device label).
pub fn eval_nurbs_grid(
    surf: &agcad_geo::nurbs::NurbsSurface,
    us: &[f64],
    vs: &[f64],
) -> DriverResult<(Vec<(V3, V3, V3)>, &'static str)> {
    let (nu, nv) = (surf.nu, surf.nv);
    let (su, sv) = (us.len(), vs.len());
    let mut input = Vec::with_capacity(4 + 2 * su * nu + 2 * sv * nv + 4 * nu * nv);
    input.extend([nu as f64, nv as f64, su as f64, sv as f64]);
    let mut bu = vec![0.0; su * nu];
    let mut dbu = vec![0.0; su * nu];
    for (iu, &u) in us.iter().enumerate() {
        let span = agcad_geo::nurbs::find_span(nu - 1, surf.du, u, &surf.ku);
        let d = agcad_geo::nurbs::ders_basis(span, u, surf.du, 1, &surf.ku);
        for k in 0..=surf.du {
            let col = span - surf.du + k;
            bu[iu * nu + col] = d[0][k];
            dbu[iu * nu + col] = d[1][k];
        }
    }
    let mut bv = vec![0.0; sv * nv];
    let mut dbv = vec![0.0; sv * nv];
    for (iv, &v) in vs.iter().enumerate() {
        let span = agcad_geo::nurbs::find_span(nv - 1, surf.dv, v, &surf.kv);
        let d = agcad_geo::nurbs::ders_basis(span, v, surf.dv, 1, &surf.kv);
        for k in 0..=surf.dv {
            let col = span - surf.dv + k;
            bv[iv * nv + col] = d[0][k];
            dbv[iv * nv + col] = d[1][k];
        }
    }
    input.extend(bu);
    input.extend(dbu);
    input.extend(bv);
    input.extend(dbv);
    for (p, &w) in surf.poles.iter().zip(&surf.weights) {
        input.extend([p.x, p.y, p.z, w]);
    }
    let count = su * sv;
    let (out, device) = dispatch(&input, count, "ag_nurbs_eval_grid", 9)?;
    let jets = out
        .chunks_exact(9)
        .map(|o| (v3(o[0], o[1], o[2]), v3(o[3], o[4], o[5]), v3(o[6], o[7], o[8])))
        .collect();
    Ok((jets, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_geo::nurbs::NurbsSurface;

    #[test]
    fn nurbs_grid_kernel_matches_exact_jet() {
        // Bilinear twisted saddle patch evaluated via device kernel vs NurbsSurface::jet
        let surf = NurbsSurface {
            du: 1, dv: 1, nu: 2, nv: 2,
            ku: vec![0.0, 0.0, 1.0, 1.0],
            kv: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![v3(0.0, 0.0, 0.0), v3(0.0, 10.0, 2.0), v3(10.0, 0.0, 1.0), v3(10.0, 10.0, 5.0)],
            weights: vec![1.0, 1.2, 0.9, 1.1],
        };
        let dev = agcad_driver_cpu::CpuDevice::new();
        let us = [0.0, 0.25, 0.5, 0.75, 1.0];
        let vs = [0.0, 0.3, 0.7, 1.0];
        let mut input = Vec::new();
        input.extend([2.0, 2.0, us.len() as f64, vs.len() as f64]);
        let mut bu = vec![0.0; us.len() * 2];
        let mut dbu = vec![0.0; us.len() * 2];
        for (iu, &u) in us.iter().enumerate() {
            let span = agcad_geo::nurbs::find_span(1, 1, u, &surf.ku);
            let d = agcad_geo::nurbs::ders_basis(span, u, 1, 1, &surf.ku);
            for k in 0..=1 { bu[iu * 2 + span - 1 + k] = d[0][k]; dbu[iu * 2 + span - 1 + k] = d[1][k]; }
        }
        let mut bv = vec![0.0; vs.len() * 2];
        let mut dbv = vec![0.0; vs.len() * 2];
        for (iv, &v) in vs.iter().enumerate() {
            let span = agcad_geo::nurbs::find_span(1, 1, v, &surf.kv);
            let d = agcad_geo::nurbs::ders_basis(span, v, 1, 1, &surf.kv);
            for k in 0..=1 { bv[iv * 2 + span - 1 + k] = d[0][k]; dbv[iv * 2 + span - 1 + k] = d[1][k]; }
        }
        input.extend(bu); input.extend(dbu); input.extend(bv); input.extend(dbv);
        for (p, &w) in surf.poles.iter().zip(&surf.weights) { input.extend([p.x, p.y, p.z, w]); }
        let out = launch(&*dev, &input, us.len() * vs.len(), "ag_nurbs_eval_grid", 9).unwrap();
        for (iu, &u) in us.iter().enumerate() {
            for (iv, &v) in vs.iter().enumerate() {
                let o = &out[9 * (iu * vs.len() + iv)..9 * (iu * vs.len() + iv) + 9];
                let j = surf.jet(u, v);
                assert!((v3(o[0], o[1], o[2]) - j.p).norm() < 1e-12);
                assert!((v3(o[3], o[4], o[5]) - j.du).norm() < 1e-12);
                assert!((v3(o[6], o[7], o[8]) - j.dv).norm() < 1e-12);
            }
        }
    }
}
