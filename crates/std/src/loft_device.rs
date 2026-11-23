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

fn launch<D: AgDevice + KernelExt>(
    device: &D,
    input: &[f64],
    count: usize,
    name: &'static str,
    stride: usize,
) -> DriverResult<Vec<f64>> {
    let lib = device.load_library(&LibrarySource {
        name: NAME,
        cuda_source: Some(include_str!("loft.cu").into()),
        cpu_kernels: &[(NAME, reference), ("ag_surface_rows", rows_reference)],
        options: Vec::new(),
    })?;
    let stream = device.stream(Priority::Normal)?;
    let input = bytemuck::cast_slice(input);
    let src = device.alloc_bytes(&stream, input.len())?;
    let bytes = count * stride * size_of::<f64>();
    let out = device.alloc_bytes(&stream, bytes)?;
    let params = device.alloc_bytes(&stream, 24)?;
    let addresses = [src.device_ptr(), out.device_ptr(), count as u64];
    device.upload_from_host_async(&stream, input, &src, 0)?;
    device.upload_from_host_async(&stream, bytemuck::bytes_of(&addresses), &params, 0)?;
    device.launch(
        &stream,
        lib,
        name,
        &LaunchConfig::linear(count as u64, 64),
        &params,
    )?;
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
