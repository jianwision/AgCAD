#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn dimension(v: f64, low: usize, high: usize) -> Result<usize> {
    if !v.is_finite() || v.fract() != 0. || v < (low as f64) || v > (high as f64) {
        return Err("Invalid Class-A dimension".into());
    }
    Ok(v as usize)
}
fn validate(input: &[f64]) -> Result<(usize, usize)> {
    if input.len() < 16
        || input[0] != 20261121.
        || input[1] != 1.
        || input.iter().any(|v| !v.is_finite())
        || input[4..16].iter().any(|v| *v != 0.)
    {
        return Err("Invalid Class-A input".into());
    }
    let count = dimension(input[2], 1, 32)?;
    let samples = dimension(input[3], 5, 65)?;
    if input.len() != 16 + 4360 * count {
        return Err("Class-A payload length mismatch".into());
    }
    for record in input[16..].chunks_exact(4360) {
        dimension(record[0], 4, 8)?;
        dimension(record[1], 4, 8)?;
        dimension(record[2], 0, 512)?;
        if !(0. ..=100.).contains(&record[3])
            || record[5..8].iter().any(|v| *v != 0.)
            || record[8..200].iter().any(|v| v.abs() > 1e6)
            || record[200..264].iter().any(|v| *v != 0. && *v != 1.)
        {
            return Err("Invalid fairing/geometry parameters".into());
        }
        let h = &record[264..];
        let bound = h
            .chunks_exact(64)
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(1_f64, f64::max);
        if record[4] <= 0. || record[4] > 0.5 / bound + 1e-15 || h.iter().any(|v| v.abs() > 256.) {
            return Err("Invalid fairing operator/step".into());
        }
        for i in 0..64 {
            for j in 0..64 {
                if (h[64 * i + j] - h[64 * j + i]).abs() > 1e-12 {
                    return Err("Asymmetric fairing operator".into());
                }
            }
        }
    }
    Ok((count, samples))
}
fn main() -> Result<()> {
    let wall = Instant::now();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or("Use class_a_suite OUTPUT_DIRECTORY")?);
    if args.next().is_some() {
        return Err("Unexpected Class-A argument".into());
    }
    let bytes = fs::read(root.join("batch.bin"))?;
    if bytes.len() % 8 != 0 || bytes.len() > 2 * 1024 * 1024 {
        return Err("Invalid Class-A payload bytes".into());
    }
    let input: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (count, samples) = validate(&input)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let source = fs::read_to_string("cuda/class_a.cu")?;
    let compile = Instant::now();
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("class_a.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Class-A NVRTC: {e:?}"))?;
    let compile_ms = compile.elapsed().as_secs_f64() * 1000.;
    fs::write(root.join("source.cu"), &source)?;
    fs::write(root.join("kernel.ptx"), ptx.to_src())?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("class_a_batch")?;
    let data = stream.clone_htod(&input)?;
    let words = count * (192 + 48 * samples * samples);
    let mut result = stream.alloc_zeros::<f64>(words)?;
    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    // SAFETY: all dimensions, strides and allocations are bounded by validate.
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&data)
            .arg(&mut result)
            .launch(LaunchConfig {
                grid_dim: (count as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let values = stream.clone_dtoh(&result)?;
    stream.synchronize()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Nonfinite Class-A result".into());
    }
    fs::write(
        root.join("results.bin"),
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    fs::write(
        root.join("gpu-report.json"),
        format!(
            "{{\"patches\":{count},\"samples_per_patch\":{},\"context_creations\":1,\"final_readbacks\":1,\"intermediate_readbacks\":0,\"upload_bytes\":{},\"download_bytes\":{},\"compile_ms\":{compile_ms},\"kernel_ms\":{},\"wall_ms\":{}}}",
            samples * samples,
            bytes.len(),
            words * 8,
            start.elapsed_ms(&end)?,
            wall.elapsed().as_secs_f64() * 1000.
        ),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::{dimension, validate};
    #[test]
    fn rejects_bad_dimensions() {
        assert!(validate(&[]).is_err());
        assert!(dimension(f64::NAN, 1, 32).is_err());
        assert!(dimension(3.5, 1, 32).is_err());
        assert!(dimension(33., 1, 32).is_err());
    }
}
