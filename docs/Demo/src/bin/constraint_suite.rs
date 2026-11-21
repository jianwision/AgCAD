#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn integer(v: f64, max: usize) -> Result<usize> {
    if !v.is_finite() || v < 1. || v.fract() != 0. || v > max as f64 {
        return Err("Invalid packed constraint dimension".into());
    }
    Ok(v as usize)
}
fn grid(count: usize, batch: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count as u32).div_ceil(256), batch as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn reduction(batch: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or("output/constraints".into()));
    if args.next().is_some() {
        return Err("Use constraint_suite [OUTPUT_DIRECTORY]".into());
    }
    let bytes = fs::read(out.join("batch.bin"))?;
    if bytes.len() < 128 || bytes.len() % 8 != 0 || bytes.len() > 256 * 1024 * 1024 {
        return Err("Invalid constraint binary size".into());
    }
    let input: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if input.iter().any(|v| !v.is_finite()) || input[0] != 20261120. || input[1] != 1. {
        return Err("Invalid constraint binary header/data".into());
    }
    let batch = integer(input[2], 64)?;
    let n = integer(input[3], 1024)?;
    let m = if input[4] == 0. {
        0
    } else {
        integer(input[4], 8192)?
    };
    let k = if input[5] == 0. {
        0
    } else {
        integer(input[5], 512)?
    };
    let rows = integer(input[6], 8192)?;
    let iterations = integer(input[7], 2048)?;
    let tol = input[8];
    let istride = m * k + m + n * k + n + rows * n + rows + k;
    let wstride = 7 * k + 2 * m + 8 + n + rows;
    if input.len() != 16 + batch * istride || !(0. < tol && tol < 1.) || k > n {
        return Err("Constraint binary dimensions disagree".into());
    }
    for c in 0..batch {
        let start = 16 + (c + 1) * istride - k;
        if input[start..start + k].iter().any(|v| *v <= 0.) {
            return Err("Nonpositive PCG diagonal".into());
        }
    }
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: all buffers, copies, graph nodes and events use this one stream.
    // No allocation is accessed by another stream while event tracking is off.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let source = fs::read_to_string("cuda/constraint_solver.cu")?;
    let timer = Instant::now();
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("constraint_solver.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Constraint NVRTC: {e:?}"))?;
    let compile_ms = timer.elapsed().as_secs_f64() * 1000.;
    let module = ctx.load_module(ptx)?;
    let names = [
        "initialize",
        "initialize_reduce",
        "matvec",
        "transpose",
        "step_size",
        "update",
        "convergence",
        "direction",
        "reconstruct",
        "residuals",
        "probes",
        "finalize",
    ];
    let functions = names
        .iter()
        .map(|s| module.load_function(s))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let device_input = stream.clone_htod(&input)?;
    let mut arena = stream.alloc_zeros::<f64>(batch * wstride)?;
    // SAFETY: the complete pinned buffer is populated before any host read.
    let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f64>(batch * wstride, 0)? };
    stream.synchronize()?;
    let (ni, mi, ki, ri) = (n as i32, m as i32, k as i32, rows as i32);
    let timer = Instant::now();
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    // SAFETY: dimensions are bounded above and packed input/output strides match
    // constraint_solver.cu. Captured buffers and functions outlive graph launch.
    unsafe {
        macro_rules! run {
            ($f:expr, $count:expr) => {
                if $count > 0 {
                    stream
                        .launch_builder(&functions[$f])
                        .arg(&device_input)
                        .arg(&mut arena)
                        .arg(&ni)
                        .arg(&mi)
                        .arg(&ki)
                        .arg(&ri)
                        .launch(grid($count, batch))?;
                }
            };
            ($f:expr, $count:expr, $flag:expr) => {
                if $count > 0 {
                    stream
                        .launch_builder(&functions[$f])
                        .arg(&device_input)
                        .arg(&mut arena)
                        .arg(&ni)
                        .arg(&mi)
                        .arg(&ki)
                        .arg(&ri)
                        .arg(&$flag)
                        .launch(grid($count, batch))?;
                }
            };
        }
        macro_rules! reduce {
            ($f:expr) => {
                stream
                    .launch_builder(&functions[$f])
                    .arg(&mut arena)
                    .arg(&ni)
                    .arg(&mi)
                    .arg(&ki)
                    .arg(&ri)
                    .launch(reduction(batch))?;
            };
            ($f:expr, $tol:expr) => {
                stream
                    .launch_builder(&functions[$f])
                    .arg(&mut arena)
                    .arg(&ni)
                    .arg(&mi)
                    .arg(&ki)
                    .arg(&ri)
                    .arg(&$tol)
                    .launch(reduction(batch))?;
            };
        }
        run!(0, k);
        reduce!(1, tol);
        for _ in 0..iterations {
            run!(2, m, 0_i32);
            run!(3, k, 0_i32);
            reduce!(4);
            run!(5, k);
            reduce!(6, tol);
            if k > 0 {
                stream
                    .launch_builder(&functions[7])
                    .arg(&mut arena)
                    .arg(&ni)
                    .arg(&mi)
                    .arg(&ki)
                    .arg(&ri)
                    .launch(grid(k, batch))?;
            }
        }
        run!(2, m, 1_i32);
        run!(3, k, 1_i32);
        run!(8, n);
        run!(9, rows);
        run!(10, m.max(k));
        reduce!(11, tol);
    }
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags(0))?
        .ok_or("Empty graph")?;
    graph.upload()?;
    stream.synchronize()?;
    let graph_setup_ms = timer.elapsed().as_secs_f64() * 1000.;
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    graph.launch()?;
    let solved = stream.record_event(flags)?;
    stream.memcpy_dtoh(&arena, &mut pinned)?;
    let copied = stream.record_event(flags)?;
    copied.synchronize()?;
    let data = pinned.as_slice()?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err("Nonfinite constraint solver result".into());
    }
    let mut output = Vec::with_capacity(data.len() * 8);
    for value in data {
        output.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(out.join("result.bin"), output)?;
    let device = ctx.name()?.replace('"', "");
    let report = format!(
        "{{\"device\":\"{device}\",\"batch\":{batch},\"variables\":{n},\"soft_rows\":{m},\"reduced_dof\":{k},\"all_rows\":{rows},\"iteration_capacity\":{iterations},\"relative_tolerance\":{tol},\"compile_ms\":{compile_ms},\"graph_setup_ms\":{graph_setup_ms},\"gpu_graph_ms\":{},\"copy_ms\":{},\"htod_copies\":1,\"dtoh_copies\":1,\"intermediate_readbacks\":0,\"download_bytes\":{},\"graph_launches\":1}}",
        start.elapsed_ms(&solved)?,
        solved.elapsed_ms(&copied)?,
        data.len() * 8,
    );
    fs::write(out.join("gpu_report.json"), &report)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::integer;
    #[test]
    fn rejects_untrusted_dimensions() {
        for bad in [0., -1., 1.5, f64::NAN, f64::INFINITY, 1025.] {
            assert!(integer(bad, 1024).is_err());
        }
        assert_eq!(integer(512., 512).unwrap(), 512);
    }
}
