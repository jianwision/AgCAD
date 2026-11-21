#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Input {
    bytes: Vec<u8>,
    n: usize,
    candidates: usize,
    rounds: usize,
    stride: usize,
    output_words: usize,
    shared_bytes: usize,
}
fn u32_at(data: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize
}
fn f64_at(data: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}
fn parse(bytes: Vec<u8>) -> Result<Input> {
    if bytes.len() < 128 || bytes.len() > 256 * 1024 * 1024 || &bytes[..8] != b"GPHARN01" {
        return Err("Invalid GPU harness binary".into());
    }
    let fields: Vec<_> = (0..12).map(|i| u32_at(&bytes, 8 + 4 * i)).collect();
    let (version, n, m, k, r, candidates, rounds) = (
        fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
    );
    if version != 1
        || n != 192
        || m == 0
        || m > 8192
        || k == 0
        || k > 192
        || r < m
        || r > 8192
        || !(1..=64).contains(&candidates)
        || !(1..=32).contains(&rounds)
        || fields[10] != 24
    {
        return Err("Unsupported harness dimensions/ABI (four bicubic patches required)".into());
    }
    let config: Vec<_> = (0..9).map(|i| f64_at(&bytes, 56 + 8 * i)).collect();
    if config.iter().any(|v| !v.is_finite())
        || config[0] >= config[1]
        || !(0. < config[2] && config[2] < 1.)
        || config[3] < 0.
        || config[4] < 0.
        || config[5] < 0.
        || config[6] <= 0.
        || config[8] <= 0.
    {
        return Err("Invalid harness numerical configuration".into());
    }
    let template_words = n * k + n + 2 * k + r * n + 5 * r + m * k + 2 * m;
    let shared_bytes = 128 + 8 * template_words;
    if bytes.len() != shared_bytes + 24 * candidates + 32 * candidates * rounds {
        return Err("Harness binary length disagrees with dimensions".into());
    }
    for offset in (128..shared_bytes).step_by(8) {
        if !f64_at(&bytes, offset).is_finite() {
            return Err("Nonfinite shared harness template".into());
        }
    }
    let tol_offset = 128 + 8 * (n * k + n + 2 * k + r * n + 2 * r);
    for i in 0..r {
        let tol = f64_at(&bytes, tol_offset + 8 * i);
        let hard = f64_at(&bytes, tol_offset + 8 * (r + i));
        let weight = f64_at(&bytes, tol_offset + 8 * (2 * r + i));
        if tol <= 0. || (hard != 0. && hard != 1.) || weight < 0. {
            return Err("Invalid shared constraint tolerance/mask/weight".into());
        }
    }
    for c in 0..candidates {
        let offset = shared_bytes + 24 * c;
        for value in [f64_at(&bytes, offset), f64_at(&bytes, offset + 8)] {
            if !value.is_finite() || value < config[0] || value > config[1] {
                return Err("Invalid immutable goal or initial parameter".into());
            }
        }
        if u32_at(&bytes, offset + 16) > 256 || u32_at(&bytes, offset + 20) != fields[9] {
            return Err("Invalid candidate budget/goal revision".into());
        }
    }
    // Command values intentionally reach the device validator, including NaN
    // negative fixtures. Byte ranges are validated here; semantics on the GPU.
    let stride = 2 * n + 2 * k + 32 + rounds * 24;
    Ok(Input {
        bytes,
        n,
        candidates,
        rounds,
        stride,
        output_words: candidates * stride + 8,
        shared_bytes,
    })
}
fn grid(n: usize, candidates: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), candidates as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn blocks(candidates: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (candidates as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or("output/harness".into()));
    if args.next().is_some() {
        return Err("Use gpu_harness [OUTPUT_DIRECTORY]".into());
    }
    let input = parse(fs::read(out.join("input.bin"))?)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: every operation uses this single dedicated stream.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let timer = Instant::now();
    let source = fs::read_to_string("cuda/gpu_harness.cu")?;
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("gpu_harness.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Harness NVRTC: {e:?}"))?;
    let compile_ms = timer.elapsed().as_secs_f64() * 1000.;
    fs::write(out.join("harness-source.cu"), source)?;
    fs::write(out.join("harness.ptx"), ptx.to_src())?;
    let module = ctx.load_module(ptx)?;
    let functions = [
        "harness_initialize",
        "harness_verify",
        "harness_baseline",
        "harness_propose",
        "harness_construct",
        "harness_decide",
        "harness_commit",
        "harness_trace",
        "harness_select",
        "harness_compare",
    ]
    .iter()
    .map(|name| module.load_function(name))
    .collect::<std::result::Result<Vec<_>, _>>()?;
    let device_input = stream.clone_htod(&input.bytes)?;
    let mut arena = stream.alloc_zeros::<f64>(input.output_words)?;
    let mut snapshot = stream.alloc_zeros::<f64>(input.output_words)?;
    // SAFETY: a full DtoH copy completes before reading this pinned allocation.
    let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f64>(input.output_words, 0)? };
    stream.synchronize()?;
    let timer = Instant::now();
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    stream.memset_zeros(&mut arena)?;
    // SAFETY: parse() checks every packed byte range, dimensions and template
    // values. All graph-owned argument allocations/functions outlive execution.
    unsafe {
        macro_rules! call {
            ($f:expr, $grid:expr) => {
                stream
                    .launch_builder(&functions[$f])
                    .arg(&device_input)
                    .arg(&mut arena)
                    .launch($grid)?;
            };
            ($f:expr, $grid:expr, $value:expr) => {
                stream
                    .launch_builder(&functions[$f])
                    .arg(&device_input)
                    .arg(&mut arena)
                    .arg(&$value)
                    .launch($grid)?;
            };
        }
        call!(0, grid(input.n, input.candidates));
        call!(1, blocks(input.candidates), 1_u32);
        call!(2, grid(input.candidates, 1));
        for round in 0..input.rounds as u32 {
            call!(3, grid(input.candidates, 1), round);
            call!(4, grid(input.n, input.candidates), round);
            call!(1, blocks(input.candidates), 0_u32);
            call!(5, grid(input.candidates, 1));
            call!(6, grid(input.n, input.candidates));
            call!(7, grid(input.candidates, 1), round);
        }
        call!(8, grid(1, 1));
    }
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags(0))?
        .ok_or("Empty harness graph")?;
    graph.upload()?;
    stream.synchronize()?;
    let graph_setup_ms = timer.elapsed().as_secs_f64() * 1000.;
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    graph.launch()?;
    let first = stream.record_event(flags)?;
    stream.memcpy_dtod(&arena, &mut snapshot)?;
    graph.launch()?;
    // SAFETY: both arrays have the checked output_words length. The comparison
    // kernel reports bit mismatches after replay without an intermediate DtoH.
    unsafe {
        stream
            .launch_builder(&functions[9])
            .arg(&device_input)
            .arg(&mut arena)
            .arg(&snapshot)
            .launch(blocks(1))?;
    }
    let replayed = stream.record_event(flags)?;
    stream.memcpy_dtoh(&arena, &mut pinned)?;
    let copied = stream.record_event(flags)?;
    copied.synchronize()?;
    let data = pinned.as_slice()?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err("Nonfinite GPU harness output".into());
    }
    let mut output = Vec::with_capacity(data.len() * 8);
    for value in data {
        output.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(out.join("result.bin"), output)?;
    let device = ctx.name()?.replace('"', "");
    let report = format!(
        "{{\"device\":\"{device}\",\"candidates\":{},\"rounds\":{},\"stride\":{},\"output_words\":{},\"shared_template_bytes\":{},\"upload_bytes\":{},\"download_bytes\":{},\"compile_ms\":{compile_ms},\"graph_setup_ms\":{graph_setup_ms},\"first_graph_ms\":{},\"replay_copy_compare_ms\":{},\"copy_ms\":{},\"graph_launches\":2,\"htod_copies\":1,\"dtoh_copies\":1,\"intermediate_readbacks\":0,\"replay_mismatched_words\":{}}}",
        input.candidates,
        input.rounds,
        input.stride,
        input.output_words,
        input.shared_bytes,
        input.bytes.len(),
        data.len() * 8,
        start.elapsed_ms(&first)?,
        first.elapsed_ms(&replayed)?,
        replayed.elapsed_ms(&copied)?,
        data[input.output_words - 4],
    );
    fs::write(out.join("gpu_report.json"), &report)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse;
    #[test]
    fn rejects_truncated_and_unknown_harness_input() {
        assert!(parse(vec![]).is_err());
        assert!(parse(vec![0; 128]).is_err());
        let mut input = vec![0; 128];
        input[..8].copy_from_slice(b"GPHARN01");
        assert!(parse(input).is_err());
    }
}
