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
    patches: usize,
    systems: usize,
    repeats: usize,
    programs: usize,
    rounds: usize,
    words: usize,
}
fn parse(bytes: Vec<u8>) -> Result<Input> {
    if bytes.len() < 32 || &bytes[..8] != b"GPDIA001" {
        return Err("Invalid diagnostic header".into());
    }
    let fields: Vec<usize> = (0..6)
        .map(|i| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize)
        .collect();
    let (patches, systems, repeats) = (fields[1], fields[2], fields[3]);
    let (programs, rounds) = (fields[4], fields[5]);
    if fields[0] != 1
        || !(1..=2048).contains(&patches)
        || !(1..=256).contains(&systems)
        || !(2..=256).contains(&repeats)
        || !(1..=64).contains(&programs)
        || rounds != 16
        || bytes.len()
            != 32 + 8 * (48 * patches + 49 * systems + 2958 + (13 + 7 * rounds) * programs)
    {
        return Err("Invalid diagnostic dimensions/version/length".into());
    }
    // Nonfinite patch coordinates intentionally exercise the device finite check.
    // Systems have no such fixture and must remain finite before launching a solve.
    let program_start = 32 + 8 * (48 * patches + 49 * systems + 2958);
    for offset in (32 + 8 * 48 * patches..program_start).step_by(8) {
        if !f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()).is_finite() {
            return Err("Nonfinite diagnostic solve input".into());
        }
    }
    for id in 0..programs {
        let start = program_start + 8 * (13 + 7 * rounds) * id;
        for offset in (start..start + 13 * 8).step_by(8) {
            if !f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()).is_finite() {
                return Err("Nonfinite proposal goal/initial/mode".into());
            }
        }
        let mode = f64::from_le_bytes(bytes[start + 96..start + 104].try_into().unwrap());
        if mode != 0. && mode != 1. {
            return Err("Invalid proposal adapter mode".into());
        }
        for round in 0..rounds {
            let offset = start + 104 + 56 * round;
            let opcode = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            if !opcode.is_finite() || !(0. ..=65535.).contains(&opcode) || opcode.fract() != 0. {
                return Err("Invalid proposal opcode encoding".into());
            }
        }
    }
    Ok(Input {
        bytes,
        patches,
        systems,
        repeats,
        programs,
        rounds,
        words: 8 * patches + 20 * systems + programs * (404 + 20 * rounds) + 1,
    })
}
fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(32), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn main() -> Result<()> {
    let wall = Instant::now();
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or("output/diagnostics".into()));
    if args.next().is_some() {
        return Err("Use diagnostic_suite [OUTPUT_DIRECTORY]".into());
    }
    let input = parse(fs::read(out.join("input.bin"))?)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: all graph operations and copies use this dedicated stream.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let start = Instant::now();
    let source = fs::read_to_string("cuda/diagnostic_suite.cu")?;
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("diagnostic_suite.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Diagnostic NVRTC: {e:?}"))?;
    let compile_ms = start.elapsed().as_secs_f64() * 1000.;
    fs::write(out.join("diagnostic-source.cu"), source)?;
    fs::write(out.join("diagnostic.ptx"), ptx.to_src())?;
    let module = ctx.load_module(ptx)?;
    let geometry = module.load_function("diagnostic_geometry")?;
    let solve = module.load_function("diagnostic_solve")?;
    let proposals = module.load_function("diagnostic_proposals")?;
    let compare = module.load_function("diagnostic_compare")?;
    let device_input = stream.clone_htod(&input.bytes)?;
    let mut arena = stream.alloc_zeros::<f64>(input.words)?;
    let mut previous = stream.alloc_zeros::<f64>(input.words)?;
    // SAFETY: read only after the final copy event has completed.
    let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f64>(input.words, 0)? };
    stream.synchronize()?;
    let setup = Instant::now();
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    stream.memset_zeros(&mut arena)?;
    // SAFETY: parse checks all buffer extents, and kernels additionally bound ids.
    unsafe {
        stream
            .launch_builder(&geometry)
            .arg(&device_input)
            .arg(&mut arena)
            .launch(grid(input.patches))?;
        for method in 0..2_u32 {
            stream
                .launch_builder(&solve)
                .arg(&device_input)
                .arg(&mut arena)
                .arg(&method)
                .launch(grid(input.systems))?;
        }
        stream
            .launch_builder(&proposals)
            .arg(&device_input)
            .arg(&mut arena)
            .launch(grid(input.programs))?;
    }
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags(0))?
        .ok_or("Empty diagnostic graph")?;
    graph.upload()?;
    stream.synchronize()?;
    let graph_setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    graph.launch()?;
    stream.memcpy_dtod(&arena, &mut previous)?;
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    for _ in 0..input.repeats {
        graph.launch()?;
    }
    let ran = stream.record_event(flags)?;
    // Profile the same batched kernels, excluding compilation/transfers.
    unsafe {
        stream
            .launch_builder(&geometry)
            .arg(&device_input)
            .arg(&mut arena)
            .launch(grid(input.patches))?;
    }
    let geometry_done = stream.record_event(flags)?;
    unsafe {
        stream
            .launch_builder(&solve)
            .arg(&device_input)
            .arg(&mut arena)
            .arg(&0_u32)
            .launch(grid(input.systems))?;
    }
    let direct_done = stream.record_event(flags)?;
    unsafe {
        stream
            .launch_builder(&solve)
            .arg(&device_input)
            .arg(&mut arena)
            .arg(&1_u32)
            .launch(grid(input.systems))?;
    }
    let iterative_done = stream.record_event(flags)?;
    unsafe {
        stream
            .launch_builder(&proposals)
            .arg(&device_input)
            .arg(&mut arena)
            .launch(grid(input.programs))?;
    }
    let proposals_done = stream.record_event(flags)?;
    unsafe {
        stream
            .launch_builder(&compare)
            .arg(&mut arena)
            .arg(&previous)
            .arg(&(input.words as u32))
            .launch(grid(1))?;
    }
    stream.memcpy_dtoh(&arena, &mut pinned)?;
    let copied = stream.record_event(flags)?;
    copied.synchronize()?;
    let values = pinned.as_slice()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Nonfinite diagnostic result".into());
    }
    fs::write(
        out.join("result.bin"),
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let report = format!(
        concat!(
            "{{\"device\":\"{}\",\"patches\":{},\"systems\":{},\"output_words\":{},",
            "\"compile_ms\":{},\"graph_setup_ms\":{},\"mean_graph_ms\":{},\"geometry_profile_ms\":{},",
            "\"direct_profile_ms\":{},\"iterative_profile_ms\":{},\"proposal_programs\":{},\"proposal_rounds\":{},\"proposals_profile_ms\":{},\"profile_samples\":1,",
            "\"wall_ms_including_context_compile_io\":{},\"graph_launches\":{},",
            "\"htod_copies\":1,\"dtoh_copies\":1,\"intermediate_readbacks\":0,",
            "\"upload_bytes\":{},\"download_bytes\":{},\"replay_mismatched_words\":{}}}"
        ),
        ctx.name()?.replace('"', ""),
        input.patches,
        input.systems,
        input.words,
        compile_ms,
        graph_setup_ms,
        start.elapsed_ms(&ran)? / input.repeats as f32,
        ran.elapsed_ms(&geometry_done)?,
        geometry_done.elapsed_ms(&direct_done)?,
        direct_done.elapsed_ms(&iterative_done)?,
        input.programs,
        input.rounds,
        iterative_done.elapsed_ms(&proposals_done)?,
        wall.elapsed().as_secs_f64() * 1000.,
        input.repeats + 1,
        input.bytes.len(),
        input.words * 8,
        values[input.words - 1]
    );
    fs::write(out.join("gpu_report.json"), &report)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse;
    fn valid() -> Vec<u8> {
        let mut v = b"GPDIA001".to_vec();
        for x in [1_u32, 1, 1, 2, 1, 16] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.resize(32 + 8 * (48 + 49 + 2958 + 125), 0);
        v
    }
    #[test]
    fn validates_external_diagnostic_protocol() {
        assert!(parse(valid()).is_ok());
        assert!(parse(vec![]).is_err());
        let mut v = valid();
        v.pop();
        assert!(parse(v).is_err());
        let mut v = valid();
        v[12..16].copy_from_slice(&2049_u32.to_le_bytes());
        assert!(parse(v).is_err());
        let mut v = valid();
        v[20..24].copy_from_slice(&1_u32.to_le_bytes());
        assert!(parse(v).is_err());
        let mut v = valid();
        v[28] = 17;
        assert!(parse(v).is_err());
        let mut v = valid();
        v[32 + 48 * 8..40 + 48 * 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(parse(v).is_err());
        let mut v = valid();
        v[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(parse(v).is_ok());
    }
}
