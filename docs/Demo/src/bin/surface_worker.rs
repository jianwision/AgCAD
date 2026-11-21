//! Persistent surface arena, cached basis tables and CUDA graph execution.
#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
use cudarc::{
    driver::{CudaContext, CudaGraph, CudaSlice, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts},
};
use std::{
    collections::HashMap,
    error::Error,
    fs,
    hash::{Hash, Hasher},
    io::{self, Read, Write},
    path::PathBuf,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SLOTS: usize = 128;
const SLOT: usize = 4096;
const MAX_INPUT: usize = 1_048_576;
const MAX_POINTS: usize = 4225;
#[derive(Clone, Copy)]
struct Surface {
    nu: usize,
    nv: usize,
    generation: u32,
    uniform_weights: bool,
}
struct Basis {
    nu: usize,
    nv: usize,
    p: usize,
    q: usize,
    count: usize,
    host: Vec<f64>,
    device: CudaSlice<f64>,
}
fn integer(x: f64, lo: usize, hi: usize) -> Result<usize> {
    if !x.is_finite() || x.fract() != 0. || x < (lo as f64) || x > (hi as f64) {
        return Err("Invalid dimension/index".into());
    }
    Ok(x as usize)
}
fn u32_at(b: &[u8], k: usize) -> u32 {
    u32::from_le_bytes(b[k..k + 4].try_into().unwrap())
}
fn send(
    output: &mut impl Write,
    status: u32,
    slot: usize,
    generation: u32,
    body: &[f64],
) -> Result<()> {
    output.write_all(b"SURFRSP1")?;
    for v in [status, slot as u32, generation, 0, body.len() as u32, 0] {
        output.write_all(&v.to_le_bytes())?;
    }
    for v in body {
        output.write_all(&v.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
fn surface(
    slots: &[Option<Surface>],
    index: usize,
    generation: u32,
    revision: u32,
) -> Result<Surface> {
    let s = slots
        .get(index)
        .and_then(|s| *s)
        .ok_or("Unknown surface handle")?;
    if s.generation != generation || revision != 0 {
        return Err("Stale surface handle".into());
    }
    Ok(s)
}
fn validate_basis(v: &[f64]) -> Result<(usize, usize, usize, usize, usize)> {
    if v.len() < 5 {
        return Err("Missing basis dimensions".into());
    }
    let nu = integer(v[0], 2, 32)?;
    let nv = integer(v[1], 2, 32)?;
    let p = integer(v[2], 1, 7)?;
    let q = integer(v[3], 1, 7)?;
    let count = integer(v[4], 1, MAX_POINTS)?;
    if nu <= p || nv <= q || v.len() != 5 + 67 * count {
        return Err("Basis dimensions/length mismatch".into());
    }
    for row in v[5..].chunks_exact(67) {
        integer(row[0], 0, nu - p - 1)?;
        integer(row[1], 0, nv - q - 1)?;
        if row[2..66].iter().any(|x| x.abs() > 1e12) || row[66] < 0. || row[66] > 1. {
            return Err("Invalid basis/weight".into());
        }
    }
    Ok((nu, nv, p, q, count))
}
fn main() -> Result<()> {
    let wall = Instant::now();
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("Use surface_worker OUTPUT_DIRECTORY")?,
    );
    fs::create_dir_all(&root)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: one serialized stream; no resources are shared with other contexts.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let source = fs::read_to_string("cuda/resident_surface.cu")?;
    let capability = ctx.compute_capability()?;
    let options = format!(
        "--fmad=false;compute_{}{};cudarc-0.19.10;CUDA-12.6",
        capability.0, capability.1
    );
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    options.hash(&mut hasher);
    let cache = PathBuf::from("output/gpu-cache");
    fs::create_dir_all(&cache)?;
    let prefix = cache.join(format!("surface-{:016x}", hasher.finish()));
    let signature = format!("{options}\n{source}");
    let cache_hit = prefix.with_extension("key").exists()
        && prefix.with_extension("ptx").exists()
        && fs::read_to_string(prefix.with_extension("key"))? == signature;
    let compile = Instant::now();
    let ptx = if cache_hit {
        Ptx::from_file(prefix.with_extension("ptx"))
    } else {
        let ptx = compile_ptx_with_opts(
            &source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!(
                    "--gpu-architecture=compute_{}{}",
                    capability.0, capability.1
                )],
                name: Some("resident_surface.cu".into()),
                ..Default::default()
            },
        )?;
        fs::write(prefix.with_extension("ptx"), ptx.to_src())?;
        fs::write(prefix.with_extension("key"), &signature)?;
        ptx
    };
    let compile_ms = compile.elapsed().as_secs_f64() * 1000.;
    fs::write(root.join("worker-source.cu"), &source)?;
    fs::write(root.join("worker.ptx"), ptx.to_src())?;
    let module = ctx.load_module(ptx)?;
    let eval = module.load_function("surface_eval")?;
    let summary = module.load_function("surface_summary")?;
    let clone_kernel = module.load_function("surface_clone")?;
    let refine = module.load_function("surface_refine")?;
    let gather = module.load_function("network_gather")?;
    let project = module.load_function("network_project")?;
    let prepare = module.load_function("network_prepare")?;
    let neteval = module.load_function("network_eval")?;
    let score = module.load_function("network_score")?;
    let choose = module.load_function("network_choose")?;
    let publish = module.load_function("network_publish")?;
    // Stable allocation addresses let cached executable graphs survive requests.
    let mut arena = stream.alloc_zeros::<f64>(SLOTS * SLOT)?;
    let mut command = stream.alloc_zeros::<f64>(MAX_INPUT)?;
    let mut grid = stream.alloc_zeros::<f64>(MAX_POINTS * 48)?;
    let mut stats_buffer = stream.alloc_zeros::<f64>(8)?;
    let mut seed = stream.alloc_zeros::<f64>(384 * 4)?;
    let mut current = stream.alloc_zeros::<f64>(384 * 4)?;
    let mut candidates = stream.alloc_zeros::<f64>(384 * 4 * 3)?;
    let mut reference = stream.alloc_zeros::<f64>(6 * 625 * 48)?;
    let mut candidate_grid = stream.alloc_zeros::<f64>(3 * 6 * 625 * 48)?;
    let mut scores = stream.alloc_zeros::<f64>(24)?;
    let mut state = stream.alloc_zeros::<f64>(18)?;
    let mut slots = vec![None; SLOTS];
    let mut generations = [0_u32; SLOTS];
    let mut bases: Vec<Basis> = Vec::new();
    let mut graphs: HashMap<(usize, usize, usize), CudaGraph> = HashMap::new();
    let mut requests = 0_usize;
    let mut rejects = 0_usize;
    let mut uploads = 0_usize;
    let mut downloads = 0_usize;
    let mut summary_reads = 0_usize;
    let mut grid_reads = 0_usize;
    let mut geometry_reads = 0_usize;
    let mut basis_hits = 0_usize;
    let mut graph_builds = 0_usize;
    let mut graph_hits = 0_usize;
    let mut graph_launches = 0_usize;
    let mut kernels = 0_usize;
    let mut network_runs = 0_usize;
    let mut basis_bytes = 0_usize;
    let mut network_tail: Vec<f64> = Vec::new();
    let mut network_prefix = 0;
    let mut plan_uploads = 0;
    let mut plan_hits = 0;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    send(
        &mut output,
        100,
        usize::MAX,
        0,
        &[
            1.,
            f64::from(!cache_hit),
            f64::from(cache_hit),
            compile_ms,
            SLOTS as f64,
        ],
    )?;
    macro_rules! launch { ($function:expr,$n:expr,$($arg:expr),+ $(,)?) => {{
        // SAFETY: host validates all lengths/indices and owns all stable buffers.
        unsafe { stream.launch_builder($function)$(.arg($arg))+.launch(LaunchConfig{grid_dim:((($n) as u32).div_ceil(64),1,1),block_dim:(64,1,1),shared_mem_bytes:0}) }?; kernels+=1;
    }}; }
    loop {
        let mut header = [0_u8; 32];
        match input.read_exact(&mut header) {
            Ok(()) => (),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if &header[..8] != b"SURFREQ1" {
            return Err("Bad surface frame magic".into());
        }
        let op = u32_at(&header, 8);
        let slot = u32_at(&header, 12) as usize;
        let generation = u32_at(&header, 16);
        let revision = u32_at(&header, 20);
        let length = u32_at(&header, 24) as usize;
        if length > MAX_INPUT || u32_at(&header, 28) != 0 {
            return Err("Oversized/unsupported surface frame".into());
        }
        let mut bytes = vec![0_u8; length * 8];
        input.read_exact(&mut bytes)?;
        let mut v: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        if op == 0 {
            send(&mut output, 0, slot, 0, &[])?;
            break;
        }
        requests += 1;
        let result = (|| -> Result<(usize, u32, Vec<f64>)> {
            if requests > 4096 || v.iter().any(|x| !x.is_finite()) {
                return Err("Request budget/nonfinite payload".into());
            }
            if op == 1 {
                if v.len() < 2 {
                    return Err("Missing control dimensions".into());
                }
                let nu = integer(v[0], 2, 32)?;
                let nv = integer(v[1], 2, 32)?;
                if v.len() != 2 + 4 * nu * nv
                    || v[2..]
                        .chunks_exact(4)
                        .any(|p| p[..3].iter().any(|x| x.abs() > 1e6) || p[3] < 1e-4 || p[3] > 1e4)
                {
                    return Err("Invalid control net/weights".into());
                }
                let dest = slots
                    .iter()
                    .position(Option::is_none)
                    .ok_or("Surface arena full")?;
                stream.memcpy_htod(
                    &v[2..],
                    &mut arena.slice_mut(dest * SLOT..dest * SLOT + 4 * nu * nv),
                )?;
                uploads += 4 * nu * nv * 8;
                let uniform_weights = v[2..].chunks_exact(4).all(|p| p[3] == v[5]);
                generations[dest] += 1;
                slots[dest] = Some(Surface {
                    nu,
                    nv,
                    generation: generations[dest],
                    uniform_weights,
                });
                return Ok((dest, generations[dest], vec![nu as f64, nv as f64]));
            }
            if op == 2 {
                let (nu, nv, p, q, count) = validate_basis(&v)?;
                if let Some(id) = bases.iter().position(|b| b.host == v) {
                    basis_hits += 1;
                    return Ok((usize::MAX, 0, vec![id as f64]));
                }
                if bases.len() >= 64 || basis_bytes + v.len() * 8 > 64 * 1024 * 1024 {
                    return Err("Basis cache capacity exhausted".into());
                }
                let device = stream.clone_htod(&v[5..])?;
                uploads += (v.len() - 5) * 8;
                basis_bytes += (v.len() - 5) * 8;
                bases.push(Basis {
                    nu,
                    nv,
                    p,
                    q,
                    count,
                    host: v,
                    device,
                });
                return Ok((usize::MAX, 0, vec![(bases.len() - 1) as f64]));
            }
            if op == 9 {
                return Ok((
                    usize::MAX,
                    0,
                    vec![
                        1.,
                        f64::from(!cache_hit),
                        f64::from(cache_hit),
                        compile_ms,
                        requests as f64,
                        rejects as f64,
                        uploads as f64,
                        downloads as f64,
                        summary_reads as f64,
                        grid_reads as f64,
                        geometry_reads as f64,
                        bases.len() as f64,
                        basis_hits as f64,
                        graph_builds as f64,
                        graph_hits as f64,
                        graph_launches as f64,
                        kernels as f64,
                        network_runs as f64,
                        basis_bytes as f64,
                        plan_uploads as f64,
                        plan_hits as f64,
                    ],
                ));
            }
            if op == 8 {
                if v.len() < 16 {
                    return Err("Missing network header".into());
                }
                let np = integer(v[0], 2, 6)?;
                let n = integer(v[1], 2, 384)?;
                let df = integer(v[2], 0, n)?;
                let m = integer(v[3], 9, 625)?;
                let steps = integer(v[4], 0, 512)?;
                let compatibility = integer(v[11], 0, n)?;
                let base = 16 + 8 * np + n * n + n * df + n * compatibility;
                if v.len() != base + np * m * 67
                    || v[5] <= 0.
                    || v[5] > 10.
                    || v[6] <= 0.
                    || v[6] > 100.
                    || v[7] <= 0.
                    || v[7] > 1e6
                    || v[8..11].iter().any(|x| *x < 0. || *x > 100.)
                    || v[12..16].iter().any(|x| *x != 0.)
                {
                    return Err("Invalid network shape/budget".into());
                }
                if v[16 + 8 * np..base].iter().any(|x| x.abs() > 1e4) {
                    return Err("Invalid constraint projection".into());
                }
                let free: Vec<usize> = slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.is_none().then_some(i))
                    .take(np)
                    .collect();
                if free.len() != np {
                    return Err("Not enough result slots".into());
                }
                let mut count = 0;
                let mut outputs = Vec::new();
                for (i, &dest) in free.iter().enumerate() {
                    let at = 16 + 8 * i;
                    let src = integer(v[at], 0, SLOTS - 1)?;
                    let source_generation = integer(v[at + 1], 1, u32::MAX as usize)? as u32;
                    let s = surface(&slots, src, source_generation, 0)?;
                    if integer(v[at + 2], 0, n)? != count
                        || integer(v[at + 3], 2, 32)? != s.nu
                        || integer(v[at + 4], 2, 32)? != s.nv
                        || s.nu * s.nv > 64
                    {
                        return Err("Network control layout mismatch".into());
                    }
                    let p = integer(v[at + 5], 1, 7)?;
                    let q = integer(v[at + 6], 1, 7)?;
                    if s.nu != p + 1 || s.nv != q + 1 || v[at + 7] != 0. || !s.uniform_weights {
                        return Err("Network supports polynomial Bezier layouts".into());
                    }
                    let mut b = vec![s.nu as f64, s.nv as f64, p as f64, q as f64, m as f64];
                    b.extend_from_slice(&v[base + i * m * 67..base + (i + 1) * m * 67]);
                    validate_basis(&b)?;
                    v[at + 1] = dest as f64;
                    count += s.nu * s.nv;
                    outputs.push((dest, s));
                }
                if count != n {
                    return Err("Network point count mismatch".into());
                }
                let prefix = 16 + 8 * np;
                if network_prefix == prefix && network_tail == v[prefix..] {
                    stream.memcpy_htod(&v[..prefix], &mut command.slice_mut(..prefix))?;
                    uploads += prefix * 8;
                    plan_hits += 1;
                } else {
                    stream.memcpy_htod(&v, &mut command.slice_mut(..v.len()))?;
                    uploads += v.len() * 8;
                    network_prefix = prefix;
                    network_tail = v[prefix..].to_vec();
                    plan_uploads += 1;
                }
                stream.memcpy_htod(&[0_f64; 18], &mut state)?;
                uploads += 18 * 8;
                launch!(&gather, n, &arena, &command, &mut seed);
                launch!(&project, n, &command, &seed, &mut current);
                launch!(&neteval, np * m, &command, &seed, &1_i32, &mut reference);
                launch!(&prepare, n, &command, &current, &state, &mut candidates);
                launch!(
                    &neteval,
                    3 * np * m,
                    &command,
                    &candidates,
                    &3_i32,
                    &mut candidate_grid
                );
                // score needs exactly three blocks of 64 threads.
                unsafe {
                    stream
                        .launch_builder(&score)
                        .arg(&command)
                        .arg(&seed)
                        .arg(&reference)
                        .arg(&candidates)
                        .arg(&candidate_grid)
                        .arg(&mut scores)
                        .launch(LaunchConfig {
                            grid_dim: (3, 1, 1),
                            block_dim: (64, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                kernels += 1;
                launch!(
                    &choose,
                    4 * n,
                    &command,
                    &candidates,
                    &scores,
                    &mut current,
                    &mut state,
                    &1_i32
                );
                let key = (np, n, m);
                if let std::collections::hash_map::Entry::Vacant(entry) = graphs.entry(key) {
                    if graph_builds >= 32 {
                        return Err("Graph cache capacity exhausted".into());
                    }
                    stream.synchronize()?;
                    stream.begin_capture(
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    launch!(&prepare, n, &command, &current, &state, &mut candidates);
                    launch!(
                        &neteval,
                        3 * np * m,
                        &command,
                        &candidates,
                        &3_i32,
                        &mut candidate_grid
                    );
                    unsafe {
                        stream
                            .launch_builder(&score)
                            .arg(&command)
                            .arg(&seed)
                            .arg(&reference)
                            .arg(&candidates)
                            .arg(&candidate_grid)
                            .arg(&mut scores)
                            .launch(LaunchConfig {
                                grid_dim: (3, 1, 1),
                                block_dim: (64, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    launch!(
                        &choose,
                        4 * n,
                        &command,
                        &candidates,
                        &scores,
                        &mut current,
                        &mut state,
                        &0_i32
                    );
                    let graph = stream
                        .end_capture(sys::CUgraphInstantiate_flags(0))?
                        .ok_or("Empty graph")?;
                    entry.insert(graph);
                    graph_builds += 1;
                } else {
                    graph_hits += 1;
                }
                for _ in 0..steps {
                    graphs[&key].launch()?;
                    graph_launches += 1;
                }
                let mut feedback = stream.clone_dtoh(&state)?;
                stream.synchronize()?;
                downloads += 18 * 8;
                summary_reads += 1;
                network_runs += 1;
                if feedback.iter().any(|x| !x.is_finite()) || feedback[10] >= 1e299 {
                    return Err(
                        "Network projection/search violates seam/lock compatibility, regularity or displacement budget"
                            .into(),
                    );
                }
                launch!(&publish, n, &mut arena, &command, &current);
                stream.synchronize()?;
                for (dest, s) in outputs {
                    generations[dest] += 1;
                    slots[dest] = Some(Surface {
                        generation: generations[dest],
                        ..s
                    });
                    feedback.extend([dest as f64, generations[dest] as f64]);
                }
                return Ok((usize::MAX, 0, feedback));
            }
            let s = surface(&slots, slot, generation, revision)?;
            match op {
                10 => {
                    if v.len() < 4 {
                        return Err("Missing refinement layout".into());
                    }
                    let nu = integer(v[0], 2, 32)?;
                    let nv = integer(v[1], 2, 32)?;
                    if v[2] != s.nu as f64
                        || v[3] != s.nv as f64
                        || v.len() != 4 + nu * nv * s.nu * s.nv
                    {
                        return Err("Refinement layout mismatch".into());
                    }
                    for row in v[4..].chunks_exact(s.nu * s.nv) {
                        if row.iter().any(|x| *x < 0. || *x > 1.)
                            || (row.iter().sum::<f64>() - 1.).abs() > 1e-10
                        {
                            return Err("Invalid homogeneous refinement map".into());
                        }
                    }
                    let dest = slots
                        .iter()
                        .position(Option::is_none)
                        .ok_or("Surface arena full")?;
                    stream.memcpy_htod(&v, &mut command.slice_mut(..v.len()))?;
                    uploads += v.len() * 8;
                    network_tail.clear();
                    network_prefix = 0;
                    launch!(
                        &refine,
                        nu * nv,
                        &mut arena,
                        &(slot as i32),
                        &(dest as i32),
                        &command
                    );
                    generations[dest] += 1;
                    slots[dest] = Some(Surface {
                        nu,
                        nv,
                        generation: generations[dest],
                        uniform_weights: s.uniform_weights,
                    });
                    Ok((dest, generations[dest], vec![]))
                }
                3 | 4 => {
                    if v.len() != 1 {
                        return Err("Expected one basis handle".into());
                    }
                    let id = integer(v[0], 0, bases.len().saturating_sub(1))?;
                    let b = bases.get(id).ok_or("Unknown basis handle")?;
                    if (s.nu, s.nv) != (b.nu, b.nv) {
                        return Err("Basis does not match surface dimensions".into());
                    }
                    launch!(
                        &eval,
                        b.count,
                        &arena,
                        &(slot as i32),
                        &(s.nv as i32),
                        &(b.p as i32),
                        &(b.q as i32),
                        &b.device,
                        &(b.count as i32),
                        &mut grid
                    );
                    let body = if op == 3 {
                        launch!(&summary, 1, &grid, &(b.count as i32), &mut stats_buffer);
                        summary_reads += 1;
                        stream.clone_dtoh(&stats_buffer)?
                    } else {
                        grid_reads += 1;
                        stream.clone_dtoh(&grid.slice(..48 * b.count))?
                    };
                    stream.synchronize()?;
                    downloads += body.len() * 8;
                    if body.iter().any(|x| !x.is_finite()) {
                        return Err("Nonfinite surface result".into());
                    }
                    Ok((slot, generation, body))
                }
                5 => {
                    if !v.is_empty() {
                        return Err("Unexpected fetch payload".into());
                    }
                    let body = stream
                        .clone_dtoh(&arena.slice(slot * SLOT..slot * SLOT + 4 * s.nu * s.nv))?;
                    stream.synchronize()?;
                    geometry_reads += 1;
                    downloads += body.len() * 8;
                    Ok((slot, generation, body))
                }
                6 => {
                    if !v.is_empty() {
                        return Err("Unexpected fork payload".into());
                    }
                    let dest = slots
                        .iter()
                        .position(Option::is_none)
                        .ok_or("Surface arena full")?;
                    launch!(
                        &clone_kernel,
                        4 * s.nu * s.nv,
                        &mut arena,
                        &(slot as i32),
                        &(dest as i32),
                        &((4 * s.nu * s.nv) as i32)
                    );
                    generations[dest] += 1;
                    slots[dest] = Some(Surface {
                        generation: generations[dest],
                        ..s
                    });
                    Ok((dest, generations[dest], vec![]))
                }
                7 => {
                    if !v.is_empty() {
                        return Err("Unexpected drop payload".into());
                    }
                    slots[slot] = None;
                    Ok((slot, generation, vec![]))
                }
                _ => Err("Unsupported surface operation".into()),
            }
        })();
        match result {
            Ok((s, g, body)) => send(&mut output, 0, s, g, &body)?,
            Err(error) => {
                rejects += 1;
                let message = error.to_string();
                fs::write(root.join("last-rejection.txt"), &message)?;
                send(
                    &mut output,
                    1,
                    slot,
                    generation,
                    &message.bytes().map(f64::from).collect::<Vec<_>>(),
                )?;
            }
        }
    }
    stream.synchronize()?;
    fs::write(
        root.join("worker-report.json"),
        format!(
            "{{\"context_creations\":1,\"compilations\":{},\"compile_cache_hit\":{},\"compile_ms\":{compile_ms},\"requests\":{requests},\"rejections\":{rejects},\"upload_bytes\":{uploads},\"download_bytes\":{downloads},\"summary_readbacks\":{summary_reads},\"grid_readbacks\":{grid_reads},\"geometry_readbacks\":{geometry_reads},\"basis_entries\":{},\"basis_cache_hits\":{basis_hits},\"basis_bytes\":{basis_bytes},\"graph_builds\":{graph_builds},\"graph_cache_hits\":{graph_hits},\"graph_launches\":{graph_launches},\"explicit_kernel_submissions\":{kernels},\"network_runs\":{network_runs},\"optimization_intermediate_readbacks\":0,\"wall_ms\":{}}}",
            usize::from(!cache_hit),
            cache_hit,
            bases.len(),
            wall.elapsed().as_secs_f64() * 1000.
        ),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_dimensions_and_stale_handles() {
        assert!(integer(f64::NAN, 1, 32).is_err());
        assert!(integer(1.5, 1, 32).is_err());
        assert!(validate_basis(&[4., 4., 3., 3., 1.]).is_err());
        assert!(
            surface(
                &[Some(Surface {
                    nu: 4,
                    nv: 4,
                    generation: 2,
                    uniform_weights: true
                })],
                0,
                1,
                0
            )
            .is_err()
        );
    }
}
