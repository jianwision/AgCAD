//! Persistent CUDA worker. Binary IPC keeps geometry on device across decisions.
#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{
    error::Error,
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TEMPLATE_WORDS: usize = 2958;
const MAX_STEPS: usize = 64;
const SLOT_WORDS: usize = 448;
const FEEDBACK_WORDS: usize = 64 + 26 * MAX_STEPS + 192;
fn word(v: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(v[offset..offset + 4].try_into().unwrap())
}
fn double(v: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(v[offset..offset + 8].try_into().unwrap())
}
fn scene_dimensions(v: &[u8]) -> Result<(usize, u32)> {
    if v.len() != 24 + 8 * TEMPLATE_WORDS
        || &v[..8] != b"GPCAD001"
        || word(v, 8) != 1
        || word(v, 20) != 0
    {
        return Err("Invalid persistent scene ABI".into());
    }
    let slots = word(v, 12) as usize;
    let budget = word(v, 16);
    if !(1..=64).contains(&slots) || !(1..=4096).contains(&budget) {
        return Err("Invalid scene capacity/budget".into());
    }
    if (24..v.len()).step_by(8).any(|i| !double(v, i).is_finite()) {
        return Err("Nonfinite scene template".into());
    }
    // Check the physical-coordinate right inverse; geometry is additionally
    // checked on device when creating/committing each object.
    for i in 0..6 {
        for j in 0..6 {
            let value = (0..192)
                .map(|k| {
                    double(v, 24 + 8 * (192 + 1152 + 192 * i + k))
                        * double(v, 24 + 8 * (192 + 6 * k + j))
                })
                .sum::<f64>();
            if (value - if i == j { 1. } else { 0. }).abs() > 1e-8 {
                return Err("Invalid physical probe map".into());
            }
        }
    }
    Ok((slots, budget))
}
fn validate_header(v: &[u8]) -> bool {
    v.len() == 160
        && &v[..8] == b"CADREQ01"
        && word(v, 8) == 1
        && word(v, 12) <= 7
        && word(v, 36) == 1
        && word(v, 40) <= 1
        && word(v, 44) == 0
        && v[144..].iter().all(|b| *b == 0)
        && (48..144).step_by(8).all(|i| double(v, i).is_finite())
}
struct Reply<'a> {
    status: u32,
    slot: u32,
    generation: u32,
    revision: u32,
    remaining: u32,
    body: &'a [f64],
}
fn reply(output: &mut impl Write, r: Reply<'_>) -> Result<()> {
    output.write_all(b"CADRSP01")?;
    for n in [
        r.status,
        r.slot,
        r.generation,
        r.revision,
        r.remaining,
        r.body.len() as u32,
    ] {
        output.write_all(&n.to_le_bytes())?;
    }
    for value in r.body {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
fn main() -> Result<()> {
    let wall = Instant::now();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or("Use cad_worker SESSION_DIRECTORY")?);
    if args.next().is_some() {
        return Err("Unexpected worker argument".into());
    }
    let template = fs::read(root.join("template.bin"))?;
    let (slots, budget) = scene_dimensions(&template)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: every operation is ordered on this one stream; IPC is serialized.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let compile = Instant::now();
    let source = fs::read_to_string("cuda/hood_contract.cuh")?
        + "\n"
        + &fs::read_to_string("cuda/cad_session.cu")?;
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("cad_session.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Session NVRTC: {e:?}"))?;
    let compile_ms = compile.elapsed().as_secs_f64() * 1000.;
    fs::write(root.join("worker-source.cu"), source)?;
    fs::write(root.join("worker.ptx"), ptx.to_src())?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("session_execute")?;
    let scene = stream.clone_htod(&template)?;
    let mut arena = stream.alloc_zeros::<f64>(slots * SLOT_WORDS)?;
    let mut command = stream.alloc_zeros::<u8>(160 + 64 * MAX_STEPS)?;
    let mut feedback = stream.alloc_zeros::<f64>(FEEDBACK_WORDS)?;
    let live_bytes =
        template.len() + slots * SLOT_WORDS * 8 + 160 + 64 * MAX_STEPS + FEEDBACK_WORDS * 8;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut generations = vec![0_u32; slots];
    let mut revisions = vec![0_u32; slots];
    let mut occupied = vec![false; slots];
    let mut draft = vec![false; slots];
    let mut reserved = 0_u32;
    let mut messages = 0;
    let mut rejections = 0;
    let mut calls = 0;
    let mut download_bytes = 0;
    let mut upload_bytes = template.len();
    let mut full_downloads = 0;
    let mut peak = 0;
    let mut gpu_ms = 0_f64;
    reply(
        &mut output,
        Reply {
            status: 100,
            slot: u32::MAX,
            generation: 0,
            revision: 0,
            remaining: budget,
            body: &[
                1.,
                slots as f64,
                budget as f64,
                MAX_STEPS as f64,
                live_bytes as f64,
                compile_ms,
            ],
        },
    )?;
    loop {
        let mut header = [0_u8; 160];
        match input.read_exact(&mut header) {
            Ok(()) => (),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        messages += 1;
        let op = word(&header, 12);
        let slot = word(&header, 16) as usize;
        let other = word(&header, 20) as usize;
        let expected_revision = word(&header, 24);
        let generation = word(&header, 28);
        let steps = word(&header, 32) as usize;
        let mut packet = header.to_vec();
        if op == 1 {
            if steps == 0 || steps > MAX_STEPS {
                return Err("Unrecoverable action frame length".into());
            }
            packet.resize(160 + 64 * steps, 0);
            input.read_exact(&mut packet[160..])?;
        }
        let error = if !validate_header(&header)
            || slot >= slots
            || (op == 3 && other >= slots)
            || ((op == 1 || op == 2) && (steps == 0 || steps > MAX_STEPS))
            || ((op != 1 && op != 2) && steps != 0)
        {
            200
        } else if op == 7 {
            0
        } else if messages > 1024 {
            207
        } else if op == 0 {
            if occupied[slot] {
                204
            } else if (48..144).step_by(8).any(|i| double(&header, i).abs() > 1e4) {
                200
            } else {
                0
            }
        } else if op == 3 {
            if occupied[slot] || slot == other {
                204
            } else if !occupied[other] || generations[other] != generation {
                201
            } else if revisions[other] != expected_revision {
                202
            } else {
                0
            }
        } else if !occupied[slot] || generations[slot] != generation {
            201
        } else if revisions[slot] != expected_revision {
            202
        } else if op == 5 && word(&header, 40) == 1 && !draft[slot] {
            205
        } else if (op == 1 || op == 2) && reserved + steps as u32 > budget {
            203
        } else {
            0
        };
        if error != 0 {
            rejections += 1;
            reply(
                &mut output,
                Reply {
                    status: error,
                    slot: slot as u32,
                    generation: 0,
                    revision: 0,
                    remaining: budget - reserved,
                    body: &[],
                },
            )?;
            continue;
        }
        if op == 7 {
            reply(
                &mut output,
                Reply {
                    status: 0,
                    slot: slot as u32,
                    generation: 0,
                    revision: 0,
                    remaining: budget - reserved,
                    body: &[],
                },
            )?;
            break;
        }
        if op == 0 || op == 3 {
            generations[slot] += 1;
            packet[28..32].copy_from_slice(&generations[slot].to_le_bytes());
        }
        if op == 1 || op == 2 {
            reserved += steps as u32;
        }
        if op == 1 {
            for action in packet[160..].chunks_exact(64) {
                if word(action, 56) != 0 || word(action, 60) != 0 {
                    return Err("Invalid action reserved fields".into());
                }
            }
        }
        stream.memcpy_htod(&packet, &mut command.slice_mut(..packet.len()))?;
        stream.memset_zeros(&mut feedback)?;
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        // SAFETY: headers, dimensions, slots and buffer lengths were checked;
        // all allocations and module functions remain alive for the worker.
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&command)
                .arg(&scene)
                .arg(&mut arena)
                .arg(&mut feedback)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let words = 64
            + if op == 1 || op == 2 {
                26 * steps
            } else if op == 5 {
                192
            } else {
                0
            };
        let values = stream.clone_dtoh(&feedback.slice(..words))?;
        stream.synchronize()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite session feedback".into());
        }
        gpu_ms += f64::from(start.elapsed_ms(&end)?);
        calls += 1;
        upload_bytes += packet.len();
        download_bytes += 8 * words;
        if op == 5 {
            full_downloads += 1;
        }
        occupied[slot] = values[29] == 1.;
        revisions[slot] = values[18] as u32;
        draft[slot] = values[20] != 0.;
        peak = peak.max(occupied.iter().filter(|v| **v).count());
        reply(
            &mut output,
            Reply {
                status: 0,
                slot: slot as u32,
                generation: generations[slot],
                revision: revisions[slot],
                remaining: budget - reserved,
                body: &values,
            },
        )?;
    }
    let report = format!(
        concat!(
            "{{\"context_creations\":1,\"template_uploads\":1,\"command_htod_copies\":{},",
            "\"feedback_dtoh_copies\":{},\"geometry_downloads\":{},\"kernel_launches\":{},\"upload_bytes\":{},",
            "\"download_bytes\":{},\"requested_device_bytes_excludes_context\":{},\"budget_reserved\":{},",
            "\"budget_limit\":{},\"protocol_rejections\":{},\"messages\":{},\"peak_live_objects\":{},",
            "\"compile_ms\":{},\"kernel_ms_total\":{},\"wall_ms\":{}}}"
        ),
        calls,
        calls,
        full_downloads,
        calls,
        upload_bytes,
        download_bytes,
        live_bytes,
        reserved,
        budget,
        rejections,
        messages,
        peak,
        compile_ms,
        gpu_ms,
        wall.elapsed().as_secs_f64() * 1000.
    );
    fs::write(root.join("worker-report.json"), report)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{scene_dimensions, validate_header};
    #[test]
    fn rejects_untrusted_session_frames() {
        assert!(!validate_header(&[]));
        assert!(scene_dimensions(&[]).is_err());
        let mut v = vec![0; 160];
        v[..8].copy_from_slice(b"CADREQ01");
        v[8..12].copy_from_slice(&1_u32.to_le_bytes());
        v[36..40].copy_from_slice(&1_u32.to_le_bytes());
        assert!(validate_header(&v));
        v[40] = 2;
        assert!(!validate_header(&v));
        v[40] = 0;
        v[48..56].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(!validate_header(&v));
    }
}
