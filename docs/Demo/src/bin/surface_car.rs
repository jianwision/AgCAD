#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
#[path = "../mesh.rs"]
mod mesh;

use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, result, sys::CUevent_flags},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CP: usize = 14 * 17 * 4;
const HEADER: usize = 64;

fn root() -> Result<PathBuf> {
    if PathBuf::from("cuda/car_surfaces.cu").is_file() {
        return Ok(PathBuf::from("."));
    }
    for dir in std::env::current_exe()?.ancestors().skip(1) {
        if dir.join("cuda/car_surfaces.cu").is_file() {
            return Ok(dir.to_owned());
        }
    }
    Err("Run from the demo directory, containing cuda/car_surfaces.cu".into())
}

fn launch(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(64), 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() -> Result<()> {
    let total = Instant::now();
    let mut out = PathBuf::from("output/car");
    let (mut steps, mut around) = (128_u32, 96_u32);
    let mut input = [0., 0., 0., 0.4, 1.];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--no-wheels" {
            input[4] = -1.;
            continue;
        }
        if arg == "--help" {
            println!(
                "surface_car [--out DIR] [--steps 32.0.512] [--around 32.0.256] [--length 3680.0.5520] [--width 1520.0.2280] [--height 1080.0.1620] [--style 0.0.1] [--no-wheels]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("Option requires a value")?;
        match arg.as_str() {
            "--out" => out = value.into(),
            "--steps" => steps = value.parse()?,
            "--around" => around = value.parse()?,
            "--length" => input[0] = (value.parse::<f64>()? / 4600. - 1.) / 0.2,
            "--width" => input[1] = (value.parse::<f64>()? / 1900. - 1.) / 0.2,
            "--height" => input[2] = (value.parse::<f64>()? / 1350. - 1.) / 0.2,
            "--style" => input[3] = 2. * value.parse::<f64>()? - 1.,
            _ => return Err(format!("Unknown option {arg}").into()),
        }
    }
    if !(32..=512).contains(&steps) || !(32..=256).contains(&around) {
        return Err("Sampling resolution out of range".into());
    }
    fs::create_dir_all(&out)?;
    let root = root()?;
    let weight_bytes = fs::read(root.join("models/geometry_policy.f64"))?;
    if weight_bytes.len() != 769 * 8 {
        return Err("Expected a trained 5->24->25 policy with 769 parameters".into());
    }
    let weights: Vec<f64> = weight_bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    fs::write(out.join("geometry_policy.f64"), &weight_bytes)?;
    fs::copy(
        root.join("models/training_report.json"),
        out.join("training_report.json"),
    )?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    let device = ctx.name()?;
    let stream = ctx.default_stream();
    let source = [
        "surface_primitives.cuh",
        "geometry_policy.cuh",
        "car_surfaces.cu",
    ]
    .iter()
    .map(|p| fs::read_to_string(root.join("cuda").join(p)))
    .collect::<std::io::Result<Vec<_>>>()?
    .join("\n");
    let t = Instant::now();
    let compiled = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("surface_car.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("NVRTC: {e:?}"))?;
    let compile_ms = t.elapsed().as_secs_f64() * 1000.;
    fs::write(out.join("surface_car.ptx"), compiled.to_src())?;
    fs::write(out.join("surface_car.cu"), &source)?;
    let module = ctx.load_module(compiled)?;
    let infer = module.load_function("infer_commands")?;
    let net = module.load_function("make_control_net")?;
    let evaluate = module.load_function("evaluate_car")?;
    let profiles = module.load_function("cad_profiles")?;
    let checks = module.load_function("primitive_checks")?;
    let triangles = (2 * steps * around + 2 * around + 4 * 64 * 24 * 2) as usize;
    let tri_offset = (HEADER + CP) as u32;
    let profiles_offset = tri_offset + (triangles * 9) as u32;
    let checks_offset = profiles_offset + 25 * 48 * 3;
    let packet_len = checks_offset as usize + 54;
    // Two initial uploads. All subsequent command and geometry communication is device-to-device.
    let weight_gpu = stream.clone_htod(&weights)?;
    let input_gpu = stream.clone_htod(&input)?;
    let mut packet = stream.alloc_zeros::<f64>(packet_len)?;
    // SAFETY: initialized by the full DtoH copy before any host read.
    let mut host = unsafe { ctx.alloc_pinned_with_flags::<f64>(packet_len, 0)? };
    let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    // SAFETY: argument order and buffer sizes match the bounded kernels.
    unsafe {
        stream
            .launch_builder(&infer)
            .arg(&weight_gpu)
            .arg(&input_gpu)
            .arg(&mut packet)
            .launch(launch(1))?;
    }
    let infer_end = stream.record_event(flags)?;
    unsafe {
        stream
            .launch_builder(&net)
            .arg(&mut packet)
            .launch(launch(14 * 17))?;
        stream
            .launch_builder(&evaluate)
            .arg(&mut packet)
            .arg(&steps)
            .arg(&around)
            .arg(&tri_offset)
            .launch(launch(steps * around + 2 * around + 4 * 64 * 24))?;
        stream
            .launch_builder(&profiles)
            .arg(&mut packet)
            .arg(&profiles_offset)
            .launch(launch(25 * 48))?;
        stream
            .launch_builder(&checks)
            .arg(&mut packet)
            .arg(&checks_offset)
            .launch(launch(1))?;
    }
    let geometry_end = stream.record_event(flags)?;
    stream.memcpy_dtoh(&packet, &mut host)?;
    let end = stream.record_event(flags)?;
    end.synchronize()?; // First host dependency after inference starts.
    let data = host.as_slice()?;
    if data[32] != 0. {
        return Err(format!(
            "GPU instruction validation rejected policy output, code {}",
            data[32]
        )
        .into());
    }
    let opcodes: Vec<u32> = (0..4).map(|i| data[8 * i] as u32).collect();
    let expected_opcodes = vec![1, 2, if input[4] > 0. { 3 } else { 0 }, 4];
    if opcodes != expected_opcodes {
        return Err(format!("Policy opcode mismatch: {opcodes:?}").into());
    }
    let check = &data[checks_offset as usize..];
    let q = 0.5_f64.sqrt();
    let expected = [
        [1.5, 1.5, 0.],
        [1., -2., 3.],
        [0., 1., 2.],
        [0.5, 1., 1.5],
        [3., 5., 3.],
        [0.3, 0.7, 0.],
        [0.25, 0.75, 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.],
        [0.25, 0.75, 2.],
        [0.4, 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 0., 0.],
        [0.4, q, q],
        [5., 9., 15.],
    ];
    let max_error = check
        .iter()
        .zip(expected.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    if check.iter().any(|x| !x.is_finite()) || !max_error.is_finite() || max_error > 1e-7 {
        return Err(format!(
            "GPU primitive / seam check failed: max absolute error {max_error}; values {check:?}"
        )
        .into());
    }
    let used_triangles = triangles - if data[16] == 0. { 4 * 64 * 24 * 2 } else { 0 };
    let mesh: Vec<f32> = data[tri_offset as usize..tri_offset as usize + used_triangles * 9]
        .iter()
        .map(|&a| a as f32)
        .collect();
    let stats = mesh::analyze(&mesh)?;
    if stats.boundary_edges
        + stats.nonmanifold_edges
        + stats.orientation_errors
        + stats.degenerate_triangles
        > 0
        || stats.volume <= 0.
        || stats.euler != 2
    {
        return Err(format!("GPU mesh validation failed: {stats:?}").into());
    }
    mesh::write_stl(&out.join("model.stl"), &mesh)?;
    mesh::write_viewer(&out.join("viewer.html"), &mesh, 2300.)?;
    let viewer = fs::read_to_string(out.join("viewer.html"))?
        .replace("让代码成为形状", "小模型驱动 GPU 建模")
        .replace("沿不同角度检查表面和贯通孔", "检查 NURBS 车身和旋转面轮胎")
        .replace(
            "在 GPU 上生成的三维网格",
            "769 参数策略网络 → 四个 GPU 原语",
        );
    fs::write(out.join("viewer.html"), viewer)?;
    let ku: Vec<f64> = (0..18)
        .map(|i| {
            if i < 4 {
                0.
            } else if i >= 14 {
                1.
            } else {
                (i - 3) as f64 / 11.
            }
        })
        .collect();
    let kv: Vec<f64> = (0..21).map(|i| i as f64).collect();
    let poles: Vec<Vec<f64>> = data[HEADER..HEADER + CP]
        .chunks_exact(4)
        .map(|p| p.to_vec())
        .collect();
    fs::write(
        out.join("car_surface.json"),
        format!(
            "{{\n\"units\":\"mm\",\"degree\":[3,3],\"size\":[14,17],\"domain\":[[0,1],[3,17]],\"u_knots\":{ku:?},\"v_knots\":{kv:?},\"poles_xyzw\":{poles:?},\"commands\":{:?},\"profiles_xyz\":{:?}\n}}",
            &data[..32],
            &data[profiles_offset as usize..checks_offset as usize]
        ),
    )?;
    // SAFETY: these events are alive, ordered in one stream and already complete.
    let elapsed = |a: &cudarc::driver::CudaEvent, b: &cudarc::driver::CudaEvent| -> Result<f64> {
        Ok(unsafe { result::event::elapsed(a.cu_event(), b.cu_event())? } as f64)
    };
    let infer_ms = elapsed(&start, &infer_end)?;
    let geometry_ms = elapsed(&infer_end, &geometry_end)?;
    let download_ms = elapsed(&geometry_end, &end)?;
    let seam_position = check[36..39].iter().copied().fold(0., f64::max);
    let seam_d1 = check[39..42].iter().copied().fold(0., f64::max);
    let seam_d2 = check[42..45].iter().copied().fold(0., f64::max);
    let report = format!(
        "{{\n\"device\":\"{device}\",\"model_parameters\":769,\"input\":{input:?},\"opcodes\":{opcodes:?},\"predicted_scales\":{:?},\"tyre_scale\":{},\"host_to_device_uploads\":2,\"intermediate_device_to_host_copies\":0,\"final_device_to_host_copies\":1,\"download_bytes\":{},\"compile_ms\":{compile_ms},\"inference_ms\":{infer_ms},\"geometry_ms\":{geometry_ms},\"download_ms\":{download_ms},\"triangles\":{},\"vertices\":{},\"boundary_edges\":{},\"nonmanifold_edges\":{},\"orientation_errors\":{},\"degenerate_triangles\":{},\"euler\":{},\"signed_volume_mm3\":{},\"extents_mm\":{:?},\"primitive_max_abs_error\":{max_error},\"periodic_seam_samples\":33,\"seam_position_error_mm\":{seam_position},\"seam_d1_error\":{seam_d1},\"seam_d2_error\":{seam_d2},\"pipeline\":\"host-enqueued fixed kernels, GPU-resident neural commands gate geometry, one final readback\"\n}}",
        &data[1..5],
        data[17],
        packet_len * 8,
        stats.triangles,
        stats.vertices,
        stats.boundary_edges,
        stats.nonmanifold_edges,
        stats.orientation_errors,
        stats.degenerate_triangles,
        stats.euler,
        stats.volume,
        stats.extents
    );
    fs::write(out.join("report.json"), &report)?;
    println!(
        "{report}\nTotal {:0.1} ms; output {}",
        total.elapsed().as_secs_f64() * 1000.,
        out.display()
    );
    Ok(())
}
