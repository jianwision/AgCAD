#[path = "../advanced_output.rs"]
mod advanced_output;
#[cfg(windows)]
#[path = "../cuda_loader.rs"]
mod cuda_loader;
#[path = "../extension_output.rs"]
mod extension_output;
#[path = "../mesh.rs"]
mod mesh;
#[path = "../operation_output.rs"]
mod operation_output;
#[path = "../topology_output.rs"]
mod topology_output;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, result, sys::CUevent_flags},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SURF_COUNT: usize = 14;
const SURF_WORDS: usize = 4192;
const MV: usize = 4096;
const MT: usize = 4096;
const MESH_WORDS: usize = 8 + 6 * MV + 3 * MT;
const MESH_COUNT: usize = 16;
fn launch(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(32), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn main() -> Result<()> {
    let mut out = PathBuf::from("output/primitives");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a != "--out" {
            return Err("Use primitive_suite [--out DIR] from the demo directory".into());
        }
        out = args.next().ok_or("Missing output directory")?.into();
    }
    fs::create_dir_all(&out)?;
    #[cfg(windows)]
    let _nvrtc = cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let device = ctx.name()?;
    let source = [
        "surface_primitives.cuh",
        "curve_ops.cuh",
        "surface_records.cuh",
        "surface_builders.cuh",
        "surface_suite.cu",
        "polygon_ops.cuh",
        "extended_curves.cuh",
        "extended_solids.cuh",
        "extension_suite.cu",
        "modeling_operations.cuh",
        "operation_suite.cu",
        "topology_operations.cuh",
        "advanced_operations.cuh",
        "motion_operations.cuh",
    ]
    .iter()
    .map(|p| fs::read_to_string(PathBuf::from("cuda").join(p)))
    .collect::<std::io::Result<Vec<_>>>()?
    .join("\n");
    let timer = Instant::now();
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("primitive_suite.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("NVRTC: {e:?}"))?;
    fs::write(out.join("primitive_suite.ptx"), ptx.to_src())?;
    fs::write(out.join("primitive_suite.cu"), &source)?;
    let compile_ms = timer.elapsed().as_secs_f64() * 1000.;
    let module = ctx.load_module(ptx)?;
    let build = module.load_function("build_surfaces")?;
    let curves = module.load_function("sample_curves")?;
    let surfaces = module.load_function("sample_surfaces")?;
    let metrics = module.load_function("sample_metrics")?;
    let builder_checks = module.load_function("check_surface_builders")?;
    let base = module.load_function("mesh_base")?;
    let derived = module.load_function("mesh_derived")?;
    let finalize = module.load_function("mesh_finalize")?;
    let ext_build = module.load_function("build_extension_curves")?;
    let ext_sample = module.load_function("sample_extension_curves")?;
    let ext_ops = module.load_function("extension_curve_ops")?;
    let ext_solids = module.load_function("build_extension_solids")?;
    let op_mesh = module.load_function("build_operation_meshes")?;
    let op_loft = module.load_function("build_operation_loft")?;
    let op_sample = module.load_function("sample_operation_surfaces")?;
    let op_checks = module.load_function("operation_resample_checks")?;
    let top_build = module.load_function("build_topology_meshes")?;
    let top_checks = module.load_function("topology_descriptors_checks")?;
    let advanced_build = module.load_function("build_advanced_surfaces")?;
    let advanced_sample = module.load_function("sample_advanced_surfaces")?;
    let motion = module.load_function("motion_suite")?;
    let so = 0_u32;
    let co = (SURF_COUNT * SURF_WORDS) as u32;
    let po = co + 8 * 33 * 3;
    let mo = po + (SURF_COUNT * 81 * 3) as u32;
    let me = mo + 320;
    let ext_offset = me + (MESH_WORDS * MESH_COUNT) as u32;
    let op_offset = ext_offset + extension_output::WORDS as u32;
    let top_offset = op_offset + operation_output::WORDS as u32;
    let adv_offset = top_offset + topology_output::WORDS as u32;
    let motion_offset = adv_offset + advanced_output::WORDS as u32;
    let count = motion_offset as usize + 1024;
    let mut arena = stream.alloc_zeros::<f64>(count)?;
    // SAFETY: all bytes are populated by one full copy before any host read.
    let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f64>(count, 0)? };
    let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    // SAFETY: the arena partitions match the fixed bounded CUDA record layouts.
    unsafe {
        stream
            .launch_builder(&build)
            .arg(&mut arena)
            .arg(&so)
            .launch(launch(SURF_COUNT as u32))?;
        stream
            .launch_builder(&curves)
            .arg(&mut arena)
            .arg(&co)
            .launch(launch(8 * 33))?;
        stream
            .launch_builder(&surfaces)
            .arg(&mut arena)
            .arg(&so)
            .arg(&po)
            .launch(launch((SURF_COUNT * 81) as u32))?;
        stream
            .launch_builder(&metrics)
            .arg(&mut arena)
            .arg(&so)
            .arg(&mo)
            .launch(launch(33))?;
        stream
            .launch_builder(&builder_checks)
            .arg(&mut arena)
            .arg(&so)
            .arg(&(mo + 315))
            .launch(launch(1))?;
        stream
            .launch_builder(&base)
            .arg(&mut arena)
            .arg(&me)
            .launch(launch(9))?;
        stream
            .launch_builder(&derived)
            .arg(&mut arena)
            .arg(&me)
            .launch(launch(7))?;
        stream
            .launch_builder(&finalize)
            .arg(&mut arena)
            .arg(&me)
            .launch(launch(MESH_COUNT as u32))?;
        stream
            .launch_builder(&ext_build)
            .arg(&mut arena)
            .arg(&ext_offset)
            .launch(launch(9))?;
        stream
            .launch_builder(&ext_sample)
            .arg(&mut arena)
            .arg(&ext_offset)
            .launch(launch(9 * 513))?;
        stream
            .launch_builder(&ext_ops)
            .arg(&mut arena)
            .arg(&ext_offset)
            .launch(launch(65))?;
        stream
            .launch_builder(&ext_solids)
            .arg(&mut arena)
            .arg(&ext_offset)
            .launch(launch(10))?;
        stream
            .launch_builder(&op_mesh)
            .arg(&mut arena)
            .arg(&op_offset)
            .launch(launch(7))?;
        stream
            .launch_builder(&op_loft)
            .arg(&mut arena)
            .arg(&op_offset)
            .launch(launch(1))?;
        stream
            .launch_builder(&op_sample)
            .arg(&mut arena)
            .arg(&op_offset)
            .launch(launch(2 * 81))?;
        stream
            .launch_builder(&op_checks)
            .arg(&mut arena)
            .arg(&op_offset)
            .launch(launch(1))?;
        stream
            .launch_builder(&top_build)
            .arg(&mut arena)
            .arg(&top_offset)
            .launch(launch(5))?;
        stream
            .launch_builder(&top_checks)
            .arg(&mut arena)
            .arg(&top_offset)
            .launch(launch(1))?;
        stream
            .launch_builder(&advanced_build)
            .arg(&mut arena)
            .arg(&adv_offset)
            .launch(launch(1))?;
        stream
            .launch_builder(&advanced_sample)
            .arg(&mut arena)
            .arg(&adv_offset)
            .launch(launch(9 * 81))?;
        stream
            .launch_builder(&motion)
            .arg(&mut arena)
            .arg(&motion_offset)
            .launch(launch(1))?;
    }
    let geometry_end = stream.record_event(flags)?;
    stream.memcpy_dtoh(&arena, &mut pinned)?;
    let end = stream.record_event(flags)?;
    end.synchronize()?;
    let data = pinned.as_slice()?;
    if data.iter().any(|p| !p.is_finite()) {
        return Err("GPU suite produced a nonfinite value".into());
    }
    let mut surface_json = Vec::new();
    for i in 0..SURF_COUNT {
        let s = &data[i * SURF_WORDS..(i + 1) * SURF_WORDS];
        if s[11] != 0. {
            return Err("GPU control-net builder rejected fixture".into());
        }
        let (nu, nv, pu, pv) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
        if nu > 32 || nv > 32 || pu > 5 || pv > 5 {
            return Err("Invalid GPU surface descriptor".into());
        }
        surface_json.push(if s[8]==1.{format!("{{\"id\":{i},\"kind\":\"offset\",\"base\":{},\"distance\":{}}}",s[9],s[10])}else{format!("{{\"id\":{i},\"kind\":\"nurbs\",\"size\":[{nu},{nv}],\"degree\":[{pu},{pv}],\"u_knots\":{:?},\"v_knots\":{:?},\"poles_xyzw\":{:?}}}",&s[16..16+nu+pu+1],&s[56..56+nv+pv+1],&s[96..96+4*nu*nv])});
    }
    fs::write(
        out.join("geometry.json"),
        format!(
            "{{\"surfaces\":[{}],\"curve_samples\":{:?},\"surface_samples\":{:?},\"metrics\":{:?}}}",
            surface_json.join(","),
            &data[co as usize..po as usize],
            &data[po as usize..mo as usize],
            &data[mo as usize..me as usize]
        ),
    )?;
    let names = [
        "box",
        "uv_sphere",
        "cylinder",
        "cone",
        "torus",
        "grid",
        "triangulate_polygon",
        "extrude_face",
        "inset_face",
        "subdivide",
        "weld",
        "normals",
        "bevel_vertices",
    ];
    let mut records = Vec::new();
    let mut gallery = Vec::new();
    for id in 0..MESH_COUNT {
        let s = &data[me as usize + id * MESH_WORDS..me as usize + (id + 1) * MESH_WORDS];
        if id >= 13 {
            let expected = [102., 104., 105.][id - 13];
            if s[2] != expected {
                return Err(
                    format!("Negative mesh case {id}: expected {expected}, got {}", s[2]).into(),
                );
            }
            continue;
        }
        if s[2] != 0. {
            return Err(format!("Mesh operator {} failed with status {}", names[id], s[2]).into());
        }
        let (nv, nt) = (s[0] as usize, s[1] as usize);
        if nv == 0 || nv > MV || nt == 0 || nt > MT {
            return Err("Invalid mesh bounds".into());
        }
        let positions = &s[8..8 + 3 * nv];
        let normals = &s[8 + 3 * MV..8 + 3 * MV + 3 * nv];
        let indices = &s[8 + 6 * MV..8 + 6 * MV + 3 * nt];
        let mut soup = Vec::with_capacity(9 * nt);
        let mut area = 0.;
        for &index in indices {
            let k = index as usize;
            if k >= nv || index != k as f64 {
                return Err("GPU mesh index out of range".into());
            }
            soup.extend(positions[3 * k..3 * k + 3].iter().map(|x| *x as f32));
        }
        for p in normals.chunks_exact(3) {
            let length = p.iter().map(|x| x * x).sum::<f64>().sqrt();
            if (length - 1.).abs() > 1e-10 {
                return Err(format!("Nonunit normal in {}", names[id]).into());
            }
        }
        for t in soup.chunks_exact(9) {
            let mut a = [0.; 3];
            let mut b = [0.; 3];
            for k in 0..3 {
                a[k] = f64::from(t[3 + k] - t[k]);
                b[k] = f64::from(t[6 + k] - t[k]);
            }
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            area += 0.5 * c.iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        let st = mesh::analyze(&soup)?;
        if st.triangles != nt || st.vertices != nv {
            return Err(
                "Indexed vertex/triangle counts differ from independent mesh analysis".into(),
            );
        }
        let boundary = match id {
            5 => 32,
            6 => 6,
            8 => 5,
            _ => 0,
        };
        let euler = match id {
            4 | 11 => 0,
            5 | 6 | 8 => 1,
            _ => 2,
        };
        if st.boundary_edges != boundary
            || st.euler != euler
            || st.nonmanifold_edges + st.orientation_errors + st.degenerate_triangles > 0
        {
            return Err(format!("{} topology failed: {st:?}", names[id]).into());
        }
        if boundary == 0 && st.volume <= 0. {
            return Err("Closed mesh has nonpositive volume".into());
        }
        if (id == 0 || id == 9 || id == 10) && (st.volume - 8.).abs() > 1e-10 {
            return Err("Box/subdivide/weld volume mismatch".into());
        }
        if id == 6 && (area - 3.).abs() > 1e-10 {
            return Err("Concave polygon area mismatch".into());
        }
        if id == 7 && (st.volume - 6.).abs() > 1e-5 {
            return Err("Extruded concave polygon volume mismatch".into());
        }
        if id == 9 && nv != 26 || id == 10 && nv != 8 || id == 12 && nv != 24 {
            return Err("Unexpected indexed topology after mesh operation".into());
        }
        mesh::write_stl(&out.join(format!("{}.stl", names[id])), &soup)?;
        for p in soup.chunks_exact(3) {
            gallery.extend([
                p[0] + 4. * (id % 4) as f32 - 6.,
                p[1] + 4. * (id / 4) as f32 - 6.,
                p[2],
            ]);
        }
        records.push(format!("{{\"id\":{id},\"name\":\"{}\",\"vertices\":{nv},\"triangles\":{nt},\"positions\":{positions:?},\"normals\":{normals:?},\"indices\":{indices:?},\"boundary_edges\":{},\"euler\":{},\"volume\":{},\"area\":{area},\"extents\":{:?}}}",names[id],st.boundary_edges,st.euler,st.volume,st.extents));
    }
    fs::write(out.join("meshes.json"), format!("[{}]", records.join(",")))?;
    mesh::write_stl(&out.join("model.stl"), &gallery)?;
    mesh::write_viewer(&out.join("viewer.html"), &gallery, 8.)?;
    let html = fs::read_to_string(out.join("viewer.html"))?
        .replace("让代码成为形状", "多边形原语测试集")
        .replace(
            "沿不同角度检查表面和贯通孔",
            "检查基础网格、挤出、内插、细分和截角",
        );
    let html = html.replace("</aside>", "<details class=\"small\"><summary>排列顺序（逐行）</summary>box · sphere · cylinder · cone<br>torus · grid · triangulate · extrude<br>inset · subdivide · weld · normals<br>bevel</details></aside>");
    fs::write(out.join("viewer.html"), html)?;
    let samples = &data[po as usize..mo as usize];
    let mut surface_gallery = Vec::new();
    for id in 0..SURF_COUNT {
        for i in 0..8 {
            for j in 0..8 {
                let indices = [
                    9 * i + j,
                    9 * (i + 1) + j,
                    9 * (i + 1) + j + 1,
                    9 * i + j,
                    9 * (i + 1) + j + 1,
                    9 * i + j + 1,
                ];
                for k in indices {
                    let p = &samples[3 * (81 * id + k)..3 * (81 * id + k) + 3];
                    surface_gallery.extend([
                        p[0] as f32 + 4. * (id % 4) as f32 - 6.,
                        p[1] as f32 + 4. * (id / 4) as f32 - 6.,
                        p[2] as f32,
                    ]);
                }
            }
        }
    }
    mesh::write_stl(&out.join("surface_model.stl"), &surface_gallery)?;
    mesh::write_viewer(&out.join("surface_viewer.html"), &surface_gallery, 8.)?;
    let html = fs::read_to_string(out.join("surface_viewer.html"))?
        .replace("让代码成为形状", "曲面原语测试集")
        .replace(
            "沿不同角度检查表面和贯通孔",
            "检查 NURBS、放样、Coons 和 G1/G2 连接带",
        )
        .replace("href=\"model.stl\"", "href=\"surface_model.stl\"")
        .replace("</aside>", "<details class=\"small\"><summary>排列顺序（逐行）</summary>plane · Bezier · NURBS · ruled<br>extrude · revolve · sweep · loft<br>Coons · offset · G1 · G2<br>left support · right support</details></aside>");
    fs::write(out.join("surface_viewer.html"), html)?;
    extension_output::write(&data[ext_offset as usize..op_offset as usize], &out)?;
    operation_output::write(&data[op_offset as usize..top_offset as usize], &out)?;
    topology_output::write(&data[top_offset as usize..adv_offset as usize], &out)?;
    advanced_output::write(&data[adv_offset as usize..motion_offset as usize], &out)?;
    let motion_data = &data[motion_offset as usize..];
    if motion_data[1000] != 0. || motion_data[992..996] != [1., 1., 1., 1.] {
        return Err("GPU motion operation rejected fixture or accepted invalid input".into());
    }
    fs::write(out.join("motion.json"), format!("{motion_data:?}"))?;
    // SAFETY: events have completed in the same context/stream.
    let gpu_ms = unsafe { result::event::elapsed(start.cu_event(), geometry_end.cu_event())? };
    let copy_ms = unsafe { result::event::elapsed(geometry_end.cu_event(), end.cu_event())? };
    let report = format!(
        "{{\"device\":\"{device}\",\"compile_ms\":{compile_ms},\"gpu_ms\":{gpu_ms},\"copy_ms\":{copy_ms},\"download_bytes\":{},\"final_readback_calls\":1,\"intermediate_readback_calls\":0,\"curve_cases\":8,\"surface_objects\":14,\"positive_mesh_cases\":13,\"negative_mesh_cases\":3,\"extension_curve_objects\":9,\"extension_solid_objects\":10,\"extension_invalid_inputs_rejected\":8,\"operation_mesh_objects\":7,\"operation_surface_objects\":2,\"operation_invalid_inputs_rejected\":7,\"mesh_checks_passed\":true}}",
        count * 8
    );
    let report = report.replace(
        "\"mesh_checks_passed\":true",
        "\"topology_mesh_objects\":5,\"advanced_surface_objects\":9,\"mesh_checks_passed\":true",
    );
    fs::write(out.join("gpu_report.json"), &report)?;
    println!("{report}");
    Ok(())
}
