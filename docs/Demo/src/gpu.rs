use crate::{Options, Result};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, result, sys::CUevent_flags},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{fs, time::Instant};

const PROBES: usize = 64;
const HEADER_WORDS: usize = 16 + PROBES;

pub struct GpuOutput {
    pub device: String,
    pub triangles: Vec<f32>,
    pub pixels: Vec<u8>,
    pub ptx: String,
    pub source: String,
    pub step: f32,
    pub boundary_inside: usize,
    pub probe_error: f64,
    pub download_bytes: usize,
    pub padding_bytes: usize,
    pub field_bytes_not_downloaded: usize,
    pub compile_ms: f64,
    pub sampling_ms: f64,
    pub meshing_ms: f64,
    pub render_ms: f64,
    pub download_ms: f64,
}

pub fn execute(options: &Options) -> Result<GpuOutput> {
    // Preload by absolute path on Windows, where a DLL name alone may not search
    // CUDA_PATH. Keep the library loaded throughout compilation and execution.
    #[cfg(windows)]
    let _nvrtc = crate::cuda_loader::preload_nvrtc()?;
    let ctx = CudaContext::new(0)?;
    let device = ctx.name()?;
    println!("CUDA device: {device}");
    let stream = ctx.default_stream();
    let source = include_str!("../cuda/kernels.cu")
        .replace("// __MODEL_SOURCE__", &fs::read_to_string(&options.model)?);
    let timer = Instant::now();
    let compiled = compile_ptx_with_opts(
        &source,
        CompileOptions {
            fmad: Some(false),
            name: Some("model.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|err| {
        format!(
            "NVRTC compilation failed for {}:\n{err:?}",
            options.model.display()
        )
    })?;
    let ptx = compiled.to_src();
    let compile_ms = timer.elapsed().as_secs_f64() * 1000.0;
    let module = ctx.load_module(compiled)?;
    let sample = module.load_function("sample_field")?;
    let polygonize = module.load_function("polygonize")?;
    let render = module.load_function("render")?;
    let gather = module.load_function("gather_probes")?;
    let n = options.grid;
    let nodes = n + 1;
    let count = nodes.pow(3);
    // Custom model zero sets must stay strictly within this domain.
    let bound = options.radius * 1.2;
    let step = 2.0 * bound / n as f32;
    let capacity = n * n * 32;
    let pixel_offset = HEADER_WORDS + capacity as usize * 9;
    let packet_words = pixel_offset + options.width.pow(2) as usize;
    let pixel_offset_arg = pixel_offset as u32;
    let mut field = stream.alloc_zeros::<f32>(count as usize)?;
    let mut packet = stream.alloc_zeros::<u32>(packet_words)?;
    // SAFETY: the entire pinned allocation is initialized by the one DtoH copy
    // before it is read. Flags=0 avoids write-combined memory, which is slow to read.
    let mut host_packet = unsafe { ctx.alloc_pinned_with_flags::<u32>(packet_words, 0)? };
    let event_flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let sampling_start = stream.record_event(event_flags)?;
    // SAFETY: types/order match the kernels. The u32 arena stores float bit
    // patterns in its triangle section, and every kernel bounds-checks writes.
    unsafe {
        stream
            .launch_builder(&sample)
            .arg(&mut field)
            .arg(&nodes)
            .arg(&bound)
            .arg(&step)
            .arg(&options.radius)
            .arg(&options.hole)
            .arg(&mut packet)
            .launch(LaunchConfig::for_num_elems(count))?;
    }
    let meshing_start = stream.record_event(event_flags)?;
    unsafe {
        stream
            .launch_builder(&polygonize)
            .arg(&field)
            .arg(&n)
            .arg(&bound)
            .arg(&step)
            .arg(&mut packet)
            .arg(&capacity)
            .launch(LaunchConfig::for_num_elems(n.pow(3)))?;
    }
    let rendering_start = stream.record_event(event_flags)?;
    unsafe {
        stream
            .launch_builder(&render)
            .arg(&mut packet)
            .arg(&pixel_offset_arg)
            .arg(&options.width)
            .arg(&options.radius)
            .arg(&options.hole)
            .launch(LaunchConfig::for_num_elems(options.width.pow(2)))?;
    }
    let rendering_end = stream.record_event(event_flags)?;
    unsafe {
        stream
            .launch_builder(&gather)
            .arg(&field)
            .arg(&count)
            .arg(&mut packet)
            .launch(LaunchConfig::for_num_elems(PROBES as u32))?;
    }
    let copy_start = stream.record_event(event_flags)?;
    // One application-level DtoH copy. Unused triangle slots travel too, avoiding
    // an intermediate triangle-count readback and the resulting host dependency.
    stream.memcpy_dtoh(&packet, &mut host_packet)?;
    let copy_end = stream.record_event(event_flags)?;
    // First host-visible completion point in the geometry/render pipeline.
    copy_end.synchronize()?;
    let words = host_packet.as_slice()?;
    let num_triangles = words[0];
    if words[2] > 0 {
        return Err(format!("GPU found {} non-finite field samples", words[2]).into());
    }
    if num_triangles == 0 || num_triangles > capacity {
        return Err(format!(
            "GPU produced {num_triangles} triangles (capacity {capacity}); adjust model or capacity"
        )
        .into());
    }
    let triangle_words = num_triangles as usize * 9;
    let triangles_host = words[HEADER_WORDS..HEADER_WORDS + triangle_words]
        .iter()
        .map(|&x| f32::from_bits(x))
        .collect();
    let pixels_host = words[pixel_offset..]
        .iter()
        .flat_map(|word| {
            let rgba = word.to_le_bytes();
            [rgba[0], rgba[1], rgba[2]]
        })
        .collect();
    let mut probe_error = 0.0_f64;
    if options.check {
        for q in 0..PROBES {
            let i = (q as u32).wrapping_mul(2_654_435_761) % count;
            let x = i % nodes;
            let y = (i / nodes) % nodes;
            let z = i / (nodes * nodes);
            // f64 host reference is used only for independent sparse validation.
            let px = -f64::from(bound) + f64::from(step) * f64::from(x);
            let py = -f64::from(bound) + f64::from(step) * f64::from(y);
            let pz = -f64::from(bound) + f64::from(step) * f64::from(z);
            let sphere = (px * px + py * py + pz * pz).sqrt() - f64::from(options.radius);
            let cylinder = (px * px + py * py).sqrt() - f64::from(options.hole);
            probe_error = probe_error
                .max((f64::from(f32::from_bits(words[16 + q])) - sphere.max(-cylinder)).abs());
        }
    }
    // SAFETY: these events are alive, in one context, and already complete.
    // Raw elapsed queries avoid adding per-stage waits to the timed pipeline.
    let sampling_ms =
        unsafe { result::event::elapsed(sampling_start.cu_event(), meshing_start.cu_event())? }
            as f64;
    let meshing_ms =
        unsafe { result::event::elapsed(meshing_start.cu_event(), rendering_start.cu_event())? }
            as f64;
    let render_ms =
        unsafe { result::event::elapsed(rendering_start.cu_event(), rendering_end.cu_event())? }
            as f64;
    let download_ms =
        unsafe { result::event::elapsed(copy_start.cu_event(), copy_end.cu_event())? } as f64;
    Ok(GpuOutput {
        device,
        triangles: triangles_host,
        pixels: pixels_host,
        ptx,
        source,
        step,
        boundary_inside: words[1] as usize,
        probe_error,
        download_bytes: packet_words * 4,
        padding_bytes: (capacity as usize * 9 - triangle_words) * 4,
        field_bytes_not_downloaded: count as usize * 4,
        compile_ms,
        sampling_ms,
        meshing_ms,
        render_ms,
        download_ms,
    })
}
