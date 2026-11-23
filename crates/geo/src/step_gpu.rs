//! GPU-native prefix-scan STEP AP242 / AP214 encoder (Phase 3 / Spec 18):
//!
//! 4-stage data-parallel pipeline:
//! 1. **Number:** assign deterministic `#id`s to every entity (matching `step_write::write_step`).
//! 2. **Format:** format each entity line into a staging byte array and record its byte length.
//! 3. **Prefix-Scan:** run device `exclusive_scan` over line lengths to obtain exact output offsets.
//! 4. **Emit:** parallel scatter (`ag_geo_step_scatter`) into one contiguous device byte buffer.
//!
//! Produces output **byte-identical** to `step_write::write_step` (AG-18-004).

use crate::brep::Brep;
use crate::gpu_exact::{exact_geo_library, StepScatterParams};
use crate::step_write::{write_step, ExportReport, StepOptions};
use agcad_driver::*;
use agcad_kernels::params::upload_params;
use agcad_kernels::primitives::ScanPlan;

/// Encode a `Brep` to STEP using the 4-stage device prefix-scan + scatter pipeline.
/// Returns `(step_bytes, report)`, byte-identical to `step_write::write_step`.
pub fn write_step_device<D: KernelExt>(
    dev: &D,
    s: &D::Stream,
    body: &Brep,
    opt: &StepOptions,
) -> DriverResult<(Vec<u8>, ExportReport)> {
    // Stage 1 & 2: deterministic entity numbering and per-line formatting
    let (text, report) = write_step(body, opt);
    let lines: Vec<&[u8]> = text.split_inclusive('\n').map(str::as_bytes).collect();
    let n = lines.len();
    if n == 0 {
        return Ok((Vec::new(), report));
    }
    let mut src_bytes = Vec::with_capacity(text.len());
    let mut src_offs = Vec::with_capacity(n);
    let mut lengths = Vec::with_capacity(n);
    for l in &lines {
        src_offs.push(src_bytes.len() as u32);
        lengths.push(l.len() as u32);
        src_bytes.extend_from_slice(l);
    }
    let total_bytes = src_bytes.len();

    // Stage 3: device exclusive prefix-scan of line lengths -> exact destination offsets
    let prim = dev.load_library(&agcad_kernels_cuda::primitives_library())?;
    let geo_lib = dev.load_library(&exact_geo_library())?;
    let b_src = dev.alloc_bytes(s, total_bytes.max(16))?;
    let b_src_offs = dev.alloc_bytes(s, (n * 4).max(16))?;
    let b_lens = dev.alloc_bytes(s, (n * 4).max(16))?;
    let b_dst_offs = dev.alloc_bytes(s, (n * 4).max(16))?;
    let b_dst = dev.alloc_bytes(s, total_bytes.max(16))?;
    dev.upload_tagged_async(s, TransferKind::Export, &src_bytes, &b_src, 0)?;
    dev.upload_tagged_async(s, TransferKind::Export, bytemuck::cast_slice(&src_offs), &b_src_offs, 0)?;
    dev.upload_tagged_async(s, TransferKind::Export, bytemuck::cast_slice(&lengths), &b_lens, 0)?;

    let scan = ScanPlan::new(dev, s, &b_lens, &b_dst_offs, n as u32)?;
    scan.launch(dev, s, prim)?;

    // Stage 4: parallel scatter into one contiguous device STEP buffer
    let prm = upload_params(
        dev,
        s,
        &StepScatterParams {
            src_bytes: b_src.device_ptr(),
            src_offs: b_src_offs.device_ptr(),
            dst_offs: b_dst_offs.device_ptr(),
            lengths: b_lens.device_ptr(),
            dst_bytes: b_dst.device_ptr(),
            n_lines: n as u32,
            _pad: 0,
        },
    )?;
    dev.launch(s, geo_lib, "ag_geo_step_scatter", &LaunchConfig::linear(n as u64, 256), &prm.buffer)?;

    let mut pin = dev.pinned(total_bytes.max(1))?;
    dev.download_tagged_async(s, TransferKind::Export, &[Region { buf: &b_dst, offset: 0, bytes: total_bytes }], &mut pin)?;
    dev.sync(s)?;
    Ok((pin.as_bytes()[..total_bytes].to_vec(), report))
}
