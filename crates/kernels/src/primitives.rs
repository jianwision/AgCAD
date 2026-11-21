//! Deterministic parallel primitives (review F8: no float atomics, fixed-order reductions):
//! exclusive scan (u32), stream compaction, stable LSD radix sort (u64 keys + u32 values),
//! segmented reduction (f64), stable top-k.

use crate::params::{upload_params, ParamUpload};
use agcad_driver::*;
use bytemuck::{Pod, Zeroable};

/// Elements per scan block (CUDA kernels use 256 threads × 4 items).
pub const SCAN_TILE: u32 = 1024;
/// Elements per radix-sort tile.
pub const SORT_TILE: u32 = 1024;
pub const RADIX_BITS: u32 = 8;
pub const RADIX_BINS: u32 = 1 << RADIX_BITS;

pub mod names {
    pub const SCAN_BLOCKS: &str = "ag_scan_blocks";
    pub const SCAN_ADD: &str = "ag_scan_add";
    pub const COMPACT_SCATTER: &str = "ag_compact_scatter";
    pub const RADIX_HISTOGRAM: &str = "ag_radix_histogram";
    pub const RADIX_SCATTER: &str = "ag_radix_scatter";
    pub const SEGMENTED_REDUCE: &str = "ag_segmented_reduce_f64";
    pub const ORDER_KEYS_F64: &str = "ag_order_keys_f64";
    pub const MESH_TERMS: &str = "ag_mesh_terms";
}

/// Per-triangle mass-property terms (spec 14 `evaluate.measure_inertia`, GPU hot path).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshTermsParams {
    /// f64 xyz triples.
    pub positions: u64,
    /// u32 index triples (global vertex indices into `positions`).
    pub tris: u64,
    /// SoA output, 5 × n_tris f64: 6·signed tetra volume, area, (a+b+c)·6V (x, y, z).
    pub out: u64,
    pub n_tris: u32,
    pub _pad: u32,
}

/// Number of SoA term arrays written by [`names::MESH_TERMS`].
pub const MESH_TERMS: usize = 5;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ScanParams {
    pub input: u64,
    pub output: u64,
    pub block_sums: u64,
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ScanAddParams {
    pub output: u64,
    pub block_offsets: u64,
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CompactParams {
    pub flags: u64,
    pub offsets: u64,
    pub src: u64,
    pub dst: u64,
    pub n: u32,
    pub elem_bytes: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RadixParams {
    pub keys_in: u64,
    pub vals_in: u64,
    pub keys_out: u64,
    pub vals_out: u64,
    /// `hist[digit * tiles + tile]` (digit-major so one exclusive scan yields global offsets).
    pub hist: u64,
    pub n: u32,
    pub shift: u32,
    pub tiles: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SegReduceParams {
    pub values: u64,
    /// `segments + 1` offsets (CUB convention).
    pub offsets: u64,
    pub out: u64,
    pub segments: u32,
    /// 0 = sum, 1 = min, 2 = max.
    pub op: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrderKeysParams {
    pub scores: u64,
    pub keys: u64,
    pub vals: u64,
    pub n: u32,
    /// 0 = ascending (minimize), 1 = descending.
    pub descending: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Min = 1,
    Max = 2,
}

type Buf<D> = <D as AgDevice>::Buffer;

/// Exclusive scan of `n` u32 values. Returns the total (one small readback — host-side
/// orchestration only; device harness code uses the in-graph variant without readback).
pub fn exclusive_scan_u32<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, input: &Buf<D>, output: &Buf<D>, n: u32) -> DriverResult<()> {
    if n == 0 {
        return Ok(());
    }
    let blocks = n.div_ceil(SCAN_TILE);
    let sums = dev.alloc_bytes(s, blocks as usize * 4)?;
    let p = upload_params(dev, s, &ScanParams { input: input.device_ptr(), output: output.device_ptr(), block_sums: sums.device_ptr(), n, _pad: 0 })?;
    dev.launch(s, lib, names::SCAN_BLOCKS, &LaunchConfig { grid: Dim3::x(blocks), block: Dim3::x(256), shared_mem_bytes: 0, cluster: None }, &p.buffer)?;
    if blocks > 1 {
        let offs = dev.alloc_bytes(s, blocks as usize * 4)?;
        exclusive_scan_u32(dev, s, lib, &sums, &offs, blocks)?;
        let pa = upload_params(dev, s, &ScanAddParams { output: output.device_ptr(), block_offsets: offs.device_ptr(), n, _pad: 0 })?;
        dev.launch(s, lib, names::SCAN_ADD, &LaunchConfig { grid: Dim3::x(blocks), block: Dim3::x(256), shared_mem_bytes: 0, cluster: None }, &pa.buffer)?;
    }
    Ok(())
}

/// Stable compaction: keep elements whose flag is non-zero. `offsets` receives the scan.
pub fn compact<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, flags: &Buf<D>, src: &Buf<D>, dst: &Buf<D>, n: u32, elem_bytes: u32) -> DriverResult<Buf<D>> {
    let offsets = dev.alloc_bytes(s, (n as usize).max(1) * 4)?;
    exclusive_scan_u32(dev, s, lib, flags, &offsets, n)?;
    let p = upload_params(dev, s, &CompactParams { flags: flags.device_ptr(), offsets: offsets.device_ptr(), src: src.device_ptr(), dst: dst.device_ptr(), n, elem_bytes })?;
    dev.launch(s, lib, names::COMPACT_SCATTER, &LaunchConfig::linear(n as u64, 256), &p.buffer)?;
    Ok(offsets)
}

/// Stable LSD radix sort of `n` (u64 key, u32 value) pairs over `key_bits` low bits.
/// Result ends in (keys, vals) — buffers are ping-ponged internally.
pub fn radix_sort_pairs<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, keys: &Buf<D>, vals: &Buf<D>, n: u32, key_bits: u32) -> DriverResult<()> {
    if n <= 1 {
        return Ok(());
    }
    let tiles = n.div_ceil(SORT_TILE);
    let k2 = dev.alloc_bytes(s, n as usize * 8)?;
    let v2 = dev.alloc_bytes(s, n as usize * 4)?;
    let hist_n = RADIX_BINS * tiles;
    let hist = dev.alloc_bytes(s, hist_n as usize * 4)?;
    let hist_scan = dev.alloc_bytes(s, hist_n as usize * 4)?;
    let passes = key_bits.div_ceil(RADIX_BITS);
    let (mut ki, mut vi, mut ko, mut vo) = (keys, vals, &k2, &v2);
    for pass in 0..passes {
        let shift = pass * RADIX_BITS;
        let prm = RadixParams { keys_in: ki.device_ptr(), vals_in: vi.device_ptr(), keys_out: ko.device_ptr(), vals_out: vo.device_ptr(), hist: hist.device_ptr(), n, shift, tiles, _pad: 0 };
        let p = upload_params(dev, s, &prm)?;
        let cfg = LaunchConfig { grid: Dim3::x(tiles), block: Dim3::x(RADIX_BINS), shared_mem_bytes: SORT_TILE * 12, cluster: None };
        dev.launch(s, lib, names::RADIX_HISTOGRAM, &cfg, &p.buffer)?;
        exclusive_scan_u32(dev, s, lib, &hist, &hist_scan, hist_n)?;
        let p2 = upload_params(dev, s, &RadixParams { hist: hist_scan.device_ptr(), ..prm })?;
        dev.launch(s, lib, names::RADIX_SCATTER, &cfg, &p2.buffer)?;
        core::mem::swap(&mut ki, &mut ko);
        core::mem::swap(&mut vi, &mut vo);
    }
    if passes % 2 == 1 {
        // Result is in the scratch buffers; copy back so callers see it in place.
        dev.copy_device_async(s, ki, 0, keys, 0, n as usize * 8)?;
        dev.copy_device_async(s, vi, 0, vals, 0, n as usize * 4)?;
    }
    Ok(())
}

/// Deterministic segmented reduction (one block per segment, fixed tree).
pub fn segmented_reduce_f64<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, values: &Buf<D>, offsets: &Buf<D>, out: &Buf<D>, segments: u32, op: ReduceOp) -> DriverResult<()> {
    let p = upload_params(dev, s, &SegReduceParams { values: values.device_ptr(), offsets: offsets.device_ptr(), out: out.device_ptr(), segments, op: op as u32 })?;
    dev.launch(s, lib, names::SEGMENTED_REDUCE, &LaunchConfig { grid: Dim3::x(segments.max(1)), block: Dim3::x(256), shared_mem_bytes: 256 * 8, cluster: None }, &p.buffer)?;
    Ok(())
}

/// Stable top-k ordering: keys = order-preserving bits of the score, values = configuration
/// ids `0..n`; a stable sort keeps ties in id order (spec 06 RANK, AG-05-001).
/// After the call, `vals[0..k]` hold the selected configuration ids.
pub fn stable_order_by_score<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, scores: &Buf<D>, keys: &Buf<D>, vals: &Buf<D>, n: u32, descending: bool) -> DriverResult<()> {
    let p = upload_params(dev, s, &OrderKeysParams { scores: scores.device_ptr(), keys: keys.device_ptr(), vals: vals.device_ptr(), n, descending: descending as u32 })?;
    dev.launch(s, lib, names::ORDER_KEYS_F64, &LaunchConfig::linear(n as u64, 256), &p.buffer)?;
    radix_sort_pairs(dev, s, lib, keys, vals, n, 64)
}

/// Order-preserving map f64 → u64 (NaN sorts last); shared with the CUDA implementation.
pub fn f64_order_key(x: f64, descending: bool) -> u64 {
    let x = if x.is_nan() { f64::INFINITY } else { x };
    let x = if descending { -x } else { x };
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1u64 << 63) }
}

/// Batched mass properties of many meshes (one per configuration) in one pass: per-triangle
/// terms, then 5 deterministic segmented sums (one segment per mesh). `tri_offsets` holds
/// `meshes + 1` triangle offsets. Returns a buffer of `5 × meshes` f64 sums laid out
/// `[term][mesh]` (see [`mass_properties_from_sums`]). Bit-identical on every device.
pub fn mesh_mass_sums<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, positions: &Buf<D>, tris: &Buf<D>, n_tris: u32, tri_offsets: &Buf<D>, meshes: u32) -> DriverResult<Buf<D>> {
    let terms = dev.alloc_bytes(s, (n_tris as usize).max(1) * MESH_TERMS * 8)?;
    let p = upload_params(dev, s, &MeshTermsParams { positions: positions.device_ptr(), tris: tris.device_ptr(), out: terms.device_ptr(), n_tris, _pad: 0 })?;
    dev.launch(s, lib, names::MESH_TERMS, &LaunchConfig::linear(n_tris as u64, 256), &p.buffer)?;
    let out = dev.alloc_bytes(s, (meshes as usize).max(1) * MESH_TERMS * 8)?;
    for k in 0..MESH_TERMS {
        let prm = SegReduceParams { values: terms.device_ptr() + (k * n_tris as usize * 8) as u64, offsets: tri_offsets.device_ptr(), out: out.device_ptr() + (k * meshes as usize * 8) as u64, segments: meshes, op: ReduceOp::Sum as u32 };
        let pk = upload_params(dev, s, &prm)?;
        dev.launch(s, lib, names::SEGMENTED_REDUCE, &LaunchConfig { grid: Dim3::x(meshes.max(1)), block: Dim3::x(256), shared_mem_bytes: 256 * 8, cluster: None }, &pk.buffer)?;
    }
    Ok(out)
}

/// (volume, area, centroid) of mesh `m` from the sums of [`mesh_mass_sums`].
pub fn mass_properties_from_sums(sums: &[f64], meshes: usize, m: usize) -> (f64, f64, [f64; 3]) {
    let g = |k: usize| sums[k * meshes + m];
    let vol = g(0) / 6.0;
    let c = if vol != 0.0 { [g(2) / (24.0 * vol), g(3) / (24.0 * vol), g(4) / (24.0 * vol)] } else { [0.0; 3] };
    (vol, g(1), c)
}

/// Pre-planned exclusive scan of a fixed-size u32 buffer: scratch buffers and parameter blocks
/// are allocated up front, so [`ScanPlan::launch`] only enqueues kernels — safe inside graph
/// capture and conditional-node bodies (which may not contain allocations). Same kernels and
/// association order as [`exclusive_scan_u32`].
pub struct ScanPlan<D: KernelExt> {
    steps: Vec<(&'static str, LaunchConfig, ParamUpload<D>)>,
    _scratch: Vec<Buf<D>>,
}

impl<D: KernelExt> ScanPlan<D> {
    pub fn new(dev: &D, s: &D::Stream, input: &Buf<D>, output: &Buf<D>, n: u32) -> DriverResult<Self> {
        let mut steps = Vec::new();
        let mut scratch = Vec::new();
        fn plan<D: KernelExt>(dev: &D, s: &D::Stream, input: u64, output: u64, n: u32, steps: &mut Vec<(&'static str, LaunchConfig, ParamUpload<D>)>, scratch: &mut Vec<Buf<D>>) -> DriverResult<()> {
            if n == 0 { return Ok(()); }
            let blocks = n.div_ceil(SCAN_TILE);
            let sums = dev.alloc_bytes(s, blocks as usize * 4)?;
            let p = upload_params(dev, s, &ScanParams { input, output, block_sums: sums.device_ptr(), n, _pad: 0 })?;
            steps.push((names::SCAN_BLOCKS, LaunchConfig { grid: Dim3::x(blocks), block: Dim3::x(256), shared_mem_bytes: 0, cluster: None }, p));
            if blocks > 1 {
                let offs = dev.alloc_bytes(s, blocks as usize * 4)?;
                plan(dev, s, sums.device_ptr(), offs.device_ptr(), blocks, steps, scratch)?;
                let pa = upload_params(dev, s, &ScanAddParams { output, block_offsets: offs.device_ptr(), n, _pad: 0 })?;
                steps.push((names::SCAN_ADD, LaunchConfig { grid: Dim3::x(blocks), block: Dim3::x(256), shared_mem_bytes: 0, cluster: None }, pa));
                scratch.push(offs);
            }
            scratch.push(sums);
            Ok(())
        }
        plan(dev, s, input.device_ptr(), output.device_ptr(), n, &mut steps, &mut scratch)?;
        Ok(ScanPlan { steps, _scratch: scratch })
    }

    pub fn launch(&self, dev: &D, s: &D::Stream, lib: LibraryId) -> DriverResult<()> {
        for (k, cfg, p) in &self.steps { dev.launch(s, lib, k, cfg, &p.buffer)?; }
        Ok(())
    }
}
