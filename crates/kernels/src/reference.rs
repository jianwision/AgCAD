//! CPU reference kernels (conformance oracle). Each mirrors its CUDA twin's **exact
//! association order**, so floating-point results are bit-identical (AG-06-006, AG-05-001).
//!
//! Safety contract for every kernel: `params` points at the documented `repr(C)` block, and
//! every address in it is a valid host address of the stated length (CPU reference device).

use crate::primitives::*;
use agcad_driver::{CpuKernelFn, LaunchConfig};
use core::ptr::read_unaligned;

unsafe fn slice<'a, T>(addr: u64, n: usize) -> &'a [T] {
    if n == 0 { return &[]; }
    unsafe { core::slice::from_raw_parts(addr as *const T, n) }
}
#[allow(clippy::mut_from_ref)]
unsafe fn slice_mut<'a, T>(addr: u64, n: usize) -> &'a mut [T] {
    if n == 0 { return &mut []; }
    unsafe { core::slice::from_raw_parts_mut(addr as *mut T, n) }
}

/// # Safety: see module docs.
pub unsafe fn ag_scan_blocks(params: *const u8, _cfg: &LaunchConfig) {
    let p: ScanParams = unsafe { read_unaligned(params as *const ScanParams) };
    let n = p.n as usize;
    let tile = SCAN_TILE as usize;
    let input = unsafe { slice::<u32>(p.input, n) };
    let output = unsafe { slice_mut::<u32>(p.output, n) };
    let blocks = n.div_ceil(tile);
    let sums = unsafe { slice_mut::<u32>(p.block_sums, blocks) };
    for b in 0..blocks {
        let mut acc = 0u32;
        for i in b * tile..((b + 1) * tile).min(n) {
            output[i] = acc;
            acc = acc.wrapping_add(input[i]);
        }
        sums[b] = acc;
    }
}

/// # Safety: see module docs.
pub unsafe fn ag_scan_add(params: *const u8, _cfg: &LaunchConfig) {
    let p: ScanAddParams = unsafe { read_unaligned(params as *const ScanAddParams) };
    let n = p.n as usize;
    let tile = SCAN_TILE as usize;
    let out = unsafe { slice_mut::<u32>(p.output, n) };
    let offs = unsafe { slice::<u32>(p.block_offsets, n.div_ceil(tile)) };
    for (i, o) in out.iter_mut().enumerate() {
        *o = o.wrapping_add(offs[i / tile]);
    }
}

/// # Safety: see module docs.
pub unsafe fn ag_compact_scatter(params: *const u8, _cfg: &LaunchConfig) {
    let p: CompactParams = unsafe { read_unaligned(params as *const CompactParams) };
    let n = p.n as usize;
    let eb = p.elem_bytes as usize;
    let flags = unsafe { slice::<u32>(p.flags, n) };
    let offs = unsafe { slice::<u32>(p.offsets, n) };
    let src = unsafe { slice::<u8>(p.src, n * eb) };
    let kept = flags.iter().filter(|f| **f != 0).count();
    let dst = unsafe { slice_mut::<u8>(p.dst, kept * eb) };
    for i in 0..n {
        if flags[i] != 0 {
            let o = offs[i] as usize * eb;
            dst[o..o + eb].copy_from_slice(&src[i * eb..(i + 1) * eb]);
        }
    }
}

/// # Safety: see module docs.
pub unsafe fn ag_radix_histogram(params: *const u8, _cfg: &LaunchConfig) {
    let p: RadixParams = unsafe { read_unaligned(params as *const RadixParams) };
    let n = p.n as usize;
    let tiles = p.tiles as usize;
    let tile = SORT_TILE as usize;
    let keys = unsafe { slice::<u64>(p.keys_in, n) };
    let hist = unsafe { slice_mut::<u32>(p.hist, RADIX_BINS as usize * tiles) };
    hist.fill(0);
    for t in 0..tiles {
        for &k in &keys[t * tile..((t + 1) * tile).min(n)] {
            let d = ((k >> p.shift) & (RADIX_BINS as u64 - 1)) as usize;
            hist[d * tiles + t] += 1;
        }
    }
}

/// # Safety: see module docs. `hist` holds the exclusive scan of the histogram.
pub unsafe fn ag_radix_scatter(params: *const u8, _cfg: &LaunchConfig) {
    let p: RadixParams = unsafe { read_unaligned(params as *const RadixParams) };
    let n = p.n as usize;
    let tiles = p.tiles as usize;
    let tile = SORT_TILE as usize;
    let ki = unsafe { slice::<u64>(p.keys_in, n) };
    let vi = unsafe { slice::<u32>(p.vals_in, n) };
    let ko = unsafe { slice_mut::<u64>(p.keys_out, n) };
    let vo = unsafe { slice_mut::<u32>(p.vals_out, n) };
    let hist = unsafe { slice::<u32>(p.hist, RADIX_BINS as usize * tiles) };
    for t in 0..tiles {
        let mut run = [0u32; RADIX_BINS as usize];
        for i in t * tile..((t + 1) * tile).min(n) {
            let d = ((ki[i] >> p.shift) & (RADIX_BINS as u64 - 1)) as usize;
            let dst = (hist[d * tiles + t] + run[d]) as usize;
            run[d] += 1;
            ko[dst] = ki[i];
            vo[dst] = vi[i];
        }
    }
}

const REDUCE_THREADS: usize = 256;

fn combine(op: u32, a: f64, b: f64) -> f64 {
    match op {
        1 => if b < a { b } else { a },
        2 => if b > a { b } else { a },
        _ => a + b,
    }
}
fn identity(op: u32) -> f64 {
    match op { 1 => f64::INFINITY, 2 => f64::NEG_INFINITY, _ => 0.0 }
}

/// # Safety: see module docs. Mirrors: per-thread strided accumulation, then shared-memory tree.
pub unsafe fn ag_segmented_reduce_f64(params: *const u8, _cfg: &LaunchConfig) {
    let p: SegReduceParams = unsafe { read_unaligned(params as *const SegReduceParams) };
    let segs = p.segments as usize;
    let offs = unsafe { slice::<u32>(p.offsets, segs + 1) };
    let total = if segs > 0 { offs[segs] as usize } else { 0 };
    let vals = unsafe { slice::<f64>(p.values, total) };
    let out = unsafe { slice_mut::<f64>(p.out, segs) };
    for seg in 0..segs {
        let (b, e) = (offs[seg] as usize, offs[seg + 1] as usize);
        let mut part = [identity(p.op); REDUCE_THREADS];
        for (t, acc) in part.iter_mut().enumerate() {
            let mut j = b + t;
            while j < e {
                *acc = combine(p.op, *acc, vals[j]);
                j += REDUCE_THREADS;
            }
        }
        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            for t in 0..stride {
                part[t] = combine(p.op, part[t], part[t + stride]);
            }
            stride /= 2;
        }
        out[seg] = part[0];
    }
}

/// # Safety: see module docs.
pub unsafe fn ag_order_keys_f64(params: *const u8, _cfg: &LaunchConfig) {
    let p: OrderKeysParams = unsafe { read_unaligned(params as *const OrderKeysParams) };
    let n = p.n as usize;
    let scores = unsafe { slice::<f64>(p.scores, n) };
    let keys = unsafe { slice_mut::<u64>(p.keys, n) };
    let vals = unsafe { slice_mut::<u32>(p.vals, n) };
    for i in 0..n {
        keys[i] = f64_order_key(scores[i], p.descending != 0);
        vals[i] = i as u32;
    }
}

/// CPU reference table for the primitives library.
pub const PRIMITIVES_CPU: &[(&str, CpuKernelFn)] = &[
    (names::SCAN_BLOCKS, ag_scan_blocks),
    (names::SCAN_ADD, ag_scan_add),
    (names::COMPACT_SCATTER, ag_compact_scatter),
    (names::RADIX_HISTOGRAM, ag_radix_histogram),
    (names::RADIX_SCATTER, ag_radix_scatter),
    (names::SEGMENTED_REDUCE, ag_segmented_reduce_f64),
    (names::ORDER_KEYS_F64, ag_order_keys_f64),
    (names::MESH_TERMS, ag_mesh_terms),
];

/// # Safety: see module docs. Same operation order as the CUDA kernel (no FMA contraction).
pub unsafe fn ag_mesh_terms(params: *const u8, _cfg: &LaunchConfig) {
    let p: MeshTermsParams = unsafe { read_unaligned(params as *const MeshTermsParams) };
    let n = p.n_tris as usize;
    let tris = unsafe { slice::<u32>(p.tris, 3 * n) };
    let nv = tris.iter().copied().max().map_or(0, |m| m as usize + 1);
    let pos = unsafe { slice::<f64>(p.positions, 3 * nv) };
    let out = unsafe { slice_mut::<f64>(p.out, MESH_TERMS * n) };
    for i in 0..n {
        let v = |k: usize| { let j = tris[3 * i + k] as usize; [pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]] };
        let (a, b, d) = (v(0), v(1), v(2));
        // b × d, then a · (b × d)
        let bxd = [b[1] * d[2] - b[2] * d[1], b[2] * d[0] - b[0] * d[2], b[0] * d[1] - b[1] * d[0]];
        let v6 = a[0] * bxd[0] + a[1] * bxd[1] + a[2] * bxd[2];
        let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [d[0] - a[0], d[1] - a[1], d[2] - a[2]]);
        let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let area = 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        out[i] = v6;
        out[n + i] = area;
        for k in 0..3 { out[(2 + k) * n + i] = (a[k] + b[k] + d[k]) * v6; }
    }
}
