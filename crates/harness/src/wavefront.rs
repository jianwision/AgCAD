//! GPU-resident harness wavefront (spec 06 § GPU-resident harness; review F2). Stages run as
//! uniform kernels over records/configurations; CPU references here are the oracle and the
//! host-loop path on devices without conditional graph nodes (`CondSupport::HostEmulated`).

use agcad_driver::abi::{CommandRecord, StatusRecord};
use agcad_driver::*;
use agcad_kernels::primitives;
use agcad_kernels::upload_params;
use bytemuck::{Pod, Zeroable};

pub const CONFIG_ALL: u32 = u32::MAX;

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct ValidateParams { pub cmds: u64, pub status: u64, pub allow: u64, pub budget: u64, pub revisions: u64, pub keys: u64, pub n: u32, pub configs: u32 }
#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct BinBoundsParams { pub sorted_keys: u64, pub bin_starts: u64, pub bin_count: u64, pub n: u32, pub max_bins: u32 }
#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct DecideParams { pub trial_flags: u64, pub current: u64, pub trial: u64, pub accepted: u64, pub status: u64, pub configs: u32, pub required_flags: u32 }
#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeterParams { pub used: u64, pub remaining: u64, pub active: u64, pub status: u64, pub configs: u32, pub _pad: u32 }
#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct MaskScoresParams { pub scores: u64, pub active: u64, pub flags: u64, pub masked: u64, pub configs: u32, pub required_flags: u32 }

/// Trial-flag bits written by check kernels.
pub mod flag {
    pub const NUMERIC_OK: u32 = 1 << 0;
    pub const REQUIREMENTS_OK: u32 = 1 << 1;
    pub const GEOMETRY_OK: u32 = 1 << 2;
    pub const NEEDS_EXACT: u32 = 1 << 3;
}

unsafe fn sl<'a, T>(a: u64, n: usize) -> &'a [T] { if n == 0 { &[] } else { unsafe { core::slice::from_raw_parts(a as *const T, n) } } }
#[allow(clippy::mut_from_ref)]
unsafe fn slm<'a, T>(a: u64, n: usize) -> &'a mut [T] { if n == 0 { &mut [] } else { unsafe { core::slice::from_raw_parts_mut(a as *mut T, n) } } }

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_hv_validate(params: *const u8, _: &LaunchConfig) {
    let p: ValidateParams = unsafe { core::ptr::read_unaligned(params as *const ValidateParams) };
    let n = p.n as usize;
    let cmds = unsafe { sl::<CommandRecord>(p.cmds, n) };
    let st = unsafe { slm::<StatusRecord>(p.status, n) };
    let allow = unsafe { sl::<u32>(p.allow, 2048) };
    let budget = unsafe { sl::<u32>(p.budget, p.configs.max(1) as usize) };
    let rev = unsafe { sl::<u32>(p.revisions, p.configs.max(1) as usize) };
    let keys = unsafe { slm::<u64>(p.keys, n) };
    for i in 0..n {
        let c = cmds[i];
        let s = if (allow[(c.opcode >> 5) as usize] >> (c.opcode & 31)) & 1 == 0 { AgStatus::InvalidOpcode }
            else if c.configuration != CONFIG_ALL && c.configuration >= p.configs { AgStatus::InvalidParameter }
            else {
                let cfg = if c.configuration == CONFIG_ALL { 0 } else { c.configuration as usize };
                if rev[cfg] != c.expected_revision { AgStatus::StaleRevision }
                else if budget[cfg] == 0 { AgStatus::BudgetExhausted }
                else if !c.imm[0].is_finite() || !c.imm[1].is_finite() { AgStatus::NonfiniteParameter }
                else { AgStatus::Accepted }
            };
        st[i] = StatusRecord { status: s.value(), flags: 0, dst: c.dst, budget_used: (s == AgStatus::Accepted) as u32, aux: 0 };
        keys[i] = if s == AgStatus::Accepted { ((c.opcode as u64) << 32) | i as u64 } else { u64::MAX };
    }
}

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_hv_bin_bounds(params: *const u8, _: &LaunchConfig) {
    let p: BinBoundsParams = unsafe { core::ptr::read_unaligned(params as *const BinBoundsParams) };
    let n = p.n as usize;
    let k = unsafe { sl::<u64>(p.sorted_keys, n) };
    let starts = unsafe { slm::<u32>(p.bin_starts, p.max_bins as usize) };
    let count = unsafe { slm::<u32>(p.bin_count, 1) };
    let mut rank = 0u32;
    for i in 0..n {
        if k[i] == u64::MAX { continue; }
        if i == 0 || (k[i] >> 32) != (k[i - 1] >> 32) {
            if rank < p.max_bins { starts[rank as usize] = i as u32; }
            rank += 1;
        }
    }
    if rank > 0 { count[0] = rank; }
}

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_hv_decide(params: *const u8, _: &LaunchConfig) {
    let p: DecideParams = unsafe { core::ptr::read_unaligned(params as *const DecideParams) };
    let c = p.configs as usize;
    let f = unsafe { sl::<u32>(p.trial_flags, c) };
    let cur = unsafe { slm::<u32>(p.current, c) };
    let tri = unsafe { slm::<u32>(p.trial, c) };
    let acc = unsafe { slm::<u32>(p.accepted, c) };
    let st = unsafe { slm::<u16>(p.status, c) };
    for i in 0..c {
        let ok = f[i] & p.required_flags == p.required_flags;
        acc[i] = ok as u32;
        if ok { core::mem::swap(&mut cur[i], &mut tri[i]); }
        st[i] = if ok { AgStatus::Accepted } else if f[i] & 1 == 0 { AgStatus::NumericalCheckFailed } else { AgStatus::RequirementFailed }.value();
    }
}

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_hv_meter(params: *const u8, _: &LaunchConfig) {
    let p: MeterParams = unsafe { core::ptr::read_unaligned(params as *const MeterParams) };
    let c = p.configs as usize;
    let used = unsafe { sl::<u32>(p.used, c) };
    let rem = unsafe { slm::<u32>(p.remaining, c) };
    let act = unsafe { slm::<u32>(p.active, c) };
    let st = unsafe { slm::<u16>(p.status, c) };
    for i in 0..c {
        rem[i] = rem[i].saturating_sub(used[i]);
        if rem[i] == 0 { act[i] = 0; st[i] = AgStatus::BudgetExhausted.value(); }
    }
}

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_hv_mask_scores(params: *const u8, _: &LaunchConfig) {
    let p: MaskScoresParams = unsafe { core::ptr::read_unaligned(params as *const MaskScoresParams) };
    let c = p.configs as usize;
    let s = unsafe { sl::<f64>(p.scores, c) };
    let a = unsafe { sl::<u32>(p.active, c) };
    let f = unsafe { sl::<u32>(p.flags, c) };
    let m = unsafe { slm::<f64>(p.masked, c) };
    for i in 0..c { m[i] = if a[i] != 0 && f[i] & p.required_flags == p.required_flags { s[i] } else { f64::INFINITY }; }
}

pub const HARNESS_CPU: &[(&str, CpuKernelFn)] = &[
    ("ag_hv_validate", ag_hv_validate),
    ("ag_hv_bin_bounds", ag_hv_bin_bounds),
    ("ag_hv_decide", ag_hv_decide),
    ("ag_hv_meter", ag_hv_meter),
    ("ag_hv_mask_scores", ag_hv_mask_scores),
];

pub fn harness_library() -> LibrarySource {
    LibrarySource { name: "ag_harness", cuda_source: Some(agcad_kernels_cuda::bundle(include_str!("../cu/wavefront.cu"), &[])), cpu_kernels: HARNESS_CPU, options: vec![] }
}

/// Opcode allow-list bitmap (65536 bits) from a capability manifest.
pub fn allow_bitmap(opcodes: &[u16]) -> Vec<u32> {
    let mut b = vec![0u32; 2048];
    for &o in opcodes { b[(o >> 5) as usize] |= 1 << (o & 31); }
    b
}

/// Device state of one rollout segment.
pub struct WavefrontState<B: RawBuffer> {
    pub configs: u32,
    pub revisions: B,
    pub budget: B,
    pub active: B,
    pub current: B,
    pub trial: B,
    pub cfg_status: B,
}

pub struct RoundResult {
    pub statuses: Vec<StatusRecord>,
    /// Opcode bins: (first sorted index, count).
    pub bins: Vec<(u32, u32)>,
}

/// Validate + bin one block of command records (wavefront steps 1–2). Returns statuses and
/// the opcode bin structure (one readback; used by the host-loop path and by conformance).
pub fn validate_and_bin<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, prim: LibraryId, cmds: &[CommandRecord], allow: &[u32], st: &WavefrontState<D::Buffer>) -> DriverResult<RoundResult> {
    let n = cmds.len() as u32;
    let up = |bytes: &[u8]| -> DriverResult<D::Buffer> { let b = dev.alloc_bytes(s, bytes.len().max(1))?; dev.upload_from_host_async(s, bytes, &b, 0)?; Ok(b) };
    let cb = up(bytemuck::cast_slice(cmds))?;
    let ab = up(bytemuck::cast_slice(allow))?;
    let status = dev.alloc_bytes(s, (n as usize).max(1) * 16)?;
    let keys = dev.alloc_bytes(s, (n as usize).max(1) * 8)?;
    let vals = dev.alloc_bytes(s, (n as usize).max(1) * 4)?;
    let p = upload_params(dev, s, &ValidateParams { cmds: cb.device_ptr(), status: status.device_ptr(), allow: ab.device_ptr(), budget: st.budget.device_ptr(), revisions: st.revisions.device_ptr(), keys: keys.device_ptr(), n, configs: st.configs })?;
    dev.launch(s, lib, "ag_hv_validate", &LaunchConfig::linear(n as u64, 256), &p.buffer)?;
    // values = record indices (sort carries them; keys already embed the index for stability)
    let idx: Vec<u32> = (0..n).collect();
    dev.upload_from_host_async(s, bytemuck::cast_slice(&idx), &vals, 0)?;
    primitives::radix_sort_pairs(dev, s, prim, &keys, &vals, n, 64)?;
    let max_bins = 1024u32;
    let starts = dev.alloc_zeroed::<u32>(s, max_bins as usize)?;
    let count = dev.alloc_zeroed::<u32>(s, 1)?;
    let pb = upload_params(dev, s, &BinBoundsParams { sorted_keys: keys.device_ptr(), bin_starts: starts.device_ptr(), bin_count: count.device_ptr(), n, max_bins })?;
    dev.launch(s, lib, "ag_hv_bin_bounds", &LaunchConfig::linear(n as u64, 256), &pb.buffer)?;
    let total = 16 * n as usize + 4 * max_bins as usize + 4 + 8 * n as usize;
    let mut pin = dev.pinned(total)?;
    let ev = dev.download_packed_async(s, &[
        Region { buf: &status, offset: 0, bytes: 16 * n as usize },
        Region { buf: starts.raw(), offset: 0, bytes: 4 * max_bins as usize },
        Region { buf: count.raw(), offset: 0, bytes: 4 },
        Region { buf: &keys, offset: 0, bytes: 8 * n as usize },
    ], &mut pin)?;
    dev.sync(s)?;
    let _ = dev.query(&ev)?;
    let bytes = pin.as_bytes();
    let statuses: Vec<StatusRecord> = bytemuck::pod_collect_to_vec(&bytes[..16 * n as usize]);
    let o = 16 * n as usize;
    let starts_h: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes[o..o + 4 * max_bins as usize]);
    let nb = u32::from_le_bytes(bytes[o + 4 * max_bins as usize..o + 4 * max_bins as usize + 4].try_into().unwrap()) as usize;
    let ko = o + 4 * max_bins as usize + 4;
    let sorted: Vec<u64> = bytemuck::pod_collect_to_vec(&bytes[ko..ko + 8 * n as usize]);
    let valid = sorted.iter().filter(|k| **k != u64::MAX).count() as u32;
    let mut bins = Vec::with_capacity(nb);
    for b in 0..nb.min(max_bins as usize) {
        let start = starts_h[b];
        let end = if b + 1 < nb { starts_h[b + 1] } else { valid };
        bins.push((start, end - start));
    }
    Ok(RoundResult { statuses, bins })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_driver_cpu::CpuDevice;

    fn lib(dev: &CpuDevice) -> (LibraryId, LibraryId) {
        let h = dev.load_library(&LibrarySource { name: "ag_harness", cuda_source: None, cpu_kernels: HARNESS_CPU, options: vec![] }).unwrap();
        let p = dev.load_library(&LibrarySource { name: agcad_kernels::PRIMITIVES, cuda_source: None, cpu_kernels: agcad_kernels::reference::PRIMITIVES_CPU, options: vec![] }).unwrap();
        (h, p)
    }

    #[test]
    fn validate_rejects_and_bins_by_opcode() {
        let dev = CpuDevice::new();
        let s = dev.stream(Priority::High).unwrap();
        let (h, p) = lib(&dev);
        let configs = 4u32;
        let up = |v: &[u32]| { let b = dev.alloc_bytes(&s, v.len() * 4).unwrap(); dev.upload_from_host_async(&s, bytemuck::cast_slice(v), &b, 0).unwrap(); b };
        let st = WavefrontState { configs, revisions: up(&[7, 7, 7, 7]), budget: up(&[10, 10, 0, 10]), active: up(&[1, 1, 1, 1]), current: up(&[0, 1, 2, 3]), trial: up(&[4, 5, 6, 7]), cfg_status: up(&[0, 0, 0, 0]) };
        let mk = |op: u16, cfg: u32, rev: u32| { let mut c: CommandRecord = Zeroable::zeroed(); c.opcode = op; c.configuration = cfg; c.expected_revision = rev; c };
        let cmds = vec![mk(0x0700, 0, 7), mk(0x0E02, 1, 7), mk(0x0700, 3, 7), mk(0x0700, 2, 7), mk(0x9999, 0, 7), mk(0x0700, 1, 6)];
        let allow = allow_bitmap(&[0x0700, 0x0E02]);
        let r = validate_and_bin(&*dev, &s, h, p, &cmds, &allow, &st).unwrap();
        let codes: Vec<u16> = r.statuses.iter().map(|x| x.status).collect();
        assert_eq!(codes, vec![0, 0, 0, AgStatus::BudgetExhausted.value(), AgStatus::InvalidOpcode.value(), AgStatus::StaleRevision.value()]);
        assert_eq!(r.bins, vec![(0, 2), (2, 1)]); // 0x0700 ×2 (records 0, 2) then 0x0E02 ×1
    }
}
