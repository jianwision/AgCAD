//! CPU reference kernels of the device operation layer (conformance oracle, AG-08-005). Each
//! function has a twin in `cu/dops.cu` with the same operation order; integer results (cell
//! counts, bounding boxes, statuses) and every f64 written are bit-identical.
//!
//! Round pipeline (spec 06 wavefront): policy → validate (harness) → apply → eval_clear →
//! eval → decide → continue. Safety contract: `params` points at a [`DopsArgs`] whose
//! addresses are valid host buffers of the documented sizes (CPU reference device).

use crate::abi::*;
use crate::math::*;
use agcad_driver::abi::{CommandRecord, StatusRecord};
use agcad_driver::{AgStatus, CpuKernelFn, LaunchConfig};
use bytemuck::Zeroable;

unsafe fn sl<'a, T>(a: u64, n: usize) -> &'a [T] { if n == 0 { &[] } else { unsafe { core::slice::from_raw_parts(a as *const T, n) } } }
#[allow(clippy::mut_from_ref)]
unsafe fn slm<'a, T>(a: u64, n: usize) -> &'a mut [T] { if n == 0 { &mut [] } else { unsafe { core::slice::from_raw_parts_mut(a as *mut T, n) } } }
unsafe fn args(p: *const u8) -> DopsArgs { unsafe { core::ptr::read_unaligned(p as *const DopsArgs) } }

const ACCEPTED: u16 = AgStatus::Accepted.value();

fn nop(cfg: u32, rev: u32, update: u32) -> CommandRecord {
    let mut c = CommandRecord::zeroed();
    c.opcode = update as u16;
    c.configuration = cfg;
    c.expected_revision = rev;
    c
}

/// Mock GPU model: a deterministic goal-seeking controller that writes command records and
/// arg-arena payloads straight into the device ring (stand-in for a learned policy; same
/// inputs and outputs). One thread per configuration.
pub unsafe fn ag_dop_policy(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, rn, kn) = (a.configs as usize, a.records as usize, a.params as usize);
    let g: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let rev = unsafe { sl::<u32>(a.rev, cn) };
    let budget = unsafe { sl::<u32>(a.budget, cn) };
    let done = unsafe { sl::<u32>(a.done, cn) };
    let obs = unsafe { sl::<f64>(a.obs, cn * OBS) };
    let pr = unsafe { sl::<f64>(a.cur_params, cn * kn) };
    let count = unsafe { sl::<u32>(a.cur_count, cn) };
    let cmds = unsafe { slm::<CommandRecord>(a.cmds, cn * rn) };
    let arena = unsafe { slm::<FeatureArgs>(a.arena, cn * rn) };
    let round = unsafe { sl::<u32>(a.round, 1)[0] };
    for cfg in 0..cn {
        let base = cfg * rn;
        let gcfg = cfg as u32 + a.cfg_offset;
        for s in 0..rn {
            cmds[base + s] = nop(cfg as u32, rev[cfg], a.ops.update);
            arena[base + s] = FeatureArgs::zeroed();
        }
        if done[cfg] != 0 || budget[cfg] == 0 { continue; }
        let np = (g.n_params as usize).min(kn).min(MAX_PARAMS);
        if np == 0 { continue; }
        if round == 0 {
            // exploration: configuration 0 keeps the base design, the others sample the box
            if gcfg != 0 {
                for j in 0..np.min(rn) {
                    let v = g.param_lo[j] + unit(hash3(g.seed, gcfg, j as u32)) * (g.param_hi[j] - g.param_lo[j]);
                    let c = &mut cmds[base + j];
                    c.opcode = a.ops.parameter as u16;
                    c.args[0] = j as u32;
                    c.args[1] = rec::PARAM_SET;
                    c.imm[0] = v;
                }
            }
            continue;
        }
        let o = &obs[cfg * OBS..cfg * OBS + OBS];
        let t = 0.5 * (g.vol_lo + g.vol_hi);
        let err = (o[12] - t) / t;
        let mut slot = 0usize;
        let acts = if np >= 2 { 2 } else { 1 };
        for act in 0..acts {
            if slot >= rn { break; }
            let j = (gcfg as usize + round as usize + act) % np;
            let (lo, hi) = (g.param_lo[j], g.param_hi[j]);
            let cur = pr[cfg * kn + j];
            let d = -err * g.param_sign[j] * g.gain * (hi - lo);
            let mut nv = cur + d;
            if nv < lo { nv = lo; }
            if nv > hi { nv = hi; }
            if (nv - cur).abs() <= 1e-12 * (hi - lo) {
                // parameter saturated: structural action (remove material with a new hole)
                if err > 0.0 && act == 0 && count[cfg] < a.nodes && g.hole_r > 0.0 {
                    let ux = unit(hash3(g.seed ^ 0xA5A5_A5A5, gcfg, round));
                    let uy = unit(hash3(g.seed ^ 0x5A5A_5A5A, gcfg, round));
                    let (px, py) = (g.place[0] + ux * (g.place[2] - g.place[0]), g.place[1] + uy * (g.place[3] - g.place[1]));
                    let rect = (gcfg + round) % 2 == 1 && slot + 1 < rn;
                    let c = &mut cmds[base + slot];
                    c.opcode = a.ops.pocket as u16;
                    let f = &mut arena[base + slot];
                    f.pref = [NO_PARAM; NVALS];
                    if rect {
                        // rounded rectangular pocket: part.pocket (rectangle) + part.edge_fillet
                        c.args[0] = profile::RECT;
                        let h = g.hole_r;
                        f.val = [px - h, py - h, px + h, py + h, g.hole_z0, g.hole_z1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                        slot += 1;
                        let c2 = &mut cmds[base + slot];
                        c2.opcode = a.ops.fillet as u16;
                        c2.args[0] = rec::LAST;
                        let f2 = &mut arena[base + slot];
                        f2.pref = [NO_PARAM; NVALS];
                        f2.val[10] = 0.4 * h;
                    } else {
                        c.args[0] = profile::CIRCLE;
                        f.val = [px, py, g.hole_r, 0.0, g.hole_z0, g.hole_z1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    }
                    slot += 1;
                }
                continue;
            }
            let c = &mut cmds[base + slot];
            c.opcode = a.ops.parameter as u16;
            c.args[0] = j as u32;
            c.args[1] = rec::PARAM_SET;
            c.imm[0] = nv;
            slot += 1;
        }
    }
}

/// EXECUTE: per configuration, copy current → trial and apply its accepted records in record
/// order (edits are tiny writes; the uniform, heavy work is evaluation).
pub unsafe fn ag_dop_apply(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn, kn, rn) = (a.configs as usize, a.nodes as usize, a.params as usize, a.records as usize);
    let cur_n = unsafe { sl::<Node>(a.cur_nodes, cn * nn) };
    let cur_c = unsafe { sl::<u32>(a.cur_count, cn) };
    let cur_p = unsafe { sl::<f64>(a.cur_params, cn * kn) };
    let tr_n = unsafe { slm::<Node>(a.tr_nodes, cn * nn) };
    let tr_c = unsafe { slm::<u32>(a.tr_count, cn) };
    let tr_p = unsafe { slm::<f64>(a.tr_params, cn * kn) };
    let cmds = unsafe { sl::<CommandRecord>(a.cmds, cn * rn) };
    let arena = unsafe { sl::<FeatureArgs>(a.arena, cn * rn) };
    let status = unsafe { slm::<StatusRecord>(a.status, cn * rn) };
    for cfg in 0..cn {
        for i in 0..nn { tr_n[cfg * nn + i] = cur_n[cfg * nn + i]; }
        for k in 0..kn { tr_p[cfg * kn + k] = cur_p[cfg * kn + k]; }
        let mut count = cur_c[cfg];
        for r in 0..rn {
            let idx = cfg * rn + r;
            if status[idx].status != ACCEPTED { continue; }
            let c = cmds[idx];
            let op = c.opcode as u32;
            let mut st = ACCEPTED;
            if op == a.ops.update {
            } else if op == a.ops.parameter {
                let k = c.args[0] as usize;
                if k >= kn { st = AgStatus::InvalidParameter.value(); } else {
                    let v = if c.args[1] == rec::PARAM_ADD { tr_p[cfg * kn + k] + c.imm[0] } else { c.imm[0] };
                    tr_p[cfg * kn + k] = v;
                }
            } else if op == a.ops.pad || op == a.ops.pocket || op == a.ops.hole {
                let f = arena[idx];
                let bad_ref = f.pref.iter().any(|&p| p != NO_PARAM && p as usize >= kn);
                if c.args[0] > profile::POLYGON || bad_ref { st = AgStatus::InvalidParameter.value(); }
                else if count as usize >= nn { st = AgStatus::CapacityExceeded.value(); }
                else {
                    tr_n[cfg * nn + count as usize] = Node { kind: c.args[0], op: if op == a.ops.pad { fop::ADD } else { fop::REMOVE }, flags: nflags::ACTIVE, src: r as u32, pref: f.pref, val: f.val };
                    count += 1;
                }
            } else if op == a.ops.shell || op == a.ops.body || op == a.ops.add || op == a.ops.remove || op == a.ops.intersect {
                // structural nodes: body stack depth checked against the trial tape
                let mut depth = 0i32;
                for k in 0..count as usize {
                    let n = &tr_n[cfg * nn + k];
                    if n.flags & nflags::ACTIVE == 0 { continue; }
                    if n.kind == profile::BODY { depth += 1; } else if n.kind == profile::COMBINE { depth -= 1; }
                }
                let f = arena[idx];
                if count as usize >= nn { st = AgStatus::CapacityExceeded.value(); }
                else if op == a.ops.body && depth as usize >= MAX_BODIES { st = AgStatus::CapacityExceeded.value(); }
                else if (op == a.ops.add || op == a.ops.remove || op == a.ops.intersect) && depth <= 0 { st = AgStatus::InvalidParameter.value(); }
                else if op == a.ops.shell && f.pref[0] != NO_PARAM && f.pref[0] as usize >= kn { st = AgStatus::InvalidParameter.value(); }
                else {
                    let (kind, fo) = if op == a.ops.shell { (profile::SHELL, fop::ADD) } else if op == a.ops.body { (profile::BODY, fop::ADD) } else if op == a.ops.add { (profile::COMBINE, fop::ADD) } else if op == a.ops.remove { (profile::COMBINE, fop::REMOVE) } else { (profile::COMBINE, fop::INTERSECT) };
                    let mut nd = Node { kind, op: fo, flags: nflags::ACTIVE, src: r as u32, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
                    if op == a.ops.shell { nd.pref[0] = f.pref[0]; nd.val[0] = f.val[0]; }
                    tr_n[cfg * nn + count as usize] = nd;
                    count += 1;
                }
            } else if op == a.ops.circular_pattern || op == a.ops.rect_pattern || op == a.ops.fillet || op == a.ops.chamfer || op == a.ops.deactivate {
                let fi = if c.args[0] == rec::LAST { count.wrapping_sub(1) } else { c.args[0] };
                if count == 0 || fi >= count { st = AgStatus::InvalidParameter.value(); } else {
                    let n = &mut tr_n[cfg * nn + fi as usize];
                    let f = arena[idx];
                    // value slots this op writes
                    let (lo, hi) = if op == a.ops.circular_pattern { (6, 8) } else if op == a.ops.rect_pattern { (6, 10) } else if op == a.ops.deactivate { (0, 0) } else { (10, 11) };
                    let bad_ref = (lo..hi).any(|j| f.pref[j] != NO_PARAM && f.pref[j] as usize >= kn);
                    let is_profile_op = op == a.ops.fillet || op == a.ops.chamfer;
                    if bad_ref || (is_profile_op && n.kind != profile::RECT) { st = if bad_ref { AgStatus::InvalidParameter.value() } else { AgStatus::UnsupportedOp.value() }; }
                    else if op == a.ops.deactivate { if c.args[1] != 0 { n.flags |= nflags::ACTIVE; } else { n.flags &= !nflags::ACTIVE; } }
                    else {
                        for j in lo..hi { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                        if op == a.ops.circular_pattern { n.flags = (n.flags & !nflags::PATTERN_RECT) | nflags::PATTERN; }
                        else if op == a.ops.rect_pattern { n.flags = (n.flags & !nflags::PATTERN) | nflags::PATTERN_RECT; }
                        else if op == a.ops.fillet { n.flags = (n.flags & !nflags::CHAMFER) | nflags::FILLET; }
                        else { n.flags = (n.flags & !nflags::FILLET) | nflags::CHAMFER; }
                    }
                }
            } else {
                st = AgStatus::UnsupportedOp.value();
            }
            status[idx].status = st;
        }
        tr_c[cfg] = count;
    }
}

/// COMPILE: resolve parameters and precompute trig tables of the tapes to be evaluated
/// (current for observe, trial for decide). One thread per configuration.
pub unsafe fn ag_dop_compile(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn, kn) = (a.configs as usize, a.nodes as usize, a.params as usize);
    let (nodes, cnt, prm) = if a.eval_trial != 0 { (a.tr_nodes, a.tr_count, a.tr_params) } else { (a.cur_nodes, a.cur_count, a.cur_params) };
    let nodes = unsafe { sl::<Node>(nodes, cn * nn) };
    let cnt = unsafe { sl::<u32>(cnt, cn) };
    let prm = unsafe { sl::<f64>(prm, cn * kn) };
    let done = unsafe { sl::<u32>(a.done, cn) };
    let out = unsafe { slm::<CNode>(a.compiled, cn * nn) };
    let tabs = unsafe { slm::<f32>(a.tables, cn * nn * TABLE) };
    for c in 0..cn {
        if a.eval_trial != 0 && done[c] != 0 { continue; }
        for i in 0..cnt[c] as usize {
            let t = &mut tabs[(c * nn + i) * TABLE..(c * nn + i + 1) * TABLE];
            out[c * nn + i] = compile_node(&nodes[c * nn + i], &prm[c * kn..c * kn + kn], t);
        }
    }
}

/// Reset per-configuration evaluation accumulators.
pub unsafe fn ag_dop_eval_clear(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cn = a.configs as usize;
    let counts = unsafe { slm::<u64>(a.counts, 4 * cn) };
    let dom = unsafe { slm::<u32>(a.domain_hit, cn) };
    let bb = unsafe { slm::<u32>(a.bbox, 12 * cn) };
    for c in 0..cn {
        for j in 0..4 { counts[4 * c + j] = 0; }
        dom[c] = 0;
        for k in 0..3 { bb[12 * c + k] = u32::MAX; bb[12 * c + 3 + k] = 0; bb[12 * c + 6 + k] = u32::MAX; bb[12 * c + 9 + k] = 0; }
    }
}

/// Evaluation grid: domain, root resolution, levels; volumes are counted in finest cells.
#[derive(Clone, Copy)]
pub struct Grid { pub lo: [f64; 3], pub size: [f64; 3], pub fine: [f64; 3], pub eps: f64, pub r: [u32; 3], pub levels: u32 }

/// Root cells per axis: `r0` on every axis. (Measured: for plate-like parts the cells that
/// are thin along the short axis bound the dominant faces best; near-cubic cells were 3×
/// wider. Orientation-aware anisotropic splitting is the planned improvement.)
pub fn grid(goal: &Goal, r0: u32, levels: u32) -> Grid {
    let size = [goal.domain_hi[0] - goal.domain_lo[0], goal.domain_hi[1] - goal.domain_lo[1], goal.domain_hi[2] - goal.domain_lo[2]];
    let r = [r0; 3];
    let fine = [size[0] / (r[0] << (levels - 1)) as f64, size[1] / (r[1] << (levels - 1)) as f64, size[2] / (r[2] << (levels - 1)) as f64];
    // f32 evaluation error bound: ~40 rounded ops on coordinates ≤ the domain's reach
    let reach = goal.domain_lo.iter().chain(&goal.domain_hi).fold(0.0f64, |m, v| m.max(v.abs())) + (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt();
    let eps = 4.0e-6 * reach + 1e-6;
    Grid { lo: goal.domain_lo, size, fine, eps, r, levels }
}

/// Cell box at `level` as f32 intervals (corners computed in f64, then rounded).
pub fn cell_box(gr: &Grid, level: u32, ix: u32, iy: u32, iz: u32) -> [Iv; 3] {
    let idx = [ix, iy, iz];
    let mut b = [iv(0.0, 0.0); 3];
    for k in 0..3 {
        let w = gr.size[k] / (gr.r[k] << level) as f64;
        b[k] = iv((gr.lo[k] + idx[k] as f64 * w) as f32, (gr.lo[k] + (idx[k] + 1) as f64 * w) as f32);
    }
    b
}

/// Fill fraction of a boundary cell (cell dims `d`), quantized to 2⁻²⁰: 0.5 − f / w with w
/// the cell's width along the SDF gradient. Exact for axis-aligned planar faces.
pub fn fill_q(nodes: &[CNode], tabs: &[f32], x: f32, y: f32, z: f32, d: [f32; 3]) -> u64 {
    let f = tape_sdf_c(nodes, tabs, x, y, z);
    let h = 0.25f32 * d[0].min(d[1]).min(d[2]);
    let gx = tape_sdf_c(nodes, tabs, x + h, y, z) - tape_sdf_c(nodes, tabs, x - h, y, z);
    let gy = tape_sdf_c(nodes, tabs, x, y + h, z) - tape_sdf_c(nodes, tabs, x, y - h, z);
    let gz = tape_sdf_c(nodes, tabs, x, y, z + h) - tape_sdf_c(nodes, tabs, x, y, z - h);
    let gl = (gx * gx + gy * gy + gz * gz).sqrt();
    let mut frac = 0.5f32;
    if gl > 1e-30 {
        let w = (gx.abs() * d[0] + gy.abs() * d[1] + gz.abs() * d[2]) / gl;
        frac = 0.5 - f / w;
        if frac < 0.0 { frac = 0.0; }
        if frac > 1.0 { frac = 1.0; }
    }
    (frac as f64 * 1048576.0 + 0.5).floor() as u64
}

fn decode(e: u64) -> (usize, u32, u32, u32) { ((e >> 32) as usize, ((e >> 20) & 1023) as u32, ((e >> 10) & 1023) as u32, (e & 1023) as u32) }
fn encode(cfg: u32, ix: u32, iy: u32, iz: u32) -> u64 { ((cfg as u64) << 32) | ((ix as u64) << 20) | ((iy as u64) << 10) | iz as u64 }

struct Acc<'a> { counts: &'a mut [u64], dom: &'a mut [u32], bb: &'a mut [u32] }

impl Acc<'_> {
    fn bbox(&mut self, cfg: usize, lo: [u32; 3], n: u32, inside: bool) {
        for k in 0..3 {
            if inside { self.bb[12 * cfg + k] = self.bb[12 * cfg + k].min(lo[k]); self.bb[12 * cfg + 3 + k] = self.bb[12 * cfg + 3 + k].max(lo[k] + n); }
            self.bb[12 * cfg + 6 + k] = self.bb[12 * cfg + 6 + k].min(lo[k]);
            self.bb[12 * cfg + 9 + k] = self.bb[12 * cfg + 9 + k].max(lo[k] + n);
        }
    }
}

/// Does the solid reach the domain face(s) that cell (level, idx) lies on? Inside cells at the
/// border always do; boundary cells test the interval over each border face patch.
fn border_hit(gr: &Grid, nodes: &[CNode], tabs: &[f32], level: u32, idx: [u32; 3], b: &[Iv; 3], inside: bool) -> bool {
    for ax in 0..3 {
        let res = gr.r[ax] << level;
        for side in 0..2u32 {
            if (side == 0 && idx[ax] != 0) || (side == 1 && idx[ax] != res - 1) { continue; }
            if inside { return true; }
            let mut f = *b;
            let face = if side == 0 { b[ax].lo } else { b[ax].hi };
            f[ax] = iv(face, face);
            let r = tape_iv(nodes, tabs, f[0], f[1], f[2]);
            if r.lo - gr.eps as f32 <= 0.0 { return true; }
        }
    }
    false
}

/// Level-0 worklist: every root cell of every configuration.
pub unsafe fn ag_ad_seed(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let r = grid(&goal, a.coarse, a.levels).r;
    let per = (r[0] * r[1] * r[2]) as usize;
    let n = a.configs as usize * per;
    let wl = unsafe { slm::<u64>(a.wl0, n) };
    for (i, e) in wl.iter_mut().enumerate() {
        let (cfg, cell) = ((i / per) as u32, (i % per) as u32);
        *e = encode(cfg, cell % r[0], (cell / r[0]) % r[1], cell / (r[0] * r[1]));
    }
    unsafe { slm::<u32>(a.lvl, a.levels as usize + 1)[0] = n as u32; }
}

/// CLASSIFY (level l): interval over each cell → outside / inside (counted) / ambiguous.
pub unsafe fn ag_ad_classify(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn) = (a.configs as usize, a.nodes as usize);
    let cnt = unsafe { sl::<u32>(if a.eval_trial != 0 { a.tr_count } else { a.cur_count }, cn) };
    let cnodes = unsafe { sl::<CNode>(a.compiled, cn * nn) };
    let tabs = unsafe { sl::<f32>(a.tables, cn * nn * TABLE) };
    let done = unsafe { sl::<u32>(a.done, cn) };
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let gr = grid(&goal, a.coarse, a.levels);
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let cls = unsafe { slm::<u32>(a.cls, a.capacity as usize) };
    for x in cls[n..].iter_mut() { *x = 0; }
    let mut acc = Acc { counts: unsafe { slm::<u64>(a.counts, 4 * cn) }, dom: unsafe { slm::<u32>(a.domain_hit, cn) }, bb: unsafe { slm::<u32>(a.bbox, 12 * cn) } };
    let sh = a.levels - 1 - l;
    let e = gr.eps as f32;
    for i in 0..n {
        let (cfg, ix, iy, iz) = decode(wl[i]);
        cls[i] = 0;
        if a.eval_trial != 0 && done[cfg] != 0 { continue; }
        let tape = &cnodes[cfg * nn..cfg * nn + cnt[cfg] as usize];
        let tb = &tabs[cfg * nn * TABLE..(cfg * nn + cnt[cfg] as usize) * TABLE];
        let b = cell_box(&gr, l, ix, iy, iz);
        let r = tape_iv(tape, tb, b[0], b[1], b[2]);
        if r.lo - e > 0.0 { continue; }
        if r.hi + e < 0.0 {
            acc.counts[4 * cfg] += 1u64 << (3 * sh);
            acc.bbox(cfg, [ix << sh, iy << sh, iz << sh], 1 << sh, true);
            if border_hit(&gr, tape, tb, l, [ix, iy, iz], &b, true) { acc.dom[cfg] = 1; }
            continue;
        }
        cls[i] = 1;
    }
}

/// RESOLVE (level l): refine ambiguous cells while the configuration is within its cap and
/// levels remain; otherwise count them as boundary (with fill estimate and border test).
pub unsafe fn ag_ad_resolve(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn) = (a.configs as usize, a.nodes as usize);
    let cnt = unsafe { sl::<u32>(if a.eval_trial != 0 { a.tr_count } else { a.cur_count }, cn) };
    let cnodes = unsafe { sl::<CNode>(a.compiled, cn * nn) };
    let tabs = unsafe { sl::<f32>(a.tables, cn * nn * TABLE) };
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let gr = grid(&goal, a.coarse, a.levels);
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let cap = a.capacity as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let cls = unsafe { sl::<u32>(a.cls, n) };
    let flag = unsafe { slm::<u32>(a.flag, cap) };
    let crank = unsafe { sl::<u32>(a.crank, cap) };
    let start = unsafe { sl::<u32>(a.start, cn) };
    let mut acc = Acc { counts: unsafe { slm::<u64>(a.counts, 4 * cn) }, dom: unsafe { slm::<u32>(a.domain_hit, cn) }, bb: unsafe { slm::<u32>(a.bbox, 12 * cn) } };
    let sh = a.levels - 1 - l;
    for i in 0..cap {
        flag[i] = 0;
        if i >= n || cls[i] != 1 { continue; }
        let (cfg, ix, iy, iz) = decode(wl[i]);
        let rank = crank[i] - crank[start[cfg] as usize];
        if l + 1 < a.levels && (rank as u64 + 1) * 8 <= a.cap as u64 { flag[i] = 1; continue; }
        let tape = &cnodes[cfg * nn..cfg * nn + cnt[cfg] as usize];
        let tb = &tabs[cfg * nn * TABLE..(cfg * nn + cnt[cfg] as usize) * TABLE];
        let b = cell_box(&gr, l, ix, iy, iz);
        let w = 1u64 << (3 * sh);
        acc.counts[4 * cfg + 1] += w;
        let d = [b[0].hi - b[0].lo, b[1].hi - b[1].lo, b[2].hi - b[2].lo];
        let c = [0.5f32 * (b[0].lo + b[0].hi), 0.5f32 * (b[1].lo + b[1].hi), 0.5f32 * (b[2].lo + b[2].hi)];
        acc.counts[4 * cfg + 2] += fill_q(tape, tb, c[0], c[1], c[2], d) * w;
        acc.bbox(cfg, [ix << sh, iy << sh, iz << sh], 1 << sh, false);
        if border_hit(&gr, tape, tb, l, [ix, iy, iz], &b, false) { acc.dom[cfg] = 1; }
    }
}

/// START (level l): first worklist index of every configuration (the list is sorted by it).
pub unsafe fn ag_ad_start(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let start = unsafe { slm::<u32>(a.start, a.configs as usize) };
    for i in 0..n {
        let c = (wl[i] >> 32) as usize;
        if i == 0 || (wl[i - 1] >> 32) as usize != c { start[c] = i as u32; }
    }
}

/// EMIT (level l): write the 8 children of every refined cell at its scan offset.
pub unsafe fn ag_ad_emit(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let cap = a.capacity as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let out = unsafe { slm::<u64>(if l % 2 == 0 { a.wl1 } else { a.wl0 }, cap) };
    let flag = unsafe { sl::<u32>(a.flag, cap) };
    let offs = unsafe { sl::<u32>(a.offs, cap) };
    for i in 0..n.min(cap) {
        if flag[i] == 0 { continue; }
        let (cfg, ix, iy, iz) = decode(wl[i]);
        let base = offs[i] as usize * 8;
        for k in 0..8u32 { out[base + k as usize] = encode(cfg as u32, 2 * ix + (k & 1), 2 * iy + ((k >> 1) & 1), 2 * iz + (k >> 2)); }
    }
}

/// TOTAL (level l): next level's entry count from the scan.
pub unsafe fn ag_ad_total(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cap = a.capacity as usize;
    let flag = unsafe { sl::<u32>(a.flag, cap) };
    let offs = unsafe { sl::<u32>(a.offs, cap) };
    let lvl = unsafe { slm::<u32>(a.lvl, a.levels as usize + 1) };
    lvl[a.level as usize + 1] = (offs[cap - 1] + flag[cap - 1]) * 8;
}

/// Bounds of one configuration from its accumulators: (vol_lo, vol_hi, vol_est, ext_lo,
/// ext_hi). The interval is rigorous (bounded evidence); the estimate is sampled.
pub fn bounds(gr: &Grid, counts: &[u64], bb: &[u32]) -> (f64, f64, f64, [f64; 3], [f64; 3]) {
    let vc = gr.fine[0] * gr.fine[1] * gr.fine[2];
    let vol_lo = counts[0] as f64 * vc;
    let vol_hi = (counts[0] + counts[1]) as f64 * vc;
    let vol_est = (counts[0] as f64 + counts[2] as f64 * (1.0 / 1048576.0)) * vc;
    let mut el = [0.0; 3];
    let mut eh = [0.0; 3];
    for k in 0..3 {
        if bb[3 + k] > bb[k] { el[k] = (bb[3 + k] - bb[k]) as f64 * gr.fine[k]; }
        if bb[9 + k] > bb[6 + k] { eh[k] = (bb[9 + k] - bb[6 + k]) as f64 * gr.fine[k]; }
    }
    (vol_lo, vol_hi, vol_est, el, eh)
}

/// CHECK + DECIDE (eval_trial = 1) or OBSERVE (eval_trial = 0), one thread per configuration:
/// interval checks against the goal and hard limits, transactional commit of the trial tape,
/// budget metering, observation row, ranking score and trace row.
pub unsafe fn ag_dop_decide(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn, kn, rn) = (a.configs as usize, a.nodes as usize, a.params as usize, a.records as usize);
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let gr = grid(&goal, a.coarse, a.levels);
    let counts = unsafe { sl::<u64>(a.counts, 4 * cn) };
    let dom = unsafe { sl::<u32>(a.domain_hit, cn) };
    let bb = unsafe { sl::<u32>(a.bbox, 12 * cn) };
    let cmds = unsafe { sl::<CommandRecord>(a.cmds, cn * rn) };
    let status = unsafe { slm::<StatusRecord>(a.status, cn * rn) };
    let cur_n = unsafe { slm::<Node>(a.cur_nodes, cn * nn) };
    let cur_c = unsafe { slm::<u32>(a.cur_count, cn) };
    let cur_p = unsafe { slm::<f64>(a.cur_params, cn * kn) };
    let tr_n = unsafe { sl::<Node>(a.tr_nodes, cn * nn) };
    let tr_c = unsafe { sl::<u32>(a.tr_count, cn) };
    let tr_p = unsafe { sl::<f64>(a.tr_params, cn * kn) };
    let rev = unsafe { slm::<u32>(a.rev, cn) };
    let budget = unsafe { slm::<u32>(a.budget, cn) };
    let done = unsafe { slm::<u32>(a.done, cn) };
    let obs = unsafe { slm::<f64>(a.obs, cn * OBS) };
    let score = unsafe { slm::<f64>(a.score, cn) };
    let round = unsafe { sl::<u32>(a.round, 1)[0] };
    let trace = unsafe { slm::<TraceRow>(a.trace, a.max_rounds as usize * cn) };
    let t = 0.5 * (goal.vol_lo + goal.vol_hi);
    for c in 0..cn {
        if a.eval_trial != 0 && done[c] != 0 { continue; }
        let (vl, vh, ve, el, eh) = bounds(&gr, &counts[4 * c..4 * c + 4], &bb[12 * c..12 * c + 12]);
        let empty = counts[4 * c] + counts[4 * c + 1] == 0;
        let hard = el[0] > goal.max_ext[0] || el[1] > goal.max_ext[1] || el[2] > goal.max_ext[2];
        let mut write = a.eval_trial == 0;
        if a.eval_trial != 0 {
            let (mut acc, mut rej) = (0u32, 0u32);
            for r in 0..rn {
                let i = c * rn + r;
                if cmds[i].opcode as u32 == a.ops.update { continue; }
                if status[i].status == ACCEPTED { acc += 1; } else { rej += 1; }
            }
            let geometry_ok = !empty && dom[c] == 0 && !hard;
            let commit = geometry_ok && acc > 0;
            if !geometry_ok && acc > 0 {
                let code = if empty { AgStatus::TopologyFailed } else if dom[c] != 0 { AgStatus::CapacityExceeded } else { AgStatus::RequirementFailed }.value();
                for r in 0..rn {
                    let i = c * rn + r;
                    if cmds[i].opcode as u32 != a.ops.update && status[i].status == ACCEPTED { status[i].status = code; }
                }
            }
            if commit {
                for i in 0..nn { cur_n[c * nn + i] = tr_n[c * nn + i]; }
                for k in 0..kn { cur_p[c * kn + k] = tr_p[c * kn + k]; }
                cur_c[c] = tr_c[c];
                rev[c] = rev[c].wrapping_add(1);
                budget[c] = if budget[c] > acc { budget[c] - acc } else { 0 };
                write = true;
            }
            let o = &obs[c * OBS..c * OBS + OBS];
            let (ol, oh) = if write { (vl, vh) } else { (o[0], o[1]) };
            if (round as usize) < a.max_rounds as usize {
                trace[round as usize * cn + c] = TraceRow { round, committed: commit as u32, accepted: if commit { acc } else { 0 }, rejected: if commit { rej } else { rej + acc }, vol_lo: ol, vol_hi: oh };
            }
        }
        if write {
            let o = &mut obs[c * OBS..c * OBS + OBS];
            o[0] = vl; o[1] = vh;
            o[2] = el[0]; o[3] = el[1]; o[4] = el[2];
            o[5] = eh[0]; o[6] = eh[1]; o[7] = eh[2];
            o[8] = if dom[c] != 0 { 1.0 } else { 0.0 };
            o[9] = cur_c[c] as f64;
            // goal: sampled volume estimate in the target interval; extents bounded (rigorous)
            let met = !empty && dom[c] == 0 && ve >= goal.vol_lo && ve <= goal.vol_hi && eh[0] <= goal.max_ext[0] && eh[1] <= goal.max_ext[1] && eh[2] <= goal.max_ext[2];
            o[10] = if met { 1.0 } else { 0.0 };
            let err = (ve - t) / t;
            o[11] = err;
            o[12] = ve;
            if met { done[c] = 1; }
            score[c] = if met { err.abs() } else { 1000.0 + err.abs() };
        }
    }
}

/// Loop control: advance the round, decide whether another iteration is needed and (CUDA)
/// set the enclosing WHILE node's condition. Single thread.
pub unsafe fn ag_dop_continue(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cn = a.configs as usize;
    let done = unsafe { sl::<u32>(a.done, cn) };
    let budget = unsafe { sl::<u32>(a.budget, cn) };
    let round = unsafe { slm::<u32>(a.round, 1) };
    let active = unsafe { slm::<u32>(a.active, 1) };
    let mut any = 0u32;
    for c in 0..cn { if done[c] == 0 && budget[c] > 0 { any = 1; } }
    let next = round[0] + 1;
    round[0] = next;
    active[0] = if any != 0 && next < a.max_rounds { 1 } else { 0 };
}

/// Timestamp of a stage boundary (CPU: monotonic ns since first use; CUDA: %globaltimer).
pub unsafe fn ag_dop_stamp(params: *const u8, _: &LaunchConfig) {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let p: StampParams = unsafe { core::ptr::read_unaligned(params as *const StampParams) };
    let round = unsafe { sl::<u32>(p.round, 1)[0] };
    if round >= p.max_rounds { return; }
    let st = unsafe { slm::<u64>(p.stamps, p.max_rounds as usize * STAGES) };
    st[round as usize * STAGES + p.stage as usize] = T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64;
}

pub const DOPS_CPU: &[(&str, CpuKernelFn)] = &[
    ("ag_dop_stamp", ag_dop_stamp),
    ("ag_dop_policy", ag_dop_policy),
    ("ag_dop_apply", ag_dop_apply),
    ("ag_dop_compile", ag_dop_compile),
    ("ag_dop_eval_clear", ag_dop_eval_clear),
    ("ag_ad_seed", ag_ad_seed),
    ("ag_ad_classify", ag_ad_classify),
    ("ag_ad_resolve", ag_ad_resolve),
    ("ag_ad_start", ag_ad_start),
    ("ag_ad_emit", ag_ad_emit),
    ("ag_ad_total", ag_ad_total),
    ("ag_dop_decide", ag_dop_decide),
    ("ag_dop_continue", ag_dop_continue),
];
