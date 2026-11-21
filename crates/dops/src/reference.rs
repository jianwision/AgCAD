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
        if a.policy == policy::CMAES { unsafe { cma_sample(&a, &g, cfg, gcfg, round, &mut cmds[base..base + rn]) }; continue; }
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
        let err = if g.vol_hi > g.vol_lo { (o[12] - t) / t } else { 0.0 };
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
                // parameter saturated: structural action (exact-path features only)
                if act == 0 && count[cfg] < a.nodes && g.hole_r > 0.0 && err != 0.0 {
                    let ux = unit(hash3(g.seed ^ 0xA5A5_A5A5, gcfg, round));
                    let uy = unit(hash3(g.seed ^ 0x5A5A_5A5A, gcfg, round));
                    let (px, py) = (g.place[0] + ux * (g.place[2] - g.place[0]), g.place[1] + uy * (g.place[3] - g.place[1]));
                    let h = g.hole_r;
                    let two = slot + 1 < rn && (count[cfg] as usize) + 1 < a.nodes as usize;
                    let blank = |f: &mut FeatureArgs| { f.pref = [NO_PARAM; NVALS]; f.val = [0.0; NVALS]; };
                    if err > 0.0 {
                        // remove material: hole | rounded pocket | slot | bolt circle (4 holes)
                        let kind = if two { (gcfg + round) % 4 } else { 0 };
                        let c = &mut cmds[base + slot];
                        c.opcode = a.ops.pocket as u16;
                        let f = &mut arena[base + slot];
                        blank(f);
                        match kind {
                            1 | 2 => {
                                c.args[0] = profile::RECT;
                                let (hx, hy) = if kind == 1 { (h, h) } else { (2.0 * h, 0.5 * h) };
                                f.val[0] = px - hx; f.val[1] = py - hy; f.val[2] = px + hx; f.val[3] = py + hy; f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
                                slot += 1;
                                let c2 = &mut cmds[base + slot];
                                c2.opcode = a.ops.fillet as u16;
                                c2.args[0] = rec::LAST;
                                let f2 = &mut arena[base + slot];
                                blank(f2);
                                // rounded corners, or fully rounded ends (the device clamps to min(hx, hy))
                                f2.val[10] = if kind == 1 { 0.4 * h } else { 0.5 * h };
                            }
                            3 => {
                                c.args[0] = profile::CIRCLE;
                                f.val[0] = px; f.val[1] = py; f.val[2] = 0.5 * h; f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
                                slot += 1;
                                let c2 = &mut cmds[base + slot];
                                c2.opcode = a.ops.circular_pattern as u16;
                                c2.args[0] = rec::LAST;
                                let f2 = &mut arena[base + slot];
                                blank(f2);
                                f2.val[6] = 4.0;
                                f2.val[7] = core::f64::consts::TAU;
                            }
                            _ => {
                                c.args[0] = profile::CIRCLE;
                                f.val[0] = px; f.val[1] = py; f.val[2] = h; f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
                            }
                        }
                        slot += 1;
                    } else {
                        // add material: a boss on top of the base pad
                        let nodes = unsafe { sl::<Node>(a.cur_nodes, cn * a.nodes as usize) };
                        let base_pad = nodes[cfg * a.nodes as usize..cfg * a.nodes as usize + count[cfg] as usize].iter().find(|n| n.op == fop::ADD && n.kind <= profile::SKETCH && n.flags & nflags::ACTIVE != 0);
                        if let Some(bp) = base_pad {
                            let top = bp.val[5] + if bp.pref[5] != NO_PARAM && (bp.pref[5] as usize) < kn { pr[cfg * kn + bp.pref[5] as usize] } else { 0.0 };
                            let c = &mut cmds[base + slot];
                            c.opcode = a.ops.pad as u16;
                            c.args[0] = profile::CIRCLE;
                            let f = &mut arena[base + slot];
                            blank(f);
                            f.val[0] = px; f.val[1] = py; f.val[2] = h; f.val[4] = top; f.val[5] = top + h;
                            slot += 1;
                        }
                    }
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

/// Free (non-fixed) parameter j of the goal.
fn free(g: &Goal, j: usize) -> bool { g.param_hi[j] > g.param_lo[j] }

/// Parameter j searched by a CMA-ES island: a free design parameter, or the structural slot
/// of a move the island has made (slots n_params + 3k … of move k; see `structural_slots`).
pub fn active(g: &Goal, st: &OptState, j: usize, kn: usize) -> bool {
    let n = g.n_params as usize;
    if j >= kn.min(MAX_PARAMS) || !free(g, j) { return false; }
    j < n || (j - n) / 3 < st.structs as usize
}

/// Irwin–Hall standard-normal approximation from 12 hashed uniforms (no transcendentals).
pub fn normal12(seed: u32, gcfg: u32, idx: u32) -> f64 {
    let mut s = 0.0f64;
    let mut k = 0u32;
    while k < 12 { s += unit(hash3(seed ^ 0x51ED_270B, gcfg, idx.wrapping_mul(12).wrapping_add(k))); k += 1; }
    s - 6.0
}

/// CMA-ES policy, one configuration: sample u = clamp(mean + σ·√diag·z, 0, 1) from its
/// island and set every free parameter. Done / out-of-budget members report their current
/// point and score instead, so the island update sees λ evaluated points.
unsafe fn cma_sample(a: &DopsArgs, g: &Goal, cfg: usize, gcfg: u32, round: u32, cmds: &mut [CommandRecord]) {
    let (cn, kn) = (a.configs as usize, a.params as usize);
    let lam = a.island as usize;
    let st: OptState = unsafe { sl::<OptState>(a.opt, cn.div_ceil(lam))[cfg / lam] };
    let samp = unsafe { slm::<f64>(a.samp, cn * MAX_PARAMS) };
    let tscore = unsafe { slm::<f64>(a.tscore, cn) };
    let pr = unsafe { sl::<f64>(a.cur_params, cn * kn) };
    let np = (g.n_params as usize).min(kn).min(MAX_PARAMS);
    let idle = unsafe { sl::<u32>(a.done, cn)[cfg] != 0 || sl::<u32>(a.budget, cn)[cfg] == 0 };
    let _ = np;
    if idle {
        for j in 0..MAX_PARAMS { samp[cfg * MAX_PARAMS + j] = if active(g, &st, j, kn) { (pr[cfg * kn + j] - g.param_lo[j]) / (g.param_hi[j] - g.param_lo[j]) } else { 0.0 }; }
        tscore[cfg] = unsafe { sl::<f64>(a.score, cn)[cfg] };
        return;
    }
    let mut slot = 0usize;
    if st.pending != 0 {
        // the island's structural move: the same feature for every member
        let gi = gcfg / lam as u32;
        let arena = unsafe { slm::<FeatureArgs>(a.arena, cn * a.records as usize) };
        let base = cfg * a.records as usize;
        slot = unsafe { struct_move(a, g, cfg, gi, st.structs, st.pending == 1, cmds, &mut arena[base..base + a.records as usize]) };
    }
    for j in 0..MAX_PARAMS {
        let mut u = 0.0;
        if active(g, &st, j, kn) {
            let z = normal12(g.seed, gcfg, round.wrapping_mul(16).wrapping_add(j as u32));
            u = st.mean[j] + st.sigma * st.diag[j].sqrt() * z;
            if u < 0.0 { u = 0.0; }
            if u > 1.0 { u = 1.0; }
            if slot < cmds.len() {
                let c = &mut cmds[slot];
                c.opcode = a.ops.parameter as u16;
                c.args[0] = j as u32;
                c.args[1] = rec::PARAM_SET;
                c.imm[0] = g.param_lo[j] + u * (g.param_hi[j] - g.param_lo[j]);
                slot += 1;
            }
        }
        samp[cfg * MAX_PARAMS + j] = u;
    }
}

/// A structural move of a CMA-ES island (exact-path features only): remove material with a
/// hole / rounded pocket / slot / 4-hole bolt circle, or add a boss on the base pad. Location
/// and kind are hashed from (island, move count), so every member adds the same feature.
/// Returns the records used.
unsafe fn struct_move(a: &DopsArgs, g: &Goal, cfg: usize, gi: u32, k: u32, remove: bool, cmds: &mut [CommandRecord], arena: &mut [FeatureArgs]) -> usize {
    let (cn, kn, nn) = (a.configs as usize, a.params as usize, a.nodes as usize);
    let count = unsafe { sl::<u32>(a.cur_count, cn)[cfg] } as usize;
    if count >= nn || cmds.is_empty() { return 0; }
    let ux = unit(hash3(g.seed ^ 0x5717_0C7A, gi, 2 * k));
    let uy = unit(hash3(g.seed ^ 0x5717_0C7A, gi, 2 * k + 1));
    let (px, py) = (g.place[0] + ux * (g.place[2] - g.place[0]), g.place[1] + uy * (g.place[3] - g.place[1]));
    let h = g.hole_r;
    let two = cmds.len() >= 2 && count + 1 < nn;
    let blank = |f: &mut FeatureArgs| { f.pref = [NO_PARAM; NVALS]; f.val = [0.0; NVALS]; };
    blank(&mut arena[0]);
    // structural parameter slots of this move (radius, x, y), when the design has them
    let n0 = g.n_params as usize + 3 * k as usize;
    let slot = |w: usize| -> u32 { let j = n0 + w; if j < kn.min(MAX_PARAMS) && free(g, j) { j as u32 } else { NO_PARAM } };
    let (jr, jx, jy) = (slot(0), slot(1), slot(2));
    // a value (offset, literal fallback) bound to slot j
    let bind = |f: &mut FeatureArgs, i: usize, j: u32, off: f64, lit: f64| { if j != NO_PARAM { f.pref[i] = j; f.val[i] = off; } else { f.val[i] = lit; } };
    if remove {
        let kind = if two { (gi + k) % 4 } else { 0 };
        cmds[0].opcode = a.ops.pocket as u16;
        let f = &mut arena[0];
        if kind == 1 || kind == 2 {
            cmds[0].args[0] = profile::RECT;
            let (hx, hy) = if kind == 1 { (h, h) } else { (2.0 * h, 0.5 * h) };
            bind(f, 0, jx, -hx, px - hx); bind(f, 1, jy, -hy, py - hy); bind(f, 2, jx, hx, px + hx); bind(f, 3, jy, hy, py + hy);
            f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
            cmds[1].opcode = a.ops.fillet as u16;
            cmds[1].args[0] = rec::LAST;
            blank(&mut arena[1]);
            arena[1].val[10] = if kind == 1 { 0.4 * h } else { 0.5 * h };
            return 2;
        }
        cmds[0].args[0] = profile::CIRCLE;
        bind(f, 0, jx, 0.0, px); bind(f, 1, jy, 0.0, py);
        f.val[4] = g.hole_z0; f.val[5] = g.hole_z1;
        if kind == 3 {
            f.val[2] = 0.5 * h;
            cmds[1].opcode = a.ops.circular_pattern as u16;
            cmds[1].args[0] = rec::LAST;
            blank(&mut arena[1]);
            arena[1].val[6] = 4.0;
            arena[1].val[7] = core::f64::consts::TAU;
            return 2;
        }
        bind(f, 2, jr, 0.0, h);
        return 1;
    }
    let nodes = unsafe { sl::<Node>(a.cur_nodes, cn * nn) };
    let pr = unsafe { sl::<f64>(a.cur_params, cn * kn) };
    let Some(bp) = nodes[cfg * nn..cfg * nn + count].iter().find(|n| n.op == fop::ADD && n.kind <= profile::SKETCH && n.flags & nflags::ACTIVE != 0) else { return 0 };
    let top = bp.val[5] + if bp.pref[5] != NO_PARAM && (bp.pref[5] as usize) < kn { pr[cfg * kn + bp.pref[5] as usize] } else { 0.0 };
    cmds[0].opcode = a.ops.pad as u16;
    cmds[0].args[0] = profile::CIRCLE;
    let f = &mut arena[0];
    bind(f, 0, jx, 0.0, px); bind(f, 1, jy, 0.0, py); bind(f, 2, jr, 0.0, h);
    f.val[4] = top; f.val[5] = top + h;
    1
}

/// CMA-ES island initialization (before the first round): island 0 starts at the base
/// design, the others at hashed points of the box; σ₀ = 0.3, diag = 1.
pub unsafe fn ag_opt_init(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    if a.policy != policy::CMAES { return; }
    let (cn, kn) = (a.configs as usize, a.params as usize);
    let lam = a.island as usize;
    let g: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let pr = unsafe { sl::<f64>(a.cur_params, cn * kn) };
    let ni = cn.div_ceil(lam);
    let opt = unsafe { slm::<OptState>(a.opt, ni) };
    let np = (g.n_params as usize).min(kn).min(MAX_PARAMS);
    for (i, st) in opt.iter_mut().enumerate() {
        let gi = (a.cfg_offset as usize / lam + i) as u32;
        *st = OptState::zeroed();
        for j in 0..MAX_PARAMS {
            st.diag[j] = 1.0;
            if j < np && free(&g, j) {
                st.mean[j] = if gi == 0 { (pr[i * lam * kn + j] - g.param_lo[j]) / (g.param_hi[j] - g.param_lo[j]) } else { unit(hash3(g.seed ^ 0x0CA1_5EED, gi, j as u32)) };
            }
        }
        st.sigma = 0.3;
        st.best = 1.0e30;
    }
}

/// CMA-ES island update (after decide): rank the λ members by (trial score, index), move
/// the mean to the linearly weighted top μ = λ/2, rank-μ diagonal update, 1/5-success step
/// size; restart a stalled infeasible island at a hashed point.
pub unsafe fn ag_opt_update(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    if a.policy != policy::CMAES { return; }
    let cn = a.configs as usize;
    let lam = a.island as usize;
    let g: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let round = unsafe { sl::<u32>(a.round, 1)[0] };
    let samp = unsafe { sl::<f64>(a.samp, cn * MAX_PARAMS) };
    let tscore = unsafe { sl::<f64>(a.tscore, cn) };
    let obs = unsafe { sl::<f64>(a.obs, cn * OBS) };
    let ni = cn.div_ceil(lam);
    let opt = unsafe { slm::<OptState>(a.opt, ni) };
    for (i, st) in opt.iter_mut().enumerate() {
        let m0 = i * lam;
        let m = lam.min(cn - m0);
        let errs: Vec<f64> = (m0..m0 + m).map(|c| obs[c * OBS + ob::ERR]).collect();
        island_update(st, &samp[m0 * MAX_PARAMS..(m0 + m) * MAX_PARAMS], &tscore[m0..m0 + m], &errs, &g, a.params as usize, (a.cfg_offset as usize / lam + i) as u32, round);
    }
}

/// One island's update (shared by the kernel and the reference test).
pub fn island_update(st: &mut OptState, samp: &[f64], fs: &[f64], errs: &[f64], g: &Goal, kn: usize, gi: u32, round: u32) {
    let np = kn.min(MAX_PARAMS);
    let m = fs.len();
    if m == 0 { return; }
    // a structural round just happened: new structure, new baseline
    let was_struct = st.pending != 0;
    st.pending = 0;
    if was_struct { st.best = 1.0e30; }
    let best_before = st.best;
    // insertion sort by (score, index): deterministic, λ ≤ 64
    let mut order = [0usize; MAX_ISLAND];
    for k in 0..m {
        let mut p = k;
        while p > 0 && (fs[order[p - 1]] > fs[k] || (fs[order[p - 1]] == fs[k] && order[p - 1] > k)) { order[p] = order[p - 1]; p -= 1; }
        order[p] = k;
    }
    let mu = (m / 2).max(1);
    let wsum = (mu * (mu + 1) / 2) as f64;
    let mut succ = 0usize;
    for k in 0..m { if fs[k] < st.best { succ += 1; } }
    if fs[order[0]] < st.best { st.best = fs[order[0]]; }
    let old = st.mean;
    for j in 0..np {
        if !active(g, st, j, kn) { continue; }
        let mut mj = 0.0;
        let mut var = 0.0;
        for r in 0..mu {
            let w = (mu - r) as f64 / wsum;
            let x = samp[order[r] * MAX_PARAMS + j];
            mj += w * x;
            let d = (x - old[j]) / st.sigma;
            var += w * d * d;
        }
        st.mean[j] = mj;
        let mut dg = 0.8 * st.diag[j] + 0.2 * var;
        if dg < 1e-8 { dg = 1e-8; }
        if dg > 1e4 { dg = 1e4; }
        st.diag[j] = dg;
    }
    st.sigma *= if succ * 5 > m { 1.2 } else { 0.85 };
    if st.sigma < 1e-4 { st.sigma = 1e-4; }
    if st.sigma > 1.0 { st.sigma = 1.0; }
    st.generation += 1;
    if was_struct { st.sigma = 0.3; for j in 0..np { st.diag[j] = 1.0; } }
    st.stall = if st.best < best_before && !was_struct { 0 } else { st.stall + 1 };
    if was_struct { st.stall = 0; }
    // infeasible and not improving: change the structure (direction from the best member's
    // volume error) — at most 6 moves per island
    if g.hole_r > 0.0 && st.best >= 1000.0 && st.stall >= 3 && st.structs < 6 {
        let e = errs.get(order[0]).copied().unwrap_or(0.0);
        if e != 0.0 {
            st.pending = if e > 0.0 { 1 } else { 2 };
            // the new move's slots start at its hashed placement and nominal radius
            let n0 = g.n_params as usize + 3 * st.structs as usize;
            let (ux, uy) = (unit(hash3(g.seed ^ 0x5717_0C7A, gi, 2 * st.structs)), unit(hash3(g.seed ^ 0x5717_0C7A, gi, 2 * st.structs + 1)));
            for (w, u) in [0.5, ux, uy].into_iter().enumerate() { if n0 + w < MAX_PARAMS { st.mean[n0 + w] = u; st.diag[n0 + w] = 1.0; } }
            st.structs += 1;
            st.stall = 0;
            return;
        }
    }
    if st.sigma < 1e-3 && st.best >= 1000.0 {
        for j in 0..np { if active(g, st, j, kn) { st.mean[j] = unit(hash3(g.seed ^ 0x0CA1_5EED, gi, (round.wrapping_add(1) << 4) | j as u32)); st.diag[j] = 1.0; } }
        st.sigma = 0.3;
        st.restarts += 1;
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
    let npool = unsafe { sl::<f32>(a.pool, 1)[0] } as u32;
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
            } else if op == a.ops.pad || op == a.ops.pocket || op == a.ops.hole || op == a.ops.shaft || op == a.ops.groove {
                let f = arena[idx];
                let bad_ref = f.pref.iter().any(|&p| p != NO_PARAM && p as usize >= kn);
                let revolve = op == a.ops.shaft || op == a.ops.groove;
                let kind = if revolve { profile::REVOLVE } else { c.args[0] };
                // sketch / revolve profiles must name an existing pool profile (literal index)
                let pooled = kind == profile::SKETCH || kind == profile::REVOLVE;
                let pool_ok = !pooled || (f.pref[15] == NO_PARAM && f.val[15] >= 0.0 && (f.val[15].floor() as u64) < npool as u64);
                if (!revolve && kind > profile::POLYGON && kind != profile::SKETCH) || bad_ref || !pool_ok { st = AgStatus::InvalidParameter.value(); }
                else if pooled && a.general_ok == 0 { st = AgStatus::UnsupportedOp.value(); }
                else if count as usize >= nn { st = AgStatus::CapacityExceeded.value(); }
                else {
                    let add = op == a.ops.pad || op == a.ops.shaft;
                    tr_n[cfg * nn + count as usize] = Node { kind, op: if add { fop::ADD } else { fop::REMOVE }, flags: nflags::ACTIVE, src: r as u32, pref: f.pref, val: f.val };
                    count += 1;
                }
            } else if op == a.ops.mirror {
                let f = arena[idx];
                let mut depth = 0i32;
                for k in 0..count as usize { let n = &tr_n[cfg * nn + k]; if n.flags & nflags::ACTIVE != 0 { if n.kind == profile::BODY { depth += 1; } else if n.kind == profile::COMBINE { depth -= 1; } } }
                if a.general_ok == 0 { st = AgStatus::UnsupportedOp.value(); }
                else if count as usize >= nn { st = AgStatus::CapacityExceeded.value(); }
                else if depth != 0 { st = AgStatus::InvalidParameter.value(); }
                else {
                    let mut nd = Node { kind: profile::MIRROR, op: fop::ADD, flags: nflags::ACTIVE, src: r as u32, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
                    nd.val[0] = if f.val[0] > 0.5 { 1.0 } else { 0.0 };
                    nd.val[1] = if f.val[1] > 0.5 { 1.0 } else { 0.0 };
                    tr_n[cfg * nn + count as usize] = nd;
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
            } else if op == a.ops.circular_pattern || op == a.ops.rect_pattern || op == a.ops.fillet || op == a.ops.chamfer || op == a.ops.deactivate || op == a.ops.draft {
                let fi = if c.args[0] == rec::LAST { count.wrapping_sub(1) } else { c.args[0] };
                if count == 0 || fi >= count { st = AgStatus::InvalidParameter.value(); } else {
                    let n = &mut tr_n[cfg * nn + fi as usize];
                    let f = arena[idx];
                    // value slots this op writes
                    let edges_z = (op == a.ops.fillet || op == a.ops.chamfer) && c.args[1] == rec::EDGES_Z;
                    let (lo, hi) = if op == a.ops.circular_pattern { (6, 8) } else if op == a.ops.rect_pattern { (6, 10) } else if op == a.ops.deactivate { (0, 0) } else if op == a.ops.draft { (12, 13) } else if edges_z { (13, 14) } else { (10, 11) };
                    let bad_ref = (lo..hi).any(|j| f.pref[j] != NO_PARAM && f.pref[j] as usize >= kn);
                    let extrusion = n.kind <= profile::POLYGON || n.kind == profile::SKETCH;
                    let unsupported = if op == a.ops.draft || edges_z { !extrusion } else if op == a.ops.fillet || op == a.ops.chamfer { n.kind != profile::RECT } else { false };
                    // literal range checks (|draft| ≤ 30°, non-negative cap size)
                    let out_of_range = (op == a.ops.draft && f.pref[12] == NO_PARAM && !(f.val[12].abs() <= 0.57735)) || (edges_z && f.pref[13] == NO_PARAM && !(f.val[13] >= 0.0));
                    if bad_ref || unsupported || ((op == a.ops.draft || edges_z) && a.general_ok == 0) { st = if bad_ref { AgStatus::InvalidParameter.value() } else { AgStatus::UnsupportedOp.value() }; }
                    else if out_of_range { st = AgStatus::GeometryCheckFailed.value(); }
                    else if op == a.ops.deactivate { if c.args[1] != 0 { n.flags |= nflags::ACTIVE; } else { n.flags &= !nflags::ACTIVE; } }
                    else {
                        for j in lo..hi { n.pref[j] = f.pref[j]; n.val[j] = f.val[j]; }
                        if op == a.ops.circular_pattern { n.flags = (n.flags & !nflags::PATTERN_RECT) | nflags::PATTERN; }
                        else if op == a.ops.rect_pattern { n.flags = (n.flags & !nflags::PATTERN) | nflags::PATTERN_RECT; }
                        else if op == a.ops.draft { n.flags |= nflags::DRAFT; }
                        else if edges_z { n.flags = (n.flags & !(nflags::ROUND_Z | nflags::CHAMFER_Z)) | if op == a.ops.fillet { nflags::ROUND_Z } else { nflags::CHAMFER_Z }; }
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
    let np = unsafe { sl::<f32>(a.pool, 1)[0] } as usize;
    let pool = unsafe { sl::<f32>(a.pool + 4, np * PSTRIDE) };
    for c in 0..cn {
        if a.eval_trial != 0 && done[c] != 0 { continue; }
        for i in 0..cnt[c] as usize {
            let t = &mut tabs[(c * nn + i) * TABLE..(c * nn + i + 1) * TABLE];
            out[c * nn + i] = compile_node_pool(&nodes[c * nn + i], &prm[c * kn..c * kn + kn], pool, t);
        }
        if cnt[c] > 0 {
            let m = scan_mirror(&out[c * nn..c * nn + cnt[c] as usize]);
            out[c * nn]._pad[0] = 0x8000_0000 | m;
        }
    }
}

/// Reset per-configuration evaluation accumulators.
pub unsafe fn ag_dop_eval_clear(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cn = a.configs as usize;
    let counts = unsafe { slm::<u64>(a.counts, CS * cn) };
    let dom = unsafe { slm::<u32>(a.domain_hit, cn) };
    let bb = unsafe { slm::<u32>(a.bbox, 12 * cn) };
    for c in 0..cn {
        for j in 0..CS { counts[CS * c + j] = 0; }
        counts[CS * c + cnt::MIN_WALL] = u64::MAX;
        counts[CS * c + cnt::MIN_GAP] = u64::MAX;
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

/// SDF value, central-difference gradient and gradient width at a boundary cell centre.
pub struct Surface { pub f: f32, pub g: [f32; 3], pub gl: f32, pub w: f32 }

pub fn surface(nodes: &[CNode], tabs: &[f32], c: [f32; 3], d: [f32; 3]) -> Surface { surface_p(nodes, tabs, &Prune::NONE, c, d) }

/// `surface` with the cell's pruned tape (all samples lie inside the cell).
pub fn surface_p(nodes: &[CNode], tabs: &[f32], pr: &Prune, c: [f32; 3], d: [f32; 3]) -> Surface {
    let (x, y, z) = (c[0], c[1], c[2]);
    let e = |x: f32, y: f32, z: f32| tape_sdf_p(nodes, tabs, pr, x, y, z);
    let f = e(x, y, z);
    let h = 0.25f32 * d[0].min(d[1]).min(d[2]);
    let gx = e(x + h, y, z) - e(x - h, y, z);
    let gy = e(x, y + h, z) - e(x, y - h, z);
    let gz = e(x, y, z + h) - e(x, y, z - h);
    let gl = (gx * gx + gy * gy + gz * gz).sqrt();
    // width of the cell along the gradient direction
    let w = if gl > 1e-30 { (gx.abs() * d[0] + gy.abs() * d[1] + gz.abs() * d[2]) / gl } else { 0.0 };
    Surface { f, g: [gx, gy, gz], gl, w }
}

/// Fill fraction of a boundary cell, quantized to 2⁻²⁰: 0.5 − f / w with w the cell's width
/// along the SDF gradient. Exact for axis-aligned planar faces.
pub fn fill_of(s: &Surface) -> u64 {
    let mut frac = 0.5f32;
    if s.gl > 1e-30 {
        frac = 0.5 - s.f / s.w;
        if frac < 0.0 { frac = 0.0; }
        if frac > 1.0 { frac = 1.0; }
    }
    (frac as f64 * 1048576.0 + 0.5).floor() as u64
}

pub fn fill_q(nodes: &[CNode], tabs: &[f32], x: f32, y: f32, z: f32, d: [f32; 3]) -> u64 { fill_of(&surface(nodes, tabs, [x, y, z], d)) }

/// Rigorous inside fraction of an ambiguous cell `b` (units 2⁻²⁴): repeatedly split the
/// largest ambiguous sub-box along the axis where the field varies most (|g_k|·width_k,
/// effectively anisotropic: a planar face converges geometrically) and classify the halves
/// by interval evaluation, within `budget` evaluations. Volumes use the actual f32 box
/// widths (lower bound floored, upper bound ceiled). Returns (certified inside, inside or
/// ambiguous).
pub fn tighten(nodes: &[CNode], tabs: &[f32], pr: &Prune, b: &[Iv; 3], g: [f32; 3], e: f32, budget: u32) -> (u64, u64) {
    let full = 1u64 << TQ;
    if budget < 2 { return (0, full); }
    let cd = [(b[0].hi - b[0].lo) as f64, (b[1].hi - b[1].lo) as f64, (b[2].hi - b[2].lo) as f64];
    let cv = cd[0] * cd[1] * cd[2];
    if !(cv > 0.0) { return (0, full); }
    let q = |x: &[Iv; 3]| -> f64 { (x[0].hi - x[0].lo) as f64 * (x[1].hi - x[1].lo) as f64 * (x[2].hi - x[2].lo) as f64 / cv * full as f64 };
    let mut bx = [[iv(0.0, 0.0); 3]; TK];
    let mut vol = [0.0f64; TK];
    bx[0] = *b;
    vol[0] = full as f64;
    let mut n = 1usize;
    let (mut lo, mut hi) = (0u64, 0u64);
    let mut evals = 0u32;
    while n > 0 && evals + 2 <= budget && n < TK {
        // largest ambiguous sub-box (first on ties)
        let mut k = 0;
        for i in 1..n { if vol[i] > vol[k] { k = i; } }
        let pb = bx[k];
        // split axis: largest |g|·width; stop when the box is near f32 resolution
        let mut ax = 0;
        let mut best = -1.0f32;
        for j in 0..3 {
            let v = g[j].abs() * (pb[j].hi - pb[j].lo);
            if v > best { best = v; ax = j; }
        }
        let mid = 0.5f32 * (pb[ax].lo + pb[ax].hi);
        if !(mid > pb[ax].lo && mid < pb[ax].hi) || ((pb[ax].hi - pb[ax].lo) as f64) < 1e-4 * cd[ax] { break; }
        n -= 1;
        bx[k] = bx[n];
        vol[k] = vol[n];
        for half in 0..2 {
            let mut c = pb;
            if half == 0 { c[ax].hi = mid; } else { c[ax].lo = mid; }
            evals += 1;
            let r = tape_iv_p(nodes, tabs, pr, c[0], c[1], c[2]);
            if r.lo - e > 0.0 { continue; }
            let v = q(&c);
            if r.hi + e < 0.0 { lo += v.floor() as u64; hi += v.ceil() as u64; continue; }
            bx[n] = c;
            vol[n] = v;
            n += 1;
        }
    }
    for i in 0..n { hi += vol[i].ceil() as u64; }
    (lo, hi.min(full))
}

/// Area quantization: 2⁻¹⁰ mm².
pub fn qarea(a: f32) -> u64 { (a as f64 * 1024.0 + 0.5).floor() as u64 }

/// Maximum sphere-tracing steps of a probe ray.
pub const MARCH_STEPS: u32 = 160;

/// Sphere-trace from `p` along unit `dir` while the SDF keeps its sign (`inside`: negative);
/// returns the distance of the first sign change, or `tmax`. Steps never exceed |f| (a
/// distance bound), so thin features are not skipped; `hmin` guarantees progress.
pub fn march(nodes: &[CNode], tabs: &[f32], p: [f32; 3], dir: [f32; 3], inside: bool, t0: f32, tmax: f32, hmin: f32) -> f32 {
    let mut t = t0;
    let mut i = 0u32;
    while i < MARCH_STEPS {
        if t >= tmax { return tmax; }
        let f = tape_sdf_c(nodes, tabs, p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t);
        let s = if inside { -f } else { f };
        if s <= 0.0 { return t; }
        t = t + if s > hmin { s } else { hmin };
        i += 1;
    }
    tmax
}

/// Engineering probe of one boundary cell (sampled evidence): surface area share, wall
/// thickness along the inward normal, clearance along the outward normal, and 3-axis tool
/// access (a ray toward +z from up-facing points, toward −z from down-facing ones: a down-facing
/// point whose downward ray is blocked is an overhang the tool cannot reach).
pub struct Probe { pub area: f32, pub wall: f32, pub gap: f32, pub thin: bool, pub narrow: bool, pub undercut: bool }

pub fn probe(nodes: &[CNode], tabs: &[f32], goal: &Goal, s: &Surface, c: [f32; 3], d: [f32; 3]) -> Probe {
    let mut pr = Probe { area: 0.0, wall: FAR32, gap: FAR32, thin: false, narrow: false, undercut: false };
    // only cells the zero set actually crosses (interval-ambiguous cells near the surface
    // are boundary for the rigorous bounds but carry no surface)
    if !(s.gl > 1e-30) || !(s.f.abs() < 0.5f32 * s.w) { return pr; }
    pr.area = d[0] * d[1] * d[2] / s.w;
    let n = [s.g[0] / s.gl, s.g[1] / s.gl, s.g[2] / s.gl];
    // surface point: the centre projected along the gradient; rejected where the bound is not
    // a distance (near edges the projection can miss the surface)
    if !(goal.min_wall > 0.0) && !(goal.min_gap > 0.0) && goal.machining_3ax == 0 { return pr; }
    let p = [c[0] - n[0] * s.f, c[1] - n[1] * s.f, c[2] - n[2] * s.f];
    let dm = d[0].min(d[1]).min(d[2]);
    let hmin = 0.1f32 * dm;
    if !(tape_sdf_c(nodes, tabs, p[0], p[1], p[2]).abs() <= hmin) { return pr; }
    let at = |t: f32| tape_sdf_c(nodes, tabs, p[0] + n[0] * t, p[1] + n[1] * t, p[2] + n[2] * t);
    // a valid probe starts inside just below the surface and outside just above it
    let (fin, fout) = (at(-hmin), at(hmin));
    if !(fin < 0.0 && fout > 0.0) { return pr; }
    if goal.min_wall > 0.0 {
        let tw = goal.min_wall as f32;
        pr.wall = march(nodes, tabs, p, [-n[0], -n[1], -n[2]], true, hmin, tw, hmin);
        pr.thin = pr.wall < tw;
    }
    if goal.min_gap > 0.0 {
        let tg = goal.min_gap as f32;
        pr.gap = march(nodes, tabs, p, n, false, hmin, tg, hmin);
        pr.narrow = pr.gap < tg;
    }
    if goal.machining_3ax != 0 {
        let q = [p[0] + n[0] * dm, p[1] + n[1] * dm, p[2] + n[2] * dm];
        let up = n[2] >= -0.25f32;
        let (dir, tmax) = if up { ([0.0f32, 0.0, 1.0], goal.domain_hi[2] as f32 - q[2]) } else { ([0.0f32, 0.0, -1.0], q[2] - goal.domain_lo[2] as f32) };
        if tmax > 0.0 && tape_sdf_c(nodes, tabs, q[0], q[1], q[2]) > 0.0 { pr.undercut = march(nodes, tabs, q, dir, false, 0.0, tmax, 0.5f32 * dm) < tmax; }
    }
    pr
}

fn decode(e: u64) -> (usize, u32, u32, u32) { ((e >> 32) as usize, ((e >> 20) & 1023) as u32, ((e >> 10) & 1023) as u32, (e & 1023) as u32) }
fn encode(cfg: u32, ix: u32, iy: u32, iz: u32) -> u64 { ((cfg as u64) << 32) | ((ix as u64) << 20) | ((iy as u64) << 10) | iz as u64 }

struct Acc<'a> { counts: &'a mut [u64], dom: &'a mut [u32], bb: &'a mut [u32] }

impl Acc<'_> {
    /// Mass moments of a counted cell (q20 fill weight; integer, order-independent).
    fn moments(&mut self, cfg: usize, idx: [u32; 3], sh: u32, q20: u64) {
        let w = 1u64 << (3 * sh);
        let q6 = q20 >> 14;
        let ss = 4u64 << (2 * sh); // (doubled side)²
        let o = CS * cfg;
        for k in 0..3 {
            let c2 = ((2 * idx[k] + 1) as u64) << sh;
            self.counts[o + cnt::M1 + k] += q20 * w * c2;
            self.counts[o + cnt::M2 + k] += q6 * w * (12 * c2 * c2 + ss);
        }
        self.counts[o + cnt::V6] += q6 * w;
    }
    fn probe(&mut self, cfg: usize, p: &Probe) {
        let o = CS * cfg;
        let aq = qarea(p.area);
        self.counts[o + cnt::AREA] += aq;
        if p.thin { self.counts[o + cnt::THIN_AREA] += aq; }
        if p.narrow { self.counts[o + cnt::GAP_AREA] += aq; }
        if p.undercut { self.counts[o + cnt::UNDERCUT_AREA] += aq; }
        if p.wall < FAR32 { self.counts[o + cnt::MIN_WALL] = self.counts[o + cnt::MIN_WALL].min(p.wall.to_bits() as u64); }
        if p.gap < FAR32 { self.counts[o + cnt::MIN_GAP] = self.counts[o + cnt::MIN_GAP].min(p.gap.to_bits() as u64); }
    }
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
    let mut acc = Acc { counts: unsafe { slm::<u64>(a.counts, CS * cn) }, dom: unsafe { slm::<u32>(a.domain_hit, cn) }, bb: unsafe { slm::<u32>(a.bbox, 12 * cn) } };
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
            acc.counts[CS * cfg] += 1u64 << (3 * sh);
            acc.moments(cfg, [ix, iy, iz], sh, 1 << 20);
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
    let cn = a.configs as usize;
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let cap = a.capacity as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let cls = unsafe { sl::<u32>(a.cls, n) };
    let flag = unsafe { slm::<u32>(a.flag, cap) };
    let term = unsafe { slm::<u32>(a.term, cap) };
    let crank = unsafe { sl::<u32>(a.crank, cap) };
    let start = unsafe { sl::<u32>(a.start, cn) };
    for i in 0..cap {
        flag[i] = 0;
        term[i] = 0;
        if i >= n || cls[i] != 1 { continue; }
        let (cfg, _, _, _) = decode(wl[i]);
        let rank = crank[i] - crank[start[cfg] as usize];
        if l + 1 < a.levels && (rank as u64 + 1) * 8 <= a.cap as u64 { flag[i] = 1; } else { term[i] = 1; }
    }
}

/// CLIP: refinements whose children would not fit in the global cell budget become terminal
/// (a suffix in entry order, i.e. in configuration order: deterministic, and sound — such
/// cells are counted as boundary). Also sets the next level's entry count.
pub unsafe fn ag_ad_clip(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cap = a.capacity as usize;
    let flag = unsafe { slm::<u32>(a.flag, cap) };
    let term = unsafe { slm::<u32>(a.term, cap) };
    let offs = unsafe { sl::<u32>(a.offs, cap) };
    let keep = (a.capacity / 8) as u32;
    let total = offs[cap - 1] + flag[cap - 1];
    for i in 0..cap { if flag[i] != 0 && offs[i] >= keep { flag[i] = 0; term[i] = 1; } }
    unsafe { slm::<u32>(a.lvl, a.levels as usize + 1)[a.level as usize + 1] = total.min(keep) * 8; }
}

/// COMPACT: terminal entries → dense list (ascending entry index, so sorted by configuration).
pub unsafe fn ag_ad_compact(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let cap = a.capacity as usize;
    let term = unsafe { sl::<u32>(a.term, cap) };
    let toffs = unsafe { sl::<u32>(a.toffs, cap) };
    let tlist = unsafe { slm::<u32>(a.tlist, cap) };
    for i in 0..cap { if term[i] != 0 { tlist[toffs[i] as usize] = i as u32; } }
    unsafe { slm::<u32>(a.tcount, 1)[0] = toffs[cap - 1] + term[cap - 1]; }
}

/// CUDA splits mirrored tapes into their own launch; the reference fill covers all cells.
pub unsafe fn ag_ad_fill_mirror(_: *const u8, _: &LaunchConfig) {}

/// FILL (level l): terminal boundary cells — rigorous tightened fractions, sampled fill,
/// moments, engineering probes, extents and the border test.
pub unsafe fn ag_ad_fill(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn) = (a.configs as usize, a.nodes as usize);
    let cnt = unsafe { sl::<u32>(if a.eval_trial != 0 { a.tr_count } else { a.cur_count }, cn) };
    let cnodes = unsafe { sl::<CNode>(a.compiled, cn * nn) };
    let tabs = unsafe { sl::<f32>(a.tables, cn * nn * TABLE) };
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let gr = grid(&goal, a.coarse, a.levels);
    let l = a.level;
    let n = unsafe { sl::<u32>(a.lvl, a.levels as usize + 1)[l as usize] } as usize;
    let wl = unsafe { sl::<u64>(if l % 2 == 0 { a.wl0 } else { a.wl1 }, n) };
    let tn = unsafe { sl::<u32>(a.tcount, 1)[0] } as usize;
    let tlist = unsafe { sl::<u32>(a.tlist, tn) };
    let mut acc = Acc { counts: unsafe { slm::<u64>(a.counts, CS * cn) }, dom: unsafe { slm::<u32>(a.domain_hit, cn) }, bb: unsafe { slm::<u32>(a.bbox, 12 * cn) } };
    let sh = a.levels - 1 - l;
    let e = gr.eps as f32;
    for &ti in tlist {
        let (cfg, ix, iy, iz) = decode(wl[ti as usize]);
        let tape = &cnodes[cfg * nn..cfg * nn + cnt[cfg] as usize];
        let tb = &tabs[cfg * nn * TABLE..(cfg * nn + cnt[cfg] as usize) * TABLE];
        let b = cell_box(&gr, l, ix, iy, iz);
        let w = 1u64 << (3 * sh);
        acc.counts[CS * cfg + cnt::BOUNDARY] += w;
        let d = [b[0].hi - b[0].lo, b[1].hi - b[1].lo, b[2].hi - b[2].lo];
        let c = [0.5f32 * (b[0].lo + b[0].hi), 0.5f32 * (b[1].lo + b[1].hi), 0.5f32 * (b[2].lo + b[2].hi)];
        // pruned tape for this cell (bit-identical point values inside it)
        let mut pru = Prune::NONE;
        let _ = tape_iv_prune(tape, tb, b[0], b[1], b[2], e, &mut pru);
        let sf = surface_p(tape, tb, &pru, c, d);
        let (tlo, thi) = tighten(tape, tb, &pru, &b, sf.g, e, a.tighten);
        acc.counts[CS * cfg + cnt::BLO] += tlo * w;
        acc.counts[CS * cfg + cnt::BHI] += thi * w;
        // sampled fill, kept inside the certified fraction interval
        let mut fq = fill_of(&sf);
        if fq < tlo >> 4 { fq = tlo >> 4; }
        if fq > (thi + 15) >> 4 { fq = (thi + 15) >> 4; }
        acc.counts[CS * cfg + cnt::FILL] += fq * w;
        acc.moments(cfg, [ix, iy, iz], sh, fq);
        let pr = probe(tape, tb, &goal, &sf, c, d);
        acc.probe(cfg, &pr);
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
    let q = 1.0 / (1u64 << TQ) as f64;
    let vol_lo = (counts[0] as f64 + counts[cnt::BLO] as f64 * q) * vc;
    let vol_hi = (counts[0] as f64 + counts[cnt::BHI] as f64 * q) * vc;
    let vol_est = (counts[0] as f64 + counts[2] as f64 * (1.0 / 1048576.0)) * vc;
    let mut el = [0.0; 3];
    let mut eh = [0.0; 3];
    for k in 0..3 {
        if bb[3 + k] > bb[k] { el[k] = (bb[3 + k] - bb[k]) as f64 * gr.fine[k]; }
        if bb[9 + k] > bb[6 + k] { eh[k] = (bb[9 + k] - bb[6 + k]) as f64 * gr.fine[k]; }
    }
    (vol_lo, vol_hi, vol_est, el, eh)
}

/// Engineering metrics of one configuration from its accumulators (all sampled).
pub struct Eng { pub centroid: [f64; 3], pub inertia: [f64; 3], pub area: f64, pub thin: f64, pub min_wall: f64, pub gap: f64, pub min_gap: f64, pub undercut: f64, pub violation: f64, pub centroid_ok: bool }

pub fn engineering(gr: &Grid, goal: &Goal, cc: &[u64], ve: f64) -> Eng {
    let qa = 1.0 / 1024.0;
    let v20 = cc[cnt::INSIDE] as f64 * 1048576.0 + cc[cnt::FILL] as f64;
    let v6 = cc[cnt::V6] as f64;
    let mut centroid = [0.0; 3];
    let mut var = [0.0; 3];
    let mut centroid_ok = true;
    for k in 0..3 {
        let half = 0.5 * gr.fine[k];
        let m1 = if v20 > 0.0 { cc[cnt::M1 + k] as f64 / v20 } else { 0.0 };
        let m2 = if v6 > 0.0 { cc[cnt::M2 + k] as f64 / (12.0 * v6) } else { 0.0 };
        centroid[k] = gr.lo[k] + m1 * half;
        let vv = m2 - m1 * m1;
        var[k] = if vv > 0.0 { vv * half * half } else { 0.0 };
        if goal.centroid_hi[k] > goal.centroid_lo[k] && !(centroid[k] >= goal.centroid_lo[k] && centroid[k] <= goal.centroid_hi[k]) { centroid_ok = false; }
    }
    let inertia = [ve * (var[1] + var[2]), ve * (var[0] + var[2]), ve * (var[0] + var[1])];
    let mw = |slot: usize| if cc[slot] == u64::MAX { -1.0 } else { f32::from_bits(cc[slot] as u32) as f64 };
    let thin = cc[cnt::THIN_AREA] as f64 * qa;
    let gap = cc[cnt::GAP_AREA] as f64 * qa;
    let undercut = cc[cnt::UNDERCUT_AREA] as f64 * qa;
    let over = undercut - goal.undercut_tol;
    let violation = thin + gap + if goal.machining_3ax != 0 && over > 0.0 { over } else { 0.0 };
    Eng { centroid, inertia, area: cc[cnt::AREA] as f64 * qa, thin, min_wall: if goal.min_wall > 0.0 { mw(cnt::MIN_WALL) } else { -1.0 }, gap, min_gap: if goal.min_gap > 0.0 { mw(cnt::MIN_GAP) } else { -1.0 }, undercut, violation, centroid_ok }
}

impl Eng {
    pub fn write(&self, o: &mut [f64]) {
        for k in 0..3 { o[ob::CENTROID + k] = self.centroid[k]; o[ob::INERTIA + k] = self.inertia[k]; }
        o[ob::AREA] = self.area;
        o[ob::THIN_AREA] = self.thin;
        o[ob::MIN_WALL] = self.min_wall;
        o[ob::GAP_AREA] = self.gap;
        o[ob::MIN_GAP] = self.min_gap;
        o[ob::UNDERCUT_AREA] = self.undercut;
        o[ob::VIOLATION] = self.violation;
        o[ob::CENTROID_OK] = if self.centroid_ok { 1.0 } else { 0.0 };
    }
}

/// Goal judgement of one evaluation: (met, relative volume error, score). Satisfy mode:
/// met = feasible and the estimate in the target interval, score |err| (1000 + … if not
/// met). Objective modes: met = feasible (and in the interval if one is given), score
/// ±volume / obj_scale.
pub fn judge(goal: &Goal, empty: bool, dom: bool, ve: f64, eh: [f64; 3], en: &Eng) -> (bool, f64, f64) {
    let has_iv = goal.vol_hi > goal.vol_lo;
    let t = 0.5 * (goal.vol_lo + goal.vol_hi);
    let err = if has_iv { (ve - t) / t } else { 0.0 };
    let in_iv = !has_iv || (ve >= goal.vol_lo && ve <= goal.vol_hi);
    let met = !empty && !dom && in_iv && eh[0] <= goal.max_ext[0] && eh[1] <= goal.max_ext[1] && eh[2] <= goal.max_ext[2] && en.violation == 0.0 && en.centroid_ok;
    let pen = if en.area > 0.0 { en.violation / en.area } else { 0.0 } + if en.centroid_ok { 0.0 } else { 1.0 };
    let scale = if goal.obj_scale > 0.0 { goal.obj_scale } else { 1.0 };
    let sc = if !met { 1000.0 + err.abs() + pen } else if goal.objective == 1 { ve / scale } else if goal.objective == 2 { -(ve / scale) } else { err.abs() };
    (met, err, sc)
}

/// CHECK + DECIDE (eval_trial = 1) or OBSERVE (eval_trial = 0), one thread per configuration:
/// interval checks against the goal and hard limits, transactional commit of the trial tape,
/// budget metering, observation row, ranking score and trace row.
pub unsafe fn ag_dop_decide(params: *const u8, _: &LaunchConfig) {
    let a = unsafe { args(params) };
    let (cn, nn, kn, rn) = (a.configs as usize, a.nodes as usize, a.params as usize, a.records as usize);
    let goal: Goal = unsafe { sl::<Goal>(a.goal, 1)[0] };
    let gr = grid(&goal, a.coarse, a.levels);
    let counts = unsafe { sl::<u64>(a.counts, CS * cn) };
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
    let tscore = unsafe { slm::<f64>(a.tscore, cn) };
    let cma = a.policy == policy::CMAES;
    for c in 0..cn {
        if a.eval_trial != 0 && done[c] != 0 { continue; }
        let cc = &counts[CS * c..CS * c + CS];
        let (vl, vh, ve, el, eh) = bounds(&gr, cc, &bb[12 * c..12 * c + 12]);
        let en = engineering(&gr, &goal, cc, ve);
        let empty = cc[cnt::INSIDE] + cc[cnt::BOUNDARY] == 0;
        let hard = el[0] > goal.max_ext[0] || el[1] > goal.max_ext[1] || el[2] > goal.max_ext[2];
        let (met, err, sc) = judge(&goal, empty, dom[c] != 0, ve, eh, &en);
        let mut write = a.eval_trial == 0;
        if a.eval_trial != 0 {
            let (mut acc, mut rej) = (0u32, 0u32);
            for r in 0..rn {
                let i = c * rn + r;
                if cmds[i].opcode as u32 == a.ops.update { continue; }
                if status[i].status == ACCEPTED { acc += 1; } else { rej += 1; }
            }
            // constraint violations may not grow (non-worsening edits only)
            // (mock policy; the CMA-ES policy is elitist on the full score instead)
            let worse = !cma && en.violation > obs[c * OBS + ob::VIOLATION];
            let geometry_ok = !empty && dom[c] == 0 && !hard && !worse;
            if cma { tscore[c] = if geometry_ok { sc } else { 1.0e30 }; }
            let structural = (0..rn).any(|r| { let o = cmds[c * rn + r].opcode as u32; o != a.ops.update && o != a.ops.parameter && status[c * rn + r].status == ACCEPTED });
            let commit = geometry_ok && acc > 0 && (!cma || sc < score[c] || structural);
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
            o[10] = if met { 1.0 } else { 0.0 };
            o[11] = err;
            o[12] = ve;
            en.write(o);
            for k in 0..3 { o[ob::BOX_LO + k] = if eh[k] > 0.0 { gr.lo[k] + bb[12 * c + 6 + k] as f64 * gr.fine[k] } else { 0.0 }; }
            if met && goal.objective == 0 { done[c] = 1; }
            score[c] = sc;
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
    ("ag_ad_clip", ag_ad_clip),
    ("ag_ad_compact", ag_ad_compact),
    ("ag_ad_fill", ag_ad_fill),
    ("ag_ad_fill_mirror", ag_ad_fill_mirror),
    ("ag_ad_start", ag_ad_start),
    ("ag_ad_emit", ag_ad_emit),
    ("ag_ad_total", ag_ad_total),
    ("ag_dop_decide", ag_dop_decide),
    ("ag_dop_continue", ag_dop_continue),
    ("ag_opt_init", ag_opt_init),
    ("ag_opt_update", ag_opt_update),
    ("ag_fea_voxelize", crate::fea::ag_fea_voxelize),
    ("ag_fea_nodes", crate::fea::ag_fea_nodes),
    ("ag_fea_init", crate::fea::ag_fea_init),
    ("ag_fea_dot_part", crate::fea::ag_fea_dot_part),
    ("ag_fea_dot_final", crate::fea::ag_fea_dot_final),
    ("ag_fea_start", crate::fea::ag_fea_start),
    ("ag_fea_check", crate::fea::ag_fea_check),
    ("ag_fea_matvec", crate::fea::ag_fea_matvec),
    ("ag_fea_update1", crate::fea::ag_fea_update1),
    ("ag_fea_update2", crate::fea::ag_fea_update2),
    ("ag_fea_step", crate::fea::ag_fea_step),
    ("ag_fea_finish", crate::fea::ag_fea_finish),
    ("ag_fea_maxdisp", crate::fea::ag_fea_maxdisp),
    ("ag_mesh_sample", crate::mesh::ag_mesh_sample),
];
