//! Device-resident rollout ("studio"): C configurations of a design, a device policy (the
//! mock GPU model) and the harness, iterating rounds without host involvement:
//!
//! ```text
//! WHILE active:                                         (CUDA: one graph, conditional WHILE)
//!   policy   → command records + arg arena               (mock model; learned model later)
//!   validate → status per record (allow-list, revision, budget, finiteness)   [harness]
//!   apply    → trial tapes (record order per configuration)
//!   compile  → parameters resolved, pattern / polygon trig tables precomputed
//!   eval     → rigorous volume / extent intervals (Lipschitz cell classification)
//!   decide   → interval checks vs goal + hard limits, commit/reject, meter, observe, score
//!   continue → round++, active flag, cudaGraphSetConditional
//! ```
//!
//! Host interaction per segment: none in graph mode; a 4-byte flag per round in host-loop mode
//! (CPU reference device and conformance). Results (top-k tapes) cross once, at the end.

use crate::abi::*;
use agcad_driver::abi::{CommandRecord, StatusRecord};
use agcad_driver::*;
use agcad_kernels::params::{upload_params, ParamUpload};
use bytemuck::Zeroable;
use std::cell::Cell;
use std::sync::Arc;

const NODE_B: usize = std::mem::size_of::<Node>();

pub fn dops_library() -> LibrarySource {
    LibrarySource { name: "ag_dops", cuda_source: Some(agcad_kernels_cuda::bundle(include_str!("../cu/dops.cu"), &[])), cpu_kernels: crate::reference::DOPS_CPU, options: vec![] }
}

/// A base design: the feature tape, its parameter values and their names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Design {
    pub nodes: Vec<Node>,
    pub params: Vec<f64>,
    pub names: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct StudioConfig {
    pub configs: u32,
    pub nodes: u32,
    pub records: u32,
    pub max_rounds: u32,
    /// Root grid per axis and adaptive refinement levels (finest resolution =
    /// coarse << (levels − 1)); `cap` bounds cells per configuration and level.
    pub coarse: u32,
    pub levels: u32,
    pub cap: u32,
    /// Accepted (non-no-op) edits per configuration.
    pub budget: u32,
    /// Interval evaluations per terminal boundary cell to tighten the rigorous volume
    /// bounds (0 = boundary cells count as [0, 1]).
    pub tighten: u32,
    /// Global worklist budget in cells (0 = configs × cap). Memory scales with this instead of
    /// configs × cap; when exhausted, the remaining refinements of a level become boundary
    /// cells (still rigorous, just wider bounds), in configuration order.
    pub max_cells: u32,
    /// `policy::MOCK` or `policy::CMAES`; `island` = CMA-ES population λ (configurations
    /// per island, 2 ..= 64).
    pub policy: u32,
    pub island: u32,
}

impl Default for StudioConfig {
    fn default() -> Self { StudioConfig { configs: 64, nodes: 16, records: 4, max_rounds: 32, coarse: 8, levels: 5, cap: 4096, budget: 64, tighten: 0, max_cells: 0, policy: 0, island: 16 } }
}

/// One configuration's result after a rollout.
#[derive(Clone, Debug)]
pub struct Survivor {
    pub configuration: u32,
    pub score: f64,
    pub obs: [f64; OBS],
    pub design: Design,
}

#[derive(Clone, Debug)]
pub struct RolloutSummary {
    pub rounds: u32,
    pub done: u32,
    pub top: Vec<Survivor>,
    /// Bytes read back by the host while rounds were executing (0 in graph mode).
    pub in_loop_readback_bytes: u64,
}

struct Bufs<B> {
    goal: B, cur_nodes: B, cur_count: B, cur_params: B, tr_nodes: B, tr_count: B, tr_params: B,
    rev: B, budget: B, done: B, cmds: B, arena: B, status: B, counts: B, domain_hit: B, bbox: B,
    obs: B, score: B, trace: B, round: B, active: B, allow: B, keys: B, compiled: B, tables: B,
    // ranking scratch
    rank_keys: B, rank_vals: B,
    stamps: B,
    wl0: B, wl1: B, cls: B, flag: B, offs: B, crank: B, start: B, lvl: B, term: B, toffs: B, tlist: B, tcount: B, opt: B, samp: B, tscore: B,
}

pub struct Studio<D: KernelExt> {
    pub dev: Arc<D>,
    pub s: D::Stream,
    lib: LibraryId,
    hv: LibraryId,
    prim: LibraryId,
    pub cfg: StudioConfig,
    pub base: Design,
    b: Bufs<D::Buffer>,
    a_observe: ParamUpload<D>,
    a_round: ParamUpload<D>,
    /// Per-level argument blocks: [eval_trial][level].
    a_level: [Vec<ParamUpload<D>>; 2],
    scan: agcad_kernels::primitives::ScanPlan<D>,
    scan_cls: agcad_kernels::primitives::ScanPlan<D>,
    scan_term: agcad_kernels::primitives::ScanPlan<D>,
    a_validate: ParamUpload<D>,
    host_args: DopsArgs,
    a_stamp: Vec<ParamUpload<D>>,
    readback: Cell<u64>,
}

fn up<D: AgDevice>(d: &D, s: &D::Stream, bytes: &[u8]) -> DriverResult<D::Buffer> {
    let b = d.alloc_bytes(s, bytes.len().max(16))?;
    if !bytes.is_empty() { d.upload_from_host_async(s, bytes, &b, 0)?; }
    Ok(b)
}

impl<D: KernelExt> Studio<D> {
    pub fn new(dev: Arc<D>, base: &Design, goal: &Goal, cfg: StudioConfig) -> DriverResult<Self> { Self::new_shard(dev, base, goal, cfg, 0) }

    /// A shard holding global configurations `cfg_offset .. cfg_offset + cfg.configs`.
    pub fn new_shard(dev: Arc<D>, base: &Design, goal: &Goal, cfg: StudioConfig, cfg_offset: u32) -> DriverResult<Self> { Self::new_with(dev, base, goal, cfg, cfg_offset, Priority::High) }

    /// Full constructor: shard offset and stream priority (rollouts high; certification and
    /// escalation low, so they overlap without delaying rollouts).
    pub fn new_with(dev: Arc<D>, base: &Design, goal: &Goal, cfg: StudioConfig, cfg_offset: u32, priority: Priority) -> DriverResult<Self> {
        let s = dev.stream(priority)?;
        let lib = dev.load_library(&dops_library())?;
        let hv = dev.load_library(&agcad_harness::wavefront::harness_library())?;
        let prim = dev.load_library(&agcad_kernels_cuda::primitives_library())?;
        let (c, n, r) = (cfg.configs as usize, cfg.nodes as usize, cfg.records as usize);
        assert!(cfg.levels >= 1 && (cfg.coarse << (cfg.levels - 1)) <= 1024, "finest resolution must be ≤ 1024 per axis");
        assert!(cfg.cap >= cfg.coarse.pow(3), "cap must hold the root grid of one configuration");
        let island = (cfg.island as usize).clamp(2, MAX_ISLAND);
        assert!(cfg.policy != policy::CMAES || cfg_offset as usize % island == 0, "shards must start on a CMA-ES island boundary");
        let full = c * cfg.cap as usize;
        let capacity = if cfg.max_cells > 0 { full.min(cfg.max_cells as usize) } else { full };
        assert!(capacity >= c * cfg.coarse.pow(3) as usize, "max_cells must hold the root grids of all configurations");
        let k = base.params.len().max(1);
        assert!(base.nodes.len() <= n, "base design has more features than node capacity");
        assert!(base.params.len() <= MAX_PARAMS, "at most {MAX_PARAMS} parameters");
        let mut nodes = vec![Node::zeroed(); c * n];
        let mut params = vec![0.0f64; c * k];
        for i in 0..c {
            nodes[i * n..i * n + base.nodes.len()].copy_from_slice(&base.nodes);
            params[i * k..i * k + base.params.len()].copy_from_slice(&base.params);
        }
        let d = &*dev;
        // device-side zeroing (stream-ordered memset; no host staging buffers)
        let zeros = |bytes: usize| -> DriverResult<D::Buffer> {
            let b = d.alloc_bytes(&s, bytes.max(16))?;
            d.memset_async(&s, &b, 0, bytes.max(16), 0)?;
            Ok(b)
        };
        let b = Bufs {
            goal: up(d, &s, bytemuck::bytes_of(goal))?,
            cur_nodes: up(d, &s, bytemuck::cast_slice(&nodes))?,
            cur_count: up(d, &s, bytemuck::cast_slice(&vec![base.nodes.len() as u32; c]))?,
            cur_params: up(d, &s, bytemuck::cast_slice(&params))?,
            tr_nodes: zeros(c * n * NODE_B)?,
            tr_count: zeros(c * 4)?,
            tr_params: zeros(c * k * 8)?,
            rev: zeros(c * 4)?,
            budget: up(d, &s, bytemuck::cast_slice(&vec![cfg.budget; c]))?,
            done: zeros(c * 4)?,
            cmds: zeros(c * r * 64)?,
            arena: zeros(c * r * std::mem::size_of::<FeatureArgs>())?,
            status: zeros(c * r * 16)?,
            counts: zeros(c * CS * 8)?,
            domain_hit: zeros(c * 4)?,
            bbox: zeros(c * 48)?,
            obs: zeros(c * OBS * 8)?,
            score: zeros(c * 8)?,
            trace: zeros(cfg.max_rounds as usize * c * 32)?,
            round: zeros(4)?,
            active: zeros(4)?,
            allow: up(d, &s, bytemuck::cast_slice(&crate::registry::allow_list(agcad_ir::Catalog::global())))?,
            keys: zeros(c * r * 8)?,
            compiled: zeros(c * n * std::mem::size_of::<CNode>())?,
            tables: zeros(c * n * TABLE * 8)?,
            rank_keys: zeros(c * 8)?,
            rank_vals: zeros(c * 4)?,
            stamps: zeros(cfg.max_rounds as usize * STAGES * 8)?,
            wl0: zeros(capacity * 8)?,
            wl1: zeros(capacity * 8)?,
            cls: zeros(capacity * 4)?,
            flag: zeros(capacity * 4)?,
            offs: zeros(capacity * 4)?,
            crank: zeros(capacity * 4)?,
            start: zeros(c * 4)?,
            lvl: zeros((cfg.levels as usize + 1) * 4)?,
            term: zeros(capacity * 4)?,
            toffs: zeros(capacity * 4)?,
            tlist: zeros(capacity * 4)?,
            tcount: zeros(4)?,
            opt: zeros(c.div_ceil(island) * std::mem::size_of::<OptState>())?,
            samp: zeros(c * MAX_PARAMS * 8)?,
            tscore: zeros(c * 8)?,
        };
        let mk = |eval_trial: u32, handle: u64, level: u32| DopsArgs {
            goal: b.goal.device_ptr(), cur_nodes: b.cur_nodes.device_ptr(), cur_count: b.cur_count.device_ptr(), cur_params: b.cur_params.device_ptr(),
            tr_nodes: b.tr_nodes.device_ptr(), tr_count: b.tr_count.device_ptr(), tr_params: b.tr_params.device_ptr(),
            rev: b.rev.device_ptr(), budget: b.budget.device_ptr(), done: b.done.device_ptr(), cmds: b.cmds.device_ptr(), arena: b.arena.device_ptr(),
            status: b.status.device_ptr(), counts: b.counts.device_ptr(), domain_hit: b.domain_hit.device_ptr(), bbox: b.bbox.device_ptr(),
            obs: b.obs.device_ptr(), score: b.score.device_ptr(), trace: b.trace.device_ptr(), round: b.round.device_ptr(), active: b.active.device_ptr(), compiled: b.compiled.device_ptr(), tables: b.tables.device_ptr(),
            wl0: b.wl0.device_ptr(), wl1: b.wl1.device_ptr(), cls: b.cls.device_ptr(), flag: b.flag.device_ptr(), offs: b.offs.device_ptr(), crank: b.crank.device_ptr(), start: b.start.device_ptr(), lvl: b.lvl.device_ptr(), term: b.term.device_ptr(), toffs: b.toffs.device_ptr(), tlist: b.tlist.device_ptr(), tcount: b.tcount.device_ptr(), opt: b.opt.device_ptr(), samp: b.samp.device_ptr(), tscore: b.tscore.device_ptr(),
            cond_handle: handle, configs: cfg.configs, nodes: cfg.nodes, params: k as u32, records: cfg.records, coarse: cfg.coarse, levels: cfg.levels,
            eval_trial, max_rounds: cfg.max_rounds, level, capacity: capacity as u32, cap: cfg.cap, cfg_offset, tighten: cfg.tighten, policy: cfg.policy, island: island as u32, _pad2: 0, ops: crate::registry::op_table(agcad_ir::Catalog::global()),
        };
        let a_observe = upload_params(d, &s, &mk(0, 0, 0))?;
        let host_args = mk(1, 0, 0);
        let a_level = [(0..cfg.levels).map(|l| upload_params(d, &s, &mk(0, 0, l))).collect::<DriverResult<Vec<_>>>()?, (0..cfg.levels).map(|l| upload_params(d, &s, &mk(1, 0, l))).collect::<DriverResult<Vec<_>>>()?];
        let scan = agcad_kernels::primitives::ScanPlan::new(d, &s, &b.flag, &b.offs, capacity as u32)?;
        let scan_cls = agcad_kernels::primitives::ScanPlan::new(d, &s, &b.cls, &b.crank, capacity as u32)?;
        let scan_term = agcad_kernels::primitives::ScanPlan::new(d, &s, &b.term, &b.toffs, capacity as u32)?;
        let a_round = upload_params(d, &s, &host_args)?;
        let a_validate = upload_params(d, &s, &agcad_harness::wavefront::ValidateParams {
            cmds: b.cmds.device_ptr(), status: b.status.device_ptr(), allow: b.allow.device_ptr(), budget: b.budget.device_ptr(), revisions: b.rev.device_ptr(), keys: b.keys.device_ptr(), n: cfg.configs * cfg.records, configs: cfg.configs,
        })?;
        let a_stamp = (0..STAGES as u32).map(|stage| upload_params(d, &s, &StampParams { stamps: b.stamps.device_ptr(), round: b.round.device_ptr(), stage, max_rounds: cfg.max_rounds })).collect::<DriverResult<Vec<_>>>()?;
        Ok(Studio { dev, s, lib, hv, prim, cfg, base: base.clone(), b, a_observe, a_round, a_level, scan, scan_cls, scan_term, a_validate, host_args, a_stamp, readback: Cell::new(0) })
    }

    /// Re-initialize the rollout state in place for a new base design and goal (same shape:
    /// ≤ node capacity, same parameter count). Buffers and their addresses are kept, so a
    /// captured graph stays valid.
    pub fn reset(&mut self, base: &Design, goal: &Goal) -> DriverResult<()> {
        let (c, n) = (self.cfg.configs as usize, self.cfg.nodes as usize);
        let k = self.base.params.len().max(1);
        if base.nodes.len() > n || base.params.len().max(1) != k {
            return Err(EnqueueError::InvalidConfig(format!("reset: shape changed ({} nodes / {} params vs capacity {n} / {k})", base.nodes.len(), base.params.len())).into());
        }
        let d = &*self.dev;
        let mut nodes = vec![Node::zeroed(); c * n];
        let mut params = vec![0.0f64; c * k];
        for i in 0..c {
            nodes[i * n..i * n + base.nodes.len()].copy_from_slice(&base.nodes);
            params[i * k..i * k + base.params.len()].copy_from_slice(&base.params);
        }
        d.upload_from_host_async(&self.s, bytemuck::cast_slice(&nodes), &self.b.cur_nodes, 0)?;
        d.upload_from_host_async(&self.s, bytemuck::cast_slice(&vec![base.nodes.len() as u32; c]), &self.b.cur_count, 0)?;
        d.upload_from_host_async(&self.s, bytemuck::cast_slice(&params), &self.b.cur_params, 0)?;
        d.upload_from_host_async(&self.s, bytemuck::bytes_of(goal), &self.b.goal, 0)?;
        d.upload_from_host_async(&self.s, bytemuck::cast_slice(&vec![self.cfg.budget; c]), &self.b.budget, 0)?;
        for (buf, bytes) in [(&self.b.rev, c * 4), (&self.b.done, c * 4), (&self.b.round, 4), (&self.b.active, 4), (&self.b.trace, self.cfg.max_rounds as usize * c * 32), (&self.b.stamps, self.cfg.max_rounds as usize * STAGES * 8), (&self.b.obs, c * OBS * 8), (&self.b.score, c * 8), (&self.b.samp, c * MAX_PARAMS * 8), (&self.b.tscore, c * 8)] {
            d.memset_async(&self.s, buf, 0, bytes, 0)?;
        }
        d.sync(&self.s)?;
        self.base = base.clone();
        self.readback.set(0);
        Ok(())
    }

    fn per_cfg(&self) -> LaunchConfig { LaunchConfig::linear(self.cfg.configs as u64, 128) }
    /// Grid-stride launch over worklist entries (sizes live on the device).
    fn per_entry(&self) -> LaunchConfig { LaunchConfig::linear((self.cfg.configs as u64 * self.cfg.cap as u64).min(8192 * 256), 256) }

    /// Adaptive interval evaluation of the current (t = 0) or trial (t = 1) tapes.
    fn enqueue_eval(&self, t: usize) -> DriverResult<()> {
        let d = &*self.dev;
        let a0 = &self.a_level[t][0].buffer;
        d.launch(&self.s, self.lib, "ag_ad_seed", &self.per_entry(), a0)?;
        for l in 0..self.cfg.levels as usize {
            let a = &self.a_level[t][l].buffer;
            d.launch(&self.s, self.lib, "ag_ad_classify", &self.per_entry(), a)?;
            d.launch(&self.s, self.lib, "ag_ad_start", &self.per_entry(), a)?;
            self.scan_cls.launch(d, &self.s, self.prim)?;
            d.launch(&self.s, self.lib, "ag_ad_resolve", &self.per_entry(), a)?;
            self.scan.launch(d, &self.s, self.prim)?;
            d.launch(&self.s, self.lib, "ag_ad_clip", &self.per_entry(), a)?;
            self.scan_term.launch(d, &self.s, self.prim)?;
            d.launch(&self.s, self.lib, "ag_ad_compact", &self.per_entry(), a)?;
            d.launch(&self.s, self.lib, "ag_ad_fill", &self.per_entry(), a)?;
            if l + 1 < self.cfg.levels as usize {
                d.launch(&self.s, self.lib, "ag_ad_emit", &self.per_entry(), a)?;
            }
        }
        Ok(())
    }

    /// Evaluate the current tapes and write observations (before the first round).
    pub fn observe(&self) -> DriverResult<()> {
        let d = &*self.dev;
        d.launch(&self.s, self.lib, "ag_dop_compile", &self.per_cfg(), &self.a_observe.buffer)?;
        d.launch(&self.s, self.lib, "ag_dop_eval_clear", &self.per_cfg(), &self.a_observe.buffer)?;
        self.enqueue_eval(0)?;
        d.launch(&self.s, self.lib, "ag_dop_decide", &self.per_cfg(), &self.a_observe.buffer)?;
        d.launch(&self.s, self.lib, "ag_opt_init", &LaunchConfig::linear(1, 32), &self.a_observe.buffer)?;
        Ok(())
    }

    /// Enqueue one round (no allocation, no host synchronization: capturable into a graph).
    pub fn enqueue_round(&self, cont: &D::Buffer) -> DriverResult<()> {
        let d = &*self.dev;
        let a = &self.a_round.buffer;
        let one = LaunchConfig::linear(1, 32);
        let stamp = |k: usize| d.launch(&self.s, self.lib, "ag_dop_stamp", &one, &self.a_stamp[k].buffer);
        stamp(0)?;
        d.launch(&self.s, self.lib, "ag_dop_policy", &self.per_cfg(), a)?;
        stamp(1)?;
        d.launch(&self.s, self.hv, "ag_hv_validate", &LaunchConfig::linear((self.cfg.configs * self.cfg.records) as u64, 256), &self.a_validate.buffer)?;
        stamp(2)?;
        d.launch(&self.s, self.lib, "ag_dop_apply", &self.per_cfg(), a)?;
        stamp(3)?;
        d.launch(&self.s, self.lib, "ag_dop_compile", &self.per_cfg(), a)?;
        stamp(4)?;
        d.launch(&self.s, self.lib, "ag_dop_eval_clear", &self.per_cfg(), a)?;
        stamp(5)?;
        self.enqueue_eval(1)?;
        stamp(6)?;
        d.launch(&self.s, self.lib, "ag_dop_decide", &self.per_cfg(), a)?;
        stamp(7)?;
        let islands = (self.cfg.configs as u64).div_ceil(self.cfg.island.clamp(2, MAX_ISLAND as u32) as u64);
        d.launch(&self.s, self.lib, "ag_opt_update", &LaunchConfig::linear(islands, 64), a)?;
        stamp(8)?;
        d.launch(&self.s, self.lib, "ag_dop_continue", &LaunchConfig::linear(1, 32), cont)?;
        Ok(())
    }

    fn read<T: bytemuck::Pod>(&self, buf: &D::Buffer, offset: usize, n: usize) -> DriverResult<Vec<T>> {
        let bytes = n * std::mem::size_of::<T>();
        let mut pin = self.dev.pinned(bytes.max(1))?;
        self.dev.download_packed_async(&self.s, &[Region { buf, offset, bytes }], &mut pin)?;
        self.dev.sync(&self.s)?;
        Ok(bytemuck::pod_collect_to_vec(&pin.as_bytes()[..bytes]))
    }

    /// Host-driven loop (CPU reference device, eager CUDA): one 4-byte flag read per round.
    pub fn run_host_loop(&self) -> DriverResult<u32> {
        self.observe()?;
        let cont = &self.a_round.buffer; // cond_handle = 0
        let mut rounds = 0;
        loop {
            self.enqueue_round(cont)?;
            rounds += 1;
            let flag: Vec<u32> = self.read(&self.b.active, 0, 1)?;
            self.readback.set(self.readback.get() + 4);
            if flag[0] == 0 { break; }
        }
        Ok(rounds)
    }

    /// Rank by score on the device, then read back the top-k designs (the only bulk transfer).
    pub fn results(&self, k: usize) -> DriverResult<RolloutSummary> {
        let d = &*self.dev;
        let c = self.cfg.configs as usize;
        agcad_kernels::primitives::stable_order_by_score(d, &self.s, self.prim, &self.b.score, &self.b.rank_keys, &self.b.rank_vals, self.cfg.configs, false)?;
        let order: Vec<u32> = self.read(&self.b.rank_vals, 0, k.min(c))?;
        let done: Vec<u32> = self.read(&self.b.done, 0, c)?;
        let round: Vec<u32> = self.read(&self.b.round, 0, 1)?;
        let (n, kp) = (self.cfg.nodes as usize, self.base.params.len().max(1));
        let mut top = Vec::new();
        for &id in &order {
            let i = id as usize;
            let count: Vec<u32> = self.read(&self.b.cur_count, i * 4, 1)?;
            let nodes: Vec<Node> = self.read(&self.b.cur_nodes, i * n * NODE_B, count[0] as usize)?;
            let params: Vec<f64> = self.read(&self.b.cur_params, i * kp * 8, kp)?;
            let obs: Vec<f64> = self.read(&self.b.obs, i * OBS * 8, OBS)?;
            let score: Vec<f64> = self.read(&self.b.score, i * 8, 1)?;
            let mut o = [0.0; OBS];
            o.copy_from_slice(&obs);
            top.push(Survivor { configuration: id, score: score[0], obs: o, design: Design { nodes, params: params[..self.base.params.len()].to_vec(), names: self.base.names.clone() } });
        }
        Ok(RolloutSummary { rounds: round[0], done: done.iter().filter(|x| **x != 0).count() as u32, top, in_loop_readback_bytes: self.readback.get() })
    }

    /// Per-round stage durations (ns) from the device timestamps: [round][stage] in the
    /// order of [`STAGE_NAMES`].
    pub fn stage_times(&self, rounds: u32) -> DriverResult<Vec<[u64; STAGES - 1]>> {
        let st: Vec<u64> = self.read(&self.b.stamps, 0, self.cfg.max_rounds as usize * STAGES)?;
        Ok((0..rounds.min(self.cfg.max_rounds) as usize).map(|r| {
            let mut d = [0u64; STAGES - 1];
            for k in 0..STAGES - 1 { d[k] = st[r * STAGES + k + 1].saturating_sub(st[r * STAGES + k]); }
            d
        }).collect())
    }

    /// Trace ring (rounds × configurations): training data for learned device policies.
    pub fn trace(&self) -> DriverResult<Vec<TraceRow>> { self.read(&self.b.trace, 0, self.cfg.max_rounds as usize * self.cfg.configs as usize) }
    /// Command records and statuses of the last round (conformance / debugging).
    pub fn last_round(&self) -> DriverResult<(Vec<CommandRecord>, Vec<StatusRecord>)> {
        let n = (self.cfg.configs * self.cfg.records) as usize;
        Ok((self.read(&self.b.cmds, 0, n)?, self.read(&self.b.status, 0, n)?))
    }
    /// Raw device state for parity tests: (current nodes, params, obs, trace) bytes.
    pub fn state_bytes(&self) -> DriverResult<Vec<u8>> {
        let c = self.cfg.configs as usize;
        let mut out: Vec<u8> = self.read(&self.b.cur_nodes, 0, c * self.cfg.nodes as usize * NODE_B)?;
        out.extend(self.read::<u8>(&self.b.cur_params, 0, c * self.base.params.len().max(1) * 8)?);
        out.extend(self.read::<u8>(&self.b.obs, 0, c * OBS * 8)?);
        out.extend(self.read::<u8>(&self.b.trace, 0, self.cfg.max_rounds as usize * c * 32)?);
        out.extend(self.read::<u8>(&self.b.cmds, 0, c * self.cfg.records as usize * 64)?);
        out.extend(self.read::<u8>(&self.b.status, 0, c * self.cfg.records as usize * 16)?);
        out.extend(self.read::<u8>(&self.b.samp, 0, c * MAX_PARAMS * 8)?);
        out.extend(self.read::<u8>(&self.b.tscore, 0, c * 8)?);
        Ok(out)
    }
}

impl<D: KernelExt + ConditionalExt> Studio<D> {
    /// Capture the round body into a graph with a device-driven WHILE node and instantiate it.
    fn build_graph(&self) -> DriverResult<(D::Graph, D::GraphExec, ParamUpload<D>)> {
        let d = &*self.dev;
        let mut g = d.new_graph()?;
        let (h, bodies) = d.conditional(&mut g, CondKind::While, 1)?;
        // continue-kernel arguments carrying the WHILE handle (allocated before capture)
        let mut a = self.host_args;
        a.cond_handle = h.0;
        let cont = upload_params(d, &self.s, &a)?;
        d.sync(&self.s)?;
        d.capture_into(&self.s, &bodies[0], &mut || self.enqueue_round(&cont.buffer))?;
        let exec = d.instantiate(&g)?;
        Ok((g, exec, cont))
    }

    /// Whole rollout as ONE graph launch: rounds run inside a device-driven conditional WHILE
    /// node; the host neither reads nor writes anything until the segment ends.
    pub fn run_graph(&self) -> DriverResult<()> {
        self.observe()?;
        let (_g, exec, _cont) = self.build_graph()?;
        self.dev.launch_graph(&self.s, &exec)?;
        self.dev.sync(&self.s)?;
        Ok(())
    }
}

/// A studio with its instantiated rollout graph, reused across rollouts of the same shape:
/// [`GraphRollout::run`] after [`GraphRollout::reset`] only rewrites device buffers and
/// relaunches (no capture, no instantiation, no allocation).
pub struct GraphRollout<D: KernelExt + ConditionalExt> {
    pub studio: Studio<D>,
    _graph: D::Graph,
    exec: D::GraphExec,
    _cont: ParamUpload<D>,
}

impl<D: KernelExt + ConditionalExt> GraphRollout<D> {
    pub fn new(studio: Studio<D>) -> DriverResult<Self> {
        let (g, exec, cont) = studio.build_graph()?;
        Ok(GraphRollout { studio, _graph: g, exec, _cont: cont })
    }

    /// New base design / goal of the same shape (feature capacity, parameter count).
    pub fn reset(&mut self, base: &Design, goal: &Goal) -> DriverResult<()> { self.studio.reset(base, goal) }

    pub fn run(&self) -> DriverResult<()> {
        let st = &self.studio;
        st.observe()?;
        st.dev.launch_graph(&st.s, &self.exec)?;
        st.dev.sync(&st.s)?;
        Ok(())
    }
}

/// Escalation step of the evidence chain (spec 14): re-evaluate one design alone with a deep,
/// high-capacity adaptive pass. Returns its observation row (rigorous volume / extent bounds,
/// sampled estimate). Memory is O(cap) for one configuration instead of configs × cap.
pub fn certify<D: KernelExt>(dev: Arc<D>, design: &Design, goal: &Goal, levels: u32, cap: u32) -> DriverResult<[f64; OBS]> { certify_with(dev, design, goal, levels, cap, CERTIFY_TIGHTEN) }

/// Interval evaluations per boundary cell used by [`certify`].
pub const CERTIFY_TIGHTEN: u32 = 96;

pub fn certify_with<D: KernelExt>(dev: Arc<D>, design: &Design, goal: &Goal, levels: u32, cap: u32, tighten: u32) -> DriverResult<[f64; OBS]> {
    let cfg = StudioConfig { configs: 1, nodes: design.nodes.len().max(1) as u32, records: 1, max_rounds: 1, coarse: 8, levels, cap, budget: 1, tighten, max_cells: 0, policy: 0, island: 16 };
    let st = Studio::new_with(dev, design, goal, cfg, 0, Priority::Low)?;
    st.observe()?;
    let o: Vec<f64> = st.read(&st.b.obs, 0, OBS)?;
    let mut out = [0.0; OBS];
    out.copy_from_slice(&o);
    Ok(out)
}

/// Multi-device rollout (spec 05 § Multi-GPU): configurations are sharded contiguously over
/// `devs` (`sched::shard`), shards run concurrently, and survivors are merged by (score order
/// key, global configuration id) with `sched::merge_topk` — identical to one device.
pub fn rollout_sharded<D: KernelExt + Send + Sync>(devs: &[Arc<D>], base: &Design, goal: &Goal, cfg: StudioConfig, top_k: usize, run: &(dyn Fn(&Studio<D>) -> DriverResult<()> + Sync)) -> DriverResult<RolloutSummary>
where D::Stream: Send, D::Buffer: Send {
    // CMA-ES islands stay whole: shard in units of islands
    let unit = if cfg.policy == policy::CMAES { (cfg.island as usize).clamp(2, MAX_ISLAND) as u32 } else { 1 };
    let ranges: Vec<std::ops::Range<u32>> = agcad_runtime::sched::shard(cfg.configs.div_ceil(unit), devs.len() as u32).into_iter().map(|r| (r.start * unit).min(cfg.configs)..(r.end * unit).min(cfg.configs)).collect();
    let shards: Vec<DriverResult<RolloutSummary>> = std::thread::scope(|sc| {
        let hs: Vec<_> = ranges.iter().zip(devs).filter(|(r, _)| !r.is_empty()).map(|(r, d)| {
            let (d, r) = (d.clone(), r.clone());
            sc.spawn(move || -> DriverResult<RolloutSummary> {
                let st = Studio::new_shard(d, base, goal, StudioConfig { configs: r.end - r.start, ..cfg }, r.start)?;
                run(&st)?;
                let mut s = st.results(top_k)?;
                for t in &mut s.top { t.configuration += r.start; }
                Ok(s)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().expect("shard thread panicked")).collect()
    });
    let shards: Vec<RolloutSummary> = shards.into_iter().collect::<DriverResult<_>>()?;
    let keyed: Vec<Vec<(u64, u32)>> = shards.iter().map(|s| s.top.iter().map(|t| (agcad_kernels::primitives::f64_order_key(t.score, false), t.configuration)).collect()).collect();
    let order = agcad_runtime::sched::merge_topk(&keyed, top_k);
    let all: Vec<&Survivor> = shards.iter().flat_map(|s| s.top.iter()).collect();
    let top = order.iter().filter_map(|id| all.iter().find(|t| t.configuration == *id).map(|t| (*t).clone())).collect();
    Ok(RolloutSummary {
        rounds: shards.iter().map(|s| s.rounds).max().unwrap_or(0),
        done: shards.iter().map(|s| s.done).sum(),
        top,
        in_loop_readback_bytes: shards.iter().map(|s| s.in_loop_readback_bytes).sum(),
    })
}
