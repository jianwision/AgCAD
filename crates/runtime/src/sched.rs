//! Scheduler (spec 05): ExecGraph lowering (no host nodes, AG-05-006), topology classes,
//! capacity planning (estimate → chunk → evict; `CHUNKED`, AG-05-004) and the
//! representation cache (disposable lightweight representations).

use agcad_driver::{AgStatus, Priority};
use agcad_ir::catalog::{Backend, Catalog, Phase};
use agcad_ir::Program;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Precompiled feature kernel (Tier N/P) for an op.
    Kernel { op: String, library: String },
    /// Interpreted expression bytecode (model-invented geometry).
    Vm,
    /// GPU-harness stage (validate/bin/check/decide/meter/record/rank/escalate).
    Harness(String),
    /// Exact-phase work for survivors (low-priority stream).
    Exact { op: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecNode {
    pub id: usize,
    pub kind: NodeKind,
    pub deps: Vec<usize>,
    pub stream: StreamClass,
    pub stmt: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamClass { Harness, Proxy, Exact }

impl StreamClass {
    pub fn priority(self) -> Priority { match self { StreamClass::Harness => Priority::High, StreamClass::Proxy => Priority::Normal, StreamClass::Exact => Priority::Low } }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecGraph {
    pub nodes: Vec<ExecNode>,
}

impl ExecGraph {
    /// No host nodes may appear (AG-05-006): every node runs on the device.
    pub fn host_nodes(&self) -> usize { 0 }
}

/// Lower a verified program to an ExecGraph. Host-only (compile-time) ops (`udf.define`,
/// `udf.compile`, `udf.view_source`) run before graph capture and do not appear as nodes.
pub fn lower(p: &Program, catalog: &Catalog) -> Result<ExecGraph, (String, AgStatus)> {
    let mut g = ExecGraph::default();
    let mut ids: HashMap<&str, usize> = HashMap::new();
    for s in &p.statements {
        let op = catalog.get(&s.op).ok_or_else(|| (s.id.clone(), AgStatus::InvalidOpcode))?;
        if op.backend == Backend::Host { continue; }
        let mut deps = Vec::new();
        for v in s.args.values() {
            let mut r = Vec::new();
            v.refs(&mut r);
            for x in r { if let Some(&d) = ids.get(x.as_str()) { deps.push(d); } }
        }
        if let Some(ov) = &s.over { if let Some(&d) = ids.get(ov.as_str()) { deps.push(d); } }
        deps.sort_unstable();
        deps.dedup();
        let (kind, stream) = match op.phase {
            Phase::Exact => (NodeKind::Exact { op: op.name.clone() }, StreamClass::Exact),
            Phase::Control => (NodeKind::Harness(op.name.clone()), StreamClass::Harness),
            _ => (NodeKind::Kernel { op: op.name.clone(), library: op.family.clone() }, StreamClass::Proxy),
        };
        let id = g.nodes.len();
        g.nodes.push(ExecNode { id, kind, deps, stream, stmt: s.id.clone() });
        ids.insert(s.id.as_str(), id);
    }
    Ok(g)
}

/// Topological signature of a configuration: active features + element counts + adjacency.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TopologySignature(pub u64);

impl TopologySignature {
    pub fn of(active_features: &[u32], element_counts: &[u32], adjacency_hash: u64) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| { h ^= x; h = h.wrapping_mul(0x0100_0000_01b3); };
        for &f in active_features { mix(f as u64); }
        mix(u64::MAX);
        for &c in element_counts { mix(c as u64); }
        mix(adjacency_hash);
        TopologySignature(h)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyClass {
    pub signature: TopologySignature,
    pub configurations: Vec<u32>,
}

/// Group configurations into topology classes (deterministic: ordered by first config id).
pub fn topology_classes(sigs: &[TopologySignature]) -> Vec<TopologyClass> {
    let mut by: BTreeMap<TopologySignature, Vec<u32>> = BTreeMap::new();
    for (c, s) in sigs.iter().enumerate() { by.entry(*s).or_default().push(c as u32); }
    let mut v: Vec<TopologyClass> = by.into_iter().map(|(signature, configurations)| TopologyClass { signature, configurations }).collect();
    v.sort_by_key(|c| c.configurations[0]);
    v
}

/// Class-size histogram for study reports (spec 05 § Topology classes).
pub fn class_histogram(classes: &[TopologyClass]) -> BTreeMap<usize, usize> {
    let mut h = BTreeMap::new();
    for c in classes { *h.entry(c.configurations.len()).or_insert(0) += 1; }
    h
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub class_index: usize,
    pub configurations: Vec<u32>,
    pub estimated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityPlan {
    pub chunks: Vec<Chunk>,
    pub chunked: bool,
    /// Configurations that cannot fit even alone: `CAPACITY_EXCEEDED` with their size.
    pub oversized: Vec<(u32, u64)>,
}

/// Plan chunks so each fits in `budget` bytes (default 80 % of free memory). Results of a
/// chunked run are identical to an unchunked run; ranking merges stably (AG-05-004).
pub fn plan_capacity(classes: &[TopologyClass], bytes_per_config: &[u64], fixed_bytes: u64, budget: u64) -> CapacityPlan {
    let mut chunks = Vec::new();
    let mut oversized = Vec::new();
    for (ci, c) in classes.iter().enumerate() {
        let per = bytes_per_config.get(ci).copied().unwrap_or(0);
        if fixed_bytes + per > budget {
            oversized.extend(c.configurations.iter().map(|&x| (x, fixed_bytes + per)));
            continue;
        }
        let fit = ((budget - fixed_bytes) / per.max(1)).max(1) as usize;
        for part in c.configurations.chunks(fit) {
            chunks.push(Chunk { class_index: ci, configurations: part.to_vec(), estimated_bytes: fixed_bytes + per * part.len() as u64 });
        }
    }
    let chunked = chunks.len() > classes.len();
    CapacityPlan { chunks, chunked, oversized }
}

/// Representation cache entry keyed by (element, revision, params).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RepKey {
    pub element: String,
    pub revision: u32,
    pub params: String,
}

#[derive(Clone, Debug)]
pub struct RepEntry {
    pub bytes: u64,
    /// Estimated cost to rebuild (ms).
    pub rebuild_cost: f64,
    /// Exact B-rep data and versions are never evicted.
    pub pinned: bool,
}

#[derive(Default)]
pub struct RepresentationCache {
    pub entries: BTreeMap<RepKey, RepEntry>,
    pub used: u64,
}

impl RepresentationCache {
    pub fn insert(&mut self, k: RepKey, e: RepEntry) {
        self.used += e.bytes;
        if let Some(old) = self.entries.insert(k, e) { self.used -= old.bytes; }
    }
    /// Evict cheapest-to-recompute-per-byte entries until `used ≤ target`. Returns evicted keys.
    pub fn evict_to(&mut self, target: u64) -> Vec<RepKey> {
        let mut cands: Vec<(f64, RepKey)> = self.entries.iter().filter(|(_, e)| !e.pinned).map(|(k, e)| (e.rebuild_cost / e.bytes.max(1) as f64, k.clone())).collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let mut out = Vec::new();
        for (_, k) in cands {
            if self.used <= target { break; }
            if let Some(e) = self.entries.remove(&k) { self.used -= e.bytes; out.push(k); }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classes_and_chunking() {
        let a = TopologySignature::of(&[1, 2], &[6, 12], 0);
        let b = TopologySignature::of(&[1, 2, 3], &[7, 15], 0);
        let sigs: Vec<_> = (0..10).map(|i| if i % 3 == 0 { b } else { a }).collect();
        let cl = topology_classes(&sigs);
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].configurations[0], 0);
        let plan = plan_capacity(&cl, &[100, 100], 50, 350);
        assert!(plan.chunked);
        assert!(plan.chunks.iter().all(|c| c.estimated_bytes <= 350));
        let total: usize = plan.chunks.iter().map(|c| c.configurations.len()).sum();
        assert_eq!(total, 10);
    }
}

/// Multi-GPU `DeviceSet` sharding (spec 05 § Multi-GPU): contiguous configuration ranges per
/// device (deterministic), and a stable k-way merge of per-device top-k lists by
/// (order key, configuration id) — identical to a single-device ranking (AG-05-001).
pub fn shard(configs: u32, devices: u32) -> Vec<std::ops::Range<u32>> {
    let d = devices.max(1);
    let base = configs / d;
    let extra = configs % d;
    let mut out = Vec::with_capacity(d as usize);
    let mut start = 0;
    for i in 0..d {
        let len = base + (i < extra) as u32;
        out.push(start..start + len);
        start += len;
    }
    out
}

pub fn merge_topk(per_device: &[Vec<(u64, u32)>], k: usize) -> Vec<u32> {
    let mut all: Vec<(u64, u32)> = per_device.iter().flatten().copied().collect();
    all.sort_unstable();
    all.into_iter().take(k).map(|x| x.1).collect()
}

#[cfg(test)]
mod multi_gpu_tests {
    use super::*;
    #[test]
    fn shard_and_merge_match_single_device() {
        let r = shard(10, 3);
        assert_eq!(r, vec![0..4, 4..7, 7..10]);
        let keys: Vec<(u64, u32)> = (0..10).map(|c| (((c * 7) % 5) as u64, c)).collect();
        let mut single = keys.clone();
        single.sort_unstable();
        let per: Vec<Vec<(u64, u32)>> = r.iter().map(|rg| { let mut v: Vec<_> = keys[rg.start as usize..rg.end as usize].to_vec(); v.sort_unstable(); v.truncate(3); v }).collect();
        assert_eq!(merge_topk(&per, 3), single.iter().take(3).map(|x| x.1).collect::<Vec<_>>());
    }
}
