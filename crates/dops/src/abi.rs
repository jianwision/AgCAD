//! Device ABI of the operation layer (spec 04 bytecode, spec 05 proxy phase). All structs are
//! `repr(C)` Pod and mirrored field-for-field in `cu/dops.cu`.
//!
//! A configuration's design is a **feature tape**: the specification tree of Part Design
//! features in order (pad / pocket / hole over rectangle, circle and regular-polygon sketches,
//! optional circular pattern, active flag). Every numeric value is a literal or a reference to
//! the configuration's parameter table (`value = param[pref] + val`), so a design-table row
//! changes the geometry without changing the tape.

use bytemuck::{Pod, Zeroable};

/// `pref` sentinel: the value is the literal `val` only.
pub const NO_PARAM: u32 = u32::MAX;
/// Value slots per feature: 0..=5 profile + limits; 6, 7 pattern (circular: count, total
/// angle; rectangular: count₁, spacing₁ along x); 8, 9 rectangular count₂, spacing₂ along y;
/// 10 corner size (fillet radius / chamfer length of a rectangle profile); 11 reserved.
pub const NVALS: usize = 12;
/// Observation row width (f64) per configuration: 0 vol_lo, 1 vol_hi (bounded), 2–4 extent lo,
/// 5–7 extent hi (bounded), 8 domain hit, 9 features, 10 goal met, 11 relative error,
/// 12 volume estimate (sampled).
pub const OBS: usize = 16;
/// Maximum parameters per configuration.
pub const MAX_PARAMS: usize = 8;
/// Maximum pattern instances evaluated per feature.
pub const MAX_PATTERN: u32 = 64;

/// Sketch profile kinds (the `kind` of a feature node).
pub mod profile {
    /// v0..v3 = x0, y0, x1, y1 (corners), v4, v5 = z0, z1.
    pub const RECT: u32 = 0;
    /// v0, v1 = centre, v2 = radius, v4, v5 = z0, z1.
    pub const CIRCLE: u32 = 1;
    /// v0, v1 = centre, v2 = circumradius, v3 = sides (3..=64), v4, v5 = z0, z1.
    pub const POLYGON: u32 = 2;
    /// part.shell of the current body: v0 = wall thickness (inward). Device-only (no exact
    /// layered offset yet).
    pub const SHELL: u32 = 3;
    /// document.body: start a new body (the current one is pushed, depth ≤ [`super::MAX_BODIES`]).
    pub const BODY: u32 = 4;
    /// boolean.add / remove / intersect: combine the pushed body (target) with the current
    /// one (tool); `op` = fop::ADD / REMOVE / INTERSECT.
    pub const COMBINE: u32 = 5;
}

/// Bodies on the evaluation stack (document.body nesting).
pub const MAX_BODIES: usize = 4;

/// Feature boolean against the body so far.
pub mod fop {
    pub const ADD: u32 = 0; // part.pad
    pub const REMOVE: u32 = 1; // part.pocket / part.hole
    pub const INTERSECT: u32 = 2; // boolean.intersect
}

pub mod nflags {
    pub const ACTIVE: u32 = 1;
    /// Circular pattern about z (v6 count, v7 total angle).
    pub const PATTERN: u32 = 2;
    /// Rectangular pattern along x / y (v6 count₁, v7 spacing₁, v8 count₂, v9 spacing₂).
    pub const PATTERN_RECT: u32 = 4;
    /// Rounded vertical edges of a rectangle profile (v10 radius).
    pub const FILLET: u32 = 8;
    /// Chamfered vertical edges of a rectangle profile (v10 length).
    pub const CHAMFER: u32 = 16;
}

/// One feature of the specification tree on the device (160 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Node {
    pub kind: u32,
    pub op: u32,
    pub flags: u32,
    /// Index of the command record that created the feature (provenance, stable naming).
    pub src: u32,
    pub pref: [u32; NVALS],
    pub val: [f64; NVALS],
}

/// A feature compiled for evaluation (128 B): parameter references resolved and values
/// rounded to f32 (evaluation precision on consumer GPUs, spec 08 precision policy); its
/// pattern instance transforms and polygon edge normals live in a [`TABLE`]-f32 table per
/// node: 0..256 instances as (cos, sin, tx, ty) — the instance is evaluated at R(−θ)(p − t) —
/// and 256..384 polygon normal (cos, sin) pairs.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct CNode {
    pub kind: u32,
    pub op: u32,
    pub active: u32,
    /// Pattern instances (1 without a pattern).
    pub count: u32,
    /// Polygon sides (0 for other profiles).
    pub sides: u32,
    /// Profile modifiers (`nflags::FILLET` / `nflags::CHAMFER`).
    pub mods: u32,
    pub _pad: [u32; 2],
    pub v: [f32; NVALS],
    pub apothem: f32,
    pub _pad2: [f32; 11],
}

/// f32 entries of a node's table.
pub const TABLE: usize = 384;
/// Offset of the polygon normals in a node's table.
pub const POLY: usize = 256;

/// Arg-arena payload of a feature-creating command record (144 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct FeatureArgs {
    pub pref: [u32; NVALS],
    pub val: [f64; NVALS],
}

/// Goal and limits of a device rollout. Set by the LLM (`set_goal`) or a study; read by the
/// device policy and by the device checks.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Goal {
    /// Evaluation domain (mm); geometry reaching its border is rejected (DOMAIN).
    pub domain_lo: [f64; 3],
    pub domain_hi: [f64; 3],
    /// Target volume interval (mm³): the configuration is done when its rigorous volume
    /// interval lies inside it.
    pub vol_lo: f64,
    pub vol_hi: f64,
    /// Hard limit on the bounding-box extent (mm): edits that definitely exceed it are rejected.
    pub max_ext: [f64; 3],
    pub _pad0: f64,
    /// Where structural actions may place new holes (x0, y0, x1, y1) and their size / depth.
    pub place: [f64; 4],
    pub hole_r: f64,
    pub hole_z0: f64,
    pub hole_z1: f64,
    /// Controller gain (fraction of a parameter's range per unit relative error).
    pub gain: f64,
    pub param_lo: [f64; MAX_PARAMS],
    pub param_hi: [f64; MAX_PARAMS],
    /// +1 if increasing the parameter increases volume, -1 otherwise (the mock model's prior).
    pub param_sign: [f64; MAX_PARAMS],
    pub n_params: u32,
    pub seed: u32,
    pub _pad1: [u32; 2],
}

/// Per configuration and round (trace ring; training data for learned policies).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct TraceRow {
    pub round: u32,
    pub committed: u32,
    pub accepted: u32,
    pub rejected: u32,
    pub vol_lo: f64,
    pub vol_hi: f64,
}

/// Opcodes the kernels dispatch on, resolved from the catalog at run time (never hard-coded
/// in device code, so catalog renumbering cannot desynchronize host and device).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OpTable {
    pub update: u32,
    pub parameter: u32,
    pub pad: u32,
    pub pocket: u32,
    pub hole: u32,
    pub circular_pattern: u32,
    pub deactivate: u32,
    pub rect_pattern: u32,
    pub fillet: u32,
    pub chamfer: u32,
    pub shell: u32,
    pub body: u32,
    pub add: u32,
    pub remove: u32,
    pub intersect: u32,
    pub _pad: u32,
}

/// Every device pointer and size; one block shared by all dops kernels.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct DopsArgs {
    pub goal: u64,
    pub cur_nodes: u64,
    pub cur_count: u64,
    pub cur_params: u64,
    pub tr_nodes: u64,
    pub tr_count: u64,
    pub tr_params: u64,
    pub rev: u64,
    pub budget: u64,
    pub done: u64,
    pub cmds: u64,
    pub arena: u64,
    pub status: u64,
    /// u64 × 4 per configuration: inside cells, boundary cells, quantized fill of boundary cells
    /// (2⁻²⁰ cell units), reserved.
    pub counts: u64,
    /// u32 per configuration: geometry reached the domain border.
    pub domain_hit: u64,
    /// u32 × 12 per configuration: inside-cell bbox min xyz, max xyz; any-cell bbox min, max.
    pub bbox: u64,
    pub obs: u64,
    pub score: u64,
    pub trace: u64,
    pub round: u64,
    pub active: u64,
    /// Compiled tapes ([`CNode`] × nodes per configuration) and their trig tables.
    pub compiled: u64,
    pub tables: u64,
    /// Adaptive evaluation: ping-pong cell worklists (u64 entries: cfg << 32 | ix << 20 |
    /// iy << 10 | iz, sorted by configuration), per-entry class (1 = ambiguous) and its
    /// exclusive scan (`crank`), refine flag and its scan (`offs`), per-configuration first
    /// worklist index (`start`), per-level entry counts (levels + 1).
    pub wl0: u64,
    pub wl1: u64,
    pub cls: u64,
    pub flag: u64,
    pub offs: u64,
    pub crank: u64,
    pub start: u64,
    pub lvl: u64,
    /// CUgraphConditionalHandle of the enclosing WHILE node (0 = host-driven loop).
    pub cond_handle: u64,
    pub configs: u32,
    pub nodes: u32,
    pub params: u32,
    pub records: u32,
    /// Root grid resolution per axis (level 0).
    pub coarse: u32,
    /// Refinement levels (finest resolution = coarse << (levels − 1)).
    pub levels: u32,
    /// 0 = evaluate the current tapes (observe), 1 = evaluate the trial tapes (decide).
    pub eval_trial: u32,
    pub max_rounds: u32,
    /// Level of this launch (adaptive kernels).
    pub level: u32,
    /// Worklist capacity (= configs × cap).
    pub capacity: u32,
    /// Max cells per configuration and level: a configuration refines its first cap / 8
    /// ambiguous cells (worklist order); the rest are counted as boundary.
    pub cap: u32,
    /// Global id of local configuration 0 (multi-device shards): the policy hashes and
    /// alternates on global ids, so trajectories do not depend on the sharding.
    pub cfg_offset: u32,
    pub ops: OpTable,
}

/// Command-record argument conventions (`CommandRecord.args`, `imm`).
pub mod rec {
    /// knowledge.parameter: args[0] = parameter index, args[1] = 0 set / 1 add, imm[0] = value.
    pub const PARAM_SET: u32 = 0;
    pub const PARAM_ADD: u32 = 1;
    /// part.pad / part.pocket / part.hole: args[0] = profile kind, payload in the arena slot of
    /// the same record index.
    /// transform.circular_pattern: args[0] = feature index (u32::MAX = last), arena v6 / v7.
    /// transform.rectangular_pattern: args[0] = feature index, arena v6..v9.
    /// part.edge_fillet / part.chamfer: args[0] = feature index (a rectangle feature), arena v10.
    /// document.deactivate: args[0] = feature index (u32::MAX = last), args[1] = 1 active / 0.
    pub const LAST: u32 = u32::MAX;
}

/// Stage timestamps per round (ns): 0 start, 1 policy, 2 validate, 3 apply, 4 compile,
/// 5 clear, 6 eval, 7 decide. Not part of the deterministic state.
pub const STAGES: usize = 8;
pub const STAGE_NAMES: [&str; STAGES - 1] = ["policy", "validate", "apply", "compile", "clear", "eval", "decide"];

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct StampParams {
    pub stamps: u64,
    pub round: u64,
    pub stage: u32,
    pub max_rounds: u32,
}
