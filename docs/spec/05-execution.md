# 05 — Execution & Parallelism

> Status: **Normative** · Spec v0.3.1 · Area: runtime / scheduler

## Batch axes

| Axis | What varies | Example |
|---|---|---|
| `configuration` | Design-table row: parameter values, branch, active features | 4,096 flange variants |
| `entity` | Topological or procedural element within one configuration | every edge of a fillet chain; every scattered point |
| `sample` | Evaluation point within an element | UV samples on a face; voxel probes |

Four levels of parallelism follow from these axes:
- **L0:** within a sample.
- **L1:** across entities.
- **L2:** across configurations.
- **L3:** across sessions (multi-tenant packing on a shared GPU).

The scheduler flattens `configuration × entity × sample` into one launch whenever the feature's kernel is shape-polymorphic. That holds for every expression-level UDF and most proxy-phase features. Ragged sizes use SoA with segment offsets.

## Batched data layout

Every `Configurations<T>` and per-entity buffer carries a `BatchLayout` that follows NVIDIA library conventions, so cuBLAS/cuSOLVER/CUB/OptiX/NanoVDB interop needs no repacking ([API review](13-api-review.md) A2-6):

```rust
pub enum BatchLayout {
    Strided { count: u32, stride_elems: u64, ld: u32 },    // cuBLAS strided-batched
    Segmented { count: u32, offsets: DeviceSlice<u32> },  // CUB segmented: count+1 offsets [begin_i, begin_{i+1})
    Grouped { groups: Vec<(u32, Shape)> },                // cuBLAS grouped GEMM
}
```

- Array starts are 128-byte aligned.
- df64 values are stored as separate `hi[]` / `lo[]` SoA arrays.
- Uniform shapes (implicit samples, sketch solves of one UDF) use `Strided`. Ragged shapes (faces per configuration) use `Segmented`.

## Multi-GPU

- A `DeviceSet` shards configurations across devices: one context per device, and a read-only document replica on each.
- Each device runs its own GPU-harness wavefront.
- `study.rank` takes a per-device top-k, then a stable k-way merge by configuration id, via NCCL all-gather or on the host for small k. This keeps results deterministic.
- wgpu multi-adapter is host-orchestrated only.
- Milestone: M4.

## Phases

| Phase | Runs on | Purpose | Examples |
|---|---|---|---|
| `control` | GPU harness (host only launches) | transactions, studies, reports | `document.update`, `study.rank` |
| `proxy` | GPU | lightweight evaluation of all configurations | implicit eval, tessellation, inertia from mesh, thickness, clash, render |
| `exact` | GPU | exact B-rep work for survivors | SSI marching, topology builders, booleans, fillets, `convert.implicit_to_solid`, STEP encoding |
| `both` | GPU: lightweight first, exact on demand | most Part Design / GSD features | `part.pad`, `part.edge_fillet` |

**Preview cheaply, rebuild exactly:**
1. A `both` feature first produces its lightweight result for every configuration, for example a pad as an implicit or tessellated prism.
2. GPU harness checks filter the configurations.
3. Exact rebuild runs, also on the GPU, only for configurations selected by `"over"` subsets, `study.rank` survivors or `document.save_version`.

## Topology classes

A design table's rows do not all share one structure. Changing a hole count, or a fillet that consumes a face, gives each row a different feature tree, face count and name set. Batching must respect that:
- Each configuration gets a **topological signature**: a hash of the active feature list, element counts per feature and the adjacency fingerprint. The proxy phase computes it cheaply.
- Configurations are **grouped by signature into topology classes**. Each class runs as its own wave with `Strided` layouts; different classes run concurrently as separate bins.
- The study report includes the **class count and class-size histogram**. A study that fragments into thousands of classes is reported as such, because the GPU advantage is gone. Models can then restructure the table, for example by pattern-instancing instead of varying the count.
- Stable names are resolved per class and mapped back to configurations.

## Proxy fidelity

The lightweight-then-exact strategy only works if lightweight rankings predict exact rankings. So fidelity is measured, not assumed:
- **Rank agreement:** Kendall τ between proxy and exact scores on survivors, plus a periodic random audit sample.
- **Survivor failure rate:** the share of top-k configurations whose exact rebuild fails (`TOPOLOGY_FAILED`) or flips a requirement verdict.
- **Fallback:** when a survivor fails, the next-ranked configuration is promoted automatically, and the report says so. This is part of `study.rank` semantics, not hidden repair.
- Fidelity statistics are recorded per feature and per topology class. Low-fidelity features get a tighter proxy (finer tessellation, smaller bricks) on the next study, and the policy is stated in the report.

## Lowering

```
AgIR program
  → feature expansion (stdlib/UDF bodies inlined or called)
  → ExecGraph: GPU kernel nodes | VM nodes | harness nodes   (no host nodes)
  → fusion of elementwise chains (implicit trees, wrangles, objectives)
  → graph capture (CUDA Graph / prerecorded wgpu command buffers), keyed by
    (program shape, feature versions, capacities, topology class)
  → on repeat runs upload only changed parameters (Demo template-cache result)
```

## Overlap on the GPU (no CPU pipeline)

```
stream hi:  [harness wavefront k] ─▶ [harness wavefront k+1] ─▶ …
stream mid: [proxy batch k+1] ───────▶ [proxy batch k+2] ─▶ …
stream lo:  [exact rebuild survivors k] ─▶ [incremental re-tessellation] ─▶ …
            (event dependencies only; green-context partitions where available)
```

- Exact work for survivors overlaps the next proxy batch on lower-priority streams of the same GPU.
- Exact results are re-tessellated incrementally: only dirty faces, with shared-edge discretization kept watertight.
- `PENDING_EXACT` means the exact result is scheduled in a later wavefront. Configurations keep it until that wavefront completes.

## Host-interaction budget

The host is a control plane only. It is measured, and regressions fail CI:

| Metric | Target |
|---|---|
| Geometry bytes read or written by the host per program | **0**. Only report/trace bytes and final file bytes are copied, and file bytes go to disk via GPUDirect Storage where available |
| Host launches per `run_program` | O(segments), independent of configuration and feature count |
| Host CPU-ms per program (excluding compilation and network) | < 5 % of GPU-ms |
| Compilation | Cached; first-run UDFs served by the device VM while compiling |

## Memory capacity planning

Device memory is finite: 8 GB on the Demo laptop, shared system RAM on the Intel iGPU. Everything is device-resident, so capacity is planned rather than discovered:
- **Estimate.** Before launch, the planner estimates bytes per configuration for each topology class and feature, using the calibrated element counts from the proxy phase: B-rep arrays, tessellation, bricks, BVH and evaluation workspaces.
- **Chunk.** If the total exceeds the budget (default 80 % of free device memory), the configuration axis is split into chunks that fit. Each chunk runs as its own segment, and ranking merges across chunks stably by configuration id. The report carries `CHUNKED` (`0x0005`) with the chunk count. Chunking is scheduling, not hidden repair: results are identical to an unchunked run.
- **Evict.** Representations are disposable ([02](02-geometry-model.md#lightweight-representations-gpu-derived-disposable)). Under pressure, the planner evicts by `rebuild_cost / bytes` (cheapest-to-recompute first). Exact B-rep data and versions are never evicted; they spill to pinned host memory only as opaque pages, which are never parsed on the host.
- **Fail explicitly.** If a single configuration cannot fit, the result is `CAPACITY_EXCEEDED` with the estimated bytes and the largest consumers.

## Memory

- **Typed slab arenas** per element kind (curves, surfaces, B-rep topology, meshes, attributes, implicit bricks, expression graphs). Capacities are bounded and overflow returns `CAPACITY_EXCEEDED`.
- **CoW pages:** branches and configurations share pages until written. On CUDA this is implemented with virtual memory management (`cuMemMap`): shared physical pages are mapped read-only into each branch's address range, and a write maps a fresh page. On wgpu it uses index indirection.
- **Transients** (scan buffers, bins, queues) come from stream-ordered memory pools (`cudaMallocAsync`) and never from per-call `malloc`.
- **Sparse volumes:** implicit-body bricks and voxel bodies use the **NanoVDB** layout (PNanoVDB on wgpu), so OpenVDB/Houdini round trips are lossless (review F5).
- **Residency:** documents, lightweight representations and compiled kernels stay on the GPU across calls. This persistent worker was validated in the Demo (`MODEL_SESSIONS.md`).
- **Readback:** small status/report packets per step, with geometry only on demand or export. Each rollout or program ends with one packed readback (Demo `TRANSFER_STRATEGIES.md`).

## Determinism

- Reductions use a fixed order. Ranking breaks ties stably by configuration id. RNGs are counter-based and keyed by `(seed, feature_id, configuration, element)`.
- **CUDA rules (review F8):**
  - No floating-point atomics on deterministic paths; use CCCL deterministic or fixed-tree reductions.
  - Configure cuBLAS for deterministic algorithms (no atomics modes, fixed workspace).
  - Pin library versions per harness release.
  - Record device, driver, CUDA, library and kernel hashes in every trace.
- Same device, same binary and same input give **bit-identical** results, verified by replay-and-compare as in the Demo (`HARNESS.md`).
- Across devices or backends, results agree within each op's declared tolerance ([09](09-validation.md)). Bit identity is not claimed.

## Throughput targets (M4 exit; measured, not assumed)

| Workload (consumer/workstation targets use the df64 policy, [08](08-hal-backends.md)) | Target |
|---|---|
| Implicit body eval, 4,096 configurations × 64³ samples | ≤ 50 ms on an RTX 4070-class GPU |
| Parametric part (≈20 features) proxy update, 4,096 configurations | ≤ 200 ms |
| GPU harness check + rank over 4,096 configurations | ≤ 5 ms |
| Exact GPU rebuild of top-64 survivors (≈20 features, booleans + fillets) | ≤ 1 s |

The targets are reported per backend. Intel iGPU and wgpu numbers are measured and published, not extrapolated.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-05-001 | Same device, binary and input MUST produce bit-identical results and traces. |
| AG-05-002 | Floating-point atomics MUST NOT be used on deterministic paths. |
| AG-05-003 | Study reports MUST include topology-class counts and proxy-fidelity metrics. |
| AG-05-004 | Chunked execution MUST produce results identical to an unchunked run and MUST be reported with `CHUNKED`. |
| AG-05-005 | Host geometry bytes per program MUST be 0 (CI gate). |
| AG-05-006 | ExecGraphs MUST contain no host nodes. |
