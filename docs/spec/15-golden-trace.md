# 15 — Golden Trace: one program through every layer

> Status: **Informative** · Spec v0.3.1 · Area: integration test reference

This follows the [04](04-agir.md#json-encoding-llm-batch-program) flange study (4,096 configurations → rank → exact top-4 → STEP) through every layer. It is the reference for M0/M1 integration tests: each numbered step is an assertion point. Rows are tagged **[GPU]** or **[host: control plane]**.

## 0. Session setup

| # | Layer | Where | What happens | Bytes crossing host↔GPU |
|---|---|---|---|---|
| 0.1 | `agcad-server` | host | MCP `get_requirements` → requirements YAML (rev 3, `process: machining_3ax`) | — |
| 0.2 | `agcad-runtime` | host | Default session on device 0 (cf. primary context); `CudaDevice::open`; caps: `fp64_rate = 1/64` → **df64 policy** ([08](08-hal-backends.md)) | — |
| 0.3 | `agcad-driver` | host→GPU | Upload requirements, empty Part, precompiled stdlib library (LTO-IR → cubin cache hit) | requirements + document header (KB) |

## 1. `run_program` decode and compile

| # | Layer | Where | What happens | Assertion |
|---|---|---|---|---|
| 1.1 | `agcad-ir` | host | Decode AgIR JSON; units → internal; `expected_revision = 41` matches; `requirements_rev = 3` matches | otherwise `STALE_REVISION` / `STALE_REQUIREMENTS` |
| 1.2 | `agcad-rtc` | host | `udf.define(bolted_flange.ag.rs)`: parse the Rust subset, typecheck, verify sandbox; `udf.compile`: lift over the configuration axis → CubeCL + Tier N CUDA C++ → NVRTC `-dlto` → nvJitLink | `COMPILE_FAILED` diagnostics carry spans; cache key recorded in the trace |
| 1.3 | `agcad-sched` (in runtime) | host | Lower to ExecGraph; capture the graph keyed by (program shape, versions, capacities); **no host nodes** | graph has 0 host nodes |
| 1.4 | `agcad-driver` | host→GPU | Write the **parameter block** (design-table grid: 16 × 16 × 16) | parameter block only (KB) |

## 2. Batched update (proxy phase) [GPU]

| # | Step | Kernels / libraries | Assertion |
|---|---|---|---|
| 2.1 | `study.configurations` | expand the grid into `Strided` SoA parameters (4,096) | count = 4,096 |
| 2.2 | `sketch.solve` | `solver.graph` (DOF = 0 after constraints) → batched LM, 4,096 tiny dense blocks in registers | all `ACCEPTED`, residual < tol |
| 2.3 | `part.pad` (lightweight) | analytic prism, plus implicit view for checks | — |
| 2.4 | `part.hole` + `transform.circular_pattern` (lightweight) | pattern instancing; **topology signature** per configuration | **topology classes = 16** (one per hole count n); class histogram in report |
| 2.5 | evaluations | per class: `measure_inertia` (tessellation), `thickness_analysis` (NanoVDB bricks → interval bound), `bounding_box` | evidence `bounded` where requirement R2 demands it |

## 3. GPU harness wavefront [GPU]

| # | Stage | Assertion |
|---|---|---|
| 3.1 | VALIDATE / BIN | 0 invalid records; bins by opcode |
| 3.2 | CHECK (R1 bbox, R2 thickness ≥ 3 mm bounded) | margins = (value − bound)/tol; failures → `REQUIREMENT_FAILED` per configuration |
| 3.3 | DECIDE / METER / RECORD | budget decremented; status records written |
| 3.4 | RANK | `study.rank` k = 4 by mass among passing configurations; stable tie-break by configuration id |
| 3.5 | ESCALATE | top-4 → exact wavefront (low-priority stream) |

## 4. Exact rebuild of survivors [GPU]

| # | Step | Assertion |
|---|---|---|
| 4.1 | `part.pad` exact: planar + cylindrical faces (analytic) | topology builders: Euler characteristic and closure OK |
| 4.2 | holes: `boolean.remove` cylinders | SSI analytic plane–cylinder (closed form), builders, classification via ray parity; robust predicates take the exact path 0 times (non-degenerate case) |
| 4.3 | `evaluate.check` (exact) | valid solid; tolerance growth 0 |
| 4.4 | Proxy fidelity | exact mass vs proxy mass rank: τ = 1.0 on the 4 survivors; survivor failure rate 0 |
| 4.5 | Fallback (test variant) | inject a failure on #2 → #5 is promoted, and the report states it |

## 5. Report, renders, checkpoint

| # | Where | What | Bytes host←GPU |
|---|---|---|---|
| 5.1 | GPU | `study.render` iso + top views with **name-annotated face ids**; PNG encode on GPU | 4 × 2 PNGs |
| 5.2 | GPU | UpdateReport packed: top-4 details, distributions, class histogram, margins, budget | report (KB) |
| 5.3 | GPU→host | **one** `download_packed_async` into pinned staging; one event wait | report + PNGs only; **geometry bytes = 0** |
| 5.4 | host | `document.checkpoint` → MCP response to the LLM | — |

## 6. Save and export (next call)

| # | Where | What | Assertion |
|---|---|---|---|
| 6.1 | GPU | `document.save_version` on the chosen configuration: save-phase checks (R1–R3) | version created |
| 6.2 | GPU | `file.export_step`: entity numbering by scan, number formatting, concatenation into one buffer | AP242 syntax valid |
| 6.3 | GPU→disk | GPUDirect Storage write, or one copy of the file bytes to an fd | host never parses or creates geometry |
| 6.4 | test only | OCCT reimport: 1 solid, face count, 324 off-grid probes ≤ 1e-9 mm | conformance |

## 7. Replay

- `ReplayModel` re-runs the trace on the same device. The full trace and output buffers must be **bit-identical**.
- Cross-backend (wgpu) results must agree within each op's declared tolerance, with identical verdicts.

## Host-interaction tally (asserted in CI)

| Metric | Value |
|---|---|
| Host launches | 2 graph launches (study, save/export) + parameter writes |
| Host geometry bytes | **0** |
| Host-ms (excluding NVRTC compile on cache miss and network) | < 5 % of GPU-ms |
