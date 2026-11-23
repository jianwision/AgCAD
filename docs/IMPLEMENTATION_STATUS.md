# Implementation Status

Spec v0.3.1 → code, as of the implementation passes P1–P12 (build & test deferred by request).

**Build verified** (2026-11-20) on Windows 11 / MSVC 14.44 / CUDA 12.6 / RTX 2000 Ada and on
Linux: `cargo build --workspace --all-targets` with 0 warnings; `cargo test --workspace --
--include-ignored` 102 passed / 0 failed. GPU coverage:
- `crates/conformance/tests/parity_cuda.rs`: CUDA vs CPU bit parity for scan, compaction,
  radix sort, top-k, segmented reduce, expression VM, batched LM, MLP inference, harness
  validate/bin, and batched mesh mass properties (64 configurations in one pass).
- `crates/driver-cuda/tests/graphs_vmm_cuda.rs`: graph capture/replay/update, IF and SWITCH
  conditional nodes, a device-driven WHILE loop (`cudaGraphSetConditional` from NVRTC code),
  VMM reserve/map/use.
- `crates/server/tests/golden_trace.rs`: spec 15 flange study over MCP JSON-RPC (64
  configurations → 4 topology classes → rank matching an exact-volume oracle → exact survivors,
  τ = 1 → render → save → STEP export/reimport → delivery), replayed bit-identically. Steps
  needing the GPU feature data plane are printed as pending.

The CAD fuzzer runs **10 000** layered programs with 0 failures on both machines. Every program also passes these metamorphic relations:
- inclusion–exclusion at each boolean;
- translation invariance;
- commutation of subtractive ops;
- STEP write → read.

The generator deliberately places features in near-degenerate ways: concentric, tangent, centred on walls, on corners, and with fillets and chamfers. Tangential (zero-thickness) contacts are rejected explicitly with `TOPOLOGY_FAILED` and the contact point. `agcad conformance --capture <file>` turns shrunk failures into regression tests.

**AI ergonomics:**
- the MCP tools `check_program` (all diagnostics with fixes) and `describe` (holes, walls and planes as text, plus a semantic diff against a version);
- `$param` and `=formula` arguments;
- in-place parameter edits;
- requirement satisfiability warnings and per-cause infeasibility attribution in study reports;
- one seamed face per full cylinder (CATIA convention). `cargo xtask check` matches
`check_spec.py`. `cargo clippy` reports ~80 style lints (no errors).

## GPU-model-driven device operations (`crates/dops`)

This is the loop **LLM observes → sets a goal → GPU model plans and executes CAD operations under the GPU harness → survivors return as exact documents → LLM observes**:

- **MCP tools:** `set_goal` and `device_rollout`, then the existing `describe`, `render`, `export`.
- **Device design:** a *feature tape* per configuration, mirroring the Part Design tree. Values are literals or parameter + offset.
  - **Profile ops:** pad, pocket and hole over rectangle, circle and regular-polygon sketches.
  - **Patterns:** circular, and rectangular (x / y, ≤ 64 instances).
  - **Corners:** vertical-edge fillet and chamfer of rectangle profiles.
  - **Shell (closed):** exact through a 2D inward offset (`region2d::offset_inward`, `LayeredSolid::shell_closed`; std `part.shell` with `remove: []`) when the body is one convex pad minus cuts. In that case the device field is an exact interior distance. Any other body is delivered as a **certified implicit** body: a watertight mesh within h·√3 of the surface with certified volume bounds, exportable as 3MF/STL, with STEP refused.
  - **Multi-body:** `document.body` plus `boolean.add/remove/intersect`. Lowered with `$body` containers.
  - **General sketches:** lines and arcs, one outer loop plus holes, from a per-design profile pool. The device uses an exact transcendental-free SDF: crossing parity combined with circular-segment parity. `raise` reads LLM sketches built from `sketch.line` / `arc` / `profile` / `elongated_hole` (slots) and mixed primitives. They lower exactly to layered extrusions and back to std sketches.
  - **Revolve** (`part.shaft` / `part.groove` about the vertical axis): exact as a stack of annuli when the meridian is axis-aligned (`LayeredSolid::revolve_rectilinear`, also used by std for composable shafts). Sloped or curved meridians are device-only.
  - **Mirror** (`transform.mirror`, planes x = 0 / y = 0): exact layered reflection plus union. It follows document order: the body built so far is reflected, and later features are not mirrored (one mirror stage per design). This also fixes std, which dropped the original body for layered mirrors.
  - **Revolve about an off-origin vertical axis:** via `datum.plane {mode: "through"}` (new std mode).
  - **Exact closed shell:** cross-checked against OCCT's own inward offset on 30 fuzzed holed plates, worst relative difference 1.5e-15 (`agcad-geo-occt::offset_volume`).
  - **Draft** and **rounded / chamfered top and bottom edges:** device-only (Lipschitz-1 fields), delivered as certified implicit bodies. The document keeps the exact base features.
  - **Fast path:** tapes with general features run in their own CUDA launch, so plain designs keep full speed.
  - **Kernel fix:** a smooth (G1) junction of the same wall, continuing across a layer boundary, is no longer misreported as a tangential contact. Carrier-checked: genuinely tangent operands are still rejected.
  - **Also:** activation and parameters.
  
  Each op has a point SDF, an interval version, an exact lowering and a raise. Each is tested for soundness, round trip and CUDA ≡ CPU bitwise. Each device op is a catalog op; opcodes are resolved from the catalog, never hard-coded.
- **Round:** policy (**mock GPU model**, a deterministic goal-seeking controller writing command records and arg-arena payloads into the device ring) → validate (harness) → apply (trial tapes) → compile (resolve parameters, precompute trig tables) → eval → decide (commit or reject, meter, observe, score) → continue (`cudaGraphSetConditional`). The whole rollout is **one graph launch with a device-driven WHILE**, and the host reads 0 bytes until it ends.
- **Evaluation:** Lipschitz cell classification of exact distance bounds, evaluated in f32 with an explicit error margin (spec 08 precision policy). It produces:
  - **rigorous volume and extent intervals** (`bounded` evidence), used for hard rejections;
  - a gradient-based **volume estimate** (`sampled` evidence), used for steering.
  
  All accumulation uses integer atomics, so CUDA and the CPU reference are bit-identical.
- **Escalation:** survivors lower to exact layered B-reps (soundness oracle: exact volume ∈ device bounds) and to AgIR programs of real features on a new branch. `raise` maps a document back to a device design.
- **Runtime:**
  - `memset_async` on every device;
  - `Studio::reset`, plus `GraphRollout`, which caches the instantiated graph (the server reuses it across `device_rollout` calls);
  - certification on a low-priority stream, overlapped with host exact escalation;
  - multi-device sharding (`studio::rollout_sharded`) with global configuration ids in the policy, giving results identical to one device.
- **Evaluation pipeline (G1/G2):**
  - **Adaptive interval evaluation:** the f32 interval tape is evaluated over each cell's box, which classifies it as inside, outside or ambiguous.
  - **Worklists:** per level, ambiguous cells are refined through deterministic scan-compacted worklists (`ScanPlan`: allocation-free, so it is legal inside the graph's WHILE body). Each configuration refines its first cap/8 ambiguous cells in worklist order.
  - **Counters:** all counts are integer atomics, so CUDA and the CPU reference stay bit-identical.
- **Evidence chain (spec 14):**
  - **Proxy rounds:** these use small caps. The policy steers with the sampled estimate, and the bounds reject definite violations.
  - **Survivors:** `studio::certify` re-evaluates them alone with a deep pass plus **per-cell bound tightening**: interval refinement of each terminal boundary cell (largest ambiguous sub-box split along the axis of largest |g|·width) with fixed-point certified fractions. Flange: ±23.9 % (quick) → ±3.5 % (deep) → **±0.3 %** (tightened). Plate ±0.19 %, union ±0.12 %, shell ±0.24 %. Soundness is checked against exact volumes.
- **Engineering evaluators** (sampled, CUDA ≡ CPU):
  - integer mass moments → centroid and inertia;
  - surface area;
  - wall thickness and clearance (sphere-traced probes along ∓normal);
  - 3-axis tool access;
  - centroid limits.

  `set_goal` takes `min_wall`, `min_clearance`, `machining: "3ax"` (defaulting from the requirements' process), `centroid`, and `load` / `max_compliance`. Edits may not worsen violations.
- **On-device optimizer policy** (`policy: "cmaes"`, the server default when parameters are free):
  - **Islands:** λ configurations per island run sep-CMA-ES on the normalized parameters. Sampling uses Irwin–Hall normals (no transcendentals). Members are elitist, the rank-μ diagonal update is linearly weighted, and step size follows the 1/5-success rule with restarts. The update kernel runs one thread per island.
  - **Parity:** CUDA ≡ CPU bitwise, graph ≡ host loop, and sharding keeps islands whole.
  - **Objectives:** `satisfy`, `min_volume` and `max_volume`, subject to every constraint.
  - **Measured** at 4096 configurations with every LLM sign prior wrong: mock meets the goal in 181 / 4096, CMA-ES in 4095 / 4096 (24 rounds, 16–17k config-rounds/s).
  - **min_volume** with an 8 mm wall picks the lightest feasible disc. The mock policy remains for structural moves (new holes / pockets).
- **Structural search:**
  - **Mock policy:** when a parameter saturates it adds exact-path features: a hole, a rounded pocket, a slot, or a 4-hole bolt circle to remove material, or a boss on the base pad to add it.
  - **CMA-ES islands:** an island that stays infeasible for 3 generations makes the same structural move on every member (≤ 6 per island). The move's radius and x / y are bound to spare parameter slots `s{k}_r / _x / _y` (`studio::structural_slots`), so CMA-ES tunes the added structure and the lowered documents stay parametric.
  - **Measured:** on a goal unreachable by parameters alone, plain CMA-ES meets it in 0 / 64 configurations and structural CMA-ES in 64 / 64. Survivors are exact-sound and round-trip through std documents.
- **Ray queries (`agcad-raytrace`):**
  - **OptiX:** RTX hardware ray tracing through the `agcad-optix-sys` shim (feature `optix`; SDK from `OPTIX_ROOT` or the default install path). Device programs are compiled to PTX with NVRTC.
  - **CPU reference:** an exact f64 Möller–Trumbore BVH tracer for machines without OptiX and for parity checks.
  - **Parity:** OptiX ≡ CPU on hit / miss and distance (1e-4) for 20k random rays.
  - **Speed:** 4 M rays in about 130–150 ms on the RTX 2000 Ada, including transfers.
  - **Analyses** (MCP tool `dfm`): tool access for 3-axis (fixture face excluded), 3+2 and 5-axis direction sets; wall thickness and clearance along normals.
  - **Survivor escalation:** with `set_goal` `machining` (3ax / 3+2 / 5ax), `min_wall` or `min_clearance`, `device_rollout` ray-checks the survivors (top-k, or up to 16 Pareto candidates).
    - Failures get `goal_met: false` and are excluded from `apply_best` and from the Pareto set.
    - `ray_undercut` is available as a Pareto objective; `ray_check: false` turns the check off.
    - Catches what the device's +z probe cannot: sealed cavities, and 3+2 / 5-axis reach.
  - **Renders:** ray-traced, shadowed renders (`render {mode: "raytraced"}`, Pareto candidates).
- **Multigrid stiffness solver** (`fea`, `Precond`):
  - **Method:** geometric multigrid V-cycle as the PCG preconditioner, up to 6 levels. Coarse levels re-discretize on doubled voxels, weighted by solid fraction; trilinear prolongation with transpose restriction; ℓ1-Jacobi smoothing, which is convergent for any geometry, 2 pre / 2 post sweeps at ω 1.3; 32 coarse sweeps.
  - **Determinism:** gather-only kernels in a fixed order, so CUDA ≡ CPU bitwise.
  - **Iterations are resolution-independent:**
    - cantilever: 60 → 64 iterations from 8.8k to 265k DOFs (Jacobi: 292 → 890);
    - flange: 20 → 20 (Jacobi: 135 → 371).
  - **RTX 2000 Ada wall-clock:** 2.3× faster at 265k DOFs and 2.6× at 688k. Below about 100k DOFs a GPU V-cycle is launch-bound, so `Auto` (the default) uses Jacobi there.
  - **Reporting:** the server's stiffness report names the solver.
- **Stiffness proxy** (`fea`): voxel hex8 elasticity solved by matrix-free Jacobi-PCG on the device. Fixed-shape reductions make it bit-identical CUDA ≡ CPU, and it matches `agcad_std::physics` to 1e-6. It runs on survivors.
- **Measured** on an RTX 2000 Ada laptop GPU (`agcad bench-device`; eval is ≥ 99.9 % of each round):

  | Design | Eval per round (4096 configurations) | Rigorous volume interval | 4096 × 7-round rollout |
  |---|---|---|---|
  | f64 point / Lipschitz, 96³ | — | — | 710 s |
  | f32 point / Lipschitz, 96³ (G0 baseline) | 1361 ms | ±40 % | 11.9 s |
  | f32 adaptive interval, 128³ (all-or-nothing cap) | 166 ms | — | — |
  | f32 adaptive interval, 256³ (cap 4096, partial refinement) | 360 ms | ±30 % | 2.8 s |
  | + new ops and engineering accumulators (post-N3) | 585 ms | | 5.9k config-rounds/s |
  | + warp-aggregated integer accumulation | 399 ms | | |
  | + per-cell tape pruning (exact: bit-identical point values) | 385 ms | | |
  | + `__launch_bounds__(256, 3)` (143 regs → 16.7 % occupancy before) | 274 ms | | 10.7k |
  | + compacted terminal-cell fill kernel (resolve → clip → compact → fill) | 227 ms | | **12.5–13.9k config-rounds/s** |

  **Global cell budget** (`max_cells`): memory follows the budget instead of configs × cap. 4096 configurations at cap 65 536 / 512³ finest run in 2.7 s under a 32 M-cell budget; before, this exceeded VRAM at 12 s per round.

  Ruled out by measurement: near-cubic root cells. They made plate bounds 3× wider, because the thin-z cells bound the dominant faces best. The global worklist now splits along the axes that affect the distance interval. On the pocketed thin plate at a 4,096-cell cap, this narrowed the relative volume half-width from ±42.37% to ±19.89%. Use `bench-device --no-aniso` or `device_rollout` with `"aniso": false` to select uniform splitting.
- **Next:**
  - a learned structural policy over feature-tape tokens, one token per node (CANTO-style native tokenization), trained from rollout traces; CMA-ES then tunes its parameters;
  - shared-memory tapes;
  - open shells and multi-slab offsets (the general offset path);
  - an OCCT parity fuzz for `offset_inward`;
  - a Pareto front of survivors with renders back to the LLM.

## Op coverage (293 catalog ops)

| Family | Feature impl | Harness/runtime | `UNSUPPORTED_OP` |
|---|---|---|---|
| `document` | 0 | 14 | 10 |
| `knowledge` | 1 | 0 | 9 |
| `udf` | 0 | 4 | 2 |
| `proc` | 9 | 0 | 9 |
| `convert` | 1 | 0 | 8 |
| `study` | 1 | 11 | 1 |
| `evaluate` | 12 | 0 | 10 |
| `query` | 5 | 0 | 7 |
| `file` | 5 | 2 | 6 |
| `datum` | 3 | 0 | 1 |
| `sketch` | 11 | 0 | 10 |
| `constraint` | 12 | 0 | 11 |
| `wireframe` | 0 | 0 | 22 |
| `surface` | 0 | 0 | 25 |
| `part` | 10 | 0 | 19 |
| `boolean` | 4 | 0 | 3 |
| `transform` | 7 | 0 | 5 |
| `direct` | 0 | 0 | 4 |
| `mesh` | 1 | 0 | 9 |
| `assembly` | 7 | 0 | 2 |
| **total (293)** | **89** | **31** | **173** |

- *Feature impl*: a registered `FeatureImpl` (agcad-std / agcad-proc).
- *Harness/runtime*: handled by the harness session, study engine or runtime document/file layer.
- *Unsupported*: returns `UNSUPPORTED_OP` with its manifest (AG-07-002).

## What is implemented, by pass

| Pass | Crates | Highlights |
|---|---|---|
| P1 | driver, ir, xtask | AgStatus (generated from spec 06), bytecode ABI, device traits, arenas/handles, BatchLayout, caps/precision policy; catalog loader, units, types, AgIR JSON ⇄ bytecode, verifier; spec tooling in Rust |
| P2 | driver-cpu, driver-cuda, kernels, kernels-cuda | CPU reference device; CUDA device on cudarc raw API (priority streams, stream-ordered buffers, pinned, events, NVRTC + disk cache, graphs + IF/WHILE/SWITCH, VMM + FD export, sticky-fault poisoning); deterministic scan/compaction/radix sort/segmented reduce/stable top-k (CUDA + bit-identical CPU refs) |
| P3 | rtc, rtc-macros, vm | AgLang strict-Rust-subset front end (syn), sandbox verifier, unit-aware expression graphs, forward/reverse autodiff, CUDA codegen vectorized over configurations, rustc-style diagnostics; expression VM (CUDA + CPU), proc-macros |
| P4 | solver | Dense Cholesky/QR/SVD/nullspace, batched LM (CUDA + CPU), constraint-graph analysis (Hopcroft–Karp, DM, Tarjan plans), PCG, affine spaces, Newton/interval Newton, Adam/NM/CMA-ES/SA/augmented Lagrangian, interval & affine arithmetic, df64 |
| P5 | geo | NURBS (Piegl–Tiller), analytic curves/surfaces with jets, robust predicates (Shewchuk), half-edge B-rep + validity, builders (prism/revolve/primitives), exact 2D region booleans + layered extrusion solids, analytic intersection table, watertight tessellation, STEP AP242/AP214 write/read with validation properties, 3MF, deterministic zip, ExactKernel |
| P6 | runtime, trace | CATIA document model (CoW versions/branches/workspaces/undo), stable naming + signature recovery, rebuild engine + registry, ExecGraph lowering, topology classes, capacity planner, representation cache, .agcad save/open/migrations/snapshots, sessions, multi-GPU sharding |
| P7 | std | Sketcher + constraints (LM), Part Design (pad/pocket/shaft/groove/hole/primitive/implicit/lattice/edge_fillet/chamfer), booleans (+ smooth implicit), patterns/transforms, meshing (marching tetrahedra), evaluations incl. DFM, queries, knowledge/datum, STEP/STL/3MF/USD/CATIA-VBA export, STEP import, physics (voxel FEA, modal estimate, SIMP), assembly mates + solve |
| P8 | harness | Requirements + budget, Design Checker with evidence enforcement, transactions/session, GPU wavefront stages (CUDA + CPU), reports with untrusted quoting, review log |
| P9 | harness | Study engine (configurations, batched update, topology classes, rank + fallback, proxy fidelity, sensitivity, optimize, tolerance analysis, observe), offscreen rasterizer + PNG |
| P10 | server, cli, model | MCP JSON-RPC over stdio (9 tools), `agcad` CLI, ModelAdaptor/Llm/ReplayModel/CuModel (safetensors, MLP kernels), Candle trainer binary, geometry descriptor |
| P11 | geo, proc, std | General SSI (seeding + marching + verification), corner fillets/chamfers on layered bodies, Houdini-style procedural networks, physics, assembly, exports |
| P12 | conformance, improve, geo-occt, capi, optix-sys, driver-wgpu | Requirement-keyed suites, CAD fuzzer with shrinking, OCCT oracle (subprocess), gates/archive/promotion/rollback/held-out accounting, C ABI + header, OptiX shim bindings (feature-gated), wgpu placeholder |

## Known gaps (next passes)

- **GPU data plane for features:** feature kernels (tessellation, SSI, topology builders, STEP encode/decode, rasterization) exist as CPU references; their CUDA versions are the main remaining work for the GPU-only data plane (AG-05-005 is not yet met end-to-end).
- **General exact kernel:** booleans/fillets beyond same-direction layered solids return `TOPOLOGY_FAILED` (SSI marching exists; topology builders for general faces do not yet).
- **Surfaces/wireframe GSD families, direct edit, mesh editing, most converters:** `UNSUPPORTED_OP`.
- **OptiX C++ shim source**, NanoVDB layout, wgpu/CubeCL Tier P, nvJitLink/LTO-IR linking, cuBLAS/cuSOLVER/cuDSS handles.
- **Autodiff through features** (sensitivity uses finite differences, evidence `sampled`); learned geometry encoder.
- **SolidWorks receiver profile** remains `experimental` (no B-spline conversion of procedural surfaces; no SolidWorks CI runner).

## Verification commands

1. `cargo build --workspace --all-targets` (Windows: run inside `vcvars64.bat`).
2. `cargo test --workspace` (CPU paths); add `-- --include-ignored` on an NVIDIA machine.
3. `cargo xtask check` (needs `docs/spec`).
4. `agcad conformance --fuzz 3000` (OCCT oracle used if `python3 -c "import OCP"` works).
5. Next: move the feature data plane to the GPU so the golden trace's pending steps (1.2, 1.3, 2.x/4.x, 5.1, 5.3, 6.3) become assertions.
