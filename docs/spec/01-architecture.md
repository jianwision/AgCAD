# 01 — Architecture

> Status: **Normative** · Spec v0.3.1 · Area: architecture

Target: a **CATIA-class agentic GPU CAD**. It covers:
- Part Design, Generative Shape Design, Knowledgeware, Assembly Design
- **Houdini-style procedural modeling** (networks, attributes, wrangles, for-each/solver), with both living in one specification tree

The first MVP is **headless**: a pure-Rust library plus the `agcad` CLI and a headless MCP server. Python bindings are optional and come after the MVP. There is no GUI. Rendering is offscreen only, for vision feedback and reports ([10](10-roadmap.md)).

## Layers

```
┌─────────────────────────────── Callers ─────────────────────────────────┐
│ Llm: AgLang / AgIR JSON (MCP, Rust API) CuModel: AgIR bytecode   Replay │
└───────────────┬─────────────────────────────┬────────────────────┬──────┘
                └──── ModelAdaptor.decide(Observation) -> Program ─┘
┌─────────────────────────────── Harness ─────────────────────────────────┐
│ Requirements · Design Checker · Budget/sandbox · Workspace/Version/      │
│ Branch transactions · Update reports · Trace · UDF publication gate      │
│ GPU-resident: per-configuration requirement checks (proxy and exact),   │
│ budget, commit/rollback, rank, observe, delivery checks                 │
│ host (control plane): requirements upload, publication registry, trace  │
│ persistence (bytes only)                                                │
└─────────────────────────────────────────────────────────────────────────┘
┌──────────────────── Compiler (agcad-rtc / agcad-ir) ────────────────────┐
│ AgLang tracer → AgIR (typed SSA, units, effects, batch axes) → verify   │
│ (sandbox) → autodiff → vectorize over configurations → fuse → codegen:  │
│ CubeCL kernels | device bytecode (GPU VM)                               │
└───────────────────────────────┬─────────────────────────────────────────┘
┌──────────────────────────── Feature table ──────────────────────────────┐
│ core: document · knowledge · udf · proc · convert · study · evaluate ·  │
│       query · file                                                      │
│ stdlib: datum · sketch · constraint · wireframe · surface · part ·      │
│         boolean · transform · direct · mesh · assembly (AgLang on core) │
│ model: UDFs published at runtime                                        │
└───────────────────────────────┬─────────────────────────────────────────┘
┌──────────────────────────── Scheduler ──────────────────────────────────┐
│ ExecGraph: GPU kernel nodes, VM nodes, harness nodes · graph capture   │
│ cache · topology classes · stream-level overlap · multi-session packing│
└───────────────────────────────┬─────────────────────────────────────────┘
┌───────────────────────────────┴ Runtime ────────────────────────────────┐
│ Document: specification tree + parameters + UDF defs + networks (CoW,  │
│ device-resident) · Element store (typed slab arenas) · Stable naming   │
└───────────────────────────────┬─────────────────────────────────────────┘
┌─────────────────────── GPU data plane (everything) ─────────────────────┐
│ UDF/wrangle kernels · device VM · curve/surface derivatives ·          │
│ tessellation · NanoVDB bricks · OptiX/BVH · agcad-solver (NLS, graphs, │
│ linear, root, opt, interval) · exact B-rep kernel: SSI marching,       │
│ topology builders, booleans, fillets, sewing/healing, validity ·       │
│ evaluations & constraints · GPU harness · STEP/3MF/USD encoders ·      │
│ raster/ray-traced renders                                              │
└───────────────────────────────┬─────────────────────────────────────────┘
┌───────────────┴──── Driver: CudaDevice (Tier P+N) | WgpuDevice | CpuDevice (test oracle) ┐
└──────────────────────────────────────────────────────────────────────────────────────────┘
Host = control plane only: network (MCP), AgLang/NVRTC compilation, graph launches, file handles
(GDS on Linux where available). The host never reads or writes geometry.
Test-only oracle: OCCT for validity, booleans, STEP reimport
```

## Responsibilities

| Layer | Owns | Must not |
|---|---|---|
| **Model** | Requirements → features/UDFs/networks; body-type choice; search; inventing UDFs; repair; when to finish | Rewrite requirements or requirement checks |
| **ModelAdaptor** | Observation/decision translation; model-side state | Pick "better" geometry for the model |
| **Harness** (GPU; host only for control plane) | Requirements; Design Checker; budget/sandbox; workspace/version transactions; publication gate; reports; trace | Optimize, repair, retry or restrict strategy silently |
| **Compiler** | Tracing, typing, verification, autodiff, vectorization, fusion, codegen | Change semantics |
| **Feature table** | Signatures, manifests, naming roles, evidence declarations, aliases | Hide a feature from either front door |
| **Scheduler** | Placement, overlap, caching | Change results |
| **Runtime** | Specification tree, elements, names, representation cache, memory | Decide validity |
| **Kernels / VM** | Numerics with declared precision and evidence | Return values without evidence |
| **HAL** | Devices, memory, launches, sync | Leak backend semantics |

## One flange, three ways

1. **Part Design:** `sketch.circle` → `part.pad` → `part.hole` → `transform.circular_pattern`. The result is exact, named and editable.
2. **Implicit / UDF:** `part.implicit_body` built from a smooth-union blend tree (disc ⊕ bosses ⊖ six holes, 0.5 mm blends), then `convert.implicit_to_solid(tol = 0.01 mm)` at delivery.
3. **Procedural:** a `proc.network`: `proc.scatter` bolt points on the PCD → `proc.copy_to_points` hole tools → `boolean.remove`, with a `proc.wrangle` setting per-hole diameter from an attribute.

All three can run under a `study.configurations` design table of 4,096 rows on the GPU. The GPU-resident harness checks the requirements of every row and commits the top 4. All three appear in the specification tree and are replayable from either front door.

## Data flow — LLM path (headless MCP / Rust API)

```
LLM ──run_program(AgLang | AgIR JSON)──▶ agcad-server (headless)
  harness: requirements/revision/budget → compiler: trace, verify, typecheck
  scheduler: GPU batched update over configurations (lightweight representations)
             → GPU harness checks → survivors → GPU exact rebuild/convert → GPU exact checks
  harness: save_version / keep workspace → UpdateReport (+ offscreen renders)
LLM ◀── report JSON + PNG views ── one round trip per program
```

## Data flow — GPU model path

```
host: upload requirements, document, compiled stdlib/UDF kernels once; launch rollout graph
on device, N steps, zero host round trips:
   CuModel reads Observation → writes AgIR bytecode into the command ring
   GPU harness: opcode allow-list, revision, budget, sandbox caps → execute
       (precompiled feature kernels, or VM-interpreted model expressions)
   GPU harness: per-configuration Design Checker rules → commit/rollback → rank
   study.observe packs the next Observation
exact-phase work (booleans, fillets, SSI, validity) runs in later wavefronts of the same graph on the GPU
single packed readback at rollout end (trace + summary); geometry stays resident
```

This is the pattern already proven in `docs/Demo/cuda/gpu_harness.cu`: device-side propose → check → commit/rollback for 32 configurations × 8 rounds with a single readback. AgCAD generalizes it from a fixed template to the full feature table.

## Project layout: driver / runtime split (CUDA-style)

AgCAD's crates are layered the way the CUDA toolkit is. Each layer has its own API contract and versioning, so callers choose explicit control (driver) or convenience (runtime), and the two interoperate.

| CUDA toolkit | AgCAD layer | Contract |
|---|---|---|
| `libcuda` — **driver API** (`cuInit`, explicit `CUcontext`, `cuModule*`, `cuLaunchKernel`) | **`agcad-driver`** | Explicit, handle-based, no hidden state. Devices, streams, events, memory/arenas, graphs, libraries/functions, launches. Implementations per backend. Stable `AgStatus`/`DeviceFault` error model. Owns the **bytecode ABI** (block header, record and status layouts) |
| `cudart` — **runtime API** (implicit init, primary context, `<<<>>>`) | **`agcad-runtime`** | Convenient and implicit: lazy init, a **default session per thread/device** (like the primary context). Documents, specification tree, feature dispatch, scheduler, naming, representation cache, requirements registry. Every runtime object exposes its driver handles (as `cudart` interoperates with the driver API) |
| NVRTC / nvJitLink / NVCC | **`agcad-rtc`** (+ `agcad-ir`, `agcad-vm`) | Runtime compiler: AgLang → AgIR → CubeCL / CUDA C++ / LTO-IR → link; device VM for interpreted expression graphs |
| cuSOLVER / cuDSS / cuBLAS (numerical libraries) | **`agcad-solver`** | Batched numerical solvers for all of AgCAD: nonlinear least squares, constraint-graph analysis, sparse/dense linear solves, root finding, optimizers. Depends only on `agcad-kernels` + driver, so it is usable without a document ([§ Solver library](#solver-library-agcad-solver)) |
| cuBLAS / CUB / cuDNN (domain libraries over runtime) | **`agcad-geo`, `agcad-std`, `agcad-proc`, `agcad-kernels`** | Libraries: exact geometry kernel (behind the `ExactKernel` trait), standard features, procedural networks, core GPU kernels. They use only the runtime + driver public APIs |
| Nsight / compute-sanitizer / cuobjdump (tools) | **`agcad-cli`, `agcad-trace`, `agcad-conformance`** | `agcad` CLI, trace inspect/replay/diff, conformance + benchmarks |
| Application frameworks over CUDA | **`agcad-harness`, `agcad-model`, `agcad-improve`, `agcad-server`** | GPU harness (host: control plane only), ModelAdaptor/Candle, self-improvement, headless MCP server |

**Layering rules:**
- Dependencies point downward only: `server/cli → harness/model/improve → std/proc → runtime → geo → solver → kernels → rtc/ir → driver`. `agcad-solver` never depends on documents or features; everything that solves calls it.
- The **driver never depends on the runtime**. Libraries never call backend crates directly. Only `agcad-driver-*` crates touch `cudarc`, `wgpu`, CubeCL runtimes or OptiX.
- **Versioning follows CUDA's compatibility rule:**
  - The driver ABI version (`AGB` bytecode, `AgStatus` values, record layouts) is backward compatible: a newer driver runs older runtimes and older bytecode.
  - The runtime declares the minimum driver version it needs (cf. `cudaDriverGetVersion` ≥ runtime version).
  - The feature catalog version is separate and is checked by `catalog_hash` in bytecode block headers.
- The **GPU harness kernels** live in `agcad-harness` but are built only on driver primitives, so device models can run rollouts through the driver API alone, with no runtime-level host logic in the loop.
- A **C ABI** (`agcad-capi`, `include/agcad.h`, cf. `cuda.h`) exposes the driver API and a small runtime subset for non-Rust callers. It is post-MVP.

```
AgCAD/
├─ Cargo.toml                    # workspace
├─ include/agcad.h               # C ABI (post-MVP)
├─ crates/
│  ├─ driver/                    # agcad-driver: AgDevice + extension traits, AgStatus, DeviceFault,
│  │                             #   arenas/handles, bytecode ABI structs (repr(C), versioned)
│  ├─ driver-cuda/               # agcad-driver-cuda: cudarc, NVRTC, nvJitLink, cuLibrary, graphs +
│  │                             #   conditionals, VMM/IPC, green contexts, cuBLAS/cuSOLVER/cuDSS/CUB handles
│  ├─ driver-wgpu/               # agcad-driver-wgpu: CubeCL wgpu runtime, df64, host-emulated conditionals
│  ├─ driver-cpu/                # agcad-driver-cpu: CubeCL CPU / reference interpreter (oracle)
│  ├─ optix-sys/                 # agcad-optix-sys: thin C++ shim (M2)
│  ├─ ir/                        # agcad-ir: AgIR types, JSON + bytecode codecs, verifier, catalog loader
│  ├─ rtc/                       # agcad-rtc: AgLang parser (syn), proc-macros, autodiff, vectorize, codegen, cache
│  ├─ vm/                        # agcad-vm: device bytecode VM + host reference interpreter
│  ├─ runtime/                   # agcad-runtime: sessions (default per thread/device), document/spec tree,
│  │                             #   feature dispatch, scheduler/ExecGraph, naming, representation cache
│  ├─ kernels/                   # agcad-kernels: Tier P CubeCL core kernels
│  ├─ kernels-cuda/              # Tier N CUDA C++ sources (.cu/.cuh) compiled to LTO-IR at build time
│  ├─ solver/                    # agcad-solver: batched LM/Gauss–Newton, constraint graphs (DOF, redundancy,
│  │                             #   decomposition), dense/sparse linear (Cholesky/QR, PCG, cuDSS), Newton root
│  │                             #   finding, optimizers (Adam, CMA-ES, Nelder–Mead), autodiff Jacobian plumbing
│  ├─ geo/                       # agcad-geo: ExactKernel trait + GPU-native B-rep: SSI marching, topology
│  │                             #   builders, booleans, fillets, sewing, validity, STEP/IGES/USD encoders;
│  │                             #   agcad-geo-occt (test-oracle impl of ExactKernel, never in production)
│  ├─ std/                       # agcad-std: standard features (AgLang) + manifests; op-catalog.json source
│  ├─ proc/                      # agcad-proc: procedural networks, attributes, groups, cook
│  ├─ harness/                   # agcad-harness: requirements, Design Checker, transactions, GPU harness kernels
│  ├─ model/                     # agcad-model: ModelAdaptor, Llm, CuModel, ReplayModel, Candle training (separate process)
│  ├─ improve/                   # agcad-improve: archive, evaluator, gates, promotion
│  ├─ server/                    # agcad-server: headless MCP + JSON-RPC
│  ├─ cli/                       # agcad-cli: `agcad` binary (run, trace, improve, conformance)
│  ├─ trace/                     # agcad-trace: trace format, replay, diff
│  ├─ conformance/               # agcad-conformance: per-op suites, backend parity, benchmarks
│  └─ capi/                      # agcad-capi (post-MVP)
├─ xtask/                        # catalog generation, spec checks (replaces docs/spec/tools/*.py)
└─ docs/{spec,Demo}
```

### Solver library (`agcad-solver`)

Every solve in AgCAD goes through one library, as cuSOLVER/cuDSS serve CUDA applications. Before this, solving was scattered: sketch DOF, assembly constraints, design optimization, surface fairing, SSI seeding, face fitting and Knowledgeware formulas. Centralizing it gives a single place for batching, precision policy, determinism and self-improvement (L4) of numerics.

| Module | Scope | Users |
|---|---|---|
| `nls` | Batched Levenberg–Marquardt / Gauss–Newton with trust region and robust losses; Jacobians from AgIR autodiff | `sketch.solve`, `assembly.update`, `constraint.residual`, face/curve fitting |
| `graph` | Constraint-graph analysis: DOF counting, structural rank, redundancy/conflict sets, decomposition into rigid clusters / solve plans (DR-planning), so large sketches and assemblies solve as small independent blocks | sketch, assembly, `knowledge.formula` cycles |
| `linear` | Dense batched Cholesky/QR/SVD (register/shared-memory kernels ≤64 DOF; cuSOLVER batched beyond); sparse PCG (matrix-free, the Demo's approach), cuDSS direct; nullspace/projection for hard constraints | `nls`, `surface.fair`, `surface.match_surface`, `study.sensitivity` adjoints |
| `root` | Batched Newton / interval Newton for point inversion, curve–surface and surface–surface seeds; certified variants return `bounded` evidence | `agcad-geo` SSI seeding, `query.closest_to`, `evaluate.measure_between` |
| `opt` | Adam, L-BFGS, CMA-ES, Nelder–Mead, simulated annealing, with per-configuration batching and budget hooks | `study.optimize`, CuModel training baselines |
| `interval` | Interval and affine arithmetic for certified bounds on implicit bodies and residuals | Evidence levels ([02](02-geometry-model.md#evidence-levels)) |

**Contract:**
- Inputs and outputs use the driver's `BatchLayout`.
- Results are deterministic under the [05](05-execution.md#determinism) rules.
- Every result reports convergence status, residual norms, iterations, rank/DOF diagnostics and evidence level.
- Failures map to `SOLVER_DIVERGED`, `UNDER_CONSTRAINED` and `OVER_CONSTRAINED`, with the conflict set attached.
- Each module has a Tier P (CubeCL) and an optional Tier N (cuSOLVER/cuDSS/cuBLAS) implementation, promoted by parity.

**Using both levels** (cf. runtime vs driver API):

```rust
// Runtime API: implicit default session on device 0, CATIA vocabulary.
let mut part = agcad::Part::new("flange")?;
let report = part.run_program(include_str!("flange.ag.rs"), &requirements)?;

// Driver API: explicit device, stream, arenas and graph for a device-model rollout.
let dev = agcad_driver_cuda::CudaDevice::open(0)?;
let s = dev.stream(Priority::High)?;
let doc = part.document().driver_view(&dev)?;      // runtime → driver interop
let exec = harness::build_rollout(&dev, &doc, &cumodel, RolloutSpec { segments: 4, ..Default::default() })?;
dev.launch_graph(&s, &exec)?;
let ev = dev.download_packed_async(&s, &exec.outputs(), &mut pinned)?;
```

## Normative requirements

| ID | Requirement |
|---|---|
| AG-01-001 | Dependencies MUST point downward only, in the layer order of the project layout; `agcad-solver` MUST NOT depend on documents or features. |
| AG-01-002 | Only `agcad-driver-*` crates MAY use `cudarc`, wgpu, CubeCL runtimes or OptiX directly. |
| AG-01-003 | Every runtime object MUST expose its underlying driver handles. |
| AG-01-004 | The driver ABI MUST remain backward compatible within a MAJOR spec version. |
| AG-01-005 | The host MUST NOT read or write geometry; it is a control plane only. |
