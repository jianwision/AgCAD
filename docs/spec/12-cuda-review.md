# 12 — Review: NVIDIA Frameworks & CUDA

> Status: **Informative** · Spec v0.3.1 · Area: review

> **Superseded in part by v0.3:** exact-topology work moved from CPU to GPU ([02](02-geometry-model.md#exact-geometry-solidsurfacewire-bodies), [05](05-execution.md)). Where this review mentions CPU exact work or host queues, read "GPU exact wavefronts".

Review of spec v0.2 (docs 00–11) from the CUDA and NVIDIA-ecosystem side, dated 2026-11-20.

The section headed "Amendments applied" lists the changes already folded into the spec. Everything else is a recommendation, and its M0 spike outcome decides it.

Severity:
- **S1:** the architecture is wrong or will badly underperform as written.
- **S2:** a significant gap.
- **S3:** an improvement.

## Summary

The overall shape fits CUDA well:
- resident documents
- batched configurations as the leading axis
- captured graphs
- one packed readback
- GPU-seed / CPU-exact topology

The Demo proved all of these on an RTX 2000 Ada.

Four places in the spec would work against NVIDIA hardware:
1. **FP64 policy.** It assumes datacenter FP64 rates.
2. **The device VM as the default execution model for heterogeneous commands.** It causes warp divergence.
3. **Hand-rolled BVH and brick formats.** They ignore RT cores, OptiX and NanoVDB.
4. **Fault isolation.** One bad model-written kernel can kill the CUDA context for every session.

## Findings

### F1 (S1) — FP64 throughput on GeForce/RTX workstation parts

**Problem:**
- [08](08-hal-backends.md) says "f64 on CUDA" for exact-adjacent work, solvers and harness checks.
- On Ada/Blackwell consumer and workstation GPUs, FP64 runs at **1/64 of FP32**. That includes the Demo's RTX 2000 Ada and the "RTX 4070-class" throughput target.
- Only datacenter parts (A100/H100 class) have high native FP64, and NVIDIA's newest datacenter parts lean on emulation instead.

**Recommendation:** make precision a **per-device capability decision**, not a per-backend one:
- `Caps` reports `fp64_rate`, `has_tensor_fp64` and `fp64_emulation` (cuBLAS).
- Use **df64 (float-float) on CUDA as well** when `fp64_rate < 1/8`. It is usually faster than native f64 on consumer parts, and it keeps CUDA and wgpu on the same numeric path, which simplifies parity.
- Dense linear algebra (least-squares face fitting, normal equations, Gauss–Newton steps) goes through **cuBLAS GEMM with FP64 emulation**. That means Ozaki schemes, available since CUDA 13.0 Update 2, with Ozaki-II added in 13.4.2. Automatic dynamic precision falls back to native FP64 when the inputs need it.
- Pattern: f32/df64 bulk work, then f64 **refinement of residuals only**. Iterative refinement keeps the exact-adjacent evidence honest.

### F2 (S1) — The device VM must not be the default executor of heterogeneous commands

**Problem:**
- [06](06-harness.md) and [07](07-model-interfaces.md) have every configuration's command record validated and executed per thread block. With 4,096 configurations issuing different opcodes, that becomes an interpreter megakernel.
- On SIMT hardware this diverges badly and blocks per-op tuning: shared-memory layout, tensor cores, library calls.

**Recommendation:** run each round as a **wavefront**, like wavefront path tracing:
1. **Validate** all records. This is uniform work, one thread per record.
2. **Bin by opcode** with a CUB segmented radix sort or histogram + scan, producing per-opcode work queues.
3. **Launch one specialized kernel per non-empty opcode bin.** In a captured graph this is a **SWITCH/IF conditional node per bin** (CUDA 12.8+), so empty bins cost almost nothing.
4. **Check → decide → meter → record → rank** as uniform kernels.
5. Wrap the round in a **WHILE conditional node** that loops until every configuration is stopped or out of budget. This replaces the Demo's fixed 8-round unroll.

Keep the VM only for **small model-invented expression graphs** (implicit functions, wrangles), and evaluate them per sample, where the graph is uniform across the batch. Promote hot graphs to NVRTC-compiled kernels asynchronously (see F7).

**Do not rely on device-side graph launch** with conditional nodes. Its support is unconfirmed, so keep host-launched graphs with device-set conditionals.

### F3 (S1) — Fault containment for model-written and model-optimized kernels

**Problem:**
- An out-of-bounds access in a CUDA kernel is a **sticky error**. It invalidates the whole context, so every resident document and every session sharing that context is lost.
- L4 self-improvement ([11](11-self-improvement.md)) explicitly runs candidate kernels.

**Recommendation:**
- **Codegen guarantees.** AgLang kernels use checked indexing, no raw pointers, and capacity-checked writes. The verifier is the first line of defence.
- **Process isolation.** Run one worker process per session or tenant. The Demo already used a persistent Rust worker, so make this the rule. Run untrusted or candidate kernels (L4, first-run UDFs) in a **separate sandbox worker with its own context**, and promote them only after conformance under **compute-sanitizer** (memcheck, racecheck, initcheck).
- Use MPS for many small workers on one GPU. Use MIG or green contexts for hard partitions where available ([F10](#f10-s3--multi-tenancy-and-co-scheduling)).

### F4 (S2) — Use RT cores / OptiX for ray-type queries on NVIDIA

**Problem:** the spec hand-rolls an LBVH for raycast, picking, clash, thickness, inside/outside tests and rendering. On RTX GPUs, RT cores do BVH traversal and triangle tests in hardware.

**Recommendation:**
- On `CudaBackend`, ray queries use **OptiX**. Inside/outside uses ray parity plus winding-number fallback. Thickness and clearance use ray casts, and offscreen renders are ray-traced and need no display, which fits the headless MVP.
- Point/AABB/closest-point queries, which RT cores don't accelerate, use **cuBQL**, NVIDIA's header-only CUDA BVH library, compiled through NVRTC. NVIDIA Warp has adopted it as an experimental backend.
- `WgpuBackend` keeps the CubeCL LBVH, with Vulkan ray queries where exposed.
- OptiX has no maintained Rust binding, so plan a thin C++ shim (`agcad-optix-sys`) and budget for it in M2.

### F5 (S2) — Adopt NanoVDB for sparse implicit/voxel storage

**Problem:** [02](02-geometry-model.md) invents a hashed 8³ brick format.

**Recommendation:**
- Use **NanoVDB** (OpenVDB's GPU-friendly, pointer-free tree) as the implicit-brick and voxel-body layout.
- It already has GPU builders, sampling and ray marching (HDDA). It is portable to shaders through PNanoVDB, which serves wgpu.
- It round-trips with OpenVDB, so `file.export_vdb` becomes lossless.
- **It directly serves the Houdini goal**, since Houdini's volume workflow is VDB-native.
- Treat **fVDB** (sparse convolution and ops on NanoVDB) as a reference for learned implicit operators in L3.

### F6 (S2) — HAL trait is too thin for CUDA-native features and NVIDIA libraries

**Problem:**
- The `Backend` trait ([08](08-hal-backends.md)) covers only alloc, upload, launch and capture.
- CubeCL is the right portable layer, but it won't give access to cuBLAS/cuBLASLt, cuSOLVER, cuDSS, CUB, OptiX, conditional graph nodes, stream-ordered memory pools, VMM or green contexts.

**Recommendation:** use a **two-tier kernel strategy**:
- **Tier P (portable):** CubeCL kernels for all backends. This is the reference behaviour.
- **Tier N (NVIDIA-native):** NVRTC-compiled CUDA C++ (the Demo-proven path) plus NVIDIA libraries, behind the same op manifests. A Tier N kernel is promoted per op only by parity with Tier P (the [09](09-validation.md) backend-parity rule).
- **Extend the trait** with:
  - `mem_pool()`: stream-ordered `cudaMallocAsync` pools for transients
  - `vmm_reserve/map()`: growable arenas and page-level CoW
  - `conditional_handle()`
  - `event()`
  - `stream_priority()`
  - `library::<Blas|Solver|Dss|Optix|Cub>()` handles
  - `caps()` with FP64 rate, RT cores, tensor-core types, conditional-node support and green-context support

| Need in spec | NVIDIA component |
|---|---|
| Batched small dense solves (sketch LM, assembly, face fit) | cuBLAS/cuSOLVER batched (`potrf`/`getrf` batched), or register/shared-memory Cholesky kernels for ≤64 DOF |
| Sparse systems (fairing networks, large constraint graphs) | **cuDSS** (sparse direct), cuSPARSE + PCG (the Demo's matrix-free approach), AmgX for large Laplacian-type fairing |
| Scan, sort, compaction, segmented reduce, top-k | **CUB / CCCL** (deterministic variants; see F8) |
| Dense LA in exact-adjacent work | cuBLAS with FP64 emulation (F1) |
| CuModel inference matmuls inside graphs | **cuBLASLt** (capture-safe with preallocated workspace) or CUTLASS |
| Rays, picking, renders | OptiX (F4) |
| Sparse volumes | NanoVDB (F5) |

### F7 (S2) — Runtime compilation latency and caching

**Problem:**
- The Demo concatenated sources for NVRTC, which is fine for one model but slow when every `udf.define` recompiles stdlib headers.
- Hundreds of model-written UDFs per session are expected.

**Recommendation:**
- Precompile stdlib and core device code to **LTO-IR**.
- Compile only the UDF with NVRTC (`-dlto`), then link with **nvJitLink**.
- Cache by `(source hash, arch, flags, catalog version)` in a persistent on-disk cache shared across sessions.
- Serve the first evaluation from the VM while compilation finishes, as the spec already says. Budget compile time as its own resource in requirements.

### F8 (S2) — Determinism rules are under-specified for CUDA

**Problem:** the [05](05-execution.md) claim of bit-identical replay on the same device needs explicit rules. Floating `atomicAdd`, cuBLAS split-K/atomics modes and some library algorithms are nondeterministic.

**Recommendation:**
- No float atomics on deterministic paths.
- Use CCCL's deterministic reductions, or fixed-tree reductions.
- Configure cuBLAS for deterministic algorithms (no atomics, fixed workspace).
- Pin library versions per harness release.
- Record the device, driver, CUDA, library and kernel hashes in the trace.
- Tensor-core results, including FP64 emulation, are architecture-dependent. That fits the existing "no cross-device bit claim", but say explicitly that **promotion gates compare within architecture**.

### F9 (S2) — Asynchronous exact verdicts and the GPU harness

**Problem:** writing exact (CPU) verdicts back "asynchronously into device memory" while a captured graph runs means one of two things:
- device kernels spin on mapped host memory, which needs system-scope atomics and is fragile under graphs
- host nodes in the graph, which serialize the stream

Neither exists on wgpu.

**Recommendation:**
- Rollouts run in **segments**. Each segment is one graph launch, internally looping with WHILE.
- Exact verdicts that complete during segment *k* are applied by a small parameter-patch upload before segment *k+1*.
- Until then, configurations awaiting exact verdicts keep `PENDING_EXACT`. Models may keep working on other configurations.

This keeps the zero-round-trip inner loop and stays portable.

### F10 (S3) — Multi-tenancy and co-scheduling

- The GPU harness, CuModel inference and background proxy rebuilds compete for SMs.
- Use **stream priorities**: the harness gets high priority and rebuilds low.
- On supporting drivers, use **green contexts** to reserve SM partitions per role or tenant.
- Use **MPS** for many lightweight session workers.
- Use **MIG** on datacenter GPUs for hard isolation in the multi-tenant server (M6).

### F11 (S3) — Use tensor cores where geometry is secretly a GEMM

- Tensor-product surface sampling of a whole UV grid is `B_u · P · B_vᵀ` per surface, which is batched GEMM.
  - Proxy-phase sampling, tessellation and render previews use TF32/FP16 tensor cores.
  - Exact-adjacent sampling uses FP64 emulation or df64.
- The same applies to batched B-spline least-squares fitting and to CuModel MLP/GNN layers.
- The CubeCL matmul component can cover wgpu.

### F12 (S3) — NVIDIA Warp as oracle and design reference (not a dependency)

- NVIDIA Warp already covers much of what AgLang needs: mesh BVH queries with tile/cooperative variants, sparse Lipschitz-octree marching cubes for implicit functions, SDF utilities and kernel autodiff.
- AgCAD stays pure Rust, so Warp is not a runtime dependency. Use it as:
  - an **independent oracle** in conformance (marching cubes, mesh queries, autodiff gradients), run offline in CI
  - a **design reference** for the AgLang kernel model, especially the tile API and its autodiff rules

### F13 (S3) — Ecosystem interop: OpenUSD / Omniverse

- Add `file.export_usd`: geometry, components, variants and attributes.
- USD is the interchange layer for NVIDIA Omniverse, Isaac Sim and Houdini Solaris.
- Design-table configurations map naturally onto USD **variant sets**, and procedural attributes map onto primvars.
- PhysX/Newton/Warp-based simulation of `assembly.mechanism_joint` and PhysicsNeMo surrogate evaluators are later, optional UDF-level integrations.

### F14 (S3) — Observability and budget metering

- **Profiling:** emit **NVTX** ranges per feature/opcode/segment, so Nsight Systems/Compute profile per feature. Run Nsight in the perf CI.
- **Device-side budget metering:** use `%globaltimer`/`clock64` per block with a calibrated per-op cost model. CUPTI is unsuitable inside the loop. Report metered time against CUDA-event wall time per segment.

### F15 (S3) — Rust toolchain specifics

- **Shared CUDA context with Candle.** Candle's CUDA backend depends on `cudarc`. To share one `CudaContext` and stream with AgCAD (zero-copy weights, one graph), **pin AgCAD to Candle's `cudarc` version** or isolate Candle in the training process only. Recommended: Candle trains in a separate process and AgCAD's Tier N/P kernels do inference ([08](08-hal-backends.md)), which avoids the version lock.
- **CUDA 13 minimum arch is Turing (sm_75).** The Demo's Ada (sm_89) is fine. State the support floor: sm_75+ on Tier N, and anything wgpu supports on Tier P.
- **Linux enables GPUDirect Storage** for large scan/VDB IO, which was out of scope on the Demo's Windows box. It is low priority.

## Amendments applied to the spec

| Doc | Change |
|---|---|
| [08](08-hal-backends.md) | Two-tier kernels (P: CubeCL, N: NVRTC + NVIDIA libraries); precision chosen per device from `fp64_rate`; df64 also on CUDA consumer parts; FP64-emulated GEMM; extended trait capabilities; NVIDIA library map; Candle process separation |
| [06](06-harness.md) | GPU harness rounds as validate → bin-by-opcode → per-opcode kernels (SWITCH/IF nodes) → check/decide/meter/record/rank inside a WHILE node; exact verdicts applied at segment boundaries |
| [05](05-execution.md) | CUDA determinism rules; NanoVDB layout; OptiX/cuBQL for queries; isolation rule |
| [02](02-geometry-model.md) | Implicit/voxel bricks use NanoVDB; RT-core ray queries |
| [11](11-self-improvement.md) | L4 candidates run in isolated sandbox workers under compute-sanitizer |
| [`op-catalog.json`](op-catalog.json) | `file.export_usd` (configurations → USD variant sets) |
| [10](10-roadmap.md) | M0 spikes: CubeCL-CUDA feature gaps, conditional nodes, df64 vs f64 per device; M2: OptiX shim |

## Sources

- FP64 emulation in cuBLAS (Ozaki; CUDA 13.0u2): https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
- Ozaki-II / 2M in cuBLAS 13.4.2: https://developer.nvidia.com/blog/faster-scientific-simulations-with-new-fp64-algorithms-in-nvidia-cublas/
- Conditional graph nodes (IF/WHILE 12.3; IF-ELSE/SWITCH 12.8): https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes and https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/
- Device launch of conditional nodes unconfirmed: https://forums.developer.nvidia.com/t/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/292531
- NVIDIA Warp geometry, BVH/cuBQL, tile queries: https://nvidia.github.io/warp/latest/api_reference/warp_geometry.html and https://nvidia.github.io/warp/v1.17/project/changelog.html
