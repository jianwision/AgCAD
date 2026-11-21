# 13 — Review: APIs & Abstractions vs NVIDIA / CUDA Conventions

> Status: **Informative** · Spec v0.3.1 · Area: review

> **Superseded in part by v0.3:** exact-topology work moved from CPU to GPU ([02](02-geometry-model.md#exact-geometry-solidsurfacewire-bodies), [05](05-execution.md)). Where this review mentions CPU exact work or host queues, read "GPU exact wavefronts".

This reviews the spec's programmatic surfaces, as of spec v0.2 plus the [CUDA review](12-cuda-review.md) amendments. It complements [12](12-cuda-review.md), which covered performance and ecosystem; this document covers **API shape**. Dated 2026-11-20.

The surfaces reviewed:
- the HAL `Backend` trait ([08](08-hal-backends.md))
- the bytecode and device-buffer ABI ([04](04-agir.md), [07](07-model-interfaces.md))
- status and error codes ([06](06-harness.md))
- batching layouts ([05](05-execution.md))

The reference conventions are:
- the CUDA driver/runtime APIs
- NVIDIA library APIs (cuBLAS, cuSOLVER, cuDSS, CUB/CCCL, OptiX)
- the Rust crates AgCAD actually sits on: `cudarc`, which the Demo used and Candle uses, and CubeCL

The "Amendments applied" section lists what was folded back into the spec.

Severity:
- **A1:** the API is wrong or unsafe as written.
- **A2:** it diverges from convention in a way that will cost later.
- **A3:** polish.

## Findings

### A1-1 — One fat `Backend` trait with `Option` features is the wrong abstraction

**Problem:**
- [08](08-hal-backends.md) puts device memory, VMM, graphs, conditionals, green contexts and libraries on a single trait, with `library::<L>() -> Option<&L>`.
- That forces every backend to stub CUDA-only features, and it hides capability at the type level.
- It also **duplicates CubeCL's runtime** (`Runtime`, `ComputeClient`, handles) for the portable tier.

**Convention:** CUDA gates features by compute capability and driver version, Vulkan by extensions, and CubeCL by runtime type. Rust expresses this as **extension traits** implemented only where they apply.

**Recommendation:** use a small core plus typed extensions:
- **Tier P** is CubeCL's `ComputeClient<R: Runtime>` directly, with no wrapper.
- `AgDevice` is a thin AgCAD core: arenas, packed readback, segments.
- CUDA-only capability lives in extension traits implemented by `CudaDevice`: `GraphExt`, `ConditionalExt`, `VmmExt`, `LibraryExt`, `PartitionExt`, `RayTracingExt`.
- Features that need an extension take `D: AgDevice + GraphExt`. Absence is a compile-time or dispatch-time fact, not a runtime `None` discovered mid-rollout.

### A1-2 — Asynchronous error semantics are missing

**Problem:**
- `launch(...) -> Result<()>` reads as if it reports kernel failure. In CUDA, a launch returns **enqueue** errors only. Execution faults surface later, at a sync, event query or next API call.
- Faults like illegal address are **sticky**: they poison the context.

**Convention:** CUDA distinguishes non-sticky errors (bad launch config, out of memory) from sticky ones. Libraries return status enums (`cublasStatus_t`) and never panic.

**Recommendation:**
- `enqueue_*` returns `Result<(), EnqueueError>`.
- `sync` and `event.query()` return `Result<(), DeviceFault>`, where `DeviceFault { sticky: bool, code, stream, segment_id }`.
- A sticky fault transitions the worker to `Poisoned`. Harness policy then kills the worker process and restores the session from its last version, which ties into review F3's process isolation.
- **Device-side command failures are data, not errors.** They are `AgStatus` codes in status records (A1-4), never Rust errors.

### A1-3 — Buffer lifetime and drop must be stream-ordered

**Problem:**
- The trait has `alloc` but no free, and `Buffer` is untyped bytes.
- In CUDA, freeing memory that in-flight work still uses is undefined behaviour. Rust's `Drop` runs immediately.

**Convention:** `cudarc`'s `CudaSlice<T>` is typed (`DeviceRepr`/`ValidAsZeroBits`) and stream-associated. Stream-ordered `cudaFreeAsync` makes deferred frees safe.

**Recommendation:**
- Use typed `DeviceSlice<T: Pod>` bound to an owning stream. `Drop` enqueues `free_async` on that stream, or defers to a per-segment retire list for arena sub-allocations.
- Cross-stream use requires an explicit `wait_event`.
- Arena handles (slot + generation) never expose raw device pointers to model code.

### A1-4 — Status and error codes need fixed numeric values and categories

**Problem:**
- [06](06-harness.md) lists names only, and they mix several kinds: per-command outcomes (`ACCEPTED`, `NO_IMPROVEMENT`), harness/API errors (`STALE_REVISION`, `COMPILE_FAILED`) and transient states (`PENDING_EXACT`).
- The device status record needs stable numbers.

**Convention:** `cudaError_t` / `cublasStatus_t` have fixed values, `0 == success`, append-only numbering and `*GetErrorString`/`*GetErrorName`.

**Recommendation:** a single `AgStatus: u16` with fixed values. The **high byte is the category**:

| Range | Category |
|---|---|
| `0x00xx` | success/info |
| `0x01xx` | lifecycle |
| `0x02xx` | revision |
| `0x03xx` | parameter |
| `0x04xx` | check |
| `0x05xx` | reference |
| `0x06xx` | resource |
| `0x07xx` | code |
| `0x08xx` | solver |

- Values are append-only.
- `ag_status_name` / `ag_status_string` exist on host and device.
- `0x0000` is `ACCEPTED`, matching the Demo's `ACCEPTED=0`.

### A2-1 — Graphs: separate template, executable and parameter patching

**Problem:** `capture() -> Graph` followed by `replay(graph, patch)` conflates two CUDA objects:
- `cudaGraph_t`, the editable template
- `cudaGraphExec_t`, the instantiated executable

Patching node parameters per launch (`cudaGraphExecKernelNodeSetParams`) is also slow and brittle.

**Convention:**
- `instantiate` once.
- `cudaGraphExecUpdate` handles topology-compatible changes.
- Kernels read **parameters through a device-resident parameter block**, so a patch is an ordinary `memcpy_async` before launch and the graph never changes.

**Recommendation:**
- `GraphExt::begin_capture(stream, mode) → end_capture() → Graph → instantiate() → GraphExec → launch(exec, stream)`.
- Capture mode is thread-local by default.
- All kernels take `const SegmentParams* __restrict__`, or a `__grid_constant__` struct for small blocks.
- `ParamPatch` becomes "write into the parameter block", and the graph is reused across segments.
- Conditional handles are created on the `Graph` and set on device via `cudaGraphSetConditional`. Verify in M0 that conditional nodes can be added during stream capture with the target CUDA version; otherwise build those subgraphs explicitly.

### A2-2 — Launch configuration should use CUDA's shape vocabulary

**Problem:** `launch(stream, module, grid: Grid, args: &Args)` has no kernel name, block shape, dynamic shared memory or cluster dimensions.

**Convention:**
- `cudaLaunchKernelEx` takes `cudaLaunchConfig_t`: grid/block `dim3`, dynamic shared memory bytes, stream, and attributes such as cluster dimension and priority.
- `cudarc` provides `LaunchConfig` plus `stream.launch_builder(&func).arg(..).launch(cfg)`, which the Demo used.
- CubeCL uses `CubeCount` and `CubeDim`.

**Recommendation:**
- `LaunchConfig { grid: Dim3, block: Dim3, shared_mem_bytes, cluster: Option<Dim3>, attrs }`.
- Kernels are looked up by name from a loaded library: `CudaFunction`.
- Kernel argument size stays small. Bulk arguments travel by pointer to the parameter block (A2-1), which avoids per-launch parameter-size limits.

### A2-3 — Modules: use the context-independent library API and name kernels

**Problem:** `load(&KernelIr) -> Module` is fine for Tier P, but Tier N needs a convention for kernels compiled from NVRTC, LTO-IR and fatbins.

**Convention:** CUDA 12's `cuLibraryLoadData` / `cuLibraryGetKernel` loads context-independently, with lazy loading. nvJitLink produces the final cubin from LTO-IR.

**Recommendation:**
- `LibraryExt::load_library(image: CubinOrPtx) → CudaLibrary`, then `library.kernel("name") → CudaFunction`.
- The cache key includes the compute capability (`sm_XY`), toolkit/driver versions, flags and source hash, as [08](08-hal-backends.md) specified.
- Host code must embed **PTX for forward compatibility** alongside the cubins.

### A2-4 — Transfers: pinned staging, not `Vec<u8>`

**Problem:** `download_packed(...) -> Vec<u8>` implies pageable host memory. A device-to-host copy into pageable memory is synchronous and staged, which defeats the one-readback design.

**Convention:** asynchronous copies need **page-locked** host memory (`cudaMallocHost` / `cudaHostAlloc`).

**Recommendation:**
- `PinnedBuffer` is a reusable per-session staging ring.
- `download_packed_async(stream, regions, &mut PinnedBuffer) → Event`.
- The host parses only after the event completes, as the Demo did: one packed copy, then wait.

### A2-5 — Events and cross-stream dependencies are first-class

**Problem:** the CPU–GPU pipelining ([05](05-execution.md)), segment boundaries and priority streams all need dependencies, and the trait only has `sync`.

**Convention:** `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaEventElapsedTime`, and `cudaLaunchHostFunc` for host callbacks.

**Recommendation:**
- `record(stream) → Event`, `wait(stream, &Event)`, `elapsed(&a, &b)` and `event.query()`.
- The scheduler builds its ExecGraph dependencies from events.
- Host callbacks are not used inside harness graphs. They serialize the stream.

### A2-6 — Batched data layout should follow cuBLAS and CUB conventions

**Problem:** "SoA with segment offsets" ([05](05-execution.md)) is underspecified. Every library interop point (cuBLAS batched, cuSOLVER batched, CUB segmented, NanoVDB, OptiX build inputs) would otherwise need a conversion.

**Convention:**
- **Uniform batches** use cuBLAS *strided-batched*: base pointer, leading dimension, batch stride and batch count.
- **Ragged batches** use CUB *segmented* offsets: `n+1` offsets, `[begin_i, begin_{i+1})`, or cuBLAS *grouped* GEMM for groups of uniform sizes.
- Pointer-array batched calls only where unavoidable.

**Recommendation:** every `Configurations<T>` buffer carries a `BatchLayout`:

```rust
pub enum BatchLayout {
    Strided { count: u32, stride_elems: u64, ld: u32 },     // cuBLAS strided-batched convention
    Segmented { count: u32, offsets: DeviceSlice<u32> },   // CUB: count+1 offsets
    Grouped { groups: Vec<(u32 /*count*/, Shape)> },       // cuBLAS grouped-GEMM convention
}
```

- Alignment is 128 B per array start, for coalescing and TMA eligibility on Hopper+.
- df64 values are stored as **two separate SoA arrays (hi[], lo[])**, not interleaved, so f32 vector loads stay coalesced.

### A2-7 — Library handles: per stream, created once, not thread-shared

**Problem:** `library::<L>()` returning a shared reference suggests one global handle.

**Convention:**
- cuBLAS, cuSOLVER and cuDSS handles are expensive to create, bound to a stream with `SetStream`, and given an explicit workspace.
- A handle must not be used concurrently from multiple host threads.
- cuBLAS also needs a capture-safe preconfigured workspace for use in graphs.

**Recommendation:**
- `LibraryExt::blas(&stream) → BlasHandle`, cached per `(stream, thread)`.
- Handles are `Send + !Sync`.
- Workspace comes from the stream's memory pool and is fixed before capture.
- Determinism flags are set at creation (review F8).

### A2-8 — Multi-GPU is absent

**Problem:** the spec models one device. The configuration axis is embarrassingly data-parallel across GPUs, and the convention for this is well established.

**Convention:**
- One context (primary context) per device.
- Peer access between devices (`cudaDeviceEnablePeerAccess`).
- **NCCL** collectives for cross-device reductions.

**Recommendation:**
- `DeviceSet` shards configurations across devices. Each device runs its own wavefront and holds a replica of the read-only document.
- Global `study.rank` is a per-device top-k followed by a k-way merge, either with NCCL all-gather or on the host for small k. Determinism comes from a stable merge by configuration id.
- On wgpu, multiple adapters are host-orchestrated only.
- The milestone is M4.

### A2-9 — Tensor interop should use DLPack

**Problem:** observation, command and status buffers ([07](07-model-interfaces.md)) are bespoke. External models and frameworks would need adapters.

**Convention:** **DLPack** (`DLManagedTensor`: device type and id, dtype, shape, strides, byte offset) is the cross-framework zero-copy tensor standard, used by PyTorch, JAX, CuPy, Warp and others. The CUDA Array Interface is the Python-side equivalent.

**Recommendation:**
- Every device-visible tensor (observation, command ring, status ring, sensitivity outputs) exposes a **DLPack descriptor**.
- Candle has no native DLPack, so `agcad-model` converts safetensors to and from it on the host.
- Optional PyO3 bindings later get zero-copy access for free.

### A2-10 — Cross-process sharing for sandbox workers

**Problem:** review F3 puts untrusted kernels in a separate process, which would duplicate read-only document pages.

**Convention:**
- CUDA IPC: `cudaIpcGetMemHandle`, or with VMM `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`, using POSIX file descriptors on Linux.
- Mapping read-only (`cuMemSetAccess`) prevents writes from the sandbox.

**Recommendation:**
- The main worker exports document pages read-only to sandbox workers.
- Sandboxes write only into their own arenas.
- Results come back through a shared, exported output arena, validated by the main worker.

### A3-1 — Bytecode record framing and versioning

**Problems:**
- Mixing the `AGB1` magic into the first record's bytes conflates framing with records.
- There is no size field, so layout cannot evolve.

**Convention:** CUDA and Vulkan structures carry a type/size or `sType` field for forward compatibility, and stream protocols use a separate block header.

**Recommendation:**
- A 64-byte **block header** `{magic 'AGB1', abi_version u16, record_size u16, record_count u32, requirements_rev u32, catalog_hash u64, …}` precedes a block of records.
- Records keep a clean 64-byte layout aligned to 16 bytes, and `imm: f64[2]` stays 8-byte aligned.
- The status record becomes `{status: AgStatus u16, flags u16, dst u32, budget_used u32, aux u32}` = 16 B.

### A3-2 — Device queries: attribute style

**Problem:** the `Caps` struct is fine as a cache, but it should be filled from the CUDA convention.

**Convention:** `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_*)`, compute capability major/minor, driver version, `cudaDeviceProp`.

**Recommendation:**
- Add `compute_capability: (u8, u8)` and `driver_version`.
- Derive `fp64_rate` from a per-architecture table plus a microbenchmark at startup.
- Record all of it in traces.

### A3-3 — Naming

Follow CUDA and cudarc naming in Rust APIs:

| Use | Avoid |
|---|---|
| `*_async` for stream-ordered operations | — |
| `Stream` | queue |
| `Event` | fence |
| `Library` / `Function` | module/kernel ambiguity |
| `Graph` / `GraphExec` | — |
| "pinned" | "page-locked" variants |

The CAD-facing API keeps CATIA vocabulary ([02](02-geometry-model.md)). CUDA vocabulary stays below the HAL line.

### A2-11 — Project layout should mirror the CUDA toolkit's driver/runtime split

**Problem:** the earlier crate list was flat (`hal`, `runtime`, `sched`, …). It had no API-contract boundary between explicit low-level control and convenient high-level use.

**Convention:** the CUDA toolkit has distinct layers, each with its own contract:
- a **driver API** (`libcuda`: explicit contexts, modules, launches)
- a **runtime API** (`cudart`: implicit init, primary context, interoperable with the driver)
- a **runtime compiler** (NVRTC, nvJitLink)
- **libraries** (cuBLAS, CUB, …) built on runtime + driver
- **tools** (Nsight, compute-sanitizer)

The driver is backward compatible with older runtimes.

**Recommendation (applied in [01](01-architecture.md#project-layout-driver--runtime-split-cuda-style)):**
- **`agcad-driver` (+ `-cuda`, `-wgpu`, `-cpu`)**: explicit and handle-based. It owns the bytecode ABI and the `AgStatus` values.
- **`agcad-runtime`**: implicit default session per thread/device, the CATIA-vocabulary API, and runtime→driver handle interop.
- **`agcad-rtc`**: the runtime compiler.
- **Libraries:** `geom`, `std`, `proc`, `kernels`.
- **Tools:** `cli`, `trace`, `conformance`.
- **Frameworks:** `harness`, `model`, `improve`, `server`.

Downward-only dependencies, a CUDA-style version compatibility rule, and a C ABI (`agcad.h`) post-MVP.

## Amendments applied

| Doc | Change |
|---|---|
| [08](08-hal-backends.md) | Trait replaced by core `AgDevice` + extension traits (`GraphExt`, `ConditionalExt`, `VmmExt`, `LibraryExt`, `PartitionExt`, `RayTracingExt`); Tier P uses CubeCL `ComputeClient` directly; typed stream-ordered `DeviceSlice<T>`; pinned staging; events; `LaunchConfig`; graph template/exec + parameter block; error model (`EnqueueError` / sticky `DeviceFault`) |
| [06](06-harness.md) | `AgStatus` numeric values and categories |
| [04](04-agir.md) | Block header + record/status layouts |
| [05](05-execution.md) | `BatchLayout` (strided / segmented / grouped), alignment, df64 SoA; multi-GPU `DeviceSet` |
| [07](07-model-interfaces.md) | DLPack descriptors for device tensors; CUDA IPC for sandbox workers |
| [01](01-architecture.md) | Project layout as driver / runtime / rtc / libraries / tools / frameworks; versioning and layering rules; driver-vs-runtime usage example |
