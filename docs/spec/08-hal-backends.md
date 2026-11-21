# 08 — HAL, Backends & Precision

> Status: **Normative** · Spec v0.3.1 · Area: driver

## Kernel authoring: two tiers

Revised after the [CUDA review](12-cuda-review.md) (F6).

| Tier | What | Backends | Role |
|---|---|---|---|
| **P (portable)** | **CubeCL** kernels, written once in Rust | CUDA, wgpu (Vulkan/Metal/DX12); CPU target for conformance only | Reference behaviour for every op. AgLang UDFs and wrangles compile here |
| **N (NVIDIA-native)** | NVRTC-compiled CUDA C++ (the Demo-proven path) + NVIDIA libraries: cuBLAS/cuBLASLt, cuSOLVER, cuDSS, CUB/CCCL, OptiX, NanoVDB, cuBQL | CUDA only | Per-op acceleration behind the same manifest. Promoted only by parity with Tier P ([09](09-validation.md)) |

- Core kernels start in Tier P. Hot paths get Tier N variants: SSI seeding, tessellation, ray queries, batched solves, scan/sort/compaction, CuModel matmuls, implicit bricks.
- **Compilation:**
  - Stdlib device code is precompiled to LTO-IR.
  - Model UDFs are compiled alone with NVRTC (`-dlto`) and linked with **nvJitLink**.
  - Compiled artifacts are cached on disk by `(source hash, arch, flags, catalog version)`.
  - The device VM serves first evaluations while compilation finishes (review F7).
- **Support floor:** Tier N needs sm_75+ (the CUDA 13 floor). Tier P runs on anything wgpu supports.

## Device API (core + extension traits)

Revised after the [API review](13-api-review.md). The portable tier uses CubeCL's `ComputeClient<R: Runtime>` directly. AgCAD adds a thin core, and CUDA-only capability lives in **extension traits** implemented only by `CudaDevice`, so absent features are known at compile time or dispatch time, never discovered as a runtime `None`.

```rust
/// Core: implemented by CudaDevice, WgpuDevice, CpuDevice.
pub trait AgDevice: Send + Sync {
    type Stream: Clone; type Event;
    fn compute_client(&self) -> &ComputeClient<Self::Runtime>;   // Tier P (CubeCL)
    fn caps(&self) -> &Caps;                                        // filled from device attributes
    fn stream(&self, priority: Priority) -> Result<Self::Stream>;
    fn alloc<T: Pod>(&self, s: &Self::Stream, n: usize) -> Result<DeviceSlice<T>>;   // stream-ordered; Drop = free_async
    fn arena<T: Pod>(&self, cap: usize, layout: BatchLayout) -> Result<Arena<T>>;   // resident slab, slot+generation handles
    fn pinned(&self, bytes: usize) -> Result<PinnedBuffer>;                          // page-locked staging
    fn copy_to_device_async<T: Pod>(&self, s: &Self::Stream, src: &PinnedBuffer, dst: &mut DeviceSlice<T>) -> Result<(), EnqueueError>;
    fn download_packed_async(&self, s: &Self::Stream, regions: &[Region], dst: &mut PinnedBuffer) -> Result<Self::Event, EnqueueError>;
    fn record(&self, s: &Self::Stream) -> Result<Self::Event, EnqueueError>;
    fn wait(&self, s: &Self::Stream, e: &Self::Event) -> Result<(), EnqueueError>;
    fn query(&self, e: &Self::Event) -> Result<bool, DeviceFault>;     // async faults surface here
    fn sync(&self, s: &Self::Stream) -> Result<(), DeviceFault>;
    fn elapsed_ms(&self, a: &Self::Event, b: &Self::Event) -> Result<f32, DeviceFault>;
}

/// CUDA-only extensions (CudaDevice).
pub trait LibraryExt: AgDevice {        // Tier N kernels + NVIDIA libraries
    fn load_library(&self, image: CubinOrPtx) -> Result<CudaLibrary>;      // cuLibraryLoadData (context-independent)
    fn launch(&self, s: &Self::Stream, f: &CudaFunction, cfg: LaunchConfig, params: &DeviceSlice<u8>) -> Result<(), EnqueueError>;
    fn blas(&self, s: &Self::Stream) -> Result<BlasHandle>;      // per (stream, thread); Send + !Sync; fixed workspace
    fn solver(&self, s: &Self::Stream) -> Result<SolverHandle>;  // cuSOLVER / cuDSS
    fn cub(&self) -> &CubKernels;                                // precompiled CCCL scan/sort/segmented reduce
}
pub trait GraphExt: AgDevice {
    fn begin_capture(&self, s: &Self::Stream, mode: CaptureMode) -> Result<()>;   // thread-local by default
    fn end_capture(&self, s: &Self::Stream) -> Result<Graph>;
    fn instantiate(&self, g: &Graph) -> Result<GraphExec>;
    fn update(&self, exec: &mut GraphExec, g: &Graph) -> Result<UpdateResult>;   // cudaGraphExecUpdate
    fn launch_graph(&self, s: &Self::Stream, exec: &GraphExec) -> Result<(), EnqueueError>;
}
pub trait ConditionalExt: GraphExt { fn conditional(&self, g: &mut Graph, kind: CondKind /*If|IfElse|While|Switch(n)*/) -> Result<CondHandle>; }
pub trait VmmExt: AgDevice {        // growable arenas, page-level CoW, IPC to sandbox workers
    fn reserve(&self, va_bytes: usize) -> Result<VaRange>;
    fn map(&self, va: &VaRange, off: usize, page: &PhysPage, access: Access) -> Result<()>;
    fn export_fd(&self, page: &PhysPage) -> Result<OwnedFd>;   // cuMemExportToShareableHandle (POSIX fd)
    fn import_fd(&self, fd: OwnedFd) -> Result<PhysPage>;
}
pub trait PartitionExt: AgDevice { fn green_context(&self, sms: u32) -> Result<Partition>; }
pub trait RayTracingExt: AgDevice { fn optix(&self) -> &OptixContext; }   // agcad-optix-sys shim

pub struct LaunchConfig { pub grid: Dim3, pub block: Dim3, pub shared_mem_bytes: u32,
                          pub cluster: Option<Dim3>, pub attrs: LaunchAttrs }
pub enum EnqueueError { OutOfMemory, InvalidConfig, NotPermittedDuringCapture, Other(i32) }
pub struct DeviceFault { pub sticky: bool, pub code: i32, pub segment: Option<SegmentId> }
// sticky ⇒ worker becomes Poisoned ⇒ harness kills the worker process and restores the last version

pub struct Caps {                       // filled from cuDeviceGetAttribute / wgpu adapter limits
    pub compute_capability: Option<(u8, u8)>, pub driver_version: u32,
    pub fp64_rate: f32,                 // native FP64:FP32 (≈1/64 GeForce/RTX workstation, ≥1/2 A100/H100 class); arch table + startup microbenchmark
    pub fp64_gemm_emulation: bool,      // cuBLAS Ozaki emulation (CUDA 13.0u2+)
    pub tensor_types: TensorTypes,      // fp16/bf16/tf32/fp8/int8/fp64-mma
    pub rt_cores: bool,                 // OptiX hardware traversal
    pub conditional_nodes: CondSupport, // IF/WHILE (12.3+), IF-ELSE/SWITCH (12.8+); host-emulated on wgpu
    pub green_contexts: bool, pub subgroup_size: u32, pub atomics64: bool, pub max_alloc: u64,
}
```

**Conventions:**
- **Kernel parameters** are read from a device-resident **parameter block** (`const SegmentParams* __restrict__`). Patching a segment is a `copy_to_device_async` into that block, and graphs are instantiated once and reused.
- **Command failures are data:** they are `AgStatus` codes in status records ([06](06-harness.md#error-codes)), never Rust errors.
- **Buffers** are typed (`T: Pod`), stream-bound and freed stream-ordered. Arena handles never expose raw device pointers to model code.
- **Embedded binaries:** cubins for target architectures plus PTX for forward compatibility.
- **Naming:** `_async` marks stream-ordered calls. CUDA vocabulary (Stream, Event, Library/Function, Graph/GraphExec, pinned) stays below the HAL line.

| Backend | Where | Notes |
|---|---|---|
| `CudaDevice` | NVIDIA (production) | cudarc (`CudaContext`/`CudaStream`/`CudaSlice`, as in the Demo); `AgDevice` + all extension traits; Tier P + Tier N; OptiX via a thin C++ shim (`agcad-optix-sys`) |
| `WgpuDevice` | Vulkan/Metal/DX12, **including this dev box's Intel UHD iGPU** | Tier P only; f32 + df64; prerecorded command buffers stand in for graphs, and conditionals become host-side segment loops; Vulkan ray queries where exposed |
| `CpuDevice` | reference/testing | Tier P through CubeCL's CPU target or the host interpreter; the conformance oracle |

## Precision policy (chosen per device, not per backend)

Revised after the review (F1). Consumer and workstation NVIDIA GPUs run native FP64 at about 1/64 of the FP32 rate, so "CUDA ⇒ f64" would make the main dev targets slow.

| Work | `fp64_rate ≥ 1/8` (A100/H100 class) | `fp64_rate < 1/8` (GeForce/RTX, Demo's RTX 2000 Ada) | wgpu |
|---|---|---|---|
| Exact-adjacent: SSI seeds, NURBS derivatives for fitting, residuals | f64 | **df64**, then f64 residual refinement | df64 |
| Dense LA (face fitting least squares, normal equations, Gauss–Newton) | cuBLAS f64 / FP64-emulated GEMM | **cuBLAS FP64-emulated GEMM** (Ozaki, adaptive-precision fallback) or df64 | df64 (CubeCL matmul) |
| Constraint solvers (sketch, assembly, optimize) | f64 | df64 + f64 refinement | df64 |
| Lightweight: tessellation, implicit bricks, BVH, render, proxy surface sampling | f32 / TF32 tensor-core GEMM (`B_u·P·B_vᵀ`) | same | f32 |
| Harness numeric checks | f64 | df64; final verdict margins recomputed in f64 for the survivors | df64 |
| Exact topology decisions (predicates) | filtered df64 → exact expansion arithmetic | same | same |
| Exact topology construction (SSI marching, builders, fillets, sewing) | f64 | df64 + f64 refinement | df64 |

- Every kernel declares a tolerance against the `CpuDevice` f64 reference.
- Conformance runs every op on every available backend and tier ([09](09-validation.md)).
- A backend that can't meet a requested evidence level returns `unknown` or `UNSUPPORTED_OP`. It is never silently downgraded.
- FP64 emulation and tensor-core results depend on the architecture, so promotion gates compare within an architecture (review F8).

## Models: Candle + HAL

- CuModel policies are defined and trained with **Candle**, on its CUDA backend. CPU training is a fallback for GPU-less development only. Weights are stored as safetensors.
- **Candle runs in a separate training process.** This avoids pinning AgCAD's `cudarc` to Candle's version and keeps training allocations out of the harness context (review F15).
- For rollouts, `agcad-model` loads the same safetensors weights into AgCAD kernels: Tier P (CubeCL matmul, activations, GNN message passing) or Tier N (cuBLASLt, capture-safe with preallocated workspace). Inference runs inside the captured graph next to the GPU harness, on CUDA and on wgpu.
- Candle has no wgpu backend, so this also lets the Intel iGPU dev box run device policies.
- A Candle-vs-HAL inference parity check runs for every checkpoint. The Demo measured that parity at ≈1e-15 mm on proposals.

## Headless operation

- The MVP needs no window system. wgpu runs with no surface. On CUDA, offscreen renders are ray-traced with OptiX (or use the Tier P rasterizer), and PNGs travel in the packed readback.
- **Observability:** NVTX ranges are emitted per feature, opcode and rollout segment for Nsight Systems/Compute. Device budget metering uses `%globaltimer`/`clock64` with a calibrated per-op cost model (review F14).
- The server runs on CI machines with a GPU. `CpuDevice` exists only as a conformance oracle and is never a production execution path.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-08-001 | Every Tier N kernel MUST pass parity with its Tier P counterpart before use. |
| AG-08-002 | Precision MUST be chosen from device capabilities (`fp64_rate`), not from backend identity. |
| AG-08-003 | Enqueue errors and device faults MUST be distinct; sticky faults MUST poison the worker. |
| AG-08-004 | Device buffers MUST be freed stream-ordered. |
| AG-08-005 | `CpuDevice` MUST NOT be used as a production execution path. |
