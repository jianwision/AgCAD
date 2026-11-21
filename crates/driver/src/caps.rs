//! Device capabilities (spec 08 § Device API; A3-2). Filled from device attributes plus an
//! architecture table and a startup FP64 microbenchmark.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TensorTypes {
    pub fp16: bool,
    pub bf16: bool,
    pub tf32: bool,
    pub fp8: bool,
    pub int8: bool,
    pub fp64_mma: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CondSupport {
    /// No device conditionals: loops are host-side segment loops (wgpu, CPU).
    #[default]
    HostEmulated,
    /// IF and WHILE nodes (CUDA 12.3+).
    IfWhile,
    /// IF, IF-ELSE, WHILE, SWITCH (CUDA 12.8+).
    Full,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Caps {
    pub name: String,
    pub backend: BackendKind,
    pub compute_capability: Option<(u8, u8)>,
    pub driver_version: u32,
    /// Native FP64 : FP32 throughput.
    pub fp64_rate: f32,
    pub fp64_gemm_emulation: bool,
    pub tensor_types: TensorTypes,
    pub rt_cores: bool,
    pub conditional_nodes: CondSupport,
    pub green_contexts: bool,
    pub subgroup_size: u32,
    pub atomics64: bool,
    pub max_alloc: u64,
    pub total_memory: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Wgpu,
    /// Reference oracle only; never a production path (AG-08-005).
    CpuReference,
}

/// Numeric policy chosen per device, not per backend (AG-08-002, review F1).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionPolicy {
    /// High native FP64 (A100/H100 class).
    NativeF64,
    /// df64 bulk + f64 residual refinement (GeForce/RTX, wgpu).
    Df64Refined,
}

impl Caps {
    pub fn precision_policy(&self) -> PrecisionPolicy {
        if self.backend == BackendKind::Cuda && self.fp64_rate >= 0.125 {
            PrecisionPolicy::NativeF64
        } else {
            PrecisionPolicy::Df64Refined
        }
    }

    /// FP64:FP32 rate by architecture (CUDA compute capability). Startup microbenchmark may refine it.
    pub fn fp64_rate_for_cc(cc: (u8, u8)) -> f32 {
        match cc {
            (6, 0) | (7, 0) | (8, 0) | (9, 0) => 0.5, // P100, V100, A100, H100
            (10, 0) => 0.5,                           // datacenter Blackwell (refined by benchmark)
            _ => 1.0 / 64.0,                          // GeForce / RTX workstation parts
        }
    }
}
