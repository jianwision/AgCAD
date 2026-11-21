# Glossary

> Status: **Informative** · Spec v0.3.1 · Area: all

CAD terms follow CATIA (see the ontology table in [02](02-geometry-model.md#ontology) for SolidWorks/Onshape/Houdini equivalents). This glossary defines AgCAD-specific and cross-cutting terms.

| Term | Definition |
|---|---|
| **AgIR** | Typed SSA intermediate representation of programs. JSON form for LLMs; bytecode form for device models ([04](04-agir.md)) |
| **AgLang** | Strict subset of Rust used to write UDFs, wrangles, rules, checks and evaluations; compiled at runtime by `agcad-rtc` |
| **AgStatus** | `u16` status code with fixed values; category in the high byte ([06](06-harness.md#error-codes)) |
| **Block / record** | Bytecode framing: a 64-byte block header followed by 64-byte command records ([04](04-agir.md#bytecode-device-resident-models)) |
| **Configuration** | One row of a design table; the leading batch axis |
| **Constraint** | An evaluation compared with a bound; driving (solved) or checked ([14](14-constraints-evaluations.md)) |
| **Control plane** | Host-side work limited to network, compilation, graph launches and file handles. Never touches geometry |
| **CuModel** | GPU-resident model (Candle-trained) that writes bytecode on device ([07](07-model-interfaces.md)) |
| **Data plane** | All geometry, solving, evaluation, harness and encoding work. Runs on the GPU |
| **Design Checker** | The set of versioned checks run at declared phases by the harness |
| **Driver / runtime** | `agcad-driver` (explicit, handle-based) vs `agcad-runtime` (implicit default session), mirroring CUDA's driver and runtime APIs |
| **Evaluation** | A measured quantity with value, evidence, optional gradient and cost ([14](14-constraints-evaluations.md)) |
| **Evidence level** | `exact`, `bounded`, `sampled` or `unknown`: how strongly a result is established |
| **Escalation chain** | The ordered methods that raise an evaluation's evidence level |
| **ExactKernel** | Trait for the exact B-rep kernel. GPU-native implementation in production; OCCT implementation for tests only |
| **Kernel primitive** | A geometry op implemented natively by the kernel; all others are UDF compositions (`kernel_primitive` in the catalog) |
| **Lightweight representation** | Derived GPU representation (tessellation, NanoVDB bricks, sample grids, BVH); disposable |
| **Margin** | Signed, tolerance-normalized distance of an evaluation from its bound |
| **Process** | Declared manufacturing process in requirements; selects delivery representation and DFM evaluations |
| **Proxy fidelity** | Measured agreement between lightweight and exact rankings ([05](05-execution.md#proxy-fidelity)) |
| **Requirement** | Harness-owned constraint with phase and required evidence; immutable during a session |
| **Segment** | One graph launch of the GPU harness. Exact verdicts and parameter patches apply between segments |
| **Tier P / Tier N** | Portable CubeCL kernels vs NVIDIA-native kernels and libraries ([08](08-hal-backends.md)) |
| **Topology class** | Configurations sharing a topological signature; the unit of uniform batching ([05](05-execution.md#topology-classes)) |
| **UDF** | User Defined Feature written in AgLang (CATIA term); standard features are UDFs too |
| **Wavefront** | One harness round: validate → bin → execute → check → decide → meter → record → observe → rank → escalate |
