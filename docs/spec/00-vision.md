# 00 — Vision & Principles

> Status: **Informative** · Spec v0.3.1 · Area: vision

## The bet

LLMs can already create shape **by writing code**: implicit (signed-distance) functions, parametric surfaces, procedural generators, CadQuery-style scripts. A conventional CAD API offers them only a fixed menu of features, called one at a time, each blocking on exact topology. That menu is a *ceiling*:
- **TPMS lattices and graded infill** cannot be built from extrude and fillet.
- **Organic or generative forms** lose information when forced into sketch-based features.
- **Field-driven and topology-optimized parts** live naturally as functions.
- **Operators a model invents** have nowhere to live if the feature list is fixed.

At the same time, everyone who uses the result — engineers, PLM, CAM, suppliers — thinks in the **traditional CAD ontology**: parts, bodies, faces/edges/vertices, sketches, features, a feature tree, parameters, configurations, assemblies and mates. That vocabulary must not change.

AgCAD therefore takes this position:

> **Traditional CAD ontology, unlimited feature capability, executed on the GPU, with a GPU-resident harness and recursive self-improvement.**

- **Ontology.** Part, Body, Face/Edge/Vertex, Sketch, Feature, Feature tree, Global variables, Configurations, Assembly, Mate, Version/Branch keep their conventional meaning ([02](02-geometry-model.md#ontology)).
- **Capability.** Every feature is code. The standard features (extrude, revolve, fillet, shell, pattern, …) are written in AgLang and can be viewed, copied and modified. Models write **custom features** in the same language: implicit bodies, lattices, generative patterns, new blends, new converters. Custom features sit in the feature tree exactly like standard ones. (Code-defined custom features are an established CAD idea. AgCAD makes them first-class, GPU-compiled and model-authored.)
- **Body types beyond B-rep.** Solid and surface bodies (exact B-rep) remain the delivery format for manufacturing. Mesh, implicit, voxel and point-cloud bodies are peers. Conversions are ordinary `convert.*` features with stated deviation.
- **GPU execution.** A configuration is a row of a design table, and the GPU rebuilds thousands of configurations in one pass. Design studies, sensitivity and optimization are native.
- **GPU harness.** Requirements checks, Design Checker rules, budget metering and workspace commit/rollback run **on the GPU, per configuration, inside the inner loop** ([06](06-harness.md#gpu-resident-harness)). There is no per-step host round trip, and the host only handles exact-topology work.
- **Recursive self-improvement.** Models improve the feature library, the converters, the GPU kernels, the harness's checker implementations and themselves. Frozen reference checks and held-out benchmarks gate every promotion ([11](11-self-improvement.md)).

## Implementation stance

- **Pure Rust**, as in the Demo: kernel, stdlib, harness, compiler, server and model adaptors. AgLang is a Rust-syntax subset compiled at runtime by AgCAD itself.
- **Device models are trained with Candle**, as in the Demo, and run inside the GPU rollout from the same safetensors weights.
- **The MVP is headless:** library, `agcad` CLI and MCP server. There is no GUI.
- **GPU-only data plane.** Every geometry computation runs on the GPU:
  - exact B-rep (SSI marching, topology builders, booleans, fillets, sewing)
  - solvers
  - evaluations and constraints
  - the harness
  - file encoding

  The host is a control plane only (network, compilation, graph launches, file handles), and it never reads or writes geometry ([05](05-execution.md#host-interaction-budget)).

## Thesis and kill criteria

**Thesis:** for agent-driven design, a GPU-resident CAD system that evaluates thousands of configurations per step, under a GPU-resident harness, beats an LLM writing CPU CAD scripts. It wins on task success, cost and time, and the gap widens with self-improvement.

**Baseline** (benchmarked from M1 on identical tasks and requirements): an LLM writing CadQuery/build123d scripts against OCCT on CPU, with the same LLM, the same budget and the same checks.

**Kill/redirect criteria (checked at the M2 exit):**
- If AgCAD does not beat the baseline on **at least two benchmark families** in success rate at equal cost, or in cost at equal success rate, the plan is re-scoped before M3.
- If proxy fidelity ([05](05-execution.md#proxy-fidelity)) on survivors is below τ = 0.7 for the core Part Design features, the "preview cheaply" strategy is re-evaluated per feature.
- If topology-class fragmentation makes most studies effectively serial, the configuration-batch thesis is restricted to the families where it holds.
- If the GPU exact kernel fails OCCT parity on more than 1 % of the boolean/fillet conformance corpus, its ops stay `experimental`, and exact delivery is limited to the ops that pass.

## Principles

1. **The feature tree is the source of truth.**
   - A part is a feature tree driven by global variables and equations.
   - Every feature is a pure, GPU-compilable function of its inputs and parameters.
   - Bodies are the rebuild result.
2. **Conventional ontology, open capability.** New capability arrives as new *features* and *body types*, never as new top-level concepts that CAD users wouldn't recognize.
3. **Standard features are a library, not a cage.** They are complete and convention-faithful, but they are not mandatory. A model may bypass, copy or extend any of them.
4. **Configurations are the unit of work.**
   - Every feature has batch axes `[configuration, entity, sample]`.
   - A single edit is a design table with one row.
   - The compiler lifts custom-feature code over configurations automatically.
5. **Preview cheaply, rebuild exactly.**
   - Explore with GPU lightweight representations (tessellation, implicit bricks, sample grids).
   - Pay for exact B-rep rebuild only for surviving configurations, and only when the requirements need it.
6. **One feature language, two front doors.**
   - LLMs write AgLang source or AgIR JSON.
   - GPU-resident models write AgIR **bytecode** into device memory, executed by an on-device VM.
   - Both front doors share the same feature tree, harness and history ([04](04-agir.md)).
7. **Differentiable by default.**
   - Design sensitivity (∂measure/∂global variable) is available everywhere geometry allows.
   - Topology changes are reported as discontinuities.
8. **The harness checks outcomes, not methods.**
   - Requirements define what must be delivered and how it is checked. Budget and sandbox define what execution may consume.
   - The model chooses features, body types, strategy and inventions.
   - The harness never secretly optimizes, repairs or substitutes ([06](06-harness.md)).
9. **Evidence, not assertion.**
   - Every measurement and check reports its confidence (`exact | bounded | sampled | unknown`).
   - A model's own checks can add evidence but never replace requirement checks.
10. **Self-improvement is a system feature, not an afterthought.**
    - Every op, kernel, converter and checker implementation is replaceable by a better verified version.
    - Every run produces trace data for training.
    - Promotion is gated by frozen references ([11](11-self-improvement.md)).

## Conventional vs AgCAD

| Conventional CAD | AgCAD (same terms) |
|---|---|
| Fixed feature set + optional custom-feature scripting | Standard features *are* custom features; models author and publish new ones |
| Solid/surface bodies only (mesh as import) | Solid, surface, wire, mesh, implicit, voxel and point-cloud bodies; `convert.*` features between them |
| Configurations: a handful, rebuilt one at a time | Configurations: thousands, rebuilt in one GPU pass |
| Rebuild errors → user fiddles | Implicit-body booleans/blends never fail. Exact rebuild errors come with names, causes and evidence |
| Design checker: a manual tool | Design Checker rules run on the GPU per configuration, at declared phases, every step |
| Design study/optimization: separate module | `study.*` features with autodiff sensitivity in the same kernels |
| Software improves by releases | Feature library, kernels, checks and models improve continuously under gates |

The conventional path stays first-class. Take "a flange with 6 M8 holes on a 100 mm PCD, export STEP": `sketch.circle` → `part.pad` → `part.hole` → `transform.circular_pattern` → `file.export_step` must give named, editable, exact results.

## Non-goals (v1)

- 2D drawings and GD&T layout. STEP AP242 PMI pass-through comes later.
- Certified FEA/CFD/CAM. These are expressible as custom-feature analysis code, but uncertified.
- Cross-device bit-identical results. Determinism is per device and binary ([05](05-execution.md#determinism)).
- Certification of Class-A quality or manufacturability.

## Lessons from `docs/Demo`

The Demo's first experiment was already code-first: Rust compiled LLM-written CUDA implicit-body code (`cuda/model.cuh`) at runtime. Its GPU harness (`cuda/gpu_harness.cu`) already committed and rolled back 32 configurations per round on the device. Later work drifted into fixed templates.

**Ported:**

| From | What survives |
|---|---|
| `README.md`, `cuda/model.cuh` | Runtime-compiled model-written GPU geometry code; single packed readback |
| `HARNESS.md`, `cuda/gpu_harness.cu` | **Device-side** propose → check → commit/rollback per configuration; error taxonomy; bit-compare replay; budget never refunded |
| `MODEL_SESSIONS.md`, `scripts/model_session.py` | Handles `(session, slot, generation, revision)`; workspace edits may be invalid; failed save keeps the workspace; branch/switch/delete |
| `SELF_IMPROVEMENT_PLAN.md`, `SELF_IMPROVEMENT_REVIEW_V2.md` | Responsibility split; frozen oracles; archive of diverse solutions; independent promotion evaluation; "Stop ≠ success" |
| `scripts/model_adaptor.py` | `ModelAdaptor.decide(observation)`, with `Llm`, `CuModel` and `ReplayModel` interchangeable |
| `cuda/surface_primitives.cuh`, `cuda/curve_ops.cuh` | NURBS/Bezier derivatives to order 3, curvature and continuity metrics |
| `scripts/feature_graph.py` | Stable feature ids, revision-bound selections, ambiguity is an error |
| `primitive_*.json`, `CAPABILITIES_V4.md` | Coverage checklist for standard features |

**Avoided:**

| Demo symptom | AgCAD rule |
|---|---|
| 134 primitives unreachable from the model session | One feature table, callable from both front doors |
| Fixed 4-patch hood template, 6 DOF, 4-slot neural action space | No templates. A device model's action space is AgIR bytecode over the whole feature table |
| CPU OCCT as topology substrate | GPU-native exact kernel; OCCT is a test oracle only; the host never touches geometry |
| CUDA-only | HAL with CUDA + wgpu ([08](08-hal-backends.md)) |
| Self-improvement plan never started | Self-improvement loops are milestones with exit criteria ([10](10-roadmap.md)) |
