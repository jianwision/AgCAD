# 07 — Model Interfaces

> Status: **Normative** · Spec v0.3.1 · Area: model / server

## ModelAdaptor

```rust
pub trait ModelAdaptor: Send {
    fn capabilities(&self) -> &CapabilityManifest; // allowed families/opcodes, precision, context, device?
    fn decide(&mut self, obs: &Observation) -> Result<Decision>; // Decision::Program | Decision::DeviceRollout
}

pub struct Llm { /* MCP/HTTP client to a local or remote LLM; emits AgLang or AgIR JSON */ }
pub struct CuModel { /* Candle-trained policy; safetensors weights resident on the GPU; writes bytecode on device */ }
pub struct ReplayModel { /* replays a trace; regression and cross-adaptor checks */ }
```

This is ported from Demo `scripts/model_adaptor.py` (Python there, Rust here), where the contract was already validated.

- Switching adaptors keeps the requirements, feature semantics and report definitions.
- A missing capability returns `UNSUPPORTED_OP`. It is never silently reinterpreted.
- `Llm` and `CuModel` can be composed inside a model: an LLM writes UDFs and design tables, and a CuModel explores parameters on device. That pairing is a strategy choice, not a fixed hierarchy.

## LLM path (headless MCP server / Rust API)

The MCP tools are few and coarse, so the per-op tool-call overhead is amortized:

| Tool | Purpose |
|---|---|
| `get_feature_table(filter?)` | Families, ops, aliases, signatures (generated from [`op-catalog.json`](op-catalog.json)) |
| `view_source(feature)` | AgLang source of a standard feature or UDF |
| `get_requirements()` | The immutable requirements |
| `run_program(program)` | Run an AgLang module or AgIR JSON program until the next `document.checkpoint` or the end; returns an `UpdateReport` |
| `specification_tree(target?)` | Tree, parameters, bodies, names, versions |
| `query(expr)` | Resolve names/queries; return elements with measurements |
| `render(items, views, mode)` | Offscreen PNGs (shaded, edges, face-id false colour, curvature, highlight lines) |
| `export(items, format)` | STEP / 3MF / STL / glTF / VDB |
| `finish()` | Delivery checks → `DeliveryReport` |

- **Typical loop:** read requirements → write UDFs/networks → one `run_program` with a design table of hundreds to thousands of configurations → read the ranked report and renders → refine code → repeat.
- **Compiler diagnostics** (`COMPILE_FAILED`, `SANDBOX_VIOLATION`) carry source spans and fix hints, because most LLM errors are code errors.
- **Rust in-process:** the `agcad` crate exposes the same API (`Session::run_program`, …) for Rust agent harnesses, with no server needed. PyO3 bindings are optional, post-MVP.
- **CLI:** `agcad run program.py --requirements req.yaml --out out/` is the headless batch entry point for CI and benchmarks.

### Diagnostics and perception (the LLM's real bottleneck)

LLM CAD agents fail mainly at spatial reasoning and reference selection, not at throughput. On the LLM path, latency is dominated by LLM calls, so the investment goes into **what the model sees**:

| Report element | Purpose |
|---|---|
| **Name-annotated renders** | Face/edge provenance names overlaid on shaded views (false colour per face id), generated on the GPU |
| **Diff renders** | Before/after highlight of created, modified and deleted elements between versions |
| **Section views** | Automatic sections through violated-constraint locations |
| **Causal failure reports** | For every failed feature or constraint: cause, involved elements, quantitative limit, suggestion. Example: "edge_fillet r=6 infeasible: adjacent edge `part.pad#3/side.2` is 4.1 mm; max feasible ≈ 3.9 mm; try r ≤ 3.9 or `part.variable_fillet`" |
| **Constraint margins** | Top violated constraints/requirements, with signed normalized margins and the gradient direction in parameter space ([14](14-constraints-evaluations.md)) |
| **Scene summary** | Bodies, bounding boxes, key dimensions, topology-class stats, DOF status, in text |

### Observation for LLMs

The report is JSON, sized to a token budget:
- the top-k configurations in detail
- distributions for the rest
- element names, not ids
- residuals with units
- optional images

The full data stays queryable through `query` and `specification_tree`.

## GPU model path (device ring)

```
device buffers (resident for the session):
  requirements (read-only) · document + compiled kernels (read-only for the model)
  command ring   [rounds × configurations × 64 B]       written by CuModel
  arg arena      typed records (point lists, expression graphs)
  status ring    [rounds × configurations × 16 B]       written by GPU harness
  observation    [configurations × obs_dim] f32/f64     written by study.observe
  trace ring                                          written by GPU harness
```

- Every buffer above publishes a **DLPack** descriptor (device, dtype, shape, strides), so external frameworks can attach zero-copy. Candle has no native DLPack, so `agcad-model` bridges via safetensors on the host.
- **Sandbox workers** (first-run UDFs, L4 candidates) receive read-only document pages through CUDA VMM IPC (`cuMemExportToShareableHandle` POSIX fds, `cuMemSetAccess` read-only) and write only into their own arenas ([API review](13-api-review.md) A2-10).

**Rollout:**

```
host: capture graph { repeat N: CuModel.forward → GPU harness (validate/execute/check/decide/meter/record) → observe }
host: launch graph; wait; one packed readback (trace + summary)
```

- **CuModel forward** is any **Candle**-defined model (MLP, GNN over the specification tree, small transformer).
  - **Training** uses Candle's CUDA backend, with on-device rollouts generating the data. Candle CPU training is a fallback for GPU-less development only.
  - **Inference inside the rollout** uses AgCAD's GPU kernels, which load the same safetensors weights, so the policy runs in the captured graph with no host hop.
  - This is the Demo's validated split: Candle training, then GPU inference with CPU/GPU proposal parity checked to ≈1e-15.
  - Conformance re-checks Candle-vs-kernel inference parity for every checkpoint.
- **Geometry encoder:** CuModels share a learned encoder rather than bespoke features:
  - a face/edge **UV-graph network** over B-rep elements (per-face UV grids of points, normals and curvature, with message passing over adjacency)
  - an **SDF/brick encoder** for implicit and voxel bodies

  It is pre-trained self-supervised on reconstruction tasks and fuzzer output with Candle. The same embeddings back `query.similar` (retrieval of parts, features and skills).
- **Output head** writes command records and, optionally, expression-graph bytecode for model-invented implicit bodies or wrangles ([04](04-agir.md#bytecode-device-resident-models)).
- **Action space** is the opcode allow-list in its capability manifest. Any feature-table op and any published UDF can be allowed, so there is no fixed slot count (the Demo's 4-slot limit is gone).
- **Observation layout** is declared by `ObsSpec`, which covers:
  - parameter values
  - requirement margins
  - verdict codes
  - measurements
  - sensitivity vectors
  - fixed-size spatial encodings (implicit samples on a grid, curvature histograms)
  - per-element feature vectors for GNN models
- **Exact work** during a rollout runs in later GPU exact wavefronts (`PENDING_EXACT` until done), and the model may continue with other configurations.

## Multi-agent work: skeleton + publications

Large designs split across agents the way CATIA top-down design splits across engineers:
- A **skeleton part** holds the master geometry (layout planes, axes, interface surfaces, envelopes) and exposes it through `document.publication`.
- Each agent owns a part or product branch that references **only publications**. Publications are the contracts between agents.
- A change to a publication bumps its revision. Dependent agents receive `STALE_REFERENCE` with lineage and re-resolve; there are no feature-level merges across agents.
- The harness enforces the boundaries: an agent cannot write outside its owned branch. Clash and clearance requirements across agents are evaluated on the assembled product on the GPU.

## Cross-path equivalence

- A CuModel rollout's trace converts to AgIR JSON, and an LLM program compiles to bytecode. `ReplayModel` can replay either through the other path.
- Conformance replays a sample of traces across both paths and checks equal verdicts and equal geometry within tolerance.
- The Demo validated this for its fixed template: CuModel with 21 programs and ReplayModel with 22 programs ran through the same executor. AgCAD extends it to the full feature table.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-07-001 | Switching `ModelAdaptor` implementations MUST NOT change requirements, op semantics or report definitions. |
| AG-07-002 | Missing capabilities MUST return `UNSUPPORTED_OP`. |
| AG-07-003 | Reports MUST present untrusted strings only as quoted data (see AG-17-003). |
| AG-07-004 | Every device-visible tensor MUST publish a DLPack descriptor. |
| AG-07-005 | Candle-vs-kernel inference parity MUST be checked for every CuModel checkpoint. |

## LLM ⇄ GPU-model loop (implemented: `crates/dops`, MCP `set_goal` / `device_rollout`)

1. **Observe:** the LLM observes the design with `describe` (holes, walls, planes, parameters) and `render`.
2. **Goal:** it states a goal with `set_goal`:
   - a target volume interval;
   - extent limits;
   - which parameters may vary, with their ranges and the expected sign of their effect;
   - optionally, where new holes may be added.
3. **Rollout:** `device_rollout` raises the document into a device design and replicates it over C configurations. The device policy (mock GPU model today; a learned CuModel next) emits command records each round. The GPU harness validates, applies, evaluates (bounded plus sampled evidence), decides and meters them inside one graph WHILE node.
4. **Return:** the best survivor comes back as Part Design features on a new branch `device-N`. Its exact volume is checked against the device bounds, so the LLM can render, describe, check, export, or set a new goal.
