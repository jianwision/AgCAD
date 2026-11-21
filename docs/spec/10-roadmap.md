# 10 — Roadmap

> Status: **Normative** · Spec v0.3.1 · Area: program

The Demo (`docs/Demo`) already validated the risky *control* mechanisms on real hardware:
- runtime-compiled model-written GPU code
- the device-side harness (propose → check → commit/rollback per configuration, replay bit-compare, single readback)
- persistent GPU sessions with drafts and branches
- NURBS derivatives and G1–G3 tooling
- GPU SSI seeding

Those are **ported**. New risk is concentrated in four places:
1. **The GPU-native exact kernel** (SSI marching, topology builders, fillets, sewing). The Demo did this part on the CPU.
2. The thesis against the CPU-script baseline ([00](00-vision.md#thesis-and-kill-criteria)).
3. CubeCL/wgpu portability.
4. The open feature table.

**Infrastructure prerequisite:** the local dev box has only an Intel iGPU, so it runs the wgpu tier only. M0 requires an **NVIDIA CI runner from week one** (the Demo's RTX machine or a cloud GPU, sm_75+) for the CUDA tier, NVRTC, conditional graph nodes, the CAD fuzzer and performance gates. No CUDA-tier change merges without a green NVIDIA run.

**Rules:**
- Build a **walking skeleton first**: every layer exists thinly before any layer is deep.
- **The spec is frozen at v0.3.** Changes come from what implementation teaches, recorded as spec diffs.
- **Headless MVP = M0 + M1**, in pure Rust: library, `agcad` CLI and MCP server. Rendering is offscreen only.
- **GPU-only data plane from day one.** The host-interaction budget ([05](05-execution.md#host-interaction-budget)) is a CI gate from M0.

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0 — Walking skeleton** (≈3–4 weeks) | Driver/runtime workspace layout ([01](01-architecture.md#project-layout-driver--runtime-split-cuda-style)); `agcad-driver` core + extension traits with `CudaDevice` (port the Demo NVRTC/cudarc path) and `WgpuDevice` (Intel iGPU dev box); `AgStatus` + bytecode ABI structs; AgIR JSON + bytecode codecs; `agcad-rtc` for expression UDFs (strict Rust subset); device VM; `agcad-solver` batched LM (small dense) + `interval`; **one thin path through every layer**: `sketch.circle` → `part.pad` (analytic, exact on GPU) + `part.implicit_body` (gyroid) → `study.configurations` (4,096) → GPU harness wavefront (validate/bin/execute/check/decide/rank in a WHILE node) → GPU STEP encoder (planes/cylinders) → replay. Spikes: f64 vs df64 per device, conditional nodes, NVRTC + nvJitLink latency, CubeCL CUDA interop with CUB/cuBLAS | The skeleton runs end-to-end on CUDA and wgpu; host geometry bytes = 0; replay bit-identical; spec checks in CI; **baseline harness** (LLM + CadQuery/OCCT) runs the same toy task |
| **M1 — Headless MVP** | `document` (part, body, geometrical set, edit/update/save_version/branch); `knowledge.parameter`/`formula`/`design_table`; full `sketch` + `constraint` (incl. `inequality`, `analyze`) on `agcad-solver`; `part.pad`/`pocket`/`shaft`/`groove`/`primitive`/`implicit_body`; `boolean.smooth_boolean`; wireframe/surface NURBS + derivatives (port Demo); `mesh.tessellate`; `convert.implicit_to_mesh`; core `evaluate.*` with escalation chains; `study.*` minus optimize; GPU harness v1 with requirements, process-driven delivery for `additive` (3MF/VDB) and analytic `machining` parts; **diagnostics**: name-annotated renders, causal failure reports; MCP server + CLI + Rust API; trace + `ReplayModel`; topology classes and proxy-fidelity metrics | An LLM, through MCP only, builds a parametric flange and a gyroid-filled block from requirements; a 4,096-configuration study in one GPU pass; STEP and 3MF verified (OCCT reimport/probes); **first baseline comparison published** with LLM compile-error and repair KPIs |
| **M2 — GPU exact kernel I + naming** | Robust predicates; GPU SSI seeding + **marching**; **topology builders**; booleans; `part.hole`, `transform.*` patterns; stable naming + signature recovery; `query.*`; `document.publication`; `evaluate.check`/`clash`/DFM machining set; STEP import (GPU lexer); **reconstruction tasks** online; **L1 session skills + L2 library growth** | Prismatic family passes; **OCCT parity ≥ 99 % on the boolean corpus**; naming stability KPI baseline; **kill/redirect review against the baseline** ([00](00-vision.md#thesis-and-kill-criteria)) |
| **M3 — GPU exact kernel II, procedural, conversion** | Fillets, chamfer, shell, draft, thickness, rib/slot/multi-sections; GSD surfaces (sweep, multi-sections, blend, fill, join, split, trim); GPU sewing/healing; `proc.*` networks; `part.lattice`; `convert.implicit_to_solid`/`to_solid`/`mesh_to_solid`; NanoVDB bricks; OptiX shim; DFM molding + additive sets; `file.export_usd`; **L4 starts**: model-proposed fixes for exact-kernel ops below 100 % OCCT parity | Sheet-like, lattice and procedural families pass; fillet/shell OCCT parity ≥ 99 %; converters meet declared deviation |
| **M4 — Agentic GPU at scale** | `study.optimize`/`sensitivity` (autodiff through features), `study.iterate`, `study.observe`; device command ring + **CuModel** (Candle CUDA training, kernel inference); multi-GPU `DeviceSet`; `udf.publish` gate; archive; multi-agent skeleton/publication workflow | Throughput targets ([05](05-execution.md)); CuModel and Llm both solve identical tasks; published model UDFs pass transfer tests; L3 first CuModel promotion |
| **M5 — CATIA-class surfacing & assembly** | Class-A tooling (match, fair, connect checker, highlight lines; port Demo G2/G3), direct edit, defeature, threads, IGES; products + assembly constraints on `agcad-solver`; clash in products | Surfacing and assembly families pass; G2/G3 evidence reported with scope |
| **M6 — Self-improvement at full depth** | L3 at scale (expert iteration + DAgger), L4 for kernels/converters/GPU-harness checks, L5 meta (human-approved); held-out rotation; canary + rollback; multi-tenant server (MPS/MIG/green contexts); optional web viewer | At least one promoted improvement per level L2–L5 with gate reports; no held-out leakage; rollback drill passes |

## Capability tracks (scheduled into milestones)

| Track | Ops | Milestone |
|---|---|---|
| Human review workflow | `document.comment`, `document.approve`, `document.diff`; HTML diff reports from the trace | M1 |
| CAD fuzzer + counterexample corpus | `agcad-conformance` ([09](09-validation.md#cad-fuzzer)) | M2 (nightly) |
| Document format + migrations | `.agcad`, `file.save`/`open`, `MIGRATION_FAILED` ([16](16-document-format.md)) | M1 (format), M2 (migrations) |
| Statistical tolerance analysis | `knowledge.tolerance`, `study.tolerance_analysis` | M3 |
| Physics evaluations + topology optimization | `evaluate.stiffness`, `evaluate.modal`, `study.topology_optimization` | M4 |
| Standard-part catalogs + shape retrieval | `knowledge.catalog_insert`, `query.similar`, geometry encoder | M4 |
| History-preserving handoff | `file.export_feature_script` (CATIA EKL/VBA, SolidWorks API, FeatureScript) | M5 |
| Security hardening | threat model ([17](17-security.md)) | M1 (baseline), M6 (multi-tenant) |

## Normative requirements

| ID | Requirement |
|---|---|
| AG-10-001 | CUDA-tier changes MUST NOT merge without a green NVIDIA CI run. |
| AG-10-002 | The kill/redirect review MUST take place at the M2 exit. |
| AG-10-003 | Exact ops below their OCCT parity threshold MUST remain `experimental`. |
