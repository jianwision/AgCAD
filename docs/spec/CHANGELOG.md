# Changelog

> Status: **Informative** · Spec v0.3.1 · Area: all

## v0.3.1 — 2026-11-20

- **Standardization:**
  - [CONVENTIONS](CONVENTIONS.md): RFC 2119 keywords, document classes and status lines, requirement IDs `AG-<doc>-<nnn>`, naming, units, versioning and deprecation, change control.
  - [GLOSSARY](GLOSSARY.md).
  - Normative requirement tables in every normative document.
  - Checker enforcement of all of the above.
- **New documents:**
  - [18 STEP interchange, receiver profiles & manufacturing outputs](18-interchange-manufacturing.md): GPU STEP import/export pipeline with a device-memory sizing model and chunked streaming; AP203/214/242 coverage; `generic`/`solidworks`/`catia` receiver profiles (SolidWorks profile `experimental` until verified on a SolidWorks CI runner); validation properties; 3MF (Materials, Beam Lattice, Slice), slicing, PMI for machining
  - [16 Document format, versioning & migration](16-document-format.md)
  - [17 Security & threat model](17-security.md), including prompt injection through geometry metadata
- **Execution:** memory capacity planning (estimate, chunk, evict) with status `CHUNKED` (`0x0005`).
- **Geometry:** analytic-first intersection (co-surface detection plus a closed-form table for analytic surface pairs); NURBS marching becomes the fallback.
- **Validation:** CAD fuzzer with shrinking against OCCT.
- **Models:** shared geometry encoder (UV-graph + SDF) for CuModels and retrieval.
- **Roadmap:** NVIDIA CI runner prerequisite; capability tracks.
- **Ops added:** 13, for 293 in total. `file.export_step` and `file.import_step` gained schema/profile/PMI options and reports; new `file.export_3mf` and `file.export_slices`; plus:
  - `knowledge.tolerance`, `knowledge.catalog_insert`
  - `study.tolerance_analysis`, `study.topology_optimization`
  - `evaluate.stiffness`, `evaluate.modal`
  - `query.similar`
  - `file.export_feature_script`
  - `document.comment`, `document.approve`, `document.diff`
- **Status codes:** `MIGRATION_FAILED` (`0x0103`), `CHUNKED` (`0x0005`).
- **Tooling fix:** `check_spec.py` exited before running its later checks, so the host-backend guard and the conventions checks were never enforced. Fixed; it also caught two alias conflicts (`boundary surface`, `measure`), now resolved.

## v0.3 — 2026-11-20

- **GPU-only data plane:** the exact kernel moves to the GPU; the host becomes a control plane with a CI-gated host-interaction budget.
- **New library and rename:** `agcad-solver`; `agcad-geom` → `agcad-geo`.
- **Unified model:** constraints and evaluations ([14](14-constraints-evaluations.md)); process-driven delivery and DFM.
- **Thesis and kill criteria:** baseline comparison against an LLM writing CadQuery/OCCT scripts.
- **Execution:** topology classes; proxy fidelity.
- **Naming:** recovery by geometric signature.
- **LLM support:** diagnostics; AgLang as a strict real-Rust subset.
- **Validation and self-improvement:** reconstruction tasks; anti-gaming rules; earlier start for self-improvement.
- **Roadmap:** re-cut around the walking skeleton; [15 Golden trace](15-golden-trace.md).

## v0.2 — 2026-11-20

- **Ontology and capability:**
  - CATIA canonical ontology with SolidWorks/Onshape/Houdini aliases.
  - `proc` family (Houdini-style procedural networks).
  - Code-first UDFs.
  - Body types beyond B-rep.
- **Harness:** GPU-resident harness; recursive self-improvement ([11](11-self-improvement.md)).
- **Stack:** pure Rust; Candle-trained CuModel; headless MVP.
- **Reviews:**
  - NVIDIA/CUDA review ([12](12-cuda-review.md)): FP64 policy, wavefront harness, fault isolation, OptiX/NanoVDB/cuBLAS/CUB.
  - API review ([13](13-api-review.md)): driver/runtime layering, extension traits, `AgStatus`, batch layouts, DLPack.

## v0.1 — 2026-11-20

- Initial architecture spec and op catalog.
