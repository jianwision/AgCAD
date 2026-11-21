# AgCAD Architecture Specification

> Status: **Informative** · Spec v0.3.1 · Area: index

AgCAD is an **agentic, GPU-native, CATIA-class CAD system with Houdini-style procedural modeling**, written in **pure Rust**.
- **Ontology:** traditional CAD (CATIA canonical names, with SolidWorks/Onshape/Houdini aliases).
- **Capability:** open. Every feature is code, and models write and publish their own UDFs.
- **Callers:** LLM agents send batch programs (AgLang / AgIR JSON over MCP). GPU-resident models, trained with **Candle**, write AgIR bytecode directly into device memory.
- **Harness:** both callers share one feature table and one harness. The harness runs **on the GPU** in the inner loop and on the host for exact delivery.
- **Self-improvement:** recursive, under frozen oracles and held-out evaluation.
- **First MVP:** headless.

Status: **design spec v0.3.1 (2026-11-20), frozen for M0. Not implemented yet.** See the [CHANGELOG](CHANGELOG.md). The spec follows [CONVENTIONS](CONVENTIONS.md) (RFC 2119 keywords, requirement IDs `AG-<doc>-<nnn>`), and terms are defined in the [GLOSSARY](GLOSSARY.md). v0.3 added:
- a GPU-only data plane, with the exact kernel on the GPU
- unified constraints and evaluations
- `agcad-solver`, plus the rename of `agcad-geom` to `agcad-geo`
- thesis and kill criteria
- topology classes and proxy fidelity
- process-driven delivery and DFM
- diagnostics for LLMs
- reconstruction tasks and anti-gaming rules
- a golden trace
The validated prototype in [`../Demo`](../Demo/README.md) is the baseline whose mechanisms the roadmap ports ([00](00-vision.md#lessons-from-docsdemo)).

| # | Document | Question it answers |
|---|---|---|
| 00 | [Vision & principles](00-vision.md) | What are we building, and why this shape? |
| 01 | [Architecture](01-architecture.md) | Layers, responsibilities, data flow, driver/runtime project layout |
| 02 | [Ontology & geometry model](02-geometry-model.md) | CAD vocabulary, body types, specification tree, exact and lightweight representations, naming |
| 03 | [Feature table](03-op-catalog.md) · [`op-catalog.json`](op-catalog.json) | 293 core + standard ops with aliases, signatures, phases, axes |
| 04 | [AgLang, AgIR & bytecode](04-agir.md) | The Rust-syntax feature language, the batch-program JSON, the device ABI |
| 05 | [Execution & parallelism](05-execution.md) | Configuration/entity/sample axes, preview-then-exact, pipelining, memory, determinism |
| 06 | [Harness](06-harness.md) | Requirements, Design Checker, transactions, **GPU-resident harness**, reports, error codes |
| 07 | [Model interfaces](07-model-interfaces.md) | `ModelAdaptor`; MCP batch API; device ring; Candle CuModel |
| 08 | [HAL & precision](08-hal-backends.md) | Two kernel tiers (CubeCL portable + NVIDIA-native libraries); per-device precision (f64/df64/FP64 emulation); Candle + HAL inference |
| 09 | [Validation](09-validation.md) | Oracles, conformance suite, agentic benchmarks |
| 10 | [Roadmap](10-roadmap.md) | M0–M6; the MVP is M0+M1, headless |
| 11 | [Recursive self-improvement](11-self-improvement.md) | L0–L5 improvement loops, archive, gates, invariants |
| 12 | [Review: NVIDIA frameworks & CUDA](12-cuda-review.md) | FP64 rates, wavefront harness, fault isolation, OptiX/NanoVDB/cuBLAS/CUB, determinism; amendments applied |
| 13 | [Review: APIs & abstractions vs CUDA conventions](13-api-review.md) | Driver/runtime layering, extension traits, async errors, stream-ordered buffers, graphs, `AgStatus` values, batch layouts, DLPack, multi-GPU |
| 14 | [Constraints & evaluations](14-constraints-evaluations.md) | One model for evaluations, driving/inequality constraints, checks, requirements, objectives; GPU pipeline; DFM; process-driven delivery |
| 15 | [Golden trace](15-golden-trace.md) | One program through every layer, with assertions and the host-interaction tally |
| 16 | [Document format, versioning & migration](16-document-format.md) | `.agcad` container, four versioned artifacts, migrations, snapshots and crash recovery |
| 17 | [Security & threat model](17-security.md) | Sandboxing, fault isolation, prompt injection through geometry metadata, tenancy, MCP auth |
| 18 | [STEP interchange & manufacturing](18-interchange-manufacturing.md) | GPU STEP load/save and memory model, AP242 coverage, SolidWorks/CATIA receiver profiles, validation properties, 3MF/slicing/PMI |
| — | [CONVENTIONS](CONVENTIONS.md) · [GLOSSARY](GLOSSARY.md) · [CHANGELOG](CHANGELOG.md) | Standardization rules, terms, version history |

## Tooling

```sh
python3 docs/spec/tools/catalog_source.py > docs/spec/op-catalog.json   # regenerate catalog
python3 docs/spec/tools/render_catalog.py                               # refresh tables in 03
python3 docs/spec/tools/check_spec.py                                   # consistency checks
```

`op-catalog.json` is normative, and the tables in 03 are generated from it. These spec-maintenance scripts are the only Python in the project. They will move to an `xtask` once the Rust workspace exists.
