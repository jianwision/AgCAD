# 09 — Validation

> Status: **Informative** · Spec v0.3.1 · Area: conformance

## Oracles

| Oracle | Used for |
|---|---|
| Analytic formulas | Primitive volumes/areas, curve lengths, curvature of analytic surfaces |
| `CpuDevice` f64 reference | Every GPU kernel, including GPU harness checks (bit-equal verdicts) |
| Exact host checks (`agcad-geo`) | B-rep validity, closure, orientation, tolerances |
| **OCCT** (test-only, via `cadquery-ocp`) | Independent validity, booleans, STEP export→reimport, distance probes |
| Commercial import (optional, manual) | STEP opened in CATIA/SolidWorks where licences exist; recorded per release |

The Demo already established this oracle style: OCCT reimport, off-grid probes and independent KKT references. It also established the rule that sampled evidence is never presented as a continuous-domain proof.

## Conformance suite (`agcad-conformance`)

Every feature-table entry and every published UDF must have:
1. **Golden cases** with expected geometry/measurements and tolerances.
2. **Expected rejections:** invalid inputs and the exact error code.
3. **Lightweight-vs-exact bound:** the proxy result lies within the declared bound of the exact one.
4. **Backend parity:** CUDA, wgpu and CPU agree within the op's tolerance, with an identical verdict for checks.
5. **Determinism:** run twice on the same device, bit-compare.
6. **Naming stability:** parameter perturbations that keep topology keep names; topology changes produce correct lineage.
7. **Differentiability** (if declared): finite-difference agreement of sensitivities away from discontinuities.
8. **Attribute propagation** (proc/mesh): declared rules hold.

For L4 implementation swaps, the suite is extended with differential, metamorphic and adversarial fuzzing against the frozen reference, plus the permanent counterexample corpus ([11](11-self-improvement.md)).

## CAD fuzzer

`agcad-conformance` includes a **generator of random valid AgIR programs**: random feature trees, degenerate placements (coplanar, coaxial, tangent, near-zero edges), near-limit fillets and shells, and random design tables.
- Every program runs on the GPU kernel and on the OCCT oracle.
- Disagreements are **shrunk** to minimal reproducers (delta debugging over the AgIR) and added to the permanent counterexample corpus.
- The fuzzer runs nightly on NVIDIA CI, and its corpus is a gate for every exact-kernel change.
- It also produces training and red-team data for [11](11-self-improvement.md).

## Baseline

Every benchmark family also runs against the **baseline**: the same LLM writing CadQuery/build123d scripts against OCCT on CPU, with the same requirements, budget and checks. The result feeds the kill criteria in [00](00-vision.md#thesis-and-kill-criteria).

## Reconstruction tasks (self-verifying, from M2)

These tasks are mined from public CAD datasets (ABC, Fusion 360 Gallery, DeepCAD, Text2CAD), subject to their licences. A task is "rebuild this part within δ mm", verified automatically by `evaluate.distance_analysis` against ground truth. It needs no human labels, which provides thousands of tasks with dense rewards for self-improvement ([11](11-self-improvement.md)). Held-out splits are by source dataset and design family.

## Benchmarks (agentic)

| Family | Examples | Required body type | Key requirements |
|---|---|---|---|
| Prismatic parts | brackets, flanges, housings, shafts with keyways | Solid | dimensions, holes, thickness, STEP |
| Sheet-like / shelled | enclosures, covers with bosses and ribs | Solid | wall thickness, draft, fillets |
| Class-A-style surfacing | hood, fender, handle patches | Surface | G2/G3 connect checker, highlight lines |
| Lattice / implicit | graded gyroid infill, conformal lattices | Implicit → Solid / Mesh | min ligament, porosity, mass |
| Procedural | façade panels, perforation patterns, cable trays, generative brackets | Mesh / Solid | counts, spacing, attribute-driven variation |
| Assemblies | hinge, gearbox housing + cover | Product | clash-free, constraint DOF |
| Repair / edit | import STEP, defeature, modify design intent | Solid | deviation bounds, naming lineage |

Reported per run:
- success: stop / execution / requirements / delivery are four separate fields
- cost: GPU-ms, host-ms, host geometry bytes (target 0), exact rebuilds, LLM tokens, round trips
- configurations explored
- evidence levels achieved

There are LLM-path vs GPU-model-path comparisons under identical requirements.

**System KPIs** (tracked per release):
- LLM compile-error rate and reference-error rate
- repair iterations per task
- proxy fidelity (τ, survivor failure rate)
- topology-class fragmentation
- naming stability under perturbation
- host-interaction budget
- exact-kernel OCCT parity rate Held-out families follow the [11](11-self-improvement.md) rules.
