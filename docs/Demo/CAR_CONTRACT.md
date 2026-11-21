# GPU surface toolkit and streamlined car demo

Date: 2026-11-19. Source: current user requests in this chat.

## Scope and authorization
- New demo files only. Preserve existing CAD documents.
- User narrowed the MVP to 3–5 primitives and STEP delivery; native SLDPRT is no
  longer an acceptance requirement. Surface architecture follows NURBS/CATIA
  concepts; no dependence on SolidWorks loft capabilities.
- Four command primitives: NURBS_PATCH, MIRROR_Y, REVOLVE_Y, TESSELLATE.
- Train a 769-parameter policy with Rust Candle. CUDA inference emits commands
  and parameters into device memory; geometry kernels consume them directly.
- A CPU CAD kernel constructs B-rep topology and STEP from the exact GPU poles.
- Do not label tessellated STL as the CAD deliverable.

## Concept assumptions (reversible, not user-specified hard requirements)
- Vehicle longitudinal X; lateral Y; vertical Z; symmetry about Y=0.
- Nominal length 4600 mm; body width about 1900 mm; overall concept height about
  1350 mm; ground Z=0. Four wheel positions use a nominal 2900 mm wheelbase.
- Styling model, no material/process/structural/CFD targets; no mating interfaces.
- No claim of automotive Class-A surface certification, G2 at all joins, road
  legality, manufacturing readiness or aerodynamic performance.

## Outputs
- CUDA surface library, composable example, Rust CLI and reproducible handoff script.
- GPU mesh and interactive preview.
- car.step with smooth B-rep, control-net JSON and trained weights.
- Independent CAD evaluation, validity, solid count and STEP roundtrip checks.

## Acceptance
- Actual GPU execution, with no CPU geometry-generation fallback.
- Independent analytic tests for basic surface operators.
- CAD validity and expected solid count: one car body plus four tyre solids, or
  just the body if the neural policy emits NOOP for REVOLVE_Y.
- Geometry profiles and output files retained for review and future edits.
- Inspect seams and surface quality; report sampled grades and unsampled limits.
- Save STEP; reopen and compare validity, body count and volume.

## Plan review status
The user authorized this demo and requested composable curve/surface support.
These proportions are implementation assumptions; no separate styling approval or
production design review has been claimed.
