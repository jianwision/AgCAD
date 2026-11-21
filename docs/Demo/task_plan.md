# Rust + CUDA implicit modeling demo

Date: 2026-11-19 (Asia/Shanghai)

## Goal
Build and run a Rust host that runtime-compiles editable CUDA geometry code,
generates a sphere with a through-hole on the GPU, extracts a mesh on the GPU,
and exports an STL and GPU-rendered preview. Verify geometry independently.

## Phases
1. Inspect local tools and verify API/platform requirements: complete
2. Implement Rust host, CUDA kernels, CLI, preview and documentation: complete
3. Build and run on the actual NVIDIA GPU; repair failures: complete
4. Verify original geometry, mesh, parameter changes and viewer: complete
5. Implement four neural command primitives and bounded GPU execution: complete
6. Train a small policy with Rust Candle; run GPU inference driving car geometry: complete (CPU and CUDA training both verified)
7. Export exact GPU NURBS as STEP using CAD topology kernel: complete
8. Verify variants, mesh, STEP roundtrip and document surface quality limits: complete
9. Implement user-confirmed automotive surface primitive baseline: complete (46 scoped operators, parameterized API)
10. Validate every baseline operator, compositions and STEP roundtrip: complete (46/46, four STEP roundtrips, 10 unit tests, lint and browser checks)
11. Expand reusable curve construction/analysis and exact solid primitives: complete (12 curve operations + 10 solid types, 68/68 and six STEP roundtrips passed)
12. Regress all 68 scoped items, STEP roundtrips, previews and delivery package: complete (consolidated into v4)
13. Add bend/sweep/loft/subdivision/resampling operations from latest steering: complete (9 items; 77 total)
14. Final 77-item regression, STEP delivery, API docs and package v3: complete (consolidated into v4)
15. Add revolve/boolean/delete/hole-fill/smoothing requested during v3: complete (8 items; 85/85)
16. Fill the missing capability families explicitly authorized by the user: complete (44 new, 129/129; four additional STEP roundtrips)
17. Validate all additions, composable selection/history and final expanded delivery: complete (134/134, 16 STEP roundtrips, composed panel, API/docs, browser and verified v4 package)
18. Add positioning/attachment operations from latest steering: complete (5 items, 134 total; curve/mesh/surface reference tests and instance STEP passed)

## Decisions
- Isolate files here; preserve existing research and root planning documents.
- Rust host with cudarc; compile editable CUDA source through NVRTC at runtime.
- Keep field evaluation and triangle generation on the GPU; read back outputs for validation and file export.
- Use millimetres by convention. STL itself does not encode units.
- No LLM API integration is required: the editable model function is the LLM-generated-code boundary.
- Native Windows demo uses ordinary export I/O. GDS is unavailable on Windows.

## Errors
| Error | Resolution |
|---|---|
| Initial cargo fetch ran before main.rs existed | Added the executable target before retrying |
| crates.io connection denied inside sandbox | Approved download with require_escalated; project-local Cargo cache populated |
| apply_patch rejected delete+add of the same file in one patch | Split into sequential delete and add patches |
| Rust auto-selected an incomplete VS 18 linker environment; msvcrt.lib unavailable | Added run.ps1 to initialize installed VS 2022 C++ environment and select its linker |
| Dynamic NVRTC lookup failed even with CUDA bin in PATH; absolute-path ctypes load succeeded | Preload the installed NVRTC DLL by absolute CUDA_PATH and retain its library handle |
| NVRTC then reported its builtins DLL missing | Preload installed nvrtc-builtins DLLs by absolute path as well |
| Documentation patch used a println line reformatted by cargo fmt | Reapplied docs independently and used a literal label replacement |
| Ellipsoid volume check differed by 0.044% with default fixed CAD integration | Switched to adaptive volume integration; independently checked CAD surface against ellipsoid equation to 1e-12 |
| STEP free-curve writer omitted a native offset curve (9 inputs became 8 edges) | Explicit bounded spline conversion; retain native definition and verify off-grid approximation plus 9-edge reimport |

## Additional user direction
- Reduce device-host trips: use one stream and one packed download, with GPU-side
  status reduction and a small set of reference probes instead of downloading the full field.
- Keep GDS out of the Windows implementation; document supported-platform extension.
- User expanded demo to complex streamlined vehicle surfaces, required SolidWorks
  delivery, and requested reusable/composable basic surface modeling support.
- See CAR_CONTRACT.md. Reversible concept dimensions are assumptions, not a production vehicle specification.
- Latest scope: small model drives GPU primitives; 3–5 primitives for MVP; STEP is
  sufficient delivery. Native SLDPRT is no longer required.
- User then requested all primitives; confirmed scope through async answer:
  automotive surfaces: curves, loft/sweep/fill, G1/G2, trimming/intersection/sewing.
  Expand and verify this concrete baseline; never imply all CATIA functionality.

## Baseline implementation plan (latest user direction)
- 46 operator acceptance items in SURFACE_BASELINE.md: curves 8, surfaces 11,
  continuity/curvature 4, CAD topology 6, polygon 13, transforms 4.
- Implement generic GPU curve/surface helpers and batch fixtures/control nets.
- Implement bounded indexed polygon operators; one packed GPU download.
- Add Rust primitive_suite binary, independent Python/OCP references, gallery STEP.
- Test each operator and compositions, including negative/degenerate inputs.
- Preserve the working four-slot neural MVP; new primitives are library support,
  not falsely presented as already learned by its unchanged model head.

## Extension v2 (user requested more curves and basic bodies)
- Add Hermite, parabola, hyperbola, helix, Archimedean spiral, three-point arc,
  natural interpolating spline, reversal, arc length, uniform-distance sampling,
  bounded nearest-point search and planar offset.
- Add exact B-rep sphere, ellipsoid, cylinder, cone, frustum, torus, capsule,
  regular prism, pyramid and wedge; GPU creates definitions and preview meshes,
  CPU CAD kernel assembles valid closed solids and exports STEP.
- Extend the existing GPU batch arena and final readback, rather than introducing
  a host roundtrip between individual operations. Preserve the four-slot model.
- Helix/spiral require approximate STEP spline curves with measured error;
  preserve analytic descriptors, do not claim exact rational representation.

## Operation extension v3 (new steering during v2 delivery)
- Add bend, twist, taper point/mesh deformations with explicit coordinate frames.
- Add GPU rotation-minimizing-frame path sweep and chord-parameterized natural
  cubic compatible-section loft; preserve CPU general CAD topology boundary.
- Add closed-manifold quad Catmull-Clark and triangle Loop subdivision.
- Add bilinear surface-grid resampling and uniform-distance polyline resampling.
- Integrate into the same GPU arena/readback. Export operation surfaces as native
  GPU control nets where available and deformed/smoothed meshes as faceted B-rep.
- Finish v2 docs/package together with v3; do not drop the successful 68-item work.

## Topology/smoothing extension (latest steering)
- Full positive-radius closed-profile revolution, GPU mesh + exact CAD revolve.
- CPU robust Boolean union, cut and common, with analytic volume references.
- GPU selected triangle deletion and planar boundary-loop filling; additionally
  native CAD shell-face deletion, planar capping and sewing roundtrip composition.
- GPU Laplacian and Taubin smoothing, independent adjacency/matrix reference;
  topology preserved, volume preservation is not guaranteed.
- Consolidate all 85 items, docs and source/STEP package in this active task.

## User-authorized missing capability families (latest instruction)
- Curve editing: knot insertion/removal, degree elevation/reduction, exact split,
  reparameterization, fitting and smoothing.
- Advanced sweep/loft: guides, changing sections/scales, closed-path frame
  correction and compatibility matching.
- Surface edits: extension/untrim/replace, automatic G1/G2 boundary matching,
  multi-patch fairing and quality diagnostics.
- Solid features: fillets/chamfers, thickening/shell, draft and ribs.
- Mesh edits: decimation, adaptive remeshing, edge flip/bridge, region extrusion,
  crease-aware subdivision.
- Repair: self-intersection and nonmanifold detection/repair, tolerance unification,
  small-edge/sliver-face cleanup; bounded repair must report unsupported cases.
- Orchestration: stable typed object handles and point/edge/face selections,
  dependency DAG, undo/redo and reproducible rebuilding.
- Implement/test concrete bounded variants for each family, never present a stub
  or metadata-only entry as accepted. Keep four-slot policy limits explicit until
  its action space/training is actually expanded.

## GPU constraint architecture extension (2026-11-20)
19. Implement typed affine constraint graph, tolerance normalization, hard-equality elimination and rank/conflict diagnostics: complete
20. Implement batched CUDA residual/JVP/VJP and nullspace PCG in a captured execution graph, one final readback: complete
21. Validate derivatives, CPU independent solves, hard constraints, negative cases, multi-patch seams and native STEP roundtrip: complete
22. Document bounded scope, integrate reproducible command, viewer and versioned delivery: complete

Scope: linear constraints on fixed-topology polynomial Bezier patches and point variables. Compatible C0/C1/C2 seams imply respective geometric continuity, but arbitrary reparameterized G1/G2, nonlinear dimensional constraints and collision inequalities remain future work. CPU graph compilation/SVD and final CAD topology are explicit. No acceleration claim without a matched baseline.

## Initial GPU harness (2026-11-20)
23. Define shared-template/command/state ABI, cached affine solve and immutable goals: complete
24. Implement CUDA candidate transactions, independent checks, revision/budget control and traces: complete
25. Validate 32 candidates, rejection/rollback paths, cache sharing and replay; export selected STEP: complete
26. Integrate command, documentation, interactive trace viewer and versioned package: complete

Initial harness uses deterministic device control and cached exact affine least-squares maps for the existing fixed-topology hood. Learned policy integration and nonlinear/topology-changing operations remain explicit future work. Do not add pointless learned iterations to the fixed linear problem.

## Recursive improvement planning (2026-11-20)
27. Audit implemented harness/model boundaries and identify measurable next tasks: complete
28. Design model/data/curriculum and harness improvement loops with independent promotion gates: complete
29. Write versioned roadmap, experiment budgets, acceptance criteria and first implementation slice: complete

This request authorizes planning artifacts. Training runs, background recurrence, runtime changes and promotion are future planned work, not part of this turn.

## Deeper recursive-improvement review (2026-11-20)
30. Challenge representation, observability, teacher ceiling, evaluator independence and economics: complete
31. Audit basis-coordinate stability and reachable objective floor on current data: complete
32. Revise roadmap around diagnostic interventions, transferable skills and calibrated promotion: complete

Scope remains design and diagnostic calculations, with runtime/training unchanged.

## Execute diagnostic foundation (user authorized 2026-11-20)
33. Implement executable D0-D4 protocol, fixtures and independent references: complete
34. Run Rust/CUDA diagnostics, coordinate/rank/reachability/observation and baseline checks: complete
35. Find and minimize verifier disagreements, repair scoped checks and re-audit: complete
36. Integrate reproducible command, reports, STEP validation and delivery documentation: complete

Scope: execute the v2 diagnostic foundation before freezing a feedback ABI or starting recursive promotion. Preserve v6 artifacts. Report unavailable or unproven model comparisons explicitly.

User clarification: harness enforces user/geometry/resource constraints; maximize model autonomy. Add an open proposal-program adapter and an actual small GPU proposal model. Feasible score-worsening exploration is permitted, best incumbent kept separately; STOP never implies task success. The model may use richer observations than its pilot adapter. Synthetic proposal training is authorized as part of this execution, not a recursive promotion claim.

37. Rewrite review around ModelAdaptor with concrete Llm/CuModel, scope and execution evidence: complete
38. Verify common protocol, model-context integrity and switching to ReplayModel on the same GPU executor: complete

Diagnostic acceptance remains a bounded hood pilot: CuModel has a learned six-parameter ADD head. Persistent feedback sessions and private drafts are implemented in phases 39–42 below; live external LLM and general primitive DAG integration remain unverified/future. No automatic promotion or general CAD learning gain claimed.

## Persistent model-driven execution (2026-11-20)
39. Implement persistent Rust/CUDA worker, versioned object handles and constraint-only transactions: complete
40. Implement shared model decisions, rich observations, feedback sessions and Llm/CuModel switching: complete
41. Validate repairable drafts, rollback, branches, budgets, stale handles and device-only rollouts: complete (12 session tests, six legacy adapter tests, 14 Rust test executions and Clippy)
42. Integrate reproducible session demo, independent STEP checks and current design documentation: complete (repair/switch and changed-target CuModel runs passed; current source hashes, 27 Markdown links and v6 preservation verified; reproducible package script added)

User authorizes best-judgment architecture rather than literal adherence to earlier proposals. Unify model intent/feedback while retaining host program and device producer execution paths. Existing five-operation diagnostic interface stays compatible. Current geometry backend is explicitly the constrained hood family, with capability discovery and unsupported results instead of claiming all library primitives are integrated.

## Class-A, design intent and topology extension (2026-11-20)
43. Audit existing capabilities and define reusable, independently verifiable service contracts: complete
44. Implement higher-degree Bezier jets, controlled G1/G2/G3 matching, GPU quality analysis and bounded fairing: complete
45. Implement unit-aware parameter/formula graph, nonlinear geometric constraints, solving and conflict/DOF diagnostics: complete
46. Implement tolerance-budgeted trimming, splitting, Boolean, sewing/healing and topology history/inspection: complete
47. Expose new services through ModelAdaptor tool programs and a document runtime; demonstrate combined automotive surface delivery: complete (initial combined STEP passed; final declared quality checks added)
48. Run positive/negative/independent GPU and CAD checks, regress session interfaces, document scope and deliver STEP/source package: complete (20 engineering cases, 12 session cases, six adapter cases, 15 Rust test executions and Clippy; six STEP files reopened; 475-file package verified)

Scope: usable support for the three requested domains, with explicit bounded mathematics and CPU robust-topology boundary. Class-A analysis and compatible reparameterization matching do not constitute universal Class-A certification. Constraint solving returns independently assessed candidate feasibility, not global infeasibility claims. Tolerance policies cannot silently expand to make an operation pass. Preserve prior session and v6 delivery artifacts.

## Resident engineering GPU and multi-surface improvement (user selects 1 and 3)
49. Define resident surface ABI, immutable device handles, metrics and NURBS/network scope: complete
50. Implement persistent Rust/CUDA surface service, reusable allocations/compilation and summary-only observations: complete
51. Implement rational B-spline/NURBS jets, shape-preserving knot insertion and adaptive quality checks: complete
52. Implement coupled multi-patch optimization with hard seam/lock constraints and geometric quality objectives: complete
53. Integrate model tools and combined STEP demo; validate independent references, residency, negatives and regressions: complete (18 resident cases, 20 engineering regressions, 16 Rust test executions and Clippy passed; current-source G3 demo, native references and benchmark verified)
54. Document measured performance/scope and deliver a new verified package without replacing previous archives: complete (560-file / 17,193,445-byte ZIP; CRC and all manifest SHA-256 values passed; three resident STEP files reopened, 58 documentation links checked and all prior ZIPs preserved)

Selected scope is resident execution and higher-quality multi-surface construction. Feature dependency graphs, broader topology repair and new model training are separate work. CPU may compile constraint/nullspace structures and perform robust STEP topology; GPU executes repeated surface evaluation and candidate optimization. Preserve explicit sampled/unknown quality status and do not equate compatible seam jets with global Class-A certification.
