# Progress

## 2026-11-19
- Read planning skill and existing plans; created an isolated demo plan.
- Verified installed Rust/CUDA/GPU/MSVC and consulted official NVRTC/cudarc docs.
- Answered GDS follow-up: unavailable on this Windows host; keep generated data
  on the GPU until final validation/export instead.
- Implementing the runnable project and a model-code replacement interface.
- Implemented one GPU-resident pipeline and one packed DtoH call, pinned memory,
  device reductions, 64 probes, CUDA event timing and capacity overflow checks.
- Fixed MSVC environment and NVRTC/builtins DLL lookup.
- First GPU run: 390936 triangles; no boundary/nonmanifold/orientation errors;
  Euler 0, volume error 0.016162%, max extent error 0.03420 mm.
- All 4 unit tests passed. Checking parameters, alternate code, viewer and lint.

- MVP complete: Rust Candle CPU and CUDA training, 769 parameters; CUDA policy
  emits four instruction slots consumed by geometry without intermediate DtoH.
- Default GPU mesh 37056 triangles; no-wheels variant 24768; both closed and
  consistently oriented. GPU rejects NaN inputs before geometry export.
- Exact STEP export and OCCT reimport pass: 5 vs 1 valid solids; GPU/CAD
  evaluator agrees within 2e-12 mm. User relaxed native SLDPRT requirement.
- CUDA training completed 4000 steps, held-out instruction accuracy 100%; final
  artifacts regenerated from CUDA-trained weights. See VALIDATION.md.
- User authorized next stage: complete automotive surface primitive baseline
  (curves, loft/sweep/fill, G1/G2, trim/intersect/sew). MVP remains reviewable.

## 2026-11-20
- Completed 46 scoped operators including 13 polygon operators, with parameterized
  CUDA construction APIs and reusable CPU CAD topology wrappers.
- Batch arena: zero intermediate DtoH, one final 5,225,232-byte download; GPU
  composition of compatible sections, G1/G2 strips and derived mesh operations.
- Independent batch acceptance passed 46/46; five invalid surface builders,
  three invalid mesh inputs and disconnected CAD curves rejected as expected.
- Fixed native splitter API mismatch and tightened filling solver settings to
  meet the existing geometric tolerance. No test was removed to hide failures.
- Perturbed weld test now preserves exact first representatives and perturbs
  duplicate vertices; verifies radius welding without assuming centroid averaging.
- Four STEP galleries exported and independently reimported with valid topology,
  matching face/solid counts and geometry; curve gallery additionally checks 11
  edges and length. Polygon STEP contains 10 valid solids and 2,491 faceted faces.
- Final main-project unit checks 10/10, Clippy -D warnings and Python compilation
  passed. Actual WebGL galleries, rotation/reset and error logs verified.
- Added PRIMITIVE_API.md and machine-readable definitions/catalog; current
  neural head remains four slots, separate from the expanded 46-operator library.
- User requested further curve/basic-body primitives. Expanding by 12 curve
  operations and 10 exact-solid types; regression target becomes 68 items.
- V2 actual acceptance complete: 68/68, six STEP roundtrips, 10 unit checks,
  Clippy and Python compilation passed. New STEP contains 10 solids /34 faces;
  extended curve STEP contains 9 edges. One final readback 8,528,408 bytes.
- In-app preview creation timed out; recovery found no tab. V2 browser QA is
  still pending and must not be claimed complete. Core geometry/STEP verified.
- User steered active work to bend/sweep/loft and subdivision/resample. Add 9
  operations and finish consolidated 77-item regression/docs/package.
- Implemented CUDA bend/twist/taper, RMF path sweep, nonuniform chordal loft,
  Catmull-Clark/Loop subdivision and grid/polyline resampling; Rust integration
  compiled and first expanded GPU run is pending (session 6031).
- Additional steering: revolve/cut/delete/hole-fill/boolean/union and smoothing.
  Adding 8 items; total acceptance target is now 85, with all earlier scope retained.
- User then explicitly authorized the missing capability families from the
  inventory: curve editing, advanced surfacing/solid features, mesh edits/repair
  and stable selection/history. Continue autonomously after completing the
  in-progress 85-item baseline; acceptance must be concrete and bounded.
- Completed 85/85 and then 44 missing-family additions: 129/129, 14 STEP
  roundtrips. Fixed mesh station density, floating comparisons, inward thickening
  orientation and adaptive two-edge split rotation. Replaced seam overwrite
  fairing with orthogonally projected bending-gradient descent; independent
  Hessian reference, fixed boundaries and C2 constraints all passed.
- Added composed_panel.py: two GPU C2 patches, sew/thicken, two Boolean holes,
  one valid solid after STEP reimport. Volume conservation residual 3.11e-11 mm3.
- User asked about snap/merge and following curves/meshes/surfaces. Adding five
  explicit bounded placement/movement operations (target 134), retaining prior
  successes. Rust unit tests passed; final lint/package/browser QA pending.
- Final placement suite passed: grid/vertex and native edge/trimmed-face snaps;
  merged/welded panels; 33 arc-length helix poses (7.62e-13 mm); 33 intrinsic
  mesh walk positions across four triangles (2.22e-16 mm); 17 NURBS UV poses
  (1.11e-16 mm). Native placed marker STEP has five valid solids.
- Final combined suite 134/134 passed, 15 primitive STEP files reimported plus
  composed_panel.step (16 total). GPU batch: 12,505,680-byte final download,
  one DtoH, zero intermediate, kernels 796.927 ms, copy 1.142 ms.
- Clippy -D warnings and Python compilation passed. Retained neural car MVP
  rerun/export passed: 5 solids/7 faces, 1200 GPU/CAD probes, max 1.811e-12 mm.
- Browser index/search/dropdown, actual motion lines, composed panel and drag
  verified. Fixed iframe/mobile layout and preview triangle count/source labels.
  Saved workbench_preview.jpg. Browser logs had two MutationObserver errors;
  application #error was empty and rendering worked; recorded in VALIDATION.md.
- Delivered output/delivery/rust-cuda-modeling-demo-v4.zip: 23,681,861 bytes,
  195 files including manifest; CRC and every SHA256 entry verified.
  ZIP SHA256 fcc41632a370840b1212c71863c74e3e6dd579488e61efa9cf402b7d1f580735.
  Planning notes are intentionally outside the distributable source package.

## 2026-11-20 GPU constraint continuation
- User authorized continuation. Building a reusable affine constraint compiler and batched GPU least-squares solver rather than adding isolated fixture operators.

- Added affine constraint graph, locks/links/coincidence/fixed planes/symmetry, line sliding, fixed-t curve and fixed-UV surface attachment, polynomial seam and fairing goals. CUDA Graph batched PCG passed 15 positive cases, 10 invalid graph checks and 4 rejected iteration limits. Full-coordinate KKT, derivative/adjoint, C0/C1/C2 and 4-face STEP roundtrip passed. Rust tests/lint and Python compileall passed.
- Corrected a potential shared-memory reduction reuse race with an explicit barrier, then reran all GPU cases. Supported fully locked (zero DOF) and zero-RHS cases explicitly. Patch application order was rejected once with no mutation; reordered and applied. Preview server had stopped; restarting local-only server for QA.

- Final validation additionally checked 196 source points against reopened STEP, max distance 6.61e-12 mm. Interactive viewer switched candidate 8 and restored candidate 6, WebGL rendered, page error empty and console errors empty. Saved preview.jpg and kept deliverable tab.
- Completed v5 documentation and package: 9124127 bytes, 227 files; CRC and every SHA256 manifest entry passed. SHA256 ec67714f9eb560f6c2579f8bcf62133ef9a41c5f8f1cb893430c7a5622143dcd. Existing v4 ZIP retained.

## Initial GPU harness implementation
- User explicitly requested implementation. Reuse hood constraints, share one compiled template across 32 candidates, cache the affine optimum map, and execute bounded transactions with GPU verification/commit/rollback and immutable goal scoring. Add device-side replay comparison and selected STEP validation.
- First GPU run passed 32 candidates × 8 rounds, 13 reason paths, independent full-coordinate KKT and immutable-goal scoring. One final 158,784-byte readback, zero intermediate readbacks; device bit replay matched. Four-patch STEP reimport and 196 distance probes passed (6.61e-12 mm max). Adding isolated fault-injection tests for geometry/numerical/invalid-baseline checks and malformed host payloads.
- Geometry, numerical and invalid-baseline GPU injections passed. Host-negative test initially rejected the correct Rust "Nonfinite" diagnostic because its whitelist omitted that word; corrected the diagnostic assertion and rerunning acceptance. No GPU rollback mismatch found.
- Full harness passed after correcting the diagnostic assertion: 32 main candidates, three dual-candidate verifier injections, ten malformed input rejections, independent scoring/KKT and STEP checks. Rust 12 test executions, Clippy -D warnings and Python compileall passed. Added HARNESS.md and offline final-control-net / per-round trace viewer; browser QA and v6 package pending.
- Browser preview initially used the wrong server root and returned 404. Directory inspection showed the existing server serves output/ directly; opened its observed harness/ directory. No source or validation failure.
- Browser rendered actual GPU final control nets and trace rows. Illegal opcode fixture exposed a viewer-only undefined operation label; added a numeric fallback and explicit inactive dash before final browser verification.
- Final integrated run passed including per-round budget assertions and viewer generation. Browser verified candidate 22 invalid opcode detail, candidate 27 budget stop and candidate 32 two commits/reset; error text empty and console errors empty. Saved browser-validation.json and preview.jpg; kept deliverable tab. Final main GPU graph 2.323232 ms; source/PTX snapshots retained. Packaging v6 next.
- Delivered v6 ZIP: 9,574,196 bytes, 268 files including manifest. ZIP CRC and every manifest SHA-256 passed. SHA256 c1307c5c53e32269554c1109c85fd6eb448663ce8dd87316657c4a7a6f5d4fac. Existing v4/v5 packages preserved; planning notes excluded. HARNESS.md documents fixed-topology affine scope and deterministic controller; neural feedback, general nonlinear constraints and topology transactions remain future work.

## Recursive improvement planning
- User asked for a plan for recursive improvement of both harness and model. Audited current v6 harness, 769-parameter open-loop model and explicit fixed-topology limitations. Preparing an implementation roadmap and measurable promotion protocol; no training or autonomous recurring run started.
- Wrote SELF_IMPROVEMENT_PLAN.md and non-executable configs/improvement-plan.v1.json: nonlinear bootstrap before recursion, 72→64→64→12 feedback model, teacher relabeling, failure/diversity replay, bounded harness variants, immutable oracle contract, 2×2 attribution, fresh qualification/final audit, resource limits and rollback. Proposed initial implementation is S0+S1, not the entire experiment loop at once.
- Plan consistency checks passed: 72 observation slots, 12 output slots, 9,612 parameters, replay ratios sum to 1, all split counts agree, 1 initialization + up to 128 subsequent geometry queries = 129 total, baseline artifact paths exist. Clarified finite-difference accounting and isolated-audit limitations. Linked plan from README; v6 archive/runtime/weights unchanged. Planning complete; no improvement loop or training started.

## Deeper plan review
- User requested deeper thinking. Reviewing whether improvement transfers beyond fixed tasks, whether SVD coordinates have stable semantics, whether the teacher/representation creates a quality ceiling, and how evaluator and resource costs affect promotion. Runtime changes and training remain outside this design continuation.
- Implemented a reproducible CPU-only assumptions audit. Existing quadratic floor equals GPU score; all 32 cases share one template/goal. Selected six physical probes with condition 4.030897; 64 gauge changes yield at most 2.8866e-14 mm step deviation. Wrote revision v2 covering causal failure routing, semantic coordinates, observability, search/teacher improvement, reusable skills, verifier repair/re-audit and calibrated economic/statistical gates. Added proposed config v2 while retaining v1.
- Revision v2 consistency passed: inherited v1 hash unchanged, 72/12 candidate feature/output counts and both architecture parameter counts agree, alpha-budget arithmetic agrees, links/artifact paths resolve, diagnostic script and NPZ source hashes match. Diagnostic Python compiles. Runtime, neural weights and v6 ZIP unchanged. Next planned implementation starts with D0–D4 diagnostic infrastructure before ABI freeze and S1 task expansion; design review complete.

## Execute diagnostic foundation
- User authorized execution. Implement D0-D4 as a reproducible diagnostic pipeline using actual Rust/CUDA evaluation and independent CPU geometry, with explicit evidence limits and preserved v6 artifacts. Reusing planning-with-files to track implementation and checks.
- User clarified that the harness is a constraint layer and should maximize LLM/model capability. Adjusting execution to include open proposal programs and a separate 462-parameter Candle/GPU proposer. Legal exploratory score increases will commit; quality feedback and best-so-far tracking are separate from admission. Preserve the old four-slot weights and v6 semantics.
- User requested unified ModelAdaptor with concrete Llm and CuModel implementations, then asked to rewrite/review the design. Rewrote SELF_IMPROVEMENT_REVIEW_V2.md around those boundaries, model autonomy, proposal contracts, validation phases, explicit transfer paths and capability-based switching. Archived preceding review for traceability.
- Initial GPU proposal training launched through cargo failed with STATUS_DLL_NOT_FOUND. Direct launch with explicit CUDA bin PATH starts successfully; changed the training wrapper to build then launch explicitly. CPU D0/D1/D2 checks passed; D4 reference exposed a NumPy bool JSON serialization error, fixed by returning a native bool.
- Explicit PATH/direct launch also exited with the same loader status; the earlier apparent start was only an asynchronous process handle. Every directly imported DLL can load in Python, but the linked Candle CUDA executable still fails at startup. Made train-proposals an explicit CPU-training action (no silent CUDA fallback); CuModel inference remains CUDA. GPU training is not claimed for the new model.
- Explicit CPU Candle training passed (462 parameters, 4000 steps, held-out normalized delta MSE 3.3193e-5). Rust 13 tests, Clippy and six ModelAdaptor boundary tests passed. Actual CUDA D0-D4 pilot passed: 14 geometry cases, 12 systems, 21 proposal programs, 16/16 neural probe goals, zero intermediate readbacks, bit-exact replay. CPU/GPU proposal parity 2.66e-15 mm; STEP 4 faces/0 solids and 324 probes passed at 6.24e-12 mm. Adding per-adaptor evidence snapshots before testing ReplayModel.
- ReplayModel switch passed on the unchanged GPU executor: 22 programs, 16 diagnostic neural reference cases, D0-D4, bit replay and STEP all passed. Preserved independent CuModel/ReplayModel snapshots. Review now distinguishes target ModelAdaptor architecture from current Python adapter and fixed-hood implementation, including live LLM not connected, CPU-trained/GPU-inferred model and unlearned opcode selection. Existing v6 archive hash remains unchanged.
- Final comparison check initially assumed POSIX source-hash keys; Windows reports store backslashes. Normalized keys in the check, then verified identical CUDA/Rust/adaptor source hashes across both runs, input/STEP snapshot hashes, review Markdown fences and all relative links. Wrote adaptor-comparison.json; design rewrite and bounded execution complete.

## Persistent model-driven execution
- User requested implementation and authorized architectural judgment. Implementing a persistent Rust/CUDA geometry worker with versioned handles, private drafts and feedback. ModelAdaptor will unify semantic decisions while retaining separate host-program and device-producer execution. No external model credentials are present in the inspected environment; implement provider injection and a usable optional transport without treating replay as live LLM inference.
- Persistent Rust worker compiled and passed an actual CUDA smoke test: invalid draft commit retained, inverse raw edit repaired it, committed-view fork stayed independent, 16 device inference rounds reached 0.056 mm probe error, and committed geometry downloaded successfully. Revisions cover draft edits as well as commits, and slot generations prevent ABA reuse. Added typed session decisions, feedback-driven ModelSession and provider-neutral HTTP transport; full acceptance running next.
- Nine protocol/persistent-GPU/HTTP-provider tests passed. Integrated proposal-parity check then caught overlapping trace columns (draft coordinates overwrote two proposed deltas); geometry and transaction tests were unaffected. Expanded the trace to 26 fields across CUDA/Rust/Python and will rerun the affected checks. Added explicit caller-owned final-goal tolerance and requirement; final delivery cannot downgrade a declared hard goal.
- Corrected trace parity and reran actual CUDA acceptance: 12 session tests passed, including full fork/select/inspect/drop flow, constraint-rejection explanations, declared-goal protection and startup process cleanup. Six legacy adapter tests and all 14 Rust test executions passed; Clippy -D warnings and Python compilation passed.
- Persistent repair/switch integration passed with required final goal: 16 GPU inferences, 0.055946 mm max probe error, proposal parity 8.88e-16 mm, and four-face STEP reimport/324-point error 5.01e-12 mm. Wrote MODEL_SESSIONS.md and updated architecture review to revision 4; completing final artifact/source consistency checks.
- Changed-target CuModel CLI passed at 0.044006 mm max error with 16 device inferences and required delivery goal. Final artifact checker verified both STEP digests, all recorded source digests, one final rollout feedback frame, 27 documentation links and unchanged v6 archive. Added reproducible source/result packaging as model-sessions-v1, retaining fixed-hood scope and no live external LLM claim.
- Delivered rust-cuda-modeling-sessions-v1.zip: 10,503,083 bytes, 383 files; ZIP CRC and all manifest SHA-256 values passed. Archive SHA-256 e1fa066bb83ad58f527d795231b3aa70553eaa65375c69fc1fb8d343b1b5a0f3. cargo fmt --check passed. All four persistent-session phases complete.

## Class-A / design intent / topology support
- User authorized all three capability families. Audited bounded matching/fairing, affine graph and CPU CAD wrappers. Extending reusable services and model tool programs rather than duplicating the existing 134 acceptance items. Keep GPU differential analysis/fairing separate from CPU nonlinear solving and robust B-rep topology, and preserve prior delivery evidence.
- Implemented degree 3–7 jets and affine-map G1/G2/G3 construction, one-readback GPU quality/fairing, unit-aware formula and nonlinear constraint graph, and tolerance-budgeted B-rep wrappers. Initial 15-case acceptance passed 14 cases; split-solid case exposed a checker scope issue: shared edges across distinct valid solids were counted as one nonmanifold shell. Changed incidence checking to each shell plus unshelled faces; rerunning the affected case before document-runtime integration.
- Split STEP legitimately duplicates shared inter-solid topology (11 unique input faces become 12); changed roundtrip validation to per-solid face counts plus volume/area/geometry, explicitly reporting topology identity changes. Affected case then passed. Full 18-case support suite passed after adding typed ToolProgram and EngineeringRuntime; atomic failure repair, branch isolation, fixed-target/tolerance protection and delivered-support binding verified. Combined two-quintic panel with polygonal opening passed G3 and STEP at 8.25e-12 mm sampled discrepancy. Added final curvature/regularity checks and a 19th acceptance case; final validation in progress.
- Completed 20/20 engineering acceptance cases, including an evaluation-budget test that counts finite-difference residual calls. Demo solve used 19 of 400 residual calls, plus one independent final constraint verification. Refreshed combined demo evidence and source hashes; G3 jet residual 7.91e-13 mm, STEP sampled discrepancy 8.25e-12 mm. GPU D3/OCP maximum difference 8.56e-12 mm. Four-patch diagnostic batch used one final readback and no intermediate readbacks.
- Regressed 12 persistent-session cases, six adapter cases, 15 Rust test executions and Clippy; final Python compileall and Rust formatting checks passed. Refreshed both saved hood demos with current shared sources, including the changed-target CuModel run. Generated and visually inspected actual reflection/curvature preview. ENGINEERING_SUPPORT.md documents API, CPU/GPU boundaries, bounded reparameterization, sampled checks, fixed contracts, atomic tools and remaining limitations.
- engineering_delivery.py verified current test/demo source digests, reopened six engineering STEP files, checked 42 Markdown links and preserved previous v6/session ZIP hashes. Delivered output/delivery/rust-cuda-engineering-support-v1.zip: 13,173,194 bytes, 475 files including manifest; SHA256 6fc04a6f56ed64e089b428f7dae948a5bb7297c02925bb37a58ab9abc4b16135. ZIP CRC and every manifest SHA-256 passed. No live LLM call, SolidWorks import, global Class-A certification or general resident GPU CAD DAG claimed.

## Resident engineering GPU and multi-surface improvement
- User selected improvements 1 and 3. Audit confirmed on-demand class_a batches each create a context and compile; engineering runtime materializes CPU patch results. Extending with a separate persistent surface worker, rational B-spline support and jointly constrained multi-patch optimization, while retaining old batch and hood APIs.
- Added positive-weight NURBS definition/basis compilation/native conversion and homogeneous knot insertion; persistent binary surface arena, cached tables/PTX, compact diagnostics, explicit geometry fetch and CUDA graph coordinate search. CPU compiles a joint seam/lock nullspace; GPU gathers and projects input geometry and optimizes curvature-gradient, reflection-gradient and shape-deviation objectives without intermediate geometry download. Initial Rust build caught the Rust 2024 reserved identifier gen; renamed it before retrying.
- Fixed launch-argument temporary lifetimes and replaced the default 1024-thread launch with 64-thread blocks after actual CUDA reported register resource exhaustion. Twelve initial independent tests then passed. Extended all-four-edge joint constraints and shared corner coupling, resident homogeneous refinement, reusable network matrix uploads and model tool/STEP integration. Seventeen tests passed, including actual G3 and four-patch corners, cache reuse, atomic resource cleanup and delivered-support binding. Closed subprocess pipes explicitly after resource warnings in the first runtime test run.
- Combined G3 demo passed: objective 7.14946 -> 1.08804, dense independent 33x33 curvature-gradient metric reduction 88.60%, max control displacement 4.4261 mm within 5 mm, seam jets 4.58e-11 mm and STEP sampled roundtrip 9.20e-12 mm. Added seed compatibility directions so a fully locked valid resident seed passes while incompatible locks reject; 18/18 tests passed. Actual CPU/GPU NURBS D3 error 4.43e-12 mm. Original 20 engineering cases, current 16 Rust test executions and Clippy passed.
- Measured current-code same-grid workflow: legacy cold-process median 726.13 ms; cached-PTX resident startup 613.77 ms, then full-grid warm median 1.6161 ms and summary median 0.6126 ms. Warm calls uploaded zero bytes and downloaded no control nets. Common fields 0..44 were bit-equal in this fixture. These are process-amortization/full-output measurements, not kernel throughput claims. Generated and visually inspected reflection/curvature-gradient preview; final source/STEP/package verification follows.
- Completed current-source delivery verification: reopened all three resident STEP files with valid topology and bounded native tolerances; verified preview source/grid/image hashes, 58 documentation links and all previous artifact digests. Rust formatting and Python compilation checks passed. Delivered output/delivery/rust-cuda-resident-surfaces-v1.zip, 17,193,445 bytes, 560 files including manifest; SHA-256 a85bb9996529a1b125e7e7c19b2580430956a4356bc26c383d9b37d72cba30be. ZIP CRC and every manifest hash passed. Selected improvements 1 and 3 are complete; general rational network optimization and teaching the new tools to CuModel remain explicit next work.
