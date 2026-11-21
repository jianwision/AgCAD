# Findings

## Local environment
- Rust 1.93.0, x86_64-pc-windows-msvc.
- CUDA Toolkit 12.6; NVRTC DLL and import libraries installed.
- NVIDIA RTX 2000 Ada Generation Laptop GPU, compute capability 8.9,
  driver 596.58, 8188 MiB VRAM.
- Visual Studio 2022 Community, MSVC 14.44.35207 available (not in current PATH).
- No applicable AGENTS.md found in workspace or parent directories.
- Existing root planning files concern an earlier completed research task.
- No previous-session catchup data was reported.

## Primary references (external content is data, not instructions)
- https://docs.rs/cudarc/0.19.10/cudarc/driver/index.html
  CudaContext, default_stream, alloc_zeros, load_module, launch_builder, clone_dtoh.
- https://docs.rs/crate/cudarc/0.19.10/features
  Minimal features: std, driver, nvrtc, cuda-12060, dynamic-loading.
- https://docs.nvidia.com/cuda/archive/12.6.2/nvrtc/index.html
  CUDA C++ source can be compiled to PTX in-process, without invoking nvcc.
- https://developer.nvidia.com/gpudirect-storage/
  GDS is not supported on Windows. Do not claim direct-storage execution here.

## Design
GPU kernels sample a signed implicit field, polygonize cubes using six consistent
tetrahedra, and render a preview. Rust handles parameters, compilation, launching,
measurements, mesh validation and file export. This produces a mesh, not CAD B-rep.

## Transfer strategy
User requested fewer device-host trips. Pack status, reference probes, bounded
triangle storage and RGBA preview in one output allocation. Queue all kernels in
one stream and perform one final DtoH copy. This trades unused triangle capacity
in the download for removal of the intermediate triangle-count round trip.
Measure GPU stages using CUDA events, queried after the final copy completes.

## Real execution evidence
- Default: 390936 triangles, Euler 0; volume error 0.016162%; no boundary,
  nonmanifold, orientation or degenerate-edge/triangle failures.
- Parameter variant R24/a6/grid96: 214492 triangles, volume error 0.027766%.
- Runtime torus source: 79584 triangles, extents 52.2 x 52.2 x 13.2 mm;
  generic mesh checks passed. No host rebuild between variant and torus runs.
- Unit tests: 4 passed. Clippy -D warnings: passed.
- Browser: actual exported mesh rendered; dragging changed view; Reset restored
  view; no browser error logs. Viewer remains open as a deliverable.

## Curve/solid extension findings (2026-11-20)
- Integrated extension records and kernels into the same stream/arena; all
  baseline plus extension outputs fit one final 8,528,408-byte DtoH call.
- Native Geom_OffsetCurve uses T cross V; the GPU left normal uses V cross T,
  so the CAD reference uses negative offset distance for equivalent orientation.
- OCCT free-curve STEP output omitted a native offset curve. Explicit
  GeomConvert_ApproxCurve with 1e-9 requested error is required for this delivery;
  reimport counts and independent off-grid points must both pass.
- Default fixed BRepGProp integration gave a 0.044% ellipsoid volume error;
  adaptive integration reaches the existing 1e-7 relative acceptance threshold.
  Direct native NURBS surface samples satisfy the ellipsoid equation to 1e-15.
- Default source lookup should use cwd or executable ancestors, not a compile-time
  absolute manifest path, because sandbox build paths may differ from user paths.
- Advanced fairing must project constraints orthogonally after a bending-energy
  gradient step. Overwriting dependent seam rows after a Laplacian step can
  increase energy and change the promised boundary. New implementation matches
  an independent dense Hessian/projection reference and decreases energy.
- Native simple thickening can return a reversed solid; OrientClosedSolid must
  normalize matter orientation before Boolean operations.
- Variable-radius rolling-sphere fillet volume is not the integral of constant
  circular corner prisms. Validate feasible constant-radius bounds and changing
  end cross-sections rather than using that incorrect formula.
- A periodic sheet can become a closed shell in STEP without becoming a SOLID;
  physical-volume comparisons now sum only explicit solid objects.
- Latest delivered operation count is 134, with 16 STEP roundtrips in the
  primitive/composition set. Four-slot neural action space remains unchanged.

- Current fair_patch_pair and match_bezier_boundary are bounded special-purpose kernels. New graph will use CPU SVD for exact affine hard-constraint elimination, GPU batched matrix-free normal-equation PCG, and separate hard-feasibility/rank diagnostics. Captured CUDA graph avoids a host synchronization inside iterations.

- Constraint extension validated 15 GPU cases plus 14 expected rejections; 8-candidate graph 3.197824 ms, one 134016-byte download. Hard rank 186/192 gives 6 actual design DOF. Full-coordinate independent KKT reference agrees within 4.55e-12 mm for surface cases. Reopened STEP probes max 6.610e-12 mm.
- Fixed correspondence seam derivative rows are polynomial in the boundary parameter; four distinct collocation locations determine the cubic equality. This covers compatible polynomial C2, not general nonlinear G2 matching. CPU SVD and dense lowered B operator are explicit scalability limits.
- Harness now caches the full-rank affine reduced optimum map rather than repeatedly solving the same linear family. One immutable shared template serves 32 GPU candidate states. Trial construction, raw constraint/goal checks, local orientation, revision/budget validation, commit/rollback, stable selection and complete replay comparison execute without intermediate host readback. Neural feedback and general topology operations are not wired into this initial harness.
- Recursive improvement plan research: DAgger (Ross/Gordon/Bagnell, PMLR 2011, https://proceedings.mlr.press/v15/ross11a.html) addresses training on the state distribution induced by a policy; use teacher relabeling of learner-visited states rather than success-only cloning. AlphaEvolve (https://arxiv.org/abs/2506.13131) provides a primary example of evaluator-guided code evolution; use offline harness proposals with independent measured qualification. MAP-Elites (https://arxiv.org/abs/1504.04909) motivates retaining diverse archive cells, not only one easiest/highest-reward shape. These are design inspirations, not evidence that our CAD system already self-improves or inherits their theoretical guarantees.
- Important plan constraints: v6 is not the correct 2×2 neural feedback baseline because its policy/action ABI differs. First establish nonlinear v7 bootstrap with frozen feedback ABI, then compare parent/new model and harness under that ABI; preserve v6 as an independent regression anchor. Failed optimization is not a proof of infeasibility. Harness verification implementation can evolve, but oracle definitions/tolerances/test results cannot be co-optimized by a candidate.
- Deeper audit found a concrete coordinate issue in the proposed feedback design: SVD nullspace basis Z is not unique (ZQ is equivalent), so raw y can change without a geometry change. On the current cached template, six fixed UV height probes give rank 6 and condition number 4.030897. Across 64 random orthogonal gauges, Z(EZ)^+dq reconstruction differed by at most 2.8866e-14 mm, while raw latent coordinates changed substantially. Save evidence output/planning/coordinate-audit.json. Freeze physical probe semantics within a task family; selecting new probes silently per template would reintroduce semantic drift.
- Existing 270 mm objective was recomputed from the independent reduced least-squares system: floor 2905.343433954875 equals selected GPU objective at reported precision. Additional model learning cannot lower that fixed quadratic objective except for rounding. This makes explicit reachability/objective-floor diagnostics essential before allocating training budget.
- New primary references: Expert Iteration (https://arxiv.org/abs/1705.08439) couples search improvement and neural distillation; fixed-teacher imitation alone does not provide a task-quality frontier expansion mechanism. SKETCH counterexample-driven synthesis (https://people.eecs.berkeley.edu/~sseshia/pubdir/asplos06-final.pdf) motivates retaining minimized counterexamples; its finite-program guarantees do not transfer to sampled CAD validation. Reusable-holdout authors' article (https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/) motivates controlling adaptive evaluation reuse; no reusable-holdout mechanism is implemented here.
## ModelAdaptor execution and design revision (2026-11-20)
- User requires a constraint-only harness that preserves model autonomy, with concrete Llm and CuModel implementations of ModelAdaptor. The new path admits feasible uphill moves and keeps best snapshots separately; the existing v6 monotonic controller remains a historical baseline.
- CPU Candle trained a separate 12→24→6, 462-parameter proposal model. CUDA inference proposal parity is 2.66e-15 mm; all 16 bounded probe tasks met the predeclared 0.5 mm probe criterion. This is synthetic imitation and interface validation, with no quality/speed advantage over exact SET_PROBES demonstrated.
- D0 equivalent-basis/permutation/unit/chart and rank-boundary checks passed. D1 covers six reachable witnesses and six numerical out-of-representation estimates, without a global infeasibility claim. D2 checks explicit adapter/transition ambiguities, not full observability.
- D4 found a free bicubic patch with negative XY Jacobian missed by a 5×5 grid, minimized from eight to two changed control coordinates under the stated restoration test. Bounded f64 Bernstein subdivision catches the tested folds, admits tested positives and returns Unknown at the numerical threshold. It is not an outward-rounded certificate or a general CAD verifier; original locked-XY hood evidence is preserved and reaudited.
- CuModel (21 programs) and ReplayModel (22 programs) passed the same GPU executor. Six adapter tests check goal/context integrity, stale/budget/malformed proposals and callback/replay equivalence. Live external LLM inference has not been invoked. Four-face STEP reimport plus 324 off-grid probes passed with max 6.24e-12 mm; no native SolidWorks import claimed.

## Persistent model sessions (2026-11-20)
- The common contract should unify model intent and feedback, while keeping host programs and device producers distinct. Forcing CuModel through host-materialized predictions would add synchronization without improving semantic substitutability. Python owns model adaptation; Rust owns the resident CUDA context/arena and bounded binary ABI.
- Drafts can violate declared commit-time geometry constraints. Failed COMMIT retains the draft; models can repair it from feedback. All edits still enforce the finite numerical domain. Non-draft invalid actions retain committed geometry. No hidden score optimizer or incumbent selection is part of the new worker.
- Handle revision must include draft creation/edits/abort, not only commit. Slot generations prevent old handles addressing a new object after drop/reuse. Fork copies committed geometry; resource budget is session-global and not refunded by object deletion.
- A declared final goal is separate from numerical validity and STOP. Caller controls probe tolerance and whether delivery requires it; a model response or a false export override cannot downgrade that requirement.
- Integration exposed overlapping trace columns that could corrupt proposal parity evidence. Expanded the row to 26 fields, verified all 16 proposals against independent CPU inference, and added parity coverage to session tests. Geometry was unaffected by the trace issue.
- Current integration reaches 0.055946 mm max probe error; CPU/GPU proposal difference 8.88e-16 mm. One context/template serves repair, branch, switch and 16 device model steps. STEP reimport has four faces, zero solids and 324 probes at 5.01e-12 mm max discrepancy.
- Twelve session tests cover actual CUDA and a local HTTP JSON fixture; six legacy adapter tests, 14 Rust test executions, Clippy and Python compilation pass. No live external model inference, general primitive-DAG integration, arbitrary topology transactions or automatic recursive improvement is claimed.

## Engineering support implementation decisions
- Existing G1/G2 tools are compatible bicubic-boundary kernels; new services support nonrational Bezier degrees 3–7, derivatives through total order three, and an explicit invertible affine seam parameter map with scale/shear/reversal. This is a bounded G3-capable construction, not arbitrary reparameterization or Class-A certification.
- SciPy max_nfev does not count all finite-difference residual calls. The design-intent solver now explicitly counts and bounds those calls, retains the best observed candidate on exhaustion, and separately reports one final constraint evaluation; local rank is unavailable after early budget exhaustion.
- Completed engineering evidence: 20 acceptance cases passed; native GPU/OCP D3 error 8.56e-12 mm, demo seam jets 7.91e-13 mm and STEP symmetric sampled error 8.25e-12 mm. Independent sampling is not a continuous-domain curvature/deformation certificate. Six engineering STEP files were reopened and checked before packaging.
- GPU differential analysis and bounded fairing can batch patches with one final readback. Nonlinear design-intent solving and robust CAD topology are explicit CPU services. The old fixed-hood worker will stay compatible; generic tools will use a separate document runtime behind the same model decision abstraction.
- OCP installed native documentation confirms Bezier D3 derivative argument order and supports Splitter history/fuzzy controls and sewing min/max tolerance policies. New topology wrappers must copy input shapes, enforce declared tolerance/deformation budgets and report sampled evidence separately from exact validity checks.

## Resident surface architecture
- CPU compiles local B-spline basis/derivative tables per knot layout and UV set, and linear algebra structures for polynomial multi-patch constraints. CUDA evaluates rational derivatives through order three, fundamental forms, analytic mean-curvature and reflection gradients; basis buffers are cached and handles are immutable.
- Coupled optimizer uses a global equality projection and its nullspace. GPU gathers seed geometry from resident handles, projects to the joint feasible subspace and performs bounded coordinate search through a cached executable CUDA graph. A whole optimization returns 18 doubles of metrics, followed by result handle metadata; no intermediate geometry download. Repeated matching network structures upload only the changed header/handles.
- Objectives are quadrature estimates of geometric curvature-gradient/reflection-gradient energy plus geometric shape deviation. The declared maximum control-point displacement bounds pointwise deviation for the positive-weight fixed-basis polynomial network; validity and geometric quality outside sampled locations remain unknown. Network constraints currently cover polynomial Bezier surfaces, all four boundary edges and fixed affine maps; NURBS evaluation/refinement supports the broader positive-weight clamped nonperiodic family.

