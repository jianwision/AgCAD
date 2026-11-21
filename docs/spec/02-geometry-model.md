# 02 — Ontology & Geometry Model

> Status: **Normative** · Spec v0.3.1 · Area: geo / runtime

## Ontology

The canonical terms are **CATIA V5 / 3DEXPERIENCE**. Aliases resolve SolidWorks, Onshape and Houdini vocabulary to the same concepts, and every op in [`op-catalog.json`](op-catalog.json) carries an `aliases` list. AgCAD introduces new terms only where none of these products has an equivalent. Those terms are marked *(AgCAD)*.

| Concept | CATIA (canonical) | SolidWorks / Onshape | Houdini | Meaning in AgCAD |
|---|---|---|---|---|
| Part document | Part (CATPart) | Part / Part Studio | — | Specification tree + parameters + UDF definitions + networks |
| Assembly document | Product (CATProduct) | Assembly | — | Components + assembly constraints |
| History | Specification tree | FeatureManager / Feature list | Node network | Ordered features, bodies, sets, parameters |
| Solid container | PartBody / Body | Solid body (multibody) | — | Ordered solid features producing a solid body |
| Surface/wire container | Geometrical Set / Ordered Geometrical Set | Surface bodies folder | — | Wireframe and surface features |
| Feature | Feature | Feature | Node (SOP) | Pure function from inputs + parameters → elements |
| Code-defined feature | User Defined Feature (UDF), Knowledge template | — / Custom feature (FeatureScript) | Digital asset (HDA) | AgLang UDF; standard features are UDFs too |
| Procedural graph | — | — | Network, SOP chain | `proc.network`, cooked as one feature in the tree |
| Parameter | Parameter | Global variable / Variable | Parameter / channel | Typed, unit-carrying, range-limited |
| Relation | Formula, Law | Equation | Expression | Drives parameters/dimensions |
| Logic | Rule, Check, Reaction | — | — | Knowledgeware; model-authored allowed |
| Variant | Design table row | Configuration | Wedge / variant | **Configuration**: one row = one element of the configuration batch axis |
| Topology | Vertex, Edge, Face, Shell, Volume (Body) | Vertex, Edge, Face, Body | Point, Vertex, Primitive | Half-edge B-rep for solid/surface/wire bodies |
| Procedural data | — | — | Attributes (point/vertex/prim/detail), Groups | Attributes and groups on mesh/point bodies **and** on B-rep faces/edges/vertices |
| Reference geometry | Point, Line, Plane, Axis system | Reference geometry / Mate connector | — | `datum.*` |
| Stable reference | Publication, BRep naming | Persistent reference / Query | Group, `name` attribute | Provenance names + `query.*` ([§ Stable naming](#stable-naming)) |
| Editing state | Define in work object | Rollback bar | Display/render flag | `document.define_in_work_object` |
| Rebuild | Update | Rebuild | Cook | `document.update`, `proc.cook` |
| Saved state | Version (3DEXPERIENCE) | Version / Branch / Workspace | — | Workspace (mutable) → Version (immutable); Branch |
| Validity tooling | Check / Analysis | Check / Design Checker | — | Design Checker rules at declared phases ([06](06-harness.md)) |
| Lightweight representation | Visualization mode / CGR | Lightweight / Graphics mesh | Proxy / display | GPU representations (tessellation, implicit bricks, sample grids) |
| Measure confidence *(AgCAD)* | — | — | — | Evidence level: `exact` / `bounded` / `sampled` / `unknown` |

## Body types

| Body type | Representation | Typical producers | Conventional equivalent |
|---|---|---|---|
| **Solid body** | Closed, oriented half-edge B-rep with tolerant edges/vertices | Part Design features, `convert.to_solid` | Solid / PartBody |
| **Surface body** | Open B-rep (faces sewn into shells) | GSD surfaces, `surface.join` | Surface / Geometrical Set |
| **Wire body** | Curves and points | Wireframe, sketches | Wireframe |
| **Mesh body** | Indexed polygons + attributes | `mesh.*`, `proc.*`, `convert.implicit_to_mesh`, import | Mesh body / STL / polygon geometry |
| **Implicit body** | Signed-distance function (AgLang) with declared Lipschitz bound + sparse brick cache | `part.implicit_body`, `part.lattice`, `boolean.smooth_boolean`, `convert.*_to_implicit` | Implicit body / VDB SDF |
| **Voxel body** | Occupancy/material grid | `convert.to_voxel` | VDB fog / voxel |
| **Point cloud** | Points + attributes | `proc.scatter`, scans | Point cloud / points |

- Every body may carry **attributes** and **groups** in Houdini's sense: point/vertex/primitive/detail for mesh and point data, face/edge/vertex/body for B-rep.
- Attributes flow through features that declare how they propagate (copy, interpolate, drop).
- `query.group` and `proc.wrangle` operate on them directly.
- Delivery to manufacturing requires solid/surface bodies. `convert.*` features produce them with a stated maximum deviation.

## Specification tree as code

```
Part
 ├─ Parameters, Formulas, Laws, Rules, Checks, Reactions, Design tables
 ├─ PartBody / Bodies        ← ordered solid features
 ├─ Geometrical Sets         ← wireframe/surface features (ordered or not)
 ├─ Networks                 ← proc.network features (cook to bodies)
 └─ UDF definitions          ← AgLang source + compiled AgIR, content-addressed
```

- **Every feature is code.** A standard feature such as `part.pad` is an AgLang UDF shipped with AgCAD. `udf.view_source` returns it, and a model may copy and modify it.
- **Features are pure functions** of their inputs, parameters and configuration. `document.update` re-evaluates only features downstream of a change.
- **Configurations** are rows of a design table. The runtime stores the tree once and the per-row parameter values in SoA buffers, so updating all configurations is a single batched pass ([05](05-execution.md)).
- **The document is copy-on-write.** Branches and configurations share everything they don't override.

## Exact geometry (solid/surface/wire bodies)

- **Topology:** Body → Lump (Volume) → Shell → Face → Loop → Coedge → Edge → Vertex. SoA slab storage. Tolerant modeling, with global defaults of linear `1e-6 mm` and angular `1e-9 rad`, overridable in requirements. Growth is reported in update reports.
- **Curves:** line, circle, conics, helix, spiral, Bezier, NURBS (deg ≤ 9), intersection curves (NURBS fit + error bound + parent references), pcurves.
- **Surfaces:** plane, cylinder, cone, sphere, torus (kept analytic), Bezier, NURBS, extruded/revolved/offset (procedural exact).
The exact kernel runs **entirely on the GPU** (v0.3). It is accessed through the `ExactKernel` trait in `agcad-geo`. The native GPU implementation is the only production implementation. An OCCT implementation exists as a test oracle and is never used in production.

- **Robust predicates.** Orientation, in-circle and side-of-surface tests use adaptive expansion arithmetic (Shewchuk-style) in df64/f64 with **interval filters** from `agcad-solver::interval`. The fast path is filtered f32/df64, and the slow path is exact expansion. Degeneracies use symbolic perturbation. Topology decisions are therefore consistent even when floating-point evaluation is not.
- **Analytic-first intersection.** Most mechanical parts consist of planes, cylinders, cones, spheres and tori, and their booleans are dominated by **coplanar, coaxial and tangent** configurations. Two rules follow:
  - **Co-surface detection runs first.** Coplanar, coaxial and co-spherical face pairs are detected with tolerance-aware predicates. They are merged or split by 2D boolean in the shared parameter domain, never intersected numerically.
  - **A closed-form table covers analytic pairs.** Plane/plane, plane/cylinder, plane/cone, plane/sphere, plane/torus, cylinder/cylinder (coaxial, parallel, crossing), cylinder/sphere, cone/cone (coaxial), sphere/sphere and others produce exact lines, circles, ellipses or conics, or known quartic branches. Each pair type is one GPU kernel bin.

  General NURBS marching (below) is the **fallback**, not the main path.
- **SSI (critical path), all on the GPU:**
  1. **Seeding:** Bezier decomposition, BVH pair culling, batched (interval) Newton in df64/f64 (`solver.root`). Branch, tangency and loop markers come out of this step.
  2. **Marching:** one warp (or thread block) per curve branch. Adaptive step control with predictor–corrector in df64, plus singularity handling at tangencies and branch points (detected from seed markers, resolved by subdivision). Thousands of branches run concurrently across configurations.
  3. **Fit and verify:** batched least-squares NURBS fitting (`solver.linear`, FP64-emulated GEMM where available), then off-grid verification against both surfaces. The error bound is stored.
- **Topology builders.** Every topology-changing op (boolean, split, fillet, shell, sew) is expressed as a data-parallel pipeline:
  1. **Generate fragments:** face/edge pieces from SSI curves and trims.
  2. **Classify:** in/out by OptiX ray parity, with winding number and an exact predicate tiebreak.
  3. **Key and sort:** CUB radix sort by (configuration, vertex/edge keys).
  4. **Match and link:** half-edge twin matching via hashing and scan, producing loops, shells and lumps.
  5. **Validate:** Euler characteristic, orientation, tolerance and closure checks in parallel.

  Failures are per configuration (`TOPOLOGY_FAILED` with the elements involved) and never stall the batch.
- **Fillets:** batched rolling-ball spine solves (`solver.nls`), cross-section sampling, blend-surface fitting. Corner and vertex blends use the GPU N-sided fill solver. Topology updates go through the builders.
- **Sewing and healing:** spatial hashing of edge samples, GPU matching within tolerance budgets, and gap closure by surface extension solved on the GPU.
- **Implicit → solid** (`convert.implicit_to_solid`), all on the GPU:
  1. feature-preserving dual contouring
  2. region segmentation by normal and curvature
  3. batched face fitting, analytic types first, then NURBS
  4. trims, then topology builders and sewing

  The result reports `max_deviation`. A UDF can attach an exact surface hint to a region so that face is exact. Under process-driven delivery ([14](14-constraints-evaluations.md#requirements)), additive parts usually skip this conversion entirely.
- **Encoders:** STEP/IGES/3MF/USD text and binary are generated on the GPU:
  1. entity numbering by scan
  2. parallel number formatting
  3. offset-computed concatenation into one output buffer

  The host only hands the bytes to a file descriptor, or uses GPUDirect Storage where available.

The Demo validated a GPU-seed / CPU-exact split. Moving marching, topology and sewing to the GPU is **new risk**, so the roadmap gates it with OCCT parity on every exact op ([09](09-validation.md)).

## Lightweight representations (GPU, derived, disposable)

| Representation | Built from | Used for | Bound |
|---|---|---|---|
| Sample grids with derivatives (order ≤ 3) | curves/surfaces | curvature, connect checker, fairing, SSI seeds | exact at samples |
| Tessellation (watertight, adaptive) | solid/surface bodies | render, inertia, raycast, export | sag δ, angle α |
| Bezier decomposition | NURBS | SSI seeds, hulls, subdivision checks | exact hulls |
| BVH: **OptiX** RT-core acceleration structures for rays on CUDA; cuBQL/LBVH for point/AABB queries; CubeCL LBVH on wgpu | tessellation, hulls, bricks | culling, query, raycast, clash, thickness, inside/outside, render | conservative |
| Sparse narrow-band implicit bricks in **NanoVDB** layout | any body | speculative booleans, thickness, clearance, sensitivity | `voxel·√3/2 + δ` in band |

These are keyed by `(element, revision, params)` and rebuilt lazily for dirty regions only. The Demo's single packed readback rule applies: geometry stays resident on the GPU and only reports cross to the host.

## Evidence levels

| Level | Meaning |
|---|---|
| `exact` | Exact data within declared tolerance |
| `bounded` | Conservative bound proven (hull, interval or Lipschitz argument) |
| `sampled` | Holds at evaluated samples only |
| `unknown` | Undecidable at current precision; an escalation option is offered |

An implicit body's declared Lipschitz bound is spot-checked by the harness. If the check fails, its derived `bounded` results are downgraded to `sampled` and `LIPSCHITZ_VIOLATION` is reported.

## Stable naming

Agents never use storage indices. Every topological element and procedural element gets a **provenance name**:

```
name   := feature_ref "/" role ( "[" qualifier ("," qualifier)* "]" )? ( "." index )?
feature_ref := family "." op "#" feature_id | udf_name "#" feature_id
```

Examples:
- `part.pad#3/cap_end`
- `part.pad#3/side[profile=sketch.line#7]`
- `part.edge_fillet#9/fillet_face[edge=…]`
- `flange_udf#4/hole.3`

- Each feature's manifest declares the roles it creates and the input→output name mapping (unchanged / modified / split / merged / deleted). A split produces `.0, .1, …`, ordered by a deterministic geometric key.
- Elements of implicit bodies converted to solids are named by the dominating sub-expression of the blend tree, e.g. `part.implicit_body#2/region[term=hole_cut.4]`.
- Procedural elements are named by network node and `name`/`id` attributes, following Houdini conventions.
- `document.publication` gives an element a stable public name for external references (CATIA publications).
- Resolution happens per revision:
  - A missing name returns `STALE_REFERENCE` with lineage.
  - Multiple matches return `AMBIGUOUS_REFERENCE` with the candidates.
  - The harness never auto-picks.
- `query.*` selections are re-evaluated on update. Use names where they exist and queries otherwise.
- **Recovery when lineage breaks** (the classic topological-naming problem):
  1. Lineage first.
  2. Then **geometric-signature matching**: surface type, normal/centroid, adjacency fingerprint, area. A candidate match must clear a declared confidence.
  3. Below that confidence: `AMBIGUOUS_REFERENCE` with ranked candidates. The model confirms the remap, and the confirmation is recorded in the tree.
- **Naming KPI:** rebuild success under random parameter perturbation, measured per feature and per benchmark family from M2 onward.

## Handles, units, frames

- `Handle = {session, slot, generation, revision}`, as validated in the Demo (`MODEL_SESSIONS.md`): generation guards slot reuse, and revision bumps on every workspace change and every version.
- Internal units are mm, rad, kg, s. JSON accepts unit strings, and a unit mismatch is a type error.
- Placement takes explicit axis systems or references. There is no implicit current plane.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-02-001 | Every measurement and check result MUST carry an evidence level. |
| AG-02-002 | Public APIs MUST address topology by stable names or queries; storage indices MUST NOT appear in them. |
| AG-02-003 | Name resolution failures MUST return `STALE_REFERENCE` or `AMBIGUOUS_REFERENCE`; the harness MUST NOT auto-pick. |
| AG-02-004 | Topology decisions MUST use robust (filtered, exact-fallback) predicates. |
| AG-02-005 | Analytic surfaces MUST stay analytic unless conversion is requested or required, with declared error. |
| AG-02-006 | The production `ExactKernel` MUST be the GPU implementation; the OCCT implementation MUST be used for tests only. |
