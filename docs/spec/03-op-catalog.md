# 03 — Feature Table: Core, Standard, Model

> Status: **Normative** · Spec v0.3.1 · Area: feature table

The feature table is **open** and uses **CATIA canonical names** with SolidWorks / Onshape / Houdini **aliases**. An agent that asks for `extrude`, `boss-extrude` or `pad` reaches the same `part.pad`.

| Layer | Families | Who writes it | Bypassable? |
|---|---|---|---|
| `core` | `document`, `knowledge`, `udf`, `proc`, `convert`, `study`, `evaluate`, `query`, `file` | AgCAD | No: this is the execution and harness interface |
| `stdlib` | `datum`, `sketch`, `constraint`, `wireframe`, `surface`, `part`, `boolean`, `transform`, `direct`, `mesh`, `assembly` | AgCAD, **as AgLang UDFs on core** (exact-topology internals in Rust) | Yes: call, copy (`udf.view_source`), modify, or ignore |
| `model` | Anything published through `udf.publish` | Models | Yes |

[`op-catalog.json`](op-catalog.json) is normative for `core` and `stdlib`. Model UDFs live in the session feature table and in the feature catalog / skill library ([11](11-self-improvement.md)).

## Feature manifest

Every feature, in every layer, declares a manifest:

| Field | Meaning |
|---|---|
| `name`, `opcode`, `version`, `aliases` | `family.op`. Opcode = `(family << 8) \| index`, append-only. Model UDFs get session opcodes `0x8000–0xFFFF` |
| `inputs`, `outputs` | Typed signature (`name:Type`, `?` optional, `[]` array, `[,]` grid, `A\|B` either, `Configurations<T>`) |
| `batch_axes` | Which of `configuration`, `entity`, `sample` the feature vectorizes over |
| `phase` | `control`, `proxy` (lightweight), `exact` or `both` ([05](05-execution.md)) |
| `backend` | `gpu`, or `host` for compile-time control-plane ops only (`udf.define`, `udf.compile`, `udf.view_source`) |
| `effect` | `pure`, `doc_write` or `io` |
| `deterministic`, `differentiable` | Per [05](05-execution.md#determinism) and [04](04-agir.md#autodiff) |
| `failure_codes` | From [06](06-harness.md#error-codes) |
| `naming` | Roles created, input→output name mapping ([02](02-geometry-model.md#stable-naming)) |
| `attributes` | Attribute propagation rules (copy / interpolate / drop) |
| `evidence` | Evidence level of each output and the bounds it promises |
| `milestone`, `status`, `layer` | Planning metadata |

## Family notes

- **`document`:** Part/Product, Body, Geometrical Set, define-in-work-object, deactivate, publication. Also the transaction verbs: `edit`, `update`, `save_version`, `discard_edit`, `branch`, `merge`.
- **`knowledge`:** CATIA Knowledgeware: parameters, formulas, laws, rules, checks, reactions, design tables, PowerCopy. Model-authored rules and checks are allowed. They add evidence but never weaken requirements.
- **`udf`:** the code-first heart of the system: define → compile → instantiate → publish. Equivalent to CATIA UDFs and knowledge templates, Onshape custom features and Houdini digital assets, but GPU-compiled and model-authored ([04](04-agir.md)).
- **`proc`:** Houdini-style procedural modeling with networks, attributes, groups, wrangles, scatter, copy-to-points, for-each and solver blocks. A network is one feature in the specification tree, and it cooks for all configurations at once.
- **`convert`:** between body types, always with a declared maximum deviation. `convert.body_type` dispatches to the best published converter. Model-published converters can win after conformance.
- **`study`:** configurations, batched update, evaluate, filter, rank, optimize, sensitivity, render, iterate, observe. These are the parallel-workflow primitives, and they run against the GPU harness.
- **`evaluate`:** CATIA measure and analysis vocabulary (measure between/item/inertia, curvature, connect checker, highlight lines, draft/thickness/distance analysis, sectioning, clash, check).
- **`part`, `surface`, `wireframe`:** Part Design and Generative Shape Design. `part` also includes `implicit_body` and `lattice` as standard features.

Legend:
- **Axes:** C = configuration, E = entity, S = sample.
- **∂:** differentiable.
- **MS:** milestone ([10](10-roadmap.md)).

<!-- BEGIN GENERATED CATALOG -->

### `document` (core, family 0x00, 24 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `document.new_part` |  | `0x0000` | Create a Part (CATPart): origin, default planes, PartBody, feature tree. | new part studio | `name:String, units:Enum? → part:Part` | control | GPU | C |  | M1 |
| `document.new_product` |  | `0x0001` | Create a Product (assembly document). | new assembly | `name:String → product:Product` | control | GPU | C |  | M1 |
| `document.body` |  | `0x0002` | Insert a Body (solid feature container) into the part. | insert body, multibody | `part:Part, name:String → body:BodyRef` | control | GPU | C |  | M1 |
| `document.geometrical_set` |  | `0x0003` | Insert a Geometrical Set (unordered wireframe/surface container) or Ordered Geometrical Set. | surface body folder | `part:Part, name:String, ordered:Bool → set:SetRef` | control | GPU | C |  | M1 |
| `document.define_in_work_object` |  | `0x0004` | Make a feature the current work object (rollback bar); later features are not rebuilt. | rollback, rollback bar | `feature:FeatureRef → —` | control | GPU | C |  | M1 |
| `document.deactivate` |  | `0x0005` | Deactivate or activate features. | suppress, unsuppress | `features:FeatureRef[], active:Bool → —` | control | GPU | C |  | M1 |
| `document.material` |  | `0x0006` | Apply material (density, mechanical properties, appearance) to bodies or parts. | — | `targets:Ref[], material:Material → —` | control | GPU | C |  | M1 |
| `document.publication` |  | `0x0007` | Publish an element under a stable public name for external references. | published geometry | `element:Ref, name:String → pub:Publication` | control | GPU | C |  | M1 |
| `document.edit` |  | `0x0008` | Open the workspace for editing (private, may be temporarily invalid). | — | `branch:Branch → workspace:Workspace` | control | GPU | C |  | M1 |
| `document.update` |  | `0x0009` | Update (rebuild) the feature tree and run edit-phase Design Checker rules. | rebuild, regenerate, cook | `workspace:Workspace? → report:UpdateReport` | control | GPU | C |  | M1 |
| `document.save_version` |  | `0x000A` | Run save-phase checks; on success create an immutable version (optional label), on failure keep the workspace. | commit | `workspace:Workspace?, label:String? → version:Version` | control | GPU | C |  | M1 |
| `document.discard_edit` |  | `0x000B` | Discard workspace changes since the last version. | abort, revert | `workspace:Workspace? → —` | control | GPU | C |  | M1 |
| `document.checkpoint` |  | `0x000C` | Return an update report to the caller and wait for the next decision. | — | `label:String → report:UpdateReport` | control | GPU | C |  | M1 |
| `document.branch` |  | `0x000D` | Create branches from a version, sharing history copy-on-write. | fork | `version:Version, count:Int → branches:Configurations<Branch>` | control | GPU | C |  | M1 |
| `document.switch_branch` |  | `0x000E` | Make a branch the active branch. | — | `branch:Branch → —` | control | GPU | C |  | M1 |
| `document.merge` |  | `0x000F` | Merge feature-level changes between branches. | — | `from:Branch, into:Branch → version:Version` | control | GPU | C |  | M1 |
| `document.delete` |  | `0x0010` | Delete a feature, body, set or branch handle; budget is not refunded. | — | `handle:Handle → —` | control | GPU | C |  | M1 |
| `document.specification_tree` |  | `0x0011` | Return the specification tree: features, parameters, bodies, names, versions. | feature tree, feature manager, node network view | `target:Handle → tree:SpecTree` | control | GPU | C |  | M1 |
| `document.undo` |  | `0x0012` | Undo the last workspace change. | — | `workspace:Workspace? → —` | control | GPU | C |  | M1 |
| `document.redo` |  | `0x0013` | Redo an undone change. | — | `workspace:Workspace? → —` | control | GPU | C |  | M1 |
| `document.finish` |  | `0x0014` | Caller declares the design finished; delivery checks run; does not imply success. | — | `— → report:DeliveryReport` | control | GPU | C |  | M1 |
| `document.comment` |  | `0x0015` | Attach a review comment to a version and stable element names. | — | `version:Version, elements:String[], text:String → comment:Comment` | control | GPU | C |  | M1 |
| `document.approve` |  | `0x0016` | Record a human or delegated approval/rejection of a version (gates delivery when requirements demand it). | — | `version:Version, decision:Enum, note:String? → approval:Approval` | control | GPU | C |  | M1 |
| `document.diff` |  | `0x0017` | Diff two versions: features, parameters, elements (with lineage), evaluations; optional diff renders. | — | `a:Version, b:Version → r:DiffReport` | control | GPU | C |  | M1 |

### `knowledge` (core, family 0x01, 10 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `knowledge.parameter` |  | `0x0100` | Create/modify a parameter (length, angle, real, integer, boolean, string, list) with units and range. | global variable, channel, spare parameter | `name:String, type:Enum?, value:Any, range:Interval? → param:Parameter` | both | GPU | C | ✓ | M1 |
| `knowledge.formula` |  | `0x0101` | Formula driving a parameter or feature dimension from other parameters. | equation, expression | `target:Ref, expr:String → f:Formula` | both | GPU | C | ✓ | M1 |
| `knowledge.law` |  | `0x0102` | Law curve (value along a parameter) for variable sweeps, fillets, gradings. | — | `kind:Enum, points:Vec2[]/Expr → law:Law` | both | GPU | C | ✓ | M1 |
| `knowledge.rule` |  | `0x0103` | Knowledge rule: conditional logic setting parameters/activation on update. | design rule | `source:String → rule:Rule` | both | GPU | C | ✓ | M1 |
| `knowledge.check` |  | `0x0104` | Knowledge check: user/model-authored pass/fail condition evaluated on update (adds evidence, never replaces requirements). | — | `source:String, severity:Enum → check:Check` | both | GPU | C | ✓ | M1 |
| `knowledge.reaction` |  | `0x0105` | Reaction: action script triggered by an event (parameter change, feature update). | — | `event:Event, source:String → reaction:Reaction` | both | GPU | C |  | M1 |
| `knowledge.design_table` |  | `0x0106` | Design table: named rows of parameter values; rows are configurations. | configurations table, excel design table | `params:Parameter[], rows:Any[][] → table:DesignTable` | both | GPU | C | ✓ | M1 |
| `knowledge.power_copy` |  | `0x0107` | Instantiate a PowerCopy (feature group template) with new inputs. | copy features, derived | `template:Ref, inputs:Map → features:FeatureRef[]` | both | GPU | C | ✓ | M1 |
| `knowledge.tolerance` |  | `0x0108` | Dimensional or GD&T tolerance on a parameter, dimension or feature (size, position, profile, flatness, ...), with datum references and distribution. | gd&t, fta, 3d annotation | `target:Ref, spec:ToleranceSpec, datums:Ref[]?, distribution:Enum? → tol:Tolerance` | both | GPU | C | ✓ | M3 |
| `knowledge.catalog_insert` |  | `0x0109` | Insert a standard part from a catalog (ISO/DIN fasteners, bearings, profiles) by designation and parameters. | toolbox, standard parts, catalog browser | `catalog:String, designation:String, params:Map? → feature:FeatureRef` | both | GPU | C | ✓ | M4 |

### `udf` (core, family 0x11, 6 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `udf.define` |  | `0x1100` | Define a User Defined Feature in AgLang (inputs, parameters, outputs, naming roles, checks) or as a procedural network. | user feature, custom feature, knowledge template, digital asset, hda | `source:String/Network, lang:Enum → definition:UdfDef` | control | host (control plane) | C |  | M1 |
| `udf.compile` |  | `0x1101` | Compile a UDF to batched GPU kernels (vectorized over configurations) or device bytecode. | — | `definition:UdfDef, target:Enum → kernel:Kernel` | control | host (control plane) | C |  | M1 |
| `udf.instantiate` |  | `0x1102` | Instantiate a UDF in the specification tree with inputs and parameter values. | insert custom feature | `definition:UdfDef, inputs:Map, params:Map → feature:FeatureRef` | control | GPU | C,E | ✓ | M1 |
| `udf.sample` |  | `0x1103` | Sample a UDF's geometry functions (implicit value, surface point, derivatives) at points/parameters. | — | `feature:FeatureRef, at:Tensor, order:Int? → v:Tensor` | proxy | GPU | C,E,S | ✓ | M1 |
| `udf.publish` |  | `0x1104` | Publish a UDF to a feature catalog after conformance checks pass. | catalog, asset library | `definition:UdfDef, manifest:FeatureManifest, scope:Enum → feature:FeatureTypeRef` | exact | GPU | C |  | M4 |
| `udf.view_source` |  | `0x1105` | Return the AgLang source of any standard feature or UDF (it may be copied and modified). | — | `feature:FeatureTypeRef → source:String` | control | host (control plane) | C |  | M1 |

### `proc` (core, family 0x0C, 18 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `proc.network` |  | `0x0C00` | Create a procedural network (node graph) that cooks to one or more bodies; appears as one feature in the tree. | sop network, geometry node, houdini engine asset | `name:String, inputs:Ref[] → net:Network` | both | GPU | C,E,S | ✓ | M3 |
| `proc.node` |  | `0x0C01` | Add a node to a network: any catalog op or UDF, wired by input/output ports. | — | `net:Network, op:OpRef, inputs:Port[], params:Map → node:Node` | both | GPU | C,E,S | ✓ | M3 |
| `proc.wrangle` |  | `0x0C02` | Run an AgLang snippet over points, vertices, primitives, faces, edges or detail, reading/writing attributes. | attribute wrangle, vex, point wrangle | `net:Network, over:Enum, source:String → node:Node` | both | GPU | C,E,S | ✓ | M3 |
| `proc.attribute_create` |  | `0x0C03` | Create an attribute (point/vertex/primitive/face/edge/detail) with type and default. | — | `target:Ref, class:Enum, name:String, type:Enum → attr:Attribute` | both | GPU | C,E,S | ✓ | M3 |
| `proc.attribute_promote` |  | `0x0C04` | Promote/demote attributes between classes with a reduction. | — | `target:Ref, name:String, from:Enum, to:Enum, method:Enum → attr:Attribute` | both | GPU | C,E,S | ✓ | M3 |
| `proc.attribute_transfer` |  | `0x0C05` | Transfer attributes from one geometry to another by proximity. | — | `src:Ref, dst:Ref, names:String[], radius:Length → attr:Attribute[]` | both | GPU | C,E,S | ✓ | M3 |
| `proc.group` |  | `0x0C06` | Create a group (named element set) by expression, query or bounding region. | group create, selection set | `target:Ref, class:Enum, pred:Predicate/Query → group:Group` | both | GPU | C,E,S | ✓ | M3 |
| `proc.blast` |  | `0x0C07` | Delete or keep elements of a group. | delete by group | `target:Ref, group:Group, keep:Bool → geo:Geometry` | both | GPU | C,E,S | ✓ | M3 |
| `proc.scatter` |  | `0x0C08` | Scatter points on curves, surfaces or volumes by density attribute, with seed. | — | `target:Ref, count:Int/DensityAttr, seed:Int → pts:PointCloud` | both | GPU | C,E,S | ✓ | M3 |
| `proc.copy_to_points` |  | `0x0C09` | Instance or copy geometry onto points, oriented by point attributes (N, up, orient, pscale). | copy stamp, instancer, user pattern | `geo:Ref, pts:Ref, pack:Bool → geo:Geometry` | both | GPU | C,E,S | ✓ | M3 |
| `proc.for_each` |  | `0x0C0A` | For-each block over pieces, points, groups or a count, with feedback option. | foreach loop, for each named primitive | `net:Network, over:Enum, body:Network, feedback:Bool → node:Node` | both | GPU | C,E,S | ✓ | M3 |
| `proc.solver` |  | `0x0C0B` | Iterative solver block: the network's output feeds its next iteration (growth, relaxation, simulation-like rules). | solver sop, feedback loop | `net:Network, body:Network, iterations:Int, until:Predicate? → node:Node` | both | GPU | C,E,S | ✓ | M3 |
| `proc.merge` |  | `0x0C0C` | Merge geometry streams. | — | `inputs:Ref[] → geo:Geometry` | both | GPU | C,E,S | ✓ | M3 |
| `proc.switch` |  | `0x0C0D` | Choose an input by parameter or expression (procedural variant). | — | `inputs:Ref[], index:Expr → geo:Geometry` | both | GPU | C,E,S |  | M3 |
| `proc.poly_extrude` |  | `0x0C0E` | Extrude polygon faces with inset, twist, per-element attributes. | — | `mesh:Ref, group:Group?, distance:Length/Attr, inset:Length? → mesh:MeshBody` | both | GPU | C,E,S | ✓ | M3 |
| `proc.resample` |  | `0x0C0F` | Resample curves by length or count; add tangent/curve-u attributes. | — | `curves:Ref, seg:Length/Int → curves:Geometry` | both | GPU | C,E,S | ✓ | M3 |
| `proc.sweep_along` |  | `0x0C10` | Sweep cross-sections along curves using point attributes for frames and scale. | sweep sop | `curves:Ref, section:Ref → geo:Geometry` | both | GPU | C,E,S | ✓ | M3 |
| `proc.cook` |  | `0x0C11` | Cook a network (or node) for all configurations; outputs flow into the specification tree. | — | `net:Network → bodies:Body[]` | both | GPU | C,E,S | ✓ | M3 |

### `convert` (core, family 0x12, 9 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `convert.implicit_to_mesh` | ● | `0x1200` | Convert an implicit body to a watertight mesh body (dual contouring / adaptive marching). | convert to mesh, convert vdb | `body:Ref, res:Length, bounds:Box → mesh:MeshBody` | both | GPU | C,E |  | M3 |
| `convert.implicit_to_solid` |  | `0x1201` | Convert an implicit body to an exact solid body: feature-preserving mesh then analytic/NURBS face fitting with deviation bound. | implicit to brep | `body:Ref, tol:Length → solid:SolidBody` | exact | GPU | C,E |  | M3 |
| `convert.body_to_implicit` | ● | `0x1202` | Signed-distance implicit view of a solid/surface body (sparse bricks + exact refinement). | vdb from polygons, sdf | `body:Ref, voxel:Length → implicit:ImplicitBody` | both | GPU | C,E |  | M3 |
| `convert.mesh_to_implicit` | ● | `0x1203` | Implicit body from a mesh body (generalized winding number / signed distance). | — | `mesh:Ref, voxel:Length → implicit:ImplicitBody` | both | GPU | C,E |  | M3 |
| `convert.point_cloud_to_implicit` |  | `0x1204` | Implicit body fitted to a point cloud with normals (RBF or neural). | — | `pts:Ref, tol:Length → implicit:ImplicitBody` | both | GPU | C,E |  | M5 |
| `convert.to_voxel` |  | `0x1205` | Voxel body (occupancy/material grid) from any body. | — | `body:Ref, voxel:Length → vox:VoxelBody` | both | GPU | C,E |  | M3 |
| `convert.mesh_to_solid` |  | `0x1206` | Convert a closed mesh body to a faceted or surface-fitted solid body. | mesh to brep, convert to solid | `mesh:Ref, mode:Enum, tol:Length? → solid:SolidBody` | exact | GPU | C,E |  | M3 |
| `convert.to_solid` | ● | `0x1207` | Convert any closed body (mesh, implicit, voxel) to an exact solid within a deviation budget. | — | `body:Ref, tol:Length, prefer:Enum[] → solid:SolidBody` | exact | GPU | C,E |  | M3 |
| `convert.body_type` |  | `0x1208` | Convert a body to a requested body type using the best published converter. | — | `body:Ref, type:BodyType, tol:Length → body:Body` | both | GPU | C,E |  | M3 |

### `study` (core, family 0x10, 13 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `study.configurations` |  | `0x1000` | Create configurations from a design table, parameter grid, sampled range or explicit list. | design of experiments, wedge, variants | `table:DesignTable/GridSpec → cfgs:Configurations` | control | GPU | C |  | M4 |
| `study.update_configurations` |  | `0x1001` | Update the specification tree for every configuration in one batched GPU pass. | rebuild all configurations | `cfgs:Configurations → results:Configurations<UpdateReport>` | control | GPU | C |  | M4 |
| `study.evaluate` |  | `0x1002` | Evaluate objectives and requirements per configuration. | — | `cfgs:Configurations, objectives:Objective[] → scores:Configurations<Score>` | control | GPU | C | ✓ | M4 |
| `study.filter` |  | `0x1003` | Keep configurations satisfying requirements/predicates. | — | `cfgs:Configurations, pred:Predicate → cfgs:Configurations` | control | GPU | C |  | M4 |
| `study.rank` |  | `0x1004` | Stable top-k configurations by score (tie-break by configuration id). | — | `cfgs:Configurations, scores:Configurations<Score>, k:Int → cfgs:Configurations` | control | GPU | C |  | M4 |
| `study.summarize` |  | `0x1005` | Summarize over configurations (min/max/best/mean/table). | — | `cfgs:Configurations, op:Enum → r:Any` | control | GPU | C |  | M4 |
| `study.optimize` |  | `0x1006` | Optimization of parameters (lm/adam/cma/nelder-mead/simulated annealing) within budget. | product engineering optimizer, design study | `params:Parameter[], objective:Objective, method:Enum, iters:Int → r:OptReport` | both | GPU | C |  | M4 |
| `study.sensitivity` |  | `0x1007` | Design sensitivity d(measure)/d(parameters) by adjoint/forward autodiff. | — | `measure:Ref, params:Parameter[] → g:Tensor` | proxy | GPU | C | ✓ | M4 |
| `study.render` |  | `0x1008` | Render named views (shaded/edges/face ids/depth) per configuration for vision feedback. | — | `items:Ref[], views:View[], mode:Enum → imgs:Image[]` | proxy | GPU | C |  | M4 |
| `study.iterate` |  | `0x1009` | Bounded loop over a sub-program with optional stop condition; consumes budget each pass. | — | `n:Int, body:Program, until:Predicate? → r:Any` | control | GPU | C |  | M4 |
| `study.observe` |  | `0x100A` | Pack a fixed-layout observation tensor of the design state for device-resident models. | — | `items:Ref[], spec:ObsSpec → obs:Tensor` | proxy | GPU | C |  | M4 |
| `study.tolerance_analysis` |  | `0x100B` | Statistical tolerance stack-up: Monte Carlo configurations sampled from tolerance distributions; reports fit/clearance/clash distributions and yield. | stack-up, monte carlo tolerance | `tolerances:Tolerance[], evaluations:EvalRef[], samples:Int, seed:Int → r:ToleranceReport` | control | GPU | C |  | M3 |
| `study.topology_optimization` |  | `0x100C` | Density-based topology optimization (SIMP) of a design region under loads and constraints; produces an implicit body. | generative design, simp | `region:Ref, loads:Load[], supports:Ref[], volume_fraction:Scalar, iters:Int → body:ImplicitBody` | both | GPU | C | ✓ | M4 |

### `evaluate` (core, family 0x0E, 22 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `evaluate.measure_between` | ● | `0x0E00` | Minimum distance and angle between two elements, with witness points. | measure distance, measure angle | `a:Ref, b:Ref → m:Measure` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.measure_item` |  | `0x0E01` | Length, area, radius, angle or position of one element. | measure | `item:Ref → m:Measure` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.measure_inertia` | ● | `0x0E02` | Volume, area, mass, center of gravity, inertia matrix (from material density). | mass properties | `items:Ref[], density:Scalar? → m:MassProps` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.bounding_box` |  | `0x0E03` | Axis-aligned or oriented bounding box. | — | `items:Ref[], oriented:Bool → b:Box` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.curvature_analysis` |  | `0x0E04` | Gaussian/mean/min/max curvature maps on faces; porcupine on curves. | porcupine, curvature map | `items:Ref[], kind:Enum, samples:Int? → k:Field` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.connect_checker` |  | `0x0E05` | G0/G1/G2/G3 continuity deviations along shared or adjacent boundaries. | continuity check, deviation between edges | `a:Ref, b:Ref?, samples:Int → r:Report` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.highlight_lines` |  | `0x0E06` | Reflection/highlight/zebra-line analysis. | zebra stripes, reflection lines | `faces:Ref[], view:View → img:Image` | proxy | GPU | C,E,S |  | M2 |
| `evaluate.draft_analysis` |  | `0x0E07` | Per-face/per-point draft angle relative to pull direction. | — | `faces:Ref[], pull:Vec3, min:Angle → r:Report` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.thickness_analysis` |  | `0x0E08` | Wall-thickness field (ray/SDF based) with min threshold. | — | `body:Ref, min:Length? → t:Field` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.distance_analysis` |  | `0x0E09` | Distance field between two elements (e.g. scan vs CAD). | deviation analysis | `a:Ref, b:Ref → d:Field` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.sectioning` |  | `0x0E0A` | Planar section curves and section properties. | section properties, section view | `items:Ref[], plane:Ref → s:Section` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.clash` |  | `0x0E0B` | Clash/interference and clearance between bodies or products. | interference detection, clearance | `items:Ref[], clearance:Length? → r:ClashReport` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.check` | ● | `0x0E0C` | Geometric/topological validity: closure, orientation, tolerances, self-intersection. | check entity, geometry check | `body:Ref → r:Report` | exact | GPU | C,E,S |  | M2 |
| `evaluate.sample` | ● | `0x0E0D` | Sample signed distance / implicit value or attribute of any body at points. | — | `body:Ref, pts:Vec3[] → v:Scalar[]` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.raycast` | ● | `0x0E0E` | Batched ray casts returning hit element names and parameters. | — | `rays:Ray[] → hits:Hit[]` | proxy | GPU | C,E,S |  | M2 |
| `evaluate.overhang_analysis` |  | `0x0E0F` | Additive manufacturing: overhang angle map and unsupported area for a build direction. | am overhang, support analysis | `body:Ref, build_dir:Vec3, max_angle:Angle → r:Field` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.undercut_analysis` |  | `0x0E10` | Molding: undercut regions for a pull direction (visibility via ray casting). | — | `body:Ref, pull:Vec3 → r:Report` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.tool_accessibility` |  | `0x0E11` | Machining: per-face accessibility from a set of tool axes and tool envelope. | accessibility, 3-axis/5-axis reach | `body:Ref, axes:Vec3[], tool:ToolSpec → r:Report` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.internal_radius_analysis` |  | `0x0E12` | Machining: internal corner radii vs minimum tool radius. | — | `body:Ref, min_r:Length → r:Report` | proxy | GPU | C,E,S | ✓ | M2 |
| `evaluate.stiffness` |  | `0x0E13` | Linear-elastic static analysis (matrix-free CG on voxel/implicit representation): compliance, max displacement, max von Mises stress. Evidence: estimated, discretization reported. | fea, static analysis | `body:Ref, material:Material, loads:Load[], supports:Ref[], voxel:Length → r:PhysicsReport` | proxy | GPU | C,E,S | ✓ | M4 |
| `evaluate.modal` |  | `0x0E14` | Lowest natural frequencies and mode shapes (batched Lanczos/LOBPCG) on voxel/implicit representation. | frequency analysis | `body:Ref, material:Material, supports:Ref[], modes:Int → r:PhysicsReport` | proxy | GPU | C,E,S | ✓ | M4 |
| `evaluate.evaluation` |  | `0x0E15` | Define a named, reusable evaluation (measure expression in AgLang over any evaluate.* results) with declared evidence and gradient. | sensor, knowledge parameter measure, custom measure | `source:String, evidence:Enum → e:EvalRef` | proxy | GPU | C,E,S | ✓ | M2 |

### `query` (core, family 0x0F, 12 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `query.named` |  | `0x0F00` | Resolve stable names (provenance paths or publications) at a revision. | — | `names:String[], scope:Ref? → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.element_type` |  | `0x0F01` | Filter by element type (vertex, edge, face, body, sketch element, point/primitive). | — | `scope:Ref, type:Enum → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.created_by` |  | `0x0F02` | Elements created or modified by a feature, optionally by role. | — | `feature:FeatureRef, role:String? → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.parallel_to` |  | `0x0F03` | Faces/edges whose normal/tangent lies within an angle of a direction. | — | `scope:Ref, dir:Vec3, tol:Angle → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.geometry` |  | `0x0F04` | Predicate on geometry: radius, area, length, convexity, surface type, position. | — | `scope:Ref, pred:Predicate → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.adjacent` |  | `0x0F05` | Topological neighbours (faces of edge, edges of face, tangent-continuous chain). | tangent propagation | `sel:Selection, kind:Enum, depth:Int? → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.contained_in` |  | `0x0F06` | Elements inside a box, sphere, half-space or body. | — | `scope:Ref, region:Region → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.closest_to` |  | `0x0F07` | Elements closest to a point. | — | `scope:Ref, p:Vec3, k:Int? → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.similar` |  | `0x0F08` | Shape/requirement retrieval: k most similar parts or features from libraries and archives by learned embedding. | — | `probe:Ref/Requirements, scope:Enum, k:Int → hits:Hit[]` | proxy | GPU | C,E |  | M4 |
| `query.nth` |  | `0x0F09` | Order a selection by a measure and take a slice. | — | `sel:Selection, key:Enum, take:Int → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.group` |  | `0x0F0A` | Elements of a procedural group or attribute predicate. | — | `scope:Ref, group:String/Predicate → sel:Selection` | proxy | GPU | C,E |  | M2 |
| `query.combine` |  | `0x0F0B` | Union/intersection/difference of selections. | — | `a:Selection, b:Selection, op:Enum → sel:Selection` | proxy | GPU | C,E |  | M2 |

### `file` (core, family 0x13, 13 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `file.export_step` |  | `0x1300` | STEP export (AP242 ed.2+ default, AP214 option) of bodies and products: names, colours/layers, assembly structure, validation properties, optional semantic PMI; receiver profile (generic/solidworks/catia). GPU-encoded (doc 18). | — | `items:Ref[], path:String, schema:Enum?, profile:Enum?, pmi:Bool? → file:File, report:ExportReport` | exact | GPU | — |  | M1 |
| `file.import_step` |  | `0x1301` | STEP AP203/AP214/AP242 import on GPU (lexer → parser → resolver → topology builders) with healing report, product structure, names, colours, PMI and validation-property comparison (doc 18). | — | `path:String, options:ImportOptions? → bodies:Body[], report:ImportReport` | exact | GPU | — |  | M2 |
| `file.export_3mf` |  | `0x1302` | 3MF export for additive manufacturing: Core + Materials & Properties + Beam Lattice + Slice extensions; per-body materials; lattice as beams or mesh. | additive export | `items:Ref[], path:String, extensions:Enum[]? → file:File, report:ExportReport` | exact | GPU | — |  | M3 |
| `file.export_slices` |  | `0x1303` | GPU slicing of bodies (incl. implicit/lattice) into layer contours for AM (3MF Slice extension or CLI/SVG stacks). | slice, layer export | `items:Ref[], layer:Length, build_dir:Vec3, path:String → file:File` | exact | GPU | — |  | M5 |
| `file.export_iges` |  | `0x1304` | IGES export of surfaces/curves. | — | `items:Ref[], path:String → file:File` | exact | GPU | — |  | M5 |
| `file.export_mesh` |  | `0x1305` | STL/3MF/OBJ export from tessellation or mesh bodies. | — | `items:Ref[], format:Enum, path:String → file:File` | exact | GPU | — |  | M1 |
| `file.export_gltf` |  | `0x1306` | glTF scene for viewers with stable-name metadata. | — | `items:Ref[], path:String → file:File` | exact | GPU | — |  | M1 |
| `file.import_mesh` |  | `0x1307` | Import STL/OBJ/3MF/PLY as mesh body or point cloud. | — | `path:String → body:Body` | exact | GPU | — |  | M1 |
| `file.export_vdb` |  | `0x1308` | Export implicit/voxel bodies as OpenVDB. | — | `items:Ref[], path:String → file:File` | exact | GPU | — |  | M3 |
| `file.export_feature_script` |  | `0x1309` | History-preserving handoff: export the specification tree as a replayable script (CATIA EKL/VBA, SolidWorks API, Onshape FeatureScript) via the alias table. | macro export, tree export | `part:Ref, target:Enum, path:String → file:File, report:ExportReport` | exact | GPU | — |  | M5 |
| `file.export_usd` |  | `0x130A` | OpenUSD export: products/components, bodies, attributes as primvars, configurations as variant sets (Omniverse/Houdini Solaris interop). | usd, omniverse | `items:Ref[], path:String, configurations:Configurations? → file:File` | exact | GPU | — |  | M3 |
| `file.save` |  | `0x130B` | Save the document as .agcad (specification tree, UDFs, versions, trace refs); see doc 16. | — | `path:String → file:File` | exact | GPU | — |  | M1 |
| `file.open` |  | `0x130C` | Open a saved .agcad document, migrating older schema/op versions. | — | `path:String → doc:Document` | exact | GPU | — |  | M1 |

### `datum` (stdlib, family 0x02, 4 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `datum.point` |  | `0x0200` | Point: coordinates, on curve/surface, center, tangent, between, intersection. | reference point | `mode:Enum, refs:Ref[], value:Any? → pt:Point` | both | GPU | C,E | ✓ | M1 |
| `datum.line` |  | `0x0201` | Line: point-point, point-direction, angle/normal to curve, tangent, normal to surface, bisecting. | axis, reference axis | `mode:Enum, refs:Ref[], value:Any? → ln:Line` | both | GPU | C,E | ✓ | M1 |
| `datum.plane` |  | `0x0202` | Plane: offset, parallel through point, angle/normal, three points, two lines, normal to curve, mean through points. | reference plane, datum plane | `mode:Enum, refs:Ref[], value:Any? → pl:Plane` | both | GPU | C,E | ✓ | M1 |
| `datum.axis_system` |  | `0x0203` | Axis system (coordinate system) from origin and axes. | coordinate system, frame | `origin:Ref, x:Ref, y:Ref → axs:AxisSystem` | both | GPU | C,E | ✓ | M1 |

### `sketch` (stdlib, family 0x03, 21 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `sketch.create` |  | `0x0300` | Sketch on a plane or planar face, positioned by an axis system. | positioned sketch | `support:Ref, axis:Ref? → sketch:Sketch` | both | GPU | C,E | ✓ | M1 |
| `sketch.point` |  | `0x0301` | Sketch point. | — | `sketch:Sketch, p:Vec2 → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.line` |  | `0x0302` | Line or construction line. | — | `sketch:Sketch, a:Vec2, b:Vec2, construction:Bool? → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.arc` |  | `0x0303` | Arc by center/ends, three points, or tangent. | — | `sketch:Sketch, mode:Enum, pts:Vec2[] → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.circle` |  | `0x0304` | Circle by center-radius, three points, or tritangent. | — | `sketch:Sketch, mode:Enum, pts:Vec2[], r:Length? → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.ellipse` |  | `0x0305` | Ellipse or elliptical arc. | — | `sketch:Sketch, center:Vec2, major:Vec2, ratio:Scalar → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.conic` |  | `0x0306` | Conic by endpoints, tangents and parameter (parabola/hyperbola by rho). | — | `sketch:Sketch, a:Vec2, b:Vec2, shoulder:Vec2, rho:Scalar → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.spline` |  | `0x0307` | Interpolating or control-point spline. | — | `sketch:Sketch, mode:Enum, pts:Vec2[], degree:Int? → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.profile` |  | `0x0308` | Connected line/arc profile (polyline with tangent arcs). | — | `sketch:Sketch, pts:Vec2[], arcs:Int[]? → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.rectangle` |  | `0x0309` | Rectangle, oriented rectangle, parallelogram. | — | `sketch:Sketch, mode:Enum, pts:Vec2[] → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.polygon` |  | `0x030A` | Regular polygon (hexagon etc.). | — | `sketch:Sketch, center:Vec2, sides:Int, r:Length → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.elongated_hole` |  | `0x030B` | Elongated hole / slot (straight or cylindrical), keyhole profile. | slot | `sketch:Sketch, mode:Enum, pts:Vec2[], width:Length → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.text` |  | `0x030C` | Text outline profile. | — | `sketch:Sketch, text:String, font:String, height:Length → e:SketchElement[]` | both | GPU | C,E |  | M5 |
| `sketch.offset` |  | `0x030D` | Offset a chain of sketch elements. | — | `elements:SketchElement[], d:Length → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.trim` |  | `0x030E` | Trim/break/quick-trim elements. | — | `element:SketchElement, pick:Vec2 → e:SketchElement[]` | both | GPU | C,E |  | M1 |
| `sketch.corner` |  | `0x030F` | Sketch corner (fillet) between two elements. | sketch fillet | `a:SketchElement, b:SketchElement, r:Length → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.chamfer` |  | `0x0310` | Sketch chamfer between two elements. | — | `a:SketchElement, b:SketchElement, d:Length → e:SketchElement` | both | GPU | C,E | ✓ | M1 |
| `sketch.symmetry` |  | `0x0311` | Mirror elements about a line. | mirror entities | `elements:SketchElement[], axis:SketchElement → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.project_3d` |  | `0x0312` | Project 3D elements (edges, silhouettes, intersections) into the sketch. | convert entities, use edges | `sketch:Sketch, elements:Ref[], mode:Enum → e:SketchElement[]` | both | GPU | C,E | ✓ | M1 |
| `sketch.solve` | ● | `0x0313` | Batched Levenberg-Marquardt solve of all constraints; reports DOF, over/under-constrained elements. | — | `sketch:Sketch → status:SolveReport` | both | GPU | C,E | ✓ | M1 |
| `sketch.regions` | ● | `0x0314` | Extract closed regions from the solved sketch. | — | `sketch:Sketch → regions:Region[]` | both | GPU | C,E | ✓ | M1 |

### `constraint` (stdlib, family 0x04, 23 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `constraint.coincidence` |  | `0x0400` | Point-point or point-on-curve coincidence. | coincident | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.horizontal` |  | `0x0401` | Horizontal. | — | `a:Ref, b:Ref? → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.vertical` |  | `0x0402` | Vertical. | — | `a:Ref, b:Ref? → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.parallelism` |  | `0x0403` | Parallel lines. | parallel | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.perpendicularity` |  | `0x0404` | Perpendicular lines. | perpendicular | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.tangency` |  | `0x0405` | Tangency (G1). | tangent | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.curvature_continuity` |  | `0x0406` | Curvature continuity (G2) between curves. | curvature | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.equal` |  | `0x0407` | Equal length or radius. | — | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.concentricity` |  | `0x0408` | Shared center. | concentric | `a:Ref, b:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.symmetry` |  | `0x0409` | Symmetric about a line. | symmetric | `a:Ref, b:Ref, axis:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.midpoint` |  | `0x040A` | Point at midpoint. | — | `p:Ref, line:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.fix` |  | `0x040B` | Fix element. | fixed, lock | `a:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.pierce` |  | `0x040C` | Point pierces a 3D curve. | — | `p:Ref, curve:Ref → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.distance` |  | `0x040D` | Distance dimension (driving or reference). | — | `a:Ref, b:Ref, value:Length → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.angle` |  | `0x040E` | Angle dimension. | — | `a:Ref, b:Ref, value:Angle → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.radius` |  | `0x040F` | Radius dimension. | — | `a:Ref, value:Length → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.diameter` |  | `0x0410` | Diameter dimension. | — | `a:Ref, value:Length → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.length` |  | `0x0411` | Length / arc length dimension. | — | `a:Ref, value:Length → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.residual` |  | `0x0412` | Arbitrary differentiable AgLang residual as a constraint. | — | `source:String → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.inequality` |  | `0x0413` | Inequality constraint on any evaluation (e.g. min wall >= 2 mm, clearance >= 0.5 mm); driving in solves/optimization, checked otherwise. | limit, bound | `evaluation:EvalRef, op:Enum, bound:Scalar, tol:Scalar? → c:Constraint` | both | GPU | C,E | ✓ | M1 |
| `constraint.geometric_3d` |  | `0x0414` | 3D geometric constraint between part elements (coincident, parallel, perpendicular, tangent, distance, angle) solved with the part's parameters. | 3d constraint, part constraint | `kind:Enum, a:Ref, b:Ref, value:Scalar? → c:Constraint` | both | GPU | C,E | ✓ | M3 |
| `constraint.set` |  | `0x0415` | Named constraint set (sketch, part, assembly or study scope) with priority/weights for soft constraints. | — | `scope:Ref, constraints:Constraint[], weights:Scalar[]? → set:ConstraintSet` | both | GPU | C,E | ✓ | M2 |
| `constraint.analyze` |  | `0x0416` | Constraint-graph analysis: DOF, structural rank, redundancy and conflict sets, decomposition plan. | — | `set:Ref → r:ConstraintReport` | both | GPU | C,E | ✓ | M1 |

### `wireframe` (stdlib, family 0x05, 22 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `wireframe.line` | ● | `0x0500` | 3D line segment. | — | `a:Ref/Vec3, b:Ref/Vec3 → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.polyline` |  | `0x0501` | Polyline through points with optional corner radii. | — | `pts:Ref[], radii:Length[]? → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.circle` | ● | `0x0502` | Circle or arc (center-radius, three points, tangent, bitangent). | — | `mode:Enum, args:Any → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.conic` | ● | `0x0503` | Ellipse, parabola, hyperbola, general conic. | — | `kind:Enum, support:Ref, p:Scalar[] → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.helix` | ● | `0x0504` | Helix (constant/variable pitch, taper, law). | — | `axis:Ref, start:Ref, pitch:Length/Law, height:Length, taper:Angle? → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.spiral` |  | `0x0505` | Planar spiral. | — | `support:Ref, center:Ref, kind:Enum, p:Scalar[] → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.spline` |  | `0x0506` | 3D spline through points with tangent/curvature conditions. | — | `pts:Ref[], tangents:Vec3[]?, param:Enum? → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.nurbs_curve` | ● | `0x0507` | Explicit NURBS curve (control points, weights, knots). | bspline, bezier | `poles:Vec4[], degree:Int, knots:Scalar[] → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.connect_curve` |  | `0x0508` | Connect curve between two curve ends with G0..G3 conditions. | blend curve | `a:Ref, b:Ref, order:Int → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.corner` |  | `0x0509` | 3D corner (fillet) between two curves. | — | `a:Ref, b:Ref, r:Length → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.parallel_curve` |  | `0x050A` | Offset curve on a support surface. | offset curve | `c:Ref, support:Ref, d:Length/Law → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.projection` | ● | `0x050B` | Project curve/point onto surface (normal or along direction). | — | `c:Ref, support:Ref, dir:Vec3? → curve:Curve[]` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.intersection` | ● | `0x050C` | Intersection of two elements (surfaces, curves). | — | `a:Ref, b:Ref → curve:Curve[]` | both | GPU | C,E,S |  | M1 |
| `wireframe.combine` |  | `0x050D` | Combined curve from two planar-projected curves. | — | `a:Ref, b:Ref → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.reflect_line` |  | `0x050E` | Reflect/silhouette lines of a surface for a direction. | silhouette | `s:Ref, dir:Vec3, angle:Angle → curve:Curve[]` | both | GPU | C,E,S |  | M1 |
| `wireframe.isoparametric` | ● | `0x050F` | Isoparametric curve on a surface. | — | `s:Ref, dir:Enum, t:Scalar → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.spine` |  | `0x0510` | Spine curve for sweeps (normal to sections). | — | `sections:Ref[] → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.fit` | ● | `0x0511` | Least-squares curve fit to points within tolerance. | — | `pts:Ref[], tol:Length, degree:Int → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.join_curves` | ● | `0x0512` | Join connected curves into one wire. | — | `cs:Ref[], tol:Length? → curve:Curve` | both | GPU | C,E,S |  | M1 |
| `wireframe.split_curve` | ● | `0x0513` | Split/trim a curve by points or elements. | — | `c:Ref, tools:Ref[] → curve:Curve[]` | both | GPU | C,E,S |  | M1 |
| `wireframe.curve_smooth` |  | `0x0514` | Smooth a curve within deviation (curvature cleanup). | — | `c:Ref, max_dev:Length → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |
| `wireframe.rebuild_curve` |  | `0x0515` | Rebuild curve with given control-point count/degree; degree elevation; knot insertion. | refit, elevate degree, insert knot | `c:Ref, mode:Enum, args:Any → curve:Curve` | both | GPU | C,E,S | ✓ | M1 |

### `surface` (stdlib, family 0x06, 25 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `surface.extrude` | ● | `0x0600` | Extruded surface from a profile. | — | `profile:Ref, dir:Ref, limits:Interval → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M2 |
| `surface.revolve` | ● | `0x0601` | Revolved surface (exact rational). | — | `profile:Ref, axis:Ref, angles:Interval → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M2 |
| `surface.sphere` |  | `0x0602` | Spherical surface. | — | `center:Ref, r:Length, limits:Any? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M2 |
| `surface.cylinder` |  | `0x0603` | Cylindrical surface. | — | `center:Ref, dir:Ref, r:Length, limits:Interval → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M2 |
| `surface.offset` | ● | `0x0604` | Offset surface (constant or variable) with self-intersection check. | — | `s:Ref, d:Length/Law → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.sweep` | ● | `0x0605` | Sweep: explicit, linear, circular, conic profiles; guides, spine, law, twist. | — | `profile:Ref, guides:Ref[], spine:Ref?, laws:Map? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.multi_sections_surface` | ● | `0x0606` | Multi-sections surface (loft) with guides, spine, tangency and coupling. | loft, lofted surface | `sections:Ref[], guides:Ref[]?, coupling:Enum, ruled:Bool? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.blend` |  | `0x0607` | Blend surface between two curves on supports with G0..G3 continuity and tension. | — | `a:Ref, b:Ref, order:Int[], tension:Scalar[]? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.fill` | ● | `0x0608` | Fill an N-sided boundary with G0..G2 conditions to support surfaces. | n-sided patch, fill surface | `boundary:Ref[], supports:Ref[]?, order:Int[]? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.net_surface` |  | `0x0609` | Surface from a network of curves in two directions (Gordon). | boundary surface, network surface | `u:Ref[], v:Ref[] → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.ruled` |  | `0x060A` | Ruled surface between two curves. | — | `a:Ref, b:Ref → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.planar` |  | `0x060B` | Planar surface from a closed planar boundary. | — | `boundary:Ref → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.nurbs_surface` | ● | `0x060C` | Explicit NURBS/Bezier surface (control net, weights, knots). | — | `poles:Vec4[,], degree:Int2, knots_u:Scalar[], knots_v:Scalar[] → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M1 |
| `surface.join` | ● | `0x060D` | Join surfaces into one surface body within merging distance. | knit, sew | `surfaces:Ref[], tol:Length → surf:SurfaceBody` | both | GPU | C,E,S |  | M3 |
| `surface.healing` |  | `0x060E` | Heal gaps/tangency defects between surfaces within tolerance budget. | heal | `surfaces:Ref[], tol:Length, tangency:Angle? → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.split` | ● | `0x060F` | Split a surface/curve by a cutting element; keep chosen side. | — | `element:Ref, cutter:Ref, keep:Enum → surf:SurfaceBody` | both | GPU | C,E,S |  | M3 |
| `surface.trim` | ● | `0x0610` | Mutually trim two surfaces/curves. | — | `a:Ref, b:Ref, keep:Enum[] → surf:SurfaceBody` | both | GPU | C,E,S |  | M3 |
| `surface.untrim` | ● | `0x0611` | Restore underlying surface bounds or remove inner loops. | — | `face:Ref, loops:Ref[]? → surf:SurfaceBody` | both | GPU | C,E,S |  | M3 |
| `surface.boundary` |  | `0x0612` | Extract boundary curves of a surface. | — | `s:Ref, propagation:Enum → curve:Curve[]` | both | GPU | C,E,S | ✓ | M3 |
| `surface.extract` | ● | `0x0613` | Extract faces/edges (with propagation) as a new element. | — | `elements:Ref[], propagation:Enum → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.extrapolate` |  | `0x0614` | Extrapolate surface/curve boundary (tangent/curvature continuity). | extend surface | `boundary:Ref, s:Ref, length:Length, mode:Enum → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.shape_fillet` | ● | `0x0615` | Surface-surface fillet (bitangent, tritangent, variable, chordal). | — | `a:Ref, b:Ref, r:Length/Law, mode:Enum → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.match_surface` |  | `0x0616` | Deform a surface edge to match another at G0..G3. | — | `s:Ref, edge:Ref, target:Ref, order:Int → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |
| `surface.fair` |  | `0x0617` | Curvature-gradient fairing of a surface network under fixed boundaries and max deviation. | — | `faces:Ref[], fixed:Ref[], max_dev:Length → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M5 |
| `surface.rebuild_surface` |  | `0x0618` | Rebuild surface with given control-point counts/degrees within tolerance. | — | `s:Ref, poles:Int2, degree:Int2 → surf:SurfaceBody` | both | GPU | C,E,S | ✓ | M3 |

### `part` (stdlib, family 0x07, 29 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `part.pad` |  | `0x0700` | Pad: extrude a profile to add material (dimension, up to next/last/plane/surface, mirrored, thick, draft). | extrude, boss-extrude, extruded boss | `profile:Ref, limits:Limits, dir:Ref?, thick:Length? → feature:FeatureRef` | both | GPU | C | ✓ | M1 |
| `part.pocket` |  | `0x0701` | Pocket: extrude a profile to remove material. | extruded cut, cut-extrude | `profile:Ref, limits:Limits, dir:Ref? → feature:FeatureRef` | both | GPU | C | ✓ | M1 |
| `part.shaft` |  | `0x0702` | Shaft: revolve a profile to add material. | revolve, revolved boss | `profile:Ref, axis:Ref, angles:Interval → feature:FeatureRef` | both | GPU | C | ✓ | M1 |
| `part.groove` |  | `0x0703` | Groove: revolve a profile to remove material. | revolved cut | `profile:Ref, axis:Ref, angles:Interval → feature:FeatureRef` | both | GPU | C | ✓ | M1 |
| `part.hole` |  | `0x0704` | Hole: simple, tapered, counterbored, countersunk, counterdrilled; threaded; standard tables. | hole wizard | `support:Ref, positions:Ref[], spec:HoleSpec → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.rib` |  | `0x0705` | Rib: sweep a profile along a center curve to add material. | sweep, swept boss | `profile:Ref, center:Ref, orientation:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.slot` |  | `0x0706` | Slot: sweep a profile along a center curve to remove material. | swept cut | `profile:Ref, center:Ref, orientation:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.multi_sections_solid` |  | `0x0707` | Multi-sections solid (loft) through sections with guides and coupling. | loft, lofted boss | `sections:Ref[], guides:Ref[]?, coupling:Enum, ruled:Bool? → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.removed_multi_sections_solid` |  | `0x0708` | Removed multi-sections solid. | lofted cut | `sections:Ref[], guides:Ref[]?, coupling:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.stiffener` |  | `0x0709` | Stiffener from an open profile (from side / from top). | rib (solidworks), web | `profile:Ref, thickness:Length, mode:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.solid_combine` |  | `0x070A` | Solid from the intersection of two extruded profiles. | — | `a:Ref, b:Ref → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.thread` |  | `0x070B` | Thread/tap on a cylindrical face (cosmetic specification or modeled geometry). | cosmetic thread | `face:Ref, spec:ThreadSpec, modeled:Bool → feature:FeatureRef` | both | GPU | C | ✓ | M5 |
| `part.edge_fillet` | ● | `0x070C` | Edge fillet (constant radius, conic, chordal) with propagation and setbacks. | fillet, round | `edges:Ref[], r:Length, propagation:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.variable_fillet` |  | `0x070D` | Variable-radius fillet with radius law or points. | variable radius fillet | `edges:Ref[], radii:Law → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.face_face_fillet` |  | `0x070E` | Fillet between two faces. | face fillet | `a:Ref, b:Ref, r:Length → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.tritangent_fillet` |  | `0x070F` | Tritangent fillet: remove a face by a fillet tangent to three faces. | full round fillet | `faces:Ref[], remove:Ref → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.chamfer` | ● | `0x0710` | Chamfer (length-angle, length-length) on edges. | — | `edges:Ref[], mode:Enum, a:Scalar, b:Scalar? → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.draft` | ● | `0x0711` | Draft angle (neutral element, parting element, variable, reflect line). | — | `faces:Ref[], neutral:Ref, pull:Ref, angle:Angle/Law → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.shell` | ● | `0x0712` | Shell: hollow with faces to remove and per-face thickness. | — | `remove:Ref[], inner:Length, outer:Length?, per_face:Map? → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.thickness` | ● | `0x0713` | Offset selected faces of a solid. | offset face, move face | `faces:Ref[], d:Length → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.split` |  | `0x0714` | Split a body with a plane or surface; keep a side. | split body | `body:Ref, cutter:Ref, keep:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.thick_surface` | ● | `0x0715` | Thicken a surface into a solid. | thicken | `s:Ref, t1:Length, t2:Length? → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.close_surface` | ● | `0x0716` | Close a surface body into a solid. | knit to solid | `s:Ref → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.sew_surface` |  | `0x0717` | Sew a surface onto a solid (add/remove material up to it). | replace face, surface cut | `s:Ref, keep:Enum → feature:FeatureRef` | both | GPU | C | ✓ | M2 |
| `part.remove_face` | ● | `0x0718` | Remove faces and heal by extending neighbours. | delete face, defeature | `faces:Ref[], keep:Ref[]? → feature:FeatureRef` | both | GPU | C |  | M2 |
| `part.replace_face` |  | `0x0719` | Replace faces with a surface. | — | `faces:Ref[], target:Ref → feature:FeatureRef` | both | GPU | C |  | M2 |
| `part.lattice` |  | `0x071A` | Lattice infill of a region (TPMS gyroid/diamond, strut, custom UDF cell) with grading law; produces an implicit body. | infill, gyroid | `region:Ref, cell:Enum/UdfDef, size:Length, grading:Law? → body:ImplicitBody` | both | GPU | C | ✓ | M3 |
| `part.implicit_body` | ● | `0x071B` | Implicit body defined by an AgLang distance function or blend tree (smooth union/subtract/intersect, shell, offset). | sdf, field, vdb | `source:String/Network, bounds:Box → body:ImplicitBody` | both | GPU | C | ✓ | M1 |
| `part.primitive` | ● | `0x071C` | Solid primitives: box, cylinder, cone, sphere, torus, wedge, prism. | — | `kind:Enum, axis:Ref, p:Scalar[] → feature:FeatureRef` | both | GPU | C | ✓ | M1 |

### `boolean` (stdlib, family 0x08, 7 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `boolean.assemble` |  | `0x0800` | Assemble a body into another (operation sign follows feature type). | — | `target:Ref, tool:Ref → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.add` | ● | `0x0801` | Add (union) bodies. | union, combine add | `target:Ref, tools:Ref[] → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.remove` | ● | `0x0802` | Remove (subtract) bodies. | subtract, difference, combine subtract | `target:Ref, tools:Ref[] → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.intersect` | ● | `0x0803` | Intersect bodies (empty result is valid). | common, combine common | `target:Ref, tools:Ref[] → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.union_trim` |  | `0x0804` | Union trim: union while choosing faces/volumes to keep or remove. | — | `target:Ref, tool:Ref, remove:Ref[], keep:Ref[] → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.remove_lump` |  | `0x0805` | Remove disconnected lumps from a body. | — | `body:Ref, remove:Ref[] → feature:FeatureRef` | both | GPU | C |  | M2 |
| `boolean.smooth_boolean` | ● | `0x0806` | Blended boolean on implicit bodies (smooth union/subtract/intersect with radius). | — | `a:Ref, b:Ref, op:Enum, r:Length → body:ImplicitBody` | proxy | GPU | C | ✓ | M1 |

### `transform` (stdlib, family 0x09, 12 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `transform.translation` |  | `0x0900` | Translate bodies or elements (copy optional). | move, move/copy | `items:Ref[], dir:Ref, d:Length, copy:Bool → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.rotation` |  | `0x0901` | Rotate bodies or elements about an axis. | — | `items:Ref[], axis:Ref, angle:Angle, copy:Bool → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.symmetry` |  | `0x0902` | Symmetry of bodies or elements about a point, line or plane. | — | `items:Ref[], ref:Ref, copy:Bool → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.mirror` |  | `0x0903` | Mirror features or bodies about a plane and add to the body. | — | `items:Ref[], plane:Ref → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.rectangular_pattern` |  | `0x0904` | Rectangular pattern in one or two directions with instance skipping and staggering. | linear pattern | `items:Ref[], dirs:Ref[], counts:Int[], spacing:Length[] → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.circular_pattern` |  | `0x0905` | Circular pattern (angular and radial crowns). | — | `items:Ref[], axis:Ref, count:Int, angle:Angle → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.user_pattern` |  | `0x0906` | User pattern: instances at sketch points or table positions. | sketch driven pattern, table driven pattern, copy to points | `items:Ref[], positions:Ref → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.curve_pattern` |  | `0x0907` | Pattern along a curve. | curve driven pattern | `items:Ref[], curve:Ref, count:Int, align:Enum → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.fill_pattern` |  | `0x0908` | Fill a region with a seed pattern (square, hex, circular layouts). | — | `items:Ref[], region:Ref, layout:Enum, spacing:Length → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.scaling` |  | `0x0909` | Uniform scaling about a point/plane. | scale | `items:Ref[], ref:Ref, ratio:Scalar → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.affinity` | ● | `0x090A` | Non-uniform scaling in an axis system. | non-uniform scale | `items:Ref[], axis:Ref, ratios:Vec3 → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |
| `transform.axis_to_axis` | ● | `0x090B` | Transform elements from one axis system to another. | — | `items:Ref[], from:Ref, to:Ref → feature:FeatureRef` | both | GPU | C,E | ✓ | M2 |

### `direct` (stdlib, family 0x0A, 4 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `direct.push_pull` |  | `0x0A00` | Push/pull faces along their normals. | live shape, press pull | `faces:Ref[], d:Length → feature:FeatureRef` | both | GPU | C | ✓ | M5 |
| `direct.move_face` |  | `0x0A01` | Rigidly move/rotate faces, re-solving neighbours. | — | `faces:Ref[], xform:Xform → feature:FeatureRef` | both | GPU | C | ✓ | M5 |
| `direct.defeature` |  | `0x0A02` | Remove fillets/holes/chamfers below size thresholds. | — | `body:Ref, kinds:Enum[], max_size:Length → feature:FeatureRef` | both | GPU | C |  | M5 |
| `direct.heal_body` |  | `0x0A03` | Heal imported bodies: close gaps, merge tolerant edges, remove slivers within budget. | import diagnostics | `body:Ref, tol:Length → feature:FeatureRef` | both | GPU | C | ✓ | M5 |

### `mesh` (stdlib, family 0x0B, 10 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `mesh.primitive` |  | `0x0B00` | Mesh box/sphere/cylinder/cone/torus/grid. | — | `kind:Enum, axis:Ref, p:Scalar[], res:Int[] → mesh:MeshBody` | proxy | GPU | C,E,S |  | M1 |
| `mesh.tessellate` | ● | `0x0B01` | Adaptive watertight tessellation of bodies with chordal/angular bounds. | — | `items:Ref[], sag:Length, angle:Angle → mesh:MeshBody` | proxy | GPU | C,E,S |  | M1 |
| `mesh.weld` |  | `0x0B02` | Weld coincident vertices within tolerance (spatial hash). | fuse points | `mesh:Ref, tol:Length → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.normals` |  | `0x0B03` | Vertex normals with crease angle. | — | `mesh:Ref, crease:Angle → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.subdivide` |  | `0x0B04` | Linear, Loop or Catmull-Clark subdivision with creases. | — | `mesh:Ref, scheme:Enum, levels:Int → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.decimate` | ● | `0x0B05` | Quadric-error decimation to target count or error. | polyreduce | `mesh:Ref, target:Scalar → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.remesh` | ● | `0x0B06` | Isotropic remesh to edge length. | — | `mesh:Ref, edge:Length → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.smooth` |  | `0x0B07` | Laplacian/Taubin smoothing. | — | `mesh:Ref, scheme:Enum, iters:Int → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.mesh_boolean` | ● | `0x0B08` | Robust mesh boolean (winding-number based). | — | `a:Ref, b:Ref, op:Enum → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |
| `mesh.repair` |  | `0x0B09` | Fill holes, fix flipped/non-manifold elements. | — | `mesh:Ref → mesh:MeshBody` | proxy | GPU | C,E,S |  | M3 |

### `assembly` (stdlib, family 0x0D, 9 ops)

| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |
|---|---|---|---|---|---|---|---|---|---|---|
| `assembly.component` |  | `0x0D00` | Insert a component (part or product instance) with a position. | instance, insert part | `ref:Ref, xform:Xform → comp:Component` | both | GPU | C,E | ✓ | M5 |
| `assembly.coincidence` |  | `0x0D01` | Coincidence constraint (axes, planes, points). | coincident mate, concentric mate | `a:Ref, b:Ref → c:AsmConstraint` | both | GPU | C,E | ✓ | M5 |
| `assembly.contact` |  | `0x0D02` | Contact constraint (planar, line, point contact). | — | `a:Ref, b:Ref → c:AsmConstraint` | both | GPU | C,E | ✓ | M5 |
| `assembly.offset` |  | `0x0D03` | Offset constraint. | distance mate | `a:Ref, b:Ref, d:Length → c:AsmConstraint` | both | GPU | C,E | ✓ | M5 |
| `assembly.angle` |  | `0x0D04` | Angle constraint (incl. parallel, perpendicular). | angle mate | `a:Ref, b:Ref, angle:Angle → c:AsmConstraint` | both | GPU | C,E | ✓ | M5 |
| `assembly.fix` |  | `0x0D05` | Fix component in space. | ground | `comp:Ref → c:AsmConstraint` | both | GPU | C,E | ✓ | M5 |
| `assembly.mechanism_joint` |  | `0x0D06` | Kinematic joint (revolute, prismatic, cylindrical, gear, rack, cam). | mate, joint | `kind:Enum, a:Ref, b:Ref, params:Map? → j:Joint` | both | GPU | C,E | ✓ | M6 |
| `assembly.update` |  | `0x0D07` | Batched assembly constraint solve; reports DOF and conflicts. | solve mates | `product:Ref → status:SolveReport` | both | GPU | C,E | ✓ | M5 |
| `assembly.exploded_view` |  | `0x0D08` | Exploded view scenes (presentation only). | — | `product:Ref, steps:Xform[] → view:View` | both | GPU | C,E |  | M5 |
<!-- END GENERATED CATALOG -->

## Normative requirements

| ID | Requirement |
|---|---|
| AG-03-001 | Every op MUST declare all manifest fields in `op-catalog.json`. |
| AG-03-002 | Opcodes MUST be append-only; semantic changes MUST bump the op `version`. |
| AG-03-003 | Only `udf` compile-time ops MAY have backend `host`. |
| AG-03-004 | Every alias MUST resolve to exactly one canonical op within its family. |
