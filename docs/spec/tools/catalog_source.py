"""Authoring source for docs/spec/op-catalog.json (normative op list).

Ontology: canonical names follow CATIA V5 / 3DEXPERIENCE (Part Design, Generative Shape
Design, Knowledgeware, Assembly Design). `aliases` carry SolidWorks / Onshape / Houdini /
generic names so that agents trained on any of them resolve to the same op.
The `proc` family adds Houdini-style procedural networks (attributes, groups, wrangles,
for-each, solver) on the same bodies.

Layers:
  core   = document/history, knowledgeware, UDFs, procedural networks, conversion,
           design studies, analysis, queries, files: the substrate everything builds on
  stdlib = standard features (sketcher, wireframe, surfaces, part design, booleans,
           transformations, direct edit, mesh, assembly), written in AgLang on core;
           viewable, copyable and modifiable by models
  model  = UDFs published at runtime via `udf.publish` (never listed here)

Regenerate:
    python3 docs/spec/tools/catalog_source.py > docs/spec/op-catalog.json
    python3 docs/spec/tools/render_catalog.py
    python3 docs/spec/tools/check_spec.py

Opcode = (family_id << 8) | index. Append-only once M0 ships; deprecate, never renumber.
"""
import json
import sys

CATALOG_VERSION = "0.3.1"
C, E, S = "configuration", "entity", "sample"

FAMILIES = {
    # name:       (id,   phase,     backend,  axes,      diff,  milestone, effect)
    "document":   (0x00, "control", "gpu",    [C],       False, "M1", "doc_write"),
    "knowledge":  (0x01, "both",    "gpu",    [C],       True,  "M1", "doc_write"),
    "datum":      (0x02, "both",    "gpu",    [C, E],    True,  "M1", "doc_write"),
    "sketch":     (0x03, "both",    "gpu",    [C, E],    True,  "M1", "doc_write"),
    "constraint": (0x04, "both",    "gpu",    [C, E],    True,  "M1", "doc_write"),
    "wireframe":  (0x05, "both",    "gpu",    [C, E, S], True,  "M1", "doc_write"),
    "surface":    (0x06, "both",    "gpu",    [C, E, S], True,  "M3", "doc_write"),
    "part":       (0x07, "both",    "gpu",    [C],       True,  "M2", "doc_write"),
    "boolean":    (0x08, "both",    "gpu",    [C],       False, "M2", "doc_write"),
    "transform":  (0x09, "both",    "gpu",    [C, E],    True,  "M2", "doc_write"),
    "direct":     (0x0A, "both",    "gpu",    [C],       True,  "M5", "doc_write"),
    "mesh":       (0x0B, "proxy",   "gpu",    [C, E, S], False, "M3", "doc_write"),
    "proc":       (0x0C, "both",    "gpu",    [C, E, S], True,  "M3", "doc_write"),
    "assembly":   (0x0D, "both",    "gpu",    [C, E],    True,  "M5", "doc_write"),
    "evaluate":   (0x0E, "proxy",   "gpu",    [C, E, S], True,  "M2", "pure"),
    "query":      (0x0F, "proxy",   "gpu",    [C, E],    False, "M2", "pure"),
    "study":      (0x10, "control", "gpu",    [C],       False, "M4", "pure"),
    "udf":        (0x11, "control", "gpu",    [C],       False, "M1", "doc_write"),
    "convert":    (0x12, "both",    "gpu",    [C, E],    False, "M3", "doc_write"),
    "file":       (0x13, "exact",   "gpu",       [],        False, "M1", "io"),
}
CORE = {"document", "knowledge", "proc", "evaluate", "query", "study", "udf", "convert", "file"}
ORDER = ["document", "knowledge", "udf", "proc", "convert", "study", "evaluate", "query", "file",
         "datum", "sketch", "constraint", "wireframe", "surface", "part", "boolean", "transform",
         "direct", "mesh", "assembly"]

# Kernel primitives: geometry ops implemented natively on the GPU exact/proxy kernel.
# Every other geometry op is a standard UDF composed from these (readable via udf.view_source).
PRIMITIVES = {
    "sketch.solve", "sketch.regions",
    "wireframe.line", "wireframe.circle", "wireframe.conic", "wireframe.helix", "wireframe.nurbs_curve",
    "wireframe.intersection", "wireframe.projection", "wireframe.isoparametric", "wireframe.join_curves",
    "wireframe.split_curve", "wireframe.fit",
    "surface.extrude", "surface.revolve", "surface.nurbs_surface", "surface.offset", "surface.sweep",
    "surface.multi_sections_surface", "surface.fill", "surface.join", "surface.split", "surface.trim",
    "surface.untrim", "surface.extract", "surface.shape_fillet",
    "part.primitive", "part.edge_fillet", "part.chamfer", "part.shell", "part.draft", "part.thickness",
    "part.thick_surface", "part.close_surface", "part.remove_face", "part.implicit_body",
    "boolean.add", "boolean.remove", "boolean.intersect", "boolean.smooth_boolean",
    "transform.axis_to_axis", "transform.affinity",
    "mesh.tessellate", "mesh.mesh_boolean", "mesh.remesh", "mesh.decimate",
    "convert.implicit_to_mesh", "convert.body_to_implicit", "convert.mesh_to_implicit", "convert.to_solid",
    "evaluate.measure_between", "evaluate.measure_inertia", "evaluate.check", "evaluate.raycast", "evaluate.sample",
}

CF = ["INVALID_PARAMETER", "NONFINITE_PARAMETER", "STALE_REVISION", "STALE_REFERENCE", "BUDGET_EXHAUSTED"]
TF = CF + ["TOPOLOGY_FAILED", "GEOMETRY_CHECK_FAILED", "CAPACITY_EXCEEDED"]
SF = CF + ["SOLVER_DIVERGED", "UNDER_CONSTRAINED", "OVER_CONSTRAINED"]
SAVE = CF + ["REQUIREMENT_FAILED", "GEOMETRY_CHECK_FAILED", "TOPOLOGY_FAILED", "PENDING_EXACT"]
CODE = CF + ["COMPILE_FAILED", "SANDBOX_VIOLATION"]


def A(*names):
    return {"aliases": list(names)}


def M(d, **kw):
    d = dict(d)
    d.update(kw)
    return d


# (name, summary, inputs, outputs, overrides)
OPS = {
 "document": [
  ("new_part", "Create a Part (CATPart): origin, default planes, PartBody, feature tree.", ["name:String", "units:Enum?"], ["part:Part"], A("new part studio")),
  ("new_product", "Create a Product (assembly document).", ["name:String"], ["product:Product"], A("new assembly")),
  ("body", "Insert a Body (solid feature container) into the part.", ["part:Part", "name:String"], ["body:BodyRef"], A("insert body", "multibody")),
  ("geometrical_set", "Insert a Geometrical Set (unordered wireframe/surface container) or Ordered Geometrical Set.", ["part:Part", "name:String", "ordered:Bool"], ["set:SetRef"], A("surface body folder")),
  ("define_in_work_object", "Make a feature the current work object (rollback bar); later features are not rebuilt.", ["feature:FeatureRef"], [], A("rollback", "rollback bar")),
  ("deactivate", "Deactivate or activate features.", ["features:FeatureRef[]", "active:Bool"], [], A("suppress", "unsuppress")),
  ("material", "Apply material (density, mechanical properties, appearance) to bodies or parts.", ["targets:Ref[]", "material:Material"], [], {}),
  ("publication", "Publish an element under a stable public name for external references.", ["element:Ref", "name:String"], ["pub:Publication"], A("published geometry")),
  ("edit", "Open the workspace for editing (private, may be temporarily invalid).", ["branch:Branch"], ["workspace:Workspace"], {}),
  ("update", "Update (rebuild) the feature tree and run edit-phase Design Checker rules.", ["workspace:Workspace?"], ["report:UpdateReport"], M(A("rebuild", "regenerate", "cook"), fail=SAVE)),
  ("save_version", "Run save-phase checks; on success create an immutable version (optional label), on failure keep the workspace.", ["workspace:Workspace?", "label:String?"], ["version:Version"], M(A("commit"), fail=SAVE)),
  ("discard_edit", "Discard workspace changes since the last version.", ["workspace:Workspace?"], [], A("abort", "revert")),
  ("checkpoint", "Return an update report to the caller and wait for the next decision.", ["label:String"], ["report:UpdateReport"], {}),
  ("branch", "Create branches from a version, sharing history copy-on-write.", ["version:Version", "count:Int"], ["branches:Configurations<Branch>"], A("fork")),
  ("switch_branch", "Make a branch the active branch.", ["branch:Branch"], [], {}),
  ("merge", "Merge feature-level changes between branches.", ["from:Branch", "into:Branch"], ["version:Version"], M({}, fail=CF + ["MERGE_CONFLICT"])),
  ("delete", "Delete a feature, body, set or branch handle; budget is not refunded.", ["handle:Handle"], [], {}),
  ("specification_tree", "Return the specification tree: features, parameters, bodies, names, versions.", ["target:Handle"], ["tree:SpecTree"], M(A("feature tree", "feature manager", "node network view"), effect="pure")),
  ("undo", "Undo the last workspace change.", ["workspace:Workspace?"], [], {}),
  ("redo", "Redo an undone change.", ["workspace:Workspace?"], [], {}),
  ("finish", "Caller declares the design finished; delivery checks run; does not imply success.", [], ["report:DeliveryReport"], {}),
  ("comment", "Attach a review comment to a version and stable element names.", ["version:Version", "elements:String[]", "text:String"], ["comment:Comment"], M({}, milestone="M1")),
  ("approve", "Record a human or delegated approval/rejection of a version (gates delivery when requirements demand it).", ["version:Version", "decision:Enum", "note:String?"], ["approval:Approval"], M({}, milestone="M1")),
  ("diff", "Diff two versions: features, parameters, elements (with lineage), evaluations; optional diff renders.", ["a:Version", "b:Version"], ["r:DiffReport"], M({}, effect="pure", milestone="M1")),
 ],
 "knowledge": [
  ("parameter", "Create/modify a parameter (length, angle, real, integer, boolean, string, list) with units and range.", ["name:String", "type:Enum?", "value:Any", "range:Interval?"], ["param:Parameter"], A("global variable", "channel", "spare parameter")),
  ("formula", "Formula driving a parameter or feature dimension from other parameters.", ["target:Ref", "expr:String"], ["f:Formula"], A("equation", "expression")),
  ("law", "Law curve (value along a parameter) for variable sweeps, fillets, gradings.", ["kind:Enum", "points:Vec2[]|Expr"], ["law:Law"], {}),
  ("rule", "Knowledge rule: conditional logic setting parameters/activation on update.", ["source:String"], ["rule:Rule"], M(A("design rule"), fail=CODE)),
  ("check", "Knowledge check: user/model-authored pass/fail condition evaluated on update (adds evidence, never replaces requirements).", ["source:String", "severity:Enum"], ["check:Check"], M({}, fail=CODE)),
  ("reaction", "Reaction: action script triggered by an event (parameter change, feature update).", ["event:Event", "source:String"], ["reaction:Reaction"], M({}, fail=CODE, differentiable=False)),
  ("design_table", "Design table: named rows of parameter values; rows are configurations.", ["params:Parameter[]", "rows:Any[][]"], ["table:DesignTable"], A("configurations table", "excel design table")),
  ("power_copy", "Instantiate a PowerCopy (feature group template) with new inputs.", ["template:Ref", "inputs:Map"], ["features:FeatureRef[]"], A("copy features", "derived")),
  ("tolerance", "Dimensional or GD&T tolerance on a parameter, dimension or feature (size, position, profile, flatness, ...), with datum references and distribution.", ["target:Ref", "spec:ToleranceSpec", "datums:Ref[]?", "distribution:Enum?"], ["tol:Tolerance"], M(A("gd&t", "fta", "3d annotation"), milestone="M3")),
  ("catalog_insert", "Insert a standard part from a catalog (ISO/DIN fasteners, bearings, profiles) by designation and parameters.", ["catalog:String", "designation:String", "params:Map?"], ["feature:FeatureRef"], M(A("toolbox", "standard parts", "catalog browser"), milestone="M4")),
 ],
 "udf": [
  ("define", "Define a User Defined Feature in AgLang (inputs, parameters, outputs, naming roles, checks) or as a procedural network.", ["source:String|Network", "lang:Enum"], ["definition:UdfDef"], M(A("user feature", "custom feature", "knowledge template", "digital asset", "hda"), backend="host", fail=CODE)),
  ("compile", "Compile a UDF to batched GPU kernels (vectorized over configurations) or device bytecode.", ["definition:UdfDef", "target:Enum"], ["kernel:Kernel"], M({}, backend="host", fail=CF + ["COMPILE_FAILED"])),
  ("instantiate", "Instantiate a UDF in the specification tree with inputs and parameter values.", ["definition:UdfDef", "inputs:Map", "params:Map"], ["feature:FeatureRef"], M(A("insert custom feature"), batch_axes=[C, E], differentiable=True)),
  ("sample", "Sample a UDF's geometry functions (implicit value, surface point, derivatives) at points/parameters.", ["feature:FeatureRef", "at:Tensor", "order:Int?"], ["v:Tensor"], M({}, batch_axes=[C, E, S], phase="proxy", effect="pure", differentiable=True)),
  ("publish", "Publish a UDF to a feature catalog after conformance checks pass.", ["definition:UdfDef", "manifest:FeatureManifest", "scope:Enum"], ["feature:FeatureTypeRef"], M(A("catalog", "asset library"), phase="exact", milestone="M4", fail=CF + ["COMPILE_FAILED", "CONFORMANCE_FAILED"])),
  ("view_source", "Return the AgLang source of any standard feature or UDF (it may be copied and modified).", ["feature:FeatureTypeRef"], ["source:String"], M({}, effect="pure", backend="host")),
 ],
 "proc": [
  ("network", "Create a procedural network (node graph) that cooks to one or more bodies; appears as one feature in the tree.", ["name:String", "inputs:Ref[]"], ["net:Network"], A("sop network", "geometry node", "houdini engine asset")),
  ("node", "Add a node to a network: any catalog op or UDF, wired by input/output ports.", ["net:Network", "op:OpRef", "inputs:Port[]", "params:Map"], ["node:Node"], {}),
  ("wrangle", "Run an AgLang snippet over points, vertices, primitives, faces, edges or detail, reading/writing attributes.", ["net:Network", "over:Enum", "source:String"], ["node:Node"], M(A("attribute wrangle", "vex", "point wrangle"), fail=CODE)),
  ("attribute_create", "Create an attribute (point/vertex/primitive/face/edge/detail) with type and default.", ["target:Ref", "class:Enum", "name:String", "type:Enum"], ["attr:Attribute"], {}),
  ("attribute_promote", "Promote/demote attributes between classes with a reduction.", ["target:Ref", "name:String", "from:Enum", "to:Enum", "method:Enum"], ["attr:Attribute"], {}),
  ("attribute_transfer", "Transfer attributes from one geometry to another by proximity.", ["src:Ref", "dst:Ref", "names:String[]", "radius:Length"], ["attr:Attribute[]"], {}),
  ("group", "Create a group (named element set) by expression, query or bounding region.", ["target:Ref", "class:Enum", "pred:Predicate|Query"], ["group:Group"], A("group create", "selection set")),
  ("blast", "Delete or keep elements of a group.", ["target:Ref", "group:Group", "keep:Bool"], ["geo:Geometry"], A("delete by group")),
  ("scatter", "Scatter points on curves, surfaces or volumes by density attribute, with seed.", ["target:Ref", "count:Int|DensityAttr", "seed:Int"], ["pts:PointCloud"], {}),
  ("copy_to_points", "Instance or copy geometry onto points, oriented by point attributes (N, up, orient, pscale).", ["geo:Ref", "pts:Ref", "pack:Bool"], ["geo:Geometry"], A("copy stamp", "instancer", "user pattern")),
  ("for_each", "For-each block over pieces, points, groups or a count, with feedback option.", ["net:Network", "over:Enum", "body:Network", "feedback:Bool"], ["node:Node"], A("foreach loop", "for each named primitive")),
  ("solver", "Iterative solver block: the network's output feeds its next iteration (growth, relaxation, simulation-like rules).", ["net:Network", "body:Network", "iterations:Int", "until:Predicate?"], ["node:Node"], A("solver sop", "feedback loop")),
  ("merge", "Merge geometry streams.", ["inputs:Ref[]"], ["geo:Geometry"], {}),
  ("switch", "Choose an input by parameter or expression (procedural variant).", ["inputs:Ref[]", "index:Expr"], ["geo:Geometry"], M({}, differentiable=False)),
  ("poly_extrude", "Extrude polygon faces with inset, twist, per-element attributes.", ["mesh:Ref", "group:Group?", "distance:Length|Attr", "inset:Length?"], ["mesh:MeshBody"], {}),
  ("resample", "Resample curves by length or count; add tangent/curve-u attributes.", ["curves:Ref", "seg:Length|Int"], ["curves:Geometry"], {}),
  ("sweep_along", "Sweep cross-sections along curves using point attributes for frames and scale.", ["curves:Ref", "section:Ref"], ["geo:Geometry"], A("sweep sop")),
  ("cook", "Cook a network (or node) for all configurations; outputs flow into the specification tree.", ["net:Network"], ["bodies:Body[]"], M({}, fail=CF + ["TOPOLOGY_FAILED", "CAPACITY_EXCEEDED"])),
 ],
 "convert": [
  ("implicit_to_mesh", "Convert an implicit body to a watertight mesh body (dual contouring / adaptive marching).", ["body:Ref", "res:Length", "bounds:Box"], ["mesh:MeshBody"], A("convert to mesh", "convert vdb")),
  ("implicit_to_solid", "Convert an implicit body to an exact solid body: feature-preserving mesh then analytic/NURBS face fitting with deviation bound.", ["body:Ref", "tol:Length"], ["solid:SolidBody"], M(A("implicit to brep"), backend="gpu", phase="exact", fail=TF)),
  ("body_to_implicit", "Signed-distance implicit view of a solid/surface body (sparse bricks + exact refinement).", ["body:Ref", "voxel:Length"], ["implicit:ImplicitBody"], A("vdb from polygons", "sdf")),
  ("mesh_to_implicit", "Implicit body from a mesh body (generalized winding number / signed distance).", ["mesh:Ref", "voxel:Length"], ["implicit:ImplicitBody"], {}),
  ("point_cloud_to_implicit", "Implicit body fitted to a point cloud with normals (RBF or neural).", ["pts:Ref", "tol:Length"], ["implicit:ImplicitBody"], M({}, milestone="M5")),
  ("to_voxel", "Voxel body (occupancy/material grid) from any body.", ["body:Ref", "voxel:Length"], ["vox:VoxelBody"], {}),
  ("mesh_to_solid", "Convert a closed mesh body to a faceted or surface-fitted solid body.", ["mesh:Ref", "mode:Enum", "tol:Length?"], ["solid:SolidBody"], M(A("mesh to brep", "convert to solid"), backend="gpu", phase="exact", fail=TF)),
  ("to_solid", "Convert any closed body (mesh, implicit, voxel) to an exact solid within a deviation budget.", ["body:Ref", "tol:Length", "prefer:Enum[]"], ["solid:SolidBody"], M({}, backend="gpu", phase="exact", fail=TF)),
  ("body_type", "Convert a body to a requested body type using the best published converter.", ["body:Ref", "type:BodyType", "tol:Length"], ["body:Body"], M({}, fail=TF + ["UNSUPPORTED_OP"])),
 ],
 "study": [
  ("configurations", "Create configurations from a design table, parameter grid, sampled range or explicit list.", ["table:DesignTable|GridSpec"], ["cfgs:Configurations"], A("design of experiments", "wedge", "variants")),
  ("update_configurations", "Update the specification tree for every configuration in one batched GPU pass.", ["cfgs:Configurations"], ["results:Configurations<UpdateReport>"], M(A("rebuild all configurations"), effect="doc_write")),
  ("evaluate", "Evaluate objectives and requirements per configuration.", ["cfgs:Configurations", "objectives:Objective[]"], ["scores:Configurations<Score>"], M({}, differentiable=True)),
  ("filter", "Keep configurations satisfying requirements/predicates.", ["cfgs:Configurations", "pred:Predicate"], ["cfgs:Configurations"], {}),
  ("rank", "Stable top-k configurations by score (tie-break by configuration id).", ["cfgs:Configurations", "scores:Configurations<Score>", "k:Int"], ["cfgs:Configurations"], {}),
  ("summarize", "Summarize over configurations (min/max/best/mean/table).", ["cfgs:Configurations", "op:Enum"], ["r:Any"], {}),
  ("optimize", "Optimization of parameters (lm/adam/cma/nelder-mead/simulated annealing) within budget.", ["params:Parameter[]", "objective:Objective", "method:Enum", "iters:Int"], ["r:OptReport"], M(A("product engineering optimizer", "design study"), phase="both", effect="doc_write", fail=SF)),
  ("sensitivity", "Design sensitivity d(measure)/d(parameters) by adjoint/forward autodiff.", ["measure:Ref", "params:Parameter[]"], ["g:Tensor"], M({}, phase="proxy", differentiable=True)),
  ("render", "Render named views (shaded/edges/face ids/depth) per configuration for vision feedback.", ["items:Ref[]", "views:View[]", "mode:Enum"], ["imgs:Image[]"], M({}, phase="proxy")),
  ("iterate", "Bounded loop over a sub-program with optional stop condition; consumes budget each pass.", ["n:Int", "body:Program", "until:Predicate?"], ["r:Any"], M({}, effect="doc_write")),
  ("observe", "Pack a fixed-layout observation tensor of the design state for device-resident models.", ["items:Ref[]", "spec:ObsSpec"], ["obs:Tensor"], M({}, phase="proxy")),
  ("tolerance_analysis", "Statistical tolerance stack-up: Monte Carlo configurations sampled from tolerance distributions; reports fit/clearance/clash distributions and yield.", ["tolerances:Tolerance[]", "evaluations:EvalRef[]", "samples:Int", "seed:Int"], ["r:ToleranceReport"], M(A("stack-up", "monte carlo tolerance"), milestone="M3")),
  ("topology_optimization", "Density-based topology optimization (SIMP) of a design region under loads and constraints; produces an implicit body.", ["region:Ref", "loads:Load[]", "supports:Ref[]", "volume_fraction:Scalar", "iters:Int"], ["body:ImplicitBody"], M(A("generative design", "simp"), phase="both", effect="doc_write", differentiable=True, milestone="M4", fail=SF)),
 ],
 "evaluate": [
  ("measure_between", "Minimum distance and angle between two elements, with witness points.", ["a:Ref", "b:Ref"], ["m:Measure"], A("measure distance", "measure angle")),
  ("measure_item", "Length, area, radius, angle or position of one element.", ["item:Ref"], ["m:Measure"], A("measure")),
  ("measure_inertia", "Volume, area, mass, center of gravity, inertia matrix (from material density).", ["items:Ref[]", "density:Scalar?"], ["m:MassProps"], A("mass properties")),
  ("bounding_box", "Axis-aligned or oriented bounding box.", ["items:Ref[]", "oriented:Bool"], ["b:Box"], {}),
  ("curvature_analysis", "Gaussian/mean/min/max curvature maps on faces; porcupine on curves.", ["items:Ref[]", "kind:Enum", "samples:Int?"], ["k:Field"], A("porcupine", "curvature map")),
  ("connect_checker", "G0/G1/G2/G3 continuity deviations along shared or adjacent boundaries.", ["a:Ref", "b:Ref?", "samples:Int"], ["r:Report"], A("continuity check", "deviation between edges")),
  ("highlight_lines", "Reflection/highlight/zebra-line analysis.", ["faces:Ref[]", "view:View"], ["img:Image"], M(A("zebra stripes", "reflection lines"), differentiable=False)),
  ("draft_analysis", "Per-face/per-point draft angle relative to pull direction.", ["faces:Ref[]", "pull:Vec3", "min:Angle"], ["r:Report"], {}),
  ("thickness_analysis", "Wall-thickness field (ray/SDF based) with min threshold.", ["body:Ref", "min:Length?"], ["t:Field"], {}),
  ("distance_analysis", "Distance field between two elements (e.g. scan vs CAD).", ["a:Ref", "b:Ref"], ["d:Field"], A("deviation analysis")),
  ("sectioning", "Planar section curves and section properties.", ["items:Ref[]", "plane:Ref"], ["s:Section"], A("section properties", "section view")),
  ("clash", "Clash/interference and clearance between bodies or products.", ["items:Ref[]", "clearance:Length?"], ["r:ClashReport"], A("interference detection", "clearance")),
  ("check", "Geometric/topological validity: closure, orientation, tolerances, self-intersection.", ["body:Ref"], ["r:Report"], M(A("check entity", "geometry check"), phase="exact", backend="gpu", differentiable=False)),
  ("sample", "Sample signed distance / implicit value or attribute of any body at points.", ["body:Ref", "pts:Vec3[]"], ["v:Scalar[]"], {}),
  ("raycast", "Batched ray casts returning hit element names and parameters.", ["rays:Ray[]"], ["hits:Hit[]"], M({}, differentiable=False)),
  ("overhang_analysis", "Additive manufacturing: overhang angle map and unsupported area for a build direction.", ["body:Ref", "build_dir:Vec3", "max_angle:Angle"], ["r:Field"], A("am overhang", "support analysis")),
  ("undercut_analysis", "Molding: undercut regions for a pull direction (visibility via ray casting).", ["body:Ref", "pull:Vec3"], ["r:Report"], {}),
  ("tool_accessibility", "Machining: per-face accessibility from a set of tool axes and tool envelope.", ["body:Ref", "axes:Vec3[]", "tool:ToolSpec"], ["r:Report"], A("accessibility", "3-axis/5-axis reach")),
  ("internal_radius_analysis", "Machining: internal corner radii vs minimum tool radius.", ["body:Ref", "min_r:Length"], ["r:Report"], {}),
  ("stiffness", "Linear-elastic static analysis (matrix-free CG on voxel/implicit representation): compliance, max displacement, max von Mises stress. Evidence: estimated, discretization reported.", ["body:Ref", "material:Material", "loads:Load[]", "supports:Ref[]", "voxel:Length"], ["r:PhysicsReport"], M(A("fea", "static analysis"), milestone="M4", fail=SF)),
  ("modal", "Lowest natural frequencies and mode shapes (batched Lanczos/LOBPCG) on voxel/implicit representation.", ["body:Ref", "material:Material", "supports:Ref[]", "modes:Int"], ["r:PhysicsReport"], M(A("frequency analysis"), milestone="M4", fail=SF)),
  ("evaluation", "Define a named, reusable evaluation (measure expression in AgLang over any evaluate.* results) with declared evidence and gradient.", ["source:String", "evidence:Enum"], ["e:EvalRef"], M(A("sensor", "knowledge parameter measure", "custom measure"), fail=CODE)),
 ],
 "query": [
  ("named", "Resolve stable names (provenance paths or publications) at a revision.", ["names:String[]"], ["sel:Selection"], M({}, fail=["STALE_REFERENCE", "AMBIGUOUS_REFERENCE"])),
  ("element_type", "Filter by element type (vertex, edge, face, body, sketch element, point/primitive).", ["scope:Ref", "type:Enum"], ["sel:Selection"], {}),
  ("created_by", "Elements created or modified by a feature, optionally by role.", ["feature:FeatureRef", "role:String?"], ["sel:Selection"], {}),
  ("parallel_to", "Faces/edges whose normal/tangent lies within an angle of a direction.", ["scope:Ref", "dir:Vec3", "tol:Angle"], ["sel:Selection"], {}),
  ("geometry", "Predicate on geometry: radius, area, length, convexity, surface type, position.", ["scope:Ref", "pred:Predicate"], ["sel:Selection"], {}),
  ("adjacent", "Topological neighbours (faces of edge, edges of face, tangent-continuous chain).", ["sel:Selection", "kind:Enum", "depth:Int?"], ["sel:Selection"], A("tangent propagation")),
  ("contained_in", "Elements inside a box, sphere, half-space or body.", ["scope:Ref", "region:Region"], ["sel:Selection"], {}),
  ("closest_to", "Elements closest to a point.", ["scope:Ref", "p:Vec3", "k:Int?"], ["sel:Selection"], {}),
  ("similar", "Shape/requirement retrieval: k most similar parts or features from libraries and archives by learned embedding.", ["probe:Ref|Requirements", "scope:Enum", "k:Int"], ["hits:Hit[]"], M({}, milestone="M4")),
  ("nth", "Order a selection by a measure and take a slice.", ["sel:Selection", "key:Enum", "take:Int"], ["sel:Selection"], {}),
  ("group", "Elements of a procedural group or attribute predicate.", ["scope:Ref", "group:String|Predicate"], ["sel:Selection"], {}),
  ("combine", "Union/intersection/difference of selections.", ["a:Selection", "b:Selection", "op:Enum"], ["sel:Selection"], {}),
 ],
 "file": [
  ("export_step", "STEP export (AP242 ed.2+ default, AP214 option) of bodies and products: names, colours/layers, assembly structure, validation properties, optional semantic PMI; receiver profile (generic/solidworks/catia). GPU-encoded (doc 18).", ["items:Ref[]", "path:String", "schema:Enum?", "profile:Enum?", "pmi:Bool?"], ["file:File", "report:ExportReport"], {}),
  ("import_step", "STEP AP203/AP214/AP242 import on GPU (lexer → parser → resolver → topology builders) with healing report, product structure, names, colours, PMI and validation-property comparison (doc 18).", ["path:String", "options:ImportOptions?"], ["bodies:Body[]", "report:ImportReport"], M({}, effect="doc_write", milestone="M2")),
  ("export_3mf", "3MF export for additive manufacturing: Core + Materials & Properties + Beam Lattice + Slice extensions; per-body materials; lattice as beams or mesh.", ["items:Ref[]", "path:String", "extensions:Enum[]?"], ["file:File", "report:ExportReport"], M(A("additive export"), milestone="M3")),
  ("export_slices", "GPU slicing of bodies (incl. implicit/lattice) into layer contours for AM (3MF Slice extension or CLI/SVG stacks).", ["items:Ref[]", "layer:Length", "build_dir:Vec3", "path:String"], ["file:File"], M(A("slice", "layer export"), milestone="M5")),
  ("export_iges", "IGES export of surfaces/curves.", ["items:Ref[]", "path:String"], ["file:File"], M({}, milestone="M5")),
  ("export_mesh", "STL/3MF/OBJ export from tessellation or mesh bodies.", ["items:Ref[]", "format:Enum", "path:String"], ["file:File"], {}),
  ("export_gltf", "glTF scene for viewers with stable-name metadata.", ["items:Ref[]", "path:String"], ["file:File"], {}),
  ("import_mesh", "Import STL/OBJ/3MF/PLY as mesh body or point cloud.", ["path:String"], ["body:Body"], M({}, effect="doc_write")),
  ("export_vdb", "Export implicit/voxel bodies as OpenVDB.", ["items:Ref[]", "path:String"], ["file:File"], M({}, milestone="M3")),
  ("export_feature_script", "History-preserving handoff: export the specification tree as a replayable script (CATIA EKL/VBA, SolidWorks API, Onshape FeatureScript) via the alias table.", ["part:Ref", "target:Enum", "path:String"], ["file:File", "report:ExportReport"], M(A("macro export", "tree export"), milestone="M5")),
  ("export_usd", "OpenUSD export: products/components, bodies, attributes as primvars, configurations as variant sets (Omniverse/Houdini Solaris interop).", ["items:Ref[]", "path:String", "configurations:Configurations?"], ["file:File"], M(A("usd", "omniverse"), milestone="M3")),
  ("save", "Save the document as .agcad (specification tree, UDFs, versions, trace refs); see doc 16.", ["path:String"], ["file:File"], {}),
  ("open", "Open a saved .agcad document, migrating older schema/op versions.", ["path:String"], ["doc:Document"], M({}, effect="doc_write", fail=CF + ["MIGRATION_FAILED"])),
 ],
 "datum": [
  ("point", "Point: coordinates, on curve/surface, center, tangent, between, intersection.", ["mode:Enum", "refs:Ref[]", "value:Any?"], ["pt:Point"], A("reference point")),
  ("line", "Line: point-point, point-direction, angle/normal to curve, tangent, normal to surface, bisecting.", ["mode:Enum", "refs:Ref[]", "value:Any?"], ["ln:Line"], A("axis", "reference axis")),
  ("plane", "Plane: offset, parallel through point, angle/normal, three points, two lines, normal to curve, mean through points.", ["mode:Enum", "refs:Ref[]", "value:Any?"], ["pl:Plane"], A("reference plane", "datum plane")),
  ("axis_system", "Axis system (coordinate system) from origin and axes.", ["origin:Ref", "x:Ref", "y:Ref"], ["axs:AxisSystem"], A("coordinate system", "frame")),
 ],
 "sketch": [
  ("create", "Sketch on a plane or planar face, positioned by an axis system.", ["support:Ref", "axis:Ref?"], ["sketch:Sketch"], A("positioned sketch")),
  ("point", "Sketch point.", ["sketch:Sketch", "p:Vec2"], ["e:SketchElement"], {}),
  ("line", "Line or construction line.", ["sketch:Sketch", "a:Vec2", "b:Vec2", "construction:Bool?"], ["e:SketchElement"], {}),
  ("arc", "Arc by center/ends, three points, or tangent.", ["sketch:Sketch", "mode:Enum", "pts:Vec2[]"], ["e:SketchElement"], {}),
  ("circle", "Circle by center-radius, three points, or tritangent.", ["sketch:Sketch", "mode:Enum", "pts:Vec2[]", "r:Length?"], ["e:SketchElement"], {}),
  ("ellipse", "Ellipse or elliptical arc.", ["sketch:Sketch", "center:Vec2", "major:Vec2", "ratio:Scalar"], ["e:SketchElement"], {}),
  ("conic", "Conic by endpoints, tangents and parameter (parabola/hyperbola by rho).", ["sketch:Sketch", "a:Vec2", "b:Vec2", "shoulder:Vec2", "rho:Scalar"], ["e:SketchElement"], {}),
  ("spline", "Interpolating or control-point spline.", ["sketch:Sketch", "mode:Enum", "pts:Vec2[]", "degree:Int?"], ["e:SketchElement"], {}),
  ("profile", "Connected line/arc profile (polyline with tangent arcs).", ["sketch:Sketch", "pts:Vec2[]", "arcs:Int[]?"], ["e:SketchElement[]"], {}),
  ("rectangle", "Rectangle, oriented rectangle, parallelogram.", ["sketch:Sketch", "mode:Enum", "pts:Vec2[]"], ["e:SketchElement[]"], {}),
  ("polygon", "Regular polygon (hexagon etc.).", ["sketch:Sketch", "center:Vec2", "sides:Int", "r:Length"], ["e:SketchElement[]"], {}),
  ("elongated_hole", "Elongated hole / slot (straight or cylindrical), keyhole profile.", ["sketch:Sketch", "mode:Enum", "pts:Vec2[]", "width:Length"], ["e:SketchElement[]"], A("slot")),
  ("text", "Text outline profile.", ["sketch:Sketch", "text:String", "font:String", "height:Length"], ["e:SketchElement[]"], M({}, differentiable=False, milestone="M5")),
  ("offset", "Offset a chain of sketch elements.", ["elements:SketchElement[]", "d:Length"], ["e:SketchElement[]"], {}),
  ("trim", "Trim/break/quick-trim elements.", ["element:SketchElement", "pick:Vec2"], ["e:SketchElement[]"], M({}, differentiable=False)),
  ("corner", "Sketch corner (fillet) between two elements.", ["a:SketchElement", "b:SketchElement", "r:Length"], ["e:SketchElement"], A("sketch fillet")),
  ("chamfer", "Sketch chamfer between two elements.", ["a:SketchElement", "b:SketchElement", "d:Length"], ["e:SketchElement"], {}),
  ("symmetry", "Mirror elements about a line.", ["elements:SketchElement[]", "axis:SketchElement"], ["e:SketchElement[]"], A("mirror entities")),
  ("project_3d", "Project 3D elements (edges, silhouettes, intersections) into the sketch.", ["sketch:Sketch", "elements:Ref[]", "mode:Enum"], ["e:SketchElement[]"], A("convert entities", "use edges")),
  ("solve", "Batched Levenberg-Marquardt solve of all constraints; reports DOF, over/under-constrained elements.", ["sketch:Sketch"], ["status:SolveReport"], M({}, fail=SF)),
  ("regions", "Extract closed regions from the solved sketch.", ["sketch:Sketch"], ["regions:Region[]"], M({}, fail=CF + ["TOPOLOGY_FAILED"])),
 ],
 "constraint": [
  ("coincidence", "Point-point or point-on-curve coincidence.", ["a:Ref", "b:Ref"], ["c:Constraint"], A("coincident")),
  ("horizontal", "Horizontal.", ["a:Ref", "b:Ref?"], ["c:Constraint"], {}),
  ("vertical", "Vertical.", ["a:Ref", "b:Ref?"], ["c:Constraint"], {}),
  ("parallelism", "Parallel lines.", ["a:Ref", "b:Ref"], ["c:Constraint"], A("parallel")),
  ("perpendicularity", "Perpendicular lines.", ["a:Ref", "b:Ref"], ["c:Constraint"], A("perpendicular")),
  ("tangency", "Tangency (G1).", ["a:Ref", "b:Ref"], ["c:Constraint"], A("tangent")),
  ("curvature_continuity", "Curvature continuity (G2) between curves.", ["a:Ref", "b:Ref"], ["c:Constraint"], A("curvature")),
  ("equal", "Equal length or radius.", ["a:Ref", "b:Ref"], ["c:Constraint"], {}),
  ("concentricity", "Shared center.", ["a:Ref", "b:Ref"], ["c:Constraint"], A("concentric")),
  ("symmetry", "Symmetric about a line.", ["a:Ref", "b:Ref", "axis:Ref"], ["c:Constraint"], A("symmetric")),
  ("midpoint", "Point at midpoint.", ["p:Ref", "line:Ref"], ["c:Constraint"], {}),
  ("fix", "Fix element.", ["a:Ref"], ["c:Constraint"], A("fixed", "lock")),
  ("pierce", "Point pierces a 3D curve.", ["p:Ref", "curve:Ref"], ["c:Constraint"], {}),
  ("distance", "Distance dimension (driving or reference).", ["a:Ref", "b:Ref", "value:Length"], ["c:Constraint"], {}),
  ("angle", "Angle dimension.", ["a:Ref", "b:Ref", "value:Angle"], ["c:Constraint"], {}),
  ("radius", "Radius dimension.", ["a:Ref", "value:Length"], ["c:Constraint"], {}),
  ("diameter", "Diameter dimension.", ["a:Ref", "value:Length"], ["c:Constraint"], {}),
  ("length", "Length / arc length dimension.", ["a:Ref", "value:Length"], ["c:Constraint"], {}),
  ("residual", "Arbitrary differentiable AgLang residual as a constraint.", ["source:String"], ["c:Constraint"], M({}, fail=CODE)),
  ("inequality", "Inequality constraint on any evaluation (e.g. min wall >= 2 mm, clearance >= 0.5 mm); driving in solves/optimization, checked otherwise.", ["evaluation:EvalRef", "op:Enum", "bound:Scalar", "tol:Scalar?"], ["c:Constraint"], M(A("limit", "bound"), batch_axes=[C, E])),
  ("geometric_3d", "3D geometric constraint between part elements (coincident, parallel, perpendicular, tangent, distance, angle) solved with the part's parameters.", ["kind:Enum", "a:Ref", "b:Ref", "value:Scalar?"], ["c:Constraint"], M(A("3d constraint", "part constraint"), milestone="M3")),
  ("set", "Named constraint set (sketch, part, assembly or study scope) with priority/weights for soft constraints.", ["scope:Ref", "constraints:Constraint[]", "weights:Scalar[]?"], ["set:ConstraintSet"], M({}, milestone="M2")),
  ("analyze", "Constraint-graph analysis: DOF, structural rank, redundancy and conflict sets, decomposition plan.", ["set:Ref"], ["r:ConstraintReport"], M({}, effect="pure", fail=SF)),
 ],
 "wireframe": [
  ("line", "3D line segment.", ["a:Ref|Vec3", "b:Ref|Vec3"], ["curve:Curve"], {}),
  ("polyline", "Polyline through points with optional corner radii.", ["pts:Ref[]", "radii:Length[]?"], ["curve:Curve"], {}),
  ("circle", "Circle or arc (center-radius, three points, tangent, bitangent).", ["mode:Enum", "args:Any"], ["curve:Curve"], {}),
  ("conic", "Ellipse, parabola, hyperbola, general conic.", ["kind:Enum", "support:Ref", "p:Scalar[]"], ["curve:Curve"], {}),
  ("helix", "Helix (constant/variable pitch, taper, law).", ["axis:Ref", "start:Ref", "pitch:Length|Law", "height:Length", "taper:Angle?"], ["curve:Curve"], {}),
  ("spiral", "Planar spiral.", ["support:Ref", "center:Ref", "kind:Enum", "p:Scalar[]"], ["curve:Curve"], {}),
  ("spline", "3D spline through points with tangent/curvature conditions.", ["pts:Ref[]", "tangents:Vec3[]?", "param:Enum?"], ["curve:Curve"], {}),
  ("nurbs_curve", "Explicit NURBS curve (control points, weights, knots).", ["poles:Vec4[]", "degree:Int", "knots:Scalar[]"], ["curve:Curve"], A("bspline", "bezier")),
  ("connect_curve", "Connect curve between two curve ends with G0..G3 conditions.", ["a:Ref", "b:Ref", "order:Int"], ["curve:Curve"], A("blend curve")),
  ("corner", "3D corner (fillet) between two curves.", ["a:Ref", "b:Ref", "r:Length"], ["curve:Curve"], {}),
  ("parallel_curve", "Offset curve on a support surface.", ["c:Ref", "support:Ref", "d:Length|Law"], ["curve:Curve"], A("offset curve")),
  ("projection", "Project curve/point onto surface (normal or along direction).", ["c:Ref", "support:Ref", "dir:Vec3?"], ["curve:Curve[]"], M({}, backend="gpu")),
  ("intersection", "Intersection of two elements (surfaces, curves).", ["a:Ref", "b:Ref"], ["curve:Curve[]"], M({}, backend="gpu", differentiable=False)),
  ("combine", "Combined curve from two planar-projected curves.", ["a:Ref", "b:Ref"], ["curve:Curve"], {}),
  ("reflect_line", "Reflect/silhouette lines of a surface for a direction.", ["s:Ref", "dir:Vec3", "angle:Angle"], ["curve:Curve[]"], M(A("silhouette"), backend="gpu", differentiable=False)),
  ("isoparametric", "Isoparametric curve on a surface.", ["s:Ref", "dir:Enum", "t:Scalar"], ["curve:Curve"], {}),
  ("spine", "Spine curve for sweeps (normal to sections).", ["sections:Ref[]"], ["curve:Curve"], {}),
  ("fit", "Least-squares curve fit to points within tolerance.", ["pts:Ref[]", "tol:Length", "degree:Int"], ["curve:Curve"], {}),
  ("join_curves", "Join connected curves into one wire.", ["cs:Ref[]", "tol:Length?"], ["curve:Curve"], M({}, differentiable=False)),
  ("split_curve", "Split/trim a curve by points or elements.", ["c:Ref", "tools:Ref[]"], ["curve:Curve[]"], M({}, differentiable=False)),
  ("curve_smooth", "Smooth a curve within deviation (curvature cleanup).", ["c:Ref", "max_dev:Length"], ["curve:Curve"], {}),
  ("rebuild_curve", "Rebuild curve with given control-point count/degree; degree elevation; knot insertion.", ["c:Ref", "mode:Enum", "args:Any"], ["curve:Curve"], A("refit", "elevate degree", "insert knot")),
 ],
 "surface": [
  ("extrude", "Extruded surface from a profile.", ["profile:Ref", "dir:Ref", "limits:Interval"], ["surf:SurfaceBody"], M({}, milestone="M2")),
  ("revolve", "Revolved surface (exact rational).", ["profile:Ref", "axis:Ref", "angles:Interval"], ["surf:SurfaceBody"], M({}, milestone="M2")),
  ("sphere", "Spherical surface.", ["center:Ref", "r:Length", "limits:Any?"], ["surf:SurfaceBody"], M({}, milestone="M2")),
  ("cylinder", "Cylindrical surface.", ["center:Ref", "dir:Ref", "r:Length", "limits:Interval"], ["surf:SurfaceBody"], M({}, milestone="M2")),
  ("offset", "Offset surface (constant or variable) with self-intersection check.", ["s:Ref", "d:Length|Law"], ["surf:SurfaceBody"], {}),
  ("sweep", "Sweep: explicit, linear, circular, conic profiles; guides, spine, law, twist.", ["profile:Ref", "guides:Ref[]", "spine:Ref?", "laws:Map?"], ["surf:SurfaceBody"], {}),
  ("multi_sections_surface", "Multi-sections surface (loft) with guides, spine, tangency and coupling.", ["sections:Ref[]", "guides:Ref[]?", "coupling:Enum"], ["surf:SurfaceBody"], A("loft", "lofted surface")),
  ("blend", "Blend surface between two curves on supports with G0..G3 continuity and tension.", ["a:Ref", "b:Ref", "order:Int[]", "tension:Scalar[]?"], ["surf:SurfaceBody"], {}),
  ("fill", "Fill an N-sided boundary with G0..G2 conditions to support surfaces.", ["boundary:Ref[]", "supports:Ref[]?", "order:Int[]?"], ["surf:SurfaceBody"], M(A("n-sided patch", "fill surface"), backend="gpu")),
  ("net_surface", "Surface from a network of curves in two directions (Gordon).", ["u:Ref[]", "v:Ref[]"], ["surf:SurfaceBody"], A("boundary surface", "network surface")),
  ("ruled", "Ruled surface between two curves.", ["a:Ref", "b:Ref"], ["surf:SurfaceBody"], {}),
  ("planar", "Planar surface from a closed planar boundary.", ["boundary:Ref"], ["surf:SurfaceBody"], {}),
  ("nurbs_surface", "Explicit NURBS/Bezier surface (control net, weights, knots).", ["poles:Vec4[,]", "degree:Int2", "knots_u:Scalar[]", "knots_v:Scalar[]"], ["surf:SurfaceBody"], M({}, backend="gpu", milestone="M1")),
  ("join", "Join surfaces into one surface body within merging distance.", ["surfaces:Ref[]", "tol:Length"], ["surf:SurfaceBody"], M(A("knit", "sew"), backend="gpu", differentiable=False, fail=TF)),
  ("healing", "Heal gaps/tangency defects between surfaces within tolerance budget.", ["surfaces:Ref[]", "tol:Length", "tangency:Angle?"], ["surf:SurfaceBody"], M(A("heal"), backend="gpu", fail=TF)),
  ("split", "Split a surface/curve by a cutting element; keep chosen side.", ["element:Ref", "cutter:Ref", "keep:Enum"], ["surf:SurfaceBody"], M({}, backend="gpu", differentiable=False)),
  ("trim", "Mutually trim two surfaces/curves.", ["a:Ref", "b:Ref", "keep:Enum[]"], ["surf:SurfaceBody"], M({}, backend="gpu", differentiable=False)),
  ("untrim", "Restore underlying surface bounds or remove inner loops.", ["face:Ref", "loops:Ref[]?"], ["surf:SurfaceBody"], M({}, backend="gpu", differentiable=False)),
  ("boundary", "Extract boundary curves of a surface.", ["s:Ref", "propagation:Enum"], ["curve:Curve[]"], {}),
  ("extract", "Extract faces/edges (with propagation) as a new element.", ["elements:Ref[]", "propagation:Enum"], ["surf:SurfaceBody"], {}),
  ("extrapolate", "Extrapolate surface/curve boundary (tangent/curvature continuity).", ["boundary:Ref", "s:Ref", "length:Length", "mode:Enum"], ["surf:SurfaceBody"], A("extend surface")),
  ("shape_fillet", "Surface-surface fillet (bitangent, tritangent, variable, chordal).", ["a:Ref", "b:Ref", "r:Length|Law", "mode:Enum"], ["surf:SurfaceBody"], M({}, fail=TF)),
  ("match_surface", "Deform a surface edge to match another at G0..G3.", ["s:Ref", "edge:Ref", "target:Ref", "order:Int"], ["surf:SurfaceBody"], {}),
  ("fair", "Curvature-gradient fairing of a surface network under fixed boundaries and max deviation.", ["faces:Ref[]", "fixed:Ref[]", "max_dev:Length"], ["surf:SurfaceBody"], M({}, backend="gpu", milestone="M5")),
  ("rebuild_surface", "Rebuild surface with given control-point counts/degrees within tolerance.", ["s:Ref", "poles:Int2", "degree:Int2"], ["surf:SurfaceBody"], M({}, backend="gpu")),
 ],
 "part": [
  ("pad", "Pad: extrude a profile to add material (dimension, up to next/last/plane/surface, mirrored, thick, draft).", ["profile:Ref", "limits:Limits", "dir:Ref?", "thick:Length?"], ["feature:FeatureRef"], M(A("extrude", "boss-extrude", "extruded boss"), milestone="M1", fail=TF)),
  ("pocket", "Pocket: extrude a profile to remove material.", ["profile:Ref", "limits:Limits", "dir:Ref?"], ["feature:FeatureRef"], M(A("extruded cut", "cut-extrude"), milestone="M1", fail=TF)),
  ("shaft", "Shaft: revolve a profile to add material.", ["profile:Ref", "axis:Ref", "angles:Interval"], ["feature:FeatureRef"], M(A("revolve", "revolved boss"), milestone="M1", fail=TF)),
  ("groove", "Groove: revolve a profile to remove material.", ["profile:Ref", "axis:Ref", "angles:Interval"], ["feature:FeatureRef"], M(A("revolved cut"), milestone="M1", fail=TF)),
  ("hole", "Hole: simple, tapered, counterbored, countersunk, counterdrilled; threaded; standard tables.", ["support:Ref", "positions:Ref[]", "spec:HoleSpec"], ["feature:FeatureRef"], M(A("hole wizard"), fail=TF)),
  ("rib", "Rib: sweep a profile along a center curve to add material.", ["profile:Ref", "center:Ref", "orientation:Enum"], ["feature:FeatureRef"], M(A("sweep", "swept boss"), fail=TF)),
  ("slot", "Slot: sweep a profile along a center curve to remove material.", ["profile:Ref", "center:Ref", "orientation:Enum"], ["feature:FeatureRef"], M(A("swept cut"), fail=TF)),
  ("multi_sections_solid", "Multi-sections solid (loft) through sections with guides and coupling.", ["sections:Ref[]", "guides:Ref[]?", "coupling:Enum"], ["feature:FeatureRef"], M(A("loft", "lofted boss"), fail=TF)),
  ("removed_multi_sections_solid", "Removed multi-sections solid.", ["sections:Ref[]", "guides:Ref[]?", "coupling:Enum"], ["feature:FeatureRef"], M(A("lofted cut"), fail=TF)),
  ("stiffener", "Stiffener from an open profile (from side / from top).", ["profile:Ref", "thickness:Length", "mode:Enum"], ["feature:FeatureRef"], M(A("rib (solidworks)", "web"), fail=TF)),
  ("solid_combine", "Solid from the intersection of two extruded profiles.", ["a:Ref", "b:Ref"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("thread", "Thread/tap on a cylindrical face (cosmetic specification or modeled geometry).", ["face:Ref", "spec:ThreadSpec", "modeled:Bool"], ["feature:FeatureRef"], M(A("cosmetic thread"), milestone="M5", fail=TF)),
  ("edge_fillet", "Edge fillet (constant radius, conic, chordal) with propagation and setbacks.", ["edges:Ref[]", "r:Length", "propagation:Enum"], ["feature:FeatureRef"], M(A("fillet", "round"), fail=TF)),
  ("variable_fillet", "Variable-radius fillet with radius law or points.", ["edges:Ref[]", "radii:Law"], ["feature:FeatureRef"], M(A("variable radius fillet"), fail=TF)),
  ("face_face_fillet", "Fillet between two faces.", ["a:Ref", "b:Ref", "r:Length"], ["feature:FeatureRef"], M(A("face fillet"), fail=TF)),
  ("tritangent_fillet", "Tritangent fillet: remove a face by a fillet tangent to three faces.", ["faces:Ref[]", "remove:Ref"], ["feature:FeatureRef"], M(A("full round fillet"), fail=TF)),
  ("chamfer", "Chamfer (length-angle, length-length) on edges.", ["edges:Ref[]", "mode:Enum", "a:Scalar", "b:Scalar?"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("draft", "Draft angle (neutral element, parting element, variable, reflect line).", ["faces:Ref[]", "neutral:Ref", "pull:Ref", "angle:Angle|Law"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("shell", "Shell: hollow with faces to remove and per-face thickness.", ["remove:Ref[]", "inner:Length", "outer:Length?", "per_face:Map?"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("thickness", "Offset selected faces of a solid.", ["faces:Ref[]", "d:Length"], ["feature:FeatureRef"], M(A("offset face", "move face"), fail=TF)),
  ("split", "Split a body with a plane or surface; keep a side.", ["body:Ref", "cutter:Ref", "keep:Enum"], ["feature:FeatureRef"], M(A("split body"), fail=TF)),
  ("thick_surface", "Thicken a surface into a solid.", ["s:Ref", "t1:Length", "t2:Length?"], ["feature:FeatureRef"], M(A("thicken"), fail=TF)),
  ("close_surface", "Close a surface body into a solid.", ["s:Ref"], ["feature:FeatureRef"], M(A("knit to solid"), fail=TF)),
  ("sew_surface", "Sew a surface onto a solid (add/remove material up to it).", ["s:Ref", "keep:Enum"], ["feature:FeatureRef"], M(A("replace face", "surface cut"), fail=TF)),
  ("remove_face", "Remove faces and heal by extending neighbours.", ["faces:Ref[]", "keep:Ref[]?"], ["feature:FeatureRef"], M(A("delete face", "defeature"), differentiable=False, fail=TF)),
  ("replace_face", "Replace faces with a surface.", ["faces:Ref[]", "target:Ref"], ["feature:FeatureRef"], M({}, differentiable=False, fail=TF)),
  ("lattice", "Lattice infill of a region (TPMS gyroid/diamond, strut, custom UDF cell) with grading law; produces an implicit body.", ["region:Ref", "cell:Enum|UdfDef", "size:Length", "grading:Law?"], ["body:ImplicitBody"], M(A("infill", "gyroid"), backend="gpu", milestone="M3")),
  ("implicit_body", "Implicit body defined by an AgLang distance function or blend tree (smooth union/subtract/intersect, shell, offset).", ["source:String|Network", "bounds:Box"], ["body:ImplicitBody"], M(A("sdf", "field", "vdb"), backend="gpu", milestone="M1", fail=CODE)),
  ("primitive", "Solid primitives: box, cylinder, cone, sphere, torus, wedge, prism.", ["kind:Enum", "axis:Ref", "p:Scalar[]"], ["feature:FeatureRef"], M({}, backend="gpu", milestone="M1")),
 ],
 "boolean": [
  ("assemble", "Assemble a body into another (operation sign follows feature type).", ["target:Ref", "tool:Ref"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("add", "Add (union) bodies.", ["target:Ref", "tools:Ref[]"], ["feature:FeatureRef"], M(A("union", "combine add"), fail=TF)),
  ("remove", "Remove (subtract) bodies.", ["target:Ref", "tools:Ref[]"], ["feature:FeatureRef"], M(A("subtract", "difference", "combine subtract"), fail=TF)),
  ("intersect", "Intersect bodies (empty result is valid).", ["target:Ref", "tools:Ref[]"], ["feature:FeatureRef"], M(A("common", "combine common"), fail=TF)),
  ("union_trim", "Union trim: union while choosing faces/volumes to keep or remove.", ["target:Ref", "tool:Ref", "remove:Ref[]", "keep:Ref[]"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("remove_lump", "Remove disconnected lumps from a body.", ["body:Ref", "remove:Ref[]"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("smooth_boolean", "Blended boolean on implicit bodies (smooth union/subtract/intersect with radius).", ["a:Ref", "b:Ref", "op:Enum", "r:Length"], ["body:ImplicitBody"], M({}, backend="gpu", phase="proxy", differentiable=True, milestone="M1")),
 ],
 "transform": [
  ("translation", "Translate bodies or elements (copy optional).", ["items:Ref[]", "dir:Ref", "d:Length", "copy:Bool"], ["feature:FeatureRef"], A("move", "move/copy")),
  ("rotation", "Rotate bodies or elements about an axis.", ["items:Ref[]", "axis:Ref", "angle:Angle", "copy:Bool"], ["feature:FeatureRef"], {}),
  ("symmetry", "Symmetry of bodies or elements about a point, line or plane.", ["items:Ref[]", "ref:Ref", "copy:Bool"], ["feature:FeatureRef"], {}),
  ("mirror", "Mirror features or bodies about a plane and add to the body.", ["items:Ref[]", "plane:Ref"], ["feature:FeatureRef"], {}),
  ("rectangular_pattern", "Rectangular pattern in one or two directions with instance skipping and staggering.", ["items:Ref[]", "dirs:Ref[]", "counts:Int[]", "spacing:Length[]"], ["feature:FeatureRef"], A("linear pattern")),
  ("circular_pattern", "Circular pattern (angular and radial crowns).", ["items:Ref[]", "axis:Ref", "count:Int", "angle:Angle"], ["feature:FeatureRef"], {}),
  ("user_pattern", "User pattern: instances at sketch points or table positions.", ["items:Ref[]", "positions:Ref"], ["feature:FeatureRef"], A("sketch driven pattern", "table driven pattern", "copy to points")),
  ("curve_pattern", "Pattern along a curve.", ["items:Ref[]", "curve:Ref", "count:Int", "align:Enum"], ["feature:FeatureRef"], A("curve driven pattern")),
  ("fill_pattern", "Fill a region with a seed pattern (square, hex, circular layouts).", ["items:Ref[]", "region:Ref", "layout:Enum", "spacing:Length"], ["feature:FeatureRef"], {}),
  ("scaling", "Uniform scaling about a point/plane.", ["items:Ref[]", "ref:Ref", "ratio:Scalar"], ["feature:FeatureRef"], A("scale")),
  ("affinity", "Non-uniform scaling in an axis system.", ["items:Ref[]", "axis:Ref", "ratios:Vec3"], ["feature:FeatureRef"], A("non-uniform scale")),
  ("axis_to_axis", "Transform elements from one axis system to another.", ["items:Ref[]", "from:Ref", "to:Ref"], ["feature:FeatureRef"], {}),
 ],
 "direct": [
  ("push_pull", "Push/pull faces along their normals.", ["faces:Ref[]", "d:Length"], ["feature:FeatureRef"], M(A("live shape", "press pull"), fail=TF)),
  ("move_face", "Rigidly move/rotate faces, re-solving neighbours.", ["faces:Ref[]", "xform:Xform"], ["feature:FeatureRef"], M({}, fail=TF)),
  ("defeature", "Remove fillets/holes/chamfers below size thresholds.", ["body:Ref", "kinds:Enum[]", "max_size:Length"], ["feature:FeatureRef"], M({}, differentiable=False, fail=TF)),
  ("heal_body", "Heal imported bodies: close gaps, merge tolerant edges, remove slivers within budget.", ["body:Ref", "tol:Length"], ["feature:FeatureRef"], M(A("import diagnostics"), backend="gpu", fail=TF)),
 ],
 "mesh": [
  ("primitive", "Mesh box/sphere/cylinder/cone/torus/grid.", ["kind:Enum", "axis:Ref", "p:Scalar[]", "res:Int[]"], ["mesh:MeshBody"], M({}, milestone="M1")),
  ("tessellate", "Adaptive watertight tessellation of bodies with chordal/angular bounds.", ["items:Ref[]", "sag:Length", "angle:Angle"], ["mesh:MeshBody"], M({}, milestone="M1")),
  ("weld", "Weld coincident vertices within tolerance (spatial hash).", ["mesh:Ref", "tol:Length"], ["mesh:MeshBody"], A("fuse points")),
  ("normals", "Vertex normals with crease angle.", ["mesh:Ref", "crease:Angle"], ["mesh:MeshBody"], {}),
  ("subdivide", "Linear, Loop or Catmull-Clark subdivision with creases.", ["mesh:Ref", "scheme:Enum", "levels:Int"], ["mesh:MeshBody"], {}),
  ("decimate", "Quadric-error decimation to target count or error.", ["mesh:Ref", "target:Scalar"], ["mesh:MeshBody"], A("polyreduce")),
  ("remesh", "Isotropic remesh to edge length.", ["mesh:Ref", "edge:Length"], ["mesh:MeshBody"], {}),
  ("smooth", "Laplacian/Taubin smoothing.", ["mesh:Ref", "scheme:Enum", "iters:Int"], ["mesh:MeshBody"], {}),
  ("mesh_boolean", "Robust mesh boolean (winding-number based).", ["a:Ref", "b:Ref", "op:Enum"], ["mesh:MeshBody"], M({}, backend="gpu")),
  ("repair", "Fill holes, fix flipped/non-manifold elements.", ["mesh:Ref"], ["mesh:MeshBody"], M({}, backend="gpu")),
 ],
 "assembly": [
  ("component", "Insert a component (part or product instance) with a position.", ["ref:Ref", "xform:Xform"], ["comp:Component"], A("instance", "insert part")),
  ("coincidence", "Coincidence constraint (axes, planes, points).", ["a:Ref", "b:Ref"], ["c:AsmConstraint"], A("coincident mate", "concentric mate")),
  ("contact", "Contact constraint (planar, line, point contact).", ["a:Ref", "b:Ref"], ["c:AsmConstraint"], {}),
  ("offset", "Offset constraint.", ["a:Ref", "b:Ref", "d:Length"], ["c:AsmConstraint"], A("distance mate")),
  ("angle", "Angle constraint (incl. parallel, perpendicular).", ["a:Ref", "b:Ref", "angle:Angle"], ["c:AsmConstraint"], A("angle mate")),
  ("fix", "Fix component in space.", ["comp:Ref"], ["c:AsmConstraint"], A("ground")),
  ("mechanism_joint", "Kinematic joint (revolute, prismatic, cylindrical, gear, rack, cam).", ["kind:Enum", "a:Ref", "b:Ref", "params:Map?"], ["j:Joint"], M(A("mate", "joint"), milestone="M6")),
  ("update", "Batched assembly constraint solve; reports DOF and conflicts.", ["product:Ref"], ["status:SolveReport"], M(A("solve mates"), fail=SF)),
  ("exploded_view", "Exploded view scenes (presentation only).", ["product:Ref", "steps:Xform[]"], ["view:View"], M({}, differentiable=False)),
 ],
}


def build():
    assert sorted(ORDER) == sorted(FAMILIES)
    known = {f"{f}.{o[0]}" for f in OPS for o in OPS[f]}
    assert PRIMITIVES <= known, PRIMITIVES - known
    ops = []
    for fam in ORDER:
        fid, phase, backend, axes, diff, ms, effect = FAMILIES[fam]
        for idx, (name, summary, inputs, outputs, ov) in enumerate(OPS[fam]):
            ops.append({
                "name": f"{fam}.{name}",
                "family": fam,
                "opcode": (fid << 8) | idx,
                "version": 1,
                "aliases": ov.get("aliases", []),
                "summary": summary,
                "inputs": inputs,
                "outputs": outputs,
                "batch_axes": ov.get("batch_axes", axes),
                "phase": ov.get("phase", phase),
                "backend": ov.get("backend", backend),
                "effect": ov.get("effect", effect),
                "deterministic": ov.get("deterministic", True),
                "differentiable": ov.get("differentiable", diff),
                "failure_codes": ov.get("fail", CF),
                "milestone": ov.get("milestone", ms),
                "status": "planned",
                "layer": "core" if fam in CORE else "stdlib",
                "kernel_primitive": f"{fam}.{name}" in PRIMITIVES,
            })
    return {
        "catalog_version": CATALOG_VERSION,
        "execution": "GPU data plane; host = control plane only (network, compilation, launches, file handles)",
        "ontology": "CATIA V5/3DEXPERIENCE canonical names; aliases for SolidWorks/Onshape/Houdini/generic",
        "schema": {
            "phase": ["control", "proxy", "exact", "both"],
            "backend": ["gpu", "host"],
            "effect": ["pure", "doc_write", "io"],
            "batch_axes": [C, E, S],
            "status": ["planned", "experimental", "stable", "deprecated"],
            "layer": ["core", "stdlib", "model"],
            "type_notation": "name:Type; '?' optional; '[]' array; '[,]' grid; 'A|B' either; Configurations<T> carries the configuration axis",
        },
        "families": {f: v[0] for f, v in FAMILIES.items()},
        "ops": ops,
    }


if __name__ == "__main__":
    json.dump(build(), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
