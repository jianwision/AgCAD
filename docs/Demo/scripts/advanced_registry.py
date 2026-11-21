"""Contracts for the 44 added, bounded acceptance items."""
GROUPS={
 'curve_edit':{
 'knot_insert':('cad_advanced.knot_insert','Exact interior B-spline insertion; nonperiodic tested.'),
 'knot_remove':('cad_advanced.knot_remove','Native removal with a requested tolerance; can reject removal.'),
 'degree_elevate':('cad_advanced.degree_elevate','Exact B-spline degree elevation to <=25.'),
 'degree_reduce':('cad_advanced.degree_reduce','Single nonrational Bezier span; constrained LS and 2048 off-grid checks, sampled error only.'),
 'curve_split':('cad_advanced.curve_split','Exact two-way split strictly inside the domain.'),
 'curve_reparameterize':('cad_advanced.reparameterize','Affine increasing finite domain map.'),
 'fit_curve':('cad_advanced.fit_curve','Nonperiodic cubic least squares, chord parameters, fixed endpoints.'),
 'fair_curve':('cad_advanced.fair_curve','Penalized second control-pole differences; fixed endpoints, no interpolation promise.')},
 'advanced_sweep_loft':{
 'sweep_guided':('cuda/advanced_operations.cuh:sweep_sections','CUDA projected guide rail defines profile frame; nonzero projected rail required.'),
 'sweep_scaled':('cuda/advanced_operations.cuh:sweep_sections','CUDA positive scalar per path station; <=32 stations and <=31 corners.'),
 'sweep_variable':('cuda/advanced_operations.cuh:sweep_sections','CUDA compatible polygon profile per station; bilinear side sheet, no self-intersection guarantee.'),
 'sweep_closed':('cuda/advanced_operations.cuh:sweep_sections','CUDA RMF holonomy distributed by chord length; duplicate endpoint/profile/scale; G0 closed seam, no continuous G1 guarantee.'),
 'loft_compatible':('cad_advanced.compatible_sections + cad_ops.loft','Exact degree/knot/domain harmonization then native loft; caller supplies matching orientation/correspondence.')},
 'surface_edit':{
 'surface_extend':('cad_advanced.extend_bezier_u','Exact polynomial U extrapolation of a nonrational Bezier patch with original [0,1] domain.'),
 'surface_untrim':('cad_advanced.untrim_surface','Rebuild the natural support-domain face; does not restore discarded upstream geometry.'),
 'surface_replace':('cad_advanced.replace_surface_face','Isolated untrimmed rectangular patch, matching sampled boundary; arbitrary solid-face replacement excluded.'),
 'match_g1':('cuda/advanced_operations.cuh:match_bezier_boundary','Compatible bicubic Bezier U border, same V orientation; positive derivative scale. Unit scale tested C1.'),
 'match_g2':('cuda/advanced_operations.cuh:match_bezier_boundary','Same compatible border; position/D1/D2 rows matched. Unit scale tested C2.'),
 'fair_patches':('cuda/advanced_operations.cuh:fair_patch_pair','Joint 7x4 composite net; <=64 projected bending-gradient steps, C2-compatible locked outer boundary and orthogonal seam projection.'),
 'surface_quality':('cad_advanced.surface_quality','Sampled Jacobian/principal curvature diagnostics; not global regularity/Class-A certification.')},
 'solid_feature':{
 'fillet':('cad_advanced.fillet','Native selected-edge constant positive radius; infeasible blends reject.'),
 'variable_fillet':('cad_advanced.fillet','Native linear R1/R2 law along selected contour; positive feasible radii.'),
 'chamfer':('cad_advanced.chamfer','Native symmetric selected-edge chamfer.'),
 'thicken':('cad_advanced.thicken','Native simple face/shell offset and side walls; no general offset self-intersection solver.'),
 'shell':('cad_advanced.shell','Native remove-selected-face and join offset; feasible test box, negative inward thickness.'),
 'draft':('cad_advanced.draft','Native planar/cylindrical/conical face draft, abs(angle)<45deg, explicit neutral plane.'),
 'rib':('cad_advanced.rib','Thicken an explicit planar rib profile and Boolean-fuse with base; not automatic skeleton ribs.')},
 'mesh_edit':{
 'decimate':('mesh_advanced.decimate','CPU shortest-edge collapse; closed oriented manifold, link/no-flip tests and vertex-displacement budget; not a Hausdorff guarantee.'),
 'adaptive_remesh':('mesh_advanced.adaptive_remesh','CPU conforming length-driven midpoint refinement; piecewise planar geometry preserved; <=16 passes.'),
 'edge_flip':('mesh_advanced.edge_flip','CPU two-triangle planar convex interior edge; reject duplicate diagonal.'),
 'bridge_loops':('mesh_advanced.bridge_loops','CPU disjoint equally sized oriented boundary loops with caller correspondence; may return open surface.'),
 'region_extrude':('mesh_advanced.region_extrude','CPU selected triangle region translated outward with boundary walls; vector must point out of every selected face.'),
 'crease_subdivide':('mesh_advanced.crease_subdivide','CPU one Loop step with hard edge/boundary rules and >=3-crease corner lock.')},
 'repair':{
 'self_intersection_detect':('mesh_advanced.self_intersections','CPU nonadjacent triangle crossing pairs; O(T^2), coplanar overlapping candidate rejects as unsupported.'),
 'self_intersection_repair':('mesh_advanced.repair_self_intersections','CPU explicit face-discard policy covering detected crossings; reports open result, no automatic closed reconstruction.'),
 'nonmanifold_repair':('mesh_advanced.repair_nonmanifold','CPU vertex-fan splitting, same face geometry, explicit new boundaries.'),
 'tolerance_unify':('cad_advanced.unify_tolerance','Copied native shape tolerances 1e-9..1e-3 mm; does not heal larger geometric gaps.'),
 'small_feature_cleanup':('cad_advanced.cleanup_small_features','Native same-domain unification and small-edge/wireframe repair; coplanar sliver test. Arbitrary curved sliver removal not guaranteed.')},
 'orchestration':{
 'stable_handles':('feature_graph.Handle','Typed feature ids stable through parameter/history/save-load; never reused in live graph.'),
 'select_subshape':('feature_graph.FeatureGraph.select/resolve','Unique geometric centroid point/edge/face query; revision-stale or ambiguous selections reject.'),
 'feature_dag':('feature_graph.FeatureGraph','Typed DAG, cycle rejection, descendant-only invalidation and memoized execution.'),
 'history_undo':('feature_graph.FeatureGraph.undo','Undo graph edit, clear caches, invalidate all prior subshape selections.'),
 'history_redo':('feature_graph.FeatureGraph.redo','Redo graph edit; branching edit clears redo.'),
 'feature_rebuild':('feature_graph.FeatureGraph.rebuild/loads','Deterministic registered operations and JSON graph; same geometry checked after fresh load/rebuild.')}
}
OPERATORS={name:{'family':family,'entry':entry,'limits':limits,'source':entry.split(':')[0]} for family,ops in GROUPS.items() for name,(entry,limits) in ops.items()}
if __name__=='__main__':
    import json
    from pathlib import Path
    (Path(__file__).resolve().parents[1]/'primitive_advanced.json').write_text(json.dumps({'version':4,'operators':OPERATORS},indent=2),encoding='utf-8')
    print(len(OPERATORS))
