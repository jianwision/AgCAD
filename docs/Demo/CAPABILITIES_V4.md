# 可组合建模能力 v4

实际验收：**134/134**，**16 份 STEP** 导出并由 OCCT 重读。
Rust/NVRTC/CUDA 几何 + 显式 CPU CAD/网格拓扑后端。当前神经策略仍为 769 参数、四指令槽；扩展库尚未全部训练进模型。

## 本轮扩展

在 85 项基础上补齐 44 项缺少的能力；再加入吸附、网格合并、沿曲线、沿网格、沿曲面移动 5 项。
GPU 曲面可组合为缝合、加厚、打孔后的有效实体，实例见 `output/primitives/composed_panel.step`。

## 执行与复现

```powershell
.\run.ps1 suite
.\run.ps1 test
.\run.ps1 lint
python scripts/build_delivery.py
python scripts/build_delivery.py --package
```

最终 GPU 批次下载 12,505,680 字节，1 次最终读回、0 次中间读回。
CAD 内核和部分网格拓扑操作在 CPU 上执行，不包含在这个 GPU 批次计时内。

## 原语目录

每行对应实际执行过的验收项。入口、后端、范围、独立误差和负例详见 `primitive_catalog.json` 与 `output/primitives/validation*.json`。

### curve（15）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| line | `line_point(a,b,t)` | CUDA + independent CAD evaluator；Finite points; t in [0,1]. |
| polyline | `polyline_point(points,count,t)` | CUDA + independent CAD evaluator；At least 2 points; t is normalized segment index, not arc length. |
| circle_arc | `ellipse_point(origin,x,y,r,r,angle)` | CUDA + independent CAD evaluator；Caller supplies orthonormal plane axes and positive radius; angles in radians. |
| ellipse | `ellipse_point(origin,x,y,a,b,angle)` | CUDA + independent CAD evaluator；Caller supplies orthonormal plane axes and positive radii; angles in radians. |
| bezier_curve | `bezier(poles,degree,t)` | CUDA + independent CAD evaluator；Degree 1..5; t in [0,1]; finite control points. |
| nurbs_curve | `nurbs_curve(xyzw,count,degree,knots,t)` | CUDA + independent CAD evaluator；Degree 1..5; poles <=32; ordinary XYZ plus positive weight; valid full knot vector required. Returns p,d,dd. |
| catmull_rom | `catmull_rom(p0,p1,p2,p3,t)` | CUDA + independent CAD evaluator；One uniform Catmull-Rom segment between p1 and p2; not centripetal parameterization. |
| curve_trim | `trimmed_parameter(start,end,t)` | CUDA + independent CAD evaluator；Parameter interval mapping for an existing curve; caller checks interval is in the curve domain. |
| hermite_curve | `make_hermite_curve(record,a,b,da,db)` | CUDA construction/evaluation + independent CAD or analytic reference；Cubic curve with finite endpoint positions and parameter derivatives; ordinary XYZW control points. |
| parabola_curve | `make_conic_curve(record,1,origin,x,y,focal,0,0,first,last)` | CUDA construction/evaluation + independent CAD or analytic reference；Positive focal distance; local coordinates (t^2/(4*focal),t); caller supplies orthonormal axes and finite bounded domain. |
| hyperbola_curve | `make_conic_curve(record,2,origin,x,y,a,b,0,first,last)` | CUDA construction/evaluation + independent CAD or analytic reference；Positive semi-axes; one hyperbola branch (a*cosh(t),b*sinh(t)); bounded parameters must avoid floating-point overflow. |
| helix_curve | `make_conic_curve(record,3,origin,x,y,radius,0,pitch,first,last)` | CUDA construction/evaluation + independent CAD or analytic reference；Circular helix; pitch is axial distance per full turn; finite bounded angle domain; STEP uses a separately error-checked spline approximation. |
| spiral_curve | `make_conic_curve(record,4,origin,x,y,r0,slope,0,first,last)` | CUDA construction/evaluation + independent CAD or analytic reference；Planar Archimedean spiral r=r0+slope*t; strictly positive radius over the domain; STEP spline approximation has measured error. |
| three_point_arc | `make_three_point_arc(record,a,middle,b)` | CUDA construction/evaluation + independent CAD or analytic reference；Three distinct noncollinear 3D points; arc passes through the middle point; absolute degeneracy threshold 1e-20 on squared cross product. |
| interpolate_curve | `make_interpolating_curve(record,points,count)` | CUDA construction/evaluation + independent CAD or analytic reference；Natural cubic interpolation through 2..11 finite points, uniform parameters; <=31 poles; not chord-length or periodic interpolation. |

### surface（13）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| plane | `make_plane(record,origin,u,v)` | CUDA control net/evaluation + CAD topology；Finite independent nonzero direction vectors; rectangular domain [0,1]^2. |
| bezier_patch | `make_bezier_patch(record,poles,nu,nv)` | CUDA control net/evaluation + CAD topology；Nonrational tensor patch, 2..6 poles per direction, row-major U then V. |
| nurbs_patch | `make_nurbs_patch(record,nu,nv,pu,pv,ku,kv,xyzw)` | CUDA control net/evaluation + CAD topology；2..32 poles per axis, degree 1..5 below pole count; finite positive weights and monotonic full knots; caller ensures legal multiplicities. |
| ruled_surface | `make_ruled(record,a,b,n,degree,knots)` | CUDA control net/evaluation + CAD topology；Two compatible nonrational B-spline curves sharing degree, knots and pole count <=32. |
| extrude_surface | `make_extruded(record,xyzw,n,degree,knots,direction)` | CUDA control net/evaluation + CAD topology；Rational profile <=32 poles, degree 1..5; nonzero finite extrusion; valid profile required. |
| revolve_surface | `make_revolved(record,xyzw,n,degree,knots,start,end)` | CUDA control net/evaluation + CAD topology；Exact rational rotation about Y; profile (radius,axial_y,0,weight); nonzero span <=2pi split into <=4 arcs. U is rational arc parameter, not linear angle. |
| sweep_surface | `make_translation_sweep(record,a,nu,pu,ku,b,nv,pv,kv); sweep(spine_wire,profile)` | CUDA control net/evaluation + CAD topology；GPU exact translation surface A(u)+B(v), <=32 poles/axis; general path sweep uses CPU CAD kernel. No arbitrary GPU moving-frame sweep claim. |
| loft_surface | `make_loft(record,sections,rows,nv,degree,knots); loft(section_wires,solid=False,tolerance=1e-7)` | CUDA control net/evaluation + CAD topology；GPU natural cubic interpolation through 2..11 uniformly spaced, compatible nonrational sections; <=32 poles/section. General wires use CPU CAD kernel. |
| coons_fill | `make_coons(record,c0,c1,d0,d1)` | CUDA control net/evaluation + CAD topology；Four cubic nonrational Bezier boundaries with matching corners; exact bicubic Coons construction. |
| offset_surface | `make_offset(record,base_id,distance); surface_point(arena,id,u,v)` | CUDA control net/evaluation + CAD topology；One-level offset of a regular NURBS surface, sampled by its normal; current arena holds 14 objects. STEP retains native offset definition; no global self-intersection guarantee. |
| fill_nside | `fill_nside(edges,orders=None,supports=None,tolerance=1e-7)` | CAD kernel；CPU variational CAD filling; C0/G1/G2 constraints need appropriate support faces. Batch verifies a warped five-edge C0 boundary. |
| sweep_path | `make_path_sweep(record,mesh,path,tangents,count,profile,sides,initial_normal)` | CUDA discrete parallel transport and control net + CPU caps/sew；2..32 stations; convex origin-containing CCW profile with 3..31 corners; discrete parallel transport, no antiparallel tangents. Exact piecewise bilinear GPU net exported to CAD; not a certified continuous RMF solver. |
| loft_chordal | `make_chordal_loft(record,sections,rows,nv,degree,knots)` | CUDA nonuniform natural cubic solve + independent dense solve；2..11 compatible nonrational sections; centroid chord parameters, natural cubic C2 interpolation. Coincident adjacent centroids rejected. |

### continuity（4）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| blend_g1 | `make_hermite_blend(record,1,a,b,da,db,dda,ddb,nv,degree,knots)` | CUDA Hermite control net；Cubic strip matching compatible rail positions and cross derivatives; derivatives supplied by caller, no automatic global shape optimization. |
| blend_g2 | `make_hermite_blend(record,2,a,b,da,db,dda,ddb,nv,degree,knots)` | CUDA quintic Hermite control net；Quintic strip matching compatible rail positions and first/second cross derivatives; degree V <=5, poles V <=32. |
| continuity_check | `continuity_metrics(jet_a,jet_b)` | CUDA differential geometry；Regular coincident seam samples. Returns gap, unsigned tangent-plane angle and curvature mismatch in 3 tangent directions; sample evidence is not global Class-A certification. |
| curvature | `curvature(jet)` | CUDA differential geometry + CAD reference；Returns Gaussian K, signed mean H and surface Jacobian. Zero Jacobian flags a singular/degenerate point where K/H are not valid. |

### topology（11）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| project_curve | `project_curve(curve,surface)` | CAD kernel；CPU normal projection of a native CAD curve; failure raises ValueError. |
| intersect_surfaces | `intersect_surfaces(a,b,tolerance=1e-7)` | CAD kernel；CPU intersection of native untrimmed surfaces; returns native 3D curves; not a full solid Boolean operation. |
| trim_surface | `trim_surface(surface,outer,holes=())` | CAD kernel；CPU B-rep construction from simple closed UV polygon loops; straight UV edges, optional holes; caller provides noncrossing loops within the parameter domain. |
| split_surface | `split_surface(face,tools)` | CAD kernel；CPU native CAD splitter with caller-supplied tool shapes; result must be a valid B-rep. |
| join_curves | `join_curves(curves,tolerance=1e-7)` | CAD kernel；CPU composite B-spline conversion; ordered, connected compatible native curves; rejects disconnected input. |
| sew | `sew_surfaces(faces,tolerance=1e-7,solid=False)` | CAD kernel；CPU face sewing; solid=True requires one closed shell, followed by orientation/validity check; does not union overlapping solids. |
| boolean_union | `boolean_union(a,b)` | CPU native CAD Boolean；CPU OCCT fuse of valid native B-reps; no GPU Boolean claim; failures raise. |
| boolean_cut | `boolean_cut(a,b)` | CPU native CAD Boolean；CPU native A minus B; input topology must be valid; the result may legitimately be empty. |
| boolean_common | `boolean_common(a,b)` | CPU native CAD Boolean；CPU native intersection, including valid empty intersection; distinct from untrimmed surface intersection. |
| delete_faces | `delete_triangles(mesh,mask,count); delete_cad_faces(shell,faces)` | CUDA triangle mask + native CAD shell-face deletion；GPU triangle-mask deletion compacts triangles but leaves vertex storage; CAD deletion takes an open/closed shell, not a solid that would become invalid. Does not automatically cap holes. |
| fill_hole | `fill_planar_hole(mesh,oriented_loop,count); cap_planar_wire(wire)` | CUDA oriented boundary filling + native planar cap/sew；Single simple planar boundary, <=64 mesh vertices; loop opposes remaining halfedges; 1e-12 weld tolerance. Native planar cap can then be sewn. Curved filling uses the separate fill_nside primitive. |

### polygon（13）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| box | `make_box(mesh,minimum,maximum)` | CUDA indexed mesh；Axis-aligned finite strict min/max; 8 vertices, 12 triangles. |
| uv_sphere | `make_uv_sphere(mesh,center,radius,segments,rings)` | CUDA indexed mesh；Positive radius; segments>=3, rings>=2; count must fit 4096 vertices/4096 triangles. |
| cylinder | `make_frustum(mesh,origin,r,r,height,segments)` | CUDA indexed mesh；Z-aligned closed faceted cylinder; positive radius/height; segments>=3. |
| cone | `make_frustum(mesh,origin,r,0,height,segments)` | CUDA indexed mesh；Z-aligned closed cone with one apex; positive bottom radius/height; segments>=3. Same function also supports truncated cones. |
| torus | `make_torus(mesh,center,major,minor,u_segments,v_segments)` | CUDA indexed mesh；Z-axis ring torus; major>minor>0; both segment counts>=3; bounded mesh capacity. |
| grid | `make_grid(mesh,origin,u,v,u_steps,v_steps,warp)` | CUDA indexed mesh；Independent plane vectors; steps>=1; optional sin(pi*u)sin(pi*v) displacement along the plane normal; open mesh. |
| triangulate_polygon | `triangulate_polygon(mesh,points,count)` | CUDA indexed mesh；Ear clipping of one planar simple convex/concave polygon, 3..64 distinct vertices, no holes or collinear overlaps; absolute geometric tolerances. |
| extrude_face | `extrude_polygon(mesh,points,count,direction)` | CUDA indexed mesh；Fresh destination; one planar simple polygon <=64 corners; positive extrusion dot polygon normal; creates caps and walls. |
| inset_face | `inset_polygon(mesh,points,count,fraction)` | CUDA indexed mesh；Fresh destination; strictly convex planar polygon <=64; fraction in (0,1); shrinks toward vertex centroid, not constant-distance edge offset. |
| subdivide | `subdivide_triangles(source,destination)` | CUDA indexed mesh；Fresh separate destination; linear 1-to-4 triangle subdivision and shared-midpoint welding; preserves shape, not Catmull-Clark smoothing. |
| weld | `weld_vertices(mesh,tolerance)` | CUDA indexed mesh；Positive finite tolerance; first representative within radius; O(n^2); does not remove degenerate faces created by a large tolerance. |
| normals | `mesh_normals(mesh)` | CUDA indexed mesh；Area-weighted vertex normals; valid oriented indexed triangles required; sharp edges require split vertices. |
| bevel_vertices | `bevel_vertices(mesh,points,nv,face_indices,face_offsets,nf,fraction)` | CUDA indexed mesh；Vertex truncation of a closed convex tri-valent polyhedron: <=64 vertices, <=96 edges, convex faces <=16 corners; fraction in (0,0.5). Unsupported valence rejected; not general edge filleting. |

### transform（4）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| translate | `plus(point,translation); affine(point,matrix3x4)` | CUDA point transform；Finite coordinates. Apply to control points or mesh vertices; vectors/derivatives omit translation. |
| rotate | `rotate_axis(point,center,axis,angle)` | CUDA point transform；Finite nonzero arbitrary axis; radians; Rodrigues rotation. |
| scale | `scale_about(point,center,scale_xyz)` | CUDA point transform；Finite component scales; negative scales reflect orientation; caller updates face winding and normals for whole meshes. |
| mirror | `mirror_plane(point,origin,normal)` | CUDA point transform；Finite nonzero arbitrary plane normal; caller updates face winding and normals for whole meshes. |

### curve_operation（5）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| curve_reverse | `reverse_curve_jet(base,t)` | CUDA curve jet reversal；Reverses parameter domain and first derivative; second derivative is preserved; base kinds 0..6. |
| curve_length | `curve_arc_length(record,first,last,panels)` | CUDA quadrature + independent CAD/analytic lengths；Composite five-point Gaussian quadrature of speed; positive panel count and regular finite base curve required; numerical estimate, not certified adaptive integration. |
| resample_curve | `curve_distance_parameter(record,fraction)` | CUDA arc-length inversion；Numerical arc-length inversion by bisection, fraction in [0,1]; regular nonzero-length base curve; fixed integration work bounds accuracy. |
| closest_point_curve | `closest_curve_parameter(record,query)` | CUDA bounded search + CAD projector；128-interval coarse basin selection plus golden-section refinement and endpoint checks; approximate nearest point, not a global proof for arbitrary oscillatory/self-crossing curves. |
| planar_offset_curve | `planar_curve_offset(jet,normal,distance)` | CUDA point offset + native CAD offset；Regular planar curve with nonzero plane normal; left offset normal=cross(normal,tangent); returns a point, not offset derivatives; no self-intersection cleanup. |

### solid（11）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| solid_sphere | `make_solid_definition(record,0,r,0,0,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive radius; GPU definition/mesh and exact analytic CPU B-rep topology; orthonormal right-handed frame. |
| solid_ellipsoid | `make_solid_definition(record,1,rx,ry,rz,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Three positive radii; affine-transformed exact spherical B-rep represented as rational surfaces; GPU faceted preview is separate. |
| solid_cylinder | `make_solid_definition(record,2,r,height,0,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive radius/height; closed analytic side and planar caps; local axis Z=cross(X,Y). |
| solid_cone | `make_solid_definition(record,3,r,height,0,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive base radius/height, one apex; closed exact conical B-rep. |
| solid_frustum | `make_solid_definition(record,4,r_bottom,r_top,height,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive radii/height; fixture uses unequal radii; equal radii should use cylinder rather than OCCT cone builder. |
| solid_torus | `make_solid_definition(record,5,major,minor,0,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Ring torus major>minor>0; exact native toroidal B-rep, separate GPU preview mesh. |
| solid_capsule | `make_solid_definition(record,6,r,cylinder_height,0,0,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive radius/cylinder height; hemispherical ends, exact revolved CAD profile and bounded GPU latitude mesh. |
| solid_prism | `make_solid_definition(record,7,circumradius,height,0,sides,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Regular polygon prism, integer sides 3..64; positive circumradius/height; exact planar CAD faces. |
| solid_pyramid | `make_solid_definition(record,8,circumradius,height,0,sides,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Regular polygon base with centered apex, sides 3..64; exact planar CAD faces; not a general arbitrary-base pyramid. |
| solid_wedge | `make_solid_definition(record,9,width,depth,height,top_width,origin,x,y); solid_from_definition(record)` | CUDA definition/mesh + native CPU CAD solid；Positive sizes and 0<top_width<width; one sloped side, rectangular bottom/top and trapezoidal end faces. |
| revolve_profile | `revolve_profile(mesh,profile,count,segments); revolve_face(face,origin,axis,angle)` | CUDA profile revolution + exact native CAD revolve；GPU full 2pi revolution of a simple CCW closed positive-radius RZ profile, <=32 corners. Native CAD supports a planar closed face and arbitrary axis/angle; mesh is a separate faceted preview. |

### deformation（3）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| bend | `bend_point(point,radius)` | CUDA point deformation + independent invariants；Local X bends into XZ about Y; radius>0 and radius+Z>0; caller constrains angle to avoid overlaps. STEP mesh fixture is faceted, not an exact deformed NURBS claim. |
| twist | `twist_point(point,radians_per_length)` | CUDA point deformation + independent invariants；Rotation about local Z by rate*Z; geometry may self-intersect for excessive deformation. Mesh STEP is faceted. |
| taper | `taper_point(point,rate)` | CUDA point deformation + independent invariants；Local XY scale=1+rate*Z; strictly positive scale required; mesh STEP is faceted. |

### mesh_operation（4）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| catmull_clark | `catmull_clark(destination,points,nv,quad_indices,nf)` | CUDA quad smoothing subdivision；One step on closed consistently oriented manifold quads, <=128 vertices/faces, <=256 edges; fresh destination; boundary/crease rules are separate operations. |
| loop_subdivide | `loop_subdivide(destination,points,nv,triangle_indices,nt)` | CUDA triangle smoothing subdivision；One original Loop step, closed consistently oriented manifold triangles; <=128 vertices, <=256 triangles, <=384 edges; fresh destination. |
| smooth_laplacian | `laplacian_smooth(mesh,iterations,lambda)` | CUDA uniform adjacency smoothing + dense matrix reference；Uniform adjacency Laplacian, <=512 vertices/1024 triangles, 1..64 iterations, 0<lambda<1; boundaries move and volume can shrink. |
| smooth_taubin | `taubin_smooth(mesh,iterations,lambda,mu)` | CUDA two-pass smoothing + dense matrix reference；Alternating lambda/mu uniform Laplacian, -1<mu<0; bounded mesh as above. Reduces shrinkage in the fixture but does not guarantee exact volume preservation. |

### resampling（2）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| resample_grid | `resample_grid(destination,points,nu,nv,u_steps,v_steps)` | CUDA bilinear grid resampling；Bilinear sampling of a 2..32 by 2..32 source grid; output within bounded mesh capacity; preserves bilinear geometry, no curvature reconstruction. |
| resample_polyline | `resample_polyline(output,samples,points,count)` | CUDA accumulated chord-distance resampling；Uniform accumulated chord distance, 2..128 finite input points, 2..4096 outputs; zero-length adjacent segments rejected. |

### curve_edit（8）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| knot_insert | `cad_advanced.knot_insert` | CPU OCCT / constrained NumPy fit；Exact interior B-spline insertion; nonperiodic tested. |
| knot_remove | `cad_advanced.knot_remove` | CPU OCCT / constrained NumPy fit；Native removal with a requested tolerance; can reject removal. |
| degree_elevate | `cad_advanced.degree_elevate` | CPU OCCT / constrained NumPy fit；Exact B-spline degree elevation to <=25. |
| degree_reduce | `cad_advanced.degree_reduce` | CPU OCCT / constrained NumPy fit；Single nonrational Bezier span; constrained LS and 2048 off-grid checks, sampled error only. |
| curve_split | `cad_advanced.curve_split` | CPU OCCT / constrained NumPy fit；Exact two-way split strictly inside the domain. |
| curve_reparameterize | `cad_advanced.reparameterize` | CPU OCCT / constrained NumPy fit；Affine increasing finite domain map. |
| fit_curve | `cad_advanced.fit_curve` | CPU OCCT / constrained NumPy fit；Nonperiodic cubic least squares, chord parameters, fixed endpoints. |
| fair_curve | `cad_advanced.fair_curve` | CPU OCCT / constrained NumPy fit；Penalized second control-pole differences; fixed endpoints, no interpolation promise. |

### advanced_sweep_loft（5）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| sweep_guided | `cuda/advanced_operations.cuh:sweep_sections` | CUDA frame/profile construction + CPU native B-spline sheet；CUDA projected guide rail defines profile frame; nonzero projected rail required. |
| sweep_scaled | `cuda/advanced_operations.cuh:sweep_sections` | CUDA frame/profile construction + CPU native B-spline sheet；CUDA positive scalar per path station; <=32 stations and <=31 corners. |
| sweep_variable | `cuda/advanced_operations.cuh:sweep_sections` | CUDA frame/profile construction + CPU native B-spline sheet；CUDA compatible polygon profile per station; bilinear side sheet, no self-intersection guarantee. |
| sweep_closed | `cuda/advanced_operations.cuh:sweep_sections` | CUDA frame/profile construction + CPU native B-spline sheet；CUDA RMF holonomy distributed by chord length; duplicate endpoint/profile/scale; G0 closed seam, no continuous G1 guarantee. |
| loft_compatible | `cad_advanced.compatible_sections + cad_ops.loft` | CPU exact B-spline compatibility + native loft；Exact degree/knot/domain harmonization then native loft; caller supplies matching orientation/correspondence. |

### surface_edit（7）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| surface_extend | `cad_advanced.extend_bezier_u` | CUDA control net / CPU OCCT surface and independent NumPy reference；Exact polynomial U extrapolation of a nonrational Bezier patch with original [0,1] domain. |
| surface_untrim | `cad_advanced.untrim_surface` | CUDA control net / CPU OCCT surface and independent NumPy reference；Rebuild the natural support-domain face; does not restore discarded upstream geometry. |
| surface_replace | `cad_advanced.replace_surface_face` | CUDA control net / CPU OCCT surface and independent NumPy reference；Isolated untrimmed rectangular patch, matching sampled boundary; arbitrary solid-face replacement excluded. |
| match_g1 | `cuda/advanced_operations.cuh:match_bezier_boundary` | CUDA control net / CPU OCCT surface and independent NumPy reference；Compatible bicubic Bezier U border, same V orientation; positive derivative scale. Unit scale tested C1. |
| match_g2 | `cuda/advanced_operations.cuh:match_bezier_boundary` | CUDA control net / CPU OCCT surface and independent NumPy reference；Same compatible border; position/D1/D2 rows matched. Unit scale tested C2. |
| fair_patches | `cuda/advanced_operations.cuh:fair_patch_pair` | CUDA control net / CPU OCCT surface and independent NumPy reference；Joint 7x4 composite net; <=64 projected bending-gradient steps, C2-compatible locked outer boundary and orthogonal seam projection. |
| surface_quality | `cad_advanced.surface_quality` | CUDA control net / CPU OCCT surface and independent NumPy reference；Sampled Jacobian/principal curvature diagnostics; not global regularity/Class-A certification. |

### solid_feature（7）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| fillet | `cad_advanced.fillet` | CPU native B-rep feature；Native selected-edge constant positive radius; infeasible blends reject. |
| variable_fillet | `cad_advanced.fillet` | CPU native B-rep feature；Native linear R1/R2 law along selected contour; positive feasible radii. |
| chamfer | `cad_advanced.chamfer` | CPU native B-rep feature；Native symmetric selected-edge chamfer. |
| thicken | `cad_advanced.thicken` | CPU native B-rep feature；Native simple face/shell offset and side walls; no general offset self-intersection solver. |
| shell | `cad_advanced.shell` | CPU native B-rep feature；Native remove-selected-face and join offset; feasible test box, negative inward thickness. |
| draft | `cad_advanced.draft` | CPU native B-rep feature；Native planar/cylindrical/conical face draft, abs(angle)<45deg, explicit neutral plane. |
| rib | `cad_advanced.rib` | CPU native B-rep feature；Thicken an explicit planar rib profile and Boolean-fuse with base; not automatic skeleton ribs. |

### mesh_edit（6）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| decimate | `mesh_advanced.decimate` | CPU indexed mesh topology；CPU shortest-edge collapse; closed oriented manifold, link/no-flip tests and vertex-displacement budget; not a Hausdorff guarantee. |
| adaptive_remesh | `mesh_advanced.adaptive_remesh` | CPU indexed mesh topology；CPU conforming length-driven midpoint refinement; piecewise planar geometry preserved; <=16 passes. |
| edge_flip | `mesh_advanced.edge_flip` | CPU indexed mesh topology；CPU two-triangle planar convex interior edge; reject duplicate diagonal. |
| bridge_loops | `mesh_advanced.bridge_loops` | CPU indexed mesh topology；CPU disjoint equally sized oriented boundary loops with caller correspondence; may return open surface. |
| region_extrude | `mesh_advanced.region_extrude` | CPU indexed mesh topology；CPU selected triangle region translated outward with boundary walls; vector must point out of every selected face. |
| crease_subdivide | `mesh_advanced.crease_subdivide` | CPU indexed mesh topology；CPU one Loop step with hard edge/boundary rules and >=3-crease corner lock. |

### repair（5）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| self_intersection_detect | `mesh_advanced.self_intersections` | CPU explicit mesh repair / native shape healing；CPU nonadjacent triangle crossing pairs; O(T^2), coplanar overlapping candidate rejects as unsupported. |
| self_intersection_repair | `mesh_advanced.repair_self_intersections` | CPU explicit mesh repair / native shape healing；CPU explicit face-discard policy covering detected crossings; reports open result, no automatic closed reconstruction. |
| nonmanifold_repair | `mesh_advanced.repair_nonmanifold` | CPU explicit mesh repair / native shape healing；CPU vertex-fan splitting, same face geometry, explicit new boundaries. |
| tolerance_unify | `cad_advanced.unify_tolerance` | CPU explicit mesh repair / native shape healing；Copied native shape tolerances 1e-9..1e-3 mm; does not heal larger geometric gaps. |
| small_feature_cleanup | `cad_advanced.cleanup_small_features` | CPU explicit mesh repair / native shape healing；Native same-domain unification and small-edge/wireframe repair; coplanar sliver test. Arbitrary curved sliver removal not guaranteed. |

### orchestration（6）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| stable_handles | `feature_graph.Handle` | CPU typed feature graph；Typed feature ids stable through parameter/history/save-load; never reused in live graph. |
| select_subshape | `feature_graph.FeatureGraph.select/resolve` | CPU typed feature graph；Unique geometric centroid point/edge/face query; revision-stale or ambiguous selections reject. |
| feature_dag | `feature_graph.FeatureGraph` | CPU typed feature graph；Typed DAG, cycle rejection, descendant-only invalidation and memoized execution. |
| history_undo | `feature_graph.FeatureGraph.undo` | CPU typed feature graph；Undo graph edit, clear caches, invalidate all prior subshape selections. |
| history_redo | `feature_graph.FeatureGraph.redo` | CPU typed feature graph；Redo graph edit; branching edit clears redo. |
| feature_rebuild | `feature_graph.FeatureGraph.rebuild/loads` | CPU typed feature graph；Deterministic registered operations and JSON graph; same geometry checked after fresh load/rebuild. |

### placement（5）

| 原语 | 入口 | 后端 / 范围 |
|---|---|---|
| snap_point | `snap_grid / snap_vertices / snap_point` | CUDA grid/vertex + CPU native CAD；CUDA grid and <=4096 nearest vertices; CPU native CAD nearest point on edge/trimmed face. Capture radius can reject. Native ties return first solution. |
| merge_mesh | `merge_mesh(meshes,tolerance)` | CPU indexed merge/weld；CPU concatenate indexed objects; optional first-representative weld. Internal faces remain; solid union uses boolean_union, surface merge uses sew_surfaces. |
| move_along_curve | `move_curve_arc(curve,start_parameter,signed_distance,clamp,...)` | CUDA arc-length positioning + CPU instance transform；CUDA bounded analytic/B-spline curve records; numerical Gauss arc length and inverse solve; open-domain clamp or reject, unit tangent frame. No periodic wrapping. |
| move_on_mesh | `move_mesh_path(points,nv,triangles,nt,start_face,start_point,direction,distance,...)` | CUDA intrinsic triangle walk + CPU instance transform；CUDA straightest face walk with edge parallel transport; <=128 vertices,256 triangles/crossings. Ambiguous vertex hits, nonmanifold crossings and degenerate faces reject; open boundary clamps with status=2. |
| move_on_surface | `move_surface_uv(surface,u,v,du,dv,...)` | CUDA UV placement/frame + CPU instance transform；CUDA NURBS UV displacement returns attached point, path tangent and normal; finite domain, regular surface and nonzero displacement. UV trajectory is not a world-space geodesic. |

## STEP 交付

| 文件 | 实体数 | 面数 | 字节 |
|---|---:|---:|---:|
| surface_primitives.step | 1 | 30 | 378804 |
| curve_primitives.step | 0 | 0 | 24981 |
| g2_transition.step | 0 | 3 | 13904 |
| polygon_primitives.step | 10 | 2491 | 5787882 |
| extended_curves.step | 0 | 0 | 123620 |
| solid_primitives.step | 10 | 34 | 96602 |
| operation_surfaces.step | 1 | 4 | 69362 |
| operation_meshes.step | 6 | 2256 | 5208082 |
| boolean_features.step | 5 | 43 | 117467 |
| topology_meshes.step | 4 | 854 | 1942497 |
| curve_editing.step | 0 | 0 | 23080 |
| advanced_surfaces.step | 0 | 12 | 169953 |
| solid_features.step | 10 | 84 | 241008 |
| advanced_meshes.step | 3 | 176 | 384287 |
| placement_operations.step | 5 | 34 | 103623 |
| composed_panel.step | 1 | 12 | 83049 |

另保留 `output/car/car.step` 与 `output/car-variant/car.step` 的神经网络车身示例。
解析/B-spline 曲面按原生 CAD 几何交付；多边形编辑结果导出为分面 B-rep，不能将其当作自动恢复的光顺 NURBS。
STEP 已由 OCCT 校验，未声称在本机 SolidWorks/CATIA 中完成实机导入验证。

## 关键边界

- G1/G2 自动匹配限兼容 bicubic Bezier 边界；双片平顺保持 C2 与兼容外边界，不代表全车 Class-A。
- 闭环 sweep 修正离散帧扭转，几何接缝验收 G0；任意自交、拐角奇异和全局连续性需进一步处理。
- 沿曲线按数值弧长移动；沿网格使用跨边平行输运的直线走步，顶点歧义拒绝、开边界截停；沿曲面按 UV 轨迹贴附，不是世界坐标测地线。
- merge 明确区分索引合并/焊点、曲面缝合和实体 Boolean union；索引 merge 不自动删除内部面。
- 自交修复是显式删面策略，非流形修复是分裂顶点扇；均报告新边界，不冒充任意闭实体重构。
- 小特征修复支持同域/共面碎片和原生小边修复，任意曲面薄片无法保证自动消除。
- 对象 id 稳定；几何子选择属于结果 revision，重建后须重选，不承诺通用持久拓扑命名。
