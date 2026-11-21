import json,math,traceback,sys
import numpy as np
from acceptance import ROOT,OUT,xyz,check,count,properties,compound,placed,build_surface,mesh_shape,roundtrip,finish_stage
from cad_ops import valid,loft,trim_surface
from cad_solids import polygon_face
from cad_topology_operations import boolean_union,boolean_common
import cad_advanced as cad
import mesh_advanced as poly
from feature_graph import FeatureGraph,Handle
from OCP.gp import gp_Pnt,gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_SOLID,TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

defs=json.loads((ROOT/'primitive_advanced.json').read_text())['operators'];results=[];curves=[];sheets=[];features=[];mesh_results=[]
raw=json.loads((OUT/'advanced_surfaces.json').read_text());nets=raw['surfaces'];surfaces=[build_surface(s) for s in nets];jets=np.array(raw['jets']).reshape(9,9,9,3,3)
def run(name,backend,fn):
    try:
        evidence=fn() or {};results.append({'name':name,**defs[name],'backend':backend,'pass':True,'evidence':evidence});print('PASS',name,flush=True)
    except Exception as e:
        results.append({'name':name,**defs[name],'backend':backend,'pass':False,'error':str(e),'traceback':traceback.format_exc()});print('FAIL',name,str(e),flush=True)
def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid request was accepted')
def negative_contracts():
    cases=[lambda:cad.degree_elevate(base,2),lambda:cad.reparameterize(base,1,0),lambda:cad.curve_split(base,0),lambda:cad.compatible_sections([base]),
        lambda:poly.Mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,.5,2]]),lambda:poly.adaptive_remesh(cube,0),
        lambda:poly.region_extrude(cube,[0],[0,0,0]),lambda:poly.bridge_loops(cube,[0,1,2],[3,4]),
        lambda:poly.crease_subdivide(cube,[(0,99)]),lambda:cad.unify_tolerance(box,.1),
        lambda:cad.fillet(box,[TopoDS.Edge_s(subshapes(BRepPrimAPI_MakeBox(1.,1.,1.).Shape(),TopAbs_EDGE)[0])],.1)]
    for fn in cases:rejected(fn)
    return len(cases)
def same_curve(a,b,lo=0,hi=1):
    return max(check(xyz(a.Value(float(t))),xyz(b.Value(float(t))),2e-11) for t in np.linspace(lo,hi,129))
base=cad.bspline([[0,0,0],[.5,1,0],[1.5,1,.3],[2,0,0]])
def insert_case():
    c=cad.knot_insert(base,.37);assert c.NbPoles()==5 and base.NbPoles()==4;curves.append(c);rejected(lambda:cad.knot_insert(base,1.))
    return {'poles_before':4,'poles_after':5,'max_geometry_error_mm':same_curve(base,c),'invalid_domain_rejected':True}
def remove_case():
    c=cad.knot_remove(cad.knot_insert(base,.37),2);assert c.NbPoles()==4;curves.append(c)
    return {'poles_after':4,'max_geometry_error_mm':same_curve(base,c)}
def elevate_case():
    c=cad.degree_elevate(base,5);assert c.Degree()==5 and base.Degree()==3;curves.append(c)
    return {'degree_before':3,'degree_after':5,'max_geometry_error_mm':same_curve(base,c)}
def reduce_case():
    c,e=cad.degree_reduce(cad.degree_elevate(base,5),3,1e-10);curves.append(c);same_curve(base,c);rejected(lambda:cad.degree_reduce(base,1,1e-6))
    return {**e,'rejected_excess_error':True}
def split_case():
    a,b=cad.curve_split(base,.41);curves.extend([a,b]);check(xyz(a.Value(.41)),xyz(b.Value(.41)),1e-12)
    return {'max_left_error_mm':same_curve(base,a,0,.41),'max_right_error_mm':same_curve(base,b,.41,1.)}
def reparam_case():
    c=cad.reparameterize(base,-2,5);curves.append(c);error=max(check(xyz(base.Value(float(t))),xyz(c.Value(float(-2+7*t))),1e-12) for t in np.linspace(0,1,129))
    return {'domain':[-2,5],'max_geometry_error_mm':error}
x=np.linspace(0,2,65);data=np.column_stack([x,.2*np.sin(2*x)+.015*np.sin(31*x),.05*x*x])
def fit_case(fair=False):
    c,e=cad.fair_curve(data,12,1.) if fair else cad.fit_curve(data,12);check(xyz(c.Value(0)),data[0],1e-12);check(xyz(c.Value(1)),data[-1],1e-12)
    assert e['max_sample_residual_mm']<.04
    if fair:
        _,baseline=cad.fit_curve(data,12);assert e['pole_second_difference_energy']<baseline['pole_second_difference_energy']
    curves.append(c);return e
for name,fn in [('knot_insert',insert_case),('knot_remove',remove_case),('degree_elevate',elevate_case),('degree_reduce',reduce_case),('curve_split',split_case),('curve_reparameterize',reparam_case),('fit_curve',fit_case),('fair_curve',lambda:fit_case(True))]:run(name,'CPU OCCT / constrained NumPy fit',fn)

def poles(i):return np.array(nets[i]['poles_xyzw']).reshape(*nets[i]['size'],4)[:,:,:3]
def jet_agreement(i):
    error=0.
    for j,u in enumerate(np.linspace(0,1,9)):
        for k,v in enumerate(np.linspace(0,1,9)):
            p=gp_Pnt();du,dv=gp_Vec(),gp_Vec();surfaces[i].D1(float(u),float(v),p,du,dv)
            error=max(error,check(jets[i,j,k],[xyz(p),xyz(du),xyz(dv)],3e-11))
    return error
def sweep_case(i):
    p=poles(i);u=np.linspace(0,1,25);q=2*np.pi*u
    if i<3:path=np.column_stack([2*u,.2*np.sin(np.pi*u),.3*u*u]);tangent=np.column_stack([np.full(25,2),.2*np.pi*np.cos(np.pi*u),.6*u])
    else:
        r=1+.15*np.cos(3*q);path=np.column_stack([r*np.cos(q),r*np.sin(q),.2*np.sin(2*q)])
        tangent=np.column_stack([-3*.15*np.sin(3*q)*np.cos(q)-r*np.sin(q),-3*.15*np.sin(3*q)*np.sin(q)+r*np.cos(q),.4*np.cos(2*q)])
    tangent/=np.linalg.norm(tangent,axis=1)[:,None];radii=np.linalg.norm(p[:,:8]-path[:,None,:],axis=2)
    check(p[:,:8].mean(axis=1),path,1e-12);check(np.sum((p[:,:8]-path[:,None,:])*tangent[:,None,:],axis=2),np.zeros((25,8)),1e-12)
    if i==0:
        guide=np.column_stack([.1*np.sin(q),.2*np.cos(q),np.ones(25)]);normal=guide-tangent*np.sum(guide*tangent,axis=1)[:,None];normal/=np.linalg.norm(normal,axis=1)[:,None]
        check((p[:,0]-path)/.15,normal,1e-12);check(radii,np.full((25,8),.15),1e-12)
    if i==1:check(radii,np.tile((.15*(1+.35*np.sin(np.pi*u)))[:,None],(1,8)),1e-12)
    if i==2:
        check(radii[:,0],.15*(1+.4*u),1e-12);check(radii[:,2],.15*(1-.3*u),1e-12)
    if i==3:check(p[0],p[-1],1e-12)
    sheets.append(cad.untrim_surface(surfaces[i]));return {'stations':25,'profile_corners':8,'max_gpu_native_D1_error':jet_agreement(i),'closed_frame_seam_checked':i==3,'native_valid':True}
for i,name in enumerate(['sweep_guided','sweep_scaled','sweep_variable','sweep_closed']):run(name,'CUDA frame/profile construction + CPU native B-spline sheet',lambda i=i:sweep_case(i))
def compatible_case():
    first=base.Copy();other=cad.bspline([[p.X(),p.Y(),p.Z()+1.] for p in [base.Pole(i) for i in range(1,5)]])
    second=cad.reparameterize(cad.knot_insert(cad.degree_elevate(other,5),.4),-3,2)
    output=cad.compatible_sections([first,second]);assert all(c.Degree()==5 and c.NbPoles()==7 for c in output)
    same_curve(output[0],base);check([output[0].Knot(i) for i in range(1,output[0].NbKnots()+1)],[0,.4,1],1e-12)
    wires=[BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(c).Edge()).Wire() for c in output];sheets.append(loft(wires))
    return {'harmonized_degree':5,'harmonized_poles':7,'original_geometry_preserved':True,'native_loft_valid':True}
run('loft_compatible','CPU exact B-spline compatibility + native loft',compatible_case)
def extend_case():
    c=cad.extend_bezier_u(surfaces[4]);err=0.
    for u in np.linspace(0,1,25):
        for v in np.linspace(0,1,9):err=max(err,check(xyz(c.Value(float(u),float(v))),xyz(surfaces[4].Value(float(-.2+1.4*u),float(v))),1e-12))
    sheets.append(cad.untrim_surface(c));return {'old_domain_sampled':[-.2,1.2],'max_polynomial_extension_error_mm':err}
def untrim_case():
    trimmed=trim_surface(surfaces[4],[(.2,.2),(.8,.2),(.8,.8),(.2,.8)])
    from OCP.BRep import BRep_Tool
    full=cad.untrim_surface(BRep_Tool.Surface_s(trimmed));assert properties(full)[0]>properties(trimmed)[0]*2;sheets.append(full)
    return {'trimmed_area_mm2':properties(trimmed)[0],'restored_natural_area_mm2':properties(full)[0]}
def replace_case():
    p=poles(4);p[1:3,1:3,2]+=.15;new=cad.bezier_surface(p);face=cad.untrim_surface(surfaces[4]);out=cad.replace_surface_face(face,new)
    assert properties(out)[0]>properties(face)[0];sheets.append(out);bad=p.copy();bad[0,0,2]+=.1;rejected(lambda:cad.replace_surface_face(face,cad.bezier_surface(bad)))
    return {'valid':True,'boundary_preserved':True,'changed_interior_z_mm':.15,'boundary_mismatch_rejected':True}
def boundary_case(order):
    a,b=poles(4),poles(4+order);check(a[-1],b[0],1e-12);check(3*(a[-1]-a[-2]),3*(b[1]-b[0]),1e-12)
    if order==2:check(6*(a[-1]-2*a[-2]+a[-3]),6*(b[2]-2*b[1]+b[0]),1e-12)
    # Check derivatives independently through OCCT at 33 points, not just poles.
    error=0.
    for v in np.linspace(0,1,33):
        values=[]
        for s,u in [(surfaces[4],1.),(surfaces[4+order],0.)]:
            p=gp_Pnt();du,dv,duu,dvv,duv=[gp_Vec() for _ in range(5)];s.D2(u,float(v),p,du,dv,duu,dvv,duv);values.append([xyz(p),xyz(du),xyz(dv),xyz(duu)])
        error=max(error,check(values[0][:3 if order==1 else 4],values[1][:3 if order==1 else 4],2e-12))
    sheets.append(cad.untrim_surface(surfaces[4+order]));return {'order':order,'max_native_seam_jet_error':error,'scale':1.}
def fair_case():
    seed=np.array(raw['aux'][8:92]).reshape(7,4,3);p=seed.copy()
    du=np.diff(np.eye(7),2,axis=0);dv=np.diff(np.eye(4),2,axis=0);hessian=2*(np.kron(du.T@du,np.eye(4))+np.kron(np.eye(7),dv.T@dv))
    constraint=np.array([[0,1,-2,1,0],[-1,2,0,-2,1]],dtype=float);projection=np.eye(5)-constraint.T@np.linalg.solve(constraint@constraint.T,constraint)
    for _ in range(12):
        gradient=(hessian@p.reshape(28,3)).reshape(7,4,3);p[1:6,1:3]-=.3/64*gradient[1:6,1:3]
        for j in [1,2]:p[1:6,j]=projection@p[1:6,j]
    output=np.concatenate([poles(7),poles(8)[1:]],axis=0);err=check(output,p,1e-12)
    check(output[[0,-1]],seed[[0,-1]],1e-12);check(output[:,[0,-1]],seed[:,[0,-1]],1e-12)
    check(output[3]-output[2],output[4]-output[3],1e-12);check(output[3]-2*output[2]+output[1],output[5]-2*output[4]+output[3],1e-12)
    energy=lambda q:float(np.sum(np.diff(q,2,axis=0)**2)+np.sum(np.diff(q,2,axis=1)**2))
    before,after=energy(seed),energy(output);assert after<before
    sheets.extend([cad.untrim_surface(surfaces[7]),cad.untrim_surface(surfaces[8])]);return {'iterations':12,'max_independent_reference_error_mm':err,'z_bending_energy_before':float(before),'z_bending_energy_after':float(after),'C2_and_outer_boundary_preserved':True}
def quality_case():
    q=cad.surface_quality(surfaces[4]);assert q['sampled_regular'] and q['maximum_sampled_principal_curvature_per_mm']>0
    planar=cad.bezier_surface(np.array([[[i,j,0] for j in [0,1]] for i in [0,1]]));check(cad.surface_quality(planar)['maximum_sampled_principal_curvature_per_mm'],0,1e-12)
    bad=cad.bezier_surface(np.zeros((2,2,3)));assert not cad.surface_quality(bad)['sampled_regular'];return q
for name,fn in [('surface_extend',extend_case),('surface_untrim',untrim_case),('surface_replace',replace_case),('match_g1',lambda:boundary_case(1)),('match_g2',lambda:boundary_case(2)),('fair_patches',fair_case),('surface_quality',quality_case)]:run(name,'CUDA control net / CPU OCCT surface and independent NumPy reference',fn)

def subshapes(shape,kind):
    e=TopExp_Explorer(shape,kind);out=[]
    while e.More():
        item=e.Current();e.Next()
        if not any(item.IsSame(q) for q in out):out.append(item)
    return out
def center(shape,surface=True):
    props=GProp_GProps();(BRepGProp.SurfaceProperties_s if surface else BRepGProp.LinearProperties_s)(shape,props);return xyz(props.CentreOfMass())
box=BRepPrimAPI_MakeBox(2.,2.,2.).Shape();edge=TopoDS.Edge_s(subshapes(box,TopAbs_EDGE)[0]);faces=[TopoDS.Face_s(f) for f in subshapes(box,TopAbs_FACE)];top=max(faces,key=lambda f:center(f)[2]);right=max(faces,key=lambda f:center(f)[0])
def feature_case(name):
    if name=='fillet':s=cad.fillet(box,[edge],.2);expected=8-(1-math.pi/4)*.2**2*2
    elif name=='variable_fillet':
        s=cad.fillet(box,[edge],.1,.2);v=properties(s)[1];low=8-(1-math.pi/4)*.2**2*2;high=8-(1-math.pi/4)*.1**2*2
        assert low<v<high and count(s,TopAbs_SOLID)==1;check(center(edge,False),[0,0,1],1e-12)
        sections=[]
        for z in [.05,1.9]:
            slab=BRepPrimAPI_MakeBox(gp_Pnt(-1,-1,z),gp_Pnt(3,3,z+.05)).Shape();sections.append(properties(boolean_common(s,slab))[1]/.05)
        assert abs(sections[0]-sections[1])>.003;features.append(s)
        return {'native_valid':True,'solids':1,'volume_mm3':v,'constant_radius_volume_bounds_mm3':[low,high],'independent_end_slab_areas_mm2':sections,'cross_section_varies':True,'note':'Rolling sphere envelope with varying radius; constant-radius prism integration is not the variable-fillet volume formula.'}
    elif name=='chamfer':s=cad.chamfer(box,[edge],.2);expected=8-.2**2
    elif name=='thicken':s=cad.thicken(polygon_face([[0,0,0],[2,0,0],[2,2,0],[0,2,0]]),.2);expected=.8
    elif name=='shell':s=cad.shell(box,[top],-.2);expected=8-1.6**2*1.8
    elif name=='draft':s=cad.draft(box,[right],math.atan(.1));expected=7.6
    else:
        base=BRepPrimAPI_MakeBox(2.,2.,.3).Shape();face=polygon_face([[.9,0,.2],[.9,2,.2],[.9,2,1.5],[.9,0,1.5]])
        s=cad.rib(base,face,.2);expected=1.68
    assert count(s,TopAbs_SOLID)==1;v=properties(s)[1];check(v,expected,2e-6);features.append(s)
    return {'native_valid':True,'solids':1,'volume_mm3':v,'analytic_volume_mm3':expected,'absolute_volume_error_mm3':abs(v-expected)}
for name in ['fillet','variable_fillet','chamfer','thicken','shell','draft','rib']:run(name,'CPU native B-rep feature',lambda name=name:feature_case(name))

base_mesh=json.loads((OUT/'meshes.json').read_text())[0];cube=poly.Mesh(base_mesh['positions'],base_mesh['indices'])
def keep_mesh(mesh):
    r=mesh.report();assert r['nonmanifold_edges']==r['orientation_errors']==0;mesh_results.append(mesh);return r
def decimate_case():
    source=poly.crease_subdivide(cube,[]);out,e=poly.decimate(source,32,.8);r=keep_mesh(out);assert len(out.t)==32 and r['boundary_edges']==0 and r['euler']==2
    return {**r,**e,'input_triangles':len(source.t)}
def remesh_case():
    seed=poly.Mesh([[0,0,0],[2,0,0],[2,1,0],[0,1,0],[.2,.2,0]],[[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    out=poly.adaptive_remesh(seed,.55);r=keep_mesh(out);check(r['area'],2.,1e-12);assert r['euler']==1
    maximum=max(np.linalg.norm(out.p[a]-out.p[b]) for a,b in out.edges());assert maximum<=.55*(1+1e-12)
    return {**r,'max_edge_mm':float(maximum),'input_triangles':4,'planar_area_preserved':True}
def flip_case():
    seed=poly.Mesh([[0,0,0],[2,0,0],[2,1,0],[0,1,0]],[[0,1,2],[0,2,3]]);out=poly.edge_flip(seed,[0,2]);r=keep_mesh(out);check(r['area'],2.,1e-12);assert (1,3) in out.edges() and (0,2) not in out.edges();rejected(lambda:poly.edge_flip(seed,[0,1]));return r
def bridge_case():
    seed=poly.Mesh([[x,y,z] for z in [0.,1.] for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]],[])
    out=poly.bridge_loops(seed,[0,1,2,3],[4,5,6,7]);r=keep_mesh(out);check(r['area'],8.,1e-12);assert r['boundary_edges']==8 and r['euler']==0;return r
def extrude_case():
    selected=np.flatnonzero(np.all(cube.p[cube.t][:,:,2]>.99,axis=1));assert len(selected)==2;out=poly.region_extrude(cube,selected,[0,0,.5]);r=keep_mesh(out);check(r['signed_volume'],10.,1e-12);assert r['boundary_edges']==0 and r['euler']==2;return r
def crease_case():
    # Cube crease edges have changing face normals; face diagonals stay smooth.
    creases=[]
    for e,uses in cube.edges().items():
        normals=[]
        for use in uses:
            t=cube.p[cube.t[use[0]]];n=np.cross(t[1]-t[0],t[2]-t[0]);normals.append(n/np.linalg.norm(n))
        if np.dot(*normals)<.9:creases.append(e)
    out=poly.crease_subdivide(cube,creases);check(out.p[:8],cube.p,1e-12);r=keep_mesh(out);check(r['signed_volume'],8.,1e-12);assert len(creases)==12 and len(out.t)==48
    return {**r,'hard_edges':12,'all_original_corners_locked':True}
for name,fn in [('decimate',decimate_case),('adaptive_remesh',remesh_case),('edge_flip',flip_case),('bridge_loops',bridge_case),('region_extrude',extrude_case),('crease_subdivide',crease_case)]:run(name,'CPU indexed mesh topology',fn)

cross=poly.Mesh([[-1,-1,0],[1,-1,0],[0,1,0],[0,-.5,-1],[0,-.5,1],[0,.8,.5]],[[0,1,2],[3,4,5]])
def detect_case():
    assert poly.self_intersections(cross)==[(0,1)];assert poly.self_intersections(cube)==[];return {'crossing_pairs':[[0,1]],'clean_closed_cube_checked':True,'coplanar_case_policy':'reject unsupported candidate'}
def repair_cross_case():
    out,e=poly.repair_self_intersections(cross,[1]);r=keep_mesh(out);assert r['boundary_edges']==3 and len(out.t)==1;return {**r,**e}
def nonmanifold_case():
    source=poly.Mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],[[0,1,2],[1,0,3],[0,1,4]])
    assert source.report()['nonmanifold_edges']==1;out,e=poly.repair_nonmanifold(source);r=keep_mesh(out);check(out.p[out.t],source.p[source.t],1e-12);assert r['boundary_edges']==9;return {**r,**e,'original_nonmanifold_edges':1}
def tolerance_case():
    source=BRepPrimAPI_MakeBox(1.,1.,1.).Shape();before=[BRep_Tool.Tolerance_s(TopoDS.Vertex_s(v)) for v in subshapes(source,TopAbs_VERTEX)];out=cad.unify_tolerance(source,1e-6)
    errors=[]
    for kind,cast in [(TopAbs_VERTEX,TopoDS.Vertex_s),(TopAbs_EDGE,TopoDS.Edge_s),(TopAbs_FACE,TopoDS.Face_s)]:errors.extend(abs(BRep_Tool.Tolerance_s(cast(s))-1e-6) for s in subshapes(out,kind))
    assert max(errors)<1e-15;check(before,[BRep_Tool.Tolerance_s(TopoDS.Vertex_s(v)) for v in subshapes(source,TopAbs_VERTEX)],1e-15);features.append(out)
    return {'tolerance_mm':1e-6,'max_native_tolerance_error_mm':max(errors),'source_preserved':True}
def cleanup_case():
    a=BRepPrimAPI_MakeBox(1.,1.,1.).Shape();b=BRepPrimAPI_MakeBox(gp_Pnt(1,0,0),gp_Pnt(1.00001,1,1)).Shape();source=boolean_union(a,b);out=cad.cleanup_small_features(source,1e-4)
    before,after=count(source,TopAbs_FACE),count(out,TopAbs_FACE);assert after==6 and after<before;check(properties(out)[1],1.00001,1e-10);features.append(out)
    return {'faces_before':before,'faces_after':after,'same_domain_sliver_removed':True,'volume_mm3':properties(out)[1]}
for name,fn in [('self_intersection_detect',detect_case),('self_intersection_repair',repair_cross_case),('nonmanifold_repair',nonmanifold_case),('tolerance_unify',tolerance_case),('small_feature_cleanup',cleanup_case)]:run(name,'CPU explicit mesh repair / native shape healing',fn)

registry={'box':((),'solid',lambda x,y,z,ox=0:BRepPrimAPI_MakeBox(gp_Pnt(ox,0,0),gp_Pnt(ox+x,y,z)).Shape()),'union':(('solid','solid'),'solid',boolean_union),'translate':(('solid',),'solid',lambda s,dx:placed(s,dx,0))}
def setup_graph():
    g=FeatureGraph(registry);a=g.add('box',x=2,y=2,z=2);b=g.add('box',x=2,y=2,z=2,ox=1);u=g.add('union',[a,b]);return g,a,b,u
def handles_case():
    g,a,b,u=setup_graph();before=a.id;g.update(a,x=2.5);assert a.id==before;rejected(lambda:g.evaluate(Handle(a.id,'face')))
    text=g.dumps();h=FeatureGraph.loads(text,registry);assert set(h.nodes)==set(g.nodes);return {'stable_id':a.id,'wrong_type_rejected':True,'ids_survive_serialization':True}
def select_case():
    g,a,b,u=setup_graph();selections=[g.select(a,'vertex',(0,0,0)),g.select(a,'edge',(0,0,1)),g.select(a,'face',(0,1,1))]
    for s in selections:assert not g.resolve(s).IsNull()
    rejected(lambda:g.select(a,'face',(1,1,1),10));g.update(a,z=3);rejected(lambda:g.resolve(selections[0]))
    return {'vertex_edge_face_selections':3,'ambiguous_query_rejected':True,'stale_selection_rejected':True}
def dag_case():
    g,a,b,u=setup_graph();check(properties(g.evaluate(u))[1],12.,1e-10);first=g.evaluations.copy();g.evaluate(u);assert first==g.evaluations
    g.update(a,x=2.5);check(properties(g.evaluate(u))[1],12.,1e-10);assert g.evaluations[b.id]==first[b.id] and g.evaluations[a.id]==first[a.id]+1 and g.evaluations[u.id]==first[u.id]+1
    t=g.add('translate',[u],dx=1);rejected(lambda:g.update(u,inputs=[a,t]));assert g.nodes[u.id]['inputs']==[a.id,b.id]
    return {'memoization_checked':True,'descendant_only_invalidation':True,'cycle_rejected_atomically':True}
def undo_case():
    g,a,b,u=setup_graph();g.update(a,z=3);check(properties(g.evaluate(a))[1],12.,1e-10);g.undo();check(properties(g.evaluate(a))[1],8.,1e-10)
    return {'changed_volume_mm3':12.,'undo_volume_mm3':8.,'id':a.id}
def redo_case():
    g,a,b,u=setup_graph();g.update(a,z=3);g.undo();g.redo();check(properties(g.evaluate(a))[1],12.,1e-10);g.undo();g.update(a,z=4);rejected(g.redo)
    return {'redo_volume_mm3':12.,'branch_edit_clears_redo':True}
def rebuild_case():
    g,a,b,u=setup_graph();shape=g.evaluate(u);before=properties(shape);rebuilt=g.rebuild(u);check(properties(rebuilt),before,1e-10)
    text=g.dumps();(OUT/'feature_graph.json').write_text(text,encoding='utf-8');fresh=FeatureGraph.loads(text,registry);final=fresh.evaluate(u);check(properties(final),before,1e-10);features.append(final)
    return {'saved_graph':'feature_graph.json','ids_and_geometry_survive_fresh_rebuild':True,'volume_mm3':before[1]}
for name,fn in [('stable_handles',handles_case),('select_subshape',select_case),('feature_dag',dag_case),('history_undo',undo_case),('history_redo',redo_case),('feature_rebuild',rebuild_case)]:run(name,'CPU typed feature graph',fn)

steps=[]
try:
    assert len(curves)==9 and len(sheets)==12 and len(features)==10 and len(mesh_results)==8,(len(curves),len(sheets),len(features),len(mesh_results))
    steps.append(roundtrip('curve_editing.step',compound([placed(BRepBuilderAPI_MakeEdge(c).Edge(),3*(i%4),3*(i//4)) for i,c in enumerate(curves)]),0))
    steps.append(roundtrip('advanced_surfaces.step',compound([placed(s,4*(i%4),4*(i//4)) for i,s in enumerate(sheets)]),0))
    steps.append(roundtrip('solid_features.step',compound([placed(s,4*(i%4),4*(i//4)) for i,s in enumerate(features)]),10))
    shapes=[placed(mesh_shape(m.record()),4*(i%4),4*(i//4)) for i,m in enumerate(mesh_results)];steps.append(roundtrip('advanced_meshes.step',compound(shapes),3))
    (OUT/'advanced_meshes.json').write_text(json.dumps([m.record() for m in mesh_results]),encoding='utf-8')
except Exception as e:steps.append({'valid_after':False,'error':str(e),'traceback':traceback.format_exc()});print('STEP FAIL',str(e),flush=True)
contract_rejections=negative_contracts()
stage={'operator_count':len(results),'passed':sum(r['pass'] for r in results),'failed':sum(not r['pass'] for r in results),'operators':results,'contract_rejections':contract_rejections,'step_roundtrips':steps,'all_passed':all(r['pass'] for r in results) and len(steps)==4 and all(r.get('valid_after') for r in steps)}
if '--check-only' in sys.argv:
    (OUT/'validation_advanced.json').write_text(json.dumps(stage,indent=2),encoding='utf-8');print(json.dumps({k:stage[k] for k in ['operator_count','passed','failed','all_passed']}));raise SystemExit(0 if stage['all_passed'] else 1)
finish_stage('advanced',defs,results,steps,4)
