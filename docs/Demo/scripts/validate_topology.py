import json,math,traceback
import numpy as np
from acceptance import ROOT,OUT,check,count,properties,compound,placed,mesh_shape,mesh_volume,roundtrip,finish_stage
from cad_ops import sew_surfaces
from cad_solids import polygon_face
from cad_topology_operations import boolean_union,boolean_cut,boolean_common,delete_cad_faces,cap_planar_wire,revolve_face
from OCP.gp import gp_Pnt
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_SHELL,TopAbs_WIRE,TopAbs_SOLID
from OCP.TopoDS import TopoDS

defs=json.loads((ROOT/'primitive_topology.json').read_text())['operators'];raw=json.loads((OUT/'topology_operations.json').read_text())
meshes=raw['meshes'];aux=np.array(raw['aux']);results=[];native=[]
def run(name,backend,fn):
    try:
        evidence=fn() or {};results.append({'name':name,**defs[name],'backend':backend,'pass':True,'evidence':evidence});print('PASS',name,flush=True)
    except Exception as e:
        results.append({'name':name,**defs[name],'backend':backend,'pass':False,'error':str(e),'traceback':traceback.format_exc()});print('FAIL',name,str(e),flush=True)
check(aux[:2],np.ones(2),1e-12);check(aux[16:28],[0,0,0,2,2,2,1,.5,.5,3,1.5,1.5],1e-12)
a=BRepPrimAPI_MakeBox(gp_Pnt(*aux[16:19]),gp_Pnt(*aux[19:22])).Shape()
b=BRepPrimAPI_MakeBox(gp_Pnt(*aux[22:25]),gp_Pnt(*aux[25:28])).Shape()

def boolean_case(fn,volume):
    shape=fn(a,b);assert count(shape,TopAbs_SOLID)==1;check(properties(shape)[1],volume,1e-10);native.append(shape)
    if fn==boolean_common:
        empty=fn(a,placed(b,10,0));assert count(empty,TopAbs_SOLID)==0;check(properties(empty)[1],0,1e-12)
    return {'valid':True,'closed_solids':1,'volume_mm3':volume,'empty_intersection_checked':fn==boolean_common}
for name,fn,v in [('boolean_union',boolean_union,9),('boolean_cut',boolean_cut,7),('boolean_common',boolean_common,1)]:run(name,'CPU native CAD Boolean',lambda fn=fn,v=v:boolean_case(fn,v))

box=BRepPrimAPI_MakeBox(gp_Pnt(-1,-1,-1),gp_Pnt(1,1,1)).Shape();e=TopExp_Explorer(box,TopAbs_FACE);faces=[]
while e.More():faces.append(TopoDS.Face_s(e.Current()));e.Next()
def center_z(face):
    p=GProp_GProps();BRepGProp.SurfaceProperties_s(face,p);return p.CentreOfMass().Z()
top=max(faces,key=center_z);shell=TopoDS.Shell_s(TopExp_Explorer(box,TopAbs_SHELL).Current());open_shell=None
def delete_case():
    global open_shell
    m=meshes[0];assert m['triangles']==10 and m['boundary_edges']==4 and m['euler']==1
    open_shell=delete_cad_faces(shell,[top]);assert count(open_shell,TopAbs_FACE)==5;check(properties(open_shell)[0],20.,1e-12);native.append(open_shell)
    return {'gpu_triangles':10,'gpu_boundary_edges':4,'native_open_faces':5,'native_area_mm2':20.}
def fill_case():
    m=meshes[1];assert m['vertices']==8 and m['triangles']==12 and m['boundary_edges']==0;check(mesh_volume(m),8.,1e-12)
    wire=TopoDS.Wire_s(TopExp_Explorer(top,TopAbs_WIRE).Current());cap=cap_planar_wire(wire)
    remain=[f for f in faces if not f.IsSame(top)];closed=sew_surfaces([*remain,cap],solid=True)
    check(properties(closed)[1],8.,1e-12);assert count(closed,TopAbs_SOLID)==1;native.append(closed)
    return {'gpu_vertices_after_weld':8,'gpu_closed_volume_mm3':8.,'native_cap_sew_volume_mm3':8.}
run('delete_faces','CUDA triangle mask + native CAD shell-face deletion',delete_case)
run('fill_hole','CUDA oriented boundary filling + native planar cap/sew',fill_case)

def revolve_case():
    profile=aux[32:44].reshape(4,3);check(profile,[[.6,0,-.5],[1.2,0,-.5],[1.2,0,.5],[.6,0,.5]],1e-12)
    shape=revolve_face(polygon_face(profile));expected=math.pi*(1.2**2-.6**2);check(properties(shape)[1],expected,1e-10)
    p=np.array(meshes[2]['positions']).reshape(48,4,3)
    check(p[:,:,2],np.tile(profile[:,2],(48,1)),1e-12);check(np.linalg.norm(p[:,:,:2],axis=2),np.tile(profile[:,0],(48,1)),1e-12)
    assert abs(mesh_volume(meshes[2])-expected)/expected<.003;assert meshes[2]['euler']==0;native.append(shape)
    return {'profile_vertices':4,'angular_stations':48,'native_volume_mm3':expected,'native_closed_solids':1,'mesh_relative_volume_error':abs(mesh_volume(meshes[2])-expected)/expected}
run('revolve_profile','CUDA profile revolution + exact native CAD revolve',revolve_case)

def sphere_seed():
    p=[[0.,0.,1.]]
    for i in range(1,8):
        phi=math.pi*i/8
        for j in range(16):
            theta=2*math.pi*j/16;q=np.array([math.sin(phi)*math.cos(theta),math.sin(phi)*math.sin(theta),math.cos(phi)])
            p.append(q*(1+.12*math.sin(3*theta)*math.sin(2*phi)))
    p.append([0.,0.,-1.]);return np.array(p)
seed=sphere_seed()
def smooth_case(i):
    m=meshes[i];p=np.array(m['positions']).reshape(-1,3);ids=np.array(m['indices'],dtype=int).reshape(-1,3);n=len(p);adj=np.zeros((n,n))
    for tri in ids:
        for j in range(3):a,b=tri[j],tri[(j+1)%3];adj[a,b]=adj[b,a]=1
    adj/=adj.sum(axis=1)[:,None];q=seed.copy()
    for _ in range(6):
        q+=.3*(adj@q-q)
        if i==4:q+=-.31*(adj@q-q)
    err=check(p,q,2e-12);before=float(np.mean(np.linalg.norm(seed,axis=1)));after=float(np.mean(np.linalg.norm(p,axis=1)))
    assert np.linalg.norm(adj@p-p)<np.linalg.norm(adj@seed-seed)
    if i==4:
        lap=np.array(meshes[3]['positions']).reshape(-1,3)
        assert abs(after-before)<abs(np.mean(np.linalg.norm(lap,axis=1))-before)
    return {'iterations':6,'max_matrix_reference_error_mm':err,'mean_radius_before_mm':before,'mean_radius_after_mm':after,'topology_preserved':True,'volume_mm3':mesh_volume(m)}
run('smooth_laplacian','CUDA uniform adjacency smoothing + dense matrix reference',lambda:smooth_case(3))
run('smooth_taubin','CUDA two-pass smoothing + dense matrix reference',lambda:smooth_case(4))

steps=[]
try:
    assert len(native)==6,'Missing native topology fixtures'
    steps.append(roundtrip('boolean_features.step',compound([placed(s,4*(i%3),4*(i//3)) for i,s in enumerate(native)]),5))
    steps.append(roundtrip('topology_meshes.step',compound([placed(mesh_shape(m),4*(i%3),4*(i//3)) for i,m in enumerate(meshes)]),4))
except Exception as e:steps.append({'valid_after':False,'error':str(e),'traceback':traceback.format_exc()})
finish_stage('topology',defs,results,steps,2)
