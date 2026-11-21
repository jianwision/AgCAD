import json,math,traceback
import numpy as np
from acceptance import ROOT,OUT,check,xyz,properties,compound,placed,mesh_shape,roundtrip,finish_stage
from cad_ops import trim_surface
from cad_advanced import bezier_surface
from cad_solids import polygon_face
from mesh_advanced import Mesh
from placement_ops import snap_point,merge_mesh,place_frame
from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.TColgp import TColgp_Array1OfPnt

defs=json.loads((ROOT/'primitive_motion.json').read_text())['operators'];data=np.array(json.loads((OUT/'motion.json').read_text()));results=[];deliver=[]
def run(name,backend,fn):
    try:
        evidence=fn() or {};results.append({'name':name,**defs[name],'backend':backend,'pass':True,'evidence':evidence});print('PASS',name,flush=True)
    except Exception as e:results.append({'name':name,**defs[name],'backend':backend,'pass':False,'error':str(e),'traceback':traceback.format_exc()});print('FAIL',name,str(e),flush=True)
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected invalid request rejection')
def snap_case():
    check(data[992:996],np.ones(4),1e-12);check(data[1000],0,1e-12)
    grid,e=snap_point([.24,-.26,1.12],step=.1);check(data[:3],grid,1e-12);check(data[3:6],[1,0,0],1e-12);assert data[6]==0
    edge=BRepBuilderAPI_MakeEdge(gp_Pnt(0,0,0),gp_Pnt(1,0,0)).Edge();p,e=snap_point([.4,.2,0],edge);check(p,[.4,0,0],1e-12)
    face=trim_surface(bezier_surface(np.array([[[u,v,0] for v in [0,1]] for u in [0,1]])),[(.2,.2),(.8,.2),(.8,.8),(.2,.8)])
    p,e=snap_point([.95,.5,.3],face);check(p,[.8,.5,0],1e-12);reject(lambda:snap_point([10,0,0],edge,max_distance=.1))
    return {'grid_gpu_cpu_match':True,'nearest_vertex_checked':True,'native_edge_and_trimmed_face_snap_checked':True,'capture_radius_rejection_checked':True,'invalid_gpu_requests_rejected':4}
def merge_case():
    a=Mesh([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],[[0,1,2],[0,2,3]]);b=Mesh([[1+1e-12,0,0],[2,0,0],[2,1,0],[1+1e-12,1,0]],[[0,1,2],[0,2,3]])
    separate,_=merge_mesh([a,b]);out,e=merge_mesh([a,b],1e-9);r=out.report();assert len(separate.p)==8 and len(out.p)==6 and r['boundary_edges']==6 and r['euler']==1 and r['nonmanifold_edges']==r['orientation_errors']==0
    check(r['area'],2.,1e-12);deliver.append(mesh_shape(out.record()));return {**r,**e}
def path_curve_case():
    p=data[16:247].reshape(33,7);speed=math.sqrt(1.2**2+(.4/(2*math.pi))**2);distance=np.linspace(0,2,33);t=.4+distance/speed
    ref=np.column_stack([1.2*np.cos(t),1.2*np.sin(t),.4*t/(2*math.pi)]);tangent=np.column_stack([-1.2*np.sin(t),1.2*np.cos(t),np.full(33,.4/(2*math.pi))])/speed
    err=check(p[:,:3],ref,1e-10);check(p[:,3:6],tangent,1e-10);check(p[:,6],t,1e-10)
    marker=BRepPrimAPI_MakeBox(.06,.04,.03).Shape();deliver.extend(place_frame(marker,p[k,:3],p[k,3:6],[0,0,1]) for k in [0,16,32]);return {'samples':33,'distance_mm':2.,'max_analytic_arc_length_error_mm':err,'CAD_instances_attached':3}
def path_mesh_case():
    p=data[448:712].reshape(33,8);d=np.linspace(0,1.5,33);x=np.where(d<=.8,.2+d,1+(d-.8)/math.sqrt(2));z=np.where(d<=.8,0,(d-.8)/math.sqrt(2));ref=np.column_stack([x,np.full(33,.3),z]);err=check(p[:,:3],ref,1e-11)
    expected_t=np.array([[1,0,0] if s<=.8 else [1/math.sqrt(2),0,1/math.sqrt(2)] for s in d]);check(p[:,3:6],expected_t,1e-11);check(p[:,7],np.ones(33),1e-12);check(data[720:723],[2,.3,1],1e-12);assert data[723]==2
    deliver.append(place_frame(BRepPrimAPI_MakeBox(.06,.04,.03).Shape(),p[-1,:3],p[-1,3:6],[-1,0,1]));return {'samples':33,'distance_mm':1.5,'max_unfolded_reference_error_mm':err,'crossed_face_ids':sorted(set(map(int,p[:,6]))),'boundary_clamp_checked':True}
def path_surface_case():
    p=data[736:889].reshape(17,9);u=.1+np.linspace(0,.6,17);v=.2+np.linspace(0,.3,17)+1e-10;ref=np.column_stack([u,v,.2*u*v]);err=check(p[:,:3],ref,1e-12)
    normal=np.column_stack([-.2*v,-.2*u,np.ones(17)]);normal/=np.linalg.norm(normal,axis=1)[:,None];check(p[:,6:9],normal,1e-12);check(np.einsum('ij,ij->i',p[:,3:6],p[:,6:9]),np.zeros(17),1e-12)
    deliver.append(place_frame(BRepPrimAPI_MakeBox(.06,.04,.03).Shape(),p[-1,:3],p[-1,3:6],p[-1,6:9]));return {'samples':17,'uv_displacement':[.6,.3],'max_analytic_attached_point_error_mm':err,'unit_frame_error':check(np.linalg.norm(p[:,3:6],axis=1),np.ones(17),1e-12),'world_geodesic':False}
for name,backend,fn in [('snap_point','CUDA grid/vertex + CPU native CAD',snap_case),('merge_mesh','CPU indexed merge/weld',merge_case),('move_along_curve','CUDA arc-length positioning + CPU instance transform',path_curve_case),('move_on_mesh','CUDA intrinsic triangle walk + CPU instance transform',path_mesh_case),('move_on_surface','CUDA UV placement/frame + CPU instance transform',path_surface_case)]:run(name,backend,fn)
steps=[]
try:
    assert len(deliver)==6;steps.append(roundtrip('placement_operations.step',compound([placed(s,3*(i%3),3*(i//3)) for i,s in enumerate(deliver)]),5))
except Exception as e:steps.append({'valid_after':False,'error':str(e),'traceback':traceback.format_exc()})
finish_stage('motion',defs,results,steps,1)
