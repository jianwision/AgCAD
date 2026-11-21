import json,math,traceback
import numpy as np
from acceptance import ROOT,OUT,xyz,check,count,properties,compound,placed,build_surface,mesh_shape,mesh_volume,roundtrip,finish_stage
from cad_ops import valid,sew_surfaces
from cad_solids import polygon_face
from OCP.gp import gp_Pnt,gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TopoDS import TopoDS
from OCP.TopAbs import TopAbs_SOLID

defs=json.loads((ROOT/'primitive_operations.json').read_text())['operators']
raw=json.loads((OUT/'operations.json').read_text());meshes=raw['meshes'];aux=np.array(raw['aux']);results=[]
surfaces=[build_surface(s) for s in raw['surfaces']];samples=np.array(raw['surface_samples']).reshape(2,9,9,3)
native=[]

def run(name,backend,fn):
    try:
        evidence=fn() or {};results.append({'name':name,**defs[name],'backend':backend,'pass':True,'evidence':evidence});print('PASS',name,flush=True)
    except Exception as e:
        results.append({'name':name,**defs[name],'backend':backend,'pass':False,'error':str(e),'traceback':traceback.format_exc()});print('FAIL',name,str(e),flush=True)
def points(i):return np.array(meshes[i]['positions']).reshape(-1,3)
def deformation(i):
    p=points(i);base=[]
    corners=[[-.4,-.3],[.4,-.3],[.4,.3],[-.4,.3]]
    for t in np.linspace(-1.5,1.5,64):
        for a,b in corners:base.append([t,a,b] if i==0 else [a,b,t])
    base.extend([[-1.5,0,0],[1.5,0,0]] if i==0 else [[0,0,-1.5],[0,0,1.5]])
    base=np.array(base);reference=base.copy()
    if i==0:
        angle=base[:,0]/2.6;reference[:,0]=(2.6+base[:,2])*np.sin(angle);reference[:,2]=(2.6+base[:,2])*np.cos(angle)-2.6
        check(np.hypot(p[:,0],p[:,2]+2.6)-2.6,base[:,2],1e-12)
    if i==1:
        z=(base[:,0]+1j*base[:,1])*np.exp(1j*.9*base[:,2]);reference[:,0]=z.real;reference[:,1]=z.imag
        check(np.linalg.norm(p[:,:2],axis=1),np.linalg.norm(base[:,:2],axis=1),1e-12)
    if i==2:reference[:,:2]*=(1+.18*base[:,2,None])
    err=check(p,reference,1e-12);v=mesh_volume(meshes[i]);theory=1.44 if i<2 else .48*(3+.18**2*27/12)
    assert abs(v-theory)/theory<(.02 if i<2 else 1e-12)
    return {'vertices':len(p),'triangles':meshes[i]['triangles'],'max_mapping_error_mm':err,'mesh_volume_mm3':v,'analytic_deformation_volume_mm3':theory,'representation':'faceted B-rep at sampled stations'}
for i,name in enumerate(['bend','twist','taper']):run(name,'CUDA point deformation + independent invariants',lambda i=i:deformation(i))

def surface_samples(i):
    ref=[[xyz(surfaces[i].Value(float(u),float(v))) for v in np.linspace(0,1,9)] for u in np.linspace(0,1,9)]
    return check(samples[i],ref,1e-12)
def sweep_case():
    cp=np.array(raw['surfaces'][0]['poles_xyzw']).reshape(17,9,4);angle=1.4*math.pi;normals=[]
    us=np.linspace(0,1,17);path=np.column_stack([1.2*np.cos(angle*us),1.2*np.sin(angle*us),1.5*us])
    tangents=np.column_stack([-1.2*angle*np.sin(angle*us),1.2*angle*np.cos(angle*us),np.full(17,1.5)]);tangents/=np.linalg.norm(tangents,axis=1)[:,None]
    n=np.array([0.,0.,1.]);n-=tangents[0]*np.dot(n,tangents[0]);n/=np.linalg.norm(n)
    for i,t in enumerate(tangents):
        if i:
            axis=np.cross(tangents[i-1],t);s=np.linalg.norm(axis);axis/=s;c=np.dot(tangents[i-1],t)
            n=c*n+s*np.cross(axis,n)+(1-c)*axis*np.dot(axis,n)
        n-=t*np.dot(n,t);n/=np.linalg.norm(n);normals.append(n.copy())
        b=np.cross(t,n)
        for j in range(9):check(cp[i,j,:3],path[i]+.22*(math.cos(2*math.pi*j/8)*n+math.sin(2*math.pi*j/8)*b),1e-12)
    check(np.linalg.norm(cp[:,:8,:3]-path[:,None,:],axis=2),np.full((17,8),.22),1e-12)
    side=TopoDS.Face_s(BRepBuilderAPI_MakeFace(surfaces[0],0.,1.,0.,1.,1e-7).Face().Reversed())
    caps=[polygon_face(cp[0,:8,:3][::-1]),polygon_face(cp[-1,:8,:3])]
    body=sew_surfaces([side,*caps],solid=True);assert count(body,TopAbs_SOLID)==1;native.append(body)
    return {'stations':17,'profile_corners':8,'max_gpu_cad_error_mm':surface_samples(0),'native_closed_solids':1,'cad_volume_mm3':properties(body)[1]}
run('sweep_path','CUDA discrete parallel transport and control net + CPU caps/sew',sweep_case)

def loft_case():
    sections=np.array([[[cx,j/3-.5,z+.18*math.sin(j*math.pi/3)*(1+.2*i)] for j in range(4)] for i,(cx,z) in enumerate(zip([0,.3,1.4,2],[0,.4,-.1,.5]))])
    centers=sections.mean(axis=1);t=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]);t/=t[-1];h=np.diff(t)
    matrix=np.zeros((4,4));matrix[0,0]=matrix[-1,-1]=1;rhs=np.zeros((4,4,3))
    for i in [1,2]:matrix[i,i-1:i+2]=[h[i-1],2*(h[i-1]+h[i]),h[i]];rhs[i]=6*((sections[i+1]-sections[i])/h[i]-(sections[i]-sections[i-1])/h[i-1])
    m=np.linalg.solve(matrix,rhs.reshape(4,-1)).reshape(4,4,3)
    def ref(u,v):
        i=min(np.searchsorted(t,u,side='right')-1,2);q=(u-t[i])/h[i]
        poles=(1-q)*sections[i]+q*sections[i+1]+h[i]**2/6*(((1-q)**3-(1-q))*m[i]+(q**3-q)*m[i+1])
        return (1-v)**3*poles[0]+3*v*(1-v)**2*poles[1]+3*v*v*(1-v)*poles[2]+v**3*poles[3]
    err=0
    for u in np.unique(np.r_[np.linspace(0,1,33),t]):
        for v in np.linspace(0,1,9):err=max(err,check(xyz(surfaces[1].Value(float(u),float(v))),ref(u,v),2e-12))
    for u in t[1:-1]:
        p=gp_Pnt();du,dv,duu,dvv,duv=[gp_Vec() for _ in range(5)]
        surfaces[1].D2(float(u-1e-8),.43,p,du,dv,duu,dvv,duv);left=np.r_[xyz(du),xyz(duu)]
        surfaces[1].D2(float(u+1e-8),.43,p,du,dv,duu,dvv,duv);check(left,np.r_[xyz(du),xyz(duu)],3e-5)
    assert np.max(h)-np.min(h)>.05
    native.append(valid(BRepBuilderAPI_MakeFace(surfaces[1],0.,1.,0.,1.,1e-7).Face()))
    return {'sections':4,'centroid_chord_parameters':t.tolist(),'C2_internal_seams':2,'max_reference_error_mm':err,'max_gpu_cad_error_mm':surface_samples(1)}
run('loft_chordal','CUDA nonuniform natural cubic solve + independent dense solve',loft_case)

def cc_case():
    p=points(4);expected=[]
    for i in range(8):expected.append([(5/9 if i&1 else -5/9),(5/9 if i&2 else -5/9),(5/9 if i&4 else -5/9)])
    for zero in range(3):
        for a in [-.75,.75]:
            for b in [-.75,.75]:q=[a,b];q.insert(zero,0.);expected.append(q)
    for axis in range(3):
        for sign in [-1,1]:q=[0.,0.,0.];q[axis]=sign;expected.append(q)
    assert sorted(tuple(np.round(q,12)) for q in p)==sorted(tuple(np.round(q,12)) for q in expected)
    assert meshes[4]['vertices']==26 and meshes[4]['triangles']==48
    return {'vertices':26,'triangles':48,'matches_original_Catmull_Clark_rules':True}
def loop_case():
    original=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    expected=[p*.25 for p in original]+[(original[i]+original[j])*.25 for i in range(4) for j in range(i+1,4)]
    assert sorted(map(tuple,np.round(points(5),12)))==sorted(map(tuple,np.round(expected,12)));assert meshes[5]['triangles']==16
    return {'vertices':10,'triangles':16,'matches_original_Loop_rules':True}
def grid_case():
    ref=[[u,v,.2*u*v] for u in np.linspace(0,1,17) for v in np.linspace(0,1,13)]
    return {'vertices':221,'triangles':384,'boundary_edges':56,'max_bilinear_error_mm':check(points(6),ref,1e-12)}
def polyline_case():
    p=np.array([[0,0,0],[1,0,0],[1,2,0],[1,2,1],[2,2,1]])
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];target=np.linspace(0,lengths[-1],65)
    reference=np.column_stack([np.interp(target,lengths,p[:,i]) for i in range(3)])
    check(aux[:7],np.ones(7),1e-12)
    return {'output_points':65,'total_polyline_length_mm':5.,'max_reference_error_mm':check(aux[16:211].reshape(65,3),reference,1e-12),'invalid_inputs_rejected':7}
run('catmull_clark','CUDA quad smoothing subdivision',cc_case)
run('loop_subdivide','CUDA triangle smoothing subdivision',loop_case)
run('resample_grid','CUDA bilinear grid resampling',grid_case)
run('resample_polyline','CUDA accumulated chord-distance resampling',polyline_case)

steps=[]
try:
    assert len(native)==2,'Missing native operation surfaces'
    steps.append(roundtrip('operation_surfaces.step',compound([placed(native[0],0,0),placed(native[1],4,0)]),1))
    shapes=[placed(mesh_shape(m),4*(i%4),4*(i//4)) for i,m in enumerate(meshes)]
    steps.append(roundtrip('operation_meshes.step',compound(shapes),6))
except Exception as e:steps.append({'valid_after':False,'error':str(e),'traceback':traceback.format_exc()})
finish_stage('operations',defs,results,steps,2)
