"""Batch acceptance for every item in SURFACE_BASELINE.md, with independent
Open CASCADE evaluators, analytic references, native topology and STEP reimport.
Run `run.ps1 suite` to regenerate the CUDA outputs first.
"""
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np
from cad_ops import ROOT, valid, project_curve, intersect_surfaces, trim_surface, split_surface, join_curves, sew_surfaces, fill_nside, sweep, loft
from OCP.gp import gp_Pnt,gp_Vec,gp_Dir,gp_Ax1,gp_Ax2,gp_Ax3,gp_Trsf,gp_Pln,gp_Circ
from OCP.Geom import Geom_BSplineSurface,Geom_BezierCurve,Geom_BSplineCurve,Geom_Circle,Geom_Ellipse,Geom_TrimmedCurve,Geom_OffsetSurface,Geom_Plane,Geom_CylindricalSurface
from OCP.TColgp import TColgp_Array1OfPnt,TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger,TColStd_Array2OfReal
from OCP.BRep import BRep_Tool,BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakePolygon,BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_SOLID,TopAbs_WIRE
from OCP.TopoDS import TopoDS,TopoDS_Compound
from OCP.STEPControl import STEPControl_Writer,STEPControl_Reader,STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from export_step import knots

OUT=ROOT/'output/primitives'
geometry=json.loads((OUT/'geometry.json').read_text())
meshes=json.loads((OUT/'meshes.json').read_text())
gpu=json.loads((OUT/'gpu_report.json').read_text())
definitions=json.loads((ROOT/'primitive_definitions.json').read_text(encoding='utf-8'))['operators']
curve_samples=np.array(geometry['curve_samples']).reshape(8,33,3)
surface_samples=np.array(geometry['surface_samples']).reshape(14,9,9,3)
metrics=np.array(geometry['metrics'])
results=[];surfaces=[];surface_faces=[];extra_faces=[];extra_curves=[]
P=np.array([[0,0,0],[.3,.7,.1],[.7,-.2,.4],[1,1,.2]],dtype=float)


def array_points(points):
    a=TColgp_Array1OfPnt(1,len(points))
    for i,p in enumerate(points,1):a.SetValue(i,gp_Pnt(*p))
    return a


def bezier(points):return Geom_BezierCurve(array_points(points))


def xyz(point):return np.array([point.X(),point.Y(),point.Z()])


def scalar_check(a,b,tol=1e-8):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert math.isfinite(error) and error<=tol,(error,tol)
    return error


def run(name,backend,fn,limits=''):
    definition=definitions[name]
    try:
        details=fn() or {};results.append({'name':name,**definition,'backend':backend,'pass':True,'test_scope':limits,'evidence':details})
        print('PASS',name,flush=True)
    except Exception as exc:
        results.append({'name':name,**definition,'backend':backend,'pass':False,'test_scope':limits,'error':str(exc),'traceback':traceback.format_exc()})
        print('FAIL',name,str(exc),flush=True)


def count(shape,kind):
    ex=TopExp_Explorer(shape,kind);n=0
    while ex.More():n+=1;ex.Next()
    return n


def properties(shape):
    a=GProp_GProps();v=GProp_GProps();BRepGProp.SurfaceProperties_s(shape,a);BRepGProp.VolumeProperties_s(shape,v)
    return a.Mass(),v.Mass()


def build_surface(record):
    if record['kind']=='offset':return Geom_OffsetSurface(surfaces[int(record['base'])],record['distance'])
    nu,nv=record['size'];pu,pv=record['degree'];cp=np.array(record['poles_xyzw']).reshape(nu,nv,4)
    points=TColgp_Array2OfPnt(1,nu,1,nv);weights=TColStd_Array2OfReal(1,nu,1,nv)
    for i in range(nu):
        for j in range(nv):points.SetValue(i+1,j+1,gp_Pnt(*cp[i,j,:3]));weights.SetValue(i+1,j+1,float(cp[i,j,3]))
    uk,um=knots(record['u_knots']);vk,vm=knots(record['v_knots'])
    return Geom_BSplineSurface(points,weights,uk,vk,um,vm,pu,pv,False,False)


for record in geometry['surfaces']:
    s=build_surface(record);surfaces.append(s)
    face=BRepBuilderAPI_MakeFace(s,0.,1.,0.,1.,1e-7).Face();valid(face);surface_faces.append(face)


def curve_reference(i):
    if i==0:return bezier([P[0],P[3]])
    if i==1:
        uk,um=knots([0,0,1/3,2/3,1,1]);return Geom_BSplineCurve(array_points(P),uk,um,1,False)
    if i==2:return Geom_Circle(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(0,0,1)),1)
    if i==3:return Geom_Ellipse(gp_Ax2(gp_Pnt(0,0,.2),gp_Dir(0,0,1)),2,.5)
    if i==4:return bezier(P)
    if i==5:
        w=TColStd_Array1OfReal(1,3)
        for k,q in enumerate([1,math.sqrt(.5),1],1):w.SetValue(k,q)
        uk,um=knots([0,0,0,1,1,1]);return Geom_BSplineCurve(array_points([[1,0,0],[1,1,0],[0,1,0]]),w,uk,um,2,False)
    if i==6:return bezier([P[1],P[1]+(P[2]-P[0])/6,P[2]-(P[3]-P[1])/6,P[2]])
    return Geom_TrimmedCurve(bezier(P),.2,.8)


def curve_check(i):
    curve=curve_reference(i);reference=[]
    for t in np.linspace(0,1,33):
        parameter=2*math.pi*t if i in [2,3] else .2+.6*t if i==7 else t
        reference.append(xyz(curve.Value(parameter)))
    error=scalar_check(curve_samples[i],reference,1e-12)
    if i==5:scalar_check(np.sum(curve_samples[i,:,:2]**2,axis=1),np.ones(33),1e-12)
    extra_curves.append(curve)
    return {'samples':33,'max_cad_error_mm':error}


for i,name in enumerate(['line','polyline','circle_arc','ellipse','bezier_curve','nurbs_curve','catmull_rom','curve_trim']):
    run(name,'CUDA + independent CAD evaluator',lambda i=i:curve_check(i),'degree<=5, poles<=32 for NURBS; polyline parameter is by segment index')


def surface_check(i):
    ref=np.array([[xyz(surfaces[i].Value(u,v)) for v in np.linspace(0,1,9)] for u in np.linspace(0,1,9)])
    err=scalar_check(surface_samples[i],ref,1e-11)
    return {'samples':81,'max_cad_error_mm':err,'brep_valid':True}


def surface_op(i):
    evidence=surface_check(i);samples=surface_samples[i]
    if i==0:
        for a,u in enumerate(np.linspace(0,1,9)):
            for b,v in enumerate(np.linspace(0,1,9)):scalar_check(samples[a,b],[u,v,0],1e-12)
    if i==1:
        poles=np.array(geometry['surfaces'][i]['poles_xyzw']).reshape(4,4,4)
        for a in range(4):
            for b in range(4):scalar_check(poles[a,b],[a/3,b/3,.4*math.sin(a*math.pi/3)*math.sin(b*math.pi/3),1],1e-12)
    if i==2:
        scalar_check(samples[:,:,1]**2+samples[:,:,2]**2,np.ones((9,9)),1e-12)
        scalar_check(metrics[315:320],np.ones(5),1e-12)
        evidence['invalid_builder_inputs_rejected']=5
    if i==3:
        c0=bezier([[0,j/3,.1*(j in [1,2])] for j in range(4)])
        c1=bezier([[1,j/3,.2+.1*j] for j in range(4)])
        for a,u in enumerate(np.linspace(0,1,9)):
            for b,v in enumerate(np.linspace(0,1,9)):scalar_check(samples[a,b],(1-u)*xyz(c0.Value(v))+u*xyz(c1.Value(v)),1e-12)
    if i==4:
        c=bezier([[0,0,0],[.3,.2,.5],[.7,-.2,.5],[1,0,0]])
        for a,u in enumerate(np.linspace(0,1,9)):
            for b,v in enumerate(np.linspace(0,1,9)):scalar_check(samples[a,b],xyz(c.Value(u))+v*np.array([0,1,.2]),1e-12)
    if i==5:
        rho=np.sqrt(samples[:,:,0]**2+samples[:,:,2]**2)
        scalar_check((rho-1)**2+samples[:,:,1]**2,np.full((9,9),.25**2),1e-12)
        scalar_check(samples[0],samples[-1],1e-12);scalar_check(samples[:,0],samples[:,-1],1e-12)
    if i==6:
        spine=bezier([[0,0,0],[.3,0,.2],[.7,0,.4],[1,0,.1]])
        profile=bezier([[0,0,0],[0,.3,.2],[0,.7,.2],[0,1,0]])
        for a,u in enumerate(np.linspace(0,1,9)):
            for b,v in enumerate(np.linspace(0,1,9)):scalar_check(samples[a,b],xyz(spine.Value(u))+xyz(profile.Value(v)),1e-12)
        path=BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Pnt(0,0,0),gp_Pnt(0,0,2)).Edge()).Wire()
        ring=Geom_Circle(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(0,0,1)),.2)
        pipe=sweep(path,BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(ring).Edge()).Wire())
        scalar_check(properties(pipe)[0],.8*math.pi,1e-7);extra_faces.append(pipe);evidence['cad_path_sweep_valid']=True
    if i==7:
        wires=[]
        for n,u in enumerate([0,1/3,2/3,1]):
            c=bezier([[n/3,j/3,.1*n*n+.15*math.sin(j*math.pi/3)*(1+.2*n)] for j in range(4)])
            for v in np.linspace(0,1,17):scalar_check(xyz(surfaces[i].Value(u,v)),xyz(c.Value(v)),1e-12)
            wires.append(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(c).Edge()).Wire())
        skin=loft(wires);extra_faces.append(skin);evidence['cad_general_loft_valid']=True
        for u in [1/3,2/3]:
            p=gp_Pnt();a,b,aa,bb,ab=[gp_Vec() for _ in range(5)]
            surfaces[i].D2(u-1e-8,.43,p,a,b,aa,bb,ab);left=np.array([xyz(a),xyz(aa)])
            surfaces[i].D2(u+1e-8,.43,p,a,b,aa,bb,ab);scalar_check(left,[xyz(a),xyz(aa)],2e-6)
        evidence['interpolated_sections']=4
    if i==8:
        c0=bezier([[0,0,0],[1/3,0,.2],[2/3,0,.2],[1,0,0]])
        c1=bezier([[0,1,0],[1/3,1,.4],[2/3,1,.4],[1,1,0]])
        d0=bezier([[0,0,0],[0,1/3,.1],[0,2/3,.1],[0,1,0]])
        d1=bezier([[1,0,0],[1,1/3,.3],[1,2/3,.3],[1,1,0]])
        for a,u in enumerate(np.linspace(0,1,9)):
            for b,v in enumerate(np.linspace(0,1,9)):
                q=(1-v)*xyz(c0.Value(u))+v*xyz(c1.Value(u))+(1-u)*xyz(d0.Value(v))+u*xyz(d1.Value(v))-np.array([u,v,0])
                scalar_check(samples[a,b],q,1e-12)
    return evidence


for i,name in enumerate(['plane','bezier_patch','nurbs_patch','ruled_surface','extrude_surface','revolve_surface','sweep_surface','loft_surface','coons_fill','offset_surface']):
    run(name,'CUDA control net/evaluation + CAD topology',lambda i=i:surface_op(i),'sweep: exact translation on GPU, native path sweep on CPU; loft: compatible sections on GPU; offset is an offset definition, not a fitted net')


def filling_case():
    pts=[gp_Pnt(math.cos(2*math.pi*i/5),math.sin(2*math.pi*i/5),.08*math.sin(4*math.pi*i/5)) for i in range(5)]
    edges=[BRepBuilderAPI_MakeEdge(pts[i],pts[(i+1)%5]).Edge() for i in range(5)]
    face=fill_nside(edges);surf=BRep_Tool.Surface_s(TopoDS.Face_s(face));error=0.
    for i in range(5):
        a,b=xyz(pts[i]),xyz(pts[(i+1)%5])
        for t in np.linspace(0,1,9):
            proj=GeomAPI_ProjectPointOnSurf(gp_Pnt(*((1-t)*a+t*b)),surf)
            assert proj.NbPoints()>0;error=max(error,proj.LowerDistance())
    assert error<1e-5,error
    extra_faces.append(face)
    return {'boundary_edges':5,'edge_samples':45,'max_surface_distance_mm':error,'valid':True}


run('fill_nside','CAD kernel',filling_case,'native boundary constraints C0/G1/G2; five-edge C0 composition is the batch fixture')


def blend_case(kind):
    m=metrics[:297].reshape(33,3,3)
    cols=[0] if kind=='g1' else [1,2]
    for k in cols:
        assert np.max(m[:,k,0])<1e-11
        assert np.max(m[:,k,1])<1e-5
        if kind=='g2':assert np.max(m[:,k,2])<1e-10
    if kind=='g1':assert np.min(m[:,0,2])>.01
    p=gp_Pnt();a,b,aa,bb,ab=[gp_Vec() for _ in range(5)]
    surfaces[10 if kind=='g1' else 11].D2(0.,.43,p,a,b,aa,bb,ab)
    scalar_check(xyz(a),[1,0,.7],1e-12)
    if kind=='g2':scalar_check(xyz(aa),[0,0,.3],1e-12)
    return {'samples_per_seam':33,'max_gap_mm':float(np.max(m[:,cols,0])),'max_angle_deg':float(np.max(m[:,cols,1])),'max_curvature_jump':float(np.max(m[:,cols,2]))}


run('blend_g1','CUDA Hermite control net',lambda:blend_case('g1'),'cubic strip with prescribed rail positions and cross derivatives')
run('blend_g2','CUDA quintic Hermite control net',lambda:blend_case('g2'),'quintic strip with prescribed rail positions, first/second cross derivatives')


def continuity_case():
    blend_case('g1');blend_case('g2');scalar_check(metrics[312],.1,1e-12)
    return {'distinguishes_G1_from_G2':True,'detects_G0_gap_mm':float(metrics[312]),'directions_for_curvature_comparison':3}


def curvature_case():
    props=GeomLProp_SLProps(surfaces[2],.4,.5,2,1e-9)
    assert props.IsCurvatureDefined()
    scalar_check(metrics[309:311],[props.GaussianCurvature(),props.MeanCurvature()],1e-12)
    scalar_check(metrics[311],4*(math.sqrt(2)-1),1e-12)
    return {'gaussian':float(metrics[309]),'mean':float(metrics[310]),'jacobian':float(metrics[311])}


run('continuity_check','CUDA differential geometry',continuity_case,'sampled geometry, not a global proof of Class-A quality')
run('curvature','CUDA differential geometry + CAD reference',curvature_case,'regular surface points; singular points require explicit handling')


def projection_case():
    plane=Geom_Plane(gp_Pln(gp_Pnt(0,0,0),gp_Dir(0,0,1)))
    curve=bezier([[0,0,1],[.3,.2,1.3],[.7,.8,.8],[1,1,1]])
    projected=project_curve(curve,plane);err=0.
    for t in np.linspace(0,1,33):
        a,b=xyz(curve.Value(t)),xyz(projected.Value(t));err=max(err,abs(b[2]));scalar_check(a[:2],b[:2],1e-7)
    assert err<1e-7;extra_curves.append(projected)
    return {'samples':33,'max_plane_residual_mm':err}


def intersection_case():
    cylinder=Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0,0,0),gp_Dir(0,0,1)),1)
    plane=Geom_Plane(gp_Pln(gp_Pnt(0,0,.5),gp_Dir(0,0,1)))
    curves=intersect_surfaces(cylinder,plane);assert len(curves)==1
    residual=0.
    for c in curves:
        for t in np.linspace(c.FirstParameter(),c.LastParameter(),65):
            x,y,z=xyz(c.Value(t));residual=max(residual,abs(x*x+y*y-1),abs(z-.5))
    assert residual<1e-7;extra_curves.extend(curves)
    return {'intersection_curves':len(curves),'samples':65,'max_equation_residual':residual}


def trim_case():
    face=trim_surface(surfaces[0],[(.1,.1),(.9,.1),(.9,.9),(.1,.9)],[[(.4,.4),(.6,.4),(.6,.6),(.4,.6)]])
    area=properties(face)[0];scalar_check(area,.6,1e-10);assert count(face,TopAbs_WIRE)==2
    curved=trim_surface(surfaces[1],[(.1,.1),(.8,.1),(.8,.9),(.1,.9)])
    extra_faces.extend([face,curved]);return {'plane_area_mm2':area,'boundary_loops':2,'curved_trim_valid':True}


def split_case():
    cutter=Geom_Plane(gp_Pln(gp_Pnt(.5,0,0),gp_Dir(1,0,0)))
    tool=BRepBuilderAPI_MakeFace(cutter,-2.,2.,-2.,2.,1e-7).Face()
    shape=split_surface(surface_faces[0],[tool]);assert count(shape,TopAbs_FACE)==2
    scalar_check(properties(shape)[0],1.,1e-10);extra_faces.append(shape)
    return {'result_faces':2,'total_area_mm2':1.}


def join_case():
    joined=join_curves([bezier([[0,0,0],[1,0,0]]),bezier([[1,0,0],[2,0,0]])])
    scalar_check(xyz(joined.Value(joined.FirstParameter())),[0,0,0],1e-10)
    scalar_check(xyz(joined.Value(joined.LastParameter())),[2,0,0],1e-10)
    try:join_curves([bezier([[0,0,0],[1,0,0]]),bezier([[3,0,0],[4,0,0]])])
    except ValueError:pass
    else:raise AssertionError('Disconnected curves should be rejected')
    extra_curves.append(joined);return {'degree':joined.Degree(),'poles':joined.NbPoles(),'disconnected_input_rejected':True}


def sewing_case():
    box=BRepPrimAPI_MakeBox(1,1,1).Shape();ex=TopExp_Explorer(box,TopAbs_FACE);faces=[]
    while ex.More():faces.append(TopoDS.Face_s(ex.Current()));ex.Next()
    solid=sew_surfaces(faces,solid=True);assert count(solid,TopAbs_SOLID)==1
    scalar_check(properties(solid)[1],1.,1e-10)
    transition=sew_surfaces([surface_faces[12],surface_faces[11],surface_faces[13]])
    assert count(transition,TopAbs_FACE)==3
    extra_faces.extend([solid,transition]);return {'closed_solids':1,'volume_mm3':1.,'G2_composition_faces':3}


for name,fn in [('project_curve',projection_case),('intersect_surfaces',intersection_case),('trim_surface',trim_case),('split_surface',split_case),('join_curves',join_case),('sew',sewing_case)]:run(name,'CAD kernel',fn,'B-rep topology executes explicitly on CPU, after GPU batch readback')


def mesh_case(i):
    m=meshes[i];p=np.array(m['positions']).reshape(-1,3);ids=np.array(m['indices'],dtype=int).reshape(-1,3);n=np.array(m['normals']).reshape(-1,3)
    scalar_check(np.linalg.norm(n,axis=1),np.ones(len(n)),1e-12)
    a,b,c=p[ids[:,0]],p[ids[:,1]],p[ids[:,2]]
    area=float(np.linalg.norm(np.cross(b-a,c-a),axis=1).sum()/2)
    volume=float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6)
    expected={0:8.,2:24*math.sin(2*math.pi/24),3:8*math.sin(2*math.pi/24),7:6.,9:8.,10:8.,12:8-8*(.4**3/6)}
    if i in expected:scalar_check(volume,expected[i],1e-10)
    if i==1:assert abs(volume-4*math.pi/3)/(4*math.pi/3)<.05
    if i in [4,11]:
        expected_vol=2*math.pi**2*.3**2;assert abs(volume-expected_vol)/expected_vol<.07
        radial=np.hypot(p[:,0],p[:,1]);theory=p-np.column_stack((p[:,0]/radial,p[:,1]/radial,np.zeros(len(p))))
        theory/=np.linalg.norm(theory,axis=1)[:,None];assert np.min(np.sum(n*theory,axis=1))>.98
    if i==6:scalar_check(area,3.,1e-12)
    if i==8:scalar_check(area,2.5*math.sin(2*math.pi/5),1e-12)
    if i==8:scalar_check(p[5:10],p[:5]*.75,1e-12)
    if i==9:
        expected_points={(x,y,z) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1] if max(abs(x),abs(y),abs(z))==1}
        assert {tuple(q) for q in p}==expected_points
    if i in [0,9,10]:scalar_check(area,24.,1e-12)
    return {'vertices':m['vertices'],'triangles':m['triangles'],'boundary_edges':m['boundary_edges'],'euler':m['euler'],'area_mm2':area,'volume_mm3':volume,'finite_unit_normals':True}


for i,m in enumerate(meshes):run(m['name'],'CUDA indexed mesh',lambda i=i:mesh_case(i),'ear clipping: planar simple polygon <=64 corners; inset is fractional; bevel: convex tri-valent polyhedron; welding uses representative tolerance')


def transform_case(i):
    p=gp_Pnt(1,2,3);t=gp_Trsf();center=np.array([.1,.2,.3]);q=np.array([1,2,3])
    if i==0:expected=q+np.array([2,-1,.5])
    if i==1:
        t.SetRotation(gp_Ax1(gp_Pnt(*center),gp_Dir(1,2,-1)),.7);p.Transform(t);expected=xyz(p)
    if i==2:expected=center+(q-center)*np.array([2,.5,-1])
    if i==3:
        t.SetMirror(gp_Ax2(gp_Pnt(*center),gp_Dir(1,2,-1)));p.Transform(t);expected=xyz(p)
    err=scalar_check(metrics[297+3*i:300+3*i],expected,1e-12)
    return {'max_reference_error_mm':err}


for i,name in enumerate(['translate','rotate','scale','mirror']):run(name,'CUDA point transform',lambda i=i:transform_case(i),'rotation/mirror accept arbitrary axes/planes; zero normal is outside the valid input domain')


def compound(shapes):
    c=TopoDS_Compound();b=BRep_Builder();b.MakeCompound(c)
    for s in shapes:b.Add(c,s)
    return c


def placed(shape,x,y,z=0):
    t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,z));return BRepBuilderAPI_Transform(shape,t,True).Shape()


def step_roundtrip(name,shape):
    valid(shape);Interface_Static.SetCVal_s('write.step.schema','AP214IS');Interface_Static.SetCVal_s('write.step.unit','MM')
    writer=STEPControl_Writer();assert writer.Transfer(shape,STEPControl_AsIs)==IFSelect_RetDone
    path=OUT/name;assert writer.Write(str(path))==IFSelect_RetDone
    reader=STEPControl_Reader();assert reader.ReadFile(str(path))==IFSelect_RetDone;assert reader.TransferRoots()>0
    reopened=reader.OneShape();valid(reopened)
    a,v=properties(shape);aa,vv=properties(reopened)
    assert abs(a-aa)<max(1e-6,abs(a)*1e-6),(name,a,aa)
    assert abs(v-vv)<max(1e-6,abs(v)*1e-6),(name,v,vv)
    counts={}
    for label,kind in [('solids',TopAbs_SOLID),('faces',TopAbs_FACE)]:
        before,after=count(shape,kind),count(reopened,kind);assert before==after,(name,label,before,after);counts[label]=before
    evidence={'file':str(path),'bytes':path.stat().st_size,'valid_before':True,'valid_after':True,'counts':counts,'area_before':a,'area_after':aa,'signed_surface_volume_before':v,'signed_surface_volume_after':vv,'volume_note':'Only a closed, oriented solid has physical volume; galleries can contain open or overlapping shapes.'}
    if counts['faces']==0:
        before,after=count(shape,TopAbs_EDGE),count(reopened,TopAbs_EDGE)
        assert before==after and before>0,(name,before,after)
        p,q=GProp_GProps(),GProp_GProps();BRepGProp.LinearProperties_s(shape,p);BRepGProp.LinearProperties_s(reopened,q)
        assert abs(p.Mass()-q.Mass())<max(1e-7,abs(p.Mass())*1e-7),(name,p.Mass(),q.Mass())
        counts['edges']=before;evidence['length_before_mm']=p.Mass();evidence['length_after_mm']=q.Mass()
    return evidence


roundtrips=[]
try:
    gallery=[placed(f,3*(i%5),3*(i//5)) for i,f in enumerate(surface_faces+extra_faces)]
    roundtrips.append(step_roundtrip('surface_primitives.step',compound(gallery)))
    curves=[placed(BRepBuilderAPI_MakeEdge(c).Edge(),3*(i%4),3*(i//4)) for i,c in enumerate(extra_curves)]
    roundtrips.append(step_roundtrip('curve_primitives.step',compound(curves)))
    roundtrips.append(step_roundtrip('g2_transition.step',sew_surfaces([surface_faces[12],surface_faces[11],surface_faces[13]])))
    polygon=[]
    for i,m in enumerate(meshes):
        p=np.array(m['positions']).reshape(-1,3);ids=np.array(m['indices'],dtype=int).reshape(-1,3);faces=[]
        for indices in ids:
            w=BRepBuilderAPI_MakePolygon()
            for k in indices:w.Add(gp_Pnt(*p[k]))
            w.Close();faces.append(BRepBuilderAPI_MakeFace(w.Wire(),True).Face())
        shape=sew_surfaces(faces,tolerance=1e-8,solid=m['boundary_edges']==0)
        polygon.append(placed(shape,4*(i%4),4*(i//4)))
    roundtrips.append(step_roundtrip('polygon_primitives.step',compound(polygon)))
except Exception as exc:
    roundtrips.append({'pass':False,'error':str(exc),'traceback':traceback.format_exc()})

assert len(results)==46,len(results)
assert {r['name'] for r in results}==set(definitions)
report={'catalog_version':1,'operator_count':46,'passed':sum(r['pass'] for r in results),'failed':sum(not r['pass'] for r in results),'gpu':gpu,'operators':results,'step_roundtrips':roundtrips,'all_passed':all(r['pass'] for r in results) and len(roundtrips)==4 and all(r.get('valid_after',False) for r in roundtrips),'scope':'Automotive surface + bounded polygon baseline; hybrid GPU geometry and CPU B-rep topology. Neural policy still has its four MVP slots.'}
(OUT/'validation_baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(ROOT/'primitive_catalog.json').write_text(json.dumps({'version':1,'operators':results},indent=2),encoding='utf-8')
print(json.dumps({'operators':46,'passed':report['passed'],'failed':report['failed'],'step_roundtrips':len(roundtrips),'all_passed':report['all_passed']},indent=2))
if not report['all_passed']:raise SystemExit(1)
