"""Side-effect-free shared acceptance helpers for incremental primitive suites."""
import json
import math
import numpy as np
from cad_ops import ROOT,valid,sew_surfaces
from cad_solids import polygon_face
from export_step import knots
from OCP.gp import gp_Pnt,gp_Vec,gp_Trsf
from OCP.Geom import Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array2OfReal
from OCP.TopoDS import TopoDS_Compound
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_SOLID,TopAbs_EDGE
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Writer,STEPControl_Reader,STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone

OUT=ROOT/'output/primitives'

def xyz(p):return np.array([p.X(),p.Y(),p.Z()])
def check(a,b,tol=1e-10):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert math.isfinite(error) and error<=tol,(error,tol)
    return error
def count(shape,kind):
    e=TopExp_Explorer(shape,kind);n=0
    while e.More():n+=1;e.Next()
    return n
def properties(shape):
    a=GProp_GProps();BRepGProp.SurfaceProperties_s(shape,a,1e-10,False)
    # A periodic sheet can become a closed shell during STEP transfer. Physical
    # solid volume must only count SOLID objects, not incidental closed shells.
    total=0.;e=TopExp_Explorer(shape,TopAbs_SOLID)
    while e.More():
        v=GProp_GProps();BRepGProp.VolumeProperties_s(e.Current(),v,1e-10,True,False);total+=v.Mass();e.Next()
    return a.Mass(),total
def compound(shapes):
    result=TopoDS_Compound();b=BRep_Builder();b.MakeCompound(result)
    for shape in shapes:b.Add(result,shape)
    return result
def placed(shape,x,y,z=0):
    t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,z));return BRepBuilderAPI_Transform(shape,t,True).Shape()
def build_surface(record):
    nu,nv=record['size'];pu,pv=record['degree'];cp=np.array(record['poles_xyzw']).reshape(nu,nv,4)
    points=TColgp_Array2OfPnt(1,nu,1,nv);weights=TColStd_Array2OfReal(1,nu,1,nv)
    for i in range(nu):
        for j in range(nv):points.SetValue(i+1,j+1,gp_Pnt(*cp[i,j,:3]));weights.SetValue(i+1,j+1,float(cp[i,j,3]))
    uk,um=knots(record['u_knots']);vk,vm=knots(record['v_knots'])
    return Geom_BSplineSurface(points,weights,uk,vk,um,vm,pu,pv,False,False)
def mesh_shape(mesh):
    p=np.array(mesh['positions']).reshape(-1,3);ids=np.array(mesh['indices'],dtype=int).reshape(-1,3)
    return sew_surfaces([polygon_face(p[t]) for t in ids],tolerance=1e-8,solid=mesh['boundary_edges']==0)
def mesh_volume(mesh):
    p=np.array(mesh['positions']).reshape(-1,3);ids=np.array(mesh['indices'],dtype=int).reshape(-1,3)
    return float(np.einsum('ij,ij->i',p[ids[:,0]],np.cross(p[ids[:,1]],p[ids[:,2]])).sum()/6)
def roundtrip(name,shape,expected_solids=None):
    valid(shape);Interface_Static.SetCVal_s('write.step.schema','AP214IS');Interface_Static.SetCVal_s('write.step.unit','MM')
    writer=STEPControl_Writer();assert writer.Transfer(shape,STEPControl_AsIs)==IFSelect_RetDone
    path=OUT/name;assert writer.Write(str(path))==IFSelect_RetDone
    reader=STEPControl_Reader();assert reader.ReadFile(str(path))==IFSelect_RetDone;assert reader.TransferRoots()>0
    reopened=reader.OneShape();valid(reopened);counts={}
    for label,kind in [('solids',TopAbs_SOLID),('faces',TopAbs_FACE)]:
        before,after=count(shape,kind),count(reopened,kind);assert before==after,(name,label,before,after);counts[label]=before
    if expected_solids is not None:assert counts['solids']==expected_solids,(name,counts)
    a,v=properties(shape);aa,vv=properties(reopened);check([a,v],[aa,vv],max(1e-7,abs(a)*1e-7,abs(v)*1e-7))
    result={'file':str(path),'bytes':path.stat().st_size,'valid_before':True,'valid_after':True,'counts':counts,'area_before_mm2':a,'area_after_mm2':aa,'closed_volume_before_mm3':v,'closed_volume_after_mm3':vv}
    if counts['faces']==0:
        assert count(shape,TopAbs_EDGE)==count(reopened,TopAbs_EDGE)>0
        p,q=GProp_GProps(),GProp_GProps();BRepGProp.LinearProperties_s(shape,p);BRepGProp.LinearProperties_s(reopened,q)
        check(p.Mass(),q.Mass(),max(1e-7,p.Mass()*1e-8));result.update({'edges':count(shape,TopAbs_EDGE),'length_before_mm':p.Mass(),'length_after_mm':q.Mass()})
    return result
def finish_stage(name,definitions,results,roundtrips,expected_steps):
    assert {r['name'] for r in results}==set(definitions) and len(results)==len(definitions)
    stage={'stage':name,'operator_count':len(results),'passed':sum(r['pass'] for r in results),'failed':sum(not r['pass'] for r in results),'operators':results,'step_roundtrips':roundtrips,'all_passed':all(r['pass'] for r in results) and len(roundtrips)==expected_steps and all(r.get('valid_after',False) for r in roundtrips)}
    (OUT/f'validation_{name}.json').write_text(json.dumps(stage,indent=2),encoding='utf-8')
    parent=json.loads((OUT/'validation.json').read_text());all_results=parent['operators']+results
    assert len({r['name'] for r in all_results})==len(all_results),'Duplicate stage; rerun the full suite'
    report={**parent,'catalog_version':4,'operator_count':len(all_results),'passed':sum(r['pass'] for r in all_results),'failed':sum(not r['pass'] for r in all_results),'operators':all_results,'step_roundtrips':parent['step_roundtrips']+roundtrips,'all_passed':parent['all_passed'] and stage['all_passed'],'scope':parent['scope']+'; '+name}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'primitive_catalog.json').write_text(json.dumps({'version':4,'operators':all_results},indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['operator_count','passed','failed','all_passed']},indent=2))
    if not report['all_passed']:raise SystemExit(1)
