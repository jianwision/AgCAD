"""Independent curve/body acceptance and combined 68-item delivery report."""
import json
import math
from pathlib import Path
import traceback
import numpy as np
from cad_ops import ROOT,valid
from cad_solids import solid_from_definition
from cad_curve_extensions import curve_from_record,curve_for_step
from OCP.gp import gp_Pnt,gp_Vec,gp_Dir,gp_Ax2,gp_Trsf
from OCP.Geom import Geom_BezierCurve,Geom_Parabola,Geom_Hyperbola,Geom_TrimmedCurve,Geom_OffsetCurve
from OCP.GC import GC_MakeArcOfCircle
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.GeomAdaptor import GeomAdaptor_Curve
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRep import BRep_Builder,BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_SOLID
from OCP.TopoDS import TopoDS_Compound,TopoDS
from OCP.STEPControl import STEPControl_Writer,STEPControl_Reader,STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_RetDone

OUT=ROOT/'output/primitives'
defs=json.loads((ROOT/'primitive_extensions.json').read_text())['operators']
raw=json.loads((OUT/'extension_curves.json').read_text());records=np.array(raw['records'])
samples=np.array(raw['samples']).reshape(9,513,9);ops=np.array(raw['ops'])
solids=json.loads((OUT/'extension_solids.json').read_text());results=[];native_curves=[];native_solids=[]
o=np.array([.1,.2,.3]);x=np.array([.6,.8,0]);y=np.array([-.8,.6,0]);z=np.array([0,0,1])
points=np.array([[0,0,0],[.4,.8,.2],[1,.3,.6],[1.6,1,.4],[2,0,.8]])


def xyz(p):return np.array([p.X(),p.Y(),p.Z()])


def check(a,b,tol=1e-10):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert math.isfinite(error) and error<=tol,(error,tol)
    return error


def run(name,backend,fn):
    try:
        evidence=fn() or {};results.append({'name':name,**defs[name],'backend':backend,'pass':True,'evidence':evidence});print('PASS',name,flush=True)
    except Exception as exc:
        results.append({'name':name,**defs[name],'backend':backend,'pass':False,'error':str(exc),'traceback':traceback.format_exc()});print('FAIL',name,str(exc),flush=True)


def natural_reference(t):
    h=.25;matrix=np.zeros((5,5));rhs=np.zeros((5,3));matrix[0,0]=matrix[-1,-1]=1
    for i in range(1,4):matrix[i,i-1:i+2]=[h,4*h,h];rhs[i]=6*(points[i+1]-2*points[i]+points[i-1])/h
    m=np.linalg.solve(matrix,rhs);i=min(int(t/h),3);q=(t-i*h)/h
    p=(1-q)*points[i]+q*points[i+1]+h*h/6*(((1-q)**3-(1-q))*m[i]+(q**3-q)*m[i+1])
    d=(points[i+1]-points[i])/h+h/6*((1-3*(1-q)**2)*m[i]+(3*q*q-1)*m[i+1])
    dd=(1-q)*m[i]+q*m[i+1]
    return np.concatenate([p,d,dd])


def analytic_reference(i,t):
    if i==0:
        a=np.array([0,0,0]);b=np.array([2,.3,.5]);da=np.array([1,2,0]);db=np.array([1,-1,.6])
        p=(2*t**3-3*t*t+1)*a+(t**3-2*t*t+t)*da+(-2*t**3+3*t*t)*b+(t**3-t*t)*db
        d=(6*t*t-6*t)*a+(3*t*t-4*t+1)*da+(-6*t*t+6*t)*b+(3*t*t-2*t)*db
        dd=(12*t-6)*a+(6*t-4)*da+(-12*t+6)*b+(6*t-2)*db
        return np.concatenate([p,d,dd])
    if i in [3,4]:
        if i==3:
            r=.8;pitch=.7/(2*math.pi);p=o+r*np.cos(t)*x+r*np.sin(t)*y+pitch*t*z
            d=-r*np.sin(t)*x+r*np.cos(t)*y+pitch*z;dd=-r*np.cos(t)*x-r*np.sin(t)*y
        else:
            r=.4+.08*t;p=o+r*np.cos(t)*x+r*np.sin(t)*y
            d=(.08*np.cos(t)-r*np.sin(t))*x+(.08*np.sin(t)+r*np.cos(t))*y
            dd=(-.16*np.sin(t)-r*np.cos(t))*x+(.16*np.cos(t)-r*np.sin(t))*y
        return np.concatenate([p,d,dd])
    if i==6:return natural_reference(t)
    if i==7:
        q=natural_reference(1-t);q[3:6]*=-1;return q
    axis=gp_Ax2(gp_Pnt(*o),gp_Dir(*z),gp_Dir(*x))
    if i==1:curve=Geom_Parabola(axis,.7)
    if i==2:curve=Geom_Hyperbola(axis,1.2,.55)
    if i==5:
        origin=np.array([.3,-.2,.5]);xx=np.array([.6,.8,0]);yy=np.array([0,0,1])
        q=[gp_Pnt(*(origin+2*math.cos(a)*xx+2*math.sin(a)*yy)) for a in [.2,1.4,3.7]]
        curve=GC_MakeArcOfCircle(*q).Value()
        t=curve.FirstParameter()+(curve.LastParameter()-curve.FirstParameter())*t/3.5
    if i==8:curve=Geom_OffsetCurve(Geom_Parabola(axis,.7),-.15,gp_Dir(*z))
    p=gp_Pnt();d,dd=gp_Vec(),gp_Vec();curve.D2(float(t),p,d,dd)
    return np.concatenate([xyz(p),xyz(d),xyz(dd)])


for i in range(9):native_curves.append(curve_from_record(records,samples,i))
step_curves=[curve_for_step(c,int(records[i,0])) for i,c in enumerate(native_curves)]


def curve_case(i):
    first,last=records[i,3:5];assert first<last
    expected_domains=[(0,1),(-1.2,1.4),(-.9,.9),(-.4,4*math.pi+.3),(.2,4*math.pi),(0,3.5),(0,1),(0,1),(-1.2,1.4)]
    check([first,last],expected_domains[i],1e-12)
    reference=np.array([analytic_reference(i,t) for t in np.linspace(first,last,513)])
    error=check(samples[i,:,:3],reference[:,:3],2e-12)
    evidence={'gpu_samples':513,'max_position_error_mm':error}
    if i!=8:evidence['max_derivative_error']=check(samples[i,:,3:],reference[:,3:],2e-11)
    curve=step_curves[i]
    if i in [3,4,8]:
        # Half-way between every pair of supplied samples; never just refit nodes.
        ts=first+(last-first)*(np.arange(512)+.5)/512
        err=max(float(np.linalg.norm(xyz(curve.Value(float(t)))-analytic_reference(i,t)[:3])) for t in ts)
        assert err<1e-5,err
        evidence.update({'step_representation':'interpolated B-spline approximation','off_grid_samples':512,'max_step_approximation_error_mm':err,'step_tolerance_mm':1e-5})
    else:
        error=max(float(np.linalg.norm(xyz(curve.Value(float(t)))-analytic_reference(i,t)[:3])) for t in np.linspace(first,last,129))
        assert error<1e-9,error;evidence.update({'step_representation':'native exact curve definition','max_step_curve_error_mm':error})
    if i==6:
        check([xyz(curve.Value(j/4)) for j in range(5)],points,1e-12)
        for t in [.25,.5,.75]:
            a=natural_reference(t-1e-8);b=natural_reference(t+1e-8);check(a[3:],b[3:],3e-6)
        evidence['interpolation_points']=5
    return evidence


curve_names=['hermite_curve','parabola_curve','hyperbola_curve','helix_curve','spiral_curve','three_point_arc','interpolate_curve']
for i,name in enumerate(curve_names):run(name,'CUDA construction/evaluation + independent CAD or analytic reference',lambda i=i:curve_case(i))
run('curve_reverse','CUDA curve jet reversal',lambda:curve_case(7))


def length_case():
    errors=[]
    for i in range(8):
        first,last=records[i,3:5]
        if i==3:reference=(last-first)*math.hypot(.8,.7/(2*math.pi))
        elif i==4:
            def primitive(t):
                r=.4+.08*t;return (r*math.hypot(r,.08)+.08**2*math.asinh(r/.08))/(2*.08)
            reference=primitive(last)-primitive(first)
        else:reference=GCPnts_AbscissaPoint.Length_s(GeomAdaptor_Curve(native_curves[i]),float(first),float(last),1e-11)
        errors.append(check(ops[i],reference,1e-8))
    check(ops[13:21],np.ones(8),1e-12)
    return {'curves':8,'max_independent_length_error_mm':max(errors),'invalid_inputs_rejected':8}


def parabola_length_primitive(t):
    u=t/(2*.7);return .7*(u*np.sqrt(1+u*u)+np.arcsinh(u))


def resample_case():
    p=ops[32:227].reshape(65,3);local=(p-o)@np.array([x,y,z]).T;t=local[:,1]
    check(local[:,0],t*t/(4*.7),1e-12);check(local[:,2],np.zeros(65),1e-12)
    lengths=parabola_length_primitive(t)-parabola_length_primitive(-1.2)
    target=np.linspace(0,parabola_length_primitive(1.4)-parabola_length_primitive(-1.2),65)
    error=check(lengths,target,2e-10)
    assert np.max(np.diff(t))-np.min(np.diff(t))>1e-3
    return {'points':65,'max_arc_length_spacing_error_mm':error,'nonuniform_parameters_confirmed':True}


def closest_case():
    query=gp_Pnt(.8,-.3,.5);projection=GeomAPI_ProjectPointOnCurve(query,native_curves[1])
    assert projection.NbPoints()>0
    e=check(ops[10:13],xyz(projection.NearestPoint()),1e-7)
    check(ops[9],projection.LowerDistance(),1e-12)
    check(ops[8],projection.LowerDistanceParameter(),1e-7)
    return {'max_cad_point_error_mm':e,'distance_mm':float(ops[9]),'parameter':float(ops[8])}


run('curve_length','CUDA quadrature + independent CAD/analytic lengths',length_case)
run('resample_curve','CUDA arc-length inversion',resample_case)
run('closest_point_curve','CUDA bounded search + CAD projector',closest_case)
run('planar_offset_curve','CUDA point offset + native CAD offset',lambda:curve_case(8))


def count(shape,kind):
    ex=TopExp_Explorer(shape,kind);n=0
    while ex.More():n+=1;ex.Next()
    return n


def properties(shape):
    area=GProp_GProps();volume=GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape,area,1e-10,False)
    BRepGProp.VolumeProperties_s(shape,volume,1e-10,True,False)
    return area.Mass(),volume.Mass()


def volume_area(kind,a,b,c,d):
    pi=math.pi
    if kind==0:return 4*pi*a**3/3,4*pi*a*a
    if kind==1:return 4*pi*a*b*c/3,None
    if kind==2:return pi*a*a*b,2*pi*a*(a+b)
    if kind==3:return pi*a*a*b/3,pi*a*(a+math.hypot(a,b))
    if kind==4:return pi*c*(a*a+a*b+b*b)/3,pi*(a*a+b*b+(a+b)*math.hypot(a-b,c))
    if kind==5:return 2*pi*pi*a*b*b,4*pi*pi*a*b
    if kind==6:return pi*a*a*b+4*pi*a**3/3,2*pi*a*b+4*pi*a*a
    if kind in [7,8]:
        n=int(d);base=n*a*a*math.sin(2*pi/n)/2;side=2*a*math.sin(pi/n)
        return (base*b,2*base+n*side*b) if kind==7 else (base*b/3,base+n*side*math.hypot(b,a*math.cos(pi/n))/2)
    return (a+d)*b*c/2,b*(a+d+c+math.hypot(a-d,c))+(a+d)*c


expected_sizes=[[1.2,0,0,0],[1.4,.8,1.1,0],[.8,2.3,0,0],[1,2.4,0,0],[1.1,.5,1.9,0],[1.4,.35,0,0],[.65,1.5,0,0],[1,1.8,0,6],[1.1,1.7,0,5],[2,1.2,1.6,.7]]


def solid_case(i):
    m=solids[i];s=np.array(m['spec']);check(s[:5],[i,*expected_sizes[i]],1e-12)
    check(s[8:20],[.1,-.2,.3,.6,.8,0,-.48,.36,.8,.64,-.48,.6],1e-12)
    shape=solid_from_definition(s);assert count(shape,TopAbs_SOLID)==1
    p=np.array(m['positions']).reshape(-1,3);ids=np.array(m['indices'],dtype=int).reshape(-1,3)
    n=np.array(m['normals']).reshape(-1,3);check(np.linalg.norm(n,axis=1),np.ones(len(n)),1e-12)
    local=(p-s[8:11])@np.array([s[11:14],s[14:17],s[17:20]]).T
    a,b,c,d=s[1:5];xx,yy,zz=local.T;rho=np.hypot(xx,yy)
    if i==0:check(np.sum(local*local,axis=1),np.full(len(p),a*a),1e-12)
    if i==1:check(np.sum((local/[a,b,c])**2,axis=1),np.ones(len(p)),1e-12)
    if i==1:
        ex=TopExp_Explorer(shape,TopAbs_FACE);surface=BRep_Tool.Surface_s(TopoDS.Face_s(ex.Current()))
        u0,u1,v0,v1=surface.Bounds();residual=[]
        for u in np.linspace(u0,u1,13):
            for v in np.linspace(v0,v1,11):
                q=(xyz(surface.Value(float(u),float(v)))-s[8:11])@np.array([s[11:14],s[14:17],s[17:20]]).T
                residual.append(abs(np.sum((q/[a,b,c])**2)-1))
        check(residual,np.zeros(len(residual)),1e-12)
    if i in [2,3,4]:
        height=c if i==4 else b;top=a if i==2 else b if i==4 else 0
        residual=np.minimum(np.minimum(np.abs(zz),np.abs(zz-height)),np.abs(rho-(a+(top-a)*zz/height)))
        check(residual,np.zeros(len(p)),1e-12)
    if i==5:check((rho-a)**2+zz**2,np.full(len(p),b*b),1e-12)
    if i==6:
        center_z=np.clip(zz,0,b);check(rho*rho+(zz-center_z)**2,np.full(len(p),a*a),1e-12)
    if i==7:check(np.minimum(abs(zz),abs(zz-b)),np.zeros(len(p)),1e-12);check(rho,np.full(len(p),a),1e-12)
    if i==8:check(np.sort(zz),[0]*int(d)+[b],1e-12)
    if i==9:
        expected=np.array([[(d if k&4 else a) if k&1 else 0,b if k&2 else 0,c if k&4 else 0] for k in range(8)])
        check(local,expected,1e-12)
    u,v,w=p[ids[:,0]],p[ids[:,1]],p[ids[:,2]]
    mesh_volume=float(np.einsum('ij,ij->i',u,np.cross(v,w)).sum()/6)
    theoretical,area_ref=volume_area(i,a,b,c,d);area,volume=properties(shape)
    volume_error=abs(volume-theoretical)/theoretical
    assert volume_error<1e-7,(i,volume,theoretical,volume_error)
    if area_ref is not None:check(area,area_ref,max(1e-8,abs(area_ref)*1e-8))
    relative_mesh_error=abs(mesh_volume-theoretical)/theoretical
    assert relative_mesh_error<(1e-12 if i>=7 else .025),(i,relative_mesh_error)
    # Topological metrics were independently checked by Rust from the exported triangles.
    assert m['boundary_edges']==0 and m['euler']==(0 if i==5 else 2)
    native_solids.append((i,shape))
    return {'brep_valid':True,'closed_solids':1,'faces':count(shape,TopAbs_FACE),'vertices':len(p),'mesh_triangles':len(ids),'analytic_volume_mm3':theoretical,'cad_volume_mm3':volume,'cad_relative_volume_error':volume_error,'mesh_relative_volume_error':relative_mesh_error,'cad_area_mm2':area}


for i,m in enumerate(solids):run(m['name'],'CUDA definition/mesh + native CPU CAD solid',lambda i=i:solid_case(i))


def compound(shapes):
    result=TopoDS_Compound();b=BRep_Builder();b.MakeCompound(result)
    for shape in shapes:b.Add(result,shape)
    return result


def placed(shape,x,y):
    t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));return BRepBuilderAPI_Transform(shape,t,True).Shape()


def roundtrip(name,shape,expected_solids=None):
    valid(shape);Interface_Static.SetCVal_s('write.step.schema','AP214IS');Interface_Static.SetCVal_s('write.step.unit','MM')
    writer=STEPControl_Writer();assert writer.Transfer(shape,STEPControl_AsIs)==IFSelect_RetDone
    path=OUT/name;assert writer.Write(str(path))==IFSelect_RetDone
    reader=STEPControl_Reader();assert reader.ReadFile(str(path))==IFSelect_RetDone;assert reader.TransferRoots()>0
    reopened=reader.OneShape();valid(reopened);counts={}
    for label,kind in [('solids',TopAbs_SOLID),('faces',TopAbs_FACE),('edge_uses',TopAbs_EDGE)]:
        before,after=count(shape,kind),count(reopened,kind);assert before==after,(name,label,before,after);counts[label]=before
    if expected_solids is not None:assert counts['solids']==expected_solids
    a,v=properties(shape);aa,vv=properties(reopened);check([a,v],[aa,vv],max(1e-7,abs(a)*1e-7,abs(v)*1e-7))
    result={'file':str(path),'bytes':path.stat().st_size,'valid_before':True,'valid_after':True,'counts':counts,'area_before_mm2':a,'area_after_mm2':aa,'volume_before_mm3':v,'volume_after_mm3':vv}
    if counts['faces']==0:
        p,q=GProp_GProps(),GProp_GProps();BRepGProp.LinearProperties_s(shape,p);BRepGProp.LinearProperties_s(reopened,q)
        check(p.Mass(),q.Mass(),max(1e-7,p.Mass()*1e-8));result.update({'length_before_mm':p.Mass(),'length_after_mm':q.Mass()})
    return result


roundtrips=[]
try:
    curves=[placed(BRepBuilderAPI_MakeEdge(c).Edge(),5*(i%3),5*(i//3)) for i,c in enumerate(step_curves)]
    roundtrips.append(roundtrip('extended_curves.step',compound(curves),0))
    assert len(native_solids)==10,'Cannot deliver missing solid fixtures'
    shapes=[placed(s,4.5*(i%5),4.5*(i//5)) for i,s in native_solids]
    roundtrips.append(roundtrip('solid_primitives.step',compound(shapes),10))
except Exception as exc:roundtrips.append({'valid_after':False,'error':str(exc),'traceback':traceback.format_exc()})

assert len(results)==22 and {r['name'] for r in results}==set(defs)
extension={'version':2,'operator_count':22,'passed':sum(r['pass'] for r in results),'failed':sum(not r['pass'] for r in results),'operators':results,'step_roundtrips':roundtrips,'all_passed':all(r['pass'] for r in results) and len(roundtrips)==2 and all(r['valid_after'] for r in roundtrips)}
(OUT/'validation_extensions.json').write_text(json.dumps(extension,indent=2),encoding='utf-8')
baseline=json.loads((OUT/'validation_baseline.json').read_text());all_results=baseline['operators']+results
report={'catalog_version':2,'operator_count':68,'passed':sum(r['pass'] for r in all_results),'failed':sum(not r['pass'] for r in all_results),'gpu':baseline['gpu'],'operators':all_results,'step_roundtrips':baseline['step_roundtrips']+roundtrips,'all_passed':baseline['all_passed'] and extension['all_passed'],'scope':'46 original automotive surface/polygon items plus 12 curve and 10 exact-solid items. GPU geometry and CPU B-rep topology; neural policy remains four slots.'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(ROOT/'primitive_catalog.json').write_text(json.dumps({'version':2,'operators':all_results},indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['operator_count','passed','failed','all_passed']},indent=2))
if not report['all_passed']:raise SystemExit(1)
