"""Composable CPU B-rep/curve operations complementing the CUDA geometry library.

Every editing function returns a fresh object. Approximate edits report sampled
error and reject the request if its requested tolerance is not met.
"""
import math
import numpy as np
from cad_ops import valid,sew_surfaces
from cad_topology_operations import boolean_union
from OCP.gp import gp_Pnt,gp_Dir,gp_Pln
from OCP.Geom import Geom_BSplineCurve,Geom_BSplineSurface
from OCP.TColgp import TColgp_Array1OfPnt,TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet,BRepFilletAPI_MakeChamfer
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid,BRepOffsetAPI_DraftAngle
from OCP.TopTools import TopTools_ListOfShape
from OCP.ShapeFix import ShapeFix_ShapeTolerance,ShapeFix_Wireframe
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.GeomAbs import GeomAbs_Intersection

def xyz(p):return np.array([p.X(),p.Y(),p.Z()])
def array(values,integer=False):
    a=(TColStd_Array1OfInteger if integer else TColStd_Array1OfReal)(1,len(values))
    for i,v in enumerate(values):a.SetValue(i+1,int(v) if integer else float(v))
    return a
def bspline(poles,degree=3,knots=None,mults=None):
    p=np.asarray(poles,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)) or not 1<=degree<len(p):raise ValueError('Invalid spline')
    if knots is None:
        spans=len(p)-degree;knots=np.linspace(0,1,spans+1);mults=[degree+1]+[1]*(spans-1)+[degree+1]
    a=TColgp_Array1OfPnt(1,len(p))
    for i,q in enumerate(p):a.SetValue(i+1,gp_Pnt(*q))
    return Geom_BSplineCurve(a,array(knots),array(mults,True),degree,False)
def copy_curve(c):return c.Copy()
def knot_insert(c,u,multiplicity=1):
    if not c.FirstParameter()<u<c.LastParameter() or not 1<=multiplicity<=c.Degree():raise ValueError('Interior knot/multiplicity required')
    out=copy_curve(c);out.InsertKnot(float(u),int(multiplicity),1e-12,True);return out
def knot_remove(c,index,multiplicity=0,tolerance=1e-10):
    if not 1<index<c.NbKnots() or tolerance<=0:raise ValueError('Invalid removal request')
    out=copy_curve(c)
    if not out.RemoveKnot(index,multiplicity,tolerance):raise ValueError('Knot cannot be removed at requested tolerance')
    return out
def degree_elevate(c,degree):
    if not c.Degree()<=degree<=25:raise ValueError('Degree must increase and be <=25')
    out=copy_curve(c);out.IncreaseDegree(degree);return out
def curve_split(c,u):
    lo,hi=c.FirstParameter(),c.LastParameter()
    if not lo<u<hi:raise ValueError('Split must be interior')
    a,b=copy_curve(c),copy_curve(c);a.Segment(lo,u,1e-12);b.Segment(u,hi,1e-12);return a,b
def reparameterize(c,low,high):
    if not math.isfinite(low+high) or high<=low:raise ValueError('Increasing finite domain required')
    out=copy_curve(c);a,b=c.FirstParameter(),c.LastParameter()
    out.SetKnots(array([low+(c.Knot(i)-a)*(high-low)/(b-a) for i in range(1,c.NbKnots()+1)]));return out
def bernstein(degree,u):
    u=np.asarray(u);return np.column_stack([math.comb(degree,j)*u**j*(1-u)**(degree-j) for j in range(degree+1)])
def degree_reduce(c,degree,tolerance):
    if c.IsRational() or c.NbKnots()!=2 or not 1<=degree<c.Degree() or tolerance<=0:raise ValueError('Reduction supports one nonrational Bezier span')
    a,b=c.FirstParameter(),c.LastParameter();u=np.linspace(0,1,257);points=np.array([xyz(c.Value(float(a+t*(b-a)))) for t in u])
    matrix=bernstein(degree,u);ends=points[[0,-1]];rhs=points-matrix[:,[0,-1]]@ends
    cp=np.vstack([ends[0],np.linalg.lstsq(matrix[:,1:-1],rhs,rcond=None)[0],ends[1]]) if degree>1 else ends
    out=bspline(cp,degree,[a,b],[degree+1,degree+1]);q=(np.arange(2048)+.37)/2048
    error=max(np.linalg.norm(xyz(c.Value(float(a+t*(b-a))))-xyz(out.Value(float(a+t*(b-a))))) for t in q)
    if error>tolerance:raise ValueError(f'Reduction sampled error {error} exceeds {tolerance}')
    return out,{'sampled_max_error_mm':float(error),'off_grid_samples':2048,'certified_bound':False}
def fit_curve(points,pole_count=8,smoothing=0.):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)) or not 4<=pole_count<=len(p) or smoothing<0:raise ValueError('Invalid fit')
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if np.any(np.diff(distance)<1e-12):raise ValueError('Repeated sample')
    u=distance/distance[-1];basis=[]
    for j in range(pole_count):
        cp=np.zeros((pole_count,3));cp[j,0]=1;c=bspline(cp)
        basis.append([c.Value(float(t)).X() for t in u])
    matrix=np.array(basis).T;fixed=p[[0,-1]];rhs=p-matrix[:,[0,-1]]@fixed
    d=np.diff(np.eye(pole_count),2,axis=0);m=matrix[:,1:-1];r=rhs
    if smoothing:m=np.vstack([m,math.sqrt(smoothing)*d[:,1:-1]]);r=np.vstack([r,-math.sqrt(smoothing)*d[:,[0,-1]]@fixed])
    cp=np.vstack([p[0],np.linalg.lstsq(m,r,rcond=None)[0],p[-1]]);curve=bspline(cp)
    error=float(np.max(np.linalg.norm(np.array([xyz(curve.Value(float(t))) for t in u])-p,axis=1)))
    return curve,{'max_sample_residual_mm':error,'pole_second_difference_energy':float(np.sum(np.diff(cp,2,axis=0)**2))}
def fair_curve(points,pole_count=8,strength=.1):return fit_curve(points,pole_count,strength)
def compatible_sections(curves):
    if len(curves)<2:raise ValueError('At least two sections required')
    out=[reparameterize(c,0,1) for c in curves];degree=max(c.Degree() for c in out)
    for c in out:c.IncreaseDegree(degree)
    union={}
    for c in out:
        for i in range(2,c.NbKnots()):union[round(c.Knot(i),12)]=max(union.get(round(c.Knot(i),12),0),c.Multiplicity(i))
    for c in out:
        for k,m in sorted(union.items()):c.InsertKnot(k,m,1e-10,False)
    if any(c.NbPoles()!=out[0].NbPoles() for c in out):raise ValueError('Incompatible sections after harmonization')
    return out
def bezier_surface(poles):
    p=np.asarray(poles,dtype=float);nu,nv,_=p.shape;arr=TColgp_Array2OfPnt(1,nu,1,nv)
    if nu<2 or nv<2 or nu>26 or nv>26 or not np.all(np.isfinite(p)):raise ValueError('Invalid Bezier net')
    for i in range(nu):
        for j in range(nv):arr.SetValue(i+1,j+1,gp_Pnt(*p[i,j]))
    return Geom_BSplineSurface(arr,array([0,1]),array([0,1]),array([nu,nu],True),array([nv,nv],True),nu-1,nv-1,False,False)
def extend_bezier_u(surface,low=-.2,high=1.2):
    if surface.NbUKnots()!=2 or surface.NbVKnots()!=2 or surface.IsURational() or surface.IsVRational() or not low<0<1<high:raise ValueError('Nonrational Bezier extension across U=[0,1] required')
    p=surface.UDegree();q=surface.VDegree();u=np.linspace(0,1,p+1);v=np.linspace(0,1,q+1)
    vals=np.array([[xyz(surface.Value(float(low+t*(high-low)),float(s))) for s in v] for t in u])
    cp=np.linalg.solve(bernstein(p,u),vals.reshape(p+1,-1)).reshape(p+1,q+1,3)
    cp=np.linalg.solve(bernstein(q,v),cp.transpose(1,0,2).reshape(q+1,-1)).reshape(q+1,p+1,3).transpose(1,0,2)
    return bezier_surface(cp)
def untrim_surface(surface):return valid(BRepBuilderAPI_MakeFace(surface,*surface.Bounds(),1e-7).Face())
def replace_surface_face(face,replacement,tolerance=1e-7):
    # Bounded replacement of an isolated natural rectangular patch. Rejects
    # boundary drift; arbitrary solid-face extension/intersection is separate.
    from OCP.BRep import BRep_Tool
    source=BRep_Tool.Surface_s(face)
    from OCP.BRepTools import BRepTools
    bounds=BRepTools.UVBounds_s(face)
    if np.max(np.abs(np.array(bounds)-source.Bounds()))>1e-10:raise ValueError('Replacement requires an untrimmed rectangular patch')
    for t in np.linspace(0,1,33):
        for u,v in [(0,t),(1,t),(t,0),(t,1)]:
            if np.linalg.norm(xyz(source.Value(float(u),float(v)))-xyz(replacement.Value(float(u),float(v))))>tolerance:raise ValueError('Replacement boundary mismatch')
    return valid(BRepBuilderAPI_MakeFace(replacement,*replacement.Bounds(),tolerance).Face())
def require_members(shape,items,kind):
    from OCP.TopExp import TopExp_Explorer
    members=[];e=TopExp_Explorer(shape,kind)
    while e.More():members.append(e.Current());e.Next()
    if not items or any(not any(item.IsSame(m) for m in members) for item in items):raise ValueError('Selection does not belong to input shape')
def fillet(shape,edges,radius,end_radius=None):
    from OCP.TopAbs import TopAbs_EDGE
    require_members(shape,edges,TopAbs_EDGE)
    if radius<=0 or (end_radius is not None and end_radius<=0) or not edges:raise ValueError('Positive radii and edges required')
    op=BRepFilletAPI_MakeFillet(shape)
    op.SetParams(1e-6,1e-5,1e-7,1e-7,1e-7,1e-5)
    for edge in edges:
        if end_radius is None:op.Add(radius,edge)
        else:op.Add(radius,end_radius,edge)
    op.Build()
    if not op.IsDone():raise ValueError('Fillet failed')
    return valid(op.Shape())
def chamfer(shape,edges,distance):
    from OCP.TopAbs import TopAbs_EDGE
    require_members(shape,edges,TopAbs_EDGE)
    if distance<=0 or not edges:raise ValueError('Positive distance/edges required')
    op=BRepFilletAPI_MakeChamfer(shape)
    for edge in edges:op.Add(distance,edge)
    op.Build()
    if not op.IsDone():raise ValueError('Chamfer failed')
    return valid(op.Shape())
def thicken(face,offset):
    if not math.isfinite(offset) or abs(offset)<1e-9:raise ValueError('Nonzero thickness required')
    op=BRepOffsetAPI_MakeThickSolid();op.MakeThickSolidBySimple(face,offset)
    if not op.IsDone():raise ValueError('Thickening failed')
    from OCP.BRepLib import BRepLib
    from OCP.TopoDS import TopoDS
    result=TopoDS.Solid_s(op.Shape())
    if not BRepLib.OrientClosedSolid_s(result):raise ValueError('Thickened shell is not an orientable closed solid')
    return valid(result)
def shell(shape,closing_faces,offset,tolerance=1e-7):
    from OCP.TopAbs import TopAbs_FACE
    require_members(shape,closing_faces,TopAbs_FACE)
    if not closing_faces or abs(offset)<tolerance:raise ValueError('Closing faces and nonzero offset required')
    faces=TopTools_ListOfShape()
    for f in closing_faces:faces.Append(f)
    op=BRepOffsetAPI_MakeThickSolid();op.MakeThickSolidByJoin(shape,faces,offset,tolerance,Join=GeomAbs_Intersection)
    if not op.IsDone():raise ValueError('Shell failed')
    return valid(op.Shape())
def draft(shape,faces,angle,direction=(0,0,1),neutral_origin=(0,0,0)):
    if not faces or not 0<abs(angle)<math.pi/4:raise ValueError('Faces and a bounded nonzero draft angle required')
    op=BRepOffsetAPI_DraftAngle(shape);axis=gp_Dir(*direction);plane=gp_Pln(gp_Pnt(*neutral_origin),axis)
    for face in faces:
        op.Add(face,axis,angle,plane)
        if not op.AddDone():raise ValueError('Draft rejected selected face')
    op.Build()
    if not op.IsDone():raise ValueError('Draft failed')
    return valid(op.Shape())
def rib(base,profile_face,thickness):return boolean_union(base,thicken(profile_face,thickness))
def unify_tolerance(shape,tolerance):
    if not 1e-9<=tolerance<=1e-3:raise ValueError('Tolerance outside supported mm range')
    # Copy geometry/topology first; this operation must not alter the input.
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    out=BRepBuilderAPI_Copy(shape,True,False).Shape();ShapeFix_ShapeTolerance().SetTolerance(out,tolerance);return valid(out)
def cleanup_small_features(shape,tolerance=1e-6):
    if not 1e-9<=tolerance<=1e-2:raise ValueError('Unsupported cleanup tolerance')
    unify=ShapeUpgrade_UnifySameDomain(shape,True,True,False);unify.SetLinearTolerance(tolerance);unify.Build()
    repair=ShapeFix_Wireframe(unify.Shape());repair.SetPrecision(tolerance);repair.SetMaxTolerance(tolerance);repair.FixSmallEdges();repair.FixWireGaps()
    return valid(repair.Shape())
def surface_quality(surface,samples=17):
    from OCP.gp import gp_Vec
    if not 3<=samples<=129:raise ValueError('Quality grid must have 3..129 samples per axis')
    low,high,bottom,top=surface.Bounds();minimum=float('inf');maximum_curvature=0.
    for u in np.linspace(low,high,samples):
        for v in np.linspace(bottom,top,samples):
            p=gp_Pnt();du,dv,duu,dvv,duv=[gp_Vec() for _ in range(5)];surface.D2(float(u),float(v),p,du,dv,duu,dvv,duv)
            a,b=xyz(du),xyz(dv);cross=np.cross(a,b);jac=np.linalg.norm(cross);minimum=min(minimum,jac)
            if jac<1e-12:continue
            normal=cross/jac;E,F,G=np.dot(a,a),np.dot(a,b),np.dot(b,b);e,f,g=np.dot(normal,xyz(duu)),np.dot(normal,xyz(duv)),np.dot(normal,xyz(dvv))
            H=(e*G-2*f*F+g*E)/(2*(E*G-F*F));K=(e*g-f*f)/(E*G-F*F);k=abs(H)+math.sqrt(max(0,H*H-K));maximum_curvature=max(maximum_curvature,k)
    return {'samples':samples*samples,'minimum_sampled_jacobian':float(minimum),'maximum_sampled_principal_curvature_per_mm':float(maximum_curvature),'sampled_regular':bool(minimum>1e-12),'global_injectivity_certified':False}
