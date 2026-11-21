"""Non-destructive B-rep operations with explicit native/sampled tolerance budgets."""
from dataclasses import dataclass,asdict
import math
import numpy as np
import cad_ops
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut,BRepAlgoAPI_Common,BRepAlgoAPI_Splitter,BRepAlgoAPI_Section
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy,BRepBuilderAPI_MakeVertex,BRepBuilderAPI_Sewing,BRepBuilderAPI_MakeSolid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt,gp_Pnt2d
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopAbs import TopAbs_VERTEX,TopAbs_EDGE,TopAbs_FACE,TopAbs_SHELL,TopAbs_SOLID,TopAbs_IN,TopAbs_ON
from OCP.TopExp import TopExp,TopExp_Explorer
from OCP.TopTools import TopTools_IndexedMapOfShape,TopTools_ListOfShape
from OCP.TopoDS import TopoDS


@dataclass(frozen=True)
class TolerancePolicy:
    absolute_mm:float=1e-6
    relative:float=1e-10
    maximum_native_mm:float=1e-3
    maximum_sampled_deformation_mm:float=1e-3
    minimum_feature_mm:float=1e-5

    def __post_init__(self):
        values=asdict(self)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values.values()):raise ValueError('Invalid tolerance policy')
        if not 1e-9<=self.absolute_mm<=self.maximum_native_mm<=1 or self.relative>1e-3 or self.maximum_sampled_deformation_mm>10:
            raise ValueError('Unsupported tolerance budget')

    def resolve(self,shapes):
        scale=0.
        for shape in shapes:
            report=inspect_shape(shape,self)
            if not report['valid'] or report['nonmanifold_edges']:raise ValueError('Input topology is invalid/nonmanifold')
            if report['maximum_native_tolerance_mm']>self.maximum_native_mm*(1+1e-10):raise ValueError('Input native tolerance exceeds user budget')
            scale=max(scale,report['bbox_diagonal_mm'])
        linear=max(self.absolute_mm,self.relative*scale)
        if linear>self.maximum_native_mm:raise ValueError('Scale-resolved tolerance exceeds user budget')
        return linear


def subshapes(shape,kind):
    mapping=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(shape,kind,mapping)
    casts={TopAbs_VERTEX:TopoDS.Vertex_s,TopAbs_EDGE:TopoDS.Edge_s,TopAbs_FACE:TopoDS.Face_s,TopAbs_SHELL:TopoDS.Shell_s,TopAbs_SOLID:TopoDS.Solid_s}
    return [casts[kind](mapping.FindKey(i)) for i in range(1,mapping.Extent()+1)]


def inspect_shape(shape,policy=None):
    policy=policy or TolerancePolicy()
    if shape.IsNull():return {'valid':False,'reason':'NULL_SHAPE'}
    vertices=subshapes(shape,TopAbs_VERTEX);edges=subshapes(shape,TopAbs_EDGE);faces=subshapes(shape,TopAbs_FACE)
    native=[BRep_Tool.Tolerance_s(s) for s in [*vertices,*edges,*faces]]
    lengths=[];uses=np.zeros(len(edges),dtype=int);seams=set()
    index=TopTools_IndexedMapOfShape()
    for edge in edges:index.Add(edge)
    for face in faces:
        for edge in subshapes(face,TopAbs_EDGE):
            i=index.FindIndex(edge)-1;uses[i]+=1
            if BRep_Tool.IsClosed_s(edge,face):seams.add(i)
    degenerate=set()
    for i,edge in enumerate(edges):
        if BRep_Tool.Degenerated_s(edge):degenerate.add(i);continue
        p=GProp_GProps();BRepGProp.LinearProperties_s(edge,p);lengths.append(p.Mass())
    bounds=Bnd_Box();BRepBndLib.AddOptimal_s(shape,bounds,False,False)
    xyz=np.array(bounds.Get()) if not bounds.IsVoid() else np.zeros(6)
    area=GProp_GProps();volume=GProp_GProps();BRepGProp.SurfaceProperties_s(shape,area)
    solids=subshapes(shape,TopAbs_SOLID)
    # Shell surface integration is not solid volume; sum only explicit solids.
    solid_volume=0.
    for solid in solids:BRepGProp.VolumeProperties_s(solid,volume);solid_volume+=volume.Mass()
    shells=subshapes(shape,TopAbs_SHELL);nonmanifold=0;owned_faces=TopTools_IndexedMapOfShape()
    for shell in shells:
        local=np.zeros(len(edges),dtype=int)
        for face in subshapes(shell,TopAbs_FACE):
            owned_faces.Add(face)
            for edge in subshapes(face,TopAbs_EDGE):local[index.FindIndex(edge)-1]+=1
        nonmanifold+=int(np.sum(local>2))
    local=np.zeros(len(edges),dtype=int)
    for face in faces:
        if not owned_faces.Contains(face):
            for edge in subshapes(face,TopAbs_EDGE):local[index.FindIndex(edge)-1]+=1
    nonmanifold+=int(np.sum(local>2))
    return {'valid':bool(BRepCheck_Analyzer(shape).IsValid()),'vertices':len(vertices),'edges':len(edges),'faces':len(faces),
        'shells':len(shells),'solids':len(solids),
        'faces_per_solid':[len(subshapes(s,TopAbs_FACE)) for s in solids],
        'free_edges':sum(int(n==1 and i not in seams and i not in degenerate) for i,n in enumerate(uses)),
        'nonmanifold_edges':nonmanifold,'edge_incidence_scope':'per shell; remaining unshelled faces together',
        'unowned_edges':int(np.sum(uses==0)),
        'periodic_seam_edges':len(seams),'degenerated_edges':len(degenerate),
        'maximum_native_tolerance_mm':max(native,default=0.),'minimum_edge_length_mm':min(lengths,default=0.),
        'small_edges':sum(v<policy.minimum_feature_mm for v in lengths),
        'bbox_diagonal_mm':float(np.linalg.norm(xyz[3:]-xyz[:3])),
        'area_mm2':area.Mass(),'solid_volume_mm3':solid_volume}


def copy_shape(shape):return BRepBuilderAPI_Copy(shape,True,False).Shape()


def compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    builder=BRep_Builder();result=TopoDS_Compound();builder.MakeCompound(result)
    for shape in shapes:builder.Add(result,shape)
    return result


def sample_shape(shape):
    points=[BRep_Tool.Pnt_s(v) for v in subshapes(shape,TopAbs_VERTEX)]
    for edge in subshapes(shape,TopAbs_EDGE):
        if BRep_Tool.Degenerated_s(edge):continue
        curve=BRepAdaptor_Curve(edge);low,high=curve.FirstParameter(),curve.LastParameter()
        if not math.isfinite(low+high):raise ValueError('Unbounded topology is unsupported')
        points.extend(curve.Value(float(t)) for t in np.linspace(low,high,7))
    for face in subshapes(shape,TopAbs_FACE):
        surface=BRep_Tool.Surface_s(face);low,high,bottom,top=BRepTools.UVBounds_s(face)
        if not all(math.isfinite(v) for v in (low,high,bottom,top)):raise ValueError('Unbounded face is unsupported')
        for u in np.linspace(low,high,5):
            for v in np.linspace(bottom,top,5):
                if BRepClass_FaceClassifier(face,gp_Pnt2d(float(u),float(v)),1e-9).State() in (TopAbs_IN,TopAbs_ON):
                    points.append(surface.Value(float(u),float(v)))
    if len(points)>20000:raise ValueError('Topology verification sampling budget exceeded')
    return points


def sampled_deformation(before,after):
    maximum=0.;count=0
    for a,b in ((before,after),(after,before)):
        for point in sample_shape(a):
            distance=BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(point).Vertex(),b);distance.Perform()
            if not distance.IsDone() or distance.NbSolution()<1:raise ValueError('Cannot evaluate topology deformation')
            maximum=max(maximum,distance.Value());count+=1
    return {'symmetric_sampled_max_mm':maximum,'samples':count,'global_deformation_certified':False}


def finish(shape,policy,linear,*,operation,inputs,history=None,preserve=False,require_solid=False):
    report=inspect_shape(shape,policy)
    if not report['valid'] or report['nonmanifold_edges']:raise ValueError('Result topology invalid/nonmanifold')
    if report['maximum_native_tolerance_mm']>policy.maximum_native_mm*(1+1e-10):raise ValueError('Result native tolerance exceeds user budget')
    if require_solid and (report['solids']<1 or report['free_edges'] or report['solid_volume_mm3']<=0):raise ValueError('Result is not a positive closed solid')
    evidence={'operation':operation,'backend':'cpu-occt','policy':asdict(policy),'resolved_linear_mm':linear,'topology':report,
              'history':history or [],'input_shapes_copied':True}
    if preserve:
        source=compound(inputs) if len(inputs)>1 else inputs[0]
        deformation=sampled_deformation(source,shape);evidence['deformation']=deformation
        if deformation['symmetric_sampled_max_mm']>policy.maximum_sampled_deformation_mm:raise ValueError('Sampled deformation exceeds user budget')
    return shape,evidence


def shape_list(values):
    result=TopTools_ListOfShape()
    for value in values:result.Append(value)
    return result


def operation_history(operation,inputs):
    result=[]
    for number,shape in enumerate(inputs):
        for i,face in enumerate(subshapes(shape,TopAbs_FACE)):
            result.append({'input':number,'face_index_at_input_revision':i,'modified':len(list(operation.Modified(face))),
                           'generated':len(list(operation.Generated(face))),'deleted':bool(operation.IsDeleted(face))})
    return result


def boolean(a,b,*,operation='union',policy=None,require_solid=True):
    policy=policy or TolerancePolicy();linear=policy.resolve([a,b]);copies=[copy_shape(a),copy_shape(b)]
    builders={'union':BRepAlgoAPI_Fuse,'cut':BRepAlgoAPI_Cut,'common':BRepAlgoAPI_Common}
    if operation not in builders:raise ValueError('Unknown Boolean operation')
    op=builders[operation]();op.SetArguments(shape_list(copies[:1]));op.SetTools(shape_list(copies[1:]))
    op.SetNonDestructive(True);op.SetRunParallel(False);op.SetFuzzyValue(linear);op.SetToFillHistory(True);op.Build()
    if not op.IsDone():raise ValueError('Boolean failed at declared tolerance')
    return finish(op.Shape(),policy,linear,operation=operation,inputs=[a,b],history=operation_history(op,copies),require_solid=require_solid)


def split(shape,tools,*,policy=None):
    if not tools:raise ValueError('Split requires tools')
    policy=policy or TolerancePolicy();linear=policy.resolve([shape,*tools]);copies=[copy_shape(s) for s in [shape,*tools]]
    op=BRepAlgoAPI_Splitter();op.SetArguments(shape_list(copies[:1]));op.SetTools(shape_list(copies[1:]));op.SetFuzzyValue(linear)
    op.SetNonDestructive(True);op.SetRunParallel(False);op.SetToFillHistory(True);op.Build()
    if not op.IsDone():raise ValueError('Split failed at declared tolerance')
    return finish(op.Shape(),policy,linear,operation='split',inputs=[shape,*tools],history=operation_history(op,copies))


def section(a,b,*,policy=None):
    policy=policy or TolerancePolicy();linear=policy.resolve([a,b]);copies=[copy_shape(s) for s in [a,b]]
    op=BRepAlgoAPI_Section();op.Init1(copies[0]);op.Init2(copies[1]);op.SetFuzzyValue(linear);op.Approximation(True)
    op.ComputePCurveOn1(True);op.ComputePCurveOn2(True);op.Build()
    if not op.IsDone():raise ValueError('Section failed at declared tolerance')
    return finish(op.Shape(),policy,linear,operation='section',inputs=[a,b])


def sew(shapes,*,policy=None,solid=False,connected=True):
    if not shapes:raise ValueError('No shapes to sew')
    policy=policy or TolerancePolicy();linear=policy.resolve(shapes)
    op=BRepBuilderAPI_Sewing(linear);op.SetMinTolerance(min(1e-7,linear));op.SetMaxTolerance(policy.maximum_native_mm)
    for shape in shapes:op.Add(copy_shape(shape))
    op.Perform();result=op.SewedShape()
    if connected and len(shapes)>1 and len(subshapes(result,TopAbs_SHELL))!=1:
        raise ValueError('Sewing did not produce one connected shell at declared tolerance')
    if solid:
        shells=subshapes(result,TopAbs_SHELL)
        if len(shells)!=1:raise ValueError('Expected one shell for solid construction')
        result=BRepBuilderAPI_MakeSolid(shells[0]).Solid()
        if not BRepLib.OrientClosedSolid_s(result):raise ValueError('Sewn shell is not closed/orientable')
    return finish(result,policy,linear,operation='sew',inputs=shapes,preserve=True,require_solid=solid)


def heal(shape,*,policy=None):
    policy=policy or TolerancePolicy();linear=policy.resolve([shape]);copy=copy_shape(shape)
    op=ShapeFix_Shape(copy);op.SetPrecision(linear);op.SetMinTolerance(min(1e-7,linear));op.SetMaxTolerance(policy.maximum_native_mm);op.Perform()
    return finish(op.Shape(),policy,linear,operation='heal',inputs=[shape],preserve=True,
                  require_solid=bool(subshapes(shape,TopAbs_SOLID)))


def _cross(a,b,c):return float(np.cross(np.r_[b-a,0],np.r_[c-a,0])[2])


def _intersects(a,b,c,d):
    def on(a,b,p):return abs(_cross(a,b,p))<1e-12 and np.all(p>=np.minimum(a,b)-1e-12) and np.all(p<=np.maximum(a,b)+1e-12)
    return (_cross(a,b,c)*_cross(a,b,d)<0 and _cross(c,d,a)*_cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)))


def _inside(point,polygon):
    x,y=point;inside=False
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def trim(patch,outer,holes=(),*,policy=None):
    policy=policy or TolerancePolicy();loops=[]
    if len(holes)>8:raise ValueError('Too many trim holes')
    for values in [outer,*holes]:
        p=np.asarray(values,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=128 or not np.isfinite(p).all() or np.min(p)<0 or np.max(p)>1:
            raise ValueError('Expected bounded unit-domain trim polygons')
        if np.min(np.linalg.norm(p-np.roll(p,1,axis=0),axis=1))<1e-10:raise ValueError('Collapsed trim edge')
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if j==i+1 or (i==0 and j==len(p)-1):continue
                if _intersects(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]):raise ValueError('Self-intersecting trim loop')
        loops.append(p)
    for i,a in enumerate(loops):
        for b in loops[i+1:]:
            if any(_intersects(p,q,r,s) for p,q in zip(a,np.roll(a,-1,axis=0)) for r,s in zip(b,np.roll(b,-1,axis=0))):raise ValueError('Intersecting trim loops')
    for i,hole in enumerate(loops[1:]):
        if not all(_inside(p,loops[0]) for p in hole):raise ValueError('Hole is outside outer trim')
        if any(_inside(hole[0],other) or _inside(other[0],hole) for other in loops[1:i+1]):raise ValueError('Nested/overlapping trim holes')
    source=patch.native()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    support=BRepBuilderAPI_MakeFace(source,1e-7).Face();linear=policy.resolve([support])
    result=cad_ops.trim_surface(source,loops[0].tolist(),[p.tolist() for p in loops[1:]])
    return finish(result,policy,linear,operation='trim_uv',inputs=[support])


def export_step_checked(shape,path,*,policy=None):
    import hashlib
    from pathlib import Path
    from OCP.STEPControl import STEPControl_Writer,STEPControl_Reader,STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    policy=policy or TolerancePolicy();policy.resolve([shape]);before=inspect_shape(shape,policy)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Interface_Static.SetCVal_s('write.step.unit','MM')
    writer=STEPControl_Writer()
    if writer.Transfer(shape,STEPControl_AsIs)!=IFSelect_RetDone or writer.Write(str(path))!=IFSelect_RetDone:
        raise ValueError('STEP export failed')
    reader=STEPControl_Reader()
    if reader.ReadFile(str(path))!=IFSelect_RetDone or reader.TransferRoots()<1:raise ValueError('STEP reimport failed')
    reopened=reader.OneShape();after=inspect_shape(reopened,policy)
    same_faces=(sorted(before['faces_per_solid'])==sorted(after['faces_per_solid'])) if before['solids'] else before['faces']==after['faces']
    if not after['valid'] or after['nonmanifold_edges'] or not same_faces or before['solids']!=after['solids']:
        raise ValueError('STEP topology mismatch')
    if after['maximum_native_tolerance_mm']>policy.maximum_native_mm*(1+1e-10):raise ValueError('STEP native tolerance exceeds user budget')
    for field in ('area_mm2','solid_volume_mm3'):
        if abs(after[field]-before[field])>1e-7*max(1,abs(before[field])):raise ValueError('STEP measure mismatch')
    deformation=sampled_deformation(shape,reopened)
    if deformation['symmetric_sampled_max_mm']>policy.maximum_sampled_deformation_mm:raise ValueError('STEP sampled geometry mismatch')
    return {'pass':True,'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'before':before,'after':after,'deformation':deformation,'solidworks_import_performed':False,
            'cross_solid_topology_identity_preserved':before['faces']==after['faces'] and before['edges']==after['edges']}
