"""Reusable CPU B-rep operations for the automotive baseline.

CUDA owns the source surface definitions; this module explicitly owns robust
topology. Native failures raise exceptions and never produce a success receipt.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.python-deps'))
from OCP.gp import gp_Pnt2d
from OCP.GCE2d import GCE2d_MakeSegment
from OCP.GeomProjLib import GeomProjLib
from OCP.GeomAPI import GeomAPI_IntSS
from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCP.GeomAbs import GeomAbs_C0
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace,BRepBuilderAPI_Sewing,BRepBuilderAPI_MakeSolid
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling,BRepOffsetAPI_MakePipe,BRepOffsetAPI_ThruSections
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_ListOfShape


def valid(shape):
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():raise ValueError('Invalid CAD result')
    return shape


def project_curve(curve,surface):
    projected=GeomProjLib.Project_s(curve,surface)
    if projected is None:raise ValueError('Normal projection failed')
    return projected


def intersect_surfaces(a,b,tolerance=1e-7):
    intersection=GeomAPI_IntSS(a,b,tolerance)
    if not intersection.IsDone():raise ValueError('Surface intersection failed')
    return [intersection.Line(i) for i in range(1,intersection.NbLines()+1)]


def _uv_wire(surface,points,ccw=True):
    pts=list(points)
    if len(pts)<3:raise ValueError('A trim loop needs at least three UV points')
    signed=sum(pts[i][0]*pts[(i+1)%len(pts)][1]-pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
    if abs(signed)<1e-14:raise ValueError('Degenerate UV trim loop')
    if (signed>0)!=ccw:pts.reverse()
    builder=BRepBuilderAPI_MakeWire()
    for i,a in enumerate(pts):
        b=pts[(i+1)%len(pts)]
        if a==b:raise ValueError('Duplicate UV trim vertex')
        segment=GCE2d_MakeSegment(gp_Pnt2d(*a),gp_Pnt2d(*b)).Value()
        builder.Add(BRepBuilderAPI_MakeEdge(segment,surface).Edge())
    if not builder.IsDone():raise ValueError('Unable to join UV trim edges')
    wire=builder.Wire();BRepLib.BuildCurves3d_s(wire)
    return wire


def trim_surface(surface,outer,holes=()):
    builder=BRepBuilderAPI_MakeFace(surface,_uv_wire(surface,outer),True)
    for hole in holes:builder.Add(_uv_wire(surface,hole,False))
    if not builder.IsDone():raise ValueError('UV trim failed')
    face=builder.Face();BRepLib.BuildCurves3d_s(face)
    return valid(face)


def split_surface(face,tools):
    if not tools:raise ValueError('At least one splitting tool required')
    arguments=TopTools_ListOfShape();arguments.Append(face)
    tool_shapes=TopTools_ListOfShape()
    for tool in tools:tool_shapes.Append(tool)
    splitter=BRepAlgoAPI_Splitter();splitter.SetArguments(arguments);splitter.SetTools(tool_shapes)
    splitter.Build()
    if not splitter.IsDone():raise ValueError('Surface split failed')
    return valid(splitter.Shape())


def join_curves(curves,tolerance=1e-7):
    if not curves:raise ValueError('No curves to join')
    builder=GeomConvert_CompCurveToBSplineCurve(curves[0])
    for curve in curves[1:]:
        if not builder.Add(curve,tolerance,True,True,0):raise ValueError('Curves do not meet at the requested tolerance')
    return builder.BSplineCurve()


def sew_surfaces(faces,tolerance=1e-7,solid=False):
    if not faces:raise ValueError('No faces to sew')
    sewing=BRepBuilderAPI_Sewing(tolerance)
    for face in faces:sewing.Add(face)
    sewing.Perform();shape=sewing.SewedShape()
    if solid:
        shells=TopExp_Explorer(shape,TopAbs_SHELL)
        if not shells.More():raise ValueError('Sewing did not form a shell')
        shell=TopoDS.Shell_s(shells.Current());shells.Next()
        if shells.More():raise ValueError('Expected one sewn shell')
        shape=BRepBuilderAPI_MakeSolid(shell).Solid()
        if not BRepLib.OrientClosedSolid_s(shape):raise ValueError('Sewn shell is not closed')
    return valid(shape)


def fill_nside(edges,orders=None,supports=None,tolerance=1e-7):
    if len(edges)<3:raise ValueError('At least three boundary edges required')
    filling=BRepOffsetAPI_MakeFilling()
    filling.SetConstrParam(tolerance/10,tolerance,1e-4,1e-3)
    filling.SetResolParam(4,25,4,False)
    filling.SetApproxParam(10,30)
    for i,edge in enumerate(edges):
        order=orders[i] if orders else GeomAbs_C0
        if supports and supports[i] is not None:filling.Add(edge,supports[i],order,True)
        else:filling.Add(edge,order,True)
    filling.Build()
    if not filling.IsDone():raise ValueError('N-sided fill failed')
    if filling.G0Error()>tolerance*10:raise ValueError('N-sided fill did not reach the boundary tolerance')
    return valid(filling.Shape())


def sweep(spine_wire,profile):
    builder=BRepOffsetAPI_MakePipe(spine_wire,profile)
    if not builder.IsDone():raise ValueError('Sweep failed')
    return valid(builder.Shape())


def loft(section_wires,solid=False,tolerance=1e-7):
    if len(section_wires)<2:raise ValueError('Loft requires at least two sections')
    builder=BRepOffsetAPI_ThruSections(solid,False,tolerance)
    builder.CheckCompatibility(True)
    for wire in section_wires:builder.AddWire(wire)
    builder.Build()
    if not builder.IsDone():raise ValueError('Loft failed')
    return valid(builder.Shape())
