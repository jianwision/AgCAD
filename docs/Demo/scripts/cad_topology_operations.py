"""Explicit CPU B-rep Boolean/delete/cap/revolve primitives."""
from cad_ops import valid
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut,BRepAlgoAPI_Common
from OCP.BRepTools import BRepTools_ReShape
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.gp import gp_Ax1,gp_Pnt,gp_Dir


def boolean_union(a,b):return _boolean(BRepAlgoAPI_Fuse,a,b)
def boolean_cut(a,b):return _boolean(BRepAlgoAPI_Cut,a,b)
def boolean_common(a,b):return _boolean(BRepAlgoAPI_Common,a,b)


def _boolean(builder,a,b):
    valid(a);valid(b);operation=builder(a,b);operation.SetRunParallel(False);operation.Build()
    if not operation.IsDone():raise ValueError('CAD Boolean failed')
    return valid(operation.Shape())


def delete_cad_faces(shell,faces):
    if not faces:raise ValueError('No faces selected')
    reshape=BRepTools_ReShape()
    for face in faces:reshape.Remove(face)
    return valid(reshape.Apply(shell))


def cap_planar_wire(wire):
    cap=BRepBuilderAPI_MakeFace(wire,True)
    if not cap.IsDone():raise ValueError('Unable to cap planar wire')
    return valid(cap.Face())


def revolve_face(face,origin=(0,0,0),axis=(0,0,1),angle=6.283185307179586):
    valid(face);operation=BRepPrimAPI_MakeRevol(face,gp_Ax1(gp_Pnt(*origin),gp_Dir(*axis)),angle,True)
    if not operation.IsDone():raise ValueError('Profile revolution failed')
    return valid(operation.Shape())
