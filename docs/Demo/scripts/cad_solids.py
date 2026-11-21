"""Exact closed CAD solids from the GPU's typed definition records.

GPU owns parameters/frame and preview geometry; OCCT owns B-rep topology.
This module never converts a sampled sphere/torus mesh into a claimed analytic solid.
"""
import math
from cad_ops import valid, sew_surfaces
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Dir, gp_Trsf, gp_GTrsf
from OCP.GC import GC_MakeArcOfCircle
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform


def polygon_face(points):
    wire=BRepBuilderAPI_MakePolygon()
    for p in points:wire.Add(gp_Pnt(*p))
    wire.Close()
    if not wire.IsDone():raise ValueError('Unable to close polygon')
    return valid(BRepBuilderAPI_MakeFace(wire.Wire(),True).Face())


def capsule(radius,height):
    r=radius;h=height;q=r/math.sqrt(2)
    bottom=GC_MakeArcOfCircle(gp_Pnt(0,0,-r),gp_Pnt(q,0,-q),gp_Pnt(r,0,0)).Value()
    top=GC_MakeArcOfCircle(gp_Pnt(r,0,h),gp_Pnt(q,0,h+q),gp_Pnt(0,0,h+r)).Value()
    wire=BRepBuilderAPI_MakeWire()
    wire.Add(BRepBuilderAPI_MakeEdge(bottom).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(r,0,0),gp_Pnt(r,0,h)).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(top).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0,0,h+r),gp_Pnt(0,0,-r)).Edge())
    face=BRepBuilderAPI_MakeFace(wire.Wire(),True).Face()
    return valid(BRepPrimAPI_MakeRevol(face,gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1))).Shape())


def solid_from_definition(s):
    if len(s)!=64 or not all(math.isfinite(x) for x in s):raise ValueError('Invalid solid record')
    kind=int(s[0]);a,b,c,d=s[1:5]
    if s[0]!=kind or not 0<=kind<=9 or s[20]!=0 or a<=0:raise ValueError('Invalid solid kind/status/size')
    if kind in [1,2,3,4,5,6,7,8,9] and b<=0:raise ValueError('Invalid second size')
    if kind in [1,4,9] and c<=0:raise ValueError('Invalid third size')
    if kind in [7,8] and (d!=int(d) or not 3<=d<=64):raise ValueError('Invalid side count')
    if kind==5 and a<=b:raise ValueError('Self-intersecting torus is outside this API')
    if kind==9 and not 0<d<a:raise ValueError('Invalid wedge top width')
    if kind==0:shape=BRepPrimAPI_MakeSphere(a).Shape()
    if kind==1:
        transform=gp_GTrsf()
        for i,v in enumerate([a,b,c],1):transform.SetValue(i,i,v)
        shape=BRepBuilderAPI_GTransform(BRepPrimAPI_MakeSphere(1.).Shape(),transform,True).Shape()
    if kind==2:shape=BRepPrimAPI_MakeCylinder(a,b).Shape()
    if kind==3:shape=BRepPrimAPI_MakeCone(a,0.,b).Shape()
    if kind==4:shape=(BRepPrimAPI_MakeCylinder(a,c) if a==b else BRepPrimAPI_MakeCone(a,b,c)).Shape()
    if kind==5:shape=BRepPrimAPI_MakeTorus(a,b).Shape()
    if kind==6:shape=capsule(a,b)
    if kind in [7,8]:
        n=int(d);points=[(a*math.cos(2*math.pi*i/n),a*math.sin(2*math.pi*i/n),0.) for i in range(n)]
        if kind==7:shape=BRepPrimAPI_MakePrism(polygon_face(points),gp_Vec(0,0,b)).Shape()
        else:
            faces=[polygon_face(list(reversed(points)))]+[polygon_face([points[i],points[(i+1)%n],(0.,0.,b)]) for i in range(n)]
            shape=sew_surfaces(faces,solid=True)
    if kind==9:
        points=[(((d if i&4 else a) if i&1 else 0.),b if i&2 else 0.,c if i&4 else 0.) for i in range(8)]
        faces=[(0,2,3,1),(4,5,7,6),(0,4,6,2),(1,3,7,5),(0,1,5,4),(2,6,7,3)]
        shape=sew_surfaces([polygon_face([points[i] for i in face]) for face in faces],solid=True)
    axes=[s[11:14],s[14:17],s[17:20]];origin=s[8:11]
    for i in range(3):
        for j in range(3):
            dot=sum(axes[i][k]*axes[j][k] for k in range(3))
            if abs(dot-(i==j))>1e-10:raise ValueError('Solid frame must be orthonormal')
    det=(axes[0][1]*axes[1][2]-axes[0][2]*axes[1][1])*axes[2][0]+(axes[0][2]*axes[1][0]-axes[0][0]*axes[1][2])*axes[2][1]+(axes[0][0]*axes[1][1]-axes[0][1]*axes[1][0])*axes[2][2]
    if det<1-1e-10:raise ValueError('Solid frame must be right-handed')
    placement=gp_Trsf();placement.SetValues(*[v for i in range(3) for v in [axes[0][i],axes[1][i],axes[2][i],origin[i]]])
    return valid(BRepBuilderAPI_Transform(shape,placement,True).Shape())
