"""CPU topology adapters for GPU placement primitives and CAD snapping."""
import math
import numpy as np
from cad_ops import valid
from mesh_advanced import Mesh
from OCP.gp import gp_Pnt,gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,BRepBuilderAPI_Transform
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

def snap_point(point,target=None,step=None,origin=(0,0,0),max_distance=float('inf')):
    p=np.asarray(point,float);o=np.asarray(origin,float)
    if p.shape!=(3,) or o.shape!=(3,) or not np.all(np.isfinite([p,o])) or max_distance<0:raise ValueError('Invalid point/origin/radius')
    if step is not None:
        if target is not None or not math.isfinite(step) or step<=0:raise ValueError('Choose one grid or CAD target')
        result=o+step*np.floor((p-o)/step+.5);mode='grid'
    else:
        valid(target);query=BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex(),target);query.Perform()
        if not query.IsDone() or query.NbSolution()<1:raise ValueError('No nearest CAD solution')
        q=query.PointOnShape2(1);result=np.array([q.X(),q.Y(),q.Z()]);mode='native CAD point/edge/trimmed face'
    distance=float(np.linalg.norm(result-p))
    if distance>max_distance:raise ValueError('Snap exceeds capture radius')
    return result,{'mode':mode,'distance_mm':distance,'tie_policy':'first native solution or grid half towards positive infinity'}
def merge_mesh(meshes,tolerance=0.):
    if len(meshes)<2 or tolerance<0 or not math.isfinite(tolerance):raise ValueError('At least two meshes and finite nonnegative tolerance required')
    positions=[];triangles=[];offset=0
    for mesh in meshes:
        positions.extend(mesh.p);triangles.extend(mesh.t+offset);offset+=len(mesh.p)
    if not tolerance:return Mesh(positions,triangles),{'mode':'combine indexed objects','welded_vertices':0}
    representatives=[];mapping=[]
    for p in positions:
        index=next((i for i,q in enumerate(representatives) if np.linalg.norm(p-q)<=tolerance),len(representatives))
        if index==len(representatives):representatives.append(p)
        mapping.append(index)
    ids=np.array(mapping)[triangles];out=Mesh(representatives,ids)
    return out,{'mode':'combine and first-representative weld','welded_vertices':len(positions)-len(representatives),'does_not_remove_internal_faces':True}
def place_frame(shape,position,tangent,normal):
    p=np.asarray(position,float);x=np.asarray(tangent,float);z=np.asarray(normal,float)
    if not np.all(np.isfinite([p,x,z])) or np.linalg.norm(x)<1e-12 or np.linalg.norm(z)<1e-12:raise ValueError('Finite nonzero placement frame required')
    x=x/np.linalg.norm(x);z=z-x*np.dot(x,z)
    if np.linalg.norm(z)<1e-12:raise ValueError('Normal must not be parallel to tangent')
    z/=np.linalg.norm(z);y=np.cross(z,x);r=np.column_stack([x,y,z]);transform=gp_Trsf();transform.SetValues(*np.column_stack([r,p]).ravel())
    return valid(BRepBuilderAPI_Transform(shape,transform,True).Shape())
