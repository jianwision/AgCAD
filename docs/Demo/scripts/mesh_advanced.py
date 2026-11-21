"""Bounded CPU topology edits for indexed triangle meshes, NumPy positions.

These are explicit CPU backends; they are not silently advertised as CUDA.
Inputs are copied, indices stay local to the returned object, and unsupported
topology is rejected. General arbitrary self-intersection remeshing is excluded.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Mesh:
    p:np.ndarray
    t:np.ndarray
    def __post_init__(self):
        indices=np.asarray(self.t)
        if not np.all(np.isfinite(indices)) or np.any(indices!=np.floor(indices)):raise ValueError('Integer indices required')
        self.p=np.asarray(self.p,float).reshape(-1,3).copy();self.t=np.asarray(self.t,int).reshape(-1,3).copy()
        if len(self.p)>4096 or len(self.t)>8192:raise ValueError('CPU mesh budget exceeded (4096 vertices / 8192 triangles)')
        if not np.all(np.isfinite(self.p)) or len(self.t) and (self.t.min()<0 or self.t.max()>=len(self.p)):raise ValueError('Invalid mesh indices/positions')
        if len(self.t) and np.any(np.linalg.norm(np.cross(self.p[self.t[:,1]]-self.p[self.t[:,0]],self.p[self.t[:,2]]-self.p[self.t[:,0]]),axis=1)<1e-12):raise ValueError('Degenerate triangle')
    def edges(self):
        edges={}
        for k,tri in enumerate(self.t):
            for j in range(3):a,b=int(tri[j]),int(tri[(j+1)%3]);edges.setdefault(tuple(sorted((a,b))),[]).append((k,a,b,int(tri[(j+2)%3])))
        return edges
    def compact(self):
        ids=np.unique(self.t);lookup=np.full(len(self.p),-1,int);lookup[ids]=np.arange(len(ids));return Mesh(self.p[ids],lookup[self.t])
    def report(self):
        e=self.edges();boundary=sum(len(v)==1 for v in e.values());nonmanifold=sum(len(v)>2 for v in e.values());orientation=sum(len(v)==2 and v[0][1:3]!=v[1][2:0:-1] for v in e.values())
        p=self.p[self.t];area=float(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()/2)
        volume=float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
        return dict(vertices=len(self.p),triangles=len(self.t),boundary_edges=boundary,nonmanifold_edges=nonmanifold,orientation_errors=orientation,euler=len(np.unique(self.t))-len(e)+len(self.t),area=area,signed_volume=volume)
    def record(self):
        return {**self.report(),'positions':self.p.ravel().tolist(),'indices':self.t.ravel().tolist()}
def require_manifold(m,closed=False):
    r=m.report()
    if r['nonmanifold_edges'] or r['orientation_errors'] or closed and r['boundary_edges']:raise ValueError('Oriented manifold required')
def decimate(mesh,target_triangles,max_displacement):
    require_manifold(mesh,True)
    if not 4<=target_triangles<len(mesh.t) or max_displacement<=0:raise ValueError('Invalid collapse budget')
    m=Mesh(mesh.p,mesh.t);origins=[{i} for i in range(len(m.p))];original=mesh.p.copy();collapses=0
    while len(m.t)>target_triangles:
        edges=m.edges();neighbors=[set() for _ in m.p]
        for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
        candidates=sorted(edges,key=lambda e:float(np.linalg.norm(m.p[e[0]]-m.p[e[1]])))
        for a,b in candidates:
            edge=edges[a,b]
            if len(edge)!=2 or neighbors[a]&neighbors[b]!={v[3] for v in edge}:continue
            point=(m.p[a]+m.p[b])/2;source=origins[a]|origins[b]
            if np.max(np.linalg.norm(original[list(source)]-point,axis=1))>max_displacement:continue
            p=m.p.copy();p[a]=point;tri=m.t.copy();tri[tri==b]=a;keep=np.array([len(set(t))==3 for t in tri]);tri=tri[keep]
            old=m.p[m.t[keep]];new=p[tri];n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);n1=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
            if np.any(np.einsum('ij,ij->i',n0,n1)<=1e-12):continue
            try:candidate=Mesh(p,tri);require_manifold(candidate,True)
            except ValueError:continue
            m=candidate;origins[a]=source;origins[b]=set();collapses+=1;break
        else:raise ValueError('Target cannot be reached within collapse/topology/error constraints')
    displacement=max(np.max(np.linalg.norm(original[list(ids)]-m.p[i],axis=1)) for i,ids in enumerate(origins) if ids)
    return m.compact(),{'collapses':collapses,'max_source_vertex_displacement_mm':float(displacement),'surface_Hausdorff_bound':False}
def adaptive_remesh(mesh,max_edge,iterations=8):
    require_manifold(mesh)
    if max_edge<=0 or not 1<=iterations<=16:raise ValueError('Invalid remesh request')
    m=Mesh(mesh.p,mesh.t)
    for _ in range(iterations):
        marked={e for e in m.edges() if np.linalg.norm(m.p[e[0]]-m.p[e[1]])>max_edge*(1+1e-12)}
        if not marked:return m
        p=list(m.p);mid={e:len(p)+i for i,e in enumerate(sorted(marked))};p.extend((m.p[a]+m.p[b])/2 for a,b in sorted(marked));out=[]
        for tri in m.t:
            a,b,c=map(int,tri);ab=mid.get(tuple(sorted((a,b))));bc=mid.get(tuple(sorted((b,c))));ca=mid.get(tuple(sorted((c,a))));mask=[ab is not None,bc is not None,ca is not None];n=sum(mask)
            if n==0:out.append([a,b,c])
            elif n==3:out.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            elif n==1:
                if ab is not None:out.extend([[a,ab,c],[ab,b,c]])
                elif bc is not None:out.extend([[b,bc,a],[bc,c,a]])
                else:out.extend([[c,ca,b],[ca,a,b]])
            else:
                # Rotate so a is the common corner of the two split edges.
                if ab is None:a,b,c=c,a,b;ab,bc,ca=ca,ab,bc
                elif ca is None:a,b,c=b,c,a;ab,bc,ca=bc,ca,ab
                out.extend([[a,ab,ca],[ab,b,c],[ab,c,ca]])
        m=Mesh(p,out)
    raise ValueError('Remesh iteration budget exhausted')
def edge_flip(mesh,edge,tolerance=1e-9):
    require_manifold(mesh);a,b=sorted(map(int,edge));uses=mesh.edges().get((a,b),[])
    if len(uses)!=2:raise ValueError('Interior manifold edge required')
    k,a,b,c=uses[0];l,rb,ra,d=uses[1]
    if (ra,rb)!=(a,b) or tuple(sorted((c,d))) in mesh.edges():raise ValueError('Flip would duplicate an edge')
    normal=np.cross(mesh.p[b]-mesh.p[a],mesh.p[c]-mesh.p[a]);normal/=np.linalg.norm(normal)
    if abs(np.dot(mesh.p[d]-mesh.p[a],normal))>tolerance:raise ValueError('Planar flip only')
    t=mesh.t.copy();t[k]=[c,d,b];t[l]=[d,c,a]
    for tri in t[[k,l]]:
        if np.dot(np.cross(mesh.p[tri[1]]-mesh.p[tri[0]],mesh.p[tri[2]]-mesh.p[tri[0]]),normal)<=1e-12:raise ValueError('Nonconvex quadrilateral')
    out=Mesh(mesh.p,t);require_manifold(out);return out
def bridge_loops(mesh,loop_a,loop_b):
    a,b=list(map(int,loop_a)),list(map(int,loop_b))
    if len(a)!=len(b) or len(a)<3 or len(set(a+b))!=2*len(a) or min(a+b)<0 or max(a+b)>=len(mesh.p):raise ValueError('Disjoint equal-sized loops required')
    edges=mesh.edges();t=list(mesh.t)
    for j in range(len(a)):
        k=(j+1)%len(a)
        for x,y in [(a[j],a[k]),(b[k],b[j])]:
            uses=edges.get(tuple(sorted((x,y))),[])
            if uses and (len(uses)!=1 or uses[0][1:3]!=(y,x)):raise ValueError('Loop must oppose an existing boundary halfedge')
        t.extend([[a[j],a[k],b[k]],[a[j],b[k],b[j]]])
    result=Mesh(mesh.p,t);require_manifold(result);return result
def region_extrude(mesh,faces,vector):
    require_manifold(mesh);selected=set(map(int,faces));v=np.asarray(vector,float)
    if not selected or min(selected)<0 or max(selected)>=len(mesh.t) or v.shape!=(3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v)<1e-12:raise ValueError('Invalid extrusion selection/vector')
    p=list(mesh.p);region=mesh.t[sorted(selected)];ids=np.unique(region);mapping={int(i):len(p)+k for k,i in enumerate(ids)};p.extend(mesh.p[ids]+v);t=[tri.tolist() for k,tri in enumerate(mesh.t) if k not in selected]
    normals=np.cross(mesh.p[region[:,1]]-mesh.p[region[:,0]],mesh.p[region[:,2]]-mesh.p[region[:,0]])
    if np.any(normals@v<=1e-12):raise ValueError('Vector must point out of all selected faces')
    t.extend([[mapping[int(i)] for i in tri] for tri in region]);edges={}
    for tri in region:
        for j in range(3):a,b=map(int,(tri[j],tri[(j+1)%3]));edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    for uses in edges.values():
        if len(uses)==1:
            a,b=uses[0];t.extend([[a,b,mapping[b]],[a,mapping[b],mapping[a]]])
    out=Mesh(p,t).compact();require_manifold(out);return out
def crease_subdivide(mesh,creases):
    require_manifold(mesh);edges=mesh.edges();sharp={tuple(sorted(map(int,e))) for e in creases}
    if not sharp<=set(edges):raise ValueError('Unknown crease edge')
    sharp|={e for e,u in edges.items() if len(u)==1};neighbors=[set() for _ in mesh.p];feature=[set() for _ in mesh.p]
    for (a,b),uses in edges.items():
        neighbors[a].add(b);neighbors[b].add(a)
        if (a,b) in sharp:feature[a].add(b);feature[b].add(a)
    p=[]
    for i,q in enumerate(mesh.p):
        if len(feature[i])>=3:p.append(q)
        elif len(feature[i])==2:p.append(.75*q+.125*mesh.p[list(feature[i])].sum(axis=0))
        elif feature[i]:p.append(q)
        else:
            n=len(neighbors[i]);beta=(.625-(.375+.25*np.cos(2*np.pi/n))**2)/n;p.append((1-n*beta)*q+beta*mesh.p[list(neighbors[i])].sum(axis=0))
    mid={}
    for (a,b),uses in edges.items():
        mid[a,b]=len(p)
        p.append((mesh.p[a]+mesh.p[b])/2 if (a,b) in sharp else .375*(mesh.p[a]+mesh.p[b])+.125*(mesh.p[uses[0][3]]+mesh.p[uses[1][3]]))
    t=[]
    for a,b,c in mesh.t:
        ab=mid[tuple(sorted((a,b)))];bc=mid[tuple(sorted((b,c)))];ca=mid[tuple(sorted((c,a)))];t.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
    out=Mesh(p,t);require_manifold(out);return out

def self_intersections(mesh,tolerance=1e-9):
    # Nonadjacent triangle pairs; coplanar overlap is deliberately rejected as
    # unsupported instead of being reported collision-free.
    if len(mesh.t)>1024 or tolerance<=0:raise ValueError('Crossing check supports <=1024 triangles and positive tolerance')
    hits=[]
    def segment_triangle(a,b,tri):
        e1,e2=tri[1]-tri[0],tri[2]-tri[0];direction=b-a;h=np.cross(direction,e2);det=np.dot(e1,h)
        if abs(det)<tolerance:return False
        inv=1/det;s=a-tri[0];u=inv*np.dot(s,h);q=np.cross(s,e1);v=inv*np.dot(direction,q);t=inv*np.dot(e2,q)
        return -tolerance<=u<=1+tolerance and -tolerance<=v and u+v<=1+tolerance and tolerance<t<1-tolerance
    for i in range(len(mesh.t)):
        a=mesh.p[mesh.t[i]]
        for j in range(i+1,len(mesh.t)):
            if set(mesh.t[i])&set(mesh.t[j]):continue
            b=mesh.p[mesh.t[j]]
            if np.any(a.max(axis=0)<b.min(axis=0)-tolerance) or np.any(b.max(axis=0)<a.min(axis=0)-tolerance):continue
            na=np.cross(a[1]-a[0],a[2]-a[0]);na/=np.linalg.norm(na)
            if np.max(np.abs((b-a[0])@na))<tolerance:raise ValueError('Coplanar overlap needs a planar arrangement backend')
            if any(segment_triangle(tri[k],tri[(k+1)%3],other) for tri,other in [(a,b),(b,a)] for k in range(3)):hits.append((i,j))
    return hits
def repair_self_intersections(mesh,discard_faces):
    # Explicit destructive repair policy: remove user-selected crossing faces,
    # preserve remaining geometry and report open boundaries. No auto-closed claim.
    hits=self_intersections(mesh);discard=set(map(int,discard_faces))
    if not hits or not discard or not discard<=set(range(len(mesh.t))) or any(not ({a,b}&discard) for a,b in hits):raise ValueError('Selection must cover every detected crossing')
    result=Mesh(mesh.p,[t for i,t in enumerate(mesh.t) if i not in discard]).compact()
    if self_intersections(result):raise ValueError('Crossings remain')
    return result,{'removed_faces':sorted(discard),'original_crossing_pairs':hits,'closed_repair':False}
def repair_nonmanifold(mesh):
    # Split disconnected/oversubscribed vertex fans. Face geometry is unchanged;
    # cuts become explicit boundaries, preserving oriented manifold components.
    p=list(mesh.p);t=mesh.t.copy();edges=mesh.edges();cuts=0
    for vertex in range(len(mesh.p)):
        faces=set(np.flatnonzero(np.any(mesh.t==vertex,axis=1)).tolist());adj={f:set() for f in faces}
        for edge,uses in edges.items():
            if vertex in edge and len(uses)==2 and uses[0][1:3]==uses[1][2:0:-1]:
                a,b=uses[0][0],uses[1][0];adj[a].add(b);adj[b].add(a)
        first=True
        while faces:
            component=set();stack=[min(faces)]
            while stack:
                f=stack.pop()
                if f in component:continue
                component.add(f);stack.extend(adj[f]-component)
            faces-=component
            if first:first=False;continue
            replacement=len(p);p.append(mesh.p[vertex]);cuts+=1
            for f in component:t[f,t[f]==vertex]=replacement
    out=Mesh(p,t);require_manifold(out)
    return out,{'duplicated_vertices':cuts,'geometry_preserved':True,'cuts_create_boundaries':True}
