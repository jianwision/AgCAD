"""Compile polynomial seam/lock equalities into a coupled feasible subspace.

Only numerical structure is built here. Current control points are gathered and
projected on GPU; optimization does not download them between iterations.
"""
import math
import numpy as np
from scipy.interpolate import BSpline


def compile_network(layouts,seams,locks=None):
    if not 2<=len(layouts)<=6 or not isinstance(seams,list) or not seams or len(seams)>12:
        raise ValueError('Expected 2..6 polynomial Bezier patches and 1..12 seams')
    if any(not p.polynomial_bezier for p in layouts):raise ValueError('Coupled optimizer currently requires polynomial Bezier patches')
    sizes=[int(np.prod(p.poles.shape[:2])) for p in layouts];offsets=np.cumsum([0,*sizes]);n=int(offsets[-1])
    if n>384:raise ValueError('Network exceeds 384 control points')
    rows=[];targets=[]
    def line(patch,u,v,du,dv):
        p=layouts[patch];bu=BSpline(p.knots_u,np.eye(p.poles.shape[0]),p.degree_u)
        bv=BSpline(p.knots_v,np.eye(p.poles.shape[1]),p.degree_v)
        au=bu(u,nu=du) if du<=p.degree_u else np.zeros(p.poles.shape[0])
        av=bv(v,nu=dv) if dv<=p.degree_v else np.zeros(p.poles.shape[1])
        out=np.zeros(n);out[offsets[patch]:offsets[patch+1]]=np.outer(au,av).ravel();return out
    for seam in seams:
        if set(seam)-{'left','right','order','scale','shear','reverse','left_edge','right_edge'}:raise ValueError('Unsupported seam fields')
        a,b=seam['left'],seam['right'];order=seam.get('order',2);scale=seam.get('scale',1.);shear=seam.get('shear',0.);reverse=seam.get('reverse',False)
        if any(type(i) is not int or not 0<=i<len(layouts) for i in (a,b)) or a==b or type(order) is not int or not 0<=order<=3 or type(reverse) is not bool:
            raise ValueError('Invalid network seam')
        if not math.isfinite(scale) or not .05<=scale<=20 or not math.isfinite(shear) or abs(shear)>5:
            raise ValueError('Unsupported seam parameter map')
        ae=seam.get('left_edge','u1');be=seam.get('right_edge','u0')
        if ae not in ('u0','u1','v0','v1') or be not in ('u0','u1','v0','v1'):raise ValueError('Invalid seam boundary')
        def port(patch,edge,t,dr,dt,inward=False):
            sign=(1 if edge[1]=='1' else -1)*(-1 if inward else 1)
            return sign**dr*(line(patch,int(edge[1]),t,dr,dt) if edge[0]=='u' else line(patch,t,int(edge[1]),dt,dr))
        along_a=layouts[a].degree_v if ae[0]=='u' else layouts[a].degree_u
        along_b=layouts[b].degree_v if be[0]=='u' else layouts[b].degree_u
        for t in np.linspace(0,1,max(along_a,along_b)+1):
            for r in range(order+1):
                expected=sum(math.comb(r,k)*scale**(r-k)*shear**k*port(a,ae,1-t if reverse else t,r-k,k) for k in range(r+1))
                row=port(b,be,t,r,0,True)-expected
                norm=np.linalg.norm(row)
                if norm:rows.append(row/norm);targets.append(np.zeros(n))
    locks=[np.zeros(p.poles.shape[:2],dtype=bool) for p in layouts] if locks is None else locks
    if len(locks)!=len(layouts):raise ValueError('Lock count mismatch')
    locked=[]
    for i,(patch,mask) in enumerate(zip(layouts,locks)):
        mask=np.asarray(mask)
        if mask.dtype!=bool or mask.shape!=patch.poles.shape[:2]:raise ValueError('Expected boolean lock masks')
        for j in np.flatnonzero(mask.ravel()):
            row=np.zeros(n);row[offsets[i]+j]=1;rows.append(row);targets.append(row);locked.append(int(offsets[i]+j))
    C=np.array(rows);D=np.array(targets)
    u,s,vh=np.linalg.svd(C,full_matrices=True);threshold=max(C.shape)*s[0]*np.finfo(float).eps*32
    rank=int(np.sum(s>threshold));inverse=(vh[:rank].T/s[:rank])@u[:,:rank].T
    projection=np.eye(n)+inverse@(D-C);nullspace=vh[rank:].T
    residual=float(np.max(np.abs(C@projection-D)));null_error=float(np.max(np.abs(C@nullspace))) if nullspace.size else 0.
    if null_error>1e-8:raise ValueError('Numerically unstable seam/lock system')
    # Redundant locks can be compatible for a particular resident seed even when
    # C P != D as operators. Test that seed on GPU without downloading it.
    _,compat_s,compat_vh=np.linalg.svd(C@projection-D,full_matrices=False)
    compatibility=compat_vh[compat_s>1e-9]
    return {'projection':projection,'nullspace':nullspace,'constraint':C,'seed_target':D,'offsets':offsets,
            'compatibility':compatibility,
            'report':{'patches':len(layouts),'control_points':n,'constraint_rows':len(C),'constraint_rank':rank,
                      'free_modes_per_axis':n-rank,'projection_residual':residual,'nullspace_residual':null_error,
                      'locked_indices':locked,'dependent_constraint_rows':len(C)-rank,'seed_compatibility_rank':len(compatibility),
                      'corner_constraints_solved_jointly':True,'scope':'joint polynomial boundary-jet equalities on all four edges; fixed affine maps'}}


def network_seam_report(patches,seams,*,samples=67):
    if type(samples) is not int or not 3<=samples<=1025:raise ValueError('Invalid seam sampling budget')
    reports=[]
    for seam in seams:
        a,b=patches[seam['left']],patches[seam['right']];order=seam.get('order',2);scale=seam.get('scale',1.);shear=seam.get('shear',0.)
        ae=seam.get('left_edge','u1');be=seam.get('right_edge','u0');reverse=seam.get('reverse',False)
        def port(patch,edge,t,dr,dt,inward=False):
            sign=(1 if edge[1]=='1' else -1)*(-1 if inward else 1)
            return sign**dr*(patch.jet(int(edge[1]),t,dr,dt) if edge[0]=='u' else patch.jet(t,int(edge[1]),dt,dr))
        errors=np.zeros(order+1);angle=0.
        for t in np.r_[0.,(np.arange(samples)+.371)/(samples+.742),1.]:
            s=1-t if reverse else t
            for r in range(order+1):
                expected=sum(math.comb(r,k)*scale**(r-k)*shear**k*port(a,ae,s,r-k,k) for k in range(r+1))
                errors[r]=max(errors[r],np.linalg.norm(port(b,be,t,r,0,True)-expected))
            na=np.cross(port(a,ae,s,1,0),port(a,ae,s,0,1));nb=np.cross(port(b,be,t,1,0,True),port(b,be,t,0,1))
            if min(np.linalg.norm(na),np.linalg.norm(nb))<1e-12:raise ValueError('Singular seam tangent plane')
            na/=np.linalg.norm(na);nb/=np.linalg.norm(nb)
            angle=max(angle,math.atan2(np.linalg.norm(np.cross(na,nb)),abs(float(na@nb))))
        reports.append({'order':order,'jet_max_mm':errors.tolist(),'normal_angle_max_rad':angle,'samples':samples+2,'global_class_a_certified':False})
    return {'seams':reports,'maximum_jet_residual_mm':max(max(r['jet_max_mm']) for r in reports),'maximum_normal_angle_rad':max(r['normal_angle_max_rad'] for r in reports),'global_class_a_certified':False}
