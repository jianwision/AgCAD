"""Polynomial Bezier surfacing services; explicit numerical scope, no Class-A certificate."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
JETS=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3))
RECORD_WORDS=4360


@dataclass(frozen=True)
class BezierPatch:
    poles:np.ndarray

    def __post_init__(self):
        p=np.array(self.poles,dtype=float,copy=True)
        if p.ndim!=3 or p.shape[2]!=3 or not all(4<=n<=8 for n in p.shape[:2]) or not np.isfinite(p).all() or np.max(np.abs(p))>1e6:
            raise ValueError('Expected finite 4..8 by 4..8 nonrational Bezier control points, mm')
        p.flags.writeable=False;object.__setattr__(self,'poles',p)

    @property
    def degrees(self):return self.poles.shape[0]-1,self.poles.shape[1]-1

    def jet(self,u,v,du=0,dv=0):
        if not all(math.isfinite(t) and 0<=t<=1 for t in (u,v)) or du<0 or dv<0 or du+dv>3:
            raise ValueError('Expected unit-domain parameters and derivative total order <=3')
        p=self.poles
        for axis,times in ((0,du),(1,dv)):
            for _ in range(times):p=np.diff(p,axis=axis)*(p.shape[axis]-1)
        def basis(n,t):return np.array([math.comb(n,i)*t**i*(1-t)**(n-i) for i in range(n+1)])
        return np.einsum('i,j,ijc->c',basis(p.shape[0]-1,u),basis(p.shape[1]-1,v),p)

    def native(self):
        import cad_ops  # initializes the project-local OCP path
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.Geom import Geom_BezierSurface
        from OCP.gp import gp_Pnt
        a=TColgp_Array2OfPnt(1,self.poles.shape[0],1,self.poles.shape[1])
        for i,row in enumerate(self.poles):
            for j,p in enumerate(row):a.SetValue(i+1,j+1,gp_Pnt(*p))
        return Geom_BezierSurface(a)


def elevate_curve(poles,degree):
    p=np.asarray(poles,dtype=float)
    while len(p)-1<degree:
        n=len(p)-1;q=np.empty((n+2,3));q[0]=p[0];q[-1]=p[-1]
        for i in range(1,n+1):q[i]=i/(n+1)*p[i-1]+(1-i/(n+1))*p[i]
        p=q
    return p


def match_boundary(left,right,*,order=2,scale=1.,shear=0.,reverse=False):
    """Match B(u,v) jets to A(1+scale*u, (1-v if reverse else v)+shear*u).

    Positive scale and constant shear define an invertible local parameter map.
    This covers a useful G1/G2/G3 subset, not arbitrary nonlinear matching.
    """
    if type(order) is not int or not 0<=order<=3 or not isinstance(reverse,bool):raise ValueError('Invalid continuity order/orientation')
    if not math.isfinite(scale) or not .05<=scale<=20 or not math.isfinite(shear) or abs(shear)>5:raise ValueError('Unsupported seam parameter map')
    n,m=right.degrees
    if right.degrees[1]<left.degrees[1]:raise ValueError('Destination V degree must cover source boundary degree')
    q=right.poles.copy();rows=[]
    for r in range(order+1):
        jet=np.zeros((m+1,3))
        for k in range(r+1):
            p=left.poles
            for _ in range(r-k):p=np.diff(p,axis=0)*(p.shape[0]-1)
            for _ in range(k):p=np.diff(p,axis=1)*(p.shape[1]-1)
            line=elevate_curve(p[-1],m)
            jet+=math.comb(r,k)*scale**(r-k)*shear**k*line
        if reverse:jet=jet[::-1]
        divisor=math.prod(range(n-r+1,n+1)) if r else 1
        q[r]=jet/divisor
        for k in range(r):q[r]-=(-1)**(r-k)*math.comb(r,k)*q[k]
        rows.append(jet)
    return BezierPatch(q)


def seam_report(left,right,*,order=2,scale=1.,shear=0.,reverse=False,samples=97):
    # Independent sampled evaluation of the parameter-map chain rule.
    if type(order) is not int or not 0<=order<=3 or not 3<=samples<=1025:raise ValueError('Invalid seam analysis settings')
    errors=np.zeros(order+1);angle=0.;curvature=0.
    for t in (np.arange(samples)+.371)/(samples+.742):
        s=1-t if reverse else t
        for r in range(order+1):
            expected=sum(math.comb(r,k)*scale**(r-k)*shear**k*left.jet(1,s,r-k,k) for k in range(r+1))
            errors[r]=max(errors[r],np.linalg.norm(right.jet(0,t,r,0)-expected))
        au,av=left.jet(1,s,1),left.jet(1,s,0,1);bu,bv=right.jet(0,t,1),right.jet(0,t,0,1)
        na=np.cross(au,av);nb=np.cross(bu,bv)
        if min(np.linalg.norm(na),np.linalg.norm(nb))<1e-12:raise ValueError('Singular seam tangent plane')
        na/=np.linalg.norm(na);nb/=np.linalg.norm(nb)
        if na@nb<0:nb=-nb
        angle=max(angle,math.atan2(np.linalg.norm(np.cross(na,nb)),float(na@nb)))
        tangent=scale*au+shear*av
        d2=scale**2*left.jet(1,s,2)+2*scale*shear*left.jet(1,s,1,1)+shear**2*left.jet(1,s,0,2)
        ka=na@d2/(tangent@tangent);kb=nb@right.jet(0,t,2)/(bu@bu)
        curvature=max(curvature,abs(ka-kb))
    return {'scope':'sampled affine-reparameterized boundary jets','samples':samples,'order':order,
            'jet_max_mm':errors.tolist(),'normal_angle_max_rad':angle,
            'transverse_normal_curvature_gap_per_mm':float(curvature),'global_class_a_certified':False}


def fairness_operator(shape):
    nu,nv=shape;rows=[]
    for axis in (0,1):
        for i in range(nu-(2 if axis==0 else 0)):
            for j in range(nv-(2 if axis==1 else 0)):
                row=np.zeros(64)
                for k,c in enumerate((1,-2,1)):row[8*(i+k if axis==0 else i)+(j+k if axis==1 else j)]=c
                rows.append(row)
    for i in range(nu-1):
        for j in range(nv-1):
            row=np.zeros(64)
            for di,dj,c in ((0,0,1),(1,0,-1),(0,1,-1),(1,1,1)):row[8*(i+di)+j+dj]=math.sqrt(2)*c
            rows.append(row)
    d=np.array(rows);return d.T@d


def fairness_energy(patch):
    p=np.zeros((8,8,3));p[:patch.poles.shape[0],:patch.poles.shape[1]]=patch.poles
    return float(np.sum(p.reshape(64,3)*(fairness_operator(patch.poles.shape[:2])@p.reshape(64,3))))


def assess_quality(report,criteria):
    allowed={'require_sampled_regular','maximum_abs_curvature_per_mm','maximum_fairness_energy'}
    if set(criteria)-allowed:raise ValueError('Unsupported surface quality criterion')
    checks=[]
    for name,limit in criteria.items():
        if name=='require_sampled_regular':
            if not isinstance(limit,bool):raise ValueError('Expected regularity requirement flag')
            checks.append({'criterion':name,'pass':not limit or report['all_samples_regular']})
        else:
            if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<0:raise ValueError('Invalid quality tolerance')
            actual=max(map(abs,report['principal_curvature_range_per_mm'])) if name=='maximum_abs_curvature_per_mm' else report['fairness_energy_after']
            checks.append({'criterion':name,'actual':actual,'limit':limit,'pass':actual<=limit})
    return {'pass':all(c['pass'] for c in checks),'checks':checks,'evidence':'sampled differential analysis','class_a_certified':False}


def gpu_batch(patches,directory,*,samples=17,fairing=None):
    if not 1<=len(patches)<=32 or type(samples) is not int or not 5<=samples<=65:raise ValueError('GPU batch capacity exceeded')
    fairing=[{} for _ in patches] if fairing is None else fairing
    if len(fairing)!=len(patches):raise ValueError('Fairing specification count mismatch')
    records=[]
    for patch,settings in zip(patches,fairing):
        nu,nv=patch.poles.shape[:2];iterations=settings.get('iterations',0);budget=settings.get('max_displacement_mm',0.)
        if type(iterations) is not int or not 0<=iterations<=512 or not math.isfinite(budget) or not 0<=budget<=100:
            raise ValueError('Invalid fairing budget')
        locked=settings.get('locked',np.zeros((nu,nv),dtype=bool));locked=np.asarray(locked)
        if locked.shape!=(nu,nv) or locked.dtype!=bool:raise ValueError('Expected boolean locked-pole mask')
        p=np.zeros((8,8,3));p[:nu,:nv]=patch.poles
        mask=np.ones((8,8));mask[:nu,:nv]=locked
        h=fairness_operator((nu,nv));step=.4/max(1,np.max(np.sum(np.abs(h),axis=1)))
        records.extend([np.array([nu,nv,iterations,budget,step,0,0,0]),p.ravel(),mask.ravel(),h.ravel()])
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    header=np.array([20261121,1,len(patches),samples]+[0]*12,dtype='<f8')
    np.concatenate([header,*records]).astype('<f8').tofile(directory/'batch.bin')
    result=subprocess.run([str(ROOT/'target/release/class_a_suite.exe'),str(directory)],cwd=ROOT,capture_output=True,text=True,timeout=120)
    (directory/'worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Class-A CUDA batch failed: '+result.stderr[-2000:])
    raw=np.fromfile(directory/'results.bin',dtype='<f8').reshape(len(patches),192+48*samples*samples)
    outputs=[]
    for original,row in zip(patches,raw):
        nu,nv=original.poles.shape[:2];p=BezierPatch(row[:192].reshape(8,8,3)[:nu,:nv])
        grid=row[192:].reshape(samples,samples,48)
        outputs.append({'patch':p,'grid':grid,'report':{'scope':'sampled polynomial Bezier diagnostics',
            'samples':samples*samples,'regular_samples':int(np.sum(grid[:,:,42])),
            'all_samples_regular':bool(np.all(grid[:,:,42]==1)),
            'principal_curvature_range_per_mm':[float(np.min(grid[:,:,36])),float(np.max(grid[:,:,37]))],
            'fairness_energy_before':fairness_energy(original),'fairness_energy_after':fairness_energy(p),
            'max_control_displacement_mm':float(np.max(np.linalg.norm(p.poles-original.poles,axis=2))),
            'global_class_a_certified':False}})
    return outputs,json.loads((directory/'gpu-report.json').read_text())
