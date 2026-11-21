"""Clamped positive-weight tensor B-splines, basis compilation and refinement.

CPU basis compilation is cached by the resident client. Repeated rational jets,
curvature and network objective evaluation run in CUDA.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.interpolate import BSpline

JETS=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3))


def _knots(values,n,degree):
    k=np.array(values,dtype=float,copy=True)
    if type(degree) is not int or not 1<=degree<=7 or not degree+1<=n<=32:
        raise ValueError('Expected degree 1..7 and at most 32 poles per direction')
    if k.shape!=(n+degree+1,) or not np.isfinite(k).all() or np.any(np.diff(k)<0):
        raise ValueError('Invalid full knot vector')
    if not np.all(k[:degree+1]==0) or not np.all(k[-degree-1:]==1):
        raise ValueError('Expected clamped unit-domain knots')
    internal=k[degree+1:-degree-1]
    if len(internal) and (np.any(internal<=0) or np.any(internal>=1) or max(np.unique(internal,return_counts=True)[1])>degree):
        raise ValueError('Invalid interior knot multiplicity')
    k.flags.writeable=False
    return k


@dataclass(frozen=True,eq=False)
class NurbsPatch:
    poles:np.ndarray
    weights:np.ndarray
    knots_u:np.ndarray
    knots_v:np.ndarray
    degree_u:int
    degree_v:int

    def __post_init__(self):
        p=np.array(self.poles,dtype=float,copy=True);w=np.array(self.weights,dtype=float,copy=True)
        if p.ndim!=3 or p.shape[2]!=3 or not np.isfinite(p).all() or np.max(np.abs(p))>1e6:
            raise ValueError('Invalid surface control points')
        if w.shape!=p.shape[:2] or not np.isfinite(w).all() or np.any(w<1e-4) or np.any(w>1e4):
            raise ValueError('Expected positive weights in [1e-4,1e4]')
        ku=_knots(self.knots_u,len(p),self.degree_u);kv=_knots(self.knots_v,p.shape[1],self.degree_v)
        p.flags.writeable=False;w.flags.writeable=False
        for name,value in (('poles',p),('weights',w),('knots_u',ku),('knots_v',kv)):object.__setattr__(self,name,value)

    @property
    def degrees(self):return self.degree_u,self.degree_v

    @classmethod
    def from_bezier(cls,patch):
        p,q=patch.degrees
        return cls(patch.poles,np.ones(patch.poles.shape[:2]),[0]*(p+1)+[1]*(p+1),[0]*(q+1)+[1]*(q+1),p,q)

    @property
    def polynomial_bezier(self):
        return self.poles.shape[:2]==(self.degree_u+1,self.degree_v+1) and np.all(self.weights==self.weights.flat[0])

    def descriptor(self):
        return {'shape':list(self.poles.shape[:2]),'degrees':list(self.degrees),
                'knots_u':self.knots_u.tolist(),'knots_v':self.knots_v.tolist(),'weights':self.weights.tolist()}

    def jet(self,u,v,du=0,dv=0):
        if type(du) is not int or type(dv) is not int or du<0 or dv<0 or du+dv>3 or not all(math.isfinite(t) and 0<=t<=1 for t in (u,v)):
            raise ValueError('Invalid surface jet request')
        bu=BSpline(self.knots_u,np.eye(len(self.poles)),self.degree_u)
        bv=BSpline(self.knots_v,np.eye(self.poles.shape[1]),self.degree_v)
        homogeneous=np.concatenate((self.poles*self.weights[:,:,None],self.weights[:,:,None]),axis=2)
        raw={};result={}
        for a,b in JETS:
            au=bu(u,nu=a) if a<=self.degree_u else np.zeros(len(self.poles))
            av=bv(v,nu=b) if b<=self.degree_v else np.zeros(self.poles.shape[1])
            raw[a,b]=np.einsum('i,j,ijc->c',au,av,homogeneous)
            r=raw[a,b][:3].copy()
            for i in range(a+1):
                for j in range(b+1):
                    if i or j:r-=math.comb(a,i)*math.comb(b,j)*raw[i,j][3]*result[a-i,b-j]
            result[a,b]=r/raw[0,0][3]
        return result[du,dv]

    def insert_knot(self,axis,value,multiplicity=1):
        if axis not in (0,1) or type(multiplicity) is not int or not 1<=multiplicity<=7 or not math.isfinite(value) or not 0<value<1:
            raise ValueError('Invalid knot insertion')
        knots=(self.knots_u,self.knots_v)[axis];degree=self.degrees[axis]
        if np.count_nonzero(knots==value)+multiplicity>degree:raise ValueError('Interior multiplicity exceeds degree')
        if self.poles.shape[axis]+multiplicity>32:raise ValueError('Refinement exceeds pole capacity')
        h=np.concatenate((self.poles*self.weights[:,:,None],self.weights[:,:,None]),axis=2)
        spline=BSpline(knots,h,degree,axis=axis).insert_knot(value,multiplicity)
        # SciPy stores the interpolation axis first internally.
        refined=np.moveaxis(spline.c,0,axis)
        ku,kv=(spline.t,self.knots_v) if axis==0 else (self.knots_u,spline.t)
        return NurbsPatch(refined[:,:,:3]/refined[:,:,3,None],refined[:,:,3],ku,kv,*self.degrees)

    def native(self):
        import cad_ops
        from OCP.Geom import Geom_BSplineSurface
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.TColStd import TColStd_Array2OfReal,TColStd_Array1OfReal,TColStd_Array1OfInteger
        from OCP.gp import gp_Pnt
        nu,nv=self.poles.shape[:2];p=TColgp_Array2OfPnt(1,nu,1,nv);w=TColStd_Array2OfReal(1,nu,1,nv)
        for i in range(nu):
            for j in range(nv):p.SetValue(i+1,j+1,gp_Pnt(*self.poles[i,j]));w.SetValue(i+1,j+1,float(self.weights[i,j]))
        arrays=[]
        for knots in (self.knots_u,self.knots_v):
            unique,counts=np.unique(knots,return_counts=True);k=TColStd_Array1OfReal(1,len(unique));m=TColStd_Array1OfInteger(1,len(unique))
            for i,(a,b) in enumerate(zip(unique,counts)):k.SetValue(i+1,float(a));m.SetValue(i+1,int(b))
            arrays.extend((k,m))
        return Geom_BSplineSurface(p,w,arrays[0],arrays[2],arrays[1],arrays[3],self.degree_u,self.degree_v,False,False)


def basis_table(surface,uv,quadrature=None):
    """Each row: local support starts, 4x8 U/V derivatives, area quadrature."""
    uv=np.asarray(uv,dtype=float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not 1<=len(uv)<=4225 or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1):
        raise ValueError('Expected 1..4225 UV samples in unit domain')
    result=np.zeros((len(uv),67));result[:,-1]=1/len(uv) if quadrature is None else quadrature
    for axis,(knots,degree,n) in enumerate(zip((surface.knots_u,surface.knots_v),surface.degrees,surface.poles.shape[:2])):
        start=np.clip(np.searchsorted(knots,uv[:,axis],side='right')-1,degree,n-1)-degree
        result[:,axis]=start;bs=BSpline(knots,np.eye(n),degree)
        for d in range(4):
            if d<=degree:
                values=bs(uv[:,axis],nu=d)
                result[:,2+axis*32+d*8:2+axis*32+d*8+degree+1]=values[np.arange(len(uv))[:,None],start[:,None]+np.arange(degree+1)]
    return result


def grid_uv(samples):
    if type(samples) is not int or not 3<=samples<=65:raise ValueError('Expected 3..65 grid samples')
    t=np.linspace(0,1,samples);u,v=np.meshgrid(t,t,indexing='ij')
    w=np.ones(samples)/(samples-1);w[[0,-1]]*=.5
    return np.column_stack((u.ravel(),v.ravel())),np.outer(w,w).ravel()
