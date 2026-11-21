"""Analytic polynomial surfaces and user-owned intent fixtures for acceptance."""
import math
import numpy as np
from class_a import BezierPatch


def polynomial_patch(degree=5,*,x0=0.,width=100.,depth=60.,bump=0.):
    # Exact degree elevation of x=x0+width*u, y=depth*v, z=4u²+3v²+2uv.
    p=np.zeros((degree+1,degree+1,3))
    for i in range(degree+1):
        for j in range(degree+1):
            u,v=i/degree,j/degree
            p[i,j]=[x0+width*u,depth*v,4*i*(i-1)/(degree*(degree-1))+3*j*(j-1)/(degree*(degree-1))+2*u*v]
    p[degree//2,degree//2,2]+=bump
    return BezierPatch(p)


def intent_fixture():
    return {'schema':'cad-design-intent-v1','parameters':{
        'zero':{'value':0.,'unit':'mm','fixed':True},'ten':{'value':10.,'unit':'mm','fixed':True},
        'radius':{'value':10.,'unit':'cm','fixed':True},'angle':{'value':30.,'unit':'deg','fixed':True},
        'x':{'value':65.,'unit':'mm','bounds':[1.,200.]},'y':{'value':40.,'unit':'mm','bounds':[1.,150.]}},
        'formulas':{'diameter':{'expression':'2*radius','unit':'mm'},'rise':{'expression':'radius*sin(angle)','unit':'mm'}},
        'points':{'origin':['zero','zero','zero'],'axis':['ten','zero','zero'],'p':['x','y','zero']},
        'constraints':[{'id':'radial_dimension','kind':'distance','a':'origin','b':'p','target':'radius','tolerance':{'value':1e-5,'unit':'mm'}},
            {'id':'styling_angle','kind':'angle','a0':'origin','a1':'axis','b0':'origin','b1':'p','target':'angle','tolerance':{'value':1e-5,'unit':'deg'}}]}
