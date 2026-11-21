"""Reproducible geometric inputs, independent of the model/worker implementation."""
import numpy as np
from class_a import BezierPatch
from engineering_fixtures import polynomial_patch
from nurbs_surface import NurbsPatch


def rational_fixture():
    p=polynomial_patch(3).poles.copy();w=np.array([[1.,.9,1.2,1.],[1.1,.8,1.05,1.2],[1.,1.3,.95,1.],[1.2,1.,.9,1.1]])
    return NurbsPatch(p,w,[0]*4+[1]*4,[0]*4+[1]*4,3,3).insert_knot(0,.37).insert_knot(1,.62)


def cylinder_fixture():
    p=np.array([[[10,0,z],[10,10,z],[0,10,z]] for z in (0.,20.)])
    return NurbsPatch(p,np.tile([1,2**-.5,1],(2,1)),[0,0,1,1],[0,0,0,1,1,1],1,2)


def network_fixture(degree=5,order=2):
    patches=[]
    for i in range(3):
        base=polynomial_patch(degree,x0=i*100).poles.copy();u=(base[:,:,0]-i*100)/100;v=base[:,:,1]/60
        base[:,:,2]+=8*i*u+4*i*i+2*i*v;base[2:4,2:4,2]+=(i+1)*2
        patches.append(BezierPatch(base))
    masks=[np.zeros(p.poles.shape[:2],dtype=bool) for p in patches];masks[0][0]=True;masks[-1][-1]=True
    seams=[{'left':0,'right':1,'order':order},{'left':1,'right':2,'order':order}]
    return patches,seams,masks


def corner_fixture(order=2):
    patches=[]
    for i,j in ((0,0),(1,0),(0,1),(1,1)):
        base=polynomial_patch(5,x0=i*100).poles.copy();u=(base[:,:,0]-i*100)/100;v=base[:,:,1]/60
        base[:,:,1]+=j*60;base[:,:,2]+=8*i*u+4*i*i+2*i*(v+j)+6*j*v+3*j*j+2*j*u
        base[2:4,2:4,2]+=1;patches.append(BezierPatch(base))
    seams=[{'left':0,'right':1,'order':order},{'left':2,'right':3,'order':order},
           {'left':0,'right':2,'left_edge':'v1','right_edge':'v0','order':order},
           {'left':1,'right':3,'left_edge':'v1','right_edge':'v0','order':order}]
    return patches,seams


def nurbs_args(patch):
    return {'poles':patch.poles.tolist(),'weights':patch.weights.tolist(),'knots_u':patch.knots_u.tolist(),
            'knots_v':patch.knots_v.tolist(),'degree_u':patch.degree_u,'degree_v':patch.degree_v}
