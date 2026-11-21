"""Unit-aware design parameters/formulas and bounded nonlinear geometric constraints.

Solving is an explicit CPU tool. A returned candidate is checked against each
hard tolerance; numerical failure never proves global infeasibility.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import numpy as np
from scipy.optimize import least_squares
from class_a import BezierPatch

UNITS={'1':(1.,(0,0)),'mm':(1.,(1,0)),'cm':(10.,(1,0)),'m':(1000.,(1,0)),
       'rad':(1.,(0,1)),'deg':(math.pi/180,(0,1))}


@dataclass(frozen=True)
class Quantity:
    value:float
    dimension:tuple=(0,0)

    def __post_init__(self):
        if not math.isfinite(self.value):raise ValueError('Nonfinite formula value')


def quantity(value,unit):
    if unit not in UNITS or isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError('Invalid scalar/unit')
    scale,dimension=UNITS[unit];return Quantity(value*scale,dimension)


def expression(text,context):
    if not isinstance(text,str) or len(text)>2048:raise ValueError('Invalid formula expression')
    tree=ast.parse(text,mode='eval')
    if sum(1 for _ in ast.walk(tree))>256:raise ValueError('Formula capacity exceeded')
    def run(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return Quantity(float(node.value))
        if isinstance(node,ast.Name):
            if node.id not in context:raise ValueError('Unknown formula parameter: '+node.id)
            return context[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            a=run(node.operand);return Quantity(a.value*(-1 if isinstance(node.op,ast.USub) else 1),a.dimension)
        if isinstance(node,ast.BinOp):
            a,b=run(node.left),run(node.right)
            if isinstance(node.op,(ast.Add,ast.Sub)):
                if a.dimension!=b.dimension:raise ValueError('Formula dimension mismatch')
                return Quantity(a.value+(b.value if isinstance(node.op,ast.Add) else -b.value),a.dimension)
            if isinstance(node.op,ast.Mult):return Quantity(a.value*b.value,tuple(x+y for x,y in zip(a.dimension,b.dimension)))
            if isinstance(node.op,ast.Div):return Quantity(a.value/b.value,tuple(x-y for x,y in zip(a.dimension,b.dimension)))
            if isinstance(node.op,ast.Pow) and b.dimension==(0,0) and b.value.is_integer() and abs(b.value)<=4:
                return Quantity(a.value**int(b.value),tuple(x*int(b.value) for x in a.dimension))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            args=[run(v) for v in node.args];name=node.func.id
            if name=='sqrt' and len(args)==1:
                a=args[0]
                if any(d%2 for d in a.dimension) or a.value<0:raise ValueError('Invalid dimensional square root')
                return Quantity(math.sqrt(a.value),tuple(d//2 for d in a.dimension))
            if name in ('sin','cos') and len(args)==1 and args[0].dimension==(0,1):
                return Quantity(getattr(math,name)(args[0].value))
            if name=='abs' and len(args)==1:return Quantity(abs(args[0].value),args[0].dimension)
        raise ValueError('Unsupported formula syntax')
    try:return run(tree.body)
    except (ZeroDivisionError,OverflowError) as error:raise ValueError('Formula numerical domain error') from error


class DesignIntent:
    def __init__(self,spec):
        self.spec=deepcopy(spec)
        if self.spec.get('schema')!='cad-design-intent-v1':raise ValueError('Unknown design-intent schema')
        self.parameters=self.spec.get('parameters',{});self.formulas=self.spec.get('formulas',{})
        self.constraints=self.spec.get('constraints',[]);self.points=self.spec.get('points',{})
        if not 1<=len(self.parameters)<=64 or len(self.formulas)>128 or not 1<=len(self.constraints)<=128:
            raise ValueError('Design-intent capacity exceeded')
        if set(self.parameters)&set(self.formulas):raise ValueError('Parameter/formula name collision')
        for name in [*self.parameters,*self.formulas]:
            if not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}',name):raise ValueError('Invalid parameter identifier')
        for p in self.parameters.values():
            q=quantity(p['value'],p['unit'])
            if not isinstance(p.get('fixed',False),bool):raise ValueError('Invalid fixed parameter flag')
            bounds=p.get('bounds',[-1e6,1e6])
            if len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or not bounds[0]<=p['value']<=bounds[1] or bounds[0]>=bounds[1]:
                raise ValueError('Invalid parameter bounds')
            if abs(q.value)>1e6:raise ValueError('Parameter outside numerical domain')
        ids=[c.get('id') for c in self.constraints]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid/duplicate constraint id')
        for c in self.constraints:
            if not isinstance(c.get('hard',True),bool):raise ValueError('Invalid constraint hardness')
            if c.get('relation','equal') not in ('equal','le','ge'):raise ValueError('Invalid constraint relation')
            if not isinstance(c.get('weight',1),(int,float)) or c.get('weight',1)<=0:raise ValueError('Invalid constraint weight')
            tol=c['tolerance'];q=quantity(tol['value'],tol['unit'])
            if q.value<=0:raise ValueError('Constraint tolerance must be positive')
        self.surfaces={k:BezierPatch(v) for k,v in self.spec.get('surfaces',{}).items()}
        self.curves={}
        for name,poles in self.spec.get('curves',{}).items():
            p=np.asarray(poles,dtype=float)
            if p.ndim!=2 or p.shape[1]!=3 or not 2<=len(p)<=8 or not np.isfinite(p).all():raise ValueError('Invalid Bezier guide curve')
            self.curves[name]=p.copy()
        self.evaluate()  # Validate dependencies, units and geometry references before solving.

    @property
    def contract_sha256(self):return hashlib.sha256(self.dumps().encode()).hexdigest()

    def dumps(self):return json.dumps(self.spec,sort_keys=True,allow_nan=False)

    @classmethod
    def loads(cls,text):return cls(json.loads(text))

    def with_parameters(self,changes):
        """Explicit caller design change creates a new immutable contract instance."""
        spec=deepcopy(self.spec)
        for name,value in changes.items():
            if name not in spec['parameters']:raise ValueError('Unknown user parameter')
            spec['parameters'][name]['value']=value
        return DesignIntent(spec)

    def _context(self,values):
        if set(values)-set(self.parameters):raise ValueError('Unknown proposed parameter')
        context={}
        for name,p in self.parameters.items():
            if name in values and p.get('fixed',False):raise ValueError('Model cannot override fixed design parameters')
            v=values.get(name,p['value']);lo,hi=p.get('bounds',[-1e6,1e6])
            if not lo<=v<=hi:raise ValueError('Proposed parameter outside bounds')
            context[name]=quantity(v,p['unit'])
        pending=dict(self.formulas)
        while pending:
            progressed=False
            for name,f in list(pending.items()):
                try:q=expression(f['expression'],context)
                except ValueError as error:
                    if str(error).startswith('Unknown formula parameter:'):continue
                    raise
                if q.dimension!=UNITS[f['unit']][1]:raise ValueError('Derived parameter unit mismatch')
                context[name]=q;del pending[name];progressed=True
            if not progressed:raise ValueError('Cyclic or unresolved formula dependencies: '+','.join(pending))
        return context

    def _residual(self,c,context):
        def value(text,dim):
            q=expression(text,context)
            if q.dimension!=dim:raise ValueError('Geometric constraint unit mismatch')
            return q.value
        def point(name):
            values=self.points[name]
            if len(values)!=3:raise ValueError('Expected XYZ point expressions')
            return np.array([value(v,(1,0)) for v in values])
        kind=c['kind'];dim=(1,0)
        if kind=='expression':
            a,b=expression(c['lhs'],context),expression(c['rhs'],context)
            if a.dimension!=b.dimension:raise ValueError('Constraint expression dimensions differ')
            raw=np.array([a.value-b.value]);dim=a.dimension
        elif kind=='distance':raw=np.array([np.linalg.norm(point(c['a'])-point(c['b']))-value(c['target'],dim)])
        elif kind=='coincident':raw=point(c['a'])-point(c['b'])
        elif kind in ('angle','parallel','perpendicular'):
            a=point(c['a1'])-point(c['a0']);b=point(c['b1'])-point(c['b0'])
            if min(np.linalg.norm(a),np.linalg.norm(b))<1e-10:raise ValueError('Degenerate constraint direction')
            angle=math.atan2(np.linalg.norm(np.cross(a,b)),float(a@b));dim=(0,1)
            residual=angle-value(c['target'],dim) if kind=='angle' else min(angle,math.pi-angle) if kind=='parallel' else angle-math.pi/2
            raw=np.array([residual])
        elif kind=='on_curve':
            t=value(c['t'],(0,0));p=self.curves[c['curve']];n=len(p)-1
            if not 0<=t<=1:raise ValueError('Curve attachment parameter outside [0,1]')
            q=sum(math.comb(n,i)*t**i*(1-t)**(n-i)*p[i] for i in range(n+1));raw=point(c['point'])-q
        elif kind=='on_surface':
            u,v=value(c['u'],(0,0)),value(c['v'],(0,0));raw=point(c['point'])-self.surfaces[c['surface']].jet(u,v)
        elif kind=='symmetry':
            normal=np.asarray(c.get('normal',[0,1,0]),dtype=float)
            if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-10:raise ValueError('Invalid symmetry normal')
            normal/=np.linalg.norm(normal);a=point(c['a']);b=point(c['b']);offset=value(c['offset'],dim)
            raw=b-(a-2*(normal@a-offset)*normal)
        else:raise ValueError('Unsupported geometric constraint: '+str(kind))
        tolerance=quantity(c['tolerance']['value'],c['tolerance']['unit'])
        if tolerance.dimension!=dim:raise ValueError('Constraint tolerance dimension mismatch')
        relation=c.get('relation','equal')
        violation=raw if relation=='equal' else np.maximum(raw,0) if relation=='le' else np.minimum(raw,0)
        return raw,violation/tolerance.value

    def evaluate(self,values=None):
        context=self._context(values or {});reports=[]
        for c in self.constraints:
            try:raw,ratio=self._residual(c,context)
            except KeyError as error:raise ValueError('Unknown geometric reference: '+str(error)) from error
            maximum=float(np.max(np.abs(ratio)))
            reports.append({'id':c['id'],'kind':c['kind'],'hard':c.get('hard',True),
                'relation':c.get('relation','equal'),'residual_canonical':raw.tolist(),
                'tolerance_ratio':maximum,'pass':maximum<=1.,'phase':c.get('phase','commit')})
        values={}
        for name,q in context.items():
            unit=(self.parameters.get(name) or self.formulas[name])['unit'];values[name]={'value':q.value/UNITS[unit][0],'unit':unit}
        return {'hard_constraints_pass':all(r['pass'] for r in reports if r['hard']),
                'constraints':reports,'parameters':values,'canonical_values':{k:q.value for k,q in context.items()},
                'contract_sha256':self.contract_sha256,'backend':'cpu-design-intent-v1'}

    def solve(self,initial=None,*,max_evaluations=400):
        if type(max_evaluations) is not int or not 1<=max_evaluations<=2000:raise ValueError('Invalid nonlinear solve budget')
        initial=initial or {};self.evaluate(initial)
        names=[name for name,p in self.parameters.items() if not p.get('fixed',False)]
        x=np.array([initial.get(n,self.parameters[n]['value']) for n in names])
        bounds=np.array([self.parameters[n].get('bounds',[-1e6,1e6]) for n in names])
        hard=[c for c in self.constraints if c.get('hard',True)]
        evaluations=0;best=x.copy();best_cost=float('inf')
        class EvaluationLimit(Exception):pass
        def residual(v):
            nonlocal evaluations,best,best_cost
            if evaluations>=max_evaluations:raise EvaluationLimit
            evaluations+=1
            context=self._context(dict(zip(names,map(float,v))))
            values=np.concatenate([self._residual(c,context)[1] for c in hard]) if hard else np.zeros(1)
            cost=float(values@values)
            if cost<best_cost:best_cost=cost;best=v.copy()
            return values
        if names:
            try:
                result=least_squares(residual,x,bounds=(bounds[:,0],bounds[:,1]),max_nfev=max_evaluations,
                                     xtol=1e-12,ftol=1e-12,gtol=1e-12,x_scale='jac')
                x=result.x;success=bool(result.success)
                singular=np.linalg.svd(result.jac,compute_uv=False)
                cutoff=max(result.jac.shape)*np.finfo(float).eps*singular.max(initial=0)*100
                rank=int(np.sum(singular>cutoff))
            except EvaluationLimit:x=best;success=False;rank=None
        else:success=True;evaluations=1;rank=0
        report=self.evaluate(dict(zip(names,map(float,x))))
        report.update({'solver_terminated':success,'function_evaluations':evaluations,'solved_parameters':dict(zip(names,map(float,x))),
            'evaluation_budget':max_evaluations,'finite_difference_calls_included':True,'final_verification_evaluations':1,
            'free_variables':len(names),'local_hard_jacobian_rank':rank,'local_remaining_dof':len(names)-rank if rank is not None else None,
            'suspected_conflicts':[r['id'] for r in report['constraints'] if r['hard'] and not r['pass']],
            'soft_constraints_optimized':False,'global_infeasibility_proven':False})
        return report
