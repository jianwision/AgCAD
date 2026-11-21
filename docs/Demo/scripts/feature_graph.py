"""Typed feature DAG with deterministic serialization and transactional history.

Object ids survive parameter edits, undo/redo and save/load. Subshape selections
are revision-scoped geometric queries: topology changes require re-selection;
this is intentionally not a claim of general persistent topological naming.
"""
from dataclasses import dataclass
import copy,json
import numpy as np
from cad_ops import valid
from OCP.TopAbs import TopAbs_VERTEX,TopAbs_EDGE,TopAbs_FACE,TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

@dataclass(frozen=True)
class Handle:
    id:str
    kind:str
@dataclass(frozen=True)
class Selection:
    parent:Handle
    revision:int
    kind:str
    center:tuple
    tolerance:float
class FeatureGraph:
    def __init__(self,registry):
        # registry[name]=(input kinds,output kind,callable(*inputs,**params))
        self.registry=registry;self.nodes={};self.cache={};self.revisions={};self.next_id=1;self.epoch=0;self.past=[];self.future=[];self.evaluations={}
    def _snapshot(self):return copy.deepcopy(self.nodes)
    def _commit(self,before):self.past.append(before);self.future.clear()
    def _node(self,handle):
        node=self.nodes.get(handle.id)
        if node is None or node['kind']!=handle.kind:raise ValueError('Unknown handle or type mismatch')
        return node
    def _validate(self,nodes):
        active=set();done=set()
        def visit(k):
            if k in active:raise ValueError('Dependency cycle')
            if k in done:return
            node=nodes[k];spec=self.registry[node['operation']];inputs=node['inputs']
            if len(inputs)!=len(spec[0]) or node['kind']!=spec[1]:raise ValueError('Operation signature mismatch')
            active.add(k)
            for child,kind in zip(inputs,spec[0]):
                if child not in nodes or nodes[child]['kind']!=kind:raise ValueError('Typed input mismatch')
                visit(child)
            active.remove(k);done.add(k)
        for k in nodes:visit(k)
    def add(self,operation,inputs=(),**params):
        if operation not in self.registry:raise ValueError('Unknown operation')
        for handle in inputs:self._node(handle)
        before=self._snapshot();key=f'object-{self.next_id:08d}';self.next_id+=1
        self.nodes[key]={'operation':operation,'inputs':[h.id for h in inputs],'params':copy.deepcopy(params),'kind':self.registry[operation][1]}
        try:self._validate(self.nodes)
        except Exception:self.nodes=before;raise
        self._commit(before);return Handle(key,self.nodes[key]['kind'])
    def update(self,handle,inputs=None,**params):
        self._node(handle);before=self._snapshot();node=self.nodes[handle.id]
        if inputs is not None:
            for h in inputs:self._node(h)
            node['inputs']=[h.id for h in inputs]
        node['params'].update(copy.deepcopy(params))
        try:self._validate(self.nodes)
        except Exception:self.nodes=before;raise
        dirty={handle.id}
        while True:
            expanded=dirty|{k for k,v in self.nodes.items() if set(v['inputs'])&dirty}
            if dirty==expanded:break
            dirty=expanded
        for k in dirty:self.cache.pop(k,None);self.revisions.pop(k,None)
        self._commit(before)
    def evaluate(self,handle):
        node=self._node(handle)
        if handle.id in self.cache:return self.cache[handle.id]
        args=[self.evaluate(Handle(k,self.nodes[k]['kind'])) for k in node['inputs']]
        result=self.registry[node['operation']][2](*args,**copy.deepcopy(node['params']))
        if node['kind'] in ['solid','face']:
            valid(result)
            if not TopExp_Explorer(result,TopAbs_SOLID if node['kind']=='solid' else TopAbs_FACE).More():raise ValueError('Registered operation returned the wrong geometry kind')
        self.epoch+=1;self.revisions[handle.id]=self.epoch;self.cache[handle.id]=result;self.evaluations[handle.id]=self.evaluations.get(handle.id,0)+1
        return result
    def rebuild(self,handle):
        self.cache.clear();self.revisions.clear();return self.evaluate(handle)
    def undo(self):
        if not self.past:raise ValueError('No undo history')
        self.future.append(self._snapshot());self.nodes=self.past.pop();self.cache.clear();self.revisions.clear()
    def redo(self):
        if not self.future:raise ValueError('No redo history')
        self.past.append(self._snapshot());self.nodes=self.future.pop();self.cache.clear();self.revisions.clear()
    def dumps(self):
        return json.dumps({'schema':1,'next_id':self.next_id,'nodes':self.nodes},sort_keys=True,indent=2,allow_nan=False)
    @classmethod
    def loads(cls,text,registry):
        data=json.loads(text)
        if data['schema']!=1:raise ValueError('Unknown graph schema')
        g=cls(registry);g.nodes=data['nodes'];g.next_id=data['next_id'];g._validate(g.nodes);return g
    def select(self,handle,kind,center,tolerance=1e-7):
        if kind not in ['vertex','edge','face'] or tolerance<=0:raise ValueError('Invalid selection kind/tolerance')
        shape=self.evaluate(handle);selected=Selection(handle,self.revisions[handle.id],kind,tuple(center),tolerance);self._resolve(shape,selected);return selected
    def resolve(self,selection):
        shape=self.evaluate(selection.parent)
        if self.revisions[selection.parent.id]!=selection.revision:raise ValueError('Stale selection: reselect after rebuild or topology edit')
        return self._resolve(shape,selection)
    @staticmethod
    def _resolve(shape,selection):
        enum={'vertex':TopAbs_VERTEX,'edge':TopAbs_EDGE,'face':TopAbs_FACE}[selection.kind];e=TopExp_Explorer(shape,enum);matches=[]
        while e.More():
            item=e.Current();e.Next()
            if any(item.IsSame(x) for x in matches):continue
            if selection.kind=='vertex':q=BRep_Tool.Pnt_s(TopoDS.Vertex_s(item))
            else:
                props=GProp_GProps()
                (BRepGProp.LinearProperties_s if selection.kind=='edge' else BRepGProp.SurfaceProperties_s)(item,props);q=props.CentreOfMass()
            if np.linalg.norm(np.array([q.X(),q.Y(),q.Z()])-selection.center)<=selection.tolerance:matches.append(item)
        if len(matches)!=1:raise ValueError(f'Selection expected one {selection.kind}, got {len(matches)}')
        return {'vertex':TopoDS.Vertex_s,'edge':TopoDS.Edge_s,'face':TopoDS.Face_s}[selection.kind](matches[0])
