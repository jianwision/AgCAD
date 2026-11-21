"""Typed engineering document backend for the existing ModelSession/ModelAdaptor.

Objects in each document are immutable results. Tool programs stage all writes
and commit atomically; user intent/tolerance/delivery rules stay caller-owned.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
from class_a import BezierPatch,gpu_batch,match_boundary,seam_report,assess_quality
from design_intent import DesignIntent,expression
import tolerance_topology as topo
from session_protocol import ObjectHandle,ToolProgram,OperationProgram,integer

TOOLS={
    'intent.solve':{'args':['initial?','max_evaluations?'],'result':'intent_solution','backend':'CPU nonlinear solve'},
    'intent.evaluate':{'args':['values?'],'result':'intent_measurement','backend':'CPU named constraints'},
    'patch.create':{'args':['poles'],'result':'bezier_patch','backend':'CPU definition'},
    'surface.match':{'args':['left','right','order?','scale?','shear?','reverse?'],'result':'bezier_patch','backend':'CPU exact polynomial construction'},
    'surface.analyze':{'args':['patch','samples?'],'result':'quality_report','backend':'CUDA batch jets/reflection/curvature'},
    'surface.fair':{'args':['patch','iterations','max_displacement_mm','locked?','samples?'],'result':'bezier_patch','backend':'CUDA bounded fairing'},
    'surface.seam':{'args':['left','right','order?','scale?','shear?','reverse?','samples?'],'result':'seam_report','backend':'CPU independent seam analysis'},
    'topology.face':{'args':['patch'],'result':'shape','backend':'CPU OCCT'},
    'topology.trim':{'args':['patch','outer','holes?'],'result':'shape','backend':'CPU OCCT'},
    'topology.sew':{'args':['shapes','solid?','connected?'],'result':'shape','backend':'CPU OCCT'},
    'topology.boolean':{'args':['a','b','operation?','require_solid?'],'result':'shape','backend':'CPU OCCT'},
    'topology.split':{'args':['shape','tools'],'result':'shape','backend':'CPU OCCT'},
    'topology.section':{'args':['a','b'],'result':'shape','backend':'CPU OCCT'},
    'topology.heal':{'args':['shape'],'result':'shape','backend':'CPU OCCT'},
    'topology.inspect':{'args':['shape'],'result':'topology_report','backend':'CPU OCCT'}}


class EngineeringRuntime:
    backend='engineering-document-v1'

    def __init__(self,root,intent_spec,*,policy=None,delivery_rules=None,budget=128,slots=8):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.intent=DesignIntent(intent_spec);self.policy=policy or topo.TolerancePolicy();self.rules=deepcopy(delivery_rules or {})
        self.budget=integer(budget,1,4096);self.remaining=budget;self.slots=integer(slots,1,64)
        self.session_id=uuid.uuid4().hex;self.pid=None;self.events=[];self.gpu_reports=[];self._live={};self._documents={};self._generations={}
        self._closed=False;self.scene={'backend':self.backend,'object_kind':'engineering_document','units':'mm',
            'user_intent':deepcopy(intent_spec),'intent_sha256':self.intent.contract_sha256,'tolerance_policy':asdict(self.policy),
            'delivery_rules':self.rules,'goal_is_immutable':True,'quality_improvement_required_for_commit':False,
            'device_models':{},'atomic_tool_programs':True,'gpu_execution':'on-demand batched analysis/fairing; explicit geometry readback'}
        self.requests=0
        (self.root/'scene.json').write_text(json.dumps(self.scene,indent=2),encoding='utf-8')

    def _document(self,handle):
        if self._closed:raise ValueError('Runtime closed')
        if handle.session_id!=self.session_id:raise ValueError('Object belongs to another session')
        if self._live.get(handle.slot)!=handle:raise ValueError('Stale document handle or revision')
        return self._documents[handle.slot]

    def create(self):
        self._request()
        free=next((i for i in range(self.slots) if i not in self._live),None)
        if free is None:raise ValueError('Document capacity exhausted')
        generation=self._generations.get(free,0)+1;self._generations[free]=generation
        h=ObjectHandle(self.session_id,free,generation,0);self._live[free]=h
        self._documents[free]={'objects':{},'feedback':None,'stopped':False,'solution':None}
        feedback=self._feedback(h,self._documents[free]);return h,feedback

    def _feedback(self,handle,document,*,reason='ACCEPTED',trace=None,error=None,geometry=False):
        feedback={'object_id':handle.object_id,'revision':handle.revision,'reason':reason,'remaining_actions':self.remaining,
            'state':{'draft_active':0,'live':1,'stopped':document['stopped']},'stopped':document['stopped'],
            'all_actions_admitted':error is None,'objects':{name:self._describe(value,geometry) for name,value in document['objects'].items()},
            'intent_solution':deepcopy(document['solution']),'trace':trace or [],'goal_met':None}
        if error:feedback['tool_error']=error
        document['feedback']=deepcopy(feedback);return feedback

    @staticmethod
    def _describe(record,geometry=False):
        kind,value=record['kind'],record['value'];result={'kind':kind,'evidence':deepcopy(record.get('evidence',{}))}
        if kind=='bezier_patch':
            result['degrees']=value.degrees
            if geometry:result['poles']=value.poles.tolist()
        elif kind=='shape':result['topology']=topo.inspect_shape(value)
        else:result['value']=deepcopy(value)
        return result

    def model_capabilities(self,handle):
        return {'schema':'cad-decision-v1','decision_kinds':['tools','inspect','fork','select','drop','program'],
            'tools':deepcopy(TOOLS),'program_operations':['MEASURE','STOP'],'tool_program_rule':'atomic; all staged objects discarded on failure',
            'response_example':{'schema':'cad-decision-v1','kind':'tools','object_id':handle.object_id,'scene_revision':handle.revision,
                'calls':[{'op':'intent.solve','args':{},'out':'dimensions'}]},'max_tool_calls':32,
            'admission_requires_quality_improvement':False,'final_rules_rechecked_on_export':True}

    def model_resources(self):
        return {'max_program_actions':32,'max_backend_requests':1024,'remaining_backend_requests':1024-self.requests,
                'object_capacity':self.slots,'objects_per_document':128,'nonlinear_evaluation_limit':2000}

    def _request(self):
        if self.requests>=1024:raise ValueError('Backend request budget exhausted')
        self.requests+=1

    def _call(self,call,objects,document):
        if call.op not in TOOLS:raise ValueError('Unsupported engineering operation: '+call.op)
        args=deepcopy(call.args);schema=TOOLS[call.op]
        allowed={k.rstrip('?') for k in schema['args']};required={k for k in schema['args'] if not k.endswith('?')}
        if set(args)-allowed or not required<=set(args):raise ValueError('Invalid arguments for '+call.op)
        def take(key,kind):
            name=args.pop(key)
            if not isinstance(name,str) or name not in objects or objects[name]['kind']!=kind:raise ValueError('Unknown/wrong-type geometry reference: '+str(name))
            return objects[name]['value']
        evidence={};op=call.op
        if op=='intent.solve':
            value=self.intent.solve(**args)
            if value['hard_constraints_pass']:document['solution']=value
        elif op=='intent.evaluate':value=self.intent.evaluate(**args)
        elif op=='patch.create':value=BezierPatch(args['poles'])
        elif op in ('surface.match','surface.seam'):
            left,right=take('left','bezier_patch'),take('right','bezier_patch')
            value=(match_boundary if op=='surface.match' else seam_report)(left,right,**args)
        elif op in ('surface.analyze','surface.fair'):
            patch=take('patch','bezier_patch');samples=args.pop('samples',17)
            if 'locked' in args:args['locked']=np.asarray(args['locked'])
            results,gpu=gpu_batch([patch],self.root/f'gpu-{len(self.gpu_reports):04d}',samples=samples,fairing=[args if op=='surface.fair' else {}])
            self.gpu_reports.append(gpu);evidence=results[0]['report']
            value=results[0]['patch'] if op=='surface.fair' else evidence
        elif op=='topology.face':
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            value=BRepBuilderAPI_MakeFace(take('patch','bezier_patch').native(),1e-7).Face()
            self.policy.resolve([value]);evidence=topo.inspect_shape(value,self.policy)
        elif op=='topology.trim':value,evidence=topo.trim(take('patch','bezier_patch'),policy=self.policy,**args)
        elif op=='topology.sew':
            refs=args.pop('shapes');shapes=[]
            for name in refs:
                if name not in objects or objects[name]['kind']!='shape':raise ValueError('Invalid sewing shape reference')
                shapes.append(objects[name]['value'])
            value,evidence=topo.sew(shapes,policy=self.policy,**args)
        elif op in ('topology.boolean','topology.section'):
            a,b=take('a','shape'),take('b','shape');fn=topo.boolean if op=='topology.boolean' else topo.section
            value,evidence=fn(a,b,policy=self.policy,**args)
        elif op=='topology.split':
            shape=take('shape','shape');refs=args.pop('tools');tools=[]
            for name in refs:
                if name not in objects or objects[name]['kind']!='shape':raise ValueError('Invalid splitting tool reference')
                tools.append(objects[name]['value'])
            value,evidence=topo.split(shape,tools,policy=self.policy)
        elif op=='topology.heal':value,evidence=topo.heal(take('shape','shape'),policy=self.policy)
        else:value=topo.inspect_shape(take('shape','shape'),self.policy)
        return {'kind':schema['result'],'value':value,'evidence':evidence}

    def execute(self,handle,decision):
        self._request()
        document=self._document(handle)
        if isinstance(decision,OperationProgram):
            if any(a.op not in ('MEASURE','STOP') for a in decision.actions):raise ValueError('This backend requires typed engineering tools')
            if len(decision.actions)>self.remaining:raise ValueError('Action budget exhausted')
            self.remaining-=len(decision.actions);document['stopped']=any(a.op=='STOP' for a in decision.actions)
            return handle,self._feedback(handle,document,reason='STOPPED' if document['stopped'] else 'ACCEPTED')
        if not isinstance(decision,ToolProgram):raise ValueError('Device rollout unavailable for this engineering backend')
        if len(decision.calls)>self.remaining:raise ValueError('Action budget exhausted')
        self.remaining-=len(decision.calls);stage={**document,'objects':dict(document['objects']),'solution':deepcopy(document['solution']),'stopped':False}
        trace=[];error=None
        for call in decision.calls:
            try:
                stage['objects'][call.out]=self._call(call,stage['objects'],stage)
                if len(stage['objects'])>128:raise ValueError('Document object capacity exceeded')
                trace.append({'operation':call.op,'output':call.out,'accepted':True})
            except (ValueError,KeyError,TypeError,RuntimeError) as failure:
                error=str(failure);trace.append({'operation':call.op,'output':call.out,'accepted':False,'error':error});break
        if error is None:
            handle=ObjectHandle(handle.session_id,handle.slot,handle.generation,handle.revision+1)
            self._live[handle.slot]=handle;self._documents[handle.slot]=stage;document=stage
        feedback=self._feedback(handle,document,reason='TOOL_FAILED' if error else 'ACCEPTED',trace=trace,error=error)
        self.events.append({'revision':handle.revision,'committed':error is None,'reserved_actions':len(decision.calls),'trace':trace})
        return handle,feedback

    def observe(self,handle,*,geometry=False,draft=False):
        self._request()
        if draft:raise ValueError('Tool programs are atomic; no persistent private draft in this backend')
        return handle,self._feedback(handle,self._document(handle),geometry=geometry)

    def objects(self):return tuple(self._live.values())
    def latest(self,handle):
        if handle.session_id!=self.session_id or handle.slot not in self._live or self._live[handle.slot].generation!=handle.generation:raise ValueError('Unknown/stale document')
        return self._live[handle.slot]
    def cached_view(self,handle):return deepcopy(self._document(handle)['feedback'])
    def fork(self,handle):
        source=self._document(handle);child,_=self.create()
        self._documents[child.slot]={**source,'objects':dict(source['objects']),'solution':deepcopy(source['solution']),'stopped':False}
        return child,self._feedback(child,self._documents[child.slot])
    def drop(self,handle):
        self._request()
        self._document(handle);self._live.pop(handle.slot);self._documents.pop(handle.slot)
        return {'object_id':handle.object_id,'dropped':True}

    def export_step(self,handle,path,*,shape='result'):
        self._request()
        document=self._document(handle);objects=document['objects'];checks=[];solution=document['solution']
        if shape not in objects or objects[shape]['kind']!='shape':raise ValueError('No shape selected for delivery')
        delivered=objects[shape]['value']
        def require_support(patch,tolerance):
            # Bind design checks to actual delivered supports, not unrelated scene objects.
            from OCP.Geom import Geom_BezierSurface
            for face in topo.subshapes(delivered,topo.TopAbs_FACE):
                native=topo.BRep_Tool.Surface_s(face)
                if not isinstance(native,Geom_BezierSurface) or (native.NbUPoles(),native.NbVPoles())!=patch.poles.shape[:2]:continue
                errors=[]
                for i,row in enumerate(patch.poles):
                    for j,p in enumerate(row):
                        q=native.Pole(i+1,j+1);errors.append(np.linalg.norm(p-[q.X(),q.Y(),q.Z()]))
                if max(errors)<=tolerance:return
            raise ValueError('Declared design surface is absent from delivered shape')
        if self.rules.get('require_intent',True):
            if solution is None:raise ValueError('Required design intent has not been solved')
            fresh=self.intent.evaluate(solution['solved_parameters'])
            if not fresh['hard_constraints_pass']:raise ValueError('Final design-intent constraints failed')
            checks.append({'kind':'intent','pass':True,'contract_sha256':self.intent.contract_sha256})
        for rule in self.rules.get('coordinates',[]):
            patch=objects[rule['patch']]['value']
            if not isinstance(patch,BezierPatch) or solution is None:raise ValueError('Missing linked design geometry')
            require_support(patch,rule['tolerance_mm'])
            target=expression(rule['expression'],self.intent._context(solution['solved_parameters']))
            if target.dimension!=(1,0):raise ValueError('Linked geometry target must have length dimension')
            residual=abs(patch.jet(rule['u'],rule['v'])[rule['axis']]-target.value)
            if residual>rule['tolerance_mm']:raise ValueError('Final geometry/design-intent binding failed')
            checks.append({'kind':'coordinate_binding','residual_mm':float(residual),'pass':True})
        for rule in self.rules.get('seams',[]):
            require_support(objects[rule['left']]['value'],rule['tolerance_mm'])
            require_support(objects[rule['right']]['value'],rule['tolerance_mm'])
            args={k:v for k,v in rule.items() if k not in ('left','right','tolerance_mm','maximum_normal_angle_rad','maximum_curvature_gap_per_mm')}
            report=seam_report(objects[rule['left']]['value'],objects[rule['right']]['value'],**args)
            if max(report['jet_max_mm'])>rule['tolerance_mm']:raise ValueError('Final declared seam constraint failed')
            if report['normal_angle_max_rad']>rule.get('maximum_normal_angle_rad',float('inf')):raise ValueError('Final seam normal-angle constraint failed')
            if report['transverse_normal_curvature_gap_per_mm']>rule.get('maximum_curvature_gap_per_mm',float('inf')):raise ValueError('Final seam curvature constraint failed')
            checks.append({'kind':'seam','pass':True,'report':report})
        for rule in self.rules.get('quality',[]):
            patch=objects[rule['patch']]['value'];require_support(patch,self.policy.absolute_mm)
            results,gpu=gpu_batch([patch],self.root/f'gpu-{len(self.gpu_reports):04d}',samples=rule.get('samples',25))
            self.gpu_reports.append(gpu);quality=assess_quality(results[0]['report'],rule['criteria'])
            if not quality['pass']:raise ValueError('Final declared surface quality failed')
            checks.append({'kind':'quality','pass':True,'report':quality})
        result=topo.export_step_checked(delivered,path,policy=self.policy)
        result['user_contract_checks']=checks;return result

    def close(self):
        if self._closed:return
        (self.root/'runtime-report.json').write_text(json.dumps({'backend':self.backend,'budget':self.budget,
            'remaining':self.remaining,'events':self.events,'gpu_batches':self.gpu_reports},indent=2),encoding='utf-8');self._closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
