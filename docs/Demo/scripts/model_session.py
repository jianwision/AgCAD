"""Model sessions own decision flow; the runtime owns objects and constraints."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
from model_adaptor import ModelAdaptor
from session_protocol import (DeviceRollout, DropObject, Fork, Inspect, OperationProgram,
                              OPERATIONS, OPERATION_DETAILS, SelectObject, ToolProgram)
from session_runtime import RuntimeRejected


class ModelSession:
    def __init__(self,runtime,model:ModelAdaptor,handle,feedback,*,max_decisions=32):
        if not isinstance(model,ModelAdaptor): raise TypeError('Expected ModelAdaptor')
        if not 1<=max_decisions<=256: raise ValueError('Invalid model decision budget')
        self.runtime=runtime;self.model=model;self.handle=handle;self.feedback=deepcopy(feedback)
        self.max_decisions=max_decisions;self.history=[];self.decisions=0;self.provider_calls=0
        self.switches=[];self.device_inferences=0

    def observation(self):
        observation={'schema':'cad-observation-v1','session_id':self.runtime.session_id,
            'object':{'id':self.handle.object_id,'revision':self.handle.revision,
                      'generation':self.handle.generation,'kind':self.runtime.scene['object_kind'],'backend':self.runtime.backend},
            'scene':self.runtime.scene,'feedback':self.feedback,
            'objects':[{'id':h.object_id,'revision':h.revision,'generation':h.generation,
                        'draft_active':bool(self.runtime.cached_view(h)['state']['draft_active'])}
                       for h in self.runtime.objects()],
            'resources':{'remaining_actions':self.runtime.remaining,'remaining_model_decisions':self.max_decisions-self.decisions,
                         'max_program_actions':64,'object_capacity':self.runtime.slots,'max_backend_requests':1024},
            'capabilities':{'operations':OPERATIONS,'operation_details':OPERATION_DETAILS,
                'active_geometry_rule':'draft if open, otherwise committed; actions without BEGIN commit individually',
                'decision_kinds':['program','device_rollout','inspect','fork','select','drop'],
                'schema':'cad-decision-v1','error_policy':['yield','continue'],
                'response_example':{'schema':'cad-decision-v1','kind':'program','object_id':self.handle.object_id,
                    'scene_revision':self.handle.revision,'actions':[{'op':'MEASURE'}],'on_error':'yield'},
                'decision_parameters':{'device_rollout':['steps','model_id','weights_sha256'],
                    'inspect':['geometry','draft'],'fork':['select'],
                    'select':['selected_object_id'],'drop':['drop_object_id']},
                'draft_rule':'geometry constraints at commit; finite domain at every action',
                'goal_status_rule':'STOP does not imply success','admission_requires_quality_improvement':False},
            'model':self.model.capabilities,'event_cursor':len(self.history),
            'history_reference':str(self.runtime.root/'model-events.json')}
        if hasattr(self.runtime,'model_capabilities'):observation['capabilities']=self.runtime.model_capabilities(self.handle)
        if hasattr(self.runtime,'model_resources'):observation['resources'].update(self.runtime.model_resources())
        return deepcopy(observation)

    def switch_model(self,model):
        if not isinstance(model,ModelAdaptor): raise TypeError('Expected ModelAdaptor')
        self.switches.append({'from':self.model.capabilities['name'],'to':model.capabilities['name'],
                              'object_id':self.handle.object_id,'revision':self.handle.revision,'worker_pid':self.runtime.pid})
        self.model=model
        self.save()

    def select(self,object_id):
        candidates=[h for h in self.runtime.objects() if h.object_id==object_id]
        if not candidates: raise ValueError('Unknown or stale scene object')
        self.handle=candidates[0];self.feedback=self.runtime.cached_view(self.handle)

    def refresh(self):
        self.handle=self.runtime.latest(self.handle)
        self.handle,self.feedback=self.runtime.observe(self.handle)
        return deepcopy(self.feedback)

    def step(self):
        if self.decisions>=self.max_decisions: raise RuntimeError('Model decision budget exhausted')
        before=self.observation();name=self.model.capabilities['name'];self.decisions+=1
        start=time.perf_counter();decision=None
        if self.model.capabilities['inference_location']=='host':self.provider_calls+=1
        try:
            decision=self.model.decide(before)
            model_ms=(time.perf_counter()-start)*1000
            if isinstance(decision,(OperationProgram,DeviceRollout,ToolProgram)):
                self.handle,self.feedback=self.runtime.execute(self.handle,decision)
                if isinstance(decision,DeviceRollout):self.device_inferences+=int(self.feedback['state']['executed_this_call'])
            elif isinstance(decision,Inspect):
                self.handle,self.feedback=self.runtime.observe(self.handle,geometry=decision.geometry,draft=decision.draft)
            elif isinstance(decision,Fork):
                child,view=self.runtime.fork(self.handle)
                if decision.select:self.handle,self.feedback=child,view
                else:self.feedback={**self.feedback,'created_object':child.object_id}
            elif isinstance(decision,SelectObject):self.select(decision.object_id)
            elif isinstance(decision,DropObject):
                if decision.object_id==self.handle.object_id: raise ValueError('Select another object before dropping current focus')
                candidates=[h for h in self.runtime.objects() if h.object_id==decision.object_id]
                if not candidates:raise ValueError('Unknown object to drop')
                self.runtime.drop(candidates[0]);self.feedback={**self.feedback,'dropped_object':decision.object_id}
            else:raise TypeError('Unsupported model decision type')
            result={'ok':True,'model':name,'decision_kind':type(decision).__name__,'model_ms':model_ms,
                    'feedback':deepcopy(self.feedback),'worker_pid':self.runtime.pid}
        except (ValueError,TypeError,RuntimeRejected) as e:
            self.feedback={**self.feedback,'model_or_runtime_error':str(e),'remaining_actions':self.runtime.remaining}
            result={'ok':False,'model':name,'decision_kind':type(decision).__name__ if decision else None,
                    'model_ms':(time.perf_counter()-start)*1000,'error':str(e),'feedback':deepcopy(self.feedback)}
        result['decision']=asdict(decision) if decision is not None else None
        self.history.append(result);self.save();return result

    def run(self,max_turns=None):
        if max_turns is not None and (type(max_turns) is not int or max_turns<0):
            raise ValueError('Expected a nonnegative model turn limit')
        remaining=self.max_decisions-self.decisions
        limit=remaining if max_turns is None else min(max_turns,remaining)
        results=[]
        for _ in range(limit):
            if self.runtime.remaining<1:break
            value=self.step();results.append(value)
            if value['ok'] and self.feedback.get('stopped'):break
        return results

    def save(self):
        path=self.runtime.root/'model-events.json'
        path.write_text(json.dumps({'schema':'model-session-events-v1','model_decisions':self.decisions,
            'host_provider_calls':self.provider_calls,'device_model_inferences':self.device_inferences,
            'switches':self.switches,'events':self.history},indent=2,allow_nan=False),encoding='utf-8')
