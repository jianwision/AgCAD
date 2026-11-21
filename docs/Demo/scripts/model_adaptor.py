"""One model role and proposal contract for CUDA models, LLMs and replay.

Provider responses cannot redefine goals/initial geometry/constraints. A live LLM
provider is injected as a callback; this module never sends network requests.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

OPCODES = {'ADD_PROBES':0,'SET_PROBES':1,'RESTORE_BEST':2,'RAW_CONTROL_DELTA':3,'STOP':4}


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    initial: tuple[float, ...]
    target: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self,'initial',tuple(self.initial))
        object.__setattr__(self,'target',tuple(self.target))
        if not self.request_id or len(self.initial)!=6 or len(self.target)!=6:
            raise ValueError('Model request requires id and six initial/target measurements')
        if not all(math.isfinite(v) for v in (*self.initial,*self.target)):
            raise ValueError('Model request measurements must be finite')


@dataclass(frozen=True)
class ModelContext:
    scene_reference: str
    constraint_contract: str = 'hood-fixed-xy-perimeter-C2-symmetry-v1'
    contract_revision: int = 1
    max_rounds: int = 16
    units: str = 'mm'


@dataclass(frozen=True)
class Proposal:
    op: str
    values: tuple[float, ...] = (0.,)*6

    def __post_init__(self):
        if self.op not in OPCODES or len(self.values)!=6 or not all(math.isfinite(v) for v in self.values):
            raise ValueError('Unsupported operation or invalid six-argument proposal')

    def packed(self):
        return [OPCODES[self.op], *self.values]


@dataclass(frozen=True)
class ProposalBatch:
    adaptor: str
    programs: tuple[dict, ...]
    host_model_calls: int
    device_model_inference: bool
    execution: str


class ModelAdaptor(ABC):
    """propose returns explicit actions OR a device-resident action producer.

    Both use identical operation and feedback semantics. This abstraction does
    not require a device model to materialize predictions on the host.
    """
    def propose(self, context: ModelContext, requests: tuple[ModelRequest, ...]) -> ProposalBatch:
        raise NotImplementedError('Legacy one-shot diagnostic interface is not supported by this model')

    @abstractmethod
    def decide(self, observation:dict):
        """Return a semantic session decision; models may retain their own context."""
        raise NotImplementedError

    @property
    def capabilities(self):
        return {'name':type(self).__name__,'inference_location':'custom','decisions':[]}


def observation(context, requests):
    return {'schema':'model-observation-v1','context':dict(context.__dict__),
            'requests':[dict(r.__dict__) for r in requests], 'operations':dict(OPCODES),
            'quality_feedback':'squared probe error; legal uphill moves allowed',
            'proposal_schema':{'contract':'hood-proposals-v1','contract_revision':context.contract_revision,
                'programs':[{'request_id':'id from requests','actions':[{'op':'ADD_PROBES','values':[0.]*6}]}]}}


def parse_response(response, context, requests):
    if isinstance(response,str): response=json.loads(response)
    if not isinstance(response,dict) or set(response)!={'contract','contract_revision','programs'}:
        raise ValueError('Response must contain contract, contract_revision and programs only')
    if response['contract']!='hood-proposals-v1' or response['contract_revision']!=context.contract_revision:
        raise ValueError('Stale or unknown model proposal contract')
    registry={r.request_id:r for r in requests}; seen=set(); programs=[]
    if not isinstance(response['programs'],list) or not response['programs']:
        raise ValueError('Model must return a nonempty program list')
    for item in response['programs']:
        if not isinstance(item,dict) or set(item)!={'request_id','actions'}:
            raise ValueError('Model cannot override target, initial state or constraint context')
        key=item['request_id']
        if key not in registry or key in seen: raise ValueError('Unknown or duplicate model request')
        seen.add(key); request=registry[key]
        if not isinstance(item['actions'],list) or not 1<=len(item['actions'])<=context.max_rounds:
            raise ValueError('Model program exceeds resource budget or is empty')
        actions=[]
        for a in item['actions']:
            if not isinstance(a,dict) or not {'op'}<=set(a)<= {'op','values'}:
                raise ValueError('Invalid action schema')
            values=tuple(a.get('values',(0.,)*6))
            actions.append(Proposal(a['op'],values).packed())
        programs.append({'name':'adapter_'+key,'request_id':key,'mode':'script',
                         'initial':list(request.initial),'target':list(request.target),'actions':actions})
    return tuple(programs)


class CuModel(ModelAdaptor):
    def __init__(self, weights: Path, steps=16):
        self.weights=Path(weights)
        from session_protocol import integer
        self.steps=integer(steps,1,64)

    @property
    def capabilities(self):
        return {'name':'CuModel','inference_location':'device','decisions':['device_rollout'],
                'learned_operations':['ADD_PROBES'],'model_id':'probe-policy-v1'}

    def decide(self,observation):
        from session_protocol import DeviceRollout, digest
        remaining=observation['resources']['remaining_actions']
        if remaining<1: raise ValueError('No action budget remains')
        return DeviceRollout(min(self.steps,remaining),'probe-policy-v1',digest(self.weights))

    def propose(self, context, requests):
        if not self.weights.is_file() or self.weights.stat().st_size!=462*8:
            raise ValueError('CUDA probe proposal model requires 462 f64 weights')
        programs=tuple({'name':'adapter_'+r.request_id,'request_id':r.request_id,'mode':'neural',
                        'initial':list(r.initial),'target':list(r.target),'actions':[]} for r in requests)
        return ProposalBatch('CuModel',programs,0,True,'device_resident_closed_loop')


class Llm(ModelAdaptor):
    """provider(observation_dict) -> structured proposal JSON, synchronous outer batch."""
    def __init__(self, provider: Callable[[dict],dict | str]):
        self.provider=provider

    @property
    def capabilities(self):
        return {'name':'Llm','inference_location':'host','decisions':['program','device_rollout','tools','inspect','fork','select','drop']}

    def decide(self,observation):
        from session_protocol import parse_decision
        return parse_decision(self.provider(deepcopy(observation)),observation)

    def propose(self, context, requests):
        result=self.provider(observation(context,requests))
        return ProposalBatch('Llm',parse_response(result,context,requests),1,False,
                             'host_llm_plan_then_device_batch')


class ReplayModel(ModelAdaptor):
    def __init__(self, response_path: Path):
        self.response_path=Path(response_path)
        self._cursor=0

    @property
    def capabilities(self):
        return {'name':'ReplayModel','inference_location':'replay','decisions':['program','device_rollout','tools','inspect','fork','select','drop']}

    def decide(self,observation):
        from session_protocol import Action, OperationProgram, decision_envelope, parse_decision
        document=json.loads(self.response_path.read_text(encoding='utf-8'))
        if document.get('schema')!='cad-replay-v1': return parse_decision(document,observation)
        if self._cursor>=len(document['decisions']): return OperationProgram((Action('STOP'),))
        body=deepcopy(document['decisions'][self._cursor]);self._cursor+=1
        kind=body.pop('kind')
        return parse_decision(decision_envelope(observation,kind,**body),observation)

    def propose(self, context, requests):
        programs=parse_response(json.loads(self.response_path.read_text(encoding='utf-8')),context,requests)
        return ProposalBatch('ReplayModel',programs,0,False,'recorded_llm_response_then_device_batch')


def load_adaptor(kind, weights, response_path=None):
    if kind=='cuda': return CuModel(weights)
    if kind=='replay' and response_path: return ReplayModel(response_path)
    raise ValueError('Use cuda or replay with --proposals; live LLM providers use Llm(callback)')
