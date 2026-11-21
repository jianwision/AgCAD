"""Versioned semantic decisions, independent of model provider and transport."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import re
from copy import deepcopy

OPERATIONS={'ADD_PROBES':0,'SET_PROBES':1,'RAW_CONTROL_DELTA':2,'BEGIN':3,
            'COMMIT':4,'ABORT':5,'STOP':6,'MEASURE':7,'BLEND_REFERENCE':8}
REASONS=['ACCEPTED','UNSUPPORTED_OPERATION','NONFINITE','CONSTRAINT_FAILED','STOPPED',
         'NOT_RUN','NO_DRAFT','OPEN_DRAFT','STALE_REVISION','STALE_HANDLE','INVALID_PARAMETER']
CONSTRAINT_BITS={1:'numerical_domain',2:'fixed_xy',4:'fixed_perimeter',8:'compatible_C2_seams',16:'mirror_symmetry'}
OPERATION_DETAILS={
    'ADD_PROBES':{'values':'six probe displacement increments in mm','effect':'add the compatible lifting W*values to active geometry'},
    'SET_PROBES':{'values':'six absolute probe displacements from reference in mm','effect':'lift values minus current probes; raw draft constraint violations are retained'},
    'RAW_CONTROL_DELTA':{'values':'values[0] is the coordinate increment in mm; remaining values unused',
        'index':'0..191, flattened [patch,u,v,xyz], index=48*patch+12*u+3*v+xyz',
        'effect':'edit one coordinate; can violate geometric constraints inside a draft'},
    'BEGIN':{'values':'unused','effect':'copy committed geometry into a private draft; reject if already open'},
    'COMMIT':{'values':'unused','effect':'check draft constraints and commit; failure retains the draft for repair'},
    'ABORT':{'values':'unused','effect':'discard the draft and preserve committed geometry'},
    'STOP':{'values':'unused','effect':'end this model run; reject with an open draft; does not imply goal success'},
    'MEASURE':{'values':'unused','effect':'return summary measurements without geometry changes'},
    'BLEND_REFERENCE':{'values':'values[0] in [0,1], remaining values unused',
        'effect':'subtract values[0] times current probes through W; raw draft violations are retained'}}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value,lower,upper):
    if isinstance(value,bool) or not isinstance(value,int) or not lower<=value<=upper:
        raise ValueError(f'Expected integer in [{lower}, {upper}]')
    return value


@dataclass(frozen=True)
class ObjectHandle:
    session_id:str
    slot:int
    generation:int
    revision:int

    @property
    def object_id(self): return f'{self.session_id}:{self.slot}:{self.generation}'


@dataclass(frozen=True)
class Action:
    op:str
    values:tuple[float,...]=(0.,)*6
    index:int=0

    def __post_init__(self):
        object.__setattr__(self,'values',tuple(self.values))
        if self.op not in OPERATIONS: raise ValueError(f'Unsupported operation: {self.op}')
        if len(self.values)!=6 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in self.values):
            raise ValueError('Action needs six finite numeric values')
        integer(self.index,0,191)
        if self.op=='BLEND_REFERENCE' and not 0<=self.values[0]<=1: raise ValueError('Blend fraction outside [0,1]')

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)-{'op','values','index'} or 'op' not in value:
            raise ValueError('Invalid action schema')
        return cls(value['op'],tuple(value.get('values',(0.,)*6)),value.get('index',0))

    def packed(self): return struct.pack('<2I6d2I',OPERATIONS[self.op],self.index,*self.values,0,0)

    def to_dict(self): return {'op':self.op,'values':list(self.values),'index':self.index}


@dataclass(frozen=True)
class OperationProgram:
    actions:tuple[Action,...]
    continue_on_rejection:bool=False

    def __post_init__(self):
        object.__setattr__(self,'actions',tuple(self.actions))
        if not 1<=len(self.actions)<=64 or not all(isinstance(a,Action) for a in self.actions):
            raise ValueError('Program requires 1..64 typed actions')
        if not isinstance(self.continue_on_rejection,bool): raise ValueError('Invalid error policy')


@dataclass(frozen=True)
class DeviceRollout:
    steps:int
    model_id:str
    weights_sha256:str

    def __post_init__(self): integer(self.steps,1,64)


@dataclass(frozen=True)
class Inspect:
    geometry:bool=False
    draft:bool=False

    def __post_init__(self):
        if not isinstance(self.geometry,bool) or not isinstance(self.draft,bool): raise ValueError('Inspection flags must be boolean')


@dataclass(frozen=True)
class Fork:
    select:bool=False

    def __post_init__(self):
        if not isinstance(self.select,bool): raise ValueError('Fork select flag must be boolean')


@dataclass(frozen=True)
class SelectObject:
    object_id:str


@dataclass(frozen=True)
class DropObject:
    object_id:str


@dataclass(frozen=True)
class ToolCall:
    op:str
    args:dict
    out:str

    def __post_init__(self):
        if not isinstance(self.op,str) or not isinstance(self.args,dict) or not isinstance(self.out,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}',self.out):
            raise ValueError('Invalid typed tool call')
        raw=json.dumps(self.args,allow_nan=False)
        if len(raw)>1<<20:raise ValueError('Tool arguments exceed capacity')
        object.__setattr__(self,'args',deepcopy(self.args))


@dataclass(frozen=True)
class ToolProgram:
    calls:tuple[ToolCall,...]

    def __post_init__(self):
        object.__setattr__(self,'calls',tuple(self.calls))
        if not 1<=len(self.calls)<=32 or not all(isinstance(c,ToolCall) for c in self.calls):raise ValueError('Expected 1..32 typed tool calls')


def parse_decision(response,observation):
    if isinstance(response,str): response=json.loads(response)
    if not isinstance(response,dict): raise ValueError('Model response must be an object')
    required={'schema','kind','object_id','scene_revision'}
    if not required<=set(response) or response['schema']!='cad-decision-v1': raise ValueError('Invalid decision envelope')
    integer(response['scene_revision'],0,2**32-1)
    if response['object_id']!=observation['object']['id'] or response['scene_revision']!=observation['object']['revision']:
        raise ValueError('Stale object or scene revision')
    kind=response['kind']
    if kind=='program':
        if set(response)-required-{'actions','on_error'}: raise ValueError('Model cannot override scene, goal, constraints or resources')
        if not isinstance(response.get('actions'),list): raise ValueError('Expected action list')
        policy=response.get('on_error','yield')
        if policy not in ('yield','continue'): raise ValueError('Unknown program error policy')
        return OperationProgram(tuple(Action.from_dict(a) for a in response['actions']),policy=='continue')
    if kind=='device_rollout':
        if set(response)!=required|{'steps','model_id','weights_sha256'}: raise ValueError('Invalid rollout fields')
        return DeviceRollout(response['steps'],response['model_id'],response['weights_sha256'])
    if kind=='inspect':
        if set(response)-required-{'geometry','draft'}: raise ValueError('Unexpected inspection fields')
        return Inspect(response.get('geometry',False),response.get('draft',False))
    if kind=='fork':
        if set(response)-required-{'select'}: raise ValueError('Unexpected fork fields')
        return Fork(response.get('select',False))
    if kind=='tools':
        if set(response)!=required|{'calls'} or not isinstance(response['calls'],list):raise ValueError('Invalid tool program fields')
        calls=[]
        for value in response['calls']:
            if not isinstance(value,dict) or set(value)!={'op','args','out'}:raise ValueError('Invalid tool call schema')
            calls.append(ToolCall(**value))
        return ToolProgram(tuple(calls))
    if kind in ('select','drop'):
        key='selected_object_id' if kind=='select' else 'drop_object_id'
        if set(response)!=required|{key} or not isinstance(response[key],str): raise ValueError('Invalid object selection')
        return SelectObject(response[key]) if kind=='select' else DropObject(response[key])
    raise ValueError('Unsupported model decision kind')


def decision_envelope(observation,kind='program',**body):
    return {'schema':'cad-decision-v1','object_id':observation['object']['id'],
            'scene_revision':observation['object']['revision'],'kind':kind,**body}
