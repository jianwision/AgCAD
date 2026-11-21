"""Persistent Rust/CUDA backend. Host orchestration never chooses geometry actions."""
import json
import math
from copy import deepcopy
import queue
from pathlib import Path
import struct
import subprocess
import threading
import time
import uuid
import numpy as np
from session_protocol import (Action, CONSTRAINT_BITS, DeviceRollout, ObjectHandle,
                              OperationProgram, REASONS, digest)

ROOT=Path(__file__).resolve().parents[1]
ERRORS={200:'INVALID_REQUEST',201:'STALE_HANDLE',202:'STALE_REVISION',203:'BUDGET_EXHAUSTED',
        204:'OBJECT_CAPACITY',205:'NO_DRAFT',207:'REQUEST_LIMIT'}
META=['revision','actions_executed_total','draft_active','stopped','last_reason','commits',
      'rejections','uphill_commits','score','draft_score','generation','live','executed_this_call','constraint_flags',
      'reserved_14','reserved_15']


class RuntimeRejected(RuntimeError):
    def __init__(self,code,remaining):
        self.code=ERRORS.get(code,f'WORKER_STATUS_{code}');self.remaining=remaining
        super().__init__(self.code)


def read_exact(pipe,count):
    chunks=[]
    while count:
        value=pipe.read(count)
        if not value: raise EOFError('CUDA worker closed its protocol stream')
        chunks.append(value);count-=len(value)
    return b''.join(chunks)


class GeometryRuntime:
    backend='cuda-hood-v1'

    def __init__(self,root,weights,*,budget=256,slots=16,timeout=60.,goal_tolerance_mm=.5,require_goal_for_delivery=False):
        from diagnostic_suite import template, PROBES
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.weights=Path(weights).resolve();self.weights_sha256=digest(self.weights)
        self.session_id=uuid.uuid4().hex;self.timeout=timeout;self.events=[];self.views={}
        self._closed=False;self._lock=threading.Lock();self._queue=queue.Queue();self._live={}
        if not 1<=budget<=4096 or not 1<=slots<=64: raise ValueError('Invalid runtime budget/capacity')
        if not math.isfinite(timeout) or timeout<=0: raise ValueError('Invalid worker response timeout')
        if not math.isfinite(goal_tolerance_mm) or goal_tolerance_mm<=0: raise ValueError('Invalid user goal tolerance')
        if not isinstance(require_goal_for_delivery,bool): raise ValueError('Expected final goal requirement flag')
        self.goal_tolerance_mm=goal_tolerance_mm;self.require_goal_for_delivery=require_goal_for_delivery
        self.slots=slots;self.remaining=budget;self.budget=budget
        data,e,w=template();weights=np.fromfile(self.weights,dtype='<f8')
        if weights.shape!=(462,) or not np.isfinite(weights).all(): raise ValueError('Expected probe-policy-v1 weights')
        packed=struct.pack('<8s4I',b'GPCAD001',1,slots,budget,0)
        packed+=b''.join(np.asarray(v,dtype='<f8').tobytes() for v in [data['x0'],w,e,weights])
        (self.root/'template.bin').write_bytes(packed)
        self.scene={'backend':self.backend,'object_kind':'bicubic_patch_set','units':'mm','probe_uv':PROBES,
            'reference_poles':data['x0'].reshape(4,4,4,3).tolist(),
            'coordinates':'six physical probe displacements from reference, in mm',
            'constraints':{'fixed_xy':{'phase':'commit','tolerance_mm':1e-7},
                'fixed_perimeter':{'phase':'commit','tolerance_mm':1e-7},
                'compatible_C2_seams':{'phase':'commit','tolerance':1e-7},
                'mirror_symmetry':{'phase':'commit','tolerance_mm':1e-7},
                'numerical_domain':{'phase':'every_operation','absolute_coordinate_limit_mm':1e6}},
            'initial_and_target_probe_limit_mm':1e4,
            'goal_criterion':{'metric':'max_probe_error_mm','tolerance_mm':goal_tolerance_mm,
                             'required_for_delivery':require_goal_for_delivery,'phase':'final'},
            'goal_is_immutable':True,'quality_improvement_required_for_commit':False,
            'device_models':{'probe-policy-v1':{'weights_sha256':self.weights_sha256,'learned_operations':['ADD_PROBES']}}}
        (self.root/'scene.json').write_text(json.dumps(self.scene,indent=2),encoding='utf-8')
        executable=ROOT/'target/release/cad_worker.exe'
        self._stderr_log=(self.root/'worker-stderr.log').open('wb')
        try:
            self.process=subprocess.Popen([str(executable),str(self.root)],cwd=ROOT,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr_log)
        except Exception:
            self._stderr_log.close();raise
        self.pid=self.process.pid
        def receive():
            try:
                while True:
                    h=read_exact(self.process.stdout,32)
                    magic,status,slot,generation,revision,remaining,words=struct.unpack('<8s6I',h)
                    if magic!=b'CADRSP01' or words>1920: raise ValueError('Invalid worker response frame')
                    body=np.frombuffer(read_exact(self.process.stdout,8*words),dtype='<f8').copy()
                    self._queue.put((status,slot,generation,revision,remaining,body))
            except Exception as e: self._queue.put(e)
        self._reader=threading.Thread(target=receive,daemon=True);self._reader.start()
        try:
            ready=self._response()
            if ready[0]!=100: raise RuntimeError('Expected worker readiness')
            self.ready={'requested_device_bytes_excludes_context':int(ready[-1][4]),'compile_ms':float(ready[-1][5]),'pid':self.pid}
        except Exception:
            self._dispose();raise

    def _response(self):
        try: response=self._queue.get(timeout=self.timeout)
        except queue.Empty:
            self.process.kill();self.process.wait(timeout=5)
            raise TimeoutError('CUDA worker response timeout')
        if isinstance(response,Exception):
            if self.process.poll() is None:self.process.kill();self.process.wait(timeout=5)
            self._stderr_log.flush()
            detail=(self.root/'worker-stderr.log').read_text(encoding='utf-8',errors='replace')[-3000:]
            raise RuntimeError(f'{response}: {detail}')
        return response

    def _exchange(self,op,handle=None,*,slot=None,other=0,steps=0,values=None,flags=0,payload=b''):
        if self._closed: raise RuntimeError('Runtime is closed')
        if handle is not None:
            if handle.session_id!=self.session_id: raise ValueError('Object belongs to another session')
            if slot is None: slot=handle.slot
        if slot is None: slot=0
        values=np.zeros(12) if values is None else np.asarray(values,dtype=float)
        if values.shape!=(12,) or not np.isfinite(values).all(): raise ValueError('Expected twelve finite command values')
        packet=struct.pack('<8s10I12d16x',b'CADREQ01',1,op,slot,other,
            handle.revision if handle else 0,handle.generation if handle else 0,steps,1,flags,0,*values)+payload
        start=time.perf_counter()
        with self._lock:
            self.process.stdin.write(packet);self.process.stdin.flush()
            status,rs,gen,rev,remaining,body=self._response()
        self.remaining=remaining
        event={'operation':op,'slot':slot,'status':status,'remaining':remaining,'request_bytes':len(packet),
               'response_bytes':32+8*len(body),'wall_ms':(time.perf_counter()-start)*1000}
        self.events.append(event)
        if status: raise RuntimeRejected(status,remaining)
        return ObjectHandle(self.session_id,rs,gen,rev),body

    def _feedback(self,handle,body,steps=0):
        state={key:float(body[18+i]) for i,key in enumerate(META)}
        reason=int(state['last_reason'])
        flags=int(body[35] if state['draft_active'] else body[34])
        rejected_flags=int(state['constraint_flags']) if reason==3 else flags
        feedback={'object_id':handle.object_id,'revision':handle.revision,
            'committed_probes':body[:6].tolist(),'draft_probes':body[6:12].tolist(),'target':body[12:18].tolist(),
            'state':state,'reason':REASONS[reason] if reason<len(REASONS) else str(reason),
            'violated_constraints':[name for bit,name in CONSTRAINT_BITS.items() if rejected_flags&bit],
            'active_geometry_violations':[name for bit,name in CONSTRAINT_BITS.items() if flags&bit],
            'committed_valid':bool(body[34]==0),'draft_valid':bool(body[35]==0) if state['draft_active'] else None,
            'probe_max_error_mm':float(body[36]),'goal_met':bool(body[36]<=self.goal_tolerance_mm),
            'stopped':bool(state['stopped']),'remaining_actions':self.remaining,
            'trace':body[64:64+26*steps].reshape(steps,26).tolist() if steps else []}
        feedback['all_actions_admitted']=all(int(row[1]) in (0,4,5) for row in feedback['trace'])
        self.views[handle.object_id]=feedback
        if state['live']: self._live[handle.slot]=handle
        return feedback

    def _free_slot(self):
        for i in range(self.slots):
            if i not in self._live: return i
        raise RuntimeRejected(204,self.remaining)

    def create(self,initial,target):
        initial,target=np.asarray(initial,dtype=float),np.asarray(target,dtype=float)
        if initial.shape!=(6,) or target.shape!=(6,): raise ValueError('Initial and target require six measurements')
        handle,body=self._exchange(0,slot=self._free_slot(),values=np.r_[initial,target])
        feedback=self._feedback(handle,body)
        if not feedback['state']['live']: raise ValueError('Initial geometry violates hood contract')
        return handle,feedback

    def execute(self,handle,decision):
        if isinstance(decision,OperationProgram):
            steps=len(decision.actions)
            h,body=self._exchange(1,handle,steps=steps,flags=int(decision.continue_on_rejection),
                                  payload=b''.join(a.packed() for a in decision.actions))
        elif isinstance(decision,DeviceRollout):
            if decision.model_id!='probe-policy-v1' or decision.weights_sha256!=self.weights_sha256:
                raise ValueError('Unsupported device model or weight identity')
            steps=decision.steps;h,body=self._exchange(2,handle,steps=steps)
        else: raise TypeError('Expected OperationProgram or DeviceRollout')
        return h,self._feedback(h,body,steps)

    def observe(self,handle,*,geometry=False,draft=False):
        h,body=self._exchange(5 if geometry else 4,handle,flags=int(draft))
        feedback=self._feedback(h,body)
        if geometry: feedback['poles']=body[64:].reshape(4,4,4,3).tolist();feedback['geometry_view']='draft' if draft else 'committed'
        return h,feedback

    def fork(self,source):
        # Source is the committed view, even when its private draft is open.
        destination=self._free_slot()
        h,body=self._exchange(3,source,other=source.slot,slot=destination)
        return h,self._feedback(h,body)

    def drop(self,handle):
        h,body=self._exchange(6,handle);self._live.pop(handle.slot,None)
        return self._feedback(h,body)

    def latest(self,handle):
        if handle.session_id!=self.session_id: raise ValueError('Object belongs to another session')
        value=self._live.get(handle.slot)
        if value is None or value.generation!=handle.generation: raise ValueError('Unknown or stale scene object')
        return value

    def objects(self): return tuple(self._live.values())

    def cached_view(self,handle):
        if self.latest(handle)!=handle: raise ValueError('Stale cached scene revision')
        return deepcopy(self.views[handle.object_id])

    def export_step(self,handle,path,*,require_goal=None):
        from diagnostic_geometry import hood_semantics
        from diagnostic_suite import export_step
        handle,feedback=self.observe(handle,geometry=True)
        if feedback['state']['draft_active']: raise ValueError('OPEN_DRAFT: commit or abort before delivery')
        if (self.require_goal_for_delivery or require_goal) and not feedback['goal_met']:
            raise ValueError('Final user goal criterion failed')
        poles=np.asarray(feedback['poles']).reshape(192)
        independent=hood_semantics(poles)
        if not independent['pass']: raise ValueError('Independent final geometry check failed')
        result=export_step(poles,destination=path)
        return {**result,'object_id':handle.object_id,'revision':handle.revision,
                'goal_met':feedback['goal_met'],'independent_semantics':independent}

    def close(self):
        if self._closed:return
        try:
            if self.process.poll() is None:
                self._exchange(7)
                self.process.wait(timeout=10)
            if self.process.returncode!=0: raise RuntimeError('CUDA worker failed')
        finally:
            self._dispose()

    def _dispose(self):
        self._closed=True
        if self.process.poll() is None:self.process.kill();self.process.wait(timeout=5)
        try:self.process.stdin.close()
        except BrokenPipeError:pass
        self._reader.join(timeout=1);self.process.stdout.close();self._stderr_log.close()
        (self.root/'ipc-events.json').write_text(json.dumps(self.events,indent=2),encoding='utf-8')

    def __enter__(self): return self
    def __exit__(self,kind,value,tb):
        if kind is None:self.close()
        else:
            try:self.close()
            except Exception as cleanup_error:value.add_note(f'Worker cleanup: {cleanup_error}')
