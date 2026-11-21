"""Persistent Rust/CUDA surface client with immutable device handles."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import uuid
import numpy as np
from nurbs_surface import NurbsPatch,basis_table,grid_uv
from surface_network import compile_network

ROOT=Path(__file__).resolve().parents[1]
STAT_NAMES=('context_creations','compilations','compile_cache_hit','compile_ms','requests','rejections','upload_bytes','download_bytes',
            'summary_readbacks','grid_readbacks','geometry_readbacks','basis_entries','basis_cache_hits','graph_builds','graph_cache_hits',
            'graph_launches','explicit_kernel_submissions','network_runs','basis_bytes','network_plan_uploads','network_plan_cache_hits')


@dataclass(frozen=True,eq=False)
class SurfaceRef:
    session_id:str
    slot:int
    generation:int
    layout:NurbsPatch

    @property
    def key(self):return self.session_id,self.slot,self.generation


def read_exact(pipe,count):
    result=bytearray()
    while len(result)<count:
        data=pipe.read(count-len(result))
        if not data:raise EOFError('Resident surface worker closed its pipe')
        result.extend(data)
    return bytes(result)


def _summary(raw):
    count,regular,mink,maxk,area,curv,reflection,minjac=raw
    return {'scope':'sampled NURBS differential diagnostics','samples':int(count),'regular_samples':int(regular),
            'all_samples_regular':count==regular,'principal_curvature_range_per_mm':[mink,maxk],
            'sampled_area_mm2':area,'mean_curvature_gradient_squared_per_mm4':curv,
            'reflection_gradient_squared_per_mm2':reflection,'minimum_normalized_jacobian':minjac,
            'global_class_a_certified':False,'global_self_intersection_status':'unknown'}


class SurfaceWorker:
    def __init__(self,directory,*,timeout=120):
        self.root=Path(directory).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.session_id=uuid.uuid4().hex;self.timeout=timeout;self._closed=False;self._live={};self._basis={};self._uploads={};self._materialized={}
        self.events=[];self.basis_cache_hits=0;self._plans={};self.plan_cache_hits=0;self._queue=queue.Queue();self._stderr=(self.root/'stderr.log').open('wb')
        self.process=subprocess.Popen([str(ROOT/'target/release/surface_worker.exe'),str(self.root)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr)
        def reader():
            try:
                while True:
                    h=read_exact(self.process.stdout,32)
                    if h[:8]!=b'SURFRSP1':raise RuntimeError('Invalid worker reply')
                    status,slot,generation,revision,count,reserved=struct.unpack('<6I',h[8:])
                    if count>1_048_576 or revision or reserved:raise RuntimeError('Invalid reply dimensions')
                    body=np.frombuffer(read_exact(self.process.stdout,count*8),dtype='<f8').copy()
                    self._queue.put((status,slot,generation,body))
            except Exception as error:self._queue.put(error)
        self._reader=threading.Thread(target=reader,daemon=True);self._reader.start()
        try:
            status,_,_,body=self._receive()
            if status!=100:raise RuntimeError('Worker startup failed')
            self.startup=dict(zip(('context_creations','compilations','compile_cache_hit','compile_ms','capacity'),body.tolist()))
        except Exception:
            self._terminate();raise

    def _receive(self):
        try:result=self._queue.get(timeout=self.timeout)
        except queue.Empty:self._terminate();raise TimeoutError('Resident surface worker timed out')
        if isinstance(result,Exception):raise RuntimeError(str(result)+'; see '+str(self.root/'stderr.log'))
        return result

    def _request(self,op,ref=None,payload=()):
        if self._closed:raise ValueError('Surface worker closed')
        if ref is not None and (ref.session_id!=self.session_id or self._live.get(ref.slot) is not ref):raise ValueError('Stale/foreign surface handle')
        values=np.asarray(payload,dtype='<f8').ravel()
        if len(values)>1_048_576 or not np.isfinite(values).all():raise ValueError('Invalid worker payload')
        header=struct.pack('<8s6I',b'SURFREQ1',op,ref.slot if ref else 0xffffffff,ref.generation if ref else 0,0,len(values),0)
        start=time.perf_counter();self.process.stdin.write(header+values.tobytes());self.process.stdin.flush()
        status,slot,generation,body=self._receive()
        self.events.append({'operation':op,'request_bytes':len(header)+values.nbytes,'response_bytes':32+body.nbytes,'wall_ms':(time.perf_counter()-start)*1000,'accepted':status==0})
        if status:raise ValueError(''.join(chr(int(c)) for c in body))
        return slot,generation,body

    def _ref(self,slot,generation,layout):
        ref=SurfaceRef(self.session_id,slot,generation,layout);self._live[slot]=ref;return ref

    def upload(self,patch):
        if not isinstance(patch,NurbsPatch):patch=NurbsPatch.from_bezier(patch)
        data=np.concatenate((patch.poles,patch.weights[:,:,None]),axis=2)
        key=hashlib.sha256(data.tobytes()+json.dumps(patch.descriptor(),sort_keys=True).encode()).hexdigest()
        if key in self._uploads and self._live.get(self._uploads[key].slot) is self._uploads[key]:return self._uploads[key]
        slot,generation,_=self._request(1,payload=np.r_[patch.poles.shape[:2],data.ravel()]);ref=self._ref(slot,generation,patch)
        self._uploads[key]=ref;return ref

    def basis(self,ref,uv,quadrature=None):
        if ref.session_id!=self.session_id or self._live.get(ref.slot) is not ref:raise ValueError('Stale/foreign surface handle')
        layout=ref.layout
        signature=json.dumps({k:v for k,v in layout.descriptor().items() if k!='weights'},sort_keys=True).encode()+np.asarray(uv,dtype='<f8').tobytes()
        if quadrature is not None:signature+=np.asarray(quadrature,dtype='<f8').tobytes()
        key=hashlib.sha256(signature).hexdigest()
        if key in self._basis:self.basis_cache_hits+=1;return self._basis[key]
        table=basis_table(layout,uv,quadrature);header=[*layout.poles.shape[:2],*layout.degrees,len(table)]
        _,_,body=self._request(2,payload=np.r_[header,table.ravel()]);self._basis[key]=(int(body[0]),table);return self._basis[key]

    def analyze(self,ref,*,samples=17,full=False,uv=None,quadrature=None):
        if uv is None:uv,quadrature=grid_uv(samples)
        basis,_=self.basis(ref,uv,quadrature)
        _,_,raw=self._request(4 if full else 3,ref,[basis])
        if full:return raw.reshape(len(uv),48)
        return _summary(raw.tolist())

    def fetch(self,ref):
        if ref.key in self._materialized:
            if self._live.get(ref.slot) is not ref:raise ValueError('Stale surface handle')
            return self._materialized[ref.key]
        _,_,raw=self._request(5,ref);layout=ref.layout;data=raw.reshape(*layout.poles.shape[:2],4)
        patch=NurbsPatch(data[:,:,:3],data[:,:,3],layout.knots_u,layout.knots_v,*layout.degrees)
        self._materialized[ref.key]=patch;return patch

    def fork(self,ref):
        slot,generation,_=self._request(6,ref);return self._ref(slot,generation,ref.layout)

    def refine(self,ref,axis,value,multiplicity=1):
        from scipy.interpolate import BSpline
        layout=ref.layout;refined=layout.insert_knot(axis,value,multiplicity)
        knots=(layout.knots_u,layout.knots_v)[axis];n=layout.poles.shape[axis]
        matrix=BSpline(knots,np.eye(n),layout.degrees[axis]).insert_knot(value,multiplicity).c
        operator=np.kron(matrix,np.eye(layout.poles.shape[1])) if axis==0 else np.kron(np.eye(layout.poles.shape[0]),matrix)
        slot,generation,_=self._request(10,ref,np.r_[refined.poles.shape[:2],layout.poles.shape[:2],operator.ravel()])
        return self._ref(slot,generation,refined)

    def drop(self,ref):
        self._request(7,ref);self._live.pop(ref.slot);self._materialized.pop(ref.key,None)

    def stats(self):
        _,_,values=self._request(9);return {**dict(zip(STAT_NAMES,values.tolist())),'host_basis_cache_hits':self.basis_cache_hits,'host_constraint_plan_cache_hits':self.plan_cache_hits}

    def optimize_network(self,refs,seams,*,locks=None,iterations=128,samples=9,step_mm=.5,max_displacement_mm=2.,length_scale_mm=100.,weights=None):
        if type(iterations) is not int or not 0<=iterations<=512 or type(samples) is not int or not 3<=samples<=25:
            raise ValueError('Invalid network iteration/sample budget')
        if not all(math.isfinite(v) and 0<v<=limit for v,limit in ((step_mm,10),(max_displacement_mm,100),(length_scale_mm,1e6))):raise ValueError('Invalid network spatial budget')
        for ref in refs:
            if ref.session_id!=self.session_id or self._live.get(ref.slot) is not ref:raise ValueError('Stale/foreign network input')
        plan_key=json.dumps({'layouts':[r.layout.descriptor() for r in refs],'seams':seams,
                            'locks':None if locks is None else [np.asarray(m).tolist() for m in locks]},sort_keys=True)
        if plan_key in self._plans:plan=self._plans[plan_key];self.plan_cache_hits+=1
        else:
            plan=compile_network([r.layout for r in refs],seams,locks)
            if len(self._plans)>=64:self._plans.pop(next(iter(self._plans)))
            self._plans[plan_key]=plan
        uv,quad=grid_uv(samples);tables=[basis_table(r.layout,uv,quad) for r in refs]
        n=plan['projection'].shape[0];df=plan['nullspace'].shape[1]
        weights={'curvature':1.,'reflection':.05,'deviation':.01} if weights is None else dict(weights)
        if set(weights)!={'curvature','reflection','deviation'} or any(not math.isfinite(v) or not 0<=v<=100 for v in weights.values()) or not any(weights.values()):
            raise ValueError('Invalid objective weights')
        header=[len(refs),n,df,len(uv),iterations,step_mm,max_displacement_mm,length_scale_mm,weights['curvature'],weights['reflection'],weights['deviation'],len(plan['compatibility']),0,0,0,0]
        meta=[]
        for i,ref in enumerate(refs):meta.extend([ref.slot,ref.generation,plan['offsets'][i],*ref.layout.poles.shape[:2],*ref.layout.degrees,0])
        _,_,raw=self._request(8,payload=np.r_[header,meta,plan['projection'].ravel(),plan['nullspace'].ravel(),plan['compatibility'].ravel(),np.array(tables).ravel()])
        outputs=[self._ref(int(raw[18+2*i]),int(raw[19+2*i]),ref.layout) for i,ref in enumerate(refs)]
        def metrics(v):return dict(zip(('objective','max_control_displacement_mm','curvature_gradient_squared_per_mm4','reflection_gradient_squared_per_mm2','mean_sampled_deviation_squared_mm2','regular_samples','sampled_area_mm2','reserved'),v.tolist()))
        report={**plan['report'],'iterations':int(raw[0]),'accepted_steps':int(raw[1]),'before':metrics(raw[2:10]),'after':metrics(raw[10:18]),
                'weights':weights,'length_scale_mm':length_scale_mm,'samples_per_patch':len(uv),'max_displacement_mm':max_displacement_mm,
                'optimization_backend':'CUDA graph coordinate search in compiled joint feasible subspace',
                'intermediate_geometry_readbacks':0,'global_class_a_certified':False}
        return outputs,report

    def adaptive_quality(self,ref,*,curvature_range_per_mm=.005,normal_angle_rad=.1,max_depth=4,max_samples=2000):
        if type(max_depth) is not int or not 0<=max_depth<=6 or type(max_samples) is not int or not 81<=max_samples<=4225:
            raise ValueError('Invalid adaptive sampling budget')
        if not all(math.isfinite(v) and v>0 for v in (curvature_range_per_mm,normal_angle_rad)):raise ValueError('Invalid adaptive thresholds')
        # Seed every knot span so a small knot span cannot disappear between grid samples.
        ku=np.unique(np.r_[np.linspace(0,1,5),ref.layout.knots_u]);kv=np.unique(np.r_[np.linspace(0,1,5),ref.layout.knots_v])
        cells=[(a,b,c,d,0) for a,b in zip(ku[:-1],ku[1:]) for c,d in zip(kv[:-1],kv[1:])]
        values={};rounds=0;unresolved=0;worst=[]
        while cells:
            requested={point for a,b,c,d,depth in cells for point in ((a,c),(a,d),(b,c),(b,d),((a+b)/2,(c+d)/2))}-values.keys()
            if len(values)+len(requested)>max_samples:unresolved+=len(cells);break
            if requested:
                uv=np.array(sorted(requested));grid=self.analyze(ref,full=True,uv=uv);values.update(zip(map(tuple,uv),grid));rounds+=1
            new=[]
            for a,b,c,d,depth in cells:
                points=((a,c),(a,d),(b,c),(b,d),((a+b)/2,(c+d)/2));rows=np.array([values[p] for p in points])
                spread=float(np.max(np.ptp(rows[:,36:38],axis=0)));dots=np.clip(rows[:,30:33]@rows[:,30:33].T,-1,1)
                angle=float(np.max(np.arccos(dots)));bad=not np.all(rows[:,42]==1)
                if bad or spread>curvature_range_per_mm or angle>normal_angle_rad:
                    if depth<max_depth:
                        u=(a+b)/2;v=(c+d)/2;new.extend((x,y,z,w,depth+1) for x,y in ((a,u),(u,b)) for z,w in ((c,v),(v,d)))
                    else:unresolved+=1;worst.append({'uv':[(a+b)/2,(c+d)/2],'curvature_range_per_mm':spread,'normal_angle_rad':angle,'regular':not bad})
            cells=new
        grid=np.array(list(values.values()))
        continuity=[]
        for knots,degree in ((ref.layout.knots_u,ref.layout.degree_u),(ref.layout.knots_v,ref.layout.degree_v)):
            internal=knots[(knots>0)&(knots<1)];continuity.extend((degree-int(n)) for n in np.unique(internal,return_counts=True)[1])
        return {'scope':'adaptive sampled diagnostics; no continuous-domain certificate','samples':len(values),'sampling_rounds':rounds,
                'unresolved_cells':unresolved,'status':'sampled_thresholds_met' if not unresolved else 'unknown_budget_or_depth_limit',
                'all_samples_regular':bool(len(grid) and np.all(grid[:,42]==1)),
                'principal_curvature_range_per_mm':[float(np.min(grid[:,36])),float(np.max(grid[:,37]))] if len(grid) else None,
                'interior_parametric_c3':all(c>=3 for c in continuity),'worst_cells':worst[:16],
                'diagnostic_grid_readbacks':rounds,'geometry_readbacks':0,'global_class_a_certified':False}

    def _terminate(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=10);self._reader.join(timeout=2)
        self.process.stdin.close();self.process.stdout.close();self._stderr.close();self._closed=True

    def close(self):
        if self._closed:return
        try:
            self.final_stats=self.stats();self._request(0);self.process.stdin.close();self.process.wait(timeout=10)
            if self.process.returncode:raise RuntimeError('Resident worker failed on shutdown')
        finally:
            if self.process.poll() is None:self.process.kill();self.process.wait(timeout=10)
            self._reader.join(timeout=2);self.process.stdin.close();self.process.stdout.close();self._stderr.close();self._closed=True
            (self.root/'client-events.json').write_text(json.dumps({'events':self.events,'host_basis_cache_hits':self.basis_cache_hits},indent=2),encoding='utf-8')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
