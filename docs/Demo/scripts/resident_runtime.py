"""ModelSession backend for resident surfaces and joint Class-A tool programs."""
from copy import deepcopy
import json
import numpy as np
from engineering_runtime import EngineeringRuntime
from nurbs_surface import NurbsPatch
from resident_surface import SurfaceWorker,SurfaceRef
from surface_network import network_seam_report
import tolerance_topology as topo

RESIDENT_TOOLS={
    'nurbs.create':{'args':['poles','weights','knots_u','knots_v','degree_u','degree_v'],'result':'resident_surface','backend':'CPU definition, CUDA immutable upload'},
    'surface.store':{'args':['patch'],'result':'resident_surface','backend':'CUDA arena'},
    'surface.refine':{'args':['patch','axis','value','multiplicity?'],'result':'resident_surface','backend':'CUDA homogeneous knot insertion; CPU map compilation'},
    'surface.analyze':{'args':['patch','samples?'],'result':'quality_report','backend':'resident CUDA jets, compact summary'},
    'surface.adaptive':{'args':['patch','curvature_range_per_mm?','normal_angle_rad?','max_depth?','max_samples?'],'result':'quality_report','backend':'CUDA point evaluation with CPU adaptive sampling; explicit diagnostic readbacks'},
    'surface.network_optimize':{'args':['patches','seams','locks?','iterations?','samples?','step_mm?','max_displacement_mm?','length_scale_mm?','weights?'],'result':'surface_network','backend':'CPU joint nullspace compilation; resident CUDA graph search'},
    'surface.network_patch':{'args':['network','index'],'result':'resident_surface','backend':'device handle selection'},
    'surface.network_check':{'args':['network','samples?'],'result':'seam_report','backend':'explicit CPU boundary, independent seam jets'},
}


class ResidentEngineeringRuntime(EngineeringRuntime):
    backend='resident-engineering-v2'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.worker=SurfaceWorker(self.root/'gpu');self.pid=self.worker.process.pid
        self.scene.update(gpu_execution='persistent CUDA context, immutable device objects, cached basis tables and executable graphs',
                          device_models={},geometry_readback='explicit inspection or CPU topology/export boundary')
        (self.root/'scene.json').write_text(json.dumps(self.scene,indent=2),encoding='utf-8')

    def model_capabilities(self,handle):
        result=super().model_capabilities(handle);result['tools'].update(deepcopy(RESIDENT_TOOLS));result['resident_surface_capacity']=128
        result['network_scope']='2..6 polynomial Bezier patches, 384 poles total, fixed affine boundary maps'
        result['nurbs_scope']='positive weights, clamped nonperiodic degree 1..7, 32x32 poles'
        return result

    def _describe(self,record,geometry=False):
        if record['kind']=='resident_surface':
            ref=record['value'];value={'kind':'resident_surface','device_handle':{'slot':ref.slot,'generation':ref.generation},
                'degrees':list(ref.layout.degrees),'shape':list(ref.layout.poles.shape[:2]),'resident':True,'evidence':deepcopy(record.get('evidence',{}))}
            if geometry:value['poles']=self.worker.fetch(ref).poles.tolist();value['geometry_readback_requested']=True
            return value
        if record['kind']=='surface_network':
            network=record['value'];return {'kind':'surface_network','patch_count':len(network['patches']),'report':deepcopy(network['report'])}
        return super()._describe(record,geometry)

    def _call(self,call,objects,document):
        args=deepcopy(call.args);op=call.op
        if op not in RESIDENT_TOOLS and op not in ('topology.face','topology.trim'):return super()._call(call,objects,document)
        if op in RESIDENT_TOOLS:
            schema=RESIDENT_TOOLS[op];required={k for k in schema['args'] if not k.endswith('?')};allowed={k.rstrip('?') for k in schema['args']}
            if set(args)-allowed or not required<=set(args):raise ValueError('Invalid arguments for '+op)
        def take(name,kind):
            if not isinstance(name,str) or name not in objects or objects[name]['kind']!=kind:raise ValueError('Unknown/wrong-type reference: '+str(name))
            return objects[name]['value']
        evidence={}
        if op=='nurbs.create':value=self.worker.upload(NurbsPatch(**args))
        elif op=='surface.store':value=self.worker.upload(take(args['patch'],'bezier_patch'))
        elif op=='surface.refine':
            ref=take(args.pop('patch'),'resident_surface');value=self.worker.refine(ref,**args)
        elif op in ('surface.analyze','surface.adaptive'):
            if op=='surface.analyze' and objects.get(args['patch'],{}).get('kind')=='bezier_patch':
                ref=self.worker.upload(take(args.pop('patch'),'bezier_patch'))
            else:ref=take(args.pop('patch'),'resident_surface')
            value=(self.worker.analyze if op=='surface.analyze' else self.worker.adaptive_quality)(ref,**args)
        elif op=='surface.network_optimize':
            refs=[take(name,'resident_surface') for name in args.pop('patches')];seams=args.pop('seams')
            outputs,report=self.worker.optimize_network(refs,seams,**args)
            value={'patches':outputs,'seeds':refs,'seams':seams,'locks':args.get('locks'),'report':report}
        elif op=='surface.network_patch':
            network=take(args['network'],'surface_network');index=args['index']
            if type(index) is not int or not 0<=index<len(network['patches']):raise ValueError('Invalid network patch index')
            value=network['patches'][index]
        elif op=='surface.network_check':
            network=take(args.pop('network'),'surface_network')
            value=network_seam_report([self.worker.fetch(ref) for ref in network['patches']],network['seams'],**args)
        else:
            if objects.get(args.get('patch'),{}).get('kind')!='resident_surface':return super()._call(call,objects,document)
            ref=take(args.pop('patch'),'resident_surface');patch=self.worker.fetch(ref)
            if op=='topology.face':
                if args:raise ValueError('Unexpected face arguments')
                from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
                value=BRepBuilderAPI_MakeFace(patch.native(),1e-7).Face();self.policy.resolve([value]);evidence=topo.inspect_shape(value,self.policy)
            else:value,evidence=topo.trim(patch,policy=self.policy,**args)
            return {'kind':'shape','value':value,'evidence':{**evidence,'cpu_geometry_boundary':True}}
        return {'kind':RESIDENT_TOOLS[op]['result'],'value':value,'evidence':evidence}

    def _collect(self):
        reachable=set()
        def walk(value):
            if isinstance(value,SurfaceRef):reachable.add(value.key)
            elif isinstance(value,dict):
                for v in value.values():walk(v)
            elif isinstance(value,(list,tuple)):
                for v in value:walk(v)
        for document in self._documents.values():walk(document['objects'])
        for ref in tuple(self.worker._live.values()):
            if ref.key not in reachable:self.worker.drop(ref)

    def execute(self,*args,**kwargs):
        try:return super().execute(*args,**kwargs)
        finally:self._collect()

    def drop(self,*args,**kwargs):
        result=super().drop(*args,**kwargs);self._collect();return result

    def export_step(self,handle,path,*,shape='result'):
        objects=self._document(handle)['objects'];checks=[]
        if shape not in objects or objects[shape]['kind']!='shape':raise ValueError('No shape selected for delivery')
        delivered=objects[shape]['value']
        for rule in self.rules.get('networks',[]):
            network=objects[rule['network']]['value'];patches=[self.worker.fetch(r) for r in network['patches']]
            seeds=[self.worker.fetch(r) for r in network['seeds']];tolerance=rule.get('tolerance_mm',1e-6)
            used=set();faces=topo.subshapes(delivered,topo.TopAbs_FACE)
            for patch in patches:
                found=None
                for i,face in enumerate(faces):
                    if i in used:continue
                    native=topo.BRep_Tool.Surface_s(face)
                    if not hasattr(native,'NbUPoles') or (native.NbUPoles(),native.NbVPoles())!=patch.poles.shape[:2] or (native.UDegree(),native.VDegree())!=patch.degrees:continue
                    # B-spline support identity includes weights, knots and multiplicities.
                    if not hasattr(native,'UKnot'):continue
                    ku=np.repeat([native.UKnot(j) for j in range(1,native.NbUKnots()+1)],[native.UMultiplicity(j) for j in range(1,native.NbUKnots()+1)])
                    kv=np.repeat([native.VKnot(j) for j in range(1,native.NbVKnots()+1)],[native.VMultiplicity(j) for j in range(1,native.NbVKnots()+1)])
                    if ku.shape!=patch.knots_u.shape or kv.shape!=patch.knots_v.shape or not np.allclose(ku,patch.knots_u,atol=1e-13,rtol=0) or not np.allclose(kv,patch.knots_v,atol=1e-13,rtol=0):continue
                    error=0.;weight_error=0.
                    for a,row in enumerate(patch.poles):
                        for b,p in enumerate(row):
                            q=native.Pole(a+1,b+1);error=max(error,float(np.linalg.norm(p-[q.X(),q.Y(),q.Z()])))
                            weight_error=max(weight_error,abs(native.Weight(a+1,b+1)-patch.weights[a,b]))
                    if error<=tolerance and weight_error<=1e-10:found=i;break
                if found is None:raise ValueError('Declared resident surface is absent from delivered shape')
                used.add(found)
            report=network_seam_report(patches,network['seams'],samples=rule.get('samples',67))
            if report['maximum_jet_residual_mm']>tolerance or report['maximum_normal_angle_rad']>rule.get('maximum_normal_angle_rad',1e-6):raise ValueError('Final joint seam constraints failed')
            displacement=max(float(np.max(np.linalg.norm(p.poles-s.poles,axis=2))) for p,s in zip(patches,seeds))
            if displacement>rule['maximum_control_displacement_mm']*(1+1e-10):raise ValueError('Final declared shape-deviation budget exceeded')
            if network['locks'] is not None:
                for p,s,mask in zip(patches,seeds,network['locks']):
                    if np.any(mask) and np.max(np.linalg.norm((p.poles-s.poles)[np.asarray(mask,dtype=bool)],axis=1))>tolerance:raise ValueError('Final locked control points moved')
            quality=[self.worker.analyze(r,samples=rule.get('quality_samples',25)) for r in network['patches']]
            if rule.get('require_sampled_regular',True) and not all(q['all_samples_regular'] for q in quality):raise ValueError('Final surface regularity failed')
            if 'maximum_abs_curvature_per_mm' in rule and max(abs(k) for q in quality for k in q['principal_curvature_range_per_mm'])>rule['maximum_abs_curvature_per_mm']:raise ValueError('Final declared curvature limit exceeded')
            checks.append({'kind':'resident_network','pass':True,'seams':report,'max_control_displacement_mm':displacement,'quality':quality})
        result=super().export_step(handle,path,shape=shape);result['user_contract_checks'].extend(checks);return result

    def close(self):
        if self._closed:return
        try:self.worker.close()
        finally:
            super().close()
            path=self.root/'runtime-report.json';report=json.loads(path.read_text(encoding='utf-8'))
            worker_path=self.root/'gpu/worker-report.json'
            if worker_path.exists():report['resident_gpu']=json.loads(worker_path.read_text(encoding='utf-8'))
            report['resident_client_final_stats']=getattr(self.worker,'final_stats',{})
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
