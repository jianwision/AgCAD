"""Design-intent → G3/fairing diagnostics → wheel-opening trim → sewn STEP."""
import json
from pathlib import Path
import hashlib
import numpy as np
from class_a import BezierPatch,gpu_batch
from engineering_fixtures import intent_fixture,polynomial_patch
from engineering_runtime import EngineeringRuntime
from model_adaptor import Llm
from model_session import ModelSession
from session_protocol import decision_envelope

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/engineering/demo'


def call(op,out,**args):return {'op':op,'out':out,'args':args}


def delivery_rules():
    return {'require_intent':True,'coordinates':[
        {'patch':'left','u':1.,'v':0.,'axis':0,'expression':'x','tolerance_mm':1e-6},
        {'patch':'left','u':0.,'v':1.,'axis':1,'expression':'y','tolerance_mm':1e-6}],
        'seams':[{'left':'left','right':'right','order':3,'tolerance_mm':1e-6,
                  'maximum_normal_angle_rad':1e-6,'maximum_curvature_gap_per_mm':1e-6}],
        'quality':[{'patch':name,'samples':25,'criteria':{'require_sampled_regular':True,'maximum_abs_curvature_per_mm':.1}} for name in ('left','right')]}


def main():
    OUT.mkdir(parents=True,exist_ok=True);turns=[]
    def provider(obs):
        turns.append(obs)
        if len(turns)==1:return decision_envelope(obs,'tools',calls=[call('intent.solve','dimensions')])
        if len(turns)==2:
            solution=obs['feedback']['intent_solution']['canonical_values'];width,depth=solution['x'],solution['y']
            left=polynomial_patch(5,width=width,depth=depth)
            right=polynomial_patch(5,x0=width,width=width,depth=depth,bump=1.)
            lock=np.zeros((6,6),dtype=bool);lock[:4]=True;lock[-1]=True;lock[:,0]=True;lock[:,-1]=True
            return decision_envelope(obs,'tools',calls=[
                call('patch.create','left',poles=left.poles.tolist()),call('patch.create','right_seed',poles=right.poles.tolist()),
                call('surface.match','right_matched',left='left',right='right_seed',order=3),
                call('surface.fair','right',patch='right_matched',iterations=96,max_displacement_mm=.5,locked=lock.tolist()),
                call('surface.analyze','quality_left',patch='left',samples=25),
                call('surface.analyze','quality_right',patch='right',samples=25),
                call('surface.seam','seam',left='left',right='right',order=3),
                call('topology.face','left_face',patch='left'),
                call('topology.trim','right_face',patch='right',outer=[[0,0],[1,0],[1,1],[0,1]],
                     holes=[[[.55+.15*np.cos(t),.52+.22*np.sin(t)] for t in np.linspace(0,2*np.pi,24,endpoint=False)]]),
                call('topology.sew','result',shapes=['left_face','right_face']),
                call('topology.inspect','topology',shape='result')])
        return decision_envelope(obs,actions=[{'op':'STOP'}])
    with EngineeringRuntime(OUT,intent_fixture(),delivery_rules=delivery_rules()) as runtime:
        handle,feedback=runtime.create();session=ModelSession(runtime,Llm(provider),handle,feedback,max_decisions=4)
        session.run()
        assert all(e['ok'] and e['feedback']['all_actions_admitted'] for e in session.history)
        cad=runtime.export_step(session.handle,OUT/'class_a_panel.step')
        geometry=runtime.observe(session.handle,geometry=True)[1]
        patches=[BezierPatch(geometry['objects'][name]['poles']) for name in ('left','right')]
        results,gpu=gpu_batch(patches,OUT/'display',samples=41)
        assert all(r['report']['all_samples_regular'] for r in results)
        np.savez(OUT/'quality-grid.npz',**{f'{name}_grid':result['grid'] for name,result in zip(('left','right'),results)})
        report={'pass':True,'scope':'two quintic concept panels with a polygonal UV wheel-opening; G3-compatible seam',
            'model_provider':'deterministic feedback-aware fixture via Llm adaptor','external_llm_invoked':False,
            'design_intent':geometry['intent_solution'],'seam':geometry['objects']['seam']['value'],
            'quality':{name:r['report'] for name,r in zip(('left','right'),results)},'cad':cad,
            'model_decisions':session.decisions,'host_provider_calls':session.provider_calls,
            'gpu_display_batch':gpu,'class_a_certified':False,'general_134_primitive_integration':False}
        sources=['scripts/class_a.py','scripts/design_intent.py','scripts/tolerance_topology.py',
            'scripts/engineering_runtime.py','scripts/engineering_demo.py','scripts/model_session.py','scripts/model_adaptor.py',
            'scripts/session_protocol.py','src/bin/class_a_suite.rs','cuda/class_a.cu']
        report['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
        (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'pass':True,'g3_jet_max_mm':report['seam']['jet_max_mm'],
            'step_faces':cad['after']['faces'],'step_error_mm':cad['deformation']['symmetric_sampled_max_mm'],
            'model_decisions':session.decisions},indent=2))


if __name__=='__main__':main()
