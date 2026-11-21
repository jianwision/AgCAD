"""Model tool program -> resident joint G3 optimization -> trimmed STEP."""
import hashlib
import json
from pathlib import Path
import numpy as np
from engineering_fixtures import intent_fixture
from model_adaptor import Llm
from model_session import ModelSession
from nurbs_surface import NurbsPatch
from resident_fixtures import network_fixture,nurbs_args
from resident_runtime import ResidentEngineeringRuntime
from session_protocol import decision_envelope

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/resident-surfaces/demo'
SOURCES=['scripts/nurbs_surface.py','scripts/surface_network.py','scripts/resident_surface.py','scripts/resident_runtime.py',
         'scripts/resident_fixtures.py','scripts/resident_demo.py','scripts/class_a.py','scripts/engineering_runtime.py',
         'scripts/design_intent.py','scripts/tolerance_topology.py','scripts/model_adaptor.py','scripts/model_session.py',
         'scripts/session_protocol.py','src/bin/surface_worker.rs','cuda/resident_surface.cu']


def call(op,out,**args):return {'op':op,'out':out,'args':args}


def main():
    OUT.mkdir(parents=True,exist_ok=True);patches,seams,locks=network_fixture(order=3);turns=[]
    def provider(obs):
        turns.append(obs)
        if len(turns)==1:return decision_envelope(obs,'tools',calls=[call('intent.solve','dimensions'),
            *[call('nurbs.create',f'seed{i}',**nurbs_args(NurbsPatch.from_bezier(p))) for i,p in enumerate(patches)]])
        if len(turns)==2:return decision_envelope(obs,'tools',calls=[call('surface.network_optimize','fair',
            patches=['seed0','seed1','seed2'],seams=seams,locks=[m.tolist() for m in locks],iterations=256,samples=9,
            step_mm=.5,max_displacement_mm=5.,weights={'curvature':1.,'reflection':.05,'deviation':.01})])
        if len(turns)==3:
            calls=[]
            for i in range(3):
                calls.extend([call('surface.network_patch',f'panel{i}',network='fair',index=i),
                              call('surface.analyze',f'quality{i}',patch=f'panel{i}',samples=25)])
                if i==1:
                    calls.append(call('topology.trim',f'face{i}',patch=f'panel{i}',outer=[[0,0],[1,0],[1,1],[0,1]],
                        holes=[[[.52+.16*np.cos(t),.55+.24*np.sin(t)] for t in np.linspace(0,2*np.pi,32,endpoint=False)]]))
                else:calls.append(call('topology.face',f'face{i}',patch=f'panel{i}'))
            calls.append(call('topology.sew','result',shapes=['face0','face1','face2']))
            return decision_envelope(obs,'tools',calls=calls)
        return decision_envelope(obs,actions=[{'op':'STOP'}])
    rules={'require_intent':True,'networks':[{'network':'fair','maximum_control_displacement_mm':5.,'tolerance_mm':1e-6,
        'maximum_normal_angle_rad':1e-6,'require_sampled_regular':True,'maximum_abs_curvature_per_mm':.2,'quality_samples':33}]}
    with ResidentEngineeringRuntime(OUT,intent_fixture(),delivery_rules=rules) as runtime:
        handle,feedback=runtime.create();session=ModelSession(runtime,Llm(provider),handle,feedback,max_decisions=5);session.run()
        if not all(e['ok'] and e['feedback']['all_actions_admitted'] for e in session.history):raise RuntimeError('Model tool program failed')
        doc=runtime._document(session.handle);network=doc['objects']['fair']['value'];before_boundary=runtime.worker.stats()
        # The model's topology tools are the first explicit CPU geometry boundary.
        cad=runtime.export_step(session.handle,OUT/'joint_g3_panel.step')
        grids={};quality_before=[];quality_after=[]
        for i,(seed,result) in enumerate(zip(network['seeds'],network['patches'])):
            grids[f'before_{i}']=runtime.worker.analyze(seed,full=True,samples=41).reshape(41,41,48)
            grids[f'after_{i}']=runtime.worker.analyze(result,full=True,samples=41).reshape(41,41,48)
            quality_before.append(runtime.worker.analyze(seed,samples=33));quality_after.append(runtime.worker.analyze(result,samples=33))
        np.savez_compressed(OUT/'quality-grid.npz',**grids)
        cb=sum(q['mean_curvature_gradient_squared_per_mm4'] for q in quality_before);ca=sum(q['mean_curvature_gradient_squared_per_mm4'] for q in quality_after)
        if ca>=cb:raise RuntimeError('Independent denser quality grid did not improve')
        report={'pass':True,'scope':'three jointly optimized quintic G3-compatible concept panels with polygonal UV opening',
            'model_provider':'deterministic feedback-aware tool fixture via Llm','external_llm_invoked':False,
            'model_decisions':session.decisions,'optimization':network['report'],'cad':cad,
            'independent_dense_quality':{'samples_per_patch':1089,'before':quality_before,'after':quality_after,'curvature_gradient_reduction_fraction':1-ca/cb},
            'stats_before_export':before_boundary,'resident_stats':runtime.worker.stats(),'class_a_certified':False,
            'new_cumodel_training_performed':False,'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}}
        (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'pass':True,'step':str(OUT/'joint_g3_panel.step'),'faces':cad['after']['faces'],
        'objective_reduction_fraction':1-network['report']['after']['objective']/network['report']['before']['objective'],
        'dense_curvature_gradient_reduction_fraction':1-ca/cb,'g3_max_mm':cad['user_contract_checks'][-1]['seams']['maximum_jet_residual_mm'],
        'resident_contexts':report['resident_stats']['context_creations'],'optimization_geometry_readbacks':0},indent=2))


if __name__=='__main__':main()
