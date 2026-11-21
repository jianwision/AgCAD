"""Persistent model session demonstration and provider-enabled command line."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from model_adaptor import CuModel, Llm, ReplayModel
from model_session import ModelSession
from session_runtime import GeometryRuntime, RuntimeRejected
from session_protocol import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['demo','cuda','llm','replay'],default='demo')
    parser.add_argument('--out',type=Path,help='Defaults to output/sessions/<model>')
    parser.add_argument('--weights',type=Path,default=ROOT/'output/diagnostics/model/probe_policy.f64')
    parser.add_argument('--replay',type=Path,default=ROOT/'configs/session-repair-demo.json')
    parser.add_argument('--provider-url')
    parser.add_argument('--provider-model',default='configured-model')
    parser.add_argument('--budget',type=int,default=256)
    parser.add_argument('--turns',type=int,default=8)
    parser.add_argument('--target',nargs=6,type=float,default=[4,-3,2,1,-2,3])
    parser.add_argument('--goal-tolerance-mm',type=float,default=.5)
    parser.add_argument('--require-goal',action='store_true',help='Require the declared probe goal before STEP delivery')
    args=parser.parse_args()
    if args.model=='llm' and not args.provider_url:parser.error('--model llm requires --provider-url')
    if not 1<=args.turns<=256:parser.error('--turns must be in [1,256]')
    if args.model=='demo' and args.turns<3:parser.error('The repair/switch demo requires at least three model decisions')
    if args.model=='demo' and args.target!=[4,-3,2,1,-2,3]:
        parser.error('The recorded repair/switch demo has a fixed target; use --model cuda, llm or replay for other targets')
    args.out=(args.out or ROOT/'output/sessions'/args.model).resolve();args.out.mkdir(parents=True,exist_ok=True)
    checks={};provider=None;parity=None
    if args.model in ('demo','replay'):model=ReplayModel(args.replay)
    elif args.model=='cuda':model=CuModel(args.weights)
    else:
        from llm_transport import HttpJsonProvider
        provider=HttpJsonProvider(args.provider_url,args.provider_model,token=os.environ.get('CAD_MODEL_TOKEN'))
        model=Llm(provider)
    with GeometryRuntime(args.out,args.weights,budget=args.budget,goal_tolerance_mm=args.goal_tolerance_mm,
                         require_goal_for_delivery=args.require_goal) as runtime:
        initial_pid=runtime.pid
        handle,view=runtime.create([0.]*6,args.target)
        session=ModelSession(runtime,model,handle,view,max_decisions=args.turns)
        if args.model=='demo':
            first=session.step()
            assert first['ok'] and session.feedback['reason']=='CONSTRAINT_FAILED'
            assert session.feedback['state']['draft_active'] and not session.feedback['draft_valid']
            assert session.feedback['committed_valid'] and session.feedback['state']['commits']==0
            checks['rejected_commit_retains_repairable_draft']=True
            frozen=session.handle
            child,child_view=runtime.fork(session.handle)
            assert child_view['committed_valid'] and child_view['state']['draft_active']==0
            assert np.max(np.abs(child_view['committed_probes']))<1e-9
            checks['fork_uses_committed_view']=True
            second=session.step()
            assert second['ok'] and not session.feedback['state']['draft_active']
            assert session.feedback['state']['uphill_commits']>=1
            checks['repair_then_legal_uphill_commit']=True
            try:runtime.observe(frozen)
            except RuntimeRejected as e:assert e.code=='STALE_REVISION'
            else:raise AssertionError('Stale revision was accepted')
            checks['stale_revision_rejected']=True
            q=np.array(session.feedback['committed_probes']);target=np.array(session.feedback['target'])
            event_before=len(runtime.events)
            session.switch_model(CuModel(args.weights,steps=16))
            result=session.step();assert result['ok']
            assert len(runtime.events)==event_before+1 and runtime.pid==initial_pid
            checks['switch_preserves_worker_and_scene']=True
            checks['device_rollout_has_one_final_feedback_readback']=True
            from diagnostic_suite import model_forward
            weights=np.fromfile(args.weights,dtype='<f8');parity=0.
            for row in session.feedback['trace']:
                expected=model_forward(weights,q,target)
                parity=max(parity,float(np.max(np.abs(expected-row[20:26]))))
                q=np.array(row[8:14])
            assert parity<1e-9 and session.feedback['goal_met']
            _,unchanged=runtime.observe(child)
            assert np.max(np.abs(unchanged['committed_probes']))<1e-9
            checks['branch_independence']=True
        else:session.run(max_turns=1 if args.model=='cuda' else args.turns)
        receipt=runtime.export_step(session.handle,args.out/'session_hood.step')
        session.save()
        final_view=runtime.cached_view(session.handle)
        result={'pass':all(e['ok'] for e in session.history) and receipt['pass'],'model_mode':args.model,
            'checks':checks,'goal_met':final_view['goal_met'],'probe_max_error_mm':final_view['probe_max_error_mm'],
            'cpu_gpu_model_proposal_max_mm':parity,'model_decisions':session.decisions,
            'host_provider_calls':session.provider_calls,'device_model_inferences':session.device_inferences,
            'external_model_service_invoked':provider is not None and provider.calls>0,
            'default_demo_is_recorded_proposals_not_live_llm':args.model=='demo',
            'switches':session.switches,'cad':receipt,
            'goal_criterion':runtime.scene['goal_criterion'],
            'weights_sha256':runtime.weights_sha256,'recursive_improvement_claim':False}
    worker=json.loads((args.out/'worker-report.json').read_text())
    assert worker['context_creations']==worker['template_uploads']==1
    result['worker']=worker
    result['sources']={p.as_posix():digest(ROOT/p) for p in map(Path,[
        'src/bin/cad_worker.rs','cuda/cad_session.cu','cuda/hood_contract.cuh',
        'scripts/model_adaptor.py','scripts/model_session.py','scripts/session_runtime.py','scripts/session_protocol.py',
        'scripts/session_demo.py','scripts/llm_transport.py','scripts/diagnostic_suite.py',
        'scripts/diagnostic_geometry.py','configs/session-repair-demo.json'])}
    (args.out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['pass','model_mode','checks','goal_met','probe_max_error_mm',
                                          'cpu_gpu_model_proposal_max_mm','device_model_inferences']},indent=2))


if __name__=='__main__':main()
