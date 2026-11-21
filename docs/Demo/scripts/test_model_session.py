"""Protocol invariants and actual persistent CUDA session acceptance."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import numpy as np
from model_adaptor import CuModel, Llm
from model_session import ModelSession
from session_protocol import (Action, DeviceRollout, Fork, Inspect, OperationProgram,
                              decision_envelope, parse_decision)
from session_runtime import GeometryRuntime, RuntimeRejected

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'output/diagnostics/model/probe_policy.f64'
TARGET=[4.,-3.,2.,1.,-2.,3.]


class ProtocolTests(unittest.TestCase):
    def setUp(self):self.observation={'object':{'id':'object:1','revision':4}}

    def test_response_cannot_redefine_contract_or_reuse_stale_view(self):
        value=decision_envelope(self.observation,actions=[{'op':'STOP'}])
        for field in ['goal','target','constraint_contract','budget']:
            changed=deepcopy(value);changed[field]=0
            with self.assertRaises(ValueError):parse_decision(changed,self.observation)
        changed=deepcopy(value);changed['scene_revision']=3
        with self.assertRaises(ValueError):parse_decision(changed,self.observation)
        missing=decision_envelope(self.observation,'device_rollout',steps=4)
        with self.assertRaisesRegex(ValueError,'rollout fields'):parse_decision(missing,self.observation)

    def test_types_finite_values_and_resources(self):
        for make in [lambda:Action('UNKNOWN'),lambda:Action('ADD_PROBES',(float('nan'),)*6),
                     lambda:Action('RAW_CONTROL_DELTA',index=-1),lambda:DeviceRollout(65,'x','x'),
                     lambda:OperationProgram(()),lambda:Inspect('yes'),lambda:Fork(1)]:
            with self.assertRaises(ValueError):make()


class PersistentSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runtime=GeometryRuntime(ROOT/'output/sessions/tests',WEIGHTS,budget=512)

    @classmethod
    def tearDownClass(cls):cls.runtime.close()

    def setUp(self):self.handle,self.feedback=self.runtime.create([0.]*6,TARGET)

    def tearDown(self):
        for handle in self.runtime.objects():self.runtime.drop(handle)

    def test_feedback_repairs_draft_then_switches_to_gpu_model(self):
        observations=[]
        def provider(obs):
            observations.append(deepcopy(obs))
            if len(observations)==1:
                actions=[{'op':'BEGIN'},{'op':'RAW_CONTROL_DELTA','index':0,'values':[12,0,0,0,0,0]},{'op':'COMMIT'}]
            else:
                self.assertEqual(obs['feedback']['reason'],'CONSTRAINT_FAILED')
                actions=[{'op':'RAW_CONTROL_DELTA','index':0,'values':[-12,0,0,0,0,0]},{'op':'COMMIT'},
                         {'op':'ADD_PROBES','values':[-x for x in TARGET]}]
            return decision_envelope(obs,actions=actions)
        session=ModelSession(self.runtime,Llm(provider),self.handle,self.feedback)
        pid=self.runtime.pid
        self.assertTrue(session.step()['ok'])
        self.assertFalse(session.feedback['all_actions_admitted'])
        self.assertTrue(session.feedback['state']['draft_active'])
        _,draft=self.runtime.observe(session.handle,geometry=True,draft=True)
        self.assertAlmostEqual(draft['poles'][0][0][0][0],-1488.)
        self.assertFalse(draft['draft_valid'])
        with self.assertRaisesRegex(ValueError,'OPEN_DRAFT'):
            self.runtime.export_step(session.handle,self.runtime.root/'should-not-export.step')
        self.assertTrue(session.step()['ok'])
        self.assertGreaterEqual(session.feedback['state']['uphill_commits'],1)
        before=len(self.runtime.events)
        q=np.array(session.feedback['committed_probes'])
        session.switch_model(CuModel(WEIGHTS))
        self.assertTrue(session.step()['ok'])
        self.assertEqual(len(self.runtime.events),before+1)
        self.assertEqual(self.runtime.pid,pid)
        self.assertTrue(session.feedback['goal_met'])
        self.assertEqual(session.device_inferences,16)
        self.assertEqual(session.provider_calls,2)
        from diagnostic_suite import model_forward
        weights=np.fromfile(WEIGHTS,dtype='<f8')
        for row in session.feedback['trace']:
            self.assertEqual(len(row),26)
            np.testing.assert_allclose(row[20:26],model_forward(weights,q,np.array(TARGET)),atol=1e-9,rtol=0)
            q=np.array(row[8:14])

    def test_branches_have_independent_geometry_and_aba_safe_handles(self):
        parent=self.handle
        child,_=self.runtime.fork(parent)
        child,view=self.runtime.execute(child,OperationProgram((Action('SET_PROBES',tuple(TARGET)),)))
        self.assertTrue(view['goal_met'])
        _,unchanged=self.runtime.observe(parent)
        np.testing.assert_allclose(unchanged['committed_probes'],np.zeros(6),atol=1e-9)
        self.runtime.drop(child)
        replacement,_=self.runtime.fork(parent)
        self.assertEqual(child.slot,replacement.slot)
        self.assertNotEqual(child.generation,replacement.generation)
        with self.assertRaises(RuntimeRejected) as error:self.runtime.observe(child)
        self.assertEqual(error.exception.code,'STALE_HANDLE')
        changed,_=self.runtime.execute(parent,OperationProgram((Action('ADD_PROBES',(1.,)*6),)))
        self.assertGreater(changed.revision,parent.revision)
        with self.assertRaises(RuntimeRejected) as error:self.runtime.observe(parent)
        self.assertEqual(error.exception.code,'STALE_REVISION')

    def test_invalid_model_decision_preserves_gpu_state(self):
        before=len(self.runtime.events)
        def provider(obs):return decision_envelope(obs,actions=[{'op':'STOP'}],target=[999]*6)
        session=ModelSession(self.runtime,Llm(provider),self.handle,self.feedback)
        self.assertFalse(session.step()['ok'])
        self.assertEqual(before,len(self.runtime.events))
        _,after=self.runtime.observe(self.handle)
        self.assertEqual(after['target'],TARGET)
        self.assertEqual(after['state']['commits'],0)

    def test_rejected_raw_edit_explains_violations_without_mutating_geometry(self):
        h,f=self.runtime.execute(self.handle,OperationProgram((Action('RAW_CONTROL_DELTA',(12.,0,0,0,0,0)),)))
        self.assertEqual(h.revision,self.handle.revision)
        self.assertEqual(f['reason'],'CONSTRAINT_FAILED')
        self.assertIn('fixed_xy',f['violated_constraints'])
        self.assertTrue(f['committed_valid']);self.assertEqual(f['active_geometry_violations'],[])

    def test_abort_continue_policy_and_stop_have_distinct_semantics(self):
        h,f=self.runtime.execute(self.handle,OperationProgram((Action('BEGIN'),Action('RAW_CONTROL_DELTA',(8.,0,0,0,0,0)),Action('STOP'))))
        self.assertEqual(f['reason'],'OPEN_DRAFT');self.assertFalse(f['stopped'])
        h,f=self.runtime.execute(h,OperationProgram((Action('ABORT'),Action('COMMIT'),Action('SET_PROBES',tuple(TARGET))),True))
        self.assertEqual(f['trace'][1][1],6)
        self.assertTrue(f['goal_met']);self.assertFalse(f['all_actions_admitted'])
        h,f=self.runtime.execute(h,OperationProgram((Action('SET_PROBES',(0.,)*6),Action('STOP'))))
        self.assertTrue(f['stopped']);self.assertFalse(f['goal_met'])
        self.assertTrue(f['committed_valid'])

    def test_model_can_fork_select_inspect_and_drop(self):
        requests=[]
        def provider(obs):
            requests.append(obs)
            if len(requests)==1:return decision_envelope(obs,'fork',select=True)
            if len(requests)==2:return decision_envelope(obs,'inspect',geometry=True)
            if len(requests)==3:return decision_envelope(obs,'select',selected_object_id=self.handle.object_id)
            child=next(h for h in obs['objects'] if h['id']!=self.handle.object_id)
            return decision_envelope(obs,'drop',drop_object_id=child['id'])
        session=ModelSession(self.runtime,Llm(provider),self.handle,self.feedback)
        self.assertTrue(session.step()['ok']);self.assertNotEqual(session.handle.object_id,self.handle.object_id)
        self.assertTrue(session.step()['ok']);self.assertIn('poles',session.feedback)
        self.assertTrue(session.step()['ok']);self.assertEqual(session.handle.object_id,self.handle.object_id)
        self.assertTrue(session.step()['ok']);self.assertEqual(len(self.runtime.objects()),1)

    def test_http_model_bridge_uses_current_feedback_and_does_not_log_token(self):
        from llm_transport import HttpJsonProvider
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                payload=json.loads(inner.rfile.read(int(inner.headers['Content-Length'])))
                captured.append(payload)
                obs=payload['observation']
                result={'decision':decision_envelope(obs,actions=[{'op':'SET_PROBES','values':obs['feedback']['target']},{'op':'STOP'}]),
                        'usage':{'test_fixture':True}}
                raw=json.dumps(result).encode();inner.send_response(200);inner.send_header('Content-Length',str(len(raw)))
                inner.end_headers();inner.wfile.write(raw)
            def log_message(inner,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            transport=HttpJsonProvider(f'http://127.0.0.1:{server.server_port}/propose','test-fixture',token='fixture-secret-token')
            session=ModelSession(self.runtime,Llm(transport),self.handle,self.feedback)
            self.assertTrue(session.step()['ok']);self.assertTrue(session.feedback['goal_met'])
            self.assertEqual(transport.calls,1);self.assertEqual(captured[0]['observation']['feedback']['target'],TARGET)
            self.assertNotIn('fixture-secret-token',json.dumps(session.history))
        finally:server.shutdown();thread.join(timeout=2);server.server_close()


class BudgetTests(unittest.TestCase):
    def test_budget_is_shared_by_branches_and_counts_rejected_attempts(self):
        with GeometryRuntime(ROOT/'output/sessions/budget-test',WEIGHTS,budget=2,slots=2) as runtime:
            h,_=runtime.create([0.]*6,TARGET)
            child,_=runtime.fork(h)
            h,f=runtime.execute(h,OperationProgram((Action('RAW_CONTROL_DELTA',(10.,0,0,0,0,0)),Action('MEASURE')),True))
            self.assertEqual(f['trace'][0][1],3);self.assertEqual(runtime.remaining,0)
            # The final MEASURE is valid, while the rejected raw edit is retained in the trace.
            with self.assertRaises(RuntimeRejected) as error:
                runtime.execute(child,OperationProgram((Action('ADD_PROBES',(1.,)*6),)))
            self.assertEqual(error.exception.code,'BUDGET_EXHAUSTED')
            runtime.drop(child);new,_=runtime.fork(h)
            with self.assertRaises(RuntimeRejected):runtime.execute(new,OperationProgram((Action('STOP'),)))
            self.assertEqual(runtime.remaining,0)

    def test_declared_final_goal_cannot_be_downgraded(self):
        with GeometryRuntime(ROOT/'output/sessions/goal-test',WEIGHTS,budget=2,
                             goal_tolerance_mm=.1,require_goal_for_delivery=True) as runtime:
            h,f=runtime.create([0.]*6,TARGET)
            self.assertFalse(f['goal_met'])
            path=runtime.root/'accepted-goal.step'
            for override in (None,False,True):
                with self.assertRaisesRegex(ValueError,'Final user goal'):
                    runtime.export_step(h,path,require_goal=override)
            h,f=runtime.execute(h,OperationProgram((Action('SET_PROBES',tuple(x+.09 for x in TARGET)),)))
            self.assertTrue(f['goal_met']);self.assertGreater(f['probe_max_error_mm'],.08)
            self.assertTrue(runtime.export_step(h,path,require_goal=False)['pass'])


class WorkerLifecycleTests(unittest.TestCase):
    def test_startup_failure_closes_process_and_pipes(self):
        import subprocess
        processes=[];popen=subprocess.Popen
        def capture(*args,**kwargs):
            process=popen(*args,**kwargs);processes.append(process);return process
        with patch('session_runtime.subprocess.Popen',side_effect=capture), \
             patch.object(GeometryRuntime,'_response',side_effect=RuntimeError('readiness fixture')):
            with self.assertRaisesRegex(RuntimeError,'readiness fixture'):
                GeometryRuntime(ROOT/'output/sessions/startup-failure-test',WEIGHTS)
        self.assertEqual(len(processes),1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed);self.assertTrue(processes[0].stdout.closed)


if __name__=='__main__':unittest.main()
