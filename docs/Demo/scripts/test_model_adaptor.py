"""Model boundary invariants: provider output cannot rewrite the user contract."""
import copy
import json
from pathlib import Path
import unittest
from model_adaptor import (CuModel, Llm, ModelAdaptor, ModelContext, ModelRequest,
                           ReplayModel, parse_response)

ROOT=Path(__file__).resolve().parents[1]


class ModelAdaptorTests(unittest.TestCase):
    def setUp(self):
        self.context=ModelContext('scene.json')
        self.requests=(ModelRequest('fit_00',(0.,)*6,(4.,-3.,2.,1.,-2.,3.)),)
        self.path=ROOT/'configs/llm-proposal-example.json'
        self.response=json.loads(self.path.read_text())

    def test_llm_callback_and_replay_have_identical_proposals(self):
        seen=[]
        def provider(observation):
            seen.append(observation)
            return self.response
        llm=Llm(provider);replay=ReplayModel(self.path)
        self.assertIsInstance(llm,ModelAdaptor)
        live=llm.propose(self.context,self.requests)
        recorded=replay.propose(self.context,self.requests)
        self.assertEqual(live.programs,recorded.programs)
        self.assertEqual(live.host_model_calls,1)
        self.assertEqual(recorded.host_model_calls,0)
        self.assertEqual(seen[0]['requests'][0]['target'],self.requests[0].target)

    def test_provider_cannot_redefine_goal_or_context(self):
        for key,value in [('target',[0.]*6),('initial',[0.]*6),('constraint_contract','relaxed')]:
            altered=copy.deepcopy(self.response);altered['programs'][0][key]=value
            with self.assertRaises(ValueError): parse_response(altered,self.context,self.requests)

    def test_provider_observation_is_not_mutable_contract_storage(self):
        def provider(obs):
            obs['context']['contract_revision']=99
            obs['requests'][0]['target']=(999.,)*6
            obs['operations']['ADD_PROBES']=99
            return self.response
        result=Llm(provider).propose(self.context,self.requests)
        self.assertEqual(self.context.contract_revision,1)
        self.assertEqual(result.programs[0]['target'],list(self.requests[0].target))
        self.assertEqual(result.programs[0]['actions'][0][0],0)

    def test_stale_duplicate_unknown_and_over_budget_are_rejected(self):
        variants=[]
        p=copy.deepcopy(self.response);p['contract_revision']=2;variants.append(p)
        p=copy.deepcopy(self.response);p['programs']*=2;variants.append(p)
        p=copy.deepcopy(self.response);p['programs'][0]['request_id']='unknown';variants.append(p)
        p=copy.deepcopy(self.response);p['programs'][0]['actions']*=6;variants.append(p)
        for p in variants:
            with self.assertRaises(ValueError): parse_response(p,self.context,self.requests)

    def test_invalid_operations_nonfinite_and_dimensions_are_rejected(self):
        for action in [{'op':'UNDECLARED'},{'op':'ADD_PROBES','values':[0.]},
                       {'op':'ADD_PROBES','values':[float('nan')]*6}]:
            p=copy.deepcopy(self.response);p['programs'][0]['actions']=[action]
            with self.assertRaises(ValueError): parse_response(p,self.context,self.requests)

    def test_cuda_is_same_role_without_host_prediction_materialization(self):
        adaptor=CuModel(ROOT/'output/diagnostics/model/probe_policy.f64')
        self.assertIsInstance(adaptor,ModelAdaptor)
        batch=adaptor.propose(self.context,self.requests)
        self.assertEqual(batch.host_model_calls,0)
        self.assertTrue(batch.device_model_inference)
        self.assertEqual(batch.programs[0]['target'],list(self.requests[0].target))
        self.assertEqual(batch.programs[0]['actions'],[])


if __name__=='__main__': unittest.main()
