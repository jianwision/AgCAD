"""Independent native/GPU checks for resident NURBS and coupled optimization."""
import json
from pathlib import Path
import unittest
import numpy as np
from class_a import BezierPatch,seam_report
from engineering_fixtures import polynomial_patch
from nurbs_surface import NurbsPatch,grid_uv
from resident_surface import SurfaceWorker
from surface_network import compile_network,network_seam_report
from resident_fixtures import corner_fixture,nurbs_args
from resident_runtime import ResidentEngineeringRuntime
from engineering_fixtures import intent_fixture
from session_protocol import ToolCall,ToolProgram
import tolerance_topology as topo
from OCP.gp import gp_Pnt,gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/resident-surfaces'


def rational_fixture():
    p=polynomial_patch(3).poles.copy();w=np.array([[1.,.9,1.2,1.],[1.1,.8,1.05,1.2],[1.,1.3,.95,1.],[1.2,1.,.9,1.1]])
    return NurbsPatch(p,w,[0]*4+[1]*4,[0]*4+[1]*4,3,3).insert_knot(0,.37).insert_knot(1,.62)


def cylinder_fixture():
    p=np.array([[[10,0,z],[10,10,z],[0,10,z]] for z in (0.,20.)])
    return NurbsPatch(p,np.tile([1,2**-.5,1],(2,1)),[0,0,1,1],[0,0,0,1,1,1],1,2)


def network_fixture(degree=5):
    patches=[]
    for i in range(3):
        base=polynomial_patch(degree,x0=i*100).poles.copy();u=(base[:,:,0]-i*100)/100;v=base[:,:,1]/60
        base[:,:,2]+=8*i*u+4*i*i+2*i*v
        base[2:4,2:4,2]+=(i+1)*2
        patches.append(BezierPatch(base))
    masks=[np.zeros(p.poles.shape[:2],dtype=bool) for p in patches];masks[0][0]=True;masks[-1][-1]=True
    seams=[{'left':0,'right':1,'order':2},{'left':1,'right':2,'order':2}]
    return patches,seams,masks


class NurbsDefinitionTests(unittest.TestCase):
    def test_homogeneous_refinement_preserves_rational_geometry_and_jets(self):
        source=rational_fixture();refined=source.insert_knot(0,.23,2).insert_knot(1,.81,2)
        maximum=0.
        for u,v in np.random.default_rng(71).uniform(.01,.99,(23,2)):
            for du,dv in ((0,0),(1,0),(0,1),(2,0),(1,1),(3,0)):
                maximum=max(maximum,float(np.max(np.abs(source.jet(u,v,du,dv)-refined.jet(u,v,du,dv)))))
        self.assertLess(maximum,2e-8)

    def test_rejects_invalid_knots_weights_and_refinement(self):
        p=polynomial_patch().poles
        with self.assertRaises(ValueError):NurbsPatch(p,np.zeros(p.shape[:2]),[0]*6+[1]*6,[0]*6+[1]*6,5,5)
        with self.assertRaises(ValueError):rational_fixture().insert_knot(0,.37,3)
        with self.assertRaises(ValueError):rational_fixture().insert_knot(1,1)
        with self.assertRaises(ValueError):compile_network([rational_fixture(),rational_fixture()],[{'left':0,'right':1}])

    def test_joint_projection_nullspace_and_conflicting_locks(self):
        patches,seams,locks=network_fixture();layouts=[NurbsPatch.from_bezier(p) for p in patches]
        plan=compile_network(layouts,seams,locks);seed=np.concatenate([p.poles.reshape(-1,3) for p in patches]);projected=plan['projection']@seed
        np.testing.assert_allclose(plan['constraint']@projected,plan['seed_target']@seed,atol=1e-10)
        np.testing.assert_allclose(plan['constraint']@plan['nullspace'],0,atol=1e-10)
        conditional=compile_network(layouts,seams,[np.ones(p.poles.shape[:2],dtype=bool) for p in patches])
        self.assertGreater(conditional['report']['seed_compatibility_rank'],0)
        self.assertGreater(np.max(np.abs(conditional['compatibility']@seed)),1e-6)


class ResidentGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.worker=SurfaceWorker(OUT/'acceptance-worker')
    @classmethod
    def tearDownClass(cls):cls.worker.close()

    def test_rational_nonuniform_d3_against_native(self):
        patch=rational_fixture();ref=self.worker.upload(patch);uv=np.random.default_rng(19).uniform(.02,.98,(31,2));grid=self.worker.analyze(ref,full=True,uv=uv)
        native=patch.native();maximum=0.
        for (u,v),row in zip(uv,grid):
            point=gp_Pnt();vectors=[gp_Vec() for _ in range(9)];native.D3(u,v,point,*vectors)
            ordered=[point,vectors[0],vectors[1],vectors[2],vectors[4],vectors[3],vectors[5],vectors[7],vectors[8],vectors[6]]
            expected=np.array([[x.X(),x.Y(),x.Z()] for x in ordered]);maximum=max(maximum,float(np.max(np.abs(expected-row[:30].reshape(10,3)))))
            np.testing.assert_allclose(row[:30].reshape(10,3),expected,rtol=2e-10,atol=2e-8)
        (OUT/'native-nurbs-validation.json').write_text(json.dumps({'pass':True,'jet_max_mm':maximum,'samples':len(uv)},indent=2),encoding='utf-8')

    def test_exact_rational_cylinder_curvature(self):
        ref=self.worker.upload(cylinder_fixture());grid=self.worker.analyze(ref,full=True,samples=13)
        np.testing.assert_allclose(np.sum(grid[:,:2]**2,axis=1),100,atol=2e-12)
        principal=np.sort(np.abs(grid[:,36:38]),axis=1)
        np.testing.assert_allclose(principal[:,0],0,atol=1e-12);np.testing.assert_allclose(principal[:,1],.1,atol=1e-12)

    def test_resident_knot_insertion_does_not_download_geometry(self):
        ref=self.worker.upload(rational_fixture());before=self.worker.stats()
        refined=self.worker.refine(self.worker.refine(ref,0,.23,2),1,.81,2)
        self.assertEqual(self.worker.stats()['geometry_readbacks'],before['geometry_readbacks'])
        uv=np.random.default_rng(11).uniform(.01,.99,(19,2))
        np.testing.assert_allclose(self.worker.analyze(ref,full=True,uv=uv)[:,:30],self.worker.analyze(refined,full=True,uv=uv)[:,:30],atol=3e-8,rtol=1e-9)

    def test_curvature_and_reflection_gradients_against_finite_differences(self):
        patch=rational_fixture();ref=self.worker.upload(patch);u,v=.21,.43;h=2e-5
        uv=np.array([[u,v],[u+h,v],[u-h,v],[u,v+h],[u,v-h]])
        rows=self.worker.analyze(ref,full=True,uv=uv);r=rows[0];E=r[3:6]@r[3:6];F=r[3:6]@r[6:9];G=r[6:9]@r[6:9];D=E*G-F*F
        hu=(rows[1,34]-rows[2,34])/(2*h);hv=(rows[3,34]-rows[4,34])/(2*h)
        expected=(G*hu*hu-2*F*hu*hv+E*hv*hv)/D
        self.assertAlmostEqual(r[45]/expected,1,delta=2e-6)
        ru=(rows[1,38:41]-rows[2,38:41])/(2*h);rv=(rows[3,38:41]-rows[4,38:41])/(2*h)
        expected=(G*(ru@ru)-2*F*(ru@rv)+E*(rv@rv))/D
        self.assertAlmostEqual(r[46]/expected,1,delta=2e-6)

    def test_summary_is_small_and_warm_execution_uploads_nothing(self):
        ref=self.worker.upload(rational_fixture());self.worker.analyze(ref,samples=17);before=self.worker.stats()
        for _ in range(4):self.assertTrue(self.worker.analyze(ref,samples=17)['all_samples_regular'])
        after=self.worker.stats();self.assertEqual(after['upload_bytes'],before['upload_bytes'])
        self.assertEqual(after['geometry_readbacks'],before['geometry_readbacks']);self.assertEqual(after['grid_readbacks'],before['grid_readbacks'])
        self.assertEqual(after['download_bytes']-before['download_bytes'],4*8*8);self.assertEqual(after['context_creations'],1)

    def test_fork_drop_reuse_and_malformed_request_recovery(self):
        ref=self.worker.upload(cylinder_fixture());child=self.worker.fork(ref);oldslot=child.slot;generation=child.generation;self.worker.drop(child)
        with self.assertRaises(ValueError):self.worker.analyze(child)
        replacement=self.worker.fork(ref);self.assertEqual(replacement.slot,oldslot);self.assertGreater(replacement.generation,generation)
        with self.assertRaises(ValueError):self.worker._request(2,payload=[4,4,3,3,100])
        self.assertTrue(self.worker.analyze(replacement)['all_samples_regular']);self.worker.drop(replacement)

    def test_adaptive_sampling_reports_unresolved_and_low_continuity(self):
        ref=self.worker.upload(rational_fixture());before=self.worker.stats()
        report=self.worker.adaptive_quality(ref,curvature_range_per_mm=1e-12,normal_angle_rad=1e-9,max_depth=0)
        self.assertGreater(report['unresolved_cells'],0);self.assertFalse(report['global_class_a_certified']);self.assertFalse(report['interior_parametric_c3'])
        self.assertEqual(self.worker.stats()['geometry_readbacks'],before['geometry_readbacks'])

    def test_joint_gpu_search_hard_seams_locks_budget_and_graph_reuse(self):
        patches,seams,locks=network_fixture();refs=[self.worker.upload(p) for p in patches];before=self.worker.stats()
        result,report=self.worker.optimize_network(refs,seams,locks=locks,iterations=192,samples=9,max_displacement_mm=5,weights={'curvature':1.,'reflection':.05,'deviation':.01})
        after=self.worker.stats();self.assertEqual(before['geometry_readbacks'],after['geometry_readbacks'])
        self.assertEqual(after['download_bytes']-before['download_bytes'],18*8)
        self.assertLess(report['after']['objective'],report['before']['objective']*.99)
        self.assertLessEqual(report['after']['max_control_displacement_mm'],5*(1+1e-9));self.assertGreater(report['accepted_steps'],0)
        materialized=[self.worker.fetch(r) for r in result]
        for original,patch,mask in zip(patches,materialized,locks):np.testing.assert_allclose(patch.poles[mask],original.poles[mask],atol=2e-10)
        for seam in seams:
            a,b=materialized[seam['left']],materialized[seam['right']]
            check=seam_report(BezierPatch(a.poles),BezierPatch(b.poles),order=2,samples=41)
            self.assertLess(max(check['jet_max_mm']),1e-7)
        _,second=self.worker.optimize_network(refs,seams,locks=locks,iterations=0,samples=9,max_displacement_mm=5)
        self.assertEqual(self.worker.stats()['graph_builds'],after['graph_builds']);self.assertGreater(self.worker.stats()['graph_cache_hits'],after['graph_cache_hits'])
        self.assertEqual(second['iterations'],0)
        cached=self.worker.stats();self.assertEqual(cached['network_plan_uploads'],after['network_plan_uploads'])
        self.assertGreater(cached['network_plan_cache_hits'],after['network_plan_cache_hits'])
        self.assertGreater(cached['host_constraint_plan_cache_hits'],after['host_constraint_plan_cache_hits'])
        (OUT/'network-validation.json').write_text(json.dumps({'pass':True,'report':report,'stats_before':before,'stats_after':after},indent=2),encoding='utf-8')

    def test_failed_projection_preserves_inputs(self):
        patches,seams,locks=network_fixture();refs=[self.worker.upload(p) for p in patches]
        with self.assertRaisesRegex(ValueError,'displacement'):self.worker.optimize_network(refs,seams,locks=locks,iterations=0,max_displacement_mm=1e-6)
        for patch,ref in zip(patches,refs):np.testing.assert_array_equal(self.worker.fetch(ref).poles,patch.poles)

    def test_fully_locked_compatible_seed_passes_and_incompatible_seed_rejects(self):
        patches,seams,locks=network_fixture();refs=[self.worker.upload(p) for p in patches]
        masks=[np.ones(p.poles.shape[:2],dtype=bool) for p in patches]
        with self.assertRaises(ValueError):self.worker.optimize_network(refs,seams,locks=masks,iterations=0,max_displacement_mm=5)
        compatible,_=self.worker.optimize_network(refs,seams,locks=locks,iterations=0,max_displacement_mm=5)
        locked,report=self.worker.optimize_network(compatible,seams,locks=masks,iterations=0,max_displacement_mm=1e-5)
        self.assertEqual(report['free_modes_per_axis'],0)
        for a,b in zip(compatible,locked):np.testing.assert_allclose(self.worker.fetch(a).poles,self.worker.fetch(b).poles,atol=1e-10)

    def test_joint_g3_and_four_patch_corner_constraints(self):
        patches,seams,locks=network_fixture()
        seams=[{**s,'order':3} for s in seams]
        refs=[self.worker.upload(p) for p in patches]
        result,_=self.worker.optimize_network(refs,seams,locks=locks,iterations=12,max_displacement_mm=5)
        self.assertLess(network_seam_report([self.worker.fetch(r) for r in result],seams)['maximum_jet_residual_mm'],2e-7)
        patches,seams=corner_fixture();refs=[self.worker.upload(p) for p in patches]
        result,report=self.worker.optimize_network(refs,seams,iterations=12,max_displacement_mm=5)
        self.assertGreater(report['dependent_constraint_rows'],0)
        self.assertLess(network_seam_report([self.worker.fetch(r) for r in result],seams)['maximum_jet_residual_mm'],2e-7)

    def test_second_worker_reuses_ptx_cache(self):
        with SurfaceWorker(OUT/'cache-worker') as worker:
            self.assertEqual(worker.startup['context_creations'],1);self.assertEqual(worker.startup['compilations'],0)
            self.assertEqual(worker.startup['compile_cache_hit'],1);self.assertTrue(worker.analyze(worker.upload(cylinder_fixture()))['all_samples_regular'])

    def test_rational_step_roundtrip(self):
        shape=BRepBuilderAPI_MakeFace(rational_fixture().native(),1e-7).Face()
        report=topo.export_step_checked(shape,OUT/'rational-skin.step');self.assertTrue(report['pass']);self.assertEqual(report['after']['faces'],1)


class ResidentRuntimeTests(unittest.TestCase):
    def test_atomic_failure_cleans_device_objects_and_fork_preserves_them(self):
        with ResidentEngineeringRuntime(OUT/'atomic-runtime',intent_fixture(),delivery_rules={'require_intent':False}) as runtime:
            handle,_=runtime.create();args=nurbs_args(cylinder_fixture())
            failed=ToolProgram((ToolCall('nurbs.create',args,'skin'),ToolCall('surface.refine',{'patch':'skin','axis':3,'value':.5},'bad')))
            handle,feedback=runtime.execute(handle,failed);self.assertEqual(feedback['reason'],'TOOL_FAILED');self.assertFalse(runtime.worker._live)
            handle,feedback=runtime.execute(handle,ToolProgram((ToolCall('nurbs.create',args,'skin'),)))
            self.assertEqual(len(runtime.worker._live),1);self.assertNotIn('poles',feedback['objects']['skin'])
            branch,_=runtime.fork(handle);runtime.drop(handle);self.assertEqual(len(runtime.worker._live),1)
            branch,feedback=runtime.execute(branch,ToolProgram((ToolCall('surface.analyze',{'patch':'skin'},'quality'),)))
            self.assertTrue(feedback['objects']['quality']['value']['all_samples_regular']);runtime.drop(branch);self.assertFalse(runtime.worker._live)

    def test_final_network_binding_and_user_deviation_limit(self):
        rules={'require_intent':False,'networks':[{'network':'fair','maximum_control_displacement_mm':5,'tolerance_mm':1e-6}]}
        with ResidentEngineeringRuntime(OUT/'delivery-runtime',intent_fixture(),delivery_rules=rules) as runtime:
            handle,_=runtime.create();patches,seams,locks=network_fixture();calls=[]
            for i,patch in enumerate(patches):calls.append(ToolCall('nurbs.create',nurbs_args(NurbsPatch.from_bezier(patch)),f'p{i}'))
            calls.append(ToolCall('surface.network_optimize',{'patches':['p0','p1','p2'],'seams':seams,'locks':[m.tolist() for m in locks],'iterations':12,'max_displacement_mm':5},'fair'))
            for i in range(3):
                calls.extend((ToolCall('surface.network_patch',{'network':'fair','index':i},f'q{i}'),ToolCall('topology.face',{'patch':f'q{i}'},f'f{i}')))
            calls.append(ToolCall('topology.sew',{'shapes':['f0','f1','f2']},'result'))
            handle,feedback=runtime.execute(handle,ToolProgram(tuple(calls)));self.assertEqual(feedback['reason'],'ACCEPTED',feedback)
            report=runtime.export_step(handle,OUT/'joint-skin.step');self.assertTrue(report['pass']);self.assertEqual(report['after']['faces'],3)
            doc=runtime._document(handle);doc['objects']['result']={'kind':'shape','value':BRepBuilderAPI_MakeFace(cylinder_fixture().native(),1e-7).Face()}
            with self.assertRaisesRegex(ValueError,'absent'):runtime.export_step(handle,OUT/'should-not-exist.step')


if __name__=='__main__':unittest.main(verbosity=2)
