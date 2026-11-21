"""Independent CUDA/OCP, nonlinear-intent and tolerance/topology acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from class_a import BezierPatch,gpu_batch,match_boundary,seam_report,JETS
from design_intent import DesignIntent
from engineering_fixtures import polynomial_patch,intent_fixture
import tolerance_topology as topo
from OCP.gp import gp_Pnt,gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TopAbs import TopAbs_FACE
from engineering_runtime import EngineeringRuntime
from model_session import ModelSession
from model_adaptor import Llm
from session_protocol import ToolCall,ToolProgram,decision_envelope

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/engineering'


def face(patch):return BRepBuilderAPI_MakeFace(patch.native(),1e-7).Face()


class ClassATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches=[polynomial_patch(3),polynomial_patch(7),polynomial_patch(5,bump=15)]
        p=polynomial_patch().poles.copy();p[:,:,1]=0;cls.patches.append(BezierPatch(p))
        lock=np.zeros((6,6),dtype=bool);lock[:2]=True;lock[-2:]=True;lock[:,:2]=True;lock[:,-2:]=True
        cls.lock=lock
        cls.results,cls.gpu=gpu_batch(cls.patches,OUT/'class-a',samples=17,fairing=[{},{},{'iterations':128,'max_displacement_mm':2.,'locked':lock},{}])

    def test_gpu_derivatives_curvature_and_reflection_against_native_d3(self):
        maximum=0.
        for patch,result in zip(self.patches[:2],self.results[:2]):
            native=patch.native();grid=result['grid']
            for i in range(17):
                for j in range(17):
                    p=gp_Pnt();vectors=[gp_Vec() for _ in range(9)]
                    native.D3(i/16,j/16,p,*vectors)
                    # Native D3 order differs from the device packing.
                    native_order=[p,vectors[0],vectors[1],vectors[2],vectors[4],vectors[3],vectors[5],vectors[7],vectors[8],vectors[6]]
                    expected=np.array([[v.X(),v.Y(),v.Z()] for v in native_order])
                    maximum=max(maximum,float(np.max(np.abs(grid[i,j,:30].reshape(10,3)-expected))))
                    u,v=i/16,j/16
                    z_u=(8*u+2*v)/100;z_v=(6*v+2*u)/60
                    denominator=1+z_u*z_u+z_v*z_v
                    gaussian=(8/100**2*6/60**2-(2/6000)**2)/denominator**2
                    self.assertAlmostEqual(grid[i,j,35],gaussian,places=15)
                    self.assertAlmostEqual(np.linalg.norm(grid[i,j,38:41]),1.,places=12)
            self.assertTrue(result['report']['all_samples_regular'])
        self.assertLess(maximum,1e-8)
        (OUT/'class-a/native-validation.json').write_text(json.dumps({'pass':True,'jet_max_mm':maximum,'gpu':self.gpu},indent=2))
        self.assertEqual(self.gpu['final_readbacks'],1);self.assertEqual(self.gpu['intermediate_readbacks'],0)

    def test_bounded_fairing_preserves_locks_and_decreases_energy(self):
        result=self.results[2];report=result['report']
        np.testing.assert_array_equal(result['patch'].poles[self.lock],self.patches[2].poles[self.lock])
        self.assertLess(report['fairness_energy_after'],report['fairness_energy_before']*.9)
        self.assertLessEqual(report['max_control_displacement_mm'],2.+1e-10)

    def test_g1_g2_g3_parameter_map_and_orientation(self):
        a=polynomial_patch(5)
        for order in (1,2,3):
            for reverse in (False,True):
                b=match_boundary(a,polynomial_patch(7,x0=100),order=order,scale=.8,shear=.12,reverse=reverse)
                report=seam_report(a,b,order=order,scale=.8,shear=.12,reverse=reverse)
                self.assertLess(max(report['jet_max_mm']),1e-8)
                self.assertLess(report['normal_angle_max_rad'],1e-10)
                if order>=2:self.assertLess(report['transverse_normal_curvature_gap_per_mm'],1e-12)
        with self.assertRaises(ValueError):match_boundary(a,a,scale=-1)

    def test_singular_surface_is_not_marked_regular(self):
        self.assertFalse(self.results[3]['report']['all_samples_regular'])
        self.assertFalse(self.results[3]['report']['global_class_a_certified'])


class IntentTests(unittest.TestCase):
    def test_units_formulas_solve_and_associative_design_change(self):
        model=DesignIntent(intent_fixture());before=model.contract_sha256;result=model.solve()
        self.assertTrue(result['hard_constraints_pass']);self.assertEqual(result['local_remaining_dof'],0)
        self.assertAlmostEqual(result['solved_parameters']['x'],100*math.cos(math.pi/6),places=8)
        self.assertAlmostEqual(result['parameters']['diameter']['value'],200.)
        changed=model.with_parameters({'radius':12.});updated=changed.solve()
        self.assertTrue(updated['hard_constraints_pass']);self.assertNotEqual(changed.contract_sha256,before)
        self.assertEqual(model.contract_sha256,before)
        self.assertAlmostEqual(updated['solved_parameters']['y'],60.,places=8)
        self.assertTrue(DesignIntent.loads(changed.dumps()).solve()['hard_constraints_pass'])
        with self.assertRaises(ValueError):model.evaluate({'radius':9.})

    def test_conflicts_report_constraint_ids_without_infeasibility_claim(self):
        spec=intent_fixture();spec['constraints']=[{'id':name,'kind':'expression','lhs':'x','rhs':rhs,'tolerance':{'value':1e-4,'unit':'mm'}} for name,rhs in [('small','ten'),('large','radius')]]
        result=DesignIntent(spec).solve()
        self.assertFalse(result['hard_constraints_pass']);self.assertTrue(result['suspected_conflicts'])
        self.assertFalse(result['global_infeasibility_proven']);self.assertGreater(result['local_remaining_dof'],0)

    def test_solver_budget_includes_finite_difference_evaluations(self):
        result=DesignIntent(intent_fixture()).solve(max_evaluations=1)
        self.assertFalse(result['solver_terminated']);self.assertEqual(result['function_evaluations'],1)
        self.assertFalse(result['hard_constraints_pass']);self.assertIsNone(result['local_hard_jacobian_rank'])
        self.assertTrue(result['finite_difference_calls_included'])

    def test_free_curve_parameter_attachment(self):
        poles=np.array([[0,0,0],[20,30,0],[40,-20,10],[60,0,0.]])
        t=.63;point=sum(math.comb(3,i)*t**i*(1-t)**(3-i)*poles[i] for i in range(4))
        parameters={name:{'value':float(v),'unit':'mm','fixed':True} for name,v in zip(('x','y','z'),point)}
        parameters['t']={'value':.2,'unit':'1','bounds':[0.,1.]}
        spec={'schema':'cad-design-intent-v1','parameters':parameters,'curves':{'guide':poles.tolist()},'points':{'p':['x','y','z']},
              'constraints':[{'id':'slide','kind':'on_curve','curve':'guide','point':'p','t':'t','tolerance':{'value':1e-6,'unit':'mm'}}]}
        result=DesignIntent(spec).solve();self.assertTrue(result['hard_constraints_pass']);self.assertAlmostEqual(result['solved_parameters']['t'],t,places=9)

    def test_free_surface_uv_attachment(self):
        patch=polynomial_patch();p=patch.jet(.37,.62)
        parameters={name:{'value':float(v),'unit':'mm','fixed':True} for name,v in zip(('x','y','z'),p)}
        parameters.update({name:{'value':.2,'unit':'1','bounds':[0.,1.]} for name in ('u','v')})
        spec={'schema':'cad-design-intent-v1','parameters':parameters,'surfaces':{'skin':patch.poles.tolist()},'points':{'p':['x','y','z']},
              'constraints':[{'id':'surface_slide','kind':'on_surface','surface':'skin','point':'p','u':'u','v':'v','tolerance':{'value':1e-6,'unit':'mm'}}]}
        result=DesignIntent(spec).solve();self.assertTrue(result['hard_constraints_pass'])
        np.testing.assert_allclose([result['solved_parameters'][k] for k in ('u','v')],[.37,.62],atol=1e-9)

    def test_dimension_cycles_and_unsafe_formulas_reject(self):
        for formulas in [{'bad':{'expression':'radius+angle','unit':'mm'}},
                         {'a':{'expression':'b','unit':'mm'},'b':{'expression':'a','unit':'mm'}},
                         {'bad':{'expression':"__import__('os')",'unit':'mm'}}]:
            spec=intent_fixture();spec['formulas']=formulas
            with self.assertRaises((ValueError,SyntaxError)):DesignIntent(spec)

    def test_inequality_symmetry_and_direction_constraints(self):
        spec=intent_fixture();spec['parameters'].update({k:{'value':v,'unit':'mm','fixed':True} for k,v in [('nx',-10),('nz',0)]})
        spec['points'].update({'q':['ten','ten','zero'],'mirror':['ten','nx','zero'],'vertical':['zero','ten','zero']})
        tol={'value':1e-6,'unit':'mm'};angle={'value':1e-6,'unit':'rad'}
        spec['constraints']=[{'id':'mirror','kind':'symmetry','a':'q','b':'mirror','offset':'zero','tolerance':tol},
            {'id':'orthogonal','kind':'perpendicular','a0':'origin','a1':'axis','b0':'origin','b1':'vertical','tolerance':angle},
            {'id':'parallel','kind':'parallel','a0':'origin','a1':'axis','b0':'vertical','b1':'q','tolerance':angle},
            {'id':'size','kind':'expression','lhs':'ten','rhs':'radius','relation':'le','tolerance':tol},
            {'id':'coincidence','kind':'coincident','a':'origin','b':'origin','tolerance':tol}]
        self.assertTrue(DesignIntent(spec).evaluate()['hard_constraints_pass'])


class TopologyTests(unittest.TestCase):
    def setUp(self):self.box=BRepPrimAPI_MakeBox(10,10,10).Shape()

    def test_boolean_split_history_and_non_destructive_inputs(self):
        tool=BRepPrimAPI_MakeBox(gp_Pnt(5,0,0),10,10,10).Shape();before=topo.inspect_shape(self.box)
        for operation,volume in [('union',1500),('cut',500),('common',500)]:
            shape,report=topo.boolean(self.box,tool,operation=operation)
            self.assertAlmostEqual(report['topology']['solid_volume_mm3'],volume,places=6)
            self.assertTrue(report['history'])
        shape,report=topo.split(self.box,[tool]);self.assertEqual(report['topology']['solids'],2)
        self.assertAlmostEqual(report['topology']['solid_volume_mm3'],1000,places=6)
        self.assertEqual(topo.inspect_shape(self.box),before)
        topo.export_step_checked(shape,OUT/'split-solid.step')

    def test_trim_holes_and_invalid_uv_topology(self):
        patch=polynomial_patch();outer=[[0,0],[1,0],[1,1],[0,1]];hole=[[.3,.3],[.3,.7],[.7,.7],[.7,.3]]
        shape,report=topo.trim(patch,outer,[hole]);self.assertTrue(report['topology']['valid']);self.assertEqual(report['topology']['faces'],1)
        topo.export_step_checked(shape,OUT/'trimmed-skin.step')
        for boundary,holes in [([[0,0],[1,1],[0,1],[1,0]],[]),(outer,[[[-.1,.2],[.1,.2],[.1,.4]]]),
                              (outer,[hole,[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]])]:
            with self.assertRaises(ValueError):topo.trim(patch,boundary,holes)

    def test_sew_closed_shell_and_tolerance_gap_budget(self):
        faces=topo.subshapes(self.box,TopAbs_FACE)
        shape,report=topo.sew(faces,solid=True)
        self.assertEqual(report['topology']['free_edges'],0);self.assertAlmostEqual(report['topology']['solid_volume_mm3'],1000,places=6)
        topo.export_step_checked(shape,OUT/'sewn-solid.step')
        def plane(x):
            p=polynomial_patch(3,x0=x,width=10,depth=10).poles.copy();p[:,:,2]=0;return face(BezierPatch(p))
        a,b=plane(0),plane(10.0002)
        with self.assertRaisesRegex(ValueError,'connected shell'):topo.sew([a,b])
        _,joined=topo.sew([a,b],policy=topo.TolerancePolicy(absolute_mm=.0003,maximum_native_mm=.0005))
        self.assertEqual(joined['topology']['free_edges'],6)
        self.assertLessEqual(joined['topology']['maximum_native_tolerance_mm'],.0005)

    def test_native_tolerance_budget_and_healing_preserve_source(self):
        from OCP.ShapeFix import ShapeFix_ShapeTolerance
        before=topo.inspect_shape(self.box);_,report=topo.heal(self.box)
        self.assertEqual(topo.inspect_shape(self.box),before);self.assertEqual(report['deformation']['symmetric_sampled_max_mm'],0.)
        bad=topo.copy_shape(self.box);ShapeFix_ShapeTolerance().SetTolerance(bad,.02)
        with self.assertRaisesRegex(ValueError,'native tolerance'):topo.heal(bad)
        with self.assertRaises(ValueError):topo.TolerancePolicy(absolute_mm=.1,maximum_native_mm=.01)

    def test_section_creates_intersection_curves(self):
        tool=BRepPrimAPI_MakeBox(gp_Pnt(5,5,-5),10,10,20).Shape();shape,report=topo.section(self.box,tool)
        self.assertGreater(report['topology']['edges'],0);self.assertTrue(report['topology']['valid'])
        topo.export_step_checked(shape,OUT/'section-curves.step')


class EngineeringRuntimeTests(unittest.TestCase):
    def test_same_model_adaptor_executes_typed_tools_and_repairs_atomic_failure(self):
        calls=[]
        def provider(obs):
            calls.append(obs)
            if len(calls)==1:return decision_envelope(obs,'tools',calls=[
                {'op':'patch.create','out':'temporary','args':{'poles':polynomial_patch().poles.tolist()}},
                {'op':'topology.face','out':'face','args':{'patch':'missing'}}])
            self.assertEqual(obs['feedback']['reason'],'TOOL_FAILED');self.assertNotIn('temporary',obs['feedback']['objects'])
            return decision_envelope(obs,'tools',calls=[{'op':'intent.solve','out':'solution','args':{}}])
        with EngineeringRuntime(OUT/'runtime-tests',intent_fixture()) as runtime:
            handle,feedback=runtime.create();session=ModelSession(runtime,Llm(provider),handle,feedback)
            self.assertIn('tools',session.observation()['capabilities'])
            self.assertTrue(session.step()['ok']);self.assertEqual(session.handle.revision,0)
            self.assertFalse(session.feedback['all_actions_admitted']);self.assertEqual(runtime.remaining,126)
            self.assertTrue(session.step()['ok']);self.assertEqual(session.handle.revision,1)
            self.assertTrue(session.feedback['intent_solution']['hard_constraints_pass'])

    def test_document_branch_and_final_geometry_binding(self):
        rules={'coordinates':[{'patch':'skin','u':1.,'v':0.,'axis':0,'expression':'x','tolerance_mm':1e-6}]}
        with EngineeringRuntime(OUT/'binding-tests',intent_fixture(),delivery_rules=rules) as runtime:
            h,_=runtime.create();h,_=runtime.execute(h,ToolProgram((ToolCall('intent.solve',{},'dimensions'),)))
            child,_=runtime.fork(h)
            h,_=runtime.execute(h,ToolProgram((ToolCall('patch.create',{'poles':polynomial_patch(width=100).poles.tolist()},'skin'),
                ToolCall('topology.face',{'patch':'skin'},'result'))))
            self.assertNotIn('skin',runtime.cached_view(child)['objects'])
            with self.assertRaisesRegex(ValueError,'binding failed'):runtime.export_step(h,OUT/'should-not-deliver.step')
            # Satisfying an unrelated scene patch must not validate a different delivered shape.
            width=runtime.cached_view(h)['intent_solution']['canonical_values']['x']
            h,_=runtime.execute(h,ToolProgram((ToolCall('patch.create',{'poles':polynomial_patch(width=width).poles.tolist()},'skin'),)))
            with self.assertRaisesRegex(ValueError,'absent from delivered'):runtime.export_step(h,OUT/'should-not-deliver.step')
            stale=h;h,_=runtime.execute(h,ToolProgram((ToolCall('topology.face',{'patch':'skin'},'result'),)))
            with self.assertRaisesRegex(ValueError,'Stale'):runtime.observe(stale)
            self.assertTrue(runtime.export_step(h,OUT/'intent-bound-skin.step')['pass'])

    def test_model_cannot_override_user_tolerance_or_fixed_targets(self):
        with EngineeringRuntime(OUT/'tool-contract-tests',intent_fixture()) as runtime:
            h,_=runtime.create()
            for call in (ToolCall('intent.solve',{'initial':{'radius':1.}},'bad'),
                         ToolCall('topology.heal',{'shape':'missing','policy':{'maximum_native_mm':1.}},'bad')):
                h,feedback=runtime.execute(h,ToolProgram((call,)))
                self.assertFalse(feedback['all_actions_admitted']);self.assertEqual(h.revision,0)

    def test_final_quality_rechecks_actual_surface(self):
        rules={'require_intent':False,'quality':[{'patch':'skin','samples':9,
                'criteria':{'require_sampled_regular':True,'maximum_abs_curvature_per_mm':1e-10}}]}
        with EngineeringRuntime(OUT/'quality-contract-tests',intent_fixture(),delivery_rules=rules) as runtime:
            h,_=runtime.create();h,_=runtime.execute(h,ToolProgram((
                ToolCall('patch.create',{'poles':polynomial_patch().poles.tolist()},'skin'),
                ToolCall('topology.face',{'patch':'skin'},'result'))))
            with self.assertRaisesRegex(ValueError,'surface quality failed'):runtime.export_step(h,OUT/'quality-should-fail.step')


if __name__=='__main__':unittest.main(verbosity=2)
