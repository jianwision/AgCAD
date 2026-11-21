"""Executable D0-D4 diagnostics and open, model-driven constraint harness pilot.

CPU reference/preprocessing and STEP topology are explicit outer stages.
All model-driven rounds and proposal transactions run on device without readbacks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
from diagnostic_geometry import (hood_semantics, jacobian_grid, measurement_row,
                                 plane_patch, quadratic_jacobian_patch, rank_status,
                                 sample_min, surface_grid)
from model_adaptor import CuModel, ModelContext, ModelRequest, load_adaptor, observation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/diagnostics'
ROUNDS = 16
STRIDE = 404+20*ROUNDS
PROBES = [(0.25,0.5),(0.75,0.5),(0.5,0.5),(0.75,0.125),(0.25,0.875),(0.5,0.125)]
OP = {'ADD_PROBES':0, 'SET_PROBES':1, 'RESTORE_BEST':2, 'RAW_CONTROL_DELTA':3, 'STOP':4}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template():
    with np.load(ROOT/'output/harness/template.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    z = data['Z']
    e = np.array([measurement_row(*uv) for uv in PROBES])
    assert z.shape == (192,6) and rank_status(e@z)['status'] == 'FULL_RANK'
    w = z@np.linalg.pinv(e@z)
    return data, e, w


def coordinates(data, e, w):
    rng = np.random.default_rng(82453)
    z, x0 = data['Z'], data['x0']
    errors = {'gauge_mm':0., 'permutation_mm':0., 'unit_conversion_mm':0.,
              'chart_position_mm':0., 'chart_derivative_mm':0.}
    for _ in range(64):
        delta = rng.normal(size=6)
        q, _ = np.linalg.qr(rng.normal(size=(6,6)))
        errors['gauge_mm'] = max(errors['gauge_mm'], float(np.max(np.abs(z@q@np.linalg.pinv(e@z@q)@delta-w@delta))))
        perm = rng.permutation(192)
        zp, ep = z[perm], e[:,perm]
        errors['permutation_mm'] = max(errors['permutation_mm'], float(np.max(np.abs(zp@np.linalg.pinv(ep@zp)@delta-(w@delta)[perm]))))
        errors['unit_conversion_mm'] = max(errors['unit_conversion_mm'], float(np.max(np.abs(1000*(w@(delta/1000))-w@delta))))
    ts = np.linspace(.03,.97,9)
    for p in x0.reshape(4,4,4,3):
        for swap in (False,True):
            for fu in (False,True):
                for fv in (False,True):
                    changed = p.transpose(1,0,2) if swap else p.copy()
                    if fu: changed = changed[::-1]
                    if fv: changed = changed[:,::-1]
                    u, v = 1-ts if fu else ts, 1-ts if fv else ts
                    reference = surface_grid(p,v,u).transpose(1,0,2) if swap else surface_grid(p,u,v)
                    errors['chart_position_mm'] = max(errors['chart_position_mm'], float(np.max(np.abs(surface_grid(changed,ts,ts)-reference))))
                    reference_d = surface_grid(p,v,u,dv=1).transpose(1,0,2) if swap else surface_grid(p,u,v,du=1)
                    reference_d *= -1 if fu else 1
                    errors['chart_derivative_mm'] = max(errors['chart_derivative_mm'], float(np.max(np.abs(surface_grid(changed,ts,ts,du=1)-reference_d))))
    rank_cases = [
        ('canonical',e@z,'FULL_RANK'), ('duplicate_probes',np.tile((e@z)[0],(6,1)),'RANK_DEFICIENT'),
        ('near_rank_boundary',np.diag([1,1,1,1,1,1e-10]),'UNKNOWN'),
        ('clear_rank_loss',np.diag([1,1,1,1,1,1e-14]),'RANK_DEFICIENT'),
        ('fully_locked',np.zeros((6,0)),'LOCKED'), ('nonfinite',np.array([[np.nan]]),'INVALID')]
    checks = []
    for name, matrix, expected in rank_cases:
        result = rank_status(matrix)
        assert result['status'] == expected, (name,result)
        checks.append({'name':name,'expected':expected,**result})
    assert max(errors.values()) < 1e-8 and np.max(np.abs(e@w-np.eye(6))) < 1e-10
    result = {'pass':True, 'probe_uv':PROBES, 'probes_frozen_per_canonical_family':True,
              'gauge_trials':64, 'permutation_trials':64, 'chart_tests':32,
              'max_errors':errors, 'rank_cases':checks, 'condition':float(np.linalg.cond(e@z)),
              'scope':'one canonical hood family; tests equivalent coordinates/charts, not unseen topology transfer'}
    dump(OUT/'D0-coordinates.json',result)
    return result


def reachable_tasks(data, w):
    rng = np.random.default_rng(572381)
    uv = np.array([(u,v) for u in np.linspace(.05,.95,10) for v in np.linspace(.05,.95,10)])
    t = np.array([measurement_row(*p) for p in uv])
    b = t@w
    tasks, systems, refs, targets = [], [], [], []
    for i in range(12):
        witness = rng.uniform(-7,7,6)
        d = b@witness
        if i >= 6:
            d = d+12*np.exp(-((uv[:,0]-.37)**2/.025**2+(uv[:,1]-.62)**2/.04**2))
        reference = np.linalg.lstsq(b,d,rcond=None)[0]
        residual = b@reference-d
        floor = float(residual@residual)
        status = 'REACHABLE_WITNESS' if floor < 1e-14 else 'OUT_OF_REPRESENTATION_ESTIMATE'
        assert (i < 6) == (status == 'REACHABLE_WITNESS')
        assert np.linalg.norm(b.T@residual) < 1e-9
        if i < 6: assert np.max(np.abs(b@witness-d)) < 1e-12
        x = data['x0']+w@reference
        assert hood_semantics(x)['pass']
        tasks.append({'id':i,'family':'probe-fit' if i<6 else 'localized-bump',
                      'classification':status,'sampled_squared_error_floor_mm2':floor,
                      'bound_kind':'numerical orthogonal projection, not a rounded interval certificate',
                      'global_infeasibility_claim':False,'witness_in_model_observation':False})
        systems.append(np.r_[(b.T@b).ravel(),b.T@d,d@d,np.zeros(6)])
        refs.append(reference);targets.append(d)
    result = {'pass':True,'tasks':tasks,'sampling_uv':uv.tolist(),
              'full_rank':rank_status(b),'hard_constraints_independently_checked':True}
    dump(OUT/'D1-reachability.json',result)
    return np.array(systems), np.array(refs), b, np.array(targets), result


def observability():
    # The harness does not impose this observation vector. This is an adapter audit.
    q, goal = np.zeros(6), np.ones(6)
    observations = np.r_[q,goal]
    pairs = [
        {'name':'hidden_action_mask','same_adapter_observation':True,'same_proposal':'ADD_PROBES',
         'transition_a':'admitted','transition_b':'user disables edit; rejected',
         'resolution':'expose user action mask/context to proposer when such a constraint exists'},
        {'name':'hidden_geometry_revision','same_adapter_observation':True,'same_proposal':'external stale edit',
         'transition_a':'admitted','transition_b':'stale; rejected',
         'resolution':'device fills current metadata for neural proposals; external programs bind revision once'},
        {'name':'multiple_valid_actions','same_adapter_observation':True,
         'transition_a':'ADD half residual','transition_b':'SET goal',
         'is_observation_alias_failure':False,'resolution':'both legal; do not penalize disagreement with one teacher'},
        {'name':'legal_uphill_exploration','same_adapter_observation':True,
         'transition_a':'score improves; admitted','transition_b':'score worsens; admitted',
         'is_observation_alias_failure':False,'resolution':'objective feedback does not define admission'}]
    assert observations.size == 12
    result = {'pass':True,'pilot_adapter_inputs':12,'harness_fixed_observation_size':None,
              'full_scene_targets_constraints_trace_available_to_external_proposer':True,
              'pairs':pairs,'trust_radius_imposed_by_harness':False,
              'metadata_feature_geometry_queries':0,
              'scope':'explicit transition counterexamples and adapter contract; no proof of full task observability'}
    dump(OUT/'D2-observability.json',result)
    return result


def blind_spots(data):
    rng = np.random.default_rng(527491)
    patch = None
    for attempt in range(128):
        center = rng.uniform(.035,.215)
        width = .7*min(center,.25-center)
        candidate = quadratic_jacobian_patch(center,width*width)
        if sample_min(candidate,5)[0] > 1e-6 and sample_min(candidate,65)[0] < -1e-4:
            patch = candidate;break
    assert patch is not None
    before = patch.copy(); baseline = plane_patch()
    # One-minimal under restoring a single changed control coordinate to baseline.
    changed = True
    while changed:
        changed = False
        for index in np.flatnonzero(np.abs(patch-baseline).ravel()>1e-12):
            trial = patch.copy();trial.ravel()[index] = baseline.ravel()[index]
            if sample_min(trial,5)[0]>1e-6 and sample_min(trial,65)[0]<-1e-4:
                patch = trial;changed = True
    assert all(not(sample_min(np.where(np.arange(48).reshape(4,4,3)==i,baseline,patch),5)[0]>1e-6
                   and sample_min(np.where(np.arange(48).reshape(4,4,3)==i,baseline,patch),65)[0]<-1e-4)
               for i in np.flatnonzero(np.abs(patch-baseline).ravel()>1e-12))
    cases, records = [], []
    def add(name,p,expected):
        cases.append(p)
        if np.isfinite(p).all():
            minimum, uv = sample_min(p,129)
            legacy = sample_min(p,5)[0]
        else: minimum, uv, legacy = None,None,None
        records.append({'name':name,'expected_guard':expected,'dense_minimum':minimum,
                        'dense_witness_uv':uv,'legacy_sample_minimum':legacy})
    add('flat_valid',plane_patch(),1)
    add('subdivision_needed_valid',quadratic_jacobian_patch(.113,.003,False),1)
    add('searched_fold',before,2)
    add('minimized_fold',patch,2)
    borderline = plane_patch();borderline[:,:,0]*=1e-6
    add('numerical_boundary_unknown',borderline,0)
    nonfinite = plane_patch();nonfinite[1,1,2] = np.nan
    add('nonfinite_patch',nonfinite,3)
    from harness_demo import read_results
    _, previous, _ = read_results()
    for family,index in [('parent',0),('alternative',31)]:
        for i,p in enumerate(previous[index]['x'].reshape(4,4,4,3)):
            add(f'{family}_hood_patch_{i}',p,1)
    # Shared-row corruption: removing every row involving variable zero fools any
    # validator that trusts that same corrupted compiler output.
    x = previous[0]['x'].copy();x[0]+=10.
    keep = (data['hard']==1)&(data['A'][:,0]==0)
    b = data['bmap'][:,0]+.6*data['bmap'][:,1]
    ratio = float(np.max(np.abs(data['A'][keep]@x-b[keep])/data['tol'][keep]))
    semantic = hood_semantics(x)
    assert ratio < 1 and not semantic['pass']
    counterexample = {'seed':527491,'search_attempts':attempt+1,
        'before_poles':before.tolist(),'after_poles':patch.tolist(),
        'changed_coordinates_before':int(np.count_nonzero(np.abs(before-baseline)>1e-12)),
        'changed_coordinates_after':int(np.count_nonzero(np.abs(patch-baseline)>1e-12)),
        'minimality':'one-minimal for tested single-coordinate baseline restorations on the fixed 65x65 witness grid',
        'not_global_minimality_or_completeness':True,'shared_dependency_fault':{
            'corrupted_row_validator_pass':True,'corrupted_row_hard_ratio':ratio,
            'independent_semantic_validator':semantic}}
    dump(OUT/'counterexample.json',counterexample)
    return np.array(cases), records, counterexample


def proposal_programs(external, adaptor_kind):
    def action(op, values=None): return [OP.get(op,op),*(np.zeros(6) if values is None else values)]
    target = np.array([4.,-3.,2.,1.,-2.,3.])
    programs = [
        {'name':'legal_uphill_then_goal','mode':'script','initial':[0.]*6,'target':target.tolist(),
         'actions':[action('ADD_PROBES',-target),action('ADD_PROBES',2*target),action('STOP')]},
        {'name':'composed_set_restore','mode':'script','initial':[0.]*6,'target':target.tolist(),
         'actions':[action('SET_PROBES',target),action('ADD_PROBES',-target),action('RESTORE_BEST'),action('STOP')]},
        {'name':'invalid_raw_rollback','mode':'script','initial':[0.]*6,'target':target.tolist(),
         'actions':[action('RAW_CONTROL_DELTA',[0,10,0,0,0,0]),action('SET_PROBES',target),action('STOP')]},
        {'name':'stop_is_not_success','mode':'script','initial':[0.]*6,'target':target.tolist(),'actions':[action('STOP')]},
        {'name':'invalid_opcode_rollback','mode':'script','initial':[0.]*6,'target':target.tolist(),
         'actions':[action(999),action('SET_PROBES',target),action('STOP')]},
    ]
    rng = np.random.default_rng(728401)
    requests=[ModelRequest('fit_00',(0.,)*6,tuple(target))]
    for i in range(1,16):
        requests.append(ModelRequest(f'fit_{i:02}',tuple(rng.uniform(-7,7,6)),tuple(rng.uniform(-7,7,6))))
    requests=tuple(requests)
    context=ModelContext('scene.json')
    weights=OUT/'model/probe_policy.f64'
    batch=load_adaptor(adaptor_kind,weights,external).propose(context,requests)
    programs.extend(batch.programs)
    if not batch.device_model_inference:
        for p in CuModel(weights).propose(context,requests).programs:
            p=dict(p);p['name']='reference_cuda_'+p['request_id'];programs.append(p)
    dump(OUT/'model-observation.json',observation(context,requests))
    dump(OUT/'adaptor.json',{'selected':batch.adaptor,'execution':batch.execution,
        'selected_programs':[p['name'] for p in batch.programs], 'host_model_calls':batch.host_model_calls,
        'device_model_inference':batch.device_model_inference,'llm_network_calls':0,
        'live_llm_provider_connected':False,'same_harness_and_operation_contract':True,
        'current_CuModel_learned_output':'six ADD_PROBES parameters; opcode selection is not learned',
        'runtime_operations':list(OP)})
    assert len(programs)<=64 and len({p['name'] for p in programs})==len(programs)
    packed = []
    for p in programs:
        if p['mode'] not in ('script','neural') or len(p['initial'])!=6 or len(p['target'])!=6:
            raise ValueError('Invalid proposal mode or initial/target dimension')
        actions = p['actions']
        if len(actions)>ROUNDS: raise ValueError('Proposal round budget exceeded')
        normalized = []
        for a in actions:
            if isinstance(a,dict): a=action(a['op'],a.get('values'))
            if len(a)!=7: raise ValueError('Every proposal has opcode and six arguments')
            normalized.append(a)
        normalized += [action('STOP') for _ in range(ROUNDS-len(normalized))]
        p['actions'] = [list(a) for a in normalized]
        packed.append(np.r_[p['target'],p['initial'],1 if p['mode']=='neural' else 0,np.array(normalized).ravel()])
    return programs,np.array(packed)


def prepare(external=None, adaptor_kind='cuda'):
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    weights_path=OUT/'model/probe_policy.f64'
    if not weights_path.exists(): raise FileNotFoundError('Run .\\run.ps1 train-proposals first')
    weights=np.fromfile(weights_path,dtype='<f8')
    assert len(weights)==462 and np.isfinite(weights).all()
    data,e,w=template()
    coordinates(data,e,w);observability()
    systems,refs,b,targets,_=reachable_tasks(data,w)
    patches,records,counterexample=blind_spots(data)
    programs,packed=proposal_programs(external,adaptor_kind)
    dump(OUT/'scene.json',{'contract':'hood-fixed-xy-perimeter-C2-symmetry-v1',
         'units':'mm','reference_poles':data['x0'].reshape(4,4,4,3).tolist(),
         'probe_uv':PROBES,'coordinates':'probe displacement relative to reference poles in mm',
         'operations':OP,'quality_is_admission_criterion':False})
    header=struct.pack('<8s6I',b'GPDIA001',1,len(patches),len(systems),16,len(programs),ROUNDS)
    binary=header+b''.join(np.asarray(a,dtype='<f8').tobytes() for a in
                          [patches,systems,data['x0'],w,e,weights,packed])
    (OUT/'input.bin').write_bytes(binary)
    np.savez(OUT/'references.npz',systems=systems,references=refs,B=b,targets=targets,
             patches=patches,x0=data['x0'],W=w,E=e)
    dump(OUT/'programs.json',{'contract':'hood-proposals-v1','opcodes':OP,'programs':programs,
         'constraint_scope':'fixed XY, perimeter, compatible C2 seams, symmetry; absolute coordinate limit 1e6 mm',
         'quality_is_admission_criterion':False,'model_can_request_legal_uphill_moves':True})
    dump(OUT/'preparation.json',{'pass':True,'patch_cases':records,'program_count':len(programs),
         'prepare_wall_ms':(time.perf_counter()-start)*1000,'input_sha256':sha(OUT/'input.bin'),
         'model_sha256':sha(weights_path),'source_template_sha256':sha(ROOT/'output/harness/template.npz'),
         'guard_scope':'signed XY Jacobian, bounded f64 Bernstein subdivision; UNKNOWN near numerical/depth limits',
         'v6_archive_sha256':sha(ROOT/'output/delivery/rust-cuda-modeling-demo-v6.zip')})
    print(json.dumps({'prepared':True,'geometry_cases':len(patches),'systems':len(systems),'proposal_programs':len(programs)}))


def model_forward(weights, q, target):
    hidden=np.tanh(np.r_[q/10,target/10]@weights[:288].reshape(12,24)+weights[288:312])
    return 10*(hidden@weights[312:456].reshape(24,6)+weights[456:])


def export_step(coordinates, destination=None):
    sys.path.insert(0,str(ROOT/'.python-deps'))
    from acceptance import build_surface, count
    from cad_ops import sew_surfaces
    from cad_advanced import untrim_surface
    from placement_ops import snap_point
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    start=time.perf_counter()
    nets=coordinates.reshape(4,4,4,3);faces=[]
    for p in nets:
        rec={'kind':'nurbs','size':[4,4],'degree':[3,3],
             'u_knots':[0.]*4+[1.]*4,'v_knots':[0.]*4+[1.]*4,
             'poles_xyzw':np.concatenate((p,np.ones((4,4,1))),axis=2).ravel().tolist()}
        faces.append(untrim_surface(build_surface(rec)))
    shape=sew_surfaces(faces,tolerance=1e-6)
    assert BRepCheck_Analyzer(shape).IsValid()
    path=OUT/'model_hood.step' if destination is None else Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=STEPControl_Writer()
    assert writer.Transfer(shape,STEPControl_AsIs)==IFSelect_RetDone
    assert writer.Write(str(path))==IFSelect_RetDone
    reader=STEPControl_Reader()
    assert reader.ReadFile(str(path))==IFSelect_RetDone and reader.TransferRoots()>0
    reopened=reader.OneShape()
    assert BRepCheck_Analyzer(reopened).IsValid() and count(reopened,TopAbs_FACE)==4 and count(reopened,TopAbs_SOLID)==0
    maximum=0.
    for p in nets:
        points=surface_grid(p,np.linspace(.021,.979,9),np.linspace(.033,.967,9))
        for point in points.reshape(-1,3):
            _,meta=snap_point(point,target=reopened,max_distance=1e-7)
            maximum=max(maximum,meta['distance_mm'])
    assert maximum<1e-7
    return {'pass':True,'file':str(path),'sha256':sha(path),'faces':4,'solids':0,
            'probe_count':324,'probe_max_mm':maximum,'wall_ms':(time.perf_counter()-start)*1000,
            'solidworks_import_performed':False}


def validate_programs(programs, results, reference, weights):
    x0,w=reference['x0'],reference['W']
    receipts=[];max_parity=0.;selected=None
    for program,result in zip(programs,results):
        q=np.array(program['initial']);target=np.array(program['target']);bestq=q.copy()
        score=float(np.sum((q-target)**2));best=score;commits=rejects=worse=0;stopped=False
        trace=result[404:].reshape(ROUNDS,20)
        for round,row in enumerate(trace):
            if stopped:
                assert row[1]==5 and row[4]==score
                continue
            if program['mode']=='neural':
                opcode=0;values=model_forward(weights,q,target)
                max_parity=max(max_parity,float(np.max(np.abs(row[12:18]-values))))
                assert np.max(np.abs(row[12:18]-values))<1e-9
            else:
                opcode=int(program['actions'][round][0]);values=np.array(program['actions'][round][1:])
            assert int(row[0])==opcode and abs(row[2]-score)<1e-7
            if opcode==4:
                assert row[1]==4;stopped=True
            elif opcode>4:
                assert row[1]==1;rejects+=1
            else:
                trialq=q+values if opcode==0 else values.copy() if opcode==1 else bestq.copy()
                trial=x0+w@trialq
                if opcode==3:
                    trial=x0+w@q;trial[int(values[0])]+=values[1]
                    trialq=reference['E']@(trial-x0)
                valid=hood_semantics(trial)['pass']
                if valid:
                    assert row[1]==0,(program['name'],round,row[1])
                    trialscore=float(np.sum((trialq-target)**2))
                    if trialscore>score+1e-10: worse+=1
                    q=trialq;score=trialscore;commits+=1
                    if score<best: best=score;bestq=q.copy()
                else:
                    assert row[1]==3;rejects+=1
            assert np.max(np.abs(row[6:12]-q))<1e-8
            assert abs(row[4]-score)<1e-7 and abs(row[5]-best)<1e-7
        assert np.max(np.abs(result[:192]-(x0+w@q)))<1e-8
        assert np.max(np.abs(result[192:384]-(x0+w@bestq)))<1e-8
        assert np.max(np.abs(result[396:399]-[commits,rejects,worse]))==0
        semantic=hood_semantics(result[192:384]);assert semantic['pass']
        max_error=float(np.max(np.abs(bestq-target)))
        receipt={'name':program['name'],'model_mode':program['mode'],'execution_valid':True,
                 'constraint_check':semantic,'commits':commits,'rejections':rejects,
                 'legal_uphill_commits':worse,'stopped':stopped,
                 'best_probe_max_error_mm':max_error,'goal_satisfied':max_error<=.5,
                 'best_score':best,'final_score':score}
        receipts.append(receipt)
        if program['mode']=='neural' and (selected is None or best<selected[0]):
            selected=(best,program['name'],result[192:384].copy())
    by_name={r['name']:r for r in receipts}
    assert by_name['legal_uphill_then_goal']['legal_uphill_commits']>=1
    assert by_name['composed_set_restore']['goal_satisfied'] and by_name['composed_set_restore']['legal_uphill_commits']>=1
    assert by_name['invalid_raw_rollback']['rejections']==1
    assert by_name['invalid_opcode_rollback']['rejections']==1
    assert by_name['stop_is_not_success']['stopped'] and not by_name['stop_is_not_success']['goal_satisfied']
    neural=[r for r in receipts if r['model_mode']=='neural']
    assert len(neural)==16 and all(r['goal_satisfied'] for r in neural)
    return receipts,max_parity,selected


def validate():
    start=time.perf_counter()
    prep=json.loads((OUT/'preparation.json').read_text())
    gpu=json.loads((OUT/'gpu_report.json').read_text())
    assert sha(OUT/'input.bin')==prep['input_sha256']
    assert sha(OUT/'model/probe_policy.f64')==prep['model_sha256']
    assert sha(ROOT/'output/delivery/rust-cuda-modeling-demo-v6.zip')==prep['v6_archive_sha256']
    assert gpu['intermediate_readbacks']==0 and gpu['dtoh_copies']==1 and gpu['replay_mismatched_words']==0
    raw=np.fromfile(OUT/'result.bin',dtype='<f8')
    assert len(raw)==gpu['output_words'] and np.isfinite(raw).all()
    with np.load(OUT/'references.npz',allow_pickle=False) as a: reference={k:a[k] for k in a.files}
    p,s=gpu['patches'],gpu['systems']
    geometry=raw[:8*p].reshape(p,8)
    solvers=raw[8*p:8*p+20*s].reshape(s,2,10)
    proposal_results=raw[8*p+20*s:-1].reshape(gpu['proposal_programs'],STRIDE)
    guard_cases=[]
    for i,case in enumerate(prep['patch_cases']):
        result=geometry[i]
        assert result[2]==case['expected_guard'],(case['name'],result)
        if case['legacy_sample_minimum'] is not None:
            assert abs(result[0]-case['legacy_sample_minimum'])<max(1e-10,abs(result[0])*1e-10)
        guard_cases.append({**case,'legacy_accept':bool(result[1]),'new_status':int(result[2]),
                            'subdivision_nodes':int(result[3]),'decision_margin':float(result[5])})
    folds=[r for r in guard_cases if 'fold' in r['name']]
    assert len(folds)==2 and all(r['legacy_accept'] and r['new_status']==2 for r in folds)
    d4={'pass':True,'cases':guard_cases,'legacy_false_accepts_found':len(folds),
        'new_false_accepts_on_tested_cases':0,'unknown_cases':sum(r['new_status']==0 for r in guard_cases),
        'evidence_ledger':[{'claim':'5x5 samples suffice for arbitrary bicubic XY orientation',
          'status':'REVOKED_EVIDENCE','counterexample':'counterexample.json'},
          {'claim':'original hood domain with fixed XY','status':'RE_AUDITED',
           'parent_and_alternative_patch_checks':8,'v6_archive_modified':False}],
        'scope':'f64 bounded Bernstein subdivision with margin, not formal interval proof or general CAD validity'}
    dump(OUT/'D4-verifiers.json',d4)
    comparisons=[]
    for i,result in enumerate(solvers):
        for method,label in enumerate(['direct-pivoted-6x6','steepest-descent-128']):
            q=result[method,:6]
            error=float(np.max(np.abs(q-reference['references'][i])))
            cost=float(np.sum((reference['B']@q-reference['targets'][i])**2))
            assert result[method,9]==1 and error<1e-5,(i,label,error)
            assert abs(cost-result[method,6])<1e-7
            comparisons.append({'task':i,'method':label,'q_reference_max_mm':error,
                                'objective':cost,'iterations':int(result[method,7]),
                                'final_gradient_norm':float(result[method,8])})
    timings=[]
    for _ in range(20):
        timer=time.perf_counter()
        for d in reference['targets']: np.linalg.lstsq(reference['B'],d,rcond=None)
        timings.append((time.perf_counter()-timer)*1000)
    programs=json.loads((OUT/'programs.json').read_text())['programs']
    receipts,parity,selected=validate_programs(programs,proposal_results,reference,
                                              np.fromfile(OUT/'model/probe_policy.f64',dtype='<f8'))
    training=json.loads((OUT/'model/training_report.json').read_text())
    d3={'pass':True,'deterministic_comparisons':comparisons,
        'cpu_qr_batch_median_ms':float(np.median(timings)),'cpu_qr_batch_samples':len(timings),
        'cpu_qr_excludes_template_construction':True,'gpu_profiles':gpu,
        'neural_pilot':{'tasks':16,'goal_successes':sum(r['goal_satisfied'] for r in receipts if r['model_mode']=='neural'),
                        'cpu_gpu_proposal_parity_max_mm':parity,'training':training,
                        'strong_reference':'SET_PROBES reaches each six-probe goal exactly in one legal action',
                        'model_quality_advantage_demonstrated':False,'model_speed_advantage_demonstrated':False},
        'cross_evaluation':{'status':'N/A','reason':'no promoted parent/new feedback-policy pair; diagnostic model is a new adapter'},
        'qualification_or_statistical_promotion_performed':False,
        'scope':'pipeline and strong-baseline diagnostics; synthetic affine tasks do not establish nonlinear CAD learning'}
    dump(OUT/'D3-baselines.json',d3)
    cad=export_step(selected[2])
    report={'pass':True,'contract':'model-adaptor-constraint-harness-v1',
        'adaptor':json.loads((OUT/'adaptor.json').read_text()),'gpu':gpu,
        'diagnostics':{name:json.loads((OUT/f'{name}-{suffix}.json').read_text())['pass'] for name,suffix in
                       [('D0','coordinates'),('D1','reachability'),('D2','observability'),('D3','baselines'),('D4','verifiers')]},
        'proposal_programs':receipts,'cpu_gpu_model_proposal_max_mm':parity,
        'selected_neural_program':selected[1],'cad':cad,
        'validation_wall_ms':(time.perf_counter()-start)*1000,
        'v6_archive_unchanged':True,'recursive_improvement_claim':False,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'cuda/diagnostic_suite.cu',
            ROOT/'src/bin/diagnostic_suite.rs',ROOT/'scripts/diagnostic_suite.py',
            ROOT/'scripts/diagnostic_geometry.py',ROOT/'scripts/model_adaptor.py']}}
    dump(OUT/'validation.json',report)
    # Preserve independently reviewable results when selecting another adaptor.
    snapshot=OUT/'runs'/report['adaptor']['selected']
    snapshot.mkdir(parents=True,exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file(): (snapshot/path.name).write_bytes(path.read_bytes())
    (snapshot/'model').mkdir(exist_ok=True)
    for path in (OUT/'model').iterdir():
        if path.is_file(): (snapshot/'model'/path.name).write_bytes(path.read_bytes())
    saved=dict(report);saved['cad']={**cad,'file':str(snapshot/'model_hood.step')}
    dump(snapshot/'validation.json',saved)
    print(json.dumps({'pass':True,'diagnostics':report['diagnostics'],'programs':len(receipts),
                      'model_proposal_parity_mm':parity,'neural_successes':d3['neural_pilot']['goal_successes'],
                      'step_probe_max_mm':cad['probe_max_mm']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','validate'])
    parser.add_argument('--proposals',type=Path)
    parser.add_argument('--adaptor',choices=['cuda','replay'],default='cuda')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.proposals,args.adaptor)
    else: validate()
