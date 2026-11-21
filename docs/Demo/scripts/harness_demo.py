"""Shared-template GPU harness preparation, independent acceptance and export."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np
from constraint_demo import ROOT, hood, independent_kkt, patch_nets, patch_value

OUT = ROOT/'output/harness'
OP = {'AUTO': 0, 'SET_PARAMETER': 1, 'EDIT_CONTROL_DELTA': 2, 'NOOP': 3, 'STOP': 4}
REASONS = ['ACCEPTED', 'CACHE_HIT', 'STOPPED', 'INACTIVE', 'BUDGET_EXHAUSTED',
           'STALE_REVISION', 'NEED_RECOMPILE', 'STALE_GOAL', 'INVALID_OPCODE',
           'INVALID_PARAMETER', 'NONFINITE_PARAMETER', 'HARD_CONSTRAINT_FAILED',
           'GEOMETRY_CHECK_FAILED', 'NUMERICAL_CHECK_FAILED', 'NO_IMPROVEMENT', 'INVALID_BASELINE']
STATE_NAMES = ['status','revision','budget','goal','parameter','objective','hard_ratio','sampled_jac_min',
               'relative_gradient','last_reason','commits','rejections','cache_hits','evaluations',
               'verification_flags','attempts','pending','trial_parameter','resolved_opcode','reason',
               'trial_objective','trial_hard_ratio','trial_sampled_jac_min','trial_gradient','trial_flags',
               'before_revision','before_budget','before_parameter','before_objective','requested_opcode','parameter_finite','reserved']
TRACE_NAMES = ['round','requested_opcode','resolved_opcode','reason','revision_before','revision_after',
               'budget_before','budget_after','parameter_before','trial_parameter','parameter_after',
               'objective_before','trial_objective','objective_after','trial_hard_ratio','trial_sampled_jac_min',
               'trial_gradient','cache_hit','evaluated','trial_flags','status_after','attempts','commits','rejections']
KEYS = ['Z', 'x0', 'ymap', 'A', 'bmap', 'tol', 'hard', 'weight', 'B', 'dmap']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def shared_template():
    base, end = hood(240.), hood(290.)
    signature = json.dumps([base.to_dict(), end.to_dict()], sort_keys=True).encode()
    for filename in ('constraint_graph.py', 'constraint_demo.py', 'harness_demo.py'):
        signature += (ROOT/'scripts'/filename).read_bytes()
    signature += np.__version__.encode()
    key = hashlib.sha256(signature).hexdigest()
    cache, meta = OUT/'template.npz', OUT/'template_cache.json'
    hit = cache.exists() and meta.exists() and json.loads(meta.read_text())['key'] == key
    if hit:
        metadata = json.loads(meta.read_text())
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == metadata['npz_sha256']
        with np.load(cache, allow_pickle=False) as archive:
            data = {k: archive[k] for k in KEYS}
        diagnostics = metadata['diagnostics']
    else:
        c0, c1 = base.compile(), end.compile()
        assert np.array_equal(c0['B'], c1['B']) and np.array_equal(c0['Z'], c1['Z'])
        a, b, hard, tol, weight = base.matrices()
        assert np.array_equal(a, end.matrices()[0])
        dmap = np.column_stack((c0['d'], c1['d']-c0['d']))
        # QR/SVD least-squares map, no explicit inverse/normal-equation solve.
        ymap = np.linalg.lstsq(c0['B'], dmap, rcond=None)[0]
        data = {'Z': c0['Z'], 'x0': c0['x0'], 'ymap': ymap, 'A': a,
                'bmap': np.column_stack((b, c1['b']-b)), 'tol': tol,
                'hard': hard.astype(float), 'weight': np.where(hard, 0., weight/tol**2),
                'B': c0['B'], 'dmap': dmap}
        diagnostics = c0['diagnostics']
        np.savez(cache, **data)
        dump(meta, {'key': key, 'diagnostics': diagnostics,
                    'npz_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()})
    assert all(np.isfinite(x).all() for x in data.values())
    return data, base, {'key': key, 'cpu_template_cache_hit': hit, 'diagnostics': diagnostics}


def command(op='AUTO', revision=0, template=1, coefficient=1, goal=1, index=0, value=0.):
    return [OP.get(op, op), revision, template, coefficient, goal, index, float(value)]


def write_input(path, data, requests, commands):
    path.mkdir(parents=True, exist_ok=True)
    n, k = data['Z'].shape
    m, r = len(data['dmap']), len(data['bmap'])
    header = struct.pack('<8s12I9d', b'GPHARN01', 1, n, m, k, r, len(requests), len(commands),
                         1, 1, 1, 24, 0, 200., 320., 1e-9, 1e-9, 1e-12, 0., 1e-6, 240., 50.)
    assert len(header) == 128
    shared = b''.join(np.asarray(data[key], dtype='<f8').tobytes() for key in KEYS)
    request_bytes = b''.join(struct.pack('<ddII', q['goal'], q['initial'], q['budget'], q['goal_revision']) for q in requests)
    command_bytes = b''.join(struct.pack('<6Id', *cmd) for row in commands for cmd in row)
    binary = header+shared+request_bytes+command_bytes
    (path/'input.bin').write_bytes(binary)
    return binary, len(header)+len(shared)


def prepare_verifier_fixtures(data):
    # Fault injection exercises independent checks, not new accepted hood designs.
    geometry = {k: v.copy() for k, v in data.items()}
    affected = (geometry['hard'] == 1) & (geometry['A'][:, 0] != 0)
    geometry['hard'][affected] = 0
    geometry['weight'][affected] = 0
    numerical = {k: v.copy() for k, v in data.items()}
    numerical['ymap'][0, 1] += 10.
    invalid = {k: v.copy() for k, v in data.items()}
    invalid['x0'][0] += 50.
    for name, template, initial, cmd in [
        ('geometry_failure', geometry, 270., command('EDIT_CONTROL_DELTA', index=0, value=2000)),
        ('numerical_failure', numerical, 240., command('SET_PARAMETER', value=270)),
        ('invalid_baseline', invalid, 270., command()),
    ]:
        requests = [{'goal':270., 'initial':initial, 'budget':1, 'goal_revision':1} for _ in range(2)]
        write_input(OUT/name, template, requests, [[cmd, command('STOP')]])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    timer = time.perf_counter()
    data, graph, cache = shared_template()
    # A second load checks cache identity and payload integrity, not just a flag.
    cached, _, warm = shared_template()
    assert warm['cpu_template_cache_hit'] and all(np.array_equal(data[k], cached[k]) for k in KEYS)
    candidates, rounds = 32, 8
    requests = [{'goal': 270., 'initial': float(theta), 'budget': 16, 'goal_revision': 1}
                for theta in np.linspace(220, 290, candidates)]
    requests[26]['budget'] = 0
    requests[27]['budget'] = 1
    commands = [[command() for _ in requests] for _ in range(rounds)]
    commands[0][14] = command('STOP', revision=99)
    commands[0][15] = command('NOOP')
    fixtures = {
        16: command('SET_PARAMETER', template=99, value=270),
        17: command('SET_PARAMETER', coefficient=99, value=270),
        18: command('SET_PARAMETER', revision=99, value=270),
        19: command('SET_PARAMETER', value=999),
        20: command('SET_PARAMETER', value=float('nan')),
        21: command(999),
        22: command('EDIT_CONTROL_DELTA', index=0, value=50),
        23: command('EDIT_CONTROL_DELTA', index=2, value=-100000),
        24: command('SET_PARAMETER', value=200),
        25: command('SET_PARAMETER', value=requests[25]['initial']),
        27: command('SET_PARAMETER', value=999),
        28: command('SET_PARAMETER', goal=99, value=270),
        29: command('EDIT_CONTROL_DELTA', index=999, value=1),
        30: command('STOP'),
        31: command('SET_PARAMETER', value=280),
    }
    for c, cmd in fixtures.items():
        commands[0][c] = cmd
    binary, shared_bytes = write_input(OUT, data, requests, commands)
    prepare_verifier_fixtures(data)
    readable = [[{'opcode': cmd[0], 'expected_revision': cmd[1], 'template_revision': cmd[2],
                 'coefficient_revision': cmd[3], 'goal_revision': cmd[4], 'index': cmd[5],
                 'value': cmd[6] if math.isfinite(cmd[6]) else None,
                 'value_class': 'finite' if math.isfinite(cmd[6]) else 'nan'} for cmd in row] for row in commands]
    dump(OUT/'program.json', {'schema': 1, 'requests': requests, 'rounds': readable,
                             'opcodes': OP, 'reason_codes': REASONS,
                             'immutable_goal': 'All candidates scored against the user target 270 mm; commands cannot change it.'})
    dump(OUT/'graph.json', graph.to_dict())
    dump(OUT/'preparation.json', {'pass': True, **cache, 'prepare_ms': (time.perf_counter()-timer)*1000,
                                'warm_cache_identity_verified': True,
                                'shared_bytes': shared_bytes, 'input_bytes': len(binary),
                                'duplicated_template_bytes_avoided': (shared_bytes-128)*(candidates-1),
                                'input_sha256': hashlib.sha256(binary).hexdigest()})
    print(json.dumps({'prepared': candidates, 'rounds': rounds, **cache, 'input_bytes': len(binary)}))


def read_results(path=OUT):
    report = json.loads((path/'gpu_report.json').read_text())
    raw = np.fromfile(path/'result.bin', dtype='<f8')
    assert len(raw) == report['output_words']
    c, stride, rounds = report['candidates'], report['stride'], report['rounds']
    body = raw[:c*stride].reshape(c, stride)
    state_offset = stride-32-rounds*24
    k = (state_offset-384)//2
    return report, [{'x': row[:192], 'trial_x': row[192:384], 'y': row[384:384+k],
                     'state': row[state_offset:state_offset+32], 'trace': row[-rounds*24:].reshape(rounds, 24)} for row in body], raw[-8:]


def export_selected(result):
    from acceptance import build_surface, roundtrip, count, properties
    from cad_ops import sew_surfaces
    from cad_advanced import untrim_surface
    from placement_ops import snap_point
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    nets = patch_nets(result['x'])
    surfaces, records = [], []
    for p in nets.reshape(4, 4, 4, 3):
        record = {'kind': 'nurbs', 'size': [4, 4], 'degree': [3, 3],
                  'u_knots': [0.]*4+[1.]*4, 'v_knots': [0.]*4+[1.]*4,
                  'poles_xyzw': np.concatenate((p, np.ones((4, 4, 1))), axis=2).ravel().tolist()}
        surfaces.append(build_surface(record))
        records.append(record)
    shell = sew_surfaces([untrim_surface(s) for s in surfaces], tolerance=1e-6)
    assert count(shell, TopAbs_FACE) == 4
    receipt = roundtrip('harness_hood.step', shell, 0)
    Path(receipt['file']).replace(OUT/'harness_hood.step')
    receipt['file'] = str(OUT/'harness_hood.step')
    reader = STEPControl_Reader()
    assert reader.ReadFile(receipt['file']) == IFSelect_RetDone and reader.TransferRoots() > 0
    reopened = reader.OneShape()
    probe_error = 0.
    for p in nets.reshape(4, 4, 4, 3):
        for u in np.linspace(.023, .977, 7):
            for v in np.linspace(.031, .969, 7):
                _, meta = snap_point(patch_value(p, u, v), target=reopened, max_distance=1e-7)
                probe_error = max(probe_error, meta['distance_mm'])
    seams = [0., 0., 0.]
    for direction in (0, 1):
        for side in range(2):
            a, b = (nets[0, side], nets[1, side]) if direction == 0 else (nets[side, 0], nets[side, 1])
            for t in np.linspace(0, 1, 101):
                ua, va, ub, vb = (1, t, 0, t) if direction == 0 else (t, 1, t, 0)
                for d in range(3):
                    du, dv = (d, 0) if direction == 0 else (0, d)
                    seams[d] = max(seams[d], float(np.linalg.norm(patch_value(a, ua, va, du, dv)-patch_value(b, ub, vb, du, dv))))
    assert max(seams) < 1e-7
    BRepMesh_IncrementalMesh(shell, 1., False, .1, False).Perform()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    assert writer.Write(shell, str(OUT/'harness_hood.stl'))
    dump(OUT/'surfaces.json', {'surfaces': records})
    return {'pass': True, 'step_roundtrip': receipt, 'step_probe_count': 196,
            'step_probe_max_mm': probe_error, 'seam_C0_C1_C2_max': seams, 'area_mm2': properties(shell)[0]}


def validate_verifier_fixtures():
    receipts = []
    for name, reason in [('geometry_failure',12), ('numerical_failure',13), ('invalid_baseline',15)]:
        gpu, results, selection = read_results(OUT/name)
        x, baseline = results
        s, trace = x['state'], x['trace'][0]
        assert gpu['replay_mismatched_words'] == 0 and trace[3] == reason
        # The second candidate executes STOP on the same baseline. Bit equality
        # checks rollback of every current pole, DOF and validity metric.
        assert np.array_equal(x['x'], baseline['x']) and np.array_equal(x['y'], baseline['y'])
        assert np.array_equal(s[[1,4,5,6,7,8,14]], baseline['state'][[1,4,5,6,7,8,14]])
        assert s[1] == s[10] == 0
        if reason == 12:
            assert int(trace[19]) & 2 and not (int(trace[19]) & 4)
            assert trace[14] <= 1 and trace[15] < 0 and s[0] == 0
            # Recompute sampled orientation from the actual rejected control net.
            jac = min(np.cross(patch_value(p,u,v,1,0),patch_value(p,u,v,0,1))[2]
                      for p in patch_nets(x['trial_x']).reshape(4,4,4,3)
                      for u in np.linspace(0,1,5) for v in np.linspace(0,1,5))
            assert abs(jac-trace[15]) < 1e-7
        elif reason == 13:
            assert int(trace[19]) == 6 and trace[16] > 1e-9 and s[0] == 0
            with np.load(OUT/'template.npz', allow_pickle=False) as data:
                b, d = data['B'], data['dmap'][:,0]+.6*data['dmap'][:,1]
                k = b.shape[1]
                # The trial-y block precedes state in result.bin.
                raw = np.fromfile(OUT/name/'result.bin', dtype='<f8')
                y = raw[384+k:384+2*k]
                gradient = np.linalg.norm(b.T@(b@y-d))/max(np.linalg.norm(b.T@d),1.)
            assert abs(gradient-trace[16]) < 1e-10
        else:
            assert s[0] == 3 and selection[2] == 0 and selection[0] == -1
        receipts.append({'fixture':name, 'pass':True, 'reason':REASONS[reason],
                         'bit_exact_rollback':True, 'replay_mismatched_words':0})
    return receipts


def validate_host_rejections():
    binary = (OUT/'input.bin').read_bytes()
    shared_bytes = json.loads((OUT/'preparation.json').read_text())['shared_bytes']
    n, m, k, r = struct.unpack_from('<4I', binary, 12)
    tol_offset = 128+8*(n*k+n+2*k+r*n+2*r)
    mutations = [('bad_magic',0,b'INVALID!'), ('unknown_version',8,struct.pack('<I',2)),
                 ('zero_dof',20,struct.pack('<I',0)), ('zero_candidates',28,struct.pack('<I',0)),
                 ('nonfinite_template',128,struct.pack('<d',float('nan'))),
                 ('negative_tolerance',tol_offset,struct.pack('<d',-1)),
                 ('nonfinite_request',shared_bytes,struct.pack('<d',float('nan'))),
                 ('oversized_budget',shared_bytes+16,struct.pack('<I',257))]
    invalids = [(name,binary[:offset]+value+binary[offset+len(value):]) for name,offset,value in mutations]
    invalids += [('truncated',binary[:-1]),('trailing_byte',binary+b'!')]
    receipts = []
    with tempfile.TemporaryDirectory(prefix='host-reject-',dir=OUT) as tmp:
        path = Path(tmp)
        for name, bad in invalids:
            (path/'input.bin').write_bytes(bad)
            result = subprocess.run([str(ROOT/'target/release/gpu_harness.exe'), str(path)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert result.returncode != 0 and not (path/'result.bin').exists(), (name,result)
            assert any(message in result.stderr.lower() for message in ('invalid','unsupported','disagrees','nonfinite')), (name,result.stderr)
            receipts.append({'fixture':name,'pass':True,'exit_code':result.returncode,'error':result.stderr.strip()})
    return receipts


def validate():
    gpu, results, selection = read_results()
    prep = json.loads((OUT/'preparation.json').read_text())
    assert hashlib.sha256((OUT/'input.bin').read_bytes()).hexdigest() == prep['input_sha256']
    assert gpu['replay_mismatched_words'] == 0 and selection[4] == 0
    assert selection[2] == 1 and 0 <= selection[0] < len(results)
    program = json.loads((OUT/'program.json').read_text())
    goal_graph = hood(270.)
    a, target, hard, tol, weight = goal_graph.matrices()
    expected_first = {14:5,15:1,16:6,17:6,18:5,19:9,20:10,21:8,22:11,23:11,24:14,25:1,26:4,27:9,28:7,29:9,30:2,31:0}
    candidate_checks, timeline = [], []
    references = {}
    reason_counts = {name: 0 for name in REASONS}
    for c, result in enumerate(results):
        s, trace = result['state'], result['trace']
        assert s[0] in (1, 2) and s[2] >= 0 and int(s[14]) & 6 == 6
        assert s[3] == program['requests'][c]['goal'] == 270., 'Immutable goal modified'
        if c in expected_first:
            assert trace[0, 3] == expected_first[c], (c, trace[0])
        if c in (26, 27):
            assert s[0] == 2
        if c == 31:
            assert s[10] == 2, s
        theta = float(s[4])
        if theta not in references:
            references[theta] = independent_kkt(hood(theta))
        error = float(np.max(np.abs(result['x']-references[theta])))
        assert error < 5e-7, error
        residual = a@result['x']-target
        hard_ratio = float(np.max(np.abs(residual[hard])/tol[hard]))
        objective = float(np.sum(weight[~hard]*(residual[~hard]/tol[~hard])**2))
        assert hard_ratio < 1 and abs(objective-s[5]) < 1e-7
        for row in trace:
            reason = int(row[3])
            reason_counts[REASONS[reason]] += 1
            assert row[13] <= row[11]+1e-7, 'Committed objective increased'
            if reason == 0:
                assert row[5] == row[4]+1 and row[18] == 1
            else:
                assert row[5] == row[4] and row[10] == row[8] and row[13] == row[11], 'Rejected/noop transaction changed state'
            expected_budget = row[6] if row[2] == OP['STOP'] or reason in (3,4,15) else row[6]-1
            assert row[7] == expected_budget, 'Budget contract violated'
            if reason == 1:
                assert row[17] == 1 and row[18] == 0
            event = {name: float(value) for name, value in zip(TRACE_NAMES, row)}
            event.update(candidate=c, reason_name=REASONS[reason])
            timeline.append(event)
        # Independently recompute every evaluated trial score from its source.
        for round_id, row in enumerate(trace):
            if not row[18]:
                continue
            cmd = program['rounds'][round_id][c]
            before = independent_kkt(hood(float(row[8])))
            if row[2] == 1:
                trial = independent_kkt(hood(float(row[9])))
            else:
                trial = before.copy()
                trial[cmd['index']] += cmd['value']
            rtrial = a@trial-target
            trial_objective = float(np.sum(weight[~hard]*(rtrial[~hard]/tol[~hard])**2))
            assert abs(trial_objective-row[12]) <= max(1e-6, abs(trial_objective)*1e-10)
        candidate_checks.append({'candidate': c, 'pass': True, 'cpu_reference_max_mm': error,
                                 'hard_tolerance_ratio': hard_ratio, 'objective': objective,
                                 'parameter': theta, 'commits': int(s[10]), 'rejections': int(s[11]),
                                 'cache_hits': int(s[12]), 'evaluations_including_baseline': int(s[13]),
                                 'state': {name: float(value) for name, value in zip(STATE_NAMES, s)}})
    assert selection[1] == min(r['state'][5] for r in results)
    selected = int(selection[0])
    assert abs(results[selected]['state'][4]-270.) < 1e-8
    assert all(reason_counts[name] > 0 for name in ['ACCEPTED','CACHE_HIT','STOPPED','INACTIVE','BUDGET_EXHAUSTED','STALE_REVISION','NEED_RECOMPILE','STALE_GOAL','INVALID_OPCODE','INVALID_PARAMETER','NONFINITE_PARAMETER','HARD_CONSTRAINT_FAILED','NO_IMPROVEMENT'])
    # Exact bit equality proves replay for the same binary on this device/run;
    # this is not a cross-driver or cross-device reproducibility claim.
    verifier = validate_verifier_fixtures()
    host_rejections = validate_host_rejections()
    cad = export_selected(results[selected])
    report = {'pass': True, 'scope': 'Fixed four-patch affine harness, deterministic device controller; CPU template compilation and CAD export',
              'gpu': gpu, 'preparation': prep, 'candidate_count': len(results), 'round_count': gpu['rounds'],
              'selected_candidate': selected, 'reason_counts': reason_counts, 'candidates': candidate_checks,
              'device_replay_bit_exact': True, 'immutable_goal_verified': True,
              'verifier_fault_injections': verifier, 'host_input_rejections':host_rejections,
              'rollback_and_monotonic_commit_verified': True, 'cad': cad}
    dump(OUT/'trace.json', timeline)
    dump(OUT/'validation.json', report)
    print(json.dumps({k: report[k] for k in ['pass','candidate_count','selected_candidate','reason_counts','device_replay_bit_exact']}, indent=2))


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) == 2 else ''
    if action == 'prepare':
        prepare()
    elif action == 'validate':
        validate()
    else:
        raise SystemExit('Use harness_demo.py prepare|validate')
