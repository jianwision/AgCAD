"""Prepare/validate the batched CUDA constraint demo. No external downloads."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from constraint_graph import ConstraintGraph, ConstraintError, bernstein, write_batch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'/'constraints'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def hood(height):
    g = ConstraintGraph()
    patches = {}
    # Initial net is the exact quadratic crown in Bezier form, with perturbed
    # interior Z coordinates to exercise hard-constraint elimination.
    for pu in range(2):
        for pv in range(2):
            patch = []
            for i in range(4):
                row = []
                for j in range(4):
                    x, y = -1500+1500*(pu+i/3), -750+750*(pv+j/3)
                    # Quadratic Bernstein coefficients (degree elevated to 3).
                    xx = [pu*pu, pu*pu+2*pu/3, pu*pu+4*pu/3+1/3, (pu+1)**2][i]
                    yy = [pv*pv, pv*pv+2*pv/3, pv*pv+4*pv/3+1/3, (pv+1)**2][j]
                    z = 240-60*(xx-2*(pu+i/3)+1)-40*(yy-2*(pv+j/3)+1)
                    p = g.point(f'p{pu}{pv}/{i}/{j}', (x, y, z))
                    g.on_plane(p.name+'/x_plane', p, (1, 0, 0), x)
                    g.lock(p.name+'/y_lock', p, axes=(1,))
                    # Fixed perimeter elevation; inner poles remain editable.
                    if (pu == 0 and i == 0) or (pu == 1 and i == 3) or (pv == 0 and j == 0) or (pv == 1 and j == 3):
                        g.lock(p.name+'/perimeter', p, axes=(2,))
                    g.lock(p.name+'/preserve', p, hard=False, tolerance=1., weight=.002)
                    row.append(p)
                patch.append(row)
            patches[pu, pv] = patch
            g.fair_patch(f'fair/{pu}/{pv}', patch, weight=.08)
            for i, u in enumerate(np.linspace(.08, .92, 5)):
                for j, v in enumerate(np.linspace(.07, .93, 5)):
                    xn, yn = -1+pu+u, -1+pv+v
                    z = 240-60*xn*xn-40*yn*yn+(height-240)*(1-xn*xn)*(1-yn*yn)
                    g.patch_target(f'section/{pu}/{pv}/{i}/{j}', patch, u, v,
                                   (0, 0, z), axes=(2,), hard=False, tolerance=.5, weight=1.)
    for pv in range(2):
        g.seam(f'U/{pv}', patches[0, pv], patches[1, pv])
    for pu in range(2):
        g.seam(f'V/{pu}', patches[pu, 0], patches[pu, 1], axis='v')
        for i in range(4):
            for j in range(4):
                g.symmetry(f'sym/{pu}/{i}/{j}', patches[pu, 0][i][j], patches[pu, 1][i][3-j])
    g.coincident('center_coincidence', patches[0, 0][3][3], patches[1, 1][0][0])
    g.link('center_height_link', patches[0, 1][3][0].indices[2], patches[1, 0][0][3].indices[2])
    # Perturb only the seed, after capturing compatible perimeter targets.
    for p in g.points.values():
        i = p.indices[2]
        g.initial[i] += 3*math.sin(.31*(i+1))
    return g


def random_graph(seed):
    rng = np.random.default_rng(seed)
    g = ConstraintGraph()
    for i in range(12):
        g.point(f'p{i}', rng.normal(size=3))
    truth = rng.normal(size=36)
    e = rng.normal(size=(8, 36))
    for i, row in enumerate(e):
        g.row(f'hard/{i}', dict(enumerate(row)), row@truth)
    for i in range(100):
        row = rng.normal(size=36)
        g.row(f'soft/{i}', dict(enumerate(row)), row@truth+rng.normal()*.01,
              hard=False, tolerance=.3, weight=1+i/100)
    return g


def attachment_graph():
    g = ConstraintGraph()
    patch = [[g.point(f'patch/{i}/{j}', (i*10., j*10., i*j*.5)) for j in range(4)] for i in range(4)]
    curve = [g.point(f'curve/{i}', (i*10., i*i, 2.*i)) for i in range(4)]
    for p in [p for row in patch for p in row]+curve:
        g.lock(p.name+'/fixed', p)
    on_curve = g.point('curve_marker', (-8, 2, 5))
    on_surface = g.point('surface_marker', (-8, 2, 5))
    on_line = g.point('line_marker', (0, 0, 0))
    free = g.point('free_marker', (0, 0, 0))
    g.on_curve('curve_attachment', on_curve, curve, .37)
    g.on_patch('surface_attachment', on_surface, patch, .29, .61)
    g.on_line('line_slide', on_line, (1, 2, 3), (1, 2, -1))
    g.lock('line_target', on_line, (20, 5, 8), hard=False, tolerance=1.)
    g.lock('free_target', free, (7, -3, 12), hard=False, tolerance=1.)
    return g


def negative_tests():
    results = []
    def reject(name, action):
        try:
            action()
        except ConstraintError as error:
            results.append({'name': name, 'pass': True, 'message': str(error),
                            'diagnostics': error.diagnostics})
            return
        raise AssertionError(f'Invalid input accepted: {name}')
    g = ConstraintGraph()
    p = g.point('p', (0, 0, 0))
    g.lock('fixed', p)
    g.row('contradiction', {0: 1.}, 10.)
    reject('hard_constraint_conflict', g.compile)
    assert 'contradiction' in results[-1]['diagnostics']['suspected_conflicts']
    g = ConstraintGraph()
    p = g.point('p', (0, 0, 0))
    g.row('only_x', {0: 1.}, hard=False)
    reject('underconstrained', g.compile)
    reject('negative_weight', lambda: g.row('bad_weight', {0: 1}, hard=False, weight=-1))
    reject('nonfinite_target', lambda: g.row('bad_target', {0: 1}, float('nan')))
    reject('invalid_reference', lambda: g.row('bad_ref', {99: 1}))
    reject('zero_plane', lambda: g.on_plane('plane', p, (0, 0, 0), 0))
    other = ConstraintGraph().point('p', (0, 0, 0))
    reject('foreign_point', lambda: g.lock('foreign', other))
    reject('bad_parameter', lambda: bernstein(1.1))
    reject('unsupported_protocol', lambda: ConstraintGraph.from_dict({'version': 9}))
    g.row('duplicate', {0: 1.})
    reject('duplicate_constraint_id', lambda: g.row('duplicate', {1: 1.}))
    return results


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    timer = time.perf_counter()
    graphs = [hood(float(h)) for h in np.linspace(220, 290, 8)]
    graphs = [ConstraintGraph.from_dict(g.to_dict()) for g in graphs]
    compiled = write_batch(OUT/'batch.bin', graphs)
    dump(OUT/'graphs.json', [g.to_dict() for g in graphs])
    random_out = OUT/'random'
    random_out.mkdir(exist_ok=True)
    randoms = [random_graph(i+17) for i in range(4)]
    write_batch(random_out/'batch.bin', randoms)
    dump(random_out/'graphs.json', [g.to_dict() for g in randoms])
    limited = OUT/'iteration_limit'
    limited.mkdir(exist_ok=True)
    write_batch(limited/'batch.bin', randoms, iterations=1)
    attached = OUT/'attachments'
    attached.mkdir(exist_ok=True)
    attachments = [attachment_graph()]
    write_batch(attached/'batch.bin', attachments)
    dump(attached/'graphs.json', [g.to_dict() for g in attachments])
    for name, fixed in [('fully_fixed', True), ('zero_rhs', False)]:
        g = ConstraintGraph()
        p = g.point('p', (0, 0, 0))
        g.lock('lock', p, (3, 4, 5) if fixed else (0, 0, 0), hard=fixed, tolerance=1e-7 if fixed else 1.)
        path = OUT/name
        path.mkdir(exist_ok=True)
        write_batch(path/'batch.bin', [g], iterations=8)
        dump(path/'graphs.json', [g.to_dict()])
    negatives = negative_tests()
    dump(OUT/'preparation.json', {'pass': True, 'negative_tests': negatives,
                                'cpu_preparation_ms': (time.perf_counter()-timer)*1000,
                                'diagnostics': compiled[0]['diagnostics']})
    print(json.dumps({'prepared': 8, 'random': 4, 'negative_tests': len(negatives),
                      'diagnostics': compiled[0]['diagnostics']}))


def read_results(path):
    report = json.loads((path/'gpu_report.json').read_text())
    n, m, k, r = [report[key] for key in ('variables', 'soft_rows', 'reduced_dof', 'all_rows')]
    stride = 7*k+2*m+8+n+r
    data = np.fromfile(path/'result.bin', dtype='<f8').reshape(report['batch'], stride)
    return report, [{'y': w[:k], 'jvp': w[7*k+m:7*k+2*m], 'vjp': w[6*k:7*k],
                     'gradient': w[5*k:6*k], 'status': w[7*k+2*m:7*k+2*m+8],
                     'x': w[7*k+2*m+8:7*k+2*m+8+n], 'residual': w[-r:]} for w in data]


def independent_kkt(g):
    # Independent reference: full-coordinate KKT with row-normalized hard E;
    # no nullspace substitution and no PCG/normal-equation recurrence reused.
    a, b, hard, tol, weight = g.matrices()
    aw = a[~hard]*np.sqrt(weight[~hard, None])/tol[~hard, None]
    bw = b[~hard]*np.sqrt(weight[~hard])/tol[~hard]
    norm = np.linalg.norm(a[hard], axis=1)
    e, f = a[hard]/norm[:, None], b[hard]/norm
    h = aw.T@aw
    system = np.block([[h, e.T], [e, np.zeros((len(e), len(e)))]])
    rhs = np.concatenate((aw.T@bw, f))
    return np.linalg.lstsq(system, rhs, rcond=1e-13)[0][:len(g.initial)]


def validate_bucket(path):
    graphs = [ConstraintGraph.from_dict(d) for d in json.loads((path/'graphs.json').read_text())]
    gpu, results = read_results(path)
    checks = []
    for i, (g, result) in enumerate(zip(graphs, results)):
        c = g.compile()
        b, d = c['B'], c['d']
        assert result['status'][0] == 1, result['status']
        reference = independent_kkt(g)
        error = float(np.max(np.abs(reference-result['x'])))
        assert error < 5e-7, error
        a, target, hard, tol, weight = g.matrices()
        residual = a@result['x']-target
        raw_error = float(np.max(np.abs(residual-result['residual'])))
        hard_ratio = float(np.max(np.abs(residual[hard])/tol[hard], initial=0.))
        assert hard_ratio < 1 and raw_error < 1e-8, (hard_ratio, raw_error)
        v = np.sin(.37*np.arange(1, b.shape[1]+1))
        w = np.cos(.23*np.arange(1, b.shape[0]+1))
        h = 1e-4
        fd = ((b@(result['y']+h*v)-d)-(b@(result['y']-h*v)-d))/(2*h)
        jvp_error = float(np.max(np.abs(result['jvp']-fd), initial=0.))
        vjp_error = float(np.max(np.abs(result['vjp']-b.T@w), initial=0.))
        adjoint_error = float(abs(w@result['jvp']-v@result['vjp']))
        assert jvp_error < 1e-7 and vjp_error < 1e-9 and adjoint_error < 1e-9
        final_objective = float(np.sum((b@result['y']-d)**2))
        initial_objective = float(d@d)
        assert final_objective <= initial_objective+1e-8
        checks.append({'candidate': i, 'pass': True, 'iterations': int(result['status'][1]),
                       'reference_max_mm': error, 'hard_tolerance_ratio': hard_ratio,
                       'gpu_residual_max_error': raw_error, 'jvp_finite_difference_max_error': jvp_error,
                       'vjp_max_error': vjp_error, 'adjoint_identity_error': adjoint_error,
                       'relative_gradient': float(result['status'][7]),
                       'initial_objective': initial_objective, 'final_objective': final_objective})
    return gpu, results, checks


def patch_nets(x):
    return np.asarray(x).reshape(2, 2, 4, 4, 3)


def patch_value(p, u, v, du=0, dv=0):
    return np.einsum('i,j,ijk->k', bernstein(u, du), bernstein(v, dv), p)


def export_surface(result):
    from acceptance import build_surface, roundtrip, count, properties
    from cad_ops import sew_surfaces
    from cad_advanced import untrim_surface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.gp import gp_Pnt
    nets = patch_nets(result['x'])
    surfaces, records = [], []
    native_error = 0.
    for pu in range(2):
        for pv in range(2):
            p = nets[pu, pv]
            record = {'kind': 'nurbs', 'size': [4, 4], 'degree': [3, 3],
                      'u_knots': [0.]*4+[1.]*4, 'v_knots': [0.]*4+[1.]*4,
                      'poles_xyzw': np.concatenate((p, np.ones((4, 4, 1))), axis=2).ravel().tolist()}
            surface = build_surface(record)
            surfaces.append(surface)
            records.append(record)
            for u in np.linspace(.013, .987, 13):
                for v in np.linspace(.017, .983, 11):
                    point = gp_Pnt()
                    surface.D0(u, v, point)
                    native_error = max(native_error, float(np.linalg.norm(np.array([point.X(), point.Y(), point.Z()])-patch_value(p, u, v))))
    seam_error = [0., 0., 0.]
    normal_error = 0.
    for direction in (0, 1):
        for side in range(2):
            a, b = (nets[0, side], nets[1, side]) if direction == 0 else (nets[side, 0], nets[side, 1])
            for t in np.linspace(0, 1, 101):
                ua, va, ub, vb = (1, t, 0, t) if direction == 0 else (t, 1, t, 0)
                for d in range(3):
                    du, dv = (d, 0) if direction == 0 else (0, d)
                    seam_error[d] = max(seam_error[d], float(np.linalg.norm(patch_value(a, ua, va, du, dv)-patch_value(b, ub, vb, du, dv))))
                normals = []
                for net, u, v in [(a, ua, va), (b, ub, vb)]:
                    normal = np.cross(patch_value(net, u, v, 1, 0), patch_value(net, u, v, 0, 1))
                    assert np.linalg.norm(normal) > 1.
                    normals.append(normal/np.linalg.norm(normal))
                normal_error = max(normal_error, float(np.linalg.norm(normals[0]-normals[1])))
    assert max(seam_error) < 1e-7 and native_error < 1e-8
    shell = sew_surfaces([untrim_surface(s) for s in surfaces], tolerance=1e-6)
    assert count(shell, TopAbs_FACE) == 4
    # acceptance.roundtrip writes under primitives; move only this new artifact.
    step = roundtrip('constraint_hood.step', shell, 0)
    source = ROOT/'output'/'primitives'/'constraint_hood.step'
    source.replace(OUT/'constraint_hood.step')
    step['file'] = str(OUT/'constraint_hood.step')
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from placement_ops import snap_point
    reader = STEPControl_Reader()
    assert reader.ReadFile(step['file']) == IFSelect_RetDone and reader.TransferRoots() > 0
    reopened = reader.OneShape()
    step_error = 0.
    for p in nets.reshape(4, 4, 4, 3):
        for u in np.linspace(.023, .977, 7):
            for v in np.linspace(.031, .969, 7):
                _, distance = snap_point(patch_value(p, u, v), target=reopened, max_distance=1e-7)
                step_error = max(step_error, distance['distance_mm'])
    dump(OUT/'surfaces.json', {'surfaces': records})
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    BRepMesh_IncrementalMesh(shell, 1., False, .1, False).Perform()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    assert writer.Write(shell, str(OUT/'constraint_hood.stl'))
    return {'pass': True, 'native_faces': 4, 'area_mm2': properties(shell)[0],
            'seam_C0_C1_C2_max': seam_error, 'seam_unit_normal_max': normal_error,
            'native_evaluation_max_mm': native_error, 'step_probe_count': 196,
            'step_probe_max_mm': step_error, 'step_roundtrip': step}


def validate():
    gpu, results, checks = validate_bucket(OUT)
    random_gpu, _, random_checks = validate_bucket(OUT/'random')
    attachment_gpu, attached, attachment_checks = validate_bucket(OUT/'attachments')
    # Independent analytic point positions, including the free parameter along a line.
    x = attached[0]['x'].reshape(-1, 3)
    curve_expected = bernstein(.37)@np.array([(i*10., i*i, 2.*i) for i in range(4)])
    surface_expected = np.array([30*.29, 30*.61, 4.5*.29*.61])
    origin, direction, target = np.array([1, 2, 3]), np.array([1, 2, -1]), np.array([20, 5, 8])
    line_expected = origin+direction*((target-origin)@direction)/(direction@direction)
    analytic_error = float(np.max(np.abs(x[-4:]-np.array([curve_expected, surface_expected, line_expected, [7, -3, 12]]))))
    assert analytic_error < 1e-9
    fixed_checks = {}
    for name in ('fully_fixed', 'zero_rhs'):
        _, fixed_result, fixed_check = validate_bucket(OUT/name)
        assert fixed_result[0]['status'][1] == 0
        fixed_checks[name] = fixed_check
    _, limited = read_results(OUT/'iteration_limit')
    assert all(r['status'][0] == 3 for r in limited), 'Iteration limit was falsely accepted'
    x = np.array([r['x'] for r in results])
    linearity = float(np.max(np.abs(x-(x[0]+np.arange(8)[:, None]/7*(x[-1]-x[0])))))
    assert linearity < 1e-7
    export = export_surface(results[5])
    negatives = json.loads((OUT/'preparation.json').read_text())['negative_tests']
    report = {'pass': True, 'scope': 'Affine fixed-topology constraints; CPU SVD compiler, batched CUDA PCG, CPU STEP topology',
              'gpu': gpu, 'candidates': checks, 'random_gpu': random_gpu, 'random_cases': random_checks,
              'attachment_gpu': attachment_gpu, 'attachment_cases': attachment_checks,
              'attachment_analytic_max_mm': analytic_error,
              'zero_iteration_cases': fixed_checks,
              'negative_tests': negatives, 'iteration_limit_rejected': len(limited),
              'candidate_linearity_max_mm': linearity, 'cad': export}
    dump(OUT/'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) == 2 else ''
    if action == 'prepare':
        prepare()
    elif action == 'validate':
        validate()
    else:
        raise SystemExit('Use constraint_demo.py prepare|validate')
