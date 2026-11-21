"""CPU-only design diagnostics on the existing v6 harness evidence."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/planning'
SEED = 82453


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def basis(t):
    return np.array([(1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3])


def main():
    path = ROOT/'output/harness/template.npz'
    with np.load(path, allow_pickle=False) as archive:
        z, b = archive['Z'], archive['B']
        d = archive['dmap'][:,0]+.6*archive['dmap'][:,1]
    assert z.shape == (192, 6), 'This diagnostic targets the v6 four-patch template'
    probes, rows = [], []
    for u in np.arange(1,8)/8:
        for v in np.arange(1,8)/8:
            pu, pv = min(int(2*u),1), min(int(2*v),1)
            row = np.zeros(192)
            weights = np.outer(basis(2*u-pu), basis(2*v-pv))
            for i in range(4):
                for j in range(4):
                    row[48*(2*pu+pv)+12*i+3*j+2] = weights[i,j]
            rows.append(row)
            probes.append([float(u),float(v)])
    e = np.array(rows)
    residual = e@z
    chosen = []
    # Choose once for the canonical template. This is not a per-task chart rule.
    for _ in range(6):
        index = int(np.argmax(np.sum(residual*residual,axis=1)))
        chosen.append(index)
        direction = residual[index]/np.linalg.norm(residual[index])
        residual -= np.outer(residual@direction,direction)
    e = e[chosen]
    f = e@z
    rng = np.random.default_rng(SEED)
    delta = rng.normal(size=6)
    step = z@np.linalg.pinv(f)@delta
    latent = rng.normal(size=6)
    errors, coordinate_changes = [], []
    for _ in range(64):
        q, _ = np.linalg.qr(rng.normal(size=(6,6)))
        errors.append(float(np.max(np.abs((z@q)@np.linalg.pinv(e@z@q)@delta-step))))
        coordinate_changes.append(float(np.linalg.norm(latent-q.T@latent)))
    solution = np.linalg.lstsq(b,d,rcond=None)[0]
    floor = float(np.linalg.norm(b@solution-d)**2)
    validation = json.loads((ROOT/'output/harness/validation.json').read_text())
    score = validation['candidates'][validation['selected_candidate']]['state']['objective']
    audit = {
        'scope':'design diagnostic; no runtime or trained model changed',
        'source_npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed':SEED, 'numpy_version':np.__version__,
        'variables':192, 'nullspace_dof':6,
        'semantic_probe_uv':[probes[i] for i in chosen],
        'probe_rank':int(np.linalg.matrix_rank(f)),
        'probe_condition_number':float(np.linalg.cond(f)),
        'gauge_trials':64,
        'gauge_invariant_step_max_error_mm':max(errors),
        'raw_latent_coordinate_change_l2_max':max(coordinate_changes),
        'old_quadratic_objective_floor':floor,
        'selected_gpu_objective':score, 'gpu_minus_floor':score-floor,
        'interpretation':'Raw y changes under equivalent ZQ. Physical probe coordinates and projected steps are invariant up to roundoff. The old quadratic benchmark is already at its optimum.'
    }
    assert audit['probe_rank'] == 6 and max(errors) < 1e-10
    assert abs(score-floor) < 1e-7
    program = json.loads((ROOT/'output/harness/program.json').read_text())
    assumptions = {
        'scope':'planning diagnostics, not a new performance claim',
        'current_harness_cases':len(program['requests']),
        'unique_user_goal_values':sorted({r['goal'] for r in program['requests']}),
        'current_template_families':1,
        'candidates_with_parameter_equal_to_goal':sum(abs(c['state']['parameter']-c['state']['goal']) < 1e-8 for c in validation['candidates']),
        'candidate_validation_pass_is_design_goal_success':False,
        'zero_failures_iid_example':{
            'n':64, 'confidence':.95,
            'one_sided_upper_failure_probability':1-.05**(1/64),
            'assumptions':'Independent Bernoulli observations only; correlated stratified STEP cases do not automatically satisfy this.'
        },
        'amortization_example':{
            'hypothetical_generation_gpu_seconds':600,
            'hypothetical_gpu_seconds_saved_per_task':.0002,
            'tasks_to_break_even':math.ceil(600/.0002), 'is_measured':False
        },
        'fixed_benchmark_progress_and_new_task_frontier_must_be_reported_separately':True
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/'coordinate-audit.json',audit)
    dump(OUT/'improvement-assumptions.json',assumptions)
    print(json.dumps({'pass':True,'condition_number':audit['probe_condition_number'],
                      'gauge_error_mm':max(errors),'gpu_minus_quadratic_floor':score-floor,
                      'goal_count':len(assumptions['unique_user_goal_values'])},indent=2))


if __name__ == '__main__':
    main()
