"""Train a small supervised geometry policy. No pretrained language model.

Synthetic teacher expresses a bounded vehicle recipe; test data use a different
seed. Training may use the CPU; inference and command consumption use CUDA.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(718)
NI, NH, NO = 5, 24, 25


def examples(count, generator):
    x = generator.uniform(-1, 1, (count, NI))
    x[:, 4] = generator.choice([-1., 1.], count)
    codes = np.tile([1, 2, 3, 4], (count, 1))
    codes[x[:, 4] < 0, 2] = 0
    y = np.full((count, NO), -2.)
    for slot in range(4):
        y[np.arange(count), slot * 5 + codes[:, slot]] = 2.
    y[:, 20:] = np.column_stack((1+.2*x[:, 0], 1+.2*x[:, 1],
                                1+.2*x[:, 2], .5+.5*x[:, 3], 1+.1*x[:, 0]))
    return x, y, codes


x, target, _ = examples(8192, rng)
params = [rng.normal(0, .12, (NI, NH)), np.zeros(NH),
          rng.normal(0, .12, (NH, NO)), np.zeros(NO)]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
for step in range(1, 4001):
    ids = rng.integers(0, len(x), 128)
    a = x[ids]
    hidden = np.tanh(a @ params[0] + params[1])
    predicted = hidden @ params[2] + params[3]
    diff = (predicted - target[ids]) / len(ids)
    # Give physical parameters more weight than the easy constant opcode slots.
    diff[:, 20:] *= 8
    dh = (diff @ params[2].T) * (1-hidden*hidden)
    grads = [a.T @ dh, dh.sum(0), hidden.T @ diff, diff.sum(0)]
    for k, g in enumerate(grads):
        m[k] = .9*m[k]+.1*g
        v[k] = .999*v[k]+.001*g*g
        params[k] -= .004 * (m[k]/(1-.9**step)) / (np.sqrt(v[k]/(1-.999**step))+1e-8)

xt, yt, codes = examples(4096, np.random.default_rng(991))
out = np.tanh(xt @ params[0] + params[1]) @ params[2] + params[3]
accuracy = float(np.mean(out[:, :20].reshape(-1, 4, 5).argmax(2) == codes))
error = float(np.max(np.abs(out[:, 20:]-yt[:, 20:])))
assert accuracy == 1., accuracy
assert error < .035, error
dest = ROOT / 'models'
dest.mkdir(exist_ok=True)
flat = np.concatenate([p.ravel() for p in params]).astype('<f8')
(dest/'geometry_policy.f64').write_bytes(flat.tobytes())
report = {'architecture': [NI, NH, NO], 'activation': 'tanh',
          'parameters': len(flat), 'weights_bytes': flat.nbytes,
          'training': 'CPU NumPy Adam, synthetic bounded vehicle recipe',
          'training_examples': len(x), 'training_steps': step,
          'independent_test_examples': len(xt), 'seed_train': 718, 'seed_test': 991,
          'opcode_slot_accuracy': accuracy, 'max_parameter_error': error,
          'opcode_names': ['NOOP', 'NURBS_PATCH', 'MIRROR_Y', 'REVOLVE_Y', 'TESSELLATE'],
          'scope': 'Learned recipe policy, not an LLM or general CAD planner'}
(dest/'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
