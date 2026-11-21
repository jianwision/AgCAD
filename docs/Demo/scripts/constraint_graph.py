"""Typed, fixed-topology affine CAD constraint compiler (CPU preprocessing).

Hard equalities are eliminated with an SVD, never approximated with large weights.
The compiled weighted least-squares operator is solved on CUDA. All coordinates
are millimetres; parameters of the supported polynomial Bezier patches are
dimensionless. General nonlinear G1/G2 and inequalities are deliberately absent.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np


class ConstraintError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class Point:
    name: str
    indices: tuple


@dataclass(frozen=True)
class Row:
    name: str
    terms: dict
    target: float
    hard: bool
    tolerance: float
    weight: float


def bernstein(t, order=0):
    if not math.isfinite(t) or not 0 <= t <= 1 or order not in (0, 1, 2):
        raise ConstraintError('Invalid bicubic parameter/derivative order')
    if order == 0:
        return np.array([(1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3])
    if order == 1:
        return np.array([-3*(1-t)**2, 3*(1-t)**2-6*t*(1-t), 6*t*(1-t)-3*t*t, 3*t*t])
    return np.array([6*(1-t), -12+18*t, 6-18*t, 6*t])


class ConstraintGraph:
    def __init__(self):
        self.points = {}
        self.initial = []
        self.rows = []
        self._names = set()

    def point(self, name, xyz):
        v = np.asarray(xyz, dtype=float)
        if not name or name in self.points or v.shape != (3,) or not np.isfinite(v).all():
            raise ConstraintError('Invalid or duplicate point')
        n = len(self.initial)
        p = Point(name, (n, n+1, n+2))
        self.initial.extend(v.tolist())
        self.points[name] = p
        return p

    def row(self, name, terms, target=0., *, hard=True, tolerance=1e-7, weight=1.):
        if (not name or name in self._names or not isinstance(hard, bool)
                or not all(math.isfinite(v) for v in (target, tolerance, weight))
                or tolerance <= 0 or weight <= 0):
            raise ConstraintError('Invalid constraint id, value, tolerance or weight')
        clean = {}
        for i, v in terms.items():
            if (not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                    or not 0 <= i < len(self.initial) or not math.isfinite(v)):
                raise ConstraintError('Invalid variable reference/coefficient')
            if v != 0:
                clean[int(i)] = float(v)
        if not clean:
            raise ConstraintError('Empty constraint row')
        self._names.add(name)
        self.rows.append(Row(name, clean, float(target), hard, float(tolerance), float(weight)))

    def _point(self, p):
        if not isinstance(p, Point) or self.points.get(p.name) is not p:
            raise ConstraintError('Point belongs to another graph')
        return p.indices

    def lock(self, name, point, target=None, axes=(0, 1, 2), **kwargs):
        ids = self._point(point)
        target = np.asarray(self.initial)[list(ids)] if target is None else np.asarray(target)
        if target.shape != (3,) or not set(axes).issubset({0, 1, 2}):
            raise ConstraintError('Invalid lock target/axes')
        for k in axes:
            self.row(f'{name}/{k}', {ids[k]: 1.}, target[k], **kwargs)

    def link(self, name, a, b, scale=1., offset=0., **kwargs):
        terms = {a: 1.}
        terms[b] = terms.get(b, 0.)-scale
        self.row(name, terms, offset, **kwargs)

    def coincident(self, name, a, b, **kwargs):
        for k, (ia, ib) in enumerate(zip(self._point(a), self._point(b))):
            self.link(f'{name}/{k}', ia, ib, **kwargs)

    def on_plane(self, name, point, normal, offset, **kwargs):
        ids = self._point(point)
        normal = np.asarray(normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-15:
            raise ConstraintError('Invalid plane normal')
        length = np.linalg.norm(normal)
        self.row(name, dict(zip(ids, normal/length)), offset/length, **kwargs)

    def symmetry(self, name, a, b, normal=(0, 1, 0), offset=0., **kwargs):
        ia, ib = self._point(a), self._point(b)
        normal = np.asarray(normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-15:
            raise ConstraintError('Invalid symmetry plane')
        length = np.linalg.norm(normal)
        n = normal/length
        reflection = np.eye(3)-2*np.outer(n, n)
        for k in range(3):
            terms = {ib[k]: 1.}
            for j in range(3):
                terms[ia[j]] = terms.get(ia[j], 0.)-reflection[k, j]
            self.row(f'{name}/{k}', terms, 2*offset/length*n[k], **kwargs)

    def on_line(self, name, point, origin, direction, **kwargs):
        ids = self._point(point)
        d, origin = np.asarray(direction, dtype=float), np.asarray(origin, dtype=float)
        if (d.shape != (3,) or origin.shape != (3,) or not np.isfinite(d).all()
                or not np.isfinite(origin).all() or np.linalg.norm(d) < 1e-15):
            raise ConstraintError('Invalid fixed line')
        d = d/np.linalg.norm(d)
        seed = np.eye(3)[np.argmin(np.abs(d))]
        a = np.cross(d, seed)
        a /= np.linalg.norm(a)
        for i, normal in enumerate((a, np.cross(d, a))):
            self.row(f'{name}/{i}', dict(zip(ids, normal)), normal@origin, **kwargs)

    def on_curve(self, name, point, poles, t, **kwargs):
        # Parameter is fixed. Solving for t would make this nonlinear.
        if len(poles) != 4:
            raise ConstraintError('Expected four cubic Bezier poles')
        basis = bernstein(t)
        for axis, idx in enumerate(self._point(point)):
            terms = {idx: 1.}
            for p, coefficient in zip(poles, basis):
                j = self._point(p)[axis]
                terms[j] = terms.get(j, 0.)-coefficient
            self.row(f'{name}/{axis}', terms, **kwargs)

    def on_patch(self, name, point, patch, u, v, **kwargs):
        # Fixed UV attachment; no nearest-point search or geodesic claim.
        for axis, idx in enumerate(self._point(point)):
            terms = {j: -value for j, value in self.patch_terms(patch, u, v, axis).items()}
            terms[idx] = terms.get(idx, 0.)+1.
            self.row(f'{name}/{axis}', terms, **kwargs)

    def patch_terms(self, patch, u, v, axis, du=0, dv=0):
        if len(patch) != 4 or any(len(row) != 4 for row in patch) or axis not in (0, 1, 2):
            raise ConstraintError('Expected a 4x4 polynomial Bezier patch')
        bu, bv = bernstein(u, du), bernstein(v, dv)
        terms = {}
        for i in range(4):
            for j in range(4):
                k = self._point(patch[i][j])[axis]
                terms[k] = terms.get(k, 0.)+bu[i]*bv[j]
        return terms

    def patch_target(self, name, patch, u, v, target, axes=(0, 1, 2), **kwargs):
        if np.asarray(target).shape != (3,):
            raise ConstraintError('Expected XYZ target')
        for k in axes:
            self.row(f'{name}/{k}', self.patch_terms(patch, u, v, k), target[k], **kwargs)

    def seam(self, name, a, b, axis='u', order=2, **kwargs):
        # Both patches use [0,1]^2, same orientation and equal parameter spans.
        if axis not in ('u', 'v') or order not in (0, 1, 2):
            raise ConstraintError('Invalid compatible seam')
        for d in range(order+1):
            for t in (0., 1/3, 2/3, 1.):
                for k in range(3):
                    if axis == 'u':
                        ta = self.patch_terms(a, 1, t, k, du=d)
                        tb = self.patch_terms(b, 0, t, k, du=d)
                    else:
                        ta = self.patch_terms(a, t, 1, k, dv=d)
                        tb = self.patch_terms(b, t, 0, k, dv=d)
                    terms = dict(ta)
                    for i, v in tb.items():
                        terms[i] = terms.get(i, 0.)-v
                    self.row(f'{name}/d{d}/t{t:.6f}/{k}', terms, **kwargs)

    def fair_patch(self, name, patch, weight=.2, tolerance=1.):
        # Control-net second differences, not a physical curvature integral.
        for direction in (0, 1):
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        pts = [patch[i+s][j] if direction == 0 else patch[j][i+s] for s in range(3)]
                        terms = {}
                        for p, c in zip(pts, (1., -2., 1.)):
                            idx = self._point(p)[k]
                            terms[idx] = terms.get(idx, 0.)+c
                        self.row(f'{name}/{direction}/{i}/{j}/{k}', terms,
                                 hard=False, weight=weight, tolerance=tolerance)

    def matrices(self):
        n = len(self.initial)
        if not 1 <= n <= 1024 or not self.rows or len(self.rows) > 8192:
            raise ConstraintError('Graph capacity exceeded (1024 variables, 8192 rows)')
        a = np.zeros((len(self.rows), n))
        for i, row in enumerate(self.rows):
            for j, value in row.terms.items():
                a[i, j] = value
        b = np.array([r.target for r in self.rows])
        hard = np.array([r.hard for r in self.rows])
        tol = np.array([r.tolerance for r in self.rows])
        weight = np.array([r.weight for r in self.rows])
        return a, b, hard, tol, weight

    def compile(self):
        a, b, hard, tol, weight = self.matrices()
        n = a.shape[1]
        e, f = a[hard]/tol[hard, None], b[hard]/tol[hard]
        initial = np.array(self.initial)
        if len(e):
            u, singular, vt = np.linalg.svd(e, full_matrices=(len(e) < n))
            cutoff = max(e.shape)*np.finfo(float).eps*singular[0]
            rank = int(np.sum(singular > cutoff))
            correction = vt[:rank].T@((u[:, :rank].T@(f-e@initial))/singular[:rank])
            x0 = initial+correction
            z = vt[rank:].T
        else:
            rank, x0, z = 0, initial.copy(), np.eye(n)
        violation = np.abs(e@x0-f)
        diagnostics = {'variables': n, 'hard_rows': len(e), 'hard_rank': rank,
                       'redundant_hard_rows': len(e)-rank, 'remaining_dof': n-rank,
                       'max_hard_tolerance_ratio': float(violation.max(initial=0.))}
        if violation.max(initial=0.) > 1.:
            worst = np.argsort(violation)[-8:][::-1]
            ids = [r.name for r in self.rows if r.hard]
            diagnostics['suspected_conflicts'] = [ids[i] for i in worst if violation[i] > 1.]
            raise ConstraintError('Inconsistent hard constraints', diagnostics)
        if z.shape[1] > 512:
            raise ConstraintError('Solver supports at most 512 remaining DOF', diagnostics)
        scale = np.sqrt(weight[~hard])/tol[~hard]
        aw, bw = a[~hard]*scale[:, None], b[~hard]*scale
        operator = aw@z
        rhs = bw-aw@x0
        soft_rank = int(np.linalg.matrix_rank(operator)) if z.shape[1] else 0
        diagnostics['soft_rank'] = soft_rank
        diagnostics['unanchored_dof'] = z.shape[1]-soft_rank
        if soft_rank != z.shape[1]:
            raise ConstraintError('Unanchored least-squares problem', diagnostics)
        if not all(np.isfinite(x).all() for x in (operator, rhs, z, x0)):
            raise ConstraintError('Nonfinite compiled system')
        return {'B': operator, 'd': rhs, 'Z': z, 'x0': x0, 'A': a, 'b': b,
                'diagonal': np.sum(operator*operator, axis=0), 'diagnostics': diagnostics}

    def to_dict(self):
        return {'version': 1, 'units': 'mm', 'points': [
            {'name': p.name, 'initial': [self.initial[i] for i in p.indices]}
            for p in self.points.values()], 'rows': [
                {'name': r.name, 'terms': {str(k): v for k, v in r.terms.items()},
                 'target': r.target, 'hard': r.hard, 'tolerance': r.tolerance, 'weight': r.weight}
                for r in self.rows]}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1 or data.get('units') != 'mm':
            raise ConstraintError('Unsupported constraint protocol')
        g = cls()
        for p in data['points']:
            g.point(p['name'], p['initial'])
        for r in data['rows']:
            g.row(r['name'], {int(k): v for k, v in r['terms'].items()}, r['target'],
                  hard=r['hard'], tolerance=r['tolerance'], weight=r['weight'])
        return g


def write_batch(path, graphs, iterations=256, relative_tolerance=1e-11):
    if not 1 <= len(graphs) <= 64 or not 1 <= iterations <= 2048 or not 0 < relative_tolerance < 1:
        raise ConstraintError('Invalid batch/iteration/tolerance')
    compiled = [g.compile() for g in graphs]
    first = compiled[0]
    n, k = first['Z'].shape
    m, r = len(first['d']), len(first['b'])
    words = 16+len(graphs)*(m*k+m+n*k+n+r*n+r+k)
    if words*8 > 256*1024*1024:
        raise ConstraintError('Packed batch exceeds the 256 MiB host input limit')
    header = [20261120., 1., len(graphs), n, m, k, r, iterations, relative_tolerance]+[0.]*7
    pieces = [np.array(header)]
    for c in compiled:
        if c['Z'].shape != (n, k) or c['B'].shape != (m, k) or c['A'].shape != (r, n):
            raise ConstraintError('Batch requires equal dimensions; bucket graphs by shape')
        pieces.extend(c[key].ravel() for key in ('B', 'd', 'Z', 'x0', 'A', 'b', 'diagonal'))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(pieces).astype('<f8').tofile(path)
    return compiled
