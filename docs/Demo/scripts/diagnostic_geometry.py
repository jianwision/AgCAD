"""Independent de Casteljau geometry reference; no constraint-row compiler imports."""
import math
import numpy as np


def casteljau_curve(poles, parameters):
    t = np.asarray(parameters, dtype=float).reshape(-1, 1, 1)
    p = np.broadcast_to(np.asarray(poles, dtype=float), (len(t), *np.shape(poles))).copy()
    for width in range(p.shape[1]-1, 0, -1):
        p = (1-t)*p[:, :width]+t*p[:, 1:width+1]
    return p[:, 0]


def surface_grid(poles, us, vs, du=0, dv=0):
    """Tensor grid, arbitrary polynomial degrees and derivatives up to their degree."""
    p = np.asarray(poles, dtype=float)
    for _ in range(du):
        p = (len(p)-1)*np.diff(p, axis=0)
    for _ in range(dv):
        p = (p.shape[1]-1)*np.diff(p, axis=1)
    # Each fixed-v row is a curve in u, then each resulting u section in v.
    columns = [casteljau_curve(p[:, j], us) for j in range(p.shape[1])]
    sections = np.stack(columns, axis=1)
    return np.stack([casteljau_curve(section, vs) for section in sections])


def jacobian_grid(poles, us, vs):
    a = surface_grid(poles, us, vs, du=1)
    b = surface_grid(poles, us, vs, dv=1)
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def sample_min(poles, samples=65):
    ts = np.linspace(0., 1., samples)
    values = jacobian_grid(poles, ts, ts)
    i, j = np.unravel_index(np.argmin(values), values.shape)
    return float(values[i, j]), [float(ts[i]), float(ts[j])]


def plane_patch():
    return np.array([[[i/3, j/3, 0.] for j in range(4)] for i in range(4)])


def quadratic_jacobian_patch(center, epsilon, folded=True):
    """dx/du proportional to (u-center)^2 +/- epsilon, dy/dv=1."""
    shift = -epsilon if folded else epsilon
    derivative = np.array([center*center+shift, center*center-center+shift,
                           (1-center)**2+shift])
    x = np.r_[0., np.cumsum(derivative)/3]
    x /= x[-1]
    p = plane_patch()
    p[:, :, 0] = x[:, None]
    return p


def hood_semantics(coordinates):
    """Reconstruct original millimetre hood requirements, independently of A/b rows.

    Fixed XY/perimeter, compatible equal-span C2 seams and mirrored V halves.
    This is deliberately scoped to the original four-patch hood contract.
    """
    nets = np.asarray(coordinates).reshape(2, 2, 4, 4, 3)
    xy, perimeter, symmetry = 0., 0., 0.
    for pu in range(2):
        for pv in range(2):
            p = nets[pu, pv]
            xx = [pu*pu, pu*pu+2*pu/3, pu*pu+4*pu/3+1/3, (pu+1)**2]
            yy = [pv*pv, pv*pv+2*pv/3, pv*pv+4*pv/3+1/3, (pv+1)**2]
            for i in range(4):
                for j in range(4):
                    xy = max(xy, abs(p[i,j,0]-(-1500+1500*(pu+i/3))),
                             abs(p[i,j,1]-(-750+750*(pv+j/3))))
                    if (pu == 0 and i == 0) or (pu == 1 and i == 3) or (pv == 0 and j == 0) or (pv == 1 and j == 3):
                        z = 240-60*(xx[i]-2*(pu+i/3)+1)-40*(yy[j]-2*(pv+j/3)+1)
                        perimeter = max(perimeter, abs(p[i,j,2]-z))
        reflected = nets[pu, 1, :, ::-1].copy()
        reflected[..., 1] *= -1
        symmetry = max(symmetry, float(np.max(np.abs(nets[pu,0]-reflected))))
    seams = [0., 0., 0.]
    ts = np.linspace(0, 1, 41)
    for axis in (0, 1):
        for side in range(2):
            a, b = (nets[0,side], nets[1,side]) if axis == 0 else (nets[side,0], nets[side,1])
            for order in range(3):
                if axis == 0:
                    av, bv = surface_grid(a, [1.], ts, du=order), surface_grid(b, [0.], ts, du=order)
                else:
                    av, bv = surface_grid(a, ts, [1.], dv=order), surface_grid(b, ts, [0.], dv=order)
                seams[order] = max(seams[order], float(np.max(np.abs(av-bv))))
    values = {'fixed_xy_max_mm': xy, 'fixed_perimeter_max_mm': perimeter,
              'symmetry_max_mm': symmetry, 'seam_C0_C1_C2_max': seams}
    return {'pass': bool(max(xy, perimeter, symmetry, *seams) < 1e-7), **values}


def bernstein(t):
    """Only for assembling measurement matrices; reference evaluation uses de Casteljau."""
    return np.array([(1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3])


def measurement_row(u, v):
    pu, pv = min(int(2*u), 1), min(int(2*v), 1)
    weights = np.outer(bernstein(2*u-pu), bernstein(2*v-pv))
    row = np.zeros(192)
    row[48*(2*pu+pv)+2:48*(2*pu+pv)+48:3] = weights.ravel()
    return row


def rank_status(matrix, relative_cutoff=1e-10, uncertain_band=100.):
    a = np.asarray(matrix)
    if not np.isfinite(a).all():
        return {'status': 'INVALID', 'rank': None}
    if a.shape[1] == 0:
        return {'status': 'LOCKED', 'rank': 0, 'condition': 1.}
    s = np.linalg.svd(a, compute_uv=False)
    if not len(s) or s[0] == 0:
        return {'status': 'RANK_DEFICIENT', 'rank': 0}
    threshold = relative_cutoff*s[0]
    rank = int(np.sum(s > threshold))
    if np.any((s >= threshold/uncertain_band) & (s <= threshold*uncertain_band)):
        status = 'UNKNOWN'
    else:
        status = 'FULL_RANK' if rank == a.shape[1] else 'RANK_DEFICIENT'
    return {'status': status, 'rank': rank, 'condition': float(s[0]/s[-1]) if s[-1] > 0 else None,
            'singular_values': s.tolist(), 'relative_cutoff': relative_cutoff}
