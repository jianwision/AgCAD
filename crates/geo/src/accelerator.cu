// All entry points use AgCAD's single device parameter-block ABI.
using U64 = unsigned long long;
struct SurfaceDesc {
  unsigned nu, nv, pu, pv;
  U64 ku, kv, cp;
  P3 origin;
};
struct Sample {
  U64 patch;
  double u, v;
};
struct SurfArgs {
  const SurfaceDesc *surfaces;
  const double *data;
  const Sample *samples;
  double *out;
  U64 count;
};

struct Grid {
  U64 patch, start, u_offset, v_offset, columns;
};
struct PointArgs {
  const SurfaceDesc *surfaces;
  const double *data;
  const Grid *grids;
  const double *stations;
  P3 *out;
  U64 grid_count, count;
};

// Evaluate only the nonzero basis functions (The NURBS Book, A2.2).
__device__ unsigned point_basis(const double *knots, unsigned count,
                                unsigned degree, double value,
                                double *basis_values) {
  unsigned span = degree;
  while (span + 1 < count && value >= knots[span + 1])
    ++span;
  double left[6], right[6];
  basis_values[0] = 1.0;
  for (unsigned j = 1; j <= degree; ++j) {
    left[j] = value - knots[span + 1 - j];
    right[j] = knots[span + j] - value;
    double saved = 0.0;
    for (unsigned r = 0; r < j; ++r) {
      double ratio = basis_values[r] / (right[r + 1] + left[j - r]);
      basis_values[r] = saved + right[r + 1] * ratio;
      saved = left[j - r] * ratio;
    }
    basis_values[j] = saved;
  }
  return span - degree;
}

extern "C" __global__ void surface_points(const PointArgs *p) {
  U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p->count)
    return;
  U64 lo = 0, hi = p->grid_count;
  while (hi - lo > 1) {
    U64 mid = lo + (hi - lo) / 2;
    if (p->grids[mid].start <= i)
      lo = mid;
    else
      hi = mid;
  }
  Grid g = p->grids[lo];
  SurfaceDesc s = p->surfaces[g.patch];
  U64 local = i - g.start;
  double bu[6], bv[6];
  unsigned u0 = point_basis(p->data + s.ku, s.nu, s.pu,
                            p->stations[g.u_offset + local % g.columns], bu);
  unsigned v0 = point_basis(p->data + s.kv, s.nv, s.pv,
                            p->stations[g.v_offset + local / g.columns], bv);
  P3 sum = {};
  double weight = 0;
  for (unsigned a = 0; a <= s.pu; ++a) {
    for (unsigned b = 0; b <= s.pv; ++b) {
      const double *pole = p->data + s.cp + 4 * ((u0 + a) * s.nv + v0 + b);
      double w = bu[a] * bv[b] * pole[3];
      sum = plus(sum, times(p3(pole[0], pole[1], pole[2]), w));
      weight += w;
    }
  }
  p->out[i] = plus(s.origin, times(sum, 1. / weight));
}

extern "C" __global__ void surface_curvature(const SurfArgs *p) {
  U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p->count)
    return;
  Sample q = p->samples[i];
  SurfaceDesc s = p->surfaces[q.patch];
  const double *ku = p->data + s.ku;
  const double *kv = p->data + s.kv;
  Jet j =
      nurbs_surface(p->data + s.cp, s.nu, s.nv, s.pu, s.pv, ku, kv, q.u, q.v);
  double u = ku[s.nu] - ku[s.pu], v = kv[s.nv] - kv[s.pv];
  P3 normal = vector_cross(j.du, j.dv);
  double len = sqrt(scalar(normal, normal));
  if (len > 0)
    normal = times(normal, 1. / len);
  double twist = fabs(scalar(j.duv, normal)) * u * v;
  p->out[2 * i] = sqrt(scalar(j.duu, j.duu)) * u * u + twist;
  p->out[2 * i + 1] = sqrt(scalar(j.dvv, j.dvv)) * v * v + twist;
}
struct Triangle {
  unsigned a, b, c;
};
struct Wall {
  const P3 *vertices;
  const Triangle *triangles;
  double *hits;
  U64 n, stride;
};
// One block owns one sampled ray. Keep the public FP64 predicate and origin
// bias.
extern "C" __global__ void wall_hits(const Wall *p) {
  unsigned lane = threadIdx.x;
  Triangle t = p->triangles[blockIdx.x * p->stride];
  P3 a = p->vertices[t.a], b = p->vertices[t.b], c = p->vertices[t.c];
  P3 n = vector_cross(minus(b, a), minus(c, a));
  double len = sqrt(scalar(n, n));
  if (len > 0)
    n = times(n, 1. / len);
  P3 dir = times(n, -1.),
     o = minus(times(plus(plus(a, b), c), 1. / 3.), times(n, 1e-7));
  double best = __longlong_as_double(0x7ff0000000000000ULL);
  for (U64 i = lane; i < p->n; i += blockDim.x) {
    t = p->triangles[i];
    a = p->vertices[t.a];
    b = p->vertices[t.b];
    c = p->vertices[t.c];
    P3 e1 = minus(b, a), e2 = minus(c, a), h = vector_cross(dir, e2);
    double det = scalar(e1, h);
    if (fabs(det) < 1e-14)
      continue;
    double inv = 1. / det;
    P3 s = minus(o, a);
    double u = scalar(s, h) * inv;
    if (u < 0 || u > 1)
      continue;
    P3 q = vector_cross(s, e1);
    double v = scalar(dir, q) * inv;
    if (v < 0 || u + v > 1)
      continue;
    double d = scalar(e2, q) * inv;
    if (d > 1e-9)
      best = fmin(best, d);
  }
  __shared__ double work[128];
  work[lane] = best;
  __syncthreads();
  for (int k = 64; k; k /= 2) {
    if (lane < k)
      work[lane] = fmin(work[lane], work[lane + k]);
    __syncthreads();
  }
  if (!lane)
    p->hits[blockIdx.x] = work[0];
}
struct MinArgs {
  const double *hits;
  double *out;
  U64 count;
};
extern "C" __global__ void wall_min(const MinArgs *p) {
  unsigned lane = threadIdx.x;
  double best = __longlong_as_double(0x7ff0000000000000ULL);
  for (U64 i = lane; i < p->count; i += blockDim.x)
    best = fmin(best, p->hits[i]);
  __shared__ double work[128];
  work[lane] = best;
  __syncthreads();
  for (int k = 64; k; k /= 2) {
    if (lane < k)
      work[lane] = fmin(work[lane], work[lane + k]);
    __syncthreads();
  }
  if (!lane)
    *p->out = work[0];
}
struct PairArgs {
  const double *bounds;
  unsigned char *keep;
  U64 rails, spans, count;
  double tol;
};
extern "C" __global__ void rail_overlap(const PairArgs *p) {
  U64 id = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= p->count)
    return;
  U64 pair = id / p->spans, span = id % p->spans;
  U64 i = (U64)((1. + sqrt(1. + 8. * (double)pair)) * .5);
  while (i * (i - 1) / 2 > pair)
    --i;
  while ((i + 1) * i / 2 <= pair)
    ++i;
  U64 j = pair - i * (i - 1) / 2;
  const double *a = p->bounds + 6 * (i * p->spans + span);
  const double *b = p->bounds + 6 * (j * p->spans + span);
  bool overlap = true;
  for (int k = 0; k < 3; k++)
    if (a[k] > b[k + 3] + p->tol || b[k] > a[k + 3] + p->tol)
      overlap = false;
  p->keep[id] = overlap;
}

// Segments use local vertex indices on upload. Translate them in parallel once.
struct RebaseArgs {
  Triangle *triangles;
  const unsigned *offsets, *bases;
  U64 segments, count;
};
extern "C" __global__ void rebase_triangles(const RebaseArgs *p) {
  U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p->count)
    return;
  unsigned lo = 0, hi = (unsigned)p->segments;
  while (hi - lo > 1) {
    unsigned mid = lo + (hi - lo) / 2;
    if (p->offsets[mid] <= i)
      lo = mid;
    else
      hi = mid;
  }
  Triangle t = p->triangles[i];
  unsigned base = p->bases[lo];
  p->triangles[i] = {t.a + base, t.b + base, t.c + base};
}

// Work on the packed scratch copy so a resident tessellation can be measured
// again.
struct LocalPositions {
  P3 *positions;
  const unsigned *offsets;
  const P3 *origins;
  U64 segments, count;
};
extern "C" __global__ void local_positions(const LocalPositions *p) {
  U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p->count)
    return;
  unsigned lo = 0, hi = (unsigned)p->segments;
  while (hi - lo > 1) {
    unsigned mid = lo + (hi - lo) / 2;
    if (p->offsets[mid] <= i)
      lo = mid;
    else
      hi = mid;
  }
  p->positions[i] = minus(p->positions[i], p->origins[lo]);
}

struct CornerArgs {
  const P3 *points;
  double *results;
  const P3 *frame;
  double size;
  U64 count;
};
extern "C" __global__ void block_edge_corners(const CornerArgs *p) {
  U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p->count) return;
  double x[3], y[3];
  for (int j = 0; j < 3; ++j) {
    P3 v = minus(p->points[3*i+j],p->frame[0]);
    P3 a = p->frame[1], b = p->frame[2];
    x[j] = v.x*a.x + v.y*a.y + v.z*a.z;
    y[j] = v.x*b.x + v.y*b.y + v.z*b.z;
  }
  double ux=x[0]-x[1], uy=y[0]-y[1], vx=x[2]-x[1], vy=y[2]-y[1];
  double lu=hypot(ux,uy), lv=hypot(vx,vy);
  ux/=lu; uy/=lu; vx/=lv; vy/=lv;
  double *o = p->results + 11*i;
  o[0]=x[1]; o[1]=y[1];
  o[2]=x[1]+p->size*ux; o[3]=y[1]+p->size*uy;
  o[4]=x[1]+p->size*vx; o[5]=y[1]+p->size*vy;
  o[6]=x[1]+p->size*(ux+vx); o[7]=y[1]+p->size*(uy+vy);
  o[8]=lu; o[9]=lv; o[10]=ux*vx+uy*vy;
}
