use crate::sketch::regions_of;
use crate::util::bad;
use agcad_geo::brep::DEFAULT_LINEAR_TOL as TOL;
use agcad_geo::builders::Seg2;
use agcad_geo::math::{v3, Frame, V3};
use agcad_geo::nurbs::NurbsCurve;
use agcad_ir::Value;
use agcad_runtime::rebuild::{BuildContext, FeatureError};
use std::f64::consts::{FRAC_PI_2, TAU};

pub struct Section {
    pub frame: Frame,
    pub edges: Vec<Seg2>,
    pub source_edges: Vec<usize>,
    pub original_seam: V3,
    pub next_vertex: V3,
    pub stops: Vec<f64>,
}

fn length(edge: &Seg2) -> f64 {
    match *edge {
        Seg2::Line { a, b } => (b[0] - a[0]).hypot(b[1] - a[1]),
        Seg2::Arc { r, a0, a1, .. } => r * (a1 - a0).abs(),
        Seg2::Circle { r, .. } => TAU * r,
    }
}
fn point(edge: &Seg2, t: f64) -> [f64; 2] {
    match *edge {
        Seg2::Line { a, b } => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])],
        Seg2::Arc { c, r, a0, a1 } => {
            let a = a0 + t * (a1 - a0);
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        }
        Seg2::Circle { .. } => unreachable!("circles expanded to four arcs"),
    }
}

impl Section {
    pub fn read(ctx: &BuildContext<'_>, val: &Value, index: usize) -> Result<Self, FeatureError> {
        let r = regions_of(ctx, val)?;
        let [region] = r.regions.as_slice() else {
            return Err(bad(format!("section {index} requires one region")));
        };
        let [ring] = region.loops.as_slice() else {
            return Err(bad(format!("section {index} has interior loops")));
        };
        if region.area().abs() < TOL * TOL || ring.is_empty() {
            return Err(bad(format!("degenerate section {index}")));
        }
        let mut edges = Vec::new();
        let mut source_edges = Vec::new();
        let close = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]) <= TOL;
        for (fallback, e) in ring.iter().enumerate() {
            let source = r
                .source_edges
                .iter()
                .find(|(_, a, b)| {
                    (close(*a, e.start()) && close(*b, e.end()))
                        || (close(*b, e.start()) && close(*a, e.end()))
                })
                .map_or(fallback, |s| s.0);
            if let Seg2::Circle { c, r, ccw } = *e {
                let da = if ccw { FRAC_PI_2 } else { -FRAC_PI_2 };
                for k in 0..4 {
                    source_edges.push(source);
                    edges.push(Seg2::Arc {
                        c,
                        r,
                        a0: k as f64 * da,
                        a1: (k + 1) as f64 * da,
                    });
                }
            } else {
                edges.push(*e);
                source_edges.push(source);
            }
        }
        if edges
            .iter()
            .any(|e| !length(e).is_finite() || length(e) < TOL)
        {
            return Err(bad(format!("short edge in section {index}")));
        }
        let (mut start, mut end) = r
            .source_edges
            .first()
            .map_or((ring[0].start(), ring[0].end()), |(_, a, b)| (*a, *b));
        if let Some((_, a, b)) = r.source_edges.get(1) {
            if (close(start, *a) || close(start, *b)) && !(close(end, *a) || close(end, *b)) {
                std::mem::swap(&mut start, &mut end);
            }
        }
        Ok(Self {
            frame: r.frame,
            original_seam: r.frame.to_world(v3(start[0], start[1], 0.)),
            next_vertex: r.frame.to_world(v3(end[0], end[1], 0.)),
            edges,
            source_edges,
            stops: Vec::new(),
        })
    }
    pub fn world(&self, p: [f64; 2]) -> V3 {
        self.frame.to_world(v3(p[0], p[1], 0.))
    }
    pub fn center(&self) -> V3 {
        let (mut mass, mut moment) = (0., V3::ZERO);
        for (i, e) in self.edges.iter().enumerate() {
            let centroid = match *e {
                Seg2::Arc { c, r, a0, a1 } => {
                    let half = (a1 - a0) * 0.5;
                    let radius = r * half.sin() / half;
                    let (sn, cs) = ((a0 + a1) * 0.5).sin_cos();
                    self.world([c[0] + radius * cs, c[1] + radius * sn])
                }
                _ => self.at(i, 0.5),
            };
            let l = length(e);
            mass += l;
            moment = moment + centroid * l;
        }
        moment * (1. / mass)
    }
    pub fn position(&self, t: f64) -> V3 {
        let (e, u, _) = self.segment(t, t);
        self.at(e, u)
    }
    /// Intersect a centroid ray with the planar profile. Multiple intersections
    /// of a concave profile are resolved by distance to the projected vertex.
    pub fn radial(&self, center: V3, direction: V3) -> Option<f64> {
        let c = self.frame.to_local(center);
        let d = v3(direction.dot(self.frame.x), direction.dot(self.frame.y), 0.);
        let z = v3(c.x, c.y, 0.);
        let norm = d.norm();
        if norm < 1e-12 {
            return None;
        }
        let d = d * (1. / norm);
        let mut best = None;
        for (i, e) in self.edges.iter().enumerate() {
            let mut candidates = Vec::new();
            match *e {
                Seg2::Line { a, b } => {
                    let start = v3(a[0], a[1], 0.);
                    let edge = v3(b[0] - a[0], b[1] - a[1], 0.);
                    let det = edge.cross(d).z;
                    if det.abs() > 1e-12 {
                        candidates.push((z - start).cross(d).z / det);
                    }
                }
                Seg2::Arc { c, r, a0, a1 } => {
                    let off = z - v3(c[0], c[1], 0.);
                    let along = off.dot(d);
                    let disc = r * r - off.cross(d).z.powi(2);
                    if disc >= 0. {
                        for s in [-along - disc.sqrt(), -along + disc.sqrt()] {
                            let p = off + d * s;
                            let a = (p.y.atan2(p.x) - a0) * (a1 - a0).signum();
                            let mut t = a.rem_euclid(TAU) / (a1 - a0).abs();
                            if (TAU - a.rem_euclid(TAU)).abs() < 1e-12 {
                                t = 0.;
                            }
                            candidates.push(t);
                        }
                    }
                }
                _ => unreachable!(),
            }
            for t in candidates {
                if !(-1e-10..=1. + 1e-10).contains(&t) {
                    continue;
                }
                let t = t.clamp(0., 1.);
                let delta = self.at(i, t) - center;
                if delta.dot(direction) < 0. {
                    continue;
                }
                let error = (delta - direction).norm();
                if best.is_none_or(|(cost, _)| error < cost) {
                    best = Some((
                        error,
                        (self.stops[i] + t * (self.stops[i + 1] - self.stops[i])) % 1.,
                    ));
                }
            }
        }
        best.map(|(_, t)| t)
    }
    pub fn at(&self, i: usize, t: f64) -> V3 {
        self.world(point(&self.edges[i], t))
    }
    pub fn orient(&mut self, normal: V3) {
        if self.frame.z.dot(normal) < 0. {
            self.edges = self.edges.iter().rev().map(Seg2::reversed).collect();
            self.source_edges.reverse();
        }
        let idx = (0..self.edges.len())
            .min_by(|&i, &j| {
                let a = self.at(i, 0.);
                let b = self.at(j, 0.);
                a.x.total_cmp(&b.x)
                    .then(a.y.total_cmp(&b.y))
                    .then(a.z.total_cmp(&b.z))
            })
            .unwrap();
        self.edges.rotate_left(idx);
        self.source_edges.rotate_left(idx);
    }
    pub fn align(&mut self, first: &Self) {
        let n = self.edges.len();
        let cost = |k: usize| {
            if n == first.edges.len() {
                (0..n)
                    .map(|i| {
                        let d = first.at(i, 0.) - self.at((i + k) % n, 0.);
                        d.dot(d)
                    })
                    .sum()
            } else {
                self.at(k, 0.).dist(first.at(0, 0.)).powi(2)
            }
        };
        let k = (0..n).min_by(|&i, &j| cost(i).total_cmp(&cost(j))).unwrap();
        self.edges.rotate_left(k);
        self.source_edges.rotate_left(k);
    }
    pub fn parameterize(&mut self, by_length: bool) {
        self.stops = vec![0.];
        for e in &self.edges {
            self.stops
                .push(self.stops.last().unwrap() + if by_length { length(e) } else { 1. });
        }
        let last = *self.stops.last().unwrap();
        for x in &mut self.stops {
            *x /= last;
        }
    }
    pub fn segment(&self, t0: f64, t1: f64) -> (usize, f64, f64) {
        let mid = (t0 + t1) * 0.5;
        let i = self
            .stops
            .partition_point(|&x| x <= mid)
            .saturating_sub(1)
            .min(self.edges.len() - 1);
        let t = |x: f64| ((x - self.stops[i]) / (self.stops[i + 1] - self.stops[i])).clamp(0., 1.);
        (i, t(t0), t(t1))
    }
    pub fn angle(&self, i: usize, t0: f64, t1: f64) -> f64 {
        if let Seg2::Arc { a0, a1, .. } = self.edges[i] {
            (a1 - a0) * (t1 - t0)
        } else {
            0.
        }
    }
    pub fn quadratic(&self, i: usize, a: f64, b: f64) -> NurbsCurve {
        match self.edges[i] {
            Seg2::Arc { c, .. } => {
                let weight = (self.angle(i, a, b) * 0.5).cos();
                let center = self.world(c);
                let middle = center + (self.at(i, (a + b) * 0.5) - center) * (1. / weight);
                let mut result = NurbsCurve::bezier(vec![self.at(i, a), middle, self.at(i, b)]);
                result.weights[1] = weight;
                result
            }
            _ => NurbsCurve::bezier(vec![
                self.at(i, a),
                self.at(i, (a + b) * 0.5),
                self.at(i, b),
            ]),
        }
    }
    /// Compatible near-uniform angular parameterization for different arcs, including
    /// arc-to-line correspondence. Squaring a complex half-angle Bezier in homogeneous
    /// coordinates preserves the circle locus exactly. Endpoint Hermite conditions
    /// through order four keep the angular speed uniform to roundoff on small spans.
    pub fn angular(&self, i: usize, a: f64, b: f64) -> NurbsCurve {
        const N: usize = 9;
        use crate::surface_spline::binomial as choose;
        let Seg2::Arc { c, r, a0, a1 } = self.edges[i] else {
            return NurbsCurve::bezier(
                (0..=2 * N)
                    .map(|k| self.at(i, a + (b - a) * k as f64 / (2 * N) as f64))
                    .collect(),
            );
        };
        let angle = (a1 - a0) * (b - a);
        let mut half = [[0.; 2]; N + 1];
        for (k, p) in half.iter_mut().enumerate() {
            let back = k > N / 2;
            let m = if back { N - k } else { k };
            let theta = (a0 + (a1 - a0) * if back { b } else { a }) / 2.;
            let (mut y, mut x) = theta.sin_cos();
            let rate = if back { -angle / 2. } else { angle / 2. };
            for j in 0..=m {
                p[0] += x;
                p[1] += y;
                let step = rate * (m - j) as f64 / ((j + 1) * (N - j)) as f64;
                (x, y) = (-y * step, x * step);
            }
        }
        let mut points = Vec::new();
        let mut weights = Vec::new();
        for k in 0..=2 * N {
            let (mut x, mut y, mut w) = (0., 0., 0.);
            for j in k.saturating_sub(N)..=N.min(k) {
                let [p, q] = half[j];
                let [s, t] = half[k - j];
                let co = choose(N, j) * choose(N, k - j) / choose(2 * N, k);
                x += co * (p * s - q * t);
                y += co * (p * t + q * s);
                w += co * (p * s + q * t);
            }
            points.push(self.world([c[0] + r * x / w, c[1] + r * y / w]));
            weights.push(w);
        }
        let mut curve = NurbsCurve::bezier(points);
        curve.weights = weights;
        curve
    }
    pub fn locate(&self, p: V3) -> Result<f64, FeatureError> {
        let q = self.frame.to_local(p);
        for (i, e) in self.edges.iter().enumerate() {
            let t = match *e {
                Seg2::Line { a, b } => {
                    let x = b[0] - a[0];
                    let y = b[1] - a[1];
                    ((q.x - a[0]) * x + (q.y - a[1]) * y) / (x * x + y * y)
                }
                Seg2::Arc { c, a0, a1, .. } => {
                    let d = a1 - a0;
                    let angle = (q.y - c[1]).atan2(q.x - c[0]);
                    ((angle - a0) * d.signum()).rem_euclid(TAU) / d.abs()
                }
                _ => unreachable!(),
            }
            .clamp(0., 1.);
            if self.at(i, t).dist(p) <= TOL {
                return Ok(self.stops[i] + t * (self.stops[i + 1] - self.stops[i]));
            }
        }
        Err(bad("guide intersection lies outside the section boundary"))
    }
}
