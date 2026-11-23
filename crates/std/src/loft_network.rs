//! Through-sections construction. All section curves are compatible rational
//! Bezier spans before the longitudinal interpolation is sent to the device.
use crate::loft_device::{Weighted, rows};
use crate::loft_sections::Section;
use crate::surface_spline::{Interpolation, curve};
use crate::util::{bad, list};
use agcad_geo::brep::{BodyKind, Brep, DEFAULT_LINEAR_TOL as TOL};
use agcad_geo::curve::Curve;
use agcad_geo::math::{Frame, V3};
use agcad_geo::nurbs::{NurbsCurve, NurbsSurface, ders_basis, find_span};
use agcad_geo::surface::Surface;
use agcad_ir::Value;
use agcad_runtime::rebuild::{BuildContext, FeatureError, Product};

type Terms = Vec<(f64, Weighted)>;
pub fn stations(values: &mut Vec<f64>) {
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-11);
}
fn interval(ts: &[f64], t: f64) -> (usize, f64) {
    let i = ts
        .partition_point(|&p| p <= t)
        .saturating_sub(1)
        .min(ts.len() - 2);
    (i, (t - ts[i]) / (ts[i + 1] - ts[i]))
}
fn map(from: &[f64], to: &[f64], t: f64) -> f64 {
    let (i, s) = interval(from, t);
    to[i] + s * (to[i + 1] - to[i])
}
struct Guide {
    curve: NurbsCurve,
    rail: Vec<f64>,
    rim: Vec<f64>,
}
fn guides(
    ctx: &BuildContext<'_>,
    value: Option<&Value>,
    sections: &[Section],
    normal: V3,
) -> Result<Vec<Guide>, FeatureError> {
    let mut curves = Vec::new();
    for val in value.map(list).unwrap_or(&[]) {
        let Product::Curve(Curve::Nurbs(c)) = ctx.product(val)? else {
            return Err(bad("guide requires an editable 3D spline"));
        };
        let mut c = c.clone();
        if c.poles.last().unwrap().dot(normal) < c.poles[0].dot(normal) {
            c.poles.reverse();
            c.weights.reverse();
            c.knots = c.knots.iter().rev().map(|x| 1. - x).collect();
        }
        if c.poles.windows(2).any(|p| (p[1] - p[0]).dot(normal) < -TOL) {
            return Err(bad(
                "guide must advance monotonically through the section planes",
            ));
        }
        let mut guide = Guide {
            curve: c,
            rail: vec![],
            rim: vec![],
        };
        for (i, r) in sections.iter().enumerate() {
            let height = |t| (guide.curve.point(t) - r.frame.origin).dot(normal);
            if height(0.) > TOL || height(1.) < -TOL {
                return Err(bad(format!("guide misses section {}", i + 1)));
            }
            let mut lo = 0.;
            let mut hi = 1.;
            for _ in 0..52 {
                let m = (lo + hi) * 0.5;
                if height(m) < 0. { lo = m } else { hi = m }
            }
            let t = (lo + hi) * 0.5;
            guide.rail.push(t);
            guide.rim.push(
                r.locate(guide.curve.point(t))
                    .map_err(|_| bad(format!("guide misses boundary of section {}", i + 1)))?
                    % 1.,
            );
        }
        curves.push(guide);
    }
    curves.sort_by(|a, b| a.rim[0].total_cmp(&b.rim[0]));
    Ok(curves)
}

struct Layout {
    common: Vec<f64>,
    local: Vec<Vec<f64>>,
    cuts: Vec<f64>,
}
impl Layout {
    fn polar(sections: &[Section]) -> Option<Self> {
        let n = sections.len();
        let centers: Vec<_> = sections.iter().map(Section::center).collect();
        let mut tracks: Vec<Vec<Option<f64>>> = Vec::new();
        let add_original = |tracks: &mut Vec<Vec<Option<f64>>>, i: usize| {
            for &t in &sections[i].stops[..sections[i].edges.len()] {
                if !tracks
                    .iter()
                    .any(|row| row[i].is_some_and(|p| (p - t).abs() < 1e-10))
                {
                    let mut track = vec![None; n];
                    track[i] = Some(t);
                    tracks.push(track);
                }
            }
        };
        let project = |tracks: &mut Vec<Vec<Option<f64>>>, from: usize, to: usize| -> Option<()> {
            let mut cuts = sections[to].stops.clone();
            let mut used = Vec::<f64>::new();
            for track in tracks.iter() {
                if let Some(t) = track[to] {
                    cuts.push(t);
                    used.push(t);
                }
            }
            stations(&mut cuts);
            let mut indices: Vec<_> = (0..tracks.len())
                .filter(|&k| tracks[k][from].is_some() && tracks[k][to].is_none())
                .collect();
            let section = &sections[from];
            let seam_index = (0..section.edges.len()).min_by(|&a, &b| {
                section
                    .at(a, 0.)
                    .dist(section.original_seam)
                    .total_cmp(&section.at(b, 0.).dist(section.original_seam))
            })?;
            let seam = section.stops[seam_index];
            let forward = section.at(seam_index, 1.).dist(section.next_vertex) <= TOL;
            let order = |t: f64| {
                if forward {
                    (t - seam).rem_euclid(1.)
                } else {
                    (seam - t).rem_euclid(1.)
                }
            };
            indices.sort_by(|&i, &j| {
                order(tracks[i][from].unwrap()).total_cmp(&order(tracks[j][from].unwrap()))
            });
            for idx in indices {
                let d = sections[from].position(tracks[idx][from]?) - centers[from];
                let mut t = sections[to].radial(centers[to], d)?;
                let (i, f) = interval(&cuts, t);
                let nearest = if f < 0.1 {
                    Some(cuts[i])
                } else if f > 0.9 {
                    Some(cuts[i + 1] % 1.)
                } else {
                    None
                };
                if let Some(p) = nearest {
                    if !used.iter().any(|&q| (q - p).abs() < 1e-10) {
                        t = p;
                    }
                }
                if used.iter().any(|&p| (p - t).abs() < 1e-10) {
                    return None;
                }
                cuts.push(t);
                stations(&mut cuts);
                used.push(t);
                tracks[idx][to] = Some(t);
            }
            Some(())
        };
        for i in (1..n).rev() {
            add_original(&mut tracks, i);
            project(&mut tracks, i, i - 1)?;
        }
        add_original(&mut tracks, 0);
        for i in 1..n {
            project(&mut tracks, i - 1, i)?;
        }
        tracks.sort_by(|a, b| a[0].unwrap().total_cmp(&b[0].unwrap()));
        // Use the authored seam as the common anchor. In a concave loop a radial
        // projection can hit a different lobe. Keep the projected cuts, then walk
        // each wire in boundary order to avoid reversing an interval of the loft.
        let anchor = (0..tracks.len()).min_by(|&a, &b| {
            let distance = |i: usize| {
                sections[0]
                    .position(tracks[i][0].unwrap())
                    .dist(sections[0].original_seam)
            };
            distance(a).total_cmp(&distance(b))
        })?;
        tracks.rotate_left(anchor);
        let common: Vec<_> = (0..=tracks.len())
            .map(|j| j as f64 / tracks.len() as f64)
            .collect();
        let mut local = Vec::new();
        for i in 0..n {
            let start = tracks[0][i]?;
            let mut marks: Vec<_> = tracks
                .iter()
                .map(|r| {
                    let t = r[i].unwrap();
                    if t < start { t + 1. } else { t }
                })
                .chain([start + 1.])
                .collect();
            marks.sort_by(f64::total_cmp);
            if marks.windows(2).any(|p| p[1] - p[0] < 1e-10) {
                return None;
            }
            local.push(marks);
        }
        Some(Self {
            cuts: common.clone(),
            common,
            local,
        })
    }
    fn new(sections: &[Section], guides: &[Guide]) -> Result<Self, FeatureError> {
        if guides.is_empty()
            && sections
                .iter()
                .any(|s| s.edges.len() != sections[0].edges.len())
        {
            if let Some(layout) = Self::polar(sections) {
                return Ok(layout);
            }
        }
        let mut common: Vec<_> = (0..=guides.len().max(1))
            .map(|i| i as f64 / guides.len().max(1) as f64)
            .collect();
        if guides.is_empty() {
            common = vec![0., 1.];
        }
        let mut cuts = common.clone();
        let mut local = vec![];
        for (i, r) in sections.iter().enumerate() {
            let mut marks: Vec<_> = guides.iter().map(|g| g.rim[i]).collect();
            if marks.is_empty() {
                marks.push(0.);
            }
            let start = marks[0];
            for x in &mut marks[1..] {
                if *x < start {
                    *x += 1.;
                }
            }
            marks.push(start + 1.);
            if marks.windows(2).any(|p| p[1] - p[0] < 1e-10) {
                return Err(bad(format!(
                    "coincident or crossing guides on section {}",
                    i + 1
                )));
            }
            for &s in &r.stops {
                let t = if s < start { s + 1. } else { s };
                cuts.push(map(&marks, &common, t));
            }
            local.push(marks);
        }
        stations(&mut cuts);
        Ok(Self {
            common,
            local,
            cuts,
        })
    }
    fn segment(&self, r: &Section, i: usize, a: f64, b: f64) -> (usize, f64, f64) {
        let a = map(&self.common, &self.local[i], a);
        let b = map(&self.common, &self.local[i], b);
        let cycle = ((a + b) * 0.5).floor();
        r.segment(a - cycle, b - cycle)
    }

    /// Measure section chords in the joint space of compatible control polygons.
    /// Uniformly spaced rim vertices alone underweight the interior of curved edges.
    fn chord_parameters(&self, sections: &[Section]) -> Result<Vec<f64>, FeatureError> {
        use crate::surface_spline::binomial;
        for degree in 1..=14 {
            let nodes: Vec<_> = (0..=degree)
                .map(|i| (1. - (std::f64::consts::PI * i as f64 / degree as f64).cos()) * 0.5)
                .collect();
            let knots = [vec![0.; degree + 1], vec![1.; degree + 1]].concat();
            let interpolation = Interpolation::from_knots(&nodes, degree, knots)?;
            let samples: Vec<_> = (0..=60)
                .map(|i| {
                    let t = i as f64 / 60.;
                    (0..=degree)
                        .map(|k| {
                            binomial(degree, k)
                                * t.powi(k as i32)
                                * (1. - t).powi((degree - k) as i32)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut polygons = Vec::new();
            let mut resolved = true;
            for (i, section) in sections.iter().enumerate() {
                let mut polygon = Vec::new();
                for pair in self.cuts.windows(2) {
                    let (edge, a, b) = self.segment(section, i, pair[0], pair[1]);
                    let point = |t| section.at(edge, a + (b - a) * t);
                    let base = point(0.);
                    let poles = if section.angle(edge, a, b) == 0. {
                        (0..=degree)
                            .map(|k| point(k as f64 / degree as f64) - base)
                            .collect::<Vec<_>>()
                    } else {
                        let values: Vec<_> = nodes.iter().map(|&t| point(t) - base).collect();
                        let poles = interpolation
                            .inverse
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .zip(&values)
                                    .fold(V3::ZERO, |p, (&c, &v)| p + v * c)
                            })
                            .collect::<Vec<_>>();
                        if degree < 14
                            && samples.iter().enumerate().any(|(s, row)| {
                                let q = row.iter().zip(&poles).fold(base, |p, (&c, &v)| p + v * c);
                                q.dist(point(s as f64 / 60.)) > TOL * 0.1
                            })
                        {
                            resolved = false;
                            break;
                        }
                        poles
                    };
                    polygon.extend(poles.into_iter().take(degree).map(|p| base + p));
                }
                if !resolved {
                    break;
                }
                polygons.push(polygon);
            }
            if !resolved {
                continue;
            }
            let first = &polygons[0];
            let seam = (0..first.len())
                .step_by(degree)
                .min_by(|&i, &j| {
                    first[i]
                        .dist(sections[0].original_seam)
                        .total_cmp(&first[j].dist(sections[0].original_seam))
                })
                .unwrap();
            let mut ts = vec![0.];
            for pair in polygons.windows(2) {
                let sum = pair[0]
                    .iter()
                    .zip(&pair[1])
                    .map(|(&a, &b)| a.dist(b).powi(2))
                    .sum::<f64>();
                ts.push(
                    ts.last().unwrap() + (sum + pair[0][seam].dist(pair[1][seam]).powi(2)).sqrt(),
                );
            }
            let total = *ts.last().unwrap();
            for t in &mut ts {
                *t /= total;
            }
            return Ok(ts);
        }
        Err(bad("could not parameterize section polygons"))
    }
}

pub fn rail(surface: &NurbsSurface, last: bool) -> NurbsCurve {
    let i = if last { surface.nu - 1 } else { 0 };
    curve(
        surface.dv,
        surface.kv.clone(),
        (0..surface.nv)
            .map(|j| {
                let k = i * surface.nv + j;
                (surface.poles[k], surface.weights[k])
            })
            .collect(),
    )
}
pub fn rim(surface: &NurbsSurface, end: bool) -> NurbsCurve {
    let j = if end { surface.nv - 1 } else { 0 };
    curve(
        surface.du,
        surface.ku.clone(),
        (0..surface.nu)
            .map(|i| {
                let k = i * surface.nv + j;
                (surface.poles[k], surface.weights[k])
            })
            .collect(),
    )
}

// Subdivide the homogeneous Bezier difference at the same parameter on both rails.
// Its projected control hull bounds every point on that interval.
fn contact_parameter(poles: &[Weighted], lo: f64, hi: f64) -> Option<f64> {
    let mut bounds = agcad_geo::math::Aabb::EMPTY;
    for &(p, w) in poles {
        bounds.add(p * (1. / w));
    }
    let axis_gap = |a: f64, b: f64| a.max(-b).max(0.);
    let distance = agcad_geo::math::v3(
        axis_gap(bounds.min.x, bounds.max.x),
        axis_gap(bounds.min.y, bounds.max.y),
        axis_gap(bounds.min.z, bounds.max.z),
    )
    .norm();
    if distance > TOL {
        return None;
    }
    if poles[0].0.norm() <= TOL * poles[0].1 {
        return Some(lo);
    }
    let mid = (lo + hi) * 0.5;
    if mid == lo || mid == hi {
        return Some(mid);
    }
    let mut work = poles.to_vec();
    let mut left = vec![work[0]];
    let mut right = vec![*work.last().unwrap()];
    for n in (1..poles.len()).rev() {
        for i in 0..n {
            work[i] = ((work[i].0 + work[i + 1].0) * 0.5, (work[i].1 + work[i + 1].1) * 0.5);
        }
        left.push(work[0]);
        right.push(work[n - 1]);
    }
    right.reverse();
    contact_parameter(&left, lo, mid).or_else(|| contact_parameter(&right, mid, hi))
}

fn check_rails(patches: &[NurbsSurface], heights: &[f64]) -> Result<(), FeatureError> {
    let mut rails = Vec::with_capacity(patches.len());
    for patch in patches {
        let mut c = rail(patch, false);
        let mut breaks = c.knots.clone();
        breaks.dedup();
        for &t in &breaks[1..breaks.len() - 1] {
            let multiplicity = c.knots.iter().filter(|&&k| k == t).count();
            for _ in multiplicity..c.degree {
                c = c.insert_knot(t);
            }
        }
        rails.push(c);
    }
    let d = rails[0].degree;
    let spans = (rails[0].poles.len() - 1) / d;
    let candidates = agcad_geo::accelerator::rail_candidates(&rails, TOL)?;
    for (i, a) in rails.iter().enumerate() {
        for (j, b) in rails[..i].iter().enumerate() {
            for start in (0..a.poles.len() - 1).step_by(d) {
                let pair = i * (i - 1) / 2 + j;
                if candidates.as_ref().is_some_and(|keep| keep[pair * spans + start / d] == 0) {
                    continue;
                }
                // Multiply the rational difference by its two positive denominators.
                // Dividing each Bernstein coefficient by the corresponding product
                // weight gives a rational curve with the same convex-hull property.
                let mut diff = Vec::with_capacity(2 * d + 1);
                for k in 0..=2 * d {
                    let mut delta = V3::ZERO;
                    let mut weight = 0.;
                    for p in k.saturating_sub(d)..=k.min(d) {
                        let q = k - p;
                        let w = crate::surface_spline::binomial(d, p)
                            * crate::surface_spline::binomial(d, q)
                            / crate::surface_spline::binomial(2 * d, k)
                            * a.weights[start + p]
                            * b.weights[start + q];
                        delta = delta + (a.poles[start + p] - b.poles[start + q]) * w;
                        weight += w;
                    }
                    diff.push((delta, weight));
                }
                if let Some(t) =
                    contact_parameter(&diff, a.knots[start + d], a.knots[start + 2 * d])
                {
                    let (section, _) = interval(heights, t);
                    return Err(FeatureError::new(
                        agcad_driver::AgStatus::TopologyFailed,
                        format!(
                            "loft rails {j} and {i} meet between sections {} and {} (v={t:.8})",
                            section + 1,
                            section + 2
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}
fn edge(b: &mut Brep, c: NurbsCurve, a: u32, z: u32, name: String) -> u32 {
    let (lo, hi) = c.domain();
    let cv = b.add_curve(Curve::Nurbs(c));
    b.add_edge(cv, a, z, lo, hi, name)
}

pub fn assemble(
    surfaces: Vec<NurbsSurface>,
    caps: Option<([Frame; 2], V3)>,
    closed_ring: bool,
) -> Result<Brep, FeatureError> {
    let mut b = Brep::new(if caps.is_some() {
        BodyKind::Solid
    } else {
        BodyKind::Sheet
    });
    let shell = b.add_shell(caps.is_some());
    let count = surfaces.len();
    let n = count + usize::from(!closed_ring);
    let mut verts = [Vec::new(), Vec::new()];
    let mut rings = [Vec::new(), Vec::new()];
    let mut rails = vec![];
    for i in 0..n {
        let c = rail(&surfaces[i.min(count - 1)], i == count);
        for (k, p) in [c.poles[0], *c.poles.last().unwrap()]
            .into_iter()
            .enumerate()
        {
            verts[k].push(b.add_vertex(p));
        }
        rails.push(edge(
            &mut b,
            c,
            verts[0][i],
            verts[1][i],
            format!("rail.{i}"),
        ));
    }
    for (i, s) in surfaces.into_iter().enumerate() {
        let j = (i + 1) % n;
        for k in 0..2 {
            rings[k].push(edge(
                &mut b,
                rim(&s, k == 1),
                verts[k][i],
                verts[k][j],
                format!("section.{k}.{i}"),
            ));
        }
        let sf = b.add_surface(Surface::Nurbs(s));
        b.add_face(
            sf,
            true,
            shell,
            &[
                (rings[0][i], true),
                (rails[j], true),
                (rings[1][i], false),
                (rails[i], false),
            ],
            &[],
            format!("side.{i}"),
            [0., 1., 0., 1.],
        );
    }
    if let Some((frames, normal)) = caps {
        for (k, f) in frames.into_iter().enumerate() {
            let mut bounds = [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];
            for &e in &rings[k] {
                if let Curve::Nurbs(c) = &b.curves[b.edges[e as usize].curve as usize] {
                    for &p in &c.poles {
                        let p = f.to_local(p);
                        bounds = [
                            bounds[0].min(p.x),
                            bounds[1].max(p.x),
                            bounds[2].min(p.y),
                            bounds[3].max(p.y),
                        ];
                    }
                }
            }
            let sf = b.add_surface(Surface::Plane { frame: f });
            let mut edges: Vec<_> = rings[k].iter().map(|&e| (e, k == 1)).collect();
            if k == 0 {
                edges.reverse();
            }
            b.add_face(
                sf,
                (k == 1) == (f.z.dot(normal) > 0.),
                shell,
                &edges,
                &[],
                if k == 0 { "cap_start" } else { "cap_end" },
                bounds,
            );
        }
    }
    let check = b.validate();
    if !check.valid {
        return Err(bad(check.problems.join(", ")));
    }
    Ok(b)
}

// Express a value of the longitudinal interpolant in terms of its section samples.
fn coefficients(fit: &Interpolation, t: f64) -> Vec<f64> {
    let n = fit.inverse.len();
    let span = find_span(n - 1, fit.degree, t, &fit.knots);
    let basis = ders_basis(span, t, fit.degree, 0, &fit.knots);
    (0..n)
        .map(|i| {
            (0..=fit.degree)
                .map(|k| basis[0][k] * fit.inverse[span - fit.degree + k][i])
                .sum()
        })
        .collect()
}
pub fn build(
    ctx: &BuildContext<'_>,
    mut sections: Vec<Section>,
    guide_refs: Option<&Value>,
    ruled: bool,
    solid: bool,
) -> Result<(Brep, usize, &'static str), FeatureError> {
    let first = sections[0].frame;
    let normal = first.z
        * (sections.last().unwrap().frame.origin - first.origin)
            .dot(first.z)
            .signum();
    let mut heights = vec![];
    let unequal = sections
        .iter()
        .any(|r| r.edges.len() != sections[0].edges.len());
    for (i, r) in sections.iter_mut().enumerate() {
        if let Some((a, b)) = agcad_geo::region2d::crossing_edges(&r.edges, TOL) {
            return Err(FeatureError::new(
                agcad_driver::AgStatus::TopologyFailed,
                format!("section {} edges {a} and {b} intersect", i + 1),
            ));
        }
        if r.frame.z.cross(first.z).norm() > 1e-12 {
            return Err(bad(format!("section {} is not parallel", i + 1)));
        }
        let h = (r.frame.origin - first.origin).dot(normal);
        if heights.last().is_some_and(|&last| h - last < TOL) {
            return Err(bad(format!(
                "section {} is coincident or out of order",
                i + 1
            )));
        }
        heights.push(h);
        r.orient(normal);
    }
    let last = *heights.last().unwrap();
    for h in &mut heights {
        *h /= last;
    }
    let (first, rest) = sections.split_first_mut().unwrap();
    for r in rest {
        r.align(first);
    }
    for r in &mut sections {
        r.parameterize(unequal);
    }
    let guides = guides(ctx, guide_refs, &sections, normal)?;
    let layout = Layout::new(&sections, &guides)?;
    let mut cuts = layout.cuts.clone();
    for pair in layout.cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let sweep = sections
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let (e, x, y) = layout.segment(r, i, a, b);
                r.angle(e, x, y).abs()
            })
            .fold(0., f64::max);
        let splits = (sweep / std::f64::consts::FRAC_PI_2 - 1e-12).ceil() as usize;
        for k in 1..splits {
            cuts.push(a + (b - a) * k as f64 / splits as f64);
        }
    }
    stations(&mut cuts);
    if sections.len() > 2 && !ruled && guides.is_empty() {
        heights = layout.chord_parameters(&sections)?;
    }
    let fit = Interpolation::new(&heights, ruled)?;
    let mut vt = heights.clone();
    vt.extend(fit.knots.iter().copied());
    for g in &guides {
        for &k in &g.curve.knots {
            if k > g.rail[0] && k < *g.rail.last().unwrap() {
                vt.push(map(&g.rail, &heights, k));
            }
        }
    }
    stations(&mut vt);
    let cubic = Interpolation::new(&[0., 1. / 3., 2. / 3., 1.], false)?;
    let mut kv = vec![0.; 4];
    for &t in &vt[1..vt.len() - 1] {
        kv.extend([t; 3]);
    }
    kv.extend([1.; 4]);
    let (dv, kv) = if guides.is_empty() {
        (fit.degree, fit.knots.clone())
    } else {
        (3, kv)
    };
    let nv = kv.len() - dv - 1;
    let mut terms: Vec<Terms> = Vec::new();
    let mut degrees = vec![];
    for ends in cuts.windows(2) {
        let sweeps: Vec<_> = sections
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let (e, a, b) = layout.segment(r, i, ends[0], ends[1]);
                r.angle(e, a, b).abs()
            })
            .collect();
        let angular = sweeps.iter().any(|s| (s - sweeps[0]).abs() > 1e-11);
        let profiles = sections
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let (edge, a, b) = layout.segment(r, i, ends[0], ends[1]);
                if angular {
                    r.angular(edge, a, b)
                } else {
                    r.quadratic(edge, a, b)
                }
            })
            .collect::<Vec<_>>();
        let du = profiles[0].degree;
        degrees.push(du);
        let rails: [Option<&Guide>; 2] = std::array::from_fn(|k| {
            guides
                .iter()
                .enumerate()
                .find(|(j, _)| (layout.common[*j] - (ends[k] % 1.)).abs() < 1e-10)
                .map(|(_, g)| g)
        });
        for u in 0..=du {
            let cp: Vec<_> = profiles
                .iter()
                .map(|p| (p.poles[u], p.weights[u]))
                .collect();
            if guides.is_empty() {
                for row in &fit.inverse {
                    terms.push(row.iter().copied().zip(cp.iter().copied()).collect());
                }
                continue;
            }
            for (span, w) in vt.windows(2).enumerate() {
                let mut samples = vec![];
                for j in 0..4 {
                    let t = w[0] + (w[1] - w[0]) * j as f64 / 3.;
                    let coeff = coefficients(&fit, t);
                    let mut row: Terms = coeff.iter().copied().zip(cp.iter().copied()).collect();
                    for (end, g) in rails.iter().enumerate() {
                        if let Some(g) = g {
                            let factor = if end == 0 {
                                1. - u as f64 / du as f64
                            } else {
                                u as f64 / du as f64
                            };
                            row.push((factor, (g.curve.point(map(&heights, &g.rail, t)), 1.)));
                            for (i, &c) in coeff.iter().enumerate() {
                                row.push((-factor * c, (profiles[i].poles[du * end], 1.)));
                            }
                        }
                    }
                    samples.push(row);
                }
                for co in cubic.inverse.iter().skip(usize::from(span > 0)) {
                    terms.push(
                        co.iter()
                            .zip(&samples)
                            .flat_map(|(&c, s)| s.iter().map(move |&(v, p)| (c * v, p)))
                            .collect(),
                    );
                }
            }
        }
    }
    let (poles, device) = rows(&terms).map_err(|e| bad(format!("loft CUDA control net: {e}")))?;
    if poles
        .iter()
        .any(|(p, w)| !w.is_finite() || *w <= 1e-12 || !p.norm().is_finite())
    {
        return Err(bad("loft interpolation produced a singular rational patch"));
    }
    let mut surfaces = vec![];
    let mut poles = poles.into_iter();
    for du in degrees {
        let (p, w) = poles.by_ref().take((du + 1) * nv).unzip();
        surfaces.push(NurbsSurface {
            du,
            dv,
            nu: du + 1,
            nv,
            ku: [vec![0.; du + 1], vec![1.; du + 1]].concat(),
            kv: kv.clone(),
            poles: p,
            weights: w,
        });
    }
    let n = surfaces.len();
    check_rails(&surfaces, &heights)?;
    let caps = solid.then(|| ([sections[0].frame, sections.last().unwrap().frame], normal));
    let mut body = assemble(surfaces, caps, true)?;
    for (ring, section_id) in [0, sections.len() - 1].into_iter().enumerate() {
        let section = &sections[section_id];
        let mut spans = std::collections::BTreeMap::new();
        for (i, pair) in cuts.windows(2).enumerate() {
            let (e, _, _) = layout.segment(section, section_id, pair[0], pair[1]);
            let source = section.source_edges[e];
            let span = spans.entry(source).or_insert(0usize);
            // Assembly inserts the longitudinal edges first, then the two rims for each face.
            body.edges[n + 2 * i + ring].name =
                format!("section.{section_id}.edge.{source}.span.{span}");
            *span += 1;
        }
    }
    Ok((body, n, device))
}
