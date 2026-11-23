//! Blend the section boundaries of two editable lofts using endpoint jets.
use crate::loft_device::{Weighted, rows};
use crate::loft_network::assemble;
use crate::util::{bad, list, req};
use agcad_geo::nurbs::{NurbsSurface, ders_basis, find_span};
use agcad_geo::surface::Surface;
use agcad_ir::Value;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use std::sync::Arc;

struct Boundary {
    patches: Vec<NurbsSurface>,
    end: bool,
    closed: bool,
}

pub fn boundary(
    ctx: &BuildContext<'_>,
    scope: &Value,
    name: &str,
) -> Result<Product, FeatureError> {
    let (prefix, endpoint) = name
        .rsplit_once('/')
        .or_else(|| name.strip_prefix("section.").map(|s| ("section", s)))
        .ok_or_else(|| bad("expected a loft section or a side endpoint"))?;
    let end = match endpoint {
        "start" => false,
        "end" => true,
        _ => return Err(bad("expected start or end boundary")),
    };
    let owned;
    let body = match ctx.product(scope)? {
        Product::Sheet(b) => b,
        Product::Solid(s) => {
            owned = s.brep();
            &owned
        }
        _ => return Err(bad("blend boundary requires a loft sheet or solid")),
    };
    let whole_section = prefix == "section";
    let mut indexed = Vec::new();
    for f in &body.faces {
        let Some(i) = f
            .name
            .strip_prefix("side.")
            .and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        if !whole_section && prefix != f.name {
            continue;
        }
        let Surface::Nurbs(s) = &body.surfaces[f.surface as usize] else {
            return Err(bad("expected a loft support patch"));
        };
        let rim = f
            .loops
            .first()
            .and_then(|&lp| body.loops.get(lp as usize))
            .and_then(|lp| lp.coedges.get(if end { 2 } else { 0 }))
            .copied()
            .ok_or_else(|| bad("expected a four-sided loft support patch"))?;
        indexed.push((i, s.clone(), body.coedge_start(rim), body.coedge_end(rim)));
    }
    indexed.sort_by_key(|p| p.0);
    if indexed.is_empty() {
        return Err(bad(format!("unknown loft boundary {name}")));
    }
    let first = &indexed[0];
    let last = indexed.last().unwrap();
    let closed = whole_section
        && if end {
            first.3 == last.2
        } else {
            first.2 == last.3
        };
    Ok(Product::Other(Arc::new(Boundary {
        patches: indexed.into_iter().map(|p| p.1).collect(),
        end,
        closed,
    })))
}

// Coefficients of a Bernstein end pole from the support's homogeneous position,
// first derivative and second derivative. Reversing the endpoint reverses the first jet.
fn end_row(
    s: &NurbsSurface,
    u: usize,
    end: bool,
    h: f64,
    level: usize,
    degree: usize,
) -> Vec<(f64, Weighted)> {
    let v = if end { 1. } else { 0. };
    let span = find_span(s.nv - 1, s.dv, v, &s.kv);
    let basis = ders_basis(span, v, s.dv, level, &s.kv);
    let a = h * level as f64 / degree as f64;
    let b = if level == 2 {
        h * h / (degree * (degree - 1)) as f64
    } else {
        0.
    };
    (0..=s.dv)
        .map(|j| {
            let w = basis[0][j]
                + basis.get(1).map_or(0., |r| r[j] * a)
                + basis.get(2).map_or(0., |r| r[j] * b);
            let k = u * s.nv + span - s.dv + j;
            (w, (s.poles[k], s.weights[k]))
        })
        .collect()
}

fn selection<'a>(ctx: &'a BuildContext<'_>, v: &Value) -> Result<&'a Boundary, FeatureError> {
    if let Product::Other(b) = ctx.product(v)? {
        if let Some(b) = b.downcast_ref::<Boundary>() {
            return Ok(b);
        }
    }
    Err(bad(
        "select a supported section.start, section.end, or side.N/start or end with query.named",
    ))
}

pub struct Blend;
impl FeatureImpl for Blend {
    fn op(&self) -> &'static str {
        "surface.blend"
    }
    fn execute(
        &self,
        ctx: &mut BuildContext<'_>,
        node: &FeatureNode,
    ) -> Result<Product, FeatureError> {
        let first = selection(ctx, req(node, "a")?)?;
        let last = selection(ctx, req(node, "b")?)?;
        let [ao, bo] = list(req(node, "order")?) else {
            return Err(bad("supply one continuity order per boundary"));
        };
        let order = |v| match ctx.number(v)? {
            1. => Ok(1usize),
            2. => Ok(2usize),
            _ => Err(bad("supported continuity orders are 1 and 2")),
        };
        let orders = [order(ao)?, order(bo)?];
        let degree = orders[0] + orders[1] + 1;
        let mut tension = [1.; 2];
        if let Some(t) = node.args.get("tension") {
            let [a, b] = list(t) else {
                return Err(bad("supply two positive tension values"));
            };
            tension = [ctx.number(a)?, ctx.number(b)?];
            if tension.iter().any(|x| !x.is_finite() || *x <= 0.) {
                return Err(bad("blend tension must be positive"));
            }
        }
        let n = first.patches.len();
        if first.closed != last.closed || n != last.patches.len() {
            return Err(bad("blend boundaries require corresponding support spans"));
        }
        let av = f64::from(first.end as u8);
        let bv = f64::from(last.end as u8);
        let mut best = (f64::INFINITY, 0, false);
        for shift in 0..n {
            for rev in [false, true] {
                let cost = (0..n)
                    .map(|i| {
                        let a = &first.patches[i];
                        let j = if rev {
                            (shift + n - i) % n
                        } else {
                            (i + shift) % n
                        };
                        let b = &last.patches[j];
                        [0., 1.]
                            .iter()
                            .map(|&u| {
                                a.jet(u, av)
                                    .p
                                    .dist(b.jet(if rev { 1. - u } else { u }, bv).p)
                                    .powi(2)
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                if cost < best.0 {
                    best = (cost, shift, rev);
                }
            }
        }
        let (_, shift, reverse) = best;
        let pair = |i| {
            if reverse {
                (shift + n - i) % n
            } else {
                (shift + i) % n
            }
        };
        let mut speed = [0.; 2];
        let mut gap = 0.;
        for (i, a) in first.patches.iter().enumerate() {
            let a = a.jet(0.5, av);
            let b = last.patches[pair(i)].jet(0.5, bv);
            speed[0] += a.dv.norm();
            speed[1] += b.dv.norm();
            gap += a.p.dist(b.p);
        }
        if gap < 1e-8 || speed.iter().any(|v| *v < 1e-9) {
            return Err(bad("coincident or singular blend supports"));
        }
        let outward = |end| if end { 1. } else { -1. };
        // At either support the control points advance out of its trimmed domain.
        let h = [
            outward(first.end) * gap / speed[0] * tension[0],
            outward(last.end) * gap / speed[1] * tension[1],
        ];
        let mut terms = Vec::new();
        let mut grids = Vec::new();
        for (i, a) in first.patches.iter().enumerate() {
            let b = &last.patches[pair(i)];
            if a.nu != a.du + 1 || b.nu != b.du + 1 {
                return Err(bad("blend requires Bezier section spans"));
            }
            let du = a.du.max(b.du);
            grids.push(du);
            for u in 0..=du {
                for v in 0..=degree {
                    let (s, x, end, scale, l) = if v <= orders[0] {
                        (a, u, first.end, h[0], v)
                    } else {
                        (
                            b,
                            if reverse { du - u } else { u },
                            last.end,
                            h[1],
                            degree - v,
                        )
                    };
                    let row = if s.du == du || x == 0 || x == du {
                        end_row(s, x * s.du / du, end, scale, l, degree)
                    } else {
                        // Degree elevation is multiplication by (u + (1-u))^(du-s.du).
                        use crate::surface_spline::binomial;
                        (x.saturating_sub(du - s.du)..=x.min(s.du))
                            .flat_map(|k| {
                                let weight = binomial(s.du, k) * binomial(du - s.du, x - k)
                                    / binomial(du, x);
                                end_row(s, k, end, scale, l, degree)
                                    .into_iter()
                                    .map(move |(c, p)| (weight * c, p))
                            })
                            .collect()
                    };
                    terms.push(row);
                }
            }
        }
        let (points, device) = rows(&terms).map_err(|e| bad(format!("blend control net: {e}")))?;
        if points
            .iter()
            .any(|(p, w)| !w.is_finite() || *w <= 1e-12 || !p.norm().is_finite())
        {
            return Err(bad("singular rational blend"));
        }
        let knots = |degree| [vec![0.; degree + 1], vec![1.; degree + 1]].concat();
        let mut points = points.into_iter();
        let mut patches = Vec::new();
        for du in grids {
            let (poles, weights) = points.by_ref().take((du + 1) * (degree + 1)).unzip();
            patches.push(NurbsSurface {
                du,
                dv: degree,
                nu: du + 1,
                nv: degree + 1,
                ku: knots(du),
                kv: knots(degree),
                poles,
                weights,
            });
        }
        let body = assemble(patches, None, first.closed)?;
        ctx.dispatches.insert(node.id, (device, n));
        Ok(Product::Sheet(body))
    }
}
