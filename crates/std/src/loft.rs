use crate::sketch::regions_of;
use crate::util::{bad, list, req};
use agcad_driver::AgStatus;
use agcad_geo::brep::{BodyKind, Brep, DEFAULT_LINEAR_TOL as TOL};
use agcad_geo::builders::Seg2;
use agcad_geo::curve::Curve;
use agcad_geo::kernel::Solid;
use agcad_geo::math::{v3, Frame, V3};
use agcad_geo::nurbs::NurbsSurface;
use agcad_geo::surface::Surface;
use agcad_ir::Value;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};

pub struct Loft(pub &'static str);
type UV = [f64; 2];

fn unsupported(message: impl Into<String>) -> FeatureError {
    FeatureError::new(AgStatus::UnsupportedOp, message)
}

fn polygon(
    ctx: &BuildContext<'_>,
    section: &Value,
    index: usize,
) -> Result<(Frame, Vec<UV>), FeatureError> {
    let framed = regions_of(ctx, section)
        .map_err(|e| FeatureError::new(e.status, format!("section {index}: {}", e.message)))?;
    let [region] = framed.regions.as_slice() else {
        return Err(unsupported(format!("section {index} requires one region")));
    };
    let [edges] = region.loops.as_slice() else {
        return Err(unsupported(format!("section {index} has interior loops")));
    };
    let mut points = Vec::with_capacity(edges.len());
    for edge in edges {
        let Seg2::Line { a, .. } = edge else {
            return Err(unsupported(format!(
                "section {index} contains a curved edge"
            )));
        };
        points.push(*a);
    }
    if points.len() < 3
        || region.area().abs() <= TOL * TOL
        || !points.iter().flatten().all(|p| p.is_finite())
    {
        return Err(bad(format!("section {index} is degenerate")));
    }
    Ok((framed.frame, points))
}

fn world(f: Frame, p: UV) -> V3 {
    f.to_world(v3(p[0], p[1], 0.0))
}

fn canonicalize(points: &mut [UV], frame: Frame, normal: V3) {
    if frame.z.dot(normal) < 0.0 {
        points.reverse();
    }
    let first = (0..points.len())
        .min_by(|&i, &j| {
            let a = world(frame, points[i]);
            let b = world(frame, points[j]);
            a.x.total_cmp(&b.x)
                .then(a.y.total_cmp(&b.y))
                .then(a.z.total_cmp(&b.z))
        })
        .unwrap();
    points.rotate_left(first);
}

fn cross(a: V3, b: V3, n: V3) -> f64 {
    a.cross(b).dot(n)
}

// Parallel section planes imply that intersecting side faces meet at equal loft height.
// Within a slice the edges are straight, with endpoints affine in that height. A first
// contact must occur when a moving endpoint lies on another moving edge.
fn check_contact(start: &[V3], end: &[V3], normal: V3) -> Result<(), FeatureError> {
    let count = start.len();
    let error = |i, j| {
        FeatureError::new(
            AgStatus::TopologyFailed,
            format!("degenerate or intersecting loft near section edges {i} and {j}"),
        )
    };
    for i in 0..count {
        let next = (i + 1) % count;
        let base = start[next] - start[i];
        let delta = (end[next] - end[i]) - base;
        let closest = if delta.dot(delta) > 0.0 {
            (-base.dot(delta) / delta.dot(delta)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if (base + delta * closest).norm() <= TOL {
            return Err(error(i, i));
        }
        for j in 0..count {
            if j == i || j == next {
                continue;
            }
            let p = start[j] - start[i];
            let dp = (end[j] - end[i]) - p;
            let (a, b, c) = (
                cross(delta, dp, normal),
                cross(base, dp, normal) + cross(delta, p, normal),
                cross(base, p, normal),
            );
            let mut times = vec![0.0, 1.0];
            if a != 0.0 {
                let disc = b * b - 4.0 * a * c;
                if disc >= 0.0 {
                    let q = -0.5 * (b + disc.sqrt().copysign(b));
                    times.push(q / a);
                    if q != 0.0 {
                        times.push(c / q);
                    }
                }
            } else if b != 0.0 {
                times.push(-c / b);
            }
            // Collinear throughout: entry into the segment requires coincident endpoints.
            for (v, d) in [(p, dp), (p - base, dp - delta)] {
                if d.dot(d) > 0.0 {
                    times.push(-v.dot(d) / d.dot(d));
                }
            }
            for t in times.into_iter().filter(|t| (0.0..=1.0).contains(t)) {
                let edge = base + delta * t;
                let point = p + dp * t;
                let along = (point.dot(edge) / edge.dot(edge)).clamp(0.0, 1.0);
                if (point - edge * along).norm() <= TOL {
                    return Err(error(i, j));
                }
            }
        }
    }
    // A self-crossing section can already have a proper intersection at the start,
    // without any endpoint touching an edge during the sweep.
    for i in 0..count {
        let k = (i + 1) % count;
        for j in i + 1..count {
            let l = (j + 1) % count;
            if j == k || l == i {
                continue;
            }
            let ab = start[k] - start[i];
            let cd = start[l] - start[j];
            if cross(ab, start[j] - start[i], normal) * cross(ab, start[l] - start[i], normal) < 0.0
                && cross(cd, start[i] - start[j], normal) * cross(cd, start[k] - start[j], normal)
                    < 0.0
            {
                return Err(error(i, j));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_intermediate_slices_for_collisions() {
        let ring = vec![
            v3(0., 0., 0.),
            v3(3., 0., 0.),
            v3(3., 2., 0.),
            v3(0., 2., 0.),
        ];
        let shifted: Vec<_> = ring.iter().map(|&p| p + v3(7., 1., 6.)).collect();
        assert!(check_contact(&ring, &shifted, V3::Z).is_ok());
        let mut folded = shifted;
        folded.rotate_left(2);
        assert!(check_contact(&ring, &folded, V3::Z).is_err());
        let crossing = [ring[0], ring[2], ring[1], ring[3]];
        assert!(check_contact(&crossing, &crossing, V3::Z).is_err());
    }
}

fn line(body: &mut Brep, a: u32, b: u32, name: String) -> u32 {
    let origin = body.vertices[a as usize].p;
    let dir = body.vertices[b as usize].p - origin;
    let curve = body.add_curve(Curve::Line { origin, dir });
    body.add_edge(curve, a, b, 0.0, 1.0, name)
}

fn body(patches: &[[V3; 4]], frames: [Frame; 2], normal: V3, solid: bool) -> Brep {
    let mut b = Brep::new(if solid {
        BodyKind::Solid
    } else {
        BodyKind::Sheet
    });
    let shell = b.add_shell(solid);
    let n = patches.len();
    let mut rails = Vec::with_capacity(n);
    let mut rings = [Vec::new(), Vec::new()];
    for section in 0..2 {
        let vertices: Vec<_> = patches.iter().map(|p| b.add_vertex(p[section])).collect();
        for i in 0..n {
            rings[section].push(line(
                &mut b,
                vertices[i],
                vertices[(i + 1) % n],
                format!("section.{section}.{i}"),
            ));
        }
    }
    for i in 0..n {
        rails.push(line(&mut b, i as u32, (n + i) as u32, format!("rail.{i}")));
    }
    for (i, p) in patches.iter().enumerate() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let surface = b.add_surface(Surface::Nurbs(NurbsSurface {
            du: 1,
            dv: 1,
            nu: 2,
            nv: 2,
            ku: knots.clone(),
            kv: knots,
            weights: vec![1.0; 4],
            poles: p.to_vec(),
        }));
        b.add_face(
            surface,
            true,
            shell,
            &[
                (rings[0][i], true),
                (rails[(i + 1) % n], true),
                (rings[1][i], false),
                (rails[i], false),
            ],
            &[],
            format!("side.{i}"),
            [0.0, 1.0, 0.0, 1.0],
        );
    }
    if solid {
        for (i, frame) in frames.into_iter().enumerate() {
            let surface = b.add_surface(Surface::Plane { frame });
            let mut edges: Vec<_> = rings[i].iter().map(|&edge| (edge, i == 1)).collect();
            if i == 0 {
                edges.reverse();
            }
            let mut uv = [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];
            for p in patches {
                let q = p[i] - frame.origin;
                let (x, y) = (q.dot(frame.x), q.dot(frame.y));
                uv = [uv[0].min(x), uv[1].max(x), uv[2].min(y), uv[3].max(y)];
            }
            b.add_face(
                surface,
                (frame.z.dot(normal) > 0.0) == (i == 1),
                shell,
                &edges,
                &[],
                if i == 0 { "cap_start" } else { "cap_end" },
                uv,
            );
        }
    }
    b
}

impl FeatureImpl for Loft {
    fn op(&self) -> &'static str {
        self.0
    }
    fn execute(
        &self,
        ctx: &mut BuildContext<'_>,
        node: &FeatureNode,
    ) -> Result<Product, FeatureError> {
        let refs = list(req(node, "sections")?);
        if refs.len() < 2 {
            return Err(bad("loft requires at least two sections"));
        }
        if !matches!(node.args.get("coupling"),Some(Value::String(s)) if s=="ratio") {
            return Err(unsupported("loft requires ratio coupling"));
        }
        let sections = refs
            .iter()
            .enumerate()
            .map(|(i, v)| crate::loft_sections::Section::read(ctx, v, i + 1))
            .collect::<Result<Vec<_>, _>>()?;
        let ruled = match node.args.get("ruled") {
            None => false,
            Some(Value::Bool(v)) => *v,
            _ => return Err(bad("ruled must be boolean")),
        };
        let fast = refs.len() == 2
            && sections[0].edges.len() == sections[1].edges.len()
            && sections
                .iter()
                .all(|s| s.edges.iter().all(|e| matches!(e, Seg2::Line { .. })))
            && node.args.get("guides").is_none_or(|v| list(v).is_empty());
        if !fast {
            let solid = self.0 == "part.multi_sections_solid";
            let (body, count, device) =
                crate::loft_network::build(ctx, sections, node.args.get("guides"), ruled, solid)?;
            ctx.dispatches.insert(node.id, (device, count));
            return if solid {
                crate::part::combine(
                    ctx,
                    node,
                    Solid::Brep(body),
                    agcad_geo::region2d::BoolOp2::Union,
                )
            } else {
                Ok(Product::Sheet(body))
            };
        }
        let (a, b) = (&refs[0], &refs[1]);
        let (f, mut a) = polygon(ctx, a, 1)?;
        let (g, mut b) = polygon(ctx, b, 2)?;
        if a.len() != b.len() {
            return Err(unsupported(
                "section 2 has a different edge count from section 1",
            ));
        }
        if f.z.cross(g.z).norm() > 1e-12 {
            return Err(unsupported("section 2 must be parallel to section 1"));
        }
        let height = (g.origin - f.origin).dot(f.z);
        if height.abs() <= TOL {
            return Err(bad("section 2 coincides with section 1"));
        }
        let normal = f.z * height.signum();
        canonicalize(&mut a, f, normal);
        canonicalize(&mut b, g, normal);
        let start: Vec<_> = a.iter().map(|&p| world(f, p)).collect();
        let mut end: Vec<_> = b.iter().map(|&p| world(g, p)).collect();
        let cost = |offset: usize| {
            start
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let d = *p - end[(i + offset) % end.len()];
                    d.dot(d)
                })
                .sum::<f64>()
        };
        let offset = (0..a.len())
            .map(|i| (i, cost(i)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        end.rotate_left(offset);
        b.rotate_left(offset);
        check_contact(&start, &end, normal)?;
        let (patches, device) = crate::loft_device::patches(&[f, g], &[a, b]).map_err(|e| {
            FeatureError::new(
                AgStatus::TopologyFailed,
                format!("loft device dispatch failed: {e}"),
            )
        })?;
        let solid = self.0 == "part.multi_sections_solid";
        let brep = body(&patches, [f, g], normal, solid);
        let check = brep.validate();
        if !check.valid {
            return Err(FeatureError::new(
                AgStatus::TopologyFailed,
                check.problems.join(", "),
            ));
        }
        ctx.dispatches.insert(node.id, (device, patches.len()));
        if solid {
            crate::part::combine(
                ctx,
                node,
                Solid::Brep(brep),
                agcad_geo::region2d::BoolOp2::Union,
            )
        } else {
            Ok(Product::Sheet(brep))
        }
    }
}
