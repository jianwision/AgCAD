use crate::{
    push_pull::polygonal_prism,
    util::{bad, vec3},
};
use agcad_driver::AgStatus;
use agcad_geo::{
    Brep,
    block_corner::corners,
    brep::DEFAULT_LINEAR_TOL as TOL,
    builders::{Profile, Seg2},
    curve::Curve,
    kernel::Solid,
    math::V3,
    surface::Surface,
};
use agcad_ir::Value;
use agcad_runtime::rebuild::{BuildContext, FeatureError, Product};
use std::collections::BTreeMap;

pub(crate) fn edit(
    ctx: &BuildContext<'_>,
    body: &Brep,
    pick: &BTreeMap<String, Value>,
    size: f64,
    round: bool,
) -> Result<Product, FeatureError> {
    let Some(Value::List(ends)) = pick.get("endpoints_mm") else {
        return Err(bad("Selected edge needs both endpoints in mm"));
    };
    let [a, b] = ends.as_slice() else {
        return Err(bad("Select exactly one edge"));
    };
    let (a, b) = (vec3(ctx, a)?, vec3(ctx, b)?);
    Ok(Product::Solid(Solid::Layered(treat_block(
        body, a, b, size, round,
    )?)))
}

fn treat_block(
    body: &Brep,
    a: V3,
    b: V3,
    size: f64,
    round: bool,
) -> Result<agcad_geo::layered::LayeredSolid, FeatureError> {
    let mut matches = body.edges.iter().filter(|e| {
        let (p, q) = (
            body.vertices[e.v0 as usize].p,
            body.vertices[e.v1 as usize].p,
        );
        (p.dist(a) <= TOL && q.dist(b) <= TOL) || (p.dist(b) <= TOL && q.dist(a) <= TOL)
    });
    let stale = || {
        FeatureError::new(
            AgStatus::StaleReference,
            "The selected edge changed or is ambiguous. Pick it again.",
        )
    };
    let edge = matches.next().ok_or_else(stale)?;
    if matches.next().is_some() {
        return Err(stale());
    }
    if !matches!(body.curves[edge.curve as usize], Curve::Line { .. }) {
        return Err(bad("Select a straight edge. Curved edges are unsupported."));
    }
    let unsupported = || {
        FeatureError::new(
            AgStatus::UnsupportedOp,
            "Single-edge edits currently require a rectangular block with six planar faces.",
        )
    };
    if body.faces.len() != 6 || body.edges.len() != 12 || body.vertices.len() != 8 {
        return Err(unsupported());
    }
    let direction = (b - a).normalized();
    let (face, mut prism) = body
        .faces
        .iter()
        .enumerate()
        .find_map(|(i, f)| {
            let Surface::Plane { frame } = body.surfaces[f.surface as usize] else {
                return None;
            };
            if frame.z.cross(direction).norm() > 1e-8 {
                return None;
            }
            polygonal_prism(body, i).map(|prism| (f, prism))
        })
        .ok_or_else(unsupported)?;
    let boundary = &body.loops[face.loops[0] as usize].coedges;
    if boundary.len() != 4 {
        return Err(unsupported());
    }
    let verts: Vec<_> = boundary.iter().map(|&c| body.coedge_start(c)).collect();
    let k = verts
        .iter()
        .position(|&v| v == edge.v0 || v == edge.v1)
        .ok_or_else(stale)?;
    let world = |i: usize| {
        let V3 { x, y, z } = body.vertices[verts[i % 4] as usize].p;
        [x, y, z]
    };
    let jobs = (0..4)
        .map(|i| [world(i + 3), world(i), world(i + 1)])
        .collect::<Vec<_>>();
    let pts = corners(&jobs, prism.frame, size).map_err(|e| bad(e.to_string()))?;
    if pts.iter().any(|p| !p[10].is_finite() || p[10].abs() > 1e-8) {
        return Err(unsupported());
    }
    let [_, _, ax, ay, bx, by, cx, cy, lu, lv, _] = pts[k];
    let limit = lu.min(lv);
    if !size.is_finite() || size <= TOL || size >= limit - TOL {
        return Err(bad(format!(
            "Edge size must exceed {TOL} mm and be smaller than the adjacent face width {limit:.6} mm."
        )));
    }
    let mut segments = Vec::with_capacity(5);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let start = if i == k {
            [bx, by]
        } else {
            [pts[i][0], pts[i][1]]
        };
        let end = if j == k {
            [ax, ay]
        } else {
            [pts[j][0], pts[j][1]]
        };
        segments.push(Seg2::Line { a: start, b: end });
        if j == k {
            let treatment = if round {
                let a0 = (ay - cy).atan2(ax - cx);
                let turn = ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)).signum();
                Seg2::Arc {
                    c: [cx, cy],
                    r: size,
                    a0,
                    a1: a0 + turn * std::f64::consts::FRAC_PI_2,
                }
            } else {
                Seg2::Line {
                    a: [ax, ay],
                    b: [bx, by],
                }
            };
            segments.push(treatment);
        }
    }
    prism.layers[0].regions = vec![
        Profile {
            loops: vec![segments],
        }
        .normalize(),
    ];
    Ok(prism)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_geo::{
        builders::prism,
        math::{Frame, Xform, v3},
        step_read::read_step,
        step_write::try_write_step,
        tessellate::{TessOptions, try_tessellate},
    };

    #[test]
    fn all_block_edges_roundtrip() {
        let native = prism(
            &Profile::rectangle(-20., -15., 20., 15.),
            &Frame::WORLD,
            10.,
        );
        let (mut imported, _) = read_step(
            include_str!("../../viewer/tests/fixtures/plate.step"),
            &Default::default(),
        )
        .unwrap();
        let plate = imported.remove(0);
        let rotated = plate.transformed(&Xform::rotation(v3(19., 3., -11.), v3(2., 3., 1.), 0.42));
        for (body, volume) in [(&native, 12000.), (&plate, 11200.), (&rotated, 11200.)] {
            for e in &body.edges {
                let (a, b) = (
                    body.vertices[e.v0 as usize].p,
                    body.vertices[e.v1 as usize].p,
                );
                for (round, size, loss) in [(true, 2., 4. - std::f64::consts::PI), (false, 1., 0.5)]
                {
                    let target = volume - loss * a.dist(b);
                    let (result, dispatch) =
                        agcad_geo::accelerator::capture(|| treat_block(body, b, a, size, round));
                    let result = result.unwrap();
                    if std::env::var("AGCAD_GEOMETRY_BACKEND").as_deref() == Ok("cuda") {
                        assert!(dispatch.iter().any(|d|d.operation=="block_edge_corners" && d.backend=="cuda"),"{:?}",dispatch.iter().map(|d|(&d.backend,&d.error)).collect::<Vec<_>>());
                    }
                    assert!((result.volume() - target).abs() < 1e-6);
                    let solid = result.to_brep();
                    let (file, _) = try_write_step(&solid, &Default::default()).unwrap();
                    let (reopened, _) = read_step(&file, &Default::default()).unwrap();
                    assert_eq!(reopened.len(), 1);
                    for shape in [&solid, &reopened[0]] {
                        let check = shape.validate();
                        assert!(check.valid && check.closed, "{:?}", check.problems);
                        assert_eq!(shape.shells.len(), 1);
                        assert_eq!(shape.faces.len(), 7);
                        let cylinders = shape
                            .faces
                            .iter()
                            .filter_map(|f| match shape.surfaces[f.surface as usize] {
                                Surface::Cylinder { r, .. } => Some(r),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(cylinders.len(), usize::from(round));
                        if round {
                            assert!((cylinders[0] - size).abs() < 1e-8);
                        }
                    }
                    let mesh = try_tessellate(
                        &reopened[0],
                        &TessOptions {
                            sag: 0.0001,
                            angle: 0.025,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let measured = mesh.mass_properties().0.abs();
                    assert!(
                        (measured - target).abs() < 0.025,
                        "volume {measured} expected {target}, {a:?} to {b:?}, round {round}"
                    );
                }
                assert!(treat_block(body, a, b, 100., true).is_err());
            }
        }
    }
}
