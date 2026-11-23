//! Signed displacement of the end face of a prism.
use crate::util::{bad, list, num, req};
use agcad_geo::{
    Brep,
    builders::{Profile, Seg2},
    curve::Curve,
    kernel::Solid,
    layered::LayeredSolid,
    math::Frame,
    surface::Surface,
};
use agcad_ir::Value;
use agcad_runtime::{
    document::FeatureNode,
    rebuild::{BuildContext, FeatureError, FeatureImpl, Product},
};

/// Recognize a straight-sided prism along the chosen face normal. Each vertex of the
/// cap must connect directly to its translated partner on the opposite plane.
pub(crate) fn polygonal_prism(body: &Brep, index: usize) -> Option<LayeredSolid> {
    let face = body.faces.get(index)?;
    let Surface::Plane { frame } = body.surfaces[face.surface as usize] else {
        return None;
    };
    let [outer] = face.loops.as_slice() else {
        return None;
    };
    let boundary = &body.loops[*outer as usize].coedges;
    let n = boundary.len();
    if n < 3
        || body.faces.len() != n + 2
        || body.vertices.len() != n * 2
        || body.edges.len() != n * 3
        || body.shells.len() != 1
        || !body.shells[0].closed
    {
        return None;
    }
    if body
        .faces
        .iter()
        .any(|f| !matches!(body.surfaces[f.surface as usize], Surface::Plane { .. }))
        || body
            .edges
            .iter()
            .any(|e| !matches!(body.curves[e.curve as usize], Curve::Line { .. }))
    {
        return None;
    }
    let frame = if face.same_sense {
        frame
    } else {
        Frame {
            y: -frame.y,
            z: -frame.z,
            ..frame
        }
    };
    let points: Vec<_> = body.vertices.iter().map(|v| frame.to_local(v.p)).collect();
    let bottom = points.iter().map(|p| p.z).fold(0.0, f64::min);
    let tol = agcad_geo::brep::DEFAULT_LINEAR_TOL;
    if bottom >= -tol {
        return None;
    }
    let mut linked = vec![false; points.len()];
    for e in &body.edges {
        let (i, j) = (e.v0 as usize, e.v1 as usize);
        let (p, q) = (points[i], points[j]);
        if (p.z - q.z).abs() <= tol {
            continue;
        }
        if linked[i]
            || linked[j]
            || (p.z - q.z).abs() + tol < -bottom
            || p.z.max(q.z).abs() > tol
            || (p.x - q.x).hypot(p.y - q.y) > tol
        {
            return None;
        }
        linked[i] = true;
        linked[j] = true;
    }
    if !linked.into_iter().all(|v| v) {
        return None;
    }
    let segments = boundary
        .iter()
        .map(|&ce| {
            let p = points[body.coedge_start(ce) as usize];
            let q = points[body.coedge_end(ce) as usize];
            Seg2::Line {
                a: [p.x, p.y],
                b: [q.x, q.y],
            }
        })
        .collect();
    Some(LayeredSolid::extrusion(
        frame,
        &Profile {
            loops: vec![segments],
        },
        bottom,
        0.0,
    ))
}

fn cap(layered: &LayeredSolid, name: &str) -> Option<(usize, bool)> {
    let (idx, top) = match name {
        "cap_end" => (layered.layers.len().checked_sub(1)?, true),
        "cap_start" => (0, false),
        _ => return None,
    };
    (layered.layers.get(idx)?.regions.len() == 1).then_some((idx, top))
}

pub fn supported(body: &Brep, layered: Option<&LayeredSolid>, index: usize) -> bool {
    layered
        .and_then(|l| cap(l, &body.faces[index].name))
        .is_some()
        || polygonal_prism(body, index).is_some()
}

pub struct PushPull;
impl FeatureImpl for PushPull {
    fn op(&self) -> &'static str {
        "direct.push_pull"
    }
    fn execute(
        &self,
        ctx: &mut BuildContext<'_>,
        node: &FeatureNode,
    ) -> Result<Product, FeatureError> {
        let faces = list(req(node, "faces")?);
        if faces.iter().any(|v| matches!(v, Value::Map(_))) {
            let body = ctx
                .bodies
                .get(&node.container)
                .ok_or_else(|| bad("Select a block"))?
                .brep();
            let block = agcad_geo::block_edit::Block::recognize(&body).map_err(bad)?;
            let mut mask = 0;
            for f in faces {
                let Value::Map(m) = f else {
                    return Err(bad("Expected selected planar faces"));
                };
                let point = crate::util::vec3(
                    ctx,
                    m.get("plane_origin")
                        .ok_or_else(|| bad("Missing selected plane"))?,
                )?;
                let normal = crate::util::vec3(
                    ctx,
                    m.get("normal")
                        .ok_or_else(|| bad("Missing selected normal"))?,
                )?;
                mask |= 1 << block.plane(point, normal).map_err(bad)?;
            }
            return Ok(Product::Solid(Solid::Brep(
                block.offset(mask, num(ctx, node, "d")?).map_err(bad)?,
            )));
        }
        let [Value::Int(i)] = faces else {
            return Err(bad("Select one planar face by its index"));
        };
        let index = usize::try_from(*i).map_err(|_| bad("Invalid face index"))?;
        let d = num(ctx, node, "d")?;
        if !d.is_finite() || d == 0. {
            return Err(bad("Enter a nonzero signed displacement"));
        }
        let solid = ctx
            .bodies
            .get(&node.container)
            .ok_or_else(|| bad("Select an existing part"))?;
        let body = solid.brep();
        let face = body
            .faces
            .get(index)
            .ok_or_else(|| bad("The selected face has changed"))?;
        let native = match solid {
            Solid::Layered(l) => cap(l, &face.name).map(|(i, top)| (l.clone(), i, top)),
            _ => None,
        };
        let (mut result, i, top) = native
            .or_else(|| polygonal_prism(&body, index).map(|l| (l, 0, true)))
            .ok_or_else(|| bad("Select a supported planar prism end face"))?;
        let layer = &mut result.layers[i];
        if top {
            layer.z1 += d
        } else {
            layer.z0 -= d
        }
        if layer.z1 - layer.z0 <= agcad_geo::brep::DEFAULT_LINEAR_TOL {
            return Err(bad("That displacement consumes the entire end layer"));
        }
        Ok(Product::Solid(Solid::Layered(result)))
    }
}
