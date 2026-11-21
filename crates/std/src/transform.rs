//! Transformations and patterns (spec 03 `transform.*`). Patterns replicate the recorded tool
//! of each referenced feature (CATIA pattern semantics) and apply the feature's operation.

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::builders::{Profile, Seg2};
use agcad_geo::kernel::Solid;
use agcad_geo::layered::{Layer, LayeredSolid};
use agcad_geo::math::{Xform, V3};
use agcad_geo::region2d::BoolOp2;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};

/// 2D rigid transform of a layered tool within its own frame (rotation about the frame z axis
/// through `center`, then translation). Exact: profiles stay lines/arcs/circles.
fn transform_layered(l: &LayeredSolid, center: [f64; 2], angle: f64, t: [f64; 2]) -> LayeredSolid {
    let (s, c) = angle.sin_cos();
    let mp = |p: [f64; 2]| { let (x, y) = (p[0] - center[0], p[1] - center[1]); [center[0] + c * x - s * y + t[0], center[1] + s * x + c * y + t[1]] };
    let ms = |g: &Seg2| match *g {
        Seg2::Line { a, b } => Seg2::Line { a: mp(a), b: mp(b) },
        Seg2::Arc { c: cc, r, a0, a1 } => Seg2::Arc { c: mp(cc), r, a0: a0 + angle, a1: a1 + angle },
        Seg2::Circle { c: cc, r, ccw } => Seg2::Circle { c: mp(cc), r, ccw },
    };
    LayeredSolid { frame: l.frame, layers: l.layers.iter().map(|ly| Layer { z0: ly.z0, z1: ly.z1, regions: ly.regions.iter().map(|r| Profile { loops: r.loops.iter().map(|lp| lp.iter().map(ms).collect()).collect() }).collect() }).collect() }
}

pub struct TransformOp(pub &'static str);

impl FeatureImpl for TransformOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "transform.circular_pattern" | "transform.rectangular_pattern" | "transform.user_pattern" => {
                let mut placements: Vec<([f64; 2], f64, [f64; 2])> = Vec::new(); // (center, angle, translation)
                match self.0 {
                    "transform.circular_pattern" => {
                        let count = num(ctx, n, "count")? as usize;
                        let total = num_or(ctx, n, "angle", core::f64::consts::TAU)?;
                        let full = (total - core::f64::consts::TAU).abs() < 1e-9;
                        let step = if full { total / count as f64 } else { total / (count.max(2) - 1) as f64 };
                        let center = match arg(n, "center") { Some(v) => vec2(ctx, v)?, None => [0.0, 0.0] };
                        for k in 1..count { placements.push((center, step * k as f64, [0.0, 0.0])); }
                    }
                    "transform.rectangular_pattern" => {
                        let counts: Vec<usize> = list(req(n, "counts")?).iter().map(|v| ctx.number(v).map(|x| x as usize)).collect::<Result<_, _>>()?;
                        let spacing: Vec<f64> = list(req(n, "spacing")?).iter().map(|v| ctx.number(v)).collect::<Result<_, _>>()?;
                        let dirs: Vec<[f64; 2]> = match arg(n, "dirs") { Some(v) => list(v).iter().map(|d| vec2(ctx, d)).collect::<Result<_, _>>()?, None => vec![[1.0, 0.0], [0.0, 1.0]] };
                        let (n1, n2) = (counts.first().copied().unwrap_or(1), counts.get(1).copied().unwrap_or(1));
                        for i in 0..n1 { for j in 0..n2 {
                            if i == 0 && j == 0 { continue; }
                            let t = [dirs[0][0] * spacing[0] * i as f64 + dirs.get(1).map(|d| d[0]).unwrap_or(0.0) * spacing.get(1).copied().unwrap_or(0.0) * j as f64,
                                     dirs[0][1] * spacing[0] * i as f64 + dirs.get(1).map(|d| d[1]).unwrap_or(1.0) * spacing.get(1).copied().unwrap_or(0.0) * j as f64];
                            placements.push(([0.0, 0.0], 0.0, t));
                        }}
                    }
                    _ => {
                        let pos: Vec<[f64; 2]> = list(req(n, "positions")?).iter().map(|v| vec2(ctx, v)).collect::<Result<_, _>>()?;
                        let origin = pos.first().copied().unwrap_or([0.0, 0.0]);
                        for p in pos.iter().skip(1) { placements.push(([0.0, 0.0], 0.0, [p[0] - origin[0], p[1] - origin[1]])); }
                    }
                }
                let mut body = ctx.bodies.get(&n.container).cloned().ok_or_else(|| FeatureError::new(AgStatus::InvalidBaseline, "pattern needs an existing body"))?;
                for item in list(req(n, "items")?) {
                    let fid: u32 = match item { agcad_ir::Value::Ref { id, .. } => id.trim_start_matches('f').parse().map_err(|_| bad("pattern item must reference a feature"))?, _ => return Err(bad("pattern item must reference a feature")) };
                    let (tool, op) = ctx.tools.get(&fid).cloned().ok_or_else(|| bad(format!("feature #{fid} has no tool to pattern (pattern pads/pockets/holes)")))?;
                    let Solid::Layered(lt) = tool else { return Err(FeatureError::new(AgStatus::TopologyFailed, "patterning non-prismatic tools requires the general boolean (P11)")) };
                    for (c, a, t) in &placements {
                        let inst = Solid::Layered(transform_layered(&lt, *c, *a, *t));
                        body = ctx.kernel.boolean(&body, &inst, op).map_err(|e| FeatureError::new(e.status, e.message))?;
                    }
                }
                Ok(Product::Solid(body))
            }
            "transform.translation" | "transform.rotation" | "transform.symmetry" | "transform.mirror" => {
                let target = match arg(n, "items") { Some(v) => solid(ctx, list(v).first().ok_or_else(|| bad("no items"))?)?, None => ctx.bodies.get(&n.container).cloned().ok_or_else(|| bad("nothing to transform"))? };
                let x = match self.0 {
                    "transform.translation" => { let d = vec3(ctx, req(n, "dir")?)?.normalized(); Xform::translation(d * num(ctx, n, "d")?) }
                    "transform.rotation" => Xform::rotation(V3::ZERO, vec3(ctx, req(n, "axis")?)?, num(ctx, n, "angle")?),
                    _ => Xform::mirror(V3::ZERO, vec3(ctx, req(n, if self.0 == "transform.mirror" { "plane" } else { "ref" })?)?),
                };
                let moved = Solid::Brep(target.brep().transformed(&x));
                if self.0 == "transform.mirror" {
                    let b = ctx.kernel.boolean(&target, &moved, BoolOp2::Union).unwrap_or(moved);
                    return Ok(Product::Solid(b));
                }
                Ok(Product::Solid(moved))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}
