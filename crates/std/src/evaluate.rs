//! Evaluations (spec 14): measures, inertia, bounding box, validity check, thickness, DFM
//! analyses (overhang, draft, internal radius), sampling, clash. Every result carries an
//! evidence level (`exact` / `bounded` / `sampled` / `unknown`; AG-02-001).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::kernel::Solid;
use agcad_geo::math::V3;
use agcad_geo::surface::Surface;
use agcad_geo::tessellate::{Mesh, TessOptions};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use serde_json::json;

fn mesh_of(ctx: &BuildContext<'_>, s: &Solid, sag: f64) -> Mesh { ctx.kernel.tessellate(s, &TessOptions { sag, ..Default::default() }) }

/// Ray/triangle intersection (Möller–Trumbore); returns t > eps.
fn ray_tri(o: V3, d: V3, a: V3, b: V3, c: V3) -> Option<f64> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 { return None; }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) { return None; }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = e2.dot(q) * inv;
    if t > 1e-9 { Some(t) } else { None }
}

pub struct EvalOp(pub &'static str);

impl FeatureImpl for EvalOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let first_solid = |ctx: &BuildContext<'_>| -> Result<Solid, FeatureError> {
            let v = arg(n, "items").or(arg(n, "body")).or(arg(n, "item")).ok_or_else(|| bad("no body input"))?;
            solid(ctx, list(v).first().ok_or_else(|| bad("empty items"))?)
        };
        match self.0 {
            "evaluate.measure_inertia" => {
                let s = first_solid(ctx)?;
                let density = num_or(ctx, n, "density", 7.85e-6)?; // kg/mm³ (steel) default
                let m = mesh_of(ctx, &s, num_or(ctx, n, "sag", 1e-3)?);
                let (vol, area, c) = m.mass_properties();
                Ok(Product::Json(json!({ "volume": vol, "area": area, "mass": vol * density, "centroid": [c.x, c.y, c.z], "evidence": "bounded", "bound": "tessellation sag" })))
            }
            "evaluate.bounding_box" => {
                let s = first_solid(ctx)?;
                let b = s.brep().bbox();
                let e = b.extent();
                Ok(Product::Json(json!({ "min": [b.min.x, b.min.y, b.min.z], "max": [b.max.x, b.max.y, b.max.z], "extent": [e.x, e.y, e.z], "max_extent": e.x.max(e.y).max(e.z), "evidence": "bounded" })))
            }
            "evaluate.check" => {
                let s = first_solid(ctx)?;
                let r = ctx.kernel.check(&s);
                let m = mesh_of(ctx, &s, 0.01);
                Ok(Product::Json(json!({ "valid": r.valid && m.is_watertight(), "topology_valid": r.valid, "watertight": m.is_watertight(), "euler": r.euler, "problems": r.problems, "evidence": "exact" })))
            }
            "evaluate.thickness_analysis" => {
                // Inward rays from triangle centroids against the mesh; min hit distance.
                let s = first_solid(ctx)?;
                let m = mesh_of(ctx, &s, num_or(ctx, n, "sag", 0.05)?);
                let mut tmin = f64::INFINITY;
                let p = |i: u32| m.positions[i as usize];
                let stride = (m.triangles.len() / 2000).max(1);
                for t in m.triangles.iter().step_by(stride) {
                    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                    let nrm = (b - a).cross(c - a).normalized();
                    let o = (a + b + c) * (1.0 / 3.0) - nrm * 1e-7;
                    for u in &m.triangles {
                        if let Some(d) = ray_tri(o, -nrm, p(u[0]), p(u[1]), p(u[2])) { tmin = tmin.min(d); }
                    }
                }
                let min_req = num_or(ctx, n, "min", 0.0)?;
                Ok(Product::Json(json!({ "min": tmin, "passes": tmin >= min_req, "evidence": "sampled", "samples": m.triangles.len() / stride })))
            }
            "evaluate.overhang_analysis" => {
                let s = first_solid(ctx)?;
                let build = vec3(ctx, req(n, "build_dir")?)?.normalized();
                let max_angle = num(ctx, n, "max_angle")?;
                let m = mesh_of(ctx, &s, 0.05);
                let (mut area, mut total) = (0.0, 0.0);
                let zmin = m.positions.iter().map(|q| q.dot(build)).fold(f64::INFINITY, f64::min);
                for t in &m.triangles {
                    let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
                    let cr = (b - a).cross(c - a);
                    let ar = 0.5 * cr.norm();
                    total += ar;
                    let nz = cr.normalized().dot(build);
                    let on_plate = ((a + b + c) * (1.0 / 3.0)).dot(build) - zmin < 1e-6;
                    // downward-facing beyond allowed angle from vertical
                    if !on_plate && nz < 0.0 && (-nz).acos() < core::f64::consts::FRAC_PI_2 - max_angle { area += ar; }
                }
                Ok(Product::Json(json!({ "unsupported_area": area, "total_area": total, "evidence": "sampled" })))
            }
            "evaluate.draft_analysis" => {
                let s = first_solid(ctx)?.brep();
                let pull = vec3(ctx, req(n, "pull")?)?.normalized();
                let min = num_or(ctx, n, "min", 0.0)?;
                let mut bad_faces = Vec::new();
                for f in &s.faces {
                    if let Surface::Plane { frame } = &s.surfaces[f.surface as usize] {
                        let nrm = if f.same_sense { frame.z } else { -frame.z };
                        let draft = core::f64::consts::FRAC_PI_2 - nrm.dot(pull).abs().clamp(-1.0, 1.0).acos();
                        if draft.abs() < min && nrm.dot(pull).abs() < 0.999_999 { bad_faces.push(json!({ "face": f.name, "draft": draft })); }
                    } else if let Surface::Cylinder { frame, .. } = &s.surfaces[f.surface as usize] {
                        if frame.z.cross(pull).norm() < 1e-9 && min > 0.0 { bad_faces.push(json!({ "face": f.name, "draft": 0.0 })); }
                    }
                }
                Ok(Product::Json(json!({ "below_min": bad_faces, "evidence": "exact" })))
            }
            "evaluate.internal_radius_analysis" => {
                let s = first_solid(ctx)?.brep();
                let min_r = num(ctx, n, "min_r")?;
                let tight: Vec<_> = s.faces.iter().filter_map(|f| match &s.surfaces[f.surface as usize] {
                    Surface::Cylinder { r, .. } if !f.same_sense && *r < min_r => Some(json!({ "face": f.name, "radius": r })),
                    _ => None,
                }).collect();
                Ok(Product::Json(json!({ "too_small": tight, "evidence": "exact" })))
            }
            "evaluate.sample" => {
                let body = ctx.product(req(n, "body")?)?.clone();
                let pts: Vec<V3> = list(req(n, "pts")?).iter().map(|v| vec3(ctx, v)).collect::<Result<_, _>>()?;
                let vals: Vec<f64> = match &body {
                    Product::Implicit(ib) => pts.iter().map(|p| ib.eval(*p)).collect(),
                    _ => return Err(bad("sample currently supports implicit bodies")),
                };
                Ok(Product::Json(json!({ "values": vals, "evidence": "exact" })))
            }
            "evaluate.clash" => {
                let items: Vec<Solid> = list(req(n, "items")?).iter().map(|v| solid(ctx, v)).collect::<Result<_, _>>()?;
                let mut pairs = Vec::new();
                for i in 0..items.len() { for j in i + 1..items.len() {
                    let (a, b) = (items[i].brep().bbox(), items[j].brep().bbox());
                    if a.overlaps(&b, 0.0) {
                        let r = ctx.kernel.boolean(&items[i], &items[j], agcad_geo::region2d::BoolOp2::Intersection);
                        let vol = match r { Ok(s) => mesh_of(ctx, &s, 0.05).mass_properties().0, Err(_) => f64::NAN };
                        pairs.push(json!({ "a": i, "b": j, "volume": vol, "evidence": if vol.is_nan() { "unknown" } else { "bounded" } }));
                    }
                }}
                Ok(Product::Json(json!({ "clashes": pairs })))
            }
            "evaluate.measure_item" => {
                let s = first_solid(ctx)?.brep();
                Ok(Product::Json(json!({ "faces": s.faces.len(), "edges": s.edges.len(), "vertices": s.vertices.len(), "evidence": "exact" })))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}

/// Queries (spec 03 `query.*`) over a solid's faces; results are lists of stable face names.
pub struct QueryOp(pub &'static str);

impl FeatureImpl for QueryOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let scope = arg(n, "scope").or(arg(n, "feature")).or(arg(n, "sel")).ok_or_else(|| bad("query needs a scope"))?;
        let s = solid(ctx, scope)?.brep();
        let names: Vec<String> = match self.0 {
            "query.parallel_to" => {
                let d = vec3(ctx, req(n, "dir")?)?.normalized();
                let tol = num_or(ctx, n, "tol", 1e-6)?;
                s.faces.iter().filter(|f| matches!(&s.surfaces[f.surface as usize], Surface::Plane { frame } if (if f.same_sense { frame.z } else { -frame.z }).dot(d) >= tol.cos())).map(|f| f.name.clone()).collect()
            }
            "query.element_type" | "query.geometry" => {
                let kind = text(n, "type").unwrap_or("any");
                s.faces.iter().filter(|f| kind == "any" || s.surfaces[f.surface as usize].kind_name() == kind).map(|f| f.name.clone()).collect()
            }
            "query.created_by" | "query.named" => {
                let role = text(n, "role").unwrap_or("");
                s.faces.iter().filter(|f| role.is_empty() || f.name == role || f.name.ends_with(&format!("/{role}"))).map(|f| f.name.clone()).collect()
            }
            _ => return Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        };
        Ok(Product::Json(json!({ "elements": names })))
    }
}

/// Strongest evidence each implemented evaluator can produce (spec 14 evidence levels). Used
/// to reject requirements that can never be satisfied with the available evaluators.
pub const MAX_EVIDENCE: &[(&str, &str)] = &[
    ("evaluate.measure_inertia", "bounded"),
    ("evaluate.bounding_box", "bounded"),
    ("evaluate.check", "exact"),
    ("evaluate.thickness_analysis", "sampled"),
    ("evaluate.overhang_analysis", "sampled"),
    ("evaluate.draft_analysis", "exact"),
    ("evaluate.internal_radius_analysis", "exact"),
    ("evaluate.sample", "exact"),
    ("evaluate.clash", "exact"),
    ("evaluate.measure_item", "exact"),
    ("evaluate.stiffness", "sampled"),
    ("evaluate.modal", "sampled"),
];
