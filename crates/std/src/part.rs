//! Part Design features (spec 03 `part.*`): pad, pocket, shaft, groove, hole, primitive,
//! implicit_body, lattice. Pads/pockets/holes in one direction stay on the exact layered path.

use crate::sketch::regions_of;
use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::builders::{self, Profile};
use agcad_geo::kernel::Solid;
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::{v3, Aabb, Frame, V3};
use agcad_geo::region2d::BoolOp2;
use agcad_ir::program::Value;
use agcad_rtc::expr::{Bin, Dim, ExprGraph, Un};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, ImplicitBody, Product};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Extent of a pad/pocket: `limits` = number (dimension), or map {type: dimension|mirrored|
/// two_sided, d, d2}, optional `dir: "reverse"`.
fn limits(ctx: &BuildContext<'_>, n: &FeatureNode) -> Result<(f64, f64), FeatureError> {
    let lim = req(n, "limits")?;
    let reverse = matches!(text(n, "dir"), Some("reverse"));
    let (mut z0, mut z1) = match lim {
        Value::Map(m) => {
            let ty = match m.get("type").or(m.get("$ctor")) { Some(Value::String(s)) => s.as_str(), _ => "dimension" };
            let first = m.get("d").or(m.get("args").and_then(|a| list(a).first())).ok_or_else(|| bad("limits needs a length"))?;
            let d = ctx.number(first)?;
            match ty {
                "mirrored" | "Limits::mirrored" => (-d / 2.0, d / 2.0),
                "two_sided" | "Limits::two_sided" => { let d2 = m.get("d2").map(|v| ctx.number(v)).transpose()?.unwrap_or(d); (-d2, d) }
                _ => (0.0, d),
            }
        }
        v => (0.0, ctx.number(v)?),
    };
    if reverse { let t = z0; z0 = -z1; z1 = -t; }
    if !(z1 > z0) { return Err(bad("extrusion length must be positive").suggest("use dir: \"reverse\" to extrude the other way")); }
    Ok((z0, z1))
}

fn combine(ctx: &mut BuildContext<'_>, n: &FeatureNode, tool: Solid, op: BoolOp2) -> Result<Product, FeatureError> {
    ctx.tools.insert(n.id, (tool.clone(), op));
    let result = match (ctx.bodies.get(&n.container), op) {
        (None, BoolOp2::Union) => tool,
        (None, _) => return Err(FeatureError::new(AgStatus::InvalidBaseline, format!("{}: no body to remove material from", n.op)).suggest("add a pad (or primitive) first")),
        (Some(body), _) => ctx.kernel.boolean(body, &tool, op).map_err(|e| FeatureError::new(e.status, e.message))?,
    };
    Ok(Product::Solid(result))
}

pub struct PartOp(pub &'static str);

impl FeatureImpl for PartOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "part.pad" | "part.pocket" => {
                let r = regions_of(ctx, req(n, "profile")?)?;
                let (z0, z1) = limits(ctx, n)?;
                let mut tool: Option<LayeredSolid> = None;
                for reg in &r.regions {
                    let l = LayeredSolid::extrusion(r.frame, reg, z0, z1);
                    tool = Some(match tool { None => l, Some(t) => t.boolean(&l, BoolOp2::Union) });
                }
                let tool = tool.ok_or_else(|| bad("profile has no regions"))?;
                combine(ctx, n, Solid::Layered(tool), if self.0 == "part.pad" { BoolOp2::Union } else { BoolOp2::Difference })
            }
            "part.shaft" | "part.groove" => {
                let r = regions_of(ctx, req(n, "profile")?)?;
                let reg = r.regions.first().ok_or_else(|| bad("profile has no regions"))?;
                // Profile coordinates in the sketch plane: x = radius, y = axial (revolve about sketch y).
                let axis_frame = Frame { origin: r.frame.origin, x: r.frame.x, y: -r.frame.z, z: r.frame.y };
                let b = builders::revolve(reg, &axis_frame);
                combine(ctx, n, Solid::Brep(b), if self.0 == "part.shaft" { BoolOp2::Union } else { BoolOp2::Difference })
            }
            "part.primitive" => {
                let kind = text(n, "kind").unwrap_or("box");
                let p: Vec<f64> = list(req(n, "p")?).iter().map(|v| ctx.number(v)).collect::<Result<_, _>>()?;
                let f = match arg(n, "axis") { Some(v) => frame(ctx, v).unwrap_or(Frame::WORLD), None => Frame::WORLD };
                let need = |k: usize| if p.len() >= k { Ok(()) } else { Err(bad(format!("{kind} needs {k} parameters"))) };
                let solid = match kind {
                    "box" => { need(3)?; Solid::Layered(LayeredSolid::extrusion(f, &Profile::rectangle(-p[0] / 2.0, -p[1] / 2.0, p[0] / 2.0, p[1] / 2.0), 0.0, p[2])) }
                    "cylinder" => { need(2)?; Solid::Layered(LayeredSolid::extrusion(f, &Profile::circle(0.0, 0.0, p[0]), 0.0, p[1])) }
                    "cone" => { need(3)?; Solid::Brep(builders::cone(&f, p[0], p[1], p[2])) }
                    "sphere" => { need(1)?; Solid::Brep(builders::sphere(&f, p[0])) }
                    "torus" => { need(2)?; Solid::Brep(builders::torus(&f, p[0], p[1])) }
                    other => return Err(bad(format!("unknown primitive '{other}'"))),
                };
                combine(ctx, n, solid, BoolOp2::Union)
            }
            "part.hole" => {
                let body = ctx.bodies.get(&n.container).cloned().ok_or_else(|| FeatureError::new(AgStatus::InvalidBaseline, "hole needs an existing body"))?;
                let Solid::Layered(l) = &body else { return Err(FeatureError::new(AgStatus::TopologyFailed, "hole on non-prismatic body requires the general boolean (P11)")) };
                let spec = match req(n, "spec")? { Value::Map(m) => m.clone(), _ => BTreeMap::new() };
                let d = spec.get("d").or(spec.get("args").and_then(|a| list(a).first())).map(|v| ctx.number(v)).transpose()?.ok_or_else(|| bad("hole spec needs a diameter 'd'"))?;
                let zmax = l.layers.iter().map(|x| x.z1).fold(f64::NEG_INFINITY, f64::max);
                let zmin = l.layers.iter().map(|x| x.z0).fold(f64::INFINITY, f64::min);
                let through = spec.get("through").is_some() || matches!(spec.get("depth"), Some(Value::String(s)) if s == "through");
                let depth = if through { zmax - zmin + 2.0 } else { spec.get("depth").map(|v| ctx.number(v)).transpose()?.unwrap_or(zmax - zmin + 2.0) };
                let mut tool: Option<LayeredSolid> = None;
                for p in list(req(n, "positions")?) {
                    let c = vec2(ctx, p)?;
                    let mut prof = Profile::circle(c[0], c[1], d / 2.0);
                    if let Some(cb) = spec.get("counterbore_d") {
                        let cbd = ctx.number(cb)?;
                        let cbh = spec.get("counterbore_depth").map(|v| ctx.number(v)).transpose()?.unwrap_or(d * 0.5);
                        let cbt = LayeredSolid::extrusion(l.frame, &Profile::circle(c[0], c[1], cbd / 2.0), zmax - cbh, zmax + 1.0);
                        tool = Some(match tool { None => cbt, Some(t) => t.boolean(&cbt, BoolOp2::Union) });
                    }
                    prof = prof.normalize();
                    let t = LayeredSolid::extrusion(l.frame, &prof, zmax - depth, zmax + 1.0);
                    tool = Some(match tool { None => t, Some(x) => x.boolean(&t, BoolOp2::Union) });
                }
                combine(ctx, n, Solid::Layered(tool.ok_or_else(|| bad("hole needs at least one position"))?), BoolOp2::Difference)
            }
            "part.edge_fillet" | "part.chamfer" => {
                let body = ctx.bodies.get(&n.container).cloned().ok_or_else(|| FeatureError::new(AgStatus::InvalidBaseline, format!("{} needs an existing body", n.op)))?;
                let Solid::Layered(l) = &body else { return Err(FeatureError::new(AgStatus::TopologyFailed, "fillets on non-prismatic bodies require the general rolling-ball path (P11 general kernel)")) };
                let size = if self.0 == "part.edge_fillet" { num(ctx, n, "r")? } else { num(ctx, n, "a")? };
                let brep = l.to_brep();
                // Resolve edge names (or "all") to 2D corner points of vertical edges.
                let mut at: Vec<[f64; 2]> = Vec::new();
                let edges = list(req(n, "edges")?);
                // vertical edges given by a 2D point they pass through (sketch-plane coordinates)
                for e in edges.iter() {
                    if let Value::List(c) = e { if c.len() == 2 { at.push([ctx.number(&c[0])?, ctx.number(&c[1])?]); } }
                }
                let names: Vec<String> = edges.iter().filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None }).collect();
                let all = names.iter().any(|s| s == "all" || s == "all_vertical");
                if names.is_empty() && at.is_empty() { return Err(bad("edges: names, \"all\" or [x, y] corner points")); }
                if !all {
                    for name in &names {
                        let short = name.rsplit('/').next().unwrap_or(name);
                        let e = brep.edges.iter().find(|e| e.name == short).ok_or_else(|| FeatureError::new(AgStatus::StaleReference, format!("edge '{name}' not found")).with_elements(vec![name.clone()]))?;
                        match &brep.curves[e.curve as usize] {
                            agcad_geo::curve::Curve::Line { origin, dir } if dir.normalized().cross(l.frame.z).norm() < 1e-9 => {
                                let q = l.frame.to_local(*origin);
                                at.push([q.x, q.y]);
                            }
                            _ => return Err(FeatureError::new(AgStatus::TopologyFailed, format!("edge '{name}' is not a vertical edge; horizontal-edge fillets need the general path")).with_elements(vec![name.clone()])),
                        }
                    }
                }
                let corner = if self.0 == "part.edge_fillet" { agcad_geo::region2d::Corner::Fillet(size) } else { agcad_geo::region2d::Corner::Chamfer(size) };
                match l.treat_vertical_edges(&at, corner) {
                    Ok(r) => Ok(Product::Solid(Solid::Layered(r))),
                    Err((_, p, max)) => Err(FeatureError::new(AgStatus::GeometryCheckFailed, format!("{} {} = {size} infeasible at corner ({:.3}, {:.3}): adjacent edges allow at most ≈ {max:.3}", n.op, if self.0 == "part.edge_fillet" { "r" } else { "a" }, p[0], p[1])).with_elements(vec![format!("{}/corner({:.3},{:.3})", n.feature_ref(), p[0], p[1])]).suggest(format!("use ≤ {max:.3}, or part.variable_fillet"))),
                }
            }
            "part.implicit_body" => {
                let src = match req(n, "source")? { Value::String(s) => s.clone(), _ => return Err(bad("implicit_body source must be AgLang text")) };
                let m = agcad_rtc::compile_module(&src, agcad_ir::Catalog::global()).map_err(|d| FeatureError::new(AgStatus::CompileFailed, d.iter().map(|x| x.render("implicit_body", &src)).collect::<Vec<_>>().join("\n")))?;
                let item = m.items.iter().find(|i| matches!(i.kind, agcad_rtc::ItemKind::Implicit { .. })).ok_or_else(|| bad("source contains no #[implicit] function"))?;
                let lip = match item.kind { agcad_rtc::ItemKind::Implicit { lipschitz, .. } => lipschitz, _ => 1.0 };
                let params: Vec<f64> = item.params.iter().map(|p| ctx.params.get(&p.name).copied().or(p.default).unwrap_or(0.0)).collect();
                let bounds = bounds_arg(ctx, n)?;
                Ok(Product::Implicit(ImplicitBody { graph: Arc::new(item.expr.clone().unwrap()), params, bounds, lipschitz: lip }))
            }
            "part.lattice" => {
                let cell = text(n, "cell").unwrap_or("gyroid");
                let size = num(ctx, n, "size")?;
                let wall = num_or(ctx, n, "wall", size * 0.1)?;
                let region = ctx.product(req(n, "region")?)?.clone();
                let (bounds, region_sdf) = match &region {
                    Product::Implicit(ib) => (ib.bounds, Some(ib.clone())),
                    Product::Solid(s) => (s.brep().bbox(), None),
                    _ => return Err(bad("lattice region must be a body or implicit body")),
                };
                let lat = tpms(cell, size, wall)?;
                let body = match region_sdf {
                    Some(r) => crate::boolean::compose(&r, &ImplicitBody { graph: Arc::new(lat), params: vec![], bounds, lipschitz: 1.5 }, Bin::Max, 0.0),
                    None => crate::boolean::compose(&box_sdf(&bounds), &ImplicitBody { graph: Arc::new(lat), params: vec![], bounds, lipschitz: 1.5 }, Bin::Max, 0.0),
                };
                Ok(Product::Implicit(body))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}

fn bounds_arg(ctx: &BuildContext<'_>, n: &FeatureNode) -> Result<Aabb, FeatureError> {
    match arg(n, "bounds") {
        Some(Value::List(l)) if l.len() == 2 => Ok(Aabb { min: vec3(ctx, &l[0])?, max: vec3(ctx, &l[1])? }),
        _ => Ok(Aabb { min: v3(-100.0, -100.0, -100.0), max: v3(100.0, 100.0, 100.0) }),
    }
}

/// Triply periodic minimal surface shell of thickness `wall` (conservative distance estimate).
pub fn tpms(cell: &str, size: f64, wall: f64) -> Result<ExprGraph, FeatureError> {
    let mut g = ExprGraph::default();
    let x = g.input("p.x", Dim::LENGTH);
    let y = g.input("p.y", Dim::LENGTH);
    let z = g.input("p.z", Dim::LENGTH);
    let k = g.konst(core::f64::consts::TAU / size, Dim::LENGTH.pow(-1));
    let ok = |r: Result<u32, agcad_rtc::expr::DimError>| r.map_err(|e| bad(e.message));
    let (kx, ky, kz) = (ok(g.bin(Bin::Mul, k, x))?, ok(g.bin(Bin::Mul, k, y))?, ok(g.bin(Bin::Mul, k, z))?);
    let s = |g: &mut ExprGraph, a| g.un(Un::Sin, a).unwrap();
    let c = |g: &mut ExprGraph, a| g.un(Un::Cos, a).unwrap();
    let f = match cell {
        "gyroid" => {
            let (sx, sy, sz, cx, cy, cz) = (s(&mut g, kx), s(&mut g, ky), s(&mut g, kz), c(&mut g, kx), c(&mut g, ky), c(&mut g, kz));
            let a = g.bin(Bin::Mul, sx, cy).unwrap(); let b = g.bin(Bin::Mul, sy, cz).unwrap(); let d = g.bin(Bin::Mul, sz, cx).unwrap();
            let t = g.bin(Bin::Add, a, b).unwrap(); g.bin(Bin::Add, t, d).unwrap()
        }
        "schwarz_p" => { let (cx, cy, cz) = (c(&mut g, kx), c(&mut g, ky), c(&mut g, kz)); let t = g.bin(Bin::Add, cx, cy).unwrap(); g.bin(Bin::Add, t, cz).unwrap() }
        "diamond" => {
            let (sx, sy, sz, cx, cy, cz) = (s(&mut g, kx), s(&mut g, ky), s(&mut g, kz), c(&mut g, kx), c(&mut g, ky), c(&mut g, kz));
            let a = { let t = g.bin(Bin::Mul, sx, sy).unwrap(); g.bin(Bin::Mul, t, sz).unwrap() };
            let b = { let t = g.bin(Bin::Mul, sx, cy).unwrap(); g.bin(Bin::Mul, t, cz).unwrap() };
            let cc = { let t = g.bin(Bin::Mul, cx, sy).unwrap(); g.bin(Bin::Mul, t, cz).unwrap() };
            let d = { let t = g.bin(Bin::Mul, cx, cy).unwrap(); g.bin(Bin::Mul, t, sz).unwrap() };
            let t1 = g.bin(Bin::Add, a, b).unwrap(); let t2 = g.bin(Bin::Add, t1, cc).unwrap(); g.bin(Bin::Add, t2, d).unwrap()
        }
        other => return Err(bad(format!("unknown lattice cell '{other}' (gyroid, schwarz_p, diamond)"))),
    };
    let af = g.un(Un::Abs, f).unwrap();
    let dist = ok(g.bin(Bin::Div, af, k))?;
    let half = g.konst(wall / 2.0, Dim::LENGTH);
    let out = ok(g.bin(Bin::Sub, dist, half))?;
    g.outputs = vec![out];
    Ok(g)
}

/// Exact box SDF over an AABB (implicit view of a solid's bounding region).
pub fn box_sdf(b: &Aabb) -> ImplicitBody {
    let mut g = ExprGraph::default();
    let p = [g.input("p.x", Dim::LENGTH), g.input("p.y", Dim::LENGTH), g.input("p.z", Dim::LENGTH)];
    let c = [(b.min.x + b.max.x) / 2.0, (b.min.y + b.max.y) / 2.0, (b.min.z + b.max.z) / 2.0];
    let h = [(b.max.x - b.min.x) / 2.0, (b.max.y - b.min.y) / 2.0, (b.max.z - b.min.z) / 2.0];
    let mut acc = None;
    for i in 0..3 {
        let ci = g.konst(c[i], Dim::LENGTH);
        let hi = g.konst(h[i], Dim::LENGTH);
        let d = g.bin(Bin::Sub, p[i], ci).unwrap();
        let a = g.un(Un::Abs, d).unwrap();
        let q = g.bin(Bin::Sub, a, hi).unwrap();
        acc = Some(match acc { None => q, Some(x) => g.bin(Bin::Max, x, q).unwrap() });
    }
    g.outputs = vec![acc.unwrap()];
    ImplicitBody { graph: Arc::new(g), params: vec![], bounds: *b, lipschitz: 1.0 }
}

#[allow(dead_code)]
fn unused(_: V3) {}
