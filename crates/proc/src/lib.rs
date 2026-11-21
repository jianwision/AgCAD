//! # agcad-proc — Houdini-style procedural networks (spec 03 `proc.*`)
//!
//! Geometry with per-point attributes and groups; ops are ordinary features in the
//! specification tree (the feature DAG *is* the network), so a network cooks for every
//! configuration like any other feature. Wrangles are AgLang `#[wrangle]` functions compiled
//! by agcad-rtc and evaluated per point (VM/codegen on the GPU; reference here).

use agcad_driver::AgStatus;
use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::{Mesh, TessOptions};
use agcad_ir::program::Value;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product, Registry};
use agcad_solver::opt::Rng;
use agcad_std::util::*;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Geometry {
    pub points: Vec<V3>,
    /// Per-point attributes (scalar lanes; vectors stored as `name.x`, `name.y`, `name.z`).
    pub attrs: BTreeMap<String, Vec<f64>>,
    /// Polygon primitives (indices into points).
    pub prims: Vec<Vec<u32>>,
    pub groups: BTreeMap<String, Vec<u32>>,
}

impl Geometry {
    pub fn from_mesh(m: &Mesh) -> Self {
        let mut g = Geometry { points: m.positions.clone(), prims: m.triangles.iter().map(|t| t.to_vec()).collect(), ..Default::default() };
        // vertex normals (area weighted)
        let mut n = vec![V3::ZERO; g.points.len()];
        for t in &m.triangles {
            let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
            let fn_ = (b - a).cross(c - a);
            for &i in t { n[i as usize] = n[i as usize] + fn_; }
        }
        g.set_vec("N", &n.iter().map(|x| x.normalized()).collect::<Vec<_>>());
        g
    }
    pub fn set_vec(&mut self, name: &str, v: &[V3]) {
        self.attrs.insert(format!("{name}.x"), v.iter().map(|p| p.x).collect());
        self.attrs.insert(format!("{name}.y"), v.iter().map(|p| p.y).collect());
        self.attrs.insert(format!("{name}.z"), v.iter().map(|p| p.z).collect());
    }
    pub fn vec(&self, name: &str, i: usize) -> Option<V3> {
        Some(v3(*self.attrs.get(&format!("{name}.x"))?.get(i)?, *self.attrs.get(&format!("{name}.y"))?.get(i)?, *self.attrs.get(&format!("{name}.z"))?.get(i)?))
    }
    pub fn to_mesh(&self) -> Mesh {
        let mut m = Mesh { positions: self.points.clone(), ..Default::default() };
        for p in &self.prims {
            for k in 1..p.len().saturating_sub(1) { m.triangles.push([p[0], p[k], p[k + 1]]); m.face_of.push(0); }
        }
        m
    }
    pub fn append(&mut self, o: &Geometry) {
        let base = self.points.len() as u32;
        let n0 = self.points.len();
        self.points.extend_from_slice(&o.points);
        let keys: Vec<String> = self.attrs.keys().chain(o.attrs.keys()).cloned().collect();
        for k in keys {
            let mine = self.attrs.entry(k.clone()).or_insert_with(|| vec![0.0; n0]);
            mine.resize(n0, 0.0);
            match o.attrs.get(&k) { Some(v) => mine.extend_from_slice(v), None => mine.extend(std::iter::repeat(0.0).take(o.points.len())) }
        }
        self.prims.extend(o.prims.iter().map(|p| p.iter().map(|i| i + base).collect()));
        for (g, ids) in &o.groups { self.groups.entry(g.clone()).or_default().extend(ids.iter().map(|i| i + base)); }
    }
}

fn geo(ctx: &BuildContext<'_>, v: &Value) -> Result<Geometry, FeatureError> {
    match ctx.product(v)? {
        Product::Other(o) => o.downcast_ref::<Geometry>().cloned().ok_or_else(|| bad("expected procedural geometry")),
        Product::Mesh(m) => Ok(Geometry::from_mesh(m)),
        Product::Solid(s) => Ok(Geometry::from_mesh(&ctx.kernel.tessellate(s, &TessOptions::default()))),
        Product::Implicit(i) => Ok(Geometry::from_mesh(&agcad_std::mesh::implicit_to_mesh(i, (i.bounds.extent().norm() / 64.0).max(0.05))?)),
        other => Err(bad(format!("expected geometry, got {other:?}"))),
    }
}

fn out(g: Geometry) -> Product { Product::Other(Arc::new(g)) }

pub struct ProcOp(pub &'static str);

impl FeatureImpl for ProcOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "proc.network" => {
                let mut g = Geometry::default();
                if let Some(ins) = arg(n, "inputs") { for v in list(ins) { g.append(&geo(ctx, v)?); } }
                Ok(out(g))
            }
            "proc.merge" => {
                let mut g = Geometry::default();
                for v in list(req(n, "inputs")?) { g.append(&geo(ctx, v)?); }
                Ok(out(g))
            }
            "proc.scatter" => {
                let src = geo(ctx, req(n, "target")?)?;
                let count = num(ctx, n, "count")? as usize;
                let mut rng = Rng::new(num_or(ctx, n, "seed", 0.0)? as u64);
                let tris: Vec<[u32; 3]> = src.prims.iter().filter(|p| p.len() >= 3).flat_map(|p| (1..p.len() - 1).map(move |k| [p[0], p[k], p[k + 1]])).collect();
                if tris.is_empty() { return Err(bad("scatter target has no surface")); }
                let areas: Vec<f64> = tris.iter().map(|t| 0.5 * (src.points[t[1] as usize] - src.points[t[0] as usize]).cross(src.points[t[2] as usize] - src.points[t[0] as usize]).norm()).collect();
                let total: f64 = areas.iter().sum();
                let mut cdf = Vec::with_capacity(areas.len());
                let mut acc = 0.0;
                for a in &areas { acc += a / total; cdf.push(acc); }
                let mut g = Geometry::default();
                let mut normals = Vec::new();
                for _ in 0..count {
                    let r = rng.uniform();
                    let k = cdf.partition_point(|c| *c < r).min(tris.len() - 1);
                    let t = tris[k];
                    let (mut u, mut v) = (rng.uniform(), rng.uniform());
                    if u + v > 1.0 { u = 1.0 - u; v = 1.0 - v; }
                    let (a, b, c) = (src.points[t[0] as usize], src.points[t[1] as usize], src.points[t[2] as usize]);
                    g.points.push(a + (b - a) * u + (c - a) * v);
                    normals.push((b - a).cross(c - a).normalized());
                }
                g.set_vec("N", &normals);
                Ok(out(g))
            }
            "proc.attribute_create" => {
                let mut g = geo(ctx, req(n, "target")?)?;
                let name = text(n, "name").ok_or_else(|| bad("attribute needs a name"))?.to_string();
                let val = num_or(ctx, n, "default", 0.0)?;
                g.attrs.insert(name, vec![val; g.points.len()]);
                Ok(out(g))
            }
            "proc.wrangle" => {
                let mut g = geo(ctx, req(n, "net")?)?;
                let src = match req(n, "source")? { Value::String(s) => s.clone(), _ => return Err(bad("wrangle source must be AgLang text")) };
                let m = agcad_rtc::compile_module(&src, agcad_ir::Catalog::global()).map_err(|d| FeatureError::new(AgStatus::CompileFailed, d.iter().map(|x| x.render("wrangle", &src)).collect::<Vec<_>>().join("\n")))?;
                let item = m.items.iter().find(|i| matches!(i.kind, agcad_rtc::ItemKind::Wrangle { .. })).ok_or_else(|| bad("source has no #[wrangle] function"))?;
                let graph = item.expr.as_ref().unwrap();
                let params: Vec<f64> = item.params.iter().map(|p| ctx.params.get(&p.name).copied().or(p.default).unwrap_or(0.0)).collect();
                let mut normals = Vec::with_capacity(g.points.len());
                for i in 0..g.points.len() {
                    let p = g.points[i];
                    let nn = g.vec("N", i).unwrap_or(V3::Z);
                    let o = graph.eval(&[p.x, p.y, p.z, nn.x, nn.y, nn.z], &params);
                    if o.len() >= 3 { g.points[i] = v3(o[0], o[1], o[2]); }
                    normals.push(if o.len() >= 6 { v3(o[3], o[4], o[5]) } else { nn });
                }
                g.set_vec("N", &normals);
                Ok(out(g))
            }
            "proc.copy_to_points" => {
                let src = geo(ctx, req(n, "geo")?)?;
                let pts = geo(ctx, req(n, "pts")?)?;
                let mut g = Geometry::default();
                for i in 0..pts.points.len() {
                    let s = pts.attrs.get("pscale").and_then(|v| v.get(i)).copied().unwrap_or(1.0);
                    let up = pts.vec("N", i).unwrap_or(V3::Z).normalized();
                    let x = up.any_perp();
                    let y = up.cross(x);
                    let mut inst = src.clone();
                    for p in &mut inst.points { *p = pts.points[i] + (x * p.x + y * p.y + up * p.z) * s; }
                    inst.groups.insert(format!("copy.{i}"), (0..inst.points.len() as u32).collect());
                    g.append(&inst);
                }
                Ok(out(g))
            }
            "proc.group" => {
                let mut g = geo(ctx, req(n, "target")?)?;
                let name = text(n, "name").unwrap_or("group").to_string();
                let (lo, hi) = match (arg(n, "min"), arg(n, "max")) { (Some(a), Some(b)) => (vec3(ctx, a)?, vec3(ctx, b)?), _ => (v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), v3(f64::INFINITY, f64::INFINITY, f64::INFINITY)) };
                let ids: Vec<u32> = g.points.iter().enumerate().filter(|(_, p)| p.x >= lo.x && p.y >= lo.y && p.z >= lo.z && p.x <= hi.x && p.y <= hi.y && p.z <= hi.z).map(|(i, _)| i as u32).collect();
                g.groups.insert(name, ids);
                Ok(out(g))
            }
            "proc.blast" => {
                let g = geo(ctx, req(n, "target")?)?;
                let group = text(n, "group").ok_or_else(|| bad("blast needs a group name"))?;
                let keep = matches!(arg(n, "keep"), Some(Value::Bool(true)));
                let sel: std::collections::BTreeSet<u32> = g.groups.get(group).cloned().unwrap_or_default().into_iter().collect();
                let mut remap = vec![u32::MAX; g.points.len()];
                let mut r = Geometry::default();
                for (i, p) in g.points.iter().enumerate() {
                    if sel.contains(&(i as u32)) == keep { remap[i] = r.points.len() as u32; r.points.push(*p); }
                }
                for (k, v) in &g.attrs { r.attrs.insert(k.clone(), v.iter().enumerate().filter(|(i, _)| remap[*i] != u32::MAX).map(|(_, x)| *x).collect()); }
                r.prims = g.prims.iter().filter(|p| p.iter().all(|i| remap[*i as usize] != u32::MAX)).map(|p| p.iter().map(|i| remap[*i as usize]).collect()).collect();
                Ok(out(r))
            }
            "proc.cook" => {
                let g = geo(ctx, req(n, "net")?)?;
                Ok(Product::Mesh(g.to_mesh()))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}

/// Register procedural ops into a feature registry.
pub fn register(r: &mut Registry) {
    for op in ["proc.network", "proc.merge", "proc.scatter", "proc.attribute_create", "proc.wrangle", "proc.copy_to_points", "proc.group", "proc.blast", "proc.cook"] {
        r.register(Arc::new(ProcOp(op)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_geo::kernel::NativeKernel;
    use agcad_ir::Quantity;
    use agcad_runtime::document::PartState;
    use agcad_runtime::rebuild::update;

    #[test]
    fn scatter_wrangle_copy_cook() {
        let mut reg = agcad_std::standard_registry();
        register(&mut reg);
        let mut s = PartState::new("proc");
        let m = |kv: &[(&str, Value)]| kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<BTreeMap<_, _>>();
        let r = |id: u32| Value::Ref { id: format!("f{id}"), field: None };
        let mm = |x: f64| Value::Quantity(Quantity::mm(x));
        let plate = s.add_feature("part.primitive", 1, m(&[("kind", Value::String("box".into())), ("p", Value::List(vec![mm(40.0), mm(40.0), mm(2.0)]))]), 0, None);
        let pts = s.add_feature("proc.scatter", 1, m(&[("target", r(plate)), ("count", Value::Int(25)), ("seed", Value::Int(7))]), 0, None);
        let src = "#[wrangle(over = \"point\")] fn lift(pt: &mut Point, #[param(default = 1.0*mm)] h: Length) { pt.p += pt.n * h; }";
        let w = s.add_feature("proc.wrangle", 1, m(&[("net", r(pts)), ("over", Value::String("point".into())), ("source", Value::String(src.into()))]), 0, None);
        let stud = s.add_feature("part.primitive", 1, m(&[("kind", Value::String("box".into())), ("p", Value::List(vec![mm(1.0), mm(1.0), mm(1.0)]))]), 1, None);
        let cp = s.add_feature("proc.copy_to_points", 1, m(&[("geo", r(stud)), ("pts", r(w)), ("pack", Value::Bool(false))]), 0, None);
        let ck = s.add_feature("proc.cook", 1, m(&[("net", r(cp))]), 0, None);
        let res = update(&s, &reg, &NativeKernel, BTreeMap::new(), vec![]);
        for o in &res.outcomes { assert!(o.status.is_ok(), "{} failed: {:?}", o.feature_ref, o.error); }
        let Product::Mesh(mesh) = &res.products[&ck] else { panic!() };
        assert!(mesh.triangles.len() >= 25 * 12);
    }
}
