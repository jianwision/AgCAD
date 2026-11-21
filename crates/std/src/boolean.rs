//! Booleans (spec 03 `boolean.*`): solid bodies via the exact kernel; implicit bodies by
//! expression-graph composition (sharp or smooth blends never fail).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::math::Aabb;
use agcad_geo::region2d::BoolOp2;
use agcad_rtc::expr::{Bin, Dim, Node};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, ImplicitBody, Product};
use std::sync::Arc;

/// Compose two implicit bodies: `op` = Min (union), Max (intersection); `r > 0` → smooth.
/// Subtraction is `Max(a, -b)` (callers negate b first via [`negate`]).
pub fn compose(a: &ImplicitBody, b: &ImplicitBody, op: Bin, r: f64) -> ImplicitBody {
    let mut g = (*a.graph).clone();
    let a_out = g.outputs[0];
    // Append b's nodes with remapped indices; inputs map by name (p.x/p.y/p.z), params appended.
    let base_params = g.param_names.len() as u16;
    for (k, name) in b.graph.param_names.iter().enumerate() {
        g.param_names.push(format!("b.{name}"));
        g.param_dims.push(b.graph.param_dims[k]);
    }
    let mut map: Vec<u32> = Vec::with_capacity(b.graph.nodes.len());
    for (i, n) in b.graph.nodes.iter().enumerate() {
        let d = b.graph.dims[i];
        let id = match n.clone() {
            Node::Input(k) => { let name = b.graph.input_names[k as usize].clone(); g.input(&name, d) }
            Node::Param(k) => g.push(Node::Param(base_params + k), d),
            Node::Const(c) => g.konst(c, d),
            Node::Un(op, x) => g.push(Node::Un(op, map[x as usize]), d),
            Node::Bin(op, x, y) => g.push(Node::Bin(op, map[x as usize], map[y as usize]), d),
            Node::Select(c, x, y) => g.push(Node::Select(map[c as usize], map[x as usize], map[y as usize]), d),
            Node::Noise(x, y, z) => g.push(Node::Noise(map[x as usize], map[y as usize], map[z as usize]), d),
        };
        map.push(id);
    }
    let b_out = map[b.graph.outputs[0] as usize];
    let out = if r > 0.0 {
        // smooth min/max (polynomial): min: mix(b, a, h) - k h (1-h)
        let k = g.konst(r, Dim::LENGTH);
        let (aa, bb) = if op == Bin::Max { (g.push(Node::Un(agcad_rtc::expr::Un::Neg, a_out), Dim::LENGTH), g.push(Node::Un(agcad_rtc::expr::Un::Neg, b_out), Dim::LENGTH)) } else { (a_out, b_out) };
        let half = g.konst(0.5, Dim::NONE);
        let one = g.konst(1.0, Dim::NONE);
        let zero = g.konst(0.0, Dim::NONE);
        let d = g.push(Node::Bin(Bin::Sub, bb, aa), Dim::LENGTH);
        let q = g.push(Node::Bin(Bin::Div, d, k), Dim::NONE);
        let q = g.push(Node::Bin(Bin::Mul, half, q), Dim::NONE);
        let h = g.push(Node::Bin(Bin::Add, half, q), Dim::NONE);
        let h = g.push(Node::Bin(Bin::Max, h, zero), Dim::NONE);
        let h = g.push(Node::Bin(Bin::Min, h, one), Dim::NONE);
        let omh = g.push(Node::Bin(Bin::Sub, one, h), Dim::NONE);
        let bm = g.push(Node::Bin(Bin::Mul, bb, omh), Dim::LENGTH);
        let am = g.push(Node::Bin(Bin::Mul, aa, h), Dim::LENGTH);
        let mix = g.push(Node::Bin(Bin::Add, bm, am), Dim::LENGTH);
        let kh = g.push(Node::Bin(Bin::Mul, k, h), Dim::LENGTH);
        let khh = g.push(Node::Bin(Bin::Mul, kh, omh), Dim::LENGTH);
        let m = g.push(Node::Bin(Bin::Sub, mix, khh), Dim::LENGTH);
        if op == Bin::Max { g.push(Node::Un(agcad_rtc::expr::Un::Neg, m), Dim::LENGTH) } else { m }
    } else {
        g.push(Node::Bin(op, a_out, b_out), Dim::LENGTH)
    };
    g.outputs = vec![out];
    let mut params = a.params.clone();
    params.extend_from_slice(&b.params);
    let bounds = if op == Bin::Min { Aabb::union(a.bounds, b.bounds) } else { a.bounds };
    ImplicitBody { graph: Arc::new(g), params, bounds, lipschitz: a.lipschitz.max(b.lipschitz) }
}

pub fn negate(a: &ImplicitBody) -> ImplicitBody {
    let mut g = (*a.graph).clone();
    let o = g.outputs[0];
    let n = g.push(Node::Un(agcad_rtc::expr::Un::Neg, o), Dim::LENGTH);
    g.outputs = vec![n];
    ImplicitBody { graph: Arc::new(g), ..a.clone() }
}

fn implicit(ctx: &BuildContext<'_>, v: &agcad_ir::Value) -> Result<ImplicitBody, FeatureError> {
    match ctx.product(v)? {
        Product::Implicit(i) => Ok(i.clone()),
        Product::Solid(s) => Ok(crate::part::box_sdf(&s.brep().bbox())),
        other => Err(bad(format!("expected an implicit body, got {other:?}"))),
    }
}

pub struct BooleanOp(pub &'static str);

impl FeatureImpl for BooleanOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "boolean.add" | "boolean.remove" | "boolean.intersect" => {
                let op = match self.0 { "boolean.add" => BoolOp2::Union, "boolean.remove" => BoolOp2::Difference, _ => BoolOp2::Intersection };
                let mut acc = solid(ctx, req(n, "target")?)?;
                for t in list(req(n, "tools")?) {
                    let tool = solid(ctx, t)?;
                    acc = ctx.kernel.boolean(&acc, &tool, op).map_err(|e| FeatureError::new(e.status, e.message).with_elements(vec![n.feature_ref()]))?;
                }
                Ok(Product::Solid(acc))
            }
            "boolean.smooth_boolean" => {
                let a = implicit(ctx, req(n, "a")?)?;
                let b = implicit(ctx, req(n, "b")?)?;
                let r = num_or(ctx, n, "r", 0.0)?;
                Ok(Product::Implicit(match text(n, "op").unwrap_or("union") {
                    "union" | "add" => compose(&a, &b, Bin::Min, r),
                    "intersect" | "intersection" => compose(&a, &b, Bin::Max, r),
                    "subtract" | "remove" | "difference" => compose(&a, &negate(&b), Bin::Max, r),
                    other => return Err(bad(format!("unknown smooth_boolean op '{other}'"))),
                }))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}
