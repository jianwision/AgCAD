//! Knowledgeware + reference geometry (spec 03 `knowledge.*`, `datum.*`).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::math::{Frame, V3};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};

pub struct KnowledgeOp(pub &'static str);

impl FeatureImpl for KnowledgeOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        match self.0 {
            "knowledge.parameter" => {
                let name = text(n, "name").ok_or_else(|| bad("parameter needs a name"))?.to_string();
                // Design-table / configuration values override the declared value.
                let v = match ctx.params.get(&name) { Some(v) => *v, None => num(ctx, n, "value")? };
                if let Some(agcad_ir::Value::List(r)) = arg(n, "range") {
                    let (lo, hi) = (ctx.number(&r[0])?, ctx.number(&r[1])?);
                    if v < lo || v > hi { return Err(FeatureError::new(AgStatus::InvalidParameter, format!("parameter {name} = {v} outside [{lo}, {hi}]"))); }
                }
                ctx.params.insert(name, v);
                Ok(Product::Scalar(v))
            }
            "datum.plane" | "datum.axis_system" => {
                let mode = text(n, "mode").unwrap_or("offset");
                let base = match arg(n, "refs") { Some(r) => frame(ctx, list(r).first().ok_or_else(|| bad("plane needs a reference"))?)?, None => Frame::WORLD };
                let off = num_or(ctx, n, "value", 0.0)?;
                match mode {
                    "offset" => Ok(Product::Frame(Frame { origin: base.origin + base.z * off, ..base })),
                    _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("datum plane mode '{mode}' not implemented yet"))),
                }
            }
            "datum.point" => Ok(Product::Frame(Frame { origin: vec3(ctx, req(n, "value")?)?, ..Frame::WORLD })),
            _ => { let _ = V3::ZERO; Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))) }
        }
    }
}
