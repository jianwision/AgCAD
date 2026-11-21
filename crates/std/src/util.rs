//! Argument helpers shared by feature implementations.

use agcad_driver::AgStatus;
use agcad_geo::kernel::Solid;
use agcad_geo::math::{Frame, V3};
use agcad_ir::program::Value;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, Product};

pub fn arg<'a>(n: &'a FeatureNode, k: &str) -> Option<&'a Value> { n.args.get(k) }

pub fn req<'a>(n: &'a FeatureNode, k: &str) -> Result<&'a Value, FeatureError> {
    n.args.get(k).ok_or_else(|| FeatureError::new(AgStatus::InvalidParameter, format!("{}: missing input '{k}'", n.op)))
}

pub fn num(ctx: &BuildContext<'_>, n: &FeatureNode, k: &str) -> Result<f64, FeatureError> { ctx.number(req(n, k)?) }

pub fn num_or(ctx: &BuildContext<'_>, n: &FeatureNode, k: &str, d: f64) -> Result<f64, FeatureError> {
    match n.args.get(k) { Some(v) => ctx.number(v), None => Ok(d) }
}

pub fn text<'a>(n: &'a FeatureNode, k: &str) -> Option<&'a str> {
    match n.args.get(k) { Some(Value::String(s)) => Some(s.as_str()), _ => None }
}

pub fn list<'a>(v: &'a Value) -> &'a [Value] { if let Value::List(l) = v { l } else { std::slice::from_ref(v) } }

pub fn vec2(ctx: &BuildContext<'_>, v: &Value) -> Result<[f64; 2], FeatureError> {
    match v {
        Value::List(l) if l.len() >= 2 => Ok([ctx.number(&l[0])?, ctx.number(&l[1])?]),
        Value::Map(m) if m.get("$ctor").is_some() => {
            let a = m.get("args").map(list).unwrap_or(&[]);
            if a.len() >= 2 { Ok([ctx.number(&a[0])?, ctx.number(&a[1])?]) } else { Err(bad("pt2 needs 2 args")) }
        }
        _ => Err(bad(format!("expected a 2D point, got {v:?}"))),
    }
}

pub fn vec3(ctx: &BuildContext<'_>, v: &Value) -> Result<V3, FeatureError> {
    match v {
        Value::List(l) if l.len() >= 3 => Ok(V3 { x: ctx.number(&l[0])?, y: ctx.number(&l[1])?, z: ctx.number(&l[2])? }),
        Value::String(s) => match s.as_str() { "x" | "+x" => Ok(V3::X), "y" | "+y" => Ok(V3::Y), "z" | "+z" => Ok(V3::Z), "-x" => Ok(-V3::X), "-y" => Ok(-V3::Y), "-z" => Ok(-V3::Z), _ => Err(bad(format!("unknown direction '{s}'"))) },
        _ => Err(bad(format!("expected a 3D vector, got {v:?}"))),
    }
}

pub fn bad(m: impl Into<String>) -> FeatureError { FeatureError::new(AgStatus::InvalidParameter, m) }

pub fn solid(ctx: &BuildContext<'_>, v: &Value) -> Result<Solid, FeatureError> {
    match ctx.product(v)? {
        Product::Solid(s) => Ok(s.clone()),
        other => Err(bad(format!("expected a solid body, got {other:?}"))),
    }
}

/// Named planes / axis systems: "xy", "yz", "zx" or a frame product.
pub fn frame(ctx: &BuildContext<'_>, v: &Value) -> Result<Frame, FeatureError> {
    match v {
        Value::String(_) => match s_of(v) {
            "xy" | "xy_plane" | "XY" => Ok(Frame::WORLD),
            "yz" | "yz_plane" | "YZ" => Ok(Frame { origin: V3::ZERO, x: V3::Y, y: V3::Z, z: V3::X }),
            "zx" | "zx_plane" | "ZX" | "xz" => Ok(Frame { origin: V3::ZERO, x: V3::Z, y: V3::X, z: V3::Y }),
            other => Err(bad(format!("unknown plane '{other}'"))),
        },
        Value::Map(m) => {
            if let Some(Value::Bool(true)) = m.get("xy_plane") { return Ok(Frame::WORLD); }
            if m.get("yz_plane").is_some() { return frame(ctx, &Value::String("yz".into())); }
            if m.get("zx_plane").is_some() { return frame(ctx, &Value::String("zx".into())); }
            Err(bad("unrecognized plane reference"))
        }
        Value::Ref { .. } => match ctx.product(v)? {
            Product::Frame(f) => Ok(*f),
            Product::Other(o) => o.downcast_ref::<crate::sketch::SketchData>().map(|s| s.frame).ok_or_else(|| bad("reference is not a plane")),
            _ => Err(bad("reference is not a plane")),
        },
        _ => Err(bad(format!("expected a plane, got {v:?}"))),
    }
}

fn s_of(v: &Value) -> &str { if let Value::String(s) = v { s } else { "" } }
