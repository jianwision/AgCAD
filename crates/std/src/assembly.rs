//! Assembly Design (spec 03 `assembly.*`): components, mates (fix, coincidence of axes,
//! contact/offset of planes, angle), batched rigid-body solve with agcad-solver LM.
//! Each component has 6 unknowns (translation, rotation vector); mates are residuals on
//! world-space frames of component-local references.

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::kernel::Solid;
use agcad_geo::math::{v3, Frame, V3, Xform};
use agcad_ir::program::Value;
use agcad_rtc::expr::{Bin, Dim, ExprGraph, Un};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use agcad_solver::nls;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    pub solid: Solid,
    pub placement: Xform,
}

#[derive(Clone, Debug)]
pub enum Mate {
    Fix(usize),
    /// Axes (origin + direction in component-local coordinates) coincide.
    Coincidence { a: usize, fa: Frame, b: usize, fb: Frame },
    /// Planes (origin + normal) at signed offset `d` (contact = 0, opposed normals).
    Offset { a: usize, fa: Frame, b: usize, fb: Frame, d: f64 },
    Angle { a: usize, fa: Frame, b: usize, fb: Frame, angle: f64 },
}

#[derive(Clone, Debug)]
pub struct ProductData {
    pub components: Vec<Component>,
    pub mates: Vec<Mate>,
}

fn product_of(ctx: &BuildContext<'_>, v: &Value) -> Result<ProductData, FeatureError> {
    match ctx.product(v)? {
        Product::Other(o) => o.downcast_ref::<ProductData>().cloned().ok_or_else(|| bad("expected a product")),
        _ => Err(bad("expected a product")),
    }
}

fn local_frame(ctx: &BuildContext<'_>, v: Option<&Value>) -> Result<Frame, FeatureError> {
    match v {
        None => Ok(Frame::WORLD),
        Some(Value::Map(m)) => {
            let o = m.get("origin").map(|x| vec3(ctx, x)).transpose()?.unwrap_or(V3::ZERO);
            let z = m.get("dir").or(m.get("normal")).map(|x| vec3(ctx, x)).transpose()?.unwrap_or(V3::Z);
            Ok(Frame::new(o, z, z.any_perp()))
        }
        Some(other) => frame(ctx, other),
    }
}

/// Rotation vector → rotation matrix (Rodrigues) in the expression graph: returns 9 nodes.
fn rot_nodes(g: &mut ExprGraph, r: [u32; 3]) -> [u32; 9] {
    let b = |g: &mut ExprGraph, op, a, c| g.bin(op, a, c).unwrap();
    let sq = |g: &mut ExprGraph, a| g.bin(Bin::Mul, a, a).unwrap();
    let tiny = g.konst(1e-30, Dim::NONE);
    let th2 = { let (a, c, d) = (sq(g, r[0]), sq(g, r[1]), sq(g, r[2])); let s = b(g, Bin::Add, a, c); let s = b(g, Bin::Add, s, d); b(g, Bin::Add, s, tiny) };
    let th = g.un(Un::Sqrt, th2).unwrap();
    let k: Vec<u32> = r.iter().map(|&x| b(g, Bin::Div, x, th)).collect();
    let s = g.un(Un::Sin, th).unwrap();
    let c = g.un(Un::Cos, th).unwrap();
    let one = g.konst(1.0, Dim::NONE);
    let v = b(g, Bin::Sub, one, c);
    let kk = |g: &mut ExprGraph, i: usize, j: usize| { let t = b(g, Bin::Mul, k[i], k[j]); b(g, Bin::Mul, t, v) };
    let ks = |g: &mut ExprGraph, i: usize| b(g, Bin::Mul, k[i], s);
    let d0 = { let t = kk(g, 0, 0); b(g, Bin::Add, c, t) };
    let d1 = { let t = kk(g, 1, 1); b(g, Bin::Add, c, t) };
    let d2 = { let t = kk(g, 2, 2); b(g, Bin::Add, c, t) };
    let m01 = { let t = kk(g, 0, 1); let u = ks(g, 2); b(g, Bin::Sub, t, u) };
    let m02 = { let t = kk(g, 0, 2); let u = ks(g, 1); b(g, Bin::Add, t, u) };
    let m10 = { let t = kk(g, 1, 0); let u = ks(g, 2); b(g, Bin::Add, t, u) };
    let m12 = { let t = kk(g, 1, 2); let u = ks(g, 0); b(g, Bin::Sub, t, u) };
    let m20 = { let t = kk(g, 2, 0); let u = ks(g, 1); b(g, Bin::Sub, t, u) };
    let m21 = { let t = kk(g, 2, 1); let u = ks(g, 0); b(g, Bin::Add, t, u) };
    [d0, m01, m02, m10, d1, m12, m20, m21, d2]
}

fn apply(g: &mut ExprGraph, rm: &[u32; 9], t: Option<[u32; 3]>, p: V3) -> [u32; 3] {
    let mut out = [0u32; 3];
    for i in 0..3 {
        let (x, y, z) = (g.konst(p.x, Dim::NONE), g.konst(p.y, Dim::NONE), g.konst(p.z, Dim::NONE));
        let a = g.bin(Bin::Mul, rm[3 * i], x).unwrap();
        let b2 = g.bin(Bin::Mul, rm[3 * i + 1], y).unwrap();
        let c = g.bin(Bin::Mul, rm[3 * i + 2], z).unwrap();
        let s = g.bin(Bin::Add, a, b2).unwrap();
        let s = g.bin(Bin::Add, s, c).unwrap();
        out[i] = match t { Some(t) => g.bin(Bin::Add, s, t[i]).unwrap(), None => s };
    }
    out
}

/// Solve placements; returns (product with solved placements, DOF report).
pub fn solve(p: &ProductData) -> Result<(ProductData, nls::LmReport), FeatureError> {
    let nc = p.components.len();
    let mut g = ExprGraph::default();
    let vars: Vec<[u32; 6]> = (0..nc).map(|i| { let mut v = [0u32; 6]; for k in 0..6 { v[k] = g.input(&format!("c{i}.{k}"), Dim::NONE); } v }).collect();
    let rms: Vec<[u32; 9]> = vars.iter().map(|v| rot_nodes(&mut g, [v[3], v[4], v[5]])).collect();
    let mut res = Vec::new();
    let world = |g: &mut ExprGraph, c: usize, f: &Frame| -> ([u32; 3], [u32; 3]) {
        // component placement P_c ∘ (current solve transform)
        let pl = p.components[c].placement;
        let (o, z) = (pl.apply(f.origin), pl.apply_dir(f.z).normalized());
        let t = [vars[c][0], vars[c][1], vars[c][2]];
        (apply(g, &rms[c], Some(t), o), apply(g, &rms[c], None, z))
    };
    let sub = |g: &mut ExprGraph, a: u32, b: u32| g.bin(Bin::Sub, a, b).unwrap();
    let dot = |g: &mut ExprGraph, a: [u32; 3], b: [u32; 3]| { let x = g.bin(Bin::Mul, a[0], b[0]).unwrap(); let y = g.bin(Bin::Mul, a[1], b[1]).unwrap(); let z = g.bin(Bin::Mul, a[2], b[2]).unwrap(); let s = g.bin(Bin::Add, x, y).unwrap(); g.bin(Bin::Add, s, z).unwrap() };
    let cross = |g: &mut ExprGraph, a: [u32; 3], b: [u32; 3]| -> [u32; 3] {
        let m = |g: &mut ExprGraph, i: u32, j: u32| g.bin(Bin::Mul, i, j).unwrap();
        let (p1, p2) = (m(g, a[1], b[2]), m(g, a[2], b[1]));
        let (q1, q2) = (m(g, a[2], b[0]), m(g, a[0], b[2]));
        let (r1, r2) = (m(g, a[0], b[1]), m(g, a[1], b[0]));
        [g.bin(Bin::Sub, p1, p2).unwrap(), g.bin(Bin::Sub, q1, q2).unwrap(), g.bin(Bin::Sub, r1, r2).unwrap()]
    };
    for m in &p.mates {
        match m {
            Mate::Fix(c) => { for k in 0..6 { res.push(vars[*c][k]); } }
            Mate::Coincidence { a, fa, b, fb } => {
                let (oa, za) = world(&mut g, *a, fa);
                let (ob, zb) = world(&mut g, *b, fb);
                let cz = cross(&mut g, za, zb);
                res.extend(cz);
                let d = [sub(&mut g, ob[0], oa[0]), sub(&mut g, ob[1], oa[1]), sub(&mut g, ob[2], oa[2])];
                let cd = cross(&mut g, d, za);
                res.extend(cd);
            }
            Mate::Offset { a, fa, b, fb, d } => {
                let (oa, za) = world(&mut g, *a, fa);
                let (ob, zb) = world(&mut g, *b, fb);
                let cz = cross(&mut g, za, zb);
                res.extend(cz);
                let dv = [sub(&mut g, ob[0], oa[0]), sub(&mut g, ob[1], oa[1]), sub(&mut g, ob[2], oa[2])];
                let h = dot(&mut g, dv, za);
                let k = g.konst(*d, Dim::NONE);
                res.push(sub(&mut g, h, k));
            }
            Mate::Angle { a, fa, b, fb, angle } => {
                let (_, za) = world(&mut g, *a, fa);
                let (_, zb) = world(&mut g, *b, fb);
                let c = dot(&mut g, za, zb);
                let k = g.konst(angle.cos(), Dim::NONE);
                res.push(sub(&mut g, c, k));
            }
        }
    }
    g.outputs = res;
    if g.outputs.is_empty() { return Err(FeatureError::new(AgStatus::UnderConstrained, "product has no constraints")); }
    let compiled = nls::compile(&g).map_err(|s| FeatureError::new(s, "assembly too large for one dense block (≤ 5 components per block until solve plans decompose products)"))?;
    let x0 = vec![1e-6; 6 * nc];
    let rep = nls::lm_solve(&compiled, &x0, &[], nls::LmOptions::default());
    let mut out = p.clone();
    for (i, c) in out.components.iter_mut().enumerate() {
        let x = &rep.x[6 * i..6 * i + 6];
        let rv = v3(x[3], x[4], x[5]);
        let ang = rv.norm();
        let rot = if ang > 1e-15 { Xform::rotation(V3::ZERO, rv, ang) } else { Xform::IDENTITY };
        let solve = Xform { t: v3(x[0], x[1], x[2]), ..rot };
        c.placement = solve.compose(&c.placement);
    }
    Ok((out, rep))
}

pub struct AssemblyOp(pub &'static str);

impl FeatureImpl for AssemblyOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let wrap = |p: ProductData| Product::Other(Arc::new(p));
        match self.0 {
            "assembly.component" => {
                // components accumulate on an optional existing product (`into`)
                let mut p = match arg(n, "into") { Some(v) => product_of(ctx, v)?, None => ProductData { components: vec![], mates: vec![] } };
                let s = solid(ctx, req(n, "ref")?)?;
                let t = match arg(n, "xform") { Some(v) => Xform::translation(vec3(ctx, v)?), None => Xform::IDENTITY };
                p.components.push(Component { name: format!("{}.{}", n.feature_ref(), p.components.len()), solid: s, placement: t });
                Ok(wrap(p))
            }
            "assembly.fix" => { let mut p = product_of(ctx, req(n, "comp")?)?; let i = num_or(ctx, n, "index", 0.0)? as usize; p.mates.push(Mate::Fix(i)); Ok(wrap(p)) }
            "assembly.coincidence" | "assembly.contact" | "assembly.offset" | "assembly.angle" => {
                let mut p = product_of(ctx, req(n, "a")?)?;
                let ia = num_or(ctx, n, "ia", 0.0)? as usize;
                let ib = num_or(ctx, n, "ib", 1.0)? as usize;
                if ia >= p.components.len() || ib >= p.components.len() { return Err(bad("mate references a missing component index")); }
                let fa = local_frame(ctx, arg(n, "fa"))?;
                let fb = local_frame(ctx, arg(n, "fb"))?;
                p.mates.push(match self.0 {
                    "assembly.coincidence" => Mate::Coincidence { a: ia, fa, b: ib, fb },
                    "assembly.contact" => Mate::Offset { a: ia, fa, b: ib, fb, d: 0.0 },
                    "assembly.offset" => Mate::Offset { a: ia, fa, b: ib, fb, d: num(ctx, n, "d")? },
                    _ => Mate::Angle { a: ia, fa, b: ib, fb, angle: num(ctx, n, "angle")? },
                });
                Ok(wrap(p))
            }
            "assembly.update" => {
                let p = product_of(ctx, req(n, "product")?)?;
                let (solved, rep) = solve(&p)?;
                if rep.status != AgStatus::Accepted { return Err(FeatureError::new(rep.status, format!("assembly constraints: max residual {:.3e} (rank {}, dof {})", rep.max_residual, rep.rank, rep.dof))); }
                Ok(wrap(solved))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        }
    }
}

/// Positioned solids of a product (for clash, export).
pub fn placed_solids(p: &ProductData) -> Vec<Solid> {
    p.components.iter().map(|c| Solid::Brep(c.solid.brep().transformed(&c.placement))).collect()
}
