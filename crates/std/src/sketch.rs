//! Sketcher (spec 03 `sketch.*`, `constraint.*`): entities, constraints solved by the batched
//! LM solver (agcad-solver), and region extraction into normalized profiles. Sketch values are
//! functional: each op returns a new sketch snapshot; entity indices only grow, so element
//! references stay valid in later snapshots (ops take the newest snapshot among their inputs).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::builders::{Profile, Seg2};
use agcad_geo::math::Frame;
use agcad_ir::program::Value;
use agcad_rtc::expr::{Bin, Dim, ExprGraph, Un};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use agcad_solver::nls;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Ent {
    Point(usize),
    Line { a: usize, b: usize, construction: bool },
    /// Arc around center point `c` from point `a` to point `b`, counter-clockwise.
    Arc { c: usize, a: usize, b: usize },
    Circle { c: usize, r: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Con {
    Coincident(usize, usize),
    Horizontal(usize, usize),
    Vertical(usize, usize),
    Parallel([usize; 4]),
    Perpendicular([usize; 4]),
    Distance(usize, usize, f64),
    Radius(usize, f64),
    Fix(usize, [f64; 2]),
    Equal([usize; 4]),
    Concentric(usize, usize),
    /// Line (a, b) tangent to circle center c, radius var r.
    Tangent { a: usize, b: usize, c: usize, r: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SketchData {
    pub frame: Frame,
    pub pts: Vec<[f64; 2]>,
    pub rads: Vec<f64>,
    pub ents: Vec<Ent>,
    pub cons: Vec<Con>,
    /// Index of the entity created by the op that produced this snapshot.
    pub last: Option<usize>,
}

impl SketchData {
    fn point(&mut self, p: [f64; 2]) -> usize {
        if let Some(i) = self.pts.iter().position(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 1e-9) { return i; }
        self.pts.push(p);
        self.pts.len() - 1
    }
    fn push(mut self, e: Ent) -> Self { self.ents.push(e); self.last = Some(self.ents.len() - 1); self }
    fn line_pts(&self, e: usize) -> Option<(usize, usize)> { match self.ents.get(e)? { Ent::Line { a, b, .. } => Some((*a, *b)), _ => None } }

    /// Closed regions from non-construction entities, nested into outer + holes.
    pub fn regions(&self) -> Result<Vec<Profile>, FeatureError> {
        let mut loops: Vec<Vec<Seg2>> = Vec::new();
        let mut open: Vec<Seg2> = Vec::new();
        for e in &self.ents {
            match *e {
                Ent::Circle { c, r } => loops.push(vec![Seg2::Circle { c: self.pts[c], r: self.rads[r], ccw: true }]),
                Ent::Line { a, b, construction: false } => open.push(Seg2::Line { a: self.pts[a], b: self.pts[b] }),
                Ent::Arc { c, a, b } => {
                    let (pc, pa, pb) = (self.pts[c], self.pts[a], self.pts[b]);
                    let r = (pa[0] - pc[0]).hypot(pa[1] - pc[1]);
                    let a0 = (pa[1] - pc[1]).atan2(pa[0] - pc[0]);
                    let mut a1 = (pb[1] - pc[1]).atan2(pb[0] - pc[0]);
                    while a1 <= a0 { a1 += core::f64::consts::TAU; }
                    open.push(Seg2::Arc { c: pc, r, a0, a1 });
                }
                _ => {}
            }
        }
        // chain open segments (allow reversal) into loops
        while let Some(first) = open.pop() {
            let mut lp = vec![first];
            let start = first.start();
            let close = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]) < 1e-7;
            loop {
                let end = lp.last().unwrap().end();
                if close(end, start) { break; }
                if let Some(k) = open.iter().position(|s| close(s.start(), end)) { lp.push(open.swap_remove(k)); }
                else if let Some(k) = open.iter().position(|s| close(s.end(), end)) { let s = open.swap_remove(k); lp.push(s.reversed()); }
                else { return Err(FeatureError::new(AgStatus::TopologyFailed, "sketch profile is not closed").suggest("add a line or coincidence to close the open chain")); }
            }
            loops.push(lp);
        }
        if loops.is_empty() { return Err(FeatureError::new(AgStatus::InvalidParameter, "sketch has no closed regions")); }
        // Nest: sort by |area| descending; each loop becomes a hole of the smallest containing outer.
        let mut items: Vec<(f64, Vec<Seg2>)> = loops.into_iter().map(|l| (Profile::loop_area(&l).abs(), l)).collect();
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut regions: Vec<Profile> = Vec::new();
        for (_, l) in items {
            let probe = l[0].start();
            let host = regions.iter_mut().filter(|r| agcad_geo::region2d::winding(&r.loops[..1], probe).abs() > 0.5 && r.loops.iter().skip(1).all(|h| agcad_geo::region2d::winding(&[h.clone()], probe).abs() < 0.5)).last();
            match host {
                Some(r) => r.loops.push(l),
                None => regions.push(Profile { loops: vec![l] }),
            }
        }
        Ok(regions.into_iter().map(Profile::normalize).collect())
    }

    /// Solve constraints with batched LM (one configuration on the host reference path).
    pub fn solve(&mut self) -> Result<nls::LmReport, FeatureError> {
        if self.cons.is_empty() {
            return Ok(nls::LmReport { status: AgStatus::Accepted, x: vec![], iterations: 0, cost: 0.0, max_residual: 0.0, rank: 0, dof: 0 });
        }
        let mut g = ExprGraph::default();
        let x: Vec<u32> = (0..self.pts.len()).map(|i| g.input(&format!("x{i}"), Dim::NONE)).collect();
        let y: Vec<u32> = (0..self.pts.len()).map(|i| g.input(&format!("y{i}"), Dim::NONE)).collect();
        let r: Vec<u32> = (0..self.rads.len()).map(|i| g.input(&format!("r{i}"), Dim::NONE)).collect();
        let mut out = Vec::new();
        let b = |g: &mut ExprGraph, op, a, c| g.bin(op, a, c).expect("dimensionless");
        for c in &self.cons {
            match *c {
                Con::Coincident(p, q) => { out.push(b(&mut g, Bin::Sub, x[p], x[q])); out.push(b(&mut g, Bin::Sub, y[p], y[q])); }
                Con::Horizontal(p, q) => out.push(b(&mut g, Bin::Sub, y[p], y[q])),
                Con::Vertical(p, q) => out.push(b(&mut g, Bin::Sub, x[p], x[q])),
                Con::Parallel([a, bb, cc, d]) | Con::Perpendicular([a, bb, cc, d]) => {
                    let dx1 = b(&mut g, Bin::Sub, x[bb], x[a]); let dy1 = b(&mut g, Bin::Sub, y[bb], y[a]);
                    let dx2 = b(&mut g, Bin::Sub, x[d], x[cc]); let dy2 = b(&mut g, Bin::Sub, y[d], y[cc]);
                    let (t1, t2) = if matches!(c, Con::Parallel(_)) { (b(&mut g, Bin::Mul, dx1, dy2), b(&mut g, Bin::Mul, dy1, dx2)) } else { (b(&mut g, Bin::Mul, dx1, dx2), { let m = b(&mut g, Bin::Mul, dy1, dy2); g.un(Un::Neg, m).unwrap() }) };
                    out.push(b(&mut g, Bin::Sub, t1, t2));
                }
                Con::Distance(p, q, d) => {
                    let dx = b(&mut g, Bin::Sub, x[q], x[p]); let dy = b(&mut g, Bin::Sub, y[q], y[p]);
                    let s = { let a = b(&mut g, Bin::Mul, dx, dx); let c2 = b(&mut g, Bin::Mul, dy, dy); b(&mut g, Bin::Add, a, c2) };
                    let len = g.un(Un::Sqrt, s).unwrap();
                    let k = g.konst(d, Dim::NONE);
                    out.push(b(&mut g, Bin::Sub, len, k));
                }
                Con::Radius(ri, v) => { let k = g.konst(v, Dim::NONE); out.push(b(&mut g, Bin::Sub, r[ri], k)); }
                Con::Fix(p, at) => { let kx = g.konst(at[0], Dim::NONE); let ky = g.konst(at[1], Dim::NONE); out.push(b(&mut g, Bin::Sub, x[p], kx)); out.push(b(&mut g, Bin::Sub, y[p], ky)); }
                Con::Equal([a, bb, cc, d]) => {
                    let l = |g: &mut ExprGraph, p: usize, q: usize| { let dx = g.bin(Bin::Sub, x[q], x[p]).unwrap(); let dy = g.bin(Bin::Sub, y[q], y[p]).unwrap(); let a = g.bin(Bin::Mul, dx, dx).unwrap(); let c = g.bin(Bin::Mul, dy, dy).unwrap(); g.bin(Bin::Add, a, c).unwrap() };
                    let l1 = l(&mut g, a, bb); let l2 = l(&mut g, cc, d);
                    out.push(b(&mut g, Bin::Sub, l1, l2));
                }
                Con::Concentric(p, q) => { out.push(b(&mut g, Bin::Sub, x[p], x[q])); out.push(b(&mut g, Bin::Sub, y[p], y[q])); }
                Con::Tangent { a, b: bb, c, r: ri } => {
                    // distance(center, line)^2 * |ab|^2 - r^2 |ab|^2 = 0  (cross product form)
                    let dx = b(&mut g, Bin::Sub, x[bb], x[a]); let dy = b(&mut g, Bin::Sub, y[bb], y[a]);
                    let ex = b(&mut g, Bin::Sub, x[c], x[a]); let ey = b(&mut g, Bin::Sub, y[c], y[a]);
                    let cr = { let p1 = b(&mut g, Bin::Mul, dx, ey); let p2 = b(&mut g, Bin::Mul, dy, ex); b(&mut g, Bin::Sub, p1, p2) };
                    let l2 = { let a2 = b(&mut g, Bin::Mul, dx, dx); let c2 = b(&mut g, Bin::Mul, dy, dy); b(&mut g, Bin::Add, a2, c2) };
                    let cr2 = b(&mut g, Bin::Mul, cr, cr);
                    let rr = b(&mut g, Bin::Mul, r[ri], r[ri]);
                    let rl = b(&mut g, Bin::Mul, rr, l2);
                    out.push(b(&mut g, Bin::Sub, cr2, rl));
                }
            }
        }
        g.outputs = out;
        let n = self.pts.len() * 2 + self.rads.len();
        if n > nls::MAX_VARS || g.outputs.len() > nls::MAX_RES {
            return Err(FeatureError::new(AgStatus::CapacityExceeded, format!("sketch too large for one dense block ({n} unknowns); solve plans via constraint.analyze land with block decomposition")));
        }
        let compiled = nls::compile(&g).map_err(|s| FeatureError::new(s, "constraint system compile failed"))?;
        let mut x0: Vec<f64> = self.pts.iter().map(|p| p[0]).collect();
        x0.extend(self.pts.iter().map(|p| p[1]));
        x0.extend(self.rads.iter().copied());
        let rep = nls::lm_solve(&compiled, &x0, &[], nls::LmOptions::default());
        let np = self.pts.len();
        for i in 0..np { self.pts[i] = [rep.x[i], rep.x[np + i]]; }
        for i in 0..self.rads.len() { self.rads[i] = rep.x[2 * np + i]; }
        Ok(rep)
    }
}

pub fn sketch_of(ctx: &BuildContext<'_>, v: &Value) -> Result<SketchData, FeatureError> {
    match ctx.product(v)? {
        Product::Other(o) => o.downcast_ref::<SketchData>().cloned().ok_or_else(|| bad("expected a sketch")),
        _ => Err(bad("expected a sketch")),
    }
}

fn newest(ctx: &BuildContext<'_>, n: &FeatureNode, keys: &[&str]) -> Result<SketchData, FeatureError> {
    let mut best: Option<SketchData> = None;
    for k in keys {
        if let Some(v) = n.args.get(*k) {
            if let Ok(s) = sketch_of(ctx, v) {
                if best.as_ref().map(|b| s.ents.len() + s.cons.len() > b.ents.len() + b.cons.len()).unwrap_or(true) { best = Some(s); }
            }
        }
    }
    best.ok_or_else(|| bad(format!("{}: no sketch input", n.op)))
}

fn elem(ctx: &BuildContext<'_>, n: &FeatureNode, k: &str) -> Result<usize, FeatureError> {
    sketch_of(ctx, req(n, k)?)?.last.ok_or_else(|| bad(format!("{}: '{k}' is not a sketch element", n.op)))
}

pub struct SketchOp(pub &'static str);

impl FeatureImpl for SketchOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let wrap = |s: SketchData| Product::Other(Arc::new(s));
        match self.0 {
            "sketch.create" => {
                let f = frame(ctx, req(n, "support")?)?;
                Ok(wrap(SketchData { frame: f, pts: vec![], rads: vec![], ents: vec![], cons: vec![], last: None }))
            }
            "sketch.point" => { let mut s = sketch_of(ctx, req(n, "sketch")?)?; let p = vec2(ctx, req(n, "p")?)?; let i = s.point(p); Ok(wrap(s.push(Ent::Point(i)))) }
            "sketch.line" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                let (a, b) = (vec2(ctx, req(n, "a")?)?, vec2(ctx, req(n, "b")?)?);
                let construction = matches!(arg(n, "construction"), Some(Value::Bool(true)));
                let (ia, ib) = (s.point(a), s.point(b));
                Ok(wrap(s.push(Ent::Line { a: ia, b: ib, construction })))
            }
            "sketch.circle" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                let pts = list(req(n, "pts")?);
                let c = vec2(ctx, pts.first().ok_or_else(|| bad("circle needs a center"))?)?;
                let r = match arg(n, "r") { Some(v) => ctx.number(v)?, None => { let p = vec2(ctx, pts.get(1).ok_or_else(|| bad("circle needs r or a point"))?)?; (p[0] - c[0]).hypot(p[1] - c[1]) } };
                if !(r > 0.0) { return Err(bad("circle radius must be positive")); }
                let ic = s.point(c);
                s.rads.push(r);
                let ri = s.rads.len() - 1;
                Ok(wrap(s.push(Ent::Circle { c: ic, r: ri })))
            }
            "sketch.arc" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                let pts = list(req(n, "pts")?);
                if pts.len() < 3 { return Err(bad("arc needs center, start, end")); }
                let (c, a, b) = (vec2(ctx, &pts[0])?, vec2(ctx, &pts[1])?, vec2(ctx, &pts[2])?);
                let (ic, ia, ib) = (s.point(c), s.point(a), s.point(b));
                Ok(wrap(s.push(Ent::Arc { c: ic, a: ia, b: ib })))
            }
            "sketch.rectangle" | "sketch.profile" | "sketch.polygon" | "sketch.elongated_hole" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                let pts: Vec<[f64; 2]> = match self.0 {
                    "sketch.rectangle" => {
                        let p = list(req(n, "pts")?);
                        let (a, b) = (vec2(ctx, &p[0])?, vec2(ctx, p.get(1).ok_or_else(|| bad("rectangle needs two corners"))?)?);
                        vec![a, [b[0], a[1]], b, [a[0], b[1]]]
                    }
                    "sketch.polygon" => {
                        let c = vec2(ctx, req(n, "center")?)?;
                        let k = num(ctx, n, "sides")? as usize;
                        let r = num(ctx, n, "r")?;
                        if k < 3 { return Err(bad("polygon needs ≥ 3 sides")); }
                        (0..k).map(|i| { let a = core::f64::consts::TAU * i as f64 / k as f64; [c[0] + r * a.cos(), c[1] + r * a.sin()] }).collect()
                    }
                    "sketch.elongated_hole" => {
                        let p = list(req(n, "pts")?);
                        let (a, b) = (vec2(ctx, &p[0])?, vec2(ctx, &p[1])?);
                        let w = num(ctx, n, "width")? / 2.0;
                        let d = [(b[0] - a[0]), (b[1] - a[1])];
                        let l = d[0].hypot(d[1]);
                        let nrm = [-d[1] / l * w, d[0] / l * w];
                        let ang = d[1].atan2(d[0]);
                        let (p1, p2, p3, p4) = ([a[0] - nrm[0], a[1] - nrm[1]], [b[0] - nrm[0], b[1] - nrm[1]], [b[0] + nrm[0], b[1] + nrm[1]], [a[0] + nrm[0], a[1] + nrm[1]]);
                        let ids = [s.point(p1), s.point(p2), s.point(p3), s.point(p4), s.point(a), s.point(b)];
                        s.ents.push(Ent::Line { a: ids[0], b: ids[1], construction: false });
                        s.ents.push(Ent::Arc { c: ids[5], a: ids[1], b: ids[2] });
                        s.ents.push(Ent::Line { a: ids[2], b: ids[3], construction: false });
                        let _ = ang;
                        s.ents.push(Ent::Arc { c: ids[4], a: ids[3], b: ids[0] });
                        s.last = Some(s.ents.len() - 1);
                        return Ok(wrap(s));
                    }
                    _ => list(req(n, "pts")?).iter().map(|v| vec2(ctx, v)).collect::<Result<_, _>>()?,
                };
                let ids: Vec<usize> = pts.iter().map(|p| s.point(*p)).collect();
                for k in 0..ids.len() { s.ents.push(Ent::Line { a: ids[k], b: ids[(k + 1) % ids.len()], construction: false }); }
                s.last = Some(s.ents.len() - 1);
                Ok(wrap(s))
            }
            "sketch.solve" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                let rep = s.solve()?;
                if rep.status != AgStatus::Accepted { return Err(FeatureError::new(rep.status, format!("sketch solve: max residual {:.3e} after {} iterations", rep.max_residual, rep.iterations))); }
                Ok(wrap(s))
            }
            "sketch.regions" => {
                let mut s = sketch_of(ctx, req(n, "sketch")?)?;
                if !s.cons.is_empty() { let rep = s.solve()?; if rep.status != AgStatus::Accepted { return Err(FeatureError::new(rep.status, "sketch constraints did not converge")); } }
                Ok(Product::Other(Arc::new(s.framed_regions()?)))
            }
            _ => Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented", self.0))),
        }
    }
}

/// Closed regions together with the sketch frame they live in (input of pad/pocket/...).
#[derive(Clone, Debug, PartialEq)]
pub struct FramedRegions {
    pub frame: Frame,
    pub regions: Vec<Profile>,
    /// Original sketch-entity indices and endpoints, before loop chaining and normalization.
    pub source_edges: Vec<(usize, [f64; 2], [f64; 2])>,
}

impl SketchData {
    fn framed_regions(&self) -> Result<FramedRegions, FeatureError> {
        let source_edges = self.ents.iter().enumerate().filter_map(|(i, entity)| {
            match *entity {
                Ent::Line { a, b, construction: false } | Ent::Arc { a, b, .. } => {
                    Some((i, self.pts[a], self.pts[b]))
                }
                Ent::Circle { c, r } => {
                    let seam = [self.pts[c][0] + self.rads[r], self.pts[c][1]];
                    Some((i, seam, seam))
                }
                _ => None,
            }
        }).collect();
        Ok(FramedRegions { frame: self.frame, regions: self.regions()?, source_edges })
    }
}

pub fn regions_of(ctx: &BuildContext<'_>, v: &Value) -> Result<FramedRegions, FeatureError> {
    match ctx.product(v)? {
        Product::Other(o) => {
            if let Some(r) = o.downcast_ref::<FramedRegions>() { return Ok(r.clone()); }
            if let Some(s) = o.downcast_ref::<SketchData>() {
                let mut s = s.clone();
                if !s.cons.is_empty() { s.solve()?; }
                return s.framed_regions();
            }
            Err(bad("expected sketch regions"))
        }
        Product::Regions(r) => Ok(FramedRegions { frame: Frame::WORLD, regions: r.clone(), source_edges:Vec::new() }),
        _ => Err(bad("expected sketch regions")),
    }
}

pub struct ConstraintOp(pub &'static str);

impl FeatureImpl for ConstraintOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let mut s = newest(ctx, n, &["a", "b", "p", "line", "axis"])?;
        let line = |s: &SketchData, e: usize| s.line_pts(e).ok_or_else(|| bad("expected a line"));
        let center = |s: &SketchData, e: usize| match s.ents.get(e) { Some(Ent::Circle { c, .. }) | Some(Ent::Arc { c, .. }) => Ok(*c), Some(Ent::Point(p)) => Ok(*p), _ => Err(bad("expected a circle/arc/point")) };
        let pt = |s: &SketchData, e: usize| match s.ents.get(e) { Some(Ent::Point(p)) => Ok(*p), Some(Ent::Circle { c, .. }) => Ok(*c), Some(Ent::Line { a, .. }) => Ok(*a), _ => Err(bad("expected a point")) };
        let c = match self.0 {
            "constraint.coincidence" => { let (a, b) = (elem(ctx, n, "a")?, elem(ctx, n, "b")?); Con::Coincident(pt(&s, a)?, pt(&s, b)?) }
            "constraint.horizontal" => { let (a, b) = line(&s, elem(ctx, n, "a")?)?; Con::Horizontal(a, b) }
            "constraint.vertical" => { let (a, b) = line(&s, elem(ctx, n, "a")?)?; Con::Vertical(a, b) }
            "constraint.parallelism" | "constraint.perpendicularity" | "constraint.equal" => {
                let (a, b) = line(&s, elem(ctx, n, "a")?)?;
                let (c2, d) = line(&s, elem(ctx, n, "b")?)?;
                match self.0 { "constraint.parallelism" => Con::Parallel([a, b, c2, d]), "constraint.perpendicularity" => Con::Perpendicular([a, b, c2, d]), _ => Con::Equal([a, b, c2, d]) }
            }
            "constraint.distance" => {
                let (ea, eb) = (elem(ctx, n, "a")?, elem(ctx, n, "b")?);
                Con::Distance(pt(&s, ea)?, pt(&s, eb)?, num(ctx, n, "value")?)
            }
            "constraint.radius" | "constraint.diameter" => {
                let e = elem(ctx, n, "a")?;
                let v = num(ctx, n, "value")? * if self.0 == "constraint.diameter" { 0.5 } else { 1.0 };
                match s.ents.get(e) { Some(Ent::Circle { r, .. }) => Con::Radius(*r, v), _ => return Err(bad("radius/diameter needs a circle")) }
            }
            "constraint.fix" => { let e = elem(ctx, n, "a")?; let p = pt(&s, e)?; Con::Fix(p, s.pts[p]) }
            "constraint.concentricity" => { let (a, b) = (elem(ctx, n, "a")?, elem(ctx, n, "b")?); Con::Concentric(center(&s, a)?, center(&s, b)?) }
            "constraint.tangency" => {
                let (ea, eb) = (elem(ctx, n, "a")?, elem(ctx, n, "b")?);
                let (l, c) = if s.line_pts(ea).is_some() { (ea, eb) } else { (eb, ea) };
                let (a, b) = line(&s, l)?;
                match s.ents.get(c) { Some(Ent::Circle { c, r }) => Con::Tangent { a, b, c: *c, r: *r }, _ => return Err(bad("tangency supports line–circle in this build")) }
            }
            _ => return Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        };
        s.cons.push(c);
        Ok(Product::Other(Arc::new(s)))
    }
}
