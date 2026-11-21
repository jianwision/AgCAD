//! AgLang → IR lowering.
//!
//! * `#[implicit]`, `#[wrangle]`, `#[evaluation]` functions → [`ExprGraph`] (pure, per-sample).
//! * `#[udf]`, `#[check]` functions → [`ProgramTemplate`]: a sequence of feature-table calls
//!   whose scalar arguments are expression graphs over the UDF parameters (evaluated per
//!   configuration by the runtime, so the whole UDF vectorizes over the design table).

use crate::diag::Diagnostic;
use crate::expr::*;
use agcad_ir::program::Value;
use agcad_ir::Catalog;
use std::collections::{BTreeMap, HashMap};
use syn::spanned::Spanned;

/// Unit identifiers usable in literals (`8.0 * mm`): (name, scale to internal, dimension).
const UNIT_IDENTS: &[(&str, f64, Dim)] = &[
    ("mm", 1.0, Dim::LENGTH),
    ("cm", 10.0, Dim::LENGTH),
    ("m", 1000.0, Dim::LENGTH),
    ("um", 1e-3, Dim::LENGTH),
    ("inch", 25.4, Dim::LENGTH),
    ("rad", 1.0, Dim::ANGLE),
    ("deg", core::f64::consts::PI / 180.0, Dim::ANGLE),
    ("kg", 1.0, Dim::MASS),
    ("g", 1e-3, Dim::MASS),
    ("s", 1.0, Dim::TIME),
];
const CONSTS: &[(&str, f64)] = &[("PI", core::f64::consts::PI), ("TAU", core::f64::consts::TAU), ("E", core::f64::consts::E)];

pub fn type_dim(ty: &syn::Type) -> Option<Dim> {
    let s = quote::quote!(#ty).to_string().replace(' ', "");
    Some(match s.as_str() {
        "Length" => Dim::LENGTH,
        "Angle" => Dim::ANGLE,
        "Mass" => Dim::MASS,
        "Time" => Dim::TIME,
        "f64" | "f32" | "Scalar" | "u32" | "i32" | "usize" | "bool" => Dim::NONE,
        _ => return None,
    })
}

/// A lowered value: scalar lane(s) in the graph.
#[derive(Clone, Copy, Debug)]
pub enum Lv {
    S(u32),
    V(u32, u32, u32),
}

pub struct ExprLowerer<'g> {
    pub g: &'g mut ExprGraph,
    pub vars: HashMap<String, Lv>,
    pub diags: Vec<Diagnostic>,
}

impl<'g> ExprLowerer<'g> {
    pub fn new(g: &'g mut ExprGraph) -> Self { ExprLowerer { g, vars: HashMap::new(), diags: Vec::new() } }

    fn err(&mut self, span: proc_macro2::Span, msg: impl Into<String>) -> Lv {
        self.diags.push(Diagnostic::compile(span, msg));
        Lv::S(self.g.konst(f64::NAN, Dim::NONE))
    }

    fn scalar(&mut self, span: proc_macro2::Span, v: Lv) -> u32 {
        match v {
            Lv::S(s) => s,
            Lv::V(..) => match self.err(span, "expected a scalar, found a Vec3") { Lv::S(s) => s, _ => unreachable!() },
        }
    }

    fn dimres(&mut self, span: proc_macro2::Span, r: Result<u32, DimError>) -> u32 {
        match r {
            Ok(i) => i,
            Err(e) => match self.err(span, format!("unit error: {}", e.message)) { Lv::S(s) => s, _ => unreachable!() },
        }
    }

    fn binop(&mut self, span: proc_macro2::Span, op: Bin, a: Lv, b: Lv) -> Lv {
        match (a, b) {
            (Lv::S(x), Lv::S(y)) => Lv::S({ let r = self.g.bin(op, x, y); self.dimres(span, r) }),
            (Lv::V(a0, a1, a2), Lv::V(b0, b1, b2)) if matches!(op, Bin::Add | Bin::Sub | Bin::Mul | Bin::Min | Bin::Max) => {
                let r0 = self.g.bin(op, a0, b0); let r1 = self.g.bin(op, a1, b1); let r2 = self.g.bin(op, a2, b2);
                Lv::V(self.dimres(span, r0), self.dimres(span, r1), self.dimres(span, r2))
            }
            (Lv::V(a0, a1, a2), Lv::S(y)) if matches!(op, Bin::Mul | Bin::Div) => {
                let r0 = self.g.bin(op, a0, y); let r1 = self.g.bin(op, a1, y); let r2 = self.g.bin(op, a2, y);
                Lv::V(self.dimres(span, r0), self.dimres(span, r1), self.dimres(span, r2))
            }
            (Lv::S(x), Lv::V(b0, b1, b2)) if op == Bin::Mul => {
                let r0 = self.g.bin(op, x, b0); let r1 = self.g.bin(op, x, b1); let r2 = self.g.bin(op, x, b2);
                Lv::V(self.dimres(span, r0), self.dimres(span, r1), self.dimres(span, r2))
            }
            _ => self.err(span, format!("operator {op:?} not defined for these operand shapes")),
        }
    }

    fn unop(&mut self, span: proc_macro2::Span, op: Un, a: Lv) -> Lv {
        match a {
            Lv::S(x) => Lv::S({ let r = self.g.un(op, x); self.dimres(span, r) }),
            Lv::V(a0, a1, a2) if matches!(op, Un::Neg | Un::Abs | Un::Floor | Un::Ceil) => {
                let r0 = self.g.un(op, a0); let r1 = self.g.un(op, a1); let r2 = self.g.un(op, a2);
                Lv::V(self.dimres(span, r0), self.dimres(span, r1), self.dimres(span, r2))
            }
            _ => self.err(span, format!("{op:?} is not defined for Vec3")),
        }
    }

    fn length(&mut self, span: proc_macro2::Span, v: Lv) -> Lv {
        match v {
            Lv::V(x, y, z) => {
                let xx = self.binop(span, Bin::Mul, Lv::S(x), Lv::S(x));
                let yy = self.binop(span, Bin::Mul, Lv::S(y), Lv::S(y));
                let zz = self.binop(span, Bin::Mul, Lv::S(z), Lv::S(z));
                let s = self.binop(span, Bin::Add, xx, yy);
                let s = self.binop(span, Bin::Add, s, zz);
                self.unop(span, Un::Sqrt, s)
            }
            s => self.unop(span, Un::Abs, s),
        }
    }

    pub fn expr(&mut self, e: &syn::Expr) -> Lv {
        let span = e.span();
        match e {
            syn::Expr::Lit(l) => match &l.lit {
                syn::Lit::Float(f) => Lv::S(self.g.konst(f.base10_parse::<f64>().unwrap_or(f64::NAN), Dim::NONE)),
                syn::Lit::Int(i) => Lv::S(self.g.konst(i.base10_parse::<f64>().unwrap_or(f64::NAN), Dim::NONE)),
                syn::Lit::Bool(b) => Lv::S(self.g.konst(b.value as i32 as f64, Dim::NONE)),
                _ => self.err(span, "only numeric and boolean literals are allowed in expressions"),
            },
            syn::Expr::Paren(p) => self.expr(&p.expr),
            syn::Expr::Group(g) => self.expr(&g.expr),
            syn::Expr::Path(p) => {
                let name = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
                if let Some(v) = self.vars.get(&name) { return *v; }
                if let Some((_, k, d)) = UNIT_IDENTS.iter().find(|(n, _, _)| *n == name) { return Lv::S(self.g.konst(*k, *d)); }
                if let Some((_, k)) = CONSTS.iter().find(|(n, _)| *n == name) { return Lv::S(self.g.konst(*k, Dim::NONE)); }
                self.err(span, format!("unknown identifier `{name}`"))
            }
            syn::Expr::Field(f) => {
                // record fields bound as `var.field` (wrangle point attributes)
                if let (syn::Expr::Path(bp), syn::Member::Named(m)) = (&*f.base, &f.member) {
                    if let Some(b) = bp.path.segments.last() {
                        if let Some(v) = self.vars.get(&format!("{}.{}", b.ident, m)) { return *v; }
                    }
                }
                let base = self.expr(&f.base);
                let member = match &f.member { syn::Member::Named(i) => i.to_string(), syn::Member::Unnamed(i) => i.index.to_string() };
                match (base, member.as_str()) {
                    (Lv::V(x, _, _), "x" | "0") => Lv::S(x),
                    (Lv::V(_, y, _), "y" | "1") => Lv::S(y),
                    (Lv::V(_, _, z), "z" | "2") => Lv::S(z),
                    _ => {
                        // Attribute-style access `pt.p`, `pt.n` resolved through flattened var names.
                        if let syn::Expr::Path(pp) = &*f.base {
                            let key = format!("{}.{}", pp.path.segments.last().unwrap().ident, member);
                            if let Some(v) = self.vars.get(&key) { return *v; }
                        }
                        self.err(span, format!("unknown field `{member}`"))
                    }
                }
            }
            syn::Expr::Unary(u) => {
                let a = self.expr(&u.expr);
                match u.op {
                    syn::UnOp::Neg(_) => self.unop(span, Un::Neg, a),
                    syn::UnOp::Not(_) => self.unop(span, Un::Not, a),
                    _ => self.err(span, "dereference is not allowed"),
                }
            }
            syn::Expr::Binary(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                let op = match b.op {
                    syn::BinOp::Add(_) => Bin::Add, syn::BinOp::Sub(_) => Bin::Sub, syn::BinOp::Mul(_) => Bin::Mul,
                    syn::BinOp::Div(_) => Bin::Div, syn::BinOp::Rem(_) => Bin::Mod, syn::BinOp::Lt(_) => Bin::Lt,
                    syn::BinOp::Le(_) => Bin::Le, syn::BinOp::Gt(_) => Bin::Gt, syn::BinOp::Ge(_) => Bin::Ge,
                    syn::BinOp::Eq(_) => Bin::Eq, syn::BinOp::Ne(_) => Bin::Ne, syn::BinOp::And(_) => Bin::And,
                    syn::BinOp::Or(_) => Bin::Or,
                    _ => return self.err(span, "compound assignment is only allowed as a statement"),
                };
                self.binop(span, op, l, r)
            }
            syn::Expr::MethodCall(m) => {
                let recv = self.expr(&m.receiver);
                let args: Vec<Lv> = m.args.iter().map(|a| self.expr(a)).collect();
                let name = m.method.to_string();
                let un = match name.as_str() {
                    "abs" => Some(Un::Abs), "sqrt" => Some(Un::Sqrt), "sin" => Some(Un::Sin), "cos" => Some(Un::Cos),
                    "tan" => Some(Un::Tan), "asin" => Some(Un::Asin), "acos" => Some(Un::Acos), "atan" => Some(Un::Atan),
                    "exp" => Some(Un::Exp), "ln" => Some(Un::Ln), "floor" => Some(Un::Floor), "ceil" => Some(Un::Ceil),
                    _ => None,
                };
                if let Some(op) = un { return self.unop(span, op, recv); }
                match (name.as_str(), args.as_slice()) {
                    ("min", [b]) => self.binop(span, Bin::Min, recv, *b),
                    ("max", [b]) => self.binop(span, Bin::Max, recv, *b),
                    ("atan2", [b]) => self.binop(span, Bin::Atan2, recv, *b),
                    ("powi" | "powf", [b]) => self.binop(span, Bin::Pow, recv, *b),
                    ("length" | "norm", []) => self.length(span, recv),
                    ("clamp", [lo, hi]) => { let t = self.binop(span, Bin::Max, recv, *lo); self.binop(span, Bin::Min, t, *hi) }
                    ("dot", [b]) => self.dot(span, recv, *b),
                    _ => self.err(span, format!("method `{name}` is not in the AgLang prelude").to_string()),
                }
            }
            syn::Expr::Call(c) => {
                let fname = match &*c.func { syn::Expr::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default(), _ => String::new() };
                let args: Vec<Lv> = c.args.iter().map(|a| self.expr(a)).collect();
                match (fname.as_str(), args.as_slice()) {
                    ("vec3", [x, y, z]) => { let (x, y, z) = (self.scalar(span, *x), self.scalar(span, *y), self.scalar(span, *z)); Lv::V(x, y, z) }
                    ("length", [v]) => self.length(span, *v),
                    ("dot", [a, b]) => self.dot(span, *a, *b),
                    ("min", [a, b]) => self.binop(span, Bin::Min, *a, *b),
                    ("max", [a, b]) => self.binop(span, Bin::Max, *a, *b),
                    ("clamp", [x, lo, hi]) => { let t = self.binop(span, Bin::Max, *x, *lo); self.binop(span, Bin::Min, t, *hi) }
                    ("smooth_min", [a, b, k]) => self.smooth_min(span, *a, *b, *k),
                    ("noise3", [v]) => match v {
                        Lv::V(x, y, z) => Lv::S(self.g.push(Node::Noise(*x, *y, *z), Dim::NONE)),
                        _ => self.err(span, "noise3 takes a Vec3"),
                    },
                    _ => self.err(span, format!("function `{fname}` is not in the AgLang prelude")),
                }
            }
            syn::Expr::If(i) => {
                let c = self.expr(&i.cond);
                let c = self.scalar(span, c);
                let a = self.block(&i.then_branch);
                let b = match &i.else_branch {
                    Some((_, e)) => self.expr(e),
                    None => return self.err(span, "`if` used as a value needs an `else`"),
                };
                match (a, b) {
                    (Lv::S(x), Lv::S(y)) => Lv::S({ let r = self.g.select(c, x, y); self.dimres(span, r) }),
                    (Lv::V(a0, a1, a2), Lv::V(b0, b1, b2)) => {
                        let r0 = self.g.select(c, a0, b0); let r1 = self.g.select(c, a1, b1); let r2 = self.g.select(c, a2, b2);
                        Lv::V(self.dimres(span, r0), self.dimres(span, r1), self.dimres(span, r2))
                    }
                    _ => self.err(span, "if/else branches have different shapes"),
                }
            }
            syn::Expr::Block(b) => self.block(&b.block),
            syn::Expr::Cast(c) => self.expr(&c.expr),
            _ => self.err(span, "expression form not supported in AgLang"),
        }
    }

    fn dot(&mut self, span: proc_macro2::Span, a: Lv, b: Lv) -> Lv {
        match (a, b) {
            (Lv::V(a0, a1, a2), Lv::V(b0, b1, b2)) => {
                let x = self.binop(span, Bin::Mul, Lv::S(a0), Lv::S(b0));
                let y = self.binop(span, Bin::Mul, Lv::S(a1), Lv::S(b1));
                let z = self.binop(span, Bin::Mul, Lv::S(a2), Lv::S(b2));
                let s = self.binop(span, Bin::Add, x, y);
                self.binop(span, Bin::Add, s, z)
            }
            _ => self.err(span, "dot needs two Vec3"),
        }
    }

    /// Polynomial smooth minimum (Quilez): h = clamp(0.5 + 0.5(b-a)/k, 0, 1); mix(b, a, h) - k h (1-h).
    fn smooth_min(&mut self, span: proc_macro2::Span, a: Lv, b: Lv, k: Lv) -> Lv {
        let half = Lv::S(self.g.konst(0.5, Dim::NONE));
        let one = Lv::S(self.g.konst(1.0, Dim::NONE));
        let zero = Lv::S(self.g.konst(0.0, Dim::NONE));
        let d = self.binop(span, Bin::Sub, b, a);
        let q = self.binop(span, Bin::Div, d, k);
        let q = self.binop(span, Bin::Mul, half, q);
        let h = self.binop(span, Bin::Add, half, q);
        let h = self.binop(span, Bin::Max, h, zero);
        let h = self.binop(span, Bin::Min, h, one);
        let omh = self.binop(span, Bin::Sub, one, h);
        let bm = self.binop(span, Bin::Mul, b, omh);
        let am = self.binop(span, Bin::Mul, a, h);
        let mix = self.binop(span, Bin::Add, bm, am);
        let kh = self.binop(span, Bin::Mul, k, h);
        let khh = self.binop(span, Bin::Mul, kh, omh);
        self.binop(span, Bin::Sub, mix, khh)
    }

    /// Lower a block; `let` bindings, field compound-assignments (wrangles), final expression.
    pub fn block(&mut self, b: &syn::Block) -> Lv {
        let mut last = None;
        for st in &b.stmts {
            match st {
                syn::Stmt::Local(l) => {
                    let ident = match &l.pat { syn::Pat::Ident(i) => Some(&i.ident), syn::Pat::Type(t) => match &*t.pat { syn::Pat::Ident(i) => Some(&i.ident), _ => None }, _ => None };
                    let Some(ident) = ident else { self.err(l.span(), "only simple `let name = ...` bindings"); continue; };
                    let name = ident.to_string();
                    let v = match &l.init { Some(init) => self.expr(&init.expr), None => self.err(l.span(), "`let` needs an initializer") };
                    self.vars.insert(name, v);
                    last = None;
                }
                syn::Stmt::Expr(e, semi) => {
                    if let syn::Expr::Assign(a) = e {
                        self.assign(&a.left, None, &a.right);
                        last = None;
                    } else if let syn::Expr::Binary(bx) = e {
                        let op = match bx.op { syn::BinOp::AddAssign(_) => Some(Bin::Add), syn::BinOp::SubAssign(_) => Some(Bin::Sub), syn::BinOp::MulAssign(_) => Some(Bin::Mul), syn::BinOp::DivAssign(_) => Some(Bin::Div), _ => None };
                        if op.is_some() { self.assign(&bx.left, op, &bx.right); last = None; } else { let v = self.expr(e); last = if semi.is_none() { Some(v) } else { None }; }
                    } else if let syn::Expr::ForLoop(f) = e {
                        self.for_loop(f);
                        last = None;
                    } else {
                        let v = self.expr(e);
                        last = if semi.is_none() { Some(v) } else { None };
                    }
                }
                _ => { self.err(st.span(), "item statements are not allowed inside AgLang functions"); }
            }
        }
        last.unwrap_or_else(|| Lv::S(self.g.konst(0.0, Dim::NONE)))
    }

    fn assign(&mut self, lhs: &syn::Expr, op: Option<Bin>, rhs: &syn::Expr) {
        let key = match lhs {
            syn::Expr::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
            syn::Expr::Field(f) => match (&*f.base, &f.member) {
                (syn::Expr::Path(p), syn::Member::Named(m)) => format!("{}.{}", p.path.segments.last().unwrap().ident, m),
                _ => { self.err(lhs.span(), "assignment target must be `var` or `var.field`"); return; }
            },
            _ => { self.err(lhs.span(), "unsupported assignment target"); return; }
        };
        let r = self.expr(rhs);
        let v = match op {
            Some(o) => match self.vars.get(&key).copied() { Some(cur) => self.binop(rhs.span(), o, cur, r), None => self.err(lhs.span(), format!("`{key}` is not defined")) },
            None => r,
        };
        self.vars.insert(key, v);
    }

    /// `for i in 0..N { ... }` with literal bounds ≤ 64 is unrolled (bounded loops, AG-04-002).
    fn for_loop(&mut self, f: &syn::ExprForLoop) {
        let name = match &*f.pat { syn::Pat::Ident(i) => i.ident.to_string(), _ => { self.err(f.span(), "loop variable must be an identifier"); return; } };
        let (lo, hi) = match &*f.expr {
            syn::Expr::Range(r) => {
                let lit = |e: &Option<Box<syn::Expr>>| e.as_ref().and_then(|x| if let syn::Expr::Lit(l) = &**x { if let syn::Lit::Int(i) = &l.lit { i.base10_parse::<i64>().ok() } else { None } } else { None });
                (lit(&r.start).unwrap_or(0), lit(&r.end))
            }
            _ => (0, None),
        };
        let Some(hi) = hi else { self.err(f.span(), "`for` bounds must be integer literals (`0..N`)"); return; };
        if hi - lo > 64 { self.err(f.span(), "unrolled loops are limited to 64 iterations"); return; }
        for i in lo..hi {
            let k = self.g.konst(i as f64, Dim::NONE);
            self.vars.insert(name.clone(), Lv::S(k));
            self.block(&f.body);
        }
        self.vars.remove(&name);
    }
}

// ----------------------------------------------------------------------------------------
// UDF / check lowering → program template
// ----------------------------------------------------------------------------------------

/// One feature call inside a UDF body.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TemplateStmt {
    pub id: String,
    pub op: String,
    pub args: BTreeMap<String, Value>,
    pub role: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProgramTemplate {
    pub params: Vec<(String, Dim, Option<f64>)>,
    pub stmts: Vec<TemplateStmt>,
    /// Statement id (or param) returned by the UDF.
    pub result: Option<String>,
    /// Scalar expressions over params referenced by `Value::Map {"$expr": idx}`.
    pub exprs: Vec<ExprGraph>,
}

pub struct UdfLowerer<'c> {
    pub catalog: &'c Catalog,
    pub t: ProgramTemplate,
    pub locals: HashMap<String, Value>,
    pub diags: Vec<Diagnostic>,
    next: usize,
}

impl<'c> UdfLowerer<'c> {
    pub fn new(catalog: &'c Catalog) -> Self { UdfLowerer { catalog, t: ProgramTemplate::default(), locals: HashMap::new(), diags: Vec::new(), next: 0 } }

    fn param_dims(&self) -> Vec<(String, Dim)> { self.t.params.iter().map(|(n, d, _)| (n.clone(), *d)).collect() }

    /// Resolve `family::op` paths (with aliases) to a canonical catalog op.
    fn resolve_feature(&mut self, path: &syn::Path) -> Option<String> {
        let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if segs.len() < 2 { return None; }
        let fam = &segs[segs.len() - 2];
        let op = &segs[segs.len() - 1];
        if !self.catalog.families.contains_key(fam) { return None; }
        let canonical = format!("{fam}.{op}");
        if self.catalog.get(&canonical).is_some() { return Some(canonical); }
        let cands = self.catalog.resolve(op);
        let in_fam: Vec<_> = cands.iter().filter(|o| &o.family == fam).collect();
        if in_fam.len() == 1 { return Some(in_fam[0].name.clone()); }
        let sugg = self.catalog.suggest(&canonical, 3);
        self.diags.push(Diagnostic::compile(path.span(), format!("unknown feature `{fam}::{op}`")).with_help(format!("did you mean: {}", sugg.join(", "))));
        Some(canonical)
    }

    /// Lower an argument expression to a Value; arithmetic over params becomes a `$expr`.
    fn value(&mut self, e: &syn::Expr) -> Value {
        match e {
            syn::Expr::Lit(l) => match &l.lit {
                syn::Lit::Str(s) => Value::String(s.value()),
                syn::Lit::Bool(b) => Value::Bool(b.value),
                syn::Lit::Int(i) => Value::Int(i.base10_parse().unwrap_or(0)),
                syn::Lit::Float(f) => Value::Float(f.base10_parse().unwrap_or(f64::NAN)),
                _ => Value::Null,
            },
            syn::Expr::Reference(r) => self.value(&r.expr),
            syn::Expr::Paren(p) => self.value(&p.expr),
            syn::Expr::Array(a) => Value::List(a.elems.iter().map(|x| self.value(x)).collect()),
            syn::Expr::Path(p) if p.path.segments.len() == 1 => {
                let name = p.path.segments[0].ident.to_string();
                if let Some(v) = self.locals.get(&name) { return v.clone(); }
                if self.t.params.iter().any(|(n, _, _)| *n == name) { return Value::Ref { id: "$param".into(), field: Some(name) }; }
                Value::String(name) // enum-like bare identifiers
            }
            syn::Expr::Path(p) => Value::String(quote::quote!(#p).to_string().replace(' ', "")),
            syn::Expr::Call(c) => {
                if let syn::Expr::Path(p) = &*c.func {
                    if let Some(op) = self.resolve_feature(&p.path) {
                        let id = self.call(&op, &c.args.iter().collect::<Vec<_>>(), None);
                        return Value::Ref { id, field: None };
                    }
                    // helper constructors: pt2(..), Limits::dimension(..), HoleSpec::simple(..)
                    let name = quote::quote!(#p).to_string().replace(' ', "");
                    let mut m = BTreeMap::new();
                    m.insert("$ctor".into(), Value::String(name));
                    m.insert("args".into(), Value::List(c.args.iter().map(|x| self.value(x)).collect()));
                    return Value::Map(m);
                }
                Value::Null
            }
            syn::Expr::MethodCall(mc) => {
                // Builder-style modifiers on constructors/features: `.through()`, `.role("x")`.
                let base = self.value(&mc.receiver);
                let mut m = match base { Value::Map(m) => m, other => { let mut m = BTreeMap::new(); m.insert("$base".into(), other); m } };
                let args: Vec<Value> = mc.args.iter().map(|x| self.value(x)).collect();
                m.insert(mc.method.to_string(), if args.len() == 1 { args[0].clone() } else if args.is_empty() { Value::Bool(true) } else { Value::List(args) });
                Value::Map(m)
            }
            _ => {
                // Scalar arithmetic over parameters → expression graph evaluated per configuration.
                let mut g = ExprGraph::default();
                let pd = self.param_dims();
                let mut lw = ExprLowerer::new(&mut g);
                for (n, d) in pd {
                    let id = lw.g.param(&n, d);
                    lw.vars.insert(n, Lv::S(id));
                }
                let out = lw.expr(e);
                self.diags.extend(lw.diags.drain(..));
                g.outputs = match out { Lv::S(s) => vec![s], Lv::V(a, b, c) => vec![a, b, c] };
                self.t.exprs.push(g);
                let mut m = BTreeMap::new();
                m.insert("$expr".into(), Value::Int((self.t.exprs.len() - 1) as i64));
                Value::Map(m)
            }
        }
    }

    fn call(&mut self, op: &str, args: &[&syn::Expr], role: Option<String>) -> String {
        let id = format!("f{}", self.next);
        self.next += 1;
        let params: Vec<String> = self.catalog.get(op).map(|m| m.params().into_iter().map(|p| p.name).collect()).unwrap_or_default();
        let mut map = BTreeMap::new();
        for (i, a) in args.iter().enumerate() {
            let key = params.get(i).cloned().unwrap_or_else(|| format!("arg{i}"));
            let v = self.value(a);
            map.insert(key, v);
        }
        self.t.stmts.push(TemplateStmt { id: id.clone(), op: op.to_string(), args: map, role });
        id
    }

    /// Lower a statement-level feature call; supports trailing `.role("name")`.
    fn stmt_expr(&mut self, e: &syn::Expr) -> Option<Value> {
        match e {
            syn::Expr::MethodCall(mc) if mc.method == "role" => {
                let v = self.stmt_expr(&mc.receiver)?;
                if let (Value::Ref { id, .. }, Some(syn::Expr::Lit(l))) = (&v, mc.args.first()) {
                    if let syn::Lit::Str(s) = &l.lit {
                        if let Some(st) = self.t.stmts.iter_mut().find(|s2| &s2.id == id) { st.role = Some(s.value()); }
                    }
                }
                Some(v)
            }
            _ => Some(self.value(e)),
        }
    }

    pub fn body(&mut self, b: &syn::Block) {
        for st in &b.stmts {
            match st {
                syn::Stmt::Local(l) => {
                    if let (syn::Pat::Ident(i), Some(init)) = (&l.pat, &l.init) {
                        if let Some(v) = self.stmt_expr(&init.expr) { self.locals.insert(i.ident.to_string(), v); }
                    } else {
                        self.diags.push(Diagnostic::compile(l.span(), "only `let name = ...` bindings are supported in UDF bodies"));
                    }
                }
                syn::Stmt::Expr(e, semi) => {
                    let v = self.stmt_expr(e);
                    if semi.is_none() {
                        self.t.result = match v { Some(Value::Ref { id, .. }) => Some(id), _ => None };
                    }
                }
                _ => self.diags.push(Diagnostic::compile(st.span(), "unsupported statement in UDF body")),
            }
        }
    }
}
