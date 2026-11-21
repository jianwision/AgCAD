//! Module compilation: parse an AgLang source file, verify the sandbox, lower each
//! attributed function (spec 04). Entry point: [`compile_module`].

use crate::diag::Diagnostic;
use crate::expr::{Dim, ExprGraph};
use crate::lower::{type_dim, ExprLowerer, Lv, ProgramTemplate, UdfLowerer};
use agcad_ir::Catalog;
use serde::{Deserialize, Serialize};
use syn::spanned::Spanned;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ItemKind {
    Udf { roles: Vec<String> },
    Implicit { lipschitz: f64, bounds: String },
    Wrangle { over: String },
    Check { severity: String, phase: String },
    Evaluation { evidence: String },
    Residual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub dim: Dim,
    pub default: Option<f64>,
    pub range: Option<(f64, f64)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledItem {
    pub name: String,
    pub kind: ItemKind,
    pub params: Vec<ParamSpec>,
    pub expr: Option<ExprGraph>,
    pub template: Option<ProgramTemplate>,
    /// Exact source text of the function (for `udf.view_source`).
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub items: Vec<CompiledItem>,
    pub hash: u64,
}

impl Module {
    pub fn get(&self, name: &str) -> Option<&CompiledItem> { self.items.iter().find(|i| i.name == name) }
}

fn attr_kind(attrs: &[syn::Attribute], diags: &mut Vec<Diagnostic>) -> Option<ItemKind> {
    for a in attrs {
        let name = a.path().segments.last()?.ident.to_string();
        let mut kv: Vec<(String, String)> = Vec::new();
        if matches!(a.meta, syn::Meta::List(_)) {
            let _ = a.parse_nested_meta(|m| {
                let key = m.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
                if m.input.peek(syn::Token![=]) {
                    let v: syn::Expr = m.value()?.parse()?;
                    let s = match &v { syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) => s.value(), other => quote::quote!(#other).to_string() };
                    kv.push((key, s));
                } else if m.input.peek(syn::token::Paren) {
                    let content;
                    syn::parenthesized!(content in m.input);
                    let list: syn::punctuated::Punctuated<syn::LitStr, syn::Token![,]> = content.parse_terminated(|p| p.parse::<syn::LitStr>(), syn::Token![,])?;
                    kv.push((key, list.iter().map(|l| l.value()).collect::<Vec<_>>().join(",")));
                } else {
                    kv.push((key, String::new()));
                }
                Ok(())
            });
        }
        let get = |k: &str| kv.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        let kind = match name.as_str() {
            "udf" => ItemKind::Udf { roles: get("roles").map(|r| r.split(',').map(str::to_string).filter(|s| !s.is_empty()).collect()).unwrap_or_default() },
            "implicit" => ItemKind::Implicit { lipschitz: get("lipschitz").and_then(|v| v.parse().ok()).unwrap_or(1.0), bounds: get("bounds").unwrap_or_else(|| "auto".into()) },
            "wrangle" => ItemKind::Wrangle { over: get("over").unwrap_or_else(|| "point".into()) },
            "check" => ItemKind::Check { severity: get("severity").unwrap_or_else(|| "error".into()), phase: get("phase").unwrap_or_else(|| "save".into()) },
            "evaluation" => ItemKind::Evaluation { evidence: get("evidence").unwrap_or_else(|| "sampled".into()) },
            "residual" => ItemKind::Residual,
            _ => continue,
        };
        if kv.iter().any(|(k, _)| k.is_empty()) {
            diags.push(Diagnostic::compile(a.span(), "malformed attribute arguments"));
        }
        return Some(kind);
    }
    None
}

/// Constant-fold a parameter default/range expression (`8.0*mm`).
fn const_eval(e: &syn::Expr, diags: &mut Vec<Diagnostic>) -> Option<f64> {
    let mut g = ExprGraph::default();
    let mut lw = ExprLowerer::new(&mut g);
    let v = lw.expr(e);
    diags.extend(lw.diags.drain(..));
    let Lv::S(id) = v else { return None };
    g.outputs = vec![id];
    if !g.input_names.is_empty() || !g.param_names.is_empty() { return None; }
    g.eval(&[], &[]).first().copied().filter(|x| x.is_finite())
}

fn params_of(f: &syn::ItemFn, skip_first: bool, diags: &mut Vec<Diagnostic>) -> Vec<ParamSpec> {
    let mut out = Vec::new();
    for (i, arg) in f.sig.inputs.iter().enumerate() {
        if skip_first && i == 0 { continue; }
        let syn::FnArg::Typed(pt) = arg else { continue };
        let name = match &*pt.pat { syn::Pat::Ident(id) => id.ident.to_string(), _ => { diags.push(Diagnostic::compile(pt.span(), "parameters must be simple identifiers")); continue; } };
        if name == "ctx" { continue; }
        let Some(dim) = type_dim(&pt.ty) else {
            diags.push(Diagnostic::compile(pt.ty.span(), "parameter type must be Length, Angle, Mass, Time, f64, u32, i32 or bool"));
            continue;
        };
        let mut spec = ParamSpec { name, dim, default: None, range: None };
        for a in &pt.attrs {
            if !a.path().is_ident("param") { continue; }
            let _ = a.parse_nested_meta(|m| {
                let key = m.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
                let v: syn::Expr = m.value()?.parse()?;
                match (key.as_str(), &v) {
                    ("default", e) => spec.default = const_eval(e, diags),
                    ("range", syn::Expr::Range(r)) => {
                        let lo = r.start.as_ref().and_then(|x| const_eval(x, diags));
                        let hi = r.end.as_ref().and_then(|x| const_eval(x, diags));
                        if let (Some(lo), Some(hi)) = (lo, hi) { spec.range = Some((lo, hi)); }
                    }
                    _ => {}
                }
                Ok(())
            });
        }
        out.push(spec);
    }
    out
}

/// Compile a module. Returns all diagnostics on failure (AG-04-002 / `COMPILE_FAILED`).
pub fn compile_module(source: &str, catalog: &Catalog) -> Result<Module, Vec<Diagnostic>> {
    let file: syn::File = syn::parse_file(source).map_err(|e| vec![Diagnostic::compile(e.span(), format!("syntax error: {e}"))])?;
    let mut diags = crate::verify::sandbox(&file);
    let mut items = Vec::new();
    for it in &file.items {
        let syn::Item::Fn(f) = it else { continue };
        let Some(kind) = attr_kind(&f.attrs, &mut diags) else { continue };
        let name = f.sig.ident.to_string();
        let text = quote::quote!(#f).to_string();
        let item = match &kind {
            ItemKind::Implicit { .. } | ItemKind::Wrangle { .. } | ItemKind::Evaluation { .. } | ItemKind::Residual => {
                let point_input = matches!(kind, ItemKind::Implicit { .. } | ItemKind::Wrangle { .. });
                let params = params_of(f, point_input, &mut diags);
                let mut g = ExprGraph::default();
                let mut lw = ExprLowerer::new(&mut g);
                match &kind {
                    ItemKind::Implicit { .. } => {
                        let pname = f.sig.inputs.first().and_then(|a| if let syn::FnArg::Typed(t) = a { if let syn::Pat::Ident(i) = &*t.pat { Some(i.ident.to_string()) } else { None } } else { None }).unwrap_or_else(|| "p".into());
                        let x = lw.g.input("p.x", Dim::LENGTH); let y = lw.g.input("p.y", Dim::LENGTH); let z = lw.g.input("p.z", Dim::LENGTH);
                        lw.vars.insert(pname, Lv::V(x, y, z));
                    }
                    ItemKind::Wrangle { .. } => {
                        let pname = f.sig.inputs.first().and_then(|a| if let syn::FnArg::Typed(t) = a { if let syn::Pat::Ident(i) = &*t.pat { Some(i.ident.to_string()) } else { None } } else { None }).unwrap_or_else(|| "pt".into());
                        let p = (lw.g.input("P.x", Dim::LENGTH), lw.g.input("P.y", Dim::LENGTH), lw.g.input("P.z", Dim::LENGTH));
                        let n = (lw.g.input("N.x", Dim::NONE), lw.g.input("N.y", Dim::NONE), lw.g.input("N.z", Dim::NONE));
                        lw.vars.insert(format!("{pname}.p"), Lv::V(p.0, p.1, p.2));
                        lw.vars.insert(format!("{pname}.n"), Lv::V(n.0, n.1, n.2));
                    }
                    _ => {}
                }
                for p in &params {
                    let id = lw.g.param(&p.name, p.dim);
                    lw.vars.insert(p.name.clone(), Lv::S(id));
                }
                let out = lw.block(&f.block);
                let outputs = if let ItemKind::Wrangle { .. } = kind {
                    let pname = lw.vars.keys().find(|k| k.ends_with(".p")).map(|k| k.trim_end_matches(".p").to_string()).unwrap_or_else(|| "pt".into());
                    let mut o = Vec::new();
                    for f2 in ["p", "n"] {
                        if let Some(Lv::V(a, b, c)) = lw.vars.get(&format!("{pname}.{f2}")).copied() { o.extend([a, b, c]); }
                    }
                    o
                } else {
                    match out { Lv::S(s) => vec![s], Lv::V(a, b, c) => vec![a, b, c] }
                };
                diags.extend(lw.diags.drain(..));
                g.outputs = outputs;
                if let ItemKind::Implicit { .. } = kind {
                    if g.outputs.len() != 1 || g.dim(g.outputs[0]) != Dim::LENGTH {
                        diags.push(Diagnostic::compile(f.sig.span(), "an #[implicit] function must return a Length (signed distance)"));
                    }
                }
                CompiledItem { name, kind, params, expr: Some(g), template: None, source: text }
            }
            ItemKind::Udf { .. } | ItemKind::Check { .. } => {
                let params = params_of(f, false, &mut diags);
                let mut ul = UdfLowerer::new(catalog);
                ul.t.params = params.iter().map(|p| (p.name.clone(), p.dim, p.default)).collect();
                ul.body(&f.block);
                diags.extend(ul.diags.drain(..));
                CompiledItem { name, kind, params, expr: None, template: Some(ul.t), source: text }
            }
        };
        items.push(item);
    }
    if diags.iter().any(|d| d.severity == crate::diag::Severity::Error) {
        return Err(diags);
    }
    Ok(Module { items, hash: agcad_ir::catalog::fnv1a64(source.as_bytes()) })
}
