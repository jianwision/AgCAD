//! Sandbox verifier on the `syn` AST (AG-04-002, AG-17-001). Rejects `unsafe`, macros,
//! unbounded loops, statics, raw pointers, extern items and non-prelude `use`.

use crate::diag::{Diagnostic, Severity};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

pub struct SandboxVisitor {
    pub diags: Vec<Diagnostic>,
}

fn has_attr(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|a| a.path().is_ident(name))
}

impl SandboxVisitor {
    fn deny(&mut self, span: proc_macro2::Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::sandbox(span, msg));
    }
}

impl<'ast> Visit<'ast> for SandboxVisitor {
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        if f.sig.unsafety.is_some() {
            self.deny(f.sig.span(), "`unsafe fn` is not allowed in AgLang");
        }
        if f.sig.abi.is_some() {
            self.deny(f.sig.span(), "`extern` functions are not allowed in AgLang");
        }
        visit::visit_item_fn(self, f);
    }
    fn visit_expr_unsafe(&mut self, e: &'ast syn::ExprUnsafe) {
        self.deny(e.span(), "`unsafe` blocks are not allowed in AgLang");
    }
    fn visit_expr_loop(&mut self, e: &'ast syn::ExprLoop) {
        self.deny(e.span(), "`loop` is unbounded; use `for i in 0..N` or `#[max_iter = N] while`");
        visit::visit_expr_loop(self, e);
    }
    fn visit_expr_while(&mut self, e: &'ast syn::ExprWhile) {
        if !has_attr(&e.attrs, "max_iter") {
            self.deny(e.span(), "`while` needs a `#[max_iter = N]` bound");
        }
        visit::visit_expr_while(self, e);
    }
    fn visit_expr_macro(&mut self, e: &'ast syn::ExprMacro) {
        self.deny(e.span(), "macros are not allowed in AgLang");
    }
    fn visit_stmt_macro(&mut self, s: &'ast syn::StmtMacro) {
        self.deny(s.span(), "macros are not allowed in AgLang");
    }
    fn visit_item_static(&mut self, s: &'ast syn::ItemStatic) {
        self.deny(s.span(), "statics are not allowed in AgLang (pure functions only)");
    }
    fn visit_item_foreign_mod(&mut self, s: &'ast syn::ItemForeignMod) {
        self.deny(s.span(), "`extern` blocks are not allowed in AgLang");
    }
    fn visit_type_ptr(&mut self, t: &'ast syn::TypePtr) {
        self.deny(t.span(), "raw pointers are not allowed in AgLang");
    }
    fn visit_item_use(&mut self, u: &'ast syn::ItemUse) {
        let txt = quote::quote!(#u).to_string().replace(' ', "");
        if !(txt.starts_with("useagcad::") || txt.starts_with("pubuseagcad::")) {
            self.deny(u.span(), "only `use agcad::...` imports are allowed (no std, IO or external crates)");
        }
    }
    fn visit_item_mod(&mut self, m: &'ast syn::ItemMod) {
        if m.content.is_none() {
            self.deny(m.span(), "out-of-line modules are not allowed");
        }
        visit::visit_item_mod(self, m);
    }
}

pub fn sandbox(file: &syn::File) -> Vec<Diagnostic> {
    let mut v = SandboxVisitor { diags: Vec::new() };
    v.visit_file(file);
    for d in &mut v.diags {
        d.severity = Severity::Error;
    }
    v.diags
}
