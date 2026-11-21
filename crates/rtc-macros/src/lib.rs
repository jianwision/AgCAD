//! # agcad-rtc-macros
//!
//! `#[udf]`, `#[implicit]`, `#[wrangle]`, `#[check]`, `#[evaluation]`, `#[residual]` for
//! ahead-of-time standard features (spec 04). Each macro:
//! 1. rejects `unsafe fn` / `extern` immediately (full sandbox checks run in agcad-rtc when
//!    the standard library registers its sources at startup);
//! 2. keeps the function so it type-checks with `rustc` (AG-04-001);
//! 3. emits `__AGCAD_SRC_<NAME>`: the AgLang source text (attribute included) used by
//!    `udf.view_source` and by the runtime compiler to build the same IR as for model code.

use proc_macro::TokenStream;
use quote::{format_ident, quote};

fn expand(kind: &str, attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: proc_macro2::TokenStream = attr.into();
    let func: syn::ItemFn = match syn::parse(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    if let Some(u) = &func.sig.unsafety {
        return syn::Error::new_spanned(u, "AgLang: `unsafe fn` is not allowed").to_compile_error().into();
    }
    if let Some(a) = &func.sig.abi {
        return syn::Error::new_spanned(a, "AgLang: `extern` functions are not allowed").to_compile_error().into();
    }
    let k = format_ident!("{}", kind);
    let src_attr = if attr2.is_empty() { quote!(#[#k]) } else { quote!(#[#k(#attr2)]) };
    let source = format!("{}\n{}", src_attr, quote!(#func));
    let konst = format_ident!("__AGCAD_SRC_{}", func.sig.ident.to_string().to_uppercase());
    // Strip `#[param(...)]` helper attributes so the function compiles as plain Rust.
    let mut clean = func.clone();
    for arg in clean.sig.inputs.iter_mut() {
        if let syn::FnArg::Typed(pt) = arg {
            pt.attrs.retain(|a| !a.path().is_ident("param"));
        }
    }
    quote! {
        #[allow(dead_code, unused_variables)]
        #clean
        #[doc(hidden)]
        pub const #konst: &str = #source;
    }
    .into()
}

#[proc_macro_attribute]
pub fn udf(attr: TokenStream, item: TokenStream) -> TokenStream { expand("udf", attr, item) }
#[proc_macro_attribute]
pub fn implicit(attr: TokenStream, item: TokenStream) -> TokenStream { expand("implicit", attr, item) }
#[proc_macro_attribute]
pub fn wrangle(attr: TokenStream, item: TokenStream) -> TokenStream { expand("wrangle", attr, item) }
#[proc_macro_attribute]
pub fn check(attr: TokenStream, item: TokenStream) -> TokenStream { expand("check", attr, item) }
#[proc_macro_attribute]
pub fn evaluation(attr: TokenStream, item: TokenStream) -> TokenStream { expand("evaluation", attr, item) }
#[proc_macro_attribute]
pub fn residual(attr: TokenStream, item: TokenStream) -> TokenStream { expand("residual", attr, item) }
