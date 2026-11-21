//! # agcad-rtc — runtime compiler (cf. NVRTC)
//!
//! AgLang is a strict subset of Rust (AG-04-001), parsed at runtime with `syn`:
//! * [`verify`]: sandbox rules (AG-04-002).
//! * [`lower`]: functions → [`expr::ExprGraph`] (implicit bodies, wrangles, evaluations,
//!   residuals) or [`lower::ProgramTemplate`] (UDFs, checks).
//! * [`autodiff`]: forward/reverse derivatives; discontinuities reported (AG-04-005).
//! * [`codegen`]: CUDA C++ kernels vectorized over `[configuration, sample]`.
//! * [`diag`]: rustc-style diagnostics with "did you mean" help.

pub mod autodiff;
pub mod codegen;
pub mod diag;
pub mod expr;
pub mod lower;
pub mod module;
pub mod verify;

pub use diag::Diagnostic;
pub use expr::{Dim, ExprGraph};
pub use module::{compile_module, CompiledItem, ItemKind, Module};

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
use agcad::prelude::*;

#[implicit(bounds = "auto", lipschitz = 1.0)]
fn gyroid_core(p: Vec3, #[param(default = 8.0*mm)] cell: Length, #[param(default = 0.8*mm)] wall: Length) -> Length {
    let k = TAU / cell;
    let g = (k*p.x).sin()*(k*p.y).cos() + (k*p.y).sin()*(k*p.z).cos() + (k*p.z).sin()*(k*p.x).cos();
    g.abs() / k - wall / 2.0
}

#[udf(name = "bolted_flange", roles("disc", "bolt_hole"))]
fn bolted_flange(ctx: &Ctx, #[param(default = 120.0*mm)] od: Length, #[param(default = 12.0*mm)] t: Length) -> FeatureRef {
    let s = sketch::create(ctx.xy_plane());
    sketch::circle(s, "center_radius", [pt2(0.0, 0.0)], od / 2.0);
    let disc = part::pad(sketch::regions(s), t).role("disc");
    disc
}
"#;

    #[test]
    fn compiles_implicit_and_udf() {
        let m = compile_module(SRC, agcad_ir::Catalog::global()).expect("compiles");
        let g = m.get("gyroid_core").unwrap().expr.as_ref().unwrap();
        let d = g.eval(&[0.0, 0.0, 0.0], &[8.0, 0.8]);
        assert!((d[0] - (-0.4)).abs() < 1e-12); // g(0)=0 → -wall/2
        let t = m.get("bolted_flange").unwrap().template.as_ref().unwrap();
        assert!(t.stmts.iter().any(|s| s.op == "part.pad" && s.role.as_deref() == Some("disc")));
        assert_eq!(m.get("gyroid_core").unwrap().params[0].default, Some(8.0));
    }

    #[test]
    fn rejects_unsafe_and_unit_errors() {
        let bad = "#[implicit] fn f(p: Vec3) -> Length { unsafe { p.x } }";
        assert!(compile_module(bad, agcad_ir::Catalog::global()).is_err());
        let units = "#[implicit] fn f(p: Vec3) -> Length { p.x + 30.0 * deg }";
        let e = compile_module(units, agcad_ir::Catalog::global()).unwrap_err();
        assert!(e.iter().any(|d| d.message.contains("unit error")));
    }

    #[test]
    fn gradient_of_sphere_is_unit_normal() {
        let src = "#[implicit] fn s(p: Vec3, #[param(default = 5.0*mm)] r: Length) -> Length { length(p) - r }";
        let m = compile_module(src, agcad_ir::Catalog::global()).unwrap();
        let g = m.get("s").unwrap().expr.as_ref().unwrap();
        let j = autodiff::jacobian(g, &[3.0, 4.0, 0.0], &[5.0], autodiff::Wrt::Inputs);
        assert!((j.jacobian[0] - 0.6).abs() < 1e-12 && (j.jacobian[1] - 0.8).abs() < 1e-12);
    }
}
