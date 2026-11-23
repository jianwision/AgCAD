//! ExprGraph → CUDA C++ (Tier N), vectorized over `[configuration, sample]` (spec 04 §
//! Vectorization). Semantics match [`ExprGraph::eval`] op by op; NVRTC compiles with
//! `fmad=false`, so results are bit-identical to the host reference interpreter in agcad-vm.

use crate::expr::*;
use std::fmt::Write;

/// Value noise shared by codegen and the device VM (mirrors `expr::value_noise`).
pub const VALUE_NOISE_CUDA: &str = r#"
__device__ __forceinline__ double ag_noise_hash(long long i, long long j, long long k) {
    unsigned long long n = ((unsigned long long)i * 73856093ULL) ^ ((unsigned long long)j * 19349663ULL) ^ ((unsigned long long)k * 83492791ULL);
    n = (n ^ (n >> 33)) * 0xff51afd7ed558ccdULL;
    n = (n ^ (n >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    n ^= n >> 33;
    return (double)(n >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}
__device__ __forceinline__ double ag_value_noise(double x, double y, double z) {
    double xi = floor(x), yi = floor(y), zi = floor(z);
    double fx = x - xi, fy = y - yi, fz = z - zi;
    double u = fx * fx * (3.0 - 2.0 * fx), v = fy * fy * (3.0 - 2.0 * fy), w = fz * fz * (3.0 - 2.0 * fz);
    long long i = (long long)xi, j = (long long)yi, k = (long long)zi;
#define AG_LERP(a, b, t) ((a) + ((b) - (a)) * (t))
    double x00 = AG_LERP(ag_noise_hash(i, j, k), ag_noise_hash(i + 1, j, k), u);
    double x10 = AG_LERP(ag_noise_hash(i, j + 1, k), ag_noise_hash(i + 1, j + 1, k), u);
    double x01 = AG_LERP(ag_noise_hash(i, j, k + 1), ag_noise_hash(i + 1, j, k + 1), u);
    double x11 = AG_LERP(ag_noise_hash(i, j + 1, k + 1), ag_noise_hash(i + 1, j + 1, k + 1), u);
    return AG_LERP(AG_LERP(x00, x10, v), AG_LERP(x01, x11, v), w);
#undef AG_LERP
}
"#;

/// Parameter block for generated expression kernels (mirrored in agcad-vm).
pub const EXPR_PARAMS_CUDA: &str = r#"
struct AgExprParams {
    unsigned long long inputs;   // double[n_inputs][(per_config ? configs : 1) * samples]
    unsigned long long params;   // double[n_params][configs]
    unsigned long long out;      // double[n_outputs][configs * samples]
    unsigned int samples, configs, n_inputs, n_params, n_outputs, inputs_per_config;
};
"#;

fn lit(x: f64) -> String {
    if x.is_nan() { "__longlong_as_double(0x7FF8000000000000LL)".into() }
    else if x.is_infinite() { if x > 0.0 { "__longlong_as_double(0x7FF0000000000000LL)".into() } else { "__longlong_as_double(0xFFF0000000000000LL)".into() } }
    else { format!("__longlong_as_double({}LL)", x.to_bits() as i64) } // exact bits, no decimal rounding
}

/// Emit `__device__ void <name>_eval(const double* in, const double* prm, double* out)`.
pub fn device_function(name: &str, g: &ExprGraph) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "__device__ __forceinline__ void {name}_eval(const double* in, const double* prm, double* out) {{");
    for (i, n) in g.nodes.iter().enumerate() {
        let e = match *n {
            Node::Const(c) => lit(c),
            Node::Input(k) => format!("in[{k}]"),
            Node::Param(k) => format!("prm[{k}]"),
            Node::Un(op, a) => {
                let a = format!("v{a}");
                match op {
                    Un::Neg => format!("-{a}"), Un::Abs => format!("fabs({a})"), Un::Sqrt => format!("sqrt({a})"),
                    Un::Sin => format!("sin({a})"), Un::Cos => format!("cos({a})"), Un::Tan => format!("tan({a})"),
                    Un::Asin => format!("asin({a})"), Un::Acos => format!("acos({a})"), Un::Atan => format!("atan({a})"),
                    Un::Exp => format!("exp({a})"), Un::Ln => format!("log({a})"), Un::Floor => format!("floor({a})"),
                    Un::Ceil => format!("ceil({a})"), Un::Not => format!("(({a}) == 0.0 ? 1.0 : 0.0)"),
                }
            }
            Node::Bin(op, a, b) => {
                let (a, b) = (format!("v{a}"), format!("v{b}"));
                match op {
                    Bin::Add => format!("{a} + {b}"), Bin::Sub => format!("{a} - {b}"), Bin::Mul => format!("{a} * {b}"),
                    Bin::Div => format!("{a} / {b}"), Bin::Min => format!("(({b}) < ({a}) ? ({b}) : ({a}))"),
                    Bin::Max => format!("(({b}) > ({a}) ? ({b}) : ({a}))"), Bin::Pow => format!("pow({a}, {b})"),
                    Bin::Atan2 => format!("atan2({a}, {b})"), Bin::Mod => format!("({a} - {b} * floor({a} / {b}))"),
                    Bin::Lt => format!("({a} < {b} ? 1.0 : 0.0)"), Bin::Le => format!("({a} <= {b} ? 1.0 : 0.0)"),
                    Bin::Gt => format!("({a} > {b} ? 1.0 : 0.0)"), Bin::Ge => format!("({a} >= {b} ? 1.0 : 0.0)"),
                    Bin::Eq => format!("({a} == {b} ? 1.0 : 0.0)"), Bin::Ne => format!("({a} != {b} ? 1.0 : 0.0)"),
                    Bin::And => format!("((({a}) != 0.0 && ({b}) != 0.0) ? 1.0 : 0.0)"),
                    Bin::Or => format!("((({a}) != 0.0 || ({b}) != 0.0) ? 1.0 : 0.0)"),
                }
            }
            Node::Select(c, a, b) => format!("(v{c} != 0.0 ? v{a} : v{b})"),
            Node::Noise(x, y, z) => format!("ag_value_noise(v{x}, v{y}, v{z})"),
        };
        let _ = writeln!(s, "    const double v{i} = {e};");
    }
    for (o, id) in g.outputs.iter().enumerate() {
        let _ = writeln!(s, "    out[{o}] = v{id};");
    }
    s.push_str("}\n");
    s
}

/// Full translation unit: device function + vectorized kernel `<name>` (one thread per
/// (configuration, sample)), with the AgExprParams block as the single argument.
pub fn kernel_source(name: &str, g: &ExprGraph) -> String {
    let ni = g.input_names.len().max(1);
    let np = g.param_names.len().max(1);
    let no = g.outputs.len().max(1);
    let mut s = String::new();
    s.push_str(VALUE_NOISE_CUDA);
    s.push_str(EXPR_PARAMS_CUDA);
    s.push_str(&device_function(name, g));
    let _ = write!(
        s,
        r#"
extern "C" __global__ void {name}(const AgExprParams* p) {{
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)p->samples * p->configs;
    if (t >= total) return;
    unsigned int c = (unsigned int)(t / p->samples), smp = (unsigned int)(t % p->samples);
    const double* ins = (const double*)p->inputs;
    const double* prs = (const double*)p->params;
    double* outs = (double*)p->out;
    unsigned long long in_stride = (unsigned long long)(p->inputs_per_config ? p->configs : 1u) * p->samples;
    unsigned long long in_base = (unsigned long long)(p->inputs_per_config ? c : 0u) * p->samples + smp;
    double in[{ni}], prm[{np}], out[{no}];
    for (unsigned int k = 0; k < p->n_inputs; ++k) in[k] = ins[k * in_stride + in_base];
    for (unsigned int k = 0; k < p->n_params; ++k) prm[k] = prs[(unsigned long long)k * p->configs + c];
    {name}_eval(in, prm, out);
    for (unsigned int o = 0; o < p->n_outputs; ++o) outs[((unsigned long long)o * p->configs + c) * p->samples + smp] = out[o];
}}
"#
    );
    s
}

/// Codegen numeric precision (spec 08 § Precision policy; AG-08-002).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Native IEEE-754 f64 (A100/H100 class where `fp64_rate >= 1/8`).
    F64,
    /// Double-float (hi + lo f32 pair): ~48-bit significand at FP32 ALU rates on RTX/GeForce.
    Df64,
    /// Fast f32 for lightweight proxy sampling.
    F32,
}

impl From<agcad_driver::PrecisionPolicy> for PrecisionMode {
    fn from(p: agcad_driver::PrecisionPolicy) -> Self {
        match p {
            agcad_driver::PrecisionPolicy::NativeF64 => PrecisionMode::F64,
            agcad_driver::PrecisionPolicy::Df64Refined => PrecisionMode::Df64,
        }
    }
}

/// Exact Dekker / Knuth double-float (`df64`) device helpers mirroring `agcad-solver::df64`.
/// Compiled with `fmad=false` for deterministic TwoSum / TwoProd rounding.
pub const DF64_CUDA: &str = r#"
struct df64 { float hi, lo; };
__device__ __forceinline__ df64 df64_from_double(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return {hi, lo};
}
__device__ __forceinline__ double df64_to_double(df64 a) { return (double)a.hi + (double)a.lo; }
__device__ __forceinline__ df64 df64_two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return {s, e};
}
__device__ __forceinline__ df64 df64_quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return {s, e};
}
__device__ __forceinline__ df64 df64_add(df64 a, df64 b) {
    df64 s = df64_two_sum(a.hi, b.hi);
    df64 t = df64_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = df64_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return df64_quick_two_sum(s.hi, s.lo);
}
__device__ __forceinline__ df64 df64_neg(df64 a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ df64 df64_sub(df64 a, df64 b) { return df64_add(a, df64_neg(b)); }
__device__ __forceinline__ df64 df64_two_prod(float a, float b) {
    float p = a * b;
    float e = fmaf(a, b, -p);
    return {p, e};
}
__device__ __forceinline__ df64 df64_mul(df64 a, df64 b) {
    df64 p = df64_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return df64_quick_two_sum(p.hi, p.lo);
}
__device__ __forceinline__ df64 df64_div(df64 a, df64 b) {
    float q1 = a.hi / b.hi;
    df64 r = df64_sub(a, df64_mul(b, {q1, 0.0f}));
    float q2 = r.hi / b.hi;
    return df64_quick_two_sum(q1, q2);
}
__device__ __forceinline__ df64 df64_sqrt(df64 a) {
    if (a.hi <= 0.0f) return {0.0f, 0.0f};
    float x = sqrtf(a.hi);
    df64 ax = df64_sub(a, df64_two_prod(x, x));
    return df64_quick_two_sum(x, ax.hi / (2.0f * x));
}
__device__ __forceinline__ df64 df64_abs(df64 a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? df64_neg(a) : a; }
__device__ __forceinline__ int df64_cmp(df64 a, df64 b) {
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}
__device__ __forceinline__ df64 df64_min(df64 a, df64 b) { return df64_cmp(b, a) < 0 ? b : a; }
__device__ __forceinline__ df64 df64_max(df64 a, df64 b) { return df64_cmp(b, a) > 0 ? b : a; }
"#;

fn df64_lit(x: f64) -> String {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    format!("df64{{__int_as_float({}), __int_as_float({})}}", hi.to_bits() as i32, lo.to_bits() as i32)
}

fn f32_lit(x: f64) -> String {
    format!("__int_as_float({})", (x as f32).to_bits() as i32)
}

/// Emit a precision-selected CUDA translation unit (spec 08 § Precision policy).
///
/// The external `AgExprParams` buffer ABI stays `double` so caller layouts are invariant,
/// while internal arithmetic runs in `f64`, `df64` (consumer RTX/GeForce), or `f32`.
pub fn kernel_source_with_precision(name: &str, g: &ExprGraph, mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::F64 => kernel_source(name, g),
        PrecisionMode::F32 => kernel_source_f32(name, g),
        PrecisionMode::Df64 => kernel_source_df64(name, g),
    }
}

/// Select codegen precision automatically from device capabilities (AG-08-002).
pub fn kernel_source_for_policy(name: &str, g: &ExprGraph, policy: agcad_driver::PrecisionPolicy) -> String {
    kernel_source_with_precision(name, g, PrecisionMode::from(policy))
}

/// Emit a vectorized GPU reverse-mode gradient kernel `<name>_grad` for `g.outputs[output]`
/// with respect to its inputs (normals, Lipschitz probes, shape sensitivity).
pub fn gradient_kernel_source(name: &str, g: &ExprGraph, output: usize, mode: PrecisionMode) -> String {
    let grad_graph = crate::autodiff::gradient_graph(g, output);
    kernel_source_with_precision(&format!("{name}_grad"), &grad_graph, mode)
}

fn kernel_source_f32(name: &str, g: &ExprGraph) -> String {
    let ni = g.input_names.len().max(1);
    let np = g.param_names.len().max(1);
    let no = g.outputs.len().max(1);
    let mut s = String::new();
    s.push_str(VALUE_NOISE_CUDA);
    s.push_str(EXPR_PARAMS_CUDA);
    let _ = writeln!(s, "__device__ __forceinline__ void {name}_eval(const float* in, const float* prm, float* out) {{");
    for (i, n) in g.nodes.iter().enumerate() {
        let e = match *n {
            Node::Const(c) => f32_lit(c),
            Node::Input(k) => format!("in[{k}]"),
            Node::Param(k) => format!("prm[{k}]"),
            Node::Un(op, a) => {
                let a = format!("v{a}");
                match op {
                    Un::Neg => format!("-{a}"), Un::Abs => format!("fabsf({a})"), Un::Sqrt => format!("sqrtf({a})"),
                    Un::Sin => format!("sinf({a})"), Un::Cos => format!("cosf({a})"), Un::Tan => format!("tanf({a})"),
                    Un::Asin => format!("asinf({a})"), Un::Acos => format!("acosf({a})"), Un::Atan => format!("atanf({a})"),
                    Un::Exp => format!("expf({a})"), Un::Ln => format!("logf({a})"), Un::Floor => format!("floorf({a})"),
                    Un::Ceil => format!("ceilf({a})"), Un::Not => format!("(({a}) == 0.0f ? 1.0f : 0.0f)"),
                }
            }
            Node::Bin(op, a, b) => {
                let (a, b) = (format!("v{a}"), format!("v{b}"));
                match op {
                    Bin::Add => format!("{a} + {b}"), Bin::Sub => format!("{a} - {b}"), Bin::Mul => format!("{a} * {b}"),
                    Bin::Div => format!("{a} / {b}"), Bin::Min => format!("(({b}) < ({a}) ? ({b}) : ({a}))"),
                    Bin::Max => format!("(({b}) > ({a}) ? ({b}) : ({a}))"), Bin::Pow => format!("powf({a}, {b})"),
                    Bin::Atan2 => format!("atan2f({a}, {b})"), Bin::Mod => format!("({a} - {b} * floorf({a} / {b}))"),
                    Bin::Lt => format!("({a} < {b} ? 1.0f : 0.0f)"), Bin::Le => format!("({a} <= {b} ? 1.0f : 0.0f)"),
                    Bin::Gt => format!("({a} > {b} ? 1.0f : 0.0f)"), Bin::Ge => format!("({a} >= {b} ? 1.0f : 0.0f)"),
                    Bin::Eq => format!("({a} == {b} ? 1.0f : 0.0f)"), Bin::Ne => format!("({a} != {b} ? 1.0f : 0.0f)"),
                    Bin::And => format!("((({a}) != 0.0f && ({b}) != 0.0f) ? 1.0f : 0.0f)"),
                    Bin::Or => format!("((({a}) != 0.0f || ({b}) != 0.0f) ? 1.0f : 0.0f)"),
                }
            }
            Node::Select(c, a, b) => format!("(v{c} != 0.0f ? v{a} : v{b})"),
            Node::Noise(x, y, z) => format!("(float)ag_value_noise((double)v{x}, (double)v{y}, (double)v{z})"),
        };
        let _ = writeln!(s, "    const float v{i} = {e};");
    }
    for (o, id) in g.outputs.iter().enumerate() { let _ = writeln!(s, "    out[{o}] = v{id};"); }
    s.push_str("}\n");
    let _ = write!(
        s,
        r#"
extern "C" __global__ void {name}(const AgExprParams* p) {{
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)p->samples * p->configs;
    if (t >= total) return;
    unsigned int c = (unsigned int)(t / p->samples), smp = (unsigned int)(t % p->samples);
    const double* ins = (const double*)p->inputs;
    const double* prs = (const double*)p->params;
    double* outs = (double*)p->out;
    unsigned long long in_stride = (unsigned long long)(p->inputs_per_config ? p->configs : 1u) * p->samples;
    unsigned long long in_base = (unsigned long long)(p->inputs_per_config ? c : 0u) * p->samples + smp;
    float in[{ni}], prm[{np}], out[{no}];
    for (unsigned int k = 0; k < p->n_inputs; ++k) in[k] = (float)ins[k * in_stride + in_base];
    for (unsigned int k = 0; k < p->n_params; ++k) prm[k] = (float)prs[(unsigned long long)k * p->configs + c];
    {name}_eval(in, prm, out);
    for (unsigned int o = 0; o < p->n_outputs; ++o) outs[((unsigned long long)o * p->configs + c) * p->samples + smp] = (double)out[o];
}}
"#
    );
    s
}

fn kernel_source_df64(name: &str, g: &ExprGraph) -> String {
    let ni = g.input_names.len().max(1);
    let np = g.param_names.len().max(1);
    let no = g.outputs.len().max(1);
    let mut s = String::new();
    s.push_str(VALUE_NOISE_CUDA);
    s.push_str(DF64_CUDA);
    s.push_str(EXPR_PARAMS_CUDA);
    let _ = writeln!(s, "__device__ __forceinline__ void {name}_eval(const df64* in, const df64* prm, df64* out) {{");
    for (i, n) in g.nodes.iter().enumerate() {
        let e = match *n {
            Node::Const(c) => df64_lit(c),
            Node::Input(k) => format!("in[{k}]"),
            Node::Param(k) => format!("prm[{k}]"),
            Node::Un(op, a) => {
                let a = format!("v{a}");
                match op {
                    Un::Neg => format!("df64_neg({a})"),
                    Un::Abs => format!("df64_abs({a})"),
                    Un::Sqrt => format!("df64_sqrt({a})"),
                    Un::Sin => format!("df64_from_double(sin(df64_to_double({a})))"),
                    Un::Cos => format!("df64_from_double(cos(df64_to_double({a})))"),
                    Un::Tan => format!("df64_from_double(tan(df64_to_double({a})))"),
                    Un::Asin => format!("df64_from_double(asin(df64_to_double({a})))"),
                    Un::Acos => format!("df64_from_double(acos(df64_to_double({a})))"),
                    Un::Atan => format!("df64_from_double(atan(df64_to_double({a})))"),
                    Un::Exp => format!("df64_from_double(exp(df64_to_double({a})))"),
                    Un::Ln => format!("df64_from_double(log(df64_to_double({a})))"),
                    Un::Floor => format!("df64_from_double(floor(df64_to_double({a})))"),
                    Un::Ceil => format!("df64_from_double(ceil(df64_to_double({a})))"),
                    Un::Not => format!("(({a}.hi == 0.0f && {a}.lo == 0.0f) ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                }
            }
            Node::Bin(op, a, b) => {
                let (a, b) = (format!("v{a}"), format!("v{b}"));
                match op {
                    Bin::Add => format!("df64_add({a}, {b})"),
                    Bin::Sub => format!("df64_sub({a}, {b})"),
                    Bin::Mul => format!("df64_mul({a}, {b})"),
                    Bin::Div => format!("df64_div({a}, {b})"),
                    Bin::Min => format!("df64_min({a}, {b})"),
                    Bin::Max => format!("df64_max({a}, {b})"),
                    Bin::Pow => format!("df64_from_double(pow(df64_to_double({a}), df64_to_double({b})))"),
                    Bin::Atan2 => format!("df64_from_double(atan2(df64_to_double({a}), df64_to_double({b})))"),
                    Bin::Mod => format!("df64_sub({a}, df64_mul({b}, df64_from_double(floor(df64_to_double(df64_div({a}, {b}))))))"),
                    Bin::Lt => format!("(df64_cmp({a}, {b}) < 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Le => format!("(df64_cmp({a}, {b}) <= 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Gt => format!("(df64_cmp({a}, {b}) > 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Ge => format!("(df64_cmp({a}, {b}) >= 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Eq => format!("(df64_cmp({a}, {b}) == 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Ne => format!("(df64_cmp({a}, {b}) != 0 ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::And => format!("((({a}.hi != 0.0f || {a}.lo != 0.0f) && ({b}.hi != 0.0f || {b}.lo != 0.0f)) ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                    Bin::Or => format!("((({a}.hi != 0.0f || {a}.lo != 0.0f) || ({b}.hi != 0.0f || {b}.lo != 0.0f)) ? df64{{1.0f, 0.0f}} : df64{{0.0f, 0.0f}})"),
                }
            }
            Node::Select(c, a, b) => format!("((v{c}.hi != 0.0f || v{c}.lo != 0.0f) ? v{a} : v{b})"),
            Node::Noise(x, y, z) => format!("df64_from_double(ag_value_noise(df64_to_double(v{x}), df64_to_double(v{y}), df64_to_double(v{z})))"),
        };
        let _ = writeln!(s, "    const df64 v{i} = {e};");
    }
    for (o, id) in g.outputs.iter().enumerate() { let _ = writeln!(s, "    out[{o}] = v{id};"); }
    s.push_str("}\n");
    let _ = write!(
        s,
        r#"
extern "C" __global__ void {name}(const AgExprParams* p) {{
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)p->samples * p->configs;
    if (t >= total) return;
    unsigned int c = (unsigned int)(t / p->samples), smp = (unsigned int)(t % p->samples);
    const double* ins = (const double*)p->inputs;
    const double* prs = (const double*)p->params;
    double* outs = (double*)p->out;
    unsigned long long in_stride = (unsigned long long)(p->inputs_per_config ? p->configs : 1u) * p->samples;
    unsigned long long in_base = (unsigned long long)(p->inputs_per_config ? c : 0u) * p->samples + smp;
    df64 in[{ni}], prm[{np}], out[{no}];
    for (unsigned int k = 0; k < p->n_inputs; ++k) in[k] = df64_from_double(ins[k * in_stride + in_base]);
    for (unsigned int k = 0; k < p->n_params; ++k) prm[k] = df64_from_double(prs[(unsigned long long)k * p->configs + c]);
    {name}_eval(in, prm, out);
    for (unsigned int o = 0; o < p->n_outputs; ++o) outs[((unsigned long long)o * p->configs + c) * p->samples + smp] = df64_to_double(out[o]);
}}
"#
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_df64_f32_and_gradient_kernels() {
        let src = "#[implicit] fn sphere(p: Vec3, #[param(default = 5.0*mm)] r: Length) -> Length { length(p) - r }";
        let m = crate::compile_module(src, agcad_ir::Catalog::global()).unwrap();
        let g = m.get("sphere").unwrap().expr.as_ref().unwrap();
        let df = kernel_source_for_policy("sphere", g, agcad_driver::PrecisionPolicy::Df64Refined);
        assert!(df.contains("struct df64") && df.contains("df64_sqrt") && df.contains("df64_sub"));
        let f32s = kernel_source_with_precision("sphere", g, PrecisionMode::F32);
        assert!(f32s.contains("sqrtf(") && f32s.contains("const float v"));
        let grad = gradient_kernel_source("sphere", g, 0, PrecisionMode::Df64);
        assert!(grad.contains("sphere_grad") && grad.contains("out[0]") && grad.contains("out[1]") && grad.contains("out[2]"));
    }
}
