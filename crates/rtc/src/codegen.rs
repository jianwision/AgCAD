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
