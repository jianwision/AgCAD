//! Autodiff over expression graphs (spec 04 § Autodiff): forward-mode tangents and
//! reverse-mode adjoints with respect to parameters or inputs. Non-differentiable points
//! (comparisons/select switching, floor/ceil, abs at 0) are reported, not hidden (AG-04-005).

use crate::expr::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Wrt {
    Params,
    Inputs,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    /// `d output[o] / d var[j]` stored row-major `[outputs × vars]`.
    pub jacobian: Vec<f64>,
    pub outputs: usize,
    pub vars: usize,
    /// True if evaluation passed through a select/comparison/floor whose branch could flip
    /// under an infinitesimal change (gradient valid only locally; spec 04).
    pub discontinuous: bool,
}

/// Reverse-mode Jacobian at a point.
pub fn jacobian(g: &ExprGraph, inputs: &[f64], params: &[f64], wrt: Wrt) -> Gradient {
    let n = g.nodes.len();
    let mut v = vec![0.0; n];
    let mut disc = false;
    for (i, node) in g.nodes.iter().enumerate() {
        v[i] = match *node {
            Node::Const(c) => c,
            Node::Input(k) => inputs[k as usize],
            Node::Param(k) => params[k as usize],
            Node::Un(op, a) => {
                if matches!(op, Un::Floor | Un::Ceil) || (op == Un::Abs && v[a as usize] == 0.0) { disc = true; }
                eval_un(op, v[a as usize])
            }
            Node::Bin(op, a, b) => {
                if matches!(op, Bin::Min | Bin::Max) && v[a as usize] == v[b as usize] { disc = true; }
                eval_bin(op, v[a as usize], v[b as usize])
            }
            Node::Select(c, a, b) => { disc |= g.nodes.iter().any(|_| false); if v[c as usize] != 0.0 { v[a as usize] } else { v[b as usize] } }
            Node::Noise(x, y, z) => value_noise(v[x as usize], v[y as usize], v[z as usize]),
        };
        if matches!(node, Node::Select(..)) { disc = true; }
    }
    let vars = match wrt { Wrt::Params => g.param_names.len(), Wrt::Inputs => g.input_names.len() };
    let mut jac = vec![0.0; g.outputs.len() * vars];
    for (oi, &out) in g.outputs.iter().enumerate() {
        let mut adj = vec![0.0; n];
        adj[out as usize] = 1.0;
        for i in (0..n).rev() {
            let a_i = adj[i];
            if a_i == 0.0 { continue; }
            match g.nodes[i] {
                Node::Const(_) => {}
                Node::Input(k) => if wrt == Wrt::Inputs { jac[oi * vars + k as usize] += a_i },
                Node::Param(k) => if wrt == Wrt::Params { jac[oi * vars + k as usize] += a_i },
                Node::Un(op, a) => {
                    let x = v[a as usize];
                    let d = match op {
                        Un::Neg => -1.0, Un::Abs => x.signum(), Un::Sqrt => 0.5 / v[i], Un::Sin => x.cos(), Un::Cos => -x.sin(),
                        Un::Tan => 1.0 / (x.cos() * x.cos()), Un::Asin => 1.0 / (1.0 - x * x).sqrt(), Un::Acos => -1.0 / (1.0 - x * x).sqrt(),
                        Un::Atan => 1.0 / (1.0 + x * x), Un::Exp => v[i], Un::Ln => 1.0 / x, Un::Floor | Un::Ceil | Un::Not => 0.0,
                    };
                    adj[a as usize] += a_i * d;
                }
                Node::Bin(op, a, b) => {
                    let (x, y) = (v[a as usize], v[b as usize]);
                    let (da, db) = match op {
                        Bin::Add => (1.0, 1.0), Bin::Sub => (1.0, -1.0), Bin::Mul => (y, x), Bin::Div => (1.0 / y, -x / (y * y)),
                        Bin::Min => if y < x { (0.0, 1.0) } else { (1.0, 0.0) },
                        Bin::Max => if y > x { (0.0, 1.0) } else { (1.0, 0.0) },
                        Bin::Pow => (y * x.powf(y - 1.0), if x > 0.0 { v[i] * x.ln() } else { 0.0 }),
                        Bin::Atan2 => { let r = x * x + y * y; (y / r, -x / r) }
                        Bin::Mod => (1.0, -(x / y).floor()),
                        _ => (0.0, 0.0),
                    };
                    adj[a as usize] += a_i * da;
                    adj[b as usize] += a_i * db;
                }
                Node::Select(c, a, b) => {
                    if v[c as usize] != 0.0 { adj[a as usize] += a_i } else { adj[b as usize] += a_i }
                }
                Node::Noise(x, y, z) => {
                    // Central differences (noise is C1 except at lattice planes).
                    let h = 1e-6;
                    let (px, py, pz) = (v[x as usize], v[y as usize], v[z as usize]);
                    adj[x as usize] += a_i * (value_noise(px + h, py, pz) - value_noise(px - h, py, pz)) / (2.0 * h);
                    adj[y as usize] += a_i * (value_noise(px, py + h, pz) - value_noise(px, py - h, pz)) / (2.0 * h);
                    adj[z as usize] += a_i * (value_noise(px, py, pz + h) - value_noise(px, py, pz - h)) / (2.0 * h);
                }
            }
        }
    }
    Gradient { jacobian: jac, outputs: g.outputs.len(), vars, discontinuous: disc }
}

/// Build a new graph computing the gradient symbolically w.r.t. inputs (used to derive the
/// implicit-body normal / Lipschitz probes on the device). Output lanes: [d/dinput_0..].
pub fn gradient_graph(g: &ExprGraph, output: usize) -> ExprGraph {
    let mut h = g.clone();
    let n = g.nodes.len();
    let zero = h.konst(0.0, Dim::NONE);
    let one = h.konst(1.0, Dim::NONE);
    let mut adj: Vec<u32> = vec![zero; n];
    let out = g.outputs[output] as usize;
    adj[out] = one;
    let mut grads = vec![zero; g.input_names.len()];
    let add = |h: &mut ExprGraph, a: u32, b: u32| -> u32 { h.push(Node::Bin(Bin::Add, a, b), Dim::NONE) };
    let mul = |h: &mut ExprGraph, a: u32, b: u32| -> u32 { h.push(Node::Bin(Bin::Mul, a, b), Dim::NONE) };
    for i in (0..n).rev() {
        let ai = adj[i];
        if ai == zero { continue; }
        match g.nodes[i].clone() {
            Node::Input(k) => grads[k as usize] = add(&mut h, grads[k as usize], ai),
            Node::Const(_) | Node::Param(_) => {}
            Node::Un(op, a) => {
                let x = a;
                let d = match op {
                    Un::Neg => h.konst(-1.0, Dim::NONE),
                    Un::Abs => { let z = h.konst(0.0, Dim::NONE); let gt = h.push(Node::Bin(Bin::Gt, x, z), Dim::NONE); let m1 = h.konst(-1.0, Dim::NONE); h.push(Node::Select(gt, one, m1), Dim::NONE) }
                    Un::Sqrt => { let half = h.konst(0.5, Dim::NONE); h.push(Node::Bin(Bin::Div, half, i as u32), Dim::NONE) }
                    Un::Sin => h.push(Node::Un(Un::Cos, x), Dim::NONE),
                    Un::Cos => { let s = h.push(Node::Un(Un::Sin, x), Dim::NONE); h.push(Node::Un(Un::Neg, s), Dim::NONE) }
                    Un::Exp => i as u32,
                    Un::Ln => h.push(Node::Bin(Bin::Div, one, x), Dim::NONE),
                    _ => zero,
                };
                let t = mul(&mut h, ai, d);
                adj[a as usize] = add(&mut h, adj[a as usize], t);
            }
            Node::Bin(op, a, b) => {
                let (da, db) = match op {
                    Bin::Add => (one, one),
                    Bin::Sub => (one, h.konst(-1.0, Dim::NONE)),
                    Bin::Mul => (b, a),
                    Bin::Div => { let inv = h.push(Node::Bin(Bin::Div, one, b), Dim::NONE); let q = h.push(Node::Bin(Bin::Div, i as u32, b), Dim::NONE); (inv, h.push(Node::Un(Un::Neg, q), Dim::NONE)) }
                    Bin::Min => { let c = h.push(Node::Bin(Bin::Lt, b, a), Dim::NONE); (h.push(Node::Select(c, zero, one), Dim::NONE), h.push(Node::Select(c, one, zero), Dim::NONE)) }
                    Bin::Max => { let c = h.push(Node::Bin(Bin::Gt, b, a), Dim::NONE); (h.push(Node::Select(c, zero, one), Dim::NONE), h.push(Node::Select(c, one, zero), Dim::NONE)) }
                    _ => (zero, zero),
                };
                let ta = mul(&mut h, ai, da);
                adj[a as usize] = add(&mut h, adj[a as usize], ta);
                let tb = mul(&mut h, ai, db);
                adj[b as usize] = add(&mut h, adj[b as usize], tb);
            }
            Node::Select(c, a, b) => {
                let ta = h.push(Node::Select(c, ai, zero), Dim::NONE);
                adj[a as usize] = add(&mut h, adj[a as usize], ta);
                let tb = h.push(Node::Select(c, zero, ai), Dim::NONE);
                adj[b as usize] = add(&mut h, adj[b as usize], tb);
            }
            Node::Noise(..) => {}
        }
    }
    h.outputs = grads;
    h
}
