//! Expression graphs: the SSA core of implicit bodies, wrangles, evaluations, residuals and
//! laws (spec 04). Nodes are scalar; `Vec3` values are lowered to three scalar lanes.
//! Every node carries a physical dimension (exponents of length, angle, mass, time) checked
//! at construction (unit-aware typecheck, AG-04 language rules).

use serde::{Deserialize, Serialize};

/// Dimension exponents `[length, angle, mass, time]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Dim(pub [i8; 4]);

impl Dim {
    pub const NONE: Dim = Dim([0, 0, 0, 0]);
    pub const LENGTH: Dim = Dim([1, 0, 0, 0]);
    pub const ANGLE: Dim = Dim([0, 1, 0, 0]);
    pub const MASS: Dim = Dim([0, 0, 1, 0]);
    pub const TIME: Dim = Dim([0, 0, 0, 1]);
    pub fn mul(self, o: Dim) -> Dim { Dim([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2], self.0[3] + o.0[3]]) }
    pub fn div(self, o: Dim) -> Dim { Dim([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2], self.0[3] - o.0[3]]) }
    pub fn pow(self, k: i8) -> Dim { Dim([self.0[0] * k, self.0[1] * k, self.0[2] * k, self.0[3] * k]) }
    /// Angles count as dimensionless for transcendental arguments (radians).
    pub fn is_dimensionless_or_angle(self) -> bool { self.0[0] == 0 && self.0[2] == 0 && self.0[3] == 0 }
    pub fn name(self) -> String {
        if self == Dim::NONE { return "dimensionless".into(); }
        let names = ["mm", "rad", "kg", "s"];
        self.0.iter().zip(names).filter(|(e, _)| **e != 0).map(|(e, n)| if *e == 1 { n.to_string() } else { format!("{n}^{e}") }).collect::<Vec<_>>().join("·")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Un { Neg, Abs, Sqrt, Sin, Cos, Tan, Asin, Acos, Atan, Exp, Ln, Floor, Ceil, Not }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Bin { Add, Sub, Mul, Div, Min, Max, Pow, Atan2, Lt, Le, Gt, Ge, Eq, Ne, And, Or, Mod }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Const(f64),
    /// Per-sample input lane (e.g. point coordinate `p.x` = Input(0)).
    Input(u16),
    /// Per-configuration parameter (design-table column).
    Param(u16),
    Un(Un, u32),
    Bin(Bin, u32, u32),
    /// `if c != 0 { a } else { b }` (predication; both sides evaluated).
    Select(u32, u32, u32),
    /// Deterministic counter-based noise: `noise(seed, x, y, z)` (Perlin-style value noise).
    Noise(u32, u32, u32),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExprGraph {
    pub nodes: Vec<Node>,
    pub dims: Vec<Dim>,
    /// Output node ids (scalar lanes).
    pub outputs: Vec<u32>,
    pub input_names: Vec<String>,
    pub param_names: Vec<String>,
    pub param_dims: Vec<Dim>,
    pub input_dims: Vec<Dim>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DimError {
    pub message: String,
}

impl ExprGraph {
    pub fn push(&mut self, n: Node, d: Dim) -> u32 {
        // Hash-consing of constants keeps graphs small; other nodes are appended.
        if let Node::Const(c) = n {
            if let Some(i) = self.nodes.iter().position(|x| matches!(x, Node::Const(k) if k.to_bits() == c.to_bits())) {
                if self.dims[i] == d { return i as u32; }
            }
        }
        self.nodes.push(n);
        self.dims.push(d);
        (self.nodes.len() - 1) as u32
    }

    pub fn dim(&self, i: u32) -> Dim { self.dims[i as usize] }

    pub fn konst(&mut self, v: f64, d: Dim) -> u32 { self.push(Node::Const(v), d) }

    pub fn input(&mut self, name: &str, d: Dim) -> u32 {
        let k = match self.input_names.iter().position(|n| n == name) {
            Some(k) => k,
            None => { self.input_names.push(name.into()); self.input_dims.push(d); self.input_names.len() - 1 }
        };
        self.push(Node::Input(k as u16), d)
    }

    pub fn param(&mut self, name: &str, d: Dim) -> u32 {
        let k = match self.param_names.iter().position(|n| n == name) {
            Some(k) => k,
            None => { self.param_names.push(name.into()); self.param_dims.push(d); self.param_names.len() - 1 }
        };
        self.push(Node::Param(k as u16), d)
    }

    /// Dimension-checked unary op.
    pub fn un(&mut self, op: Un, a: u32) -> Result<u32, DimError> {
        let da = self.dim(a);
        let d = match op {
            Un::Neg | Un::Abs | Un::Floor | Un::Ceil => da,
            Un::Sqrt => {
                if da.0.iter().any(|e| e % 2 != 0) { return Err(DimError { message: format!("sqrt of {} has no integral dimension", da.name()) }); }
                Dim([da.0[0] / 2, da.0[1] / 2, da.0[2] / 2, da.0[3] / 2])
            }
            Un::Sin | Un::Cos | Un::Tan | Un::Exp | Un::Ln => {
                if !da.is_dimensionless_or_angle() { return Err(DimError { message: format!("{op:?} needs an angle or dimensionless argument, got {}", da.name()) }); }
                Dim::NONE
            }
            Un::Asin | Un::Acos | Un::Atan => {
                if da != Dim::NONE { return Err(DimError { message: format!("{op:?} needs a dimensionless argument, got {}", da.name()) }); }
                Dim::ANGLE
            }
            Un::Not => Dim::NONE,
        };
        Ok(self.push(Node::Un(op, a), d))
    }

    /// Dimension-checked binary op.
    pub fn bin(&mut self, op: Bin, a: u32, b: u32) -> Result<u32, DimError> {
        let (da, db) = (self.dim(a), self.dim(b));
        let same = |what: &str| -> Result<Dim, DimError> {
            if da == db || self.is_zero_const(a) || self.is_zero_const(b) { Ok(if da == Dim::NONE { db } else { da }) } else { Err(DimError { message: format!("{what}: {} vs {}", da.name(), db.name()) }) }
        };
        let d = match op {
            Bin::Add | Bin::Sub | Bin::Min | Bin::Max | Bin::Mod => same("dimension mismatch")?,
            Bin::Mul => da.mul(db),
            Bin::Div => da.div(db),
            Bin::Atan2 => { same("atan2 dimension mismatch")?; Dim::ANGLE }
            Bin::Pow => {
                if db != Dim::NONE { return Err(DimError { message: "exponent must be dimensionless".into() }); }
                match self.nodes[b as usize] {
                    Node::Const(k) if k.fract() == 0.0 && k.abs() <= 8.0 => da.pow(k as i8),
                    _ if da == Dim::NONE => Dim::NONE,
                    _ => return Err(DimError { message: "non-integer power of a dimensioned value".into() }),
                }
            }
            Bin::Lt | Bin::Le | Bin::Gt | Bin::Ge | Bin::Eq | Bin::Ne => { same("comparison of different dimensions")?; Dim::NONE }
            Bin::And | Bin::Or => Dim::NONE,
        };
        Ok(self.push(Node::Bin(op, a, b), d))
    }

    pub fn select(&mut self, c: u32, a: u32, b: u32) -> Result<u32, DimError> {
        if self.dim(a) != self.dim(b) {
            return Err(DimError { message: format!("if/else branches differ: {} vs {}", self.dim(a).name(), self.dim(b).name()) });
        }
        let d = self.dim(a);
        Ok(self.push(Node::Select(c, a, b), d))
    }

    fn is_zero_const(&self, i: u32) -> bool { matches!(self.nodes[i as usize], Node::Const(z) if z == 0.0) }

    /// Reference evaluation (f64). `inputs`/`params` indexed as in the graph tables.
    pub fn eval(&self, inputs: &[f64], params: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0f64; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            v[i] = match *n {
                Node::Const(c) => c,
                Node::Input(k) => inputs[k as usize],
                Node::Param(k) => params[k as usize],
                Node::Un(op, a) => eval_un(op, v[a as usize]),
                Node::Bin(op, a, b) => eval_bin(op, v[a as usize], v[b as usize]),
                Node::Select(c, a, b) => if v[c as usize] != 0.0 { v[a as usize] } else { v[b as usize] },
                Node::Noise(x, y, z) => value_noise(v[x as usize], v[y as usize], v[z as usize]),
            };
        }
        self.outputs.iter().map(|&o| v[o as usize]).collect()
    }
}

pub fn eval_un(op: Un, a: f64) -> f64 {
    match op {
        Un::Neg => -a, Un::Abs => a.abs(), Un::Sqrt => a.sqrt(), Un::Sin => a.sin(), Un::Cos => a.cos(), Un::Tan => a.tan(),
        Un::Asin => a.asin(), Un::Acos => a.acos(), Un::Atan => a.atan(), Un::Exp => a.exp(), Un::Ln => a.ln(),
        Un::Floor => a.floor(), Un::Ceil => a.ceil(), Un::Not => (a == 0.0) as i32 as f64,
    }
}

pub fn eval_bin(op: Bin, a: f64, b: f64) -> f64 {
    match op {
        Bin::Add => a + b, Bin::Sub => a - b, Bin::Mul => a * b, Bin::Div => a / b,
        Bin::Min => if b < a { b } else { a }, Bin::Max => if b > a { b } else { a },
        Bin::Pow => a.powf(b), Bin::Atan2 => a.atan2(b), Bin::Mod => a - b * (a / b).floor(),
        Bin::Lt => (a < b) as i32 as f64, Bin::Le => (a <= b) as i32 as f64, Bin::Gt => (a > b) as i32 as f64,
        Bin::Ge => (a >= b) as i32 as f64, Bin::Eq => (a == b) as i32 as f64, Bin::Ne => (a != b) as i32 as f64,
        Bin::And => ((a != 0.0) && (b != 0.0)) as i32 as f64, Bin::Or => ((a != 0.0) || (b != 0.0)) as i32 as f64,
    }
}

/// Deterministic value noise in [-1, 1] (hash of integer lattice, smoothstep blend).
/// The CUDA VM and codegen implement the identical function (`ag_value_noise`).
pub fn value_noise(x: f64, y: f64, z: f64) -> f64 {
    fn h(i: i64, j: i64, k: i64) -> f64 {
        let mut n = (i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663) ^ k.wrapping_mul(83492791)) as u64;
        n = (n ^ (n >> 33)).wrapping_mul(0xff51afd7ed558ccd);
        n = (n ^ (n >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
        n ^= n >> 33;
        (n >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    let (xi, yi, zi) = (x.floor(), y.floor(), z.floor());
    let (fx, fy, fz) = (x - xi, y - yi, z - zi);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (u, v, w) = (s(fx), s(fy), s(fz));
    let (i, j, k) = (xi as i64, yi as i64, zi as i64);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(h(i, j, k), h(i + 1, j, k), u);
    let x10 = lerp(h(i, j + 1, k), h(i + 1, j + 1, k), u);
    let x01 = lerp(h(i, j, k + 1), h(i + 1, j, k + 1), u);
    let x11 = lerp(h(i, j + 1, k + 1), h(i + 1, j + 1, k + 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}
