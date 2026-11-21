//! Vectors and frames (f64, internal units mm/rad).

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn v3(x: f64, y: f64, z: f64) -> V3 { V3 { x, y, z } }

impl V3 {
    pub const ZERO: V3 = v3(0.0, 0.0, 0.0);
    pub const X: V3 = v3(1.0, 0.0, 0.0);
    pub const Y: V3 = v3(0.0, 1.0, 0.0);
    pub const Z: V3 = v3(0.0, 0.0, 1.0);
    pub fn dot(self, o: V3) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: V3) -> V3 { v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x) }
    pub fn norm(self) -> f64 { self.dot(self).sqrt() }
    pub fn normalized(self) -> V3 { let n = self.norm(); if n > 0.0 { self * (1.0 / n) } else { self } }
    pub fn dist(self, o: V3) -> f64 { (self - o).norm() }
    pub fn lerp(self, o: V3, t: f64) -> V3 { self + (o - self) * t }
    /// Any unit vector perpendicular to `self`.
    pub fn any_perp(self) -> V3 {
        let a = if self.x.abs() < 0.9 { V3::X } else { V3::Y };
        self.cross(a).normalized()
    }
    pub fn max_abs(self) -> f64 { self.x.abs().max(self.y.abs()).max(self.z.abs()) }
}

impl Add for V3 { type Output = V3; fn add(self, o: V3) -> V3 { v3(self.x + o.x, self.y + o.y, self.z + o.z) } }
impl Sub for V3 { type Output = V3; fn sub(self, o: V3) -> V3 { v3(self.x - o.x, self.y - o.y, self.z - o.z) } }
impl Mul<f64> for V3 { type Output = V3; fn mul(self, s: f64) -> V3 { v3(self.x * s, self.y * s, self.z * s) } }
impl Neg for V3 { type Output = V3; fn neg(self) -> V3 { v3(-self.x, -self.y, -self.z) } }

/// Right-handed orthonormal frame (CATIA "axis system").
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub origin: V3,
    pub x: V3,
    pub y: V3,
    pub z: V3,
}

impl Frame {
    pub const WORLD: Frame = Frame { origin: V3::ZERO, x: V3::X, y: V3::Y, z: V3::Z };
    /// Frame from origin + normal (z) + reference x direction (orthogonalized).
    pub fn new(origin: V3, z: V3, xref: V3) -> Frame {
        let z = z.normalized();
        let x = (xref - z * xref.dot(z)).normalized();
        let x = if x.norm() < 0.5 { z.any_perp() } else { x };
        Frame { origin, x, y: z.cross(x), z }
    }
    pub fn to_world(&self, p: V3) -> V3 { self.origin + self.x * p.x + self.y * p.y + self.z * p.z }
    pub fn to_local(&self, p: V3) -> V3 { let d = p - self.origin; v3(d.dot(self.x), d.dot(self.y), d.dot(self.z)) }
    pub fn dir_to_world(&self, d: V3) -> V3 { self.x * d.x + self.y * d.y + self.z * d.z }
}

/// Rigid transform (rotation matrix rows + translation).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Xform {
    pub r: [[f64; 3]; 3],
    pub t: V3,
    /// Uniform scale (1.0 for rigid).
    pub s: f64,
}

impl Xform {
    pub const IDENTITY: Xform = Xform { r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: V3::ZERO, s: 1.0 };
    pub fn translation(t: V3) -> Xform { Xform { t, ..Xform::IDENTITY } }
    /// Rodrigues rotation about an axis through `c`.
    pub fn rotation(c: V3, axis: V3, angle: f64) -> Xform {
        let k = axis.normalized();
        let (s, co) = angle.sin_cos();
        let v = 1.0 - co;
        let r = [
            [co + k.x * k.x * v, k.x * k.y * v - k.z * s, k.x * k.z * v + k.y * s],
            [k.y * k.x * v + k.z * s, co + k.y * k.y * v, k.y * k.z * v - k.x * s],
            [k.z * k.x * v - k.y * s, k.z * k.y * v + k.x * s, co + k.z * k.z * v],
        ];
        let x = Xform { r, t: V3::ZERO, s: 1.0 };
        Xform { t: c - x.apply_dir(c), ..x }
    }
    pub fn mirror(p: V3, n: V3) -> Xform {
        let n = n.normalized();
        let r = [
            [1.0 - 2.0 * n.x * n.x, -2.0 * n.x * n.y, -2.0 * n.x * n.z],
            [-2.0 * n.y * n.x, 1.0 - 2.0 * n.y * n.y, -2.0 * n.y * n.z],
            [-2.0 * n.z * n.x, -2.0 * n.z * n.y, 1.0 - 2.0 * n.z * n.z],
        ];
        let x = Xform { r, t: V3::ZERO, s: 1.0 };
        Xform { t: p - x.apply_dir(p), ..x }
    }
    pub fn apply_dir(&self, d: V3) -> V3 {
        v3(self.r[0][0] * d.x + self.r[0][1] * d.y + self.r[0][2] * d.z, self.r[1][0] * d.x + self.r[1][1] * d.y + self.r[1][2] * d.z, self.r[2][0] * d.x + self.r[2][1] * d.y + self.r[2][2] * d.z) * self.s
    }
    pub fn apply(&self, p: V3) -> V3 { self.apply_dir(p) + self.t }
    pub fn det(&self) -> f64 {
        let r = &self.r;
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }
    pub fn compose(&self, o: &Xform) -> Xform {
        // self ∘ o
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 { for j in 0..3 { r[i][j] = (0..3).map(|k| self.r[i][k] * o.r[k][j]).sum(); } }
        Xform { r, t: self.apply(o.t), s: self.s * o.s }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: V3,
    pub max: V3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: v3(f64::INFINITY, f64::INFINITY, f64::INFINITY), max: v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY) };
    pub fn add(&mut self, p: V3) {
        self.min = v3(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z));
        self.max = v3(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z));
    }
    pub fn union(a: Aabb, b: Aabb) -> Aabb { let mut r = a; r.add(b.min); r.add(b.max); r }
    pub fn extent(&self) -> V3 { self.max - self.min }
    pub fn overlaps(&self, o: &Aabb, tol: f64) -> bool {
        self.min.x <= o.max.x + tol && o.min.x <= self.max.x + tol && self.min.y <= o.max.y + tol && o.min.y <= self.max.y + tol && self.min.z <= o.max.z + tol && o.min.z <= self.max.z + tol
    }
}
