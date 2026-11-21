//! Deterministic device math shared bit-for-bit with `cu/dops.cu` (no FMA contraction on
//! either side, no libm transcendental calls): sincos with Cody–Waite reduction and Taylor
//! kernels, feature signed-distance bounds, a u32 hash for the mock policy.
//!
//! Every expression here has a line-by-line twin in the CUDA source; change both together.

use crate::abi::*;

const TWO_OVER_PI: f64 = 6.36619772367581382433e-01;
const PIO2_HI: f64 = 1.57079632673412561417e+00;
const PIO2_LO: f64 = 6.07710050650619224932e-11;
/// sin: coefficients of r^19, r^17, …, r^3 (Taylor, 1/n!).
pub const SIN_C: [f64; 9] = [-8.22063524662433e-18, 2.8114572543455206e-15, -7.647163731819816e-13, 1.6059043836821613e-10, -2.505210838544172e-08, 2.7557319223985893e-06, -0.0001984126984126984, 0.008333333333333333, -0.16666666666666666];
/// cos: coefficients of r^20, r^18, …, r^2.
pub const COS_C: [f64; 10] = [4.110317623312165e-19, -1.5619206968586225e-16, 4.779477332387385e-14, -1.1470745597729725e-11, 2.08767569878681e-09, -2.755731922398589e-07, 2.48015873015873e-05, -0.001388888888888889, 0.041666666666666664, -0.5];
/// "Empty" distance (no material).
pub const FAR: f64 = 1.0e30;

/// sin and cos of `x`, |error| ≲ 2 ulp for |x| < 1e6.
pub fn sincos(x: f64) -> (f64, f64) {
    let k = (x * TWO_OVER_PI + 0.5).floor();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let r2 = r * r;
    // Horner over Taylor coefficients (highest first); identical tables in cu/dops.cu
    let mut ps = SIN_C[0];
    for &c in &SIN_C[1..] { ps = ps * r2 + c; }
    let s = r + r * r2 * ps;
    let mut pc = COS_C[0];
    for &c in &COS_C[1..] { pc = pc * r2 + c; }
    let c = 1.0 + r2 * pc;
    let q = (k as i64).rem_euclid(4);
    match q {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Resolved value `j` of a feature: literal, or parameter + literal offset.
#[inline]
pub fn value(n: &Node, params: &[f64], j: usize) -> f64 {
    let p = n.pref[j];
    if p == NO_PARAM { n.val[j] } else { params[p as usize] + n.val[j] }
}

/// Extrude a 2D signed distance `d2` between `z0` and `z1` (exact composition).
#[inline]
fn extrude(d2: f64, z: f64, z0: f64, z1: f64) -> f64 {
    let dz = (z0 - z).max(z - z1);
    let a = d2.max(0.0);
    let b = dz.max(0.0);
    (a * a + b * b).sqrt() + d2.max(dz).min(0.0)
}

/// Signed-distance bound of one (unpatterned) profile prism at (x, y, z): |f| ≤ distance to
/// the surface and the sign is exact, so a cell whose centre has |f| > half-diagonal is
/// entirely inside or outside. Degenerate profiles are empty.
pub fn prism_sdf(kind: u32, v: &[f64; NVALS], x: f64, y: f64, z: f64) -> f64 {
    let (z0, z1) = (v[4], v[5]);
    if !(z1 > z0) { return FAR; }
    if kind == profile::RECT {
        let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
        if !(x1 > x0 && y1 > y0) { return FAR; }
        let qx = (x0 - x).max(x - x1);
        let qy = (y0 - y).max(y - y1);
        let qz = (z0 - z).max(z - z1);
        let a = qx.max(0.0);
        let b = qy.max(0.0);
        let c = qz.max(0.0);
        return (a * a + b * b + c * c).sqrt() + qx.max(qy.max(qz)).min(0.0);
    }
    if kind == profile::CIRCLE {
        let r = v[2];
        if !(r > 0.0) { return FAR; }
        let dx = x - v[0];
        let dy = y - v[1];
        let d2 = (dx * dx + dy * dy).sqrt() - r;
        return extrude(d2, z, z0, z1);
    }
    // regular polygon: max over edge half-planes (exact inside, a lower bound outside)
    let r = v[2];
    let sides = v[3].floor();
    if !(r > 0.0) || !(sides >= 3.0) || sides > 64.0 { return FAR; }
    let k = sides as u32;
    let (sa, ca) = sincos(core::f64::consts::PI / sides);
    let apothem = r * ca;
    let _ = sa;
    let dx = x - v[0];
    let dy = y - v[1];
    let mut d2 = -FAR;
    for i in 0..k {
        let ang = core::f64::consts::TAU * (i as f64 + 0.5) / sides;
        let (s, c) = sincos(ang);
        d2 = d2.max(dx * c + dy * s - apothem);
    }
    extrude(d2, z, z0, z1)
}

/// Distance bound of a feature's tool, including its circular pattern about the z axis.
pub fn tool_sdf(n: &Node, params: &[f64], x: f64, y: f64, z: f64) -> f64 {
    let mut v = [0.0f64; NVALS];
    for (j, vj) in v.iter_mut().enumerate() { *vj = value(n, params, j); }
    if n.flags & nflags::PATTERN == 0 { return prism_sdf(n.kind, &v, x, y, z); }
    let cnt = v[6].floor();
    let count = if cnt < 1.0 { 1 } else if cnt > MAX_PATTERN as f64 { MAX_PATTERN } else { cnt as u32 };
    let total = v[7];
    let full = (total - core::f64::consts::TAU).abs() < 1e-9;
    let step = if full { total / count as f64 } else if count > 1 { total / (count - 1) as f64 } else { 0.0 };
    let mut f = FAR;
    for i in 0..count {
        // evaluate the instance rotated by +a at the point rotated by −a
        let (s, c) = sincos(step * i as f64);
        let px = x * c + y * s;
        let py = y * c - x * s;
        f = f.min(prism_sdf(n.kind, &v, px, py, z));
    }
    f
}

/// Compile one feature: resolve values (f64 → f32), precompute pattern rotations and polygon
/// normals (computed in f64 with the deterministic `sincos`, then rounded to f32).
pub fn compile_node(n: &Node, params: &[f64], table: &mut [f32]) -> CNode { compile_node_pool(n, params, &[], table) }

/// `compile_node` with the design's profile pool (sketch / revolve segments are copied into
/// the node's table so evaluation stays self-contained).
pub fn compile_node_pool(n: &Node, params: &[f64], pool: &[f32], table: &mut [f32]) -> CNode {
    let mut v = [0.0f64; NVALS];
    for (j, vj) in v.iter_mut().enumerate() { *vj = value(n, params, j); }
    let mut vf = [0.0f32; NVALS];
    for j in 0..NVALS { vf[j] = v[j] as f32; }
    let mut c = CNode { kind: n.kind, op: n.op, active: (n.flags & nflags::ACTIVE != 0) as u32, count: 1, sides: 0, mods: n.flags & nflags::MODS, _pad: [0; 2], v: vf, apothem: 0.0, cosd: 1.0, _pad2: [0.0; 6] };
    if n.flags & nflags::DRAFT != 0 { c.cosd = (1.0 / (1.0 + v[12] * v[12]).sqrt()) as f32; }
    table[0] = 1.0;
    table[1] = 0.0;
    table[2] = 0.0;
    table[3] = 0.0;
    if n.flags & nflags::PATTERN != 0 {
        let cnt = v[6].floor();
        let count = if cnt < 1.0 { 1 } else if cnt > MAX_PATTERN as f64 { MAX_PATTERN } else { cnt as u32 };
        let total = v[7];
        let full = (total - core::f64::consts::TAU).abs() < 1e-9;
        let step = if full { total / count as f64 } else if count > 1 { total / (count - 1) as f64 } else { 0.0 };
        for i in 0..count {
            let (s, co) = sincos(step * i as f64);
            let k = 4 * i as usize;
            table[k] = co as f32;
            table[k + 1] = s as f32;
            table[k + 2] = 0.0;
            table[k + 3] = 0.0;
        }
        c.count = count;
    } else if n.flags & nflags::PATTERN_RECT != 0 {
        let c1f = v[6].floor();
        let c1 = if c1f < 1.0 { 1 } else if c1f > MAX_PATTERN as f64 { MAX_PATTERN } else { c1f as u32 };
        let c2f = v[8].floor();
        let lim = MAX_PATTERN / c1;
        let c2 = if c2f < 1.0 { 1 } else if c2f > lim as f64 { lim } else { c2f as u32 };
        let mut k = 0usize;
        for j in 0..c2 {
            for i in 0..c1 {
                table[4 * k] = 1.0;
                table[4 * k + 1] = 0.0;
                table[4 * k + 2] = (i as f64 * v[7]) as f32;
                table[4 * k + 3] = (j as f64 * v[9]) as f32;
                k += 1;
            }
        }
        c.count = c1 * c2;
    }
    if n.kind == profile::POLYGON {
        let sides = v[3].floor();
        if v[2] > 0.0 && sides >= 3.0 && sides <= 64.0 {
            let k = sides as u32;
            let (_, ca) = sincos(core::f64::consts::PI / sides);
            c.apothem = (v[2] * ca) as f32;
            for i in 0..k {
                let ang = core::f64::consts::TAU * (i as f64 + 0.5) / sides;
                let (s, co) = sincos(ang);
                table[POLY + 2 * i as usize] = co as f32;
                table[POLY + 2 * i as usize + 1] = s as f32;
            }
            c.sides = k;
        }
    }
    if n.kind == profile::SKETCH || n.kind == profile::REVOLVE {
        let idx = v[15].floor();
        if idx >= 0.0 && ((idx as usize) + 1) * PSTRIDE <= pool.len() {
            let o = idx as usize * PSTRIDE;
            let ns = (pool[o] as usize).min(MAX_SEGS);
            table[POLY..POLY + 8 * ns].copy_from_slice(&pool[o + 1..o + 1 + 8 * ns]);
            c.sides = ns as u32;
        }
    }
    c
}

/// Exact signed 2D distance to a pool sketch (`n` segments at `t[POLY..]`): minimum distance
/// to the segments; inside = chord-polygon crossing parity XOR the parity of the circular
/// segments (between chord and arc, sweep ≤ π) containing the point. No transcendentals.
pub fn sketch2d(t: &[f32], n: u32, x: f32, y: f32) -> f32 {
    let mut dmin = FAR32;
    let mut inside = false;
    for s in 0..n as usize {
        let r = &t[POLY + 8 * s..POLY + 8 * s + 8];
        let (ax, ay, bx, by) = (r[1], r[2], r[3], r[4]);
        if (ay > y) != (by > y) {
            let xi = ax + (y - ay) * (bx - ax) / (by - ay);
            if x < xi { inside = !inside; }
        }
        let d;
        if r[0] == 0.0 {
            let (dx, dy) = (bx - ax, by - ay);
            let l2 = dx * dx + dy * dy;
            let mut u = if l2 > 0.0 { ((x - ax) * dx + (y - ay) * dy) / l2 } else { 0.0 };
            if u < 0.0 { u = 0.0; }
            if u > 1.0 { u = 1.0; }
            let (px, py) = (x - (ax + u * dx), y - (ay + u * dy));
            d = (px * px + py * py).sqrt();
        } else {
            let (cx, cy, rr) = (r[5], r[6], r[7].abs());
            let sg = if r[7] > 0.0 { 1.0f32 } else { -1.0f32 };
            let (qx, qy) = (x - cx, y - cy);
            let (ux, uy, vx, vy) = (ax - cx, ay - cy, bx - cx, by - cy);
            let span = sg * (ux * qy - uy * qx) >= 0.0 && sg * (qx * vy - qy * vx) >= 0.0;
            let q2 = qx * qx + qy * qy;
            if span {
                d = (q2.sqrt() - rr).abs();
            } else {
                let (pax, pay, pbx, pby) = (x - ax, y - ay, x - bx, y - by);
                d = (pax * pax + pay * pay).sqrt().min((pbx * pbx + pby * pby).sqrt());
            }
            let side = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if side * sg < 0.0 && q2 < rr * rr { inside = !inside; }
        }
        if d < dmin { dmin = d; }
    }
    if inside { -dmin } else { dmin }
}

/// 2D signed distance of a compiled profile (FAR32 if degenerate).
pub fn profile_d2(c: &CNode, table: &[f32], x: f32, y: f32) -> f32 {
    let v = &c.v;
    match c.kind {
        profile::RECT => {
            let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
            if !(x1 > x0 && y1 > y0) { return FAR32; }
            let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
            let (hx, hy) = (0.5 * (x1 - x0), 0.5 * (y1 - y0));
            let ax = (x - cx).abs();
            let ay = (y - cy).abs();
            let k = v[10].max(0.0).min(hx.min(hy));
            if c.mods & nflags::FILLET != 0 {
                let qx = ax - (hx - k);
                let qy = ay - (hy - k);
                let a = qx.max(0.0);
                let b = qy.max(0.0);
                (a * a + b * b).sqrt() + qx.max(qy).min(0.0) - k
            } else {
                let qx = ax - hx;
                let qy = ay - hy;
                let a = qx.max(0.0);
                let b = qy.max(0.0);
                let dbox = (a * a + b * b).sqrt() + qx.max(qy).min(0.0);
                if c.mods & nflags::CHAMFER != 0 { dbox.max((ax + ay - (hx + hy - k)) * 0.70710678) } else { dbox }
            }
        }
        profile::CIRCLE => {
            let r = v[2];
            if !(r > 0.0) { return FAR32; }
            let dx = x - v[0];
            let dy = y - v[1];
            (dx * dx + dy * dy).sqrt() - r
        }
        profile::SKETCH => if c.sides == 0 { FAR32 } else { sketch2d(table, c.sides, x - v[0], y - v[1]) },
        _ => {
            if c.sides == 0 { return FAR32; }
            let dx = x - v[0];
            let dy = y - v[1];
            let mut d2 = -FAR32;
            for i in 0..c.sides as usize { d2 = d2.max(dx * table[POLY + 2 * i] + dy * table[POLY + 2 * i + 1] - c.apothem); }
            d2
        }
    }
}

/// Extrusion of a 2D distance with the z modifiers (draft, rounded / chamfered caps).
fn zmod(c: &CNode, d2: f32, z: f32) -> f32 {
    let v = &c.v;
    let (z0, z1) = (v[4], v[5]);
    let d = if c.mods & nflags::DRAFT != 0 { (d2 + (z - z0) * v[12]) * c.cosd } else { d2 };
    let dz = (z0 - z).max(z - z1);
    let r = v[13].max(0.0).min(0.5 * (z1 - z0));
    if c.mods & nflags::ROUND_Z != 0 {
        let qx = d + r;
        let qy = dz + r;
        let a = qx.max(0.0);
        let b = qy.max(0.0);
        return qx.max(qy).min(0.0) + (a * a + b * b).sqrt() - r;
    }
    if c.mods & nflags::CHAMFER_Z != 0 { return extrude32(d, z, z0, z1).max((d + dz + r) * 0.70710678); }
    extrude32(d, z, z0, z1)
}

const ZMODS: u32 = nflags::DRAFT | nflags::ROUND_Z | nflags::CHAMFER_Z;

/// Full revolution of a pool sketch in (ρ, z) about the vertical axis through (v0, v1).
fn revolve_sdf(c: &CNode, table: &[f32], x: f32, y: f32, z: f32) -> f32 {
    if c.sides == 0 { return FAR32; }
    let dx = x - c.v[0];
    let dy = y - c.v[1];
    sketch2d(table, c.sides, (dx * dx + dy * dy).sqrt(), z)
}

/// "Empty" distance in evaluation precision.
pub const FAR32: f32 = 1.0e30;

#[inline]
fn extrude32(d2: f32, z: f32, z0: f32, z1: f32) -> f32 {
    let dz = (z0 - z).max(z - z1);
    let a = d2.max(0.0);
    let b = dz.max(0.0);
    (a * a + b * b).sqrt() + d2.max(dz).min(0.0)
}

/// Distance bound of a compiled prism (f32; twin of `prism_sdf_c` in cu/dops.cu).
pub fn prism_sdf_c(c: &CNode, table: &[f32], x: f32, y: f32, z: f32) -> f32 {
    if c.kind == profile::REVOLVE { return revolve_sdf(c, table, x, y, z); }
    let v = &c.v;
    let (z0, z1) = (v[4], v[5]);
    if !(z1 > z0) { return FAR32; }
    if c.mods & ZMODS != 0 {
        let d2 = profile_d2(c, table, x, y);
        return if d2 == FAR32 { FAR32 } else { zmod(c, d2, z) };
    }
    if c.kind == profile::RECT && c.mods & (nflags::FILLET | nflags::CHAMFER) == 0 {
        let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
        if !(x1 > x0 && y1 > y0) { return FAR32; }
        let qx = (x0 - x).max(x - x1);
        let qy = (y0 - y).max(y - y1);
        let qz = (z0 - z).max(z - z1);
        let a = qx.max(0.0);
        let b = qy.max(0.0);
        let cc = qz.max(0.0);
        return (a * a + b * b + cc * cc).sqrt() + qx.max(qy.max(qz)).min(0.0);
    }
    let d2 = profile_d2(c, table, x, y);
    if d2 == FAR32 { return FAR32; }
    extrude32(d2, z, z0, z1)
}

/// Signed-distance bound of a compiled tape (f32).
fn tape_sdf_c_core(nodes: &[CNode], tables: &[f32], x: f32, y: f32, z: f32) -> f32 {
    let mut f = FAR32;
    let mut stack = [FAR32; MAX_BODIES];
    let mut depth = 0usize;
    for (i, c) in nodes.iter().enumerate() {
        if c.active == 0 || c.kind == profile::MIRROR { continue; }
        if c.kind == profile::SHELL { f = f.max(-f - c.v[0]); continue; }
        if c.kind == profile::BODY { if depth < MAX_BODIES { stack[depth] = f; depth += 1; f = FAR32; } continue; }
        if c.kind == profile::COMBINE {
            if depth > 0 {
                depth -= 1;
                let t = stack[depth];
                f = if c.op == fop::ADD { t.min(f) } else if c.op == fop::REMOVE { t.max(-f) } else { t.max(f) };
            }
            continue;
        }
        let t = &tables[i * TABLE..(i + 1) * TABLE];
        let mut tf = FAR32;
        for k in 0..c.count as usize {
            let (co, s, tx, ty) = (t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
            let qx = x - tx;
            let qy = y - ty;
            let px = qx * co + qy * s;
            let py = qy * co - qx * s;
            tf = tf.min(prism_sdf_c(c, t, px, py, z));
        }
        f = if c.op == fop::ADD { f.min(tf) } else { f.max(-tf) };
    }
    f
}

/// Signed-distance bound of a whole tape (features folded in order).
pub fn tape_sdf(nodes: &[Node], params: &[f64], x: f64, y: f64, z: f64) -> f64 {
    let mut f = FAR;
    for n in nodes {
        if n.flags & nflags::ACTIVE == 0 { continue; }
        let t = tool_sdf(n, params, x, y, z);
        f = if n.op == fop::ADD { f.min(t) } else { f.max(-t) };
    }
    f
}

/// splitmix-style u32 hash (identical wrapping arithmetic on both sides).
pub fn hash3(a: u32, b: u32, c: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77) ^ c.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    h
}

/// Uniform in [0, 1) from a hash (24 bits).
pub fn unit(h: u32) -> f64 { (h >> 8) as f64 * (1.0 / 16777216.0) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sincos_accuracy() {
        for i in -2000..2000 {
            let x = i as f64 * 0.0137 - 3.0;
            let (s, c) = sincos(x);
            assert!((s - x.sin()).abs() < 1e-15 && (c - x.cos()).abs() < 1e-15, "{x}: {s} {} {c} {}", x.sin(), x.cos());
        }
        let (s, c) = sincos(core::f64::consts::TAU);
        assert!(s.abs() < 1e-15 && (c - 1.0).abs() < 1e-15);
    }
    #[test]
    fn prism_distances_are_exact_on_axes() {
        let v = [0.0, 0.0, 10.0, 10.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(prism_sdf(profile::RECT, &v, 5.0, 5.0, 2.5), -2.5);
        assert_eq!(prism_sdf(profile::RECT, &v, 13.0, 5.0, 2.5), 3.0);
        let c = [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((prism_sdf(profile::CIRCLE, &c, 5.0, 0.0, 0.5) - 3.0).abs() < 1e-15);
        let h = [0.0, 0.0, 2.0, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = 2.0 * (core::f64::consts::PI / 6.0).cos();
        assert!((prism_sdf(profile::POLYGON, &h, 0.0, 0.0, 0.5) - (-0.5f64).max(-a)).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------------------
// Interval arithmetic over cell boxes (f32; twin of the `iv_*` functions in cu/dops.cu).
// Round-to-nearest per op; callers widen the final interval by the grid's error margin.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iv { pub lo: f32, pub hi: f32 }

#[inline] pub fn iv(lo: f32, hi: f32) -> Iv { Iv { lo, hi } }
#[inline] fn sub_ci(c: f32, x: Iv) -> Iv { iv(c - x.hi, c - x.lo) }
#[inline] fn sub_ic(x: Iv, c: f32) -> Iv { iv(x.lo - c, x.hi - c) }
#[inline] fn sub(a: Iv, b: Iv) -> Iv { iv(a.lo - b.hi, a.hi - b.lo) }
#[inline] fn add(a: Iv, b: Iv) -> Iv { iv(a.lo + b.lo, a.hi + b.hi) }
#[inline] fn imax(a: Iv, b: Iv) -> Iv { iv(a.lo.max(b.lo), a.hi.max(b.hi)) }
#[inline] fn imin(a: Iv, b: Iv) -> Iv { iv(a.lo.min(b.lo), a.hi.min(b.hi)) }
#[inline] fn maxc(a: Iv, c: f32) -> Iv { iv(a.lo.max(c), a.hi.max(c)) }
#[inline] fn minc(a: Iv, c: f32) -> Iv { iv(a.lo.min(c), a.hi.min(c)) }
#[inline] fn neg(a: Iv) -> Iv { iv(-a.hi, -a.lo) }
#[inline] fn scale(a: Iv, c: f32) -> Iv { if c >= 0.0 { iv(a.lo * c, a.hi * c) } else { iv(a.hi * c, a.lo * c) } }
#[inline] fn sq(a: Iv) -> Iv {
    if a.lo >= 0.0 { iv(a.lo * a.lo, a.hi * a.hi) } else if a.hi <= 0.0 { iv(a.hi * a.hi, a.lo * a.lo) } else { iv(0.0, (a.lo * a.lo).max(a.hi * a.hi)) }
}
#[inline] fn isqrt(a: Iv) -> Iv { iv(a.lo.max(0.0).sqrt(), a.hi.max(0.0).sqrt()) }
#[inline] fn iabs(a: Iv) -> Iv { if a.lo >= 0.0 { a } else if a.hi <= 0.0 { iv(-a.hi, -a.lo) } else { iv(0.0, (-a.lo).max(a.hi)) } }

fn extrude_iv(d2: Iv, z: Iv, z0: f32, z1: f32) -> Iv {
    let dz = imax(sub_ci(z0, z), sub_ic(z, z1));
    let a = maxc(d2, 0.0);
    let b = maxc(dz, 0.0);
    add(isqrt(add(sq(a), sq(b))), minc(imax(d2, dz), 0.0))
}

/// Interval of a compiled prism's distance bound over a box.
/// Lipschitz-1 interval from the centre point (every compiled field is 1-Lipschitz).
fn center_iv(f: f32, r: f32) -> Iv { iv(f - r, f + r) }
#[inline] fn mid(a: Iv) -> f32 { 0.5 * (a.lo + a.hi) }
#[inline] fn wid(a: Iv) -> f32 { a.hi - a.lo }

pub fn prism_iv(c: &CNode, table: &[f32], x: Iv, y: Iv, z: Iv) -> Iv {
    let v = &c.v;
    let far = iv(FAR32, FAR32);
    if c.kind == profile::REVOLVE {
        if c.sides == 0 { return far; }
        let rho = isqrt(add(sq(sub_ic(x, v[0])), sq(sub_ic(y, v[1]))));
        let (wr, wz) = (wid(rho), wid(z));
        return center_iv(sketch2d(table, c.sides, mid(rho), mid(z)), 0.5 * (wr * wr + wz * wz).sqrt());
    }
    let (z0, z1) = (v[4], v[5]);
    if !(z1 > z0) { return far; }
    if c.mods & ZMODS != 0 {
        let (wx, wy, wz) = (wid(x), wid(y), wid(z));
        let f = prism_sdf_c(c, table, mid(x), mid(y), mid(z));
        if f == FAR32 { return far; }
        return center_iv(f, 0.5 * (wx * wx + wy * wy + wz * wz).sqrt());
    }
    if c.kind == profile::SKETCH {
        if c.sides == 0 { return far; }
        let (wx, wy) = (wid(x), wid(y));
        let d = sketch2d(table, c.sides, mid(x) - v[0], mid(y) - v[1]);
        return extrude_iv(center_iv(d, 0.5 * (wx * wx + wy * wy).sqrt()), z, z0, z1);
    }
    if c.kind == profile::RECT {
        let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
        if !(x1 > x0 && y1 > y0) { return far; }
        if c.mods & (nflags::FILLET | nflags::CHAMFER) != 0 {
            let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
            let (hx, hy) = (0.5 * (x1 - x0), 0.5 * (y1 - y0));
            let ax = iabs(sub_ic(x, cx));
            let ay = iabs(sub_ic(y, cy));
            let k = v[10].max(0.0).min(hx.min(hy));
            let d2 = if c.mods & nflags::FILLET != 0 {
                let qx = sub_ic(ax, hx - k);
                let qy = sub_ic(ay, hy - k);
                sub_ic(add(isqrt(add(sq(maxc(qx, 0.0)), sq(maxc(qy, 0.0)))), minc(imax(qx, qy), 0.0)), k)
            } else {
                let qx = sub_ic(ax, hx);
                let qy = sub_ic(ay, hy);
                let dbox = add(isqrt(add(sq(maxc(qx, 0.0)), sq(maxc(qy, 0.0)))), minc(imax(qx, qy), 0.0));
                imax(dbox, scale(sub_ic(add(ax, ay), hx + hy - k), 0.70710678))
            };
            return extrude_iv(d2, z, z0, z1);
        }
        let qx = imax(sub_ci(x0, x), sub_ic(x, x1));
        let qy = imax(sub_ci(y0, y), sub_ic(y, y1));
        let qz = imax(sub_ci(z0, z), sub_ic(z, z1));
        let s = isqrt(add(add(sq(maxc(qx, 0.0)), sq(maxc(qy, 0.0))), sq(maxc(qz, 0.0))));
        return add(s, minc(imax(qx, imax(qy, qz)), 0.0));
    }
    if c.kind == profile::CIRCLE {
        let r = v[2];
        if !(r > 0.0) { return far; }
        let dx = sub_ic(x, v[0]);
        let dy = sub_ic(y, v[1]);
        let d2 = sub_ic(isqrt(add(sq(dx), sq(dy))), r);
        return extrude_iv(d2, z, z0, z1);
    }
    if c.sides == 0 { return far; }
    let dx = sub_ic(x, v[0]);
    let dy = sub_ic(y, v[1]);
    let mut d2 = iv(-FAR32, -FAR32);
    for i in 0..c.sides as usize {
        let lin = sub_ic(add(scale(dx, table[POLY + 2 * i]), scale(dy, table[POLY + 2 * i + 1])), c.apothem);
        d2 = imax(d2, lin);
    }
    extrude_iv(d2, z, z0, z1)
}

/// Interval of a compiled tape's distance bound over a box (pattern instances rotate the box
/// conservatively to its axis-aligned hull).
fn tape_iv_core(nodes: &[CNode], tables: &[f32], x: Iv, y: Iv, z: Iv) -> Iv {
    let mut f = iv(FAR32, FAR32);
    let mut stack = [iv(FAR32, FAR32); MAX_BODIES];
    let mut depth = 0usize;
    for (i, c) in nodes.iter().enumerate() {
        if c.active == 0 || c.kind == profile::MIRROR { continue; }
        if c.kind == profile::SHELL { f = imax(f, sub_ic(neg(f), c.v[0])); continue; }
        if c.kind == profile::BODY { if depth < MAX_BODIES { stack[depth] = f; depth += 1; f = iv(FAR32, FAR32); } continue; }
        if c.kind == profile::COMBINE {
            if depth > 0 {
                depth -= 1;
                let t = stack[depth];
                f = if c.op == fop::ADD { imin(t, f) } else if c.op == fop::REMOVE { imax(t, neg(f)) } else { imax(t, f) };
            }
            continue;
        }
        let t = &tables[i * TABLE..(i + 1) * TABLE];
        let mut tf = iv(FAR32, FAR32);
        for k in 0..c.count as usize {
            let (co, s, tx, ty) = (t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
            let qx = sub_ic(x, tx);
            let qy = sub_ic(y, ty);
            let px = add(scale(qx, co), scale(qy, s));
            let py = sub(scale(qy, co), scale(qx, s));
            tf = imin(tf, prism_iv(c, t, px, py, z));
        }
        f = if c.op == fop::ADD { imin(f, tf) } else { imax(f, neg(tf)) };
    }
    f
}


// ------------------------------------------------------------------------------------
// Per-cell tape pruning: terms that cannot affect the field anywhere in a cell
// ------------------------------------------------------------------------------------

/// Nodes eligible for pruning (higher node indices are always evaluated).
pub const PRUNE_NODES: usize = 32;

/// Pruned tape for one cell: skipped nodes and, per node, the pattern instances still needed.
#[derive(Clone, Copy, Debug)]
pub struct Prune { pub skip: u32, pub inst: [u64; PRUNE_NODES] }

impl Prune {
    pub const NONE: Prune = Prune { skip: 0, inst: [u64::MAX; PRUNE_NODES] };
    #[inline] fn skipped(&self, i: usize) -> bool { i < PRUNE_NODES && (self.skip >> i) & 1 != 0 }
    #[inline] fn keeps(&self, i: usize, k: usize) -> bool { i >= PRUNE_NODES || k >= 64 || (self.inst[i] >> k) & 1 != 0 }
}

/// `tape_iv` over a cell that also records which terms are dominated by more than `m` (a
/// margin above f32 rounding) everywhere in the cell: such terms cannot change any min/max
/// at any point of the cell, so point evaluations with the pruned tape are bit-identical.
fn tape_iv_prune_core(nodes: &[CNode], tables: &[f32], x: Iv, y: Iv, z: Iv, m: f32, pr: &mut Prune) -> Iv {
    *pr = Prune::NONE;
    let mut f = iv(FAR32, FAR32);
    let mut stack = [iv(FAR32, FAR32); MAX_BODIES];
    let mut depth = 0usize;
    let mut ivs = [iv(FAR32, FAR32); 64];
    for (i, c) in nodes.iter().enumerate() {
        if c.active == 0 || c.kind == profile::MIRROR { continue; }
        if c.kind == profile::SHELL { f = imax(f, sub_ic(neg(f), c.v[0])); continue; }
        if c.kind == profile::BODY { if depth < MAX_BODIES { stack[depth] = f; depth += 1; f = iv(FAR32, FAR32); } continue; }
        if c.kind == profile::COMBINE {
            if depth > 0 {
                depth -= 1;
                let t = stack[depth];
                f = if c.op == fop::ADD { imin(t, f) } else if c.op == fop::REMOVE { imax(t, neg(f)) } else { imax(t, f) };
            }
            continue;
        }
        let t = &tables[i * TABLE..(i + 1) * TABLE];
        let mut tf = iv(FAR32, FAR32);
        let cn = c.count as usize;
        for k in 0..cn {
            let (co, s, tx, ty) = (t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
            let qx = sub_ic(x, tx);
            let qy = sub_ic(y, ty);
            let px = add(scale(qx, co), scale(qy, s));
            let py = sub(scale(qy, co), scale(qx, s));
            let r = prism_iv(c, t, px, py, z);
            if k < 64 { ivs[k] = r; }
            tf = imin(tf, r);
        }
        if i < PRUNE_NODES {
            if cn > 1 && cn <= 64 {
                let mut mask = 0u64;
                for (k, r) in ivs.iter().enumerate().take(cn) { if !(r.lo > tf.hi + m) { mask |= 1 << k; } }
                pr.inst[i] = mask;
            }
            let dominated = if c.op == fop::ADD { tf.lo > f.hi + m } else { -tf.lo < f.lo - m };
            if dominated { pr.skip |= 1 << i; }
        }
        f = if c.op == fop::ADD { imin(f, tf) } else { imax(f, neg(tf)) };
    }
    f
}

/// `tape_sdf_c` with a cell's pruned tape (bit-identical inside that cell).
fn tape_sdf_p_core(nodes: &[CNode], tables: &[f32], pr: &Prune, x: f32, y: f32, z: f32) -> f32 {
    let mut f = FAR32;
    let mut stack = [FAR32; MAX_BODIES];
    let mut depth = 0usize;
    for (i, c) in nodes.iter().enumerate() {
        if c.active == 0 || c.kind == profile::MIRROR { continue; }
        if c.kind == profile::SHELL { f = f.max(-f - c.v[0]); continue; }
        if c.kind == profile::BODY { if depth < MAX_BODIES { stack[depth] = f; depth += 1; f = FAR32; } continue; }
        if c.kind == profile::COMBINE {
            if depth > 0 {
                depth -= 1;
                let t = stack[depth];
                f = if c.op == fop::ADD { t.min(f) } else if c.op == fop::REMOVE { t.max(-f) } else { t.max(f) };
            }
            continue;
        }
        if pr.skipped(i) { continue; }
        let t = &tables[i * TABLE..(i + 1) * TABLE];
        let mut tf = FAR32;
        for k in 0..c.count as usize {
            if !pr.keeps(i, k) { continue; }
            let (co, s, tx, ty) = (t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
            let qx = x - tx;
            let qy = y - ty;
            let px = qx * co + qy * s;
            let py = qy * co - qx * s;
            tf = tf.min(prism_sdf_c(c, t, px, py, z));
        }
        f = if c.op == fop::ADD { f.min(tf) } else { f.max(-tf) };
    }
    f
}

/// `tape_iv` with a cell's pruned tape, for sub-boxes of that cell.
fn tape_iv_p_core(nodes: &[CNode], tables: &[f32], pr: &Prune, x: Iv, y: Iv, z: Iv) -> Iv {
    let mut f = iv(FAR32, FAR32);
    let mut stack = [iv(FAR32, FAR32); MAX_BODIES];
    let mut depth = 0usize;
    for (i, c) in nodes.iter().enumerate() {
        if c.active == 0 || c.kind == profile::MIRROR { continue; }
        if c.kind == profile::SHELL { f = imax(f, sub_ic(neg(f), c.v[0])); continue; }
        if c.kind == profile::BODY { if depth < MAX_BODIES { stack[depth] = f; depth += 1; f = iv(FAR32, FAR32); } continue; }
        if c.kind == profile::COMBINE {
            if depth > 0 {
                depth -= 1;
                let t = stack[depth];
                f = if c.op == fop::ADD { imin(t, f) } else if c.op == fop::REMOVE { imax(t, neg(f)) } else { imax(t, f) };
            }
            continue;
        }
        if pr.skipped(i) { continue; }
        let t = &tables[i * TABLE..(i + 1) * TABLE];
        let mut tf = iv(FAR32, FAR32);
        for k in 0..c.count as usize {
            if !pr.keeps(i, k) { continue; }
            let (co, s, tx, ty) = (t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
            let qx = sub_ic(x, tx);
            let qy = sub_ic(y, ty);
            let px = add(scale(qx, co), scale(qy, s));
            let py = sub(scale(qy, co), scale(qx, s));
            tf = imin(tf, prism_iv(c, t, px, py, z));
        }
        f = if c.op == fop::ADD { imin(f, tf) } else { imax(f, neg(tf)) };
    }
    f
}


// ------------------------------------------------------------------------------------
// Mirror (body-level suffix): the field is the min over the reflection combinations
// ------------------------------------------------------------------------------------

/// Mirror planes of a tape (bit 0 = x = 0, bit 1 = y = 0), cached by the compile pass in the
/// first node's `_pad[0]` (bit 31 marks it valid).
pub fn mirror_mask(nodes: &[CNode]) -> u32 {
    match nodes.first() { Some(c) if c._pad[0] & 0x8000_0000 != 0 => c._pad[0] & 3, _ => scan_mirror(nodes) }
}

/// Scan a tape for mirror nodes (compile pass).
pub fn scan_mirror(nodes: &[CNode]) -> u32 {
    let mut m = 0u32;
    for c in nodes {
        if c.active == 0 { continue; }
        if c.kind == profile::MIRROR {
            if c.v[0] > 0.5 { m |= 1; }
            if c.v[1] > 0.5 { m |= 2; }
        }
        if c.kind == profile::SKETCH || c.kind == profile::REVOLVE || c.mods & ZMODS != 0 { m |= 4; }
    }
    m
}

/// Signed-distance bound of a compiled tape (f32).
pub fn tape_sdf_c(nodes: &[CNode], tables: &[f32], x: f32, y: f32, z: f32) -> f32 {
    let m = mirror_mask(nodes);
    if m == 0 { return tape_sdf_c_core(nodes, tables, x, y, z); }
    let mut f = FAR32;
    for k in 0..4u32 {
        if k & !m != 0 { continue; }
        f = f.min(tape_sdf_c_core(nodes, tables, if k & 1 != 0 { -x } else { x }, if k & 2 != 0 { -y } else { y }, z));
    }
    f
}

/// Interval of a compiled tape's distance bound over a box (pattern instances rotate the box
/// conservatively to its axis-aligned hull).
pub fn tape_iv(nodes: &[CNode], tables: &[f32], x: Iv, y: Iv, z: Iv) -> Iv {
    let m = mirror_mask(nodes);
    if m == 0 { return tape_iv_core(nodes, tables, x, y, z); }
    let mut f = iv(FAR32, FAR32);
    for k in 0..4u32 {
        if k & !m != 0 { continue; }
        f = imin(f, tape_iv_core(nodes, tables, if k & 1 != 0 { neg(x) } else { x }, if k & 2 != 0 { neg(y) } else { y }, z));
    }
    f
}

/// `tape_iv` over a cell recording the terms dominated everywhere in it (see `Prune`).
/// Mirrored tapes are not pruned.
pub fn tape_iv_prune(nodes: &[CNode], tables: &[f32], x: Iv, y: Iv, z: Iv, m: f32, pr: &mut Prune) -> Iv {
    if mirror_mask(nodes) != 0 { *pr = Prune::NONE; return tape_iv(nodes, tables, x, y, z); }
    tape_iv_prune_core(nodes, tables, x, y, z, m, pr)
}

/// `tape_sdf_c` with a cell's pruned tape (bit-identical inside that cell).
pub fn tape_sdf_p(nodes: &[CNode], tables: &[f32], pr: &Prune, x: f32, y: f32, z: f32) -> f32 {
    if mirror_mask(nodes) != 0 { return tape_sdf_c(nodes, tables, x, y, z); }
    tape_sdf_p_core(nodes, tables, pr, x, y, z)
}

/// `tape_iv` with a cell's pruned tape, for sub-boxes of that cell.
pub fn tape_iv_p(nodes: &[CNode], tables: &[f32], pr: &Prune, x: Iv, y: Iv, z: Iv) -> Iv {
    if mirror_mask(nodes) != 0 { return tape_iv(nodes, tables, x, y, z); }
    tape_iv_p_core(nodes, tables, pr, x, y, z)
}
