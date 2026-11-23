//! Bridges between device designs and the exact/document world:
//!
//! - [`to_layered`]: a device design → exact layered solid (escalation of survivors; also the
//!   oracle for the soundness of device interval bounds).
//! - [`to_program`]: a device design → AgIR program of real Part Design features (sketch +
//!   pad/pocket + circular pattern, parameters as `knowledge.parameter`), so survivors become
//!   ordinary documents the LLM can describe, render, edit and export.
//! - [`raise`]: a document's feature tree → device design (the LLM's base design seeds the
//!   GPU rollout). Unsupported features are reported, never silently dropped.

use crate::abi::*;
use crate::math::value;
use crate::studio::Design;
use agcad_geo::builders::{Profile, Seg2};
use agcad_geo::kernel::{ExactKernel, NativeKernel, Solid};
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::Frame;
use agcad_geo::region2d::BoolOp2;
use agcad_ir::program::Value;
use agcad_runtime::document::PartState;
use serde_json::{json, Value as J};
use std::collections::BTreeMap;

fn rot(p: [f64; 2], a: f64) -> [f64; 2] { let (s, c) = a.sin_cos(); [p[0] * c - p[1] * s, p[0] * s + p[1] * c] }

/// Rectangle with rounded (fillet) or cut (chamfer) corners of size `k`, rotated by `a` about
/// the origin. Zero-length sides are omitted, so the full-round limit is a clean stadium /
/// circle (the device clamps k to min(hx, hy) the same way).
fn treated_rect(x0: f64, y0: f64, x1: f64, y1: f64, k: f64, fillet: bool, a: f64) -> Profile {
    let mut segs: Vec<Seg2> = Vec::new();
    let line = |segs: &mut Vec<Seg2>, p: [f64; 2], q: [f64; 2]| { if (p[0] - q[0]).hypot(p[1] - q[1]) > 1e-9 { segs.push(Seg2::Line { a: rot(p, a), b: rot(q, a) }); } };
    let pi = core::f64::consts::PI;
    // corners CCW from bottom-right: (centre, start angle)
    let corners = [([x1 - k, y0 + k], -0.5 * pi), ([x1 - k, y1 - k], 0.0), ([x0 + k, y1 - k], 0.5 * pi), ([x0 + k, y0 + k], pi)];
    let at = |c: [f64; 2], t: f64| [c[0] + k * t.cos(), c[1] + k * t.sin()];
    for (i, (c, t0)) in corners.iter().enumerate() {
        let (prev_c, prev_t0) = corners[(i + 3) % 4];
        line(&mut segs, at(prev_c, prev_t0 + 0.5 * pi), at(*c, *t0));
        if fillet {
            segs.push(Seg2::Arc { c: rot(*c, a), r: k, a0: t0 + a, a1: t0 + 0.5 * pi + a });
        } else {
            line(&mut segs, at(*c, *t0), at(*c, t0 + 0.5 * pi));
        }
    }
    if fillet && segs.iter().all(|s| matches!(s, Seg2::Arc { .. })) {
        let c = rot([0.5 * (x0 + x1), 0.5 * (y0 + y1)], a);
        return Profile::circle(c[0], c[1], k);
    }
    Profile { loops: vec![segs] }.normalize()
}

fn polygon(pts: &[[f64; 2]]) -> Profile {
    let n = pts.len();
    Profile { loops: vec![(0..n).map(|i| Seg2::Line { a: pts[i], b: pts[(i + 1) % n] }).collect()] }.normalize()
}

fn seg_end(s: &Seg, end: bool) -> [f64; 2] {
    match *s {
        Seg::Line { a, b } => if end { b } else { a },
        Seg::Arc { c, r, a0, a1 } => { let t = if end { a1 } else { a0 }; [c[0] + r * t.cos(), c[1] + r * t.sin()] }
    }
}

fn seg2_of(s: &Seg, dx: f64, dy: f64, a: f64) -> Seg2 {
    match *s {
        Seg::Line { a: p, b: q } => Seg2::Line { a: rot([p[0] + dx, p[1] + dy], a), b: rot([q[0] + dx, q[1] + dy], a) },
        Seg::Arc { c, r, a0, a1 } => {
            let c = rot([c[0] + dx, c[1] + dy], a);
            if (a1 - a0).abs() >= core::f64::consts::TAU - 1e-12 { Seg2::Circle { c, r, ccw: a1 > a0 } } else { Seg2::Arc { c, r, a0: a0 + a, a1: a1 + a } }
        }
    }
}

/// Exact profile of a pool sketch (one outer loop plus holes), translated by (dx, dy), then
/// rotated by `a` about the origin (the device convention).
pub fn sketch_profile(loops: &[Vec<Seg>], dx: f64, dy: f64, a: f64) -> Result<Profile, String> {
    let mut ps = sketch_profiles(loops, dx, dy, a)?;
    if ps.len() != 1 { return Err("sketch with several outer loops: use sketch_profiles".into()); }
    Ok(ps.remove(0))
}

/// Exact regions of a pool sketch with any number of islands: loops nested at even depth are
/// outer boundaries, their direct children holes (the device's parity rule).
pub fn sketch_profiles(loops: &[Vec<Seg>], dx: f64, dy: f64, a: f64) -> Result<Vec<Profile>, String> {
    if loops.is_empty() { return Err("empty sketch".into()); }
    let ls: Vec<Vec<Seg2>> = loops.iter().map(|lp| lp.iter().map(|s| seg2_of(s, dx, dy, a)).collect()).collect();
    let probe = |lp: &[Seg2]| match lp[0] { Seg2::Line { a, b } => [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])], Seg2::Arc { c, r, a0, a1 } => { let t = 0.5 * (a0 + a1); [c[0] + r * t.cos(), c[1] + r * t.sin()] } Seg2::Circle { c, r, .. } => [c[0] + r, c[1]] };
    let inside = |p: [f64; 2], lp: &Vec<Seg2>| agcad_geo::region2d::winding(std::slice::from_ref(lp), p).round().abs() as i64 % 2 == 1;
    // containing loops of each loop
    let parents: Vec<Vec<usize>> = (0..ls.len()).map(|i| (0..ls.len()).filter(|&j| j != i && inside(probe(&ls[i]), &ls[j])).collect()).collect();
    let mut out = Vec::new();
    for i in 0..ls.len() {
        if parents[i].len() % 2 == 1 { continue; } // a hole
        let mut prof = Profile { loops: vec![ls[i].clone()] };
        for j in 0..ls.len() {
            // direct child: inside i and exactly one level deeper
            if parents[j].len() == parents[i].len() + 1 && parents[j].contains(&i) { prof.loops.push(ls[j].clone()); }
        }
        out.push(prof.normalize());
    }
    Ok(out)
}


/// Group sketch segments into closed loops (endpoints matched within 1e-6; segments are
/// reversed as needed).
pub fn close_loops(segs: &[Seg]) -> Result<Vec<Vec<Seg>>, String> {
    let near = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6;
    let rev = |s: &Seg| match *s { Seg::Line { a, b } => Seg::Line { a: b, b: a }, Seg::Arc { c, r, a0, a1 } => Seg::Arc { c, r, a0: a1, a1: a0 } };
    let mut used = vec![false; segs.len()];
    let mut loops = Vec::new();
    for i in 0..segs.len() {
        if used[i] { continue; }
        used[i] = true;
        let mut lp = vec![segs[i]];
        let start = seg_end(&segs[i], false);
        // a full circle closes on itself
        while !near(seg_end(lp.last().unwrap(), true), start) {
            let e = seg_end(lp.last().unwrap(), true);
            let Some(j) = (0..segs.len()).find(|&j| !used[j] && (near(seg_end(&segs[j], false), e) || near(seg_end(&segs[j], true), e))) else {
                return Err(format!("open sketch loop at ({:.4}, {:.4})", e[0], e[1]));
            };
            used[j] = true;
            lp.push(if near(seg_end(&segs[j], false), e) { segs[j] } else { rev(&segs[j]) });
        }
        loops.push(lp);
    }
    Ok(loops)
}

fn pool_index(d: &Design, n: &Node, i: usize) -> Result<usize, String> {
    let idx = value(n, &d.params, 15).floor();
    if idx < 0.0 || idx as usize >= d.profiles.len() { return Err(format!("feature {i}: sketch profile {idx} missing from the design")); }
    Ok(idx as usize)
}

/// Profile of one feature instance rotated by `a` about the z axis.
fn profile_of(kind: u32, v: &[f64; NVALS], a: f64) -> Result<Profile, String> {
    match kind {
        profile::RECT => {
            let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
            if !(x1 > x0 && y1 > y0) { return Err("degenerate rectangle".into()); }
            if a == 0.0 { return Ok(Profile::rectangle(x0, y0, x1, y1)); }
            Ok(polygon(&[rot([x0, y0], a), rot([x1, y0], a), rot([x1, y1], a), rot([x0, y1], a)]))
        }
        profile::CIRCLE => {
            if !(v[2] > 0.0) { return Err("non-positive radius".into()); }
            let c = rot([v[0], v[1]], a);
            Ok(Profile::circle(c[0], c[1], v[2]))
        }
        _ => {
            let k = v[3].floor();
            if !(v[2] > 0.0) || !(3.0..=64.0).contains(&k) { return Err("bad polygon".into()); }
            let k = k as usize;
            let pts: Vec<[f64; 2]> = (0..k).map(|i| { let t = core::f64::consts::TAU * i as f64 / k as f64; rot([v[0] + v[2] * t.cos(), v[1] + v[2] * t.sin()], a) }).collect();
            Ok(polygon(&pts))
        }
    }
}

/// Pattern instances as (rotation θ, translation tx, ty): instance = R(θ)·tool + t, the same
/// clamps as `compile_node` on the device.
fn instances(n: &Node, v: &[f64; NVALS], d: &Design) -> Vec<(f64, f64, f64)> {
    if n.flags & nflags::PATTERN_USER != 0 {
        let idx = v[6].floor();
        let pts: Vec<(f64, f64, f64)> = if idx >= 0.0 { d.profiles.get(idx as usize).map(|p| p.iter().flatten().take(MAX_PATTERN as usize).map(|s| match *s { Seg::Line { a, .. } => (0.0, a[0], a[1]), Seg::Arc { c, .. } => (0.0, c[0], c[1]) }).collect()).unwrap_or_default() } else { vec![] };
        return if pts.is_empty() { vec![(0.0, 0.0, 0.0)] } else { pts };
    }
    if n.flags & nflags::PATTERN != 0 {
        let count = (v[6].floor().clamp(1.0, MAX_PATTERN as f64)) as usize;
        let total = v[7];
        let step = if (total - core::f64::consts::TAU).abs() < 1e-9 { total / count as f64 } else if count > 1 { total / (count - 1) as f64 } else { 0.0 };
        return (0..count).map(|i| (step * i as f64, 0.0, 0.0)).collect();
    }
    if n.flags & nflags::PATTERN_RECT != 0 {
        let c1 = (v[6].floor().clamp(1.0, MAX_PATTERN as f64)) as usize;
        let c2 = (v[8].floor().clamp(1.0, (MAX_PATTERN as usize / c1) as f64)) as usize;
        return (0..c2).flat_map(|j| (0..c1).map(move |i| (0.0, i as f64 * v[7], j as f64 * v[9]))).collect();
    }
    vec![(0.0, 0.0, 0.0)]
}

/// Values of instance (θ, t): profile values shifted by t (rotation applied in `profile_of`).
fn shifted(kind: u32, v: &[f64; NVALS], tx: f64, ty: f64) -> [f64; NVALS] {
    let mut w = *v;
    w[0] += tx;
    w[1] += ty;
    if kind == profile::RECT { w[2] += tx; w[3] += ty; }
    w
}

/// Corner size of a rectangle feature as the device clamps it (0 = none).
fn corner_size(n: &Node, v: &[f64; NVALS]) -> f64 {
    if n.kind != profile::RECT || n.flags & (nflags::FILLET | nflags::CHAMFER) == 0 { return 0.0; }
    let (hx, hy) = (0.5 * (v[0] - v[2]).abs(), 0.5 * (v[1] - v[3]).abs());
    v[10].max(0.0).min(hx.min(hy))
}

/// 2D corner points of every instance of a rectangle feature (for std fillet/chamfer).
fn instance_corners(n: &Node, v: &[f64; NVALS], d: &Design) -> Vec<[f64; 2]> {
    let (x0, y0, x1, y1) = (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]));
    instances(n, v, d).into_iter().flat_map(|(a, tx, ty)| [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|p| { let r = rot(p, a); [r[0] + tx, r[1] + ty] })).collect()
}

/// Exact solid of a device design (same feature semantics as the device kernels).
pub fn to_layered(d: &Design) -> Result<Solid, String> {
    let k = &NativeKernel;
    let mut body: Option<Solid> = None;
    let mut stack: Vec<Option<Solid>> = Vec::new();
    // the device shell is max(f, −f − t): the geometric shell exactly when the body's field is
    // an exact interior distance — one convex unpatterned pad followed only by cuts
    let mut exact_interior = false;
    let mut mirror = 0u32;
    // a mirror reflects the body built so far (document order), when its stage ends
    let flush = |body: &mut Option<Solid>, mirror: &mut u32| -> Result<(), String> {
        for ax in 0..2 {
            if *mirror >> ax & 1 == 0 { continue; }
            *body = match body.take() {
                Some(Solid::Layered(l)) if l.frame == Frame::WORLD => { let m = l.reflected(ax); Some(Solid::Layered(l.boolean(&m, BoolOp2::Union))) }
                Some(_) => return Err("mirror of a non-prismatic body needs the general kernel".into()),
                None => None,
            };
        }
        *mirror = 0;
        Ok(())
    };
    for (i, n) in d.nodes.iter().enumerate() {
        if n.kind != profile::MIRROR && mirror != 0 { flush(&mut body, &mut mirror)?; }
        if n.flags & nflags::ACTIVE == 0 { continue; }
        if n.flags & (nflags::DRAFT | nflags::ROUND_Z | nflags::CHAMFER_Z) != 0 {
            return Err(format!("feature {i}: draft / rounded or chamfered caps are device-only (no exact layered form); certify it and deliver the certified implicit body"));
        }
        match n.kind {
            profile::MIRROR => {
                if value(n, &d.params, 0) > 0.5 { mirror |= 1; }
                if value(n, &d.params, 1) > 0.5 { mirror |= 2; }
                continue;
            }
            profile::REVOLVE => {
                let mut v = [0.0; NVALS];
                for (j, x) in v.iter_mut().enumerate() { *x = value(n, &d.params, j); }
                let meridian = sketch_profile(&d.profiles[pool_index(d, n, i)?], 0.0, 0.0, 0.0)?;
                let mut tool: Option<Solid> = None;
                for (a, tx, ty) in instances(n, &v, d) {
                    let axis = rot([v[0] + tx, v[1] + ty], a);
                    let l = LayeredSolid::revolve_rectilinear(axis, &meridian).map_err(|e| format!("feature {i}: part.shaft is device-only here ({e}); certify it and deliver the certified implicit body"))?;
                    let inst = Solid::Layered(l);
                    tool = Some(match tool { None => inst, Some(t) => k.boolean(&t, &inst, BoolOp2::Union).map_err(|e| format!("feature {i}: {}", e.message))? });
                }
                let Some(tool) = tool else { continue };
                if n.op == fop::ADD { exact_interior = false; }
                body = match (body.take(), n.op) {
                    (None, fop::ADD) => Some(tool),
                    (None, _) => None,
                    (Some(b), op) => Some(k.boolean(&b, &tool, if op == fop::ADD { BoolOp2::Union } else { BoolOp2::Difference }).map_err(|e| format!("feature {i}: {}", e.message))?),
                };
                continue;
            }
            profile::SHELL => {
                let t = value(n, &d.params, 0);
                let why = |m: String| format!("feature {i}: part.shell is device-only here ({m}); certify it and deliver the certified implicit body");
                if !exact_interior { return Err(why("the body is not a single convex pad minus cuts, so the device field is not an exact interior distance".into())); }
                let Some(Solid::Layered(l)) = &body else { return Err(why("no prismatic body".into())) };
                body = Some(Solid::Layered(l.shell_closed(t).map_err(why)?));
                exact_interior = false;
                continue;
            }
            profile::BODY => { exact_interior = false; if stack.len() < MAX_BODIES { stack.push(body.take()); } continue; }
            profile::COMBINE => {
                exact_interior = false;
                if let Some(target) = stack.pop() {
                    let tool = body.take();
                    body = match (target, tool) {
                        (Some(t), Some(b)) => Some(k.boolean(&t, &b, match n.op { fop::ADD => BoolOp2::Union, fop::REMOVE => BoolOp2::Difference, _ => BoolOp2::Intersection }).map_err(|e| format!("feature {i}: {}", e.message))?),
                        (Some(t), None) => if n.op == fop::INTERSECT { None } else { Some(t) },
                        (None, b) => if n.op == fop::ADD { b } else { None },
                    };
                }
                continue;
            }
            _ => {}
        }
        let mut v = [0.0; NVALS];
        for (j, x) in v.iter_mut().enumerate() { *x = value(n, &d.params, j); }
        if !(v[5] > v[4]) { continue; } // empty tool (as on the device)
        let corner = corner_size(n, &v);
        let mut tool: Option<Solid> = None;
        for (a, tx, ty) in instances(n, &v, d) {
            let w = shifted(n.kind, &v, tx, ty);
            let p = if corner > 0.0 {
                let (x0, y0, x1, y1) = (w[0].min(w[2]), w[1].min(w[3]), w[0].max(w[2]), w[1].max(w[3]));
                treated_rect(x0, y0, x1, y1, corner, n.flags & nflags::FILLET != 0, a)
            } else if n.kind == profile::SKETCH {
                let mut ps = sketch_profiles(&d.profiles[pool_index(d, n, i)?], w[0], w[1], a).map_err(|e| format!("feature {i}: {e}"))?;
                let first = ps.remove(0);
                for extra in ps {
                    let inst = Solid::Layered(LayeredSolid::extrusion(Frame::WORLD, &extra, v[4], v[5]));
                    tool = Some(match tool { None => inst, Some(t) => k.boolean(&t, &inst, BoolOp2::Union).map_err(|e| format!("feature {i}: {}", e.message))? });
                }
                first
            } else {
                let Ok(p) = profile_of(n.kind, &w, a) else { continue };
                p
            };
            let inst = Solid::Layered(LayeredSolid::extrusion(Frame::WORLD, &p, v[4], v[5]));
            tool = Some(match tool { None => inst, Some(t) => k.boolean(&t, &inst, BoolOp2::Union).map_err(|e| format!("feature {i}: {}", e.message))? });
        }
        let Some(tool) = tool else { continue };
        if n.op == fop::ADD { exact_interior = body.is_none() && n.flags & (nflags::PATTERN | nflags::PATTERN_RECT) == 0; }
        else if n.op != fop::REMOVE { exact_interior = false; }
        body = match (body.take(), n.op) {
            (None, fop::ADD) => Some(tool),
            (None, _) => None, // removing from nothing (as on the device: no material)
            (Some(b), op) => Some(k.boolean(&b, &tool, if op == fop::ADD { BoolOp2::Union } else { BoolOp2::Difference }).map_err(|e| format!("feature {i}: {}", e.message))?),
        };
    }
    if mirror != 0 { flush(&mut body, &mut mirror)?; }
    body.ok_or_else(|| "design has no material".into())
}

fn length(d: &Design, n: &Node, j: usize) -> J {
    match n.pref[j] {
        NO_PARAM => J::String(format!("{} mm", n.val[j])),
        p if n.val[j] == 0.0 => J::String(format!("${}", d.names[p as usize])),
        p if n.val[j] < 0.0 => J::String(format!("={} - {} mm", d.names[p as usize], -n.val[j])),
        p => J::String(format!("={} + {} mm", d.names[p as usize], n.val[j])),
    }
}
fn plain(d: &Design, n: &Node, j: usize, unit: &str) -> J {
    match n.pref[j] {
        NO_PARAM => if unit.is_empty() { json!(n.val[j]) } else { J::String(format!("{} {unit}", n.val[j])) },
        p if n.val[j] == 0.0 => J::String(format!("${}", d.names[p as usize])),
        p if n.val[j] < 0.0 => J::String(format!("={} - {}", d.names[p as usize], -n.val[j])),
        p => J::String(format!("={} + {}", d.names[p as usize], n.val[j])),
    }
}

/// AgIR program (JSON) rebuilding a device design as Part Design features.
pub fn to_program(d: &Design, requirements_rev: u32, expected_revision: u32) -> J {
    let mut st: Vec<J> = vec![json!({"id": "dev_edit", "op": "document.edit", "args": {"branch": "main"}})];
    let unit_of = |name: &str| if name.starts_with('n') || name.contains("count") { "" } else { "mm" };
    for (name, v) in d.names.iter().zip(&d.params) {
        let val = if unit_of(name).is_empty() { json!(v) } else { J::String(format!("{v} mm")) };
        st.push(json!({"id": format!("dev_p_{name}"), "op": "knowledge.parameter", "args": {"name": name, "value": val}}));
    }
    // bodies: features of body k carry "$body": k (container); boolean ops combine the last
    // feature of the pushed body (target) with the current body's last feature (tool)
    let mut cur_body = 0u32;
    let mut next_body = 1u32;
    let mut body_stack: Vec<u32> = Vec::new();
    let mut last_of: std::collections::BTreeMap<u32, String> = std::collections::BTreeMap::new();
    let tag = |mut v: J, b: u32| -> J { if b != 0 { v["args"]["$body"] = json!(b); } v };
    let mut mirror = 0u32;
    let mm = |x: f64| J::String(format!("{x} mm"));
    let chain = |st: &mut Vec<J>, sk: &str, loops: &[Vec<Seg>], dx: f64, dy: f64, pfx: &str| -> String {
        let mut prev = sk.to_string();
        let mut k = 0usize;
        for lp in loops {
            for sg in lp {
                let id = format!("{pfx}_{k}");
                k += 1;
                let stmt = match *sg {
                    Seg::Line { a, b } => json!({"id": id, "op": "sketch.line", "args": {"sketch": format!("%{prev}"), "a": [mm(a[0] + dx), mm(a[1] + dy)], "b": [mm(b[0] + dx), mm(b[1] + dy)]}}),
                    Seg::Arc { c, r, a0, a1 } if (a1 - a0).abs() >= core::f64::consts::TAU - 1e-12 => json!({"id": id, "op": "sketch.circle", "args": {"sketch": format!("%{prev}"), "mode": "center_radius", "pts": [[mm(c[0] + dx), mm(c[1] + dy)]], "r": mm(r)}}),
                    Seg::Arc { c, r, a0, a1 } => {
                        let (t0, t1) = if a1 > a0 { (a0, a1) } else { (a1, a0) }; // std arcs run counter-clockwise
                        json!({"id": id, "op": "sketch.arc", "args": {"sketch": format!("%{prev}"), "mode": "center_start_end", "pts": [[mm(c[0] + dx), mm(c[1] + dy)], [mm(c[0] + dx + r * t0.cos()), mm(c[1] + dy + r * t0.sin())], [mm(c[0] + dx + r * t1.cos()), mm(c[1] + dy + r * t1.sin())]]}})
                    }
                };
                st.push(stmt);
                prev = id;
            }
        }
        prev
    };
    // the whole body so far: mirror its latest feature's product, where the stage ends
    let emit_mirror = |st: &mut Vec<J>, last_of: &mut std::collections::BTreeMap<u32, String>, mirror: &mut u32, at: usize| {
        for (ax, plane) in [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])] {
            if *mirror >> ax & 1 == 0 { continue; }
            let Some(last) = last_of.get(&0).cloned() else { continue };
            let id = format!("dev_mir{at}_{ax}");
            st.push(json!({"id": id, "op": "transform.mirror", "args": {"items": [format!("%{last}")], "plane": plane}}));
            last_of.insert(0, id);
        }
        *mirror = 0;
    };
    for (i, n) in d.nodes.iter().enumerate() {
        if n.kind != profile::MIRROR && mirror != 0 { emit_mirror(&mut st, &mut last_of, &mut mirror, i); }
        if n.flags & nflags::ACTIVE == 0 { continue; }
        match n.kind {
            profile::MIRROR => {
                if value(n, &d.params, 0) > 0.5 { mirror |= 1; }
                if value(n, &d.params, 1) > 0.5 { mirror |= 2; }
                continue;
            }
            profile::REVOLVE => {
                let Some(loops) = d.profiles.get(value(n, &d.params, 15).floor().max(0.0) as usize) else { continue };
                let (sk, rg, ft) = (format!("dev_sk{i}"), format!("dev_rg{i}"), format!("dev_f{i}"));
                let first = st.len();
                let (cx, cy) = (value(n, &d.params, 0), value(n, &d.params, 1));
                if cx != 0.0 || cy != 0.0 {
                    // axis off the origin: sketch on a yz-parallel datum plane through it
                    let dp = format!("dev_dp{i}");
                    st.push(json!({"id": dp, "op": "datum.plane", "args": {"mode": "through", "refs": ["yz"], "value": [mm(cx), mm(cy), mm(0.0)]}}));
                    st.push(json!({"id": sk, "op": "sketch.create", "args": {"support": format!("%{dp}")}}));
                } else {
                    st.push(json!({"id": sk, "op": "sketch.create", "args": {"support": "yz"}}));
                }
                let last = chain(&mut st, &sk, loops, 0.0, 0.0, &format!("dev_rv{i}"));
                st.push(json!({"id": rg, "op": "sketch.regions", "args": {"sketch": format!("%{last}")}}));
                st.push(json!({"id": ft, "op": if n.op == fop::ADD { "part.shaft" } else { "part.groove" }, "args": {"profile": format!("%{rg}"), "axis": "z", "angles": ["0 deg", "360 deg"]}}));
                if cur_body != 0 { for x in st[first..].iter_mut() { x["args"]["$body"] = json!(cur_body); } }
                last_of.insert(cur_body, ft);
                continue;
            }
            profile::SHELL => {
                let id = format!("dev_sh{i}");
                st.push(tag(json!({"id": id, "op": "part.shell", "args": {"remove": [], "inner": length(d, n, 0)}}), cur_body));
                last_of.insert(cur_body, id);
                continue;
            }
            profile::BODY => { body_stack.push(cur_body); cur_body = next_body; next_body += 1; continue; }
            profile::COMBINE => {
                let Some(target) = body_stack.pop() else { continue };
                let op = match n.op { fop::ADD => "boolean.add", fop::REMOVE => "boolean.remove", _ => "boolean.intersect" };
                let id = format!("dev_bool{i}");
                if let (Some(t), Some(b)) = (last_of.get(&target).cloned(), last_of.get(&cur_body).cloned()) {
                    st.push(tag(json!({"id": id, "op": op, "args": {"target": format!("%{t}"), "tools": [format!("%{b}")]}}), target));
                    last_of.insert(target, id);
                }
                cur_body = target;
                continue;
            }
            _ => {}
        }
        let (sk, pr, rg, ft) = (format!("dev_sk{i}"), format!("dev_pr{i}"), format!("dev_rg{i}"), format!("dev_f{i}"));
        let first = st.len();
        st.push(json!({"id": sk, "op": "sketch.create", "args": {"support": "xy"}}));
        let prof = match n.kind {
            profile::SKETCH => {
                let Some(loops) = d.profiles.get(value(n, &d.params, 15).floor().max(0.0) as usize) else { continue };
                let last = chain(&mut st, &sk, loops, value(n, &d.params, 0), value(n, &d.params, 1), &format!("dev_sg{i}"));
                json!({"id": pr, "op": "sketch.solve", "args": {"sketch": format!("%{last}")}})
            }
            profile::RECT => json!({"id": pr, "op": "sketch.rectangle", "args": {"sketch": format!("%{sk}"), "mode": "corners", "pts": [[length(d, n, 0), length(d, n, 1)], [length(d, n, 2), length(d, n, 3)]]}}),
            profile::CIRCLE => json!({"id": pr, "op": "sketch.circle", "args": {"sketch": format!("%{sk}"), "mode": "center_radius", "pts": [[length(d, n, 0), length(d, n, 1)]], "r": length(d, n, 2)}}),
            _ => json!({"id": pr, "op": "sketch.polygon", "args": {"sketch": format!("%{sk}"), "center": [length(d, n, 0), length(d, n, 1)], "sides": plain(d, n, 3, ""), "r": length(d, n, 2)}}),
        };
        st.push(prof);
        st.push(json!({"id": rg, "op": "sketch.regions", "args": {"sketch": format!("%{pr}")}}));
        // two_sided limits (−d2, d) = (z0, z1)
        let neg_z0 = match n.pref[4] { NO_PARAM => J::String(format!("{} mm", -n.val[4])), p => J::String(format!("=-({}) - ({} mm)", d.names[p as usize], n.val[4])) };
        st.push(json!({"id": ft, "op": if n.op == fop::ADD { "part.pad" } else { "part.pocket" }, "args": {"profile": format!("%{rg}"), "limits": {"type": "two_sided", "d": length(d, n, 5), "d2": neg_z0}}}));
        if n.flags & nflags::PATTERN != 0 {
            st.push(json!({"id": format!("dev_pat{i}"), "op": "transform.circular_pattern", "args": {"items": [format!("%{ft}")], "axis": "z", "count": plain(d, n, 6, ""), "angle": plain(d, n, 7, "rad")}}));
        }
        if n.flags & nflags::PATTERN_USER != 0 {
            let mut v = [0.0; NVALS];
            for (j, x) in v.iter_mut().enumerate() { *x = value(n, &d.params, j); }
            let pos: Vec<J> = instances(n, &v, d).iter().map(|(_, tx, ty)| json!([format!("{tx} mm"), format!("{ty} mm")])).collect();
            st.push(json!({"id": format!("dev_upat{i}"), "op": "transform.user_pattern", "args": {"items": [format!("%{ft}")], "positions": pos}}));
        }
        if n.flags & nflags::PATTERN_RECT != 0 {
            st.push(json!({"id": format!("dev_rpat{i}"), "op": "transform.rectangular_pattern", "args": {"items": [format!("%{ft}")], "dirs": [[1.0, 0.0], [0.0, 1.0]], "counts": [plain(d, n, 6, ""), plain(d, n, 8, "")], "spacing": [length(d, n, 7), length(d, n, 9)]}}));
        }
        if n.kind == profile::RECT && n.flags & (nflags::FILLET | nflags::CHAMFER) != 0 {
            let mut v = [0.0; NVALS];
            for (j, x) in v.iter_mut().enumerate() { *x = value(n, &d.params, j); }
            let pts: Vec<J> = instance_corners(n, &v, d).iter().map(|p| json!([format!("{} mm", p[0]), format!("{} mm", p[1])])).collect();
            // the size as the device clamps it (≤ half the shorter side); parametric when unclamped
            let k = corner_size(n, &v);
            let size = if (k - v[10]).abs() < 1e-12 { length(d, n, 10) } else { J::String(format!("{k} mm")) };
            if n.flags & nflags::FILLET != 0 {
                st.push(json!({"id": format!("dev_fil{i}"), "op": "part.edge_fillet", "args": {"edges": pts, "r": size, "propagation": "minimal"}}));
            } else {
                st.push(json!({"id": format!("dev_cha{i}"), "op": "part.chamfer", "args": {"edges": pts, "mode": "distance", "a": size}}));
            }
        }
        // container of this feature's statements, and the body's latest feature
        let last = st.last().and_then(|x| x["id"].as_str()).unwrap_or_default().to_string();
        if cur_body != 0 { for x in st[first..].iter_mut() { x["args"]["$body"] = json!(cur_body); } }
        last_of.insert(cur_body, last);
    }
    if mirror != 0 { emit_mirror(&mut st, &mut last_of, &mut mirror, d.nodes.len()); }
    json!({"agir": "0.2", "requirements_rev": requirements_rev, "expected_revision": expected_revision, "program": st})
}

/// A constant subexpression (number or negated number).
fn literal(e: &agcad_ir::formula::Expr) -> Option<f64> {
    use agcad_ir::formula::Expr;
    match e { Expr::Num(x) => Some(*x), Expr::Neg(a) => literal(a).map(|x| -x), _ => None }
}

/// Value → (parameter index or NO_PARAM, literal) given known parameter names.
fn raise_value(v: &Value, names: &[String], params: &BTreeMap<String, f64>, warnings: &mut Vec<String>) -> Result<(u32, f64), String> {
    let pidx = |n: &str| names.iter().position(|x| x == n).map(|i| i as u32);
    Ok(match v {
        Value::Quantity(q) => (NO_PARAM, q.value),
        Value::Float(f) => (NO_PARAM, *f),
        Value::Int(i) => (NO_PARAM, *i as f64),
        Value::Ref { id, field: Some(n) } if id == agcad_ir::program::PARAM_REF => (pidx(n).ok_or(format!("unknown parameter {n}"))?, 0.0),
        Value::String(n) if pidx(n).is_some() => (pidx(n).unwrap(), 0.0),
        Value::Formula(src) => {
            use agcad_ir::formula::Expr;
            let e = agcad_ir::formula::parse(src)?;
            match &e {
                Expr::Param(n) if pidx(n).is_some() => (pidx(n).unwrap(), 0.0),
                Expr::Bin(op @ ('+' | '-'), a, b) if matches!(&**a, Expr::Param(n) if pidx(n).is_some()) && literal(b).is_some() => {
                    let Expr::Param(n) = &**a else { unreachable!() };
                    let x = literal(b).unwrap();
                    (pidx(n).unwrap(), if *op == '-' { -x } else { x })
                }
                _ => {
                    warnings.push(format!("formula '={src}' is not param ± constant; frozen at its current value on the device"));
                    (NO_PARAM, e.eval(&|n| params.get(n).copied())?)
                }
            }
        }
        other => return Err(format!("unsupported value {other:?}")),
    })
}

/// Raise a document's feature tree into a device design. Returns the design and warnings.
pub fn raise(part: &PartState) -> Result<(Design, Vec<String>), String> {
    let mut d = Design::default();
    let mut warnings = Vec::new();
    let mut pvals: BTreeMap<String, f64> = BTreeMap::new();
    let mut profiles: BTreeMap<u32, (u32, [(u32, f64); 4])> = BTreeMap::new(); // sketch-element feature → (kind, values)
    let mut regions: BTreeMap<u32, u32> = BTreeMap::new(); // regions feature → profile feature
    let mut feat_node: BTreeMap<u32, usize> = BTreeMap::new();
    // sketch statement → (support plane, accumulated segments): general sketches
    let mut chains: BTreeMap<u32, (String, Vec<Seg>)> = BTreeMap::new();
    let mut datums: BTreeMap<u32, String> = BTreeMap::new();
    let mut mirrored = false;
    // the device has one mirror stage: features may follow it, a later mirror may not
    let mut mirror_closed = false;
    let mut mirror_at = usize::MAX;
    let before_mirror = |ni: usize, mirror_at: usize, what: &str| -> Result<(), String> {
        if mirror_at != usize::MAX && ni < mirror_at { return Err(format!("{what}: modifying a feature that precedes the mirror is not device-representable")); }
        Ok(())
    };
    let refid = |v: Option<&Value>| -> Option<u32> { match v { Some(Value::Ref { id, .. }) => id.trim_start_matches('f').parse().ok(), _ => None } };
    let list = |v: Option<&Value>| -> Vec<Value> { match v { Some(Value::List(l)) => l.clone(), _ => vec![] } };
    let mut cur_body = 0usize;
    let mut body_stack: Vec<usize> = Vec::new();
    let mut seen_bodies: std::collections::BTreeSet<usize> = [0].into_iter().collect();
    let blank = |kind: u32, op: u32, src: u32| Node { kind, op, flags: nflags::ACTIVE, src, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    for id in part.rebuild_order() {
        let f = &part.features[&id];
        if !f.active { continue; }
        let a = &f.args;
        let solid_feature = matches!(f.op.as_str(), "part.pad" | "part.pocket" | "part.shell" | "part.shaft" | "part.groove" | "part.hole");
        if mirrored && solid_feature { mirror_closed = true; }
        if solid_feature && f.container != cur_body {
            if seen_bodies.insert(f.container) && body_stack.len() < MAX_BODIES {
                body_stack.push(cur_body);
                d.nodes.push(blank(profile::BODY, fop::ADD, id));
                cur_body = f.container;
            } else {
                return Err(format!("{}: body containers must be used in nested order on the device", f.feature_ref()));
            }
        }
        match f.op.as_str() {
            "boolean.add" | "boolean.remove" | "boolean.intersect" => {
                let Some(target) = body_stack.pop() else { return Err(format!("{}: boolean without a second body", f.feature_ref())) };
                let op = match f.op.as_str() { "boolean.add" => fop::ADD, "boolean.remove" => fop::REMOVE, _ => fop::INTERSECT };
                d.nodes.push(blank(profile::COMBINE, op, id));
                cur_body = target;
            }
            "part.shell" => {
                let t = raise_value(a.get("inner").ok_or("shell inner thickness")?, &d.names, &pvals, &mut warnings)?;
                let mut n = blank(profile::SHELL, fop::ADD, id);
                (n.pref[0], n.val[0]) = t;
                d.nodes.push(n);
            }
            "knowledge.parameter" => {
                let Some(Value::String(name)) = a.get("name") else { return Err("parameter without name".into()) };
                let v = match a.get("value") { Some(Value::Quantity(q)) => q.value, Some(Value::Float(x)) => *x, Some(Value::Int(i)) => *i as f64, other => return Err(format!("parameter {name}: unsupported value {other:?}")) };
                if d.names.len() >= MAX_PARAMS { return Err(format!("more than {MAX_PARAMS} parameters")); }
                d.names.push(name.clone());
                d.params.push(v);
                pvals.insert(name.clone(), v);
            }
            "sketch.create" => {
                let sup = match a.get("support") {
                    Some(Value::String(x)) => x.clone(),
                    Some(r @ Value::Ref { .. }) => refid(Some(r)).and_then(|d| datums.get(&d).cloned()).ok_or(format!("{}: sketch support must be a base plane or a datum.plane through a point", f.feature_ref()))?,
                    _ => "xy".into(),
                };
                chains.insert(id, (sup, vec![]));
            }
            "datum.plane" => {
                // mode "through": a base plane moved to a literal point → "yz@x,y,z"
                let mode = match a.get("mode") { Some(Value::String(m)) => m.as_str(), _ => "offset" };
                let base = match list(a.get("refs")).first() { Some(Value::String(b)) => b.clone(), _ => "xy".into() };
                if mode == "through" {
                    let pt = list(a.get("value"));
                    let mut o = [0.0; 3];
                    for (k, v) in pt.iter().take(3).enumerate() {
                        let (p, x) = raise_value(v, &d.names, &pvals, &mut warnings)?;
                        if p != NO_PARAM { return Err(format!("{}: parametric datum points are not device-representable", f.feature_ref())); }
                        o[k] = x;
                    }
                    datums.insert(id, format!("{base}@{},{},{}", o[0], o[1], o[2]));
                } else {
                    warnings.push(format!("{}: datum.plane mode {mode} has no device equivalent; sketches on it are not raised", f.feature_ref()));
                }
            }
            "document.edit" | "document.checkpoint" | "sketch.solve" => {
                if f.op == "sketch.solve" { if let Some(p) = refid(a.get("sketch")) { if let Some(c) = chains.get(&p).cloned() { chains.insert(id, c); } if let Some(pr) = profiles.get(&p).copied() { profiles.insert(id, pr); } } }
            }
            "sketch.line" | "sketch.arc" | "sketch.profile" | "sketch.elongated_hole" => {
                let parent = refid(a.get("sketch")).ok_or(format!("{}: no sketch", f.feature_ref()))?;
                let (sup, mut segs) = chains.get(&parent).cloned().unwrap_or(("xy".into(), vec![]));
                let lit = |v: &Value, w: &mut Vec<String>| -> Result<f64, String> {
                    let (p, x) = raise_value(v, &d.names, &pvals, w)?;
                    if p != NO_PARAM { return Err(format!("{}: parametric coordinates in general sketches are not device-representable (use rectangle / circle / polygon sketches for parametric profiles)", f.feature_ref())); }
                    Ok(x)
                };
                let pt = |v: &Value, w: &mut Vec<String>| -> Result<[f64; 2], String> { let l = match v { Value::List(l) => l.clone(), _ => vec![] }; Ok([lit(l.first().ok_or("point")?, w)?, lit(l.get(1).ok_or("point")?, w)?]) };
                match f.op.as_str() {
                    "sketch.line" => {
                        if matches!(a.get("construction"), Some(Value::Bool(true))) { chains.insert(id, (sup, segs)); continue; }
                        segs.push(Seg::Line { a: pt(a.get("a").ok_or("line a")?, &mut warnings)?, b: pt(a.get("b").ok_or("line b")?, &mut warnings)? });
                    }
                    "sketch.arc" => {
                        let ps = list(a.get("pts"));
                        if ps.len() < 3 { return Err("arc needs center, start, end".into()); }
                        let (c, p0, p1) = (pt(&ps[0], &mut warnings)?, pt(&ps[1], &mut warnings)?, pt(&ps[2], &mut warnings)?);
                        let r = (p0[0] - c[0]).hypot(p0[1] - c[1]);
                        let t0 = (p0[1] - c[1]).atan2(p0[0] - c[0]);
                        let mut t1 = (p1[1] - c[1]).atan2(p1[0] - c[0]);
                        while t1 <= t0 { t1 += core::f64::consts::TAU; } // counter-clockwise, as std
                        segs.push(Seg::Arc { c, r, a0: t0, a1: t1 });
                    }
                    "sketch.profile" => {
                        let ps: Vec<[f64; 2]> = list(a.get("pts")).iter().map(|v| pt(v, &mut warnings)).collect::<Result<_, _>>()?;
                        for k in 0..ps.len() { segs.push(Seg::Line { a: ps[k], b: ps[(k + 1) % ps.len()] }); }
                    }
                    _ => {
                        // slot (stadium): two lines and two half circles
                        let ps = list(a.get("pts"));
                        let (p0, p1) = (pt(ps.first().ok_or("slot start")?, &mut warnings)?, pt(ps.get(1).ok_or("slot end")?, &mut warnings)?);
                        let w = lit(a.get("width").ok_or("slot width")?, &mut warnings)? / 2.0;
                        let ang = (p1[1] - p0[1]).atan2(p1[0] - p0[0]);
                        let (nx, ny) = (-ang.sin() * w, ang.cos() * w);
                        let hp = core::f64::consts::FRAC_PI_2;
                        segs.push(Seg::Line { a: [p0[0] - nx, p0[1] - ny], b: [p1[0] - nx, p1[1] - ny] });
                        segs.push(Seg::Arc { c: p1, r: w, a0: ang - hp, a1: ang + hp });
                        segs.push(Seg::Line { a: [p1[0] + nx, p1[1] + ny], b: [p0[0] + nx, p0[1] + ny] });
                        segs.push(Seg::Arc { c: p0, r: w, a0: ang + hp, a1: ang + 3.0 * hp });
                    }
                }
                chains.insert(id, (sup, segs));
            }
            "transform.mirror" => {
                // the whole body: no items, or the body's latest solid feature
                if let Some(it) = a.get("items") {
                    let ids: Vec<u32> = list(Some(it)).iter().filter_map(|v| refid(Some(v))).collect();
                    let latest = d.nodes.last().map(|n| n.src);
                    if ids.len() != 1 || Some(ids[0]) != latest { return Err(format!("{}: mirroring selected items is not device-representable (only the whole body)", f.feature_ref())); }
                }
                let pl: Vec<f64> = list(a.get("plane")).iter().filter_map(|v| v.as_f64()).collect();
                let mut m = blank(profile::MIRROR, fop::ADD, id);
                match pl.as_slice() {
                    [x, y, z] if x.abs() > 1.0 - 1e-9 && y.abs() < 1e-9 && z.abs() < 1e-9 => m.val[0] = 1.0,
                    [x, y, z] if y.abs() > 1.0 - 1e-9 && x.abs() < 1e-9 && z.abs() < 1e-9 => m.val[1] = 1.0,
                    _ => return Err(format!("{}: only mirror planes x = 0 / y = 0 (normal [1,0,0] / [0,1,0]) are device-representable", f.feature_ref())),
                }
                if mirror_closed { return Err(format!("{}: a second mirror after further features is not device-representable (one mirror stage)", f.feature_ref())); }
                if body_stack.is_empty() && seen_bodies.len() == 1 { if !mirrored { mirror_at = d.nodes.len(); } d.nodes.push(m); mirrored = true; } else { return Err(format!("{}: mirror with several bodies is not device-representable", f.feature_ref())); }
            }
            "part.shaft" | "part.groove" => {
                let rg = refid(a.get("profile")).and_then(|r| regions.get(&r)).ok_or(format!("{}: profile is not a sketch region", f.feature_ref()))?;
                let (sup, segs) = chains.get(rg).cloned().ok_or(format!("{}: profile sketch not found", f.feature_ref()))?;
                let (plane, origin) = match sup.split_once('@') {
                    Some((p, o)) => { let v: Vec<f64> = o.split(',').filter_map(|x| x.parse().ok()).collect(); (p.to_string(), [v[0], v[1], v[2]]) }
                    None => (sup.clone(), [0.0; 3]),
                };
                if !matches!(plane.as_str(), "yz" | "yz_plane" | "YZ") { return Err(format!("{}: only revolutions about a vertical axis (sketch on the yz plane or a yz-parallel datum) are device-representable", f.feature_ref())); }
                let mut loops = close_loops(&segs).map_err(|e| format!("{}: {e}", f.feature_ref()))?;
                if origin[2] != 0.0 {
                    // the meridian's z is relative to the datum origin
                    for lp in loops.iter_mut() { for sg in lp.iter_mut() { match sg { Seg::Line { a, b } => { a[1] += origin[2]; b[1] += origin[2]; } Seg::Arc { c, .. } => c[1] += origin[2] } } }
                }
                if d.profiles.len() >= 64 { return Err("too many sketch profiles".into()); }
                let mut n = blank(profile::REVOLVE, if f.op == "part.shaft" { fop::ADD } else { fop::REMOVE }, id);
                n.val[0] = origin[0];
                n.val[1] = origin[1];
                n.val[15] = d.profiles.len() as f64;
                d.profiles.push(loops);
                feat_node.insert(id, d.nodes.len());
                d.nodes.push(n);
            }
            "sketch.circle" | "sketch.rectangle" | "sketch.polygon" => {
                let rv = |v: &Value, w: &mut Vec<String>| raise_value(v, &d.names, &pvals, w);
                let pts = list(a.get("pts"));
                let xy = |p: &Value, w: &mut Vec<String>| -> Result<[(u32, f64); 2], String> { let l = match p { Value::List(l) => l.clone(), _ => vec![] }; Ok([rv(l.first().ok_or("point")?, w)?, rv(l.get(1).ok_or("point")?, w)?]) };
                let entry = match f.op.as_str() {
                    "sketch.circle" => { let c = xy(pts.first().ok_or("circle centre")?, &mut warnings)?; (profile::CIRCLE, [c[0], c[1], rv(a.get("r").ok_or("circle r")?, &mut warnings)?, (NO_PARAM, 0.0)]) }
                    "sketch.rectangle" => { let p0 = xy(pts.first().ok_or("corner")?, &mut warnings)?; let p1 = xy(pts.get(1).ok_or("corner")?, &mut warnings)?; (profile::RECT, [p0[0], p0[1], p1[0], p1[1]]) }
                    _ => { let c = xy(a.get("center").ok_or("polygon centre")?, &mut warnings)?; (profile::POLYGON, [c[0], c[1], rv(a.get("r").ok_or("polygon r")?, &mut warnings)?, rv(a.get("sides").ok_or("sides")?, &mut warnings)?]) }
                };
                // also as segments, for sketches mixing primitives with lines / arcs
                let parent = refid(a.get("sketch"));
                let (sup, mut segs) = parent.and_then(|p| chains.get(&p).cloned()).unwrap_or(("xy".into(), vec![]));
                let fresh = segs.is_empty();
                let (kind, pv) = entry;
                let lits: Vec<f64> = pv.iter().map(|(_, v)| *v).collect();
                let parametric = pv.iter().any(|(p, _)| *p != NO_PARAM);
                if !parametric {
                    match kind {
                        profile::CIRCLE => segs.push(Seg::Arc { c: [lits[0], lits[1]], r: lits[2], a0: 0.0, a1: core::f64::consts::TAU }),
                        profile::RECT => { let q = [[lits[0], lits[1]], [lits[2], lits[1]], [lits[2], lits[3]], [lits[0], lits[3]]]; for k in 0..4 { segs.push(Seg::Line { a: q[k], b: q[(k + 1) % 4] }); } }
                        _ => { let kk = lits[3].floor() as usize; for k in 0..kk { let (t0, t1) = (core::f64::consts::TAU * k as f64 / kk as f64, core::f64::consts::TAU * (k + 1) as f64 / kk as f64); segs.push(Seg::Line { a: [lits[0] + lits[2] * t0.cos(), lits[1] + lits[2] * t0.sin()], b: [lits[0] + lits[2] * t1.cos(), lits[1] + lits[2] * t1.sin()] }); } }
                    }
                }
                chains.insert(id, (sup, segs));
                if fresh { profiles.insert(id, entry); }
            }
            "sketch.regions" => { if let Some(p) = refid(a.get("sketch")) { regions.insert(id, p); } }
            "part.pad" | "part.pocket" => {
                let rg = refid(a.get("profile")).and_then(|r| regions.get(&r).copied()).ok_or(format!("{}: profile is not a sketch region", f.feature_ref()))?;
                if let Some((sup, _)) = chains.get(&rg) { if !matches!(sup.as_str(), "xy" | "xy_plane" | "XY") { return Err(format!("{}: only sketches on the xy plane are device-representable for pads / pockets", f.feature_ref())); } }
                let general;
                let prof = match profiles.get(&rg) {
                    Some(p) => p,
                    None => {
                        let segs = chains.get(&rg).map(|c| c.1.clone()).unwrap_or_default();
                        let loops = close_loops(&segs).map_err(|e| format!("{}: {e}", f.feature_ref()))?;
                        if loops.is_empty() { return Err(format!("{}: empty sketch", f.feature_ref())); }
                        if d.profiles.len() >= 64 { return Err("too many sketch profiles".into()); }
                        let mut pv = [(NO_PARAM, 0.0); 4];
                        pv[3] = (NO_PARAM, d.profiles.len() as f64); // carried to v15 below
                        d.profiles.push(loops);
                        general = (profile::SKETCH, pv);
                        &general
                    }
                };
                let (z0, z1) = match a.get("limits") {
                    Some(Value::Map(m)) => {
                        let ty = match m.get("type") { Some(Value::String(s)) => s.as_str(), _ => "dimension" };
                        let dv = raise_value(m.get("d").ok_or("limits d")?, &d.names, &pvals, &mut warnings)?;
                        match ty {
                            "two_sided" => { let d2 = raise_value(m.get("d2").unwrap_or(&Value::Float(0.0)), &d.names, &pvals, &mut warnings)?; if d2.0 != NO_PARAM { return Err("parametric d2 not supported on the device".into()); } ((NO_PARAM, -d2.1), dv) }
                            "mirrored" => { if dv.0 != NO_PARAM { return Err("parametric mirrored limits not supported".into()); } ((NO_PARAM, -dv.1 / 2.0), (NO_PARAM, dv.1 / 2.0)) }
                            _ => ((NO_PARAM, 0.0), dv),
                        }
                    }
                    Some(v) => ((NO_PARAM, 0.0), raise_value(v, &d.names, &pvals, &mut warnings)?),
                    None => return Err("pad without limits".into()),
                };
                let (kind, pv) = *prof;
                let mut n = Node { kind, op: if f.op == "part.pad" { fop::ADD } else { fop::REMOVE }, flags: nflags::ACTIVE, src: id, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
                for (j, (p, v)) in pv.iter().enumerate() { n.pref[j] = *p; n.val[j] = *v; }
                if kind == profile::SKETCH { n.val[15] = pv[3].1; n.val[3] = 0.0; }
                (n.pref[4], n.val[4]) = z0;
                (n.pref[5], n.val[5]) = z1;
                feat_node.insert(id, d.nodes.len());
                d.nodes.push(n);
            }
            "transform.circular_pattern" => {
                for it in list(a.get("items")) {
                    let fid = refid(Some(&it)).ok_or("pattern item must reference a feature")?;
                    let ni = *feat_node.get(&fid).ok_or(format!("pattern item #{fid} is not a device feature"))?;
                    before_mirror(ni, mirror_at, &f.feature_ref())?;
                    let count = raise_value(a.get("count").ok_or("count")?, &d.names, &pvals, &mut warnings)?;
                    let angle = raise_value(a.get("angle").unwrap_or(&Value::Float(core::f64::consts::TAU)), &d.names, &pvals, &mut warnings)?;
                    let n = &mut d.nodes[ni];
                    n.flags |= nflags::PATTERN;
                    (n.pref[6], n.val[6]) = count;
                    (n.pref[7], n.val[7]) = angle;
                }
            }
            "part.hole" => {
                // std semantics: from the body top (zmax) down `depth` (or through), plus an
                // optional counterbore; one circular pocket per position
                let zmax = d.nodes.iter().filter(|n| n.op == fop::ADD && n.kind <= profile::SKETCH).map(|n| value(n, &d.params, 5)).fold(f64::NEG_INFINITY, f64::max);
                let zmin = d.nodes.iter().filter(|n| n.op == fop::ADD && n.kind <= profile::SKETCH).map(|n| value(n, &d.params, 4)).fold(f64::INFINITY, f64::min);
                if !zmax.is_finite() { return Err(format!("{}: hole needs an existing pad", f.feature_ref())); }
                let spec = match a.get("spec") { Some(Value::Map(m)) => m.clone(), _ => BTreeMap::new() };
                let num = |k: &str, w: &mut Vec<String>| -> Result<Option<f64>, String> { match spec.get(k) { Some(v) => { let (p, x) = raise_value(v, &d.names, &pvals, w)?; if p != NO_PARAM { return Err(format!("{}: parametric hole {k} is not device-representable", f.feature_ref())); } Ok(Some(x)) } None => Ok(None) } };
                let dia = match num("d", &mut warnings)? { Some(x) => x, None => return Err(format!("{}: hole spec needs a diameter 'd'", f.feature_ref())) };
                let through = spec.get("through").is_some() || matches!(spec.get("depth"), Some(Value::String(s)) if s == "through");
                let depth = if through { zmax - zmin + 2.0 } else { num("depth", &mut warnings)?.unwrap_or(zmax - zmin + 2.0) };
                let cb = num("counterbore_d", &mut warnings)?;
                let cbh = num("counterbore_depth", &mut warnings)?.unwrap_or(dia * 0.5);
                if spec.contains_key("countersink_d") || spec.contains_key("countersink_angle") { warnings.push(format!("{}: countersink is not modeled by the standard library; ignored on the device too", f.feature_ref())); }
                for p in list(a.get("positions")) {
                    let l = match &p { Value::List(l) => l.clone(), _ => return Err(format!("{}: hole positions must be [x, y]", f.feature_ref())) };
                    let x = raise_value(l.first().ok_or("hole x")?, &d.names, &pvals, &mut warnings)?;
                    let y = raise_value(l.get(1).ok_or("hole y")?, &d.names, &pvals, &mut warnings)?;
                    let mut mk = |r: f64, z0: f64| {
                        let mut n = blank(profile::CIRCLE, fop::REMOVE, id);
                        (n.pref[0], n.val[0]) = x;
                        (n.pref[1], n.val[1]) = y;
                        n.val[2] = r;
                        n.val[4] = z0;
                        n.val[5] = zmax + 1.0;
                        d.nodes.push(n);
                    };
                    if let Some(cbd) = cb { mk(cbd / 2.0, zmax - cbh); }
                    mk(dia / 2.0, zmax - depth);
                }
                feat_node.insert(id, d.nodes.len() - 1);
            }
            "transform.user_pattern" => {
                let pos = list(a.get("positions"));
                let mut pts: Vec<[f64; 2]> = Vec::new();
                for p in &pos {
                    let l = match p { Value::List(l) => l.clone(), _ => return Err(format!("{}: positions must be [x, y]", f.feature_ref())) };
                    let mut c = [0.0; 2];
                    for k in 0..2 {
                        let (pp, x) = raise_value(l.get(k).ok_or("position")?, &d.names, &pvals, &mut warnings)?;
                        if pp != NO_PARAM { return Err(format!("{}: parametric pattern positions are not device-representable", f.feature_ref())); }
                        c[k] = x;
                    }
                    pts.push(c);
                }
                if pts.is_empty() || pts.len() > MAX_PATTERN as usize { return Err(format!("{}: 1..={MAX_PATTERN} positions", f.feature_ref())); }
                let o = pts[0];
                let entry: Vec<Seg> = pts.iter().map(|p| { let t = [p[0] - o[0], p[1] - o[1]]; Seg::Line { a: t, b: t } }).collect();
                for it in list(a.get("items")) {
                    let fid = refid(Some(&it)).ok_or("pattern item must reference a feature")?;
                    let ni = *feat_node.get(&fid).ok_or(format!("pattern item #{fid} is not a device feature"))?;
                    before_mirror(ni, mirror_at, &f.feature_ref())?;
                    if d.profiles.len() >= 64 { return Err("too many sketch profiles".into()); }
                    let n = &mut d.nodes[ni];
                    n.flags = (n.flags & !(nflags::PATTERN | nflags::PATTERN_RECT)) | nflags::PATTERN_USER;
                    n.pref[6] = NO_PARAM;
                    n.val[6] = d.profiles.len() as f64;
                    d.profiles.push(vec![entry.clone()]);
                }
            }
            "transform.rectangular_pattern" => {
                if let Some(Value::List(dirs)) = a.get("dirs") {
                    let ok = dirs.len() <= 2 && dirs.iter().enumerate().all(|(k, dv)| matches!(dv, Value::List(c) if c.len() == 2 && c.iter().zip(if k == 0 { [1.0, 0.0] } else { [0.0, 1.0] }).all(|(x, e)| x.as_f64() == Some(e))));
                    if !ok { return Err(format!("{}: only x / y pattern directions are device-representable", f.feature_ref())); }
                }
                let counts = list(a.get("counts"));
                let spacing = list(a.get("spacing"));
                for it in list(a.get("items")) {
                    let fid = refid(Some(&it)).ok_or("pattern item must reference a feature")?;
                    let ni = *feat_node.get(&fid).ok_or(format!("pattern item #{fid} is not a device feature"))?;
                    before_mirror(ni, mirror_at, &f.feature_ref())?;
                    let c1 = raise_value(counts.first().ok_or("counts")?, &d.names, &pvals, &mut warnings)?;
                    let s1 = raise_value(spacing.first().ok_or("spacing")?, &d.names, &pvals, &mut warnings)?;
                    let c2 = match counts.get(1) { Some(v) => raise_value(v, &d.names, &pvals, &mut warnings)?, None => (NO_PARAM, 1.0) };
                    let s2 = match spacing.get(1) { Some(v) => raise_value(v, &d.names, &pvals, &mut warnings)?, None => (NO_PARAM, 0.0) };
                    let n = &mut d.nodes[ni];
                    n.flags = (n.flags & !nflags::PATTERN) | nflags::PATTERN_RECT;
                    (n.pref[6], n.val[6]) = c1;
                    (n.pref[7], n.val[7]) = s1;
                    (n.pref[8], n.val[8]) = c2;
                    (n.pref[9], n.val[9]) = s2;
                }
            }
            "part.edge_fillet" | "part.chamfer" => {
                // device form: corner points of the last rectangle feature's instances
                let size = raise_value(a.get(if f.op == "part.edge_fillet" { "r" } else { "a" }).ok_or("fillet size")?, &d.names, &pvals, &mut warnings)?;
                let pts: Vec<[f64; 2]> = list(a.get("edges")).iter().filter_map(|e| match e { Value::List(c) if c.len() == 2 => Some([c[0].as_f64()?, c[1].as_f64()?]), _ => None }).collect();
                let target = d.nodes.iter().rposition(|n| n.kind == profile::RECT);
                if let Some(t) = target { before_mirror(t, mirror_at, &f.feature_ref())?; }
                let matched = target.is_some_and(|ti| {
                    let n = &d.nodes[ti];
                    let mut v = [0.0; NVALS];
                    for (j, x) in v.iter_mut().enumerate() { *x = value(n, &d.params, j); }
                    let cs = instance_corners(n, &v, &d);
                    !pts.is_empty() && pts.len() == cs.len() && pts.iter().all(|p| cs.iter().any(|c| (c[0] - p[0]).abs() < 1e-6 && (c[1] - p[1]).abs() < 1e-6))
                });
                if !matched {
                    warnings.push(format!("{}: not the corners of one rectangle feature; skipped in the rollout", f.feature_ref()));
                } else {
                    let n = &mut d.nodes[target.unwrap()];
                    n.flags = (n.flags & !(nflags::FILLET | nflags::CHAMFER)) | if f.op == "part.edge_fillet" { nflags::FILLET } else { nflags::CHAMFER };
                    (n.pref[10], n.val[10]) = size;
                }
            }
            "part.multi_sections_solid" => return Err(format!("{}: solid loft has no optimization tape representation", f.feature_ref())),
            other if other.starts_with("evaluate.") || other.starts_with("file.") => {}
            other => warnings.push(format!("{}: {other} has no device equivalent yet; skipped in the rollout", f.feature_ref())),
        }
    }
    if d.nodes.is_empty() { return Err("no device-representable features (pad / pocket / shaft over xy / yz sketches)".into()); }
    Ok((d, warnings))
}
