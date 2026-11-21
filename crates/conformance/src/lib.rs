//! # agcad-conformance — conformance suites, CAD fuzzer, coverage (spec 09)
//!
//! Suites reference requirement IDs (CONVENTIONS § 3); untested IDs are reported as such.
//! The CAD fuzzer generates random *valid* layered programs (plates with pockets, holes and
//! patterns), checks validity, watertightness, exact-volume agreement and (when available)
//! OCCT parity, and shrinks failures by delta debugging into minimal reproducers.

use agcad_geo::builders::Profile;
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::Frame;
use agcad_geo::region2d::BoolOp2;
use agcad_geo::tessellate::{tessellate, TessOptions};
use agcad_solver::opt::Rng;
use serde_json::{json, Value};

pub struct Case {
    pub name: &'static str,
    pub requirements: &'static [&'static str],
    pub run: fn() -> Result<(), String>,
}

fn ensure(c: bool, m: impl Into<String>) -> Result<(), String> { if c { Ok(()) } else { Err(m.into()) } }

fn case_step_determinism() -> Result<(), String> {
    let b = agcad_geo::builders::prism(&Profile::rectangle(0.0, 0.0, 10.0, 5.0).with_hole(Profile::circle(5.0, 2.5, 1.0)), &Frame::WORLD, 2.0);
    let (a, _) = agcad_geo::step_write::write_step(&b, &Default::default());
    let (a2, _) = agcad_geo::step_write::write_step(&b, &Default::default());
    ensure(a == a2, "STEP export not byte-deterministic")?;
    let (bodies, _) = agcad_geo::step_read::read_step(&a, &Default::default())?;
    ensure(bodies.len() == 1 && bodies[0].face_by_name("cap_end").is_some(), "names did not round-trip")
}

fn case_status_codes() -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for s in agcad_driver::AgStatus::ALL { ensure(seen.insert(s.value()), format!("duplicate status {}", s.name()))?; }
    ensure(agcad_driver::AgStatus::Accepted.value() == 0, "ACCEPTED must be 0")
}

fn case_sandbox() -> Result<(), String> {
    let bad = "#[implicit] fn f(p: Vec3) -> Length { unsafe { p.x } }";
    ensure(agcad_rtc::compile_module(bad, agcad_ir::Catalog::global()).is_err(), "unsafe accepted")?;
    let lp = "#[implicit] fn f(p: Vec3) -> Length { loop {} }";
    ensure(agcad_rtc::compile_module(lp, agcad_ir::Catalog::global()).is_err(), "unbounded loop accepted")
}

fn case_vm_parity() -> Result<(), String> {
    let src = "#[implicit] fn s(p: Vec3, #[param(default = 5.0*mm)] r: Length) -> Length { length(p) - r + noise3(p) * 0.1 * mm }";
    let m = agcad_rtc::compile_module(src, agcad_ir::Catalog::global()).map_err(|d| format!("{d:?}"))?;
    let g = m.get("s").unwrap().expr.clone().unwrap();
    let back = agcad_vm::decode(&agcad_vm::encode(&g).map_err(|s| s.name().to_string())?).map_err(|s| s.name().to_string())?;
    for i in 0..50 {
        let p = [i as f64 * 0.37 - 9.0, (i as f64 * 1.3).sin() * 4.0, i as f64 * 0.11];
        ensure(g.eval(&p, &[5.0])[0].to_bits() == back.eval(&p, &[5.0])[0].to_bits(), "VM differs from reference")?;
    }
    Ok(())
}

fn case_layered_exactness() -> Result<(), String> {
    let f = Frame::WORLD;
    let plate = LayeredSolid::extrusion(f, &Profile::rectangle(0.0, 0.0, 100.0, 60.0), 0.0, 20.0);
    let pocket = LayeredSolid::extrusion(f, &Profile::rectangle(20.0, 10.0, 60.0, 50.0), 12.0, 25.0);
    let s = plate.boolean(&pocket, BoolOp2::Difference);
    let b = s.to_brep();
    ensure(b.validate().valid, format!("{:?}", b.validate().problems))?;
    let m = tessellate(&b, &TessOptions::default());
    ensure(m.is_watertight(), "not watertight")?;
    ensure((m.mass_properties().0 - s.volume()).abs() < 1e-6 * s.volume(), "volume mismatch")
}

fn case_requirements_immutable() -> Result<(), String> {
    let r = agcad_harness::Requirements::parse(r#"{"id":"r","rev":3,"deliver":{"body_type":"SolidBody"}}"#)?;
    ensure(r.check_rev(2) == Err(agcad_driver::AgStatus::StaleRequirements), "stale requirements accepted")
}

fn case_naming_roundtrip() -> Result<(), String> {
    for s in ["part.pad#3/cap_end", "part.pad#3/side[profile=sketch.line#7]", "flange_udf#4/hole.3"] {
        let n = agcad_runtime::naming::StableName::parse(s).ok_or("parse")?;
        ensure(n.format() == s, format!("{s} → {}", n.format()))?;
    }
    Ok(())
}

fn case_report_enums() -> Result<(), String> {
    use agcad_harness::checker::Verdict;
    use agcad_harness::requirements::Evidence;
    // spec 06: verdicts pass|fail|unknown; spec 14: evidence unknown < sampled < bounded < exact
    for (v, want) in [(Verdict::Pass, "pass"), (Verdict::Fail, "fail"), (Verdict::Unknown, "unknown")] {
        ensure(serde_json::to_value(v).map_err(|e| e.to_string())? == want, format!("verdict {v:?} must serialize as \"{want}\""))?;
    }
    for e in [Evidence::Unknown, Evidence::Sampled, Evidence::Bounded, Evidence::Exact] {
        ensure(serde_json::to_value(e).map_err(|x| x.to_string())? == e.name() && Evidence::parse(e.name()) == e, format!("evidence {e:?}"))?;
    }
    ensure(Evidence::Unknown < Evidence::Sampled && Evidence::Sampled < Evidence::Bounded && Evidence::Bounded < Evidence::Exact, "evidence order")?;
    // every evaluator's declared evidence is a valid level (catches typos such as "estimated")
    for (m, ev) in agcad_std::evaluate::MAX_EVIDENCE {
        ensure(Evidence::parse(ev).name() == *ev, format!("{m}: unknown evidence level '{ev}'"))?;
    }
    Ok(())
}

pub const CASES: &[Case] = &[
    Case { name: "step_determinism_and_names", requirements: &["AG-18-004", "AG-18-006", "AG-16-002"], run: case_step_determinism },
    Case { name: "status_codes_unique", requirements: &["AG-06-005", "AG-CV-003"], run: case_status_codes },
    Case { name: "aglang_sandbox", requirements: &["AG-04-002", "AG-17-001"], run: case_sandbox },
    Case { name: "vm_reference_parity", requirements: &["AG-06-006", "AG-08-001"], run: case_vm_parity },
    Case { name: "layered_exactness", requirements: &["AG-02-005", "AG-05-005"], run: case_layered_exactness },
    Case { name: "requirements_immutable", requirements: &["AG-06-001", "AG-17-004"], run: case_requirements_immutable },
    Case { name: "report_enums_match_spec", requirements: &["AG-06-004", "AG-CV-003"], run: case_report_enums },
    Case { name: "stable_name_roundtrip", requirements: &["AG-02-002", "AG-02-003"], run: case_naming_roundtrip },
];

/// One fuzz program: base plate + operations (kept small so shrinking is cheap).
#[derive(Clone, Debug)]
pub enum FuzzOp {
    Pocket { x0: f64, y0: f64, w: f64, h: f64, depth: f64 },
    Hole { x: f64, y: f64, r: f64, depth: f64 },
    Boss { x: f64, y: f64, r: f64, height: f64 },
    /// Round all line–line vertical edges (rejections with a max-radius are valid outcomes).
    Fillet { r: f64 },
    Chamfer { d: f64 },
}

#[derive(Clone, Debug)]
pub struct FuzzProgram {
    pub w: f64,
    pub h: f64,
    pub t: f64,
    pub ops: Vec<FuzzOp>,
}

pub fn generate(rng: &mut Rng) -> FuzzProgram {
    let (w, h, t) = (20.0 + 80.0 * rng.uniform(), 20.0 + 60.0 * rng.uniform(), 2.0 + 18.0 * rng.uniform());
    let n = 1 + (rng.uniform() * 5.0) as usize;
    let mut ops = Vec::new();
    for _ in 0..n {
        let k = rng.uniform();
        // degenerate placements on purpose: edges flush with the boundary, tangent circles
        let snap = |v: f64, lim: f64, rng: &mut Rng| if rng.uniform() < 0.2 { if rng.uniform() < 0.5 { 0.0 } else { lim } } else { v };
        if k < 0.4 {
            let (pw, ph) = (w * (0.1 + 0.4 * rng.uniform()), h * (0.1 + 0.4 * rng.uniform()));
            let x0 = snap(rng.uniform() * (w - pw), w - pw, rng);
            let y0 = snap(rng.uniform() * (h - ph), h - ph, rng);
            ops.push(FuzzOp::Pocket { x0, y0, w: pw, h: ph, depth: t * (0.2 + 0.9 * rng.uniform()) });
        } else if k < 0.8 {
            let r = 0.5 + 0.15 * w.min(h) * rng.uniform();
            ops.push(FuzzOp::Hole { x: r + (w - 2.0 * r) * rng.uniform(), y: r + (h - 2.0 * r) * rng.uniform(), r, depth: t * (0.3 + rng.uniform()) });
        } else if k < 0.92 {
            let r = 1.0 + 0.1 * w.min(h) * rng.uniform();
            ops.push(FuzzOp::Boss { x: r + (w - 2.0 * r) * rng.uniform(), y: r + (h - 2.0 * r) * rng.uniform(), r, height: 1.0 + 10.0 * rng.uniform() });
        } else if k < 0.96 {
            ops.push(FuzzOp::Fillet { r: 0.2 + 0.1 * w.min(h) * rng.uniform() });
        } else {
            ops.push(FuzzOp::Chamfer { d: 0.2 + 0.1 * w.min(h) * rng.uniform() });
        }
        // Near-degenerate bias (the bug classes found so far): re-place a circle relative to
        // an earlier feature: concentric, tangent to a wall, centred on a wall, on a corner.
        if ops.len() >= 2 && rng.uniform() < 0.3 {
            let anchor = ops[(rng.uniform() * (ops.len() - 1) as f64) as usize].clone();
            let mode = rng.uniform();
            if let Some(FuzzOp::Hole { x, y, r, .. } | FuzzOp::Boss { x, y, r, .. }) = ops.last_mut() {
                match anchor {
                    FuzzOp::Hole { x: ax, y: ay, r: ar, .. } | FuzzOp::Boss { x: ax, y: ay, r: ar, .. } => {
                        if mode < 0.5 { *x = ax; *y = ay; } // concentric
                        else { *x = ax + ar + *r; *y = ay; } // externally tangent
                    }
                    FuzzOp::Pocket { x0, y0, w: pw, h: ph, .. } => {
                        if mode < 0.33 { *x = x0 + pw + *r; *y = y0 + 0.5 * ph; } // tangent to the wall
                        else if mode < 0.66 { *x = x0; *y = y0 + 0.5 * ph; } // centred on the wall
                        else { *x = x0 + pw; *y = y0 + ph; } // on the corner
                    }
                    _ => {}
                }
            }
        }
    }
    FuzzProgram { w, h, t, ops }
}

/// Tool solid and boolean of a feature op (None for corner treatments).
fn tool(p: &FuzzProgram, op: &FuzzOp, dx: f64, dy: f64) -> Option<(LayeredSolid, BoolOp2)> {
    let f = Frame::WORLD;
    Some(match *op {
        FuzzOp::Pocket { x0, y0, w, h, depth } => (LayeredSolid::extrusion(f, &Profile::rectangle(x0 + dx, y0 + dy, x0 + w + dx, y0 + h + dy), p.t - depth, p.t + 1.0), BoolOp2::Difference),
        FuzzOp::Hole { x, y, r, depth } => (LayeredSolid::extrusion(f, &Profile::circle(x + dx, y + dy, r), p.t - depth, p.t + 1.0), BoolOp2::Difference),
        FuzzOp::Boss { x, y, r, height } => (LayeredSolid::extrusion(f, &Profile::circle(x + dx, y + dy, r), p.t, p.t + height), BoolOp2::Union),
        FuzzOp::Fillet { .. } | FuzzOp::Chamfer { .. } => return None,
    })
}

fn apply(p: &FuzzProgram, s: LayeredSolid, op: &FuzzOp, dx: f64, dy: f64) -> LayeredSolid {
    match (tool(p, op, dx, dy), op) {
        (Some((t, b)), _) => s.boolean(&t, b),
        (None, FuzzOp::Fillet { r }) => s.treat_vertical_edges(&[], agcad_geo::region2d::Corner::Fillet(*r)).unwrap_or(s),
        (None, FuzzOp::Chamfer { d }) => s.treat_vertical_edges(&[], agcad_geo::region2d::Corner::Chamfer(*d)).unwrap_or(s),
        _ => s,
    }
}

/// Build the program, optionally translated in the sketch plane.
pub fn build_at(p: &FuzzProgram, dx: f64, dy: f64) -> LayeredSolid {
    let mut s = LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(dx, dy, p.w + dx, p.h + dy), 0.0, p.t);
    for op in &p.ops { s = apply(p, s, op, dx, dy); }
    s
}

pub fn build(p: &FuzzProgram) -> LayeredSolid { build_at(p, 0.0, 0.0) }

/// Like the kernel: a boolean step whose result has a tangential (non-manifold) contact is
/// rejected explicitly. `Err` carries the rejection message.
pub fn build_checked(p: &FuzzProgram) -> Result<LayeredSolid, String> {
    let mut s = LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, p.w, p.h), 0.0, p.t);
    for (i, op) in p.ops.iter().enumerate() {
        if let Some((t, _)) = tool(p, op, 0.0, 0.0) {
            if let Some(c) = s.boolean_contacts(&t).first() { return Err(format!("op {i}: operands touch tangentially at {c:?}")); }
        }
        s = apply(p, s, op, 0.0, 0.0);
        if let Some(c) = s.tangent_contacts().first() { return Err(format!("op {i}: non-manifold tangential contact at {c:?}")); }
    }
    Ok(s)
}

/// Metamorphic relations (no oracle needed), checked on exact layered volumes:
/// inclusion–exclusion per boolean step, translation invariance, commutation of
/// subtractive ops, and STEP write → read preserving validity, face count and volume.
pub fn check_metamorphic(p: &FuzzProgram) -> Result<(), String> {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
    let mut s = LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, p.w, p.h), 0.0, p.t);
    for (i, op) in p.ops.iter().enumerate() {
        if let Some((t, _)) = tool(p, op, 0.0, 0.0).filter(|(t, _)| s.boolean_contacts(t).is_empty()) {
            let (u, x) = (s.boolean(&t, BoolOp2::Union), s.boolean(&t, BoolOp2::Intersection));
            let d = s.boolean(&t, BoolOp2::Difference);
            if !close(u.volume() + x.volume(), s.volume() + t.volume()) { return Err(format!("op {i}: inclusion–exclusion |A∪B|+|A∩B| = {} ≠ |A|+|B| = {}", u.volume() + x.volume(), s.volume() + t.volume())); }
            if !close(d.volume(), s.volume() - x.volume()) { return Err(format!("op {i}: |A−B| = {} ≠ |A|−|A∩B| = {}", d.volume(), s.volume() - x.volume())); }
        }
        s = apply(p, s, op, 0.0, 0.0);
    }
    if s.layers.is_empty() { return Ok(()); }
    let b = s.to_brep();
    let moved = build_at(p, 13.75, -7.125);
    let bm = moved.to_brep();
    if !close(moved.volume(), s.volume()) || bm.faces.len() != b.faces.len() || bm.validate().valid != b.validate().valid {
        return Err(format!("translation changed the result: volume {} → {}, faces {} → {}", s.volume(), moved.volume(), b.faces.len(), bm.faces.len()));
    }
    if p.ops.len() >= 2 && p.ops.iter().all(|o| matches!(o, FuzzOp::Pocket { .. } | FuzzOp::Hole { .. })) {
        let rev = FuzzProgram { ops: p.ops.iter().rev().cloned().collect(), ..p.clone() };
        // the reversed order may pass through a tangent (rejected) intermediate state even when
        // the forward order does not; only compare orders the kernel accepts
        let Ok(r) = build_checked(&rev) else { return Ok(()) };
        if !close(r.volume(), s.volume()) || r.to_brep().faces.len() != b.faces.len() {
            return Err(format!("subtractive ops do not commute: volume {} vs {}", s.volume(), r.volume()));
        }
    }
    let (txt, _) = agcad_geo::step_write::write_step(&b, &Default::default());
    let (bodies, _) = agcad_geo::step_read::read_step(&txt, &Default::default()).map_err(|e| format!("STEP reimport: {e}"))?;
    // one STEP solid per lump; cavities travel as voids of their lump (BREP_WITH_VOIDS)
    let shells_back: usize = bodies.iter().map(|x| x.shells.len()).sum();
    if shells_back != b.shells.len() { return Err(format!("STEP round trip: {} shells → {shells_back}", b.shells.len())); }
    let faces_back: usize = bodies.iter().map(|x| x.faces.len()).sum();
    if bodies.iter().any(|x| !x.validate().valid) || faces_back != b.faces.len() { return Err(format!("STEP round trip: valid {} faces {} → {faces_back}", bodies.iter().all(|x| x.validate().valid), b.faces.len())); }
    let v0 = tessellate(&b, &TessOptions { sag: 1e-3, ..Default::default() }).mass_properties().0;
    let v1: f64 = bodies.iter().map(|x| tessellate(x, &TessOptions { sag: 1e-3, ..Default::default() }).mass_properties().0).sum();
    if (v0 - v1).abs() > 1e-6 * v0.abs().max(1.0) { return Err(format!("STEP round trip volume {v0} → {v1}")); }
    Ok(())
}

/// Check one program; `Err` carries the failure description.
pub fn check_program(p: &FuzzProgram, occt: bool) -> Result<(), String> {
    // An explicit rejection of a non-manifold (tangent) configuration is a correct outcome.
    let Ok(s) = build_checked(p) else { return Ok(()) };
    if s.layers.is_empty() { return Ok(()); } // fully removed: valid empty result
    let b = s.to_brep();
    let v = b.validate();
    if !v.valid { return Err(format!("invalid B-rep: {:?}", v.problems.iter().take(3).collect::<Vec<_>>())); }
    let m = tessellate(&b, &TessOptions { sag: 1e-3, ..Default::default() });
    if !m.is_watertight() { return Err("mesh not watertight".into()); }
    let (vol, _, _) = m.mass_properties();
    let exact = s.volume();
    if (vol - exact).abs() > 1e-3 * exact.max(1e-9) { return Err(format!("volume {vol} vs exact {exact}")); }
    check_metamorphic(p)?;
    if occt {
        let r = agcad_geo_occt::parity(&b, 1e-3)?;
        if r["pass"] != true { return Err(format!("OCCT parity failed: {r}")); }
    }
    Ok(())
}

/// Delta debugging over the op list: smallest failing subsequence (greedy one-at-a-time).
pub fn shrink(p: &FuzzProgram, occt: bool) -> FuzzProgram {
    let mut cur = p.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..cur.ops.len() {
            let mut cand = cur.clone();
            cand.ops.remove(i);
            if check_program(&cand, occt).is_err() { cur = cand; changed = true; break; }
        }
    }
    cur
}

pub fn fuzz(n: usize, seed: u64) -> Vec<Value> {
    let occt = agcad_geo_occt::available();
    let mut rng = Rng::new(seed);
    let mut failures = Vec::new();
    for i in 0..n {
        let p = generate(&mut rng);
        if let Err(e) = check_program(&p, occt) {
            let min = shrink(&p, occt);
            // report the shrunk program's own failure (it may differ from the original's)
            let min_err = check_program(&min, occt).err().unwrap_or_default();
            let test = regression_test_source(&format!("fuzz_seed{seed:x}_case{i}"), &min_err, &min);
            failures.push(json!({ "case": i, "seed": seed, "error": min_err, "original_error": e, "minimal": format!("{min:?}"), "regression_test": test }));
        }
    }
    failures
}

/// Rust source of a regression test for a (shrunk) failing program, in the format of
/// `tests/fuzz_regressions.rs` (Debug of f64 round-trips exactly).
pub fn regression_test_source(name: &str, error: &str, p: &FuzzProgram) -> String {
    let body = format!("{p:?}").replacen("ops: [", "ops: vec![", 1);
    let first_line = error.lines().next().unwrap_or("").replace("*/", "* /");
    format!("\n/// Captured by `agcad conformance --capture`: {first_line}\n#[test]\nfn {name}() {{ ok({body}); }}\n")
}

/// All requirement IDs declared in the spec (parsed from docs at build time is avoided: the
/// list is checked by `cargo xtask check`; here we only report which IDs suites cover).
pub fn coverage() -> Value {
    let covered: std::collections::BTreeSet<&str> = CASES.iter().flat_map(|c| c.requirements.iter().copied()).collect();
    json!({ "covered_requirement_ids": covered })
}

pub fn run_all(fuzz_n: usize) -> Value {
    let mut results = Vec::new();
    let mut failed = 0;
    for c in CASES {
        let r = (c.run)();
        if r.is_err() { failed += 1; }
        results.push(json!({ "case": c.name, "requirements": c.requirements, "ok": r.is_ok(), "error": r.err() }));
    }
    let fz = if fuzz_n > 0 { fuzz(fuzz_n, 0xA9CAD) } else { vec![] };
    failed += fz.len();
    json!({ "cases": results, "fuzz": { "programs": fuzz_n, "failures": fz, "occt_oracle": agcad_geo_occt::available() }, "coverage": coverage(), "failed": failed })
}

#[cfg(test)]
mod tests {
    #[test]
    fn captured_regression_source_is_valid_test_syntax() {
        let p = super::FuzzProgram { w: 0.1 + 0.2, h: 1e-7, t: 3.0, ops: vec![super::FuzzOp::Fillet { r: 1.5 }, super::FuzzOp::Hole { x: 1.0, y: 2.0, r: 0.5, depth: 4.0 }] };
        let src = super::regression_test_source("captured_x", "invalid B-rep: */ edge", &p);
        assert!(src.contains("#[test]\nfn captured_x() { ok(FuzzProgram { w: 0.30000000000000004, h: 1e-7, t: 3.0, ops: vec![Fillet { r: 1.5 }, Hole { x: 1.0, y: 2.0, r: 0.5, depth: 4.0 }] }); }"), "{src}");
        assert!(!src.contains("*/"), "{src}");
    }

    #[test]
    fn suites_pass_and_small_fuzz() {
        let r = super::run_all(20);
        assert_eq!(r["failed"], 0, "{r:#}");
    }
}
