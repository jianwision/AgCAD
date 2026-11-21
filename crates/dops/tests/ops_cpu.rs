//! New device operations: general sketches (slots, arcs, holes), revolve, mirror (exact
//! path + document round trip) and draft / rounded caps (device-only, certified implicit).

use agcad_dops::examples::*;
use agcad_dops::lower::{raise, to_layered, to_program};
use agcad_dops::studio::certify;
use agcad_dops::Design;
use agcad_driver_cpu::CpuDevice;
use std::f64::consts::PI;

fn exact(d: &Design) -> f64 { agcad_geo::tessellate::tessellate(&to_layered(d).unwrap().brep(), &Default::default()).mass_properties().0 }

fn sound(name: &str, d: &Design, want: f64) {
    let o = certify(CpuDevice::new(), d, &ops_goal(), 6, 1 << 18).unwrap();
    eprintln!("{name}: want {want:.1} ∈ [{:.1}, {:.1}], estimate {:.1}", o[0], o[1], o[12]);
    assert!(o[0] <= want && want <= o[1], "{name}: {want} ∉ [{}, {}]", o[0], o[1]);
    assert!((o[12] - want).abs() / want < 0.02, "{name}: estimate {} vs {want}", o[12]);
}

fn round_trip(name: &str, d: &Design) {
    let v = exact(d);
    let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut sess = agcad_harness::HarnessSession::new(req, "ops");
    let prog = agcad_ir::json::decode_program(&to_program(d, 1, 0).to_string(), agcad_ir::Catalog::global()).unwrap();
    let out = sess.run_program("main", &prog, agcad_ir::Catalog::global()).unwrap();
    assert_eq!(out.report.status, "ACCEPTED", "{name}: {:?}", out.report.features);
    let (raised, warnings) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
    assert!(warnings.is_empty(), "{name}: {warnings:?}");
    let vr = exact(&raised);
    assert!((vr - v).abs() / v < 1e-9, "{name}: raised {vr} vs {v}");
}

#[test]
fn slot_sketch_is_exact_and_round_trips() {
    let d = slotted_plate();
    let want = 60_000.0 - (400.0 + 25.0 * PI) * 10.0;
    assert!((exact(&d) - want).abs() / want < 1e-3);
    sound("slot", &d, want);
    round_trip("slot", &d);
}

#[test]
fn arc_sketch_with_hole_is_exact_and_round_trips() {
    let d = l_bracket();
    // L area 60·15 + 15·35 plus the concave fillet's corner square minus its quarter disc, minus the hole
    let area = 60.0 * 15.0 + 15.0 * 35.0 - (36.0 - 9.0 * PI) * -1.0 - 9.0 * PI;
    let want = area * 12.0;
    let v = exact(&d);
    assert!((v - want).abs() / want < 2e-3, "exact {v} vs {want}");
    sound("l-bracket", &d, v);
    round_trip("l-bracket", &d);
}

#[test]
fn rectilinear_revolve_is_exact_and_round_trips() {
    let d = stepped_shaft();
    let want = PI * (400.0 * 10.0 + 144.0 * 30.0);
    assert!((exact(&d) - want).abs() / want < 2e-3);
    sound("shaft", &d, want);
    round_trip("shaft", &d);
}

#[test]
fn mirror_is_exact_and_round_trips() {
    let d = mirrored_bracket();
    let want = 2.0 * (30.0 * 40.0 - 25.0 * PI) * 8.0;
    assert!((exact(&d) - want).abs() / want < 2e-3);
    sound("mirror", &d, want);
    round_trip("mirror", &d);
}

#[test]
fn device_only_ops_are_sound_and_delivered_implicitly() {
    let g = ops_goal();
    // frustum of the drafted boss: ∫₀²⁰ (40 − 2·0.1·z)² dz
    let draft = (0..20_000).map(|i| { let z = (i as f64 + 0.5) * 1e-3; (40.0 - 0.2 * z).powi(2) * 1e-3 }).sum::<f64>();
    // rounded caps (r 4): layers inset by r − √(r² − s²) within r of either cap
    let r: f64 = 4.0;
    let round = 60.0 * 40.0 * 12.0 + 2.0 * (0..4000).map(|i| { let s = (i as f64 + 0.5) * 1e-3; let dlt = r - (r * r - (r - s).powi(2)).sqrt(); (60.0 - 2.0 * dlt) * (40.0 - 2.0 * dlt) * 1e-3 }).sum::<f64>();
    let cone = PI * 30.0 * (400.0 + 200.0 + 100.0) / 3.0;
    for (name, d, want) in [("draft", drafted_boss(), draft), ("round", rounded_block(), round), ("cone", coned_shaft(), cone)] {
        let e = to_layered(&d).unwrap_err();
        assert!(e.contains("device-only"), "{name}: {e}");
        sound(name, &d, want);
        let o = certify(CpuDevice::new(), &d, &g, 6, 1 << 18).unwrap();
        let lo = [o[27], o[28], o[29]];
        let m = agcad_dops::mesh::extract_mesh(CpuDevice::new(), &d, lo, [lo[0] + o[5], lo[1] + o[6], lo[2] + o[7]], 0.4).unwrap();
        assert!(m.mesh.is_watertight(), "{name}");
        assert!((m.volume - want).abs() <= m.area * m.deviation && (m.volume - want).abs() / want < 0.01, "{name}: mesh {} vs {want}", m.volume);
    }
}

#[test]
fn sketch_sdf_sign_and_distance_are_exact() {
    use agcad_dops::math::sketch2d;
    let d = l_bracket();
    let (cn, tabs) = d.compile();
    let prof = agcad_dops::lower::sketch_profile(&d.profiles[0], 0.0, 0.0, 0.0).unwrap();
    // boundary samples for a distance reference
    let mut bnd = Vec::new();
    for lp in &d.profiles[0] {
        for s in lp {
            for k in 0..=400 {
                let t = k as f64 / 400.0;
                bnd.push(match *s { agcad_dops::abi::Seg::Line { a, b } => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], agcad_dops::abi::Seg::Arc { c, r, a0, a1 } => { let u = a0 + t * (a1 - a0); [c[0] + r * u.cos(), c[1] + r * u.sin()] } });
            }
        }
    }
    let mut h = 12345u32;
    for _ in 0..3000 {
        h = h.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = -10.0 + 80.0 * (h >> 8) as f64 / 16_777_216.0;
        h = h.wrapping_mul(1664525).wrapping_add(1013904223);
        let y = -10.0 + 70.0 * (h >> 8) as f64 / 16_777_216.0;
        let f = sketch2d(&tabs[..agcad_dops::abi::TABLE], cn[0].sides, x as f32, y as f32) as f64;
        let inside = agcad_geo::region2d::winding(&prof.loops, [x, y]).round() as i64 > 0;
        let dref = bnd.iter().map(|p| (p[0] - x).hypot(p[1] - y)).fold(f64::INFINITY, f64::min);
        if dref > 1e-3 { assert_eq!(f < 0.0, inside, "sign at ({x}, {y}): f {f}"); }
        assert!(f.abs() <= dref + 1e-4 && f.abs() >= dref - 0.2, "distance at ({x}, {y}): {f} vs ≈{dref}");
    }
}

/// LLM-style document: counterbored + through holes, a user pattern, and one sketch of four
/// bolt circles (several outer loops). Raise must not drop anything, and the device design's
/// exact volume must equal the document's; device bounds must contain it.
#[test]
fn holes_user_patterns_and_multi_island_sketches_raise_exactly() {
    let req = agcad_harness::Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut sess = agcad_harness::HarnessSession::new(req, "plate");
    let prog = serde_json::json!({"agir": "0.2", "requirements_rev": 1, "expected_revision": 0, "program": [
        {"id": "s1", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "r1", "op": "sketch.rectangle", "args": {"sketch": "%s1", "mode": "corners", "pts": [["0 mm", "0 mm"], ["120 mm", "80 mm"]]}},
        {"id": "g1", "op": "sketch.regions", "args": {"sketch": "%r1"}},
        {"id": "pad", "op": "part.pad", "args": {"profile": "%g1", "limits": "12 mm"}},
        {"id": "h1", "op": "part.hole", "args": {"support": "%pad", "positions": [["15 mm", "15 mm"], ["105 mm", "15 mm"]], "spec": {"d": "6 mm", "through": true, "counterbore_d": "11 mm", "counterbore_depth": "4 mm"}}},
        {"id": "h2", "op": "part.hole", "args": {"support": "%pad", "positions": [["60 mm", "65 mm"]], "spec": {"d": "8 mm", "depth": "7 mm"}}},
        {"id": "s2", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "c1", "op": "sketch.circle", "args": {"sketch": "%s2", "mode": "center_radius", "pts": [["50 mm", "40 mm"]], "r": "3 mm"}},
        {"id": "c2", "op": "sketch.circle", "args": {"sketch": "%c1", "mode": "center_radius", "pts": [["70 mm", "40 mm"]], "r": "3 mm"}},
        {"id": "c3", "op": "sketch.circle", "args": {"sketch": "%c2", "mode": "center_radius", "pts": [["60 mm", "30 mm"]], "r": "3 mm"}},
        {"id": "c4", "op": "sketch.circle", "args": {"sketch": "%c3", "mode": "center_radius", "pts": [["60 mm", "50 mm"]], "r": "3 mm"}},
        {"id": "g2", "op": "sketch.regions", "args": {"sketch": "%c4"}},
        {"id": "bolts", "op": "part.pocket", "args": {"profile": "%g2", "limits": {"type": "two_sided", "d": "13 mm", "d2": "1 mm"}}},
        {"id": "s3", "op": "sketch.create", "args": {"support": "xy"}},
        {"id": "r3", "op": "sketch.rectangle", "args": {"sketch": "%s3", "mode": "corners", "pts": [["10 mm", "35 mm"], ["20 mm", "45 mm"]]}},
        {"id": "g3", "op": "sketch.regions", "args": {"sketch": "%r3"}},
        {"id": "sq", "op": "part.pocket", "args": {"profile": "%g3", "limits": {"type": "two_sided", "d": "13 mm", "d2": "-8 mm"}}},
        {"id": "up", "op": "transform.user_pattern", "args": {"items": ["%sq"], "positions": [["15 mm", "40 mm"], ["105 mm", "40 mm"], ["15 mm", "62 mm"]]}}
    ]});
    let p = agcad_ir::json::decode_program(&prog.to_string(), agcad_ir::Catalog::global()).unwrap();
    let out = sess.run_program("main", &p, agcad_ir::Catalog::global()).unwrap();
    assert_eq!(out.report.status, "ACCEPTED", "{:?}", out.report.features);
    let u = sess.last_update.as_ref().unwrap();
    let last = u.outcomes.last().unwrap().feature;
    let agcad_runtime::rebuild::Product::Solid(s) = &u.products[&last] else { panic!("no solid") };
    let vdoc = agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0;
    let (d, warnings) = raise(&sess.doc.workspace("main").unwrap().state).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let v = exact(&d);
    assert!((v - vdoc).abs() / vdoc < 1e-9, "device design {v} vs document {vdoc}");
    let mut g = ops_goal();
    g.domain_lo = [-6.0, -6.0, -4.0];
    g.domain_hi = [126.0, 86.0, 16.0];
    let o = certify(CpuDevice::new(), &d, &g, 6, 1 << 18).unwrap();
    assert!(o[0] <= v && v <= o[1]);
    round_trip("holes", &d);
}

#[test]
fn user_pattern_and_two_island_sketch_are_sound() {
    let d = user_pattern_plate();
    let want = 60_000.0 - 4.0 * 100.0 * 6.0 - core::f64::consts::PI * (16.0 + 9.0) * 10.0;
    let v = exact(&d);
    assert!((v - want).abs() / want < 2e-3, "{v} vs {want}");
    sound("user-pattern", &d, v);
    round_trip("user-pattern", &d);
}
