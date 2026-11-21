//! Engineering evaluators on the device (sampled evidence): mass properties, surface area,
//! wall thickness, clearance and 3-axis tool access, checked against analytic values and the
//! exact kernel on the CPU reference device.

use agcad_dops::abi::{fop, ob, profile, Goal, NO_PARAM};
use agcad_dops::examples::{box_goal, feature, flange, goal, shelled_box};
use agcad_dops::studio::certify;
use agcad_dops::Design;
use agcad_driver_cpu::CpuDevice;

fn plain_box() -> Design {
    let b = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    Design { nodes: vec![b], params: vec![], names: vec![] }
}

fn obs(d: &Design, g: &Goal) -> [f64; agcad_dops::abi::OBS] { certify(CpuDevice::new(), d, g, 6, 1 << 16).unwrap() }

#[test]
fn mass_properties_and_area_match_analytic_box() {
    let o = obs(&plain_box(), &box_goal());
    let (c, i) = (&o[ob::CENTROID..ob::CENTROID + 3], &o[ob::INERTIA..ob::INERTIA + 3]);
    for (k, want) in [50.0, 30.0, 5.0].iter().enumerate() { assert!((c[k] - want).abs() < 0.05, "centroid {c:?}"); }
    let v = 60_000.0;
    let want = [v * (60.0f64.powi(2) + 10.0f64.powi(2)) / 12.0, v * (100.0f64.powi(2) + 10.0f64.powi(2)) / 12.0, v * (100.0f64.powi(2) + 60.0f64.powi(2)) / 12.0];
    for k in 0..3 { assert!((i[k] - want[k]).abs() / want[k] < 0.01, "inertia {i:?} vs {want:?}"); }
    let area = 2.0 * (100.0 * 60.0 + 100.0 * 10.0 + 60.0 * 10.0);
    assert!((o[ob::AREA] - area).abs() / area < 0.05, "area {} vs {area}", o[ob::AREA]);
    assert_eq!(o[ob::MIN_WALL], -1.0, "wall check off by default");
}

#[test]
fn flange_centroid_and_area_match_the_exact_kernel() {
    let d = flange(60.0, 12.0, 6.0);
    let o = obs(&d, &goal(1.0, 2.0));
    let m = agcad_geo::tessellate::tessellate(&agcad_dops::lower::to_layered(&d).unwrap().brep(), &Default::default());
    let (_, area, c) = m.mass_properties();
    let got = &o[ob::CENTROID..ob::CENTROID + 3];
    assert!((got[0] - c.x).abs() < 0.1 && (got[1] - c.y).abs() < 0.1 && (got[2] - c.z).abs() < 0.05, "centroid {got:?} vs {c:?}");
    assert!((o[ob::AREA] - area).abs() / area < 0.05, "area {} vs exact {area}", o[ob::AREA]);
}

#[test]
fn wall_thickness_flags_thin_shells_only() {
    let mut g = box_goal();
    g.min_wall = 3.0;
    let thin = obs(&shelled_box(2.0), &g);
    assert!(thin[ob::THIN_AREA] > 1000.0, "2 mm shell vs 3 mm minimum: thin area {}", thin[ob::THIN_AREA]);
    assert!(thin[ob::MIN_WALL] > 1.5 && thin[ob::MIN_WALL] < 2.6, "sampled min wall {}", thin[ob::MIN_WALL]);
    assert!(thin[ob::VIOLATION] > 0.0);
    g.min_wall = 1.5;
    let ok = obs(&shelled_box(2.0), &g);
    assert_eq!(ok[ob::THIN_AREA], 0.0);
    assert_eq!(ok[ob::MIN_WALL], 1.5, "no sample thinner than the requirement");
    assert_eq!(ok[ob::VIOLATION], 0.0);
}

#[test]
fn clearance_measures_hole_diameter() {
    let d = flange(60.0, 12.0, 6.0); // holes of radius 4.5
    let mut g = goal(1.0, 2.0);
    g.min_gap = 12.0;
    let o = obs(&d, &g);
    assert!(o[ob::GAP_AREA] > 0.0, "9 mm holes vs 12 mm clearance");
    assert!(o[ob::MIN_GAP] > 8.5 && o[ob::MIN_GAP] < 9.6, "sampled min gap {}", o[ob::MIN_GAP]);
    g.min_gap = 8.0;
    assert_eq!(obs(&d, &g)[ob::GAP_AREA], 0.0);
}

#[test]
fn three_axis_access_finds_enclosed_cavities() {
    let mut g = box_goal();
    g.machining_3ax = 1;
    assert_eq!(obs(&plain_box(), &g)[ob::UNDERCUT_AREA], 0.0, "a block is fully accessible");
    let mut gf = goal(1.0, 2.0);
    gf.machining_3ax = 1;
    assert_eq!(obs(&flange(60.0, 12.0, 6.0), &gf)[ob::UNDERCUT_AREA], 0.0, "through holes are accessible");
    let o = obs(&shelled_box(2.0), &g);
    // cavity floor and ceiling (2 × 96 × 56) are unreachable from +z
    assert!(o[ob::UNDERCUT_AREA] > 0.8 * 2.0 * 96.0 * 56.0, "undercut {}", o[ob::UNDERCUT_AREA]);
    assert!(o[ob::VIOLATION] > 0.0);
    g.undercut_tol = 1e9;
    assert_eq!(obs(&shelled_box(2.0), &g)[ob::VIOLATION], 0.0, "tolerated");
}

#[test]
fn centroid_limits_gate_the_goal() {
    let mut g = box_goal();
    g.vol_lo = 50_000.0;
    g.vol_hi = 70_000.0;
    assert_eq!(obs(&plain_box(), &g)[ob::MET], 1.0);
    g.centroid_lo = [0.0, 0.0, 0.0];
    g.centroid_hi = [40.0, 100.0, 100.0];
    let o = obs(&plain_box(), &g);
    assert_eq!(o[ob::CENTROID_OK], 0.0);
    assert_eq!(o[ob::MET], 0.0);
}
