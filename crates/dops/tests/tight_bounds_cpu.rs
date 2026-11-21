//! Tightened rigorous volume bounds: per-cell interval refinement of boundary cells must stay
//! sound (exact / analytic volume inside) and narrow the bounds.

use agcad_dops::abi::fop;
use agcad_dops::examples::{box_goal, flange, goal, plate_goal, pocketed_plate, shelled_box, two_bodies};
use agcad_dops::lower::to_layered;
use agcad_dops::studio::certify_with;
use agcad_dops::{Design, Goal};
use agcad_driver_cpu::CpuDevice;

fn exact(d: &Design) -> f64 { agcad_geo::tessellate::tessellate(&to_layered(d).unwrap().brep(), &Default::default()).mass_properties().0 }

fn width(d: &Design, g: &Goal, levels: u32, t: u32) -> (f64, f64, f64) {
    let o = certify_with(CpuDevice::new(), d, g, levels, 1 << 18, t).unwrap();
    (o[0], o[1], (o[1] - o[0]) / (0.5 * (o[0] + o[1])))
}

#[test]
fn tightened_bounds_are_sound_and_narrow() {
    let mut g2 = box_goal();
    g2.domain_hi = [100.0, 56.0, 16.0];
    let cases: Vec<(&str, Design, Goal, Option<f64>)> = vec![
        ("flange", flange(60.0, 12.0, 6.0), goal(1.0, 2.0), None),
        ("flange-odd", flange(67.3, 9.1, 7.9), goal(1.0, 2.0), None),
        ("plate", pocketed_plate(100.0, 3.0), plate_goal(), None),
        ("union", two_bodies(fop::ADD, 12.0), g2, None),
        ("shell", shelled_box(2.0), box_goal(), Some(100.0 * 60.0 * 10.0 - 96.0 * 56.0 * 6.0)),
    ];
    for (name, d, g, analytic) in cases {
        let v = analytic.unwrap_or_else(|| exact(&d));
        let (l0, h0, w0) = width(&d, &g, 6, 0);
        let (l1, h1, w1) = width(&d, &g, 6, 48);
        eprintln!("{name}: exact {v:.1} | untightened ±{:.2}% | tightened ±{:.3}% [{l1:.1}, {h1:.1}]", 50.0 * w0, 50.0 * w1);
        assert!(l0 <= v && v <= h0, "{name}: untightened unsound");
        assert!(l1 <= v && v <= h1, "{name}: exact {v} ∉ tightened [{l1}, {h1}]");
        assert!(l1 >= l0 && h1 <= h0, "{name}: tightening widened the bounds");
        assert!(w1 < 0.25 * w0, "{name}: ±{} → ±{}", w0 / 2.0, w1 / 2.0);
    }
}
