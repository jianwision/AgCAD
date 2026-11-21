//! Exact inward offsets and closed shells of prismatic solids.

use agcad_geo::builders::Profile;
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::Frame;
use agcad_geo::region2d::{boolean, offset_inward, BoolOp2};
use std::f64::consts::PI;

fn area(ps: &[Profile]) -> f64 { ps.iter().map(|p| p.area()).sum() }

#[test]
fn rectangle_with_holes_offsets_exactly() {
    let t = 2.0;
    // rectangle: sharp inner rectangle
    assert!((area(&offset_inward(&Profile::rectangle(0.0, 0.0, 100.0, 60.0), t).unwrap()) - 96.0 * 56.0).abs() < 1e-6);
    // circular holes grow: r → r + t
    let p = Profile::rectangle(0.0, 0.0, 100.0, 60.0).with_hole(Profile::circle(30.0, 30.0, 5.0)).with_hole(Profile::circle(70.0, 30.0, 5.0));
    let want = 96.0 * 56.0 - 2.0 * PI * 49.0;
    assert!((area(&offset_inward(&p, t).unwrap()) - want).abs() < 1e-6);
    // rectangular hole: its corners are reflex for the region → quarter arcs of radius t
    let p = Profile::rectangle(0.0, 0.0, 100.0, 60.0).with_hole(Profile::rectangle(40.0, 20.0, 60.0, 40.0));
    let grown = 24.0 * 24.0 - (4.0 - PI) * t * t; // (20 + 2t)² minus the four rounded corners
    assert!((area(&offset_inward(&p, t).unwrap()) - (96.0 * 56.0 - grown)).abs() < 1e-6);
    // a grown hole reaching past the outer offset trims it (exact booleans)
    let p = Profile::rectangle(0.0, 0.0, 100.0, 60.0).with_hole(Profile::circle(50.0, 4.0, 3.0));
    let r = offset_inward(&p, t).unwrap();
    let want = area(&boolean(&Profile::rectangle(2.0, 2.0, 98.0, 58.0), &Profile::circle(50.0, 4.0, 5.0), BoolOp2::Difference));
    assert!((area(&r) - want).abs() < 1e-6, "{} vs {want}", area(&r));
    // too thick: the region vanishes
    assert!(offset_inward(&Profile::circle(0.0, 0.0, 1.5), t).unwrap().is_empty());
}

#[test]
fn closed_shell_of_a_box_and_a_holed_plate() {
    let b = LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 100.0, 60.0), 0.0, 10.0);
    let s = b.shell_closed(2.0).unwrap();
    let brep = s.to_brep();
    assert!(brep.validate().valid);
    let v = agcad_geo::tessellate::tessellate(&brep, &Default::default()).mass_properties().0;
    assert!((v - (60000.0 - 96.0 * 56.0 * 6.0)).abs() < 1e-6, "{v}");
    let p = Profile::rectangle(0.0, 0.0, 100.0, 60.0).with_hole(Profile::circle(50.0, 30.0, 8.0));
    let s = LayeredSolid::extrusion(Frame::WORLD, &p, 0.0, 10.0).shell_closed(1.5).unwrap();
    assert!(s.to_brep().validate().valid);
    let v = agcad_geo::tessellate::tessellate(&s.to_brep(), &Default::default()).mass_properties().0;
    let want = (6000.0 - PI * 64.0) * 10.0 - (97.0 * 57.0 - PI * 9.5 * 9.5) * 7.0;
    assert!((v - want).abs() / want < 1e-3, "{v} vs {want} (tessellated arcs)");
    // multi-slab bodies are refused (not approximated)
    let step = b.boolean(&LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 50.0, 60.0), 10.0, 20.0), BoolOp2::Union);
    assert!(step.shell_closed(2.0).is_err());
}
