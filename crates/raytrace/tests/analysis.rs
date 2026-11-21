//! Ray-query analyses on shapes with known answers (CPU reference tracer); OptiX parity on
//! RTX machines (`--features optix -- --ignored`).

use agcad_geo::builders::Profile;
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::Frame;
#[cfg(feature = "optix")]
use agcad_geo::math::v3;
use agcad_geo::tessellate::{tessellate, Mesh};
use agcad_raytrace::analysis::{access, along_normal, directions, render};
use agcad_raytrace::{CpuTracer, Ray, Tracer};

fn mesh(l: &LayeredSolid) -> Mesh { tessellate(&l.to_brep(), &Default::default()) }
fn plate_with_hole() -> Mesh { mesh(&LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 100.0, 60.0).with_hole(Profile::circle(50.0, 30.0, 4.5)), 0.0, 12.0)) }
fn shelled_box() -> Mesh { mesh(&LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 100.0, 60.0), 0.0, 10.0).shell_closed(2.0).unwrap()) }

#[test]
fn cpu_tracer_hits_a_box_exactly() {
    let m = mesh(&LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 10.0, 10.0), 0.0, 10.0));
    let t = CpuTracer::new(&m);
    let h = t.trace(&[Ray { origin: [5.0, 5.0, 20.0], tmin: 0.0, dir: [0.0, 0.0, -1.0], tmax: 1e30 }, Ray { origin: [50.0, 5.0, 20.0], tmin: 0.0, dir: [0.0, 0.0, -1.0], tmax: 1e30 }]).unwrap();
    assert!((h[0].t - 10.0).abs() < 1e-5 && h[0].front == 1, "{:?}", h[0]);
    assert!(h[1].t < 0.0, "miss");
}

#[test]
fn closed_cavities_are_unreachable_and_through_holes_are_reachable() {
    let five = directions("5ax").unwrap();
    let a = access(&CpuTracer::new(&plate_with_hole()), &plate_with_hole(), &directions("3+2").unwrap()).unwrap();
    assert!(a.unreachable < 1e-6 * a.area, "plate with a through hole: {a:?}");
    let sb = shelled_box();
    let a = access(&CpuTracer::new(&sb), &sb, &five).unwrap();
    let cavity = 2.0 * (96.0 * 56.0 + 96.0 * 6.0 + 56.0 * 6.0);
    assert!((a.unreachable - cavity).abs() / cavity < 1e-6, "cavity {} vs {cavity}", a.unreachable);
}

#[test]
fn wall_thickness_and_clearance_along_normals() {
    let sb = shelled_box();
    let w = along_normal(&CpuTracer::new(&sb), &sb, true).unwrap();
    assert!((w.min() - 2.0).abs() < 1e-3, "min wall {}", w.min());
    assert!(w.area_below(2.5) > 0.9 * sb.mass_properties().1, "everything is a 2 mm wall");
    let p = plate_with_hole();
    let g = along_normal(&CpuTracer::new(&p), &p, false).unwrap();
    assert!((g.min() - 9.0).abs() < 0.05, "hole diameter {}", g.min());
}

#[test]
fn render_is_a_png() {
    let p = plate_with_hole();
    let png = render(&CpuTracer::new(&p), &p, 96, 96).unwrap();
    assert_eq!(&png[1..4], b"PNG");
}

/// OptiX ≡ CPU reference on random rays (hit / miss and distances to 1e-4 relative).
#[cfg(feature = "optix")]
#[test]
#[ignore = "requires an RTX GPU"]
fn optix_matches_the_cpu_reference() {
    let m = shelled_box();
    let o = agcad_raytrace::OptixTracer::new(&m).expect("OptiX");
    let c = CpuTracer::new(&m);
    let mut s = 12345u64;
    let mut u = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 11) as f64) / ((1u64 << 53) as f64) };
    let rays: Vec<Ray> = (0..20000).map(|_| {
        let o = v3(-20.0 + 140.0 * u(), -20.0 + 100.0 * u(), -10.0 + 30.0 * u());
        let d = v3(u() - 0.5, u() - 0.5, u() - 0.5).normalized();
        Ray { origin: [o.x as f32, o.y as f32, o.z as f32], tmin: 0.0, dir: [d.x as f32, d.y as f32, d.z as f32], tmax: 1e30 }
    }).collect();
    let (a, b) = (o.trace(&rays).unwrap(), c.trace(&rays).unwrap());
    let mut same = 0;
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.t < 0.0, y.t < 0.0, "hit / miss agree");
        if x.t >= 0.0 { assert!((x.t - y.t).abs() <= 1e-4 * y.t.max(1.0), "{x:?} vs {y:?}"); if x.prim == y.prim { same += 1; } }
    }
    eprintln!("OptiX vs CPU: {} rays, {same} same-triangle hits", rays.len());
    let w = std::time::Instant::now();
    let big: Vec<Ray> = rays.iter().cycle().take(4_000_000).copied().collect();
    o.trace(&big).unwrap();
    eprintln!("OptiX: 4M rays in {:.1} ms", w.elapsed().as_secs_f64() * 1e3);
}
