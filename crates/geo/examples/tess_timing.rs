use agcad_geo::builders::{prism, Profile};
use agcad_geo::math::Frame;
use agcad_geo::tessellate::{tessellate, TessOptions};
fn main() {
    let mut p = Profile::circle(0.0, 0.0, 80.0);
    for k in 0..8 { let a = k as f64 * core::f64::consts::TAU / 8.0; p = p.with_hole(Profile::circle(40.0 * a.cos(), 40.0 * a.sin(), 4.5)); }
    let b = prism(&p, &Frame::WORLD, 10.0);
    for sag in [0.5, 0.01, 0.001] {
        let t = std::time::Instant::now();
        let m = tessellate(&b, &TessOptions { sag, ..Default::default() });
        println!("sag {sag}: {} tris, {:.1} ms, watertight {}", m.triangles.len(), t.elapsed().as_secs_f64() * 1e3, m.is_watertight());
    }
}
