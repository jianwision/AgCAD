//! OCCT oracle for the exact closed shell (`LayeredSolid::shell_closed`): on fuzzed holed
//! plates, our shell volume must equal V − V(OCCT inward offset by t). Requires cadquery-ocp
//! (`cargo test -p agcad-geo-occt -- --ignored`).

use agcad_geo::builders::Profile;
use agcad_geo::layered::LayeredSolid;
use agcad_geo::math::Frame;

fn lcg(s: &mut u64) -> f64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 11) as f64) / ((1u64 << 53) as f64) }

#[test]
#[ignore = "requires OCCT (cadquery-ocp)"]
fn shell_offset_matches_occt() {
    if !agcad_geo_occt::available() { eprintln!("OCCT not available: skipped"); return; }
    let mut seed = 0xA9CAD_5EEDu64;
    let (mut ok, mut worst) = (0, 0.0f64);
    for case in 0..30 {
        let (w, h, th) = (40.0 + 80.0 * lcg(&mut seed), 30.0 + 60.0 * lcg(&mut seed), 8.0 + 12.0 * lcg(&mut seed));
        let t = 0.8 + 2.2 * lcg(&mut seed);
        let mut p = Profile::rectangle(0.0, 0.0, w, h);
        // holes well inside, apart from each other and the outline (no degenerate offsets)
        let mut holes: Vec<([f64; 2], f64)> = Vec::new();
        for _ in 0..(lcg(&mut seed) * 4.0) as usize {
            let r = 3.0 + 5.0 * lcg(&mut seed);
            let m = r + 2.0 * t + 2.0;
            if w < 2.0 * m || h < 2.0 * m { continue; }
            let c = [m + (w - 2.0 * m) * lcg(&mut seed), m + (h - 2.0 * m) * lcg(&mut seed)];
            if holes.iter().any(|(q, rq)| ((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).sqrt() < r + rq + 2.0 * t + 2.0) { continue; }
            holes.push((c, r));
        }
        for (c, r) in &holes { p = p.with_hole(Profile::circle(c[0], c[1], *r)); }
        if lcg(&mut seed) < 0.5 {
            // a rectangular hole (its corners are concave for the solid → arcs)
            let (rx, ry) = (0.3 * w, 0.3 * h);
            let c = [0.5 * w, 0.5 * h];
            let clear = holes.iter().all(|(q, r)| (q[0] - c[0]).abs() > 0.5 * rx + r + 2.0 * t + 2.0 || (q[1] - c[1]).abs() > 0.5 * ry + r + 2.0 * t + 2.0);
            if clear { p = p.with_hole(Profile::rectangle(c[0] - 0.5 * rx, c[1] - 0.5 * ry, c[0] + 0.5 * rx, c[1] + 0.5 * ry)); }
        }
        let body = LayeredSolid::extrusion(Frame::WORLD, &p, 0.0, th);
        let shell = body.shell_closed(t).unwrap();
        let ours = shell.volume();
        let inner = agcad_geo_occt::offset_volume(&body.to_brep(), -t).unwrap_or_else(|e| panic!("case {case}: OCCT offset failed: {e}"));
        let occt = body.volume() - inner;
        let rel = (ours - occt).abs() / occt;
        worst = worst.max(rel);
        assert!(rel < 1e-4, "case {case} (w {w:.1}, h {h:.1}, th {th:.1}, t {t:.2}, {} holes): ours {ours} vs OCCT {occt}", p.loops.len() - 1);
        ok += 1;
    }
    eprintln!("shell vs OCCT offset: {ok}/30 cases, worst relative difference {worst:.2e}");
}
