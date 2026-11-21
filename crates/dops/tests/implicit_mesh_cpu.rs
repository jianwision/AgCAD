//! Certified implicit deliverable: watertight, inside the certified volume bounds, close to
//! the exact solid, and the deviation bound holds.

use agcad_dops::abi::ob;
use agcad_dops::examples::{box_goal, flange, goal, shelled_box};
use agcad_dops::mesh::extract_mesh;
use agcad_dops::studio::certify;
use agcad_driver_cpu::CpuDevice;

#[test]
fn shelled_box_mesh_is_watertight_and_certified() {
    let (d, g) = (shelled_box(2.0), box_goal());
    let o = certify(CpuDevice::new(), &d, &g, 6, 1 << 18).unwrap();
    let lo = [o[ob::BOX_LO], o[ob::BOX_LO + 1], o[ob::BOX_LO + 2]];
    let hi = [lo[0] + o[5], lo[1] + o[6], lo[2] + o[7]];
    let m = extract_mesh(CpuDevice::new(), &d, lo, hi, 0.5).unwrap();
    assert!(m.mesh.is_watertight(), "closed, consistently oriented");
    let analytic = 100.0 * 60.0 * 10.0 - 96.0 * 56.0 * 6.0;
    // a mesh within h√3 of the surface: volume within area × deviation of the solid's
    assert!((m.volume - analytic).abs() <= m.area * m.deviation, "mesh {} vs {analytic}", m.volume);
    assert!((m.volume - analytic).abs() / analytic < 0.01, "mesh {} vs {analytic}", m.volume);
    assert!((m.area - 2.0 * (6000.0 + 1000.0 + 600.0 + 96.0 * 56.0 + 96.0 * 6.0 + 56.0 * 6.0)).abs() / m.area < 0.02, "area {}", m.area);
    // two closed shells (outer surface and cavity)
    assert!(o[0] <= analytic && analytic <= o[1]);
}

#[test]
fn flange_mesh_vertices_lie_within_the_deviation_bound() {
    let d = flange(60.0, 12.0, 6.0);
    let m = extract_mesh(CpuDevice::new(), &d, [-60.0, -60.0, 0.0], [60.0, 60.0, 12.0], 1.0).unwrap();
    assert!(m.mesh.is_watertight());
    let exact = std::f64::consts::PI * (3600.0 - 6.0 * 20.25) * 12.0;
    assert!((m.volume - exact).abs() / exact < 0.01, "{} vs {exact}", m.volume);
    // analytic distance to the flange surface for a few vertices: |r − 60|, |z|, |z − 12|, holes
    for p in m.mesh.positions.iter().step_by(97) {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        let mut dist = (r - 60.0).abs().min(p.z.abs()).min((p.z - 12.0).abs());
        for k in 0..6 {
            let a = k as f64 * std::f64::consts::TAU / 6.0;
            let (cx, cy) = (45.0 * a.cos(), 45.0 * a.sin());
            dist = dist.min((((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt() - 4.5).abs());
        }
        assert!(dist <= m.deviation + 1e-9, "vertex {p:?} at {dist} > {}", m.deviation);
    }
    let _ = goal(1.0, 2.0);
}
