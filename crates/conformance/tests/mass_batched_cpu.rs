//! Batched GPU-path mass properties (CPU reference device): one pass over 64 tessellated
//! configurations must agree with `Mesh::mass_properties` per configuration.
use agcad_driver::*;
use agcad_kernels::primitives::*;

#[test]
fn batched_mass_properties_match_per_mesh() {
    use agcad_geo::builders::{prism, Profile};
    use agcad_geo::math::Frame;
    let dev = agcad_driver_cpu::CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();
    let l = dev.load_library(&LibrarySource { name: agcad_kernels::PRIMITIVES, cuda_source: None, cpu_kernels: agcad_kernels::reference::PRIMITIVES_CPU, options: vec![] }).unwrap();
    let mut meshes = Vec::new();
    for &n in &[3usize, 6] { for &r in &[50.0, 80.0] { for &t in &[6.0, 20.0] {
        let mut p = Profile::circle(0.0, 0.0, r);
        for k in 0..n { let a = k as f64 * core::f64::consts::TAU / n as f64; p = p.with_hole(Profile::circle((r - 15.0) * a.cos(), (r - 15.0) * a.sin(), 4.5)); }
        meshes.push(agcad_geo::tessellate::tessellate(&prism(&p, &Frame::WORLD, t), &Default::default()));
    }}}
    let (mut pos, mut tris, mut offs) = (Vec::<f64>::new(), Vec::<u32>::new(), vec![0u32]);
    for m in &meshes {
        let base = (pos.len() / 3) as u32;
        for p in &m.positions { pos.extend([p.x, p.y, p.z]); }
        for t in &m.triangles { tris.extend(t.iter().map(|i| i + base)); }
        offs.push((tris.len() / 3) as u32);
    }
    let up = |b: &[u8]| { let x = dev.alloc_bytes(&s, b.len()).unwrap(); dev.upload_from_host_async(&s, b, &x, 0).unwrap(); x };
    let (pb, tb, ob) = (up(bytemuck::cast_slice(&pos)), up(bytemuck::cast_slice(&tris)), up(bytemuck::cast_slice(&offs)));
    let sums_b = mesh_mass_sums(&*dev, &s, l, &pb, &tb, (tris.len() / 3) as u32, &ob, meshes.len() as u32).unwrap();
    let sums: Vec<f64> = bytemuck::pod_collect_to_vec(&sums_b.as_slice()[..MESH_TERMS * meshes.len() * 8]);
    for (m, mesh) in meshes.iter().enumerate() {
        let (v, a, c) = mass_properties_from_sums(&sums, meshes.len(), m);
        let (v0, a0, c0) = mesh.mass_properties();
        assert!((v - v0).abs() <= 1e-12 * v0 && (a - a0).abs() <= 1e-12 * a0, "{m}: {v} {v0} {a} {a0}");
        assert!((c[2] - c0.z).abs() < 1e-9, "{m}: centroid {c:?} vs {c0:?}");
        let r = [50.0, 80.0][(m / 2) % 2];
        let t = [6.0, 20.0][m % 2];
        let n = [3.0, 6.0][m / 4];
        let exact = core::f64::consts::PI * (r * r - n * 4.5 * 4.5) * t;
        assert!((v - exact).abs() / exact < 2e-3, "{m}: {v} vs exact {exact}");
    }
}
