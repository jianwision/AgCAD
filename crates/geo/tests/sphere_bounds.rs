use agcad_geo::{
    brep::{BodyKind, Brep},
    builders,
    curve::Curve,
    math::{Frame, V3, v3},
    surface::Surface,
};
use std::f64::consts::FRAC_PI_2;

#[test]
fn translated_full_sphere_has_six_extrema() {
    let center = v3(4., 5., 6.);
    let sphere = builders::sphere(
        &Frame {
            origin: center,
            ..Frame::WORLD
        },
        6.,
    );
    let bb = sphere.bbox();
    assert_eq!(bb.min, v3(-2., -1., 0.));
    assert_eq!(bb.max, v3(10., 11., 12.));
}

#[test]
fn trimmed_octant_excludes_opposite_hemisphere() {
    let mut patch = Brep::new(BodyKind::Sheet);
    let shell = patch.add_shell(false);
    let surf = patch.add_surface(Surface::Sphere {
        frame: Frame::WORLD,
        r: 1.,
    });
    let axes = [V3::X, V3::Y, V3::Z];
    let vertices = axes.map(|p| patch.add_vertex(p));
    let mut boundary = Vec::new();
    for i in 0..3 {
        let j = (i + 1) % 3;
        let (x, y) = (axes[i], axes[j]);
        let frame = Frame {
            origin: V3::ZERO,
            x,
            y,
            z: x.cross(y),
        };
        let curve = patch.add_curve(Curve::Circle { frame, r: 1. });
        boundary.push((
            patch.add_edge(curve, vertices[i], vertices[j], 0., FRAC_PI_2, "arc"),
            true,
        ));
    }
    patch.add_face(
        surf,
        true,
        shell,
        &boundary,
        &[],
        "octant",
        [0., FRAC_PI_2, 0., FRAC_PI_2],
    );
    let bb = patch.bbox();
    assert_eq!(bb.min, V3::ZERO);
    assert_eq!(bb.max, v3(1., 1., 1.));
}
