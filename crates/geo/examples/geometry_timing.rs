//! Compare warm CAD dispatch times with AGCAD_GEOMETRY_BACKEND=host and cuda.
//! Pass a native STEP file containing four-sided NURBS faces.
use agcad_geo::{
    accelerator::{Dispatch, capture, mass_properties, wall_minimum},
    step_read::{ImportLimits, read_step},
    tessellate::{TessOptions, tessellate_for_measurement, try_tessellate},
};
use std::{hint::black_box, time::Instant};
fn checked<T>(
    required: Option<&str>,
    operation: impl FnOnce() -> agcad_driver::DriverResult<T>,
) -> (T, Vec<Dispatch>, f64) {
    let ((result, elapsed), events) = capture(|| {
        let start = Instant::now();
        let value = operation();
        (value, start.elapsed().as_secs_f64() * 1000.)
    });
    let value = result.expect("geometry operation failed");
    let backend = std::env::var("AGCAD_GEOMETRY_BACKEND").unwrap_or_default();
    for event in &events {
        assert!(
            event.error.is_none(),
            "{} fell back: {:?}",
            event.operation,
            event.error
        );
        if matches!(backend.as_str(), "host" | "cuda") {
            assert_eq!(
                event.backend, backend,
                "unexpected backend for {}",
                event.operation
            );
        }
    }
    if let Some(name) = required {
        assert!(
            events.iter().any(|e| e.operation == name),
            "missing dispatch for {name}"
        );
    }
    (value, events, elapsed)
}

fn median_ms<T>(
    required: Option<&str>,
    mut operation: impl FnMut() -> agcad_driver::DriverResult<T>,
) -> f64 {
    black_box(checked(required, &mut operation).0);
    let mut samples = [0.0; 7];
    for t in &mut samples {
        let (value, _, elapsed) = checked(required, &mut operation);
        black_box(value);
        *t = elapsed;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}
fn main() {
    let file = std::env::args()
        .nth(1)
        .expect("provide the native STEP file to measure");
    let (bodies, _) = read_step(
        &std::fs::read_to_string(file).unwrap(),
        &ImportLimits::default(),
    )
    .unwrap();
    let body = &bodies[0];
    println!(
        "sag_mm,triangles,curvature_probes,interior_points,tess_ms,mass_ms,wall_ms,batch16_mass_ms"
    );
    for sag in [1., 0.25, 0.1, 0.03, 0.01, 0.002] {
        let options = TessOptions {
            sag,
            ..Default::default()
        };
        let (mesh, events, _) = checked(None, || try_tessellate(body, &options));
        let n = mesh.triangles.len();
        let work = |op| {
            events
                .iter()
                .filter(|e| e.operation == op)
                .map(|e| e.work)
                .sum::<usize>()
        };
        let tess_ms = median_ms(None, || try_tessellate(body, &options));
        let prepare = || checked(None, || tessellate_for_measurement(body, &options)).0;
        let prepared = [prepare()];
        let mass_ms = median_ms((n > 0).then_some("mesh_mass_properties"), || {
            mass_properties(&prepared)
        });
        let wall_ms = if n <= 50000 {
            median_ms((n > 0).then_some("wall_thickness"), || {
                wall_minimum(&mesh, (n / 2000).max(1))
            })
        } else {
            f64::NAN
        };
        let batch_ms = if n <= 130000 {
            let batch: Vec<_> = (0..16).map(|_| prepare()).collect();
            median_ms((n > 0).then_some("mesh_mass_properties"), || {
                mass_properties(&batch)
            })
        } else {
            f64::NAN
        };
        println!(
            "{sag},{n},{},{},{tess_ms:.4},{mass_ms:.4},{wall_ms:.4},{batch_ms:.4}",
            work("nurbs_curvature"),
            work("nurbs_interior_points")
        );
    }
}
