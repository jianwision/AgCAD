//! Project rectangular end sections and calculate equal-distance corner setbacks.
use crate::{
    accelerator::{Gpu, attempt},
    math::{Frame, V3},
};
use agcad_driver::{DriverResult, LaunchConfig, RawBuffer};

/// Previous, selected and following vertices of a cap, in world coordinates.
pub type CornerInput = [[f64; 3]; 3];
/// Corner, two tangent points, arc centre, adjacent lengths and angle cosine.
pub type CornerOutput = [f64; 11];

pub fn corners(jobs: &[CornerInput], frame: Frame, size: f64) -> DriverResult<Vec<CornerOutput>> {
    // Host-resident inputs measured faster on the host through 65,536 corners.
    // Keep CUDA available through the existing forced-backend setting.
    if let Some(points) = attempt("block_edge_corners", jobs.len(), None, || {
        let gpu = Gpu::open()?;
        let input = gpu.upload(jobs)?;
        let output = gpu.output::<CornerOutput>(jobs.len())?;
        let xyz = |v: V3| [v.x, v.y, v.z];
        let axes = gpu.upload(&[xyz(frame.origin), xyz(frame.x), xyz(frame.y)])?;
        gpu.launch(
            "block_edge_corners",
            LaunchConfig::linear(jobs.len() as u64, 128),
            &[
                input.device_ptr(),
                output.device_ptr(),
                axes.device_ptr(),
                size.to_bits(),
                jobs.len() as u64,
            ],
        )?;
        gpu.read(&output, jobs.len())
    })? {
        return Ok(points);
    }
    Ok(jobs
        .iter()
        .map(|job| {
            let [prev, p, next] = job.map(|[x, y, z]| frame.to_local(V3 { x, y, z }));
            let (u, v) = (prev - p, next - p);
            let (lu, lv) = (u.x.hypot(u.y), v.x.hypot(v.y));
            let (ux, uy, vx, vy) = (u.x / lu, u.y / lu, v.x / lv, v.y / lv);
            [
                p.x,
                p.y,
                p.x + size * ux,
                p.y + size * uy,
                p.x + size * vx,
                p.y + size * vy,
                p.x + size * (ux + vx),
                p.y + size * (uy + vy),
                lu,
                lv,
                ux * vx + uy * vy,
            ]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "geometry backend timing; run with AGCAD_GEOMETRY_BACKEND set"]
    fn benchmark_corner_batches() {
        for count in [4, 256, 4096, 65536] {
            let input = vec![[[0., 20., 0.], [0., 0., 0.], [30., 0., 0.]]; count];
            let (warm, events) = crate::accelerator::capture(|| corners(&input, Frame::WORLD, 2.));
            assert_eq!(
                warm.unwrap()[0],
                [0., 0., 0., 2., 2., 0., 2., 2., 20., 30., 0.]
            );
            let backend = events.last().unwrap().backend;
            if std::env::var("AGCAD_GEOMETRY_BACKEND").as_deref() == Ok("cuda") {
                assert_eq!(backend, "cuda", "{:?}", events.last().unwrap().error);
            }
            let now = std::time::Instant::now();
            for _ in 0..32 {
                std::hint::black_box(corners(&input, Frame::WORLD, 2.).unwrap());
            }
            eprintln!(
                "{backend} {count} corners: {:.1} us/batch",
                now.elapsed().as_secs_f64() * 1e6 / 32.
            );
        }
    }
}
