#[cfg(windows)]
mod cuda_loader;
mod gpu;
mod mesh;

use std::{error::Error, fs, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
pub struct Options {
    pub radius: f32,
    pub hole: f32,
    pub grid: u32,
    pub width: u32,
    pub out: PathBuf,
    pub model: PathBuf,
    pub check: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            radius: 30.0,
            hole: 8.0,
            grid: 128,
            width: 960,
            out: "output".into(),
            model: default_model_path(),
            check: false,
        }
    }
}

fn default_model_path() -> PathBuf {
    let relative = PathBuf::from("cuda/model.cuh");
    if relative.is_file() {
        return relative;
    }
    if let Ok(executable) = std::env::current_exe() {
        for ancestor in executable.ancestors().skip(1) {
            let candidate = ancestor.join("cuda/model.cuh");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    relative
}

impl Options {
    fn parse() -> Result<Option<Self>> {
        let mut options = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                println!(
                    "Rust + CUDA implicit modeling demo\n\n\
                    cargo run --release -- [options]\n\n\
                    --radius MM    Sphere radius (default: 30)\n\
                    --hole MM      Through-hole radius, along Z (default: 8)\n\
                    --grid N       Cells per axis, 24..256 (default: 128)\n\
                    --width PX     Square preview size, 128..2048 (default: 960)\n\
                    --model FILE   CUDA model body with float model(V3, float, float)\n\
                    --out DIR      Output directory (default: output)\n\
                    --check        Enforce analytic checks for the default model\n\n\
                    Outputs: model.stl, preview.png, viewer.html, report.json, model.ptx\n\
                    All geometry coordinates use millimetres. There is no CPU fallback."
                );
                return Ok(None);
            }
            if arg == "--check" {
                options.check = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("Missing value for {arg}"))?;
            match arg.as_str() {
                "--radius" => options.radius = value.parse()?,
                "--hole" => options.hole = value.parse()?,
                "--grid" => options.grid = value.parse()?,
                "--width" => options.width = value.parse()?,
                "--out" => options.out = value.into(),
                "--model" => options.model = value.into(),
                _ => return Err(format!("Unknown option: {arg}; use --help").into()),
            }
        }
        options.validate()?;
        Ok(Some(options))
    }

    fn validate(&self) -> Result<()> {
        if !self.radius.is_finite() || !(0.1..=10_000.0).contains(&self.radius) {
            return Err("--radius must be finite and between 0.1 and 10000 mm".into());
        }
        if !self.hole.is_finite() || self.hole <= 0.0 || self.hole >= self.radius {
            return Err("--hole must be finite, positive and smaller than --radius".into());
        }
        if !(24..=256).contains(&self.grid) || !(128..=2048).contains(&self.width) {
            return Err("--grid must be 24..256; --width must be 128..2048".into());
        }
        if self.check && fs::read_to_string(&self.model)? != include_str!("../cuda/model.cuh") {
            return Err("--check uses sphere-minus-cylinder analytic expectations; remove it for custom model code".into());
        }
        Ok(())
    }
}

fn run() -> Result<()> {
    let Some(options) = Options::parse()? else {
        return Ok(());
    };
    let total_start = Instant::now();
    fs::create_dir_all(&options.out)?;
    let result = gpu::execute(&options)?;
    let validation_start = Instant::now();
    let stats = mesh::analyze(&result.triangles)?;
    let r = f64::from(options.radius);
    let a = f64::from(options.hole);
    let half_height = (r * r - a * a).sqrt();
    // Independent napkin-ring volume formula; do not reuse CUDA field logic.
    let expected_volume = 4.0 * std::f64::consts::PI / 3.0 * half_height.powi(3);
    let volume_error = (stats.volume / expected_volume - 1.0).abs();
    let expected_extents = [2.0 * r, 2.0 * r, 2.0 * half_height];
    let max_extent_error = (0..3)
        .map(|i| (stats.extents[i] - expected_extents[i]).abs())
        .fold(0.0_f64, f64::max);
    let step = f64::from(result.step);
    if stats.boundary_edges > 0
        || stats.nonmanifold_edges > 0
        || stats.orientation_errors > 0
        || stats.degenerate_triangles > 0
        || stats.volume <= 0.0
    {
        return Err(format!(
            "Mesh validation failed: {stats:?}. Increase --grid or adjust the model."
        )
        .into());
    }
    if result.boundary_inside > 0 {
        return Err(
            "Model intersects the sampling boundary; enlarge the domain in src/gpu.rs".into(),
        );
    }
    if options.check
        && (volume_error > 0.02
            || max_extent_error > step * 2.0
            || stats.euler != 0
            || result.probe_error > 2e-4 * r)
    {
        return Err(format!("Analytic check failed: volume error={:.3}%, extent error={max_extent_error:.6} mm, Euler={}, probe error={} mm", volume_error * 100.0, stats.euler, result.probe_error).into());
    }
    let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
    let export_start = Instant::now();
    mesh::write_stl(&options.out.join("model.stl"), &result.triangles)?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(fs::File::create(options.out.join("preview.png"))?),
        options.width,
        options.width,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&result.pixels)?;
    mesh::write_viewer(
        &options.out.join("viewer.html"),
        &result.triangles,
        options.radius,
    )?;
    fs::write(options.out.join("model.ptx"), &result.ptx)?;
    fs::write(options.out.join("model-source.cu"), &result.source)?;
    let export_ms = export_start.elapsed().as_secs_f64() * 1000.0;
    let report = format!(
        concat!(
            "{{\n  \"backend\": \"CUDA + NVRTC (no CPU fallback)\",\n",
            "  \"device\": \"{}\",\n  \"units\": \"mm\",\n",
            "  \"radius_mm\": {},\n  \"hole_radius_mm\": {},\n  \"grid_cells_per_axis\": {},\n",
            "  \"grid_step_mm\": {},\n  \"triangles\": {},\n  \"vertices\": {},\n",
            "  \"boundary_edges\": {},\n  \"nonmanifold_edges\": {},\n  \"orientation_errors\": {},\n",
            "  \"degenerate_triangles\": {},\n  \"euler_characteristic\": {},\n",
            "  \"boundary_inside_samples\": {},\n  \"volume_mm3\": {},\n",
            "  \"extents_mm\": {:?},\n  \"analytic_checks_enabled\": {},\n",
            "  \"sphere_model_expected_volume_mm3\": {},\n  \"sphere_model_volume_relative_error\": {},\n",
            "  \"sphere_model_max_extent_error_mm\": {},\n  \"sphere_model_probe_max_error_mm\": {},\n",
            "  \"transfers\": {{\"application_d2h_copy_calls\": 1, \"intermediate_host_waits\": 0,\n",
            "    \"download_bytes\": {}, \"unused_triangle_capacity_bytes\": {}, \"resident_field_bytes_not_downloaded\": {} }},\n",
            "  \"timing_ms\": {{\n    \"nvrtc_compile\": {},\n    \"gpu_sampling_event\": {},\n",
            "    \"gpu_meshing_event\": {},\n    \"gpu_render_event\": {},\n    \"download_event\": {},\n",
            "    \"cpu_validation\": {},\n    \"cpu_export\": {},\n    \"total_wall\": {}\n  }}\n}}\n"
        ),
        result.device,
        options.radius,
        options.hole,
        options.grid,
        result.step,
        stats.triangles,
        stats.vertices,
        stats.boundary_edges,
        stats.nonmanifold_edges,
        stats.orientation_errors,
        stats.degenerate_triangles,
        stats.euler,
        result.boundary_inside,
        stats.volume,
        stats.extents,
        options.check,
        if options.check {
            expected_volume.to_string()
        } else {
            "null".into()
        },
        if options.check {
            volume_error.to_string()
        } else {
            "null".into()
        },
        if options.check {
            max_extent_error.to_string()
        } else {
            "null".into()
        },
        if options.check {
            result.probe_error.to_string()
        } else {
            "null".into()
        },
        result.download_bytes,
        result.padding_bytes,
        result.field_bytes_not_downloaded,
        result.compile_ms,
        result.sampling_ms,
        result.meshing_ms,
        result.render_ms,
        result.download_ms,
        validation_ms,
        export_ms,
        total_start.elapsed().as_secs_f64() * 1000.0
    );
    fs::write(options.out.join("report.json"), report)?;
    println!(
        "GPU: {}\nMesh: {} triangles, {} vertices, Euler={}\nBoundary edges: {}; nonmanifold: {}; orientation errors: {}\nVolume: {:.3} mm^3; extents: {:.3?} mm",
        result.device,
        stats.triangles,
        stats.vertices,
        stats.euler,
        stats.boundary_edges,
        stats.nonmanifold_edges,
        stats.orientation_errors,
        stats.volume,
        stats.extents
    );
    if options.check {
        println!(
            "Analytic checks PASS: volume error {:.4}%; max extent error {:.5} mm; probe error {:.6} mm",
            volume_error * 100.0,
            max_extent_error,
            result.probe_error
        );
    }
    println!(
        "GPU stages (CUDA events): sampling {:.2} ms; meshing {:.2} ms; rendering {:.2} ms\nReadback: 1 packed copy, {:.2} MiB; no intermediate host waits\nOutputs: {}",
        result.sampling_ms,
        result.meshing_ms,
        result.render_ms,
        result.download_bytes as f64 / 1048576.0,
        fs::canonicalize(&options.out)?.display()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_or_unbounded_allocations() {
        for options in [
            Options {
                hole: 31.0,
                ..Default::default()
            },
            Options {
                radius: f32::NAN,
                ..Default::default()
            },
            Options {
                grid: u32::MAX,
                ..Default::default()
            },
            Options {
                width: 0,
                ..Default::default()
            },
        ] {
            assert!(options.validate().is_err());
        }
    }
}
