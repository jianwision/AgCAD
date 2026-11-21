use crate::Result;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Debug)]
pub struct Stats {
    pub triangles: usize,
    pub vertices: usize,
    pub boundary_edges: usize,
    pub nonmanifold_edges: usize,
    pub orientation_errors: usize,
    pub degenerate_triangles: usize,
    pub euler: i64,
    pub volume: f64,
    pub extents: [f64; 3],
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn point(triangle: &[f32], vertex: usize) -> [f64; 3] {
    std::array::from_fn(|i| f64::from(triangle[3 * vertex + i]))
}

pub fn analyze(data: &[f32]) -> Result<Stats> {
    if data.is_empty() || !data.len().is_multiple_of(9) || data.iter().any(|x| !x.is_finite()) {
        return Err("Mesh contains invalid coordinates or incomplete triangles".into());
    }
    let mut vertices = HashMap::<[u32; 3], u32>::new();
    let mut edges = HashMap::<(u32, u32), (u32, i32)>::new();
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    let mut volume = 0.0;
    let mut degenerate_triangles = 0;
    for triangle in data.chunks_exact(9) {
        let points = [point(triangle, 0), point(triangle, 1), point(triangle, 2)];
        let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        if dot(normal, normal) == 0.0 {
            degenerate_triangles += 1;
        }
        volume += dot(points[0], cross(points[1], points[2])) / 6.0;
        let mut ids = [0_u32; 3];
        for (j, p) in points.iter().enumerate() {
            for i in 0..3 {
                minimum[i] = minimum[i].min(p[i]);
                maximum[i] = maximum[i].max(p[i]);
            }
            // Normalize -0, but otherwise use exact coordinates, without tolerance welding.
            let key = std::array::from_fn(|i| {
                if p[i] == 0.0 {
                    0
                } else {
                    (p[i] as f32).to_bits()
                }
            });
            let next_id = vertices.len() as u32;
            ids[j] = *vertices.entry(key).or_insert(next_id);
        }
        for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let edge = edges.entry((a.min(b), a.max(b))).or_default();
            edge.0 += 1;
            edge.1 += if a < b { 1 } else { -1 };
        }
    }
    Ok(Stats {
        triangles: data.len() / 9,
        vertices: vertices.len(),
        boundary_edges: edges.values().filter(|(count, _)| *count == 1).count(),
        nonmanifold_edges: edges.values().filter(|(count, _)| *count > 2).count(),
        orientation_errors: edges
            .values()
            .filter(|(count, balance)| *count == 2 && *balance != 0)
            .count(),
        degenerate_triangles,
        euler: vertices.len() as i64 - edges.len() as i64 + (data.len() / 9) as i64,
        volume,
        extents: std::array::from_fn(|i| maximum[i] - minimum[i]),
    })
}

pub fn write_stl(path: &Path, data: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    let mut header = [0_u8; 80];
    let label = b"Rust + CUDA implicit modeling; coordinates in mm";
    header[..label.len()].copy_from_slice(label);
    file.write_all(&header)?;
    file.write_all(&(data.len() as u32 / 9).to_le_bytes())?;
    for triangle in data.chunks_exact(9) {
        let normal = cross(
            sub(point(triangle, 1), point(triangle, 0)),
            sub(point(triangle, 2), point(triangle, 0)),
        );
        let length = dot(normal, normal).sqrt();
        for coordinate in normal {
            file.write_all(&((coordinate / length) as f32).to_le_bytes())?;
        }
        for coordinate in triangle {
            file.write_all(&coordinate.to_le_bytes())?;
        }
        file.write_all(&0_u16.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn base64(data: &[u8]) -> String {
    const DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(data.len().div_ceil(3) * 4);
    for group in data.chunks(3) {
        let value = (u32::from(group[0]) << 16)
            | (u32::from(*group.get(1).unwrap_or(&0)) << 8)
            | u32::from(*group.get(2).unwrap_or(&0));
        output.push(DIGITS[(value >> 18) as usize] as char);
        output.push(DIGITS[((value >> 12) & 63) as usize] as char);
        output.push(if group.len() > 1 {
            DIGITS[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if group.len() > 2 {
            DIGITS[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

pub fn write_viewer(path: &Path, data: &[f32], radius: f32) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    let html = include_str!("../viewer.html")
        .replace("__RADIUS__", &radius.to_string())
        .replace("__TRIANGLES__", &(data.len() / 9).to_string())
        .replace("__MESH__", &base64(&bytes));
    std::fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> Vec<f32> {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
            .into_iter()
            .flat_map(|face| face.into_iter().flat_map(|i| points[i]))
            .collect()
    }

    #[test]
    fn closed_mesh_has_correct_orientation_volume_and_topology() {
        let stats = analyze(&tetrahedron()).unwrap();
        assert_eq!((stats.vertices, stats.triangles, stats.euler), (4, 4, 2));
        assert_eq!(
            (
                stats.boundary_edges,
                stats.nonmanifold_edges,
                stats.orientation_errors
            ),
            (0, 0, 0)
        );
        assert!((stats.volume - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(stats.extents, [1.0; 3]);
    }

    #[test]
    fn detects_missing_and_reversed_faces() {
        let mut data = tetrahedron();
        assert_eq!(analyze(&data[..27]).unwrap().boundary_edges, 3);
        for i in 0..3 {
            data.swap(i + 3, i + 6);
        }
        assert_eq!(analyze(&data).unwrap().orientation_errors, 3);
    }

    #[test]
    fn nonfinite_mesh_is_rejected_before_export() {
        let mut data = tetrahedron();
        data[0] = f32::NAN;
        assert!(analyze(&data).is_err());
    }
}
