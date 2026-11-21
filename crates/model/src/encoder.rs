//! Geometry encoder (spec 07 / review "more" item 8). Baseline deterministic descriptor
//! (surface-type histogram, bbox extents, area/volume ratios, face-adjacency degree stats)
//! used for observations and `query.similar` until the learned UV-graph + SDF encoder
//! (Candle-trained on reconstruction tasks) replaces it behind the same interface.

use agcad_geo::brep::Brep;

pub const DESCRIPTOR_LEN: usize = 16;

pub fn describe(b: &Brep, volume: f64, area: f64) -> [f32; DESCRIPTOR_LEN] {
    let mut d = [0f32; DESCRIPTOR_LEN];
    for f in &b.faces {
        let k = match b.surfaces[f.surface as usize].kind_name() { "plane" => 0, "cylinder" => 1, "cone" => 2, "sphere" => 3, "torus" => 4, "nurbs" => 5, _ => 6 };
        d[k] += 1.0;
    }
    let nf = b.faces.len().max(1) as f32;
    for x in d.iter_mut().take(7) { *x /= nf; }
    let e = b.bbox().extent();
    let mut ext = [e.x, e.y, e.z];
    ext.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let scale = ext[0].max(1e-9);
    d[7] = (ext[1] / scale) as f32;
    d[8] = (ext[2] / scale) as f32;
    d[9] = (volume / (e.x * e.y * e.z).max(1e-12)) as f32;
    d[10] = (area / (scale * scale)) as f32;
    d[11] = (b.faces.len() as f32).ln_1p();
    d[12] = (b.edges.len() as f32).ln_1p();
    d[13] = b.loops.iter().filter(|l| !l.outer).count() as f32 / nf;
    d[14] = (scale as f32).ln_1p();
    d[15] = (b.validate().euler as f32) / 2.0;
    d
}

/// Cosine similarity of two descriptors (used by `query.similar`).
pub fn similarity(a: &[f32; DESCRIPTOR_LEN], b: &[f32; DESCRIPTOR_LEN]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}
