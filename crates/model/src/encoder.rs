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

/// Phase 4: Spatial boundary-cell geometry encoder (`SPATIAL_LEN = 32`).
/// Encodes 3D boundary cells directly from `dops` / NanoVDB terminal cells (no B-rep needed):
/// - `[0..3]` normalized centroid
/// - `[3..6]` principal coordinate std-dev (x, y, z)
/// - `[6..14]` 8-octant boundary-cell occupancy fractions (3D spatial awareness)
/// - `[14..20]` mean & abs-mean surface normal moments (`nx, ny, nz, |nx|, |ny|, |nz|`)
/// - `[20..24]` thin-wall hotspot `(x, y, z, min_wall_norm)`
/// - `[24..28]` high-stress / undercut hotspot `(x, y, z, max_severity_norm)`
/// - `[28..32]` area density, cell count log, aspect ratios
pub const SPATIAL_LEN: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundaryCellSample {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub area: f32,
    /// Sampled wall thickness (mm); `< 0` if unchecked.
    pub wall: f32,
    /// Local severity (e.g., von Mises stress or undercut indicator >= 0).
    pub severity: f32,
}

pub fn encode_boundary_cells(
    cells: &[BoundaryCellSample],
    domain_lo: [f32; 3],
    domain_hi: [f32; 3],
) -> [f32; SPATIAL_LEN] {
    let mut out = [0.0f32; SPATIAL_LEN];
    if cells.is_empty() { return out; }
    let ext = [
        (domain_hi[0] - domain_lo[0]).max(1e-6),
        (domain_hi[1] - domain_lo[1]).max(1e-6),
        (domain_hi[2] - domain_lo[2]).max(1e-6),
    ];
    let norm_p = |p: [f32; 3]| -> [f32; 3] {
        [
            ((p[0] - domain_lo[0]) / ext[0] * 2.0 - 1.0).clamp(-1.0, 1.0),
            ((p[1] - domain_lo[1]) / ext[1] * 2.0 - 1.0).clamp(-1.0, 1.0),
            ((p[2] - domain_lo[2]) / ext[2] * 2.0 - 1.0).clamp(-1.0, 1.0),
        ]
    };
    let mut w_sum = 0.0f32;
    let mut c = [0.0f32; 3];
    for b in cells {
        let w = b.area.max(1e-6);
        let q = norm_p(b.pos);
        for k in 0..3 { c[k] += q[k] * w; }
        w_sum += w;
    }
    let inv_w = 1.0 / w_sum.max(1e-9);
    for k in 0..3 { c[k] *= inv_w; out[k] = c[k]; }

    let mut var = [0.0f32; 3];
    let mut oct = [0.0f32; 8];
    let mut n_mean = [0.0f32; 3];
    let mut n_abs = [0.0f32; 3];
    let mut min_wall = f32::INFINITY;
    let mut thin_p = [0.0f32; 3];
    let mut max_sev = 0.0f32;
    let mut sev_p = [0.0f32; 3];
    for b in cells {
        let w = b.area.max(1e-6);
        let q = norm_p(b.pos);
        for k in 0..3 {
            let d = q[k] - c[k];
            var[k] += d * d * w;
            n_mean[k] += b.normal[k] * w;
            n_abs[k] += b.normal[k].abs() * w;
        }
        let oi = ((q[0] >= c[0]) as usize) | (((q[1] >= c[1]) as usize) << 1) | (((q[2] >= c[2]) as usize) << 2);
        oct[oi] += w;
        if b.wall >= 0.0 && b.wall < min_wall {
            min_wall = b.wall;
            thin_p = q;
        }
        if b.severity > max_sev {
            max_sev = b.severity;
            sev_p = q;
        }
    }
    for k in 0..3 { out[3 + k] = (var[k] * inv_w).sqrt(); }
    for o in 0..8 { out[6 + o] = oct[o] * inv_w; }
    for k in 0..3 {
        out[14 + k] = n_mean[k] * inv_w;
        out[17 + k] = n_abs[k] * inv_w;
    }
    if min_wall.is_finite() {
        out[20] = thin_p[0]; out[21] = thin_p[1]; out[22] = thin_p[2];
        out[23] = (min_wall / ext[0].max(ext[1]).max(ext[2])).clamp(0.0, 1.0);
    }
    if max_sev > 0.0 {
        out[24] = sev_p[0]; out[25] = sev_p[1]; out[26] = sev_p[2];
        out[27] = max_sev.ln_1p();
    }
    let scale = ext[0].max(ext[1]).max(ext[2]);
    out[28] = (w_sum / (scale * scale)).ln_1p();
    out[29] = (cells.len() as f32).ln_1p();
    out[30] = ext[1] / scale;
    out[31] = ext[2] / scale;
    out
}
