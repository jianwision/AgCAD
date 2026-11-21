//! Offscreen renders for vision feedback (`study.render`; spec 07 § Diagnostics). Z-buffer
//! rasterizer with shading or face-id false colour; legend maps colours to (untrusted) face
//! names. PNG written with stored (uncompressed) deflate blocks — valid zlib, no dependency.
//! CPU reference; the GPU path rasterizes in-kernel (OptiX / Tier P rasterizer, P11).

use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::Mesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View { Iso, Top, Front, Right }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode { Shaded, FaceIds }

fn basis(v: View) -> (V3, V3, V3) {
    match v {
        View::Top => (V3::X, V3::Y, V3::Z),
        View::Front => (V3::X, V3::Z, -V3::Y),
        View::Right => (V3::Y, V3::Z, V3::X),
        View::Iso => { let d = v3(1.0, -1.0, 1.0).normalized(); let r = V3::Z.cross(d).normalized(); let u = d.cross(r); (r, u, d) }
    }
}

pub fn face_color(id: u32) -> [u8; 3] {
    let h = id.wrapping_mul(2654435761);
    [(h >> 16) as u8 | 0x40, (h >> 8) as u8 | 0x40, h as u8 | 0x40]
}

/// Rasterize to RGB.
pub fn rasterize(m: &Mesh, view: View, mode: Mode, w: usize, h: usize) -> Vec<u8> {
    let (r, u, d) = basis(view);
    let proj: Vec<(f64, f64, f64)> = m.positions.iter().map(|p| (p.dot(r), p.dot(u), p.dot(d))).collect();
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in &proj { minx = minx.min(p.0); maxx = maxx.max(p.0); miny = miny.min(p.1); maxy = maxy.max(p.1); }
    let scale = 0.9 * (w as f64 / (maxx - minx).max(1e-9)).min(h as f64 / (maxy - miny).max(1e-9));
    let (cx, cy) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
    let sx = |p: (f64, f64, f64)| (w as f64 / 2.0 + (p.0 - cx) * scale, h as f64 / 2.0 - (p.1 - cy) * scale, p.2);
    let mut img = vec![255u8; w * h * 3];
    let mut z = vec![f64::NEG_INFINITY; w * h];
    let light = v3(0.3, 0.5, 0.8).normalized();
    for (ti, t) in m.triangles.iter().enumerate() {
        let (a, b, c) = (sx(proj[t[0] as usize]), sx(proj[t[1] as usize]), sx(proj[t[2] as usize]));
        let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area.abs() < 1e-12 { continue; }
        let n3 = (m.positions[t[1] as usize] - m.positions[t[0] as usize]).cross(m.positions[t[2] as usize] - m.positions[t[0] as usize]).normalized();
        let color = match mode {
            Mode::FaceIds => face_color(m.face_of.get(ti).copied().unwrap_or(0)),
            Mode::Shaded => { let s = (0.25 + 0.75 * n3.dot(light).abs()).min(1.0); [(150.0 * s) as u8, (170.0 * s) as u8, (200.0 * s) as u8] }
        };
        let (x0, x1) = (a.0.min(b.0).min(c.0).floor().max(0.0) as usize, (a.0.max(b.0).max(c.0).ceil() as usize).min(w - 1));
        let (y0, y1) = (a.1.min(b.1).min(c.1).floor().max(0.0) as usize, (a.1.max(b.1).max(c.1).ceil() as usize).min(h - 1));
        for y in y0..=y1 { for x in x0..=x1 {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let w0 = ((b.0 - px) * (c.1 - py) - (b.1 - py) * (c.0 - px)) / area;
            let w1 = ((c.0 - px) * (a.1 - py) - (c.1 - py) * (a.0 - px)) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }
            let depth = -(w0 * a.2 + w1 * b.2 + w2 * c.2);
            let i = y * w + x;
            if depth > z[i] { z[i] = depth; img[i * 3..i * 3 + 3].copy_from_slice(&color); }
        }}
    }
    img
}

/// Minimal PNG (RGB8, stored deflate blocks).
pub fn png(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    fn chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = Vec::with_capacity(4 + data.len());
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&agcad_geo::zip::crc32(&c).to_be_bytes());
    }
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for y in 0..h { raw.push(0); raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]); }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw { a = (a + x as u32) % 65521; b = (b + a) % 65521; }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}
