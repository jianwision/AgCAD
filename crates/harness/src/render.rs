//! Offscreen renders for vision feedback (`study.render`; spec 07 § Diagnostics). Z-buffer
//! rasterizer with shading or face-id false colour; legend maps colours to (untrusted) face
//! names. PNG written with stored (uncompressed) deflate blocks — valid zlib, no dependency.
//! CPU reference; the GPU path rasterizes in-kernel (OptiX / Tier P rasterizer, P11).

use agcad_geo::math::{v3, V3};
use agcad_geo::tessellate::Mesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View { Iso, Top, Front, Right }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode { Shaded, FaceIds, Heatmap }

/// Perceptually uniform engineering heatmap colormap (`t ∈ [0, 1]`: blue → cyan → green → yellow → red).
pub fn heatmap_color(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

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
            Mode::Heatmap => {
                // Default heatmap channel: overhang / undercut indicator from z-normal (1 = downward facing)
                let t = ((0.5 - 0.5 * n3.z) as f32).clamp(0.0, 1.0);
                let c = heatmap_color(t);
                let s = (0.35 + 0.65 * n3.dot(light).abs()).min(1.0) as f32;
                [(c[0] as f32 * s) as u8, (c[1] as f32 * s) as u8, (c[2] as f32 * s) as u8]
            }
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderParams {
    pub tris: u64,
    pub out_rgb: u64,
    pub n_tris: u32,
    pub w: u32,
    pub h: u32,
    pub _pad: u32,
}

/// CPU reference twin of `ag_rasterize` in `cu/render.cu` (pixel-parallel, race-free).
pub unsafe fn ag_rasterize_cpu(params: *const u8, _: &agcad_driver::LaunchConfig) {
    let p: RenderParams = unsafe { core::ptr::read_unaligned(params as *const RenderParams) };
    let tris = unsafe { core::slice::from_raw_parts(p.tris as *const f32, p.n_tris as usize * 12) };
    let img = unsafe { core::slice::from_raw_parts_mut(p.out_rgb as *mut u8, p.w as usize * p.h as usize * 3) };
    for y in 0..p.h {
        for x in 0..p.w {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let mut best_z = -1.0e300_f64;
            let (mut cr, mut cg, mut cb) = (255u8, 255u8, 255u8);
            for t in 0..p.n_tris as usize {
                let d = &tris[t * 12..t * 12 + 12];
                let (ax, ay, az) = (d[0] as f64, d[1] as f64, d[2] as f64);
                let (bx, by, bz) = (d[3] as f64, d[4] as f64, d[5] as f64);
                let (cx, cy, cz) = (d[6] as f64, d[7] as f64, d[8] as f64);
                let minx = ax.min(bx).min(cx);
                let maxx = ax.max(bx).max(cx);
                if px < minx || px > maxx { continue; }
                let miny = ay.min(by).min(cy);
                let maxy = ay.max(by).max(cy);
                if py < miny || py > maxy { continue; }
                let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
                if area.abs() < 1e-12 { continue; }
                let w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
                let w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }
                let depth = -(w0 * az + w1 * bz + w2 * cz);
                if depth > best_z {
                    best_z = depth;
                    cr = d[9] as u8;
                    cg = d[10] as u8;
                    cb = d[11] as u8;
                }
            }
            let idx = (y as usize * p.w as usize + x as usize) * 3;
            img[idx] = cr;
            img[idx + 1] = cg;
            img[idx + 2] = cb;
        }
    }
}

pub const RENDER_CPU: &[(&str, agcad_driver::CpuKernelFn)] = &[("ag_rasterize", ag_rasterize_cpu)];

pub fn render_library() -> agcad_driver::LibrarySource {
    agcad_driver::LibrarySource {
        name: "ag_render",
        cuda_source: Some(agcad_kernels_cuda::bundle(include_str!("../cu/render.cu"), &[])),
        cpu_kernels: RENDER_CPU,
        options: vec![],
    }
}

/// Pack mesh triangles in screen space (`[ax,ay,az, bx,by,bz, cx,cy,cz, r,g,b]`) for `ag_rasterize`.
pub fn pack_screen_tris(m: &Mesh, view: View, mode: Mode, w: usize, h: usize) -> Vec<f32> {
    if m.positions.is_empty() || m.triangles.is_empty() { return Vec::new(); }
    let (r, u, d) = basis(view);
    let proj: Vec<(f64, f64, f64)> = m.positions.iter().map(|p| (p.dot(r), p.dot(u), p.dot(d))).collect();
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in &proj { minx = minx.min(p.0); maxx = maxx.max(p.0); miny = miny.min(p.1); maxy = maxy.max(p.1); }
    let scale = 0.9 * (w as f64 / (maxx - minx).max(1e-9)).min(h as f64 / (maxy - miny).max(1e-9));
    let (cx, cy) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
    let sx = |p: (f64, f64, f64)| (w as f64 / 2.0 + (p.0 - cx) * scale, h as f64 / 2.0 - (p.1 - cy) * scale, p.2);
    let light = v3(0.3, 0.5, 0.8).normalized();
    let mut out = Vec::with_capacity(m.triangles.len() * 12);
    for (ti, t) in m.triangles.iter().enumerate() {
        let (a, b, c) = (sx(proj[t[0] as usize]), sx(proj[t[1] as usize]), sx(proj[t[2] as usize]));
        let n3 = (m.positions[t[1] as usize] - m.positions[t[0] as usize]).cross(m.positions[t[2] as usize] - m.positions[t[0] as usize]).normalized();
        let color = match mode {
            Mode::FaceIds => face_color(m.face_of.get(ti).copied().unwrap_or(0)),
            Mode::Shaded => { let s = (0.25 + 0.75 * n3.dot(light).abs()).min(1.0); [(150.0 * s) as u8, (170.0 * s) as u8, (200.0 * s) as u8] }
            Mode::Heatmap => {
                // Default heatmap channel: overhang / undercut indicator from z-normal (1 = downward facing)
                let t = ((0.5 - 0.5 * n3.z) as f32).clamp(0.0, 1.0);
                let c = heatmap_color(t);
                let s = (0.35 + 0.65 * n3.dot(light).abs()).min(1.0) as f32;
                [(c[0] as f32 * s) as u8, (c[1] as f32 * s) as u8, (c[2] as f32 * s) as u8]
            }
        };
        out.extend([a.0 as f32, a.1 as f32, a.2 as f32, b.0 as f32, b.1 as f32, b.2 as f32, c.0 as f32, c.1 as f32, c.2 as f32, color[0] as f32, color[1] as f32, color[2] as f32]);
    }
    out
}

/// Rasterize on a device (CUDA or CPU reference) into an RGB8 image.
pub fn rasterize_device<D: agcad_driver::KernelExt>(dev: &D, s: &D::Stream, m: &Mesh, view: View, mode: Mode, w: usize, h: usize) -> agcad_driver::DriverResult<Vec<u8>> {
    use agcad_driver::*;
    let lib = dev.load_library(&render_library())?;
    let tris = pack_screen_tris(m, view, mode, w, h);
    let n_tris = (tris.len() / 12) as u32;
    let tb = dev.alloc_bytes(s, (tris.len() * 4).max(16))?;
    if !tris.is_empty() { dev.upload_from_host_async(s, bytemuck::cast_slice(&tris), &tb, 0)?; }
    let img_bytes = w * h * 3;
    let ob = dev.alloc_bytes(s, img_bytes.max(16))?;
    let prm = agcad_kernels::upload_params(dev, s, &RenderParams { tris: tb.device_ptr(), out_rgb: ob.device_ptr(), n_tris, w: w as u32, h: h as u32, _pad: 0 })?;
    dev.launch(s, lib, "ag_rasterize", &LaunchConfig::linear((w * h) as u64, 256), &prm.buffer)?;
    let mut pin = dev.pinned(img_bytes.max(1))?;
    dev.download_packed_async(s, &[Region { buf: &ob, offset: 0, bytes: img_bytes }], &mut pin)?;
    dev.sync(s)?;
    Ok(pin.as_bytes()[..img_bytes].to_vec())
}

/// Rasterize an arbitrary per-triangle scalar field (e.g. von Mises stress, wall thickness,
/// undercut severity) using [`heatmap_color`] over `[min_v, max_v]`.
pub fn rasterize_heatmap(m: &Mesh, view: View, tri_values: &[f32], min_v: f32, max_v: f32, w: usize, h: usize) -> Vec<u8> {
    if m.positions.is_empty() || m.triangles.is_empty() { return vec![255u8; w * h * 3]; }
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
    let inv_span = 1.0 / (max_v - min_v).abs().max(1e-9);
    for (ti, t) in m.triangles.iter().enumerate() {
        let (a, b, c) = (sx(proj[t[0] as usize]), sx(proj[t[1] as usize]), sx(proj[t[2] as usize]));
        let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area.abs() < 1e-12 { continue; }
        let n3 = (m.positions[t[1] as usize] - m.positions[t[0] as usize]).cross(m.positions[t[2] as usize] - m.positions[t[0] as usize]).normalized();
        let val = tri_values.get(ti).copied().unwrap_or(min_v);
        let col = heatmap_color((val - min_v) * inv_span);
        let shade = (0.35 + 0.65 * n3.dot(light).abs()).min(1.0) as f32;
        let color = [(col[0] as f32 * shade) as u8, (col[1] as f32 * shade) as u8, (col[2] as f32 * shade) as u8];
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

/// Stitch 4 quadrant views (Iso, Top, Front, Right) into a single `2*tile_w × 2*tile_h` contact sheet.
fn stitch_quadrants(quads: [&[u8]; 4], tw: usize, th: usize) -> Vec<u8> {
    let (w, h) = (tw * 2, th * 2);
    let mut out = vec![220u8; w * h * 3];
    let offs = [(0, 0), (tw, 0), (0, th), (tw, th)];
    for (q, (ox, oy)) in quads.iter().zip(offs) {
        for y in 0..th {
            let src = &q[y * tw * 3..(y + 1) * tw * 3];
            let dst_off = ((oy + y) * w + ox) * 3;
            out[dst_off..dst_off + tw * 3].copy_from_slice(src);
        }
    }
    // 1-px dark gray divider cross
    for x in 0..w { let idx = (th * w + x) * 3; out[idx..idx + 3].copy_from_slice(&[80, 80, 80]); }
    for y in 0..h { let idx = (y * w + tw) * 3; out[idx..idx + 3].copy_from_slice(&[80, 80, 80]); }
    out
}

/// 2×2 multi-view contact sheet (Iso / Top / Front / Right) on host.
pub fn contact_sheet(m: &Mesh, mode: Mode, tile_w: usize, tile_h: usize) -> Vec<u8> {
    let q0 = rasterize(m, View::Iso, mode, tile_w, tile_h);
    let q1 = rasterize(m, View::Top, mode, tile_w, tile_h);
    let q2 = rasterize(m, View::Front, mode, tile_w, tile_h);
    let q3 = rasterize(m, View::Right, mode, tile_w, tile_h);
    stitch_quadrants([&q0, &q1, &q2, &q3], tile_w, tile_h)
}

/// 2×2 multi-view contact sheet (Iso / Top / Front / Right) rasterized on device.
pub fn contact_sheet_device<D: agcad_driver::KernelExt>(
    dev: &D,
    s: &D::Stream,
    m: &Mesh,
    mode: Mode,
    tile_w: usize,
    tile_h: usize,
) -> agcad_driver::DriverResult<Vec<u8>> {
    let q0 = rasterize_device(dev, s, m, View::Iso, mode, tile_w, tile_h)?;
    let q1 = rasterize_device(dev, s, m, View::Top, mode, tile_w, tile_h)?;
    let q2 = rasterize_device(dev, s, m, View::Front, mode, tile_w, tile_h)?;
    let q3 = rasterize_device(dev, s, m, View::Right, mode, tile_w, tile_h)?;
    Ok(stitch_quadrants([&q0, &q1, &q2, &q3], tile_w, tile_h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_driver::*;

    #[test]
    fn device_rasterizer_matches_host_reference() {
        let b = agcad_geo::layered::LayeredSolid::extrusion(agcad_geo::math::Frame::WORLD, &agcad_geo::builders::Profile::rectangle(0.0, 0.0, 20.0, 15.0), 0.0, 10.0).to_brep();
        let m = agcad_geo::tessellate::tessellate(&b, &Default::default());
        let host = rasterize(&m, View::Iso, Mode::Shaded, 64, 64);
        let dev = agcad_driver_cpu::CpuDevice::new();
        let s = dev.stream(Priority::Normal).unwrap();
        let gpu = rasterize_device(&*dev, &s, &m, View::Iso, Mode::Shaded, 64, 64).unwrap();
        let diff = host.iter().zip(&gpu).filter(|(a, b)| a != b).count();
        assert!(diff < 64 * 3, "device rasterizer differed on {diff} channels (f32 edge rounding only)");
    }
}
