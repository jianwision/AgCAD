//! Sparse narrow-band implicit / voxel storage in a NanoVDB-compatible 8³ leaf layout
//! (spec 02 § Lightweight representations; spec 05 § Memory; review F5), plus
//! feature-preserving Dual Contouring extraction.
//!
//! Layout: pointer-free flat leaf array (`Leaf8`: 8×8×8 f32 values + 64-byte active bitmask +
//! origin coordinate) indexed by a deterministic sorted leaf coordinate table so the exact
//! same buffer layout maps cleanly to GPU device memory (CUDA / PNanoVDB).

use crate::math::{v3, V3};
use crate::tessellate::Mesh;
use std::collections::BTreeMap;

/// Magic header for serialized AgCAD NanoVDB-layout sparse SDF grids.
pub const NANOVDB_MAGIC: [u8; 8] = *b"AGNVDB01";
pub const LEAF_DIM: usize = 8;
pub const LEAF_VOXELS: usize = LEAF_DIM * LEAF_DIM * LEAF_DIM; // 512

#[derive(Clone, Debug, PartialEq)]
pub struct Leaf8 {
    /// Leaf origin in index space (multiple of 8 on each axis).
    pub origin: [i32; 3],
    /// Active voxel bitmask (512 bits = 8 × u64).
    pub mask: [u64; 8],
    /// Narrow-band signed distance values (mm), indexed `((x*8)+y)*8+z`.
    pub values: [f32; LEAF_VOXELS],
}

impl Leaf8 {
    pub fn new(origin: [i32; 3], background: f32) -> Self {
        Leaf8 { origin, mask: [0; 8], values: [background; LEAF_VOXELS] }
    }

    #[inline]
    pub fn idx(lx: usize, ly: usize, lz: usize) -> usize {
        (lx * LEAF_DIM + ly) * LEAF_DIM + lz
    }

    #[inline]
    pub fn set(&mut self, lx: usize, ly: usize, lz: usize, v: f32) {
        let i = Self::idx(lx, ly, lz);
        self.values[i] = v;
        self.mask[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    pub fn get(&self, lx: usize, ly: usize, lz: usize) -> f32 {
        self.values[Self::idx(lx, ly, lz)]
    }

    #[inline]
    pub fn is_active(&self, lx: usize, ly: usize, lz: usize) -> bool {
        let i = Self::idx(lx, ly, lz);
        (self.mask[i >> 6] >> (i & 63)) & 1 != 0
    }
}

/// Sparse narrow-band SDF grid with deterministic 8³ leaf bricks.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoVdbGrid {
    pub voxel_size: f64,
    pub background: f32,
    pub origin_world: [f64; 3],
    pub leaves: Vec<Leaf8>,
    /// Deterministic sorted map: leaf key `(ox, oy, oz)` → index in `leaves`.
    index: BTreeMap<[i32; 3], usize>,
}

impl NanoVdbGrid {
    pub fn new(origin_world: [f64; 3], voxel_size: f64, background: f32) -> Self {
        NanoVdbGrid {
            voxel_size,
            background,
            origin_world,
            leaves: Vec::new(),
            index: BTreeMap::new(),
        }
    }

    /// Build a sparse narrow-band grid over `[lo, hi]` keeping only 8³ bricks within
    /// `band_voxels * voxel_size` of the zero-level set.
    pub fn from_sdf(
        lo: [f64; 3],
        hi: [f64; 3],
        voxel_size: f64,
        band_voxels: f64,
        sdf: impl Fn(f64, f64, f64) -> f32,
    ) -> Self {
        let half_band = (band_voxels * voxel_size) as f32;
        let bg = half_band.max((voxel_size * 4.0) as f32);
        let mut grid = Self::new(lo, voxel_size, bg);
        let dims = [0, 1, 2].map(|k| (((hi[k] - lo[k]) / voxel_size).ceil() as i32).max(1) + 2);
        let blocks = dims.map(|n| (n + LEAF_DIM as i32 - 1) / LEAF_DIM as i32);
        let brick_diag = (0.5 * (LEAF_DIM as f64) * 3.0_f64.sqrt() * voxel_size) as f32;

        for bx in 0..blocks[0] {
            for by in 0..blocks[1] {
                for bz in 0..blocks[2] {
                    let ox = bx * LEAF_DIM as i32;
                    let oy = by * LEAF_DIM as i32;
                    let oz = bz * LEAF_DIM as i32;
                    // Coarse Lipschitz brick-center cull before populating 512 voxels
                    let cx = lo[0] + (ox as f64 + 3.5) * voxel_size;
                    let cy = lo[1] + (oy as f64 + 3.5) * voxel_size;
                    let cz = lo[2] + (oz as f64 + 3.5) * voxel_size;
                    let fc = sdf(cx, cy, cz);
                    if fc.abs() > brick_diag + half_band {
                        continue;
                    }
                    let mut leaf = Leaf8::new([ox, oy, oz], if fc < 0.0 { -bg } else { bg });
                    let mut any_near = false;
                    for lx in 0..LEAF_DIM {
                        let x = lo[0] + (ox + lx as i32) as f64 * voxel_size;
                        for ly in 0..LEAF_DIM {
                            let y = lo[1] + (oy + ly as i32) as f64 * voxel_size;
                            for lz in 0..LEAF_DIM {
                                let z = lo[2] + (oz + lz as i32) as f64 * voxel_size;
                                let v = sdf(x, y, z);
                                leaf.set(lx, ly, lz, v);
                                if v.abs() <= half_band {
                                    any_near = true;
                                }
                            }
                        }
                    }
                    if any_near {
                        let id = grid.leaves.len();
                        grid.index.insert(leaf.origin, id);
                        grid.leaves.push(leaf);
                    }
                }
            }
        }
        grid
    }

    pub fn leaf_count(&self) -> usize { self.leaves.len() }

    pub fn active_voxel_count(&self) -> usize {
        self.leaves.iter().map(|l| l.mask.iter().map(|m| m.count_ones() as usize).sum::<usize>()).sum()
    }

    /// Sample exact voxel at integer grid index `(ix, iy, iz)`.
    pub fn sample_index(&self, ix: i32, iy: i32, iz: i32) -> f32 {
        let ox = ix.div_euclid(LEAF_DIM as i32) * LEAF_DIM as i32;
        let oy = iy.div_euclid(LEAF_DIM as i32) * LEAF_DIM as i32;
        let oz = iz.div_euclid(LEAF_DIM as i32) * LEAF_DIM as i32;
        match self.index.get(&[ox, oy, oz]) {
            Some(&li) => self.leaves[li].get((ix - ox) as usize, (iy - oy) as usize, (iz - oz) as usize),
            None => self.background,
        }
    }

    /// Trilinear sample in world coordinates.
    pub fn sample_world(&self, p: V3) -> f32 {
        let gx = (p.x - self.origin_world[0]) / self.voxel_size;
        let gy = (p.y - self.origin_world[1]) / self.voxel_size;
        let gz = (p.z - self.origin_world[2]) / self.voxel_size;
        let (ix, iy, iz) = (gx.floor() as i32, gy.floor() as i32, gz.floor() as i32);
        let (tx, ty, tz) = ((gx - ix as f64) as f32, (gy - iy as f64) as f32, (gz - iz as f64) as f32);
        let mut c = [0.0f32; 8];
        for k in 0..8 {
            c[k] = self.sample_index(ix + (k & 1) as i32, iy + ((k >> 1) & 1) as i32, iz + ((k >> 2) & 1) as i32);
        }
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(c[0], c[1], tx);
        let c10 = lerp(c[2], c[3], tx);
        let c01 = lerp(c[4], c[5], tx);
        let c11 = lerp(c[6], c[7], tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    }

    /// Deterministic binary serialization for GPU upload / `.vdb` cache persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.leaves.len() * (12 + 64 + LEAF_VOXELS * 4));
        out.extend_from_slice(&NANOVDB_MAGIC);
        out.extend_from_slice(&self.voxel_size.to_le_bytes());
        out.extend_from_slice(&self.background.to_le_bytes());
        out.extend_from_slice(&(self.leaves.len() as u32).to_le_bytes());
        for o in self.origin_world { out.extend_from_slice(&o.to_le_bytes()); }
        for leaf in &self.leaves {
            for c in leaf.origin { out.extend_from_slice(&c.to_le_bytes()); }
            for m in leaf.mask { out.extend_from_slice(&m.to_le_bytes()); }
            for v in leaf.values { out.extend_from_slice(&v.to_le_bytes()); }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 48 || bytes[..8] != NANOVDB_MAGIC {
            return Err("invalid NanoVdbGrid header".into());
        }
        let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let i32_at = |o: usize| i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let voxel_size = f64_at(8);
        let background = f32_at(16);
        let n_leaves = u32_at(20) as usize;
        let origin_world = [f64_at(24), f64_at(32), f64_at(40)];
        let stride = 12 + 64 + LEAF_VOXELS * 4;
        if bytes.len() != 48 + n_leaves * stride {
            return Err("truncated NanoVdbGrid payload".into());
        }
        let mut grid = Self::new(origin_world, voxel_size, background);
        let mut off = 48;
        for id in 0..n_leaves {
            let origin = [i32_at(off), i32_at(off + 4), i32_at(off + 8)];
            off += 12;
            let mut mask = [0u64; 8];
            for m in &mut mask { *m = u64_at(off); off += 8; }
            let mut values = [0.0f32; LEAF_VOXELS];
            for v in &mut values { *v = f32_at(off); off += 4; }
            grid.index.insert(origin, id);
            grid.leaves.push(Leaf8 { origin, mask, values });
        }
        Ok(grid)
    }

    /// Feature-preserving Dual Contouring over active narrow-band cells: places one vertex
    /// per sign-changing cell by solving the regularized Hermite QEF (normal equations +
    /// mass-point Tikhonov regularization, clamped to the cell), preserving sharp edges and
    /// corners while staying watertight on sign-changing edges.
    pub fn dual_contour(&self) -> Mesh {
        let h = self.voxel_size;
        let pos = |ix: i32, iy: i32, iz: i32| {
            v3(self.origin_world[0] + ix as f64 * h, self.origin_world[1] + iy as f64 * h, self.origin_world[2] + iz as f64 * h)
        };
        let grad = |p: V3| -> V3 {
            let d = 0.5 * h;
            let gx = (self.sample_world(p + v3(d, 0.0, 0.0)) - self.sample_world(p - v3(d, 0.0, 0.0))) as f64;
            let gy = (self.sample_world(p + v3(0.0, d, 0.0)) - self.sample_world(p - v3(0.0, d, 0.0))) as f64;
            let gz = (self.sample_world(p + v3(0.0, 0.0, d)) - self.sample_world(p - v3(0.0, 0.0, d))) as f64;
            let g = v3(gx, gy, gz);
            let n = g.norm();
            if n > 1e-12 { g * (1.0 / n) } else { v3(0.0, 0.0, 1.0) }
        };

        // 12 edges of a unit cube as corner pairs (bit 0=x, 1=y, 2=z)
        const EDGES: [(usize, usize); 12] = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        let mut cell_vert: BTreeMap<[i32; 3], u32> = BTreeMap::new();
        let mut mesh = Mesh { positions: Vec::new(), triangles: Vec::new(), face_of: Vec::new() };

        // Step 1: place one QEF-optimal vertex in every sign-changing active cell.
        for leaf in &self.leaves {
            for lx in 0..LEAF_DIM {
                for ly in 0..LEAF_DIM {
                    for lz in 0..LEAF_DIM {
                        let ix = leaf.origin[0] + lx as i32;
                        let iy = leaf.origin[1] + ly as i32;
                        let iz = leaf.origin[2] + lz as i32;
                        let mut vals = [0.0f32; 8];
                        let mut any_in = false;
                        let mut any_out = false;
                        for c in 0..8 {
                            let v = self.sample_index(ix + (c & 1) as i32, iy + ((c >> 1) & 1) as i32, iz + ((c >> 2) & 1) as i32);
                            vals[c] = v;
                            if v < 0.0 { any_in = true; } else { any_out = true; }
                        }
                        if !(any_in && any_out) { continue; }
                        let mut ata = [[0.0f64; 3]; 3];
                        let mut atb = [0.0f64; 3];
                        let mut mass = V3::ZERO;
                        let mut n_cross: f64 = 0.0;
                        let cmin = pos(ix, iy, iz);
                        let cmax = pos(ix + 1, iy + 1, iz + 1);
                        for &(a, b) in &EDGES {
                            if (vals[a] < 0.0) == (vals[b] < 0.0) { continue; }
                            let pa = pos(ix + (a & 1) as i32, iy + ((a >> 1) & 1) as i32, iz + ((a >> 2) & 1) as i32);
                            let pb = pos(ix + (b & 1) as i32, iy + ((b >> 1) & 1) as i32, iz + ((b >> 2) & 1) as i32);
                            let t = (vals[a] as f64 / (vals[a] as f64 - vals[b] as f64)).clamp(0.0, 1.0);
                            let p = pa + (pb - pa) * t;
                            let n = grad(p);
                            let nv = [n.x, n.y, n.z];
                            let d = n.dot(p - cmin);
                            for r in 0..3 {
                                for c in 0..3 { ata[r][c] += nv[r] * nv[c]; }
                                atb[r] += nv[r] * d;
                            }
                            mass = mass + (p - cmin);
                            n_cross += 1.0;
                        }
                        let mp = mass * (1.0 / n_cross.max(1.0));
                        // Regularize toward mass point (keeps sharp features while preventing singular solves)
                        let reg = 1e-2;
                        let mpa = [mp.x, mp.y, mp.z];
                        for r in 0..3 { ata[r][r] += reg; atb[r] += reg * mpa[r]; }
                        let rel = solve3x3(ata, atb).unwrap_or(mpa);
                        let vtx = v3(
                            (cmin.x + rel[0]).clamp(cmin.x, cmax.x),
                            (cmin.y + rel[1]).clamp(cmin.y, cmax.y),
                            (cmin.z + rel[2]).clamp(cmin.z, cmax.z),
                        );
                        let id = mesh.positions.len() as u32;
                        mesh.positions.push(vtx);
                        cell_vert.insert([ix, iy, iz], id);
                    }
                }
            }
        }

        // Step 2: emit one oriented quad (2 triangles) per sign-changing grid edge whose 4
        // incident cells all have vertices.
        let mut emit_quad = |cells: [[i32; 3]; 4], flip: bool| {
            let Some(&v0) = cell_vert.get(&cells[0]) else { return };
            let Some(&v1) = cell_vert.get(&cells[1]) else { return };
            let Some(&v2) = cell_vert.get(&cells[2]) else { return };
            let Some(&v3) = cell_vert.get(&cells[3]) else { return };
            if flip {
                mesh.triangles.push([v0, v3, v2]);
                mesh.triangles.push([v0, v2, v1]);
            } else {
                mesh.triangles.push([v0, v1, v2]);
                mesh.triangles.push([v0, v2, v3]);
            }
            mesh.face_of.push(0);
            mesh.face_of.push(0);
        };

        let keys: Vec<[i32; 3]> = cell_vert.keys().copied().collect();
        for [ix, iy, iz] in keys {
            let f0 = self.sample_index(ix, iy, iz) < 0.0;
            // X-edge (ix,iy,iz) -> (ix+1,iy,iz) shared by 4 cells around (iy-1..iy, iz-1..iz)
            let fx = self.sample_index(ix + 1, iy, iz) < 0.0;
            if f0 != fx {
                emit_quad([[ix, iy - 1, iz - 1], [ix, iy, iz - 1], [ix, iy, iz], [ix, iy - 1, iz]], !f0);
            }
            // Y-edge (ix,iy,iz) -> (ix,iy+1,iz)
            let fy = self.sample_index(ix, iy + 1, iz) < 0.0;
            if f0 != fy {
                emit_quad([[ix - 1, iy, iz - 1], [ix - 1, iy, iz], [ix, iy, iz], [ix, iy, iz - 1]], !f0);
            }
            // Z-edge (ix,iy,iz) -> (ix,iy,iz+1)
            let fz = self.sample_index(ix, iy, iz + 1) < 0.0;
            if f0 != fz {
                emit_quad([[ix - 1, iy - 1, iz], [ix, iy - 1, iz], [ix, iy, iz], [ix - 1, iy, iz]], !f0);
            }
        }
        mesh
    }
}

fn solve3x3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-18 { return None; }
    let cramer = |col: usize| {
        let mut a = m;
        for r in 0..3 { a[r][col] = b[r]; }
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    Some([cramer(0) / det, cramer(1) / det, cramer(2) / det])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nanovdb_sparse_roundtrip_and_dual_contour_box() {
        // 20 × 16 × 12 box centered at (10, 8, 6): sharp edges preserved by dual contouring
        let sdf = |x: f64, y: f64, z: f64| -> f32 {
            let qx = (x - 10.0).abs() - 10.0;
            let qy = (y - 8.0).abs() - 8.0;
            let qz = (z - 6.0).abs() - 6.0;
            let out = qx.max(0.0).hypot(qy.max(0.0)).hypot(qz.max(0.0));
            (out + qx.max(qy).max(qz).min(0.0)) as f32
        };
        let grid = NanoVdbGrid::from_sdf([-4.0, -4.0, -4.0], [24.0, 20.0, 16.0], 1.0, 2.5, sdf);
        assert!(grid.leaf_count() > 0 && grid.active_voxel_count() > 0);
        // Binary round-trip
        let raw = grid.to_bytes();
        let grid2 = NanoVdbGrid::from_bytes(&raw).unwrap();
        assert_eq!(grid, grid2);
        // Trilinear sample accuracy
        assert!((grid.sample_world(v3(10.0, 8.0, 12.0))).abs() < 1e-4);
        // Dual contouring reproduces sharp box volume accurately and watertight
        let m = grid.dual_contour();
        assert!(!m.triangles.is_empty());
        assert!(m.is_watertight(), "dual-contoured box must be watertight");
        let (vol, _, _) = m.mass_properties();
        let exact = 20.0 * 16.0 * 12.0;
        assert!((vol - exact).abs() / exact < 0.02, "dual contour box vol {vol} vs exact {exact}");
    }
}
