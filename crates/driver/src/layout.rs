//! Batched data layouts following NVIDIA library conventions (spec 05 § Batched data layout, A2-6).

use serde::{Deserialize, Serialize};

/// Array starts in device buffers are 128-byte aligned (coalescing, TMA eligibility).
pub const DEVICE_ALIGN: usize = 128;

pub const fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) / a * a
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn x(x: u32) -> Self { Dim3 { x, y: 1, z: 1 } }
    pub const fn new(x: u32, y: u32, z: u32) -> Self { Dim3 { x, y, z } }
    pub fn count(&self) -> u64 { self.x as u64 * self.y as u64 * self.z as u64 }
}

/// Logical shape of one batch element (row-major).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<u32>);

impl Shape {
    pub fn elems(&self) -> u64 { self.0.iter().map(|&d| d as u64).product() }
}

/// Layout of a buffer carrying a configuration (or entity) batch axis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchLayout {
    /// cuBLAS strided-batched: element `i` starts at `i * stride_elems`; `ld` = leading dimension.
    Strided { count: u32, stride_elems: u64, ld: u32 },
    /// CUB segmented: `count + 1` offsets, segment `i` = `[offsets[i], offsets[i+1])`.
    /// Offsets live on the device; the host keeps only the count.
    Segmented { count: u32, offsets_slot: u32 },
    /// cuBLAS grouped GEMM: groups of uniform shape.
    Grouped { groups: Vec<(u32, Shape)> },
}

impl BatchLayout {
    pub fn count(&self) -> u32 {
        match self {
            BatchLayout::Strided { count, .. } | BatchLayout::Segmented { count, .. } => *count,
            BatchLayout::Grouped { groups } => groups.iter().map(|g| g.0).sum(),
        }
    }

    /// Element count for strided/grouped layouts (segmented needs device offsets).
    pub fn total_elems(&self) -> Option<u64> {
        match self {
            BatchLayout::Strided { count, stride_elems, .. } => Some(*count as u64 * stride_elems),
            BatchLayout::Segmented { .. } => None,
            BatchLayout::Grouped { groups } => Some(groups.iter().map(|(c, s)| *c as u64 * s.elems()).sum()),
        }
    }
}

/// df64 values are stored as two SoA arrays (`hi[]`, `lo[]`), never interleaved (A2-6).
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(C)]
pub struct Df64 {
    pub hi: f32,
    pub lo: f32,
}

impl Df64 {
    pub fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        let lo = (v - hi as f64) as f32;
        Df64 { hi, lo }
    }
    pub fn to_f64(self) -> f64 { self.hi as f64 + self.lo as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn df64_roundtrip_precision() {
        let v = 1234.567_890_123_456_f64;
        let d = Df64::from_f64(v);
        assert!((d.to_f64() - v).abs() < 1e-9);
    }
    #[test]
    fn align() {
        assert_eq!(align_up(1, DEVICE_ALIGN), 128);
        assert_eq!(align_up(128, DEVICE_ALIGN), 128);
    }
}
