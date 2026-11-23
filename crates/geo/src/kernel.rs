//! `ExactKernel` trait (AG-02-006): the native GPU-oriented kernel is the only production
//! implementation; `agcad-geo-occt` implements it as a test oracle.

use crate::brep::{Brep, ValidityReport};
use crate::layered::LayeredSolid;
use crate::math::Frame;
use crate::region2d::BoolOp2;
use crate::tessellate::{Mesh, TessOptions};
use agcad_driver::AgStatus;

/// A solid value as produced by features. Keeps its constructive representation when one is
/// known (layered extrusion), so later booleans can stay on the exact fast path.
#[derive(Clone, Debug)]
pub enum Solid {
    Layered(LayeredSolid),
    Brep(Brep),
}

impl Solid {
    pub fn brep(&self) -> Brep {
        match self { Solid::Layered(l) => l.to_brep(), Solid::Brep(b) => b.clone() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelError {
    pub status: AgStatus,
    pub message: String,
}

pub trait ExactKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn boolean(&self, a: &Solid, b: &Solid, op: BoolOp2) -> Result<Solid, KernelError>;
    fn check(&self, s: &Solid) -> ValidityReport;
    fn tessellate(&self, s: &Solid, o: &TessOptions) -> Mesh;
}

pub struct NativeKernel;

fn same_frame(a: &Frame, b: &Frame) -> bool {
    a.origin.dist(b.origin) < 1e-9 && a.z.dist(b.z) < 1e-12 && a.x.dist(b.x) < 1e-12
}

impl ExactKernel for NativeKernel {
    fn name(&self) -> &'static str { "agcad-native" }

    fn boolean(&self, a: &Solid, b: &Solid, op: BoolOp2) -> Result<Solid, KernelError> {
        match (a, b) {
            (Solid::Layered(la), Solid::Layered(lb)) if same_frame(&la.frame, &lb.frame) => {
                if let Some(p) = la.boolean_contacts(lb).first() {
                    let w = la.frame.to_world(crate::math::v3(p[0], p[1], p[2]));
                    return Err(KernelError { status: AgStatus::TopologyFailed, message: format!(
                        "non-manifold contact: the operands touch tangentially at ({:.6}, {:.6}, {:.6}) mm; move or resize one feature by more than 1e-6 mm", w.x, w.y, w.z) });
                }
                let r = la.boolean(lb, op);
                if r.layers.is_empty() && op != BoolOp2::Intersection {
                    return Err(KernelError { status: AgStatus::TopologyFailed, message: "boolean produced an empty body".into() });
                }
                if let Some(p) = r.tangent_contacts().first() {
                    let w = la.frame.to_world(crate::math::v3(p[0], p[1], p[2]));
                    return Err(KernelError { status: AgStatus::TopologyFailed, message: format!(
                        "non-manifold result: boundaries touch tangentially at ({:.6}, {:.6}, {:.6}) mm (zero-thickness contact); move or resize one feature by more than 1e-6 mm", w.x, w.y, w.z) });
                }
                Ok(Solid::Layered(r))
            }
            (Solid::Layered(la), Solid::Layered(lb)) if la.frame.z.cross(lb.frame.z).norm() < 1e-12 => {
                // Parallel extrusion directions: re-express b in a's frame (2D transform).
                let d = lb.frame.origin - la.frame.origin;
                let dz = d.dot(la.frame.z) * la.frame.z.dot(lb.frame.z).signum();
                let rebased = rebase(lb, &la.frame, dz);
                match rebased {
                    Some(r) => self.boolean(a, &Solid::Layered(r), op),
                    None => Err(KernelError { status: AgStatus::TopologyFailed, message: "could not rebase layered operand".into() }),
                }
            }
            _ => {
                // Phase 3: general 3D polyhedral boolean fallback (non-parallel planar B-reps)
                let ba = a.brep();
                let bb = b.brep();
                match crate::gpu_exact::boolean_general_polyhedral(&ba, &bb, op) {
                    Ok(brep) => Ok(Solid::Brep(brep)),
                    Err(msg) => Err(KernelError { status: AgStatus::TopologyFailed, message: msg }),
                }
            }
        }
    }

    fn check(&self, s: &Solid) -> ValidityReport { s.brep().validate() }

    fn tessellate(&self, s: &Solid, o: &TessOptions) -> Mesh { crate::tessellate::tessellate(&s.brep(), o) }
}

/// Express a layered solid in another frame with parallel z (2D rigid transform of profiles).
fn rebase(l: &LayeredSolid, target: &Frame, dz: f64) -> Option<LayeredSolid> {
    use crate::builders::{Profile, Seg2};
    let flip = l.frame.z.dot(target.z) < 0.0;
    let map = |p: [f64; 2]| -> [f64; 2] {
        let w = l.frame.to_world(crate::math::v3(p[0], p[1], 0.0));
        let q = target.to_local(w);
        [q.x, q.y]
    };
    let ang_off = { let x = l.frame.x; let lx = crate::math::v3(x.dot(target.x), x.dot(target.y), 0.0); lx.y.atan2(lx.x) };
    let map_seg = |s: &Seg2| -> Seg2 {
        match *s {
            Seg2::Line { a, b } => Seg2::Line { a: map(a), b: map(b) },
            Seg2::Arc { c, r, a0, a1 } => if flip { Seg2::Arc { c: map(c), r, a0: ang_off - a0, a1: ang_off - a1 } } else { Seg2::Arc { c: map(c), r, a0: a0 + ang_off, a1: a1 + ang_off } },
            Seg2::Circle { c, r, ccw } => Seg2::Circle { c: map(c), r, ccw: if flip { !ccw } else { ccw } },
        }
    };
    let layers = l.layers.iter().map(|ly| {
        let (z0, z1) = if flip { (dz - ly.z1, dz - ly.z0) } else { (dz + ly.z0, dz + ly.z1) };
        crate::layered::Layer { z0, z1, regions: ly.regions.iter().map(|r| Profile { loops: r.loops.iter().map(|lp| lp.iter().map(map_seg).collect()).collect() }.normalize()).collect() }
    }).collect();
    Some(LayeredSolid { frame: *target, layers })
}
