//! Exact boundaries of simultaneous treatments of an orthogonal block.
use crate::{
    Brep, Frame, V3,
    brep::{BodyKind, DEFAULT_LINEAR_TOL as TOL},
    builders::{Profile, prism},
    curve::Curve,
    surface::Surface,
};

pub struct Block {
    origin: V3,
    axes: [V3; 3],
    length: [f64; 3],
}
fn side(c: usize, a: usize) -> usize {
    (c >> a) & 1
}
fn face(c: usize, a: usize) -> usize {
    2 * a + side(c, a)
}
fn edge(c: usize, a: usize) -> usize {
    4 * a + (c & ((1 << a) - 1)) + ((c >> (a + 1)) << a)
}
impl Block {
    pub fn recognize(b: &Brep) -> Result<Self, String> {
        let fail = "Select a rectangular block with six planar faces";
        if b.vertices.len() != 8
            || b.faces.len() != 6
            || b.edges.len() != 12
            || b.shells.len() != 1
            || !b.shells[0].closed
            || b.faces.iter().any(|f| {
                !matches!(b.surfaces[f.surface as usize], Surface::Plane { .. })
                    || f.loops.len() != 1
                    || b.loops[f.loops[0] as usize].coedges.len() != 4
            })
            || b.edges
                .iter()
                .any(|e| !matches!(b.curves[e.curve as usize], Curve::Line { .. }))
        {
            return Err(fail.into());
        }
        let origin = b.vertices[0].p;
        let dirs: Vec<_> = b
            .edges
            .iter()
            .filter_map(|e| {
                if e.v0 == 0 {
                    Some(e.v1)
                } else if e.v1 == 0 {
                    Some(e.v0)
                } else {
                    None
                }
            })
            .map(|i| b.vertices[i as usize].p - origin)
            .collect();
        let mut dirs: [V3; 3] = dirs.try_into().map_err(|_| fail)?;
        if dirs[0].cross(dirs[1]).dot(dirs[2]) < 0. {
            dirs.swap(0, 1);
        }
        let axes = dirs.map(V3::normalized);
        if (0..3).any(|a| axes[a].dot(axes[(a + 1) % 3]).abs() > 1e-9 || dirs[a].norm() <= TOL) {
            return Err(fail.into());
        }
        let block = Self {
            origin,
            axes,
            length: dirs.map(V3::norm),
        };
        let mut mask = 0;
        for v in &b.vertices {
            mask |= 1u16 << block.vertex(v.p)?;
        }
        if mask != 255 {
            return Err(fail.into());
        }
        Ok(block)
    }
    fn corner(&self, c: usize) -> V3 {
        (0..3).fold(self.origin, |p, a| {
            p + self.axes[a] * (side(c, a) as f64 * self.length[a])
        })
    }
    fn inward(&self, c: usize, a: usize) -> V3 {
        self.axes[a] * (1. - 2. * side(c, a) as f64)
    }
    pub fn vertex(&self, p: V3) -> Result<usize, String> {
        (0..8)
            .find(|&c| self.corner(c).dist(p) <= TOL)
            .ok_or_else(|| "The selected corner changed. Pick it again.".into())
    }
    pub fn edge(&self, a: V3, b: V3) -> Result<usize, String> {
        let (i, j) = (self.vertex(a)?, self.vertex(b)?);
        let diff = i ^ j;
        if diff.count_ones() != 1 {
            return Err("The selected edge changed. Pick it again.".into());
        }
        Ok(edge(i, diff.trailing_zeros() as usize))
    }
    pub fn plane(&self, point: V3, normal: V3) -> Result<usize, String> {
        (0..6)
            .find(|&f| {
                let a = f / 2;
                let n = self.axes[a] * (2. * (f % 2) as f64 - 1.);
                n.dot(normal) > 1. - 1e-9 && n.dot(point - self.corner((f % 2) << a)).abs() < TOL
            })
            .ok_or_else(|| "Selected face is no longer a block side".into())
    }
    pub fn offset(&self, faces: u8, d: f64) -> Result<Brep, String> {
        let mut origin = self.origin;
        let mut span = self.length;
        if !d.is_finite() || d == 0. || faces == 0 {
            return Err("Choose faces and a nonzero offset".into());
        }
        for (a, s) in span.iter_mut().enumerate() {
            if faces & (1 << (2 * a)) != 0 {
                origin = origin - self.axes[a] * d;
                *s += d;
            }
            if faces & (2 << (2 * a)) != 0 {
                *s += d;
            }
            if *s <= TOL {
                return Err("Offset collapses the distance between opposite faces".into());
            }
        }
        Ok(prism(
            &Profile::rectangle(0., 0., span[0], span[1]),
            &Frame {
                origin,
                x: self.axes[0],
                y: self.axes[1],
                z: self.axes[2],
            },
            span[2],
        ))
    }
    pub fn treat(&self, edges: u16, vertices: u8, d: f64, round: bool) -> Result<Brep, String> {
        if !d.is_finite()
            || d <= TOL
            || (edges == 0 && vertices == 0)
            || (edges != 0 && vertices != 0)
        {
            return Err("Select block edges or vertices and supply a positive setback".into());
        }
        let on = |c, a| edges & (1u16 << edge(c, a)) != 0;
        let bevel = |c: usize| vertices & (1u8 << c) != 0;
        let mut shell = Shell::new();
        let mut patches = [0usize; 12];
        let mut ends = [0usize; 8];
        for a in 0..3 {
            for s in 0..2 {
                let c = s << a;
                let n = -self.inward(c, a);
                shell.plane(self.corner(c), n, format!("side.{}", face(c, a)));
            }
        }
        for c in 0..8 {
            for a in 0..3 {
                if side(c, a) != 0 || !on(c, a) {
                    continue;
                }
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                let (x, y) = (self.inward(c, u), self.inward(c, v));
                let p = self.corner(c);
                let id = edge(c, a);
                patches[id] = if round {
                    let center = p + (x + y) * d;
                    shell.surface(
                        Surface::Cylinder {
                            frame: Frame::new(center, self.axes[a], x),
                            r: d,
                        },
                        -(x + y).normalized(),
                        format!("fillet.{id}"),
                    )
                } else {
                    shell.plane(p + x * d, -(x + y).normalized(), format!("chamfer.{id}"))
                };
            }
        }
        for c in 0..8 {
            if bevel(c) || (0..3).all(|a| on(c, a)) {
                let diag = (0..3).fold(V3::ZERO, |s, a| s + self.inward(c, a));
                let p = self.corner(c);
                ends[c] = if bevel(c) {
                    shell.plane(
                        p + self.inward(c, 0) * d,
                        -diag.normalized(),
                        format!("bevel.{c}"),
                    )
                } else if round {
                    shell.surface(
                        Surface::Sphere {
                            frame: Frame::new(
                                p + diag * d,
                                (self.inward(c, 0) - self.inward(c, 1)).normalized(),
                                -diag.normalized(),
                            ),
                            r: d,
                        },
                        -diag.normalized(),
                        format!("corner.{c}"),
                    )
                } else {
                    shell.plane(
                        p + (diag - self.inward(c, 0)) * d,
                        -diag.normalized(),
                        format!("corner.{c}"),
                    )
                };
            }
        }
        for c in 0..8 {
            for a in 0..3 {
                if side(c, a) != 0 {
                    continue;
                }
                let z = c | (1 << a);
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                let id = edge(c, a);
                let mut longitudinal = |j: usize, k: usize, treated: bool| -> Result<(), String> {
                    let tip = |t| {
                        let trim = bevel(t) || on(t, k) || (!treated && on(t, j));
                        self.corner(t)
                            + self.inward(t, a) * (if trim { d } else { 0. })
                            + self.inward(t, k) * (if treated { d } else { 0. })
                    };
                    let (p, q) = (tip(c), tip(z));
                    if (q - p).dot(self.axes[a]) <= TOL {
                        return Err(format!("Setbacks consume block edge {id} at {d} mm"));
                    }
                    shell.line(
                        p,
                        q,
                        face(c, j),
                        if treated { patches[id] } else { face(c, k) },
                    );
                    Ok(())
                };
                longitudinal(u, v, on(c, a))?;
                if on(c, a) {
                    longitudinal(v, u, true)?;
                }
            }
        }
        for c in 0..8 {
            let p = self.corner(c);
            for a in 0..3 {
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                if bevel(c) {
                    shell.line(
                        p + self.inward(c, u) * d,
                        p + self.inward(c, v) * d,
                        face(c, a),
                        ends[c],
                    );
                    continue;
                }
                if !on(c, a) {
                    continue;
                }
                let other: Vec<_> = [u, v].into_iter().filter(|&i| on(c, i)).collect();
                let mate = match other.as_slice() {
                    [] => face(c, a),
                    [b] if *b < a => continue,
                    [b] => patches[edge(c, *b)],
                    _ => ends[c],
                };
                let tip = |k| {
                    p + self.inward(c, k) * d + self.inward(c, a) * (if on(c, k) { d } else { 0. })
                };
                let (start, end) = (tip(u), tip(v));
                if round {
                    let center = p
                        + (self.inward(c, u) + self.inward(c, v)) * d
                        + self.inward(c, a) * (if other.is_empty() { 0. } else { d });
                    shell.arc(start, end, center, patches[edge(c, a)], mate);
                } else {
                    shell.line(start, end, patches[edge(c, a)], mate);
                }
            }
        }
        shell.finish()
    }
}
struct Patch {
    surface: u32,
    normal: V3,
    name: String,
    edges: Vec<u32>,
}
struct Shell {
    b: Brep,
    patches: Vec<Patch>,
}
impl Shell {
    fn new() -> Self {
        let mut b = Brep::new(BodyKind::Solid);
        b.add_shell(true);
        Self {
            b,
            patches: Vec::new(),
        }
    }
    fn surface(&mut self, s: Surface, normal: V3, name: String) -> usize {
        let i = self.patches.len();
        let surface = self.b.add_surface(s);
        self.patches.push(Patch {
            surface,
            normal,
            name,
            edges: vec![],
        });
        i
    }
    fn plane(&mut self, p: V3, n: V3, name: String) -> usize {
        self.surface(
            Surface::Plane {
                frame: Frame::new(p, n, n.any_perp()),
            },
            n,
            name,
        )
    }
    fn vertex(&mut self, p: V3) -> u32 {
        self.b
            .vertices
            .iter()
            .position(|v| v.p.dist(p) < TOL)
            .map(|i| i as u32)
            .unwrap_or_else(|| self.b.add_vertex(p))
    }
    fn boundary(&mut self, p: V3, q: V3, curve: Curve, t: f64, a: usize, b: usize) {
        let v = self.vertex(p);
        let w = self.vertex(q);
        let c = self.b.add_curve(curve);
        let e = self
            .b
            .add_edge(c, v, w, 0., t, format!("edge.{}", self.b.edges.len()));
        self.patches[a].edges.push(e);
        self.patches[b].edges.push(e);
    }
    fn line(&mut self, p: V3, q: V3, a: usize, b: usize) {
        self.boundary(
            p,
            q,
            Curve::Line {
                origin: p,
                dir: q - p,
            },
            1.,
            a,
            b,
        );
    }
    fn arc(&mut self, mut p: V3, mut q: V3, o: V3, a: usize, b: usize) {
        if p.dist(o) < q.dist(o) {
            std::mem::swap(&mut p, &mut q);
        }
        let (x, y) = ((p - o).normalized(), (q - o).normalized());
        let frame = Frame::new(o, x.cross(y), x);
        let (major, minor) = (p.dist(o), q.dist(o));
        let curve = if (major - minor).abs() < 1e-10 * major {
            Curve::Circle { frame, r: major }
        } else {
            Curve::Ellipse {
                frame,
                a: major,
                b: minor,
            }
        };
        self.boundary(p, q, curve, std::f64::consts::FRAC_PI_2, a, b);
    }
    fn finish(mut self) -> Result<Brep, String> {
        for patch in self.patches {
            let mut edges = patch.edges;
            let first = edges.pop().ok_or("Missing patch boundary")?;
            let mut loop_ = vec![(first, true)];
            let mut end = self.b.edges[first as usize].v1;
            while !edges.is_empty() {
                let i = edges
                    .iter()
                    .position(|&e| {
                        let e = &self.b.edges[e as usize];
                        e.v0 == end || e.v1 == end
                    })
                    .ok_or("Disconnected patch boundary")?;
                let id = edges.swap_remove(i);
                let e = &self.b.edges[id as usize];
                let forward = e.v0 == end;
                end = if forward { e.v1 } else { e.v0 };
                loop_.push((id, forward));
            }
            if end != self.b.edges[first as usize].v0 {
                return Err("Open patch boundary".into());
            }
            let mut area = V3::ZERO;
            let origin = self.b.vertices[end as usize].p;
            for &(id, fwd) in &loop_ {
                let e = &self.b.edges[id as usize];
                let p = self.b.vertices[e.v0 as usize].p - origin;
                let q = self.b.vertices[e.v1 as usize].p - origin;
                let m = self.b.curves[e.curve as usize].point((e.t0 + e.t1) * 0.5) - origin;
                area = area + (p.cross(m) + m.cross(q)) * (if fwd { 1. } else { -1. });
            }
            if area.dot(patch.normal) < 0. {
                loop_.reverse();
                for (_, fwd) in &mut loop_ {
                    *fwd = !*fwd;
                }
            }
            self.b.add_face(
                patch.surface,
                true,
                0,
                &loop_,
                &[],
                patch.name,
                [0., std::f64::consts::TAU, 0., 1.],
            );
        }
        let check = self.b.validate();
        if !check.valid || !check.closed {
            return Err(format!("Invalid edited block: {:?}", check.problems));
        }
        Ok(self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn step_roundtrip_batch_edits(){
        use crate::{step_read::read_step,step_write::try_write_step,tessellate::{try_tessellate,TessOptions},math::{v3,Xform}};
        let (bodies,_)=read_step(include_str!("../../viewer/tests/fixtures/plate.step"),&Default::default()).unwrap();
        let native=prism(&Profile::rectangle(0.,0.,40.,28.),&Frame::WORLD,10.);
        let rotated=bodies[0].transformed(&Xform::rotation(v3(3.,5.,4.),v3(1.,2.,3.),0.7));
        for input in [&native,&bodies[0],&rotated]{
            let block=Block::recognize(input).unwrap();
            for (mask,vertices) in [(3,0),(0x11,0),(0x111,0),(4095,0),(0,129)] {
                for round in [false,true] {
                    if vertices!=0 && round{continue}
                    let b=block.treat(mask,vertices,1.,round).unwrap();
                    let step=try_write_step(&b,&Default::default()).unwrap().0;
                    let (reopen,_)=read_step(&step,&Default::default()).unwrap();assert_eq!(reopen.len(),1);
                    let o=TessOptions{sag:0.001,angle:0.07,..Default::default()};
                    let mut volumes=Vec::new();
                    for shape in [&b,&reopen[0]]{assert!(shape.validate().valid);let mesh=try_tessellate(shape,&o).unwrap();
                        assert!(mesh.is_watertight(),"mesh {mask:x} {vertices} {round}"); volumes.push(mesh.mass_properties().0);}
                    assert!((volumes[0]-volumes[1]).abs()<0.08,"{mask:x} {round} {volumes:?}");
                    if vertices!=0 {assert!((volumes[0]-(11200.-2./6.)).abs()<1e-5);}
                    if mask==4095 && round{let (a,b,c)=(38.,26.,8.);let pi=std::f64::consts::PI;let expected=a*b*c+2.*(a*b+a*c+b*c)+pi*(a+b+c)+4.*pi/3.;
                        assert!((volumes[0]-expected).abs()<0.35,"rounded volume {volumes:?}, expected {expected}");}
                }
            }
        }
    }
    #[test]
    fn every_edge_subset_and_corner_combination() {
        let block = Block {
            origin: V3::ZERO,
            axes: [V3::X, V3::Y, V3::Z],
            length: [40., 30., 10.],
        };
        for mask in 1..4096 {
            for round in [true, false] {
                block
                    .treat(mask, 0, 2., round)
                    .unwrap_or_else(|e| panic!("mask {mask:#x} round {round}: {e}"));
            }
        }
        for mask in 1..=255 {
            block.treat(0, mask, 1., false).unwrap();
        }
    }
}
