use super::*;
use crate::media::Highlight;
use agcad_geo::{Brep, curve::Curve};

struct PickEdge {
    name: String,
    curve: &'static str,
    path: Vec<V3>,
}

pub(super) struct BodyTopology {
    id: String,
    block: bool,
    vertices: Vec<V3>,
    edges: Vec<PickEdge>,
}

impl BodyTopology {
    pub(super) fn new(id: String, body: &Brep, mesh: &Mesh, paths: Vec<Vec<u32>>) -> Self {
        let edges = body
            .edges
            .iter()
            .zip(paths)
            .map(|(e, indices)| PickEdge {
                name: e.name.clone(),
                curve: match body.curves[e.curve as usize] {
                    Curve::Line { .. } => "Line",
                    Curve::Circle { .. } => "Circle",
                    Curve::Ellipse { .. } => "Ellipse",
                    Curve::Helix { .. } => "Helix",
                    Curve::Nurbs(_) => "NURBS",
                    Curve::Intersection { .. } => "Intersection",
                },
                path: indices
                    .into_iter()
                    .map(|i| mesh.positions[i as usize])
                    .collect(),
            })
            .collect();
        Self {
            id,
            block: agcad_geo::block_edit::Block::recognize(body).is_ok(),
            edges,
            vertices: body.vertices.iter().map(|v| v.p).collect(),
        }
    }
}

// x and y here use the displayed image height as their common unit.
struct Projection<'a> {
    scene: &'a Scene,
    frame: &'a Frame,
    right: V3,
    up: V3,
    depth: V3,
}
impl Projection<'_> {
    fn project(&self, p: V3) -> [f64; 2] {
        let p = (p - self.scene.center) * self.scene.radius.recip();
        [
            (1. + self.right.dot(p) + self.frame.rows[0][3] as f64) * 8. / 9.,
            (1. - self.up.dot(p) - self.frame.rows[1][3] as f64) * 0.5,
        ]
    }

    fn ray(&self, xy: [f64; 2]) -> Ray {
        let at = self.right
            * ((xy[0] * 9. / 8. - 1. - self.frame.rows[0][3] as f64) / self.right.dot(self.right))
            + self.up * ((1. - 2. * xy[1] - self.frame.rows[1][3] as f64) / self.up.dot(self.up));
        let origin = self.scene.center + (at - self.depth * 2.) * self.scene.radius;
        Ray {
            origin: f3(origin),
            tmin: 0.,
            dir: f3(self.depth),
            tmax: (4. * self.scene.radius) as f32,
        }
    }

    fn visible(&self, point: V3) -> bool {
        let ray = self.ray(self.project(point));
        let t = self.depth.dot(point - self.scene.center) + 2. * self.scene.radius;
        if !(0.0..=ray.tmax as f64).contains(&t) {
            return false;
        }
        // On silhouettes the f32 ray can fall just outside a front triangle.
        // Probe a subpixel neighbourhood so coincident rear edges stay occluded.
        let [x,y]=self.project(point);
        let eps=2e-7*self.up.norm();
        for (dx,dy) in [(0.,0.),(-eps,-eps),(-eps,eps),(eps,-eps),(eps,eps)] {
            let hit=self.scene.tracer.trace_one(&self.ray([x+dx,y+dy]));
            if hit.t>=0. && t>hit.t as f64+self.scene.radius*2e-5{return false;}
        }
        true
    }
}

struct Candidate {
    distance: f64,
    depth: f64,
    body: usize,
    index: usize,
    point: V3,
}

impl Scene {
    pub fn pick(
        &self,
        frame: &Frame,
        x: f64,
        y: f64,
        height: Option<f64>,
    ) -> Result<Value, String> {
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return Err("Click outside the video".into());
        }
        let [right, up, d] = frame
            .rows
            .map(|r| v3(r[0] as f64, r[1] as f64, r[2] as f64));
        let view = Projection {
            scene: self,
            frame,
            right,
            up,
            depth: d.normalized(),
        };
        let cursor = [x * 16. / 9., y];
        if let Some(height) = height {
            if !height.is_finite() || height <= 0. {
                return Err("Invalid displayed image height".into());
            }
            for kind in ["vertex", "edge"] {
                let tolerance = (if kind == "vertex" { 8. } else { 6. }) / height;
                let mut candidates = Vec::new();
                let mut consider = |body, index, point, xy: [f64; 2]| {
                    let distance = (xy[0] - cursor[0]).powi(2) + (xy[1] - cursor[1]).powi(2);
                    if distance <= tolerance * tolerance {
                        candidates.push(Candidate {
                            distance,
                            depth: view.depth.dot(point - self.center),
                            body,
                            index,
                            point,
                        });
                    }
                };
                for (body_id, body) in self.topology.iter().enumerate() {
                    if kind == "vertex" {
                        for (index, &p) in body.vertices.iter().enumerate() {
                            consider(body_id, index, p, view.project(p));
                        }
                    } else {
                        for (index, edge) in body.edges.iter().enumerate() {
                            for seg in edge.path.windows(2) {
                                let a = view.project(seg[0]);
                                let b = view.project(seg[1]);
                                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                                let len2 = dx * dx + dy * dy;
                                let t = if len2 > 0. {
                                    (((cursor[0] - a[0]) * dx + (cursor[1] - a[1]) * dy) / len2)
                                        .clamp(0., 1.)
                                } else if view.depth.dot(seg[1] - seg[0]) < 0. {
                                    1.
                                } else {
                                    0.
                                };
                                consider(
                                    body_id,
                                    index,
                                    seg[0] + (seg[1] - seg[0]) * t,
                                    [a[0] + dx * t, a[1] + dy * t],
                                );
                            }
                        }
                    }
                }
                candidates.sort_by(|a, b| {
                    a.distance
                        .total_cmp(&b.distance)
                        .then(a.depth.total_cmp(&b.depth))
                });
                for c in candidates {
                    if !view.visible(c.point) {
                        continue;
                    }
                    let body = &self.topology[c.body];
                    return Ok(self.topology_pick(body, kind, c.index, c.point));
                }
            }
        }
        let ray = view.ray(cursor);
        let hit = self.tracer.trace_one(&ray);
        if hit.t < 0. {
            return Ok(Value::Null);
        }
        let face = self.face_of[hit.prim as usize];
        let mut selection = self.faces[face as usize].clone();
        selection["kind"] = json!("face");
        selection["label"] = json!(format!("Face {}", face + 1));
        selection["index"] = json!(face);
        let [x, y, z] = ray.origin;
        selection["point_mm"] = json!(v3(x as f64, y as f64, z as f64) + view.depth * hit.t as f64);
        selection["revision"] = json!(self.revision);
        Ok(selection)
    }

    pub fn block_edges(&self)->Vec<Value>{
        let [body]=self.topology.as_slice() else{return vec![]};
        if !body.block{return vec![]}
        body.edges.iter().enumerate().map(|(i,e)|self.topology_pick(body,"edge",i,e.path[0])).collect()
    }
    fn topology_pick(&self, body: &BodyTopology, kind: &str, index: usize, point: V3) -> Value {
        let mut value = json!({"kind":kind,"body":body.id,"index":index,"point_mm":point,"revision":self.revision});
        value["label"] = json!(format!(
            "{} {}",
            if kind == "edge" { "Edge" } else { "Vertex" },
            index + 1
        ));
        if kind == "edge" {
            let e = &body.edges[index];
            value["name"] = json!(e.name);
            value["curve"] = json!(e.curve);
            let xyz = |p: &V3| [p.x, p.y, p.z];
            value["endpoints_mm"] = json!([xyz(&e.path[0]), xyz(e.path.last().unwrap())]);
        }
        value
    }
    pub fn box_pick(&self, frame: &Frame, rect: [f64; 4]) -> Result<Vec<Value>, String> {
        if rect
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
        {
            return Err("Invalid selection rectangle".into());
        }
        let [x0, y0, x1, y1] = rect;
        let l = x0.min(x1) * 16. / 9.;
        let r = x0.max(x1) * 16. / 9.;
        let t = y0.min(y1);
        let b = y0.max(y1);
        let [right, up, d] = frame
            .rows
            .map(|r| v3(r[0] as f64, r[1] as f64, r[2] as f64));
        let view = Projection {
            scene: self,
            frame,
            right,
            up,
            depth: d.normalized(),
        };
        let mut out = Vec::new();
        // Resolve frontmost face IDs at the native render pixel centres inside the rectangle.
        let xlo = (l * 1080. - 0.5).ceil().max(0.) as usize;
        let xhi = (r * 1080. + 0.5).floor().min(1920.) as usize;
        let ylo = (t * 1080. - 0.5).ceil().max(0.) as usize;
        let yhi = (b * 1080. + 0.5).floor().min(1080.) as usize;
        let w = xhi.saturating_sub(xlo);
        let h = yhi.saturating_sub(ylo);
        let mut pixels = vec![(f64::INFINITY, u32::MAX); w * h];
        for tri in self.vertices.chunks_exact(3) {
            let p: [(f64, f64, f64); 3] = std::array::from_fn(|i| {
                let [x, y, z] = tri[i].xyz;
                let v = v3(x as f64, y as f64, z as f64);
                (
                    (1. + v.dot(right) + frame.rows[0][3] as f64) * 960.,
                    (1. - v.dot(up) - frame.rows[1][3] as f64) * 540.,
                    v.dot(d) + frame.rows[2][3] as f64,
                )
            });
            let [(ax, ay, az), (bx, by, bz), (cx, cy, cz)] = p;
            let det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if det.abs() < 1e-14 {
                continue;
            }
            for y in ylo.max(ay.min(by).min(cy).floor() as usize)
                ..yhi.min(ay.max(by).max(cy).ceil() as usize)
            {
                for x in xlo.max(ax.min(bx).min(cx).floor() as usize)
                    ..xhi.min(ax.max(bx).max(cx).ceil() as usize)
                {
                    let (dx, dy) = (x as f64 + 0.5 - ax, y as f64 + 0.5 - ay);
                    let u = (dx * (cy - ay) - dy * (cx - ax)) / det;
                    let v = ((bx - ax) * dy - (by - ay) * dx) / det;
                    if u < 0. || v < 0. || u + v > 1. {
                        continue;
                    }
                    let z = az + u * (bz - az) + v * (cz - az);
                    let pixel = &mut pixels[(y - ylo) * w + x - xlo];
                    if (0.0..=1.).contains(&z) && z < pixel.0 {
                        *pixel = (z, tri[0].face);
                    }
                }
            }
        }
        let mut seen = vec![false; self.faces.len()];
        for (i, (_, face)) in pixels.into_iter().enumerate() {
            if face == u32::MAX || seen[face as usize] {
                continue;
            }
            let x = (xlo + i % w) as f64 + 0.5;
            let y = (ylo + i / w) as f64 + 0.5;
            let pick = self.pick(frame, x / 1920., y / 1080., None)?;
            if pick["index"] == face {
                seen[face as usize] = true;
                out.push(pick);
            }
        }
        for body in &self.topology {
            for (i, e) in body.edges.iter().enumerate() {
                let point = e.path.windows(2).find_map(|s| {
                    let a = view.project(s[0]);
                    let z = view.project(s[1]);
                    let (dx, dy) = (z[0] - a[0], z[1] - a[1]);
                    if dx.hypot(dy)<1e-8{return None;}
                    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                    for (p, delta, min, max) in [(a[0], dx, l, r), (a[1], dy, t, b)] {
                        if delta.abs() < 1e-15 {
                            if p < min || p > max {
                                return None;
                            }
                        } else {
                            let (u, v) = ((min - p) / delta, (max - p) / delta);
                            lo = lo.max(u.min(v));
                            hi = hi.min(u.max(v));
                        }
                    }
                    if lo > hi {
                        return None;
                    }
                    let count = ((hi - lo) * dx.hypot(dy) * 1080.).ceil().max(1.) as usize;
                    (0..=count)
                        .map(|n| s[0].lerp(s[1], lo + (hi - lo) * n as f64 / count as f64))
                        .find(|&p| view.visible(p))
                });
                if let Some(p) = point {
                    out.push(self.topology_pick(body, "edge", i, p));
                }
            }
            for (i, &p) in body.vertices.iter().enumerate() {
                let [x, y] = view.project(p);
                if x >= l && x <= r && y >= t && y <= b && view.visible(p) {
                    out.push(self.topology_pick(body, "vertex", i, p));
                }
            }
        }
        Ok(out)
    }
    pub fn bounds(&self, selected: &[Value]) -> Option<[[f32; 3]; 2]> {
        let mut bb = agcad_geo::math::Aabb::EMPTY;
        let faces: std::collections::HashSet<_> = selected
            .iter()
            .filter(|v| v["kind"] == "face")
            .filter_map(|v| v["index"].as_u64())
            .collect();
        if !faces.is_empty() || selected.is_empty() {
            for v in &self.vertices {
                if selected.is_empty() || faces.contains(&(v.face as u64)) {
                    let [x, y, z] = v.xyz;
                    bb.add(v3(x as f64, y as f64, z as f64));
                }
            }
        }
        for v in selected {
            let Some(body) = self.topology.iter().find(|b| v["body"] == b.id.as_str()) else {
                continue;
            };
            let i = v["index"].as_u64()? as usize;
            let mut add = |p: V3| bb.add((p - self.center) * self.radius.recip());
            match v["kind"].as_str() {
                Some("edge") => body.edges[i].path.iter().copied().for_each(add),
                Some("vertex") => add(body.vertices[i]),
                _ => (),
            }
        }
        bb.min.x.is_finite().then(|| [f3(bb.min), f3(bb.max)])
    }
    pub fn highlights(&self, selected: &[Value]) -> Highlight {
        Highlight::combine(
            selected
                .iter()
                .filter(|v| v["kind"] != "face")
                .map(|v| self.highlight(v)),
        )
    }
    fn highlight(&self, selection: &Value) -> Highlight {
        let Some(index) = selection["index"].as_u64() else {
            return Highlight::default();
        };
        if selection["kind"] == "face" {
            return Highlight::face(index as u32);
        }
        let Some(body) = self
            .topology
            .iter()
            .find(|b| selection["body"] == b.id.as_str())
        else {
            return Highlight::default();
        };
        let normalized = |p| f3((p - self.center) * self.radius.recip());
        let index = index as usize;
        match selection["kind"].as_str() {
            Some("edge") => Highlight::edge(body.edges[index].path.iter().copied().map(normalized)),
            Some("vertex") => Highlight::vertex(normalized(body.vertices[index])),
            _ => Highlight::default(),
        }
    }
}
