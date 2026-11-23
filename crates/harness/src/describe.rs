//! `describe` (spec 07 MCP tool): a compact, structured text view of a body for language
//! models, alongside renders. Faces are summarized as engineering features: cylindrical faces
//! become holes (concave) or bosses/outer walls (convex) by diameter, axis and position
//! (half-cylinder faces of one cylinder are merged); planar faces are grouped by normal and
//! offset. `diff` compares two descriptions semantically (holes added/removed, volume delta,
//! parameter changes).

use agcad_geo::brep::Brep;
use agcad_geo::math::V3;
use agcad_geo::surface::Surface;
use agcad_geo::tessellate::{tessellate, TessOptions};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn r6(x: f64) -> f64 { (x * 1e6).round() / 1e6 }
fn v6(v: V3) -> [f64; 3] { [r6(v.x), r6(v.y), r6(v.z)] }

/// Canonical axis direction (sign-normalized so ±z group together).
fn canon(d: V3) -> V3 {
    let d = d.normalized();
    let flip = d.z < -1e-12 || (d.z.abs() <= 1e-12 && (d.y < -1e-12 || (d.y.abs() <= 1e-12 && d.x < 0.0)));
    if flip { d * -1.0 } else { d }
}

pub fn describe(b: &Brep, params: &BTreeMap<String, f64>) -> Value {
    let m = tessellate(b, &TessOptions::default());
    let (volume, area, centroid) = m.mass_properties();
    let bb = b.bbox();
    let v = b.validate();
    // cylinders keyed by (axis, center line point, radius, concave)
    let mut cyl: BTreeMap<(i64, i64, i64, i64, i64, i64, i64, bool), (f64, f64, usize)> = BTreeMap::new();
    let mut planes: BTreeMap<(i64, i64, i64, i64), (usize, Vec<String>)> = BTreeMap::new();
    let mut other: BTreeMap<String, usize> = BTreeMap::new();
    let q = |x: f64| (x * 1e4).round() as i64;
    for f in &b.faces {
        match &b.surfaces[f.surface as usize] {
            Surface::Cylinder { frame, r } => {
                let ax = canon(frame.z);
                // point on the axis closest to the origin identifies the line
                let o = frame.origin - ax * frame.origin.dot(ax);
                let concave = !f.same_sense;
                let (v0, v1) = (frame.origin.dot(ax) + f.uv[2].min(f.uv[3]) * frame.z.dot(ax), frame.origin.dot(ax) + f.uv[2].max(f.uv[3]) * frame.z.dot(ax));
                let e = cyl.entry((q(ax.x), q(ax.y), q(ax.z), q(o.x), q(o.y), q(o.z), q(*r), concave)).or_insert((f64::INFINITY, f64::NEG_INFINITY, 0));
                e.0 = e.0.min(v0.min(v1));
                e.1 = e.1.max(v0.max(v1));
                e.2 += 1;
            }
            Surface::Plane { frame } => {
                let n = if f.same_sense { frame.z } else { frame.z * -1.0 };
                let n = n.normalized();
                let e = planes.entry((q(n.x), q(n.y), q(n.z), q(frame.origin.dot(n)))).or_insert((0, vec![]));
                e.0 += 1;
                if e.1.len() < 4 { e.1.push(f.name.clone()); }
            }
            s => *other.entry(format!("{:?}", std::mem::discriminant(s)).replace("Discriminant", "surface")).or_insert(0) += 1,
        }
    }
    let mut holes = Vec::new();
    let mut walls = Vec::new();
    for ((ax, ay, az, ox, oy, oz, r, concave), (z0, z1, nfaces)) in cyl {
        let f = |x: i64| x as f64 / 1e4;
        let axis = [f(ax), f(ay), f(az)];
        let d = json!({ "diameter": r6(2.0 * f(r)), "axis": axis, "through_point": [f(ox), f(oy), f(oz)], "extent": [r6(z0), r6(z1)], "length": r6(z1 - z0), "faces": nfaces });
        if concave { holes.push(d) } else { walls.push(d) }
    }
    let planes: Vec<Value> = planes.into_iter().map(|((nx, ny, nz, off), (n, names))| json!({ "normal": [nx as f64 / 1e4, ny as f64 / 1e4, nz as f64 / 1e4], "offset": off as f64 / 1e4, "faces": n, "names": names })).collect();
    json!({
        "valid": v.valid, "problems": v.problems.iter().take(5).collect::<Vec<_>>(),
        "shells": b.shells.len(), "faces": b.faces.len(), "edges": b.edges.len(), "vertices": b.vertices.len(),
        "bounding_box": { "min": v6(bb.min), "max": v6(bb.max), "size": v6(bb.extent()) },
        "volume": (b.kind == agcad_geo::brep::BodyKind::Solid).then(|| r6(volume)), "area": r6(area),
        "centroid": if b.kind == agcad_geo::brep::BodyKind::Sheet {
            (area > 0.).then(|| v6(m.triangles.iter().fold(V3::ZERO, |sum, t| {
                let [p,q,r] = t.map(|i| m.positions[i as usize]);
                sum + (p+q+r) * ((q-p).cross(r-p).norm() / (6.*area))
            })))
        } else { Some(v6(centroid)) }, "evidence": "bounded",
        "holes": holes, "cylindrical_walls": walls, "planar_faces": planes, "other_faces": other,
        "parameters": params,
    })
}

/// Semantic difference `after − before` of two descriptions.
pub fn diff(before: &Value, after: &Value) -> Value {
    let key = |h: &Value| format!("Ø{} axis {} at {}", h["diameter"], h["axis"], h["through_point"]);
    let set = |v: &Value, k: &str| -> BTreeMap<String, Value> { v[k].as_array().map(|a| a.iter().map(|h| (key(h), h.clone())).collect()).unwrap_or_default() };
    let mut out = serde_json::Map::new();
    for k in ["holes", "cylindrical_walls"] {
        let (b, a) = (set(before, k), set(after, k));
        let added: Vec<&String> = a.keys().filter(|x| !b.contains_key(*x)).collect();
        let removed: Vec<&String> = b.keys().filter(|x| !a.contains_key(*x)).collect();
        if !added.is_empty() || !removed.is_empty() { out.insert(k.into(), json!({ "added": added, "removed": removed })); }
    }
    for k in ["volume", "area", "faces"] {
        let (b, a) = (before[k].as_f64().unwrap_or(0.0), after[k].as_f64().unwrap_or(0.0));
        if (a - b).abs() > 1e-9 { out.insert(format!("{k}_delta"), json!(r6(a - b))); }
    }
    let (pb, pa) = (before["parameters"].as_object().cloned().unwrap_or_default(), after["parameters"].as_object().cloned().unwrap_or_default());
    let changed: BTreeMap<String, Value> = pa.iter().filter(|(k, v)| pb.get(*k) != Some(*v)).map(|(k, v)| (k.clone(), json!({ "from": pb.get(k), "to": v }))).collect();
    if !changed.is_empty() { out.insert("parameters".into(), json!(changed)); }
    if out.is_empty() { json!({ "unchanged": true }) } else { Value::Object(out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_geo::builders::{prism, Profile};
    use agcad_geo::math::Frame;
    #[test]
    fn flange_holes_are_reported_as_holes() {
        let mut p = Profile::circle(0.0, 0.0, 60.0);
        for k in 0..6 { let a = k as f64 * core::f64::consts::TAU / 6.0; p = p.with_hole(Profile::circle(40.0 * a.cos(), 40.0 * a.sin(), 4.5)); }
        let b = prism(&p, &Frame::WORLD, 12.0);
        let d = describe(&b, &BTreeMap::new());
        let holes = d["holes"].as_array().unwrap();
        assert_eq!(holes.len(), 6, "{d:#}");
        assert!(holes.iter().all(|h| h["diameter"] == 9.0 && h["length"] == 12.0 && h["axis"] == json!([0.0, 0.0, 1.0])), "{d:#}");
        assert_eq!(d["cylindrical_walls"].as_array().unwrap().len(), 1);
        assert_eq!(d["cylindrical_walls"][0]["diameter"], 120.0);
        let mut p5 = Profile::circle(0.0, 0.0, 60.0);
        for k in 0..5 { let a = k as f64 * core::f64::consts::TAU / 6.0; p5 = p5.with_hole(Profile::circle(40.0 * a.cos(), 40.0 * a.sin(), 4.5)); }
        let d5 = describe(&prism(&p5, &Frame::WORLD, 12.0), &BTreeMap::new());
        let df = diff(&d, &d5);
        assert_eq!(df["holes"]["removed"].as_array().unwrap().len(), 1, "{df:#}");
        assert!(df["volume_delta"].as_f64().unwrap() > 0.0);
    }
}
