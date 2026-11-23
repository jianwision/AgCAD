use crate::document::Scene;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Finish {
    pub color: Option<String>,
    pub roughness: Option<f32>,
    pub metalness: Option<f32>,
    pub designation: Option<String>,
    pub density_kg_m3: Option<f64>,
}
impl Finish {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(c) = &self.color {
            if c.len() != 7 || !c.starts_with('#') || !c[1..].bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err("Use a six-digit hex color".into());
            }
        }
        if [self.roughness, self.metalness]
            .into_iter()
            .flatten()
            .any(|v| !(0.0..=1.0).contains(&v))
        {
            return Err("Roughness and metalness must be between zero and one".into());
        }
        if self.designation.as_ref().is_some_and(|s| s.len() > 160)
            || self
                .density_kg_m3
                .is_some_and(|x| !x.is_finite() || x <= 0.)
        {
            return Err("Invalid engineering material".into());
        }
        Ok(())
    }
}
pub type Overrides = BTreeMap<String, Finish>;
pub fn key(face: &Value, scope: &str) -> Result<String, String> {
    let id = match scope {
        "body" => &face["body"],
        "face" => &face["entity"],
        _ => return Err("Choose a face or body".into()),
    };
    Ok(format!(
        "{scope}/{}",
        id.as_str()
            .ok_or("Missing persistent face or body identity")?
    ))
}
fn rgb(f: &Value) -> Option<[f32; 3]> {
    Some([
        f.get(0)?.as_f64()? as f32,
        f.get(1)?.as_f64()? as f32,
        f.get(2)?.as_f64()? as f32,
    ])
}
pub fn appearance(face: &Value, values: &Overrides) -> ([f32; 8], Value) {
    let mut color = rgb(&face["body_color"]).unwrap_or([0.55, 0.75, 0.82]);
    let (mut rough, mut metal) = (0.55, 0.05);
    let mut eng = face["material"].clone();
    if !eng.is_object() {
        eng = json!({});
    }
    for scope in ["body", "face"] {
        if scope == "face" {
            if let Some(c) = rgb(&face["face_color"]) {
                color = c;
            }
        }
        let Some(f) = key(face, scope).ok().and_then(|k| values.get(&k)) else {
            continue;
        };
        if let Some(c) = &f.color {
            let n = u32::from_str_radix(&c[1..], 16).unwrap();
            color = [
                (n >> 16) as f32 / 255.,
                ((n >> 8) & 255) as f32 / 255.,
                (n & 255) as f32 / 255.,
            ];
        }
        if let Some(v) = f.roughness {
            rough = v
        }
        if let Some(v) = f.metalness {
            metal = v
        }
        if let Some(s) = &f.designation {
            eng["name"] = json!(s);
        }
        if let Some(v) = f.density_kg_m3 {
            eng["density"] = json!(v);
            eng["density_unit"] = json!("kg/m³");
        }
    }
    (
        [color[0], color[1], color[2], 1., rough, metal, 0., 0.],
        json!({"color":format!("#{:02x}{:02x}{:02x}",(color[0]*255.).round() as u8,(color[1]*255.).round() as u8,(color[2]*255.).round() as u8),"roughness":rough,"metalness":metal,"engineering":eng}),
    )
}
pub fn palette(scene: &Scene, values: &Overrides, selected: &[Value]) -> Vec<[f32; 8]> {
    let faces: std::collections::HashSet<_> = selected
        .iter()
        .filter(|v| v["kind"] == "face")
        .filter_map(|v| v["index"].as_u64())
        .collect();
    scene
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut a = appearance(f, values).0;
            a[7] = if faces.contains(&(i as u64)) { 1. } else { 0. };
            a
        })
        .collect()
}
