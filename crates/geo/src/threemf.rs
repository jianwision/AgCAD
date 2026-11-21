//! 3MF export for additive manufacturing (spec 18 § 6): Core + Materials (base materials)
//! + Beam Lattice extension. Deterministic output.

use crate::math::V3;
use crate::tessellate::Mesh;
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub struct Beam {
    pub a: V3,
    pub b: V3,
    pub r0: f64,
    pub r1: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreeMfObject {
    pub name: String,
    pub mesh: Mesh,
    /// Optional strut lattice written with the beam-lattice extension (vertices appended).
    pub beams: Vec<Beam>,
    /// (material name, sRGB "#RRGGBB").
    pub material: Option<(String, String)>,
}

fn num(x: f64) -> String {
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" { "0".into() } else { s }
}

fn esc(s: &str) -> String { s.replace('&', "&amp;").replace('<', "&lt;").replace('"', "&quot;") }

pub fn write_3mf(objects: &[ThreeMfObject]) -> Vec<u8> {
    let mut model = String::new();
    let has_beams = objects.iter().any(|o| !o.beams.is_empty());
    let _ = write!(model, r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"{}>
 <resources>
"#, if has_beams { r#" xmlns:b="http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02" requiredextensions="b""# } else { "" });
    let mats: Vec<&(String, String)> = objects.iter().filter_map(|o| o.material.as_ref()).collect();
    if !mats.is_empty() {
        model += "  <basematerials id=\"1\">\n";
        for (n, c) in &mats { let _ = writeln!(model, "   <base name=\"{}\" displaycolor=\"{}\"/>", esc(n), esc(c)); }
        model += "  </basematerials>\n";
    }
    let mut mat_idx = 0;
    for (oi, o) in objects.iter().enumerate() {
        let id = oi + 2;
        let matattr = if o.material.is_some() { let s = format!(" pid=\"1\" pindex=\"{mat_idx}\""); mat_idx += 1; s } else { String::new() };
        let _ = writeln!(model, "  <object id=\"{id}\" type=\"model\" name=\"{}\"{matattr}>\n   <mesh>\n    <vertices>", esc(&o.name));
        for p in &o.mesh.positions { let _ = writeln!(model, "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>", num(p.x), num(p.y), num(p.z)); }
        let base = o.mesh.positions.len();
        for bm in &o.beams {
            for p in [bm.a, bm.b] { let _ = writeln!(model, "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>", num(p.x), num(p.y), num(p.z)); }
        }
        model += "    </vertices>\n    <triangles>\n";
        for t in &o.mesh.triangles { let _ = writeln!(model, "     <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>", t[0], t[1], t[2]); }
        model += "    </triangles>\n";
        if !o.beams.is_empty() {
            let rmin = o.beams.iter().map(|b| b.r0.min(b.r1)).fold(f64::INFINITY, f64::min);
            let _ = writeln!(model, "    <b:beamlattice minlength=\"0.0001\" radius=\"{}\" clippingmode=\"none\">\n     <b:beams>", num(rmin));
            for (k, bm) in o.beams.iter().enumerate() {
                let _ = writeln!(model, "      <b:beam v1=\"{}\" v2=\"{}\" r1=\"{}\" r2=\"{}\"/>", base + 2 * k, base + 2 * k + 1, num(bm.r0), num(bm.r1));
            }
            model += "     </b:beams>\n    </b:beamlattice>\n";
        }
        model += "   </mesh>\n  </object>\n";
    }
    model += " </resources>\n <build>\n";
    for oi in 0..objects.len() { let _ = writeln!(model, "  <item objectid=\"{}\"/>", oi + 2); }
    model += " </build>\n</model>\n";
    let content_types = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>
"#;
    let rels = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>
"#;
    crate::zip::write_zip(&[("[Content_Types].xml", content_types.as_bytes()), ("_rels/.rels", rels.as_bytes()), ("3D/3dmodel.model", model.as_bytes())])
}
