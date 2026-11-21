//! File features (spec 03 `file.*`, spec 18). Features stay pure: they produce encoded bytes
//! (`FileBytes`); the harness/control plane hands them to a file descriptor or GPUDirect
//! Storage (AG-18-001: the host never builds geometry).

use crate::util::*;
use agcad_driver::AgStatus;
use agcad_geo::step_write::{write_step, ReceiverProfile, Schema, StepOptions};
use agcad_geo::tessellate::{Mesh, TessOptions};
use agcad_geo::threemf::{write_3mf, ThreeMfObject};
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct FileBytes {
    pub path: String,
    pub bytes: Vec<u8>,
    pub warnings: Vec<String>,
}

pub fn stl_binary(m: &Mesh, name: &str) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out[..name.len().min(80)].copy_from_slice(&name.as_bytes()[..name.len().min(80)]);
    out.extend_from_slice(&(m.triangles.len() as u32).to_le_bytes());
    for t in &m.triangles {
        let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
        let nrm = (b - a).cross(c - a).normalized();
        for v in [nrm, a, b, c] { for x in [v.x, v.y, v.z] { out.extend_from_slice(&(x as f32).to_le_bytes()); } }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

fn mesh_from(ctx: &BuildContext<'_>, v: &agcad_ir::Value, res: f64) -> Result<Mesh, FeatureError> {
    match ctx.product(v)? {
        Product::Mesh(m) => Ok(m.clone()),
        Product::Solid(s) => Ok(ctx.kernel.tessellate(s, &TessOptions { sag: res, ..Default::default() })),
        Product::Implicit(i) => crate::mesh::implicit_to_mesh(i, res),
        other => Err(bad(format!("cannot export {other:?} as a mesh"))),
    }
}

pub struct FileOp(pub &'static str);

impl FeatureImpl for FileOp {
    fn op(&self) -> &'static str { self.0 }
    fn execute(&self, ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let path = text(n, "path").unwrap_or("out").to_string();
        let items = list(req(n, "items")?);
        let first = items.first().ok_or_else(|| bad("no items to export"))?;
        let fb = match self.0 {
            "file.export_step" => {
                let s = solid(ctx, first)?;
                let b = s.brep();
                let m = ctx.kernel.tessellate(&s, &TessOptions { sag: 1e-3, ..Default::default() });
                let (v, a, c) = m.mass_properties();
                let opt = StepOptions {
                    schema: if text(n, "schema") == Some("ap214") { Schema::Ap214 } else { Schema::Ap242 },
                    profile: match text(n, "profile") { Some("solidworks") => ReceiverProfile::SolidWorks, Some("catia") => ReceiverProfile::Catia, _ => ReceiverProfile::Generic },
                    product_name: ctx.part.name.clone(),
                    validation: Some((v, a, c)),
                    ..Default::default()
                };
                let (txt, rep) = write_step(&b, &opt);
                let mut warnings = rep.warnings;
                if opt.profile == ReceiverProfile::SolidWorks { warnings.push("solidworks profile is experimental until verified on a SolidWorks CI runner (AG-18-007)".into()); }
                FileBytes { path, bytes: txt.into_bytes(), warnings }
            }
            "file.export_mesh" => {
                let m = mesh_from(ctx, first, num_or(ctx, n, "res", 0.01)?)?;
                FileBytes { path, bytes: stl_binary(&m, &ctx.part.name), warnings: vec![] }
            }
            "file.export_3mf" => {
                let objs = items.iter().map(|v| Ok(ThreeMfObject { name: ctx.part.name.clone(), mesh: mesh_from(ctx, v, num_or(ctx, n, "res", 0.05)?)?, beams: vec![], material: None })).collect::<Result<Vec<_>, FeatureError>>()?;
                FileBytes { path, bytes: write_3mf(&objs), warnings: vec![] }
            }
            "file.export_usd" => {
                let m = mesh_from(ctx, first, num_or(ctx, n, "res", 0.05)?)?;
                FileBytes { path, bytes: usda(&m, &ctx.part.name, &ctx.part.design_tables).into_bytes(), warnings: vec![] }
            }
            "file.export_feature_script" => {
                let target = text(n, "target").unwrap_or("catia_vba");
                if target != "catia_vba" {
                    return Err(FeatureError::new(AgStatus::UnsupportedOp, format!("feature-script target '{target}' not implemented yet (catia_vba available)")));
                }
                let (src, warnings) = catia_vba(ctx.part);
                FileBytes { path, bytes: src.into_bytes(), warnings }
            }
            _ => return Err(FeatureError::new(AgStatus::UnsupportedOp, format!("{} not implemented yet", self.0))),
        };
        Ok(Product::Other(Arc::new(fb)))
    }
}

/// OpenUSD (USDA) mesh export; design-table rows become a `configuration` variant set
/// (spec 18 § review F13). Stable names travel as a face-name primvar on the mesh.
pub fn usda(m: &Mesh, name: &str, tables: &std::collections::BTreeMap<String, (Vec<String>, Vec<Vec<agcad_ir::Value>>)>) -> String {
    use std::fmt::Write;
    let id = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>();
    let mut o = String::from("#usda 1.0\n(\n    defaultPrim = \"Part\"\n    metersPerUnit = 0.001\n    upAxis = \"Z\"\n)\n\n");
    let variants: Vec<String> = tables.values().flat_map(|(_, rows)| (0..rows.len()).map(|i| format!("row{i}"))).collect();
    if variants.is_empty() {
        let _ = writeln!(o, "def Xform \"Part\"\n{{");
    } else {
        let _ = writeln!(o, "def Xform \"Part\" (\n    variants = {{ string configuration = \"{}\" }}\n    prepend variantSets = \"configuration\"\n)\n{{", variants[0]);
        let _ = writeln!(o, "    variantSet \"configuration\" = {{");
        for v in &variants { let _ = writeln!(o, "        \"{v}\" {{ }}"); }
        let _ = writeln!(o, "    }}");
    }
    let _ = writeln!(o, "    def Mesh \"{id}\"\n    {{");
    let pts: Vec<String> = m.positions.iter().map(|p| format!("({}, {}, {})", p.x, p.y, p.z)).collect();
    let _ = writeln!(o, "        point3f[] points = [{}]", pts.join(", "));
    let _ = writeln!(o, "        int[] faceVertexCounts = [{}]", vec!["3"; m.triangles.len()].join(", "));
    let idx: Vec<String> = m.triangles.iter().flat_map(|t| t.iter().map(|i| i.to_string())).collect();
    let _ = writeln!(o, "        int[] faceVertexIndices = [{}]", idx.join(", "));
    let fo: Vec<String> = m.face_of.iter().map(|f| f.to_string()).collect();
    let _ = writeln!(o, "        int[] primvars:agcad:faceId = [{}] (\n            interpolation = \"uniform\"\n        )", fo.join(", "));
    let _ = writeln!(o, "    }}\n}}");
    o
}

/// History-preserving handoff (spec 18; capability track): replay the specification tree as a
/// CATIA V5 VBA macro (sketch circles/rectangles → pad/pocket; holes as pockets). Features the
/// generator cannot express are listed as warnings, never silently dropped.
pub fn catia_vba(part: &agcad_runtime::document::PartState) -> (String, Vec<String>) {
    use agcad_ir::Value;
    use std::fmt::Write;
    let num = |v: Option<&Value>| -> f64 { match v { Some(Value::Quantity(q)) => q.value, Some(Value::Float(f)) => *f, Some(Value::Int(i)) => *i as f64, _ => 0.0 } };
    let mut o = String::from("' Generated by AgCAD (file.export_feature_script, catia_vba)\nSub CATMain()\n    Dim doc As PartDocument: Set doc = CATIA.Documents.Add(\"Part\")\n    Dim prt As Part: Set prt = doc.Part\n    Dim body As Body: Set body = prt.MainBody\n    Dim sf As ShapeFactory: Set sf = prt.ShapeFactory\n    Dim ref As Reference: Set ref = prt.CreateReferenceFromObject(prt.OriginElements.PlaneXY)\n");
    let mut warnings = Vec::new();
    let mut last_sketch = String::new();
    for id in part.rebuild_order() {
        let f = &part.features[&id];
        match f.op.as_str() {
            "sketch.create" => { let _ = writeln!(o, "    Dim sk{id} As Sketch: Set sk{id} = body.Sketches.Add(ref)\n    Dim f2d{id} As Factory2D: Set f2d{id} = sk{id}.OpenEdition()"); last_sketch = format!("{id}"); }
            "sketch.circle" => {
                let c = match f.args.get("pts") { Some(Value::List(l)) => l.first().cloned(), _ => None };
                let (cx, cy) = match c { Some(Value::List(xy)) => (num(xy.first()), num(xy.get(1))), _ => (0.0, 0.0) };
                let _ = writeln!(o, "    f2d{last_sketch}.CreateClosedCircle {cx}, {cy}, {}", num(f.args.get("r")));
            }
            "sketch.rectangle" => {
                if let Some(Value::List(pts)) = f.args.get("pts") {
                    let p = |k: usize| match pts.get(k) { Some(Value::List(xy)) => (num(xy.first()), num(xy.get(1))), _ => (0.0, 0.0) };
                    let ((x0, y0), (x1, y1)) = (p(0), p(1));
                    for (a, b) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))] {
                        let _ = writeln!(o, "    f2d{last_sketch}.CreateLine {}, {}, {}, {}", a.0, a.1, b.0, b.1);
                    }
                }
            }
            "sketch.regions" => { let _ = writeln!(o, "    sk{last_sketch}.CloseEdition"); }
            "part.pad" | "part.pocket" => {
                let kind = if f.op == "part.pad" { "AddNewPad" } else { "AddNewPocket" };
                let _ = writeln!(o, "    Dim ft{id}: Set ft{id} = sf.{kind}(sk{last_sketch}, {})", num(f.args.get("limits")));
            }
            other => warnings.push(format!("feature {} ({other}) not expressible in the CATIA VBA generator yet; replay manually", f.feature_ref())),
        }
    }
    o.push_str("    prt.Update\nEnd Sub\n");
    (o, warnings)
}

/// `file.import_step`: the control plane reads the bytes; decoding is the reference STEP
/// pipeline (GPU lexer/parser in P11 GPU pass). Multiple bodies → first solid is returned,
/// all bodies listed in the import report.
pub struct ImportStep;

impl FeatureImpl for ImportStep {
    fn op(&self) -> &'static str { "file.import_step" }
    fn execute(&self, _ctx: &mut BuildContext<'_>, n: &FeatureNode) -> Result<Product, FeatureError> {
        let path = text(n, "path").ok_or_else(|| bad("import_step needs a path"))?;
        let txt = std::fs::read_to_string(path).map_err(|e| FeatureError::new(AgStatus::InvalidParameter, format!("{path}: {e}")))?;
        let (bodies, rep) = agcad_geo::step_read::read_step(&txt, &Default::default()).map_err(|e| FeatureError::new(AgStatus::TopologyFailed, e))?;
        let b = bodies.into_iter().next().ok_or_else(|| FeatureError::new(AgStatus::TopologyFailed, format!("no bodies in {path} ({} entities)", rep.entities)))?;
        let v = b.validate();
        if !v.valid { return Err(FeatureError::new(AgStatus::GeometryCheckFailed, format!("imported body invalid: {:?}", v.problems.iter().take(3).collect::<Vec<_>>())).suggest("run direct.heal_body")); }
        Ok(Product::Solid(agcad_geo::kernel::Solid::Brep(b)))
    }
}
