//! # agcad-geo-occt — OCCT test oracle (never production; AG-02-006)
//!
//! Runs a cadquery-ocp Python script in a subprocess (the Demo's validated oracle) on a STEP
//! file and returns validity, volume, area, solid/face counts. Interpreter from
//! `AGCAD_OCCT_PYTHON` (default `python3`; `python` on Windows); `PYTHONPATH` may point at `.python-deps`.

use serde_json::Value;
use std::process::Command;

pub const ORACLE_PY: &str = r#"
import json, sys
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
r = STEPControl_Reader()
if r.ReadFile(sys.argv[1]) != 1:
    print(json.dumps({"ok": False, "error": "read failed"})); sys.exit(0)
r.TransferRoots(); s = r.OneShape()
def count(t):
    e = TopExp_Explorer(s, t); n = 0
    while e.More(): n += 1; e.Next()
    return n
vp = GProp_GProps(); BRepGProp.VolumeProperties_s(s, vp)
ap = GProp_GProps(); BRepGProp.SurfaceProperties_s(s, ap)
c = vp.CentreOfMass()
print(json.dumps({"ok": True, "valid": BRepCheck_Analyzer(s).IsValid(), "volume": vp.Mass(), "area": ap.Mass(),
                  "centroid": [c.X(), c.Y(), c.Z()], "solids": count(TopAbs_SOLID), "faces": count(TopAbs_FACE)}))
"#;

pub fn python() -> String { std::env::var("AGCAD_OCCT_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into()) }

/// Is the oracle available on this machine?
pub fn available() -> bool {
    Command::new(python()).args(["-c", "import OCP"]).output().map(|o| o.status.success()).unwrap_or(false)
}

/// Evaluate a STEP file with OCCT.
pub fn evaluate_step(path: &std::path::Path) -> Result<Value, String> {
    let dir = std::env::temp_dir().join("agcad-occt");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let script = dir.join("oracle.py");
    std::fs::write(&script, ORACLE_PY).map_err(|e| e.to_string())?;
    let out = Command::new(python()).arg("-I").arg(&script).arg(path).output().map_err(|e| e.to_string())?;
    if !out.status.success() { return Err(String::from_utf8_lossy(&out.stderr).into_owned()); }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("oracle output: {e}"))
}

/// Compare an AgCAD body against OCCT after STEP export (parity gate, spec 09/10).
pub fn parity(body: &agcad_geo::Brep, rel_tol: f64) -> Result<Value, String> {
    let m = agcad_geo::tessellate::tessellate(body, &agcad_geo::tessellate::TessOptions { sag: 1e-4, ..Default::default() });
    let (v, a, _) = m.mass_properties();
    let (txt, _) = agcad_geo::step_write::write_step(body, &Default::default());
    let path = std::env::temp_dir().join(format!("agcad-occt-{}.stp", std::process::id()));
    std::fs::write(&path, txt).map_err(|e| e.to_string())?;
    let o = evaluate_step(&path)?;
    let ov = o["volume"].as_f64().unwrap_or(f64::NAN);
    let oa = o["area"].as_f64().unwrap_or(f64::NAN);
    let pass = o["valid"] == true && ((ov - v).abs() / v.abs().max(1e-12) <= rel_tol) && ((oa - a).abs() / a.abs().max(1e-12) <= rel_tol);
    Ok(serde_json::json!({ "pass": pass, "agcad": { "volume": v, "area": a }, "occt": o }))
}


pub const OFFSET_PY: &str = r#"
import json, sys
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCP.BRepOffset import BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Arc
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
r = STEPControl_Reader()
if r.ReadFile(sys.argv[1]) != 1:
    print(json.dumps({"ok": False, "error": "read failed"})); sys.exit(0)
r.TransferRoots(); s = r.OneShape()
m = BRepOffsetAPI_MakeOffsetShape()
m.PerformByJoin(s, float(sys.argv[2]), 1e-7, BRepOffset_Skin, False, False, GeomAbs_Arc, False)
if not m.IsDone():
    print(json.dumps({"ok": False, "error": "offset failed"})); sys.exit(0)
vp = GProp_GProps(); BRepGProp.VolumeProperties_s(m.Shape(), vp)
print(json.dumps({"ok": True, "volume": abs(vp.Mass())}))
"#;

/// Volume of OCCT's offset of a body by `d` (negative: inward; arc joins at concave edges).
/// Oracle for the exact closed shell: shell volume = V − offset_volume(−t).
pub fn offset_volume(body: &agcad_geo::Brep, d: f64) -> Result<f64, String> {
    let dir = std::env::temp_dir().join("agcad-occt");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let script = dir.join("offset.py");
    std::fs::write(&script, OFFSET_PY).map_err(|e| e.to_string())?;
    let (txt, _) = agcad_geo::step_write::write_step(body, &Default::default());
    let path = dir.join(format!("offset-{}-{}.stp", std::process::id(), d.to_bits()));
    std::fs::write(&path, txt).map_err(|e| e.to_string())?;
    let out = Command::new(python()).arg("-I").arg(&script).arg(&path).arg(format!("{d}")).output().map_err(|e| e.to_string())?;
    if !out.status.success() { return Err(String::from_utf8_lossy(&out.stderr).into_owned()); }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("oracle output: {e}"))?;
    if v["ok"] != true { return Err(format!("{}", v["error"])); }
    v["volume"].as_f64().ok_or_else(|| "no volume".into())
}
