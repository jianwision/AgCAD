//! STEP (ISO 10303-21) export, AP242 (default) / AP214 (spec 18 § 2–5).
//!
//! Deterministic entity numbering (AG-18-004), analytic entities kept analytic (AG-18-005),
//! stable provenance names in `ADVANCED_FACE.name` / `EDGE_CURVE.name` (AG-18-006),
//! validation properties (volume, area, centroid; AG-18-003).
//!
//! This is the CPU reference encoder. The GPU encoder (number → format → offsets → emit) is
//! a P11 item; it must produce byte-identical output to this function.

use crate::brep::{BodyKind, Brep};
use crate::curve::Curve;
use crate::math::{Frame, V3};
use crate::nurbs::{NurbsCurve, NurbsSurface};
use crate::surface::Surface;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Schema {
    Ap242,
    Ap214,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReceiverProfile {
    Generic,
    SolidWorks,
    Catia,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOptions {
    pub schema: Schema,
    pub profile: ReceiverProfile,
    pub product_name: String,
    /// Validation properties: (volume mm³, area mm², centroid).
    pub validation: Option<(f64, f64, V3)>,
    /// Fixed timestamp for byte-deterministic output.
    pub timestamp: String,
}

impl Default for StepOptions {
    fn default() -> Self {
        StepOptions { schema: Schema::Ap242, profile: ReceiverProfile::Generic, product_name: "part".into(), validation: None, timestamp: "2026-01-01T00:00:00".into() }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExportReport {
    pub entities: usize,
    pub warnings: Vec<String>,
}

/// STEP REAL formatting: shortest round-trip, always with a decimal point, exponent 'E'.
pub fn real(x: f64) -> String {
    if x == 0.0 { return "0.".into(); }
    let s = format!("{x:?}");
    let (mant, exp) = match s.split_once('e') { Some((m, e)) => (m.to_string(), Some(e.to_string())), None => (s, None) };
    let mant = if mant.contains('.') { mant.trim_end_matches('0').to_string() } else { format!("{mant}.") };
    match exp { Some(e) => format!("{mant}E{e}"), None => mant }
}

fn esc(s: &str) -> String { s.replace('\'', "''") }

struct W {
    out: String,
    n: usize,
}

impl W {
    fn e(&mut self, body: impl AsRef<str>) -> usize {
        self.n += 1;
        let _ = writeln!(self.out, "#{}={};", self.n, body.as_ref());
        self.n
    }
    fn pt(&mut self, p: V3) -> usize { self.e(format!("CARTESIAN_POINT('',({},{},{}))", real(p.x), real(p.y), real(p.z))) }
    fn dir(&mut self, d: V3) -> usize { let d = d.normalized(); self.e(format!("DIRECTION('',({},{},{}))", real(d.x), real(d.y), real(d.z))) }
    fn ax2(&mut self, f: &Frame) -> usize {
        let (o, z, x) = (self.pt(f.origin), self.dir(f.z), self.dir(f.x));
        self.e(format!("AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})"))
    }
    fn refs(ids: &[usize]) -> String { ids.iter().map(|i| format!("#{i}")).collect::<Vec<_>>().join(",") }
}

fn knots_mults(knots: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut k = Vec::new();
    let mut m = Vec::new();
    for &x in knots {
        if let Some(last) = k.last() { if (x - last).abs() < 1e-15 { *m.last_mut().unwrap() += 1; continue; } }
        k.push(x);
        m.push(1);
    }
    (k, m)
}

fn list_real(v: &[f64]) -> String { v.iter().map(|x| real(*x)).collect::<Vec<_>>().join(",") }
fn list_int(v: &[usize]) -> String { v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",") }

fn bspline_curve(w: &mut W, n: &NurbsCurve) -> usize {
    let pts: Vec<usize> = n.poles.iter().map(|p| w.pt(*p)).collect();
    let (k, m) = knots_mults(&n.knots);
    let rational = n.weights.iter().any(|x| (*x - 1.0).abs() > 1e-15);
    if !rational {
        return w.e(format!("B_SPLINE_CURVE_WITH_KNOTS('',{},({}),.UNSPECIFIED.,.F.,.F.,({}),({}),.UNSPECIFIED.)", n.degree, W::refs(&pts), list_int(&m), list_real(&k)));
    }
    w.e(format!(
        "(BOUNDED_CURVE() B_SPLINE_CURVE({},({}),.UNSPECIFIED.,.F.,.F.) B_SPLINE_CURVE_WITH_KNOTS(({}),({}),.UNSPECIFIED.) CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE(({})) REPRESENTATION_ITEM(''))",
        n.degree, W::refs(&pts), list_int(&m), list_real(&k), list_real(&n.weights)
    ))
}

fn bspline_surface(w: &mut W, s: &NurbsSurface) -> usize {
    let rows: Vec<String> = (0..s.nu).map(|i| format!("({})", W::refs(&(0..s.nv).map(|j| w.pt(s.poles[i * s.nv + j])).collect::<Vec<_>>()))).collect();
    let (ku, mu) = knots_mults(&s.ku);
    let (kv, mv) = knots_mults(&s.kv);
    let rational = s.weights.iter().any(|x| (*x - 1.0).abs() > 1e-15);
    if !rational {
        return w.e(format!("B_SPLINE_SURFACE_WITH_KNOTS('',{},{},({}),.UNSPECIFIED.,.F.,.F.,.F.,({}),({}),({}),({}),.UNSPECIFIED.)", s.du, s.dv, rows.join(","), list_int(&mu), list_int(&mv), list_real(&ku), list_real(&kv)));
    }
    let wrows: Vec<String> = (0..s.nu).map(|i| format!("({})", list_real(&s.weights[i * s.nv..(i + 1) * s.nv]))).collect();
    w.e(format!(
        "(BOUNDED_SURFACE() B_SPLINE_SURFACE({},{},({}),.UNSPECIFIED.,.F.,.F.,.F.) B_SPLINE_SURFACE_WITH_KNOTS(({}),({}),({}),({}),.UNSPECIFIED.) GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(({})) REPRESENTATION_ITEM('') SURFACE())",
        s.du, s.dv, rows.join(","), list_int(&mu), list_int(&mv), list_real(&ku), list_real(&kv), wrows.join(",")
    ))
}

fn curve(w: &mut W, c: &Curve, t: (f64, f64), report: &mut ExportReport) -> usize {
    match c {
        Curve::Line { origin, dir } => {
            let p = w.pt(*origin);
            let d = w.dir(*dir);
            let v = w.e(format!("VECTOR('',#{d},{})", real(dir.norm())));
            w.e(format!("LINE('',#{p},#{v})"))
        }
        Curve::Circle { frame, r } => { let a = w.ax2(frame); w.e(format!("CIRCLE('',#{a},{})", real(*r))) }
        Curve::Ellipse { frame, a, b } => { let ax = w.ax2(frame); w.e(format!("ELLIPSE('',#{ax},{},{})", real(*a), real(*b))) }
        Curve::Nurbs(n) | Curve::Intersection { fit: n, .. } => bspline_curve(w, n),
        Curve::Helix { .. } => {
            report.warnings.push("helix exported as linear B-spline approximation (64 spans)".into());
            bspline_curve(w, &c.to_nurbs(t))
        }
    }
}

fn surface(w: &mut W, s: &Surface, opt: &StepOptions, report: &mut ExportReport) -> usize {
    match s {
        Surface::Plane { frame } => { let a = w.ax2(frame); w.e(format!("PLANE('',#{a})")) }
        Surface::Cylinder { frame, r } => { let a = w.ax2(frame); w.e(format!("CYLINDRICAL_SURFACE('',#{a},{})", real(*r))) }
        Surface::Cone { frame, r, angle } => {
            // STEP semi-angle must be positive: flip the axis for narrowing cones.
            let (f, ang) = if *angle >= 0.0 { (*frame, *angle) } else { (Frame { origin: frame.origin, x: frame.x, y: -frame.y, z: -frame.z }, -*angle) };
            let a = w.ax2(&f);
            w.e(format!("CONICAL_SURFACE('',#{a},{},{})", real(*r), real(ang)))
        }
        Surface::Sphere { frame, r } => { let a = w.ax2(frame); w.e(format!("SPHERICAL_SURFACE('',#{a},{})", real(*r))) }
        Surface::Torus { frame, major, minor } => { let a = w.ax2(frame); w.e(format!("TOROIDAL_SURFACE('',#{a},{},{})", real(*major), real(*minor))) }
        Surface::Nurbs(n) => bspline_surface(w, n),
        Surface::Extrusion { curve: c, dir } => {
            let ci = curve(w, c, (0.0, 1.0), report);
            let d = w.dir(*dir);
            let v = w.e(format!("VECTOR('',#{d},{})", real(dir.norm())));
            w.e(format!("SURFACE_OF_LINEAR_EXTRUSION('',#{ci},#{v})"))
        }
        Surface::Revolution { curve: c, axis } => {
            if opt.profile == ReceiverProfile::SolidWorks {
                report.warnings.push("solidworks profile: surface of revolution emitted natively (B-spline conversion lands in P11)".into());
            }
            let ci = curve(w, c, (0.0, 1.0), report);
            let (o, z) = (w.pt(axis.origin), w.dir(axis.z));
            let a1 = w.e(format!("AXIS1_PLACEMENT('',#{o},#{z})"));
            w.e(format!("SURFACE_OF_REVOLUTION('',#{ci},#{a1})"))
        }
        Surface::Offset { base, d } => {
            if opt.profile == ReceiverProfile::SolidWorks {
                report.warnings.push("solidworks profile: offset surface emitted natively (B-spline conversion lands in P11)".into());
            }
            let b = surface(w, base, opt, report);
            w.e(format!("OFFSET_SURFACE('',#{b},{},.F.)", real(*d)))
        }
    }
}

/// Encode one body as a complete STEP file.
pub fn write_step(body: &Brep, opt: &StepOptions) -> (String, ExportReport) {
    try_write_step(body, opt).expect("CUDA worker failed during STEP export")
}

pub fn try_write_step(body: &Brep, opt: &StepOptions) -> agcad_driver::DriverResult<(String, ExportReport)> {
    let mut report = ExportReport::default();
    let schema = match opt.schema { Schema::Ap242 => "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }", Schema::Ap214 => "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }" };
    let mut w = W { out: String::new(), n: 0 };
    // --- product structure & context
    let app = w.e(format!("APPLICATION_CONTEXT('{}')", if opt.schema == Schema::Ap242 { "managed model based 3d engineering" } else { "automotive design" }));
    w.e(format!("APPLICATION_PROTOCOL_DEFINITION('international standard','{}',{},#{app})", if opt.schema == Schema::Ap242 { "ap242_managed_model_based_3d_engineering" } else { "automotive_design" }, if opt.schema == Schema::Ap242 { 2020 } else { 2001 }));
    let pctx = w.e(format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
    let prod = w.e(format!("PRODUCT('{0}','{0}','',(#{pctx}))", esc(&opt.product_name)));
    let pdf = w.e(format!("PRODUCT_DEFINITION_FORMATION('','',#{prod})"));
    let pdctx = w.e(format!("PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"));
    let pd = w.e(format!("PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})"));
    let pds = w.e(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    // --- units & geometric context (mm, rad, sr) with uncertainty
    let mm = w.e("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
    let rad = w.e("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
    let sr = w.e("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())");
    let tol = body.edges.iter().map(|e| e.tol).fold(crate::brep::DEFAULT_LINEAR_TOL, f64::max);
    let unc = w.e(format!("UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),#{mm},'distance_accuracy_value','confusion accuracy')", real(tol)));
    let ctx = w.e(format!("(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad},#{sr})) REPRESENTATION_CONTEXT('Context3D','3D Context with UNIT and UNCERTAINTY'))"));
    // --- geometry & topology
    let surf_ids: Vec<usize> = body.surfaces.iter().map(|s| surface(&mut w, s, opt, &mut report)).collect();
    let curve_ids: Vec<usize> = body.edges.iter().map(|e| curve(&mut w, &body.curves[e.curve as usize], (e.t0, e.t1), &mut report)).collect();
    let vtx: Vec<usize> = body.vertices.iter().map(|v| { let p = w.pt(v.p); w.e(format!("VERTEX_POINT('',#{p})")) }).collect();
    let edges: Vec<usize> = body.edges.iter().enumerate().map(|(i, e)| {
        let same = e.t1 >= e.t0;
        w.e(format!("EDGE_CURVE('{}',#{},#{},#{},.{}.)", esc(&e.name), vtx[e.v0 as usize], vtx[e.v1 as usize], curve_ids[i], if same { "T" } else { "F" }))
    }).collect();
    // Voids (closed cavities): shells with negative signed volume, owned by the smallest outer
    // shell whose box contains them. Written as BREP_WITH_VOIDS voids: ORIENTED_CLOSED_SHELL
    // (.F.) over a closed shell oriented outward of the cavity, i.e. the void's faces flipped.
    let nshell = body.shells.len().max(1);
    let mut shell_v6 = vec![0.0f64; nshell];
    let mut shell_box = vec![crate::math::Aabb::EMPTY; nshell];
    if body.kind == BodyKind::Solid && body.shells.len() > 1 {
        let m = crate::tessellate::try_tessellate(body, &Default::default())?;
        let mut origins = vec![None; nshell];
        for (t, &fi) in m.triangles.iter().zip(&m.face_of) {
            let si = body.faces[fi as usize].shell as usize;
            let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
            let origin = *origins[si].get_or_insert(a);
            shell_v6[si] += (a-origin).dot((b-origin).cross(c-origin));
            for q in [a, b, c] { shell_box[si].add(q); }
        }
    }
    let contains = |o: &crate::math::Aabb, i: &crate::math::Aabb| o.min.x <= i.min.x && o.min.y <= i.min.y && o.min.z <= i.min.z && o.max.x >= i.max.x && o.max.y >= i.max.y && o.max.z >= i.max.z;
    let void_owner: Vec<Option<u32>> = (0..nshell).map(|si| {
        if shell_v6[si] >= 0.0 { return None; }
        (0..nshell).filter(|&o| o != si && shell_v6[o] > 0.0 && contains(&shell_box[o], &shell_box[si])).min_by(|&a, &b| shell_v6[a].partial_cmp(&shell_v6[b]).unwrap()).map(|o| o as u32)
    }).collect();
    let mut face_ids = Vec::new();
    for f in &body.faces {
        let flip = void_owner.get(f.shell as usize).copied().flatten().is_some();
        let mut bounds = Vec::new();
        for &li in &f.loops {
            let lp = &body.loops[li as usize];
            let oes: Vec<usize> = lp.coedges.iter().map(|&ce| {
                let c = &body.coedges[ce as usize];
                w.e(format!("ORIENTED_EDGE('',*,*,#{},.{}.)", edges[c.edge as usize], if c.forward { "T" } else { "F" }))
            }).collect();
            let el = w.e(format!("EDGE_LOOP('',({}))", W::refs(&oes)));
            bounds.push(w.e(format!("{}('',#{el},.{}.)", if lp.outer { "FACE_OUTER_BOUND" } else { "FACE_BOUND" }, if flip { "F" } else { "T" })));
        }
        face_ids.push(w.e(format!("ADVANCED_FACE('{}',({}),#{},.{}.)", esc(&f.name), W::refs(&bounds), surf_ids[f.surface as usize], if f.same_sense != flip { "T" } else { "F" })));
    }
    // One CLOSED_SHELL + MANIFOLD_SOLID_BREP per shell (disjoint lumps are separate solids in
    // one representation); sheets keep one OPEN_SHELL per shell in a surface model.
    let shell_of: Vec<u32> = body.faces.iter().map(|f| f.shell).collect();
    let mut shell_ids: Vec<u32> = shell_of.clone();
    shell_ids.sort_unstable();
    shell_ids.dedup();
    let faces_in = |si: u32| -> Vec<usize> { face_ids.iter().zip(&shell_of).filter(|(_, s)| **s == si).map(|(f, _)| *f).collect() };
    let items: Vec<usize> = match body.kind {
        BodyKind::Solid => {
            let outers: Vec<u32> = shell_ids.iter().copied().filter(|&si| void_owner.get(si as usize).copied().flatten().is_none()).collect();
            outers.iter().enumerate().map(|(k, &si)| {
                let sh = w.e(format!("CLOSED_SHELL('',({}))", W::refs(&faces_in(si))));
                let name = if outers.len() == 1 { esc(&opt.product_name) } else { format!("{}.{}", esc(&opt.product_name), k + 1) };
                let voids: Vec<usize> = shell_ids.iter().copied().filter(|&v| void_owner.get(v as usize).copied().flatten() == Some(si)).map(|v| {
                    let cs = w.e(format!("CLOSED_SHELL('',({}))", W::refs(&faces_in(v))));
                    w.e(format!("ORIENTED_CLOSED_SHELL('',*,#{cs},.F.)"))
                }).collect();
                if voids.is_empty() { w.e(format!("MANIFOLD_SOLID_BREP('{name}',#{sh})")) } else { w.e(format!("BREP_WITH_VOIDS('{name}',#{sh},({}))", W::refs(&voids))) }
            }).collect()
        }
        _ => {
            let shs: Vec<usize> = shell_ids.iter().map(|&si| w.e(format!("OPEN_SHELL('',({}))", W::refs(&faces_in(si))))).collect();
            vec![w.e(format!("SHELL_BASED_SURFACE_MODEL('',({}))", W::refs(&shs)))]
        }
    };
    let rep_kind = if body.kind == BodyKind::Solid { "ADVANCED_BREP_SHAPE_REPRESENTATION" } else { "MANIFOLD_SURFACE_SHAPE_REPRESENTATION" };
    let origin = w.ax2(&Frame::WORLD);
    let rep = w.e(format!("{rep_kind}('',({},#{origin}),#{ctx})", W::refs(&items)));
    w.e(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})"));
    // --- validation properties (CAx-IF recommended practice structure)
    if let Some((vol, area, c)) = opt.validation {
        let de3 = w.e(format!("DERIVED_UNIT_ELEMENT(#{mm},3.)"));
        let vu = w.e(format!("(DERIVED_UNIT((#{de3})) VOLUME_UNIT())"));
        let de2 = w.e(format!("DERIVED_UNIT_ELEMENT(#{mm},2.)"));
        let au = w.e(format!("(AREA_UNIT() DERIVED_UNIT((#{de2})))"));
        let mv = w.e(format!("MEASURE_REPRESENTATION_ITEM('volume measure',VOLUME_MEASURE({}),#{vu})", real(vol)));
        let ma = w.e(format!("MEASURE_REPRESENTATION_ITEM('surface area measure',AREA_MEASURE({}),#{au})", real(area)));
        let cp = w.pt(c);
        let crep = w.e(format!("REPRESENTATION('centroid',(#{cp}),#{ctx})"));
        let vrep = w.e(format!("REPRESENTATION('volume',(#{mv}),#{ctx})"));
        let arep = w.e(format!("REPRESENTATION('surface area',(#{ma}),#{ctx})"));
        for (label, r) in [("centre point", crep), ("volume", vrep), ("surface area", arep)] {
            let pdef = w.e(format!("PROPERTY_DEFINITION('geometric validation property','{label}',#{pds})"));
            w.e(format!("PROPERTY_DEFINITION_REPRESENTATION(#{pdef},#{r})"));
        }
    }
    report.entities = w.n;
    let header = format!(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('AgCAD export'),'2;1');\nFILE_NAME('{}.stp','{}',(''),(''),'AgCAD','AgCAD','');\nFILE_SCHEMA(('{}'));\nENDSEC;\nDATA;\n",
        esc(&opt.product_name), opt.timestamp, schema
    );
    Ok((format!("{header}{}ENDSEC;\nEND-ISO-10303-21;\n", w.out), report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{prism, Profile};
    #[test]
    fn deterministic_and_well_formed() {
        let b = prism(&Profile::rectangle(0.0, 0.0, 10.0, 5.0).with_hole(Profile::circle(5.0, 2.5, 1.0)), &Frame::WORLD, 2.0);
        let (a, r) = write_step(&b, &StepOptions::default());
        let (a2, _) = write_step(&b, &StepOptions::default());
        assert_eq!(a, a2, "byte-deterministic");
        assert!(a.contains("ADVANCED_FACE('cap_end'"));
        assert!(a.contains("CYLINDRICAL_SURFACE"));
        assert!(r.entities > 50);
        assert_eq!(real(1.0), "1.");
        assert_eq!(real(1e-7), "1.E-7");
        assert_eq!(real(-2.5), "-2.5");
    }
}
