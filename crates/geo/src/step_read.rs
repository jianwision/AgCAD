//! STEP (ISO 10303-21) import: lexer → entity index → typed decode → B-rep build (spec 18 §
//! 3.1). CPU reference of the GPU import pipeline; limits per spec 17 (size, entity count,
//! reference depth, string length).

use crate::brep::{BodyKind, Brep};
use crate::curve::Curve;
use crate::math::{Frame, V3};
use crate::nurbs::{NurbsCurve, NurbsSurface};
use crate::surface::Surface;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ref(u64),
    Int(i64),
    Real(f64),
    Str(String),
    Enum(String),
    Kw(String),
    LParen,
    RParen,
    Comma,
    Dollar,
    Star,
    Eq,
    Semi,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Ref(u64),
    Num(f64),
    Str(String),
    Enum(String),
    List(Vec<Arg>),
    /// Typed parameter like `LENGTH_MEASURE(1.0)`.
    Typed(String, Box<Arg>),
    Null,
}

impl Arg {
    pub fn num(&self) -> Option<f64> { match self { Arg::Num(x) => Some(*x), Arg::Typed(_, a) => a.num(), _ => None } }
    pub fn r(&self) -> Option<u64> { if let Arg::Ref(r) = self { Some(*r) } else { None } }
    pub fn list(&self) -> &[Arg] { if let Arg::List(l) = self { l } else { &[] } }
    pub fn s(&self) -> &str { match self { Arg::Str(s) | Arg::Enum(s) => s, _ => "" } }
}

/// One entity: either simple `TYPE(args)` or complex `(A(..) B(..))` (parts by type name).
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub parts: Vec<(String, Vec<Arg>)>,
}

impl Entity {
    pub fn ty(&self) -> &str { if self.parts.len() == 1 { &self.parts[0].0 } else { "COMPLEX" } }
    pub fn part(&self, name: &str) -> Option<&Vec<Arg>> { self.parts.iter().find(|p| p.0 == name).map(|p| &p.1) }
    pub fn args(&self) -> &[Arg] { &self.parts[0].1 }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportLimits {
    pub max_bytes: usize,
    pub max_entities: usize,
    pub max_string: usize,
    pub max_depth: usize,
}

impl Default for ImportLimits {
    fn default() -> Self { ImportLimits { max_bytes: 2 << 30, max_entities: 50_000_000, max_string: 65_536, max_depth: 64 } }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportReport {
    pub entities: usize,
    pub bodies: usize,
    pub length_scale: f64,
    pub schema: String,
    pub warnings: Vec<String>,
    /// (property, file value, recomputed value) for validation properties found.
    pub validation: Vec<(String, f64, f64)>,
}

pub fn lex(src: &str, lim: &ImportLimits) -> Result<Vec<Tok>, String> {
    if src.len() > lim.max_bytes { return Err(format!("file exceeds {} bytes", lim.max_bytes)); }
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\n' | b'\r' | b'\t' => i += 1,
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') { i += 1; }
                i += 2;
            }
            b'#' => {
                let s = i + 1;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() { i += 1; }
                out.push(Tok::Ref(src[s..i].parse().map_err(|_| "bad #ref")?));
            }
            b'\'' => {
                let mut s = String::new();
                i += 1;
                loop {
                    if i >= b.len() { return Err("unterminated string".into()); }
                    if b[i] == b'\'' {
                        if i + 1 < b.len() && b[i + 1] == b'\'' { s.push('\''); i += 2; continue; }
                        i += 1;
                        break;
                    }
                    s.push(b[i] as char);
                    i += 1;
                    if s.len() > lim.max_string { return Err("string too long".into()); }
                }
                out.push(Tok::Str(s));
            }
            b'.' if i + 1 < b.len() && (b[i + 1].is_ascii_alphabetic() || b[i + 1] == b'_') => {
                let s = i + 1;
                i += 1;
                while i < b.len() && b[i] != b'.' { i += 1; }
                out.push(Tok::Enum(src[s..i].to_string()));
                i += 1;
            }
            b'(' => { out.push(Tok::LParen); i += 1; }
            b')' => { out.push(Tok::RParen); i += 1; }
            b',' => { out.push(Tok::Comma); i += 1; }
            b'$' => { out.push(Tok::Dollar); i += 1; }
            b'*' => { out.push(Tok::Star); i += 1; }
            b'=' => { out.push(Tok::Eq); i += 1; }
            b';' => { out.push(Tok::Semi); i += 1; }
            c if c == b'-' || c == b'+' || c.is_ascii_digit() || c == b'.' => {
                let s = i;
                i += 1;
                while i < b.len() && (b[i].is_ascii_digit() || matches!(b[i], b'.' | b'E' | b'e' | b'-' | b'+')) { i += 1; }
                let t = &src[s..i];
                if t.contains(['.', 'E', 'e']) {
                    let fixed = if t.ends_with('.') { format!("{t}0") } else { t.replace(".E", ".0E").replace(".e", ".0e") };
                    out.push(Tok::Real(fixed.parse().map_err(|_| format!("bad real '{t}'"))?));
                } else {
                    out.push(Tok::Int(t.parse().map_err(|_| format!("bad int '{t}'"))?));
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' || c == b'!' => {
                let s = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'-' || b[i] == b'!') { i += 1; }
                out.push(Tok::Kw(src[s..i].to_string()));
            }
            _ => i += 1,
        }
    }
    Ok(out)
}

fn parse_arg(t: &[Tok], i: &mut usize, depth: usize, lim: &ImportLimits) -> Result<Arg, String> {
    if depth > lim.max_depth { return Err("nesting too deep".into()); }
    let a = match t.get(*i) {
        Some(Tok::Ref(r)) => { *i += 1; Arg::Ref(*r) }
        Some(Tok::Int(v)) => { *i += 1; Arg::Num(*v as f64) }
        Some(Tok::Real(v)) => { *i += 1; Arg::Num(*v) }
        Some(Tok::Str(s)) => { *i += 1; Arg::Str(s.clone()) }
        Some(Tok::Enum(s)) => { *i += 1; Arg::Enum(s.clone()) }
        Some(Tok::Dollar) | Some(Tok::Star) => { *i += 1; Arg::Null }
        Some(Tok::LParen) => { *i += 1; Arg::List(parse_list(t, i, depth + 1, lim)?) }
        Some(Tok::Kw(k)) => {
            let k = k.clone();
            *i += 1;
            if t.get(*i) == Some(&Tok::LParen) { *i += 1; let mut l = parse_list(t, i, depth + 1, lim)?; Arg::Typed(k, Box::new(if l.len() == 1 { l.remove(0) } else { Arg::List(l) })) } else { Arg::Enum(k) }
        }
        other => return Err(format!("unexpected token {other:?}")),
    };
    Ok(a)
}

fn parse_list(t: &[Tok], i: &mut usize, depth: usize, lim: &ImportLimits) -> Result<Vec<Arg>, String> {
    let mut out = Vec::new();
    loop {
        match t.get(*i) {
            Some(Tok::RParen) => { *i += 1; return Ok(out); }
            Some(Tok::Comma) => { *i += 1; }
            None => return Err("unterminated list".into()),
            _ => out.push(parse_arg(t, i, depth, lim)?),
        }
    }
}

/// Index the DATA section: `#id = ...;`.
pub fn index(t: &[Tok], lim: &ImportLimits) -> Result<(HashMap<u64, Entity>, String), String> {
    let mut map = HashMap::new();
    let mut schema = String::new();
    let mut i = 0;
    while i < t.len() {
        if let Tok::Kw(k) = &t[i] {
            if k == "FILE_SCHEMA" {
                let mut j = i;
                while j < t.len() && !matches!(t[j], Tok::Str(_)) { j += 1; }
                if let Some(Tok::Str(s)) = t.get(j) { schema = s.clone(); }
            }
        }
        if let (Tok::Ref(id), Some(Tok::Eq)) = (&t[i], t.get(i + 1)) {
            let id = *id;
            i += 2;
            let mut parts = Vec::new();
            match t.get(i) {
                Some(Tok::Kw(k)) => {
                    let k = k.clone();
                    i += 1;
                    if t.get(i) != Some(&Tok::LParen) { return Err(format!("#{id}: expected '('")); }
                    i += 1;
                    parts.push((k, parse_list(t, &mut i, 1, lim)?));
                }
                Some(Tok::LParen) => {
                    i += 1;
                    while let Some(Tok::Kw(k)) = t.get(i) {
                        let k = k.clone();
                        i += 1;
                        if t.get(i) == Some(&Tok::LParen) { i += 1; parts.push((k, parse_list(t, &mut i, 1, lim)?)); } else { parts.push((k, vec![])); }
                    }
                    if t.get(i) == Some(&Tok::RParen) { i += 1; }
                }
                other => return Err(format!("#{id}: unexpected {other:?}")),
            }
            map.insert(id, Entity { parts });
            if map.len() > lim.max_entities { return Err("too many entities".into()); }
            while i < t.len() && t[i] != Tok::Semi { i += 1; }
        }
        i += 1;
    }
    Ok((map, schema))
}

struct Ctx<'a> {
    m: &'a HashMap<u64, Entity>,
    scale: f64,
}

impl<'a> Ctx<'a> {
    fn get(&self, r: u64) -> Result<&Entity, String> { self.m.get(&r).ok_or_else(|| format!("dangling reference #{r}")) }
    fn point(&self, r: u64) -> Result<V3, String> {
        let e = self.get(r)?;
        let c = e.args().get(1).ok_or("CARTESIAN_POINT coords")?.list();
        let v = |k: usize| c.get(k).and_then(Arg::num).unwrap_or(0.0) * self.scale;
        Ok(V3 { x: v(0), y: v(1), z: v(2) })
    }
    fn dir(&self, r: u64) -> Result<V3, String> {
        let e = self.get(r)?;
        let c = e.args().get(1).ok_or("DIRECTION ratios")?.list();
        let v = |k: usize| c.get(k).and_then(Arg::num).unwrap_or(0.0);
        Ok(V3 { x: v(0), y: v(1), z: v(2) }.normalized())
    }
    fn ax2(&self, r: u64) -> Result<Frame, String> {
        let a = self.get(r)?.args();
        let o = self.point(a[1].r().ok_or("ax2 origin")?)?;
        let z = match a.get(2).and_then(Arg::r) { Some(z) => self.dir(z)?, None => V3::Z };
        let x = match a.get(3).and_then(Arg::r) { Some(x) => self.dir(x)?, None => z.any_perp() };
        Ok(Frame::new(o, z, x))
    }
    fn len(&self, a: &Arg) -> f64 { a.num().unwrap_or(0.0) * self.scale }

    fn curve(&self, r: u64) -> Result<Curve, String> {
        let e = self.get(r)?;
        match e.ty() {
            "LINE" => {
                let a = e.args();
                let p = self.point(a[1].r().ok_or("line pnt")?)?;
                let v = self.get(a[2].r().ok_or("line vec")?)?.args();
                let d = self.dir(v[1].r().ok_or("vec dir")?)?;
                Ok(Curve::Line { origin: p, dir: d * self.len(&v[2]) })
            }
            "CIRCLE" => { let a = e.args(); Ok(Curve::Circle { frame: self.ax2(a[1].r().ok_or("circle pos")?)?, r: self.len(&a[2]) }) }
            "ELLIPSE" => { let a = e.args(); Ok(Curve::Ellipse { frame: self.ax2(a[1].r().ok_or("ellipse pos")?)?, a: self.len(&a[2]), b: self.len(&a[3]) }) }
            "B_SPLINE_CURVE_WITH_KNOTS" => {
                let a = e.args();
                Ok(Curve::Nurbs(self.bcurve(a[1].num().unwrap_or(1.0) as usize, a[2].list(), a[6].list(), a[7].list(), None)?))
            }
            "COMPLEX" if e.part("B_SPLINE_CURVE").is_some() => {
                let bc = e.part("B_SPLINE_CURVE").unwrap();
                let bk = e.part("B_SPLINE_CURVE_WITH_KNOTS").ok_or("missing knots")?;
                let w = e.part("RATIONAL_B_SPLINE_CURVE").map(|p| p[0].list().iter().filter_map(Arg::num).collect::<Vec<_>>());
                Ok(Curve::Nurbs(self.bcurve(bc[0].num().unwrap_or(1.0) as usize, bc[1].list(), bk[0].list(), bk[1].list(), w)?))
            }
            "SURFACE_CURVE" | "SEAM_CURVE" => self.curve(e.args()[1].r().ok_or("surface_curve 3d")?),
            "TRIMMED_CURVE" => self.curve(e.args()[1].r().ok_or("trimmed basis")?),
            other => Err(format!("unsupported curve type {other} (#{r})")),
        }
    }

    fn bcurve(&self, degree: usize, pts: &[Arg], mults: &[Arg], knots: &[Arg], w: Option<Vec<f64>>) -> Result<NurbsCurve, String> {
        let poles: Vec<V3> = pts.iter().map(|p| self.point(p.r().ok_or("pole ref")?)).collect::<Result<_, _>>()?;
        let mut kv = Vec::new();
        for (m, k) in mults.iter().zip(knots) { for _ in 0..m.num().unwrap_or(1.0) as usize { kv.push(k.num().unwrap_or(0.0)); } }
        let n = NurbsCurve { degree, knots: kv, weights: w.unwrap_or_else(|| vec![1.0; poles.len()]), poles };
        n.validate().map_err(|e| format!("invalid B-spline curve: {e}"))?;
        Ok(n)
    }

    fn surface(&self, r: u64) -> Result<Surface, String> {
        let e = self.get(r)?;
        match e.ty() {
            "PLANE" => Ok(Surface::Plane { frame: self.ax2(e.args()[1].r().ok_or("plane pos")?)? }),
            "CYLINDRICAL_SURFACE" => Ok(Surface::Cylinder { frame: self.ax2(e.args()[1].r().ok_or("pos")?)?, r: self.len(&e.args()[2]) }),
            "CONICAL_SURFACE" => Ok(Surface::Cone { frame: self.ax2(e.args()[1].r().ok_or("pos")?)?, r: self.len(&e.args()[2]), angle: e.args()[3].num().unwrap_or(0.0) }),
            "SPHERICAL_SURFACE" => Ok(Surface::Sphere { frame: self.ax2(e.args()[1].r().ok_or("pos")?)?, r: self.len(&e.args()[2]) }),
            "TOROIDAL_SURFACE" => Ok(Surface::Torus { frame: self.ax2(e.args()[1].r().ok_or("pos")?)?, major: self.len(&e.args()[2]), minor: self.len(&e.args()[3]) }),
            "B_SPLINE_SURFACE_WITH_KNOTS" => {
                let a = e.args();
                self.bsurf(a[1].num().unwrap_or(1.0) as usize, a[2].num().unwrap_or(1.0) as usize, a[3].list(), &a[8..12], None)
            }
            "COMPLEX" if e.part("B_SPLINE_SURFACE").is_some() => {
                let bs = e.part("B_SPLINE_SURFACE").unwrap();
                let bk = e.part("B_SPLINE_SURFACE_WITH_KNOTS").ok_or("missing knots")?;
                let w = e.part("RATIONAL_B_SPLINE_SURFACE").map(|p| p[0].list().iter().flat_map(|row| row.list().iter().filter_map(Arg::num).collect::<Vec<_>>()).collect::<Vec<_>>());
                self.bsurf(bs[0].num().unwrap_or(1.0) as usize, bs[1].num().unwrap_or(1.0) as usize, bs[2].list(), &bk[0..4], w)
            }
            "SURFACE_OF_LINEAR_EXTRUSION" => {
                let a = e.args();
                let c = self.curve(a[1].r().ok_or("swept curve")?)?;
                let v = self.get(a[2].r().ok_or("vector")?)?.args();
                Ok(Surface::Extrusion { curve: Box::new(c), dir: self.dir(v[1].r().ok_or("dir")?)? * self.len(&v[2]) })
            }
            "SURFACE_OF_REVOLUTION" => {
                let a = e.args();
                let c = self.curve(a[1].r().ok_or("swept curve")?)?;
                let ax = self.get(a[2].r().ok_or("axis")?)?.args();
                let o = self.point(ax[1].r().ok_or("axis origin")?)?;
                let z = self.dir(ax[2].r().ok_or("axis dir")?)?;
                Ok(Surface::Revolution { curve: Box::new(c), axis: Frame::new(o, z, z.any_perp()) })
            }
            "OFFSET_SURFACE" => Ok(Surface::Offset { base: Box::new(self.surface(e.args()[1].r().ok_or("base")?)?), d: self.len(&e.args()[2]) }),
            other => Err(format!("unsupported surface type {other} (#{r})")),
        }
    }

    fn bsurf(&self, du: usize, dv: usize, rows: &[Arg], kn: &[Arg], w: Option<Vec<f64>>) -> Result<Surface, String> {
        let nu = rows.len();
        let nv = rows.first().map(|r| r.list().len()).unwrap_or(0);
        let mut poles = Vec::with_capacity(nu * nv);
        for row in rows { for p in row.list() { poles.push(self.point(p.r().ok_or("pole")?)?); } }
        let expand = |m: &Arg, k: &Arg| -> Vec<f64> { let mut v = Vec::new(); for (mm, kk) in m.list().iter().zip(k.list()) { for _ in 0..mm.num().unwrap_or(1.0) as usize { v.push(kk.num().unwrap_or(0.0)); } } v };
        let s = NurbsSurface { du, dv, ku: expand(&kn[0], &kn[2]), kv: expand(&kn[1], &kn[3]), nu, nv, weights: w.unwrap_or_else(|| vec![1.0; nu * nv]), poles };
        s.validate().map_err(|e| format!("invalid B-spline surface: {e}"))?;
        Ok(Surface::Nurbs(s))
    }
}

fn length_scale(m: &HashMap<u64, Entity>) -> f64 {
    for e in m.values() {
        if e.part("LENGTH_UNIT").is_some() {
            if let Some(si) = e.part("SI_UNIT") {
                let prefix = si.first().map(|a| a.s().to_string()).unwrap_or_default();
                return match prefix.as_str() { "MILLI" => 1.0, "CENTI" => 10.0, "MICRO" => 1e-3, "KILO" => 1e6, _ => 1000.0 };
            }
            if let Some(cb) = e.part("CONVERSION_BASED_UNIT") {
                let name = cb.first().map(|a| a.s().to_uppercase()).unwrap_or_default();
                return match name.as_str() { "INCH" => 25.4, "FOOT" => 304.8, _ => 1.0 };
            }
        }
    }
    1.0
}

/// Import all MANIFOLD_SOLID_BREP / shell-based bodies.
pub fn read_step(src: &str, lim: &ImportLimits) -> Result<(Vec<Brep>, ImportReport), String> {
    let toks = lex(src, lim)?;
    let (m, schema) = index(&toks, lim)?;
    let scale = length_scale(&m);
    let cx = Ctx { m: &m, scale };
    let mut report = ImportReport { entities: m.len(), length_scale: scale, schema, ..Default::default() };
    let mut ids: Vec<&u64> = m.keys().collect();
    ids.sort();
    let mut bodies = Vec::new();
    for &id in &ids {
        let e = &m[id];
        let (shells, kind): (Vec<u64>, BodyKind) = match e.ty() {
            "MANIFOLD_SOLID_BREP" => (vec![e.args()[1].r().ok_or("msb shell")?], BodyKind::Solid),
            "BREP_WITH_VOIDS" => (std::iter::once(e.args()[1].r().ok_or("outer")?).chain(e.args()[2].list().iter().filter_map(Arg::r)).collect(), BodyKind::Solid),
            "SHELL_BASED_SURFACE_MODEL" => (e.args()[1].list().iter().filter_map(Arg::r).collect(), BodyKind::Sheet),
            _ => continue,
        };
        let mut b = Brep::new(kind);
        let mut vmap: HashMap<u64, u32> = HashMap::new();
        let mut emap: HashMap<u64, u32> = HashMap::new();
        let mut smap: HashMap<u64, u32> = HashMap::new();
        for sh in shells {
            let mut shell_e = cx.get(sh)?;
            // void shells: ORIENTED_CLOSED_SHELL('',*,#closed_shell,.F.) reverses the shell
            let mut flip = false;
            if shell_e.ty() == "ORIENTED_CLOSED_SHELL" {
                flip = shell_e.args()[3].s() == "F";
                shell_e = cx.get(shell_e.args()[2].r().ok_or("oriented shell")?)?;
            }
            let closed = shell_e.ty() == "CLOSED_SHELL";
            let si = b.add_shell(closed);
            for fref in shell_e.args()[1].list().iter().filter_map(Arg::r) {
                let fe = cx.get(fref)?;
                if fe.ty() != "ADVANCED_FACE" && fe.ty() != "FACE_SURFACE" { report.warnings.push(format!("#{fref}: face type {} skipped", fe.ty())); continue; }
                let fa = fe.args();
                let name = fa[0].s().to_string();
                let sref = fa[2].r().ok_or("face surface")?;
                let surf_i = match smap.get(&sref) { Some(&i) => i, None => { let s = cx.surface(sref)?; let i = b.add_surface(s); smap.insert(sref, i); i } };
                let same = (fa[3].s() == "T") != flip;
                let mut loops: Vec<(bool, Vec<(u32, bool)>)> = Vec::new();
                for bref in fa[1].list().iter().filter_map(Arg::r) {
                    let be = cx.get(bref)?;
                    let outer = be.ty() == "FACE_OUTER_BOUND";
                    let lref = be.args()[1].r().ok_or("bound loop")?;
                    let bound_sense = (be.args()[2].s() == "T") != flip;
                    let le = cx.get(lref)?;
                    if le.ty() == "VERTEX_LOOP" { continue; } // poles
                    let mut ces = Vec::new();
                    for oref in le.args()[1].list().iter().filter_map(Arg::r) {
                        let oa = cx.get(oref)?.args();
                        let eref = oa[3].r().ok_or("oriented edge")?;
                        let fwd = oa[4].s() == "T";
                        let ei = match emap.get(&eref) {
                            Some(&i) => i,
                            None => {
                                let ea = cx.get(eref)?.args();
                                let mut vtx = |vr: u64| -> Result<u32, String> {
                                    if let Some(&v) = vmap.get(&vr) { return Ok(v); }
                                    let p = cx.point(cx.get(vr)?.args()[1].r().ok_or("vertex point")?)?;
                                    let v = b.add_vertex(p);
                                    vmap.insert(vr, v);
                                    Ok(v)
                                };
                                let (v0, v1) = (vtx(ea[1].r().ok_or("edge start")?)?, vtx(ea[2].r().ok_or("edge end")?)?);
                                let c = cx.curve(ea[3].r().ok_or("edge curve")?)?;
                                let same_sense = ea[4].s() == "T";
                                let (p0, p1) = (b.vertices[v0 as usize].p, b.vertices[v1 as usize].p);
                                let range = match &c { Curve::Nurbs(n) => n.domain(), _ => (-1e12, 1e12) };
                                let mut t0 = c.closest(p0, range.0, range);
                                let mut t1 = c.closest(p1, range.1, range);
                                if let Curve::Circle { .. } | Curve::Ellipse { .. } = c {
                                    // closed curves: choose the sweep matching the sense
                                    let tau = core::f64::consts::TAU;
                                    if same_sense { while t1 <= t0 + 1e-12 { t1 += tau; } } else { while t1 >= t0 - 1e-12 { t1 -= tau; } }
                                    if v0 == v1 { t1 = if same_sense { t0 + tau } else { t0 - tau }; }
                                } else if !same_sense && t1 > t0 { core::mem::swap(&mut t0, &mut t1); }
                                let ci = b.add_curve(c);
                                let ename = ea[0].s().to_string();
                                let i = b.add_edge(ci, v0, v1, t0, t1, ename);
                                emap.insert(eref, i);
                                i
                            }
                        };
                        ces.push((ei, fwd));
                    }
                    if !bound_sense { ces = ces.into_iter().rev().map(|(e, f)| (e, !f)).collect(); }
                    loops.push((outer, ces));
                }
                if loops.is_empty() { continue; }
                let oi = loops.iter().position(|l| l.0).unwrap_or(0);
                let outer = loops.remove(oi).1;
                let inner: Vec<Vec<(u32, bool)>> = loops.into_iter().map(|l| l.1).collect();
                b.add_face(surf_i, same, si, &outer, &inner, name, [0.0, core::f64::consts::TAU, -1e9, 1e9]);
            }
        }
        bodies.push(b);
    }
    report.bodies = bodies.len();
    // validation properties present in the file
    for &id in &ids {
        let e = &m[id];
        if e.ty() == "MEASURE_REPRESENTATION_ITEM" {
            if let Some(v) = e.args().get(1).and_then(Arg::num) {
                report.validation.push((e.args()[0].s().to_string(), v * if e.args()[0].s().contains("volume") { scale.powi(3) } else { scale * scale }, f64::NAN));
            }
        }
    }
    Ok((bodies, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{prism, Profile};
    use crate::step_write::{write_step, StepOptions};
    use crate::tessellate::{tessellate, TessOptions};

    #[test]
    fn roundtrip_plate_with_hole() {
        let b = prism(&Profile::rectangle(0.0, 0.0, 40.0, 20.0).with_hole(Profile::circle(20.0, 10.0, 4.0)), &Frame::WORLD, 5.0);
        let m0 = tessellate(&b, &TessOptions { sag: 1e-4, ..Default::default() });
        let (v0, a0, c0) = m0.mass_properties();
        let (txt, _) = write_step(&b, &StepOptions { validation: Some((v0, a0, c0)), ..Default::default() });
        let (bodies, rep) = read_step(&txt, &ImportLimits::default()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(rep.length_scale, 1.0);
        let r = bodies[0].validate();
        assert!(r.valid, "{:?}", r.problems);
        assert!(bodies[0].face_by_name("cap_end").is_some(), "names round-trip (AG-18-006)");
        let m1 = tessellate(&bodies[0], &TessOptions { sag: 1e-4, ..Default::default() });
        let (v1, _, _) = m1.mass_properties();
        assert!((v1 - v0).abs() / v0 < 1e-6, "{v1} vs {v0}");
        assert!(rep.validation.iter().any(|(n, v, _)| n.contains("volume") && (v - v0).abs() / v0 < 1e-9));
    }
}
