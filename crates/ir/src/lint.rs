//! `check_program` (spec 07 MCP tool): lint an AgIR JSON program without running it and
//! report **every** problem at once, each with a JSON path, the expected signature and a
//! concrete fix. Errors make `run_program` fail; warnings never do.
//!
//! Checks: JSON shape, unknown/ambiguous ops (with suggestions), unknown argument names
//! (closest catalog input), missing required inputs (example value by type), formula syntax,
//! the [`crate::verify`] rules (SSA order, finiteness, unit dimensions), bare strings that name a
//! parameter (should be `"$name"`), and a size preview of `study.configurations`.

use crate::catalog::{levenshtein, Catalog};
use crate::json::decode_value;
use crate::program::{Program, Stmt, Value};
use serde::Serialize;
use serde_json::Value as J;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity { Error, Warning }

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Lint {
    pub severity: Severity,
    pub path: String,
    /// AgStatus name the problem maps to (errors) or `ADVICE` (warnings).
    pub status: String,
    pub message: String,
    /// Concrete replacement or addition, e.g. `"limits": "10 mm"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// Signature of the op, `name(input:Type, ...)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LintReport {
    pub ok: bool,
    pub errors: usize,
    pub warnings: usize,
    pub diagnostics: Vec<Lint>,
    /// Size preview: statements, configurations implied by `study.configurations`.
    pub preview: J,
}

/// Example JSON value for a catalog type (used in fixes).
pub fn example_for(ty: &str) -> String {
    let base = ty.trim_end_matches('?');
    if let Some(inner) = base.strip_suffix("[]") { return format!("[{}]", example_for(inner)); }
    match base.split('|').next().unwrap_or(base) {
        "Length" => "\"10 mm\"".into(),
        "Angle" => "\"90 deg\"".into(),
        "Mass" => "\"1 kg\"".into(),
        "Int" | "Count" => "1".into(),
        "Scalar" | "Real" | "Float" => "1.0".into(),
        "Bool" => "true".into(),
        "String" | "Enum" | "Branch" | "Workspace" => "\"...\"".into(),
        "Vec2" => "[\"0 mm\", \"0 mm\"]".into(),
        "Vec3" => "[\"0 mm\", \"0 mm\", \"0 mm\"]".into(),
        "Limits" => "\"10 mm\"".into(),
        "Interval" => "[\"0 mm\", \"10 mm\"]".into(),
        t if t.starts_with("Configurations") => "\"%cfg\"".into(),
        _ => "\"%<statement id>\"".into(),
    }
}

fn signature(op: &crate::catalog::OpManifest) -> String { format!("{}({})", op.name, op.inputs.join(", ")) }

/// Number of configurations a `study.configurations` table expands to (grid product or rows).
fn configurations_of(table: &J) -> Option<u64> {
    if let Some(rows) = table.get("rows").and_then(J::as_array) { return Some(rows.len() as u64); }
    let grid = table.get("grid").unwrap_or(table).as_object()?;
    let mut n: u64 = 1;
    for v in grid.values() {
        let k = match v {
            J::Array(a) if a.len() == 3 && a[2].as_u64().is_some_and(|x| x > 1) => a[2].as_u64().unwrap(),
            J::Array(a) => a.len() as u64,
            J::Object(o) => o.get("values").and_then(J::as_array).map(|a| a.len() as u64).unwrap_or(1),
            _ => 1,
        };
        n = n.saturating_mul(k.max(1));
    }
    Some(n)
}

/// Lint a program given as JSON text.
pub fn check_program(text: &str, catalog: &Catalog) -> LintReport {
    check_program_in_workspace(text, catalog, |_| false)
}

pub fn check_program_in_workspace(text: &str, catalog: &Catalog, exists: impl Fn(&str) -> bool) -> LintReport {
    let mut out: Vec<Lint> = Vec::new();
    let mut push = |severity, path: String, status: &str, message: String, fix: Option<String>, signature: Option<String>| out.push(Lint { severity, path, status: status.into(), message, fix, signature });
    let mut preview = serde_json::json!({});
    let root: J = match serde_json::from_str(text) {
        Ok(j) => j,
        Err(e) => {
            push(Severity::Error, "$".into(), "INVALID_PARAMETER", format!("not valid JSON: {e}"), None, None);
            return finish(out, preview);
        }
    };
    for (k, ex) in [("requirements_rev", "3"), ("expected_revision", "0")] {
        if root.get(k).and_then(J::as_u64).is_none() {
            push(Severity::Error, format!("$.{k}"), "INVALID_PARAMETER", format!("'{k}' is required (non-negative integer)"), Some(format!("\"{k}\": {ex}")), None);
        }
    }
    let Some(stmts) = root.get("program").and_then(J::as_array) else {
        push(Severity::Error, "$.program".into(), "INVALID_PARAMETER", "'program' must be an array of statements".into(), Some("\"program\": [{\"id\": \"p\", \"op\": \"document.edit\", \"args\": {\"branch\": \"main\"}}]".into()), None);
        return finish(out, preview);
    };
    let mut decoded: Vec<Stmt> = Vec::new();
    let mut json_index: BTreeMap<String, usize> = BTreeMap::new();
    // statements that failed to decode: references to them are follow-on errors, not reported
    let mut failed: BTreeSet<String> = BTreeSet::new();
    let mut params_declared: BTreeSet<String> = BTreeSet::new();
    let mut configurations: Option<u64> = None;
    for (i, s) in stmts.iter().enumerate() {
        let path = format!("$.program[{i}]");
        let Some(so) = s.as_object() else {
            push(Severity::Error, path, "INVALID_PARAMETER", "statement must be an object {id, op, args}".into(), None, None);
            continue;
        };
        let id = match so.get("id").and_then(J::as_str) {
            Some(id) => id.to_string(),
            None => { push(Severity::Error, format!("{path}.id"), "INVALID_PARAMETER", "statement 'id' is required".into(), Some(format!("\"id\": \"s{i}\"")), None); format!("s{i}") }
        };
        let Some(raw_op) = so.get("op").and_then(J::as_str) else {
            push(Severity::Error, format!("{path}.op"), "INVALID_PARAMETER", "statement 'op' is required".into(), None, None);
            continue;
        };
        let cands = catalog.resolve(raw_op);
        let op = match cands.as_slice() {
            [one] => *one,
            [] => {
                let sug = catalog.suggest(raw_op, 3);
                push(Severity::Error, format!("{path}.op"), "INVALID_OPCODE", format!("unknown op '{raw_op}'"), sug.first().map(|s| format!("\"op\": \"{s}\"")), None);
                failed.insert(id);
                continue;
            }
            many => {
                push(Severity::Error, format!("{path}.op"), "INVALID_OPCODE", format!("ambiguous alias '{raw_op}': {}", many.iter().map(|o| o.name.as_str()).collect::<Vec<_>>().join(" | ")), Some(format!("\"op\": \"{}\"", many[0].name)), None);
                failed.insert(id);
                continue;
            }
        };
        let sig = signature(op);
        let prms = op.params();
        let mut args: BTreeMap<String, Value> = BTreeMap::new();
        let jargs = so.get("args").cloned().unwrap_or(J::Object(Default::default()));
        let Some(ao) = jargs.as_object() else {
            push(Severity::Error, format!("{path}.args"), "INVALID_PARAMETER", "'args' must be an object".into(), None, Some(sig));
            continue;
        };
        for (k, v) in ao {
            let apath = format!("{path}.args.{k}");
            if !prms.iter().any(|p| &p.name == k) && k != "role" && !k.starts_with('$') {
                let best = prms.iter().map(|p| (levenshtein(k, &p.name), &p.name)).min();
                let fix = best.filter(|(d, _)| *d <= 3).map(|(_, n)| format!("rename '{k}' → '{n}'"));
                push(Severity::Error, apath.clone(), "INVALID_PARAMETER", format!("{} has no input '{k}'", op.name), fix, Some(sig.clone()));
            }
            match decode_value(v, &apath) {
                Ok(val) => {
                    if let (Value::String(s), true) = (&val, op.name != "knowledge.parameter") {
                        if params_declared.contains(s) {
                            push(Severity::Warning, apath.clone(), "ADVICE", format!("bare string \"{s}\" is read as parameter '{s}'; write \"${s}\" to make the reference explicit"), Some(format!("\"{k}\": \"${s}\"")), None);
                        }
                    }
                    args.insert(k.clone(), val);
                }
                Err(e) => push(Severity::Error, e.path, "INVALID_PARAMETER", e.message, None, Some(sig.clone())),
            }
        }
        for p in &prms {
            if !p.optional && !ao.contains_key(&p.name) {
                let ty = op.inputs.iter().find(|x| x.split(':').next() == Some(p.name.as_str())).and_then(|x| x.split_once(':')).map(|x| x.1.to_string()).unwrap_or_default();
                push(Severity::Error, format!("{path}.args"), "INVALID_PARAMETER", format!("{}: missing required input '{}' ({ty})", op.name, p.name), Some(format!("\"{}\": {}", p.name, example_for(&ty))), Some(sig.clone()));
            }
        }
        if op.name == "knowledge.parameter" {
            if let Some(J::String(n)) = ao.get("name") { params_declared.insert(n.clone()); }
        }
        if op.name == "study.configurations" {
            configurations = ao.get("table").and_then(configurations_of);
        }
        let over = so.get("over").and_then(J::as_str).map(|s| s.trim_start_matches('%').to_string());
        json_index.insert(id.clone(), i);
        decoded.push(Stmt { id, op: op.name.clone(), args, over });
    }
    // SSA order, finiteness and unit dimensions on what decoded (missing-input rules are
    // already reported above with fixes, so skip those duplicates).
    let prog = Program { agir: crate::program::AGIR_VERSION.into(), requirements_rev: 0, expected_revision: 0, statements: decoded.clone() };
    for d in crate::verify::verify_in_workspace(&prog, catalog, exists) {
        if d.message.contains("missing required input") { continue; }
        if failed.iter().any(|f| d.message.contains(&format!("'%{f}'"))) { continue; }
        let i = json_index.get(&d.stmt).copied().unwrap_or(0);
        push(Severity::Error, format!("$.program[{i}]"), d.status.name(), d.message, None, None);
    }
    preview = serde_json::json!({ "statements": stmts.len(), "configurations": configurations });
    if let Some(n) = configurations {
        if n > 1024 {
            push(Severity::Warning, "$.program".into(), "ADVICE", format!("study expands to {n} configurations; the CPU reference path rebuilds each one (seconds per hundred). Consider a coarser grid first."), None, None);
        }
    }
    finish(out, preview)
}

fn finish(mut out: Vec<Lint>, preview: J) -> LintReport {
    out.sort_by(|a, b| a.severity.cmp(&b.severity).then(a.path.cmp(&b.path)));
    let errors = out.iter().filter(|l| l.severity == Severity::Error).count();
    LintReport { ok: errors == 0, errors, warnings: out.len() - errors, diagnostics: out, preview }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_every_mistake_with_fixes() {
        let c = Catalog::global();
        let src = r#"{"requirements_rev": 3, "expected_revision": 0, "program": [
            {"id": "pr",  "op": "knowledge.parameter", "args": {"name": "r", "value": "60 mm"}},
            {"id": "sk",  "op": "sketch.create", "args": {"support": "xy"}},
            {"id": "c",   "op": "sketch.circel", "args": {"sketch": "%sk"}},
            {"id": "pad", "op": "part.pad", "args": {"profile": "%sk", "limit": "10 mm"}},
            {"id": "x",   "op": "sketch.circle", "args": {"sketch": "%sk", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "=r/2 +"}},
            {"id": "y",   "op": "sketch.circle", "args": {"sketch": "%nope", "mode": "center_radius", "pts": [["0 mm", "0 mm"]], "r": "r"}}
        ]}"#;
        let r = check_program(src, c);
        let msgs: Vec<String> = r.diagnostics.iter().map(|d| format!("{} {} {}", d.path, d.message, d.fix.clone().unwrap_or_default())).collect();
        let has = |needle: &str| msgs.iter().any(|m| m.contains(needle));
        assert!(!r.ok);
        assert!(has("unknown op 'sketch.circel'") && has("\"op\": \"sketch.circle\""), "{msgs:#?}");
        assert!(has("has no input 'limit'") && has("rename 'limit' → 'limits'"), "{msgs:#?}");
        assert!(has("missing required input 'limits' (Limits)"), "{msgs:#?}");
        assert!(has("column"), "formula syntax error: {msgs:#?}");
        assert!(has("$.program[5] sketch.circle: '%nope' used before definition"), "{msgs:#?}");
        assert!(r.diagnostics.iter().any(|d| d.severity == Severity::Warning && d.fix.as_deref() == Some("\"r\": \"$r\"")), "{msgs:#?}");
        // knowledge.parameter no longer needs `type` (inferred from the value's unit)
        assert!(!has("missing required input 'type'"), "{msgs:#?}");
    }
}
