//! JSON encoding of AgIR (the LLM batch-program format, spec 04 § JSON encoding).
//!
//! Decoding converts unit strings to internal units and resolves aliases to canonical ops.

use crate::catalog::Catalog;
use crate::program::{Program, Stmt, Value};
use crate::units::Quantity;
use serde_json::Value as J;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeError {
    pub path: String,
    pub message: String,
    pub suggestions: Vec<String>,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)?;
        if !self.suggestions.is_empty() {
            write!(f, " (did you mean: {})", self.suggestions.join(", "))?;
        }
        Ok(())
    }
}

fn err(path: impl Into<String>, message: impl Into<String>) -> DecodeError {
    DecodeError { path: path.into(), message: message.into(), suggestions: vec![] }
}

pub fn decode_value(v: &J, path: &str) -> Result<Value, DecodeError> {
    Ok(match v {
        J::Null => Value::Null,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Int(i) } else { Value::Float(n.as_f64().ok_or_else(|| err(path, "bad number"))?) }
        }
        J::String(s) => {
            if let Some(r) = s.strip_prefix('%') {
                let (id, field) = match r.split_once('.') {
                    Some((a, b)) => (a.to_string(), Some(b.to_string())),
                    None => (r.to_string(), None),
                };
                Value::Ref { id, field }
            } else if let Some(src) = s.strip_prefix('=') {
                crate::formula::parse(src).map_err(|e| err(path, e))?;
                Value::Formula(src.trim().to_string())
            } else if let Some(name) = s.strip_prefix('$').filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')) {
                Value::Ref { id: crate::program::PARAM_REF.to_string(), field: Some(name.to_string()) }
            } else if let Some(f) = s.strip_prefix("@file:") {
                Value::File(f.to_string())
            } else if s.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '.') && s.contains(|c: char| c.is_alphabetic() || c == '°') {
                match Quantity::parse(s) {
                    Ok(q) => Value::Quantity(q),
                    Err(_) => Value::String(s.clone()),
                }
            } else {
                Value::String(s.clone())
            }
        }
        J::Array(a) => Value::List(a.iter().enumerate().map(|(i, x)| decode_value(x, &format!("{path}[{i}]"))).collect::<Result<_, _>>()?),
        J::Object(o) => Value::Map(o.iter().map(|(k, x)| Ok((k.clone(), decode_value(x, &format!("{path}.{k}"))?))).collect::<Result<_, DecodeError>>()?),
    })
}

pub fn encode_value(v: &Value) -> J {
    match v {
        Value::Null => J::Null,
        Value::Bool(b) => J::Bool(*b),
        Value::Int(i) => J::from(*i),
        Value::Float(f) => J::from(*f),
        Value::Quantity(q) => J::from(q.value),
        Value::String(s) => J::String(s.clone()),
        Value::Ref { id, field: Some(f) } if id == crate::program::PARAM_REF => J::String(format!("${f}")),
        Value::Ref { id, field } => J::String(match field { Some(f) => format!("%{id}.{f}"), None => format!("%{id}") }),
        Value::Formula(src) => J::String(format!("={src}")),
        Value::File(f) => J::String(format!("@file:{f}")),
        Value::List(l) => J::Array(l.iter().map(encode_value).collect()),
        Value::Map(m) => J::Object(m.iter().map(|(k, v)| (k.clone(), encode_value(v))).collect()),
    }
}

/// Decode an AgIR JSON program. Unknown ops get "did you mean" suggestions (spec 04).
pub fn decode_program(text: &str, catalog: &Catalog) -> Result<Program, DecodeError> {
    let root: J = serde_json::from_str(text).map_err(|e| err("$", e.to_string()))?;
    let obj = root.as_object().ok_or_else(|| err("$", "program must be an object"))?;
    let agir = obj.get("agir").and_then(J::as_str).unwrap_or(crate::program::AGIR_VERSION).to_string();
    let requirements_rev = obj.get("requirements_rev").and_then(J::as_u64).ok_or_else(|| err("$.requirements_rev", "required"))? as u32;
    let expected_revision = obj.get("expected_revision").and_then(J::as_u64).ok_or_else(|| err("$.expected_revision", "required"))? as u32;
    let stmts = obj.get("program").and_then(J::as_array).ok_or_else(|| err("$.program", "required array"))?;
    let mut statements = Vec::with_capacity(stmts.len());
    for (i, s) in stmts.iter().enumerate() {
        let path = format!("$.program[{i}]");
        let so = s.as_object().ok_or_else(|| err(&path, "statement must be an object"))?;
        let id = so.get("id").and_then(J::as_str).ok_or_else(|| err(format!("{path}.id"), "required"))?.to_string();
        let raw_op = so.get("op").and_then(J::as_str).ok_or_else(|| err(format!("{path}.op"), "required"))?;
        let cands = catalog.resolve(raw_op);
        let op = match cands.as_slice() {
            [one] => one.name.clone(),
            [] => return Err(DecodeError { path: format!("{path}.op"), message: format!("unknown op '{raw_op}'"), suggestions: catalog.suggest(raw_op, 3) }),
            many => return Err(DecodeError { path: format!("{path}.op"), message: format!("ambiguous alias '{raw_op}'"), suggestions: many.iter().map(|o| o.name.clone()).collect() }),
        };
        let mut args = BTreeMap::new();
        if let Some(a) = so.get("args") {
            let ao = a.as_object().ok_or_else(|| err(format!("{path}.args"), "must be an object"))?;
            for (k, v) in ao {
                args.insert(k.clone(), decode_value(v, &format!("{path}.args.{k}"))?);
            }
        }
        let over = so.get("over").and_then(J::as_str).map(|s| s.trim_start_matches('%').to_string());
        statements.push(Stmt { id, op, args, over });
    }
    Ok(Program { agir, requirements_rev, expected_revision, statements })
}

pub fn encode_program(p: &Program) -> String {
    let stmts: Vec<J> = p.statements.iter().map(|s| {
        let mut o = serde_json::Map::new();
        o.insert("id".into(), J::String(s.id.clone()));
        o.insert("op".into(), J::String(s.op.clone()));
        o.insert("args".into(), J::Object(s.args.iter().map(|(k, v)| (k.clone(), encode_value(v))).collect()));
        if let Some(ov) = &s.over { o.insert("over".into(), J::String(format!("%{ov}"))); }
        J::Object(o)
    }).collect();
    let root = serde_json::json!({ "agir": p.agir, "requirements_rev": p.requirements_rev, "expected_revision": p.expected_revision, "program": stmts });
    serde_json::to_string_pretty(&root).expect("serializable")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_spec_example_shape() {
        let c = Catalog::global();
        let src = r#"{"agir":"0.2","requirements_rev":3,"expected_revision":41,"program":[
          {"id":"p","op":"document.edit","args":{"branch":"main"}},
          {"id":"pd","op":"extrude","args":{"profile":"%s","limits":"12 mm"}}]}"#;
        let p = decode_program(src, c).unwrap();
        assert_eq!(p.statements[1].op, "part.pad");
        assert!(matches!(p.statements[1].args["limits"], Value::Quantity(_)));
        let bad = decode_program(r#"{"requirements_rev":1,"expected_revision":1,"program":[{"id":"x","op":"part.extrud"}]}"#, c).unwrap_err();
        assert!(!bad.suggestions.is_empty());
    }
}
