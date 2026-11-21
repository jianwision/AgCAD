//! AgIR program model (spec 04 § AgIR): SSA statements `%id = op(args) {over}`.

use crate::units::Quantity;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const AGIR_VERSION: &str = "0.2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// Unit-carrying scalar, already in internal units.
    Quantity(Quantity),
    String(String),
    /// Reference to an earlier statement result: `%id` or `%id.field`.
    Ref { id: String, field: Option<String> },
    /// Content in the request bundle: `@file:name`.
    File(String),
    /// Formula over parameters, evaluated per configuration at rebuild: `"=od/2 - 5 mm"`
    /// (source without the `=`; see [`crate::formula`]).
    Formula(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

/// Pseudo statement id of parameter references: `"$r"` decodes to `Ref { id: "$param", field: "r" }`.
pub const PARAM_REF: &str = "$param";

impl Value {
    pub fn refs(&self, out: &mut Vec<String>) {
        match self {
            // `$param` references name parameters, not statements.
            Value::Ref { id, .. } if id != PARAM_REF => out.push(id.clone()),
            Value::List(v) => v.iter().for_each(|x| x.refs(out)),
            Value::Map(m) => m.values().for_each(|x| x.refs(out)),
            _ => {}
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Quantity(q) => Some(q.value),
            _ => None,
        }
    }
    pub fn is_finite_numeric(&self) -> bool {
        self.as_f64().map(f64::is_finite).unwrap_or(true)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stmt {
    pub id: String,
    /// Canonical op name (aliases are resolved during decode).
    pub op: String,
    pub args: BTreeMap<String, Value>,
    /// Restrict execution to a configuration subset (`"over": "%top"`).
    pub over: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub agir: String,
    pub requirements_rev: u32,
    pub expected_revision: u32,
    pub statements: Vec<Stmt>,
}

impl Program {
    pub fn new(requirements_rev: u32, expected_revision: u32) -> Self {
        Program { agir: AGIR_VERSION.into(), requirements_rev, expected_revision, statements: Vec::new() }
    }
    pub fn push(&mut self, id: impl Into<String>, op: impl Into<String>, args: BTreeMap<String, Value>) -> &mut Self {
        self.statements.push(Stmt { id: id.into(), op: op.into(), args, over: None });
        self
    }
    pub fn stmt(&self, id: &str) -> Option<&Stmt> {
        self.statements.iter().find(|s| s.id == id)
    }
}
