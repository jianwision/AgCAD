//! Program verifier (spec 04): ops exist, SSA references defined before use, required inputs
//! present, immediates finite, `over` subsets refer to configuration-carrying values.

use crate::catalog::Catalog;
use crate::program::{Program, Value};
use agcad_driver::AgStatus;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub stmt: String,
    pub status: AgStatus,
    pub message: String,
}

pub fn verify(p: &Program, catalog: &Catalog) -> Vec<Diagnostic> {
    verify_in_workspace(p, catalog, |_| false)
}

/// Resolve earlier batch references against the caller's active workspace.
pub fn verify_in_workspace(p: &Program, catalog: &Catalog, exists: impl Fn(&str) -> bool) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let mut defined: HashSet<&str> = HashSet::new();
    for s in &p.statements {
        let mut d = |status, message: String| out.push(Diagnostic { stmt: s.id.clone(), status, message });
        if defined.contains(s.id.as_str()) {
            d(AgStatus::InvalidParameter, format!("duplicate statement id '{}'", s.id));
        }
        let Some(op) = catalog.get(&s.op) else {
            d(AgStatus::InvalidOpcode, format!("unknown op '{}'", s.op));
            continue;
        };
        for prm in op.params() {
            if !prm.optional && !s.args.contains_key(&prm.name) {
                d(AgStatus::InvalidParameter, format!("{}: missing required input '{}'", op.name, prm.name));
            }
        }
        for (k, v) in &s.args {
            if !v.is_finite_numeric() {
                d(AgStatus::NonfiniteParameter, format!("{}: non-finite value for '{k}'", op.name));
            }
            let mut refs = Vec::new();
            v.refs(&mut refs);
            for r in refs {
                if !defined.contains(r.as_str()) && !exists(&r) {
                    d(AgStatus::StaleReference, format!("{}: '%{r}' used before definition", op.name));
                }
            }
            if let Value::Quantity(q) = v {
                if let Some(prm) = op.params().iter().find(|pp| &pp.name == k) {
                    let expected: Vec<_> = prm.ty.names().iter().filter_map(|n| crate::units::dimension_of_type(n)).collect();
                    if !expected.is_empty() && !expected.contains(&q.dim) {
                        d(AgStatus::InvalidParameter, format!("{}: '{k}' expects {:?}, got {:?}", op.name, expected, q.dim));
                    }
                }
            }
        }
        if let Some(ov) = &s.over {
            if !defined.contains(ov.as_str()) && !exists(ov) {
                d(AgStatus::StaleReference, format!("over '%{ov}' used before definition"));
            }
        }
        defined.insert(s.id.as_str());
    }
    out
}
