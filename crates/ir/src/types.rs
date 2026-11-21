//! AgIR types parsed from catalog type notation:
//! `name:Type`, `?` optional, `[]` array, `[,]` grid, `A|B` either, `Configurations<T>`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ty {
    Named(String),
    Array(Box<Ty>),
    Grid(Box<Ty>),
    Either(Vec<Ty>),
    Configurations(Box<Ty>),
    /// `Configurations` without a parameter.
    ConfigurationsAny,
}

impl Ty {
    pub fn parse(s: &str) -> Ty {
        let s = s.trim();
        if s.contains('|') && !s.contains('<') {
            return Ty::Either(s.split('|').map(Ty::parse).collect());
        }
        if let Some(inner) = s.strip_suffix("[,]") {
            return Ty::Grid(Box::new(Ty::parse(inner)));
        }
        if let Some(inner) = s.strip_suffix("[]") {
            return Ty::Array(Box::new(Ty::parse(inner)));
        }
        if let Some(rest) = s.strip_prefix("Configurations<") {
            return Ty::Configurations(Box::new(Ty::parse(rest.trim_end_matches('>'))));
        }
        if s == "Configurations" {
            return Ty::ConfigurationsAny;
        }
        Ty::Named(s.to_string())
    }

    /// Does a value of this type carry the configuration batch axis?
    pub fn is_batched(&self) -> bool {
        matches!(self, Ty::Configurations(_) | Ty::ConfigurationsAny)
    }

    pub fn names(&self) -> Vec<&str> {
        match self {
            Ty::Named(n) => vec![n.as_str()],
            Ty::Array(t) | Ty::Grid(t) | Ty::Configurations(t) => t.names(),
            Ty::Either(v) => v.iter().flat_map(|t| t.names()).collect(),
            Ty::ConfigurationsAny => vec!["Configurations"],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    pub optional: bool,
}

impl Param {
    /// Parse `"limits:Interval?"`.
    pub fn parse(s: &str) -> Option<Param> {
        let (name, ty) = s.split_once(':')?;
        let (ty, optional) = match ty.strip_suffix('?') {
            Some(t) => (t, true),
            None => (ty, false),
        };
        Some(Param { name: name.trim().to_string(), ty: Ty::parse(ty), optional })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_types() {
        assert_eq!(Param::parse("r:Length|Law").unwrap().ty, Ty::Either(vec![Ty::Named("Length".into()), Ty::Named("Law".into())]));
        assert!(Param::parse("grading:Ref?").unwrap().optional);
        assert_eq!(Ty::parse("Vec4[,]"), Ty::Grid(Box::new(Ty::Named("Vec4".into()))));
        assert!(Ty::parse("Configurations<Score>").is_batched());
    }
}
