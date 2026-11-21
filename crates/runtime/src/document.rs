//! Document model in CATIA vocabulary (spec 02 § Ontology): Part / Product, PartBody / Body,
//! Geometrical Set, specification tree of features, Knowledgeware parameters, publications.
//! States are immutable `Arc`s: versions and branches share structure copy-on-write.

use agcad_ir::program::Value;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub type FeatureId = u32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContainerKind {
    /// PartBody / Body: ordered solid features.
    Body,
    /// Geometrical Set (unordered) or Ordered Geometrical Set.
    GeometricalSet { ordered: bool },
    /// Procedural network (cooked as one feature).
    Network,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Container {
    pub name: String,
    pub kind: ContainerKind,
    pub features: Vec<FeatureId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureNode {
    pub id: FeatureId,
    /// Canonical op name (catalog) or UDF name.
    pub op: String,
    pub op_version: u32,
    pub args: BTreeMap<String, Value>,
    pub container: usize,
    pub active: bool,
    /// Role assigned by the author (e.g. "disc").
    pub role: Option<String>,
}

impl FeatureNode {
    /// `family.op#id` (spec 02 § Stable naming).
    pub fn feature_ref(&self) -> String { format!("{}#{}", self.op, self.id) }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub kind: String,
    pub value: Value,
    pub range: Option<(f64, f64)>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PartState {
    pub name: String,
    pub units: String,
    pub containers: Vec<Container>,
    pub features: BTreeMap<FeatureId, FeatureNode>,
    /// Specification tree order (CATIA history order).
    pub order: Vec<FeatureId>,
    pub parameters: BTreeMap<String, Parameter>,
    /// Formula: target parameter/dimension path → expression source.
    pub formulas: BTreeMap<String, String>,
    pub design_tables: BTreeMap<String, (Vec<String>, Vec<Vec<Value>>)>,
    pub publications: BTreeMap<String, String>,
    /// UDF sources by content hash.
    pub udfs: BTreeMap<String, String>,
    pub materials: BTreeMap<String, String>,
    /// Define-in-work-object (rollback bar): features after it are not rebuilt.
    pub work_object: Option<FeatureId>,
    pub next_id: FeatureId,
}

impl PartState {
    pub fn new(name: &str) -> Self {
        PartState {
            name: name.into(),
            units: "mm".into(),
            containers: vec![Container { name: "PartBody".into(), kind: ContainerKind::Body, features: vec![] }],
            next_id: 1,
            ..Default::default()
        }
    }

    pub fn add_feature(&mut self, op: &str, op_version: u32, args: BTreeMap<String, Value>, container: usize, role: Option<String>) -> FeatureId {
        let id = self.next_id;
        self.next_id += 1;
        let node = FeatureNode { id, op: op.into(), op_version, args, container, active: true, role };
        self.features.insert(id, node);
        // Insert after the work object if one is defined (CATIA semantics), else at the end.
        match self.work_object.and_then(|w| self.order.iter().position(|&x| x == w)) {
            Some(pos) => { self.order.insert(pos + 1, id); self.work_object = Some(id); }
            None => self.order.push(id),
        }
        if let Some(c) = self.containers.get_mut(container) { c.features.push(id); }
        id
    }

    /// Features to rebuild, in order, honoring deactivation and the work object.
    pub fn rebuild_order(&self) -> Vec<FeatureId> {
        let mut out = Vec::new();
        for &id in &self.order {
            if self.features.get(&id).is_some_and(|f| f.active) { out.push(id); }
            if Some(id) == self.work_object { break; }
        }
        out
    }
}

pub type VersionId = u32;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Version {
    pub id: VersionId,
    pub parent: Option<VersionId>,
    pub state: Arc<PartState>,
    pub requirements_rev: u32,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub head: VersionId,
}

/// Private, mutable, possibly-invalid editing state on a branch (spec 06 § Transactions).
#[derive(Clone, Debug)]
pub struct Workspace {
    pub branch: String,
    pub base: VersionId,
    pub state: Arc<PartState>,
    /// Bumps on every change (AG via Handle.revision semantics).
    pub revision: u32,
    undo: Vec<Arc<PartState>>,
    redo: Vec<Arc<PartState>>,
}

impl Workspace {
    pub fn edit(&mut self, f: impl FnOnce(&mut PartState)) {
        self.undo.push(self.state.clone());
        self.redo.clear();
        f(Arc::make_mut(&mut self.state));
        self.revision += 1;
    }
    pub fn undo(&mut self) -> bool {
        match self.undo.pop() { Some(s) => { self.redo.push(std::mem::replace(&mut self.state, s)); self.revision += 1; true } None => false }
    }
    pub fn redo(&mut self) -> bool {
        match self.redo.pop() { Some(s) => { self.undo.push(std::mem::replace(&mut self.state, s)); self.revision += 1; true } None => false }
    }
    pub fn dirty_since(&self, v: &Version) -> bool { !Arc::ptr_eq(&self.state, &v.state) }
}

/// A Part document with versions, branches and one workspace per branch.
#[derive(Debug)]
pub struct PartDocument {
    pub versions: Vec<Version>,
    pub branches: BTreeMap<String, Branch>,
    pub workspaces: BTreeMap<String, Workspace>,
}

impl PartDocument {
    pub fn new(name: &str) -> Self {
        let v0 = Version { id: 0, parent: None, state: Arc::new(PartState::new(name)), requirements_rev: 0, label: "initial".into() };
        let mut branches = BTreeMap::new();
        branches.insert("main".to_string(), Branch { name: "main".into(), head: 0 });
        PartDocument { versions: vec![v0], branches, workspaces: BTreeMap::new() }
    }

    pub fn workspace(&mut self, branch: &str) -> Option<&mut Workspace> {
        if !self.workspaces.contains_key(branch) {
            let b = self.branches.get(branch)?;
            let v = &self.versions[b.head as usize];
            self.workspaces.insert(branch.into(), Workspace { branch: branch.into(), base: v.id, state: v.state.clone(), revision: 0, undo: vec![], redo: vec![] });
        }
        self.workspaces.get_mut(branch)
    }

    /// Create an immutable version from the workspace (call after save-phase checks pass).
    pub fn save_version(&mut self, branch: &str, requirements_rev: u32, label: &str) -> Option<VersionId> {
        let ws = self.workspaces.get(branch)?;
        let id = self.versions.len() as VersionId;
        self.versions.push(Version { id, parent: Some(ws.base), state: ws.state.clone(), requirements_rev, label: label.into() });
        self.branches.get_mut(branch)?.head = id;
        if let Some(ws) = self.workspaces.get_mut(branch) { ws.base = id; }
        Some(id)
    }

    pub fn discard(&mut self, branch: &str) {
        self.workspaces.remove(branch);
    }

    /// `document.branch`: N branches from a version, sharing state (copy-on-write).
    pub fn branch(&mut self, from: VersionId, names: &[String]) -> Vec<String> {
        names.iter().map(|n| { self.branches.insert(n.clone(), Branch { name: n.clone(), head: from }); n.clone() }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_versions_and_cow() {
        let mut d = PartDocument::new("flange");
        let ws = d.workspace("main").unwrap();
        ws.edit(|s| { s.add_feature("part.pad", 1, BTreeMap::new(), 0, Some("disc".into())); });
        assert_eq!(ws.revision, 1);
        let v = d.save_version("main", 3, "first").unwrap();
        d.branch(v, &["a".into(), "b".into()]);
        let ver = d.versions[v as usize].clone();
        let wa = d.workspace("a").unwrap();
        assert!(!wa.dirty_since(&ver), "fresh branch workspace shares the version state");
        let ws = d.workspace("main").unwrap();
        ws.edit(|s| s.work_object = None);
        assert!(ws.undo());
        assert!(ws.redo());
    }
}
