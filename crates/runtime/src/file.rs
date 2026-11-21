//! `.agcad` documents (spec 16): deterministic store-only zip with manifest + content hashes,
//! specification tree, knowledge, UDF sources, versions; forward migrations
//! (`MIGRATION_FAILED`, AG-16-001..005) and crash-recovery snapshots.

use crate::document::{Branch, PartDocument, PartState, Version};
use agcad_driver::AgStatus;
use agcad_geo::zip;
use agcad_ir::catalog::fnv1a64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Manifest {
    pub schema_version: u32,
    pub catalog_version: String,
    pub abi_version: u16,
    pub units: String,
    pub producer: String,
    /// entry name → fnv1a64 hex
    pub hashes: BTreeMap<String, String>,
}

/// Migration of one feature's arguments from `from` to `from + 1`.
pub type MigrationFn = fn(op: &str, args: &mut BTreeMap<String, agcad_ir::Value>) -> Result<(), String>;

#[derive(Default)]
pub struct Migrations {
    /// (op, from_version) → function
    pub map: BTreeMap<(String, u32), MigrationFn>,
}

#[derive(Debug)]
pub struct OpenReport {
    pub migrated: Vec<(u32, String, u32, u32)>,
    /// Set when a migration failed: document opens read-only at the last migratable feature.
    pub failed: Option<(u32, String, String)>,
}

fn json<T: Serialize>(v: &T) -> Vec<u8> { serde_json::to_vec_pretty(v).expect("serializable") }

pub fn save(doc: &PartDocument, catalog_version: &str) -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    let head = &doc.versions[doc.branches["main"].head as usize];
    entries.push(("tree/specification.json".into(), json(&*head.state)));
    let knowledge = serde_json::json!({ "parameters": head.state.parameters, "formulas": head.state.formulas, "design_tables": head.state.design_tables });
    entries.push(("tree/knowledge.json".into(), serde_json::to_vec_pretty(&knowledge).unwrap()));
    for (h, src) in &head.state.udfs { entries.push((format!("udf/{h}.ag.rs"), src.clone().into_bytes())); }
    for v in &doc.versions {
        let meta = serde_json::json!({ "id": v.id, "parent": v.parent, "requirements_rev": v.requirements_rev, "label": v.label });
        entries.push((format!("versions/{}/meta.json", v.id), serde_json::to_vec_pretty(&meta).unwrap()));
        entries.push((format!("versions/{}/state.json", v.id), json(&*v.state)));
    }
    let branches: BTreeMap<&String, u32> = doc.branches.iter().map(|(k, b)| (k, b.head)).collect();
    entries.push(("versions/branches.json".into(), serde_json::to_vec_pretty(&branches).unwrap()));
    let hashes = entries.iter().map(|(n, d)| (n.clone(), format!("{:016x}", fnv1a64(d)))).collect();
    let manifest = Manifest { schema_version: SCHEMA_VERSION, catalog_version: catalog_version.into(), abi_version: agcad_driver::ABI_VERSION, units: head.state.units.clone(), producer: format!("agcad {}", env!("CARGO_PKG_VERSION")), hashes };
    let mut all = vec![("manifest.json".to_string(), json(&manifest))];
    all.extend(entries);
    let refs: Vec<(&str, &[u8])> = all.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    zip::write_zip(&refs)
}

pub fn open(bytes: &[u8], migrations: &Migrations, catalog: &agcad_ir::Catalog) -> Result<(PartDocument, OpenReport), (AgStatus, String)> {
    let files = zip::read_zip(bytes).map_err(|e| (AgStatus::InvalidParameter, e))?;
    let get = |n: &str| files.iter().find(|(k, _)| k == n).map(|(_, d)| d.as_slice());
    let manifest: Manifest = serde_json::from_slice(get("manifest.json").ok_or((AgStatus::InvalidParameter, "missing manifest".into()))?).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
    if manifest.schema_version > SCHEMA_VERSION { return Err((AgStatus::MigrationFailed, format!("document schema {} is newer than this build ({SCHEMA_VERSION})", manifest.schema_version))); }
    for (n, h) in &manifest.hashes {
        let d = get(n).ok_or((AgStatus::InvalidParameter, format!("missing entry {n}")))?;
        if &format!("{:016x}", fnv1a64(d)) != h { return Err((AgStatus::InvalidParameter, format!("content hash mismatch in {n}"))); }
    }
    let branches: BTreeMap<String, u32> = serde_json::from_slice(get("versions/branches.json").unwrap_or(b"{}")).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
    let mut versions = Vec::new();
    let mut report = OpenReport { migrated: vec![], failed: None };
    let mut id = 0u32;
    while let Some(state) = get(&format!("versions/{id}/state.json")) {
        let mut st: PartState = serde_json::from_slice(state).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
        let meta: serde_json::Value = serde_json::from_slice(get(&format!("versions/{id}/meta.json")).unwrap_or(b"{}")).unwrap_or_default();
        // Forward migrations per feature (AG-16-004/005): never silently drop features.
        let order = st.order.clone();
        'feat: for fid in order {
            let node = st.features.get_mut(&fid).unwrap();
            let current = catalog.get(&node.op).map(|m| m.version).unwrap_or(node.op_version);
            while node.op_version < current {
                match migrations.map.get(&(node.op.clone(), node.op_version)) {
                    Some(f) => match f(&node.op, &mut node.args) {
                        Ok(()) => { report.migrated.push((fid, node.op.clone(), node.op_version, node.op_version + 1)); node.op_version += 1; }
                        Err(e) => { report.failed = Some((fid, node.op.clone(), e)); st.work_object = Some(fid); break 'feat; }
                    },
                    None => { report.failed = Some((fid, node.op.clone(), format!("no migration {} v{} → v{}", node.op, node.op_version, node.op_version + 1))); st.work_object = Some(fid); break 'feat; }
                }
            }
        }
        versions.push(Version { id, parent: meta["parent"].as_u64().map(|x| x as u32), state: Arc::new(st), requirements_rev: meta["requirements_rev"].as_u64().unwrap_or(0) as u32, label: meta["label"].as_str().unwrap_or("").into() });
        id += 1;
    }
    if versions.is_empty() { return Err((AgStatus::InvalidParameter, "no versions in document".into())); }
    let branches = branches.into_iter().map(|(k, h)| (k.clone(), Branch { name: k, head: h })).collect();
    Ok((PartDocument { versions, branches, workspaces: BTreeMap::new() }, report))
}

/// Crash-recovery snapshot: tree + parameters + version pointer (spec 16 § 5).
pub fn snapshot(state: &PartState, base_version: u32) -> Vec<u8> {
    let payload = serde_json::json!({ "base_version": base_version, "state": state });
    serde_json::to_vec(&payload).expect("serializable")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn save_open_roundtrip_is_deterministic() {
        let mut d = PartDocument::new("p");
        d.workspace("main").unwrap().edit(|s| { s.add_feature("part.pad", 1, BTreeMap::new(), 0, None); });
        d.save_version("main", 1, "v1");
        let a = save(&d, "0.3.1");
        let b = save(&d, "0.3.1");
        assert_eq!(a, b, "byte-deterministic (AG-16-002)");
        let (d2, rep) = open(&a, &Migrations::default(), agcad_ir::Catalog::global()).unwrap();
        assert!(rep.failed.is_none());
        assert_eq!(d2.versions.len(), 2);
        assert_eq!(d2.versions[1].state.features.len(), 1);
    }
}
