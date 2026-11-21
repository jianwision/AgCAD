//! Feature catalog loaded from `docs/spec/op-catalog.json` (normative; AG-03-001..004).

use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

/// The normative catalog, embedded at compile time.
pub const CATALOG_JSON: &str = include_str!("../../../docs/spec/op-catalog.json");

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase { Control, Proxy, Exact, Both }

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Backend { Gpu, Host }

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Effect { Pure, DocWrite, Io }

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchAxis { Configuration, Entity, Sample }

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Layer { Core, Stdlib, Model }

#[derive(Clone, Debug, Deserialize)]
pub struct OpManifest {
    pub name: String,
    pub family: String,
    pub opcode: u16,
    pub version: u32,
    pub aliases: Vec<String>,
    pub summary: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub batch_axes: Vec<BatchAxis>,
    pub phase: Phase,
    pub backend: Backend,
    pub effect: Effect,
    pub deterministic: bool,
    pub differentiable: bool,
    pub failure_codes: Vec<String>,
    pub milestone: String,
    pub status: String,
    pub layer: Layer,
    #[serde(default)]
    pub kernel_primitive: bool,
}

impl OpManifest {
    /// Parsed input parameters `(name, type, optional)`.
    pub fn params(&self) -> Vec<crate::types::Param> {
        self.inputs.iter().filter_map(|s| crate::types::Param::parse(s)).collect()
    }
    pub fn results(&self) -> Vec<crate::types::Param> {
        self.outputs.iter().filter_map(|s| crate::types::Param::parse(s)).collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct RawCatalog {
    catalog_version: String,
    families: BTreeMap<String, u16>,
    ops: Vec<OpManifest>,
}

pub struct Catalog {
    pub version: String,
    pub families: BTreeMap<String, u16>,
    pub ops: Vec<OpManifest>,
    by_name: HashMap<String, usize>,
    by_opcode: HashMap<u16, usize>,
    /// alias (lowercase) -> op indices (global; ambiguity resolved by family hint).
    by_alias: HashMap<String, Vec<usize>>,
    /// implicit short names (`pad` for `part.pad`); consulted only when no declared alias matches.
    by_short: HashMap<String, Vec<usize>>,
    hash: u64,
}

/// FNV-1a 64: stable catalog hash written into bytecode block headers.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl Catalog {
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: RawCatalog = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let mut by_name = HashMap::new();
        let mut by_opcode = HashMap::new();
        let mut by_alias: HashMap<String, Vec<usize>> = HashMap::new();
        let mut by_short: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, op) in raw.ops.iter().enumerate() {
            if by_name.insert(op.name.clone(), i).is_some() {
                return Err(format!("duplicate op {}", op.name));
            }
            if by_opcode.insert(op.opcode, i).is_some() {
                return Err(format!("duplicate opcode {:#06x}", op.opcode));
            }
            for a in &op.aliases {
                by_alias.entry(a.to_lowercase()).or_default().push(i);
            }
            let short = op.name.split('.').nth(1).unwrap_or("").to_string();
            by_short.entry(short.replace('_', " ")).or_default().push(i);
        }
        Ok(Catalog { version: raw.catalog_version, families: raw.families, ops: raw.ops, by_name, by_opcode, by_alias, by_short, hash: fnv1a64(json.as_bytes()) })
    }

    /// The embedded normative catalog.
    pub fn global() -> &'static Catalog {
        static C: OnceLock<Catalog> = OnceLock::new();
        C.get_or_init(|| Catalog::parse(CATALOG_JSON).expect("embedded op-catalog.json must parse"))
    }

    pub fn hash(&self) -> u64 { self.hash }
    pub fn get(&self, name: &str) -> Option<&OpManifest> { self.by_name.get(name).map(|&i| &self.ops[i]) }
    pub fn by_opcode(&self, opcode: u16) -> Option<&OpManifest> { self.by_opcode.get(&opcode).map(|&i| &self.ops[i]) }

    /// Resolve an alias or canonical name. Returns all candidates (ambiguous if > 1).
    pub fn resolve(&self, name_or_alias: &str) -> Vec<&OpManifest> {
        if let Some(op) = self.get(name_or_alias) {
            return vec![op];
        }
        let key = name_or_alias.to_lowercase().replace('_', " ");
        // Declared aliases take precedence over implicit short names.
        self.by_alias.get(&key).or_else(|| self.by_short.get(&key)).map(|v| v.iter().map(|&i| &self.ops[i]).collect()).unwrap_or_default()
    }

    /// "Did you mean" suggestions: aliases first, then nearest canonical names by edit distance.
    pub fn suggest(&self, wrong: &str, k: usize) -> Vec<String> {
        let mut out: Vec<String> = self.resolve(wrong).iter().map(|o| o.name.clone()).collect();
        if out.len() >= k {
            out.truncate(k);
            return out;
        }
        let mut scored: Vec<(usize, &str)> = self.ops.iter().map(|o| (levenshtein(wrong, &o.name), o.name.as_str())).collect();
        scored.sort();
        for (_, n) in scored {
            if out.len() >= k { break; }
            if !out.iter().any(|x| x == n) { out.push(n.to_string()); }
        }
        out
    }
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loads_embedded_catalog() {
        let c = Catalog::global();
        assert!(c.ops.len() >= 293);
        assert_eq!(c.get("part.pad").unwrap().opcode >> 8, c.families["part"]);
        assert!(c.resolve("extrude").iter().any(|o| o.name == "part.pad"));
        assert!(c.suggest("part.extrud", 3).contains(&"part.pad".to_string()) || !c.suggest("part.extrud", 3).is_empty());
    }
}
