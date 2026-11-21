//! `cargo xtask <cmd>`: spec tooling (Rust ports of `docs/spec/tools/*.py`).
//!
//! * `gen-status`   regenerate `crates/driver/src/status.rs` and `crates/kernels-cuda/cu/ag_status.cuh`
//!                  from the AgStatus table in `docs/spec/06-harness.md` (AG-CV-003).
//! * `render`       regenerate the catalog tables in `docs/spec/03-op-catalog.md` (AG-CV-004).
//! * `check`        consistency checks equivalent to `docs/spec/tools/check_spec.py`.
//!
//! Catalog *authoring* stays in `docs/spec/tools/catalog_source.py` (the data source).

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}
fn spec() -> PathBuf {
    root().join("docs/spec")
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let code = match cmd.as_str() {
        "gen-status" => gen_status(),
        "render" => render(),
        "check" => check(),
        _ => {
            eprintln!("usage: cargo xtask <gen-status|render|check>");
            2
        }
    };
    std::process::exit(code);
}

// ---------------------------------------------------------------------------------------
// AgStatus table
// ---------------------------------------------------------------------------------------

struct StatusRow {
    value: u16,
    name: String,
    meaning: String,
}

fn status_rows() -> Vec<StatusRow> {
    let text = fs::read_to_string(spec().join("06-harness.md")).expect("06-harness.md");
    let mut rows = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // | `0x0000` | category | `NAME` | meaning |
        if cells.len() >= 6 && cells[1].starts_with("`0x") && cells[3].starts_with('`') {
            let hex = cells[1].trim_matches('`').trim_start_matches("0x");
            if let Ok(v) = u16::from_str_radix(hex, 16) {
                rows.push(StatusRow { value: v, name: cells[3].trim_matches('`').to_string(), meaning: cells[4].replace('`', "") });
            }
        }
    }
    rows
}

fn camel(n: &str) -> String {
    n.split('_').map(|w| {
        let mut c = w.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
            None => String::new(),
        }
    }).collect()
}

fn gen_status() -> i32 {
    let rows = status_rows();
    let cats = [(0x00, "Info"), (0x01, "Lifecycle"), (0x02, "Revision"), (0x03, "Parameter"), (0x04, "Check"), (0x05, "Reference"), (0x06, "Resource"), (0x07, "Code"), (0x08, "Solver")];
    let mut rs = String::new();
    rs += "//! `AgStatus`: fixed, append-only `u16` status codes (spec 06 § Error codes, AG-06-005, AG-CV-003).\n";
    rs += "//! GENERATED from docs/spec/06-harness.md by `cargo xtask gen-status`. Do not edit by hand.\n\n";
    rs += "/// Category = high byte of the status value.\n#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]\n#[repr(u8)]\npub enum StatusCategory {\n";
    for (k, v) in cats { rs += &format!("    {v} = 0x{k:02x},\n"); }
    rs += "}\n\n/// Command/feature outcome. `0x0000` is success. Values never change (append-only).\n#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]\n#[repr(u16)]\npub enum AgStatus {\n";
    for r in &rows { rs += &format!("    /// {}\n    {} = 0x{:04X},\n", r.meaning, camel(&r.name), r.value); }
    rs += "}\n\nimpl AgStatus {\n    /// Every defined status, in value order.\n    pub const ALL: &[AgStatus] = &[\n";
    for r in &rows { rs += &format!("        AgStatus::{},\n", camel(&r.name)); }
    rs += "    ];\n\n    /// Raw `u16` value as written in status records.\n    pub const fn value(self) -> u16 { self as u16 }\n\n";
    rs += "    /// Decode a raw value; unknown values (from a newer driver) return `None`.\n    pub fn from_value(v: u16) -> Option<Self> {\n        match v {\n";
    for r in &rows { rs += &format!("            0x{:04X} => Some(AgStatus::{}),\n", r.value, camel(&r.name)); }
    rs += "            _ => None,\n        }\n    }\n\n    /// Stable UPPER_SNAKE name (cf. `cudaGetErrorName`).\n    pub const fn name(self) -> &'static str {\n        match self {\n";
    for r in &rows { rs += &format!("            AgStatus::{} => \"{}\",\n", camel(&r.name), r.name); }
    rs += "        }\n    }\n\n    /// Human-readable description (cf. `cudaGetErrorString`).\n    pub const fn description(self) -> &'static str {\n        match self {\n";
    for r in &rows { rs += &format!("            AgStatus::{} => \"{}\",\n", camel(&r.name), r.meaning.replace('"', "\\\"")); }
    rs += "        }\n    }\n\n    pub fn category(self) -> StatusCategory {\n        match (self.value() >> 8) as u8 {\n";
    for (k, v) in cats { rs += &format!("            0x{k:02x} => StatusCategory::{v},\n"); }
    rs += "            _ => unreachable!(\"AgStatus values are defined only in known categories\"),\n        }\n    }\n\n";
    rs += "    /// Success and informational codes (category 0x00) do not reject a command.\n    pub fn is_ok(self) -> bool { matches!(self.category(), StatusCategory::Info) }\n}\n\n";
    rs += "impl core::fmt::Display for AgStatus {\n    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n        write!(f, \"{} (0x{:04X}): {}\", self.name(), self.value(), self.description())\n    }\n}\n";
    let mut cu = String::from("// AgStatus values for device code. GENERATED from docs/spec/06-harness.md by `cargo xtask gen-status`.\n#pragma once\n\n");
    for r in &rows { cu += &format!("#define AG_{} ((unsigned short)0x{:04X})\n", r.name, r.value); }
    cu += "\n#define AG_STATUS_CATEGORY(s) ((unsigned)((s) >> 8))\n#define AG_STATUS_IS_OK(s) (AG_STATUS_CATEGORY(s) == 0u)\n";
    fs::write(root().join("crates/driver/src/status_generated.rs"), rs).expect("write status");
    fs::write(root().join("crates/kernels-cuda/cu/ag_status.cuh"), cu).expect("write cuh");
    println!("generated {} status codes (status_generated.rs; diff against status.rs before replacing)", rows.len());
    0
}

// ---------------------------------------------------------------------------------------
// Catalog rendering
// ---------------------------------------------------------------------------------------

fn load_catalog() -> Value {
    serde_json::from_str(&fs::read_to_string(spec().join("op-catalog.json")).expect("op-catalog.json")).expect("valid json")
}

fn render() -> i32 {
    let cat = load_catalog();
    let begin = "<!-- BEGIN GENERATED CATALOG -->";
    let end = "<!-- END GENERATED CATALOG -->";
    let md_path = spec().join("03-op-catalog.md");
    let md = fs::read_to_string(&md_path).expect("03");
    let (head, rest) = md.split_once(begin).expect("begin marker");
    let (_, tail) = rest.split_once(end).expect("end marker");
    let mut fams: Vec<(String, Vec<&Value>)> = Vec::new();
    for op in cat["ops"].as_array().unwrap() {
        let f = op["family"].as_str().unwrap().to_string();
        match fams.iter_mut().find(|(n, _)| *n == f) {
            Some((_, v)) => v.push(op),
            None => fams.push((f, vec![op])),
        }
    }
    let mut out = String::new();
    for (fam, ops) in &fams {
        let id = cat["families"][fam].as_u64().unwrap();
        out += &format!("\n### `{fam}` ({}, family 0x{id:02X}, {} ops)\n\n", ops[0]["layer"].as_str().unwrap(), ops.len());
        out += "| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
        for o in ops {
            let j = |k: &str| o[k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect::<Vec<_>>();
            let io = format!("{} → {}", nz(j("inputs").join(", ")), nz(j("outputs").join(", "))).replace('|', "/");
            let axes: String = j("batch_axes").iter().map(|a| a[..1].to_uppercase()).collect::<Vec<_>>().join(",");
            let backend = if o["backend"] == "gpu" { "GPU" } else { "host (control plane)" };
            out += &format!(
                "| `{}` | {} | `0x{:04X}` | {} | {} | `{}` | {} | {} | {} | {} | {} |\n",
                o["name"].as_str().unwrap(),
                if o["kernel_primitive"].as_bool().unwrap_or(false) { "●" } else { "" },
                o["opcode"].as_u64().unwrap(),
                o["summary"].as_str().unwrap(),
                nz(j("aliases").join(", ")),
                io,
                o["phase"].as_str().unwrap(),
                backend,
                nz(axes),
                if o["differentiable"].as_bool().unwrap() { "✓" } else { "" },
                o["milestone"].as_str().unwrap(),
            );
        }
    }
    fs::write(&md_path, format!("{head}{begin}\n{out}{end}{tail}")).expect("write 03");
    println!("rendered {} families", fams.len());
    0
}

fn nz(s: String) -> String {
    if s.is_empty() { "—".into() } else { s }
}

// ---------------------------------------------------------------------------------------
// Consistency checks (port of docs/spec/tools/check_spec.py)
// ---------------------------------------------------------------------------------------

fn check() -> i32 {
    let mut errors: Vec<String> = Vec::new();
    let cat = load_catalog();
    let ops = cat["ops"].as_array().unwrap();
    let families = cat["families"].as_object().unwrap();
    let required = ["name", "family", "opcode", "version", "summary", "inputs", "outputs", "batch_axes", "phase", "backend", "effect", "deterministic", "differentiable", "failure_codes", "milestone", "status"];
    let mut names = BTreeSet::new();
    let mut opcodes = BTreeSet::new();
    for op in ops {
        let name = op["name"].as_str().unwrap_or("?").to_string();
        for k in required {
            if op.get(k).is_none() { errors.push(format!("{name}: missing field {k}")); }
        }
        if !names.insert(name.clone()) { errors.push(format!("duplicate name {name}")); }
        let oc = op["opcode"].as_u64().unwrap_or(0);
        if !opcodes.insert(oc) { errors.push(format!("duplicate opcode {oc:#06x}")); }
        let fam = op["family"].as_str().unwrap_or("");
        if families.get(fam).and_then(Value::as_u64) != Some(oc >> 8) { errors.push(format!("{name}: opcode family mismatch")); }
        if op["backend"] == "host" && fam != "udf" { errors.push(format!("{name}: only udf compile-time ops may run on host")); }
    }
    // aliases unique within family
    let mut fam_alias: HashMap<(String, String), usize> = HashMap::new();
    for op in ops {
        for a in op["aliases"].as_array().unwrap() {
            *fam_alias.entry((op["family"].as_str().unwrap().into(), a.as_str().unwrap().into())).or_default() += 1;
        }
    }
    for ((f, a), n) in &fam_alias {
        if *n > 1 { errors.push(format!("family {f}: alias '{a}' maps to {n} ops")); }
    }
    let catalog_md = fs::read_to_string(spec().join("03-op-catalog.md")).unwrap();
    for n in &names {
        if !catalog_md.contains(&format!("`{n}`")) { errors.push(format!("03-op-catalog.md missing {n}")); }
    }
    let harness = fs::read_to_string(spec().join("06-harness.md")).unwrap();
    let mut codes: BTreeSet<String> = BTreeSet::new();
    for op in ops { for c in op["failure_codes"].as_array().unwrap() { codes.insert(c.as_str().unwrap().into()); } }
    for c in &codes {
        if !harness.contains(&format!("`{c}`")) { errors.push(format!("error code {c} not defined in 06-harness.md")); }
    }
    let status = status_rows();
    let vals: BTreeSet<u16> = status.iter().map(|r| r.value).collect();
    if vals.len() != status.len() { errors.push("AgStatus values not unique".into()); }
    // docs: status lines, requirement IDs, links, op references
    let mut seen_req: BTreeMap<String, String> = BTreeMap::new();
    let mut versions = BTreeSet::new();
    let mut docs: Vec<PathBuf> = fs::read_dir(spec()).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "md")).collect();
    docs.sort();
    for md in &docs {
        let fname = md.file_name().unwrap().to_string_lossy().to_string();
        let text = fs::read_to_string(md).unwrap();
        let status_line = text.lines().find(|l| l.starts_with("> Status: **"));
        let normative = match status_line {
            Some(l) => {
                if let Some(v) = l.split("Spec v").nth(1).and_then(|r| r.split_whitespace().next()) { versions.insert(v.to_string()); }
                l.contains("**Normative**")
            }
            None => { errors.push(format!("{fname}: missing status line")); false }
        };
        let docnum = if fname == "CONVENTIONS.md" { "CV".to_string() } else { fname.chars().take(2).collect() };
        let mut reqs = 0;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("| AG-") {
                let id: String = format!("AG-{}", rest.split_whitespace().next().unwrap_or(""));
                let parts: Vec<&str> = id.split('-').collect();
                if parts.len() == 3 {
                    reqs += 1;
                    if parts[1] != docnum { errors.push(format!("{fname}: {id} does not match document {docnum}")); }
                    if let Some(prev) = seen_req.insert(id.clone(), fname.clone()) { errors.push(format!("duplicate requirement {id} ({fname}, {prev})")); }
                }
            }
            // links
            let mut s = line;
            while let Some(i) = s.find("](") {
                let after = &s[i + 2..];
                let end = after.find(')').unwrap_or(after.len());
                let target = after[..end].split('#').next().unwrap_or("");
                if !target.is_empty() && !target.contains("://") && !target.contains(' ') && !md.parent().unwrap().join(target).exists() {
                    errors.push(format!("{fname}: broken link {target}"));
                }
                s = &after[end.min(after.len())..];
            }
            // `"op": "family.name"` references
            let mut s = line;
            while let Some(i) = s.find("\"op\": \"") {
                let after = &s[i + 7..];
                let end = after.find('"').unwrap_or(after.len());
                let op = &after[..end];
                if op.contains('.') && !names.contains(op) { errors.push(format!("{fname}: example uses unknown op {op}")); }
                s = &after[end.min(after.len())..];
            }
        }
        if normative && reqs == 0 { errors.push(format!("{fname}: normative doc without requirement IDs")); }
    }
    if versions.len() > 1 { errors.push(format!("docs disagree on spec version: {versions:?}")); }
    if errors.is_empty() {
        let prims = ops.iter().filter(|o| o["kernel_primitive"].as_bool().unwrap_or(false)).count();
        println!("ok: {} ops ({prims} kernel primitives), {} requirement IDs, {} AgStatus codes, {} docs", names.len(), seen_req.len(), status.len(), docs.len());
        0
    } else {
        for e in &errors { eprintln!("{e}"); }
        1
    }
}
