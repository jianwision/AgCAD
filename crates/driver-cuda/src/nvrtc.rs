//! NVRTC compilation with an on-disk cache keyed by (source hash, arch, flags, mode, catalog
//! version) (spec 08 § Kernel authoring; review F7). Supports both PTX and LTO-IR (`-dlto`)
//! output modes for incremental stdlib + UDF linking.

use agcad_driver::DriverError;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompileMode {
    Ptx,
    LtoIr,
}

fn intern_arch(arch: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut table = INTERNED.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some(&s) = table.iter().find(|&&s| s == arch) {
        return s;
    }
    let leaked: &'static str = Box::leak(arch.to_string().into_boxed_str());
    table.push(leaked);
    leaked
}

pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AGCAD_CACHE") {
        return PathBuf::from(d).join("ptx");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/agcad/ptx")
}

fn fnv1a64(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in p.bytes().chain(std::iter::once(0u8)) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Compile (or fetch from cache) PTX for `source`.
pub fn compile(name: &str, source: &str, arch: &str, options: &[String]) -> Result<String, DriverError> {
    compile_with_mode(name, source, arch, options, CompileMode::Ptx)
}

/// Compile (or fetch from cache) in PTX or LTO-IR (`-dlto`) mode.
pub fn compile_with_mode(name: &str, source: &str, arch: &str, options: &[String], mode: CompileMode) -> Result<String, DriverError> {
    let opts_joined = options.join(" ");
    let (tag, ext) = match mode { CompileMode::Ptx => ("ptx", "ptx"), CompileMode::LtoIr => ("lto", "ltoir") };
    let key = fnv1a64(&[source, arch, &opts_joined, tag, env!("CARGO_PKG_VERSION")]);
    let path = cache_dir().join(format!("{name}-{key:016x}.{ext}"));
    if let Ok(ptx) = std::fs::read_to_string(&path) {
        return Ok(ptx);
    }
    let mut all = options.to_vec();
    all.push("--std=c++17".into());
    if mode == CompileMode::LtoIr && !all.iter().any(|o| o == "-dlto") {
        all.push("-dlto".into());
    }
    let compiled = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some(intern_arch(arch)),
            fmad: Some(false), // determinism: no contraction differences vs CPU reference
            name: Some(format!("{name}.cu")),
            options: all,
            ..Default::default()
        },
    )
    .map_err(|e| DriverError::Compile(format!("NVRTC failed for {name}: {e:?}")))?;
    let ptx = compiled.to_src();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &ptx).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
    Ok(ptx)
}
