//! NVRTC compilation with an on-disk cache keyed by (source hash, arch, flags, catalog
//! version) (spec 08 § Kernel authoring; review F7). nvJitLink/LTO-IR linking is added once
//! the build pass confirms cudarc's nvJitLink coverage; until then PTX per library.

use agcad_driver::DriverError;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::path::PathBuf;

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
    let opts_joined = options.join(" ");
    let key = fnv1a64(&[source, arch, &opts_joined, env!("CARGO_PKG_VERSION")]);
    let path = cache_dir().join(format!("{name}-{key:016x}.ptx"));
    if let Ok(ptx) = std::fs::read_to_string(&path) {
        return Ok(ptx);
    }
    let mut all = options.to_vec();
    all.push("--std=c++17".into());
    let compiled = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some(Box::leak(arch.to_string().into_boxed_str())),
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
