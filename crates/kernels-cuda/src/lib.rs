//! # agcad-kernels-cuda — Tier N CUDA C++ sources
//!
//! Embedded `.cu` translation units plus a tiny include resolver (NVRTC has no access to
//! our source tree). Each library pairs its CUDA source with the CPU reference table from
//! `agcad-kernels` under the same kernel names.

use agcad_driver::LibrarySource;

/// Embedded headers resolvable by `#include "name"`.
pub const HEADERS: &[(&str, &str)] = &[
    ("ag_status.cuh", include_str!("../cu/ag_status.cuh")),
    ("ag_common.cuh", include_str!("../cu/ag_common.cuh")),
];

/// Inline `#include "x"` directives from [`HEADERS`] (each header at most once).
pub fn bundle(src: &str, extra_headers: &[(&str, &str)]) -> String {
    fn go(src: &str, hdrs: &[(&str, &str)], seen: &mut Vec<String>, out: &mut String) {
        for line in src.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("#include \"") {
                let name = rest.trim_end_matches('"');
                if let Some((_, body)) = hdrs.iter().find(|(n, _)| *n == name) {
                    if !seen.iter().any(|s| s == name) {
                        seen.push(name.to_string());
                        go(body, hdrs, seen, out);
                    }
                    continue;
                }
            }
            if t == "#pragma once" {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    let mut all: Vec<(&str, &str)> = HEADERS.to_vec();
    all.extend_from_slice(extra_headers);
    let mut out = String::new();
    go(src, &all, &mut Vec::new(), &mut out);
    out
}

pub fn primitives_library() -> LibrarySource {
    LibrarySource {
        name: agcad_kernels::PRIMITIVES,
        cuda_source: Some(bundle(include_str!("../cu/primitives.cu"), &[])),
        cpu_kernels: agcad_kernels::reference::PRIMITIVES_CPU,
        options: Vec::new(),
    }
}
