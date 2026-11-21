//! Builds the OptiX shim (feature `optix`): OptiX headers from `OPTIX_ROOT` (or the default
//! install path), CUDA headers / import library from `CUDA_PATH`. OptiX itself is loaded
//! from the display driver at run time (`optixInit`), so nothing else is linked.

fn main() {
    println!("cargo:rerun-if-changed=shim/agcad_optix.cpp");
    println!("cargo:rerun-if-env-changed=OPTIX_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    #[cfg(feature = "optix")]
    build();
}

#[cfg(feature = "optix")]
fn build() {
    use std::path::PathBuf;
    let first = |cands: Vec<PathBuf>, what: &str| -> PathBuf {
        cands.into_iter().find(|p| p.exists()).unwrap_or_else(|| panic!("{what} not found: set the environment variable"))
    };
    let mut optix = Vec::new();
    if let Ok(p) = std::env::var("OPTIX_ROOT") { optix.push(PathBuf::from(p)); }
    for v in ["9.1.0", "9.0.0", "8.1.0", "8.0.0"] {
        optix.push(PathBuf::from(format!(r"C:\ProgramData\NVIDIA Corporation\OptiX SDK {v}")));
        optix.push(PathBuf::from(format!("/opt/optix-{v}")));
    }
    let optix = first(optix.into_iter().map(|p| p.join("include")).collect(), "OptiX SDK (OPTIX_ROOT)");
    let mut cuda = Vec::new();
    if let Ok(p) = std::env::var("CUDA_PATH") { cuda.push(PathBuf::from(p)); }
    for v in ["v13.0", "v12.9", "v12.8", "v12.6", "v12.4"] { cuda.push(PathBuf::from(format!(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{v}"))); }
    cuda.push(PathBuf::from("/usr/local/cuda"));
    let cuda = first(cuda, "CUDA toolkit (CUDA_PATH)");
    cc::Build::new()
        .cpp(true)
        .file("shim/agcad_optix.cpp")
        .include(&optix)
        .include(cuda.join("include"))
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .compile("agcad_optix");
    for lib in [cuda.join("lib").join("x64"), cuda.join("lib64")] { if lib.exists() { println!("cargo:rustc-link-search=native={}", lib.display()); } }
    println!("cargo:rustc-link-lib=cuda");
    // optixInit locates the driver DLL through the registry on Windows
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") { println!("cargo:rustc-link-lib=advapi32"); }
    // headers for the device programs (compiled with NVRTC at run time)
    println!("cargo:rustc-env=AGCAD_OPTIX_INCLUDE={}", optix.display());
    println!("cargo:rustc-env=AGCAD_CUDA_INCLUDE={}", cuda.join("include").display());
}
