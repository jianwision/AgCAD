use crate::Result;
use std::fs;

#[cfg(windows)]
pub fn preload_nvrtc() -> Result<Vec<libloading::Library>> {
    let root = std::env::var_os("CUDA_PATH")
        .or_else(|| std::env::var_os("CUDA_HOME"))
        .ok_or("Set CUDA_PATH to the CUDA Toolkit installation directory")?;
    let bin = std::path::PathBuf::from(root).join("bin");
    let mut libraries = Vec::new();
    for entry in fs::read_dir(&bin)? {
        let path = entry?.path();
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("nvrtc-builtins"))
            && path.extension().is_some_and(|extension| extension == "dll")
        {
            // NVRTC loads its builtins by name, so preload those by absolute path too.
            libraries.push(unsafe { libloading::Library::new(&path)? });
        }
    }
    for candidate in cudarc::get_lib_name_candidates("nvrtc") {
        let path = bin.join(candidate);
        if path.is_file() {
            // SAFETY: load the installed toolkit library and retain its handle.
            let library = unsafe { libloading::Library::new(&path) }
                .map_err(|e| format!("Unable to load {}: {e}", path.display()))?;
            libraries.push(library);
            return Ok(libraries);
        }
    }
    Err(format!("NVRTC DLL not found under {}", bin.display()).into())
}
