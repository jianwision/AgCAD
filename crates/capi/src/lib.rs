//! # agcad-capi — C ABI (`include/agcad.h`), post-MVP surface for non-Rust callers.
//! Strings returned by AgCAD must be released with `agcad_free_string`.

use std::ffi::{c_char, CStr, CString};

fn out(s: String) -> *mut c_char { CString::new(s).map(CString::into_raw).unwrap_or(std::ptr::null_mut()) }

/// Library version string.
#[unsafe(no_mangle)]
pub extern "C" fn agcad_version() -> *mut c_char { out(format!("agcad {} abi {}", env!("CARGO_PKG_VERSION"), agcad_driver::ABI_VERSION)) }

/// Stable name of an AgStatus value (cf. cudaGetErrorName); NULL if unknown.
#[unsafe(no_mangle)]
pub extern "C" fn agcad_status_name(v: u16) -> *mut c_char {
    agcad_driver::AgStatus::from_value(v).map(|s| out(s.name().into())).unwrap_or(std::ptr::null_mut())
}

/// Run an AgIR JSON program under requirements JSON; returns the report JSON (or an error JSON).
///
/// # Safety
/// `requirements` and `program` must be valid NUL-terminated UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn agcad_run_program(requirements: *const c_char, program: *const c_char) -> *mut c_char {
    let (r, p) = unsafe { (CStr::from_ptr(requirements).to_string_lossy().into_owned(), CStr::from_ptr(program).to_string_lossy().into_owned()) };
    let res = (|| -> Result<String, String> {
        let req = agcad_harness::Requirements::parse(&r)?;
        let cat = agcad_ir::Catalog::global();
        let prog = agcad_ir::json::decode_program(&p, cat).map_err(|e| e.to_string())?;
        let mut s = agcad_harness::HarnessSession::new(req, "part");
        let o = s.run_program("main", &prog, cat).map_err(|(st, m)| format!("{}: {m}", st.name()))?;
        serde_json::to_string(&serde_json::json!({ "report": o.report, "delivery": o.delivery })).map_err(|e| e.to_string())
    })();
    out(match res { Ok(s) => s, Err(e) => serde_json::json!({ "error": e }).to_string() })
}

/// Free a string returned by this library.
///
/// # Safety
/// `s` must come from an AgCAD function and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn agcad_free_string(s: *mut c_char) { if !s.is_null() { drop(unsafe { CString::from_raw(s) }); } }
