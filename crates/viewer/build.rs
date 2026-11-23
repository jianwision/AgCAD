use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(agcad_view)");
    println!("cargo:rerun-if-env-changed=GSTREAMER_1_0_ROOT_MSVC_X86_64");
    println!("cargo:rerun-if-changed=src/media.cpp");
    println!("cargo:rerun-if-changed=src/draw.hpp");
    if !env::var("TARGET").unwrap().ends_with("windows-msvc") {
        return;
    }
    let Some(root) = env::var_os("GSTREAMER_1_0_ROOT_MSVC_X86_64").map(PathBuf::from) else {
        return;
    };
    let mut build = cc::Build::new();
    for dir in [
        "include/gstreamer-1.0",
        "lib/gstreamer-1.0/include",
        "include/glib-2.0",
        "lib/glib-2.0/include",
    ] {
        build.include(root.join(dir));
    }
    build
        .cpp(true)
        .std("c++17")
        .flag("/EHsc")
        .define("GST_USE_UNSTABLE_API", None)
        .define("NOMINMAX", None)
        .file("src/media.cpp")
        .compile("agcad_view_media");
    println!("cargo:rustc-cfg=agcad_view");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    for lib in [
        "gstreamer-1.0",
        "gstwebrtc-1.0",
        "gstsdp-1.0",
        "gstd3d11-1.0",
        "glib-2.0",
        "gobject-2.0",
    ] {
        println!("cargo:rustc-link-lib={lib}");
        println!("cargo:rustc-link-arg-bin=agcad-viewer=/DELAYLOAD:{lib}-0.dll");
    }
    for lib in ["d3dcompiler", "delayimp"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
