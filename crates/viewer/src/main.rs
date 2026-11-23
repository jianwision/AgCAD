#[cfg(agcad_view)]
mod assistant;
#[cfg(agcad_view)]
mod create;
#[cfg(agcad_view)]
mod document;
#[cfg(agcad_view)]
mod materials;
#[cfg(agcad_view)]
mod media;
#[cfg(agcad_view)]
mod service;

fn main() {
    #[cfg(agcad_view)]
    if let Err(e) = service::run() {
        eprintln!("AgCAD viewport: {e}");
        std::process::exit(1);
    }
    #[cfg(not(agcad_view))]
    {
        eprintln!(
            "AgCAD native streaming requires Windows MSVC and the GStreamer development SDK."
        );
        std::process::exit(1);
    }
}
