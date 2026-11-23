use agcad_driver::{AgDevice, DeviceFault, DriverError, EnqueueError, KernelExt, LaunchConfig};
use agcad_driver_cuda::CudaDevice;
use agcad_harness::requirements::Requirements;
use agcad_server::Server;

#[test]
#[ignore = "requires CUDA and NVRTC"]
fn workers_cache_the_requested_module() {
    const NAME: &str = "geometry_cache_regression";
    const A: &str = r#"extern "C" __global__ void first(const void*) {}"#;
    const B: &str = r#"extern "C" __global__ void first(const void*) {}
                        extern "C" __global__ void second(const void*) {}"#;
    let first = std::thread::spawn(|| {
        CudaDevice::geometry_device()
            .unwrap()
            .load_cuda_source(NAME, A, ["first"].into_iter(), &[])
            .unwrap()
    })
    .join()
    .unwrap();
    std::thread::spawn(move || {
        let (device, stream) = CudaDevice::geometry_stream().unwrap();
        let load = |source, symbols: &[&str], options: &[String]| {
            device
                .load_cuda_source(NAME, source, symbols.iter().copied(), options)
                .unwrap()
        };
        let changed = load(B, &["first"], &[]);
        assert_ne!(first, changed);
        let extra = load(B, &["first", "second"], &[]);
        assert_ne!(extra, changed);
        assert_ne!(extra, load(B, &["first", "second"], &["-DCONFIG=1".into()]));
        assert_eq!(extra, load(B, &["first", "second"], &[]));
        let buffer = device.alloc_bytes(&stream, 16).unwrap();
        device
            .launch(
                &stream,
                extra,
                "second",
                &LaunchConfig::linear(1, 1),
                &buffer,
            )
            .unwrap();
        device.sync(&stream).unwrap();
    })
    .join()
    .unwrap();
}

#[test]
fn poisoned_worker_rejects_new_work() {
    // Isolate the process-lifetime fault marker from other cases in this test executable.
    if std::env::var_os("AGCAD_TEST_FAULT_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "poisoned_worker_rejects_new_work"])
            .env("AGCAD_TEST_FAULT_CHILD", "1")
            .env("AGCAD_ALLOW_CPU_REFERENCE", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let requirements =
        Requirements::parse(r#"{"id":"r","rev":1,"deliver":{"body_type":"SolidBody"}}"#).unwrap();
    let mut server = Server::new(requirements, "p");
    let recoverable = DriverError::Enqueue(EnqueueError::OutOfMemory { requested: 4096 });
    CudaDevice::note_geometry_error(&recoverable);
    agcad_geo::accelerator::ensure_worker_healthy().unwrap();
    CudaDevice::note_geometry_error(&DriverError::Fault(DeviceFault {
        sticky: true,
        code: 700,
        segment: None,
    }));
    assert!(matches!(
        CudaDevice::geometry_device(),
        Err(DriverError::Enqueue(EnqueueError::Poisoned))
    ));
    assert!(agcad_runtime::session::Session::open(0).is_err());
    assert!(agcad_runtime::session::with_default(|_| ()).is_err());
    assert!(
        server
            .call_tool("get_feature_table", &serde_json::json!({}))
            .unwrap_err()
            .contains("poisoned")
    );
    assert!(server.session.save_version("main", "after fault").is_err());
    assert!(agcad_geo::accelerator::mass_properties(&[]).is_err());
}
