//! CUDA vs CPU-reference bit parity for the primitives (AG-08-001). Requires an NVIDIA GPU:
//! run with `cargo test -p agcad-kernels-cuda -- --ignored` on the NVIDIA CI runner.

use agcad_driver::*;
use agcad_kernels::primitives::*;

#[test]
#[ignore = "requires NVIDIA GPU"]
fn segmented_reduce_bit_parity() {
    let cuda = agcad_driver_cuda::CudaDevice::open(0).expect("CUDA device");
    let cs = cuda.stream(Priority::Normal).unwrap();
    let lib = agcad_kernels_cuda::primitives_library();
    let cl = cuda.load_library(&lib).unwrap();
    let cpu = agcad_driver_cpu::CpuDevice::new();
    let ps = cpu.stream(Priority::Normal).unwrap();
    let pl = cpu.load_library(&lib).unwrap();

    let vals: Vec<f64> = (0..100_000).map(|i| ((i as f64) * 0.1).sin() * 1e3 + 1e-7 * i as f64).collect();
    let offs: Vec<u32> = (0..=100).map(|i| i * 1000).collect();
    let bytes_v: &[u8] = bytemuck::cast_slice(&vals);
    let bytes_o: &[u8] = bytemuck::cast_slice(&offs);

    let (cv, co, cout) = (cuda.alloc_bytes(&cs, bytes_v.len()).unwrap(), cuda.alloc_bytes(&cs, bytes_o.len()).unwrap(), cuda.alloc_bytes(&cs, 800).unwrap());
    cuda.upload_from_host_async(&cs, bytes_v, &cv, 0).unwrap();
    cuda.upload_from_host_async(&cs, bytes_o, &co, 0).unwrap();
    segmented_reduce_f64(&*cuda, &cs, cl, &cv, &co, &cout, 100, ReduceOp::Sum).unwrap();
    let mut pin = cuda.pinned(800).unwrap();
    let ev = cuda.download_packed_async(&cs, &[Region { buf: &cout, offset: 0, bytes: 800 }], &mut pin).unwrap();
    cuda.sync(&cs).unwrap();
    assert!(cuda.query(&ev).unwrap());
    let gpu: Vec<f64> = bytemuck::cast_slice(pin.as_bytes()).to_vec();

    let (pv, po, pout) = (cpu.alloc_bytes(&ps, bytes_v.len()).unwrap(), cpu.alloc_bytes(&ps, bytes_o.len()).unwrap(), cpu.alloc_bytes(&ps, 800).unwrap());
    cpu.upload_from_host_async(&ps, bytes_v, &pv, 0).unwrap();
    cpu.upload_from_host_async(&ps, bytes_o, &po, 0).unwrap();
    segmented_reduce_f64(&*cpu, &ps, pl, &pv, &po, &pout, 100, ReduceOp::Sum).unwrap();
    let refv: Vec<f64> = bytemuck::cast_slice(pout.as_slice()).to_vec();
    for i in 0..100 {
        assert_eq!(gpu[i].to_bits(), refv[i].to_bits(), "segment {i}");
    }
}
