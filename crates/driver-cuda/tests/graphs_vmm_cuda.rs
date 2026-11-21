//! CUDA Graphs (capture/instantiate/update/launch), conditional nodes (IF, SWITCH, WHILE with a
//! device-set condition) and virtual memory management on real hardware (spec 06 § GPU-resident
//! harness; spec 05 § Memory). Requires an NVIDIA GPU: `cargo test -p agcad-driver-cuda -- --ignored`.

use agcad_driver::*;
use agcad_driver_cuda::CudaDevice;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SRC: &str = r#"
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __attribute__((cudart_builtin)) void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
struct AddP { double* x; unsigned int n; unsigned int pad; double v; };
extern "C" __global__ void t_add(const AddP* p) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p->n) p->x[i] += p->v;
}
struct MarkP { unsigned int* out; unsigned int v; unsigned int pad; };
extern "C" __global__ void t_mark(const MarkP* p) { if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(p->out, p->v); }
struct WhileP { cudaGraphConditionalHandle h; unsigned int* counter; unsigned int limit; unsigned int pad; };
extern "C" __global__ void t_while_step(const WhileP* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int c = *p->counter + 1;
        *p->counter = c;
        cudaGraphSetConditional(p->h, c < p->limit ? 1u : 0u);
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AddP { x: u64, n: u32, pad: u32, v: f64 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MarkP { out: u64, v: u32, pad: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WhileP { h: u64, counter: u64, limit: u32, pad: u32 }

unsafe fn unused(_: *const u8, _: &LaunchConfig) { unreachable!("CUDA-only test kernels") }
static KERNELS: &[(&str, CpuKernelFn)] = &[("t_add", unused), ("t_mark", unused), ("t_while_step", unused)];

fn setup() -> (Arc<CudaDevice>, <CudaDevice as AgDevice>::Stream, LibraryId) {
    let d = CudaDevice::open(0).expect("CUDA device");
    let s = d.stream(Priority::Normal).unwrap();
    let l = d.load_library(&LibrarySource { name: "agcad_test_graphs", cuda_source: Some(SRC.to_string()), cpu_kernels: KERNELS, options: vec![] }).unwrap();
    (d, s, l)
}

fn up<T: Pod>(d: &CudaDevice, s: &<CudaDevice as AgDevice>::Stream, v: &[T]) -> <CudaDevice as AgDevice>::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    let b = d.alloc_bytes(s, bytes.len().max(1)).unwrap();
    d.upload_from_host_async(s, bytes, &b, 0).unwrap();
    b
}

fn down<T: Pod>(d: &CudaDevice, s: &<CudaDevice as AgDevice>::Stream, b: &<CudaDevice as AgDevice>::Buffer, n: usize) -> Vec<T> {
    let bytes = n * std::mem::size_of::<T>();
    let mut pin = d.pinned(bytes).unwrap();
    d.download_packed_async(s, &[Region { buf: b, offset: 0, bytes }], &mut pin).unwrap();
    d.sync(s).unwrap();
    bytemuck::pod_collect_to_vec(&pin.as_bytes()[..bytes])
}

fn one() -> LaunchConfig { LaunchConfig::linear(1, 32) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn capture_replay_and_update() {
    let (d, s, l) = setup();
    let n = 1000u32;
    let x = up(&d, &s, &vec![0.0f64; n as usize]);
    let p1 = up(&d, &s, &[AddP { x: x.device_ptr(), n, pad: 0, v: 1.5 }]);
    let p2 = up(&d, &s, &[AddP { x: x.device_ptr(), n, pad: 0, v: 10.0 }]);
    d.sync(&s).unwrap();
    // capture two launches; replay the graph 3 times → 3 × 2 × 1.5
    d.begin_capture(&s, CaptureMode::ThreadLocal).unwrap();
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &p1).unwrap();
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &p1).unwrap();
    let g = d.end_capture(&s).unwrap();
    let mut exec = d.instantiate(&g).unwrap();
    for _ in 0..3 { d.launch_graph(&s, &exec).unwrap(); }
    assert!(down::<f64>(&d, &s, &x, n as usize).iter().all(|v| *v == 9.0));
    // same topology, different parameter block → in-place update
    d.begin_capture(&s, CaptureMode::ThreadLocal).unwrap();
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &p2).unwrap();
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &p2).unwrap();
    let g2 = d.end_capture(&s).unwrap();
    assert_eq!(d.update(&mut exec, &g2).unwrap(), UpdateResult::Updated);
    d.launch_graph(&s, &exec).unwrap();
    assert!(down::<f64>(&d, &s, &x, n as usize).iter().all(|v| *v == 29.0));
    // different topology → re-instantiated
    d.begin_capture(&s, CaptureMode::ThreadLocal).unwrap();
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &p1).unwrap();
    let g3 = d.end_capture(&s).unwrap();
    assert_eq!(d.update(&mut exec, &g3).unwrap(), UpdateResult::TopologyChanged);
    d.launch_graph(&s, &exec).unwrap();
    assert!(down::<f64>(&d, &s, &x, n as usize).iter().all(|v| *v == 30.5));
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn conditional_if_by_default_value() {
    let (d, s, l) = setup();
    for (default, expect) in [(1u32, 7u32), (0, 0)] {
        let out = up(&d, &s, &[0u32]);
        let p = up(&d, &s, &[MarkP { out: out.device_ptr(), v: 7, pad: 0 }]);
        d.sync(&s).unwrap();
        let mut g = d.new_graph().unwrap();
        let (_h, bodies) = d.conditional(&mut g, CondKind::If, default).unwrap();
        assert_eq!(bodies.len(), 1);
        d.capture_into(&s, &bodies[0], &mut || { d.launch(&s, l, "t_mark", &one(), &p)?; Ok(()) }).unwrap();
        let exec = d.instantiate(&g).unwrap();
        d.launch_graph(&s, &exec).unwrap();
        assert_eq!(down::<u32>(&d, &s, &out, 1), vec![expect], "IF default {default}");
    }
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn conditional_switch_selects_branch() {
    let (d, s, l) = setup();
    if d.caps().conditional_nodes != CondSupport::Full { eprintln!("SWITCH needs a CUDA 12.8+ driver; skipped"); return; }
    let out = up(&d, &s, &[0u32]);
    let ps: Vec<_> = [1u32, 10, 100].iter().map(|&v| up(&d, &s, &[MarkP { out: out.device_ptr(), v, pad: 0 }])).collect();
    d.sync(&s).unwrap();
    let mut g = d.new_graph().unwrap();
    let (_h, bodies) = d.conditional(&mut g, CondKind::Switch(3), 2).unwrap();
    assert_eq!(bodies.len(), 3);
    for (b, p) in bodies.iter().zip(&ps) {
        d.capture_into(&s, b, &mut || { d.launch(&s, l, "t_mark", &one(), p)?; Ok(()) }).unwrap();
    }
    let exec = d.instantiate(&g).unwrap();
    d.launch_graph(&s, &exec).unwrap();
    d.launch_graph(&s, &exec).unwrap();
    assert_eq!(down::<u32>(&d, &s, &out, 1), vec![200], "only branch 2 runs, twice");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn conditional_while_device_driven() {
    // The GPU-harness round loop: the body decides on the device whether to iterate again.
    let (d, s, l) = setup();
    let counter = up(&d, &s, &[0u32]);
    let n = 256u32;
    let x = up(&d, &s, &vec![0.0f64; n as usize]);
    let add = up(&d, &s, &[AddP { x: x.device_ptr(), n, pad: 0, v: 0.25 }]);
    let mut g = d.new_graph().unwrap();
    let (h, bodies) = d.conditional(&mut g, CondKind::While, 1).unwrap();
    let step = up(&d, &s, &[WhileP { h: h.0, counter: counter.device_ptr(), limit: 37, pad: 0 }]);
    d.sync(&s).unwrap();
    d.capture_into(&s, &bodies[0], &mut || {
        d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 128), &add)?;
        d.launch(&s, l, "t_while_step", &one(), &step)?;
        Ok(())
    })
    .unwrap();
    let exec = d.instantiate(&g).unwrap();
    d.launch_graph(&s, &exec).unwrap();
    assert_eq!(down::<u32>(&d, &s, &counter, 1), vec![37]);
    assert!(down::<f64>(&d, &s, &x, n as usize).iter().all(|v| *v == 37.0 * 0.25));
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn vmm_reserve_map_and_use() {
    let (d, s, l) = setup();
    let gran = d.granularity();
    assert!(gran.is_power_of_two() && gran >= 2 << 20);
    // reserve 4 pages of VA, back only pages 1 and 2 → a contiguous 2-page window
    let va = d.reserve(4 * gran).unwrap();
    let (p1, p2) = (d.create_page(gran).unwrap(), d.create_page(gran).unwrap());
    d.map(&va, gran, &p1, Access::ReadWrite).unwrap();
    d.map(&va, 2 * gran, &p2, Access::ReadWrite).unwrap();
    assert!(d.map(&va, 4 * gran, &p1, Access::ReadWrite).is_err(), "out-of-range map must be rejected");
    let base = va.base + gran as u64;
    let n = (2 * gran / 8) as u32; // spans both physical pages
    let zero = vec![0u8; 2 * gran];
    // stream-ordered copies on the test's own stream (legacy-stream copies are not ordered
    // with non-blocking streams used concurrently by other work in the process)
    unsafe { assert_eq!(cudarc::driver::sys::cuMemsetD8Async(base, 0, zero.len(), s.raw()), cudarc::driver::sys::CUresult::CUDA_SUCCESS) };
    let p = up(&d, &s, &[AddP { x: base, n, pad: 0, v: 3.0 }]);
    d.launch(&s, l, "t_add", &LaunchConfig::linear(n as u64, 256), &p).unwrap();
    d.sync(&s).unwrap();
    let mut host = vec![0.0f64; n as usize];
    unsafe { assert_eq!(cudarc::driver::sys::cuMemcpyDtoHAsync_v2(host.as_mut_ptr() as *mut _, base, host.len() * 8, s.raw()), cudarc::driver::sys::CUresult::CUDA_SUCCESS) };
    d.sync(&s).unwrap();
    assert!(host.iter().all(|v| *v == 3.0));
}
