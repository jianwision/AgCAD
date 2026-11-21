//! Primitives on the CPU reference device against naive implementations.

use agcad_driver::*;
use agcad_driver_cpu::CpuDevice;
use agcad_kernels::primitives::*;

fn lib(dev: &CpuDevice) -> LibraryId {
    dev.load_library(&LibrarySource { name: agcad_kernels::PRIMITIVES, cuda_source: None, cpu_kernels: agcad_kernels::reference::PRIMITIVES_CPU, options: vec![] }).unwrap()
}

fn up<T: bytemuck::Pod>(dev: &CpuDevice, s: &<CpuDevice as AgDevice>::Stream, v: &[T]) -> <CpuDevice as AgDevice>::Buffer {
    let b = dev.alloc_bytes(s, std::mem::size_of_val(v).max(1)).unwrap();
    dev.upload_from_host_async(s, bytemuck::cast_slice(v), &b, 0).unwrap();
    b
}

fn down<T: bytemuck::Pod>(b: &<CpuDevice as AgDevice>::Buffer, n: usize) -> Vec<T> {
    bytemuck::cast_slice::<u8, T>(&b.as_slice()[..n * std::mem::size_of::<T>()]).to_vec()
}

#[test]
fn scan_matches_naive_across_tiles() {
    let dev = CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();
    let l = lib(&dev);
    let n = 5000u32;
    let v: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % 11).collect();
    let input = up(&dev, &s, &v);
    let out = dev.alloc_bytes(&s, n as usize * 4).unwrap();
    exclusive_scan_u32(&*dev, &s, l, &input, &out, n).unwrap();
    let got: Vec<u32> = down(&out, n as usize);
    let mut acc = 0;
    for i in 0..n as usize {
        assert_eq!(got[i], acc, "at {i}");
        acc += v[i];
    }
}

#[test]
fn radix_sort_is_stable() {
    let dev = CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();
    let l = lib(&dev);
    let n = 3000u32;
    let keys: Vec<u64> = (0..n as u64).map(|i| (i * 2654435761) % 97).collect();
    let vals: Vec<u32> = (0..n).collect();
    let kb = up(&dev, &s, &keys);
    let vb = up(&dev, &s, &vals);
    radix_sort_pairs(&*dev, &s, l, &kb, &vb, n, 64).unwrap();
    let k: Vec<u64> = down(&kb, n as usize);
    let v: Vec<u32> = down(&vb, n as usize);
    let mut expect: Vec<(u64, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
    expect.sort_by_key(|p| p.0); // stable
    assert_eq!(k, expect.iter().map(|p| p.0).collect::<Vec<_>>());
    assert_eq!(v, expect.iter().map(|p| p.1).collect::<Vec<_>>());
}

#[test]
fn top_k_ties_break_by_configuration_id() {
    let dev = CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();
    let l = lib(&dev);
    let scores = [3.0f64, 1.0, 2.0, 1.0, f64::NAN, -0.0, 0.0];
    let n = scores.len() as u32;
    let sb = up(&dev, &s, &scores);
    let kb = dev.alloc_bytes(&s, n as usize * 8).unwrap();
    let vb = dev.alloc_bytes(&s, n as usize * 4).unwrap();
    stable_order_by_score(&*dev, &s, l, &sb, &kb, &vb, n, false).unwrap();
    let ids: Vec<u32> = down(&vb, n as usize);
    assert_eq!(&ids[..4], &[5, 6, 1, 3]); // -0.0 < 0.0 by bits, then the two 1.0 in id order
    assert_eq!(ids[6], 4); // NaN last
}

#[test]
fn segmented_reduce_sum_min_max() {
    let dev = CpuDevice::new();
    let s = dev.stream(Priority::Normal).unwrap();
    let l = lib(&dev);
    let vals: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
    let offs: Vec<u32> = vec![0, 10, 10, 1000];
    let vb = up(&dev, &s, &vals);
    let ob = up(&dev, &s, &offs);
    let out = dev.alloc_bytes(&s, 3 * 8).unwrap();
    segmented_reduce_f64(&*dev, &s, l, &vb, &ob, &out, 3, ReduceOp::Max).unwrap();
    let r: Vec<f64> = down(&out, 3);
    assert_eq!(r[0], 4.5);
    assert_eq!(r[1], f64::NEG_INFINITY);
    assert_eq!(r[2], 499.5);
}
