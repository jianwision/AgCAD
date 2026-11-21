use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const NI: usize = 5;
const NH: usize = 24;
const NO: usize = 25;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 32) as u32 as f64 / u32::MAX as f64) as f32
    }
}
fn examples(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    let mut x = Vec::with_capacity(n * NI);
    let mut y = Vec::with_capacity(n * NO);
    for _ in 0..n {
        let mut a = [0.; NI];
        for p in &mut a {
            *p = 2. * rng.next() - 1.;
        }
        a[4] = if a[4] > 0. { 1. } else { -1. };
        let mut b = [-2.; NO];
        let codes = [1, 2, if a[4] > 0. { 3 } else { 0 }, 4];
        for s in 0..4 {
            b[5 * s + codes[s]] = 2.;
        }
        b[20..].copy_from_slice(&[
            1. + 0.2 * a[0],
            1. + 0.2 * a[1],
            1. + 0.2 * a[2],
            0.5 + 0.5 * a[3],
            1. + 0.1 * a[0],
        ]);
        x.extend(a);
        y.extend(b);
    }
    (x, y)
}
fn main() -> Result<()> {
    let start = Instant::now();
    let cuda = std::env::args().any(|s| s == "--cuda");
    // Explicit device selection: a requested CUDA training run never falls back.
    let dev = if cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let root = PathBuf::from(".");
    let dest = root.join("models");
    fs::create_dir_all(&dest)?;
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &dev);
    let l1 = candle_nn::linear(NI, NH, vb.pp("hidden"))?;
    let l2 = candle_nn::linear(NH, NO, vb.pp("output"))?;
    // Seeded initialization avoids backend-dependent random seeds.
    let mut init = Rng(84573);
    let mut named: Vec<_> = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, var) in named {
        let values: Vec<f32> = (0..var.elem_count())
            .map(|_| {
                if name.ends_with("bias") {
                    0.
                } else {
                    (init.next() - 0.5) * 0.4
                }
            })
            .collect();
        var.set(&Tensor::from_vec(values, var.shape(), &dev)?)?;
    }
    let forward = |x: &Tensor| l2.forward(&l1.forward(x)?.tanh()?);
    let (xs, ys) = examples(8192, 718);
    let xs = Tensor::from_vec(xs, (8192, NI), &dev)?;
    let ys = Tensor::from_vec(ys, (8192, NO), &dev)?;
    let mut weights = vec![1_f32; NO];
    weights[20..].fill(8_f32.sqrt());
    let loss_weights = Tensor::from_vec(weights, (1, NO), &dev)?;
    let mut opt = candle_nn::AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: 0.004,
            weight_decay: 0.,
            ..Default::default()
        },
    )?;
    let mut rng = Rng(78439);
    for step in 1..=4000 {
        let ids: Vec<u32> = (0..128).map(|_| (rng.next() * 8191.) as u32).collect();
        let ids = Tensor::from_vec(ids, (128,), &dev)?;
        let predicted = forward(&xs.index_select(&ids, 0)?)?;
        let loss = (predicted - ys.index_select(&ids, 0)?)?
            .broadcast_mul(&loss_weights)?
            .sqr()?
            .mean_all()?;
        opt.backward_step(&loss)?;
        if step % 1000 == 0 {
            println!(
                "Candle {dev:?}, step {step}, loss {}",
                loss.to_scalar::<f32>()?
            );
        }
    }
    let (xt, yt) = examples(4096, 991);
    let test = Tensor::from_vec(xt, (4096, NI), &dev)?;
    let predicted = forward(&test)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let mut correct = 0;
    let mut max_error = 0_f32;
    for (i, row) in predicted.iter().enumerate() {
        for s in 0..4 {
            let argmax = |a: &[f32]| {
                a.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            if argmax(&row[5 * s..5 * s + 5]) == argmax(&yt[i * NO + 5 * s..i * NO + 5 * s + 5]) {
                correct += 1;
            }
        }
        for j in 20..25 {
            max_error = max_error.max((row[j] - yt[i * NO + j]).abs());
        }
    }
    if correct != 4096 * 4 || max_error > 0.035 {
        return Err(format!(
            "Held-out validation failed: correct {correct}/16384; parameter error {max_error}"
        )
        .into());
    }
    let flat = |t: Tensor| -> candle_core::Result<Vec<f64>> {
        t.to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1()
    };
    let mut exported = flat(l1.weight().t()?.contiguous()?)?;
    exported.extend(flat(l1.bias().unwrap().clone())?);
    exported.extend(flat(l2.weight().t()?.contiguous()?)?);
    exported.extend(flat(l2.bias().unwrap().clone())?);
    if exported.len() != 769 {
        return Err("Weight layout mismatch".into());
    }
    fs::write(
        dest.join("geometry_policy.f64"),
        exported
            .iter()
            .flat_map(|a| a.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    vars.save(dest.join("geometry_policy.safetensors"))?;
    let report = format!(
        "{{\"framework\":\"Rust Candle 0.9.2\",\"device\":\"{dev:?}\",\"architecture\":[5,24,25],\"parameters\":769,\"weights_bytes\":6152,\"training_examples\":8192,\"steps\":4000,\"test_examples\":4096,\"opcode_slot_accuracy\":1.0,\"max_parameter_error\":{max_error},\"elapsed_s\":{},\"scope\":\"synthetic bounded recipe policy, not a language model\"}}",
        start.elapsed().as_secs_f64()
    );
    fs::write(dest.join("training_report.json"), &report)?;
    println!("{report}");
    Ok(())
}
