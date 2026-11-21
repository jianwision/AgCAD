//! Diagnostic proposal model, separate from the existing four-slot recipe model.
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::{error::Error, fs, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
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
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let q: Vec<f32> = (0..6).map(|_| 2. * rng.next() - 1.).collect();
        let target: Vec<f32> = (0..6)
            .map(|j| {
                if i % 3 == 0 {
                    q[j] + 0.05 * (2. * rng.next() - 1.)
                } else {
                    2. * rng.next() - 1.
                }
            })
            .collect();
        x.extend(&q);
        x.extend(&target);
        for j in 0..6 {
            y.push(0.6 * (target[j] - q[j]));
        }
    }
    (x, y)
}
fn main() -> Result<()> {
    let start = Instant::now();
    let mut args = std::env::args().skip(1);
    let dest = PathBuf::from(
        args.next()
            .ok_or("Use probe_policy OUTPUT_DIRECTORY [--cuda]")?,
    );
    let flag = args.next();
    if flag.as_deref().is_some_and(|s| s != "--cuda") || args.next().is_some() {
        return Err("Invalid proposal trainer arguments".into());
    }
    let dev = if flag.is_some() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    fs::create_dir_all(&dest)?;
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &dev);
    let a = candle_nn::linear(12, 24, vb.pp("hidden"))?;
    let b = candle_nn::linear(24, 6, vb.pp("output"))?;
    let mut named: Vec<_> = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    let mut rng = Rng(925719);
    for (name, var) in named {
        let values: Vec<_> = (0..var.elem_count())
            .map(|_| {
                if name.ends_with("bias") {
                    0.
                } else {
                    0.15 * (2. * rng.next() - 1.)
                }
            })
            .collect();
        var.set(&Tensor::from_vec(values, var.shape(), &dev)?)?;
    }
    let forward = |x: &Tensor| b.forward(&a.forward(x)?.tanh()?);
    let (x, y) = examples(8192, 9241);
    let x = Tensor::from_vec(x, (8192, 12), &dev)?;
    let y = Tensor::from_vec(y, (8192, 6), &dev)?;
    let mut opt = candle_nn::AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: 0.003,
            weight_decay: 0.,
            ..Default::default()
        },
    )?;
    for step in 1..=4000 {
        let ids: Vec<u32> = (0..128).map(|_| (rng.next() * 8191.) as u32).collect();
        let ids = Tensor::from_vec(ids, (128,), &dev)?;
        let loss = (forward(&x.index_select(&ids, 0)?)? - y.index_select(&ids, 0)?)?
            .sqr()?
            .mean_all()?;
        opt.backward_step(&loss)?;
        if step % 1000 == 0 {
            println!(
                "Proposal Candle {dev:?}, step {step}, loss {}",
                loss.to_scalar::<f32>()?
            );
        }
    }
    let (x, y) = examples(2048, 572381);
    let pred = forward(&Tensor::from_vec(x, (2048, 12), &dev)?)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mse = pred
        .iter()
        .zip(&y)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / y.len() as f64;
    if !mse.is_finite() || mse > 0.0025 {
        return Err(format!("Proposal diagnostic holdout MSE {mse}").into());
    }
    let flat = |t: Tensor| -> candle_core::Result<Vec<f64>> {
        t.to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1()
    };
    let mut weights = flat(a.weight().t()?.contiguous()?)?;
    weights.extend(flat(a.bias().unwrap().clone())?);
    weights.extend(flat(b.weight().t()?.contiguous()?)?);
    weights.extend(flat(b.bias().unwrap().clone())?);
    assert_eq!(weights.len(), 462);
    fs::write(
        dest.join("probe_policy.f64"),
        weights
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    vars.save(dest.join("probe_policy.safetensors"))?;
    let report = format!(
        "{{\"architecture\":[12,24,6],\"parameters\":462,\"device\":\"{dev:?}\",\"steps\":4000,\"seed\":925719,\"training_examples\":8192,\"heldout_examples\":2048,\"heldout_delta_mse_normalized\":{mse},\"elapsed_s\":{},\"scope\":\"diagnostic probe-delta proposal model; synthetic imitation, no recursive improvement or generalization claim\"}}",
        start.elapsed().as_secs_f64()
    );
    fs::write(dest.join("training_report.json"), &report)?;
    println!("{report}");
    Ok(())
}
