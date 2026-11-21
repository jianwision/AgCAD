//! `agcad-train`: Candle trainer for CuModel policies (separate process; spec 08, review F15).
//! Input: JSONL dataset of {"obs": [f32], "target": [f32]} (expert-iteration / DAgger labels
//! produced by `agcad-improve`). Output: safetensors with `layer{i}.weight` / `.bias`.

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (data, out) = (args.get(1).ok_or("usage: agcad-train <data.jsonl> <out.safetensors> [hidden] [epochs]")?, args.get(2).ok_or("missing output path")?);
    let hidden: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let epochs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for line in std::fs::read_to_string(data)?.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        xs.push(v["obs"].as_array().ok_or("obs")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>());
        ys.push(v["target"].as_array().ok_or("target")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>());
    }
    let (n, ni, no) = (xs.len(), xs[0].len(), ys[0].len());
    let device = Device::cuda_if_available(0)?;
    let x = Tensor::from_vec(xs.concat(), (n, ni), &device)?;
    let y = Tensor::from_vec(ys.concat(), (n, no), &device)?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let l0: Linear = linear(ni, hidden, vb.pp("layer0"))?;
    let l1: Linear = linear(hidden, hidden, vb.pp("layer1"))?;
    let l2: Linear = linear(hidden, no, vb.pp("layer2"))?;
    let mut opt = candle_nn::AdamW::new_lr(vm.all_vars(), 1e-3)?;
    for epoch in 0..epochs {
        let pred = l2.forward(&l1.forward(&l0.forward(&x)?.relu()?)?.relu()?)?;
        let loss = pred.sub(&y)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 50 == 0 { eprintln!("epoch {epoch}: loss {}", loss.to_scalar::<f32>()?); }
    }
    vm.save(out)?;
    eprintln!("saved {out} (Candle names layer{{i}}.weight/bias; parity-check with agcad-model before promotion, AG-07-005)");
    Ok(())
}
