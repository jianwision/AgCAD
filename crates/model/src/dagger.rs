//! Expert-Iteration & DAgger trajectory dataset and deterministic MLP policy trainer (Phase 4 / L3).
//!
//! Logs `(obs, target_action, reward, committed)` tuples from winning CMA-ES / LLM rollouts,
//! distills them into a `CuModel` [`Mlp`] checkpoint (serialized as standard `safetensors`),
//! and verifies host-vs-device kernel parity (`AG-07-005`) before promotion.

use crate::cumodel::{forward_device, model_library, write_safetensors, Mlp};
use agcad_driver::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// One compressed supervision tuple from a device rollout or LLM expert trajectory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySample {
    /// Input observation vector (width = `n_in`, e.g. `OBS = 32` or tokenized input).
    pub obs: Vec<f32>,
    /// Expert target action vector (normalized parameter deltas / logits, width = `n_out`).
    pub action: Vec<f32>,
    /// Final episode reward (higher is better; used for advantage weighting).
    pub reward: f32,
    /// Whether the harness committed this step.
    pub committed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RolloutDataset {
    pub samples: Vec<TrajectorySample>,
}

impl RolloutDataset {
    pub fn push(&mut self, s: TrajectorySample) { self.samples.push(s); }

    /// Append expert relabeling samples (DAgger aggregation round).
    pub fn aggregate(&mut self, mut extra: Vec<TrajectorySample>) { self.samples.append(&mut extra); }

    pub fn save_jsonl(&self, path: &Path) -> std::io::Result<()> {
        let mut out = String::new();
        for s in &self.samples {
            out.push_str(&serde_json::to_string(s).unwrap());
            out.push('\n');
        }
        std::fs::write(path, out)
    }

    pub fn load_jsonl(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut samples = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(s) = serde_json::from_str::<TrajectorySample>(line) { samples.push(s); }
        }
        Ok(RolloutDataset { samples })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub hidden: usize,
    pub epochs: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig { hidden: 64, epochs: 40, lr: 1e-2, weight_decay: 1e-4, seed: 0xA9CAD_D066 }
    }
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub initial_mse: f32,
    pub final_mse: f32,
    pub samples: usize,
    pub safetensors: Vec<u8>,
    pub mlp: Mlp,
}

fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = ((*state >> 40) as u32) as f32 / (1u32 << 24) as f32;
    u * 2.0 - 1.0
}

/// Deterministic 2-layer ReLU MLP trainer (Behavioral Cloning / advantage-weighted DAgger).
/// Produces an [`Mlp`] and standard F32 `safetensors` bytes loadable directly into `ResidentMlp`.
pub fn train_mlp_bc(dataset: &RolloutDataset, cfg: TrainConfig) -> Result<TrainReport, String> {
    let valid: Vec<&TrajectorySample> = dataset.samples.iter().filter(|s| s.committed && !s.obs.is_empty() && !s.action.is_empty()).collect();
    if valid.is_empty() { return Err("dataset has no committed samples".into()); }
    let n_in = valid[0].obs.len();
    let n_out = valid[0].action.len();
    if valid.iter().any(|s| s.obs.len() != n_in || s.action.len() != n_out) {
        return Err("inconsistent obs/action widths in dataset".into());
    }
    let h = cfg.hidden.max(4);
    let mut rng = cfg.seed;
    let s0 = (2.0 / n_in as f32).sqrt();
    let s1 = (2.0 / h as f32).sqrt();
    let mut w0: Vec<f32> = (0..h * n_in).map(|_| lcg_next(&mut rng) * s0).collect();
    let mut b0 = vec![0.0f32; h];
    let mut w1: Vec<f32> = (0..n_out * h).map(|_| lcg_next(&mut rng) * s1).collect();
    let mut b1 = vec![0.0f32; n_out];

    let mut eval_mse = |w0: &[f32], b0: &[f32], w1: &[f32], b1: &[f32]| -> f32 {
        let mut tot = 0.0f32;
        let mut hid = vec![0.0f32; h];
        for s in &valid {
            for i in 0..h {
                let mut acc = b0[i];
                for j in 0..n_in { acc += w0[i * n_in + j] * s.obs[j]; }
                hid[i] = acc.max(0.0);
            }
            for o in 0..n_out {
                let mut acc = b1[o];
                for i in 0..h { acc += w1[o * h + i] * hid[i]; }
                let e = acc - s.action[o];
                tot += e * e;
            }
        }
        tot / (valid.len() * n_out) as f32
    };

    let initial_mse = eval_mse(&w0, &b0, &w1, &b1);
    let inv_n = 1.0 / valid.len() as f32;
    let mut gw0 = vec![0.0f32; w0.len()];
    let mut gb0 = vec![0.0f32; b0.len()];
    let mut gw1 = vec![0.0f32; w1.len()];
    let mut gb1 = vec![0.0f32; b1.len()];
    let mut pre = vec![0.0f32; h];
    let mut hid = vec![0.0f32; h];
    let mut d_out = vec![0.0f32; n_out];

    for _ in 0..cfg.epochs {
        gw0.fill(0.0); gb0.fill(0.0); gw1.fill(0.0); gb1.fill(0.0);
        for s in &valid {
            let wt = s.reward.clamp(0.1, 4.0);
            for i in 0..h {
                let mut acc = b0[i];
                for j in 0..n_in { acc += w0[i * n_in + j] * s.obs[j]; }
                pre[i] = acc;
                hid[i] = acc.max(0.0);
            }
            for o in 0..n_out {
                let mut acc = b1[o];
                for i in 0..h { acc += w1[o * h + i] * hid[i]; }
                d_out[o] = 2.0 * wt * (acc - s.action[o]) * inv_n / n_out as f32;
                gb1[o] += d_out[o];
                for i in 0..h { gw1[o * h + i] += d_out[o] * hid[i]; }
            }
            for i in 0..h {
                if pre[i] <= 0.0 { continue; }
                let mut dh = 0.0f32;
                for o in 0..n_out { dh += w1[o * h + i] * d_out[o]; }
                gb0[i] += dh;
                for j in 0..n_in { gw0[i * n_in + j] += dh * s.obs[j]; }
            }
        }
        for i in 0..w0.len() { w0[i] -= cfg.lr * (gw0[i] + cfg.weight_decay * w0[i]); }
        for i in 0..b0.len() { b0[i] -= cfg.lr * gb0[i]; }
        for i in 0..w1.len() { w1[i] -= cfg.lr * (gw1[i] + cfg.weight_decay * w1[i]); }
        for i in 0..b1.len() { b1[i] -= cfg.lr * gb1[i]; }
    }

    let final_mse = eval_mse(&w0, &b0, &w1, &b1);
    let mut tensors = BTreeMap::new();
    tensors.insert("layer0.weight".into(), (vec![h, n_in], w0.clone()));
    tensors.insert("layer0.bias".into(), (vec![h], b0.clone()));
    tensors.insert("layer1.weight".into(), (vec![n_out, h], w1.clone()));
    tensors.insert("layer1.bias".into(), (vec![n_out], b1.clone()));
    let safetensors = write_safetensors(&tensors);
    let mlp = Mlp::from_safetensors(&safetensors)?;
    Ok(TrainReport { initial_mse, final_mse, samples: valid.len(), safetensors, mlp })
}

/// Automated checkpoint parity check (`AG-07-005`): confirms host `Mlp::forward` matches
/// device kernel `ag_mlp_layer` within `max_abs_err` on a batch of probes.
pub fn verify_kernel_parity<D: KernelExt>(
    dev: &D,
    s: &D::Stream,
    mlp: &Mlp,
    probes: &[f32],
    configs: u32,
    max_abs_err: f32,
) -> Result<f32, String> {
    let host = mlp.forward(probes, configs as usize);
    let lib = dev.load_library(&model_library()).map_err(|e| e.to_string())?;
    let ob = dev.alloc_bytes(s, (probes.len() * 4).max(16)).map_err(|e| e.to_string())?;
    dev.upload_tagged_async(s, TransferKind::Control, bytemuck::cast_slice(probes), &ob, 0).map_err(|e| e.to_string())?;
    let yb = forward_device(dev, s, lib, mlp, &ob, configs).map_err(|e| e.to_string())?;
    let bytes = host.len() * 4;
    let mut pin = dev.pinned(bytes.max(4)).map_err(|e| e.to_string())?;
    dev.download_tagged_async(s, TransferKind::Report, &[Region { buf: &yb, offset: 0, bytes }], &mut pin).map_err(|e| e.to_string())?;
    dev.sync(s).map_err(|e| e.to_string())?;
    let got: Vec<f32> = bytemuck::pod_collect_to_vec(&pin.as_bytes()[..bytes]);
    let mut worst = 0.0f32;
    for (a, b) in host.iter().zip(&got) {
        worst = worst.max((a - b).abs());
    }
    if worst > max_abs_err {
        return Err(format!("AG-07-005 parity failed: max_abs_err={worst} > {max_abs_err}"));
    }
    Ok(worst)
}
