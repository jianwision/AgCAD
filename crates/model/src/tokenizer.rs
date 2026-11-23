//! CANTO-style Feature-Tape Tokenizer (Phase 4 / Pillar B2):
//!
//! Converts a parametric CAD feature tape (nodes + normalized parameters) and live observation /
//! gradient rows into fixed-width `f32` token embeddings ready for `CuModel` (`ResidentMlp` or
//! sequence/Transformer policies).

use serde::{Deserialize, Serialize};

/// Width of one feature-tape token (`f32`).
pub const TOKEN_DIM: usize = 16;

/// One structured feature token (op/profile kind, boolean sense, modifier bits, normalized values).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TapeToken {
    pub data: [f32; TOKEN_DIM],
}

/// Lightweight host/device feature descriptor for tokenization (decoupled from `agcad-dops`
/// so `agcad-model` stays lower in the dependency DAG).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureTokenInput {
    pub kind: u32,
    pub op: u32,
    pub flags: u32,
    pub values: [f64; 16],
}

/// Encode a sequence of feature nodes into normalized CANTO-style tokens.
/// `domain_scale` normalizes spatial coordinates into roughly `[-1, 1]`.
pub fn tokenize_tape(features: &[FeatureTokenInput], domain_scale: f64) -> Vec<TapeToken> {
    let inv = 1.0 / domain_scale.abs().max(1.0);
    features
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let mut d = [0.0f32; TOKEN_DIM];
            d[0] = (f.kind as f32) / 10.0;
            d[1] = match f.op { 0 => 1.0, 1 => -1.0, _ => 0.0 };
            d[2] = if f.flags & 1 != 0 { 1.0 } else { 0.0 }; // active
            d[3] = ((f.flags >> 1) as f32).ln_1p() / 10.0; // pattern / modifier flags
            d[4] = (idx as f32) / (features.len().max(1) as f32);
            for k in 0..6 {
                d[5 + k] = (f.values[k] * inv).clamp(-4.0, 4.0) as f32;
            }
            d[11] = (f.values[6] / 64.0).clamp(-2.0, 2.0) as f32; // pattern count / period
            d[12] = (f.values[7] * inv).clamp(-2.0, 2.0) as f32;  // angle / spacing / wall
            d[13] = (f.values[10] * inv).clamp(-2.0, 2.0) as f32; // fillet / chamfer
            d[14] = (f.values[12]).clamp(-2.0, 2.0) as f32;       // draft tan(theta)
            d[15] = (f.values[13] * inv).clamp(-2.0, 2.0) as f32; // cap round/chamfer
            TapeToken { data: d }
        })
        .collect()
}

/// Pool a variable-length token sequence into a fixed-length `[TOKEN_DIM * 2]` summary
/// (elementwise mean + elementwise max) and concatenate normalized observation + shape-gradient
/// features for fixed-width `ResidentMlp` policies.
pub fn pack_policy_input(
    tokens: &[TapeToken],
    obs: &[f64],
    shape_grad_vol: &[f64],
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    let mut pos = 0usize;
    // 1. Live observation vector (normalized via log/clamp)
    for &v in obs {
        if pos >= out_dim { return out; }
        let x = if v.is_finite() { v as f32 } else { 0.0 };
        out[pos] = x.signum() * x.abs().ln_1p();
        pos += 1;
    }
    // 2. Mean-pooled & max-pooled CANTO feature-tape tokens
    if !tokens.is_empty() {
        let mut mean = [0.0f32; TOKEN_DIM];
        let mut maxv = [f32::NEG_INFINITY; TOKEN_DIM];
        for t in tokens {
            for k in 0..TOKEN_DIM {
                mean[k] += t.data[k];
                if t.data[k] > maxv[k] { maxv[k] = t.data[k]; }
            }
        }
        let inv_n = 1.0 / tokens.len() as f32;
        for k in 0..TOKEN_DIM {
            if pos >= out_dim { return out; }
            out[pos] = mean[k] * inv_n;
            pos += 1;
        }
        for k in 0..TOKEN_DIM {
            if pos >= out_dim { return out; }
            out[pos] = maxv[k];
            pos += 1;
        }
    }
    // 3. Normalized Reynolds-transport shape gradients ∂V/∂p_j
    let g_norm = shape_grad_vol.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
    for &g in shape_grad_vol {
        if pos >= out_dim { return out; }
        out[pos] = (g / g_norm) as f32;
        pos += 1;
    }
    out
}
