//! The LLM ⇄ GPU-model loop (spec 07): the LLM observes renders/descriptions, states a goal
//! (`set_goal`), and a device policy (today the mock GPU model) plans and executes CAD
//! operations on thousands of configurations under the GPU harness (`device_rollout`). The best
//! survivor comes back as ordinary Part Design features on a new branch, ready to render,
//! describe, check and export — and the LLM observes again.

use agcad_dops::abi::Goal;
use agcad_dops::lower::to_layered;
use agcad_dops::{RolloutSummary, Studio, StudioConfig};
use bytemuck::Zeroable;
use serde_json::{json, Value};

/// Parse a `set_goal` request against the design's parameter names.
pub fn parse_goal(args: &Value, names: &[String], body_bbox: Option<([f64; 3], [f64; 3])>) -> Result<(Goal, Vec<String>), String> {
    let mut g = Goal::zeroed();
    let mut notes = Vec::new();
    let num = |v: &Value| -> Option<f64> { v.as_f64().or_else(|| v.as_str().and_then(|s| agcad_ir::Quantity::parse(s).ok().map(|q| q.value))) };
    let tv = args.get("target_volume").and_then(Value::as_array).ok_or("target_volume: [min, max] in mm³ is required")?;
    g.vol_lo = tv.first().and_then(num).ok_or("target_volume[0]")?;
    g.vol_hi = tv.get(1).and_then(num).ok_or("target_volume[1]")?;
    if !(g.vol_hi > g.vol_lo && g.vol_lo > 0.0) { return Err("target_volume must be 0 < min < max".into()); }
    g.max_ext = [1e6; 3];
    if let Some(m) = args.get("max_extent").and_then(Value::as_array) {
        for k in 0..3 { g.max_ext[k] = m.get(k).and_then(num).ok_or("max_extent: [x, y, z]")?; }
    }
    let params = args.get("parameters").and_then(Value::as_object).ok_or("parameters: {name: {min, max, volume_effect: \"+\"|\"-\"}} is required")?;
    for (k, name) in names.iter().enumerate().take(agcad_dops::abi::MAX_PARAMS) {
        match params.get(name) {
            Some(p) => {
                g.param_lo[k] = p.get("min").and_then(num).ok_or(format!("{name}.min"))?;
                g.param_hi[k] = p.get("max").and_then(num).ok_or(format!("{name}.max"))?;
                g.param_sign[k] = match p.get("volume_effect").and_then(Value::as_str) { Some("-") | Some("decreases") => -1.0, _ => 1.0 };
            }
            None => notes.push(format!("parameter {name} is not in the goal: held fixed")),
        }
    }
    // the policy varies parameters 0..n_params; fixed ones get a zero-width range
    g.n_params = names.len().min(agcad_dops::abi::MAX_PARAMS) as u32;
    for (k, name) in names.iter().enumerate().take(agcad_dops::abi::MAX_PARAMS) {
        if !params.contains_key(name) { g.param_lo[k] = f64::NAN; }
    }
    for unknown in params.keys().filter(|n| !names.contains(n)) { return Err(format!("parameter '{unknown}' is not a parameter of the design (known: {})", names.join(", "))); }
    // evaluation domain: explicit, or the body box grown to the extent limit / 50 %
    if let (Some(lo), Some(hi)) = (args.pointer("/domain/min").and_then(Value::as_array), args.pointer("/domain/max").and_then(Value::as_array)) {
        for k in 0..3 { g.domain_lo[k] = lo.get(k).and_then(num).ok_or("domain.min")?; g.domain_hi[k] = hi.get(k).and_then(num).ok_or("domain.max")?; }
    } else {
        let (lo, hi) = body_bbox.ok_or("no current body: give domain {min, max}")?;
        for k in 0..3 {
            let c = 0.5 * (lo[k] + hi[k]);
            let half = if g.max_ext[k] < 1e5 { 0.5 * g.max_ext[k] } else { 0.75 * (hi[k] - lo[k]) };
            let half = half.max(0.5 * (hi[k] - lo[k])) + 0.1 * (hi[k] - lo[k]).max(1.0) + 2.0;
            g.domain_lo[k] = c - half;
            g.domain_hi[k] = c + half;
        }
        notes.push(format!("domain derived from the current body: {:?} – {:?}", g.domain_lo, g.domain_hi));
    }
    if let Some(h) = args.get("new_holes") {
        g.hole_r = h.get("r").and_then(num).unwrap_or(0.0);
        let reg = h.get("region").and_then(Value::as_array).ok_or("new_holes.region: [x0, y0, x1, y1]")?;
        for k in 0..4 { g.place[k] = reg.get(k).and_then(num).ok_or("new_holes.region")?; }
        g.hole_z0 = g.domain_lo[2] + 1e-3;
        g.hole_z1 = g.domain_hi[2] - 1e-3;
    }
    g.gain = args.get("gain").and_then(Value::as_f64).unwrap_or(0.5);
    g.seed = args.get("seed").and_then(Value::as_u64).unwrap_or(0xA9CAD) as u32;
    Ok((g, notes))
}

/// Fixed parameters (NaN lower bound) become a zero-width range at their base value.
pub fn freeze_fixed(g: &mut Goal, base: &[f64]) {
    for (k, v) in base.iter().enumerate().take(agcad_dops::abi::MAX_PARAMS) {
        if g.param_lo[k].is_nan() { g.param_lo[k] = *v; g.param_hi[k] = *v; g.param_sign[k] = 0.0; }
    }
}

pub struct RolloutRun {
    pub summary: RolloutSummary,
    pub device: &'static str,
    pub mode: &'static str,
    pub wall_ms: f64,
}

/// Instantiated CUDA rollout graphs kept between `device_rollout` calls, keyed by shape:
/// a repeat rollout with a new goal or base design only rewrites device buffers.
#[derive(Default)]
pub struct DeviceCache {
    cuda: Option<((u32, u32, u32, u32, u32, u32, usize), agcad_dops::GraphRollout<agcad_driver_cuda::CudaDevice>)>,
}

/// Run on CUDA (cached graph WHILE) when available and requested, else on the CPU reference.
pub fn run(cache: &mut DeviceCache, design: &agcad_dops::Design, goal: &Goal, cfg: StudioConfig, top_k: usize, prefer_cuda: bool) -> Result<RolloutRun, String> {
    let t0 = std::time::Instant::now();
    if prefer_cuda {
        if let Ok(dev) = agcad_driver_cuda::CudaDevice::open(0) {
            let key = (cfg.configs, cfg.nodes, cfg.records, cfg.max_rounds, cfg.levels, cfg.cap, design.params.len());
            let reused = matches!(&cache.cuda, Some((k, _)) if *k == key);
            if reused {
                cache.cuda.as_mut().unwrap().1.reset(design, goal).map_err(|e| e.to_string())?;
            } else {
                let st = Studio::new(dev, design, goal, cfg).map_err(|e| e.to_string())?;
                cache.cuda = Some((key, agcad_dops::GraphRollout::new(st).map_err(|e| e.to_string())?));
            }
            let gr = &cache.cuda.as_ref().unwrap().1;
            gr.run().map_err(|e| e.to_string())?;
            let summary = gr.studio.results(top_k).map_err(|e| e.to_string())?;
            return Ok(RolloutRun { summary, device: "cuda", mode: if reused { "graph (device-driven WHILE, cached)" } else { "graph (device-driven WHILE)" }, wall_ms: t0.elapsed().as_secs_f64() * 1e3 });
        }
    }
    let st = Studio::new(agcad_driver_cpu::CpuDevice::new(), design, goal, cfg).map_err(|e| e.to_string())?;
    st.run_host_loop().map_err(|e| e.to_string())?;
    let summary = st.results(top_k).map_err(|e| e.to_string())?;
    Ok(RolloutRun { summary, device: "cpu-reference", mode: "host loop", wall_ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// JSON summary for the LLM (compact; no geometry).
pub fn summarize(r: &RolloutRun, names: &[String]) -> Value {
    let s = &r.summary;
    let top: Vec<Value> = s.top.iter().map(|sv| {
        let params: serde_json::Map<String, Value> = names.iter().zip(&sv.design.params).map(|(n, v)| (n.clone(), json!((v * 1e6).round() / 1e6))).collect();
        json!({
            "configuration": sv.configuration, "goal_met": sv.obs[10] == 1.0, "parameters": params,
            "volume_estimate": (sv.obs[12] * 10.0).round() / 10.0, "volume_bounds": [sv.obs[0], sv.obs[1]],
            "extent_bounds_hi": [sv.obs[5], sv.obs[6], sv.obs[7]], "features": sv.design.nodes.len(),
            "evidence": { "volume_bounds": "bounded", "volume_estimate": "sampled" },
        })
    }).collect();
    json!({ "device": r.device, "mode": r.mode, "rounds": s.rounds, "configurations_meeting_goal": s.done, "wall_ms": (r.wall_ms * 10.0).round() / 10.0, "host_readback_bytes_during_rollout": s.in_loop_readback_bytes, "top": top })
}

/// Exact check of a survivor (escalation) without touching the document.
pub fn exact_volume(d: &agcad_dops::Design) -> Result<f64, String> {
    let s = to_layered(d)?;
    Ok(agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0)
}

pub use agcad_dops::lower::raise as raise_design;
pub use agcad_dops::lower::to_program as lower_program;
