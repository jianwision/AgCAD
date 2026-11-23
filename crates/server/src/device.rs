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
pub fn parse_goal(args: &Value, names: &[String], body_bbox: Option<([f64; 3], [f64; 3])>, process_3ax: bool) -> Result<(Goal, Vec<String>), String> {
    let mut g = Goal::zeroed();
    let mut notes = Vec::new();
    let num = |v: &Value| -> Option<f64> { v.as_f64().or_else(|| v.as_str().and_then(|s| agcad_ir::Quantity::parse(s).ok().map(|q| q.value))) };
    g.objective = match args.get("objective").and_then(Value::as_str).unwrap_or("satisfy") {
        "satisfy" => 0,
        "min_volume" => 1,
        "max_volume" => 2,
        "explore" => 3,
        "moead" | "pareto" => 4,
        o => return Err(format!("objective: \"satisfy\" | \"min_volume\" | \"max_volume\" | \"explore\" | \"moead\" (got {o})")),
    };
    match args.get("target_volume").and_then(Value::as_array) {
        Some(tv) => {
            g.vol_lo = tv.first().and_then(num).ok_or("target_volume[0]")?;
            g.vol_hi = tv.get(1).and_then(num).ok_or("target_volume[1]")?;
            if !(g.vol_hi > g.vol_lo && g.vol_lo > 0.0) { return Err("target_volume must be 0 < min < max".into()); }
        }
        None if g.objective == 0 => return Err("target_volume: [min, max] in mm³ is required (or an objective)".into()),
        None => {}
    }
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
    // engineering constraints (sampled on the device; see the summary's evidence levels)
    if let Some(v) = args.get("min_wall") { g.min_wall = num(v).filter(|x| *x > 0.0).ok_or("min_wall: thickness > 0 (mm)")?; }
    if let Some(v) = args.get("min_clearance") { g.min_gap = num(v).filter(|x| *x > 0.0).ok_or("min_clearance: gap > 0 (mm)")?; }
    match args.get("machining").and_then(Value::as_str) {
        Some("3ax") => g.machining_3ax = 1,
        // checked by ray casting on survivors only (the device probe is +z)
        Some("3+2") | Some("5ax") => notes.push("machining checked by ray casting on survivors (dfm escalation)".into()),
        Some("none") => {}
        Some(o) => return Err(format!("machining: \"3ax\" | \"3+2\" | \"5ax\" | \"none\" (got {o})")),
        None => if process_3ax { g.machining_3ax = 1; notes.push("requirements process machining_3ax: 3-axis tool access is checked".into()); },
    }
    if let Some(v) = args.get("undercut_tolerance") { g.undercut_tol = num(v).ok_or("undercut_tolerance (mm²)")?; }
    if let Some(c) = args.get("centroid") {
        let lo = c.get("min").and_then(Value::as_array).ok_or("centroid: {min: [x, y, z], max: [x, y, z]}")?;
        let hi = c.get("max").and_then(Value::as_array).ok_or("centroid.max")?;
        for k in 0..3 { g.centroid_lo[k] = lo.get(k).and_then(num).ok_or("centroid.min")?; g.centroid_hi[k] = hi.get(k).and_then(num).ok_or("centroid.max")?; }
    }
    g.gain = args.get("gain").and_then(Value::as_f64).unwrap_or(0.5);
    g.seed = args.get("seed").and_then(Value::as_u64).unwrap_or(0xA9CAD) as u32;
    Ok((g, notes))
}

/// Stiffness requirement (escalation on survivors; evidence `sampled`: voxel discretization).
#[derive(Clone, Copy, Debug)]
pub struct LoadSpec { pub support_axis: u32, pub force: [f64; 3], pub voxel: Option<f64>, pub e: f64, pub nu: f64, pub max_compliance: Option<f64> }

pub fn parse_load(args: &Value) -> Result<Option<LoadSpec>, String> {
    let Some(l) = args.get("load") else {
        return if args.get("max_compliance").is_some() { Err("max_compliance needs a load {support, force}".into()) } else { Ok(None) };
    };
    let num = |v: &Value| -> Option<f64> { v.as_f64().or_else(|| v.as_str().and_then(|s| agcad_ir::Quantity::parse(s).ok().map(|q| q.value))) };
    let support_axis = match l.get("support").and_then(Value::as_str).unwrap_or("z_min") { "x_min" => 0, "y_min" => 1, "z_min" => 2, o => return Err(format!("load.support: x_min | y_min | z_min (got {o})")) };
    let f = l.get("force").and_then(Value::as_array).ok_or("load.force: [fx, fy, fz] in N")?;
    let mut force = [0.0; 3];
    for k in 0..3 { force[k] = f.get(k).and_then(num).ok_or("load.force: [fx, fy, fz]")?; }
    let voxel = l.get("voxel").and_then(num);
    if voxel.is_some_and(|h| h <= 0.0) { return Err("load.voxel must be > 0 (mm)".into()); }
    Ok(Some(LoadSpec { support_axis, force, voxel, e: l.get("E").and_then(num).unwrap_or(210_000.0), nu: l.get("nu").and_then(num).unwrap_or(0.3), max_compliance: args.get("max_compliance").and_then(num) }))
}

/// Stiffness of a survivor on its outer box (reuses the session's shared `CudaDevice`).
pub fn stiffness(cache: &mut DeviceCache, design: &agcad_dops::Design, obs: &[f64], spec: &LoadSpec, prefer_cuda: bool) -> Result<(agcad_dops::fea::FeaReport, f64), String> {
    use agcad_dops::abi::ob;
    let lo = [obs[ob::BOX_LO], obs[ob::BOX_LO + 1], obs[ob::BOX_LO + 2]];
    let hi = [lo[0] + obs[5], lo[1] + obs[6], lo[2] + obs[7]];
    let h = spec.voxel.unwrap_or_else(|| obs[5].max(obs[6]).max(obs[7]) / 40.0).max(1e-3);
    let mut lc = agcad_dops::fea::LoadCase::over(lo, hi, h);
    lc.support_axis = spec.support_axis;
    lc.force = spec.force;
    lc.e = spec.e;
    lc.nu = spec.nu;
    let r = if let Some(dev) = prefer_cuda.then(|| cache.cuda_device()).flatten() {
        agcad_dops::fea::stiffness(dev, design, &lc)
    } else {
        agcad_dops::fea::stiffness(agcad_driver_cpu::CpuDevice::new(), design, &lc)
    };
    r.map(|r| (r, h)).map_err(|e| e.to_string())
}

/// Fixed parameters (NaN lower bound) become a zero-width range at their base value.
pub fn freeze_fixed(g: &mut Goal, base: &[f64]) {
    for (k, v) in base.iter().enumerate().take(agcad_dops::abi::MAX_PARAMS) {
        if g.param_lo[k].is_nan() { g.param_lo[k] = *v; g.param_hi[k] = *v; g.param_sign[k] = 0.0; }
    }
}

/// Fast pre-rollout reachability & sensitivity diagnostic (`set_goal` v2 / Pillar C2):
/// computes Reynolds-transport `∂V/∂p` at the base design and estimates the first-order
/// reachable volume range over `[param_lo, param_hi]` so the LLM sees immediately if a goal
/// is unreachable by continuous parameter edits alone.
pub fn probe_reachability(design: &agcad_dops::Design, goal: &Goal) -> Value {
    let grad = agcad_dops::grad::reynolds_shape_gradients(design, goal, 24);
    let mut lo_est = grad.volume;
    let mut hi_est = grad.volume;
    let mut sens = serde_json::Map::new();
    for (j, name) in design.names.iter().enumerate().take(agcad_dops::abi::MAX_PARAMS) {
        let dv = grad.d_vol.get(j).copied().unwrap_or(0.0);
        let p0 = design.params[j];
        let (plo, phi) = (goal.param_lo[j], goal.param_hi[j]);
        if phi > plo {
            let d_lo = dv * (plo - p0);
            let d_hi = dv * (phi - p0);
            lo_est += d_lo.min(d_hi);
            hi_est += d_lo.max(d_hi);
        }
        sens.insert(name.clone(), json!({
            "d_vol_dp": (dv * 100.0).round() / 100.0,
            "discontinuous": grad.discontinuous.get(j).copied().unwrap_or(false),
        }));
    }
    lo_est = lo_est.max(0.0);
    let target_reachable = if goal.vol_hi > goal.vol_lo {
        hi_est >= goal.vol_lo * 0.9 && lo_est <= goal.vol_hi * 1.1
    } else {
        true
    };
    json!({
        "base_volume_sampled": (grad.volume * 10.0).round() / 10.0,
        "linear_reachable_volume": [(lo_est * 10.0).round() / 10.0, (hi_est * 10.0).round() / 10.0],
        "target_likely_reachable": target_reachable,
        "sensitivities": sens,
        "recommendation": if target_reachable { "continuous parameters span target" } else { "target appears outside continuous parameter bounds; consider widening ranges or enabling structural moves (new_holes)" },
    })
}

pub struct RolloutRun {
    pub summary: RolloutSummary,
    pub device: &'static str,
    pub mode: &'static str,
    pub wall_ms: f64,
}

/// Shared session CUDA device + instantiated rollout graphs kept between calls:
/// a repeat rollout with a new goal or base design only rewrites device buffers, and
/// certification / FEA / OptiX / mesh extraction all share one CUDA context (no extra opens).
#[derive(Default)]
pub struct DeviceCache {
    dev: Option<Option<std::sync::Arc<agcad_driver_cuda::CudaDevice>>>,
    cuda: Option<((u32, u32, u32, u32, u32, u32, usize, u32, u32), agcad_dops::GraphRollout<agcad_driver_cuda::CudaDevice>)>,
}

impl DeviceCache {
    /// Lazily open and cache device 0 once per server session.
    pub fn cuda_device(&mut self) -> Option<std::sync::Arc<agcad_driver_cuda::CudaDevice>> {
        self.dev.get_or_insert_with(|| agcad_driver_cuda::CudaDevice::open(0).ok()).clone()
    }
}

/// Certify a design on the shared GPU low-priority stream when CUDA is available, else CPU ref.
pub fn certify(cache: &mut DeviceCache, design: &agcad_dops::Design, goal: &Goal, levels: u32, cap: u32, prefer_cuda: bool) -> Result<[f64; agcad_dops::abi::OBS], String> {
    if let Some(dev) = prefer_cuda.then(|| cache.cuda_device()).flatten() {
        agcad_dops::studio::certify(dev, design, goal, levels, cap).map_err(|e| e.to_string())
    } else {
        agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), design, goal, levels, cap).map_err(|e| e.to_string())
    }
}

/// Run on CUDA (cached graph WHILE) when available and requested, else on the CPU reference.
pub fn run(cache: &mut DeviceCache, design: &agcad_dops::Design, goal: &Goal, cfg: StudioConfig, top_k: usize, prefer_cuda: bool) -> Result<RolloutRun, String> {
    let t0 = std::time::Instant::now();
    if let Some(dev) = prefer_cuda.then(|| cache.cuda_device()).flatten() {
        let key = (cfg.configs, cfg.nodes, cfg.records, cfg.max_rounds, cfg.levels, cfg.cap, design.params.len(), cfg.policy, cfg.island);
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
            "engineering": engineering(&sv.obs),
            "evidence": { "volume_bounds": "bounded", "extent_bounds": "bounded", "volume_estimate": "sampled", "engineering": "sampled" },
        })
    }).collect();
    json!({ "device": r.device, "mode": r.mode, "rounds": s.rounds, "configurations_meeting_goal": s.done, "wall_ms": (r.wall_ms * 10.0).round() / 10.0, "host_readback_bytes_during_rollout": s.in_loop_readback_bytes, "top": top })
}

/// Engineering metrics of an observation row (sampled evidence).
pub fn engineering(o: &[f64]) -> Value {
    use agcad_dops::abi::ob;
    let r = |x: f64, k: f64| (x * k).round() / k;
    let on = |x: f64| if x < 0.0 { Value::Null } else { json!(r(x, 1e3)) };
    json!({
        "centroid": [r(o[ob::CENTROID], 1e3), r(o[ob::CENTROID + 1], 1e3), r(o[ob::CENTROID + 2], 1e3)],
        "inertia_unit_density": [r(o[ob::INERTIA], 1.0), r(o[ob::INERTIA + 1], 1.0), r(o[ob::INERTIA + 2], 1.0)],
        "surface_area": r(o[ob::AREA], 10.0),
        "min_wall_sampled": on(o[ob::MIN_WALL]), "thin_wall_area": r(o[ob::THIN_AREA], 10.0),
        "min_clearance_sampled": on(o[ob::MIN_GAP]), "narrow_gap_area": r(o[ob::GAP_AREA], 10.0),
        "undercut_area_3ax": r(o[ob::UNDERCUT_AREA], 10.0),
        "violation_area": r(o[ob::VIOLATION], 10.0), "centroid_ok": o[ob::CENTROID_OK] == 1.0,
    })
}

/// A certified implicit body: watertight mesh with rigorous evidence.
pub struct Implicit { pub mesh: agcad_dops::mesh::ImplicitMesh, pub certified: [f64; agcad_dops::abi::OBS], pub reason: String }

impl Implicit {
    pub fn summary(&self) -> Value {
        json!({ "evidence": "implicit-certified", "reason": self.reason,
                "certified_volume_bounds": [self.certified[0], self.certified[1]], "certified_extent_hi": [self.certified[5], self.certified[6], self.certified[7]],
                "mesh": { "triangles": self.mesh.mesh.triangles.len(), "watertight": self.mesh.mesh.is_watertight(), "volume": self.mesh.volume, "spacing": self.mesh.h,
                          "max_distance_to_surface": self.mesh.deviation },
                "exports": ["3mf", "stl"], "note": "STEP needs an exact B-rep; features thinner than the mesh spacing may be missing from the mesh but never from the certified bounds" })
    }
}

/// Certify a device-only survivor and extract its mesh on the shared device (spacing ≈ largest extent / 200).
pub fn deliver_implicit(cache: &mut DeviceCache, design: &agcad_dops::Design, goal: &Goal, reason: &str, prefer_cuda: bool) -> Result<Implicit, String> {
    use agcad_dops::abi::ob;
    let run = |certified: [f64; agcad_dops::abi::OBS], m: Result<agcad_dops::mesh::ImplicitMesh, String>| m.map(|mesh| Implicit { mesh, certified, reason: reason.to_string() });
    let cuda = prefer_cuda.then(|| cache.cuda_device()).flatten();
    let cert = match &cuda {
        Some(dev) => agcad_dops::studio::certify(dev.clone(), design, goal, 7, 1 << 18).map_err(|e| e.to_string())?,
        None => agcad_dops::studio::certify(agcad_driver_cpu::CpuDevice::new(), design, goal, 7, 1 << 18).map_err(|e| e.to_string())?,
    };
    let lo = [cert[ob::BOX_LO], cert[ob::BOX_LO + 1], cert[ob::BOX_LO + 2]];
    let hi = [lo[0] + cert[5], lo[1] + cert[6], lo[2] + cert[7]];
    let h = (cert[5].max(cert[6]).max(cert[7]) / 200.0).max(0.05);
    let m = match cuda {
        Some(dev) => agcad_dops::mesh::extract_mesh(dev, design, lo, hi, h),
        None => agcad_dops::mesh::extract_mesh(agcad_driver_cpu::CpuDevice::new(), design, lo, hi, h),
    };
    run(cert, m.map_err(|e| e.to_string()))
}

/// The exactly-lowerable part of a design for the document: device-only nodes removed
/// (non-exact shells, revolutions of sloped / curved meridians) and device-only modifiers
/// cleared (drafts, rounded / chamfered caps).
pub fn exact_part(d: &agcad_dops::Design) -> agcad_dops::Design {
    use agcad_dops::abi::{nflags, profile};
    let rectilinear = |n: &agcad_dops::Node| {
        let idx = n.val[15].floor().max(0.0) as usize;
        d.profiles.get(idx).and_then(|lp| agcad_dops::lower::sketch_profile(lp, 0.0, 0.0, 0.0).ok()).is_some_and(|p| agcad_geo::layered::LayeredSolid::revolve_rectilinear([0.0, 0.0], &p).is_ok())
    };
    let nodes = d.nodes.iter().filter(|n| n.kind != profile::SHELL && (n.kind != profile::REVOLVE || rectilinear(n))).map(|n| {
        let mut m = *n;
        m.flags &= !(nflags::DRAFT | nflags::ROUND_Z | nflags::CHAMFER_Z);
        m
    }).collect();
    agcad_dops::Design { nodes, ..d.clone() }
}

/// Ray-cast DFM requirement checked on survivors (escalation step, evidence `sampled`).
#[derive(Clone, Debug)]
pub struct DfmSpec { pub set: Option<String>, pub min_wall: f64, pub min_clearance: f64, pub undercut_tol: f64 }

/// From `set_goal`: machining (3ax / 3+2 / 5ax), min_wall, min_clearance (and the
/// requirements' 3-axis process); `ray_check: false` disables it.
pub fn parse_dfm(args: &Value, g: &Goal) -> Option<DfmSpec> {
    if args.get("ray_check").and_then(Value::as_bool) == Some(false) { return None; }
    let set = match args.get("machining").and_then(Value::as_str) { Some("none") => None, Some(s) => Some(s.to_string()), None => (g.machining_3ax != 0).then(|| "3ax".to_string()) };
    if set.is_none() && !(g.min_wall > 0.0) && !(g.min_gap > 0.0) { return None; }
    Some(DfmSpec { set, min_wall: g.min_wall, min_clearance: g.min_gap, undercut_tol: g.undercut_tol })
}

/// The survivor's surface mesh: its exact solid, or the certified implicit mesh.
pub fn survivor_mesh(cache: &mut DeviceCache, d: &agcad_dops::Design, goal: &Goal, prefer_cuda: bool) -> Result<agcad_geo::tessellate::Mesh, String> {
    match to_layered(d) {
        Ok(s) => Ok(agcad_geo::tessellate::tessellate(&s.brep(), &Default::default())),
        Err(_) => Ok(deliver_implicit(cache, d, goal, "dfm", prefer_cuda)?.mesh.mesh),
    }
}

/// Iso render (PNG) of a design: its exact solid, or the certified implicit mesh for
/// device-only designs. Reuses the session's shared `CudaDevice` for OptiX / GPU rasterization.
pub fn render_design(cache: &mut DeviceCache, d: &agcad_dops::Design, obs: &[f64], goal: &Goal, prefer_cuda: bool) -> Result<Vec<u8>, String> {
    use agcad_harness::render::{png, rasterize, rasterize_device, Mode, View};
    let mesh = match to_layered(d) {
        Ok(s) => agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()),
        Err(reason) => { let _ = (obs, reason); deliver_implicit(cache, d, goal, "render", prefer_cuda)?.mesh.mesh }
    };
    let (w, h) = (256, 256);
    if let Some(dev) = prefer_cuda.then(|| cache.cuda_device()).flatten() {
        if let Ok(t) = agcad_raytrace::OptixTracer::new_on(dev.clone(), &mesh) {
            if let Ok(p) = agcad_raytrace::analysis::render(&t, &mesh, w, h) { return Ok(p); }
        }
        if let Ok(s) = agcad_driver::AgDevice::stream(&*dev, agcad_driver::Priority::Low) {
            if let Ok(rgb) = rasterize_device(&*dev, &s, &mesh, View::Iso, Mode::Shaded, w, h) {
                return Ok(png(&rgb, w, h));
            }
        }
    }
    Ok(png(&rasterize(&mesh, View::Iso, Mode::Shaded, w, h), w, h))
}

/// Exact check of a survivor (escalation) without touching the document.
pub fn exact_volume(d: &agcad_dops::Design) -> Result<f64, String> {
    let s = to_layered(d)?;
    Ok(agcad_geo::tessellate::tessellate(&s.brep(), &Default::default()).mass_properties().0)
}

pub use agcad_dops::lower::raise as raise_design;
pub use agcad_dops::lower::to_program as lower_program;
