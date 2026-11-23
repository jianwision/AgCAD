//! Design studies (spec 03 `study.*`, spec 05, spec 14): configurations from design tables or
//! grids, batched updates with topology classes, evaluation through the unified
//! Evaluation/Constraint/Requirement model, stable ranking with exact-rebuild fallback,
//! proxy-fidelity metrics, optimization, sensitivity, tolerance analysis, observations.
//!
//! This is the host reference path (bit-identical semantics to the GPU wavefront for the
//! ranking/masking stages, see `wavefront`). Feature evaluation per configuration uses the
//! runtime rebuild; the fully device-resident feature kernels land with P11.

use crate::checker::{check, CheckResult, Verdict};
use crate::requirements::{HardRequirement, Requirements};
use agcad_driver::AgStatus;
use agcad_geo::kernel::{ExactKernel, Solid};
use agcad_runtime::document::PartState;
use agcad_runtime::rebuild::{update, Product, Registry, UpdateResult};
use agcad_runtime::sched::{class_histogram, topology_classes, TopologySignature};
use agcad_solver::opt;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub type Config = BTreeMap<String, f64>;

/// `study.configurations`: grid `{"name": [lo, hi, n]}` or `{"name": [v1, v2, ...]}`
/// (lists of ≠ 3 numbers or explicit `{"values": [...]}`), or explicit rows.
pub fn configurations(spec: &Value) -> Result<Vec<Config>, String> {
    if let Some(rows) = spec.get("rows").and_then(Value::as_array) {
        let cols: Vec<String> = spec.get("params").and_then(Value::as_array).ok_or("rows need params")?.iter().filter_map(|c| c.as_str().map(String::from)).collect();
        return rows.iter().map(|r| { let r = r.as_array().ok_or("row must be a list")?; Ok(cols.iter().cloned().zip(r.iter().map(|x| x.as_f64().unwrap_or(f64::NAN))).collect()) }).collect();
    }
    let grid = spec.get("grid").unwrap_or(spec).as_object().ok_or("expected a grid object")?;
    let mut axes: Vec<(String, Vec<f64>)> = Vec::new();
    for (k, v) in grid {
        let vals: Vec<f64> = match v {
            Value::Array(a) if a.len() == 3 && a[2].is_u64() && a[2].as_u64().unwrap() > 1 && !spec.get("explicit").is_some() => {
                let (lo, hi, n) = (num(&a[0])?, num(&a[1])?, a[2].as_u64().unwrap() as usize);
                (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
            }
            Value::Array(a) => a.iter().map(num).collect::<Result<_, _>>()?,
            Value::Object(o) if o.get("values").is_some() => o["values"].as_array().ok_or("values must be a list")?.iter().map(num).collect::<Result<_, _>>()?,
            other => vec![num(other)?],
        };
        axes.push((k.clone(), vals));
    }
    let mut out: Vec<Config> = vec![Config::new()];
    for (k, vals) in &axes {
        out = out.iter().flat_map(|c| vals.iter().map(move |v| { let mut c2 = c.clone(); c2.insert(k.clone(), *v); c2 })).collect();
    }
    Ok(out)
}

fn num(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => agcad_ir::Quantity::parse(s).map(|q| q.value),
        other => Err(format!("expected a number, got {other}")),
    }
}

pub struct ConfigResult {
    pub update: UpdateResult,
    pub signature: TopologySignature,
    pub body: Option<u32>,
}

impl ConfigResult {
    fn solid(&self) -> Option<&Solid> {
        match self.update.products.get(&self.body?)? {
            Product::Solid(solid) => Some(solid),
            _ => None,
        }
    }
}

/// `study.update_configurations`: rebuild every configuration; group by topology signature.
pub fn update_configurations(part: &PartState, registry: &Registry, kernel: &dyn ExactKernel, cfgs: &[Config]) -> Vec<ConfigResult> {
    cfgs.iter().map(|c| {
        let u = update(part, registry, kernel, c.clone(), vec![]);
        let active: Vec<u32> = u.outcomes.iter().filter(|o| o.status.is_ok()).map(|o| o.feature).collect();
        let body = u.products.iter().rev().find(|(_, p)| matches!(p, Product::Solid(_))).map(|(k, _)| *k);
        let counts: Vec<u32> = match body.and_then(|b| u.products.get(&b)) {
            Some(Product::Solid(s)) => { let b = s.brep(); vec![b.faces.len() as u32, b.edges.len() as u32, b.vertices.len() as u32] }
            _ => vec![],
        };
        ConfigResult { signature: TopologySignature::of(&active, &counts, 0), body, update: u }
    }).collect()
}

#[derive(Clone, Debug)]
pub struct Evaluated {
    pub configuration: usize,
    pub score: f64,
    pub checks: Vec<CheckResult>,
    pub feasible: bool,
}

/// `study.evaluate`: objective (minimize/maximize a field of an evaluation) + hard checks.
pub fn evaluate(part: &PartState, registry: &Registry, kernel: &dyn ExactKernel, results: &[ConfigResult], objective: Option<(&str, &str, bool)>, hard: &[HardRequirement]) -> Result<Vec<Evaluated>, String> {
    let mut scores = vec![None; results.len()];
    if let Some((measure, field, _)) = objective {
        if let Some(op) = registry.get(measure) {
            let (indices, solids): (Vec<_>, Vec<_>) = results.iter().enumerate()
                .filter_map(|(i, result)| Some((i, result.solid()?))).unzip();
            if let Some(values) = op.batch_objective(kernel, &solids, field).map_err(|e| e.to_string())? {
                for (index, value) in indices.into_iter().zip(values) {
                    scores[index] = Some(value);
                }
            }
        }
    }
    Ok(results.iter().enumerate().map(|(i, r)| {
        let Some(body) = r.body else { return Evaluated { configuration: i, score: f64::INFINITY, checks: vec![], feasible: false } };
        let checks: Vec<CheckResult> = hard.iter().map(|h| check(h, part, kernel, registry, &r.update.products, body)).collect();
        let feasible = r.update.ok() && checks.iter().all(|c| c.verdict == Verdict::Pass);
        let score = match objective {
            Some((measure, field, maximize)) => {
                let v = scores[i].unwrap_or_else(|| {
                    let h = HardRequirement { id: "objective".into(), measure: measure.into(), field: field.into(), le: Some(f64::MAX), ge: None, eq: None, evidence: crate::requirements::Evidence::Unknown, args: BTreeMap::new() };
                    check(&h, part, kernel, registry, &r.update.products, body).value.as_f64().unwrap_or(f64::NAN)
                });
                if maximize { -v } else { v }
            }
            None => 0.0,
        };
        Evaluated { configuration: i, score, checks, feasible }
    }).collect())
}

/// `study.rank`: stable top-k (ties by configuration id; infeasible/NaN last), identical
/// ordering to `primitives::stable_order_by_score` on the GPU (AG-05-001).
pub fn rank(ev: &[Evaluated], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ev.len()).collect();
    let key = |e: &Evaluated| agcad_kernels::primitives::f64_order_key(if e.feasible { e.score } else { f64::INFINITY }, false);
    order.sort_by_key(|&i| (key(&ev[i]), i));
    order.into_iter().take(k).collect()
}

/// Rank with exact-rebuild fallback (spec 05 § Proxy fidelity): survivors failing the exact
/// check are replaced by the next-ranked configuration; the report lists replacements.
pub fn rank_with_fallback(part: &PartState, registry: &Registry, kernel: &dyn ExactKernel, results: &[ConfigResult], ev: &[Evaluated], k: usize) -> (Vec<usize>, Vec<Value>) {
    let order = rank(ev, ev.len());
    let mut chosen = Vec::new();
    let mut fallbacks = Vec::new();
    let validity = HardRequirement { id: "exact_validity".into(), measure: "evaluate.check".into(), field: "valid".into(), le: None, ge: None, eq: Some(json!(true)), evidence: crate::requirements::Evidence::Exact, args: BTreeMap::new() };
    for i in order {
        if chosen.len() == k { break; }
        if !ev[i].feasible { continue; }
        let Some(body) = results[i].body else { continue };
        let c = check(&validity, part, kernel, registry, &results[i].update.products, body);
        if c.verdict == Verdict::Pass { chosen.push(i); } else { fallbacks.push(json!({ "configuration": i, "reason": "exact rebuild failed validity", "status": AgStatus::TopologyFailed.name() })); }
    }
    (chosen, fallbacks)
}

/// Kendall τ between two score vectors (proxy vs exact ranking agreement).
pub fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 { return 1.0; }
    let (mut c, mut d) = (0i64, 0i64);
    for i in 0..n { for j in i + 1..n {
        let s = (a[i] - a[j]).signum() * (b[i] - b[j]).signum();
        if s > 0.0 { c += 1 } else if s < 0.0 { d += 1 }
    }}
    (c - d) as f64 / (n * (n - 1) / 2) as f64
}

/// Proxy fidelity: proxy = coarse tessellation volume, exact = fine tessellation volume of
/// the survivors (spec 05 § Proxy fidelity; recorded per study).
pub fn proxy_fidelity(kernel: &dyn ExactKernel, results: &[ConfigResult], survivors: &[usize]) -> Result<Value, String> {
    let (slots, solids): (Vec<_>, Vec<_>) = survivors.iter().enumerate()
        .filter_map(|(slot, &i)| Some((slot, results[i].solid()?))).unzip();
    let volumes = |sag| -> Result<Vec<f64>, String> {
        let mut values = vec![f64::NAN; survivors.len()];
        for (&slot, (volume, _, _)) in slots.iter().zip(agcad_std::batch_mass(kernel, &solids, sag).map_err(|e| e.to_string())?) {
            values[slot] = volume;
        }
        Ok(values)
    };
    let proxy = volumes(0.5)?;
    let exact = volumes(1e-3)?;
    Ok(json!({ "kendall_tau": kendall_tau(&proxy, &exact), "survivors": survivors.len(), "proxy": "tessellation sag 0.5 mm", "exact": "tessellation sag 0.001 mm" }))
}

pub fn topology_report(results: &[ConfigResult]) -> Value {
    let sigs: Vec<TopologySignature> = results.iter().map(|r| r.signature).collect();
    let classes = topology_classes(&sigs);
    json!({ "classes": classes.len(), "histogram": class_histogram(&classes), "configurations": results.len() })
}

/// `study.sensitivity`: central finite differences of an objective w.r.t. parameters.
/// Evidence `sampled` until autodiff through features is wired (spec 04 § Autodiff).
pub fn sensitivity(f: &dyn Fn(&Config) -> f64, at: &Config, step: f64) -> Value {
    let mut grads = BTreeMap::new();
    for k in at.keys() {
        let (mut up, mut dn) = (at.clone(), at.clone());
        *up.get_mut(k).unwrap() += step;
        *dn.get_mut(k).unwrap() -= step;
        grads.insert(k.clone(), (f(&up) - f(&dn)) / (2.0 * step));
    }
    json!({ "gradient": grads, "method": "finite_difference", "evidence": "sampled" })
}

/// `study.optimize`: derivative-free optimization of parameters within a budget.
pub fn optimize(f: &dyn Fn(&Config) -> f64, start: &Config, method: &str, max_evals: u32, seed: u64) -> (Config, f64, u32) {
    let keys: Vec<String> = start.keys().cloned().collect();
    let to_cfg = |x: &[f64]| -> Config { keys.iter().cloned().zip(x.iter().copied()).collect() };
    let x0: Vec<f64> = keys.iter().map(|k| start[k]).collect();
    let mut b = opt::Budget::new(max_evals);
    let r = match method {
        "cma" | "cma_es" => opt::cma_es(&|xs: &[Vec<f64>]| xs.iter().map(|x| f(&to_cfg(x))).collect(), &x0, x0.iter().map(|v| v.abs()).fold(1.0, f64::max) * 0.1, seed, &mut b, 1e-9),
        "anneal" | "simulated_annealing" => opt::simulated_annealing(&|x: &[f64]| f(&to_cfg(x)), &x0, 0.05 * x0.iter().map(|v| v.abs()).fold(1.0, f64::max), 1.0, seed, &mut b),
        _ => opt::nelder_mead(&|x: &[f64]| f(&to_cfg(x)), &x0, 0.05 * x0.iter().map(|v| v.abs()).fold(1.0, f64::max), &mut b, 1e-9),
    };
    (to_cfg(&r.x), r.f, r.evaluations)
}

/// `study.tolerance_analysis`: Monte Carlo over normal tolerance distributions (±3σ = tol);
/// yield = fraction of samples passing `pass`.
pub fn tolerance_analysis(nominal: &Config, tolerances: &BTreeMap<String, f64>, samples: usize, seed: u64, pass: &dyn Fn(&Config) -> bool) -> Value {
    let mut rng = opt::Rng::new(seed);
    let mut ok = 0usize;
    for _ in 0..samples {
        let mut c = nominal.clone();
        for (k, t) in tolerances { if let Some(v) = c.get_mut(k) { *v += rng.normal() * t / 3.0; } }
        if pass(&c) { ok += 1; }
    }
    json!({ "samples": samples, "yield": ok as f64 / samples.max(1) as f64, "seed": seed, "evidence": "sampled" })
}

/// `study.observe`: fixed-layout observation tensor `[configs × (params + margins + score)]`
/// (f32) with a DLPack-style descriptor (spec 07; AG-07-004).
pub fn observe(cfgs: &[Config], ev: &[Evaluated]) -> (Vec<f32>, Value) {
    let keys: Vec<String> = cfgs.first().map(|c| c.keys().cloned().collect()).unwrap_or_default();
    let nm = ev.first().map(|e| e.checks.len()).unwrap_or(0);
    let width = keys.len() + nm + 1;
    let mut t = Vec::with_capacity(cfgs.len() * width);
    for (c, e) in cfgs.iter().zip(ev) {
        for k in &keys { t.push(c[k] as f32); }
        for ch in &e.checks { t.push(ch.margin.unwrap_or(if ch.verdict == Verdict::Pass { 1.0 } else { -1.0 }) as f32); }
        t.push(e.score as f32);
    }
    let desc = json!({ "dlpack": { "device_type": "kDLCPU", "dtype": { "code": "kDLFloat", "bits": 32, "lanes": 1 }, "shape": [cfgs.len(), width], "strides": [width, 1], "byte_offset": 0 }, "columns": { "params": keys, "margins": nm, "score": 1 } });
    (t, desc)
}

/// Why configurations are infeasible, grouped by cause (most frequent first): failed features
/// (by feature and status) and requirements by verdict, with an example value and remedy.
pub fn infeasible_causes(results: &[ConfigResult], ev: &[Evaluated]) -> Vec<Value> {
    let mut groups: BTreeMap<String, (usize, Value)> = BTreeMap::new();
    for (r, e) in results.iter().zip(ev) {
        if e.feasible { continue; }
        let mut causes: Vec<(String, Value)> = Vec::new();
        for o in r.update.outcomes.iter().filter(|o| !o.status.is_ok()) {
            causes.push((format!("feature {} {}", o.feature_ref, o.status.name()), json!({ "message": o.error.as_ref().map(|x| x.message.clone()) })));
        }
        if r.body.is_none() && causes.is_empty() { causes.push(("no solid body produced".into(), Value::Null)); }
        for c in e.checks.iter().filter(|c| c.verdict != Verdict::Pass) {
            let why = match (c.verdict, &c.escalation) {
                (Verdict::Unknown, Some(esc)) => format!("{} unknown: evidence '{}' < required '{}' ({esc})", c.requirement, c.evidence.name(), c.required_evidence.name()),
                (Verdict::Unknown, None) => format!("{} unknown: {}", c.requirement, c.message.clone().unwrap_or_default()),
                _ => format!("{} failed", c.requirement),
            };
            causes.push((why, json!({ "value": c.value, "margin": c.margin })));
        }
        for (k, example) in causes {
            let g = groups.entry(k).or_insert((0, Value::Null));
            g.0 += 1;
            if g.1.is_null() { g.1 = json!({ "configuration": e.configuration, "example": example }); }
        }
    }
    let mut v: Vec<(String, (usize, Value))> = groups.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(cause, (count, ex))| json!({ "cause": cause, "configurations": count, "first": ex })).collect()
}

/// Run a requirements-driven study end to end: configurations → update → evaluate → rank.
pub fn run(part: &PartState, registry: &Registry, kernel: &dyn ExactKernel, req: &Requirements, spec: &Value, objective: Option<(&str, &str, bool)>, k: usize) -> Result<Value, String> {
    let cfgs = configurations(spec)?;
    let results = update_configurations(part, registry, kernel, &cfgs);
    let ev = evaluate(part, registry, kernel, &results, objective, &req.hard)?;
    let (top, fallbacks) = rank_with_fallback(part, registry, kernel, &results, &ev, k);
    let fidelity = proxy_fidelity(kernel, &results, &top)?;
    Ok(json!({
        "configurations": cfgs.len(),
        "feasible": ev.iter().filter(|e| e.feasible).count(),
        "infeasible_causes": infeasible_causes(&results, &ev),
        "requirement_warnings": req.satisfiability(),
        "top": top.iter().map(|&i| json!({ "configuration": i, "params": cfgs[i], "score": ev[i].score })).collect::<Vec<_>>(),
        "fallbacks": fallbacks,
        "topology_classes": topology_report(&results),
        "proxy_fidelity": fidelity,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_expansion_and_rank_ties() {
        let c = configurations(&json!({ "grid": { "t": [6.0, 20.0, 3], "n": [4.0, 6.0] } })).unwrap();
        assert_eq!(c.len(), 6);
        let ev: Vec<Evaluated> = [3.0, 1.0, 1.0, f64::NAN, 2.0].iter().enumerate().map(|(i, &s)| Evaluated { configuration: i, score: s, checks: vec![], feasible: !s.is_nan() }).collect();
        assert_eq!(rank(&ev, 3), vec![1, 2, 4]);
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-12);
    }
}
