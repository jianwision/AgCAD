//! Pareto trade-offs for the LLM: instead of one "best" survivor, a small diverse set of
//! feasible designs that trade objectives off differently (non-dominated sort + crowding
//! distance), each with its engineering summary and a render, so the LLM compares and picks.

use agcad_dops::abi::ob;
use agcad_dops::Survivor;
use serde_json::{json, Value};

/// An objective over survivors: key, sign (+1 minimize, −1 maximize).
#[derive(Clone, Debug)]
pub struct Objective { pub key: String, pub sign: f64 }

pub const KEYS: &[&str] = &["volume", "surface_area", "extent", "undercut", "thin_wall", "centroid_z", "compliance", "ray_undercut"];

pub fn parse(v: &Value) -> Result<(Vec<Objective>, usize), String> {
    let list = v.get("objectives").and_then(Value::as_array).ok_or("pareto.objectives: [\"volume\", \"compliance\", …] (prefix \"-\" to maximize)")?;
    let mut out = Vec::new();
    for o in list {
        let s = o.as_str().ok_or("pareto objectives are strings")?;
        let (sign, key) = match s.strip_prefix('-') { Some(k) => (-1.0, k), None => (1.0, s) };
        if !KEYS.contains(&key) { return Err(format!("pareto objective '{key}': one of {}", KEYS.join(", "))); }
        out.push(Objective { key: key.into(), sign });
    }
    if out.len() < 2 { return Err("pareto needs ≥ 2 objectives (one objective is just the ranking)".into()); }
    let k = v.get("k").and_then(Value::as_u64).unwrap_or(4).clamp(2, 12) as usize;
    Ok((out, k))
}

/// Objective value of a survivor (compliance from the stiffness escalation, if computed).
pub fn value(sv: &Survivor, key: &str, compliance: Option<f64>) -> Option<f64> {
    let o = &sv.obs;
    Some(match key {
        "volume" => o[ob::VOL_EST],
        "surface_area" => o[ob::AREA],
        "extent" => o[5].max(o[6]).max(o[7]),
        "undercut" => o[ob::UNDERCUT_AREA],
        "thin_wall" => o[ob::THIN_AREA],
        "centroid_z" => o[ob::CENTROID + 2],
        "compliance" => compliance?,
        _ => return None,
    })
}

fn dominates(a: &[f64], b: &[f64]) -> bool { a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y) }

/// Indices of up to `k` points of the non-dominated front (minimization): only true
/// trade-offs; when the front is larger than `k`, the largest crowding distance first
/// (extremes always kept); ties by index. Deterministic.
pub fn select(points: &[Vec<f64>], k: usize) -> Vec<usize> {
    let mut left: Vec<usize> = (0..points.len()).collect();
    let mut out = Vec::new();
    while !left.is_empty() && out.len() < k {
        let front: Vec<usize> = left.iter().copied().filter(|&i| !left.iter().any(|&j| j != i && dominates(&points[j], &points[i]))).collect();
        left.retain(|i| !front.contains(i));
        if front.len() <= k { out.extend(front); break; }
        // crowding distance within the front
        let m = points[front[0]].len();
        let mut crowd = vec![0.0f64; front.len()];
        for d in 0..m {
            let mut ord: Vec<usize> = (0..front.len()).collect();
            ord.sort_by(|&a, &b| points[front[a]][d].partial_cmp(&points[front[b]][d]).unwrap().then(front[a].cmp(&front[b])));
            let (lo, hi) = (points[front[ord[0]]][d], points[front[*ord.last().unwrap()]][d]);
            crowd[ord[0]] = f64::INFINITY;
            crowd[*ord.last().unwrap()] = f64::INFINITY;
            if hi > lo { for w in 1..ord.len().saturating_sub(1) { crowd[ord[w]] += (points[front[ord[w + 1]]][d] - points[front[ord[w - 1]]][d]) / (hi - lo); } }
        }
        let mut ord: Vec<usize> = (0..front.len()).collect();
        ord.sort_by(|&a, &b| crowd[b].partial_cmp(&crowd[a]).unwrap().then(front[a].cmp(&front[b])));
        for &w in ord.iter().take(k) { out.push(front[w]); }
        break;
    }
    out
}

/// Survivors with the same objective vector (to 1e-6 relative) collapse to the first.
pub fn dedupe(points: &[Vec<f64>]) -> Vec<usize> {
    let mut keep: Vec<usize> = Vec::new();
    for i in 0..points.len() {
        let same = keep.iter().any(|&j| points[j].iter().zip(&points[i]).all(|(a, b)| (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)));
        if !same { keep.push(i); }
    }
    keep
}

pub fn summary_row(objs: &[Objective], vals: &[f64]) -> Value {
    let m: serde_json::Map<String, Value> = objs.iter().zip(vals).map(|(o, v)| (o.key.clone(), json!(v * o.sign))).collect();
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn front_and_crowding() {
        // (1,5) (2,3) (3,2) (5,1) are the front; (4,4) is dominated by (2,3)/(3,2)
        let p = vec![vec![1.0, 5.0], vec![4.0, 4.0], vec![2.0, 3.0], vec![3.0, 2.0], vec![5.0, 1.0]];
        let s = select(&p, 5);
        assert_eq!(s.len(), 4, "only the front");
        assert!(!s.contains(&1));
        let s2 = select(&p, 2);
        assert!(s2.contains(&0) && s2.contains(&4), "extremes kept: {s2:?}");
        assert_eq!(dedupe(&[vec![1.0, 2.0], vec![1.0, 2.0 + 1e-9], vec![2.0, 1.0]]), vec![0, 2]);
    }
}
