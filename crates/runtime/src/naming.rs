//! Stable naming (spec 02 § Stable naming): provenance names, resolution per revision with
//! lineage, and geometric-signature recovery with a confidence threshold. The harness never
//! auto-picks below threshold (AG-02-003).

use agcad_geo::math::V3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableName {
    /// `family.op#id` or `udf#id`.
    pub feature_ref: String,
    pub role: String,
    pub qualifiers: Vec<(String, String)>,
    pub index: Option<u32>,
}

impl StableName {
    /// Parse `part.pad#3/side[profile=sketch.line#7].2`.
    pub fn parse(s: &str) -> Option<StableName> {
        let (fr, rest) = s.split_once('/')?;
        // The index is the digits after the last '.' that follows the qualifier bracket
        // (qualifiers may themselves contain dots, e.g. `sketch.line#7`).
        let tail_start = rest.rfind(']').map(|i| i + 1).unwrap_or(0);
        let (body, index) = match rest[tail_start..].rfind('.') {
            Some(k) => {
                let k = tail_start + k;
                let idx = &rest[k + 1..];
                if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) { (&rest[..k], idx.parse().ok()) } else { (rest, None) }
            }
            None => (rest, None),
        };
        let (role, quals) = match body.split_once('[') {
            Some((r, q)) => (r, q.trim_end_matches(']')),
            None => (body, ""),
        };
        let qualifiers = quals.split(',').filter(|q| !q.is_empty()).filter_map(|q| q.split_once('=').map(|(a, b)| (a.to_string(), b.to_string()))).collect();
        Some(StableName { feature_ref: fr.to_string(), role: role.to_string(), qualifiers, index })
    }

    pub fn format(&self) -> String {
        let mut s = format!("{}/{}", self.feature_ref, self.role);
        if !self.qualifiers.is_empty() {
            s.push('[');
            s.push_str(&self.qualifiers.iter().map(|(a, b)| format!("{a}={b}")).collect::<Vec<_>>().join(","));
            s.push(']');
        }
        if let Some(i) = self.index { s.push_str(&format!(".{i}")); }
        s
    }
}

/// Geometric signature for recovery: surface kind, centroid, normal/axis, area, adjacency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Signature {
    pub kind: String,
    pub centroid: V3,
    pub normal: V3,
    pub area: f64,
    pub adjacency: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Resolution {
    Unique(u32),
    /// Name no longer exists; lineage explains why (e.g. "split by part.pocket#7").
    Stale { lineage: Vec<String> },
    /// Several candidates; confidence per candidate (model must confirm a remap).
    Ambiguous(Vec<(u32, f64)>),
}

/// Exact resolution among the current element names.
pub fn resolve(name: &str, current: &[String], lineage: &[(String, String)]) -> Resolution {
    let hits: Vec<u32> = current.iter().enumerate().filter(|(_, n)| *n == name).map(|(i, _)| i as u32).collect();
    match hits.len() {
        1 => Resolution::Unique(hits[0]),
        0 => {
            // Split children `name.k` count as ambiguity, otherwise stale with lineage.
            let children: Vec<(u32, f64)> = current.iter().enumerate().filter(|(_, n)| n.starts_with(&format!("{name}."))).map(|(i, _)| (i as u32, 1.0)).collect();
            if !children.is_empty() { return Resolution::Ambiguous(children); }
            Resolution::Stale { lineage: lineage.iter().filter(|(from, _)| from == name).map(|(_, why)| why.clone()).collect() }
        }
        _ => Resolution::Ambiguous(hits.into_iter().map(|i| (i, 1.0)).collect()),
    }
}

/// Confidence that `b` is the same element as `a` (0..1).
pub fn similarity(a: &Signature, b: &Signature, scale: f64) -> f64 {
    if a.kind != b.kind { return 0.0; }
    let dc = (a.centroid - b.centroid).norm() / scale.max(1e-9);
    let dn = 1.0 - a.normal.dot(b.normal).clamp(-1.0, 1.0);
    let da = (a.area - b.area).abs() / a.area.max(b.area).max(1e-12);
    let adj = if a.adjacency == b.adjacency { 0.0 } else { 0.2 };
    (1.0 - (0.5 * dc + 2.0 * dn + 0.5 * da + adj)).clamp(0.0, 1.0)
}

/// Signature recovery: unique only if the best candidate clears `min_confidence` and beats
/// the runner-up by a margin; else ranked ambiguity for the model to confirm.
pub fn recover(target: &Signature, candidates: &[Signature], scale: f64, min_confidence: f64) -> Resolution {
    let mut scored: Vec<(u32, f64)> = candidates.iter().enumerate().map(|(i, c)| (i as u32, similarity(target, c, scale))).filter(|x| x.1 > 0.0).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    match scored.as_slice() {
        [] => Resolution::Stale { lineage: vec!["no geometric match".into()] },
        [best, rest @ ..] if best.1 >= min_confidence && rest.first().map(|r| best.1 - r.1 > 0.1).unwrap_or(true) => Resolution::Unique(best.0),
        _ => Resolution::Ambiguous(scored.into_iter().take(5).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_format_roundtrip() {
        for s in ["part.pad#3/cap_end", "part.pad#3/side[profile=sketch.line#7]", "flange_udf#4/hole.3", "part.pad#3/side.12"] {
            let n = StableName::parse(s).unwrap();
            assert_eq!(n.format(), s, "{n:?}");
        }
        assert_eq!(StableName::parse("flange_udf#4/hole.3").unwrap().index, Some(3));
    }
}
