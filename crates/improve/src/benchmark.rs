//! Head-to-head thesis benchmark harness (`AG-10-002` / Phase 4 Pillar C4):
//!
//! Tracks AgCAD (`LLM + GPU Model + GPU CAD`) vs baseline (`Same LLM + CadQuery/OCCT on CPU`)
//! across the 4 canonical CAD workload families:
//! 1. `prismatic_machined` — bolted flange / bracket with 3-axis DFM + volume bounds
//! 2. `thin_walled_shelled` — open/multi-slab shelled enclosure with FEA stiffness + von Mises stress
//! 3. `tpms_lattice` — Gyroid / Schwarz-P / Diamond implicit lattice with NanoVDB + Dual Contouring
//! 4. `freeform_loft_enclosure` — Class-A multi-section lofted & blended enclosure (e.g. V5 front shell)

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkFamily {
    PrismaticMachined,
    ThinWalledShelled,
    TpmsLattice,
    FreeformLoftEnclosure,
}

impl BenchmarkFamily {
    pub const ALL: [BenchmarkFamily; 4] = [
        BenchmarkFamily::PrismaticMachined,
        BenchmarkFamily::ThinWalledShelled,
        BenchmarkFamily::TpmsLattice,
        BenchmarkFamily::FreeformLoftEnclosure,
    ];
    pub fn as_str(&self) -> &'static str {
        match self {
            BenchmarkFamily::PrismaticMachined => "prismatic_machined",
            BenchmarkFamily::ThinWalledShelled => "thin_walled_shelled",
            BenchmarkFamily::TpmsLattice => "tpms_lattice",
            BenchmarkFamily::FreeformLoftEnclosure => "freeform_loft_enclosure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub family: BenchmarkFamily,
    pub task_id: String,
    pub system: String,
    pub success: bool,
    pub wall_ms: f64,
    pub llm_tokens: u64,
    pub llm_turns: u32,
    pub step_valid: bool,
    pub host_geometry_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FamilySummary {
    pub tasks: usize,
    pub success_rate: f64,
    pub mean_wall_ms: f64,
    pub mean_llm_tokens: f64,
    pub step_valid_rate: f64,
    pub max_host_geometry_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkScorecard {
    pub agcad_system: String,
    pub baseline_system: String,
    pub agcad: BTreeMap<String, FamilySummary>,
    pub baseline: BTreeMap<String, FamilySummary>,
    pub agcad_wins_success: bool,
    pub agcad_faster: bool,
    pub agcad_fewer_tokens: bool,
    pub zero_host_geometry_bytes: bool,
}

fn summarize(runs: &[&BenchmarkRun]) -> FamilySummary {
    if runs.is_empty() { return FamilySummary::default(); }
    let n = runs.len() as f64;
    FamilySummary {
        tasks: runs.len(),
        success_rate: runs.iter().filter(|r| r.success).count() as f64 / n,
        mean_wall_ms: runs.iter().map(|r| r.wall_ms).sum::<f64>() / n,
        mean_llm_tokens: runs.iter().map(|r| r.llm_tokens as f64).sum::<f64>() / n,
        step_valid_rate: runs.iter().filter(|r| r.step_valid).count() as f64 / n,
        max_host_geometry_bytes: runs.iter().map(|r| r.host_geometry_bytes).max().unwrap_or(0),
    }
}

/// Build a head-to-head scorecard comparing `agcad_system` against `baseline_system`.
pub fn compare(runs: &[BenchmarkRun], agcad_system: &str, baseline_system: &str) -> BenchmarkScorecard {
    let mut agcad = BTreeMap::new();
    let mut baseline = BTreeMap::new();
    for fam in BenchmarkFamily::ALL {
        let a: Vec<&BenchmarkRun> = runs.iter().filter(|r| r.family == fam && r.system == agcad_system).collect();
        let b: Vec<&BenchmarkRun> = runs.iter().filter(|r| r.family == fam && r.system == baseline_system).collect();
        if !a.is_empty() { agcad.insert(fam.as_str().into(), summarize(&a)); }
        if !b.is_empty() { baseline.insert(fam.as_str().into(), summarize(&b)); }
    }
    let all_a: Vec<&BenchmarkRun> = runs.iter().filter(|r| r.system == agcad_system).collect();
    let all_b: Vec<&BenchmarkRun> = runs.iter().filter(|r| r.system == baseline_system).collect();
    let sa = summarize(&all_a);
    let sb = summarize(&all_b);
    BenchmarkScorecard {
        agcad_system: agcad_system.into(),
        baseline_system: baseline_system.into(),
        agcad,
        baseline,
        agcad_wins_success: sa.success_rate >= sb.success_rate,
        agcad_faster: sa.mean_wall_ms <= sb.mean_wall_ms,
        agcad_fewer_tokens: sa.mean_llm_tokens <= sb.mean_llm_tokens,
        zero_host_geometry_bytes: sa.max_host_geometry_bytes == 0,
    }
}

/// Render a Markdown comparison table for CI reports and publications.
pub fn to_markdown(sc: &BenchmarkScorecard) -> String {
    let mut s = format!(
        "### Head-to-Head CAD Benchmark (`{}` vs `{}`)\n\n\
         | Family | AgCAD Success | Baseline Success | AgCAD ms | Baseline ms | AgCAD Tokens | Baseline Tokens | Host Geom B |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|\n",
        sc.agcad_system, sc.baseline_system
    );
    for fam in BenchmarkFamily::ALL {
        let k = fam.as_str();
        let a = sc.agcad.get(k).cloned().unwrap_or_default();
        let b = sc.baseline.get(k).cloned().unwrap_or_default();
        s.push_str(&format!(
            "| `{k}` | {:.1}% | {:.1}% | {:.1} | {:.1} | {:.0} | {:.0} | {} |\n",
            a.success_rate * 100.0,
            b.success_rate * 100.0,
            a.mean_wall_ms,
            b.mean_wall_ms,
            a.mean_llm_tokens,
            b.mean_llm_tokens,
            a.max_host_geometry_bytes,
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scorecard_compares_agcad_against_baseline() {
        let runs = vec![
            BenchmarkRun { family: BenchmarkFamily::PrismaticMachined, task_id: "flange".into(), system: "agcad".into(), success: true, wall_ms: 120.0, llm_tokens: 1800, llm_turns: 2, step_valid: true, host_geometry_bytes: 0 },
            BenchmarkRun { family: BenchmarkFamily::PrismaticMachined, task_id: "flange".into(), system: "cadquery".into(), success: true, wall_ms: 950.0, llm_tokens: 4200, llm_turns: 5, step_valid: true, host_geometry_bytes: 0 },
            BenchmarkRun { family: BenchmarkFamily::TpmsLattice, task_id: "gyroid".into(), system: "agcad".into(), success: true, wall_ms: 340.0, llm_tokens: 2100, llm_turns: 2, step_valid: false, host_geometry_bytes: 0 },
            BenchmarkRun { family: BenchmarkFamily::TpmsLattice, task_id: "gyroid".into(), system: "cadquery".into(), success: false, wall_ms: 5000.0, llm_tokens: 8000, llm_turns: 8, step_valid: false, host_geometry_bytes: 0 },
        ];
        let sc = compare(&runs, "agcad", "cadquery");
        assert!(sc.agcad_wins_success && sc.agcad_faster && sc.agcad_fewer_tokens && sc.zero_host_geometry_bytes);
        let md = to_markdown(&sc);
        assert!(md.contains("prismatic_machined") && md.contains("tpms_lattice"));
    }
}
