//! Process-isolated L2 (AgLang UDF) and L4 (CUDA kernel) self-improvement sandbox
//! (Phase 4 / Spec 11 & 17: `AG-11-002`, `AG-11-003`, `AG-17-001`, `AG-17-006`).
//!
//! Executes candidate evaluations in a dedicated subprocess (`AGCAD_SANDBOX=1`), optionally
//! wrapped by NVIDIA `compute-sanitizer` (for L4 kernel candidates), checks the promotion gate,
//! and records either a reversible versioned promotion or a permanent counterexample.

use crate::{gate, Archive, Candidate, Evaluation, GatePolicy, GateReport, Level};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Wrap L4 kernel candidates under `compute-sanitizer --tool memcheck`.
    pub use_compute_sanitizer: bool,
    /// Fuzz iteration budget inside the isolated worker process.
    pub fuzz_cases: usize,
    /// Timeout in seconds (advisory metadata for runner orchestration).
    pub timeout_secs: u64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        SandboxConfig { use_compute_sanitizer: false, fuzz_cases: 1_000, timeout_secs: 120 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxOutcome {
    pub exit_code: i32,
    pub sanitizer_clean: bool,
    pub stdout_json: Option<Value>,
    pub stderr: String,
}

/// Launch `exe` in a sandboxed worker process (`AGCAD_SANDBOX=1`). When `cfg.use_compute_sanitizer`
/// is true and `level == Level::L4`, prefixes the command with `compute-sanitizer --error-exitcode 99`.
pub fn run_sandbox_worker(exe: &Path, level: Level, cfg: &SandboxConfig) -> Result<SandboxOutcome, String> {
    let mut cmd = if cfg.use_compute_sanitizer && level == Level::L4 {
        let mut c = Command::new("compute-sanitizer");
        c.args(["--tool", "memcheck", "--error-exitcode", "99", "--"]);
        c.arg(exe);
        c
    } else {
        Command::new(exe)
    };
    cmd.args(["conformance", "--fuzz", &cfg.fuzz_cases.to_string()])
        .env("AGCAD_SANDBOX", "1")
        .env("AGCAD_SANDBOX_TIMEOUT_SECS", cfg.timeout_secs.to_string());
    let out = cmd.output().map_err(|e| format!("sandbox launch failed: {e}"))?;
    let code = out.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let stdout_json = serde_json::from_slice::<Value>(&out.stdout).ok();
    let sanitizer_clean = code != 99 && !stderr.contains("ERROR SUMMARY:") || stderr.contains("ERROR SUMMARY: 0 errors");
    Ok(SandboxOutcome { exit_code: code, sanitizer_clean, stdout_json, stderr })
}

/// Evaluate a proposed L2/L4 candidate against `incumbent` and automatically either promote it
/// (when `gate` passes) or archive its failure report as a permanent counterexample (`AG-11-003`).
pub fn evaluate_and_gate(
    archive: &Archive,
    incumbent: &Evaluation,
    candidate: &Candidate,
    cand_eval: &Evaluation,
    policy: &GatePolicy,
    approver: Option<&str>,
) -> Result<GateReport, String> {
    archive.record_evaluation(cand_eval).map_err(|e| e.to_string())?;
    let report = gate(incumbent, cand_eval, policy, candidate.level, approver.is_some());
    if report.pass {
        archive.promote(candidate, &report, approver)?;
    } else {
        let ce = json!({
            "candidate_id": candidate.id,
            "level": candidate.level,
            "target": candidate.target,
            "reasons": report.reasons,
            "improvement": report.improvement,
            "conformance_failed": cand_eval.conformance_failed,
        });
        let _ = archive.add_counterexample(&candidate.id, &ce);
    }
    Ok(report)
}
