# 11 — Recursive Self-Improvement

> Status: **Normative** · Spec v0.3.1 · Area: improve

AgCAD is built so that **models improve the system they use**:
- the feature library
- converters
- GPU kernels
- the GPU harness's check implementations
- search strategies
- the models themselves

All of it happens under evaluation the improvers cannot influence.

The design builds on the Demo's research and audit (`SELF_IMPROVEMENT_PLAN.md`, `SELF_IMPROVEMENT_REVIEW_V2.md`, `findings.md`):
- expert iteration (search, then distill)
- DAgger-style relabeling of learner-visited states
- evaluator-guided code evolution (AlphaEvolve-style)
- MAP-Elites diversity archives
- counterexample-guided refinement
- reusable-holdout discipline

The Demo also left two diagnostic lessons that this design keeps:
- A failed optimization is **not** proof of infeasibility, so representation limits must be diagnosed (objective floors, reachability) before any training budget is spent.
- An evaluator must never be co-optimized with the candidates it judges.

## Improvement levels

| Level | What improves | Proposer | Unit of change | Gate |
|---|---|---|---|---|
| **L0 — search** | One design within a session | Model (LLM/CuModel) using `study.*` | Configurations, versions | Requirements ([06](06-harness.md)) |
| **L1 — session skills** | Reusable UDFs, rules, networks within a session | Model | Session UDF (`udf.define`) | Sandbox + per-use checks |
| **L2 — library growth** | Feature catalog: UDFs, converters, knowledge templates, procedural assets | Model | `udf.publish` to `team` / `global` scope | Conformance + transfer on held-out tasks |
| **L3 — models** | CuModel weights; LLM skill context, exemplars, optional fine-tunes | Training pipeline driven by traces | Model checkpoint / context pack | Held-out benchmark, cross-adaptor replay |
| **L4 — system code** | GPU kernels, stdlib feature implementations, converters, **GPU harness check implementations**, scheduler heuristics | Model (code-writing LLM) | Implementation version behind a fixed interface | Equivalence against frozen reference + performance + full regression |
| **L5 — meta** | Curricula, task generators, archive policies, budget allocation, search policies used by L2–L4 | Model | Improvement-loop configuration | Human approval + meta-held-out |

**Frozen at every level** (changed only by humans, as new versioned releases):
- requirement semantics
- reference checks (the host harness oracles)
- benchmark definitions and held-out sets
- scoring formulas
- the promotion gate itself

## The loop

```
          ┌──────────────────────────────────────────────────────────┐
          │                      Archive                             │
          │  niches = task family × body type × cost band × method   │
          │  elites, lineage, counterexamples (minimized), traces    │
          └──────────────┬───────────────────────────────▲───────────┘
             sample parents/contexts                    insert (if gate passes)
                         ▼                               │
   PROPOSE ─▶ SANDBOX BUILD ─▶ CONFORMANCE ─▶ DEV EVAL ─▶ GATE ─▶ HELD-OUT ─▶ PROMOTE ─▶ MONITOR
   (model)    (compile, verify) (per-op suite,  (GPU harness, (stats,  (isolated,   (versioned, (canary,
                                 equivalence)    thousands of  no-regress) rate-limited) reversible) rollback)
                                                 configs/s)
```

1. **Propose.** Any `ModelAdaptor` produces a candidate: a UDF, a kernel or check implementation, a converter, a CuModel checkpoint or a curriculum change. It may consult the archive, traces and `udf.view_source` of anything.
2. **Sandbox build.** Compile and verify (purity, bounds, memory). Failures return diagnostics to the proposer as data.
3. **Conformance** ([09](09-validation.md)): golden cases, expected rejections, backend parity, and for L4 **differential equivalence** against the frozen reference implementation on randomized, adversarial and metamorphic inputs. This includes the archive's accumulated counterexamples.
4. **Dev evaluation.** Benchmark task families run on the **GPU-resident harness**. Thousands of configurations per second make dense, cheap evaluation possible, and this is the reason the harness lives on the GPU. Full cost is reported: GPU-ms, host-ms (the control-plane budget), exact rebuilds, LLM tokens, round trips.
5. **Gate:**
   - **Non-regression** on the full regression suite. Every previously passing task must still pass, within noise bounds pre-registered per metric.
   - **Improvement** on the targeted metric with a pre-declared effect size and confidence.
   - **Cost**: no hidden cost shift, e.g. a faster kernel that needs more exact rebuilds.
6. **Held-out.** Evaluate on task families isolated by ancestry from everything used to propose or select. Queries to the held-out set are rate-limited and counted (reusable-holdout discipline), and the sets rotate per release.
7. **Promote.** The change is content-addressed and versioned, with lineage recorded. Promotion is reversible, and the previous version stays loadable.
8. **Monitor.** Canary in live sessions. An automatic rollback is triggered by regression signals: increased `TOPOLOGY_FAILED` rates, conformance drift, cost blowups.

## Level details

### L1 — session skills
- A model writes UDFs during a task (`udf.define`) and reuses them within the session.
- Successful UDFs and their usage traces become L2 proposals automatically. They still go through the gate.

### L2 — library growth
- **Skill manifest:**
  - applicability conditions (input types, ranges)
  - parameters
  - naming roles and attribute rules
  - postconditions and evidence
  - cost profile
  - the trace of tasks where it helped
- **Transfer test:** the skill must help on held-out task families it was not extracted from, not just reproduce its origin task.
- **Converters** (`convert.*` implementations) compete in `convert.body_type` dispatch by measured deviation and cost per input class.
- **Diversity:** the archive keeps elites per niche (MAP-Elites), so one dominant style doesn't erase alternatives.

### L3 — models
- **CuModel:**
  1. *Expert iteration:* GPU-harness search (studies, optimize, LLM proposals) finds better solutions.
  2. The CuModel is distilled on them.
  3. *DAgger:* the expert relabels states the learner actually visits.

  Training data comes from traces. Training uses **Candle** (`agcad-model`) on CUDA, with rollouts generated on device. The Demo's CPU-train / GPU-infer split remains as a fallback. Checkpoints are safetensors with a hash recorded in the promotion lineage.
- **LLM:**
  - Improvement through *context*: retrieved exemplars, skill-library entries, failure summaries.
  - Optional fine-tuning on trace-derived pairs (requirements → successful programs, error → repair).
  - The model is evaluated with and without the context pack, so the contribution of each is known.
- **Attribution:** a 2×2 cross of {old, new model} × {old, new library/harness implementation} where interfaces are compatible. Otherwise it is reported as N/A.
- **Reachability diagnostics come first** (Demo lesson): before spending training budget on a task family, compute objective floors and representation reachability. If the representation can't reach the target, the right improvement is a new UDF or representation (L2), not more training.

### L4 — system code
- **Targets:**
  - GPU kernels (tessellation, SSI seeding, implicit eval, solvers)
  - stdlib feature implementations
  - converters
  - **GPU harness check implementations**
  - scheduler heuristics
- **Isolation:** candidate kernels compile and run only in a sandbox worker process with its own GPU context. CUDA faults are sticky and would otherwise kill other sessions. Candidates must pass compute-sanitizer (memcheck, racecheck, initcheck) before dev evaluation ([12](12-cuda-review.md) F3).
- **Fixed interface:** a candidate replaces an implementation behind an unchanged manifest and semantics. Changing semantics is a new feature (L2), not an L4 replacement.
- **Equivalence:** for checks, the GPU implementation must match the frozen host reference **bit-for-bit on verdicts** over the conformance corpus, and within declared tolerance on residuals. A check implementation that is faster but rejects or accepts differently is rejected outright. Improving *what* a check means is a human-owned change to requirement semantics.
- **Performance** is measured on every backend. Promotion is per backend, so a CUDA kernel can be promoted while the wgpu kernel stays.

### L5 — meta
- **Proposals:**
  - curriculum (which task families to emphasize)
  - task generators (parametric families, perturbations, counterexample-derived tasks)
  - archive niche definitions
  - budget split across levels
  - search policies
- **Evaluation** is the rate of verified improvement per unit of compute across a meta-held-out window.
- **Approval:** a human must approve. Generated tasks may enter the *training* pool but never the held-out pool, which is drawn only from the frozen human-curated set.

## Start early, where checks are automatic

- **L1/L2 begin at M2** on **reconstruction tasks** ([09](09-validation.md#reconstruction-tasks-self-verifying-from-m2)). These are self-verifying ("rebuild within δ mm"), so they give dense rewards without human labels.
- **L4 begins at M3** with a concrete, high-value target: GPU exact-kernel ops whose OCCT parity rate is below 100 % are candidates for model-proposed fixes. They are gated by parity on the permanent counterexample corpus.

## Anti-gaming

Improvers optimize whatever is measured, so the gate makes the following hard to game:
- **Full cost vector:** GPU-ms, host-ms, exact rebuilds, LLM tokens, evidence levels achieved. A gain that shifts cost to an unmeasured dimension fails.
- **Evidence levels are pinned per benchmark.** A candidate cannot pass by settling for `sampled` where the benchmark demands `bounded`.
- **Proxy-fidelity and survivor-failure metrics are part of the gate.** A faster proxy that ranks worse fails.
- **Red-team tasks:** after each promotion, adversarial tasks targeting the promoted change (its applicability edges, degenerate inputs, naming perturbations) are generated into the *training* pool, and the change is re-checked against them.

## Invariants

1. **No evaluator capture.**
   - Candidates never see held-out tasks, held-out witnesses or reference-check internals beyond their published interfaces.
   - No candidate can modify the frozen set: oracles, benchmark definitions, scoring, gate.
2. **Counterexamples are permanent.** Every failure found anywhere (conformance, canary, user report) is minimized and added to the regression corpus. A checker bug found under the same specification triggers re-evaluation of affected promotions.
3. **Everything is replayable.** Content hashes for code, weights and data, plus full traces, make any promotion reproducible on the same device.
4. **Legal exploration is protected.** Harness improvements reduce false rejections and missed defects. They never narrow legal exploration, loosen requirements or change targets to inflate scores.
5. **Budgets are explicit** at every level, and improvement compute is reported next to the gains.
6. **Humans hold the keys** to requirement semantics, oracles, held-out sets, the gate and L5 approval.

## Data model

```
ImprovementCandidate { id, level, target, parent_ids, proposer, artifact_hash, manifest, claims }
Evaluation           { candidate_id, suite_version, backend, metrics, costs, verdict, evidence }
Promotion            { candidate_id, scope, from_version, to_version, gate_report, approver? }
ArchiveEntry         { niche, candidate_id, score_vector, lineage, counterexamples[] }
```

These live in `agcad-improve`, exposed as a headless service and CLI (`agcad improve propose|evaluate|promote|rollback|archive`).

## Normative requirements

| ID | Requirement |
|---|---|
| AG-11-001 | Candidates MUST NOT modify frozen artifacts or read them beyond their published interfaces. |
| AG-11-002 | Promotions MUST be versioned, content-addressed and reversible. |
| AG-11-003 | Counterexamples MUST be kept permanently in the regression corpus. |
| AG-11-004 | L5 changes MUST require human approval. |
| AG-11-005 | Gates MUST score the full cost vector with pinned evidence levels. |
