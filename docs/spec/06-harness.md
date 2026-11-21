# 06 — Harness

> Status: **Normative** · Spec v0.3.1 · Area: harness

The harness is the part of AgCAD that makes autonomous modeling **trustworthy without constraining strategy**. It enforces:
- **Requirements:** what must be delivered and how it is checked.
- **Budget and sandbox:** what execution may consume.
- **Transactions:** how changes become versions.

It reports evidence. It never optimizes, repairs, retries or substitutes on the model's behalf. These rules come from Demo `SELF_IMPROVEMENT_REVIEW_V2.md` and were validated there.

The harness runs **on the GPU**. Every check runs there: per-configuration, per-step, exact-phase and delivery checks alike. The host side is only the control plane: it uploads requirements, keeps the publication registry and persists trace bytes. Every GPU check has a `CpuDevice` reference implementation used **only in conformance**, where verdicts must be bit-identical ([09](09-validation.md)).

Constraints, evaluations and requirements share one model; see [14](14-constraints-evaluations.md).

## Requirements (immutable during a session)

```yaml
requirements:
  id: flange-req, rev: 3
  units: mm
  deliver: { body_type: SolidBody, count: 1, formats: [step_ap242] }
  hard:                                     # REQUIREMENT_FAILED if violated at save/finish
    - { id: R1, measure: evaluate.bounding_box, field: max_extent, le: 160 }
    - { id: R2, measure: evaluate.thickness_analysis, field: min, ge: 3, evidence: bounded }
    - { id: R3, measure: evaluate.check, field: valid, eq: true, evidence: exact }
    - { id: R4, measure: evaluate.clash, with: "@file:bolt_envelope.step", field: volume, eq: 0 }
  objectives:                               # feedback, never rejection (unless declared hard)
    - { id: O1, minimize: evaluate.measure_inertia.mass }
  tolerances: { linear: 1e-6, angular: 1e-9, conversion_max_deviation: 0.01 }
  budget: { ops: 200000, gpu_ms: 60000, host_ms: 3000, exact_rebuilds: 64, llm_calls: 40 }
  process: machining_3ax                    # selects delivery representation + DFM evaluation family (doc 14)
  phases: { edit: [numeric], update: [R1], save: [R1, R2, R3], finish: [R1, R2, R3, R4] }
```

- Callers supply requirements, and models cannot modify them. A response that tries returns `STALE_REQUIREMENTS`.
- A new user requirement means a new `rev`.
- Every check names a required **evidence level**. A `sampled` result cannot satisfy a requirement that demands `bounded`; the harness reports `unknown` together with the escalation feature that would settle it.

## Transactions

| State | Meaning |
|---|---|
| **Workspace** | Private, mutable, may be temporarily invalid (open surfaces, failing sketches, half-built bodies) |
| **Version** | Immutable. Created only by `document.save_version` when the save-phase checks pass |
| **Branch** | A line of versions. `document.branch` creates N branches sharing history copy-on-write |
| **Configuration** | A design-table row. Each configuration has its own workspace slot on the GPU |

- A failed `save_version` **keeps the workspace** so the model can repair it.
- Legal but worse geometry may be saved, unless the requirements demand monotone improvement.
- Budget is shared across branches and configurations, and it is never refunded by `document.delete` (Demo rule).
- `document.finish` runs finish-phase checks and produces a `DeliveryReport`. Finishing does not imply success: *stop*, *execution success*, *requirements met* and *delivery verified* are four separate fields.

## Design Checker

A **check** is a versioned function `(elements, requirement) → (verdict, residual, evidence, cost)`.

| Source | Can it reject? | Notes |
|---|---|---|
| Requirement checks (harness-owned) | Yes, at declared phases | Frozen implementation per harness release; improvements only via gated promotion ([11](11-self-improvement.md)) |
| Structural rules (harness-owned) | Yes | Finite values, capacity, sandbox, revision, units, naming ambiguity |
| `knowledge.check` (user- or model-authored) | Only if a requirement references it | Adds evidence and reports, and can never weaken a harness check |

Each check declares:
- the phase it runs in (`edit`, `update`, `save`, `finish`)
- whether it runs on the GPU or the host
- its evidence level
- its cost

**Escalation chain example (thickness):**
1. GPU implicit-brick estimate (`sampled`)
2. Lipschitz-certified bracket (`bounded`)
3. exact B-rep distance on the host (`exact`)

The harness runs only as far as the requirement demands.

## GPU-resident harness

This is the inner-loop harness. It is the generalization of the Demo's device harness (`cuda/gpu_harness.cu`), which already ran propose → construct trial → independent check → decide → commit/rollback → record for 32 configurations × 8 rounds on the device, with replay bit-comparison and a single readback.

Each round runs as a **wavefront** of uniform kernels, not a per-configuration interpreter. This was revised after the [CUDA review](12-cuda-review.md) (F2), because heterogeneous opcodes in one kernel diverge badly on SIMT hardware.

```
rollout segment = one graph launch:
WHILE (any configuration active && segment budget left)        ← conditional WHILE node (CUDA 12.3+)
  1. VALIDATE   one thread per command record: opcode allow-list, revisions, budget,
                arg types, finite immediates; invalid → status record, no execution
  2. BIN        bin valid records by opcode (CUB histogram + scan / segmented sort)
                → per-opcode work queues + per-bin counts
  3. EXECUTE    per opcode bin: SWITCH/IF node skips empty bins (CUDA 12.8+); each runs the
                op's specialized kernel over its queue, writing the configurations'
                trial workspace slots (never the current slot); VM kernel only for
                model-invented expression graphs, evaluated per sample
  4. CHECK      edit/update-phase checks compiled for GPU (numeric, capacity, bbox,
                implicit thickness/clash estimates, sketch DOF, constraint residuals)
  5. DECIDE     ACCEPTED → swap trial/current; else keep current, record code + residuals
  6. METER      budget (ops, device-time quanta from %globaltimer); BUDGET_EXHAUSTED masks the configuration
  7. RECORD     16-byte status + trace row into the device trace ring
  8. OBSERVE    study.observe packs the next observation; CuModel forward writes the next commands
  9. RANK       stable top-k by score / requirement margin (deterministic CUB sort)
 10. ESCALATE   configurations needing exact work/checks → exact wavefront bins (GPU, low-priority stream)
               sets the WHILE condition for the next iteration
between segments (host launches only): patch the parameter block, relaunch; no geometry crosses
```

- On wgpu, the WHILE and SWITCH nodes become host-side loops over prerecorded command buffers. The semantics are the same; only the launch overhead differs.
- Exact work (booleans, fillets, SSI, validity) runs **on the GPU** in later exact wavefronts. Configurations show `PENDING_EXACT` until their exact wavefront completes, and models may keep working on others. Device kernels never spin on host memory.
- Device-side graph launch is not relied on, because its support with conditional nodes is unconfirmed.

Properties:
- **Zero host round trips** inside a rollout segment. Exact verdicts arrive at segment boundaries.
- **Isolation:** a configuration can only write its own trial slot. Shared feature definitions and requirements are read-only device buffers. Untrusted or first-run model kernels execute in a separate sandbox worker process with its own CUDA context, so a fault (a sticky CUDA error) cannot destroy other sessions (review F3).
- **Determinism:** the same inputs give bit-identical traces, and conformance runs everything twice and compares.
- **Reference semantics:** each GPU check has a frozen `CpuDevice` reference implementation, used only in conformance, which compares them across randomized and adversarial inputs.
- **Throughput:** check + decide + rank for 4,096 configurations in ≤ 5 ms (M4 target, [05](05-execution.md#throughput-targets-m4-exit-measured-not-assumed)).
- **Self-improvable:** a faster GPU check implementation can replace the current one only by passing equivalence against the frozen reference ([11](11-self-improvement.md#l4--system-code)).

## Update report

The same report reaches both caller paths. LLMs receive it as JSON; device models receive it as an observation tensor.

```
UpdateReport
  status, revision, requirements_rev, workspace_state
  per configuration (or summarized): verdicts { check_id: pass|fail|unknown, residual, evidence }
  measurements, deltas vs. previous version, objective values
  created/modified/deleted element names (with lineage)
  sketch/assembly DOF, solver status
  attributes summary (procedural)
  budget { used, remaining } per resource
  unsupported / failed features with codes and the elements involved
  optional: offscreen renders (headless), sensitivity tensors
  trace reference
```

## Error codes

`AgStatus` is a `u16` with **fixed, append-only values**, following the `cudaError_t` / `cublasStatus_t` conventions:
- `0x0000` means success.
- The high byte is the category.
- Values are never renumbered.
- `ag_status_name()` and `ag_status_string()` exist on host and device.

Command outcomes are status *data* in status records. Host API failures use the driver's `EnqueueError` / `DeviceFault` ([08](08-hal-backends.md#device-api-core--extension-traits)).

| Value | Category | Code | Meaning |
|---|---|---|---|
| `0x0000` | success / info | `ACCEPTED` | Applied |
| `0x0001` | success / info | `CACHE_HIT` | Reused an already-checked result |
| `0x0002` | success / info | `STOPPED` | Configuration finished |
| `0x0003` | success / info | `INACTIVE` | Configuration masked |
| `0x0004` | success / info | `PENDING_EXACT` | Exact-phase result not yet available |
| `0x0005` | success / info | `CHUNKED` | Info: the capacity planner split the study into chunks ([05](05-execution.md#memory-capacity-planning)); reported, not an error |
| `0x0100` | lifecycle | `NEED_RECOMPILE` | Feature or kernel version changed; recompile |
| `0x0101` | lifecycle | `INVALID_BASELINE` | Starting state invalid for this operation |
| `0x0102` | lifecycle | `MERGE_CONFLICT` | Branch merge touched the same feature parameters or topology |
| `0x0103` | lifecycle | `MIGRATION_FAILED` | A saved document could not be migrated to the current schema/op versions (details per feature) |
| `0x0200` | revision | `STALE_REVISION` | `expected_revision` is not current |
| `0x0201` | revision | `STALE_REQUIREMENTS` | Requirements revision mismatch, or an attempt to change requirements |
| `0x0300` | parameter | `INVALID_OPCODE` | Opcode not in the allow-list or unknown |
| `0x0301` | parameter | `INVALID_PARAMETER` | Out of range, wrong type or wrong unit |
| `0x0302` | parameter | `NONFINITE_PARAMETER` | NaN or Inf |
| `0x0400` | check | `REQUIREMENT_FAILED` | A hard requirement was violated at this phase |
| `0x0401` | check | `GEOMETRY_CHECK_FAILED` | A geometric check failed (self-intersection, fold, invalid surface) |
| `0x0402` | check | `TOPOLOGY_FAILED` | Exact topology construction failed (boolean, fillet, sew, split) |
| `0x0403` | check | `NUMERICAL_CHECK_FAILED` | A numerical-consistency check failed |
| `0x0404` | check | `NO_IMPROVEMENT` | Only when requirements demand monotone improvement |
| `0x0405` | check | `LIPSCHITZ_VIOLATION` | An implicit body's declared bound failed spot checks; evidence downgraded |
| `0x0500` | reference | `STALE_REFERENCE` | A name or handle no longer resolves (lineage attached) |
| `0x0501` | reference | `AMBIGUOUS_REFERENCE` | A name or query matched several elements where one was required |
| `0x0600` | resource | `BUDGET_EXHAUSTED` | A resource budget is exhausted |
| `0x0601` | resource | `CAPACITY_EXCEEDED` | Declared capacity overflow (never silently truncated) |
| `0x0602` | resource | `UNSUPPORTED_OP` | Feature or conversion not available for these inputs or on this backend |
| `0x0700` | code | `COMPILE_FAILED` | AgLang/AgIR did not compile (diagnostics attached) |
| `0x0701` | code | `SANDBOX_VIOLATION` | Code violated purity, bounds or memory rules |
| `0x0702` | code | `CONFORMANCE_FAILED` | UDF publication or implementation promotion failed conformance |
| `0x0800` | solver | `SOLVER_DIVERGED` | Sketch, assembly or optimizer did not converge within budget |
| `0x0801` | solver | `UNDER_CONSTRAINED` | Solver: remaining DOF where full definition was required |
| `0x0802` | solver | `OVER_CONSTRAINED` | Solver: conflicting constraints (conflict set attached) |

Codes in categories `0x00`, `0x02`, `0x03`, `0x04` and `0x06` descend from the Demo's 16-code device taxonomy. That taxonomy's values (0–15) are remapped here; the Demo trace importer translates them.

## Trace

Every program, record, verdict, checker version, kernel version and budget event is recorded. The trace makes three things possible:
- **Replay:** `ReplayModel` bit-compares on the same device.
- **Attribution:** distinguishing a model gain from a harness gain ([11](11-self-improvement.md)).
- **Training data:** every run produces it.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-06-001 | Requirements MUST be immutable within a session; changes MUST arrive as a new caller-supplied revision. |
| AG-06-002 | A failed `document.save_version` MUST keep the workspace. |
| AG-06-003 | The harness MUST NOT optimize, repair or retry on the model's behalf. |
| AG-06-004 | A `sampled` result MUST NOT satisfy a requirement that demands `bounded` or `exact` evidence. |
| AG-06-005 | `AgStatus` values MUST be fixed and append-only. |
| AG-06-006 | Every GPU check MUST match its frozen reference verdict bit-for-bit in conformance. |
| AG-06-007 | Budget MUST NOT be refunded by deletion. |
