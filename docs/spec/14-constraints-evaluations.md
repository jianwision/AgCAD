# 14 — Constraints & Evaluations

> Status: **Normative** · Spec v0.3.1 · Area: solver / harness

Constraints and evaluations show up in many places across the spec:
- sketch constraints
- assembly constraints
- Knowledgeware formulas, rules and checks
- requirements
- study objectives
- design-for-manufacturing (DFM) checks
- Design Checker rules

This document unifies them into **one GPU-resident model**, solved by [`agcad-solver`](01-architecture.md#solver-library-agcad-solver) and checked by the GPU harness ([06](06-harness.md)).

## One model, three roles

```
Evaluation   m(x, geometry) → value [+ ∂value/∂x] [+ evidence, bound]
Constraint   Evaluation ⋈ bound (=, ≤, ≥, ∈)  with tolerance, scope, role
Requirement  Constraint owned by the harness, with phase + required evidence (immutable)
```

| Concept | CATIA term | Example | Who owns it | Role |
|---|---|---|---|---|
| **Evaluation** | Measure / Analysis / parameter measure | wall thickness min, mass, clearance, draft angle, G2 deviation, tool reach | Any feature, UDF, model | Produces values |
| **Driving constraint** | Sketcher constraint, 3D constraint, formula, assembly constraint | `distance = 40 mm`, `tangency`, `pcd = od − 20 mm` | Specification tree (model-editable) | **Solved**: determines parameters/geometry |
| **Inequality constraint** | (Knowledge check with optimization) | `min_wall ≥ 2 mm`, `clearance ≥ 0.5 mm` | Specification tree or study | **Solved** inside `study.optimize`; **checked** otherwise |
| **Check** | Knowledge check | `ligament ≥ 2 mm` warning | Model or user | **Checked**; adds evidence; rejects only if a requirement references it |
| **Requirement** | (Requirements / specification in PLM) | `R2: thickness_min ≥ 3 mm, evidence ≥ bounded, at save` | Caller / harness | **Checked** at declared phases; never modifiable by the model |
| **Objective** | Optimization goal | minimize mass | Study / requirements | **Optimized**; feedback only unless declared hard |

**The rule:** the *same* evaluation object can feed a driving constraint, a check, a requirement and an objective. It is defined once, evaluated once per configuration and revision, and cached.

## Evaluations

**Definition:**
- Built-in `evaluate.*` ops: measure, inertia, curvature, connect checker, draft, thickness, distance, clash, sectioning, overhang, undercut, tool accessibility, internal radius, check.
- Composed evaluations: `evaluate.evaluation` holds AgLang expressions over built-ins, for example `thickness_min / max_extent`.

**Every evaluation declares:**

| Property | Meaning |
|---|---|
| `inputs` | Elements, bodies and parameters it reads (dependency tracking) |
| `representation` | Which GPU representation it runs on: exact B-rep, tessellation, NanoVDB bricks, ray/BVH, sample grids |
| `evidence` | The best achievable level, plus the **escalation chain** to reach higher levels |
| `gradient` | `analytic`, `autodiff`, `finite_difference` or `none`; discontinuity reporting |
| `cost` | Calibrated per-configuration cost model, used by budget metering and scheduling |
| `batch_axes` | Normally `[configuration, entity, sample]` |

**Escalation chain** example for minimum wall thickness:

| Step | Method | Evidence |
|---|---|---|
| 1 | NanoVDB brick distance, sampled | `sampled` |
| 2 | Interval/affine arithmetic over the bricks | `bounded` |
| 3 | RT-core ray casts against the exact-tessellation BVH, with chordal bound | `bounded` (tighter) |
| 4 | Exact B-rep distance via batched interval Newton (`solver.root`) | `exact` |

The harness climbs only as far as the requirement's evidence demands, and only for configurations still alive.

## GPU evaluation pipeline

All evaluations for all live configurations run in **one fused pass per wavefront** inside the GPU harness ([06](06-harness.md#gpu-resident-harness)):

```
1. DEPENDENCY      revision-keyed cache lookup; only dirty evaluations proceed
2. REPRESENTATION  ensure needed representations exist (build/refresh tessellation, BVH, bricks) — shared across evaluations
3. GROUP           bin evaluations by (kind, representation) → one specialized kernel per bin
4. EVALUATE        batched over configurations × entities × samples; reductions deterministic (CUB)
5. COMPARE         constraints/requirements: margin = value − bound, normalized by tolerance
6. ESCALATE        configurations whose verdict is `unknown` at current evidence → next chain step
7. GRADIENT        on request (study.sensitivity, optimize, CuModel observation): adjoints through autodiff
8. EMIT            verdicts, margins, evidence, gradients → status/report/observation buffers
```

- Representations are built once and shared. Ten evaluations on the same tessellation pay for one tessellation.
- Margins are signed and tolerance-normalized (`margin / tol`). That makes them directly usable as ranking keys, as model observations and as optimizer penalties.

## Constraint solving on the GPU

All driving and inequality constraints in scope form a **constraint graph**: variables are parameters and element degrees of freedom; edges are constraints. `agcad-solver` runs four steps:
1. **Analyze** (`constraint.analyze`, `solver.graph`):
   - DOF count
   - structural rank
   - redundancy and conflict sets (with element names)
   - decomposition into rigid clusters and a solve plan, so large sketches and assemblies become many small dense blocks
2. **Solve:** batched LM / Gauss–Newton (`solver.nls`) over **all configurations at once**. Each block is a small dense system solved in registers or shared memory (≤64 DOF), or with cuSOLVER-batched / cuDSS beyond that.
3. **Inequalities:**
   - In `study.optimize`: augmented Lagrangian or interior point.
   - In plain updates: they are checked, and a violation returns margins with gradients so the model or an optimizer can act.
4. **Report:** per configuration, the status (`ACCEPTED`, `UNDER_CONSTRAINED` with free DOF directions, `OVER_CONSTRAINED` with a minimal conflict set, `SOLVER_DIVERGED` with residual history).

**Scopes and order:**
- Sketch → part 3D constraints → formulas/rules (Knowledgeware) → assembly → study.
- Each scope is solved on the GPU in dependency order within the same graph launch.
- Cyclic formula dependencies are detected by `solver.graph` and reported, never iterated silently.

## Requirements

Requirements ([06](06-harness.md#requirements-immutable-during-a-session)) are constraints with three extra properties: **owner = harness**, **phase**, and **required evidence**. Evaluating them is the same pipeline, but the harness's frozen implementations are used and model-authored evaluations can't substitute for them.

**Process-driven delivery.** Requirements declare the manufacturing process, and that selects both the delivery representation and the DFM evaluation family:

| `process` | Delivery | DFM evaluations (GPU) |
|---|---|---|
| `additive` | mesh / 3MF / VDB (no B-rep required) | overhang, min feature size, trapped volume, thickness |
| `machining_3ax` / `machining_5ax` | Solid body, STEP | tool accessibility, internal radius, thin walls, undercuts |
| `molding` | Solid body, STEP | draft, undercut, wall thickness uniformity |
| `sheet_metal` | Solid body, STEP (+ flat pattern later) | bend radius, relief, hole-to-edge |
| `none` | as requested | — |

So a lattice part for additive manufacturing never needs the hard implicit-to-solid conversion. Machined parts do, or they are modeled in Part Design from the start.

## Gradients and models

- Margins and their gradients go straight into the CuModel observation tensor (`study.observe`) and into `study.optimize`.
- LLM reports carry the top violated constraints with margins and **causal explanations**, for example "fillet r=6 infeasible: adjacent edge `pad#3/side.2` is 4.1 mm; max feasible ≈3.9 mm" ([07](07-model-interfaces.md)).
- Gradients through topology changes are flagged `discontinuous`, with the parameter location.

## Caching and incrementality

- The cache key is `(evaluation id, input element revisions, configuration, representation params, evidence level)`.
- On update, only evaluations downstream of dirty elements are recomputed. Across configurations that share a topology class ([05](05-execution.md)), representation work is shared where geometry is identical.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-14-001 | An evaluation MUST be defined once and shared by every constraint, check, requirement and objective that uses it. |
| AG-14-002 | Evaluations MUST declare inputs, representation, evidence, gradient and cost. |
| AG-14-003 | Escalation MUST stop at the evidence level the requirement demands. |
| AG-14-004 | Over-constrained results MUST include a minimal conflict set; under-constrained results MUST include the free DOF. |
| AG-14-005 | Requirements MUST be evaluated with frozen harness implementations, never with model-authored evaluations. |
