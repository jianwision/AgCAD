# 04 — AgLang, AgIR and the Bytecode ABI

> Status: **Normative** · Spec v0.3.1 · Area: rtc / ir / vm

There are three levels:
- **AgLang** is what models write: UDFs, wrangles, implicit bodies, rules, checks, objectives.
- **AgIR** is the typed SSA intermediate form that everything lowers to. Its JSON form is the LLM batch-program format.
- **AgIR bytecode** is AgIR in fixed-width binary records, written straight into GPU memory by device-resident models and executed by the device VM.

All three share one type system, one feature table ([`op-catalog.json`](op-catalog.json)) and one harness.

## AgLang

AgLang is a **Rust-syntax subset**, statically typed and unit-aware. AgCAD's own front end (a `syn`-based parser in `agcad-rtc`, the NVRTC-like runtime compiler) parses and compiles it **at runtime**, so model-written code loads in milliseconds and never invokes `rustc` or a toolchain.

The same syntax compiles **ahead of time** through `#[udf]`, `#[implicit]`, `#[wrangle]` and `#[check]` proc-macros. That is how the standard features ship inside the binary. A model reading `udf.view_source("part.pad")` sees exactly the code that runs, and can copy it and load it back at runtime.

**Why Rust syntax:**
- The whole system is Rust (as in the Demo), so there is one language for the kernel, the stdlib and model-authored features.
- Strong typing and explicit units catch most LLM mistakes at `udf.define` time, with precise spans.
- The CubeCL kernel style is already Rust-embedded, so the lowering is direct.

```rust
use agcad::prelude::*;            // units (mm, deg), P (parameter builders), std features

#[udf(name = "bolted_flange", outputs(body: SolidBody), roles("disc", "bolt_hole"))]
fn bolted_flange(ctx: &Ctx,
                 #[param(range = 40.0*mm..=400.0*mm)] od: Length,
                 #[param(default = 12.0*mm)] t: Length,
                 #[param(default = 100.0*mm)] pcd: Length,
                 #[param(range = 3..=24)] n: u32,
                 #[param(default = 9.0*mm)] hole: Length) -> FeatureRef {
    let s = sketch::create(ctx.xy_plane());
    sketch::circle(&s, CircleMode::CenterRadius, &[pt2(0.0, 0.0)], od / 2.0);
    let disc = part::pad(sketch::regions(&s), Limits::dimension(t)).role("disc");
    let h = part::hole(query::created_by(&disc, "cap_end"),
                       &[pt2(pcd / 2.0, 0.0)], HoleSpec::simple(hole).through());
    transform::circular_pattern(&[h], ctx.z_axis(), n, 360.0 * deg)
}

#[implicit(bounds = "auto", lipschitz = 1.0)]
fn gyroid_core(p: Vec3, #[param(default = 8.0*mm)] cell: Length,
               #[param(default = 0.8*mm)] wall: Length) -> Length {
    let k = TAU / cell;
    let g = (k*p.x).sin()*(k*p.y).cos() + (k*p.y).sin()*(k*p.z).cos() + (k*p.z).sin()*(k*p.x).cos();
    g.abs() / k - wall / 2.0                  // conservative distance estimate
}

#[wrangle(over = "point")]                    // Houdini-style per-element snippet
fn jitter(pt: &mut Point, #[param(default = 0.2*mm)] amount: Length) {
    pt.p += pt.n * amount * noise3(pt.p * 0.1);
}

#[check(severity = "error", phase = "save")]  // adds evidence; never replaces requirements
fn min_ligament(body: &Body) -> bool {
    evaluate::thickness_analysis(body).min >= 2.0 * mm
}
```

**AgLang is a strict subset of real Rust.** Every valid AgLang program also compiles with `rustc` against the `agcad` prelude crate. This means:
- Examples and stdlib sources are ordinary Rust that CI type-checks with `rustc` and tests.
- An LLM's Rust priors apply directly (traits, closures, iterators within the subset, units as newtypes).
- Diagnostics follow `rustc` style (spans, `help:` notes) and add **"did you mean"** suggestions from the alias table: `extrude` → `part::pad`, `fillet` → `part::edge_fillet`.
- **KPIs from M1:** compile-error rate per LLM program, reference-error rate, and repair iterations per task. If compile errors stay above target, the alias layer and exemplar packs are improved first (L2), before the language is changed.

**Language rules (enforced by the verifier; violations are `SANDBOX_VIOLATION`):**
- No `unsafe`, no IO, no `std` beyond the AgLang prelude, no global mutable state, no host callbacks.
- Randomness comes from an explicit `seed` and a counter-based RNG.
- Loops are bounded: static ranges, or `while` with `#[max_iter = N]`. Recursion is bounded by a declared depth.
- Memory is declared through output capacity hints. Overflow returns `CAPACITY_EXCEEDED` and nothing is silently truncated (Demo rule).
- Types: `f32`/`f64`/`Df64` scalars and vectors; integers; `bool`; fixed arrays; element handles; attributes; units-carrying `Length`/`Angle`/`Mass`; geometry values. Generics only over these.
- Control flow on per-sample values lowers to `select`/predication. Data-dependent topology goes through feature calls.
- A UDF may call any feature in the table, including other UDFs. Composition is unrestricted.

**Front ends:**
- **AgLang** (runtime-parsed Rust subset) is the default.
- **AgIR JSON** is for LLMs that prefer structured tool calls.
- **Raw CubeCL kernels** are an expert escape hatch, compiled AOT into the binary only. Model-authored code always goes through the AgLang verifier.

## AgIR

AgIR is an SSA program: `%v = op(args…) {attrs}`.

| Aspect | Rule |
|---|---|
| Ops | Any feature-table op (`part.pad`, `proc.scatter`, …), plus expression ops (`add`, `mul`, `sin`, `select`, `dot`, …) inside UDF bodies |
| Types | Scalars with units, vectors, tensors, `Element`/`Selection`, `Body` by body type, `Parameter`, `Configurations<T>`, `Program` |
| Effects | `pure`, `doc_write` or `io`, from the manifest. The harness gates `doc_write` by workspace and `io` by requirements |
| Batch axes | Inferred. A value carrying `configuration` broadcasts through every op, and `study.summarize`/`study.rank` reduce it |
| Control | `study.iterate` (bounded), `proc.for_each`, `proc.solver`, `document.checkpoint` (yield to caller) |
| Names | Each statement gets a `feature_id`, which is the root of the stable naming ([02](02-geometry-model.md#stable-naming)) |

### JSON encoding (LLM batch program)

Here is one `run_program` call. It sweeps 4,096 flange configurations, ranks them by mass under a stiffness proxy, keeps the 4 best, rebuilds them exactly and checkpoints.

```json
{
  "agir": "0.2", "requirements_rev": 3, "expected_revision": 41,
  "program": [
    {"id": "p",   "op": "document.edit", "args": {"branch": "main"}},
    {"id": "def", "op": "udf.define", "args": {"source": "@file:bolted_flange.ag.rs", "lang": "aglang"}},
    {"id": "cfg", "op": "study.configurations",
     "args": {"table": {"grid": {"od": ["100 mm", "160 mm", 16], "t": ["6 mm", "20 mm", 16], "n": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24]}}}},
    {"id": "f",   "op": "udf.instantiate", "args": {"definition": "%def", "inputs": {}, "params": "%cfg"}},
    {"id": "upd", "op": "study.update_configurations", "args": {"cfgs": "%cfg"}},
    {"id": "sc",  "op": "study.evaluate",
     "args": {"cfgs": "%cfg", "objectives": [
        {"minimize": {"op": "evaluate.measure_inertia", "of": "%f", "field": "mass"}},
        {"require":  {"op": "evaluate.thickness_analysis", "of": "%f", "min": "3 mm"}}]}},
    {"id": "top", "op": "study.rank", "args": {"cfgs": "%cfg", "scores": "%sc", "k": 4}},
    {"id": "chk", "op": "evaluate.check", "args": {"body": "%f"}, "over": "%top"},
    {"id": "img", "op": "study.render", "args": {"items": ["%f"], "views": ["iso", "top"], "mode": "shaded_edges"}, "over": "%top"},
    {"id": "cp",  "op": "document.checkpoint", "args": {"label": "top4"}}
  ]
}
```

- `"over": "%top"` restricts a statement to a configuration subset. This is how exact-phase work is limited to survivors.
- String args with units are converted at decode time.
- `@file:` references content in the request bundle.
- **References and formulas** (string prefixes):
  - `"%id"` / `"%id.field"`: result of an earlier statement.
  - `"$name"`: a parameter, re-read per configuration.
  - `"=expr"`: a formula over parameters, evaluated per configuration at rebuild. It accepts unit literals (`"=od/2 - 5 mm"`, `"=30 deg"`), the operators `+ - * / ^`, parentheses, and the functions `sqrt abs min max sin cos tan atan2 floor ceil round`. Syntax errors are reported at decode time with a column; unknown parameters are reported at rebuild together with the list of known parameters.
  - A bare string is a literal. Legacy bare parameter names still resolve, with a `check_program` warning that suggests `"$name"`.
- **`knowledge.parameter` with an existing name modifies that parameter in place.** Parameters are unique by name, as in CATIA; it never adds a duplicate.
- **Linting** (MCP `check_program`, CLI `agcad check-program`): every problem in one pass, each with a JSON path, the op signature and a concrete fix, plus a size preview (number of configurations). `run_program` runs the same lint first and returns all errors.

### Autodiff

- AgIR supports forward and reverse mode through expression ops and through features whose manifest says `differentiable: true`. Parametric features differentiate with respect to parameters while topology is fixed.
- At a topology change (a branch flip, a face split appearing), the gradient is reported as `discontinuous` together with the parameter location, and the result is never silently zeroed.
- `study.sensitivity` and `study.optimize` use this. GPU models can read gradients from `study.observe` tensors.

### Vectorization

- The compiler lifts scalar UDF code over `[configuration, entity, sample]` and emits SoA kernels with segment offsets.
- Ragged results (a different face count per configuration) use per-configuration capacity, a prefix-sum compaction pass and explicit overflow errors. This follows the Demo's bounded-arena pattern.

## Bytecode (device-resident models)

A GPU model emits AgIR as **blocks** of 64-byte command records into a device ring buffer. The layout is little-endian, `repr(C)`, owned and versioned by `agcad-driver`, and backward compatible under the driver's versioning rule ([01](01-architecture.md#project-layout-driver--runtime-split-cuda-style)). This was revised after the [API review](13-api-review.md) (A3-1).

**Block header (64 B, 16-byte aligned):**

| Bytes | Field |
|---|---|
| 0–3 | magic `AGB1` |
| 4–5 | `abi_version: u16` |
| 6–7 | `record_size: u16` (64 today; readers skip unknown tails) |
| 8–11 | `record_count: u32` |
| 12–15 | `requirements_rev: u32` |
| 16–23 | `catalog_hash: u64` (feature-table version; mismatch → `NEED_RECOMPILE`) |
| 24–27 | `round: u32` |
| 28–31 | `producer_id: u32` (which model/adaptor) |
| 32–63 | reserved, zero |

**Command record (64 B, 16-byte aligned):**

| Bytes | Field |
|---|---|
| 0–1 | `opcode: u16`. Feature-table opcode (`0x0000–0x7FFF`), session UDF opcode (`0x8000–0xFEFF`) or expression opcode (`0xFF00–0xFFFF`) |
| 2–3 | `flags: u16`. Bits: `over_subset`, `checkpoint`, `expects_exact`, `differentiable_request` |
| 4–7 | `configuration: u32` |
| 8–11 | `dst: u32`, the SSA value slot |
| 12–35 | `args: u32[6]`, value slots or arg-arena offsets |
| 36–39 | `expected_revision: u32` |
| 40–43 | `configuration_mask_slot: u32` |
| 44–47 | reserved |
| 48–63 | `imm: f64[2]`, immediate parameters in internal units (8-byte aligned) |

**Status record (16 B):** `{ status: AgStatus u16, flags: u16, dst: u32, budget_used: u32, aux: u32 }`. Status values are defined in [06](06-harness.md#error-codes).

All device tensors (command ring, status ring, observation, sensitivity) also publish a **DLPack** descriptor, so external frameworks can attach zero-copy ([07](07-model-interfaces.md)).

- Larger arguments (point lists, AgLang expression graphs) live in an **arg arena**: a device buffer of typed records referenced by offset.
- **Model-invented geometry on device:** a CuModel can write a small **expression graph** (bytecode with expression opcodes) defining an implicit function or a wrangle. The device VM interprets it directly, so no JIT is needed inside the inner loop. Hot graphs are promoted to compiled kernels asynchronously by the host. This gives device models the same open-ended capability as LLM-written AgLang, at interpreter speed.
- **Validation** happens on device before execution, as the first stage of the GPU harness ([06](06-harness.md#gpu-resident-harness)):
  - opcode in the model's allow-list
  - revisions current
  - budget available
  - argument slots in range and of the right type
  - immediates finite
- Every record produces a status record, which the model reads on its next step.

The record shape descends from the Demo `Command` ABI (`HARNESS.md`: opcode, revisions, value). AgCAD widens it to the full feature table and adds expression bytecode.

## Normative requirements

| ID | Requirement |
|---|---|
| AG-04-001 | Every valid AgLang program MUST also compile with `rustc` against the `agcad` prelude. |
| AG-04-002 | The verifier MUST reject `unsafe`, IO, unbounded loops and undeclared capacities with `SANDBOX_VIOLATION`. |
| AG-04-003 | Capacity overflow MUST return `CAPACITY_EXCEEDED`; silent truncation MUST NOT occur. |
| AG-04-004 | Bytecode blocks MUST begin with the `AGB1` header; readers MUST skip unknown record tails using `record_size`. |
| AG-04-005 | Gradient discontinuities at topology changes MUST be reported, not zeroed. |
