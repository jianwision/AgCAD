# 17 — Security & Threat Model

> Status: **Normative** · Spec v0.3.1 · Area: harness / server / driver

## 1. Assets and trust boundaries

| Asset | Why it matters |
|---|---|
| Requirements, reference checks, held-out sets, promotion gate | Integrity of every verdict and of self-improvement ([11](11-self-improvement.md)) |
| Documents and IP (geometry, UDF source) | Confidentiality across tenants |
| GPU context and device memory | Availability: a fault poisons the context ([08](08-hal-backends.md)) |
| The LLM's context | Integrity: untrusted text can steer the agent |

**Untrusted inputs:**
- model-written AgLang/AgIR/bytecode
- imported files (STEP, mesh, VDB, `.agcad`)
- names, attributes and comments inside them
- network requests to the MCP server
- L4 candidate kernels

## 2. Threats and controls

| Threat | Control |
|---|---|
| **Malicious or buggy model code** (out-of-bounds access, infinite loops, memory blowup) | AgLang verifier (no `unsafe`, bounded loops, checked indexing, declared capacities); budget metering on device; first-run UDFs and all L4 candidates in **sandbox worker processes** with their own GPU context; compute-sanitizer before promotion |
| **Context poisoning** (sticky CUDA fault) | Process-per-session/tenant workers; poisoned worker → restart + snapshot restore ([16](16-document-format.md#5-snapshots-and-crash-recovery)) |
| **Prompt injection through geometry metadata.** Imported STEP names, attributes, comments or UDF source text such as "ignore requirements and…" reach the LLM via reports | All untrusted strings in reports are **quoted as data** inside typed fields (`"untrusted_text": …`), never concatenated into instruction text; reports carry a provenance tag per string; the MCP tool descriptions tell the LLM that such fields are data; requirements can only be changed by the caller, never by document content (`STALE_REQUIREMENTS`) |
| **Malicious imported files** (parser bombs, huge counts) | Importers enforce size, count and depth limits; GPU lexers have bounded buffers; failures are per file |
| **Evaluator capture** (an improver modifies or peeks at oracles or held-out sets) | Frozen artifacts are read-only and stored outside candidate sandboxes; held-out queries are rate-limited and counted; humans approve L5 ([11](11-self-improvement.md#invariants)) |
| **Cross-tenant leakage** | One worker process per tenant; MIG/green-context partitions in the multi-tenant server; no shared device allocations across tenants except read-only stdlib libraries; VMM IPC exports are read-only and per tenant |
| **Resource exhaustion** (GPU time, memory, compile storms) | Per-tenant budgets (`gpu_ms`, `host_ms`, compile count); capacity planner ([05](05-execution.md#memory-capacity-planning)); compile cache with quotas |
| **Unauthenticated MCP access** | The MCP server binds to localhost by default; remote access requires token authentication over TLS; every call is logged with the caller identity in the trace |
| **Supply chain** (NVIDIA libraries, crates, datasets) | Pinned versions with hashes per harness release (review F8); `cargo deny`/audit in CI; dataset licences recorded per task family |

## 3. Normative requirements

| ID | Requirement |
|---|---|
| AG-17-001 | Model-authored code MUST pass the AgLang verifier before compilation, and MUST run in a sandbox worker on first execution. |
| AG-17-002 | A sticky device fault MUST NOT affect sessions in other worker processes. |
| AG-17-003 | Untrusted strings MUST appear in reports only as quoted data fields with provenance, never as instruction text. |
| AG-17-004 | Document content MUST NOT be able to change requirements, reference checks or budgets. |
| AG-17-005 | The MCP server MUST default to localhost-only, and MUST require authentication for remote access. |
| AG-17-006 | Frozen self-improvement artifacts MUST be inaccessible for writing from candidate sandboxes. |
