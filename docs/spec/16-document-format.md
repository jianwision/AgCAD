# 16 — Document Format, Versioning & Migration

> Status: **Normative** · Spec v0.3.1 · Area: runtime / file

## 1. Scope

This document defines:
- the on-disk `.agcad` document
- versioning of the four independently versioned artifacts
- migration of saved documents when op versions change
- snapshots for crash recovery

## 2. Versioned artifacts

| Artifact | Version form | Compatibility rule |
|---|---|---|
| Spec | SemVer `MAJOR.MINOR.PATCH` | MINOR adds; MAJOR may break, with migration notes in [CHANGELOG](CHANGELOG.md) |
| Feature catalog (`op-catalog.json`) | `catalog_version` (SemVer) + per-op `version` (integer) | Opcodes are append-only; an op's behaviour changes only by bumping its `version`; old versions stay loadable until deprecated for one MAJOR |
| Driver ABI (bytecode, `AgStatus`, record layouts) | `abi_version` (integer) | Backward compatible: a newer driver accepts older ABI versions ([01](01-architecture.md#project-layout-driver--runtime-split-cuda-style)) |
| Document schema (`.agcad`) | `schema_version` (integer) | Readers MUST open every schema version since 1, migrating forward; writers write only the current version |

## 3. `.agcad` container

A `.agcad` file is a **zip container**. It is deterministic: entries are sorted, timestamps are fixed and compression is fixed, so identical documents produce identical bytes.

| Entry | Content |
|---|---|
| `manifest.json` | `schema_version`, `catalog_version`, `abi_version`, units, content hashes of every entry, producer |
| `tree/specification.json` | Specification tree: features (`op`, `op_version`, inputs by stable name, parameters), bodies, sets, networks, publications |
| `tree/knowledge.json` | Parameters, formulas, laws, rules, checks, reactions, design tables, tolerances |
| `udf/<hash>.ag.rs` | UDF sources (content-addressed); compiled artifacts are **not** stored (rebuilt from cache) |
| `versions/<n>/delta.bin` | Version history as CoW deltas over the tree and parameters |
| `exact/<version>/brep.bin` | Optional cached exact B-rep pages (driver-defined binary, little-endian, versioned) to skip rebuild on open |
| `requirements/<rev>.yaml` | Requirements revisions referenced by versions |
| `review/` | Comments and approvals ([10](10-roadmap.md#capability-tracks-scheduled-into-milestones)) |
| `trace/` | Optional trace references or embedded traces for replay |

**Source of truth:** the specification tree plus UDF sources. Cached B-rep pages are an optimization. A reader MUST be able to rebuild everything without them, and MUST discard them if their hash or ABI does not match.

## 4. Migration

- Each op version bump that changes semantics MUST ship a **migration function** `vN → vN+1`, written in AgLang over the feature's parameters and inputs. It is registered with the stdlib, and conformance tests it on stored fixtures.
- `file.open` applies migrations feature by feature, in tree order, then rebuilds and **compares** the result with the cached exact pages when present. Deviations beyond tolerance are reported per feature.
- A missing or failing migration returns `MIGRATION_FAILED` (`0x0103`) with the feature and versions involved. The document opens read-only at the last migratable feature (rollback-bar semantics), and nothing is silently dropped.
- Deprecated ops keep working for one MAJOR spec version after deprecation. Migration away from them is offered, but not forced, on open.

## 5. Snapshots and crash recovery

- The runtime writes a **snapshot** (tree + parameters + version pointer, without B-rep caches) after every `document.save_version`, and at most every 60 s while a workspace is dirty.
- When a worker becomes `Poisoned` (a sticky device fault, [08](08-hal-backends.md#device-api-core--extension-traits)), the harness restarts the worker and restores the last snapshot. The restoration is recorded in the trace and reported to the caller.
- Snapshot writes come from GPU-encoded buffers. The host handles bytes only ([05](05-execution.md#host-interaction-budget)).

## 6. Normative requirements

| ID | Requirement |
|---|---|
| AG-16-001 | Readers MUST open all schema versions ≥ 1 by forward migration. |
| AG-16-002 | `.agcad` output MUST be byte-deterministic for identical documents. |
| AG-16-003 | Cached exact pages MUST NOT be trusted without a matching content hash and `abi_version`. |
| AG-16-004 | Every semantic op version bump MUST provide a tested migration. |
| AG-16-005 | Failed migration MUST return `MIGRATION_FAILED` and MUST NOT silently drop features. |
