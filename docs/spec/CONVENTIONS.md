# Spec Conventions

> Status: **Normative** · Spec v0.3.1 · Area: all

These conventions govern every document in `docs/spec`. `tools/check_spec.py` enforces the mechanical parts.

## 1. Normative language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY** and **OPTIONAL** are to be interpreted as described in RFC 2119 and RFC 8174, and only when they appear in **ALL CAPS**. Lowercase "must" in prose is descriptive.

## 2. Document classes and headers

Every document starts with its title, then a status line:

```
> Status: **Normative** | **Informative** · Spec vX.Y.Z · Area: <area>
```

| Class | Meaning | Documents |
|---|---|---|
| **Normative** | Defines required behaviour; conformance is tested against it | 01–08, 10, 11, 14, 16, 17, 18, CONVENTIONS, `op-catalog.json` |
| **Informative** | Rationale, reviews, examples; never overrides normative text | 00, 09 (benchmarks are normative only via their IDs), 12, 13, 15, GLOSSARY, CHANGELOG |

**Precedence:** when documents conflict, the order is
1. `op-catalog.json` (for op signatures and flags)
2. normative documents, with the higher spec version winning
3. informative documents

## 3. Requirement IDs

- Each normative document ends with a **"Normative requirements"** table of IDs `AG-<doc>-<nnn>`, for example `AG-06-004`. `<doc>` is the two-digit document number (`CV` for CONVENTIONS).
- IDs are **append-only**: never renumbered, never reused. Withdrawn requirements stay listed as `WITHDRAWN (vX.Y)`.
- Conformance tests ([09](09-validation.md)) reference requirement IDs. A requirement without a test is reported as `untested` in the conformance summary.

## 4. Naming

| Item | Convention | Example |
|---|---|---|
| Ops | `family.op_name`, lower snake case; canonical CATIA term; aliases in `aliases` | `part.edge_fillet` |
| Families | lower snake case, singular | `wireframe`, `query` |
| Crates | `agcad-<layer>`; backends `agcad-driver-<backend>` | `agcad-driver-cuda` |
| Rust types | UpperCamelCase; CUDA vocabulary below the driver line only (`Stream`, `Event`, `Graph`, `GraphExec`, `Library`, `Function`) | `DeviceSlice<T>` |
| Stream-ordered calls | suffix `_async` | `download_packed_async` |
| Status codes | `AgStatus`, UPPER_SNAKE, fixed `u16` values, category in the high byte, append-only | `0x0401 GEOMETRY_CHECK_FAILED` |
| Stable element names | `feature_ref/role[qualifiers].index` ([02](02-geometry-model.md#stable-naming)) | `part.pad#3/cap_end` |
| Requirement IDs | `AG-<doc>-<nnn>` | `AG-05-002` |

## 5. Units, numbers, layouts

- Internal units are **mm, rad, kg, s**. User-facing JSON accepts unit strings. Binary ABIs always use internal units.
- Binary layouts are little-endian and `repr(C)`. Records are 16-byte aligned, and array starts in device buffers are 128-byte aligned.
- Tolerances are always stated with units and an evidence level.

## 6. Versioning and deprecation

- **Spec:** SemVer. PATCH covers clarifications and new informative text. MINOR covers new ops, new optional fields and new requirement IDs. MAJOR covers removals or semantic changes, with migration notes.
- **Catalog, ABI, document schema:** independently versioned ([16](16-document-format.md#2-versioned-artifacts)).
- **Deprecation:** mark the item `status: deprecated` with a replacement, and keep it working for one MAJOR spec version.

## 7. Change control

- Every spec change updates [CHANGELOG](CHANGELOG.md) and the version in each touched document's status line.
- **Generated sections** (the tables in 03) are never hand-edited. Edit `tools/catalog_source.py` and regenerate.
- `tools/check_spec.py` MUST pass before a spec change merges.

## 8. Normative requirements

| ID | Requirement |
|---|---|
| AG-CV-001 | Every spec document MUST carry a status line with class, spec version and area. |
| AG-CV-002 | Every normative document MUST end with a Normative requirements table. |
| AG-CV-003 | Requirement IDs and `AgStatus` values MUST be unique and append-only. |
| AG-CV-004 | Generated catalog tables MUST NOT be edited by hand. |
