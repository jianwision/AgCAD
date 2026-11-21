# 18 — STEP Interchange, Receiver Profiles & Manufacturing Outputs

> Status: **Normative** · Spec v0.3.1 · Area: geo / file

## 1. Scope

This document covers:
- STEP load and save on the GPU, including their device-memory model
- STEP entity coverage
- **receiver profiles** for SolidWorks, CATIA and generic receivers
- validation properties
- manufacturing outputs: 3MF/AM, machining handoff with PMI, slicing

Delivery format follows the requirement's `process` ([14](14-constraints-evaluations.md#requirements)).

## 2. STEP coverage

| Area | Export | Import | Notes |
|---|---|---|---|
| Schemas | **AP242 ed.2+** (default), AP214 (option) | AP203, AP214, AP242 | The AP214 option exists for older receivers |
| Units, tolerance context | `SI_UNIT`/conversion-based length and angle units; `UNCERTAINTY_MEASURE_WITH_UNIT` from the document linear tolerance | same, mapped into internal units | Tolerance growth is reported |
| Solids | `MANIFOLD_SOLID_BREP`, `CLOSED_SHELL`, `ADVANCED_FACE`, `EDGE_CURVE`, `VERTEX_POINT` (advanced B-rep) | same, plus `BREP_WITH_VOIDS`, `SHELL_BASED_SURFACE_MODEL` | Faceted B-rep only for mesh-derived solids |
| Surfaces | `PLANE`, `CYLINDRICAL_SURFACE`, `CONICAL_SURFACE`, `SPHERICAL_SURFACE`, `TOROIDAL_SURFACE`, `B_SPLINE_SURFACE_WITH_KNOTS` (± rational), `SURFACE_OF_REVOLUTION`, `SURFACE_OF_LINEAR_EXTRUSION` | + `OFFSET_SURFACE`, `RECTANGULAR_TRIMMED_SURFACE` | Analytic types stay analytic |
| Curves | `LINE`, `CIRCLE`, `ELLIPSE`, `B_SPLINE_CURVE_WITH_KNOTS`, pcurves (`PCURVE`, `SEAM_CURVE`, `SURFACE_CURVE`) | + `PARABOLA`, `HYPERBOLA`, `OFFSET_CURVE_3D`, `TRIMMED_CURVE`, `COMPOSITE_CURVE` | Intersection curves export as B-spline + both pcurves |
| Sheets / wires | `SHELL_BASED_SURFACE_MODEL`, `GEOMETRIC_CURVE_SET` | same | Surface and wire bodies |
| Product structure | `PRODUCT`, `PRODUCT_DEFINITION`, `NEXT_ASSEMBLY_USAGE_OCCURRENCE`, `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` | same | Products/components with placements |
| Names | `PRODUCT.name`, `ADVANCED_FACE.name` = stable provenance names, body names | read into attributes | Names round-trip ([02](02-geometry-model.md#stable-naming)) |
| Presentation | colours/transparency via `STYLED_ITEM`; layers via `PRESENTATION_LAYER_ASSIGNMENT` | same | — |
| Validation properties | volume, surface area, centroid per solid; assembly instance counts | read and compared after import | §5 |
| PMI | semantic GD&T from `knowledge.tolerance` (AP242 semantic PMI); optional graphical PMI | read into `knowledge.tolerance` where semantic | M3+ |

**Not exported:** the feature history. Use `file.export_feature_script` for history-preserving handoff ([10](10-roadmap.md#capability-tracks-scheduled-into-milestones)).

## 3. GPU STEP pipeline and memory model

STEP is text (ISO 10303-21), but parsing and encoding are data-parallel, so both run on the GPU. The host only moves file bytes (GPUDirect Storage where available).

### 3.1 Import

```
file bytes ──(GDS / one pinned copy, chunked)──▶ device byte buffer
 1. LEX        per-chunk tokenization (strings, numbers, #refs, keywords); chunk-boundary fix-up by scan
 2. INDEX      entity table: #id → (type, token span) via sort + scan; type interning by hash
 3. PARSE      per-entity attribute decoding into typed SoA columns (one kernel bin per entity type)
 4. RESOLVE    #ref → slot index via sorted id table (binary search / hash); dangling refs → report
 5. BUILD      geometry objects (curves/surfaces) in arenas; topology via topology builders (02)
 6. HEAL       tolerance-budgeted sewing/healing on GPU; per-body report
 7. VALIDATE   evaluate.check + compare validation properties (§5)
```

**Device memory sizing (planned before launch, per [05](05-execution.md#memory-capacity-planning)):**

| Stage | Peak bytes (estimate) |
|---|---|
| Byte buffer | `F` (file size), streamed in chunks of `C` (default 256 MiB) when `F` exceeds the budget |
| Tokens | ≈ `0.6·F` (16-byte token records, typical token density) |
| Entity table | `24 B × E` (E = entity count) |
| Typed columns | ≈ `1.5·F` transient |
| Resulting geometry and topology | element-dependent; estimated from entity-type histogram after INDEX |

- **Chunked import:** for files above the budget, stages 1–4 run per chunk with a global entity table. Stage 5 runs per product/body. Token and column buffers are freed (stream-ordered) after RESOLVE.
- **Limits:** maximum entity count, maximum reference depth and maximum string length are enforced ([17](17-security.md)). Violations are per-file errors.

### 3.2 Export

```
1. NUMBER     assign #ids by scan over (products, representations, topology, geometry), deterministic order
2. FORMAT     per-entity text formatting into per-entity byte counts (shortest round-trip float formatting, 17 sig. digits max)
3. OFFSETS    exclusive scan of byte counts → write offsets
4. EMIT       write entities into one output buffer; header section from the receiver profile
5. VERIFY     (CI/conformance) re-import on GPU and through the OCCT oracle; compare validation properties
```

- Export output is **byte-deterministic** for identical input.
- Peak memory is roughly the output size plus 32 B per entity.

## 4. Receiver profiles

The `profile` argument of `file.export_step` selects receiver-specific choices. Every profile emits valid AP242/AP214. Profiles change *choices within the standard*, never correctness.

| Choice | `generic` | `solidworks` | `catia` |
|---|---|---|---|
| Schema | AP242 | AP214 (default) or AP242 | AP242 |
| Assemblies | NAUO product structure | NAUO, one file; components named from `PRODUCT.name` (mapped to SolidWorks part/assembly names on import) | NAUO; part numbers in `PRODUCT.id` |
| Offset/procedural surfaces | as native STEP entities | **converted to B-spline** within the conversion tolerance, for robust import | native where supported |
| Periodic surfaces | seam edges explicit | seam edges explicit; full-revolution faces split into two halves | seam edges explicit |
| Tolerance context | document tolerance | tolerance ≤ 1e-5 m equivalent; edges exceeding it are healed before export, or the export reports `GEOMETRY_CHECK_FAILED` | document tolerance |
| Colours | `STYLED_ITEM` per body/face | per body and face | per body/face + layers |
| Validation properties | yes | yes | yes |
| Semantic PMI | optional | off by default (graphical PMI only if requested) | on when `knowledge.tolerance` exists |

**Acceptance:**
- Every profile is verified by GPU re-import and OCCT reimport in CI.
- The **SolidWorks profile is additionally verified on a Windows runner with SolidWorks**, where licences exist (the Demo noted that this was never done). The check uses SolidWorks API automation to open the file, run Import Diagnostics (no faulty faces or gaps) and read mass properties against the validation properties within tolerance.
- Until that runner exists, the SolidWorks profile is marked `experimental` in reports.

## 5. Validation properties

- Export writes, per solid: volume, surface area and centroid (AP242 validation properties), plus instance counts for assemblies.
- Import recomputes them on the GPU and reports deviations. Defaults: relative 1e-6 for volume and area, and the linear tolerance for the centroid.
- Deviations beyond tolerance produce `GEOMETRY_CHECK_FAILED` in the import report. The body still loads, marked unvalidated.

## 6. Manufacturing outputs

| Process | Primary output | Contents |
|---|---|---|
| `additive` | **3MF** (`file.export_3mf`) | Core mesh with watertightness guaranteed; **Materials & Properties** (per-body material, colour); **Beam Lattice** extension for strut lattices (beams + radii instead of a dense mesh); **Slice** extension or `file.export_slices` for implicit/TPMS bodies too fine to mesh; VDB (`file.export_vdb`) for volumetric workflows |
| `machining_3ax` / `machining_5ax` | **STEP AP242** solid + semantic PMI (from `knowledge.tolerance`) | DFM evaluation results ([14](14-constraints-evaluations.md)) attached as a report; history via `file.export_feature_script` when the receiving CAD/CAM should edit |
| `molding` | STEP AP242 solid (+ draft/undercut report) | Parting and pull direction recorded as attributes |
| `sheet_metal` | STEP AP242 solid (flat pattern: later) | Bend data as attributes |

**GPU memory for manufacturing outputs:**
- 3MF meshes are generated from tessellation already resident on the device, then zip-encoded with deflate on the GPU or stored uncompressed (profile option).
- Slicing evaluates implicit bodies directly per layer (contouring by marching squares), streaming layer batches to the output, so implicit lattices never need a full mesh in memory.

## 7. Normative requirements

| ID | Requirement |
|---|---|
| AG-18-001 | STEP import and export MUST run on the GPU; the host MUST handle file bytes only. |
| AG-18-002 | The importer MUST plan device memory before launch and MUST stream in chunks when the file exceeds the budget. |
| AG-18-003 | Export MUST write validation properties for every solid, and import MUST compare them. |
| AG-18-004 | STEP export MUST be byte-deterministic for identical input. |
| AG-18-005 | Analytic surfaces MUST be exported as analytic STEP entities except where a receiver profile requires conversion. |
| AG-18-006 | Stable provenance names MUST be written to `ADVANCED_FACE.name` and MUST round-trip through AgCAD import. |
| AG-18-007 | The `solidworks` profile MUST be reported `experimental` until verified on a SolidWorks CI runner. |
| AG-18-008 | `additive` delivery MUST NOT require implicit-to-solid conversion; lattices MUST be exportable as 3MF beam lattices, meshes or slices. |
