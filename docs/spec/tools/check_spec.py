"""Consistency checks for docs/spec. Exit code != 0 on any failure.

1. op-catalog.json parses, required fields present, enums valid, names/opcodes unique.
2. Every catalog op appears in 03-op-catalog.md.
3. Every op referenced in an AgIR JSON example ("op": "family.name") or in backticked
   prose (`family.name`) exists.
4. Every relative Markdown link in docs/spec resolves.
5. Every error code used in the catalog is defined in 06-harness.md.
6. CONVENTIONS: every doc has a status line (class, spec version, area); normative docs end
   with a Normative requirements table; requirement IDs AG-<doc>-<nnn> are unique and match
   their document; AgStatus values are unique; aliases are unique within a family.
"""
import json
import re
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent
REQUIRED = ["name", "family", "opcode", "version", "summary", "inputs", "outputs", "batch_axes",
            "phase", "backend", "effect", "deterministic", "differentiable", "failure_codes",
            "milestone", "status"]
errors = []


def err(msg):
    errors.append(msg)


cat = json.loads((SPEC / "op-catalog.json").read_text())
schema = cat["schema"]
names, opcodes = set(), set()
for op in cat["ops"]:
    for k in REQUIRED:
        if k not in op:
            err(f"{op.get('name')}: missing field {k}")
    for k in ("phase", "backend", "effect", "status"):
        if op.get(k) not in schema[k]:
            err(f"{op['name']}: bad {k}={op.get(k)}")
    for a in op.get("batch_axes", []):
        if a not in schema["batch_axes"]:
            err(f"{op['name']}: bad batch axis {a}")
    if op["name"] in names:
        err(f"duplicate name {op['name']}")
    if op["opcode"] in opcodes:
        err(f"duplicate opcode {op['opcode']:#06x}")
    names.add(op["name"])
    opcodes.add(op["opcode"])
    if op["opcode"] >> 8 != cat["families"][op["family"]]:
        err(f"{op['name']}: opcode family mismatch")

catalog_md = (SPEC / "03-op-catalog.md").read_text()
for n in names:
    if f"`{n}`" not in catalog_md:
        err(f"03-op-catalog.md missing {n}")

op_ref = re.compile(r'"op"\s*:\s*"([a-z_]+\.[a-z_0-9]+)"')
link = re.compile(r"\]\(([^)#\s]+)(#[^)]*)?\)")
for md in sorted(SPEC.glob("*.md")):
    text = md.read_text()
    for m in op_ref.finditer(text):
        if m.group(1) not in names:
            err(f"{md.name}: example uses unknown op {m.group(1)}")
    for m in re.finditer(r"`([a-z_]+)\.([a-z_0-9]+)[`(]", text):
        fam, op = m.group(1), m.group(2)
        if fam in cat["families"] and f"{fam}.{op}" not in names and not op.endswith("_"):
            err(f"{md.name}: prose references unknown op {fam}.{op}")
    for m in link.finditer(text):
        target = m.group(1)
        if re.match(r"[a-z]+://", target):
            continue
        if not (md.parent / target).exists():
            err(f"{md.name}: broken link {target}")

harness = (SPEC / "06-harness.md").read_text()
for code in sorted({c for op in cat["ops"] for c in op["failure_codes"]}):
    if f"`{code}`" not in harness:
        err(f"error code {code} not defined in 06-harness.md")

# --- conventions ---
status_re = re.compile(r"^> Status: \*\*(Normative|Informative)\*\* · Spec v(\d+\.\d+\.\d+) · Area: .+$", re.M)
req_re = re.compile(r"^\| (AG-([0-9]{2}|CV)-([0-9]{3})) \|", re.M)
seen_req, versions = {}, set()
for md in sorted(SPEC.glob("*.md")):
    text = md.read_text()
    m = status_re.search(text)
    if not m:
        err(f"{md.name}: missing or malformed status line")
        continue
    versions.add(m.group(2))
    normative = m.group(1) == "Normative"
    reqs = req_re.findall(text)
    if normative and "Normative requirements" not in text:
        err(f"{md.name}: normative doc without Normative requirements section")
    if normative and not reqs:
        err(f"{md.name}: normative doc without requirement IDs")
    docnum = "CV" if md.name == "CONVENTIONS.md" else md.name[:2]
    for rid, num, _ in reqs:
        if num != docnum:
            err(f"{md.name}: requirement {rid} does not match document number {docnum}")
        if rid in seen_req:
            err(f"duplicate requirement id {rid} ({md.name}, {seen_req[rid]})")
        seen_req[rid] = md.name
if len(versions) > 1:
    err(f"docs disagree on spec version: {sorted(versions)}")
codes = re.findall(r"^\| `(0x[0-9A-F]{4})` \| [^|]+ \| `([A-Z_]+)` \|", harness, re.M)
if len({v for v, _ in codes}) != len(codes) or len({n for _, n in codes}) != len(codes):
    err("AgStatus values or names are not unique in 06-harness.md")
for fam in cat["families"]:
    al = [a for op in cat["ops"] if op["family"] == fam for a in op["aliases"]]
    dups = {a for a in al if al.count(a) > 1}
    if dups:
        err(f"family {fam}: aliases map to multiple ops: {sorted(dups)}")

prims = sum(1 for op in cat["ops"] if op.get("kernel_primitive"))
if any(op["backend"] == "host" and op["family"] not in ("udf",) for op in cat["ops"]):
    err("only udf compile-time ops may run on host (GPU-first data plane)")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"ok: {len(names)} ops ({prims} kernel primitives), {len(seen_req)} requirement IDs, {len(codes)} AgStatus codes, {len(list(SPEC.glob('*.md')))} docs")
