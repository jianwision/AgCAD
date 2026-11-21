"""Render op-catalog.json into the generated section of 03-op-catalog.md."""
import json
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent
BEGIN, END = "<!-- BEGIN GENERATED CATALOG -->", "<!-- END GENERATED CATALOG -->"
FLAGS = {"gpu": "GPU", "host": "host (control plane)"}


def render(cat):
    out = []
    fams = {}
    for op in cat["ops"]:
        fams.setdefault(op["family"], []).append(op)
    for fam, ops in fams.items():
        layer = ops[0]["layer"]
        out.append(f"\n### `{fam}` ({layer}, family 0x{cat['families'][fam]:02X}, {len(ops)} ops)\n")
        out.append("| Op | K | Opcode | Summary | Aliases | Inputs → Outputs | Phase | Backend | Axes | ∂ | MS |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for o in ops:
            io = f"{', '.join(o['inputs']) or '—'} → {', '.join(o['outputs']) or '—'}"
            axes = ",".join(a[0].upper() for a in o["batch_axes"]) or "—"
            out.append(
                f"| `{o['name']}` | {'●' if o.get('kernel_primitive') else ''} | `0x{o['opcode']:04X}` | {o['summary']} | {', '.join(o['aliases']) or '—'} | `{io.replace('|', '/')}` | {o['phase']} | "
                f"{FLAGS[o['backend']]} | {axes} | {'✓' if o['differentiable'] else ''} | {o['milestone']} |"
            )
    return "\n".join(out) + "\n"


def main():
    cat = json.loads((SPEC / "op-catalog.json").read_text())
    md_path = SPEC / "03-op-catalog.md"
    md = md_path.read_text()
    head, rest = md.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    md_path.write_text(head + BEGIN + "\n" + render(cat) + END + tail)


if __name__ == "__main__":
    main()
