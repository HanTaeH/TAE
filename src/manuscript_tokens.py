"""Number supplementary tables in order of first citation (v11).

The manuscript source cites tables as {T:key}; KEYS maps each key to the internal sheet id used by
supplementary_tables.py. Tables are numbered by first appearance, {T:LAST} becomes the highest number,
and the old-to-new map is written for supplementary_tables.py --ren.

Usage: python manuscript_tokens.py --src draft_v11_src.md --out draft_v11.md --ren ren_v11.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KEYS = {"tests": "S7", "lit": "S1", "exprnull": "S3", "specnull": "S12", "strength": "S15", "poscon": "S17",
        "spike": "S21", "primary": "S2", "kappa": "S16", "spv": "S8", "stageextra": "S13", "neuronadj": "S9",
        "adjnull": "S10", "stagefu": "S11", "nft": "S23", "mediation": "S5", "seaadpb": "S4", "histo": "S14",
        "regulators": "S24", "schemes": "S18", "cohortv11": "S25", "csf": "S19", "hallmark": "S6", "residual": "S20",
        "bretigea": "S26", "direction": "S27", "crossregion": "S28", "signature": "S29", "shared": "S30", "spikev12": "S31",
        "regloo": "S32"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ren", type=Path, required=True)
    ap.add_argument("--extra", nargs=2, action="append", default=[], metavar=("SRC", "OUT"),
                    help="further files (e.g. the Supplementary Note) numbered with the same map")
    a = ap.parse_args()
    s = a.src.read_text()
    order = []
    for k in re.findall(r"\{T:([A-Za-z0-9]+)\}", s):
        if k != "LAST" and k not in order:
            order.append(k)
    unknown = [k for k in order if k not in KEYS]
    assert not unknown, unknown
    num = {k: i + 1 for i, k in enumerate(order)}
    s = re.sub(r"\{T:([A-Za-z0-9]+)\}", lambda m: str(len(order)) if m.group(1) == "LAST" else f"Supplementary Table S{num[m.group(1)]}", s)
    s = s.replace("Supplementary Tables S1–S" + str(len(order)), "Supplementary Tables S1–S" + str(len(order)))
    a.out.write_text(s)
    for src, out in a.extra:
        t = Path(src).read_text()
        bad = [k for k in re.findall(r"\{T:([A-Za-z0-9]+)\}", t) if k != "LAST" and k not in num]
        assert not bad, f"keys cited only in {src}: {bad}"
        t = re.sub(r"\{T:([A-Za-z0-9]+)\}", lambda m: str(len(order)) if m.group(1) == "LAST" else f"Supplementary Table S{num[m.group(1)]}", t)
        Path(out).write_text(t)
    ren = {KEYS[k]: f"S{num[k]}" for k in order}
    a.ren.write_text(json.dumps(ren, indent=1))
    print(json.dumps({k: num[k] for k in order}))
    unused = sorted(set(KEYS) - set(order))
    print("unused keys:", unused)


if __name__ == "__main__":
    main()
