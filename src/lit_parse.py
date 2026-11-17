"""Parse a PubMed (MEDLINE format) export and apply rule-based pre-screening for the literature review.

Rules (pre-screen; every kept record is then checked by hand):
  exclude publication types Review / Systematic Review / Meta-Analysis / Editorial / Comment / Congress
  keep records whose title+abstract describe patient-level subgrouping of AD (subtype / subgroup /
  cluster / stratif / taxonomy / molecular heterogeneity applied to patients, cases, individuals,
  samples or donors) AND postmortem brain molecular data; animal-only records are excluded.

Usage: python lit_parse.py export.txt out_dir
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

EXCL_PT = {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Comment", "Congress", "Letter", "News"}


def parse(path: Path) -> pd.DataFrame:
    recs, cur, key = [], {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            if cur:
                recs.append(cur)
            cur, key = {}, None
            continue
        if len(line) > 4 and line[4] == "-" and line[:4].strip():
            key = line[:4].strip()
            val = line[6:]
            cur.setdefault(key, []).append(val)
        elif key:
            cur[key][-1] += " " + line.strip()
    if cur:
        recs.append(cur)
    rows = []
    for r in recs:
        rows.append({"PMID": r.get("PMID", [""])[0], "year": (r.get("DP", [""])[0] or "")[:4],
                     "first_author": r.get("AU", [""])[0], "journal": r.get("TA", r.get("JT", [""]))[0],
                     "title": r.get("TI", [""])[0], "abstract": " ".join(r.get("AB", [])),
                     "pub_types": "; ".join(r.get("PT", [])), "mesh": "; ".join(r.get("MH", []))})
    return pd.DataFrame(rows)


PATIENT_SUB = re.compile(r"(?:subtyp|subgroup|sub-group|clusters? of (?:patients|individuals|cases|donors|samples)|"
                         r"stratif|taxonomy|molecular heterogeneity|patient heterogeneity|distinct (?:molecular )?(?:groups|classes)|"
                         r"(?:patients|individuals|cases|donors|samples|subjects) (?:were|was) (?:clustered|classified|grouped|divided)|"
                         r"consensus clustering|unsupervised clustering|non-negative matrix|nmf)", re.I)
CELL_SUB = re.compile(r"(?:cell|neuron|neuronal|microglia\w*|astrocyt\w*|oligodendrocyt\w*|interneuron|receptor|isoform)[- ]subtype", re.I)
BRAIN_MOL = re.compile(r"(?:post-?mortem|brain tissue|brain regions?|cortex|cortical|hippocamp|gyrus|ROSMAP|MSBB|Mount Sinai|Mayo|AMP-AD|GSE\d+|brain (?:transcriptom|gene expression|RNA))", re.I)
ANIMAL = re.compile(r"\b(?:mice|mouse|rat|rats|zebrafish|drosophila|macaque)\b", re.I)
HUMAN = re.compile(r"\b(?:human|patients?|individuals|donors|subjects|postmortem|post-mortem)\b", re.I)


def prescreen(d: pd.DataFrame) -> pd.DataFrame:
    text = d["title"] + " " + d["abstract"]
    d["excl_pubtype"] = d["pub_types"].apply(lambda s: bool(set(x.strip() for x in s.split(";")) & EXCL_PT))
    d["patient_subgrouping"] = text.str.contains(PATIENT_SUB) & ~(text.apply(lambda t: len(PATIENT_SUB.findall(t)) <= len(CELL_SUB.findall(t))))
    d["brain_molecular"] = text.str.contains(BRAIN_MOL)
    d["animal_only"] = text.str.contains(ANIMAL) & ~text.str.contains(HUMAN)
    d["keep_for_manual"] = ~d["excl_pubtype"] & d["patient_subgrouping"] & d["brain_molecular"] & ~d["animal_only"]
    return d


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    d = parse(src)
    n0 = len(d)
    d = d.drop_duplicates("PMID")
    d = prescreen(d)
    d.to_csv(out / "records_prescreened.csv", index=False)
    print({"records": n0, "unique": len(d), "reviews_etc": int(d.excl_pubtype.sum()),
           "keep_for_manual": int(d.keep_for_manual.sum())})
