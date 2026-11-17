"""GSE84422 (MSBB, Affymetrix U133A = GPL96): 107 subjects x 17 brain regions, 951 samples.

Expression is log2 (series matrix). Probes -> symbols with GPL96.annot.gz ('Gene symbol'); probes
with several symbols are dropped and the probe with the highest mean is kept per gene.
The GPL97 (U133B) chips are the same RNA on the companion array and are not used; GPL570 holds only
amygdala and nucleus accumbens (102 samples) and is not used.
Metadata: subject, region, age, sex, race, PMI, pH, CDR, Braak, neuropathological category,
average neuritic plaque density, CERAD sum, NFT density sum. group: AD = definite or probable AD,
Possible = possible AD, Control = normal.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {"subject id": "subject", "brain region": "region", "Sex": "sex", "postmortem interval minutes": "pmi",
          "ph": "ph", "clinical dementia rating": "cdr", "braak neurofibrillary tangle score": "braak",
          "neuropathological category": "np_category", "average neuritic plaque density": "plaque",
          "sum of cerad rating scores in multiple brain regions": "cerad_sum",
          "sum of neurofibrillary tangles density in multiple brain regions": "nft_sum"}
NUMERIC = ["age", "pmi", "ph", "cdr", "braak", "plaque", "cerad_sum", "nft_sum"]
NP_ORDER = {"Normal": 0, "Possible AD": 1, "Probable AD": 2, "definite AD": 3}


def read_series(path: Path):
    rows, ids, skip = {}, None, 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                for j, x in enumerate(line.rstrip("\n").split("\t")[1:]):
                    x = x.strip('"')
                    if ": " in x:
                        k, v = x.split(": ", 1)
                        rows.setdefault(ids[j], {})[k] = v
            elif line.startswith("!series_matrix_table_begin"):
                skip = i + 1
                break
    meta = pd.DataFrame.from_dict(rows, orient="index").rename(columns=RENAME)
    expr = pd.read_csv(path, sep="\t", skiprows=skip, index_col=0, comment="!")
    return expr, meta


def load(data: Path):
    expr, meta = read_series(data / "GSE84422" / "GSE84422-GPL96_series_matrix.txt.gz")
    ann = pd.read_csv(data / "GPL96" / "GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"],
                      index_col=0, low_memory=False)["Gene symbol"].dropna()
    ann = ann[~ann.str.contains("///")]
    expr = expr.loc[expr.index.intersection(ann.index)]
    best = expr.mean(axis=1).groupby(ann.loc[expr.index].values).idxmax()
    expr = expr.loc[best.values]
    expr.index = best.index
    for c in NUMERIC:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["female"] = (meta["sex"].str.lower() == "female").astype(float)
    meta["np_num"] = meta["np_category"].map(NP_ORDER)
    meta["group"] = meta["np_category"].map({"Normal": "Control", "Possible AD": "Possible",
                                             "Probable AD": "AD", "definite AD": "AD"})
    return expr.astype(float), meta.loc[expr.columns]


if __name__ == "__main__":
    import sys
    e, m = load(Path(sys.argv[1]))
    print(e.shape, m.shape, m["group"].value_counts().to_dict())
    print(e.iloc[:3, :3])
