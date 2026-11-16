"""Load GSE125583 (Banner Sun Health; fusiform gyrus RNA-seq, 219 AD / 70 controls, Braak stage).

Inputs: <data>/GSE125583/GSE125583_series_matrix.txt.gz (metadata)
        <data>/GSE125583/raw/GSM*_sample*.tsv.gz (GSE125583_RAW.tar; ID_REF = Entrez gene ID,
        VALUE = size-factor-adjusted RPKM "nRPKM", count = uniquely aligned reads)
        <data>/hgnc/hgnc_complete_set.txt (HGNC; Entrez ID -> symbol)
Output: log2(nRPKM + 1), genes with nRPKM >= 1 in >= 20% of samples; metadata with group, age,
sex, braak (0-6), region.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load_illumina import read_series_matrix

ROMAN = {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, meta = read_series_matrix(data / "GSE125583" / "GSE125583_series_matrix.txt.gz")
    cols = {}
    for f in sorted((data / "GSE125583" / "raw").glob("GSM*.tsv.gz")):
        gsm = f.name.split("_")[0]
        t = pd.read_csv(f, sep="\t", comment="#", dtype={"ID_REF": str})
        cols[gsm] = t.set_index("ID_REF")["VALUE"]
    val = pd.DataFrame(cols)
    hg = pd.read_csv(data / "hgnc" / "hgnc_complete_set.txt", sep="\t", usecols=["symbol", "entrez_id"], dtype=str).dropna()
    hg = hg.drop_duplicates("entrez_id").set_index("entrez_id")["symbol"]
    val = val.loc[val.index.intersection(hg.index)]
    val = val[(val >= 1).mean(axis=1) >= 0.2]
    val.index = hg.loc[val.index].values
    val = val[~val.index.duplicated()]
    expr = np.log2(val + 1)

    m = pd.DataFrame(index=meta.index)
    m["group"] = meta["diagnosis"].map({"Alzheimer's disease": "AD", "control": "Control"})
    m["age"] = pd.to_numeric(meta["age"], errors="coerce")
    m["sex"] = meta["Sex"].map({"F": "female", "M": "male"})
    m["braak"] = meta["braak.score"].map(ROMAN)
    m["region"] = "FG"
    m = m.loc[m.index.intersection(expr.columns)]
    return expr[m.index], m
