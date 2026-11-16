"""Load GSE106241 (Agilent 8x60K custom exon array, GPL24170): inferior temporal cortex of 60 donors
with Braak NFT stage 0-6, age, sex, RNA quality, APOE e4 dose and amyloid-beta 42 levels.

Inputs: <data>/GSE106241/GSE106241_series_matrix.txt.gz, <data>/GPL24170/GPL24170_family.soft.gz
VALUE = quantile-normalised signal (linear) -> log2(x + 1); highest-variance probe per gene.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from load_illumina import read_series_matrix


def read_soft_platform(path: Path) -> pd.Series:
    rows, inside = [], False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                inside = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if inside:
                rows.append(line.rstrip("\n").split("\t"))
    t = pd.DataFrame(rows[1:], columns=rows[0])
    t = t[t["GeneSymbol"].str.len() > 0]
    return t.set_index("ID")["GeneSymbol"]


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr, meta = read_series_matrix(data / "GSE106241" / "GSE106241_series_matrix.txt.gz")
    sym = read_soft_platform(data / "GPL24170" / "GPL24170_family.soft.gz")
    expr = np.log2(expr.clip(lower=0) + 1)
    expr = expr.loc[expr.index.intersection(sym.index)]
    expr = expr[expr.isna().mean(axis=1) <= 0.1].copy()
    expr["gene"] = sym.loc[expr.index].values
    var = expr.drop(columns="gene").var(axis=1)
    expr = expr.assign(_v=var).sort_values("_v", ascending=False).drop_duplicates("gene")
    expr = expr.set_index("gene").drop(columns="_v")

    def col(prefix):
        c = next(c for c in meta.columns if c.lower().startswith(prefix))
        return pd.to_numeric(meta[c], errors="coerce")
    m = pd.DataFrame(index=meta.index)
    m["braak"] = col("braak")
    m["age"] = col("age")
    m["sex"] = col("sex").map({1: "female", 2: "male"})
    m["rna_quality"] = col("rna quality")
    m["apoe4"] = col("apoe")
    m["abeta42"] = col("amyloid-beta 42")
    m["region"] = "ITC"
    return expr, m
