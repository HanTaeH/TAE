"""Load GSE33000 (HBTRC prefrontal cortex, Rosetta/Merck two-channel arrays).

Inputs (downloaded manually from GEO, see README):
  - GSE33000_series_matrix.txt.gz : normalized log10 ratios (sample vs pool) + sample metadata
  - GSE33000_raw_data.txt.gz      : supplementary raw file; used only for its reporterID -> Gene column

Output: gene x sample expression DataFrame (AD + non-demented only) and a metadata DataFrame.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def _read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: dict[str, list[list[str]]] = {}
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                meta_rows.setdefault(key, []).append([v.strip('"') for v in vals])

    header = [h.strip('"') for h in table_lines[0].rstrip("\n").split("\t")]
    data = [l.rstrip("\n").split("\t") for l in table_lines[1:]]
    expr = pd.DataFrame(data, columns=header).set_index("ID_REF")
    expr.index = expr.index.str.strip('"')
    expr = expr.replace({"": np.nan, "null": np.nan}).astype(float)

    meta = pd.DataFrame(index=meta_rows["!Sample_geo_accession"][0])
    meta["title"] = meta_rows["!Sample_title"][0]
    for row in meta_rows.get("!Sample_characteristics_ch2", []):
        key = row[0].split(":", 1)[0].strip()
        meta[key] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in row]
    if "age" in meta:
        meta["age"] = meta["age"].str.replace(" yrs", "", regex=False).astype(float)
    return expr, meta


def _probe_to_gene(raw_path: Path) -> pd.Series:
    mapping = {}
    with gzip.open(raw_path, "rt") as fh:
        next(fh)
        for line in fh:
            rid, gene, _ = line.split("\t", 2)
            gene = gene.strip('"').split(",")[0].strip()
            # Rosetta placeholder IDs (RSE_...) carry no gene symbol
            if gene and not gene.startswith("RSE_"):
                mapping[rid] = gene
    return pd.Series(mapping, name="gene")


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr, meta = _read_series_matrix(data_dir / "GSE33000_series_matrix.txt.gz")
    p2g = _probe_to_gene(data_dir / "GSE33000_raw_data.txt.gz")

    keep = meta["disease status"].isin(["Alzheimer's disease", "non-demented"])
    meta = meta.loc[keep].copy()
    meta["group"] = np.where(meta["disease status"] == "Alzheimer's disease", "AD", "Control")
    expr = expr.loc[:, meta.index]

    expr = expr.loc[expr.index.intersection(p2g.index)]
    expr = expr.dropna(thresh=int(0.9 * expr.shape[1])).copy()
    expr["gene"] = p2g.loc[expr.index].values
    # one row per gene: keep the probe with the highest variance across samples
    expr["_var"] = expr.drop(columns="gene").var(axis=1)
    expr = expr.sort_values("_var", ascending=False).drop_duplicates("gene")
    expr = expr.drop(columns="_var").set_index("gene")
    return expr, meta
