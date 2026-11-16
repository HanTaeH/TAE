"""Load Illumina (GPL10558) GEO series: GSE132903 and GSE118553.

Inputs (downloaded manually from GEO):
  - <GSE>/<GSE>_series_matrix.txt.gz
  - GPL10558 full table (GEO platform page -> "Download full table"), any file name under <data>/GPL10558/

Output: gene x sample expression (log2, one probe per gene) and a metadata DataFrame
with harmonised columns: group (AD / Control / AsymAD), age, sex, region, subject.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: dict[str, list[list[str]]] = {}
    lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                lines.append(line)
            elif line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                meta_rows.setdefault(key, []).append([v.strip('"') for v in vals])
    header = [h.strip('"') for h in lines[0].rstrip("\n").split("\t")]
    rows = [l.rstrip("\n").split("\t") for l in lines[1:]]
    expr = pd.DataFrame(rows, columns=header).set_index("ID_REF")
    expr.index = expr.index.str.strip('"')
    expr = expr.replace({"": np.nan, "null": np.nan}).astype(float)

    meta = pd.DataFrame(index=meta_rows["!Sample_geo_accession"][0])
    meta["title"] = meta_rows["!Sample_title"][0]
    meta["source"] = meta_rows["!Sample_source_name_ch1"][0]
    for row in meta_rows.get("!Sample_characteristics_ch1", []):
        key = row[0].split(":", 1)[0].strip()
        meta[key] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in row]
    return expr, meta


def read_gpl(gpl_dir: Path) -> pd.Series:
    """Return probe ID -> gene symbol from a GEO 'full table' download."""
    files = [p for p in gpl_dir.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no GPL table in {gpl_dir}")
    path = files[0]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        skip = 0
        for line in fh:
            if line.startswith(("#", "!", "^")):
                skip += 1
            else:
                break
    tab = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, comment=None,
                      on_bad_lines="skip", engine="python")
    sym_col = next(c for c in ["Symbol", "ILMN_Gene", "Gene symbol", "GENE_SYMBOL"] if c in tab.columns)
    m = tab.set_index("ID")[sym_col].astype(str).str.strip()
    return m[(m != "") & (m != "nan")]


def collapse(expr: pd.DataFrame, p2g: pd.Series) -> pd.DataFrame:
    expr = expr.loc[expr.index.intersection(p2g.index)].dropna(thresh=int(0.9 * expr.shape[1])).copy()
    gene = p2g.loc[expr.index].values
    var = expr.var(axis=1).values
    order = np.argsort(-var)
    expr = expr.iloc[order]
    expr.index = gene[order]
    return expr[~expr.index.duplicated()]


def _age(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("+", "", regex=False), errors="coerce")


def load(data_dir: Path, gse: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr, meta = read_series_matrix(data_dir / gse / f"{gse}_series_matrix.txt.gz")
    p2g = read_gpl(data_dir / "GPL10558")
    out = pd.DataFrame(index=meta.index)
    if gse == "GSE132903":
        out["group"] = meta["diagnosis"].map({"AD": "AD", "ND": "Control"})
        out["age"] = _age(meta["expired_age (years)"])  # "90+" coded as 90
        out["sex"] = meta["Sex"].str.lower()
        out["region"] = "MTG"
        out["subject"] = meta.index
    elif gse == "GSE118553":
        out["group"] = meta["disease state"].map({"AD": "AD", "AsymAD": "AsymAD", "control": "Control"})
        out["age"] = _age(meta["age"])
        out["sex"] = meta["gender"].str.lower()
        out["region"] = meta["tissue"]
        out["subject"] = meta["individual"].str.replace(r"\s+", " ", regex=True)
    else:
        raise ValueError(gse)
    return collapse(expr, p2g), out
