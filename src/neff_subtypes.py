"""Neff et al. 2021 (Sci Adv, doi:10.1126/sciadv.abb5398) AD subtype reference.

Data file S2: 271 random-forest features with average log(CPM) per MSBB-AD subtype
(A, B1, B2, C1, C2 and the three classes A, B, C).
Data file S1 'Signatures': per-gene log fold change per subtype (A, B1, B2, C1, C2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_centroids(neff_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(neff_dir / "abb5398_data_file_s2.xlsx", header=None, skiprows=3)
    raw = raw.dropna(subset=[0])
    raw.columns = ["gene", "A", "B1", "B2", "C1", "C2", "A_cls", "B_cls", "C_cls"]
    raw = raw.set_index("gene").astype(float)
    raw = raw[~raw.index.duplicated()]
    sub = raw[["A", "B1", "B2", "C1", "C2"]]
    cls = raw[["A_cls", "B_cls", "C_cls"]].rename(columns=lambda c: c.replace("_cls", ""))
    return sub, cls


def load_logfc(neff_dir: Path) -> pd.DataFrame:
    raw = pd.read_excel(neff_dir / "abb5398_data_file_s1.xlsx", sheet_name="Signatures",
                        header=None, skiprows=3)
    raw = raw.dropna(subset=[0])
    raw.columns = ["gene", "A", "B1", "B2", "C1", "C2"]
    raw = raw.set_index("gene").astype(float)
    return raw[~raw.index.duplicated()]
