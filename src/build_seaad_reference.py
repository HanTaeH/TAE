"""Build a cell-type reference from the SEA-AD MTG single-nucleus reference (neurotypical donors).

Input : Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad
        (https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/RNAseq/, Allen Institute SEA-AD)
Output: seaad_mtg_reference.csv  genes x cell types, mean counts-per-million per nucleus
        seaad_mtg_reference_ncells.csv

X is a CSC matrix of raw counts (nuclei x genes). Every nucleus is CPM-normalised and averaged per
subclass. Subclasses are then grouped into the types used for deconvolution (weighted by nucleus
count within a group). SST and PVALB interneurons are kept apart because SEA-AD (Gabitto 2024)
reports SST interneuron loss early and excitatory neuron loss late.

Usage: python build_seaad_reference.py --h5ad ref.h5ad --out /mnt/project-files/d1-data/seaad/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


GROUPS = {
    "Exc": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
    "Inh_SST": ["Sst", "Sst Chodl"],
    "Inh_PVALB": ["Pvalb", "Chandelier"],
    "Inh_other": ["Lamp5", "Lamp5 Lhx6", "Sncg", "Vip", "Pax6"],
    "Astro": ["Astrocyte", "Astro"],
    "Oligo": ["Oligodendrocyte", "Oligo"],
    "OPC": ["OPC"],
    "Micro": ["Microglia-PVM", "Micro-PVM"],
    "Vascular": ["Endothelial", "Endo", "VLMC", "Pericyte"],
}


def read_obs_col(f: h5py.File, col: str) -> np.ndarray:
    g = f["obs"][col]
    if isinstance(g, h5py.Group):  # anndata >= 0.8 categorical
        cats = g["categories"][()].astype(str)
        return cats[g["codes"][()]]
    vals = g[()]
    if "__categories" in f["obs"] and col in f["obs"]["__categories"]:
        return f["obs"]["__categories"][col][()].astype(str)[vals]
    return vals.astype(str)


def var_names(f: h5py.File) -> np.ndarray:
    v = f["var"]
    key = v.attrs.get("_index", "_index")
    return v[key][()].astype(str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default=None, help="obs column with subclass labels")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.h5ad, "r") as f:
        cols = list(f["obs"].keys())
        label = a.label or next(c for c in ["subclass_label", "Subclass", "subclass"] if c in cols)
        sub = read_obs_col(f, label)
        genes = var_names(f)
        lut = {s: g for g, lst in GROUPS.items() for s in lst}
        unknown = sorted(set(sub) - set(lut))
        print("subclasses not grouped:", unknown)
        X = f["X"]
        enc = X.attrs.get("encoding-type", "")
        if isinstance(enc, bytes):
            enc = enc.decode()
        if enc != "csc_matrix":
            raise ValueError(f"expected csc_matrix counts, got {enc}")
        keep = np.array([s in lut for s in sub])
        subs = sorted(set(sub[keep]))
        code = np.full(len(sub), -1)
        for k, s in enumerate(subs):
            code[sub == s] = k
        indptr = X["indptr"][()]
        # pass 1: library size per nucleus
        tot = np.zeros(len(sub))
        step = 2000
        for g0 in range(0, len(genes), step):
            lo, hi = indptr[g0], indptr[min(g0 + step, len(genes))]
            tot += np.bincount(X["indices"][lo:hi], weights=X["data"][lo:hi], minlength=len(sub))
        ncell = {s: int((code == k).sum()) for k, s in enumerate(subs)}
        # pass 2: mean CPM per subclass (all nuclei)
        acc = np.zeros((len(genes), len(subs)))
        scale = np.where(tot > 0, 1e6 / np.maximum(tot, 1), 0)
        for g0 in range(0, len(genes), step):
            g1 = min(g0 + step, len(genes))
            lo, hi = indptr[g0], indptr[g1]
            idx = X["indices"][lo:hi]
            val = X["data"][lo:hi] * scale[idx]
            gcol = np.repeat(np.arange(g0, g1), np.diff(indptr[g0:g1 + 1]))
            c = code[idx]
            ok = c >= 0
            np.add.at(acc, (gcol[ok], c[ok]), val[ok])
            print(f"genes {g1}/{len(genes)}", flush=True)
        acc /= np.array([ncell[s] for s in subs])
        ref = {s: acc[:, k] for k, s in enumerate(subs)}

    sub_ref = pd.DataFrame(ref, index=genes)
    sub_ref.to_csv(a.out / "seaad_mtg_subclass_reference.csv.gz")
    n = pd.Series(ncell)
    grouped = {}
    for g, lst in GROUPS.items():
        members = [s for s in lst if s in sub_ref.columns]
        if members:
            w = n[members] / n[members].sum()
            grouped[g] = (sub_ref[members] * w.values).sum(axis=1)
    pd.DataFrame(grouped).to_csv(a.out / "seaad_mtg_reference.csv.gz")
    n.to_csv(a.out / "seaad_mtg_reference_ncells.csv")


if __name__ == "__main__":
    main()
