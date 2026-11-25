"""Neff et al.'s cell-type normalisation reproduced with the BRETIGEA marker sets (v12, review round 10, Reviewer 2).

Neff et al. estimated surrogate proportion variables (SPVs) with BRETIGEA (McKenzie et al. 2018) from
1,000 human brain markers per cell type (500 for OPCs; neurons, astrocytes, endothelial cells, microglia,
oligodendrocytes, OPCs) and normalised expression on them with BRETIGEA defaults. The earlier
approximation used one SPV from 14 neuronal markers. Here BRETIGEA's findCells/adjustCells are
re-implemented (package source 1.0.3, markers_df_human_brain; github.com/cran/BRETIGEA):
  SPV_k = first right singular vector of the row-centred log expression of the cell type's markers
          present on the platform (up to 1,000), sign set so that its mean correlation with the markers is
          positive; normalised expression = residuals of each gene on all six SPVs (addMeans = FALSE).
Covariates (age, sex; in MSBB also PMI and pH) are regressed out together with the SPVs, as before.
Per MSBB region (AD subjects) and per independent GEO cohort (AD samples):
  - R^2 of the C-A score on the six SPVs; rho of the neuronal SPV with the 14-marker neuron score;
  - Cohen's kappa between class labels before and after normalisation, against six PC1s of random
    1,000-gene sets (non-marker, non-classifier) used in the same way (200 draws);
  - Spearman of the normalised C-A score with the (unnormalised) neuron score and neuronal SPV;
  - MSBB: age/sex partial Spearman of the normalised C-A score with CDR, Braak and tangle sum.

Usage: python bretigea_normalisation_v12.py --data /mnt/project-files/d1-data/raw --out ../results/bretigea_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
from scipy import stats

import load_gse84422
import neff_subtypes
from meta_reanalysis import INDEPENDENT, rho
from msbb_neuron_normalisation import kappa, project, resid
from null_models import z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import cohorts

CELLS = ["ast", "end", "mic", "neu", "oli", "opc"]


def load_markers(path: Path) -> dict[str, list[str]]:
    df = list(rdata.read_rda(str(path)).values())[0]
    return {c: list(df.loc[df["cell"] == c, "markers"].astype(str)) for c in CELLS}


def find_cells(X: np.ndarray, rows: list[int]) -> np.ndarray:
    """BRETIGEA findCells (method SVD, scale TRUE): rows are marker indices in X (genes x samples)."""
    D = X[rows] - X[rows].mean(1, keepdims=True)
    v = np.linalg.svd(D, full_matrices=False)[2][0]
    c = np.array([np.corrcoef(v, d)[0, 1] for d in D])
    return -v if np.nanmean(c) < 0 else v


def spvs(X, index, markers, n_marker=1000):
    out, used = [], {}
    for c in CELLS:
        rows = [index[g] for g in markers[c] if g in index and np.nansum(X[index[g]] > 0) > 0 and X[index[g]].std() > 0][:n_marker]
        used[c] = len(rows)
        out.append(find_cells(X, rows))
    return np.column_stack(out), used


def analyse(X, index, gi, cent, ni, C, markers, pool, rng, n):
    X = np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X)
    X = np.nan_to_num(X)
    S, used = spvs(X, index, markers)
    ca0, lab0 = project(resid(X[gi], C), cent)
    neu14 = z_np(X[ni]).mean(0)
    ca1, lab1 = project(resid(X[gi], np.column_stack([S, C])), cent)
    D = np.column_stack([np.ones(len(ca0)), S])
    r2 = 1 - ((ca0 - D @ np.linalg.lstsq(D, ca0, rcond=None)[0]) ** 2).sum() / ((ca0 - ca0.mean()) ** 2).sum()
    k = kappa(lab0, lab1)
    kn = []
    for _ in range(n):
        R = np.column_stack([find_cells(X, list(rng.choice(pool, 1000, replace=False))) for _ in CELLS])
        kn.append(kappa(lab0, project(resid(X[gi], np.column_stack([R, C])), cent)[1]))
    kn = np.array(kn)
    return {"markers_used": used, "r2_CA_on_6_spv": round(float(r2), 3),
            "rho_neu_spv_vs_14marker": round(float(stats.spearmanr(S[:, 3], neu14)[0]), 3),
            "kappa_bretigea": round(k, 3), "kappa_random6_median": round(float(np.median(kn)), 3),
            "kappa_random6_q05_q95": [round(float(np.quantile(kn, q)), 3) for q in (.05, .95)],
            "p_kappa_le_random": round(float(((kn <= k).sum() + 1) / (n + 1)), 4),
            "label_share_before": {c: round(float((lab0 == i).mean()), 3) for i, c in enumerate("ABC")},
            "label_share_after": {c: round(float((lab1 == i).mean()), 3) for i, c in enumerate("ABC")},
            "rho_CA_raw_vs_neuron14": round(float(stats.spearmanr(ca0, neu14)[0]), 3),
            "rho_CA_norm_vs_neuron14": round(float(stats.spearmanr(ca1, neu14)[0]), 3),
            "rho_CA_norm_vs_neu_spv": round(float(stats.spearmanr(ca1, S[:, 3])[0]), 3),
            "rho_CA_norm_vs_CA_raw": round(float(stats.spearmanr(ca1, ca0)[0]), 3)}, ca1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1302)
    markers = load_markers(a.data / "bretigea" / "markers_df_human_brain.rda")
    allmk = {g for v in markers.values() for g in v} | set(NEURON_MARKERS) | {x for v in CELL_MARKERS.values() for x in v}
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    report = {"msbb": {}, "geo": {}}

    def setup(e):
        g = cls.index.intersection(e.index)
        index = {x: i for i, x in enumerate(e.index)}
        gi = e.index.get_indexer(g)
        cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
        ni = e.index.get_indexer([x for x in NEURON_MARKERS if x in e.index and x not in g])
        pool = e.index.get_indexer(e.index.difference(g).difference(list(allmk)))
        return index, gi, cent, ni, pool

    e, m = load_gse84422.load(a.data)
    index, gi, cent, ni, pool = setup(e)
    clin = []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        mm = mm.dropna(subset=["age", "female", "pmi", "ph"])
        X = e[mm.index].values
        C = mm[["age", "female", "pmi", "ph"]].values.astype(float)
        res, ca1 = analyse(X, index, gi, cent, ni, C, markers, pool, rng, a.n)
        cov = mm[["age", "female"]].values.astype(float)
        for y in ["cdr", "braak", "nft_sum"]:
            res[f"{y}_rho_norm"] = round(rho(ca1, mm[y].values.astype(float), cov), 3)
            res[f"{y}_rho_raw"] = round(rho(project(resid(X[gi], C), cent)[0], mm[y].values.astype(float), cov), 3)
        res["n_AD"] = len(mm)
        report["msbb"][reg] = res
        print(reg, json.dumps(res), flush=True)
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        index, gi, cent, ni, pool = setup(e)
        ad = (m["group"] == "AD").values
        mm = m.loc[ad]
        age = pd.to_numeric(mm["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (mm["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        res, _ = analyse(e.loc[:, ad].values, index, gi, cent, ni, np.column_stack([age, sex]), markers, pool, rng, a.n)
        res["n_AD"] = int(ad.sum())
        report["geo"][name] = res
        print(name, json.dumps(res), flush=True)
    d = pd.DataFrame(report["msbb"]).T
    num = lambda c: pd.to_numeric(d[c])
    report["msbb_summary"] = {c: round(float(num(c).median()), 3) for c in
                              ["r2_CA_on_6_spv", "rho_neu_spv_vs_14marker", "kappa_bretigea", "kappa_random6_median",
                               "rho_CA_raw_vs_neuron14", "rho_CA_norm_vs_neuron14", "rho_CA_norm_vs_neu_spv",
                               "cdr_rho_raw", "cdr_rho_norm", "braak_rho_raw", "braak_rho_norm", "nft_sum_rho_raw", "nft_sum_rho_norm"]}
    report["msbb_summary"]["n_regions_p_kappa_lt_05"] = int((num("p_kappa_le_random") < .05).sum())
    (a.out / "bretigea_normalisation_v12.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["msbb_summary"], indent=2))


if __name__ == "__main__":
    main()
