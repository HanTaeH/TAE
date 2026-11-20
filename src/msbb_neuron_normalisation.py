"""Neff 2021 considered normalising expression by a BRETIGEA neuronal surrogate proportion variable
(SPV) but chose a CDR normalisation. What happens to the subtypes if the neuronal SPV is used?

MSBB GSE84422 (GPL96), AD samples of each of 17 regions (the parahippocampal gyrus, Neff's subtyping
region, reported separately). BRETIGEA's SPV is the first principal component of the scaled marker
genes (sign set to agree with their mean); markers are NEURON_MARKERS not in the classifier.
Every gene is residualised on [SPV, age, sex, PMI, pH] (variant 'spv') or on [SPV, CDR, age, sex,
PMI, pH] (variant 'spv_cdr'), the classes are projected again, and we report:
  - class proportions and agreement (Cohen's kappa) with the labels before normalisation;
  - rho between the normalised and the original C-A score;
  - the share of C-A variance explained by the SPV before normalisation (R^2);
  - held-out check: SPV built from a random half of the markers, then rho of the normalised C-A
    score with the other half's (equally normalised) score, median over 200 splits; a value near 0
    means the SPV captured the neuronal signal the C-A score carries;
  - rho of the normalised C-A score with CDR and Braak.

Usage: python msbb_neuron_normalisation.py --data /mnt/project-files/d1-data/raw --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_gse84422
import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import NEURON_MARKERS, SEED

COV = ["age", "female", "pmi", "ph"]


def spv(Z: np.ndarray) -> np.ndarray:
    """BRETIGEA-style surrogate proportion: PC1 of scaled markers, sign aligned with the marker mean."""
    u, s, vt = np.linalg.svd(Z - Z.mean(1, keepdims=True), full_matrices=False)
    p = vt[0] * s[0]
    return p if np.corrcoef(p, Z.mean(0))[0, 1] >= 0 else -p


def resid(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    D = np.column_stack([np.ones(D.shape[0]), D])
    return X - (X @ D) @ np.linalg.pinv(D.T @ D) @ D.T


def kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in range(3))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def project(X, cent):
    r = corr_np(z_np(X), cent)
    return r[:, 2] - r[:, 0], r.argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-split", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 13)
    e, m = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    g = cls.index.intersection(e.index)
    gi = e.index.get_indexer(g)
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    ng = [x for x in NEURON_MARKERS if x in e.index and x not in g]
    ni = e.index.get_indexer(ng)
    rows = []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        mm = mm.dropna(subset=COV + ["cdr"])
        X = e[mm.index].values
        C = mm[COV].values.astype(float)
        cdr, braak = mm["cdr"].values.astype(float), mm["braak"].values.astype(float)
        Zn = z_np(X[ni])
        s = spv(Zn)
        ca0, lab0 = project(resid(X, C)[gi], cent)
        r2 = float(np.corrcoef(resid(ca0[None], C)[0], resid(s[None], C)[0])[0, 1] ** 2)
        for var, D in [("spv", np.column_stack([s, C])), ("spv_cdr", np.column_stack([s, cdr, C]))]:
            Xr = resid(X, D)
            ca, lab = project(Xr[gi], cent)
            held = []
            k = len(ni)
            for _ in range(a.n_split):
                p = rng.permutation(k)
                sa = spv(Zn[p[: k // 2]])
                Da = np.column_stack([sa, D[:, 1:]])
                Xa = resid(X, Da)
                ca_a, _ = project(Xa[gi], cent)
                hb = z_np(Xa[ni[p[k // 2:]]]).mean(0)
                held.append(stats.spearmanr(ca_a, hb)[0])
            rows.append({"region": reg, "variant": var, "n_AD": len(mm),
                         "r2_ca_by_spv_before": round(r2, 3),
                         "frac_A_before": (lab0 == 0).mean(), "frac_B_before": (lab0 == 1).mean(),
                         "frac_C_before": (lab0 == 2).mean(),
                         "frac_A": (lab == 0).mean(), "frac_B": (lab == 1).mean(), "frac_C": (lab == 2).mean(),
                         "kappa_vs_before": kappa(lab0, lab),
                         "rho_ca_norm_vs_ca_before": stats.spearmanr(ca, ca0)[0],
                         "rho_ca_norm_vs_heldout_neuron_median": float(np.median(held)),
                         "rho_ca_norm_vs_cdr": stats.spearmanr(ca, cdr)[0],
                         "rho_ca_norm_vs_braak": stats.spearmanr(ca, braak)[0],
                         "rho_ca_before_vs_cdr": stats.spearmanr(ca0, cdr)[0]})
        print(reg, "done", flush=True)
    d = pd.DataFrame(rows).round(3)
    d.to_csv(a.out / "msbb_neuron_spv_normalisation.csv", index=False)
    summ = {}
    for var, dv in d.groupby("variant"):
        num = dv.drop(columns=["region", "variant"])
        summ[var] = {"median_17_regions": num.median().round(3).to_dict(),
                     "parahippocampal_gyrus": num[dv.region.str.contains("Parahippocampal").values]
                     .iloc[0].to_dict()}
    (a.out / "msbb_neuron_spv_normalisation.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
