"""Priority 1: random gene-set null model (Venet et al. 2011 style).

Question: are the Neff classifier genes *more* tied to cell composition and disease stage
than random genes with the same expression level? If random genes behave the same way,
our earlier findings are a generic property of bulk brain data, not of the Neff subtypes.

For every cohort / region, observed values from the Neff genes are compared against N random
gene sets that
  - have the same size as the measured Neff genes,
  - are matched on mean expression (same count drawn from each expression decile),
  - exclude Neff classifier genes and the cell-marker genes used for adjustment.
Random sets inherit the Neff centroid profiles gene-by-gene (gene i of the random set gets the
centroid row of Neff gene i, matched within expression decile), so the projection machinery is
identical and only the genes change.

Metrics (all computed identically for observed and random sets):
  r2_cell      median R² of genes explained by the five cell-marker scores (AD samples)
  ari_adj      agreement (ARI) of class assignment before vs after cell-score adjustment
  stage_rho    Spearman between diagnosis order (Control < AsymAD < AD) and C-minus-A score
               (correlation to C centroid minus correlation to A centroid; all samples)
  neuron_rho   Spearman between C-minus-A score and neuron score (AD samples)

Usage: python null_models.py --data /mnt/project-files/d1-data/raw --out ../results/null --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import neff_subtypes
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import GROUP_ORDER, cohorts


def z_np(x: np.ndarray) -> np.ndarray:
    """Row-wise z-score of genes x samples; NaN -> 0."""
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nan_to_num((x - mu) / sd)


def corr_np(z: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """Pearson r, samples x centroids, between columns of z and columns of cent."""
    zc = (z - z.mean(0)) / z.std(0)
    cc = (cent - cent.mean(0)) / cent.std(0)
    return zc.T @ cc / z.shape[0]


def metrics(x_ad: np.ndarray, x_all: np.ndarray, cent: np.ndarray, cells: np.ndarray,
            neuron: np.ndarray, stage: np.ndarray) -> dict:
    """cent columns are classes A, B, C (centred per gene)."""
    z = z_np(x_ad)
    lab = corr_np(z, cent).argmax(1)
    Xd = np.column_stack([np.ones(len(cells)), cells])
    beta, *_ = np.linalg.lstsq(Xd, z.T, rcond=None)
    resid = z.T - Xd @ beta
    ss = (z.T ** 2).sum(0)
    r2 = 1 - (resid ** 2).sum(0) / np.where(ss == 0, 1, ss)
    lab_adj = corr_np(z_np(resid.T), cent).argmax(1)
    r_ad = corr_np(z, cent)
    ca_ad = r_ad[:, 2] - r_ad[:, 0]
    r_all = corr_np(z_np(x_all), cent)
    ca_all = r_all[:, 2] - r_all[:, 0]
    return {
        "r2_cell": float(np.median(r2)),
        "ari_adj": float(adjusted_rand_score(lab, lab_adj)),
        "stage_rho": float(stats.spearmanr(stage, ca_all)[0]),
        "neuron_rho": float(stats.spearmanr(neuron, ca_ad)[0]),
    }


def run_cohort(name, expr, meta, cls, n, rng):
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
    ad = meta.index[meta["group"] == "AD"]

    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    def score(lst):
        lst = [g for g in lst if g in expr.index and g not in genes]
        return z_np(expr.loc[lst, ad].values).mean(0)
    neuron = score(NEURON_MARKERS)
    cells = np.column_stack([neuron] + [score(v) for v in CELL_MARKERS.values()])
    stage = meta["group"].map(GROUP_ORDER).values

    X_ad = expr[ad]
    X_all = expr[meta.index]
    obs = metrics(X_ad.loc[genes].values, X_all.loc[genes].values, cent.values, cells, neuron, stage)

    # expression-matched sampling pool
    mean_expr = expr.mean(axis=1)
    decile = pd.qcut(mean_expr.rank(method="first"), 10, labels=False)
    pool = expr.index.difference(genes).difference(list(marker_genes))
    pool_by_dec = {d: pool[decile.loc[pool].values == d] for d in range(10)}
    neff_dec = decile.loc[genes].values

    null = []
    for _ in range(n):
        pick = np.empty(len(genes), dtype=object)
        for d in range(10):
            idx = np.where(neff_dec == d)[0]
            if len(idx):
                pick[idx] = rng.choice(pool_by_dec[d], size=len(idx), replace=False)
        pick = list(pick)
        null.append(metrics(X_ad.loc[pick].values, X_all.loc[pick].values, cent.values,
                            cells, neuron, stage))
    null = pd.DataFrame(null)

    # one-sided empirical p: is Neff more extreme than random in the direction we reported?
    direction = {"r2_cell": "greater", "ari_adj": "less", "stage_rho": "greater", "neuron_rho": "less"}
    out = {"cohort": name, "n_genes": int(len(genes)), "n_random_sets": n}
    for k, d in direction.items():
        more = (null[k] >= obs[k]).sum() if d == "greater" else (null[k] <= obs[k]).sum()
        out[k] = {"neff": round(obs[k], 3),
                  "random_median": round(float(null[k].median()), 3),
                  "random_q05_q95": [round(float(null[k].quantile(.05)), 3), round(float(null[k].quantile(.95)), 3)],
                  "p_empirical": round(float((more + 1) / (n + 1)), 4)}
    null["cohort"] = name
    return out, null


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    reports, nulls = [], []
    for name, e, m in cohorts(a.data):
        rep, null = run_cohort(name, e, m, cls, a.n, rng)
        reports.append(rep)
        nulls.append(null)
        print(json.dumps(rep, ensure_ascii=False), flush=True)

    (a.out / "null_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    nulls = pd.concat(nulls)
    nulls.to_csv(a.out / "null_distributions.csv", index=False)

    keys = [("r2_cell", "Variance explained by cell scores (median R²)"),
            ("ari_adj", "Agreement after cell adjustment (ARI)"),
            ("stage_rho", "Stage trend of C-minus-A score (ρ)"),
            ("neuron_rho", "C-minus-A score vs neuron score (ρ, AD)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    names = [r["cohort"] for r in reports]
    for ax, (k, title) in zip(axes, keys):
        data = [nulls.loc[nulls.cohort == c, k].values for c in names]
        ax.violinplot(data, showextrema=False)
        ax.scatter(range(1, len(names) + 1), [r[k]["neff"] for r in reports], color="red", zorder=3, label="Neff genes")
        ax.set_xticks(range(1, len(names) + 1),
                      [c.replace("GSE118553_", "118553\n").replace("_Cortex", "").replace("_", "\n") for c in names],
                      fontsize=7)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "null_summary.png", dpi=150)

    flat = pd.DataFrame([{"cohort": r["cohort"], **{f"{k}_{s}": r[k][s] for k, _ in keys
                          for s in ["neff", "random_median", "p_empirical"]}} for r in reports])
    flat.to_csv(a.out / "null_summary.csv", index=False)
    print(flat.to_string())


if __name__ == "__main__":
    main()
