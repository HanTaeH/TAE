"""Residual structure after cell-composition adjustment: what is left of the Neff subtypes?

AD samples only. Z = gene-wise z-scores of the measured Neff classifier genes. Five marker scores
(neuron, astrocyte, microglia, oligodendrocyte, endothelial; genes outside the classifier) are
regressed out of Z to give the residual R.

1. Which subtype dimension is composition-driven?
   ca_axis = r(C) - r(A),  b_axis = r(B) - (r(A) + r(C)) / 2  (unadjusted projection);
   R² of each axis on the five cell scores.
2. Residual subtype signal beyond chance.
   signal = mean over samples of the best correlation to a Neff class centroid, computed on Z
   (unadjusted) and on R (adjusted). Compared with 500 random gene sets matched on mean expression
   decile x standard-deviation tercile (variance-matched null), each random gene inheriting the
   centroid row of the Neff gene it replaces. One-sided empirical p (Neff greater).
   2b. Second null keeps the Neff genes (and their co-expression) but permutes centroid rows,
   so it asks whether the residual follows the Neff pattern specifically.
3. Discrete cluster structure (SigClust-style).
   Best k-means silhouette (k = 2..4) on the top 10 principal components, against 200 Gaussian
   datasets with the same covariance. Done for Z and for R.
4. Adjusted classes vs age and sex (Kruskal-Wallis, chi-square).

Usage: python residual_structure.py --data /mnt/project-files/d1-data/raw --out ../results/residual --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import cohorts

N_PC = 10
N_GAUSS = 200


def residualise(z: np.ndarray, cells: np.ndarray) -> np.ndarray:
    D = np.column_stack([np.ones(cells.shape[0]), cells])
    beta, *_ = np.linalg.lstsq(D, z.T, rcond=None)
    return (z.T - D @ beta).T


def r2(y: np.ndarray, cells: np.ndarray) -> float:
    D = np.column_stack([np.ones(len(y)), cells])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ beta
    return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())


def signal(z: np.ndarray, cent: np.ndarray, cells: np.ndarray) -> tuple[float, float]:
    raw = corr_np(z, cent).max(1).mean()
    adj = corr_np(z_np(residualise(z, cells)), cent).max(1).mean()
    return float(raw), float(adj)


def best_silhouette(x: np.ndarray, seed: int) -> float:
    best = -1.0
    for k in (2, 3, 4):
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(x)
        best = max(best, silhouette_score(x, lab))
    return float(best)


def clusterability(z: np.ndarray, rng: np.random.Generator) -> dict:
    """z genes x samples -> PCA on samples, silhouette vs Gaussian null."""
    x = z.T - z.T.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    pcs = u[:, :N_PC] * s[:N_PC]
    obs = best_silhouette(pcs, SEED)
    sd = pcs.std(0)
    null = [best_silhouette(rng.normal(size=pcs.shape) * sd, int(rng.integers(1 << 31)))
            for _ in range(N_GAUSS)]
    return {"silhouette": round(obs, 3), "null_median": round(float(np.median(null)), 3),
            "p": round(float((np.sum(np.array(null) >= obs) + 1) / (N_GAUSS + 1)), 4),
            "pc1_var_frac": round(float(s[0] ** 2 / (s ** 2).sum()), 3)}


def run_cohort(name, expr, meta, cls, n, rng):
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    ad = meta.index[meta["group"] == "AD"]
    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}

    def score(lst):
        lst = [g for g in lst if g in expr.index and g not in genes]
        return z_np(expr.loc[lst, ad].values).mean(0)
    cells = np.column_stack([score(NEURON_MARKERS)] + [score(v) for v in CELL_MARKERS.values()])

    X = expr[ad]
    z = z_np(X.loc[genes].values)
    rep: dict = {"cohort": name, "n_AD": int(len(ad)), "n_genes": int(len(genes))}

    # 1. which axis is composition-driven
    r = corr_np(z, cent)
    ca_axis = r[:, 2] - r[:, 0]
    b_axis = r[:, 1] - (r[:, 0] + r[:, 2]) / 2
    rep["r2_cells_ca_axis"] = round(r2(ca_axis, cells), 3)
    rep["r2_cells_b_axis"] = round(r2(b_axis, cells), 3)

    # 2. residual signal vs variance-matched random genes
    raw, adj = signal(z, cent, cells)
    mean_e, sd_e = X.mean(axis=1), X.std(axis=1)
    dec = pd.qcut(mean_e.rank(method="first"), 10, labels=False)
    terc = sd_e.groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    stratum = dec * 3 + terc
    pool = X.index.difference(genes).difference(list(marker_genes))
    pool_by = {s: pool[stratum.loc[pool].values == s] for s in range(30)}
    neff_s = stratum.loc[genes].values
    null = []
    for _ in range(n):
        pick = np.empty(len(genes), dtype=object)
        for s in np.unique(neff_s):
            idx = np.where(neff_s == s)[0]
            pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=False)
        null.append(signal(z_np(X.loc[list(pick)].values), cent, cells))
    null = np.array(null)
    rep["signal_raw"] = {"neff": round(raw, 3), "random_median": round(float(np.median(null[:, 0])), 3),
                         "p": round(float((np.sum(null[:, 0] >= raw) + 1) / (n + 1)), 4)}
    rep["signal_adj"] = {"neff": round(adj, 3), "random_median": round(float(np.median(null[:, 1])), 3),
                         "p": round(float((np.sum(null[:, 1] >= adj) + 1) / (n + 1)), 4)}
    excess_raw = raw - np.median(null[:, 0])
    excess_adj = adj - np.median(null[:, 1])
    rep["excess_retained_frac"] = round(float(excess_adj / excess_raw), 3) if excess_raw > 0 else None

    # 2b. same Neff genes, centroid rows permuted: is the residual aligned with the Neff pattern
    #     itself, or only with any pattern on these co-expressed genes?
    zr = z_np(residualise(z, cells))
    perm = np.array([corr_np(zr, cent[rng.permutation(len(cent))]).max(1).mean() for _ in range(n)])
    rep["signal_adj_centroid_perm"] = {"neff": round(adj, 3), "perm_median": round(float(np.median(perm)), 3),
                                       "p": round(float((np.sum(perm >= adj) + 1) / (n + 1)), 4)}
    perm_raw = np.array([corr_np(z, cent[rng.permutation(len(cent))]).max(1).mean() for _ in range(n)])
    rep["signal_raw_centroid_perm"] = {"neff": round(raw, 3), "perm_median": round(float(np.median(perm_raw)), 3),
                                       "p": round(float((np.sum(perm_raw >= raw) + 1) / (n + 1)), 4)}

    # 3. discrete clusters?
    R = residualise(z, cells)
    rep["cluster_raw"] = clusterability(z, rng)
    rep["cluster_adj"] = clusterability(z_np(R), rng)

    # 4. adjusted classes vs age / sex
    lab_adj = np.array(list("ABC"))[corr_np(z_np(R), cent).argmax(1)]
    m = meta.loc[ad]
    groups = [m["age"].values[lab_adj == k] for k in "ABC" if (lab_adj == k).sum() > 2]
    rep["adj_class_frac"] = {k: round(float((lab_adj == k).mean()), 3) for k in "ABC"}
    rep["adj_age_kruskal_p"] = round(float(stats.kruskal(*groups).pvalue), 4) if len(groups) > 1 else None
    tab = pd.crosstab(lab_adj, m["sex"].values)
    rep["adj_sex_chi2_p"] = round(float(stats.chi2_contingency(tab)[1]), 4) if tab.shape[1] > 1 else None
    rep["adj_female_frac"] = {k: round(float((m["sex"].values[lab_adj == k] == "female").mean()), 3)
                              for k in "ABC" if (lab_adj == k).any()}
    return rep, pd.DataFrame(null, columns=["signal_raw", "signal_adj"]).assign(cohort=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    reps, nulls = [], []
    for name, e, m in cohorts(a.data):
        rep, null = run_cohort(name, e, m, cls, a.n, rng)
        reps.append(rep)
        nulls.append(null)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    (a.out / "residual_report.json").write_text(json.dumps(reps, indent=2, ensure_ascii=False))
    pd.concat(nulls).to_csv(a.out / "residual_null.csv", index=False)
    flat = pd.DataFrame([{
        "cohort": r["cohort"], "r2_ca_axis": r["r2_cells_ca_axis"], "r2_b_axis": r["r2_cells_b_axis"],
        "sig_raw": r["signal_raw"]["neff"], "sig_raw_rand": r["signal_raw"]["random_median"], "sig_raw_p": r["signal_raw"]["p"],
        "sig_adj": r["signal_adj"]["neff"], "sig_adj_rand": r["signal_adj"]["random_median"], "sig_adj_p": r["signal_adj"]["p"],
        "excess_retained": r["excess_retained_frac"],
        "sig_raw_perm": r["signal_raw_centroid_perm"]["perm_median"], "sig_raw_perm_p": r["signal_raw_centroid_perm"]["p"],
        "sig_adj_perm": r["signal_adj_centroid_perm"]["perm_median"], "sig_adj_perm_p": r["signal_adj_centroid_perm"]["p"],
        "sil_raw": r["cluster_raw"]["silhouette"], "sil_raw_p": r["cluster_raw"]["p"],
        "sil_adj": r["cluster_adj"]["silhouette"], "sil_adj_p": r["cluster_adj"]["p"],
        "pc1_raw": r["cluster_raw"]["pc1_var_frac"], "pc1_adj": r["cluster_adj"]["pc1_var_frac"],
        "adj_age_p": r["adj_age_kruskal_p"], "adj_sex_p": r["adj_sex_chi2_p"],
    } for r in reps])
    flat.to_csv(a.out / "residual_summary.csv", index=False)
    print(flat.to_string())


if __name__ == "__main__":
    main()
