"""De novo whole-transcriptome subtyping in the public cohorts (v13): does the field's usual recipe recover the neuron axis?

Most de novo AD subtyping studies cluster AD samples on highly variable genes (NMF, k-means or hierarchical
clustering, k = 2-4). Here that recipe is applied to the AD samples of each independent cohort, without any
published gene list, and the resulting subtypes are compared with the five marker scores.

Per cohort: expressed genes (top 60% of mean expression), the 1,500 genes with the largest median absolute
deviation, marker genes removed.
  Methods: (a) consensus NMF (genes shifted to a minimum of 0; 50 resamples of 80% of samples; consensus
           matrix clustered by average linkage), (b) k-means on the first 10 PCs of the z-scored genes,
           (c) PC1 of the z-scored genes as a continuous axis. k = 2, 3, 4.
  Per labelling: eta^2 of the neuron score across subtypes; cross-validated balanced accuracy of predicting
  the labels from the five marker scores (multinomial logistic regression, 5-fold, repeated 5 times),
  against 100 sets of five random-gene scores matched in size and expression decile, drawn from genes
  outside the 1,500 clustered genes, (p = share of random
  accuracies >= observed); share of the 1,500 genes' between-subtype variance explained by the five scores.
  PC1: |Spearman| with the neuron score; R^2 on the five scores.
  Adjusted: the same after regressing the five scores out of every gene (residual subtypes), with ARI
  between unadjusted and residual labels and cophenetic correlation of both consensus matrices.

Usage: python denovo_clustering_v13.py --data /mnt/project-files/d1-data/raw --out ../results/denovo_v13 [--cohort NAME]
(run_all.sh runs the four cohorts in parallel and merges them with --merge)
"""
from __future__ import annotations

import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from meta_reanalysis import INDEPENDENT
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts


def consensus_nmf(X: np.ndarray, k: int, rng, n_rep: int = 50):
    """X genes x samples, non-negative. Returns labels and cophenetic correlation."""
    n = X.shape[1]
    M, I = np.zeros((n, n)), np.zeros((n, n))
    for r in range(n_rep):
        idx = rng.choice(n, int(0.8 * n), replace=False)
        W = NMF(k, init="nndsvda", max_iter=400, random_state=int(rng.integers(1e9))).fit(X[:, idx].T)
        H = W.transform(X[:, idx].T)
        lab = H.argmax(1)
        same = lab[:, None] == lab[None, :]
        M[np.ix_(idx, idx)] += same
        I[np.ix_(idx, idx)] += 1
    C = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
    np.fill_diagonal(C, 1)
    D = squareform(1 - C, checks=False)
    L = linkage(D, "average")
    return fcluster(L, k, "maxclust") - 1, float(cophenet(L, D)[0])


def eta2(y: np.ndarray, lab: np.ndarray) -> float:
    g = [y[lab == u] for u in np.unique(lab)]
    ss_b = sum(len(v) * (v.mean() - y.mean()) ** 2 for v in g)
    return float(ss_b / ((y - y.mean()) ** 2).sum())


def cv_acc(F: np.ndarray, lab: np.ndarray, seed: int) -> float:
    if np.bincount(lab).min() < 5:
        return np.nan
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, stats.zscore(F, axis=0), lab, cv=cv, scoring="balanced_accuracy").mean())


def between_share(Z: np.ndarray, lab: np.ndarray, F: np.ndarray) -> float:
    """Share of the between-subtype variance of the genes that the five scores explain."""
    D = np.column_stack([np.ones(len(lab)), F])
    fit = D @ np.linalg.lstsq(D, Z.T, rcond=None)[0]
    def between(A):
        return sum((lab == u).sum() * ((A[lab == u].mean(0) - A.mean(0)) ** 2).sum() for u in np.unique(lab))
    return float(between(fit) / between(Z.T))


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--cohort", default=None, help="run one cohort only (outputs suffixed with its name)")
    ap.add_argument("--merge", action="store_true", help="merge the per-cohort outputs and exit")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        cs = [c for c in INDEPENDENT if (a.out / f"denovo_clustering_v13_{c}.json").exists()]
        js = [json.loads((a.out / f"denovo_clustering_v13_{c}.json").read_text()) for c in cs]
        lab = [r for j in js for r in j["labellings"]]
        pcs = [r for j in js for r in j["pcs"]]
        pd.DataFrame(lab).to_csv(a.out / "denovo_clustering_v13.csv", index=False)
        pd.DataFrame(pcs).to_csv(a.out / "denovo_pcs_v13.csv", index=False)
        (a.out / "denovo_clustering_v13.json").write_text(json.dumps({"labellings": lab, "pcs": pcs}, indent=2))
        return
    rng = np.random.default_rng(SEED + 1307 + (sum(map(ord, a.cohort)) if a.cohort else 0))
    allmark = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    rows, pcrows = [], []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT or (a.cohort and name != a.cohort):
            continue
        e = e[m.index].loc[:, (m["group"] == "AD").values]
        e = e[e.notna().mean(1) > 0.9]
        e = e.apply(lambda r: r.fillna(r.mean()), axis=1)
        mu = e.mean(1)
        e = e[mu >= mu.quantile(0.4)]
        Z = zscore_rows(e)
        cells = {"neuron": Z.loc[[g for g in NEURON_MARKERS if g in Z.index]].mean().values}
        for c, v in CELL_MARKERS.items():
            cells[c] = Z.loc[[g for g in v if g in Z.index]].mean().values
        F = np.column_stack(list(cells.values()))
        neuron = cells["neuron"]
        sizes = [len([g for g in NEURON_MARKERS if g in Z.index])] + [len([g for g in v if g in Z.index]) for v in CELL_MARKERS.values()]
        mad = (e.sub(e.median(1), axis=0)).abs().median(1)
        top = mad.drop(list(allmark & set(mad.index)), errors="ignore").nlargest(1500).index
        dec = pd.qcut(mu.loc[e.index].rank(method="first"), 10, labels=False)
        pool = e.index.difference(list(allmark) + list(top))  # random genes never include the clustered genes
        mark_dec = [dec.loc[[g for g in (NEURON_MARKERS if i == 0 else list(CELL_MARKERS.values())[i - 1]) if g in Z.index]].values
                    for i in range(5)]
        null_F = []
        for _ in range(a.n_null):
            cols = []
            for d in mark_dec:
                pick = [rng.choice(pool[dec.loc[pool].values == x]) for x in d]
                cols.append(Z.loc[pick].mean().values)
            null_F.append(np.column_stack(cols))
        D = np.column_stack([np.ones(Z.shape[1]), F])
        R = Z.loc[top].values - (D @ np.linalg.lstsq(D, Z.loc[top].values.T, rcond=None)[0]).T
        for variant, Xz in [("unadjusted", Z.loc[top].values), ("cell_adjusted", R)]:
            Xz = (Xz - Xz.mean(1, keepdims=True)) / Xz.std(1, keepdims=True)
            u, s, vt = np.linalg.svd(Xz, full_matrices=False)
            pc = vt[:10].T * s[:10]
            for j in range(3):
                pcrows.append({"cohort": name, "variant": variant, "pc": j + 1,
                               "var_share": round(float(s[j] ** 2 / (s ** 2).sum()), 3),
                               "abs_rho_neuron": round(abs(float(stats.spearmanr(pc[:, j], neuron)[0])), 3),
                               "r2_on_cells": round(float(1 - np.var(pc[:, j] - D @ np.linalg.lstsq(D, pc[:, j], rcond=None)[0]) / np.var(pc[:, j])), 3)})
            Xn = Xz - Xz.min(1, keepdims=True)
            for k in (2, 3, 4):
                labs = {}
                labs["nmf"], coph = consensus_nmf(Xn, k, rng)
                labs["kmeans_pc"] = KMeans(k, n_init=20, random_state=0).fit_predict(pc)
                for meth, lab in labs.items():
                    obs = cv_acc(F, lab, 0)
                    nul = np.array([cv_acc(nf, lab, 0) for nf in null_F[:a.n_null]]) if not np.isnan(obs) else np.array([np.nan])
                    row = {"cohort": name, "n_AD": int(Z.shape[1]), "variant": variant, "method": meth, "k": k,
                           "sizes": np.bincount(lab).tolist(),
                           "cophenetic": round(coph, 3) if meth == "nmf" else None,
                           "eta2_neuron": round(eta2(neuron, lab), 3),
                           "cv_bal_acc_cells": round(obs, 3) if not np.isnan(obs) else None,
                           "chance": round(1 / k, 3),
                           "cv_bal_acc_random_median": round(float(np.nanmedian(nul)), 3) if not np.isnan(obs) else None,
                           "p_random_ge": round(float(((nul >= obs).sum() + 1) / (len(nul) + 1)), 4) if not np.isnan(obs) else None,
                           "between_var_share_cells": round(between_share(Xz, lab, F), 3)}
                    if variant == "cell_adjusted":
                        ref = next(r for r in rows if r["cohort"] == name and r["variant"] == "unadjusted"
                                   and r["method"] == meth and r["k"] == k)
                        row["ari_vs_unadjusted"] = round(float(adjusted_rand_score(ref["_lab"], lab)), 3)
                    row["_lab"] = lab
                    rows.append(row)
                    print(json.dumps({x: y for x, y in row.items() if x != "_lab"}), flush=True)
    out = [{x: y for x, y in r.items() if x != "_lab"} for r in rows]
    suf = f"_{a.cohort}" if a.cohort else ""
    pd.DataFrame(out).to_csv(a.out / f"denovo_clustering_v13{suf}.csv", index=False)
    pd.DataFrame(pcrows).to_csv(a.out / f"denovo_pcs_v13{suf}.csv", index=False)
    (a.out / f"denovo_clustering_v13{suf}.json").write_text(json.dumps({"labellings": out, "pcs": pcrows}, indent=2))


if __name__ == "__main__":
    main()
