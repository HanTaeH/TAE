"""More published subtype schemes (follow-up, 2026-11-24): do they share the neuronal axis?

Three schemes that cluster AD samples (k = 2) on a small published gene set:
  Lai 2022 Front Immunol 13:1046410  - 5 immune signature genes; subtype 1 immune-active, 2 metabolic
  Ma 2023 Front Aging Neurosci 15:1161068 - 15 m7G-related differentially expressed genes (discovery GSE33000)
  Wang 2024 Front Immunol 15:1462003 - 24 AD-related PANoptosis genes; early synaptic vs immune-metabolic
    (list taken from a page summary; to be checked against the paper before use in the manuscript)
For each independent cohort (AD samples): genes z-scored; subtype axis = PC1 of the scheme genes
(sign fixed so the axis rises with the scheme's first gene mean); labels = k-means (k = 2) and, as a check,
consensus hierarchical clustering (100 resamples of 80% samples, 1 - Pearson, average linkage), ARI between them.
Metrics: |rho| of the axis with the neuron marker score (scheme genes removed from the markers); R^2 of the
axis on the five marker scores; |AUC - 0.5| + 0.5 of the neuron score for the k-means label.
Nulls (300 sets each): random genes matched on expression decile, and on decile x SEA-AD specificity class;
p = share of random |rho| >= observed.

Usage: python other_schemes_clustered.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from classifier_specificity import NEURONAL
from meta_reanalysis import INDEPENDENT
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts

SCHEMES = {
    "Lai2022_immune": ["CXCR4", "PPP3R1", "HSP90AB1", "CXCL10", "S100A12"],
    "Ma2023_m7G": ["NUDT3", "CYFIP1", "NCBP1", "IFIT5", "METTL1", "DCPS", "NUDT10", "NUDT11", "EIF4E",
                   "EIF4E2", "EIF4E3", "LARP1", "EIF4G3", "LSM1", "NCBP2L"],
    "Wang2024_PANoptosis": ["ANXA1", "BTG2", "CAPN2", "CASP4", "CASP7", "CD44", "GSN", "HGF", "IL18", "IRAK3",
                            "LY96", "NFKBIA", "PDGFRB", "PRKX", "TAP1", "TGFB2", "TNFRSF10A", "TNFRSF10B",
                            "TNFRSF1A", "CCNA1", "ENO2", "MAGED1", "VDAC1", "YWHAZ"],
}


def pc1(Z: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(Z - Z.mean(1, keepdims=True), full_matrices=False)
    p = vt[0] * s[0]
    return p if np.corrcoef(p, Z[0])[0, 1] >= 0 else -p


def consensus_labels(Z: np.ndarray, rng, n_rep=100) -> np.ndarray:
    n = Z.shape[1]
    M, I = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(n_rep):
        idx = rng.choice(n, int(0.8 * n), replace=False)
        d = 1 - np.corrcoef(Z[:, idx].T)
        lab = fcluster(linkage(squareform(d, checks=False), "average"), 2, "maxclust")
        same = lab[:, None] == lab[None, :]
        M[np.ix_(idx, idx)] += same
        I[np.ix_(idx, idx)] += 1
    C = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
    np.fill_diagonal(C, 1)
    return fcluster(linkage(squareform(1 - C, checks=False), "average"), 2, "maxclust")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 101)
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    allmark = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index].loc[:, (m["group"] == "AD").values]
        Z = zscore_rows(e)
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        sp = spec.reindex(e.index).fillna(3).astype(int)
        for sch, gl in SCHEMES.items():
            genes = [g for g in gl if g in Z.index]
            mk = [g for g in NEURON_MARKERS if g in Z.index and g not in genes]
            neuron = Z.loc[mk].mean().values
            cells = np.column_stack([neuron] + [Z.loc[[g for g in v if g in Z.index and g not in genes]].mean().values
                                                for v in CELL_MARKERS.values()])
            X = Z.loc[genes].values
            ax = pc1(X)
            km = KMeans(2, n_init=10, random_state=0).fit_predict(X.T)
            cons = consensus_labels(X, rng)
            D = np.column_stack([np.ones(len(ax)), cells])
            fit = D @ np.linalg.lstsq(D, ax, rcond=None)[0]
            auc = roc_auc_score(km, neuron)
            obs = abs(stats.spearmanr(ax, neuron)[0])
            pool = e.index.difference(genes).difference(list(allmark))
            res = {}
            for kind, strat in [("expr", dec.astype(str)), ("spec", dec.astype(str) + "_" + sp.astype(str))]:
                by = {k: pool[strat.loc[pool].values == k] for k in strat.loc[genes].unique()}
                nl = []
                for _ in range(a.n):
                    pick = [rng.choice(by[k]) if len(by[k]) else rng.choice(pool) for k in strat.loc[genes].values]
                    nl.append(abs(stats.spearmanr(pc1(Z.loc[pick].values), neuron)[0]))
                nl = np.array(nl)
                res[kind] = {"random_median": round(float(np.median(nl)), 3),
                             "p": round(float(((nl >= obs).sum() + 1) / (a.n + 1)), 4)}
            row = {"scheme": sch, "cohort": name, "n_AD": int(e.shape[1]), "genes_found": f"{len(genes)}/{len(gl)}",
                   "spec_mix_glia_nonspec_neuron_absent": [int((sp.loc[genes] == k).sum()) for k in (0, 1, 2, 3)],
                   "abs_rho_axis_neuron": round(obs, 3), "r2_axis_on_cells": round(float(np.corrcoef(fit, ax)[0, 1] ** 2), 3),
                   "auc_neuron_for_label": round(float(max(auc, 1 - auc)), 3),
                   "ari_kmeans_vs_consensus": round(float(adjusted_rand_score(km, cons)), 3),
                   "null_expr": res["expr"], "null_spec": res["spec"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (a.out / "other_schemes_clustered.json").write_text(json.dumps(rows, indent=2))
    pd.json_normalize(rows).to_csv(a.out / "other_schemes_clustered.csv", index=False)


if __name__ == "__main__":
    main()
