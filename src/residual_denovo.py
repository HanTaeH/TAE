"""De novo structure after removing composition (follow-up, 2026-11-24): is anything subtype-like left?

Earlier (residual_structure.py) only the Neff classifier genes were examined. Here the whole
transcriptome is used, independent of any published scheme.

Per independent cohort (AD samples): genes shared by all four cohorts, expressed (top 60% of mean
expression in every cohort), marker genes removed; z-scored; five marker scores (neuron, astrocyte,
microglia, oligodendrocyte, endothelial) regressed out -> residual R. Unadjusted Z is kept for contrast.
  1 Variance: share of variance on PC1..PC5 of Z and of R.
  2 Discreteness (SigClust-style): best k-means silhouette (k = 2..4) on the top 10 PC scores against
    200 Gaussian datasets with the same PC variances. p = share of null silhouettes >= observed.
  3 Cross-cohort replication: gene loadings (correlation of each gene with the PC score) of R-PC1..5;
    for each cohort pair, the best |r| between a PC of cohort i and PCs 1..5 of cohort j. Null: the
    same with gene labels shuffled in cohort j (200 times), giving the 95th percentile.
  4 Identity of each R-PC: SEA-AD subclass sets (top 50 specific genes, 24 subclasses) and curated
    sets (immediate early, heat shock, interferon, microglial activation, reactive astrocyte, hypoxia,
    oxphos, ribosome): mean loading of the set, as z against 1,000 random sets of equal size.
  5 Clinical: |Spearman| of each R-PC score with age and sex, and Spearman with Braak (GSE125583).

Usage: python residual_denovo.py --data /mnt/project-files/d1-data/raw --ref-dir /mnt/project-files/d1-data/seaad --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from meta_reanalysis import INDEPENDENT
from null_models import z_np
from pilot_gse33000 import SEED
from residual_identity import CURATED, MARKERS, PATTERNS, cell_scores, subclass_sets
from residual_structure import residualise
from validate_cohorts import cohorts

NPC = 5


def pca(X: np.ndarray, k: int = 10):
    X = X - X.mean(1, keepdims=True)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return vt[:k].T * s[:k], var[:k]


def best_sil(S: np.ndarray) -> float:
    return max(silhouette_score(S, KMeans(k, n_init=5, random_state=0).fit_predict(S)) for k in (2, 3, 4))


def sigclust(S: np.ndarray, rng, n=200) -> dict:
    obs = best_sil(S)
    sd = S.std(0)
    nl = np.array([best_sil(rng.normal(0, sd, S.shape)) for _ in range(n)])
    return {"silhouette": round(float(obs), 3), "null_median": round(float(np.median(nl)), 3),
            "p": round(float(((nl >= obs).sum() + 1) / (n + 1)), 4)}


def loadings(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    Xs = z_np(X)
    Ss = (S - S.mean(0)) / S.std(0)
    return Xs @ Ss / X.shape[1]


def set_z(load: np.ndarray, genes: pd.Index, sets: dict, rng, n=1000) -> dict:
    pos = {g: i for i, g in enumerate(genes)}
    out = {}
    for name, gl in sets.items():
        idx = [pos[g] for g in gl if g in pos]
        if len(idx) < 5:
            continue
        obs = load[idx].mean()
        nl = np.array([load[rng.choice(len(load), len(idx), replace=False)].mean() for _ in range(n)])
        out[name] = round(float((obs - nl.mean()) / nl.std()), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 404)
    sub_ref = pd.read_csv(a.ref_dir / "seaad_mtg_subclass_reference.csv.gz", index_col=0)
    sub_ref = sub_ref[~sub_ref.index.duplicated()]
    data = {}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e.loc[e.index.notna() & ~e.index.duplicated()].dropna()
        ad = m.index[m["group"] == "AD"]
        data[name] = (e[ad], m.loc[ad])
    shared = None
    for e, _ in data.values():
        mu = e.mean(axis=1)
        ok = mu.index[mu.rank(pct=True) > 0.4]
        shared = ok if shared is None else shared.intersection(ok)
    shared = shared.difference(list(MARKERS)).sort_values()
    sets = {k: [g for g in shared if re.match(p, g)] for k, p in PATTERNS.items()}
    sets.update(CURATED)
    sets.update(subclass_sets(sub_ref, MARKERS))
    out = {"n_genes": int(len(shared)), "cohorts": {}, "replication": {}}
    L = {}
    for name, (e, m) in data.items():
        cells = cell_scores(e, e.columns, set())
        Z = z_np(e.loc[shared].values)
        R = residualise(Z, cells)
        row = {"n_AD": int(e.shape[1])}
        for lab, X in (("unadjusted", Z), ("adjusted", R)):
            S, var = pca(X)
            row[lab] = {"var_pc1_5": [round(float(v), 3) for v in var[:NPC]],
                        "pc1_abs_rho_neuron": round(float(abs(stats.spearmanr(S[:, 0], cells[:, 0])[0])), 3),
                        "sigclust": sigclust(S, rng)}
            if lab == "adjusted":
                ld = loadings(X, S[:, :NPC])
                L[name] = ld
                ident, clin = {}, {}
                for k in range(NPC):
                    z = set_z(ld[:, k], shared, sets, rng)
                    top = sorted(z.items(), key=lambda t: -abs(t[1]))[:4]
                    ident[f"PC{k + 1}"] = dict(top)
                    c = {}
                    if "age" in m:
                        age = pd.to_numeric(m["age"], errors="coerce").values
                        ok = ~np.isnan(age)
                        if ok.sum() > 10:
                            c["age"] = round(float(abs(stats.spearmanr(S[ok, k], age[ok])[0])), 3)
                    if "sex" in m and m["sex"].nunique() == 2:
                        c["sex"] = round(float(abs(stats.spearmanr(S[:, k], (m["sex"] == m["sex"].iloc[0]).astype(float))[0])), 3)
                    if "braak" in m:
                        br = pd.to_numeric(m["braak"], errors="coerce").values
                        ok = ~np.isnan(br)
                        rb = stats.spearmanr(S[ok, k], br[ok])
                        c["braak_rho"], c["braak_p"] = round(float(rb[0]), 3), round(float(rb[1]), 4)
                    clin[f"PC{k + 1}"] = c
                row["adjusted"]["identity_top_sets_z"] = ident
                row["adjusted"]["clinical_abs_rho"] = clin
        out["cohorts"][name] = row
        print(name, json.dumps(row), flush=True)
    names = list(L)
    for i in names:
        for j in names:
            if i >= j:
                continue
            C = np.abs(np.corrcoef(L[i].T, L[j].T)[:NPC, NPC:])
            nl = []
            for _ in range(200):
                p = rng.permutation(len(shared))
                nl.append(np.abs(np.corrcoef(L[i].T, L[j][p].T)[:NPC, NPC:]).max(1))
            out["replication"][f"{i}|{j}"] = {"best_abs_r_per_PC_of_first": [round(float(v), 3) for v in C.max(1)],
                                              "matched_PC_of_second": [int(v) + 1 for v in C.argmax(1)],
                                              "null_p95": round(float(np.percentile(np.array(nl), 95)), 3)}
            print(i, j, json.dumps(out["replication"][f"{i}|{j}"]), flush=True)
    (a.out / "residual_denovo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
