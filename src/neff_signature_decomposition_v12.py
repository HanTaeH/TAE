"""Split Neff et al.'s genome-wide subtype signatures into a composition part and a residual (v12, review round 10, Reviewer 2).

Data file S1 gives, for 23,199 genes, the average log fold change of each MSBB-AD subtype (A, B1, B2, C1,
C2) after CDR normalisation. The C-A signature is mean(C1, C2) - A; B1-B2 and C1-C2 are also used.
  1. Gene level: each signature is predicted across genes from the gene's SEA-AD MTG profile (neurotypical
     donors): log2(mean + 1) in the nine cell classes, the share of expression in each class and the
     largest share. Two models: linear (OLS) and gradient-boosted trees, 5-fold cross-fitted so that every
     gene's prediction comes from a model that did not see it. R^2 = share of the signature predictable
     from cell-type expression; the cross-fitted boosted residual is the composition-independent part.
  2. Sample level, AD samples of each independent cohort (genes z-scored within AD samples): sample score
     = Pearson correlation across genes between the sample's profile and the full, fitted or residual
     signature. Reported: Spearman of each score with the neuron score; R^2 of the residual score on the
     neuron and four glial scores (ranks).
  3. Coherence: does the residual signature describe co-varying genes in the cohort? Variance of the
     residual score against 200 signatures permuted within expression deciles (upper-tail p).
  4. Disease relevance: age/sex partial Spearman of the residual score with diagnosis (all samples) and,
     in GSE125583, with Braak stage within AD, also adjusted for the five cell scores; two-sided p against
     the permuted signatures.
  5. Cross-cohort agreement: gene loadings on the residual score (Spearman of each gene with it in AD
     samples, after removing the five cell scores from the genes) compared between cohorts.

Usage: python neff_signature_decomposition_v12.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out ../results/neff_signature_v12
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from meta_reanalysis import INDEPENDENT, rho
from null_models import z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts


def ols_resid(y, X):
    D = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    return y - D @ b, D @ b


def features(ref):
    L = np.log2(ref + 1)
    F = ref.div(ref.sum(axis=1), axis=0)
    return np.column_stack([L.values, F.values, F.max(axis=1).values, L.mean(axis=1).values])


def crossfit(y, X, rng, k=5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    fold = rng.integers(0, k, len(y))
    f = np.empty(len(y))
    for i in range(k):
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)
        mdl.fit(X[fold != i], y[fold != i])
        f[fold == i] = mdl.predict(X[fold == i])
    return y - f, f


def proj(Z, sig):
    """Correlation across genes of each sample (column of Z) with sig."""
    zc = (Z - Z.mean(0)) / Z.std(0)
    s = (sig - sig.mean()) / sig.std()
    return zc.T @ s / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1304)
    fc = neff_subtypes.load_logfc(a.data / "neff2021")
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ref = ref[ref.max(axis=1) > 0.05]
    g = fc.index.intersection(ref.index)
    X = features(ref.loc[g])
    sigs = {"C_A": (fc.loc[g, ["C1", "C2"]].mean(axis=1) - fc.loc[g, "A"]),
            "C1_C2": fc.loc[g, "C1"] - fc.loc[g, "C2"], "B1_B2": fc.loc[g, "B1"] - fc.loc[g, "B2"],
            **{f"{k}_vs_control": fc.loc[g, k] for k in ["A", "B1", "B2", "C1", "C2"]}}
    report = {"n_genes": len(g), "gene_level": {}}
    parts = {}
    for k, s in sigs.items():
        rl, _ = ols_resid(s.values, X)
        r, f = crossfit(s.values, X, rng)
        report["gene_level"][k] = {"r2_linear": round(float(1 - rl.var() / s.values.var()), 3),
                                   "r2_boosted_crossfit": round(float(1 - r.var() / s.values.var()), 3)}
        parts[k] = {"full": s, "fitted": pd.Series(f, index=g), "resid": pd.Series(r, index=g)}
    top = parts["C_A"]["resid"].sort_values()
    report["gene_level"]["C_A"]["resid_top_down"] = list(top.index[:25])
    report["gene_level"]["C_A"]["resid_top_up"] = list(top.index[::-1][:25])
    print(json.dumps(report["gene_level"]), flush=True)
    loadings = {}
    report["cohorts"] = {}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        gg = g.intersection(e.index)
        ad = (m["group"] == "AD").values
        Zad = zscore_rows(e.loc[gg, ad]).values
        Zall = zscore_rows(e.loc[gg]).values
        def cell(Z, idx):
            out = {}
            for k, v in {"neuron": NEURON_MARKERS, **CELL_MARKERS}.items():
                v = [x for x in v if x in e.index and x not in cls.index]
                out[k] = zscore_rows(e.loc[v, idx]).mean().values
            return out
        cad = cell(Zad, e.columns[ad])
        call = cell(Zall, e.columns)
        Cad = np.column_stack([stats.rankdata(v) for v in cad.values()])
        res = {"n_genes": len(gg), "n_ad": int(ad.sum())}
        sc = {p: proj(Zad, parts["C_A"][p].loc[gg].values) for p in ["full", "fitted", "resid"]}
        res["rho_with_neuron"] = {p: round(float(stats.spearmanr(v, cad["neuron"])[0]), 3) for p, v in sc.items()}
        rr, _ = ols_resid(stats.rankdata(sc["resid"]), Cad)
        res["resid_score_r2_on_5_cell_scores"] = round(float(1 - rr.var() / stats.rankdata(sc["resid"]).var()), 3)
        # permuted residual signatures within expression decile
        dec = pd.qcut(e.loc[gg].mean(axis=1).rank(method="first"), 10, labels=False).values
        rv = parts["C_A"]["resid"].loc[gg].values
        perms = []
        for _ in range(a.n):
            p = rv.copy()
            for d in range(10):
                w = np.where(dec == d)[0]
                p[w] = rng.permutation(rv[w])
            perms.append(p)
        var_obs = sc["resid"].var()
        var_null = np.array([proj(Zad, p).var() for p in perms])
        res["coherence_resid"] = {"var_ratio_to_null_median": round(float(var_obs / np.median(var_null)), 2),
                                  "p_upper": round(float(((var_null >= var_obs).sum() + 1) / (a.n + 1)), 4)}
        vf = sc["full"].var()
        res["coherence_full_var_ratio"] = round(float(vf / np.median(var_null)), 2)
        # disease relevance
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov = np.column_stack([age, sex])
        covc = np.column_stack([age, sex, *call.values()])
        sall = proj(Zall, rv)
        nl = np.array([[rho(proj(Zall, p), stage, cov), rho(proj(Zall, p), stage, covc)] for p in perms])
        tests = {}
        for j, (lab, cv) in enumerate({"dx_agesex": cov, "dx_agesex_cells": covc}.items()):
            o = rho(sall, stage, cv)
            tests[lab] = {"rho": round(o, 3), "null_median": round(float(np.median(nl[:, j])), 3),
                          "p_two_sided": round(float(((np.abs(nl[:, j]) >= abs(o)).sum() + 1) / (a.n + 1)), 4)}
        if "braak" in m.columns:
            br = pd.to_numeric(m["braak"], errors="coerce").values
            k = ad & ~np.isnan(br)
            if k.sum() > 30:
                ka = k[ad]
                cv = np.column_stack([age[ad][ka], sex[ad][ka]])
                cvc = np.column_stack([cv, Cad[ka]])
                for lab, c2 in {"braak_AD_agesex": cv, "braak_AD_agesex_cells": cvc}.items():
                    o = rho(sc["resid"][ka], br[k], c2)
                    nb = np.array([rho(proj(Zad, p)[ka], br[k], c2) for p in perms])
                    tests[lab] = {"n": int(k.sum()), "rho": round(o, 3), "null_median": round(float(np.median(nb)), 3),
                                  "p_two_sided": round(float(((np.abs(nb) >= abs(o)).sum() + 1) / (a.n + 1)), 4)}
        res["disease_tests"] = tests
        # gene loadings on the residual score after removing cell scores from genes
        Zr = np.array([ols_resid(stats.rankdata(z), Cad)[0] for z in Zad])
        sr = ols_resid(stats.rankdata(sc["resid"]), Cad)[0]
        loadings[name] = pd.Series([np.corrcoef(z, sr)[0, 1] for z in Zr], index=gg)
        report["cohorts"][name] = res
        print(name, json.dumps(res), flush=True)
    pairs = {}
    for x, y in itertools.combinations(loadings, 2):
        j = loadings[x].index.intersection(loadings[y].index)
        rv = parts["C_A"]["resid"].loc[j]
        # loadings partly mirror the signature by construction; report raw and after removing the signature
        lx = ols_resid(loadings[x][j].values, rv.values[:, None])[0]
        ly = ols_resid(loadings[y][j].values, rv.values[:, None])[0]
        pairs[f"{x}|{y}"] = {"n_genes": len(j), "rho_loadings": round(float(stats.spearmanr(loadings[x][j], loadings[y][j])[0]), 3),
                             "rho_loadings_signature_removed": round(float(stats.spearmanr(lx, ly)[0]), 3),
                             "rho_loading_x_vs_signature": round(float(stats.spearmanr(loadings[x][j], rv)[0]), 3)}
    report["loading_agreement"] = pairs
    (a.out / "neff_signature_decomposition_v12.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
