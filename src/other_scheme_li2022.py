"""RQ2 (supplementary): a second, independently derived subtype scheme.

Li H, Wei M, Ye T, Liu Y, Qi D, Cheng X. Identification of the molecular subgroups in Alzheimer's
disease by transcriptomic data. Front Neurol 2022;13. doi:10.3389/fneur.2022.901179
Consensus clustering of 743 AD samples (GSE1297, GSE29378, GSE84422) into subgroups I, II, III;
Table 1 lists the top-10 hub genes of each subgroup (GSEA rank metric), used here as the signature.

For each cohort / region (AD samples): subgroup score = mean gene-wise z-score of its hub genes;
assignment = highest score. Then:
  agreement with Neff classes (crosstab, ARI, Cramer's V)
  each subgroup score vs neuron marker score (Spearman)
  assignment before vs after cell-score adjustment (ARI), and R² of each score on cell scores
  random null: same number of decile-matched random genes per subgroup (500 sets) for |ρ| with
  the neuron score and for the adjusted ARI
Stage trend (all samples): Spearman between diagnosis order and each subgroup score.

Usage: python other_scheme_li2022.py --data /mnt/project-files/d1-data/raw --out ../results/rq2_li2022 --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from residual_structure import r2, residualise
from validate_cohorts import GROUP_ORDER, cohorts

HUB = {
    "I": ["PRELP", "RBM17", "AHNAK", "TAF10", "ANKRD40", "METRN", "RPL18", "TRIM8", "RPL35", "MKNK2"],
    "II": ["HINT1", "FGF12", "MAPK9", "ATP6V0B", "SLC6A15", "VPS35", "SYN2", "ATP6V1C1", "RAB6B", "ARPC5L"],
    "III": ["GAB1", "SALL1", "PHF10", "SPOCK3", "KRCC1", "PAWR", "S100PBP", "CPM", "MYLIP", "BRD7"],
}
SUB = np.array(list(HUB))


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def subgroup_scores(z_df: pd.DataFrame, sets: dict) -> np.ndarray:
    return np.column_stack([z_df.loc[g].values.mean(0) for g in sets.values()])


def run(name, expr, meta, cls, n, rng):
    ad = meta.index[meta["group"] == "AD"]
    neff_genes = cls.index.intersection(expr.index)
    cent = cls.loc[neff_genes].sub(cls.loc[neff_genes].mean(axis=1), axis=0).values
    hub = {k: [g for g in v if g in expr.index] for k, v in HUB.items()}
    excl = set(neff_genes) | {g for v in hub.values() for g in v}

    def score(lst):
        lst = [g for g in lst if g in expr.index and g not in excl]
        return z_np(expr.loc[lst, ad].values).mean(0)
    neuron = score(NEURON_MARKERS)
    cells = np.column_stack([neuron] + [score(v) for v in CELL_MARKERS.values()])

    Z = pd.DataFrame(z_np(expr.loc[:, ad].values), index=expr.index, columns=ad)
    S = subgroup_scores(Z, hub)
    lab = SUB[S.argmax(1)]
    rn = corr_np(Z.loc[neff_genes].values, cent)
    neff = np.array(list("ABC"))[rn.argmax(1)]
    ca = rn[:, 2] - rn[:, 0]

    genes_all = sorted({g for v in hub.values() for g in v})
    R = pd.DataFrame(residualise(Z.loc[genes_all].values, cells), index=genes_all, columns=ad)
    lab_adj = SUB[subgroup_scores(pd.DataFrame(z_np(R.values), index=genes_all), hub).argmax(1)]

    rep = {"cohort": name, "n_AD": int(len(ad)), "hub_genes_found": {k: len(v) for k, v in hub.items()},
           "frac": {k: round(float((lab == k).mean()), 3) for k in SUB},
           "vs_neff_ari": round(float(adjusted_rand_score(neff, lab)), 3),
           "vs_neff_cramers_v": round(cramers_v(neff, lab), 3),
           "crosstab_neff_rows": pd.crosstab(pd.Series(neff, name="neff"), pd.Series(lab, name="li")).to_dict(orient="index"),
           "score_vs_neuron_rho": {k: round(float(stats.spearmanr(S[:, i], neuron)[0]), 3) for i, k in enumerate(SUB)},
           "score_vs_neff_ca_rho": {k: round(float(stats.spearmanr(S[:, i], ca)[0]), 3) for i, k in enumerate(SUB)},
           "score_r2_cells": {k: round(r2(S[:, i], cells), 3) for i, k in enumerate(SUB)},
           "ari_adj": round(float(adjusted_rand_score(lab, lab_adj)), 3)}

    # stage trend on all samples
    Za = pd.DataFrame(z_np(expr.loc[:, meta.index].values), index=expr.index, columns=meta.index)
    Sa = subgroup_scores(Za, hub)
    stage = meta["group"].map(GROUP_ORDER).values
    rep["stage_rho"] = {k: round(float(stats.spearmanr(stage, Sa[:, i])[0]), 3) for i, k in enumerate(SUB)}

    # random null (expression-decile matched sets of the same sizes)
    dec = pd.qcut(expr[ad].mean(axis=1).rank(method="first"), 10, labels=False)
    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pool = expr.index[expr.index.notna()].difference(list(excl | marker_genes))
    pool_by = {d: pool[dec.loc[pool].values == d] for d in range(10)}
    max_abs_obs = max(abs(v) for v in rep["score_vs_neuron_rho"].values())
    null_rho, null_ari = [], []
    for _ in range(n):
        sets = {}
        for k, v in hub.items():
            sets[k] = [rng.choice(pool_by[d]) for d in dec.loc[v].values]
        Sr = subgroup_scores(Z, sets)
        null_rho.append(max(abs(stats.spearmanr(Sr[:, i], neuron)[0]) for i in range(3)))
        gl = sorted({g for v in sets.values() for g in v})
        Rr = pd.DataFrame(z_np(residualise(Z.loc[gl].values, cells)), index=gl)
        null_ari.append(adjusted_rand_score(SUB[Sr.argmax(1)], SUB[subgroup_scores(Rr, sets).argmax(1)]))
    rep["null_max_abs_neuron_rho"] = {"obs": round(max_abs_obs, 3), "random_median": round(float(np.median(null_rho)), 3),
                                      "p": round(float((np.sum(np.array(null_rho) >= max_abs_obs) + 1) / (n + 1)), 4)}
    rep["null_ari_adj"] = {"obs": rep["ari_adj"], "random_median": round(float(np.median(null_ari)), 3),
                           "p_lower": round(float((np.sum(np.array(null_ari) <= rep["ari_adj"]) + 1) / (n + 1)), 4)}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    reps = []
    for name, e, m in cohorts(a.data):
        rep = run(name, e, m, cls, a.n, rng)
        reps.append(rep)
        print(json.dumps(rep, ensure_ascii=False, default=str), flush=True)
    (a.out / "li2022_report.json").write_text(json.dumps(reps, indent=2, ensure_ascii=False, default=str))
    flat = pd.DataFrame([{
        "cohort": r["cohort"], **{f"frac_{k}": v for k, v in r["frac"].items()},
        "ari_vs_neff": r["vs_neff_ari"], "cramers_v_vs_neff": r["vs_neff_cramers_v"],
        **{f"rho_neuron_{k}": v for k, v in r["score_vs_neuron_rho"].items()},
        **{f"r2cells_{k}": v for k, v in r["score_r2_cells"].items()},
        "ari_adj": r["ari_adj"], "ari_adj_random": r["null_ari_adj"]["random_median"], "ari_adj_p": r["null_ari_adj"]["p_lower"],
        "max_rho_random": r["null_max_abs_neuron_rho"]["random_median"], "max_rho_p": r["null_max_abs_neuron_rho"]["p"],
        **{f"stage_{k}": v for k, v in r["stage_rho"].items()},
    } for r in reps])
    flat.to_csv(a.out / "li2022_summary.csv", index=False)
    print(flat.T.to_string())


if __name__ == "__main__":
    main()
