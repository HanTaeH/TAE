"""Experiment 2 (first pass): project Neff 2021 classes onto every available cohort.

For each cohort / brain region:
  - AD-only projection (z-scores within AD samples), permutation p-values
  - neuronal marker score by class
  - cell-composition-adjusted projection and its agreement with the unadjusted one
  - all-sample projection (Control / AsymAD / AD z-scored together) to test whether
    A -> B -> C tracks disease progression (Spearman trend Control < AsymAD < AD)

Usage: python validate_cohorts.py --data /mnt/project-files/d1-data/raw --out ../results/validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import load_gse125583
import load_gse33000
import load_illumina
import neff_subtypes
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, correlate, project, zscore_rows

GROUP_ORDER = {"Control": 0, "AsymAD": 1, "AD": 2}
CLASS_ORDER = {"A": 0, "B": 1, "C": 2}


def cohorts(data: Path):
    e, m = load_gse33000.load(data / "GSE33000")
    m = m.rename(columns={"gender": "sex"})
    m["region"] = "PFC"
    yield "GSE33000_PFC", e, m
    e, m = load_illumina.load(data, "GSE132903")
    yield "GSE132903_MTG", e, m
    e, m = load_illumina.load(data, "GSE118553")
    for region in ["Temporal_Cortex", "Frontal_Cortex", "Entorhinal_Cortex", "Cerebellum"]:
        idx = m.index[m["region"] == region]
        yield f"GSE118553_{region}", e[idx], m.loc[idx]
    if (data / "GSE125583" / "raw").exists():
        e, m = load_gse125583.load(data)
        yield "GSE125583_FG", e, m


def analyse(name: str, expr: pd.DataFrame, meta: pd.DataFrame, cent: pd.DataFrame,
            rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    genes = cent.index.intersection(expr.index)
    c = cent.loc[genes].sub(cent.loc[genes].mean(axis=1), axis=0)
    ad = meta.index[meta["group"] == "AD"]
    rep: dict = {"cohort": name, "n": meta["group"].value_counts().to_dict(),
                 "coverage": round(len(genes) / len(cent), 3)}

    z = zscore_rows(expr.loc[genes, ad])
    res = project(z, c, rng)
    rep["frac_p05"] = round(float((res["p_perm"] < 0.05).mean()), 3)
    rep["class_frac"] = res["subtype"].value_counts(normalize=True).reindex(list("ABC")).fillna(0).round(3).to_dict()

    scores = {}
    mk = [g for g in NEURON_MARKERS if g in expr.index and g not in genes]
    scores["neuron"] = zscore_rows(expr.loc[mk, ad]).mean()
    for ct, lst in CELL_MARKERS.items():
        lst = [g for g in lst if g in expr.index and g not in genes]
        scores[ct] = zscore_rows(expr.loc[lst, ad]).mean()
    X = pd.DataFrame(scores)
    res = res.join(X)
    rep["neuron_by_class"] = res.groupby("subtype")["neuron"].mean().round(3).to_dict()
    if res["subtype"].nunique() > 1:
        rep["neuron_kruskal_p"] = float(stats.kruskal(*[g.values for _, g in res.groupby("subtype")["neuron"]]).pvalue)

    Xd = np.column_stack([np.ones(len(X)), X.values])
    Y = z.T.values
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = pd.DataFrame((Y - Xd @ beta).T, index=z.index, columns=z.columns)
    r2 = 1 - ((Y - Xd @ beta) ** 2).sum(axis=0) / (Y ** 2).sum(axis=0)
    rep["median_r2_cell_scores"] = round(float(np.median(r2)), 3)
    adj = project(zscore_rows(resid), c, rng)
    rep["adj_frac_p05"] = round(float((adj["p_perm"] < 0.05).mean()), 3)
    rep["adj_ari_vs_unadjusted"] = round(float(adjusted_rand_score(res["subtype"], adj["subtype"])), 3)
    res["subtype_celladj"] = adj["subtype"]

    # all-sample projection: does class order follow disease stage?
    za = zscore_rows(expr.loc[genes, meta.index])
    lab = correlate(za, c).idxmax(axis=1)
    tab = pd.crosstab(meta["group"], lab).reindex(columns=list("ABC"), fill_value=0)
    rep["all_sample_class_by_group"] = (tab.div(tab.sum(axis=1), axis=0).round(3)).to_dict(orient="index")
    rho, p = stats.spearmanr(meta["group"].map(GROUP_ORDER), lab.map(CLASS_ORDER))
    rep["stage_trend_spearman"] = {"rho": round(float(rho), 3), "p": float(p)}

    res["cohort"] = name
    return rep, res.join(meta[["age", "sex", "region"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    reports, tables = [], []
    for name, e, m in cohorts(a.data):
        if (m["group"] == "AD").sum() < 10:
            continue
        rep, res = analyse(name, e, m, cls, rng)
        reports.append(rep)
        tables.append(res)
        print(json.dumps(rep, ensure_ascii=False))

    pd.concat(tables).to_csv(a.out / "projection_all_cohorts.csv")
    (a.out / "validation_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    flat = pd.DataFrame([{
        "cohort": r["cohort"], "n_AD": r["n"].get("AD", 0), "coverage": r["coverage"],
        "frac_p05": r["frac_p05"], **{f"frac_{k}": v for k, v in r["class_frac"].items()},
        **{f"neuron_{k}": v for k, v in r["neuron_by_class"].items()},
        "r2_cell": r["median_r2_cell_scores"], "ari_celladj": r["adj_ari_vs_unadjusted"],
        "stage_rho": r["stage_trend_spearman"]["rho"], "stage_p": r["stage_trend_spearman"]["p"],
    } for r in reports])
    flat.to_csv(a.out / "validation_summary.csv", index=False)
    print(flat.to_string())


if __name__ == "__main__":
    main()
