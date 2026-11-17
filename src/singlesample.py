"""Cohort-independent scoring (singscore-style) of the Neff classes.

The main projection z-scores genes within each cohort, so labels are relative to the cohort mix.
Here every sample is scored alone: genes are ranked within the sample (all measured genes), ranks are
scaled to [-0.5, 0.5], and a class score is the mean scaled rank of the class's up genes minus that
of its down genes (up / down = sign of the centred Neff centroid for that class). The C-minus-A
score uses the sign of centroid C minus centroid A. Neuron score = mean scaled rank of neuron markers
(Neff genes excluded). Class = highest of the three class scores (no cohort reference).

Per cohort: agreement with the cohort-z C-minus-A score (AD), C-minus-A vs neuron score (AD),
stage trend (all samples), class fractions by group, and a random-gene null for the neuron and stage
correlations (decile x SD tercile matched, inheriting up/down direction; --n sets).
Pooled (random effects, independent cohorts) estimates for the two primary endpoints.

Usage: python singlesample.py --data /mnt/project-files/d1-data/raw --out ../results/singlesample --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import GROUP_ORDER, cohorts

INDEP = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE125583_FG"]
CLS = np.array(list("ABC"))


def scaled_ranks(X: pd.DataFrame) -> pd.DataFrame:
    R = X.rank(axis=0, method="average")
    return R / len(X) - 0.5


def ss(R: pd.DataFrame, w: pd.Series) -> np.ndarray:
    up, dn = w.index[w > 0], w.index[w < 0]
    return R.loc[up].mean(0).values - R.loc[dn].mean(0).values


def ci(r, n):
    z, se = np.arctanh(r), 1.06 / np.sqrt(n - 3)
    return z, se


def pooled(rs, ns):
    z = np.arctanh(rs)
    v = (1.06 / np.sqrt(np.array(ns) - 3)) ** 2
    w = 1 / v
    q = np.sum(w * (z - np.sum(w * z) / w.sum()) ** 2)
    tau2 = max(0, (q - (len(z) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    m, se = np.sum(ws * z) / ws.sum(), np.sqrt(1 / ws.sum())
    return {"rho": round(float(np.tanh(m)), 3), "ci": [round(float(np.tanh(m - 1.96 * se)), 3),
            round(float(np.tanh(m + 1.96 * se)), 3)], "p": float(f"{2 * stats.norm.sf(abs(m / se)):.3g}")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    markers = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    reps, per = [], []
    for name, expr, meta in cohorts(a.data):
        expr = expr.loc[expr.index.notna() & ~expr.index.duplicated(), meta.index]
        expr = expr.dropna()
        genes = cls.index.intersection(expr.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        R = scaled_ranks(expr)
        S = np.column_stack([ss(R, cent[k]) for k in ["A", "B", "C"]])
        lab = CLS[S.argmax(1)]
        ca_w = cent["C"] - cent["A"]
        ca = ss(R, ca_w)
        neu = R.loc[[g for g in NEURON_MARKERS if g in R.index and g not in genes]].mean(0).values
        stage = meta["group"].map(GROUP_ORDER).values
        ad = (meta["group"] == "AD").values
        # cohort-z C-minus-A for comparison (AD only)
        rz = corr_np(z_np(expr.loc[genes].values[:, ad]), cent.values)
        ca_z = rz[:, 2] - rz[:, 0]
        rep = {"cohort": name, "n": int(len(meta)), "n_AD": int(ad.sum()), "neff_genes": int(len(genes)),
               "AD_ss_vs_cohortz_ca": round(float(stats.spearmanr(ca[ad], ca_z)[0]), 3),
               "AD_ca_vs_neuron": round(float(stats.spearmanr(ca[ad], neu[ad])[0]), 3),
               "stage_rho": round(float(stats.spearmanr(ca, stage)[0]), 3),
               "stage_p": float(f"{stats.spearmanr(ca, stage)[1]:.3g}"),
               "class_frac_by_group": pd.crosstab(meta["group"], lab, normalize="index").reindex(columns=list("ABC"), fill_value=0)
                   .round(3).to_dict(orient="index")}
        # random-gene null
        mean_e, sd_e = expr.mean(axis=1), expr.std(axis=1)
        dec = pd.qcut(mean_e.rank(method="first"), 10, labels=False)
        terc = sd_e.groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        strat = dec * 3 + terc
        pool = expr.index.difference(genes).difference(list(markers))
        pool_by = {s: pool[strat.loc[pool].values == s] for s in range(30)}
        ns = strat.loc[genes].values
        nn, nst = [], []
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for s in np.unique(ns):
                ii = np.where(ns == s)[0]
                pick[ii] = rng.choice(pool_by[s], size=len(ii), replace=False)
            w = pd.Series(ca_w.values, index=list(pick))
            car = ss(R, w)
            nn.append(stats.spearmanr(car[ad], neu[ad])[0])
            nst.append(stats.spearmanr(car, stage)[0])
        nn, nst = np.array(nn), np.array(nst)
        rep["null_neuron"] = {"random_median": round(float(np.median(nn)), 3),
                              "p": round(float((np.sum(nn <= rep["AD_ca_vs_neuron"]) + 1) / (a.n + 1)), 4)}
        rep["null_stage"] = {"random_median": round(float(np.median(nst)), 3),
                             "p": round(float((np.sum(nst >= rep["stage_rho"]) + 1) / (a.n + 1)), 4)}
        reps.append(rep)
        per.append(meta[["group"]].assign(cohort=name, ss_ca=ca, ss_neuron=neu, ss_class=lab,
                                          ss_A=S[:, 0], ss_B=S[:, 1], ss_C=S[:, 2]))
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    ind = [r for r in reps if r["cohort"] in INDEP]
    summ = {"pooled_AD_ca_vs_neuron": pooled([r["AD_ca_vs_neuron"] for r in ind], [r["n_AD"] for r in ind]),
            "pooled_stage": pooled([r["stage_rho"] for r in ind], [r["n"] for r in ind]),
            "cohorts": reps}
    (a.out / "singlesample_report.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    pd.concat(per).to_csv(a.out / "singlesample_per_sample.csv")
    flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} |
                         {"null_neuron_p": r["null_neuron"]["p"], "null_stage_p": r["null_stage"]["p"],
                          **{f"AD_frac_{k}": r["class_frac_by_group"].get("AD", {}).get(k) for k in "ABC"},
                          **{f"Control_frac_{k}": r["class_frac_by_group"].get("Control", {}).get(k) for k in "ABC"}}
                         for r in reps])
    flat.to_csv(a.out / "singlesample_summary.csv", index=False)
    print(flat.to_string())
    print(json.dumps({k: v for k, v in summ.items() if k != "cohorts"}, indent=1))


if __name__ == "__main__":
    main()
