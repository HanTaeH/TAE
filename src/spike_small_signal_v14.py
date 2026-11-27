"""Can a small composition-independent signal be detected? Pooled and alternative tests (v14).

The v13 spike-in (spike_in_real.py) gave 80% power only for components of 21-52% of C-A variance in the array
cohorts. Two cheap changes are evaluated on the same spiked replicates, against specificity-matched nulls:

  (1) Pooling: the four cohorts' one-sided p-values are combined by Stouffer's method (unweighted and
      sqrt(n)-weighted). Replicate i of every cohort is spiked at the same target share of C-A variance, so
      the pooled test asks whether a component of that share, present in every cohort, is detected.
  (2) Statistic: besides the neuron coupling (upper tail of Spearman rho, as in v13), the share of C-A score
      variance NOT explained by the five marker scores (1 - R^2, OLS with intercept), upper tail. A
      composition-independent component raises it directly, whereas rho is near saturation (|rho| ~ 0.9).

Design as in v13: the spike X[g, j] + f * s_j * (C - A)_g is added to the classifier genes of a bootstrap
resample of AD samples; nulls (500 specificity-matched sets: expression decile x specificity class, classifier
and marker genes excluded) are computed once on the original samples. For each cohort f is calibrated so that
s explains the target share of C-A variance (median over a pilot of 100 replicates). Size (false-positive
rate) is the detection rate at share 0; it includes the classifier's own offset from the null, as in v13.

Usage: python spike_small_signal_v14.py --data <raw> --ref <seaad_mtg_reference.csv.gz> --out <dir> [--reps 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from classifier_specificity import NEURONAL
from gse33000_null_followup import draw
from meta_reanalysis import ca_score
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts

COH = ("GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE125583_FG")
SHARES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
F_GRID = np.array([0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, .75, 1.0])


def cell_matrix(Z: pd.DataFrame, exclude: set) -> np.ndarray:
    cols = [Z.loc[[x for x in NEURON_MARKERS if x in Z.index and x not in exclude]].mean().values]
    for v in CELL_MARKERS.values():
        g = [x for x in v if x in Z.index and x not in exclude]
        cols.append(Z.loc[g].mean().values)
    return np.column_stack([np.ones(Z.shape[1])] + cols)


def unexplained(score: np.ndarray, M: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(M, score, rcond=None)
    r = score - M @ beta
    return float(r.var() / score.var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1414)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    sc = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pv = {}     # cohort -> stat -> array (len(SHARES), reps) of one-sided p
    info = {}
    for cname, ce, cm in cohorts(a.data):
        if cname not in COH:
            continue
        ce = ce[cm.index].loc[:, (cm["group"] == "AD").values]
        cg = list(cls.index.intersection(ce.index))
        cc = cls.loc[cg].sub(cls.loc[cg].mean(axis=1), axis=0)
        contrast = (cls.loc[cg, "C"] - cls.loc[cg, "A"]).values
        dec = pd.qcut(ce.mean(axis=1).rank(method="first"), 10, labels=False).astype(str)
        st_s = dec + "_" + sc.reindex(ce.index).fillna(3).astype(int).astype(str)
        pool = ce.index.difference(cg).difference(list(marker))
        Z = zscore_rows(ce)
        M = cell_matrix(Z, set(cg))
        neu = M[:, 1]

        def stats_of(score, Mx):
            return stats.spearmanr(score, Mx[:, 1])[0], unexplained(score, Mx)
        obs = stats_of(ca_score(Z.loc[cg], cc).values, M)
        nulls = np.array([stats_of(ca_score(Z.loc[g].set_axis(cg), cc).values, M)
                          for g in draw(rng, cg, st_s, pool, a.n_null)])
        X = ce.loc[cg].values.astype(float)
        nA = X.shape[1]
        cell_raw = ce.loc[[x for x in marker if x in ce.index and x not in cg]]

        def rep(f):
            b = rng.integers(0, nA, nA)
            s = rng.standard_normal(nA)
            Zb = zscore_rows(pd.DataFrame(X[:, b] + f * np.outer(contrast, s), index=cg))
            score = ca_score(Zb, cc).values
            Mb = cell_matrix(zscore_rows(cell_raw.iloc[:, b]), set(cg))
            share = np.corrcoef(score, s)[0, 1] ** 2 if f > 0 else 0.0
            return score, Mb, share
        # calibrate f -> share of C-A variance explained by s
        pilot = [np.median([rep(f)[2] for _ in range(100)]) for f in F_GRID]
        pilot = np.maximum.accumulate(pilot)
        fmap = {v: float(np.interp(v, pilot, F_GRID)) for v in SHARES}
        P = {"rho": np.zeros((len(SHARES), a.reps)), "unexpl": np.zeros((len(SHARES), a.reps))}
        got = np.zeros((len(SHARES), a.reps))
        for i, v in enumerate(SHARES):
            for j in range(a.reps):
                score, Mb, share = rep(fmap[v])
                r, u = stats_of(score, Mb)
                P["rho"][i, j] = ((nulls[:, 0] >= r).sum() + 1) / (len(nulls) + 1)
                P["unexpl"][i, j] = ((nulls[:, 1] >= u).sum() + 1) / (len(nulls) + 1)
                got[i, j] = share
        pv[cname] = P
        info[cname] = {"n_ad": nA, "obs_rho": round(float(obs[0]), 3), "obs_unexplained": round(float(obs[1]), 3),
                       "null_rho_median": round(float(np.median(nulls[:, 0])), 3),
                       "null_unexplained_median": round(float(np.median(nulls[:, 1])), 3),
                       "null_unexplained_q95": round(float(np.quantile(nulls[:, 1], .95)), 3),
                       "p_upper_rho_obs": round(float(((nulls[:, 0] >= obs[0]).sum() + 1) / (len(nulls) + 1)), 4),
                       "p_upper_unexplained_obs": round(float(((nulls[:, 1] >= obs[1]).sum() + 1) / (len(nulls) + 1)), 4),
                       "f_for_share": {str(k): round(x, 4) for k, x in fmap.items()},
                       "achieved_share_median": dict(zip(map(str, SHARES), np.round(np.median(got, axis=1), 3).tolist()))}
        print(cname, json.dumps(info[cname]), flush=True)
    names = list(pv)
    w = np.sqrt([info[c]["n_ad"] for c in names])
    rows = []
    for stat in ("rho", "unexpl"):
        for i, v in enumerate(SHARES):
            zs = np.array([stats.norm.isf(pv[c][stat][i]) for c in names])       # cohorts x reps
            p_un = stats.norm.sf(zs.sum(0) / np.sqrt(len(names)))
            p_w = stats.norm.sf((w[:, None] * zs).sum(0) / np.sqrt((w ** 2).sum()))
            row = {"statistic": stat, "share": v}
            for k, c in enumerate(names):
                row[c] = round(float((pv[c][stat][i] < .05).mean()), 3)
            row["stouffer"] = round(float((p_un < .05).mean()), 3)
            row["stouffer_sqrt_n"] = round(float((p_w < .05).mean()), 3)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "spike_small_signal_power.csv", index=False)

    def at(col, stat, x):
        d = df[df.statistic == stat]
        pw = np.maximum.accumulate(d[col].values)
        if pw[1:].max() < x:
            return None
        return round(float(np.interp(x, pw[1:] + np.arange(len(pw) - 1) * 1e-9, d.share.values[1:])), 3)
    summ = {stat: {col: {"size": float(df[(df.statistic == stat) & (df.share == 0)][col].iloc[0]),
                         "share_at_50pct": at(col, stat, .5), "share_at_80pct": at(col, stat, .8)}
                   for col in names + ["stouffer", "stouffer_sqrt_n"]} for stat in ("rho", "unexpl")}
    (a.out / "spike_small_signal_v14.json").write_text(json.dumps({"cohorts": info, "summary": summ}, indent=2))
    print(df.to_string())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
