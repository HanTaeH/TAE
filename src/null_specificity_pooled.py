"""Pooled version of the specificity-matched random-gene null (null_specificity_matched.py), round-5 review.

Draw i of each cohort is combined across the four independent cohorts with inverse-variance weights from the
observed bootstrap SEs (stage_neuron_adjusted/per_cohort.csv; tau^2 was 0 for both endpoints, so this equals the
REML estimate). The same is done for the observed values. Tails are fixed by hypothesis, not by the sign of the
observed value: neuron_rho lower tail, the stage endpoints upper tail. Per-cohort p-values are recomputed with
these fixed tails as a check.

Usage: python null_specificity_pooled.py --res ../results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAIL = {"neuron_rho": "lower", "stage_agesex": "upper", "stage_agesex_neuron": "upper"}
SE_ENDPOINT = {"stage_agesex": "ca_stage_agesex", "stage_agesex_neuron": "ca_stage_agesex_neuron"}


def emp_p(obs, null, tail):
    more = (null <= obs).sum() if tail == "lower" else (null >= obs).sum()
    return float((more + 1) / (len(null) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    a = ap.parse_args()
    dist = pd.read_csv(a.res / "null_specificity/null_specificity_matched_distributions.csv")
    obs = {r["cohort"]: r for r in json.loads((a.res / "null_specificity/null_specificity_matched_report.json").read_text())}
    se = pd.read_csv(a.res / "stage_neuron_adjusted/per_cohort.csv")
    cohorts = list(obs)
    out = {"per_cohort_fixed_tail": {}, "pooled": {}}
    for c in cohorts:
        d = dist[dist.cohort == c]
        out["per_cohort_fixed_tail"][c] = {m: round(emp_p(obs[c][m]["neff"], d[m].values, t), 4) for m, t in TAIL.items()}
    n = dist.groupby("cohort").size().min()
    for m, ep in SE_ENDPOINT.items():
        w = np.array([1 / se[(se.endpoint == ep) & (se.cohort == c)].se_boot.iloc[0] ** 2 for c in cohorts])
        z_obs = np.arctanh([obs[c][m]["neff"] for c in cohorts])
        Z = np.column_stack([np.arctanh(dist[dist.cohort == c][m].values[:n]) for c in cohorts])
        pooled_null = np.tanh(Z @ w / w.sum())
        pooled_obs = float(np.tanh(z_obs @ w / w.sum()))
        out["pooled"][m] = {"neff": round(pooled_obs, 3), "random_median": round(float(np.median(pooled_null)), 3),
                            "random_q05_q95": [round(float(np.quantile(pooled_null, .05)), 3),
                                               round(float(np.quantile(pooled_null, .95)), 3)],
                            "p_empirical": round(emp_p(pooled_obs, pooled_null, TAIL[m]), 4), "n_sets": int(n)}
    (a.res / "null_specificity/null_specificity_pooled_report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
