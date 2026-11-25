"""MSBB (GSE84422): C-A score vs plaque density within AD, before and after neuron adjustment
(v10, 2026-11-25; internal review round 7). Neff et al. reported more plaques in class C than A within
MSBB AD; the manuscript's "unrelated to plaque density" pooled all 106 subjects.

Uses results/msbb/msbb_subject_scores.csv (subject means of cortical regions). Partial Spearman on ranks,
covariates age and sex, then also the neuron score. Groups: AD (CERAD definite/probable, n = 64), AD +
possible AD, all subjects. Bootstrap 95% CI (2,000 resamples).

Usage: python msbb_plaque_within_ad.py --res ../results --out ../results/msbb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def prho(d, x, y, cov):
    r = {c: stats.rankdata(d[c]) for c in [x, y] + cov}
    D = np.column_stack([np.ones(len(d))] + [r[c] for c in cov])
    ex = r[x] - D @ np.linalg.lstsq(D, r[x], rcond=None)[0]
    ey = r[y] - D @ np.linalg.lstsq(D, r[y], rcond=None)[0]
    return stats.pearsonr(ex, ey)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s = pd.read_csv(a.res / "msbb/msbb_subject_scores.csv").dropna(subset=["ca", "neuron", "plaque", "age", "female"])
    rng = np.random.default_rng(20261125)
    out = {}
    for lab, sub in {"AD": s[s.group == "AD"], "AD_plus_possible": s[s.group.isin(["AD", "Possible"])], "all": s}.items():
        res = {"n": int(len(sub))}
        for spec, cov in {"age_sex": ["age", "female"], "age_sex_neuron": ["age", "female", "neuron"]}.items():
            r, p = prho(sub, "ca", "plaque", cov)
            bs = []
            for _ in range(2000):
                b = sub.iloc[rng.integers(0, len(sub), len(sub))]
                bs.append(prho(b, "ca", "plaque", cov)[0])
            res[spec] = {"rho": round(float(r), 3), "p": float(f"{p:.3g}"),
                         "ci95": [round(float(np.nanquantile(bs, q)), 3) for q in (.025, .975)]}
        res["neuron_vs_plaque_age_sex"] = [round(float(v), 3) for v in prho(sub, "neuron", "plaque", ["age", "female"])]
        out[lab] = res
        print(lab, json.dumps(res))
    (a.out / "msbb_plaque_within_ad.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
