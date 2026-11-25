"""MSBB (GSE84422) neuropathology follow-ups for v11 (review round 9, Reviewer 2 M1e, M2ii, M3a, M3d).

Subject-level cortical means (results/msbb/msbb_subject_scores.csv) are merged with GSE84422 subject
annotations (NFT density sum, CERAD rating sum, pH, PMI). Partial Spearman on ranks, covariates age and
sex, then also the neuron score; bootstrap 95% CI (2,000 resamples). Groups:
  AD (CERAD definite/probable), AD restricted to CDR >= 1 (closest to Neff's clinical definition), all.
Outcomes: nft_sum, cerad_sum, plaque (average neuritic plaque density), braak, cdr.
Also: share of neuron-score variance explained by pH and PMI (R^2 of the rank regression), and the
neuron score's association with NFT versus plaque density.

Usage: python msbb_neuropath_v11.py --data /mnt/project-files/d1-data/raw --res ../results --out ../results/msbb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_gse84422
from msbb_plaque_within_ad import prho

OUTC = ["nft_sum", "cerad_sum", "plaque", "braak", "cdr"]


def boot(sub, x, y, cov, rng, n=2000):
    bs = []
    for _ in range(n):
        b = sub.iloc[rng.integers(0, len(sub), len(sub))]
        try:
            bs.append(prho(b, x, y, cov)[0])
        except Exception:
            pass
    return [round(float(np.nanquantile(bs, q)), 3) for q in (.025, .975)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _, meta = load_gse84422.load(a.data)
    sm = meta.groupby("subject")[["nft_sum", "cerad_sum", "ph", "pmi"]].first()
    s = pd.read_csv(a.res / "msbb/msbb_subject_scores.csv")
    s["subject"] = s["subject"].astype(str)
    sm.index = sm.index.astype(str)
    s = s.merge(sm, left_on="subject", right_index=True, how="left")
    rng = np.random.default_rng(20261126)
    out = {}
    groups = {"AD": s[s.group == "AD"], "AD_CDR_ge1": s[(s.group == "AD") & (s.cdr >= 1)], "all": s}
    for lab, sub in groups.items():
        res = {"n": int(len(sub))}
        for y in OUTC:
            d = sub.dropna(subset=["ca", "neuron", y, "age", "female"])
            if len(d) < 10 or d[y].nunique() < 3:
                continue
            row = {"n": int(len(d))}
            for spec, cov in {"age_sex": ["age", "female"], "age_sex_neuron": ["age", "female", "neuron"]}.items():
                r, p = prho(d, "ca", y, cov)
                row[spec] = {"rho": round(float(r), 3), "p": float(f"{p:.3g}"), "ci95": boot(d, "ca", y, cov, rng)}
            rn, pn = prho(d, "neuron", y, ["age", "female"])
            row["neuron_vs_outcome_age_sex"] = {"rho": round(float(rn), 3), "p": float(f"{pn:.3g}"),
                                                "ci95": boot(d, "neuron", y, ["age", "female"], rng)}
            res[y] = row
        out[lab] = res
        print(lab, json.dumps(res), flush=True)
    d = s.dropna(subset=["neuron", "ph", "pmi"])
    R = d[["ph", "pmi"]].rank().values
    X = np.column_stack([np.ones(len(d)), R])
    yv = d["neuron"].rank().values
    fit = X @ np.linalg.lstsq(X, yv, rcond=None)[0]
    out["neuron_score_vs_ph_pmi"] = {"n": int(len(d)), "r2_rank": round(float(1 - ((yv - fit) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()), 3),
                                     "rho_ph": round(float(stats.spearmanr(d.neuron, d.ph)[0]), 3),
                                     "rho_pmi": round(float(stats.spearmanr(d.neuron, d.pmi)[0]), 3)}
    print(json.dumps(out["neuron_score_vs_ph_pmi"]))
    (a.out / "msbb_neuropath_v11.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
