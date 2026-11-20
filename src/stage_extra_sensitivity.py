"""Further sensitivity analyses of primary endpoint 2 (case-control separation), from the round-4 review.

1. NNLS-adjusted version: age/sex partial Spearman additionally adjusted for centred log-ratio NNLS
   proportions (9 SEA-AD types; deconv_seaad.py), per cohort and pooled.
2. GSE118553 region choice: the pooled endpoint with frontal or entorhinal cortex in place of
   temporal cortex (same donors, so only one region at a time).
3. A prevalence-invariant effect size: the AUC of the C-A score for AD versus control (Mann-Whitney),
   unadjusted and after residualising the score on age and sex (OLS), per cohort, with bootstrap 95% CIs.
   GSE118553 asymptomatic AD donors are not part of the AUC (AD versus control only).
4. GSE118553 AD versus control only (asymptomatic AD dropped) in place of the ordinal three-level coding.
Bootstrap SEs (2,000) and REML/HKSJ pooling as in meta_reanalysis.py.

Usage: python stage_extra_sensitivity.py --data /mnt/project-files/d1-data/raw --res ../results --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from deconv_seaad import clr
from meta_reanalysis import INDEPENDENT, boot_se, ca_score, pool, rho
from pilot_gse33000 import SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts

NNLS = ["nnls_Exc", "nnls_Inh_SST", "nnls_Inh_PVALB", "nnls_Inh_other", "nnls_Astro", "nnls_Oligo", "nnls_OPC",
        "nnls_Micro", "nnls_Vascular"]
ALT_118553 = ["GSE118553_Frontal_Cortex", "GSE118553_Entorhinal_Cortex"]


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    u = stats.mannwhitneyu(pos, neg).statistic
    return float(u / (len(pos) * len(neg)))


def auc_ci(score, y, rng, n):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(auc(score[i], y[i]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 61)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    P = pd.read_csv(a.res / "deconv/deconv_proportions.csv", index_col=0)
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT + ALT_118553:
            continue
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        ca = ca_score(zscore_rows(e.loc[genes, m.index]), cent).values
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov = np.column_stack([age, sex])
        r = rho(ca, stage, cov)
        rows.append({"spec": "agesex", "cohort": name, "n": len(ca), "rho": round(r, 3), "z": np.arctanh(r),
                     "se_boot": boot_se(ca, stage, cov, rng, a.n_boot)})
        if name in INDEPENDENT:
            pp = P.reindex(m.index)[NNLS]
            ok = pp.notna().all(1).values
            if ok.sum() > 20:
                C = np.column_stack([cov[ok], clr(pp.values[ok] / pp.values[ok].sum(1, keepdims=True))])
                r = rho(ca[ok], stage[ok], C)
                rows.append({"spec": "agesex_nnls", "cohort": name, "n": int(ok.sum()), "rho": round(r, 3),
                             "z": np.arctanh(r), "se_boot": boot_se(ca[ok], stage[ok], C, rng, a.n_boot)})
            y = (m["group"] == "AD").values.astype(int)
            keep = m["group"].isin(["AD", "Control"]).values
            D = np.column_stack([np.ones(len(ca)), cov])
            res = ca - D @ np.linalg.lstsq(D, ca, rcond=None)[0]
            rows.append({"spec": "auc", "cohort": name, "n": int(keep.sum()),
                         "auc_unadjusted": round(auc(ca[keep], y[keep]), 3),
                         "auc_unadjusted_ci": str(auc_ci(ca[keep], y[keep], rng, a.n_boot)),
                         "auc_age_sex_residual": round(auc(res[keep], y[keep]), 3),
                         "auc_age_sex_residual_ci": str(auc_ci(res[keep], y[keep], rng, a.n_boot))})
            if name == "GSE118553_Temporal_Cortex":
                r = rho(ca[keep], stage[keep], cov[keep])
                rows.append({"spec": "agesex_ad_vs_control", "cohort": name, "n": int(keep.sum()), "rho": round(r, 3),
                             "z": np.arctanh(r), "se_boot": boot_se(ca[keep], stage[keep], cov[keep], rng, a.n_boot)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out / "stage_extra_per_cohort.csv", index=False)
    base = df[(df.spec == "agesex") & df.cohort.isin(INDEPENDENT)].set_index("cohort")
    rep = {"pooled_agesex": pool(base.z.values, base.se_boot.values)}
    nn = df[df.spec == "agesex_nnls"]
    rep["pooled_agesex_nnls"] = pool(nn.z.values, nn.se_boot.values)
    for alt in ALT_118553:
        b2 = base.drop("GSE118553_Temporal_Cortex")
        r = df[(df.spec == "agesex") & (df.cohort == alt)]
        if len(r):
            z = np.r_[b2.z.values, r.z.values]
            se = np.r_[b2.se_boot.values, r.se_boot.values]
            rep[f"pooled_agesex_with_{alt}"] = pool(z, se)
    r = df[df.spec == "agesex_ad_vs_control"]
    if len(r):
        b2 = base.drop("GSE118553_Temporal_Cortex")
        rep["pooled_agesex_GSE118553_ad_vs_control"] = pool(np.r_[b2.z.values, r.z.values],
                                                            np.r_[b2.se_boot.values, r.se_boot.values])
    rep["auc"] = df[df.spec == "auc"][["cohort", "n", "auc_unadjusted", "auc_unadjusted_ci", "auc_age_sex_residual",
                                       "auc_age_sex_residual_ci"]].to_dict(orient="records")
    (a.out / "stage_extra_report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
