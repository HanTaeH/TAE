"""Cohort follow-ups for v11 (review round 9) in the four independent cohorts.

  1. Class B axis (Reviewer 2 M8): B score = r(B) - mean(r(A), r(C)) in AD samples, Spearman with the five
     marker scores, and its oligodendrocyte coupling against 300 specificity-matched random gene sets
     (both tails), as for the C-A axis.
  2. Dissection (Reviewer 2 M2iii): primary endpoint 1 after partialling out the oligodendrocyte score
     (and all four glial scores), Spearman on ranks.
  3. Neuronal subtypes (Reviewer 2 M6): C-A score against an excitatory (SLC17A7, NRGN, CALB1 where not
     classifier genes), an inhibitory (GAD1, GAD2, SST) and a pan-neuronal synaptic (SNAP25, SYT1, STMN2,
     SYN1, ENO2, RBFOX3, VSNL1, NEFL, NEFM, GAP43) marker score, and the partial association of each
     with the other two held fixed.
  4. Effect-size scale (Reviewer 1 M6): case-control partial rho with the neuron score next to the
     semipartial on the original (age/sex-residualised) C-A scale: sr = cov(y_r, e_x) / (sd(y_r) sd(x_r)),
     ranks as in the primary analysis.
  5. GSE118553 with asymptomatic AD analysed separately (Reviewer 2 M3b): AD vs control only, and
     asymptomatic AD vs AD for the neuron score and C-A score (age/sex partial).

Usage: python cohort_followups_v11.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out ../results/cohort_v11
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
from meta_reanalysis import INDEPENDENT, resid_rank, rho
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts

EXC = ["SLC17A7", "NRGN", "CALB1"]
INH = ["GAD1", "GAD2", "SST"]
PAN = ["SNAP25", "SYT1", "STMN2", "SYN1", "ENO2", "RBFOX3", "VSNL1", "NEFL", "NEFM", "GAP43"]


def axes(Z, cent):
    r = corr_np(z_np(Z), cent)
    return r[:, 2] - r[:, 0], r[:, 1] - (r[:, 0] + r[:, 2]) / 2


def prs(x, y, cov):
    return float(np.corrcoef(resid_rank(x, cov), resid_rank(y, cov))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1212)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    sc = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    out = {}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        genes = list(cls.index.intersection(e.index))
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        ad = (m["group"] == "AD").values
        Zad = zscore_rows(e.loc[:, ad])
        Zall = zscore_rows(e)

        def score(Z, gl):
            g = [x for x in gl if x in Z.index and x not in genes]
            return Z.loc[g].mean().values, g
        neu, _ = score(Zad, NEURON_MARKERS)
        cells = {k: score(Zad, v)[0] for k, v in CELL_MARKERS.items()}
        ca, bb = axes(Zad.loc[genes].values, cent)
        res = {"n_ad": int(ad.sum())}
        # 1. class B axis
        res["B_axis"] = {"vs_neuron": round(float(stats.spearmanr(bb, neu)[0]), 3),
                         **{f"vs_{k}": round(float(stats.spearmanr(bb, v)[0]), 3) for k, v in cells.items()},
                         "vs_CA": round(float(stats.spearmanr(bb, ca)[0]), 3)}
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        st = dec * 4 + sc.reindex(e.index).fillna(3).astype(int)
        pool = e.index.difference(genes).difference(list(marker))
        by = {k: pool[st.loc[pool].values == k] for k in st.loc[genes].unique()}
        gst = st.loc[genes].values
        obs = stats.spearmanr(bb, cells["oligodendrocyte"])[0]
        nl = []
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for k in np.unique(gst):
                idx = np.where(gst == k)[0]
                src = by.get(k, pool)
                pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
            nl.append(stats.spearmanr(axes(Zad.loc[list(pick)].values, cent)[1], cells["oligodendrocyte"])[0])
        nl = np.array(nl)
        res["B_axis"]["oligo_vs_spec_null"] = {"obs": round(float(obs), 3), "null_median": round(float(np.median(nl)), 3),
                                               "null_q05_q95": [round(float(np.quantile(nl, q)), 3) for q in (.05, .95)],
                                               "p_lower": round(float(((nl <= obs).sum() + 1) / (a.n + 1)), 4),
                                               "p_upper": round(float(((nl >= obs).sum() + 1) / (a.n + 1)), 4)}
        # 2. dissection
        res["P1_adjusted"] = {"raw": round(float(stats.spearmanr(ca, neu)[0]), 3),
                              "partial_oligo": round(prs(ca, neu, cells["oligodendrocyte"][:, None]), 3),
                              "partial_all_glia": round(prs(ca, neu, np.column_stack(list(cells.values()))), 3)}
        # 3. neuronal subtypes
        sub = {k: score(Zad, v) for k, v in {"excitatory": EXC, "inhibitory": INH, "pan_synaptic": PAN}.items()}
        res["neuron_subscores"] = {}
        for k, (v, g) in sub.items():
            others = np.column_stack([sub[o][0] for o in sub if o != k])
            res["neuron_subscores"][k] = {"genes": g, "rho": round(float(stats.spearmanr(ca, v)[0]), 3),
                                          "partial_given_other_two": round(prs(ca, v, others), 3)}
        # 4. semipartial for case-control
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        res["n_age_imputed"] = int(age.isna().sum())
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        nall, _ = score(Zall, NEURON_MARKERS)
        ca_all = axes(Zall.loc[genes].values, cent)[0]
        cov, covn = np.column_stack([age, sex]), np.column_stack([age, sex, nall])
        yr, xr, ex = resid_rank(stage, cov), resid_rank(ca_all, cov), resid_rank(ca_all, covn)
        sr = float(np.cov(yr, ex)[0, 1] / (yr.std(ddof=1) * xr.std(ddof=1)))
        res["case_control"] = {"partial_agesex": round(rho(ca_all, stage, cov), 3),
                               "partial_agesex_neuron": round(rho(ca_all, stage, covn), 3),
                               "semipartial_agesex_neuron": round(sr, 3),
                               "share_CA_variance_left_after_neuron": round(float(ex.var() / xr.var()), 3)}
        # 5. GSE118553 asymptomatic AD
        if "AsymAD" in set(m["group"]):
            for lab, keep in {"AD_vs_control": m["group"].isin(["AD", "Control"]).values,
                              "AsymAD_vs_AD": m["group"].isin(["AD", "AsymAD"]).values,
                              "AsymAD_vs_control": m["group"].isin(["AsymAD", "Control"]).values}.items():
                y = (m["group"].values[keep] == {"AD_vs_control": "AD", "AsymAD_vs_AD": "AD", "AsymAD_vs_control": "AsymAD"}[lab]).astype(float)
                res[f"GSE118553_{lab}"] = {"n": int(keep.sum()), "CA": round(rho(ca_all[keep], y, cov[keep]), 3),
                                           "neuron": round(rho(nall[keep], y, cov[keep]), 3),
                                           "CA_given_neuron": round(rho(ca_all[keep], y, covn[keep]), 3)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    sp = [out[c]["case_control"]["semipartial_agesex_neuron"] for c in out]
    out["pooled_semipartial_mean_fisher"] = round(float(np.tanh(np.mean(np.arctanh(sp)))), 3)
    (a.out / "cohort_followups_v11.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
