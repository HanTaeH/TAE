"""Spike-in positive control in the real cohorts and a test-inversion upper confidence limit (v11, review round 9).

Reviewer 1 (round 9) noted that simulated power does not carry over to real cohorts, whose matched nulls are
wider, and that "not significant + high simulated power" is not a confidence limit. Here a composition-
independent within-AD subtype signal is added to the OBSERVED expression of the classifier genes in AD
samples of each independent cohort:
    X[g, j] += f * s_j * contrast_g,   s_j ~ N(0, 1),  contrast = Neff centroid C - A (log CPM)
Random genes are not spiked, so the real null distributions are unchanged. For each f (and 200 replicates,
each with a bootstrap resample of AD samples and a fresh s):
  - rho_f: Spearman(C-A score, neuron score); dvar: share of C-A variance explained by s (R^2);
  - detection: upper-tail p < 0.05 against the real specificity- and strength-matched nulls;
  - the shift delta_rho(f) and its bootstrap spread, used by spike_in_bound.py, which inverts the test
    against the matched nulls (one-sided 95% upper confidence limit).
Also reported (Reviewer 1 M2): number of classifier genes whose matching stratum was empty (fallback to the
whole pool) or smaller than needed (draws with replacement), and the strength-matched null with expression
deciles instead of tertiles, so that only the specificity matching changes.

Usage: python spike_in_real.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --res ../results --out <dir>
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
FRACS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5)


def strata_report(genes, stratum, pool):
    by = {k: (stratum.loc[pool].values == k).sum() for k in stratum.loc[genes].unique()}
    gst = pd.Series(stratum.loc[genes].values)
    need = gst.value_counts()
    fallback = int(sum(need[k] for k in need.index if by.get(k, 0) == 0))
    repl = int(sum(need[k] for k in need.index if 0 < by.get(k, 0) < need[k]))
    return {"n_strata": int(len(need)), "genes_fallback_whole_pool": fallback, "genes_in_strata_drawn_with_replacement": repl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1111)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    rb = pd.cut(ratio, [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf], labels=False).astype(str)
    sc = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    dist = pd.read_csv(a.res / "null_specificity/null_specificity_matched_distributions.csv")
    out, rows = {}, []
    for cname, ce, cm in cohorts(a.data):
        if cname not in COH:
            continue
        ce = ce[cm.index].loc[:, (cm["group"] == "AD").values]
        cg = list(cls.index.intersection(ce.index))
        cc = cls.loc[cg].sub(cls.loc[cg].mean(axis=1), axis=0)
        contrast = (cls.loc[cg, "C"] - cls.loc[cg, "A"]).values
        cmk = [x for x in NEURON_MARKERS if x in ce.index and x not in cg]
        mean = ce.mean(axis=1).rank(method="first")
        ter, dec = pd.qcut(mean, 3, labels=False).astype(str), pd.qcut(mean, 10, labels=False).astype(str)
        st_t = ter + "_" + rb.reindex(ce.index).fillna("absent")
        st_d = dec + "_" + rb.reindex(ce.index).fillna("absent")
        st_s = dec + "_" + sc.reindex(ce.index).fillna(3).astype(int).astype(str)
        pool = ce.index.difference(cg).difference(list(marker))
        Z = zscore_rows(ce)
        neu = Z.loc[cmk].mean().values

        def cr(gs):
            return stats.spearmanr(ca_score(Z.loc[gs].set_axis(cg), cc).values, neu)[0]
        obs = cr(cg)
        null_spec = dist.loc[dist.cohort == cname, "neuron_rho"].values
        null_str = np.array([cr(x) for x in draw(rng, cg, st_t, pool, a.n_null)])
        null_strd = np.array([cr(x) for x in draw(rng, cg, st_d, pool, a.n_null)])
        X = ce.loc[cg].values.astype(float)
        nA = X.shape[1]
        res = {"n_ad": nA, "n_genes": len(cg), "obs_rho": round(float(obs), 3),
               "nulls": {k: {"median": round(float(np.median(v)), 3),
                             "q05_q95": [round(float(np.quantile(v, .05)), 3), round(float(np.quantile(v, .95)), 3)],
                             "offset_obs_minus_median": round(float(obs - np.median(v)), 3),
                             "p_lower": round(float(((v <= obs).sum() + 1) / (len(v) + 1)), 4),
                             "p_upper": round(float(((v >= obs).sum() + 1) / (len(v) + 1)), 4)}
                         for k, v in (("spec", null_spec), ("strength_tertile", null_str), ("strength_decile", null_strd))},
               "strata": {"spec": strata_report(cg, st_s, pool), "strength_tertile": strata_report(cg, st_t, pool),
                          "strength_decile": strata_report(cg, st_d, pool)},
               "spike": []}
        q95 = {k: np.quantile(v, .95) for k, v in (("spec", null_spec), ("strength_tertile", null_str))}
        for f in FRACS:
            rr, dv, det = [], [], {k: 0 for k in q95}
            for _ in range(a.reps):
                b = rng.integers(0, nA, nA)
                s = rng.standard_normal(nA)
                Xb = X[:, b] + f * np.outer(contrast, s)
                Zb = zscore_rows(pd.DataFrame(Xb, index=cg))
                score = ca_score(Zb, cc).values
                nb = zscore_rows(ce.loc[cmk].iloc[:, b]).mean().values
                r = stats.spearmanr(score, nb)[0]
                rr.append(r)
                dv.append(np.corrcoef(score, s)[0, 1] ** 2 if f > 0 else 0.0)
                for k in q95:
                    p_up = ((null_spec if k == "spec" else null_str) >= r).sum() + 1
                    det[k] += p_up / ((len(null_spec) if k == "spec" else len(null_str)) + 1) < 0.05
            rr = np.array(rr)
            row = {"f": f, "rho_median": round(float(np.median(rr)), 3),
                   "rho_q05": round(float(np.quantile(rr, .05)), 3),
                   "delta_rho": round(float(np.median(rr) - obs), 3),
                   "var_share_median": round(float(np.median(dv)), 3),
                   "detect_spec": round(det["spec"] / a.reps, 3),
                   "detect_strength": round(det["strength_tertile"] / a.reps, 3)}
            res["spike"].append(row)
            rows.append({"cohort": cname, **row})
            print(cname, json.dumps(row), flush=True)
        out[cname] = res
    pd.DataFrame(rows).to_csv(a.out / "spike_in_real.csv", index=False)
    (a.out / "spike_in_real.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
