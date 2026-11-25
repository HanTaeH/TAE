"""SPV-normalisation kappa against better-matched nulls (v11, review round 9: Reviewer 1 M5, Reviewer 2 M9).

For AD subjects in each MSBB region, labels before normalisation are compared (Cohen's kappa) with
labels after residualising the classifier genes on one extra covariate, with age, sex, PMI and pH:
  spv          : PC1 of the 14 neuronal markers (as in msbb_neuron_normalisation.py);
  random_gene  : the SAME 200 random non-classifier, non-marker genes in every region;
  random_pc1   : PC1 of random 14-gene sets whose R^2 with the C-A score is matched to the SPV's
                 (the 200 closest of 3,000 sets, drawn once and used in every region).
Per-region kappa has a subject-bootstrap 95% CI (resampling label pairs, 1,000 resamples).
Joint test across regions: the median SPV kappa across regions against the medians obtained with each
random covariate (the same covariate in every region). Reported for all 17 regions and for the 14
cortical regions (caudate, putamen and hippocampus excluded).

Usage: python msbb_kappa_v11.py --data /mnt/project-files/d1-data/raw --out ../results/msbb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import load_gse84422
import neff_subtypes
from msbb_neuron_normalisation import COV, kappa, project, resid, spv
from null_models import z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

NONCORTICAL = {"Caudate Nucleus", "Putamen", "Hippocampus"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-sets", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED + 911)
    e, m = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    g = cls.index.intersection(e.index)
    gi = e.index.get_indexer(g)
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    ng = [x for x in NEURON_MARKERS if x in e.index and x not in g]
    ni = e.index.get_indexer(ng)
    marker = set(NEURON_MARKERS) | {x for v in CELL_MARKERS.values() for x in v}
    pool = e.index.get_indexer(e.index.difference(g).difference(list(marker)))
    single = rng.choice(pool, a.n, replace=False)
    sets = [rng.choice(pool, len(ni), replace=False) for _ in range(a.n_sets)]
    rows, kr_single, kr_pc1 = [], {}, {}
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        mm = mm.dropna(subset=COV + ["cdr"])
        X = e[mm.index].values
        C = mm[COV].values.astype(float)
        ca0, lab0 = project(resid(X[gi], C), cent)
        s = spv(z_np(X[ni]))
        r2_spv = np.corrcoef(s, ca0)[0, 1] ** 2
        lab_spv = project(resid(X[gi], np.column_stack([s, C])), cent)[1]
        k_spv = kappa(lab0, lab_spv)
        pcs = np.array([spv(z_np(X[ix])) for ix in sets])
        r2 = np.array([np.corrcoef(p, ca0)[0, 1] ** 2 for p in pcs])
        near = np.argsort(np.abs(r2 - r2_spv))[:a.n]
        kp = np.array([kappa(lab0, project(resid(X[gi], np.column_stack([pcs[j], C])), cent)[1]) for j in near])
        ks = np.array([kappa(lab0, project(resid(X[gi], np.column_stack([X[j], C])), cent)[1]) for j in single])
        kr_single[reg], kr_pc1[reg] = ks, kp
        bs = []
        for _ in range(1000):
            b = rng.integers(0, len(lab0), len(lab0))
            bs.append(kappa(lab0[b], lab_spv[b]))
        rows.append({"region": reg, "cortical": reg not in NONCORTICAL, "n_AD": len(mm), "r2_spv": round(float(r2_spv), 3),
                     "kappa_spv": round(k_spv, 3), "kappa_spv_ci95": [round(float(np.nanquantile(bs, q)), 3) for q in (.025, .975)],
                     "kappa_single_gene_median": round(float(np.median(ks)), 3),
                     "kappa_r2_matched_pc1_median": round(float(np.median(kp)), 3),
                     "r2_matched_pc1_median_r2": round(float(np.median(r2[near])), 3),
                     "p_spv_le_single_gene": round(float(((ks <= k_spv).sum() + 1) / (a.n + 1)), 4),
                     "p_spv_le_r2_matched": round(float(((kp <= k_spv).sum() + 1) / (a.n + 1)), 4)})
        print(json.dumps(rows[-1]), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "msbb_kappa_v11.csv", index=False)
    summ = {}
    for lab, sub in {"all_17": d, "cortical_14": d[d.cortical]}.items():
        regs = list(sub.region)
        med_spv = float(np.median(sub.kappa_spv))
        med_single = np.median(np.array([kr_single[r] for r in regs]), axis=0)   # same gene j in every region
        res = {"n_regions": len(regs), "median_kappa_spv": round(med_spv, 3),
               "median_r2_spv": round(float(np.median(sub.r2_spv)), 3),
               "joint_single_gene": {"median_of_medians": round(float(np.median(med_single)), 3),
                                     "p": round(float(((med_single <= med_spv).sum() + 1) / (a.n + 1)), 4)},
               "median_kappa_r2_matched_pc1": round(float(np.median(sub.kappa_r2_matched_pc1_median)), 3),
               "n_regions_p_lt_05_single_gene": int((sub.p_spv_le_single_gene < .05).sum()),
               "n_regions_p_lt_05_r2_matched": int((sub.p_spv_le_r2_matched < .05).sum())}
        summ[lab] = res
    (a.out / "msbb_kappa_v11.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
