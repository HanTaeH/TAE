"""Guard against circularity of the neuron marker score: split-half markers.

For each cohort (AD samples) the neuron markers (Neff genes excluded) are split at random into two
halves, 500 times. Half 1 is used for adjustment, half 2 only for evaluation:
  raw      rho(C-minus-A, half-2 score)
  adjusted rho(C-minus-A after residualising every Neff gene on the half-1 score, half-2 score)
  marker   rho(half-1 score, half-2 score)  (agreement of the two independent marker sets)
If the axis is neuronal content and not an artefact of one marker set, the raw correlation with the
held-out half stays strong and adjustment with the other half removes most of it.

Usage: python marker_split.py --data /mnt/project-files/d1-data/raw --out ../results/marker_split --n 500
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
from pilot_gse33000 import NEURON_MARKERS, SEED
from residual_structure import residualise
from validate_cohorts import cohorts


def ca_of(z, cent):
    r = corr_np(z, cent)
    return r[:, 2] - r[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows = []
    for name, expr, meta in cohorts(a.data):
        ad = meta.index[meta["group"] == "AD"]
        genes = cls.index.intersection(expr.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        mk = [g for g in NEURON_MARKERS if g in expr.index and g not in genes]
        Zm = pd.DataFrame(z_np(expr.loc[mk, ad].values), index=mk)
        z = z_np(expr.loc[genes, ad].values)
        ca = ca_of(z, cent)
        raw, adj, agree = [], [], []
        for _ in range(a.n):
            p = rng.permutation(mk)
            h1, h2 = p[: len(p) // 2], p[len(p) // 2:]
            s1, s2 = Zm.loc[h1].values.mean(0), Zm.loc[h2].values.mean(0)
            raw.append(stats.spearmanr(ca, s2)[0])
            adj.append(stats.spearmanr(ca_of(z_np(residualise(z, s1[:, None])), cent), s2)[0])
            agree.append(stats.spearmanr(s1, s2)[0])
        q = lambda v: [round(float(np.median(v)), 3), round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]
        rows.append({"cohort": name, "n_AD": int(len(ad)), "n_markers": len(mk),
                     "raw_vs_heldout_median_lo_hi": q(raw), "adjusted_vs_heldout_median_lo_hi": q(adj),
                     "half_agreement_median_lo_hi": q(agree),
                     "frac_signal_removed": round(float(1 - np.median(np.abs(adj)) / np.median(np.abs(raw))), 3)})
        print(json.dumps(rows[-1]), flush=True)
    (a.out / "marker_split_report.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(a.out / "marker_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
