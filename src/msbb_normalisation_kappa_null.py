"""How unstable are projected MSBB labels to any one-covariate normalisation? (follow-up, 2026-11-24)

msbb_neuron_normalisation.py reports kappa 0.48 (median, 17 regions) between labels before and after
residualising every gene on the neuronal SPV. With ~30 AD subjects per region and nearest-centroid labels,
some reassignment follows from any extra covariate. Here, per region, the same residualisation is done with
  - CDR (the normalisation Neff et al. used),
  - 200 random genes (not classifier, not marker), each used as the single extra covariate,
and kappa against the labels before normalisation is reported, with the SPV kappa for comparison
(p = share of random genes giving kappa <= the SPV kappa).

Usage: python msbb_normalisation_kappa_null.py --data /mnt/project-files/d1-data/raw --out <dir>
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 91)
    e, m = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    g = cls.index.intersection(e.index)
    gi = e.index.get_indexer(g)
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    ng = [x for x in NEURON_MARKERS if x in e.index and x not in g]
    ni = e.index.get_indexer(ng)
    marker = set(NEURON_MARKERS) | {x for v in CELL_MARKERS.values() for x in v}
    pool = e.index.get_indexer(e.index.difference(g).difference(list(marker)))
    rows = []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        mm = mm.dropna(subset=COV + ["cdr"])
        X = e[mm.index].values
        C = mm[COV].values.astype(float)
        _, lab0 = project(resid(X, C)[gi], cent)
        s = spv(z_np(X[ni]))
        k_spv = kappa(lab0, project(resid(X, np.column_stack([s, C]))[gi], cent)[1])
        k_cdr = kappa(lab0, project(resid(X, np.column_stack([mm["cdr"].values.astype(float), C]))[gi], cent)[1])
        kr = np.array([kappa(lab0, project(resid(X, np.column_stack([X[j], C]))[gi], cent)[1])
                       for j in rng.choice(pool, a.n, replace=False)])
        rows.append({"region": reg, "n_AD": len(mm), "kappa_spv": k_spv, "kappa_cdr": k_cdr,
                     "kappa_random_median": float(np.median(kr)), "kappa_random_q05": float(np.quantile(kr, .05)),
                     "p_spv_le_random": float(((kr <= k_spv).sum() + 1) / (a.n + 1))})
        print(json.dumps(rows[-1]), flush=True)
    d = pd.DataFrame(rows).round(3)
    d.to_csv(a.out / "msbb_normalisation_kappa_null.csv", index=False)
    summ = d.drop(columns="region").median().round(3).to_dict()
    (a.out / "msbb_normalisation_kappa_null.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
