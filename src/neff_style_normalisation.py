"""Does Neff 2021's own correction remove the neuron axis? (MSBB GSE84422, AD samples per region)

Neff normalised expression by CDR (and age, sex, PMI, pH here) before subtyping, stating that this
removes the confounding signal of neuronal loss. Here every gene is residualised on CDR + age + sex +
PMI + pH within the AD samples of each region (as in Neff), then the Neff classes are projected and
compared with the neuron marker score computed on the same normalised data.

Usage: python neff_style_normalisation.py --data /mnt/project-files/d1-data/raw --out ../results/msbb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_gse84422
import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import NEURON_MARKERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    e, m = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    g = cls.index.intersection(e.index)
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    ng = [x for x in NEURON_MARKERS if x in e.index and x not in g]
    rows = []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        X = e[mm.index].values
        cdr = mm["cdr"].values.astype(float)
        D = np.column_stack([np.ones(len(cdr)), cdr, mm[["age", "female", "pmi", "ph"]].values]).astype(float)
        Xr = pd.DataFrame(X - X @ D @ np.linalg.pinv(D), index=e.index)
        neu_raw = z_np(e.loc[ng, mm.index].values).mean(0)
        neu = z_np(Xr.loc[ng].values).mean(0)
        r = corr_np(z_np(Xr.loc[g].values), cent)
        ca = r[:, 2] - r[:, 0]
        rows.append({"region": reg, "n_AD": len(mm),
                     "neuron_raw_vs_cdr": stats.spearmanr(neu_raw, cdr)[0],
                     "neuron_normalised_vs_cdr": stats.spearmanr(neu, cdr)[0],
                     "ca_normalised_vs_cdr": stats.spearmanr(ca, cdr)[0],
                     "ca_normalised_vs_neuron_normalised": stats.spearmanr(ca, neu)[0],
                     "frac_A": float((r.argmax(1) == 0).mean()), "frac_C": float((r.argmax(1) == 2).mean())})
    d = pd.DataFrame(rows).set_index("region").round(3)
    d.to_csv(a.out / "neff_style_cdr_normalisation.csv")
    summ = {k: round(float(v), 3) for k, v in d.median().items()}
    (a.out / "neff_style_cdr_normalisation.json").write_text(json.dumps({"median_over_17_regions": summ}, indent=2))
    print(d.to_string())
    print(summ)


if __name__ == "__main__":
    main()
