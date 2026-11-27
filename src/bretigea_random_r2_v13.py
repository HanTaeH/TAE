"""How much of the C-A score do six arbitrary BRETIGEA-style components explain? (v13)

bretigea_normalisation_v12.py reports R^2 of the raw C-A score on the six BRETIGEA surrogate proportion
variables (SPVs). Any six first PCs of 1,000-gene sets share the dominant expression axis, so here the same
R^2 is computed for 100 draws of six components from random 1,000-gene sets (same pool as the kappa null)
and for one-component versions (the neuron SPV alone vs one random component).

Usage: python bretigea_random_r2_v13.py --data /mnt/project-files/d1-data/raw --out ../results/bretigea_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import load_gse84422
import neff_subtypes
from bretigea_normalisation_v12 import CELLS, find_cells, load_markers, spvs
from meta_reanalysis import INDEPENDENT
from msbb_neuron_normalisation import project, resid
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import cohorts


def r2(y, F):
    D = np.column_stack([np.ones(len(y)), F])
    return float(1 - ((y - D @ np.linalg.lstsq(D, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def one(X, index, gi, cent, C, markers, pool, rng, n):
    X = np.nan_to_num(np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X))
    S, _ = spvs(X, index, markers)
    ca0, _ = project(resid(X[gi], C), cent)
    r6, r1 = [], []
    for _ in range(n):
        R = np.column_stack([find_cells(X, list(rng.choice(pool, 1000, replace=False))) for _ in CELLS])
        r6.append(r2(ca0, R))
        r1.append(r2(ca0, R[:, :1]))
    r6, r1 = np.array(r6), np.array(r1)
    obs6, obs1 = r2(ca0, S), r2(ca0, S[:, 3:4])
    return {"r2_6spv": round(obs6, 3), "r2_6random_median": round(float(np.median(r6)), 3),
            "r2_6random_q05_q95": [round(float(np.quantile(r6, q)), 3) for q in (.05, .95)],
            "p_random6_ge": round(float(((r6 >= obs6).sum() + 1) / (n + 1)), 4),
            "r2_neuron_spv": round(obs1, 3), "r2_1random_median": round(float(np.median(r1)), 3),
            "p_random1_ge": round(float(((r1 >= obs1).sum() + 1) / (n + 1)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1308)
    markers = load_markers(a.data / "bretigea" / "markers_df_human_brain.rda")
    allmk = {g for v in markers.values() for g in v} | set(NEURON_MARKERS) | {x for v in CELL_MARKERS.values() for x in v}
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    def setup(e):
        g = cls.index.intersection(e.index)
        index = {x: i for i, x in enumerate(e.index)}
        cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
        pool = e.index.get_indexer(e.index.difference(g).difference(list(allmk)))
        return index, e.index.get_indexer(g), cent, pool

    rep = {"msbb": {}, "geo": {}}
    e, m = load_gse84422.load(a.data)
    index, gi, cent, pool = setup(e)
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        mm = mm.dropna(subset=["age", "female", "pmi", "ph"])
        rep["msbb"][reg] = one(e[mm.index].values, index, gi, cent, mm[["age", "female", "pmi", "ph"]].values.astype(float),
                               markers, pool, rng, a.n)
        print(reg, rep["msbb"][reg], flush=True)
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        index, gi, cent, pool = setup(e)
        ad = (m["group"] == "AD").values
        mm = m.loc[ad]
        age = pd.to_numeric(mm["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (mm["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        rep["geo"][name] = one(e.loc[:, ad].values, index, gi, cent, np.column_stack([age, sex]), markers, pool, rng, a.n)
        print(name, rep["geo"][name], flush=True)
    d = pd.DataFrame(rep["msbb"]).T
    rep["msbb_summary"] = {c: round(float(pd.to_numeric(d[c]).median()), 3)
                           for c in ["r2_6spv", "r2_6random_median", "r2_neuron_spv", "r2_1random_median"]}
    rep["msbb_summary"]["n_regions_p_random6_lt_05"] = int((pd.to_numeric(d["p_random6_ge"]) < .05).sum())
    (a.out / "bretigea_random_r2_v13.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["msbb_summary"]))


if __name__ == "__main__":
    main()
