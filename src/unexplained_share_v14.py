"""Observed share of C-A variance not explained by the five marker scores, against matched nulls (v14).

Companion to spike_small_signal_v14.py: the observed test only, with 1,000 sets per null and both the
specificity-matched (expression decile x specificity class) and strength-matched (expression tertile x 1-unit
log2 neuron-to-glia bins) nulls.

Usage: python unexplained_share_v14.py --data <raw> --ref <seaad_mtg_reference.csv.gz> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from classifier_specificity import NEURONAL
from gse33000_null_followup import draw
from meta_reanalysis import ca_score
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from spike_small_signal_v14 import COH, cell_matrix, unexplained
from validate_cohorts import cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1616)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    sc = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    rb = pd.cut(ratio, [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf], labels=False).astype(str)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    out = {}
    for cname, ce, cm in cohorts(a.data):
        if cname not in COH:
            continue
        ce = ce[cm.index].loc[:, (cm["group"] == "AD").values]
        cg = list(cls.index.intersection(ce.index))
        cc = cls.loc[cg].sub(cls.loc[cg].mean(axis=1), axis=0)
        mean = ce.mean(axis=1).rank(method="first")
        dec, ter = pd.qcut(mean, 10, labels=False).astype(str), pd.qcut(mean, 3, labels=False).astype(str)
        st = {"spec": dec + "_" + sc.reindex(ce.index).fillna(3).astype(int).astype(str),
              "strength_tertile": ter + "_" + rb.reindex(ce.index).fillna("absent")}
        pool = ce.index.difference(cg).difference(list(marker))
        Z = zscore_rows(ce)
        M = cell_matrix(Z, set(cg))
        obs = unexplained(ca_score(Z.loc[cg], cc).values, M)
        row = {"n_ad": int(ce.shape[1]), "obs_unexplained": round(obs, 4)}
        for k, s in st.items():
            nv = np.array([unexplained(ca_score(Z.loc[g].set_axis(cg), cc).values, M) for g in draw(rng, cg, s, pool, a.n_null)])
            row[k] = {"null_median": round(float(np.median(nv)), 4), "null_q95": round(float(np.quantile(nv, .95)), 4),
                      "excess_over_median": round(float(obs - np.median(nv)), 4),
                      "p_upper": round(float(((nv >= obs).sum() + 1) / (len(nv) + 1)), 4)}
        out[cname] = row
        print(cname, json.dumps(row), flush=True)
    (a.out / "unexplained_share_v14.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
