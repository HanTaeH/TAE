"""Upper-tail p-values for the positive-control bound (v10, 2026-11-25; internal review round 7).

The positive control (positive_control_simulation.py) shows that a composition-independent subtype
signal makes the classifier LESS neuron-coupled than random genes, detected in the upper tail. The real-
cohort tests reported so far were lower-tail (MORE coupled). To bound the composition-independent signal
in real data, the observed neuron rho is compared with the same nulls in the upper tail:
  spec     : saved specificity-matched draws (null_specificity_matched_distributions.csv, 500 per cohort)
  strength : 500 fresh strength-matched draws per cohort (expression tertile x 1-unit log2 neuron/glia bin),
             as in gse33000_null_followup.py; both tails reported.
p_upper = (#{null >= obs} + 1)/(N + 1).

Usage: python bound_upper_tail.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --res ../results --out <dir>
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1010)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    rb = pd.cut(ratio, [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf], labels=False).astype(str)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    spec_rep = json.loads((a.res / "null_specificity/null_specificity_matched_report.json").read_text())
    dist = pd.read_csv(a.res / "null_specificity/null_specificity_matched_distributions.csv")
    out = {}
    for cname, ce, cm in cohorts(a.data):
        if cname not in COH:
            continue
        ce = ce[cm.index].loc[:, (cm["group"] == "AD").values]
        cg = list(cls.index.intersection(ce.index))
        cc = cls.loc[cg].sub(cls.loc[cg].mean(axis=1), axis=0)
        cmk = [x for x in NEURON_MARKERS if x in ce.index and x not in cg]
        ter = pd.qcut(ce.mean(axis=1).rank(method="first"), 3, labels=False).astype(str)
        st = ter + "_" + rb.reindex(ce.index).fillna("absent")
        pool = ce.index.difference(cg).difference(list(marker))
        Z = zscore_rows(ce)

        def cr(gs):
            return stats.spearmanr(ca_score(Z.loc[gs].set_axis(cg), cc).values, Z.loc[cmk].mean().values)[0]
        o = cr(cg)
        nl = np.array([cr(x) for x in draw(rng, cg, st, pool, 500)])
        sp = dist.loc[dist.cohort == cname, "neuron_rho"].values
        res = {"n_ad": int(ce.shape[1]), "obs": round(float(o), 3),
               "strength": {"null_median": round(float(np.median(nl)), 3),
                            "p_lower": round(float(((nl <= o).sum() + 1) / 501), 4),
                            "p_upper": round(float(((nl >= o).sum() + 1) / 501), 4)}}
        if len(sp):
            res["spec"] = {"null_median": round(float(np.median(sp)), 3), "n": int(len(sp)),
                           "p_lower": round(float(((sp <= o).sum() + 1) / (len(sp) + 1)), 4),
                           "p_upper": round(float(((sp >= o).sum() + 1) / (len(sp) + 1)), 4)}
        out[cname] = res
        print(cname, json.dumps(res), flush=True)
    (a.out / "bound_upper_tail.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
