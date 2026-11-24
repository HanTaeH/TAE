"""Tijms 2024 CSF subtypes in bulk brain, follow-up (2026-11-24).

A. Strength-matched null for the neuron coupling of r(S1)..r(S5): random genes matched on expression
   tertile x SEA-AD log2 neuron/glia ratio in 1-unit bins (as gse33000_null_followup.py), next to the
   class-matched ("spec") null of tijms_csf_projection.py.
B. Shared signal beyond composition: Tijms genes that are also Neff classifier genes are removed; the
   Tijms genes and, separately, the Neff genes are residualised on the five marker scores; r(Sk) from
   the residual Tijms genes is correlated (Spearman) with the residual Neff C-A score. Null: random genes
   (spec and strength matched) in place of the Tijms genes, same residualisation.
Two-sided p on |rho|, 300 sets per null.

Usage: python tijms_followup.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out <dir>
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
from meta_reanalysis import INDEPENDENT
from null_models import corr_np, z_np
from pilot_gse33000 import SEED
from residual_identity import MARKERS, cell_scores
from residual_structure import residualise
from tijms_csf_projection import SUB, load_tijms
from validate_cohorts import cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 505)
    cent_all = load_tijms(a.data / "tijms2024" / "43587_2023_550_MOESM3_ESM.xlsx")
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    rbin = pd.cut(ratio, [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf], labels=False)
    out = {"n_overlap_with_neff_classifier": int(len(cent_all.index.intersection(cls.index))), "cohorts": []}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e.loc[e.index.notna() & ~e.index.duplicated()].dropna()
        e = e[m.index.intersection(e.columns)]
        e = e.loc[:, (m.loc[e.columns, "group"] == "AD").values]
        ngenes = cls.index.intersection(e.index)
        cells = cell_scores(e, e.columns, set(ngenes) | set(cent_all.index))
        neuron = cells[:, 0]
        ncent = cls.loc[ngenes].sub(cls.loc[ngenes].mean(axis=1), axis=0).values
        rn = corr_np(z_np(residualise(z_np(e.loc[ngenes].values), cells)), ncent)
        neff_res = rn[:, 2] - rn[:, 0]
        tg = cent_all.index.intersection(e.index).difference(ngenes)
        tcent = cent_all.loc[tg].values
        mu = e.mean(axis=1)
        dec = pd.qcut(mu.rank(method="first"), 10, labels=False)
        ter = pd.qcut(mu.rank(method="first"), 3, labels=False)
        strata = {"spec": dec.astype(str) + "_" + spec.reindex(e.index).fillna(3).astype(int).astype(str),
                  "strength": ter.astype(str) + "_" + rbin.reindex(e.index).fillna(-1).astype(int).astype(str)}
        pool = e.index.difference(cent_all.index).difference(ngenes).difference(list(MARKERS))
        Zall = z_np(e.values)
        gpos = pd.Series(np.arange(len(e.index)), index=e.index)

        def metrics(rows):
            X = Zall[rows]
            raw = corr_np(z_np(X), tcent)
            res = corr_np(z_np(residualise(X, cells)), tcent)
            return ([stats.spearmanr(raw[:, j], neuron)[0] for j in range(len(SUB))],
                    [stats.spearmanr(res[:, j], neff_res)[0] for j in range(len(SUB))])

        obs_n, obs_r = metrics(gpos[tg].values)
        row = {"cohort": name, "n_AD": int(e.shape[1]), "tijms_genes_used": int(len(tg)), "subtypes": {}}
        nulls = {}
        for kind, st in strata.items():
            gst = st.loc[tg].values
            by = {k: gpos[pool[st.loc[pool].values == k]].values for k in np.unique(gst)}
            pool_i = gpos[pool].values
            dn, dr = [], []
            for _ in range(a.n):
                pick = np.empty(len(tg), dtype=int)
                for k in np.unique(gst):
                    idx = np.where(gst == k)[0]
                    src = by[k] if len(by[k]) else pool_i
                    pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
                n_, r_ = metrics(pick)
                dn.append(np.abs(n_))
                dr.append(np.abs(r_))
            nulls[kind] = (np.array(dn), np.array(dr))
        for j, s in enumerate(SUB):
            d = {"rho_neuron": round(float(obs_n[j]), 3), "rho_resid_vs_neff_resid_CA": round(float(obs_r[j]), 3)}
            for kind, (dn, dr) in nulls.items():
                d[f"neuron_{kind}_median"] = round(float(np.median(dn[:, j])), 3)
                d[f"neuron_{kind}_p"] = round(float(((dn[:, j] >= abs(obs_n[j])).sum() + 1) / (a.n + 1)), 4)
                d[f"resid_{kind}_median"] = round(float(np.median(dr[:, j])), 3)
                d[f"resid_{kind}_p"] = round(float(((dr[:, j] >= abs(obs_r[j])).sum() + 1) / (a.n + 1)), 4)
            row["subtypes"][s] = d
        out["cohorts"].append(row)
        print(json.dumps(row), flush=True)
    (a.out / "tijms_followup.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
