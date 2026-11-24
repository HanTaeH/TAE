"""Positive control for the specificity-matched random-gene null (follow-up, 2026-11-24).

Question: if the classifier genes DID carry a subtype or disease signal beyond cell composition, would
the specificity-matched null detect it? If yes, the negative results in real cohorts come from a test
that can discriminate, not from a test that can only return "no difference".

Simulated cohorts (each: 100 controls, 200 AD) are mixed from the SEA-AD reference as in
simulate_composition.py, with neuron loss l ~ U(0, 0.15) in controls and U(0.1, 0.6) in AD, Dirichlet
jitter of all shares and gene noise, so a neuronal axis exists as in real data. A composition-
independent program is then added to the classifier genes (log2 scale) along the published C-A contrast
(Neff centroids, C minus A, log CPM), scaled by f = 0, 0.02, 0.05, 0.1, 0.25, 0.5, 1 of that contrast:
  within_AD_subtype : each AD sample gets s ~ N(0, 1) x f x contrast (a true subtype axis)
  disease_shift     : every AD sample gets f x contrast (a true disease signal)
Projection, neuron marker score (16 markers, none in the classifier) and metrics as in real cohorts:
  within_AD_subtype : Spearman of the C-A score with the neuron score in AD; a true extra axis makes the
                      classifier LESS coupled than random genes -> upper-tail p
  disease_shift     : partial Spearman of the C-A score with diagnosis given the neuron score -> upper tail
Nulls: 200 random gene sets per simulated cohort inheriting the centroid rows, matched (a) on expression
decile and SEA-AD specificity class ("spec", as in the real cohorts) and (b) on expression tertile and the
log2 neuron/glia ratio in 1-unit bins ("strength", see gse33000_null_followup.py). Power = share of 20
simulated cohorts with p < 0.05; at f = 0 this is the false-positive rate.

Usage: python positive_control_simulation.py --data /mnt/project-files/d1-data/raw --ref /mnt/project-files/d1-data/seaad --out <dir>
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
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from simulate_composition import BASE, TYPES, mix, type_profiles

FRACS = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def props(n, lo, hi, rng):
    base = np.array([BASE[t] for t in TYPES])
    i = {t: k for k, t in enumerate(TYPES)}
    P = np.tile(base, (n, 1))
    l = rng.uniform(lo, hi, n)
    lost = P[:, i["Exc"]] * l + P[:, i["Inh"]] * 0.7 * l
    P[:, i["Exc"]] *= 1 - l
    P[:, i["Inh"]] *= 1 - 0.7 * l
    for t, f in [("Astro", .5), ("Micro", .25), ("Oligo", .15), ("OPC", .05), ("Vascular", .05)]:
        P[:, i[t]] += lost * f
    return np.array([rng.dirichlet(p * 150) for p in P])


def partial_rho(x, y, c):
    rx, ry, rc = (stats.rankdata(v) for v in (x, y, c))
    D = np.column_stack([np.ones(len(rc)), rc])
    ex = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
    ey = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 202)
    sub_ref = pd.read_csv(a.ref / "seaad_mtg_subclass_reference.csv.gz", index_col=0)
    sub_ref = sub_ref[~sub_ref.index.duplicated()]
    ncell = pd.read_csv(a.ref / "seaad_mtg_reference_ncells.csv", index_col=0).iloc[:, 0]
    prof = type_profiles(sub_ref, ncell)
    ref9 = pd.read_csv(a.ref / "seaad_mtg_reference.csv.gz", index_col=0)
    ref9 = ref9[~ref9.index.duplicated()]
    ratio = np.log2((ref9[NEURONAL].max(axis=1) + 1) / (ref9[[c for c in ref9.columns if c not in NEURONAL]].max(axis=1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref9.index).reindex(prof.index).fillna(3)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(prof.index)
    cent_df = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
    cent = cent_df.values
    contrast = (cls.loc[genes, "C"] - cls.loc[genes, "A"]).values
    gi = prof.index.get_indexer(genes)
    mk = prof.index.get_indexer([g for g in NEURON_MARKERS if g in prof.index and g not in genes])
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pool = prof.index.difference(genes).difference(list(marker))
    pool = pool[(prof.loc[pool].max(axis=1) > 1).values]           # expressed genes only
    dec = pd.qcut(np.log2(prof.mean(axis=1) + 1).rank(method="first"), 10, labels=False)
    ter = pd.qcut(np.log2(prof.mean(axis=1) + 1).rank(method="first"), 3, labels=False)
    rb = pd.cut(ratio, [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf], labels=False).reindex(prof.index)
    strata = {"spec": dec.astype(str) + "_" + spec.astype(int).astype(str),
              "strength": ter.astype(str) + "_" + rb.fillna(-1).astype(int).astype(str)}
    BY = {k: ({c: prof.index.get_indexer(pool[st.loc[pool].values == c]) for c in st.loc[genes].unique()},
              st.loc[genes].values) for k, st in strata.items()}
    pool_i = prof.index.get_indexer(pool)

    def ca(X, rows):
        r = corr_np(z_np(X[rows]), cent)
        return r[:, 2] - r[:, 0]

    out = {"n_genes": int(len(genes)), "contrast_sd_log": round(float(contrast.std()), 3), "results": {}}
    for scen in ("within_AD_subtype", "disease_shift"):
        out["results"][scen] = {}
        for f in FRACS:
            ps, obs_l, med_l = {k: [] for k in BY}, [], {k: [] for k in BY}
            for _ in range(a.reps):
                P = np.vstack([props(100, 0, .15, rng), props(200, .1, .6, rng)])
                ad = np.r_[np.zeros(100, bool), np.ones(200, bool)]
                X = mix(P, prof.values, rng)
                if scen == "within_AD_subtype":
                    s = rng.normal(0, 1, ad.sum())
                    X[np.ix_(gi, np.where(ad)[0])] += f * np.outer(contrast, s)
                else:
                    X[np.ix_(gi, np.where(ad)[0])] += f * contrast[:, None]
                Xa = X[:, ad]
                neuron_ad = z_np(Xa[mk]).mean(0)
                neuron_all = z_np(X[mk]).mean(0)

                def metric(rows):
                    if scen == "within_AD_subtype":
                        return stats.spearmanr(ca(Xa, rows), neuron_ad)[0]
                    return partial_rho(ca(X, rows), ad.astype(float), neuron_all)
                o = metric(gi)
                obs_l.append(o)
                for kind, (by, gst) in BY.items():
                    nl = []
                    for _ in range(a.n_null):
                        pick = np.empty(len(gi), dtype=int)
                        for k in np.unique(gst):
                            idx = np.where(gst == k)[0]
                            src = by.get(k, pool_i)
                            src = src if len(src) else pool_i
                            pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
                        nl.append(metric(pick))
                    nl = np.array(nl)
                    ps[kind].append(((nl >= o).sum() + 1) / (a.n_null + 1))
                    med_l[kind].append(np.median(nl))
            res = {"obs_median": round(float(np.median(obs_l)), 3)}
            for kind in BY:
                pk = np.array(ps[kind])
                res[kind] = {"null_median": round(float(np.median(med_l[kind])), 3),
                             "p_median": round(float(np.median(pk)), 3),
                             "share_p_lt_0.05": round(float((pk < .05).mean()), 2)}
            out["results"][scen][str(f)] = res
            print(scen, f, json.dumps(out["results"][scen][str(f)]), flush=True)
    (a.out / "positive_control_simulation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
