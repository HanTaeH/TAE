"""Why GSE33000 alone exceeded the specificity-matched null for the neuron association (follow-up, 2026-11-24).

Two checks on GSE33000 (AD samples):
  1. Power: subsample AD to the size of each other cohort (97, 52, 219) 30 times and recompute the
     observed neuron rho and its specificity-matched null (200 sets, same gene draws for every subsample).
  2. Finer match: replace each glia-enriched classifier gene by a gene enriched in the same top non-neuronal
     type (Astro, Oligo, OPC, Micro, Vascular) and each neuron-enriched gene by one with the same top neuronal
     type, within the same expression decile (500 sets, full sample).
  3. Strength match: classifier glia-enriched genes are far more specific than the pool (median |log2 ratio|
     3.9 vs 2.0), so match on the log2 neuron/glia ratio in 1-unit bins (open-ended beyond +/-4) within
     expression tertile (500 sets, full sample), and in the other three independent cohorts for comparison.

Usage: python gse33000_null_followup.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out <dir>
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
from meta_reanalysis import ca_score
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts


def draw(rng, genes, stratum, pool, n):
    by = {k: pool[stratum.loc[pool].values == k] for k in stratum.loc[genes].unique()}
    gst = stratum.loc[genes].values
    out = []
    for _ in range(n):
        pick = np.empty(len(genes), dtype=object)
        for k in np.unique(gst):
            idx = np.where(gst == k)[0]
            src = by.get(k, pool)
            if len(src) == 0:
                src = pool
            pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
        out.append(list(pick))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 77)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    neu, gli = ref[NEURONAL].max(axis=1), ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1)
    ratio = np.log2((neu + 1) / (gli + 1))
    spec3 = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    top = ref.idxmax(axis=1)
    fine = spec3.astype(str)
    fine[spec3 == 0] = "g_" + ref.loc[spec3 == 0, [c for c in ref.columns if c not in NEURONAL]].idxmax(axis=1)
    fine[spec3 == 2] = "n_" + ref.loc[spec3 == 2, NEURONAL].idxmax(axis=1)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    name, e, m = next(c for c in cohorts(a.data) if c[0] == "GSE33000_PFC")
    e = e[m.index]
    e = e.loc[:, (m["group"] == "AD").values]
    genes = list(cls.index.intersection(e.index))
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
    mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
    dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
    pool = e.index.difference(genes).difference(list(marker))
    s3 = dec.astype(str) + "_" + spec3.reindex(e.index).fillna(3).astype(int).astype(str)
    sf = dec.astype(str) + "_" + fine.reindex(e.index).fillna("absent")

    def nrho(Z, gs):
        return stats.spearmanr(ca_score(Z.loc[gs].set_axis(genes), cent).values, Z.loc[mk].mean().values)[0]

    out = {"n_ad": int(e.shape[1]), "fine_mix": fine.reindex(genes).fillna("absent").value_counts().to_dict()}
    # 2. finer match, full sample
    Z = zscore_rows(e)
    obs = nrho(Z, genes)
    null = np.array([nrho(Z, g) for g in draw(rng, genes, sf, pool, 500)])
    out["finer_match"] = {"neff": round(obs, 3), "random_median": round(float(np.median(null)), 3),
                          "random_q05_q95": [round(float(np.quantile(null, q)), 3) for q in (.05, .95)],
                          "p_empirical": round(float(((null <= obs).sum() + 1) / 501), 4)}
    print(json.dumps(out["finer_match"]), flush=True)
    bins = [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    rb = pd.cut(ratio, bins, labels=False).astype(str)
    out["strength_match"] = {}
    for cname, ce, cm in cohorts(a.data):
        if cname not in ("GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE125583_FG"):
            continue
        ce = ce[cm.index].loc[:, (cm["group"] == "AD").values]
        cg = list(cls.index.intersection(ce.index))
        cc = cls.loc[cg].sub(cls.loc[cg].mean(axis=1), axis=0)
        cmk = [x for x in NEURON_MARKERS if x in ce.index and x not in cg]
        ter = pd.qcut(ce.mean(axis=1).rank(method="first"), 3, labels=False).astype(str)
        st = ter + "_" + rb.reindex(ce.index).fillna("absent")
        cpool = ce.index.difference(cg).difference(list(marker))
        Zc = zscore_rows(ce)

        def cr(gs):
            return stats.spearmanr(ca_score(Zc.loc[gs].set_axis(cg), cc).values, Zc.loc[cmk].mean().values)[0]
        o = cr(cg)
        nl = np.array([cr(x) for x in draw(rng, cg, st, cpool, 500)])
        out["strength_match"][cname] = {"neff": round(o, 3), "random_median": round(float(np.median(nl)), 3),
                                        "random_q05_q95": [round(float(np.quantile(nl, q)), 3) for q in (.05, .95)],
                                        "p_empirical": round(float(((nl <= o).sum() + 1) / 501), 4)}
        print(cname, json.dumps(out["strength_match"][cname]), flush=True)
    # 1. power by subsampling
    picks = draw(rng, genes, s3, pool, 200)
    out["subsample"] = {}
    for n in (52, 97, 219):
        res = []
        for _ in range(30):
            cols = rng.choice(e.shape[1], n, replace=False)
            Zs = zscore_rows(e.iloc[:, cols])
            o = nrho(Zs, genes)
            nl = np.array([nrho(Zs, g) for g in picks])
            res.append({"obs": o, "med": float(np.median(nl)), "p": float(((nl <= o).sum() + 1) / 201)})
        r = pd.DataFrame(res)
        out["subsample"][n] = {"obs_median": round(r.obs.median(), 3), "null_median": round(r.med.median(), 3),
                               "p_median": round(r.p.median(), 3), "share_p_lt_0.05": round(float((r.p < .05).mean()), 2)}
        print(n, json.dumps(out["subsample"][n]), flush=True)
    (a.out / "gse33000_null_followup.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
