"""Does the coupling come from the direction the classifier gives each cell-type class? (v12, review round 10, Reviewer 2)

The cell-type-matched null replaces each classifier gene by a random gene of the same neuron/glia
specificity class and expression decile, and the random gene inherits the classifier gene's centroid row.
It therefore keeps the classifier's direction per class (neuron-enriched genes higher in class A,
glia-enriched genes higher in class C), which is itself the biology in question. In AD samples of the four
independent cohorts, the coupling of the C-A score with the neuron score (Spearman) is computed for:
  observed        : the classifier;
  class_mean_rows : every classifier gene gets the mean centroid row of its specificity class
                    (only the class-level direction remains, gene-level information removed);
  perm_in_class   : classifier genes, centroid rows permuted among genes of the same specificity class
                    (500 permutations; keeps the class-level direction, scrambles gene-level assignment);
  sign_random     : classifier genes, each centroid row multiplied by a random sign (500 draws;
                    removes the direction);
  spec_matched    : random genes matched on specificity class and expression decile, inheriting rows.
If observed equals class_mean_rows and lies inside perm_in_class, the gene-level information of the
classifier adds nothing to the neuronal coupling beyond the direction given to each cell-type class.
Subtype contrasts within classes, C1-C2 and B1-B2 (subtype centroids of Data file S2): Spearman with the
neuron score and the four glial scores, against perm_in_class and spec_matched (two-sided p).

Usage: python null_direction_v12.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out ../results/null_direction_v12
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
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts

SUB = ["A", "B1", "B2", "C1", "C2"]


def contrasts(Z, cent_cls, cent_sub):
    r = corr_np(z_np(Z), cent_cls)
    s = corr_np(z_np(Z), cent_sub)
    return {"CA": r[:, 2] - r[:, 0], "C1_C2": s[:, 3] - s[:, 4], "B1_B2": s[:, 1] - s[:, 2]}


def summ(obs, nl, two_sided=False):
    nl = np.asarray(nl)
    d = {"obs": round(float(obs), 3), "null_median": round(float(np.median(nl)), 3),
         "null_q05_q95": [round(float(np.quantile(nl, q)), 3) for q in (.05, .95)]}
    if two_sided:
        c = np.median(nl)
        d["p_two_sided"] = round(float(((np.abs(nl - c) >= abs(obs - c)).sum() + 1) / (len(nl) + 1)), 4)
    else:
        d["p_lower"] = round(float(((nl <= obs).sum() + 1) / (len(nl) + 1)), 4)
        d["p_upper"] = round(float(((nl >= obs).sum() + 1) / (len(nl) + 1)), 4)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1303)
    sub, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    spec_all = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    out = {}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        genes = list(cls.index.intersection(e.index))
        cc = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        cs = sub.loc[genes].sub(sub.loc[genes].mean(axis=1), axis=0).values
        ad = (m["group"] == "AD").values
        Zad = zscore_rows(e.loc[:, ad])
        X = Zad.loc[genes].values

        def sc(gl):
            g = [x for x in gl if x in Zad.index and x not in genes]
            return Zad.loc[g].mean().values
        cells = {"neuron": sc(NEURON_MARKERS), **{k: sc(v) for k, v in CELL_MARKERS.items()}}
        spec = spec_all.reindex(e.index).fillna(3).astype(int)
        gsp = spec.loc[genes].values
        sp = lambda x, y: float(stats.spearmanr(x, y)[0])
        obs = contrasts(X, cc, cs)
        res = {"n_ad": int(ad.sum()), "n_genes": len(genes),
               "specificity_mix": {int(k): int((gsp == k).sum()) for k in range(4)}}
        # class-mean rows
        cm, sm = cc.copy(), cs.copy()
        for k in np.unique(gsp):
            cm[gsp == k] = cc[gsp == k].mean(0)
            sm[gsp == k] = cs[gsp == k].mean(0)
        cmr = contrasts(X, cm, sm)
        res["CA_neuron"] = {"observed": round(sp(obs["CA"], cells["neuron"]), 3),
                            "class_mean_rows": round(sp(cmr["CA"], cells["neuron"]), 3),
                            "rho_CA_observed_vs_class_mean_rows": round(sp(obs["CA"], cmr["CA"]), 3)}
        # permutations and sign flips on the classifier genes
        perm = {k: [] for k in ["CA", "C1_C2", "B1_B2"]}
        perm_cells = {k: {c: [] for c in cells} for k in ["C1_C2", "B1_B2"]}
        sign = []
        for _ in range(a.n):
            idx = np.arange(len(genes))
            for k in np.unique(gsp):
                w = np.where(gsp == k)[0]
                idx[w] = rng.permutation(w)
            p = contrasts(X, cc[idx], cs[idx])
            perm["CA"].append(sp(p["CA"], cells["neuron"]))
            for k in perm_cells:
                for c in cells:
                    perm_cells[k][c].append(sp(p[k], cells[c]))
            f = rng.choice([-1.0, 1.0], len(genes))[:, None]
            sign.append(sp(contrasts(X, cc * f, cs * f)["CA"], cells["neuron"]))
        res["CA_neuron"]["perm_in_class"] = summ(res["CA_neuron"]["observed"], perm["CA"])
        res["CA_neuron"]["sign_random"] = summ(res["CA_neuron"]["observed"], sign)
        # specificity-matched random genes inheriting rows
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        st = dec * 4 + spec
        pool = e.index.difference(genes).difference(list(marker))
        by = {k: pool[st.loc[pool].values == k] for k in np.unique(st.loc[genes].values)}
        gst = st.loc[genes].values
        spn = {"CA": [], **{k: {c: [] for c in cells} for k in ["C1_C2", "B1_B2"]}}
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for k in np.unique(gst):
                w = np.where(gst == k)[0]
                src = by[k] if len(by[k]) else pool
                pick[w] = rng.choice(src, len(w), replace=len(w) > len(src))
            p = contrasts(Zad.loc[list(pick)].values, cc, cs)
            spn["CA"].append(sp(p["CA"], cells["neuron"]))
            for k in ["C1_C2", "B1_B2"]:
                for c in cells:
                    spn[k][c].append(sp(p[k], cells[c]))
        res["CA_neuron"]["spec_matched"] = summ(res["CA_neuron"]["observed"], spn["CA"])
        for k in ["C1_C2", "B1_B2"]:
            res[k] = {"rho_with_CA": round(sp(obs[k], obs["CA"]), 3)}
            for c in cells:
                o = sp(obs[k], cells[c])
                res[k][c] = {"perm_in_class": summ(o, perm_cells[k][c], True), "spec_matched": summ(o, spn[k][c], True)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    (a.out / "null_direction_v12.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
