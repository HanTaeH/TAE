"""Within-AD Braak test in a large RNA-seq cohort: GSE125583 (fusiform gyrus, 219 AD, 70 controls).

Neff 2021 found no Braak association within MSBB AD cases (p = 0.075); GSE106241 reproduced this in
32 pathology-positive donors. Here, with 219 AD cases (Braak I-VI, mostly IV-VI):
  AD-only projection (z-scores within AD): Spearman Braak vs C-minus-A score, plain, partial for
  age and sex, and additionally for the neuron marker score or SEA-AD NNLS proportions;
  500 variance-matched random gene sets inheriting centroid rows as a null for the plain ρ.
  All samples (AD + controls): same, to compare with the across-spectrum trend.

Usage: python braak_gse125583.py --data /mnt/project-files/d1-data/raw \
         --ref /mnt/project-files/d1-data/seaad/seaad_mtg_reference.csv.gz --out ../results/braak_gse125583 --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import load_gse125583
import neff_subtypes
from braak_gse106241 import fmt, partial_spearman, scores
from deconv_seaad import clr, est_nnls, signature_genes
from null_models import z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    expr, meta = load_gse125583.load(a.data)
    meta = meta.dropna(subset=["braak", "sex"])
    expr = expr[meta.index]
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    rank = expr.mean(axis=1).rank(pct=True)
    sig = signature_genes(ref, set(cls.index), rank.index[rank >= 0.5])
    P_all = pd.DataFrame(est_nnls((2 ** expr.loc[sig] - 1).clip(lower=0).values, ref.loc[sig].values),
                         index=meta.index, columns=ref.columns)

    rep: dict = {"n": meta["group"].value_counts().to_dict(), "coverage": round(len(genes) / len(cls), 3),
                 "braak_by_group": meta.groupby(["group", "braak"]).size().unstack(fill_value=0).to_dict(orient="index")}
    sets = {"AD_only": meta.index[meta["group"] == "AD"], "all_samples": meta.index}
    E = expr
    dec = pd.qcut(E.mean(axis=1).rank(method="first"), 10, labels=False)
    terc = E.std(axis=1).groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    stratum = dec * 3 + terc
    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pool = E.index.difference(genes).difference(list(marker_genes))
    pool_by = {s: pool[stratum.loc[pool].values == s] for s in range(30)}
    neff_s = stratum.loc[genes].values

    per = []
    for name, idx in sets.items():
        m = meta.loc[idx]
        br = m["braak"].values
        cov = np.column_stack([m["age"], (m["sex"] == "female").astype(float)])
        lst = [g for g in NEURON_MARKERS if g in expr.index and g not in genes]
        neuron = z_np(expr.loc[lst, idx].values).mean(0)
        ca, b, lab = scores(expr.loc[genes, idx].values, cent)
        null = []
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for s in np.unique(neff_s):
                ii = np.where(neff_s == s)[0]
                pick[ii] = rng.choice(pool_by[s], size=len(ii), replace=False)
            null.append(partial_spearman(scores(E.loc[list(pick), idx].values, cent)[0], br)[0])
        obs = partial_spearman(ca, br)
        rep[name] = {
            "n": int(len(idx)),
            "class_frac": {k: round(float((lab == k).mean()), 3) for k in "ABC"},
            "class_frac_by_braak": pd.crosstab(br, lab, normalize="index").round(3).to_dict(orient="index"),
            "ca_vs_braak": fmt(obs),
            "ca_vs_braak_partial_age_sex": fmt(partial_spearman(ca, br, cov)),
            "ca_vs_braak_partial_plus_neuron": fmt(partial_spearman(ca, br, np.column_stack([cov, neuron]))),
            "ca_vs_braak_partial_plus_nnls": fmt(partial_spearman(ca, br, np.column_stack([cov, clr(P_all.loc[idx].values)]))),
            "b_vs_braak": fmt(partial_spearman(b, br)),
            "neuron_vs_braak": fmt(partial_spearman(neuron, br)),
            "exc_vs_braak": fmt(partial_spearman(P_all.loc[idx, "Exc"].values, br)),
            "ca_vs_neuron": fmt(partial_spearman(ca, neuron)),
            "random_null": {"random_median": round(float(np.median(null)), 3),
                            "random_q95": round(float(np.quantile(null, .95)), 3),
                            "p": round(float((np.sum(np.array(null) >= obs[0]) + 1) / (a.n + 1)), 4)},
        }
        per.append(m[["group", "braak", "age", "sex"]].assign(set=name, ca_score=ca, b_axis=b, neff_class=lab,
                                                               neuron_score=neuron)
                   .join(P_all.loc[idx].add_prefix("nnls_")))
    pd.concat(per).to_csv(a.out / "braak_gse125583_per_sample.csv")
    (a.out / "braak_gse125583_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
