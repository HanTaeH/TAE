"""Priority 5: do Neff classes track neurofibrillary (Braak) stage? GSE106241, inferior temporal cortex,
60 donors, Braak 0-6, with age, sex, RNA quality, APOE e4 and amyloid-beta 42.

Neff 2021 reported no Braak association within MSBB AD cases (p = 0.075) after CDR normalisation.
Tests here (C-minus-A score = r(C) - r(A); B axis = r(B) - (r(A) + r(C)) / 2):
  T1  all donors: Spearman Braak vs C-minus-A, and partial (age, sex, RNA quality)
  T2  pathology-positive subsets (Braak >= 3, Braak >= 5), z-scored within the subset as in an
      AD-only projection; Spearman Braak vs C-minus-A
  T3  amyloid-beta 42 vs C-minus-A (all donors; partial)
  T4  T1 / T2 after also partialling out cell composition (neuron marker score; SEA-AD NNLS
      proportions, CLR) - does the Braak link survive once neuron loss is removed?
  T5  random gene null (500 sets, matched on mean decile x SD tercile, inheriting centroid rows)
      for the T1 and T2 correlations
  T6  class proportions by Braak group (0-2, 3-4, 5-6), chi-square

Usage: python braak_gse106241.py --data /mnt/project-files/d1-data/raw \
         --ref /mnt/project-files/d1-data/seaad/seaad_mtg_reference.csv.gz --out ../results/braak --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import load_gse106241
import neff_subtypes
from deconv_seaad import clr, est_nnls, signature_genes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

CLS = np.array(list("ABC"))


def partial_spearman(x, y, covars=None):
    """Spearman of x and y after regressing ranks on covariates (None -> plain Spearman)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if covars is not None:
        ok &= ~np.isnan(covars).any(1)
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    if covars is None:
        r, p = stats.pearsonr(rx, ry)
        return float(r), float(p), int(ok.sum())
    C = covars[ok]
    D = np.column_stack([np.ones(ok.sum())] + [stats.rankdata(c) for c in C.T])
    ex = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
    ey = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = ok.sum() - D.shape[1]
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df)), int(ok.sum())


def fmt(t):
    return {"rho": round(t[0], 3), "p": float(f"{t[1]:.3g}"), "n": t[2]}


def scores(x, cent):
    r = corr_np(z_np(x), cent)
    return r[:, 2] - r[:, 0], r[:, 1] - (r[:, 0] + r[:, 2]) / 2, CLS[r.argmax(1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    expr, meta = load_gse106241.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    braak = meta["braak"].values
    base_cov = np.column_stack([meta["age"], (meta["sex"] == "female").astype(float), meta["rna_quality"]])

    # composition
    def score(lst, idx):
        lst = [g for g in lst if g in expr.index and g not in genes]
        return z_np(expr.loc[lst, idx].values).mean(0)
    neuron = score(NEURON_MARKERS, meta.index)
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    rank = expr.mean(axis=1).rank(pct=True)
    sig = signature_genes(ref, set(cls.index), rank.index[rank >= 0.5])
    P = pd.DataFrame(est_nnls((2 ** expr.loc[sig]).values, ref.loc[sig].values),
                     index=meta.index, columns=ref.columns)

    rep: dict = {"n": int(len(meta)), "n_genes": int(len(genes)), "coverage": round(len(genes) / len(cls), 3),
                 "braak_counts": meta["braak"].value_counts().sort_index().to_dict(),
                 "n_signature_genes": len(sig)}
    rep["neuron_score_vs_braak"] = fmt(partial_spearman(neuron, braak))
    rep["exc_prop_vs_braak"] = fmt(partial_spearman(P["Exc"].values, braak))
    rep["exc_prop_vs_neuron_score"] = fmt(partial_spearman(P["Exc"].values, neuron))

    X = expr.loc[genes, meta.index].values
    ca, b, lab = scores(X, cent)
    rep["T1_all"] = {"ca_vs_braak": fmt(partial_spearman(ca, braak)),
                     "ca_vs_braak_partial_age_sex_rin": fmt(partial_spearman(ca, braak, base_cov)),
                     "b_vs_braak": fmt(partial_spearman(b, braak)),
                     "ca_vs_neuron": fmt(partial_spearman(ca, neuron))}
    rep["T3_abeta42"] = {"ca": fmt(partial_spearman(ca, meta["abeta42"].values)),
                         "ca_partial": fmt(partial_spearman(ca, meta["abeta42"].values, base_cov)),
                         "ca_partial_plus_braak": fmt(partial_spearman(ca, meta["abeta42"].values,
                                                                       np.column_stack([base_cov, braak])))}
    comp_sets = {"neuron_score": neuron[:, None], "seaad_nnls": clr(P.values)}
    rep["T4_all"] = {k: fmt(partial_spearman(ca, braak, np.column_stack([base_cov, v])))
                     for k, v in comp_sets.items()}

    # pathology-positive subsets, AD-only style projection
    subsets = {"braak_ge3": braak >= 3, "braak_ge5": braak >= 5}
    sub_res = {}
    for name, mask in subsets.items():
        ca_s, b_s, lab_s = scores(X[:, mask], cent)
        cov = base_cov[mask]
        neu_s = score(NEURON_MARKERS, meta.index[mask])
        sub_res[name] = (ca_s, mask)
        rep[f"T2_{name}"] = {
            "n": int(mask.sum()),
            "class_frac": {k: round(float((lab_s == k).mean()), 3) for k in "ABC"},
            "ca_vs_braak": fmt(partial_spearman(ca_s, braak[mask])),
            "ca_vs_braak_partial_age_sex_rin": fmt(partial_spearman(ca_s, braak[mask], cov)),
            "ca_vs_braak_partial_plus_neuron": fmt(partial_spearman(ca_s, braak[mask], np.column_stack([cov, neu_s]))),
            "ca_vs_braak_partial_plus_nnls": fmt(partial_spearman(ca_s, braak[mask],
                                                                  np.column_stack([cov, clr(P.values[mask])]))),
            "class_vs_braak_kruskal_p": float(stats.kruskal(*[braak[mask][lab_s == k] for k in "ABC"
                                                              if (lab_s == k).sum() > 1]).pvalue),
            "ca_vs_neuron": fmt(partial_spearman(ca_s, neu_s)),
        }

    # T5 random gene null
    E = expr[meta.index]
    dec = pd.qcut(E.mean(axis=1).rank(method="first"), 10, labels=False)
    terc = E.std(axis=1).groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    stratum = dec * 3 + terc
    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pool = E.index.difference(genes).difference(list(marker_genes))
    pool_by = {s: pool[stratum.loc[pool].values == s] for s in range(30)}
    neff_s = stratum.loc[genes].values
    obs = {"all": partial_spearman(ca, braak)[0],
           **{k: partial_spearman(v[0], braak[v[1]])[0] for k, v in sub_res.items()}}
    null = {k: [] for k in obs}
    for _ in range(a.n):
        pick = np.empty(len(genes), dtype=object)
        for s in np.unique(neff_s):
            idx = np.where(neff_s == s)[0]
            pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=False)
        Xr = E.loc[list(pick)].values
        null["all"].append(partial_spearman(scores(Xr, cent)[0], braak)[0])
        for k, (_, mask) in sub_res.items():
            null[k].append(partial_spearman(scores(Xr[:, mask], cent)[0], braak[mask])[0])
    rep["T5_random_null"] = {k: {"neff_rho": round(obs[k], 3), "random_median": round(float(np.median(null[k])), 3),
                                 "random_q95": round(float(np.quantile(null[k], .95)), 3),
                                 "p": round(float((np.sum(np.array(null[k]) >= obs[k]) + 1) / (a.n + 1)), 4)}
                             for k in obs}

    # T6 class by Braak group (all-donor projection)
    grp = pd.cut(meta["braak"], [-1, 2, 4, 6], labels=["0-2", "3-4", "5-6"])
    tab = pd.crosstab(grp, lab).reindex(columns=list("ABC"), fill_value=0)
    rep["T6_class_by_braak_group"] = (tab.div(tab.sum(axis=1), axis=0).round(3)).to_dict(orient="index")
    rep["T6_chi2_p"] = float(stats.chi2_contingency(tab)[1])

    (a.out / "braak_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    out = meta.assign(ca_score=ca, b_axis=b, neff_class=lab, neuron_score=neuron).join(P.add_prefix("nnls_"))
    out.to_csv(a.out / "braak_per_sample.csv")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    data = [ca[braak == s] for s in range(7)]
    ax.boxplot(data, positions=range(7), widths=0.6)
    ax.scatter(braak + rng.uniform(-.15, .15, len(braak)), ca, s=12, color="#1f4e79", alpha=.7)
    r = rep["T1_all"]["ca_vs_braak"]
    ax.set_title(f"C-minus-A score by Braak stage (all donors)\nρ = {r['rho']}, p = {r['p']}", fontsize=10)
    ax.set_xlabel("Braak NFT stage")
    ax.set_ylabel("C-minus-A score")
    ax = axes[1]
    ax.scatter(neuron, ca, c=braak, cmap="viridis", s=20)
    ax.set_xlabel("Neuron marker score")
    ax.set_ylabel("C-minus-A score")
    ax.set_title(f"C-minus-A vs neuron score (colour = Braak)\nρ = {rep['T1_all']['ca_vs_neuron']['rho']}", fontsize=10)
    ax = axes[2]
    bottom = np.zeros(3)
    colors = {"A": "#2e7d32", "B": "#f9a825", "C": "#c62828"}
    for k in "ABC":
        v = tab[k].values / tab.sum(axis=1).values
        ax.bar(tab.index.astype(str), v, bottom=bottom, color=colors[k], label=f"class {k}")
        bottom += v
    ax.set_xlabel("Braak group")
    ax.set_ylabel("Fraction of donors")
    ax.set_title(f"Neff class by Braak group (χ² p = {rep['T6_chi2_p']:.3g})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "braak_summary.png", dpi=150)


if __name__ == "__main__":
    main()
