"""Where does the replicated residual axis come from? GTEx v8 brain cortex (follow-up, 2026-11-25).

GTEx (non-AD donors) records how each donor died (Hardy scale: 0 ventilator case, 1 violent and fast,
2 fast natural, 3 intermediate, 4 slow), tissue ischaemic time, RIN and processing batch, which the AD
cohorts lack. For Brain - Cortex and Brain - Frontal Cortex (BA9): log2 CPM, five marker cell scores
regressed out exactly as in the AD cohorts, then
  template  correlation of each residual profile with the residual-axis weights
            (results/residual_quality/residual_template_weights.csv)
  ribosome, heat_shock  mean residual z of those gene groups
Checks: the axis is present (best |r| of the weights with residual PC1..5 loadings); Spearman with RIN,
ischaemic time and age; Kruskal-Wallis over Hardy classes and ventilator (0) vs fast deaths (1, 2);
OLS of each score on RIN + ischaemic time + Hardy dummies + age + sex with the R^2 drop of each term;
Kruskal-Wallis and eta^2 over expression batches (SMGEBTCH, >= 5 samples), eta^2 against 200 label permutations.

Usage: python gtex_residual_axis.py --gtex /mnt/project-files/d1-data/raw/gtex_v8 \
         --weights ../results/residual_quality/residual_template_weights.csv --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from null_models import z_np
from residual_denovo import NPC, loadings, pca
from residual_quality import resid, scores

TISSUES = {"cortex": "gene_reads_2017-06-05_v8_brain_cortex.gct.gz",
           "frontal_BA9": "gene_reads_2017-06-05_v8_brain_frontal_cortex_ba9.gct.gz"}


def load_counts(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, sep="\t", skiprows=2).drop(columns=["id", "Name"], errors="ignore")
    c = c.groupby("Description").sum()
    lib = c.sum(axis=0)
    c = c.loc[(c / lib * 1e6 > 1).mean(axis=1) > 0.5]
    return np.log2(c / lib * 1e6 + 1)


def r2(y, X):
    D = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    return 1 - ((y - D @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtex", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(a.weights, index_col=0).iloc[:, 0]
    sa = pd.read_csv(a.gtex / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t", index_col=0, low_memory=False)
    ph = pd.read_csv(a.gtex / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t", index_col=0)
    out = {}
    for tname, fn in TISSUES.items():
        e = load_counts(a.gtex / fn)
        s = sa.loc[e.columns]
        subj = ["-".join(x.split("-")[:2]) for x in e.columns]
        meta = pd.DataFrame({"rin": s["SMRIN"].values, "isch": s["SMTSISCH"].values, "batch": s["SMGEBTCH"].values,
                             "hardy": ph.reindex(subj)["DTHHRDY"].values, "sex": ph.reindex(subj)["SEX"].values,
                             "age": ph.reindex(subj)["AGE"].str[:2].astype(float).values + 5}, index=e.columns)
        R, g = resid(e, e.columns, w.index)
        S, _ = pca(R)
        ld = loadings(R, S[:, :NPC])
        wr = w.reindex(g).values
        present = [round(float(abs(np.corrcoef(ld[:, k], wr)[0, 1])), 3) for k in range(NPC)]
        sc = scores(R, g, w)
        res = {"n_samples": int(e.shape[1]), "template_genes_found": len(g), "abs_r_template_vs_residual_PC1_5": present,
               "hardy_counts": meta["hardy"].value_counts().sort_index().to_dict()}
        for name, v in sc.items():
            d = {}
            for var in ("rin", "isch", "age"):
                ok = meta[var].notna().values
                r = stats.spearmanr(v[ok], meta[var].values[ok])
                d[f"{var}_rho"], d[f"{var}_p"] = round(float(r[0]), 3), float(f"{r[1]:.3g}")
            hk = meta["hardy"].values
            groups = [v[hk == k] for k in sorted(pd.unique(hk[~pd.isna(hk)])) if (hk == k).sum() >= 5]
            d["hardy_kw_p"] = float(f"{stats.kruskal(*groups)[1]:.3g}")
            d["hardy_means"] = {int(k): round(float(v[hk == k].mean()), 3) for k in sorted(pd.unique(hk[~pd.isna(hk)]))}
            vent, fast = v[hk == 0], v[np.isin(hk, [1, 2])]
            if len(vent) >= 5 and len(fast) >= 5:
                u = stats.mannwhitneyu(vent, fast)
                d["ventilator_vs_fast_auc"] = round(float(u[0] / (len(vent) * len(fast))), 3)
                d["ventilator_vs_fast_p"] = float(f"{u[1]:.3g}")
            ok = meta[["rin", "isch", "hardy", "age", "sex"]].notna().all(axis=1).values
            m = meta[ok]
            terms = {"rin": m[["rin"]].values, "isch": m[["isch"]].values,
                     "hardy": pd.get_dummies(m["hardy"].astype(int), drop_first=True).values.astype(float),
                     "age": m[["age"]].values, "sex": (m[["sex"]].values == 1).astype(float)}
            full = np.column_stack(list(terms.values()))
            r2f = r2(v[ok], full)
            d["ols_r2_full"] = round(float(r2f), 3)
            d["ols_r2_drop"] = {k: round(float(r2f - r2(v[ok], np.column_stack([t for kk, t in terms.items() if kk != k]))), 3)
                                for k in terms}
            bt = meta["batch"].values
            bg = [v[bt == b] for b in pd.unique(bt) if (bt == b).sum() >= 5]
            d["batch_kw_p"] = float(f"{stats.kruskal(*bg)[1]:.3g}")
            d["batch_eta2"] = round(float(r2(v[np.isin(bt, [b for b in pd.unique(bt) if (bt == b).sum() >= 5])],
                                              pd.get_dummies(bt[np.isin(bt, [b for b in pd.unique(bt) if (bt == b).sum() >= 5])],
                                                             drop_first=True).values.astype(float))), 3)
            keep = np.isin(bt, [b for b in pd.unique(bt) if (bt == b).sum() >= 5])
            Db = pd.get_dummies(bt[keep], drop_first=True).values.astype(float)
            rng = np.random.default_rng(0)
            nl = [r2(rng.permutation(v[keep]), Db) for _ in range(200)]
            d["batch_eta2_perm_median"] = round(float(np.median(nl)), 3)
            d["batch_eta2_perm_p"] = round(float((np.sum(np.array(nl) >= d["batch_eta2"]) + 1) / 201), 4)
            d["n_batches"] = int(Db.shape[1] + 1)
            res[name] = d
        out[tname] = res
        print(tname, json.dumps(res), flush=True)
    (a.out / "gtex_residual_axis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
