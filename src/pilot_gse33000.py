"""Pilot (experiment 0): project Neff 2021 AD subtypes onto GSE33000 and decide go/no-go.

Go/no-go criteria (fixed before running):
  1. >= 70% of the 271 Neff classifier genes measured on the array
  2. >= 50% of AD samples assigned with permutation p < 0.05
  3. every class (A, B, C) holds >= 10% of AD samples
  4. neuronal marker score (genes outside the classifier) lower in class C than in A and B
     (Neff 2021: class C is characterised by decreased neuronal genes)

Usage: python pilot_gse33000.py --data /mnt/project-files/d1-data/raw --out ../results/pilot
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import load_gse33000
import neff_subtypes

SEED = 20261116
N_PERM = 1000
# neuronal markers chosen independently of the Neff classifier genes
CELL_MARKERS = {
    "astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A3", "GJA1", "SLC1A2"],
    "microglia": ["CX3CR1", "P2RY12", "CSF1R", "C1QA", "AIF1", "TYROBP"],
    "oligodendrocyte": ["MBP", "MOG", "PLP1", "MOBP", "OLIG2", "MAG"],
    "endothelial": ["CLDN5", "FLT1", "VWF", "PECAM1", "ESAM"],
}
NEURON_MARKERS = ["SNAP25", "SYT1", "STMN2", "GAP43", "VSNL1", "NEFL", "NEFM", "GAD1",
                  "GAD2", "SST", "NRGN", "CALB1", "SLC17A7", "RBFOX3", "SYN1", "ENO2"]


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise z-score; missing values (<10% per gene) are set to the gene mean (z = 0)."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0), axis=0)
    return z.fillna(0.0)


def correlate(z: pd.DataFrame, cent: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between each sample (columns of z) and each centroid (columns of cent)."""
    zc = (z - z.mean()) / z.std(ddof=0)
    cc = (cent - cent.mean()) / cent.std(ddof=0)
    return (zc.T @ cc) / len(z)


def project(z: pd.DataFrame, cent: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    r = correlate(z, cent)
    best = r.max(axis=1)
    null = np.empty((N_PERM, z.shape[1]))
    for i in range(N_PERM):
        perm = cent.sample(frac=1, random_state=rng.integers(1 << 31)).set_axis(cent.index)
        null[i] = correlate(z, perm).max(axis=1).values
    p = ((null >= best.values).sum(axis=0) + 1) / (N_PERM + 1)
    out = pd.DataFrame({"subtype": r.idxmax(axis=1), "r_best": best, "p_perm": p})
    return out.join(r.add_prefix("r_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    expr, meta = load_gse33000.load(a.data / "GSE33000")
    sub_c, cls_c = neff_subtypes.load_centroids(a.data / "neff2021")
    report: dict = {"n_samples": meta["group"].value_counts().to_dict(),
                    "age_mean_by_group": meta.groupby("group")["age"].mean().round(1).to_dict(),
                    "n_genes_array": int(expr.shape[0])}

    genes = sub_c.index.intersection(expr.index)
    report["classifier_genes_total"] = int(len(sub_c))
    report["classifier_genes_measured"] = int(len(genes))
    report["coverage"] = round(len(genes) / len(sub_c), 3)

    ad = meta.index[meta["group"] == "AD"]
    z_ad = zscore_rows(expr.loc[genes, ad])
    cent_sub = sub_c.loc[genes].sub(sub_c.loc[genes].mean(axis=1), axis=0)
    cent_cls = cls_c.loc[genes].sub(cls_c.loc[genes].mean(axis=1), axis=0)

    res_cls = project(z_ad, cent_cls, rng)
    res_sub = project(z_ad, cent_sub, rng)
    res = res_cls.add_prefix("cls_").join(res_sub.add_prefix("sub_"))
    res = res.join(meta[["age", "gender"]])

    sig = res["cls_p_perm"] < 0.05
    report["frac_assigned_p05"] = round(float(sig.mean()), 3)
    report["class_counts_all"] = res["cls_subtype"].value_counts().to_dict()
    report["class_counts_p05"] = res.loc[sig, "cls_subtype"].value_counts().to_dict()
    report["class_frac_all"] = res["cls_subtype"].value_counts(normalize=True).round(3).to_dict()
    report["subtype_counts_all"] = res["sub_subtype"].value_counts().to_dict()
    report["neff_msbb_frac"] = {"A": 0.311, "B": 0.357, "C": 0.331}

    # sanity check: neuronal markers not used by the classifier
    markers = [g for g in NEURON_MARKERS if g in expr.index and g not in genes]
    neuro = zscore_rows(expr.loc[markers, ad]).mean()
    res["neuron_score"] = neuro
    by = res.groupby("cls_subtype")["neuron_score"]
    report["neuron_markers_used"] = markers
    report["neuron_score_mean_by_class"] = by.mean().round(3).to_dict()
    groups = [g.values for _, g in by]
    report["neuron_kruskal_p"] = float(stats.kruskal(*groups).pvalue)
    c = res.loc[res.cls_subtype == "C", "neuron_score"]
    for other in ["A", "B"]:
        o = res.loc[res.cls_subtype == other, "neuron_score"]
        report[f"neuron_C_vs_{other}_mwu_p"] = float(stats.mannwhitneyu(c, o, alternative="less").pvalue)

    # age / sex by class (subtypes should not simply be age groups)
    report["age_mean_by_class"] = res.groupby("cls_subtype")["age"].mean().round(1).to_dict()
    report["age_kruskal_p"] = float(stats.kruskal(*[g.values for _, g in res.groupby("cls_subtype")["age"]]).pvalue)
    ct = pd.crosstab(res["cls_subtype"], res["gender"])
    report["sex_by_class"] = ct.to_dict()
    report["sex_chi2_p"] = float(stats.chi2_contingency(ct)[1])

    # preview of experiment 3: does unsupervised k-means (k=3) on the same genes agree?
    km = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit(z_ad.T.values)
    report["kmeans3_silhouette"] = round(float(silhouette_score(z_ad.T.values, km.labels_)), 3)
    report["kmeans3_ari_vs_projection"] = round(float(adjusted_rand_score(km.labels_, res["cls_subtype"])), 3)

    # preview of experiment 5: remove cell-composition proxies and project again
    scores = {"neuron": neuro}
    for ct_name, mk in CELL_MARKERS.items():
        mk = [g for g in mk if g in expr.index and g not in genes]
        scores[ct_name] = zscore_rows(expr.loc[mk, ad]).mean()
    X = pd.DataFrame(scores)
    report["cell_score_mean_by_class"] = X.join(res["cls_subtype"]).groupby("cls_subtype").mean().round(3).to_dict()
    Xd = np.column_stack([np.ones(len(X)), X.values])
    Y = z_ad.T.values
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = pd.DataFrame((Y - Xd @ beta).T, index=z_ad.index, columns=z_ad.columns)
    r2 = 1 - (resid.T.values ** 2).sum(axis=0) / (Y ** 2).sum(axis=0)
    report["adj_median_r2_classifier_genes_explained_by_cell_scores"] = round(float(np.median(r2)), 3)
    res_adj = project(zscore_rows(resid), cent_cls, rng)
    res["cls_subtype_celladj"] = res_adj["subtype"]
    report["adj_frac_assigned_p05"] = round(float((res_adj["p_perm"] < 0.05).mean()), 3)
    report["adj_class_frac"] = res_adj["subtype"].value_counts(normalize=True).round(3).to_dict()
    report["adj_ari_vs_unadjusted"] = round(float(adjusted_rand_score(res["cls_subtype"], res_adj["subtype"])), 3)
    report["adj_crosstab"] = pd.crosstab(res["cls_subtype"], res_adj["subtype"]).to_dict()

    # how controls project (descriptive only)
    ctrl = meta.index[meta["group"] == "Control"]
    z_all = zscore_rows(expr.loc[genes, list(ad) + list(ctrl)])
    r_all = correlate(z_all, cent_cls)
    lab = r_all.idxmax(axis=1)
    report["control_projection_frac"] = lab.loc[ctrl].value_counts(normalize=True).round(3).to_dict()

    crit = {
        "1_coverage>=0.70": report["coverage"] >= 0.70,
        "2_assigned_p05>=0.50": report["frac_assigned_p05"] >= 0.50,
        "3_each_class>=0.10": min(report["class_frac_all"].get(k, 0) for k in "ABC") >= 0.10,
        "4_neuron_C_lowest": all(report[f"neuron_C_vs_{o}_mwu_p"] < 0.05 for o in "AB"),
    }
    report["criteria"] = {k: bool(v) for k, v in crit.items()}
    report["decision"] = "GO" if all(crit.values()) else "REVIEW"

    res.to_csv(a.out / "gse33000_neff_projection.csv")
    (a.out / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(res["cls_r_best"], bins=30, color="#4C72B0")
    ax[0].set(title="Best correlation to Neff class centroid", xlabel="Pearson r", ylabel="AD samples")
    order = ["A", "B", "C"]
    ax[1].boxplot([res.loc[res.cls_subtype == k, "neuron_score"] for k in order], tick_labels=order)
    ax[1].set(title="Neuronal marker score by projected class", ylabel="mean z")
    cnt = res["cls_subtype"].value_counts(normalize=True).reindex(order)
    x = np.arange(3)
    ax[2].bar(x - 0.2, cnt.values, 0.4, label="GSE33000 (projected)")
    ax[2].bar(x + 0.2, [report["neff_msbb_frac"][k] for k in order], 0.4, label="MSBB (Neff 2021)")
    ax[2].set_xticks(x, order)
    ax[2].set(title="Class proportions", ylabel="fraction")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(a.out / "pilot_summary.png", dpi=150)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
