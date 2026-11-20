"""Gene sets for experiment 3b (robustness of LDSC-SEG results).
 - key cell types at top 5% and top 20% specificity (window 100 kb) and top 10% at 50 kb
 - all-genes control at 50 kb (for the 50 kb sets)
 - DRG neuron subtypes (atlas with DRG_Neuron replaced by subtypes), top 10%, 100 kb
 - 20 random sets of the top-10% size drawn from atlas genes (empirical null), 100 kb
Usage: python -I build_sets_3b.py <atlas_cpm.tsv.gz> <GSE168243.counts.h5ad> <genes_hg19.tsv> <out.tsv>
Output columns: set, SYMBOL, window
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad

atl_f, drg_f, genes_f, out = sys.argv[1:5]
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
KEY = ["Ligament_Fibro", "Disc_Fibro", "Disc_Chondro", "Disc_CEPlike", "Bone_Osteo_stromal",
       "Bone_Chondro", "DRG_Neuron", "Cord_Neuron", "Brain_ExcNeuron"]
DRG_SUB = {"DRG_PEP": [0, 1, 2, 3, 5, 8], "DRG_NP_prurito": [12, 13, 14],
           "DRG_Abeta_proprio": [7, 9], "DRG_LTMR": [4, 10, 11]}   # cluster 6 (57 cells, mixed) dropped


def top(spec, col, frac):
    return spec[col].sort_values(ascending=False).index[: int(round(frac * len(spec)))]


rows = []
spec = atlas.div(atlas.sum(1), axis=0)
for c in KEY:
    rows += [(f"{c}__top5", g, 100_000) for g in top(spec, c, 0.05)]
    rows += [(f"{c}__top20", g, 100_000) for g in top(spec, c, 0.20)]
    rows += [(f"{c}__w50", g, 50_000) for g in top(spec, c, 0.10)]
rows += [("all_genes__w50", g, 50_000) for g in atlas.index]
# DRG subtypes
a = ad.read_h5ad(drg_f)
a = a[:, [g for g in atlas.index if g in a.var_names]]
sub = {}
for s, cl in DRG_SUB.items():
    m = a.obs.cl.astype(int).isin(cl).values
    x = np.asarray(a.X[m].sum(0)).ravel()
    sub[s] = pd.Series(x / x.sum() * 1e6, index=a.var_names)
at2 = atlas.drop(columns="DRG_Neuron").join(pd.DataFrame(sub), how="inner")
at2 = at2 / at2.sum() * 1e6
spec2 = at2.div(at2.sum(1), axis=0)
for s in DRG_SUB:
    rows += [(s, g, 100_000) for g in top(spec2, s, 0.10)]
# random null sets
rng = np.random.default_rng(2026)
k = int(round(0.10 * len(atlas)))
for i in range(20):
    rows += [(f"random{i:02d}", g, 100_000) for g in rng.choice(atlas.index.to_numpy(), k, replace=False)]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(out, sep="\t", index=False)
print("sets", len(dict.fromkeys(r[0] for r in rows)), "cells per DRG subtype",
      {s: int(a.obs.cl.astype(int).isin(c).sum()) for s, c in DRG_SUB.items()})
