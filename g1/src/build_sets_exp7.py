"""Experiment 7a: is the stenosis-ligament enrichment driven by disease-state expression (reverse causation)?
Rebuild the Ligament_Fibro column of the v3 atlas from
  LDH  = ligamentum flavum from a lumbar disc herniation patient (GSE294458 'LDH' sample, non-stenotic)
  LSS  = stenotic / hypertrophied ligamentum flavum (GSE294458 'LSS' + GSE267819 LFH1-3)
EXP7_ROUND=2: Ligament_Fibro_LSS294 (GSE294458 LSS donor only, same dataset as LDH) and LF_hypertrophy_up/down
(top 10% by log2 CPM ratio LSS294 vs LDH, genes with CPM > 1 in either).
Uses the same ligament fibroblast clusters as the atlas, then make top-10% specificity sets (+/-100 kb) for
Ligament_Fibro_LDH, Ligament_Fibro_LSS and Fibro_pooled_LDH (LDH ligament + disc fibroblast), plus all_genes.
Other atlas columns are unchanged. Also writes overlap with the v3 Ligament_Fibro set.
Usage: python -I build_sets_exp7.py <sc_dir> <labels.tsv> <atlas_v3_cpm.tsv.gz> <out_dir>
"""
import sys, os
import numpy as np
import pandas as pd
import anndata as ad

scd, labf, atl_f, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
lab = pd.read_csv(labf, sep="\t", dtype=str)
n = len(atlas); k = int(round(0.10 * n))


def pb(ds, samples):
    a = ad.read_h5ad(f"{scd}/{ds}.counts.h5ad")
    cl = set(lab[(lab.dataset == ds) & (lab.label == "Ligament_Fibro")].cluster)
    m = (a.obs.cl.astype(str).isin(cl) & a.obs["sample"].isin(samples)).values
    s = pd.Series(np.asarray(a.X[m].sum(0)).ravel(), index=a.var_names)
    s = s.groupby(level=0).sum().reindex(atlas.index)
    print(ds, samples, "cells", int(m.sum()), "atlas genes present", int(s.notna().sum()))
    return s / s.sum() * 1e6, int(m.sum())


ldh, n_ldh = pb("GSE294458", ["LDH"])
lss1, n1 = pb("GSE294458", ["LSS"]); lss2, n2 = pb("GSE267819", ["LFH1", "LFH2", "LFH3"])
cols = {"Ligament_Fibro_LDH": ldh.fillna(0), "Ligament_Fibro_LSS": pd.concat([lss1, lss2], axis=1).mean(axis=1, skipna=True).fillna(0)}
if os.environ.get("EXP7_ROUND") == "2":
    # round 2: same-dataset, one-donor-each comparison, and the hypertrophy program itself
    cols = {"Ligament_Fibro_LSS294": lss1.fillna(0)}


def top(a, c):
    a = a / a.sum() * 1e6
    return list(a.div(a.sum(axis=1), axis=0)[c].sort_values(ascending=False).index[:k])


rows, sets = [], {}
for name, col in cols.items():
    a = atlas.drop(columns=["Ligament_Fibro"]).assign(**{name: col})
    sets[name] = top(a, name)
if os.environ.get("EXP7_ROUND") == "2":
    e = (ldh.fillna(0) > 1) | (lss1.fillna(0) > 1)
    lfc = (np.log2(lss1.fillna(0) + 1) - np.log2(ldh.fillna(0) + 1))[e]
    sets["LF_hypertrophy_up"] = list(lfc.sort_values(ascending=False).index[:k])
    sets["LF_hypertrophy_down"] = list(lfc.sort_values().index[:k])
if os.environ.get("EXP7_ROUND") != "2":
    a = atlas.drop(columns=["Ligament_Fibro", "Disc_Fibro"]).assign(
        Fibro_pooled_LDH=(cols["Ligament_Fibro_LDH"] / cols["Ligament_Fibro_LDH"].sum() * 1e6 + atlas.Disc_Fibro) / 2)
    sets["Fibro_pooled_LDH"] = top(a, "Fibro_pooled_LDH")
v3 = set(atlas.div(atlas.sum(axis=1), axis=0)["Ligament_Fibro"].sort_values(ascending=False).index[:k])
ov = [dict(set=s, n=len(g), overlap_with_v3_Ligament=len(v3 & set(g)) / k) for s, g in sets.items()]
if "Ligament_Fibro_LDH" in sets:
    ov.append(dict(set="LDH_vs_LSS", n=k, overlap_with_v3_Ligament=len(set(sets["Ligament_Fibro_LDH"]) & set(sets["Ligament_Fibro_LSS"])) / k))
pd.DataFrame(ov).to_csv(f"{out}/set_overlap.tsv", sep="\t", index=False)
for s, g in sets.items():
    rows += [(s, x, 100_000) for x in g]
rows += [("all_genes", x, 100_000) for x in atlas.index]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(f"{out}/genesets_exp7.tsv", sep="\t", index=False)
pd.DataFrame([dict(sample="LDH", cells=n_ldh), dict(sample="LSS_GSE294458", cells=n1), dict(sample="LFH_GSE267819", cells=n2)]).to_csv(f"{out}/ncells.tsv", sep="\t", index=False)
print(pd.DataFrame(ov).round(3).to_string(index=False))
