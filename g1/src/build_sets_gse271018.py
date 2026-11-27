"""Limitation 1A: stenotic vs non-stenotic ligament fibroblast sets from a second, multi-donor study
(GSE271018: 6 degenerative-stenosis 'DSS', 3 traumatic 'HC' spinal ligaments). Fibroblast clusters 0 and 1
(COL1A1/DCN/LUM/PDGFRA high, PTPRC low). Donors with < 50 fibroblasts are dropped (N0 has 1).
Group profile = mean of per-donor CPM (so one large donor, N5, does not dominate).
Sets (top 10% specificity after replacing the atlas Ligament_Fibro column, +/-100 kb, as in experiment 7a):
  L271_DSS, L271_HC, L271_DSS_minus_<donor> (leave-one-donor-out),
  L271_up / L271_down: top / bottom 10% of genes by donor-level Welch t on log2 CPM (DSS vs HC), CPM > 1.
Usage: python -I build_sets_gse271018.py <GSE271018.counts.h5ad> <atlas_cpm.tsv.gz> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, anndata as ad
from scipy import stats

h5, atl_f, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
atlas = pd.read_csv(atl_f, sep="\t", index_col=0); k = int(round(0.10 * len(atlas)))
a = ad.read_h5ad(h5)
f = a[a.obs.cl.astype(str).isin(["0", "1"])]
cpm, info = {}, []
for s, idx in f.obs.groupby("sample", observed=True).groups.items():
    n = len(idx)
    info.append(dict(sample=s, group=f.obs.loc[idx[0], "group"], fibroblasts=n, used=n >= 50))
    if n < 50: continue
    v = pd.Series(np.asarray(f[idx].X.sum(0)).ravel(), index=f.var_names).groupby(level=0).sum().reindex(atlas.index).fillna(0)
    cpm[s] = v / v.sum() * 1e6
info = pd.DataFrame(info); info.to_csv(f"{out}/donors.tsv", sep="\t", index=False); print(info.to_string(index=False))
C = pd.DataFrame(cpm)
dss = [s for s in C if s.startswith("N")]; hc = [s for s in C if s.startswith("C")]


def top(col, name):
    x = atlas.drop(columns=["Ligament_Fibro"]).assign(**{name: col})
    x = x / x.sum() * 1e6
    return list(x.div(x.sum(axis=1), axis=0)[name].sort_values(ascending=False).index[:k])


sets = {"L271_DSS": top(C[dss].mean(1), "L271_DSS"), "L271_HC": top(C[hc].mean(1), "L271_HC")}
for d in dss:
    sets[f"L271_DSS_minus_{d}"] = top(C[[x for x in dss if x != d]].mean(1), "x")
e = C.mean(1) > 1
L = np.log2(C[e] + 1)
t, p = stats.ttest_ind(L[dss], L[hc], axis=1, equal_var=False)
t = pd.Series(t, index=L.index).dropna()
sets["L271_up"] = list(t.sort_values(ascending=False).index[:k]); sets["L271_down"] = list(t.sort_values().index[:k])
pd.DataFrame({"t": t, "p": pd.Series(p, index=L.index)}).to_csv(f"{out}/dss_vs_hc_t.tsv", sep="\t")
v3 = set(atlas.div(atlas.sum(axis=1), axis=0)["Ligament_Fibro"].sort_values(ascending=False).index[:k])
ov = pd.DataFrame([dict(set=s, overlap_with_v3_Ligament=len(v3 & set(g)) / k) for s, g in sets.items()])
ov.loc[len(ov)] = dict(set="DSS_vs_HC", overlap_with_v3_Ligament=len(set(sets["L271_DSS"]) & set(sets["L271_HC"])) / k)
ov.to_csv(f"{out}/set_overlap.tsv", sep="\t", index=False); print(ov.round(3).to_string(index=False))
rows = [(s, g, 100_000) for s, gs in sets.items() for g in gs]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(f"{out}/genesets_gse271018.tsv", sep="\t", index=False)
for g in ["CRLF1", "MMP14", "BMP6", "SOX9", "NGF", "COL1A1"]:
    if g in t.index: print(g, round(t[g], 2))
