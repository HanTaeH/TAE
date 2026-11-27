"""GSE271018 (Tang 2024; 6 degenerative-stenosis 'DSS' and 3 traumatic 'HC' spinal ligaments, 10x v3,
CD45+/CD45- sorted 3:7). Raw counts from the authors' Seurat objects -> QC -> Harmony by sample -> Leiden,
then marker means per cluster for labelling. Writes combined counts h5ad with obs: sample, group, cl.
Usage: python -I process_gse271018.py <dir with C1,C3,C5,N0,N1,N2,N3,N5,N10 .h5ad> <out.h5ad> <markers.tsv>
"""
import sys
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, harmonypy

d, out, mk = sys.argv[1:4]
GROUP = {"C1": "HC", "C3": "HC", "C5": "HC", "N0": "DSS", "N1": "DSS", "N2": "DSS", "N3": "DSS", "N5": "DSS", "N10": "DSS"}
ads = []
for s, g in GROUP.items():
    a = ad.read_h5ad(f"{d}/{s}.h5ad"); a.obs = pd.DataFrame(index=[f"{s}_{c}" for c in a.obs_names])
    a.obs["sample"], a.obs["group"] = s, g; ads.append(a)
a = ad.concat(ads, join="outer", fill_value=0); a.X = a.X.tocsr()
a.var["mt"] = a.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
a = a[(a.obs.n_genes_by_counts >= 300) & (a.obs.pct_counts_mt < 20)].copy()
sc.pp.filter_genes(a, min_cells=10)
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample")
b = a[:, a.var.highly_variable].copy(); sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=30)
h = harmonypy.run_harmony(b.obsm["X_pca"], b.obs, "sample", max_iter_harmony=20)
Z = np.asarray(h.Z_corr); Z = Z.T if Z.shape[0] != b.n_obs else Z
a.obsm["X_harmony"] = Z
sc.pp.neighbors(a, use_rep="X_harmony", n_neighbors=20); sc.tl.leiden(a, resolution=0.6, key_added="cl", flavor="igraph", n_iterations=2)
M = {"fibro": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "COL3A1"], "immune": ["PTPRC"], "myeloid": ["CD68", "LYZ", "CD14"],
     "T": ["CD3E", "CD2"], "B": ["MS4A1", "CD79A"], "endo": ["PECAM1", "VWF"], "peri_SMC": ["RGS5", "ACTA2", "MYH11"],
     "osteo": ["SP7", "BGLAP", "RUNX2", "IBSP"], "chondro": ["COL2A1", "ACAN", "SOX9", "CRTAC1"], "neutro": ["S100A8", "CSF3R"],
     "ery": ["HBB"], "mast": ["TPSAB1", "CPA3"], "prolif": ["MKI67"]}
rows = []
for c, idx in a.obs.groupby("cl").groups.items():
    s = a[idx]; r = {"cl": c, "n": len(idx), "frac_DSS": (s.obs.group == "DSS").mean(), "n_samples": s.obs["sample"].nunique()}
    for k, g in M.items():
        g = [x for x in g if x in a.var_names]; r[k] = float(np.asarray(s[:, g].X.mean(0)).mean())
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("n", ascending=False)
tab.to_csv(mk, sep="\t", index=False)
a.X = a.layers["counts"]; del a.layers["counts"]
a.obs = a.obs[["sample", "group", "cl", "n_genes_by_counts", "total_counts", "pct_counts_mt"]]
a.write_h5ad(out, compression="gzip")
pd.set_option("display.width", 250); print(a.shape); print(tab.round(2).to_string(index=False))
