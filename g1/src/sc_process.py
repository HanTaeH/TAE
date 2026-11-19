"""Process one public single-cell dataset into broad cell types and a pseudobulk
mean-expression table (genes x cell types, CPM) for LDSC-SEG style gene sets.

Usage: python -I sc_process.py <dataset> <raw_dir> <out_dir>
dataset in: GSE165722 GSE160756 GSE169396 GSE267819 GSE294458 GSE168243 GSE190442
Steps: load per sample -> keep barcodes with >=500 UMI and >=250 genes (raw matrices are
unfiltered) -> mito filter -> doublet-agnostic clustering (log-CPM, 2000 HVG, PCA 30,
Harmony over samples, Leiden) -> each cluster labelled by the highest mean z-score of
canonical marker sets -> clusters with the same label merged.
Outputs: <out>/<dataset>.pseudobulk_cpm.tsv.gz, .cluster_labels.tsv, .cluster_markers.tsv, .umap.png
"""
import sys, os, glob, gzip, shutil, tempfile
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

ds, raw, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
sc.settings.verbosity = 1
MAXCELLS_PER_SAMPLE = 10000

MARKERS = {
    "NP_chondrocyte": ["ACAN", "COL2A1", "SOX9", "KRT19", "KRT8", "CHI3L1", "CHI3L2", "COMP", "CYTL1"],
    "AF_fibrochondrocyte": ["COL1A1", "FMOD", "TNMD", "SCX", "THBS4", "COMP", "COL12A1"],
    "Fibroblast": ["DCN", "LUM", "COL1A2", "PDGFRA", "COL3A1", "FBLN1", "MFAP5"],
    "Osteoblast_osteoprogenitor": ["RUNX2", "SP7", "BGLAP", "SPP1", "IBSP", "ALPL", "COL1A1"],
    "MSC_stromal": ["CXCL12", "LEPR", "KITLG", "VCAM1", "EBF3"],
    "Osteoclast": ["ACP5", "CTSK", "MMP9", "OSCAR", "DCSTAMP"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "FLT1", "EMCN"],
    "Pericyte_SMC": ["RGS5", "ACTA2", "MYH11", "PDGFRB", "TAGLN", "KCNJ8"],
    "Macrophage_myeloid": ["CD68", "CD14", "LYZ", "C1QA", "C1QB", "CSF1R", "AIF1"],
    "T_NK": ["CD3E", "CD3D", "CD2", "NKG7", "GNLY", "IL7R"],
    "B_plasma": ["MS4A1", "CD79A", "CD79B", "JCHAIN", "MZB1"],
    "Neutrophil": ["S100A8", "S100A9", "CSF3R", "FCGR3B"],
    "Erythroid": ["HBB", "HBA1", "HBA2", "GYPA"],
    "Neuron": ["SNAP25", "RBFOX3", "SYT1", "STMN2", "PRPH", "NEFL", "NEFH", "SCN10A"],
    "Satellite_glia": ["FABP7", "APOE", "KCNJ10", "GLUL", "S100B"],
    "Schwann": ["MPZ", "PMP22", "PRX", "NCMAP", "SOX10", "EGR2"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOG", "MOBP", "ST18"],
    "OPC": ["PDGFRA", "OLIG2", "VCAN", "PTPRZ1", "CSPG4"],
    "Astrocyte": ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1"],
    "Microglia": ["P2RY12", "CX3CR1", "TMEM119", "CSF1R", "C3"],
    "Ependymal": ["FOXJ1", "PIFO", "DYNLRB2", "CFAP43"],
    "Adipocyte": ["ADIPOQ", "PLIN1", "LEP", "CIDEA"],
    "Mast": ["TPSAB1", "CPA3", "KIT"],
}
TISSUE_ALLOWED = {  # restrict labels to cell types plausible for the tissue
    "disc": ["NP_chondrocyte", "AF_fibrochondrocyte", "Fibroblast", "Endothelial", "Pericyte_SMC", "Macrophage_myeloid", "T_NK", "B_plasma", "Neutrophil", "Erythroid", "MSC_stromal", "Mast"],
    "ligament": ["Fibroblast", "AF_fibrochondrocyte", "NP_chondrocyte", "Endothelial", "Pericyte_SMC", "Macrophage_myeloid", "T_NK", "B_plasma", "Neutrophil", "Erythroid", "Adipocyte", "Mast", "Schwann"],
    "bone": ["Osteoblast_osteoprogenitor", "MSC_stromal", "Osteoclast", "Fibroblast", "NP_chondrocyte", "Endothelial", "Pericyte_SMC", "Macrophage_myeloid", "T_NK", "B_plasma", "Neutrophil", "Erythroid", "Adipocyte", "Mast"],
    "drg": ["Neuron", "Satellite_glia", "Schwann", "Fibroblast", "Endothelial", "Pericyte_SMC", "Macrophage_myeloid", "T_NK", "Erythroid", "Adipocyte"],
    "cord": ["Neuron", "Oligodendrocyte", "OPC", "Astrocyte", "Microglia", "Ependymal", "Endothelial", "Pericyte_SMC", "Fibroblast", "Schwann", "Macrophage_myeloid", "T_NK"],
}
TISSUE = {"GSE165722": "disc", "GSE160756": "disc", "GSE169396": "bone", "GSE267819": "ligament",
          "GSE294458": "ligament", "GSE168243": "drg", "GSE190442": "cord"}


def cellfilter(a):
    a.var_names_make_unique()
    n = np.asarray(a.X.sum(1)).ravel(); g = np.asarray((a.X > 0).sum(1)).ravel()
    a = a[(n >= 500) & (g >= 250)].copy()
    if a.n_obs > MAXCELLS_PER_SAMPLE:
        sc.pp.subsample(a, n_obs=MAXCELLS_PER_SAMPLE, random_state=0)
    return a


def load():
    L = []
    if ds == "GSE165722":
        for f in sorted(glob.glob(f"{raw}/*counts.tsv.gz")):
            df = pd.read_csv(f, sep="\t", index_col=0)
            a = ad.AnnData(sp.csr_matrix(df.T.values.astype(np.float32)), obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
            a.obs["sample"] = os.path.basename(f).split("_")[1].split(".")[0]; L.append(cellfilter(a))
    elif ds == "GSE160756":
        for f in sorted(glob.glob(f"{raw}/*.loom.gz")):
            import h5py
            with tempfile.NamedTemporaryFile(suffix=".loom", dir="/tmp/claude-0") as t:
                with gzip.open(f) as fi: shutil.copyfileobj(fi, t)
                t.flush()
                with h5py.File(t.name, "r") as h:  # dense genes x cells matrix: read in column blocks
                    g = [x.decode() if isinstance(x, bytes) else x for x in h["row_attrs/gene_names"][:]]
                    c = [x.decode() if isinstance(x, bytes) else x for x in h["col_attrs/cell_names"][:]]
                    M = h["matrix"]; parts = []
                    for i in range(0, M.shape[1], 2000):
                        parts.append(sp.csr_matrix(M[:, i:i + 2000].T.astype(np.float32)))
            a = ad.AnnData(sp.vstack(parts).tocsr(), obs=pd.DataFrame(index=c), var=pd.DataFrame(index=g))
            a.obs["sample"] = os.path.basename(f).split("_umi_")[1].split(".")[0]; L.append(cellfilter(a))
    elif ds in ("GSE169396", "GSE267819", "GSE294458"):
        for m in sorted(glob.glob(f"{raw}/*_matrix.mtx.gz")):
            p = m.replace("_matrix.mtx.gz", "_")
            X = sc.read_mtx(m).X.T.tocsr().astype(np.float32)
            bc = pd.read_csv(p + "barcodes.tsv.gz", header=None, sep="\t")[0].values
            ft = pd.read_csv(p + "features.tsv.gz", header=None, sep="\t")
            a = ad.AnnData(X, obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=ft[1].values))
            a.obs["sample"] = os.path.basename(p).split("_")[1]; L.append(cellfilter(a))
    elif ds == "GSE168243":
        for f in sorted(glob.glob(f"{raw}/*.csv.gz")):
            df = pd.read_csv(f, index_col=0)
            a = ad.AnnData(sp.csr_matrix(df.T.values.astype(np.float32)), obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
            a.obs["sample"] = os.path.basename(f).split("_")[1].split(".")[0]; L.append(cellfilter(a))
    elif ds == "GSE190442":
        for f in sorted(glob.glob(f"{raw}/*.h5")):
            a = sc.read_10x_h5(f); a.X = a.X.astype(np.float32)
            a.obs["sample"] = os.path.basename(f).split("_")[1]; L.append(cellfilter(a))
    for a in L:
        a.obs_names = [f"{a.obs['sample'].iloc[0]}_{x}" for x in a.obs_names]
    a = ad.concat(L, join="inner")
    return a


a = load()
print(ds, "cells after UMI filter", a.n_obs, "genes", a.n_vars, flush=True)
a.var["mt"] = a.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
mt_max = 5 if TISSUE[ds] in ("cord",) else 20
a = a[a.obs.pct_counts_mt < mt_max].copy()
a = a[:, ~a.var["mt"]].copy()
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample" if a.obs["sample"].nunique() > 1 else None)
b = a[:, a.var.highly_variable].copy()
sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=30)
if a.obs["sample"].nunique() > 1:
    import harmonypy
    ho = harmonypy.run_harmony(b.obsm["X_pca"], b.obs, "sample", max_iter_harmony=20)
    Z = np.asarray(ho.Z_corr)
    b.obsm["X_harmony"] = Z.T if Z.shape[0] != b.n_obs else Z
    rep = "X_harmony"
else:
    rep = "X_pca"
sc.pp.neighbors(b, use_rep=rep, n_neighbors=20); sc.tl.leiden(b, resolution=1.0, flavor="igraph", n_iterations=2, key_added="cl")
sc.tl.umap(b)
a.obs["cl"] = b.obs["cl"].values
# marker scoring: per-cell score_genes (markers vs expression-matched random genes), averaged per cluster
allowed = TISSUE_ALLOWED[TISSUE[ds]]
expr = pd.DataFrame(index=a.obs.cl.cat.categories)
for t in allowed:
    g = [x for x in MARKERS[t] if x in a.var_names]
    if len(g) < 2:
        continue
    sc.tl.score_genes(a, g, score_name="s_" + t, random_state=0)
    expr[t] = a.obs.groupby("cl", observed=True)["s_" + t].mean()
lab = expr.idxmax(1); best = expr.max(1)
second = expr.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
lab_tab = pd.DataFrame(dict(label=lab, score=best.round(2), margin=(best - second).round(2),
                            n_cells=a.obs.cl.value_counts().reindex(lab.index)))
lab_tab.to_csv(f"{out}/{ds}.cluster_labels.tsv", sep="\t")
a.obs["celltype"] = a.obs.cl.map(lab).astype(str)
b.obs["celltype"] = a.obs["celltype"].values
sc.pl.umap(b, color=["celltype", "sample"], show=False, save=False, wspace=0.5)
import matplotlib.pyplot as plt
plt.savefig(f"{out}/{ds}.umap.png", dpi=80, bbox_inches="tight"); plt.close()
# top cluster markers for manual review
sc.tl.rank_genes_groups(a, "cl", method="t-test", n_genes=15)
mk = pd.DataFrame({c: a.uns["rank_genes_groups"]["names"][c] for c in a.obs.cl.cat.categories})
mk.T.assign(label=lab).to_csv(f"{out}/{ds}.cluster_markers.tsv", sep="\t")
allm = sorted({g for v in MARKERS.values() for g in v if g in a.var_names})
cm = pd.DataFrame(a[:, allm].X.toarray(), columns=allm).groupby(a.obs.cl.values, observed=True).mean().round(2)
cm.to_csv(f"{out}/{ds}.cluster_canonical_means.tsv", sep="\t")
a.obs[["sample", "cl", "celltype"]].to_csv(f"{out}/{ds}.cells.tsv.gz", sep="\t", compression="gzip")
ad.AnnData(a.layers["counts"], obs=a.obs[["sample", "cl"]], var=pd.DataFrame(index=a.var_names)).write_h5ad(f"{out}/{ds}.counts.h5ad", compression="gzip")
# pseudobulk CPM per cell type (cell types with >= 50 cells)
cnt = a.layers["counts"]
keep = a.obs.celltype.value_counts(); keep = keep[keep >= 50].index
pb = {}
for t in keep:
    s = np.asarray(cnt[(a.obs.celltype == t).values].sum(0)).ravel()
    pb[t] = s / s.sum() * 1e6
pb = pd.DataFrame(pb, index=a.var_names)
pb.to_csv(f"{out}/{ds}.pseudobulk_cpm.tsv.gz", sep="\t", compression="gzip", float_format="%.3f")
print(ds, "cells", a.n_obs, a.obs.celltype.value_counts().to_dict(), flush=True)
