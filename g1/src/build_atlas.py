"""Build a cross-tissue cell-type atlas and LDSC-SEG style gene sets.

Inputs: per-dataset counts h5ad from sc_process.py, a manual cluster -> label map
(cluster_labels_manual.tsv: dataset, cluster, label; label 'drop' removes a cluster), and the
SEA-AD MTG broad-type pseudobulk (CPM) for brain.
Labels are 'Tissue_Type' for tissue-specific cells (e.g. Disc_NP, DRG_Neuron, Cord_Oligo) and
plain names for cell types shared across tissues (Endothelial, Macrophage, T_NK, ...); shared
types are pooled across datasets so they do not compete with each other.
Each atlas column = mean of per-dataset CPM (protein-coding genes, renormalised).
Specificity (Skene et al.) = CPM / sum over columns; the top 10% of expressed genes per column
form the gene set. An 'all_genes' control set holds all protein-coding genes in the atlas.

Usage: python -I build_atlas.py <sc_dir> <labels.tsv> <seaad_ref.csv.gz> <genes_hg19.tsv> <out_dir>
"""
import sys, os
import numpy as np
import pandas as pd
import anndata as ad

scd, labf, seaad, genes_f, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
genes = pd.read_csv(genes_f, sep="\t").drop_duplicates("SYMBOL")
pc = set(genes.SYMBOL)
lab = pd.read_csv(labf, sep="\t", dtype=str)
cols, ncell = {}, []
for ds, g in lab.groupby("dataset"):
    a = ad.read_h5ad(f"{scd}/{ds}.counts.h5ad")
    m = dict(zip(g.cluster, g.label))
    a.obs["label"] = a.obs.cl.astype(str).map(m).fillna("drop")
    keep = [x for x in a.var_names if x in pc]
    a = a[:, keep]
    for l, n in a.obs.label.value_counts().items():
        if l == "drop" or n < 30:
            continue
        s = np.asarray(a.X[(a.obs.label == l).values].sum(0)).ravel()
        cols.setdefault(l, []).append(pd.Series(s / s.sum() * 1e6, index=keep))
        ncell.append(dict(dataset=ds, label=l, n_cells=n))
br = pd.read_csv(seaad, index_col=0)
br = br.loc[br.index.isin(pc)]
br["Inh"] = br[["Inh_SST", "Inh_PVALB", "Inh_other"]].mean(1)
for c, l in [("Exc", "Brain_ExcNeuron"), ("Inh", "Brain_InhNeuron"), ("Astro", "Brain_Astro"),
             ("Oligo", "Brain_Oligo"), ("OPC", "Brain_OPC"), ("Micro", "Brain_Microglia")]:
    cols.setdefault(l, []).append(br[c] / br[c].sum() * 1e6)
    ncell.append(dict(dataset="SEA-AD", label=l, n_cells=np.nan))
common = sorted(set.intersection(*[set(s.index) for v in cols.values() for s in v]))
atlas = pd.DataFrame({l: pd.concat([s.reindex(common) for s in v], axis=1).mean(1) for l, v in cols.items()})
atlas = atlas / atlas.sum() * 1e6
atlas = atlas[atlas.max(1) > 1]  # expressed (CPM > 1) in at least one column
spec = atlas.div(atlas.sum(1), axis=0)
k = int(round(0.10 * len(atlas)))
rows = []
for c in atlas.columns:
    top = spec[c].sort_values(ascending=False).index[:k]
    rows += [(c, g) for g in top]
rows += [("all_genes", g) for g in atlas.index]
pd.DataFrame(rows, columns=["set", "SYMBOL"]).to_csv(f"{out}/genesets.tsv", sep="\t", index=False)
atlas.to_csv(f"{out}/atlas_cpm.tsv.gz", sep="\t", compression="gzip", float_format="%.3f")
pd.DataFrame(ncell).to_csv(f"{out}/atlas_ncells.tsv", sep="\t", index=False)
print("atlas", atlas.shape, "top-10% size", k)
print(pd.DataFrame(ncell).groupby("label").n_cells.sum().to_string())
