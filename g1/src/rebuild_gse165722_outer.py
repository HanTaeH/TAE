"""Rebuild the GSE165722 counts matrix with an OUTER gene join across samples (missing gene = 0 counts).
GEO ships one counts table per sample with zero-count genes dropped, and sc_process.py joined samples with
join="inner", which kept only 13,887 genes. Cells and cluster labels are taken unchanged from the
existing h5ad, so the manual cluster labels still apply.
Usage: python -I rebuild_gse165722_outer.py <raw_dir> <old.counts.h5ad> <new.counts.h5ad>
"""
import sys, glob, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

raw, old_f, new_f = sys.argv[1:4]
old = ad.read_h5ad(old_f, backed="r")
keep = set(old.obs_names)
L = []
for f in sorted(glob.glob(f"{raw}/*counts.tsv.gz")):
    s = os.path.basename(f).split("_")[1].split(".")[0]
    df = pd.read_csv(f, sep="\t", index_col=0)
    df.columns = [f"{s}_{c}" for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c in keep]]
    a = ad.AnnData(sp.csr_matrix(df.T.values.astype(np.float32)), obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
    a.var_names_make_unique()
    L.append(a)
a = ad.concat(L, join="outer", fill_value=0)
a = a[old.obs_names].copy()
a.obs = old.obs.loc[a.obs_names].copy()
a.X = sp.csr_matrix(a.X)
a.write_h5ad(new_f)
print("old", old.shape, "new", a.shape)
