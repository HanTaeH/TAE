"""Read a Seurat v3/v4 object saved as .rds(.gz) without R (rdata parser) and write raw counts + meta.data
to .h5ad. Used for GSE271018 (spinal ligament scRNA-seq, Seurat objects only).
Usage: python -I seurat_rds_to_h5ad.py <in.rds.gz> <out.h5ad>
"""
import sys, gzip
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, rdata
from rdata.parser import RObjectType as T

src, out = sys.argv[1:3]
op = gzip.open if src.endswith(".gz") else open
o = rdata.parser.parse_data(op(src, "rb").read()).object


def name_of(tag):
    while tag is not None and tag.info.type == T.REF:
        tag = tag.referenced_object
    if tag is None:
        return None
    return tag.value.value.decode() if tag.info.type == T.SYM else None


def deref(x):
    while x.info.type == T.REF:
        x = x.referenced_object
    return x


def attrs(x):
    res, cur = {}, x.attributes
    while cur is not None and cur.info.type == T.LIST:
        res[name_of(cur.tag)] = deref(cur.value[0])
        cur = cur.value[1] if len(cur.value) > 1 else None
    return res


def strs(x):
    x = deref(x)
    return [None if s.value is None else s.value.decode() for s in x.value]


def named_list(x):
    x = deref(x)
    nm = strs(attrs(x)["names"])
    return dict(zip(nm, [deref(v) for v in x.value]))


def df_from(x):
    a = attrs(x)
    cols = {}
    for k, v in named_list(x).items():
        va = attrs(v)
        if v.info.type == T.STR:
            cols[k] = strs(v)
        elif "levels" in va:
            lev = strs(va["levels"]); codes = np.asarray(v.value)
            cols[k] = [lev[c - 1] if c > 0 else None for c in codes]
        else:
            cols[k] = np.asarray(v.value)
    rn = a.get("row.names")
    idx = strs(rn) if rn is not None and rn.info.type == T.STR else None
    return pd.DataFrame(cols, index=idx)


top = attrs(o)
assays = named_list(top["assays"])
print("assays", list(assays))
rna = attrs(assays.get("RNA", next(iter(assays.values()))))
print("RNA slots", list(rna))
m = attrs(rna["counts"])
i, p, x = np.asarray(m["i"].value), np.asarray(m["p"].value), np.asarray(m["x"].value)
nr, nc = np.asarray(m["Dim"].value)
dn = deref(m["Dimnames"]).value
genes, cells = strs(dn[0]), strs(dn[1])
X = sp.csc_matrix((x.astype(np.float32), i, p), shape=(nr, nc)).T.tocsr()
meta = df_from(top["meta.data"])
meta = meta.reindex(cells)
a = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=genes))
a.var_names_make_unique()
a.write_h5ad(out, compression="gzip")
print(src, "cells", nc, "genes", nr, "total counts", int(X.sum()), "is integer", bool(np.all(np.mod(x[:1000], 1) == 0)))
print(meta.head(3).to_string()); print(list(meta.columns))
