"""Stream-read only the RNA counts (dgCMatrix) from a large Seurat .rds.gz without loading the whole object.
R XDR serialization is sequential and in a Seurat v3/v4 object the first attribute is @assays, whose first
assay's first slot is @counts; we parse up to the end of counts and stop (data/scale.data never read).
Used for GSE271018 HC2 (Tendon_C3, 3.7 GB compressed), too large for the in-memory parser. No meta.data.
Usage: python -I seurat_rds_counts_stream.py <in.rds.gz> <out.h5ad>
"""
import sys, gzip, struct
import numpy as np, scipy.sparse as sp, pandas as pd, anndata as ad

f = gzip.open(sys.argv[1], "rb")
syms = []                      # reference table (symbols only are needed before counts)


def rd(n):
    b = f.read(n)
    assert len(b) == n
    return b


def i32():
    return struct.unpack(">i", rd(4))[0]


def length():
    n = i32()
    if n == -1:
        hi, lo = i32(), i32(); n = (hi << 32) + lo
    return n


def item():
    flags = i32(); t = flags & 0xFF
    has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
    if t == 254: return None                                  # NILVALUE
    if t == 255: return syms[(flags >> 8) - 1 if (flags >> 8) else i32() - 1]
    if t == 1:                                                # SYMSXP
        s = item(); syms.append(("SYM", s)); return ("SYM", s)
    if t == 9:                                                # CHARSXP
        n = i32(); return None if n == -1 else rd(n).decode("utf-8", "replace")
    if t == 2:                                                # LISTSXP (pairlist)
        out = []
        while True:
            attr = item() if has_attr else None
            tag = item() if has_tag else None
            val = item()
            out.append((tag[1] if tag else None, val))
            flags = i32(); t = flags & 0xFF
            if t == 254: break
            has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        return ("LIST", out)
    if t in (10, 13):  v = np.frombuffer(rd(4 * length()), ">i4")
    elif t == 14:      v = np.frombuffer(rd(8 * length()), ">f8")
    elif t == 16:      v = [item() for _ in range(length())]
    elif t == 19:      v = [item() for _ in range(length())]
    elif t == 25:      v = "S4"
    else: raise RuntimeError(f"type {t} not handled")
    return ("VAL", t, v, attr_list() if has_attr else None)


STOP = {}


def attr_list():
    """Read an attribute pairlist; stop the whole parse right after an S4 dgCMatrix named 'counts'."""
    flags = i32(); assert flags & 0xFF == 2
    out = {}
    while True:
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if has_attr: item()
        tag = item()[1] if has_tag else None
        val = item()
        out[tag] = val
        if tag == "counts":
            STOP["counts"] = val; raise StopIteration
        flags = i32()
        if flags & 0xFF == 254: return out


assert rd(2) == b"X\n"
ver = i32(); i32(); i32()
if ver == 3: rd(i32())                                         # native encoding
try:
    item()
except StopIteration:
    pass
c = STOP["counts"][3]
i, p, x = c["i"][2].astype(np.int32), c["p"][2].astype(np.int64), c["x"][2].astype(np.float32)
nr, nc = c["Dim"][2]
dn = c["Dimnames"][2]
genes, cells = dn[0][2], dn[1][2]
X = sp.csc_matrix((x, i, p), shape=(nr, nc)).T.tocsr()
a = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
a.var_names_make_unique(); a.write_h5ad(sys.argv[2], compression="gzip")
print(sys.argv[1], "cells", nc, "genes", nr, "total counts", int(X.sum()))
