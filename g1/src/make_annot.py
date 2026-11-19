"""Build thin S-LDSC annotation files (one binary column per gene set) from gene sets.
A SNP is in a set if it lies within +/- WINDOW bp of any gene in the set (hg19 gene spans).

Usage: python make_annot.py <genesets.tsv> <genes_hg19.tsv> <plink_prefix> <out_prefix> [window=100000]
genesets.tsv: columns set, SYMBOL. Writes <out_prefix>.<chr>.annot.gz for chr 1-22 and
<out_prefix>.names (column order).
"""
import sys
import numpy as np
import pandas as pd

gs_f, genes_f, plink, out = sys.argv[1:5]
win = int(sys.argv[5]) if len(sys.argv) > 5 else 100_000
gs = pd.read_csv(gs_f, sep="\t")
genes = pd.read_csv(genes_f, sep="\t").drop_duplicates("SYMBOL")
sets = list(dict.fromkeys(gs["set"]))
open(out + ".names", "w").write("\n".join(sets) + "\n")
for c in range(1, 23):
    bim = pd.read_csv(f"{plink}.{c}.bim", sep="\t", header=None, usecols=[3])[3].to_numpy()
    cols = {}
    for s in sets:
        g = genes[(genes.CHR.astype(str) == str(c)) & genes.SYMBOL.isin(gs.loc[gs["set"] == s, "SYMBOL"])]
        st = np.sort(np.maximum(g.START.to_numpy() - win, 0)); en = np.sort(g.END.to_numpy() + win)
        # SNP covered if #starts <= bp exceeds #ends < bp
        cols[s] = ((np.searchsorted(st, bim, side="right") - np.searchsorted(en, bim, side="left")) > 0).astype(np.int8)
    pd.DataFrame(cols).to_csv(f"{out}.{c}.annot.gz", sep="\t", index=False, compression="gzip")
print(out, len(sets), "sets")
