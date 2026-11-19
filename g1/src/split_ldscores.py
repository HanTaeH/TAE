"""Split multi-column LD-score output (<prefix>.<chr>.l2.ldscore.gz, .M, .M_5_50) into one
LD-score prefix per annotation column: <out_dir>/<set>.<chr>.l2.ldscore.gz etc.
Usage: python split_ldscores.py <prefix> <out_dir>   (set names from <prefix>.names)"""
import sys
import pandas as pd
pre, out = sys.argv[1:3]
names = [l.strip() for l in open(pre + ".names") if l.strip()]
for c in range(1, 23):
    ld = pd.read_csv(f"{pre}.{c}.l2.ldscore.gz", sep="\t")
    M = open(f"{pre}.{c}.l2.M").read().split()
    M5 = open(f"{pre}.{c}.l2.M_5_50").read().split()
    for i, s in enumerate(names):
        d = ld[["CHR", "SNP", "BP", f"{s}L2"]].rename(columns={f"{s}L2": "L2"})
        d.to_csv(f"{out}/{s}.{c}.l2.ldscore.gz", sep="\t", index=False, compression="gzip", float_format="%.3f")
        open(f"{out}/{s}.{c}.l2.M", "w").write(M[i] + "\n")
        open(f"{out}/{s}.{c}.l2.M_5_50", "w").write(M5[i] + "\n")
print("split", len(names), "sets")
