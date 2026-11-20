"""Write full-format per-set .annot.gz files (CHR BP SNP CM <set>) next to split LD scores,
needed for ldsc --h2 --overlap-annot. Usage: python write_full_annot.py <thin_prefix> <plink_prefix> <out_dir> <set> [<set> ...]"""
import sys
import pandas as pd
pre, pl, out, sets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
for c in range(1, 23):
    bim = pd.read_csv(f"{pl}.{c}.bim", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    th = pd.read_csv(f"{pre}.{c}.annot.gz", sep="\t", usecols=sets)
    for s in sets:
        d = bim[["CHR", "BP", "SNP", "CM"]].copy(); d[s] = th[s].values
        d.to_csv(f"{out}/{s}.{c}.annot.gz", sep="\t", index=False, compression="gzip")
print("wrote", sets)
