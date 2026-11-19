"""Sample-size weighted Z meta-analysis of LDSC .sumstats.gz files (Stouffer, weights sqrt(N)).
Usage: python meta_z.py <out.sumstats.gz> <in1.sumstats.gz> <in2.sumstats.gz> ...
Alleles aligned to the first file; SNPs must be present in all inputs."""
import sys
import numpy as np
import pandas as pd
out, fs = sys.argv[1], sys.argv[2:]
d = None
for i, f in enumerate(fs):
    x = pd.read_csv(f, sep="\t").dropna(subset=["Z"]).rename(columns={"A1": f"A1_{i}", "A2": f"A2_{i}", "Z": f"Z_{i}", "N": f"N_{i}"})
    d = x if d is None else d.merge(x, on="SNP")
for i in range(1, len(fs)):
    same = (d[f"A1_{i}"] == d.A1_0) & (d[f"A2_{i}"] == d.A2_0)
    flip = (d[f"A1_{i}"] == d.A2_0) & (d[f"A2_{i}"] == d.A1_0)
    d = d[same | flip].copy(); d.loc[flip[same | flip].values, f"Z_{i}"] *= -1
w = np.column_stack([np.sqrt(d[f"N_{i}"]) for i in range(len(fs))])
Z = np.column_stack([d[f"Z_{i}"] for i in range(len(fs))])
z = (w * Z).sum(1) / np.sqrt((w ** 2).sum(1))
n = d[[f"N_{i}" for i in range(len(fs))]].sum(1)
pd.DataFrame(dict(SNP=d.SNP, A1=d.A1_0, A2=d.A2_0, Z=z, N=n)).to_csv(out, sep="\t", index=False, compression="gzip")
print(out, len(d), "mean chi2", round(float(np.mean(z ** 2)), 4))
