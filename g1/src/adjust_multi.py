"""Experiment 5b: remove the genetic components of one or more covariate traits (e.g. height, BMI)
from a target trait, GWAS-by-subtraction / Cholesky generalisation to k covariates:
  b = c'a + lam_u * u,   c = S_AA^-1 S_Ab,   lam_u^2 = S_bb - S_bA S_AA^-1 S_Ab
  g_u = (beta_b - c'beta_a) / lam_u
  var(g_u) = (v_b + c' V_a c - 2 c' v_ab) / lam_u^2      (V from the LDSC intercept matrix I)
Usage: python -I adjust_multi.py <S.npz> <target> <cov1,cov2> <target.sumstats.gz> <cov1.sumstats.gz,cov2.sumstats.gz> <ldscore.gz> <out.sumstats.gz>
"""
import sys
import numpy as np
import pandas as pd

npz, B, covs, fb, fcov, ldf, out = sys.argv[1:8]
covs = covs.split(","); fcov = fcov.split(",")
r = np.load(npz, allow_pickle=True)
names = list(r["names"]); S, I = r["S"], r["I"]
ib = names.index(B); ia = [names.index(c) for c in covs]
Saa = S[np.ix_(ia, ia)]; Sab = S[ia, ib]
c = np.linalg.solve(Saa, Sab)
lam2 = S[ib, ib] - Sab @ c
print("coefficients", dict(zip(covs, c.round(4))), "share of target h2 kept", round(lam2 / S[ib, ib], 3))

d = pd.read_csv(ldf, sep="\t", usecols=["SNP"]).merge(pd.read_csv(fb, sep="\t").dropna(subset=["Z"]), on="SNP")
for k, f in enumerate(fcov):
    a = pd.read_csv(f, sep="\t").dropna(subset=["Z"]).rename(columns={"A1": f"A1_{k}", "A2": f"A2_{k}", "Z": f"Z_{k}", "N": f"N_{k}"})
    d = d.merge(a, on="SNP")
    same = (d.A1 == d[f"A1_{k}"]) & (d.A2 == d[f"A2_{k}"])
    flip = (d.A1 == d[f"A2_{k}"]) & (d.A2 == d[f"A1_{k}"])
    d = d[same | flip].copy()
    d.loc[flip[same | flip].values, f"Z_{k}"] *= -1
nb = d.N.to_numpy(float)
bb = d.Z / np.sqrt(nb)
num = bb.copy(); var = I[ib, ib] / nb
for k in range(len(covs)):
    nk = d[f"N_{k}"].to_numpy(float); bk = d[f"Z_{k}"] / np.sqrt(nk)
    num = num - c[k] * bk
    var = var - 2 * c[k] * I[ia[k], ib] / np.sqrt(nk * nb)
    for j in range(len(covs)):
        nj = d[f"N_{j}"].to_numpy(float)
        var = var + c[k] * c[j] * I[ia[k], ia[j]] / np.sqrt(nk * nj)
z = num / np.sqrt(var)                 # = g_u / se(g_u)
neff = np.median(1 / var)              # on the scale where the residual component has variance lam^2
o = pd.DataFrame(dict(SNP=d.SNP, A1=d.A1, A2=d.A2, Z=z, N=np.round(neff)))
o.to_csv(out, sep="\t", index=False, compression="gzip")
print("n", len(o), "Neff", round(neff), "mean chi2", round(float(np.mean(z ** 2)), 4))
