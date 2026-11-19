"""GWAS-by-subtraction for two traits (Demange et al. 2021 identification, Cholesky form).
Trait A (e.g. sciatica) loads only on the shared factor; trait B (e.g. stenosis) loads on
the shared factor and on its own specific factor. Loadings come from the LDSC genetic
covariance S (observed scale); per-SNP sampling covariance of the standardized effects uses
the LDSC intercept matrix I (absorbs sample overlap and inflation, as genomic SEM does).

Usage: python subtraction.py <S_cohort.npz> <A_name> <B_name> <A.sumstats.gz> <B.sumstats.gz> <ldscore.gz> <out_prefix>
Writes <out_prefix>.shared.sumstats.gz and <out_prefix>.<B>_specific.sumstats.gz
(SNP A1 A2 Z N; N = median 1/se^2 on the scale of trait B's / A's phenotype, see below).
"""
import sys
import numpy as np
import pandas as pd

npz, A, B, fa, fb, ldf, out = sys.argv[1:8]
r = np.load(npz, allow_pickle=True)
names = list(r["names"]); ia, ib = names.index(A), names.index(B)
S, I = r["S"], r["I"]
lam_a = np.sqrt(S[ia, ia])                 # A <- shared
lam_bs = S[ia, ib] / lam_a                 # B <- shared
lam_bu = np.sqrt(S[ib, ib] - lam_bs ** 2)  # B <- specific
print(f"loadings: {A}<-shared {lam_a:.4f}, {B}<-shared {lam_bs:.4f}, {B}<-specific {lam_bu:.4f}; "
      f"share of {B} h2 specific {lam_bu**2/S[ib, ib]:.3f}")

snps = pd.read_csv(ldf, sep="\t", usecols=["SNP"])
a = pd.read_csv(fa, sep="\t").dropna(subset=["Z"])
b = pd.read_csv(fb, sep="\t").dropna(subset=["Z"])
d = snps.merge(a, on="SNP").merge(b, on="SNP", suffixes=("_a", "_b"))
flip = (d.A1_a == d.A2_b) & (d.A2_a == d.A1_b)
same = (d.A1_a == d.A1_b) & (d.A2_a == d.A2_b)
d = d[same | flip].copy()
d.loc[flip[same | flip].values, "Z_b"] *= -1
na, nb = d.N_a.to_numpy(float), d.N_b.to_numpy(float)
ba, bb = d.Z_a / np.sqrt(na), d.Z_b / np.sqrt(nb)
va, vb = I[ia, ia] / na, I[ib, ib] / nb
cab = I[ia, ib] / np.sqrt(na * nb)
# shared factor effect and specific factor effect (latent SD units)
g_s = ba / lam_a
var_s = va / lam_a ** 2
c1 = lam_bs / lam_a
g_u = (bb - c1 * ba) / lam_bu
var_u = (vb + c1 ** 2 * va - 2 * c1 * cab) / lam_bu ** 2
for tag, g, v, scale in [("shared", g_s, var_s, lam_a), (f"{B}_specific", g_u, var_u, lam_bu)]:
    z = g / np.sqrt(v)
    # effective N on the scale where this component's variance is scale^2 (its observed-scale h2)
    neff = np.median(1 / (v * scale ** 2))
    o = pd.DataFrame(dict(SNP=d.SNP, A1=d.A1_a, A2=d.A2_a, Z=z, N=np.round(neff)))
    o.to_csv(f"{out}.{tag}.sumstats.gz", sep="\t", index=False, compression="gzip")
    print(tag, "n", len(o), "Neff", round(neff), "mean chi2", round(float(np.mean(z ** 2)), 4),
          "n p<5e-8", int((z ** 2 > 29.7168).sum()))
