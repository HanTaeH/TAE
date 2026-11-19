"""Multivariable LDSC in Python (genomic-SEM style): genetic covariance matrix S,
its sampling covariance V (block jackknife over all elements jointly) and the
LDSC intercept matrix I, all on the observed scale.

Usage: python mvldsc.py <ldscore.l2.ldscore.gz> <M_5_50 file> <out_prefix> <name=file.sumstats.gz> ...
Writes <out_prefix>.npz (S, V, I, I_se, S_se, names, n_snps) and <out_prefix>.tsv (S, rg, I).
Regression for pair (i,j): z_i z_j = sqrt(N_i N_j) g_ij l / M + I_ij, LDSC rg weights,
two-step (weights from an unweighted fit), 200 contiguous jackknife blocks,
SNPs with any z^2 > max(80, 0.001 N) dropped.
"""
import sys
import numpy as np
import pandas as pd

NBLOCK = 200


def load(ld_path, m_path, specs):
    ld = pd.read_csv(ld_path, sep="\t", usecols=["SNP", "L2"])
    M = float(open(m_path).read().split()[0])
    names, df = [], ld
    ref = None
    for s in specs:
        name, f = s.split("=", 1)
        x = pd.read_csv(f, sep="\t").dropna(subset=["Z"])
        x = x.rename(columns={"Z": f"Z_{name}", "N": f"N_{name}", "A1": f"A1_{name}", "A2": f"A2_{name}"})
        df = df.merge(x, on="SNP")
        if ref is None:
            ref = name
        else:  # align alleles to the first trait
            same = (df[f"A1_{name}"] == df[f"A1_{ref}"]) & (df[f"A2_{name}"] == df[f"A2_{ref}"])
            flip = (df[f"A1_{name}"] == df[f"A2_{ref}"]) & (df[f"A2_{name}"] == df[f"A1_{ref}"])
            df = df[same | flip].copy()
            df.loc[flip[same | flip].values, f"Z_{name}"] *= -1
        names.append(name)
    return df.reset_index(drop=True), names, M


def wls_blocks(x, y, w, blocks):
    """Weighted y = a*x + b with per-block sums -> full estimate and delete-one estimates."""
    X = np.column_stack([x, np.ones_like(x)])
    Xw = X * w[:, None]
    xtx = np.stack([np.add.reduceat(Xw[:, a] * X[:, b], blocks) for a in range(2) for b in range(2)], 1).reshape(-1, 2, 2)
    xty = np.stack([np.add.reduceat(Xw[:, a] * y, blocks) for a in range(2)], 1)
    full = np.linalg.solve(xtx.sum(0), xty.sum(0))
    dele = np.linalg.solve(xtx.sum(0)[None] - xtx, (xty.sum(0)[None] - xty)[..., None])[..., 0]
    return full, dele


def mvldsc(df, names, M):
    k = len(names)
    Z = df[[f"Z_{n}" for n in names]].to_numpy(float)
    N = df[[f"N_{n}" for n in names]].to_numpy(float)
    keep = (Z ** 2 < np.maximum(80, 0.001 * N)).all(1)
    Z, N, l = Z[keep], N[keep], df["L2"].to_numpy(float)[keep]
    lw = np.maximum(l, 1.0)
    n = len(l)
    blocks = np.linspace(0, n, NBLOCK + 1).astype(int)[:-1]
    # step 1: univariate h2 (unweighted then weighted) for weights
    h2, I = np.zeros(k), np.ones(k)
    for i in range(k):
        x = N[:, i] * l / M
        (a, b), _ = wls_blocks(x, Z[:, i] ** 2, 1 / lw, blocks)
        a = np.clip(a, 1e-4, 1)
        w = 1 / (lw * (b + N[:, i] * a * l / M) ** 2)
        (h2[i], I[i]), _ = wls_blocks(x, Z[:, i] ** 2, w, blocks)
    h2c = np.clip(h2, 1e-4, 1)
    est, pv, pairs = [], [], []
    S = np.zeros((k, k)); Imat = np.zeros((k, k)); Ise = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            x = np.sqrt(N[:, i] * N[:, j]) * l / M
            y = Z[:, i] * Z[:, j]
            if i == j:
                w = 1 / (lw * (I[i] + N[:, i] * h2c[i] * l / M) ** 2)
            else:
                (g0, r0), _ = wls_blocks(x, y, 1 / lw, blocks)
                vi = I[i] + N[:, i] * h2c[i] * l / M
                vj = I[j] + N[:, j] * h2c[j] * l / M
                w = 1 / (lw * (vi * vj + (x * g0 + r0) ** 2))
            full, dele = wls_blocks(x, y, w, blocks)
            S[i, j] = S[j, i] = full[0]
            Imat[i, j] = Imat[j, i] = full[1]
            pvals = NBLOCK * full[None] - (NBLOCK - 1) * dele
            Ise[i, j] = Ise[j, i] = np.sqrt(np.var(pvals[:, 1], ddof=1) / NBLOCK)
            pv.append(pvals[:, 0]); pairs.append((i, j))
    PV = np.column_stack(pv)
    V = np.cov(PV, rowvar=False) / NBLOCK
    Sse = np.zeros((k, k))
    for c, (i, j) in enumerate(pairs):
        Sse[i, j] = Sse[j, i] = np.sqrt(V[c, c])
    return dict(S=S, V=V, I=Imat, I_se=Ise, S_se=Sse, names=np.array(names), n_snps=n, pairs=np.array(pairs))


if __name__ == "__main__":
    ld_path, m_path, out = sys.argv[1:4]
    df, names, M = load(ld_path, m_path, sys.argv[4:])
    r = mvldsc(df, names, M)
    np.savez(out + ".npz", **r)
    d = np.sqrt(np.diag(r["S"]))
    rg = r["S"] / np.outer(d, d)
    with open(out + ".tsv", "w") as fh:
        fh.write(f"# n_snps {r['n_snps']}\n# S (observed-scale genetic covariance)\n")
        pd.DataFrame(r["S"], index=names, columns=names).to_csv(fh, sep="\t", float_format="%.5f")
        fh.write("# rg\n"); pd.DataFrame(rg, index=names, columns=names).to_csv(fh, sep="\t", float_format="%.3f")
        fh.write("# LDSC intercepts\n"); pd.DataFrame(r["I"], index=names, columns=names).to_csv(fh, sep="\t", float_format="%.4f")
    print(out, "n_snps", r["n_snps"]); print(pd.DataFrame(rg, index=names, columns=names).round(3))
    print("h2_obs", dict(zip(names, np.round(np.diag(r["S"]), 4))), "int", np.round(np.diag(r["I"]), 3))
