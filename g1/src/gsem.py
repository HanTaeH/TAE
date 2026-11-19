"""Genomic SEM (confirmatory factor models) on an LDSC S/V pair, DWLS estimation,
sandwich standard errors and a residual-based model chi-square (Browne 1984).
Factor variances are fixed to 1; loadings free; residual variances free (bounded >= 0).

Usage: python gsem.py <out.tsv> <S_cohort1.npz> [<S_cohort2.npz> ...]
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2d, norm

# model = list of factors -> indicator names; factor correlations free unless fixed
MODELS = {
    "1F_common": {"F": ["STEN", "SPDS", "SPLT", "SCIA", "DORS"]},
    "2F_struct_pain": {"Fstruct": ["STEN", "SPDS", "SPLT"], "Fpain": ["SCIA", "DORS"]},
    "2F_SPDS_cross": {"Fstruct": ["STEN", "SPDS", "SPLT"], "Fpain": ["SCIA", "DORS", "SPDS"]},
}


def vech_idx(k):
    return [(i, j) for i in range(k) for j in range(i, k)]


class CFA:
    def __init__(self, names, spec):
        self.names, self.k = list(names), len(names)
        self.facs = list(spec)
        self.load = [(self.names.index(v), f) for f, (fac, vs) in enumerate(spec.items()) for v in vs]
        self.ncor = len(self.facs) * (len(self.facs) - 1) // 2
        self.np = len(self.load) + self.ncor + self.k

    def sigma(self, th):
        L = np.zeros((self.k, len(self.facs)))
        for p, (i, f) in enumerate(self.load):
            L[i, f] = th[p]
        Phi = np.eye(len(self.facs))
        c = len(self.load)
        for a in range(len(self.facs)):
            for b in range(a + 1, len(self.facs)):
                Phi[a, b] = Phi[b, a] = th[c]; c += 1
        Th = np.diag(th[c:c + self.k])
        return L @ Phi @ L.T + Th

    def vs(self, th):
        s = self.sigma(th)
        return np.array([s[i, j] for i, j in vech_idx(self.k)])


def jac(f, th, eps=1e-6):
    f0 = f(th)
    J = np.zeros((len(f0), len(th)))
    for p in range(len(th)):
        t = th.copy(); t[p] += eps
        J[:, p] = (f(t) - f0) / eps
    return J


def fit(S, V, names, spec):
    m = CFA(names, spec)
    s = np.array([S[i, j] for i, j in vech_idx(m.k)])
    W = np.diag(1 / np.diag(V))
    d = np.sqrt(np.clip(np.diag(S), 1e-6, None))
    th0 = np.concatenate([[d[i] * 0.9 for i, _ in m.load], [0.8] * m.ncor, np.diag(S) * 0.2])
    bounds = [(None, None)] * len(m.load) + [(-1, 1)] * m.ncor + [(0, None)] * m.k
    obj = lambda th: (s - m.vs(th)) @ W @ (s - m.vs(th))
    best = None
    for start in range(5):
        t0 = th0 * (1 + 0.1 * start * np.random.default_rng(start).standard_normal(len(th0))) if start else th0
        r = minimize(obj, t0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=5000, ftol=1e-14, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    D = jac(m.vs, th)
    B = np.linalg.pinv(D.T @ W @ D)
    cov = B @ D.T @ W @ V @ W @ D @ B
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # residual chi-square: e' (P V P')^+ e with P = I - D (D'WD)^-1 D'W
    e = s - m.vs(th)
    P = np.eye(len(s)) - D @ B @ D.T @ W
    G = P @ V @ P.T
    df = len(s) - np.linalg.matrix_rank(D)
    w_, U = np.linalg.eigh(G)
    top = np.argsort(w_)[::-1][:df]
    Q = float(sum((U[:, t] @ e) ** 2 / w_[t] for t in top))
    srmr_d = np.sqrt(np.outer(np.diag(S), np.diag(S)))
    Sr = m.sigma(th)
    srmr = np.sqrt(np.mean([((S[i, j] - Sr[i, j]) / np.sqrt(srmr_d[i, j])) ** 2 for i, j in vech_idx(m.k)]))
    # standardized loadings
    sd = np.sqrt(np.diag(Sr))
    rows = []
    for p, (i, f) in enumerate(m.load):
        rows.append(dict(param=f"{m.facs[f]}=~{names[i]}", est=th[p] / sd[i], se=se[p] / sd[i]))
    c = len(m.load)
    for a in range(len(m.facs)):
        for b in range(a + 1, len(m.facs)):
            z1 = (1 - th[c]) / se[c] if se[c] > 0 else np.nan
            rows.append(dict(param=f"{m.facs[a]}~~{m.facs[b]}", est=th[c], se=se[c], z_vs_1=z1, p_vs_1=norm.sf(z1)))
            c += 1
    for i in range(m.k):
        rows.append(dict(param=f"resid_{names[i]}", est=th[c + i] / Sr[i, i], se=se[c + i] / Sr[i, i]))
    return dict(chisq=Q, df=df, p=chi2d.sf(Q, df) if df > 0 else np.nan, srmr=srmr, obj=best.fun, conv=best.success), pd.DataFrame(rows)


if __name__ == "__main__":
    out = sys.argv[1]
    fits, pars = [], []
    for f in sys.argv[2:]:
        r = np.load(f, allow_pickle=True)
        coh = f.split("S_")[-1].replace(".npz", "")
        for mn, spec in MODELS.items():
            st, pr = fit(r["S"], r["V"], list(r["names"]), spec)
            fits.append(dict(cohort=coh, model=mn, **st))
            pars.append(pr.assign(cohort=coh, model=mn))
    fits = pd.DataFrame(fits)
    # AIC-like comparison (chi2 + 2 * n_params ~ chi2 - 2 df up to a constant)
    fits["aic_rel"] = fits.chisq - 2 * fits.df
    fits.to_csv(out, sep="\t", index=False)
    pars = pd.concat(pars)
    pars.to_csv(out.replace(".tsv", "_params.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(fits.round(4).to_string(index=False))
    print(pars[pars.param.str.contains("~~")].round(3).to_string(index=False))
