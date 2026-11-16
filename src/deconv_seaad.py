"""Priority 4: reference-based cell-type deconvolution (SEA-AD MTG) and re-analysis.

Reference: seaad_mtg_reference.csv.gz from build_seaad_reference.py (mean CPM per cell type,
neurotypical SEA-AD MTG donors). Signature genes: for each type, up to N_SIG genes with CPM >= 10
in that type and >= 4-fold above every other type, excluding the Neff classifier genes so the
composition estimate does not reuse them; on Illumina arrays only genes in the upper half of bulk
expression are used.

Bulk scale
  Illumina cohorts (log2)        -> linear 2^x
  GSE33000 (log10 ratio to pool) -> 10^x; the pool is a mix of the same samples, so each gene
                                    is modelled as a ratio to the cohort-average mixture
                                    (iterative: fit proportions, update average, refit)
Estimators (proportions summing to 1)
  nnls  non-negative least squares on per-gene scaled signature genes
  svr   CIBERSORT-style nu-SVR (linear kernel, nu in 0.25/0.5/0.75, best RMSE)

Re-analysis per cohort (AD samples), with three covariate sets: markers5 (previous marker scores),
nnls, svr (CLR-transformed proportions)
  r2 of the C-minus-A and B axes on the covariates
  residual Neff signal vs variance-matched random genes and vs permuted centroids
  ARI of class labels before vs after adjustment
And: mean estimated proportion per Neff class (AD, baseline labels) and per diagnosis group.

Usage: python deconv_seaad.py --data /mnt/project-files/d1-data/raw \
         --ref /mnt/project-files/d1-data/seaad/seaad_mtg_reference.csv.gz --out ../results/deconv --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.metrics import adjusted_rand_score
from sklearn.svm import NuSVR

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from residual_structure import r2, residualise, signal
from validate_cohorts import cohorts

N_SIG = 50
CLS = np.array(list("ABC"))


def signature_genes(ref: pd.DataFrame, exclude: set[str], present: pd.Index) -> list[str]:
    ref = ref.loc[ref.index.intersection(present).difference(list(exclude))]
    out = []
    for t in ref.columns:
        others = ref.drop(columns=t).max(axis=1)
        fold = (ref[t] + 1) / (others + 1)
        ok = (ref[t] >= 10) & (fold >= 4)
        out += fold[ok].sort_values(ascending=False).index[:N_SIG].tolist()
    return list(dict.fromkeys(out))


def est_nnls(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """M genes x samples (linear), S genes x types, both unscaled (scaling each gene by its
    reference mean let a handful of low-expressed SST genes dominate on arrays)."""
    P = np.array([nnls(S, M[:, j])[0] for j in range(M.shape[1])])
    return P / P.sum(1, keepdims=True)


def est_svr(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    Sw, Mw = S, M
    Sz = (Sw - Sw.mean()) / Sw.std()
    P = []
    for j in range(M.shape[1]):
        y = (Mw[:, j] - Mw[:, j].mean()) / Mw[:, j].std()
        best, best_rmse = None, np.inf
        for nu in (0.25, 0.5, 0.75):
            m = NuSVR(kernel="linear", nu=nu, C=1.0).fit(Sz, y)
            coef = np.clip(m.coef_.ravel(), 0, None)
            if coef.sum() == 0:
                continue
            pred = Sz @ (coef / coef.sum())
            rmse = np.sqrt(np.mean((y - (pred - pred.mean()) / pred.std()) ** 2))
            if rmse < best_rmse:
                best, best_rmse = coef / coef.sum(), rmse
        P.append(best if best is not None else np.full(S.shape[1], 1 / S.shape[1]))
    return np.array(P)


def estimate(M: np.ndarray, S: np.ndarray, fn, ratio: bool) -> np.ndarray:
    if not ratio:
        return fn(M, S)
    p_bar = np.full(S.shape[1], 1 / S.shape[1])
    for _ in range(10):
        avg = S @ p_bar
        P = fn(M * avg[:, None], S)  # ratio x average mixture = expression up to a constant
        if np.abs(P.mean(0) - p_bar).max() < 1e-3:
            break
        p_bar = P.mean(0)
    return P


def clr(P: np.ndarray) -> np.ndarray:
    L = np.log(P + 0.01)
    return (L - L.mean(1, keepdims=True))[:, :-1]


def adj_report(z, cent, X, genes, cov, pool_by, neff_s, n, rng):
    r = corr_np(z, cent)
    lab = CLS[r.argmax(1)]
    ca, b = r[:, 2] - r[:, 0], r[:, 1] - (r[:, 0] + r[:, 2]) / 2
    R = residualise(z, cov)
    lab_adj = CLS[corr_np(z_np(R), cent).argmax(1)]
    raw, adj = signal(z, cent, cov)
    null = []
    for _ in range(n):
        pick = np.empty(len(genes), dtype=object)
        for s in np.unique(neff_s):
            idx = np.where(neff_s == s)[0]
            pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=False)
        null.append(signal(z_np(X.loc[list(pick)].values), cent, cov)[1])
    zr = z_np(R)
    perm = [corr_np(zr, cent[rng.permutation(len(cent))]).max(1).mean() for _ in range(n)]
    return {"r2_ca": round(r2(ca, cov), 3), "r2_b": round(r2(b, cov), 3),
            "ari_adj": round(float(adjusted_rand_score(lab, lab_adj)), 3),
            "sig_raw": round(raw, 3), "sig_adj": round(adj, 3),
            "sig_adj_rand": round(float(np.median(null)), 3),
            "sig_adj_rand_p": round(float((np.sum(np.array(null) >= adj) + 1) / (n + 1)), 4),
            "sig_adj_perm": round(float(np.median(perm)), 3),
            "sig_adj_perm_p": round(float((np.sum(np.array(perm) >= adj) + 1) / (n + 1)), 4),
            "adj_frac": {k: round(float((lab_adj == k).mean()), 3) for k in "ABC"}}, lab


def run_cohort(name, expr, meta, cls, ref, n, rng):
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    ratio = name.startswith("GSE33000")
    lin = 10 ** expr if ratio else 2 ** expr
    # on Illumina arrays keep signature genes in the upper half of bulk expression (lower half sits
    # near background); the log-ratio scale of GSE33000 carries no absolute level, so no filter there
    rank = expr.mean(axis=1).rank(pct=True)
    present = expr.index if ratio else rank.index[rank >= 0.5]
    sig = signature_genes(ref, set(cls.index), present)
    S = ref.loc[sig].values
    M = lin.loc[sig, meta.index].fillna(lin.loc[sig, meta.index].mean(axis=1).mean()).values
    props = {}
    for meth, fn in [("nnls", est_nnls), ("svr", est_svr)]:
        props[meth] = pd.DataFrame(estimate(M, S, fn, ratio), index=meta.index, columns=ref.columns)

    ad = meta.index[meta["group"] == "AD"]
    marker_genes = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}

    def score(lst):
        lst = [g for g in lst if g in expr.index and g not in genes]
        return z_np(expr.loc[lst, ad].values).mean(0)
    neuron = score(NEURON_MARKERS)
    markers5 = np.column_stack([neuron] + [score(v) for v in CELL_MARKERS.values()])

    rep: dict = {"cohort": name, "n_sig_genes": len(sig), "n_AD": int(len(ad))}
    for meth, P in props.items():
        rep[f"{meth}_mean_by_group"] = P.groupby(meta["group"]).mean().round(3).to_dict(orient="index")
        rep[f"{meth}_exc_vs_neuron_score_rho"] = round(float(stats.spearmanr(P.loc[ad, "Exc"], neuron)[0]), 3)

    X = expr[ad]
    z = z_np(X.loc[genes].values)
    mean_e, sd_e = X.mean(axis=1), X.std(axis=1)
    dec = pd.qcut(mean_e.rank(method="first"), 10, labels=False)
    terc = sd_e.groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    stratum = dec * 3 + terc
    pool = X.index.difference(genes).difference(list(marker_genes)).difference(sig)
    pool_by = {s: pool[stratum.loc[pool].values == s] for s in range(30)}
    neff_s = stratum.loc[genes].values

    covsets = {"markers5": markers5, "nnls": clr(props["nnls"].loc[ad].values),
               "svr": clr(props["svr"].loc[ad].values)}
    for k, cov in covsets.items():
        rep[f"adj_{k}"], lab = adj_report(z, cent, X, genes, cov, pool_by, neff_s, n, rng)

    # composition by Neff class (AD) with Kruskal-Wallis p
    for meth, P in props.items():
        Pa = P.loc[ad]
        rep[f"{meth}_by_class"] = Pa.groupby(lab).mean().round(3).to_dict(orient="index")
        rep[f"{meth}_class_kruskal_p"] = {
            t: float(stats.kruskal(*[Pa[t].values[lab == c] for c in "ABC" if (lab == c).sum() > 2]).pvalue)
            for t in P.columns}
    out = pd.concat({m: p for m, p in props.items()}, axis=1)
    out.columns = [f"{m}_{t}" for m, t in out.columns]
    out = out.join(meta[["group"]])
    out.loc[ad, "neff_class"] = lab
    out["cohort"] = name
    return rep, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    reps, tabs = [], []
    for name, e, m in cohorts(a.data):
        rep, tab = run_cohort(name, e, m, cls, ref, a.n, rng)
        reps.append(rep)
        tabs.append(tab)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    (a.out / "deconv_report.json").write_text(json.dumps(reps, indent=2, ensure_ascii=False))
    pd.concat(tabs).to_csv(a.out / "deconv_proportions.csv")
    rows = []
    for r in reps:
        for k in ["markers5", "nnls", "svr"]:
            d = r[f"adj_{k}"]
            rows.append({"cohort": r["cohort"], "covariates": k,
                         **{x: d[x] for x in ["r2_ca", "r2_b", "ari_adj", "sig_raw", "sig_adj", "sig_adj_rand",
                                              "sig_adj_rand_p", "sig_adj_perm", "sig_adj_perm_p"]}})
    flat = pd.DataFrame(rows)
    flat.to_csv(a.out / "deconv_adjustment_summary.csv", index=False)
    print(flat.to_string())


if __name__ == "__main__":
    main()
