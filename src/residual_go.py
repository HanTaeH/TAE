"""Reviewer revision: standard gene-set collections for the residual C-minus-A axis.

Replaces the curated, name-based gene lists of residual_identity.py with MSigDB GO Biological
Process (c5.go.bp) and Hallmark (h.all) collections.

Per cortical cohort-region set (as in residual_identity.py): genes are z-scored in AD samples,
residualised on the five marker scores, and correlated with the residual C-minus-A score. Each gene
set (15-500 measured genes; Neff classifier and marker genes removed) is scored by its mean gene
correlation and tested competitively against all measured genes with a CAMERA-style variance
inflation for inter-gene correlation (VIF = 1 + (k-1) * mean pairwise correlation within the set).
Benjamini-Hochberg FDR within collection and cohort. Consistency: sets significant (FDR < 0.05) in
the same direction in every array cohort and in the RNA-seq cohort, or with opposite signs between
platforms.

Usage: python residual_go.py --data /mnt/project-files/d1-data/raw --gmt-dir /mnt/project-files/d1-data/msigdb \
         --out ../results/residual_go
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from null_models import z_np
from residual_identity import MARKERS, cell_scores, residual_ca
from residual_structure import residualise
from validate_cohorts import cohorts

RNASEQ = {"GSE125583_FG"}


def read_gmt(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) > 2:
            out[f[0]] = f[2:]
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def test_sets(gc: pd.Series, Zs: np.ndarray, idx: dict, sets: dict) -> pd.DataFrame:
    mu, sd, N = gc.mean(), gc.std(), len(gc)
    n_s = Zs.shape[1]
    rows = []
    for name, genes in sets.items():
        ii = [idx[g] for g in genes if g in idx]
        k = len(ii)
        if k < 15 or k > 500:
            continue
        sub = Zs[ii]
        s = sub.sum(0)
        rbar = ((s @ s) / n_s - k) / (k * (k - 1))
        vif = max(1.0, 1 + (k - 1) * rbar)
        m = gc.values[ii].mean()
        se = sd / np.sqrt(k) * np.sqrt((N - k) / (N - 1)) * np.sqrt(vif)
        z = (m - mu) / se
        rows.append({"set": name, "n_genes": k, "mean_r": m, "z": z, "p": 2 * stats.norm.sf(abs(z)), "vif": vif})
    df = pd.DataFrame(rows)
    df["fdr"] = bh(df["p"].values)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gmt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    coll = {"GOBP": read_gmt(next(a.gmt_dir.glob("c5.go.bp*.gmt"))),
            "HALLMARK": read_gmt(next(a.gmt_dir.glob("h.all*.gmt")))}
    res = {c: [] for c in coll}
    for name, expr, meta in cohorts(a.data):
        if "Cerebellum" in name:
            continue
        expr = expr.loc[expr.index.notna() & ~expr.index.duplicated()].dropna()
        ad = meta.index[meta["group"] == "AD"]
        genes = cls.index.intersection(expr.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        cells = cell_scores(expr, ad, set(genes))
        rca = residual_ca(expr, ad, genes, cent, cells)
        keep = expr.index.difference(genes).difference(list(MARKERS))
        Z = residualise(z_np(expr.loc[keep, ad].values), cells)
        rz = (rca - rca.mean()) / rca.std()
        Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
        gc = pd.Series(Zs @ rz / len(rz), index=keep)
        idx = {g: i for i, g in enumerate(keep)}
        for c, sets in coll.items():
            df = test_sets(gc, Zs, idx, sets)
            df["cohort"] = name
            res[c].append(df)
        print(name, "done", flush=True)

    rep = {}
    for c in coll:
        df = pd.concat(res[c])
        df.round(5).to_csv(a.out / f"{c}_all_cohorts.csv", index=False)
        wide_z = df.pivot(index="set", columns="cohort", values="z")
        wide_q = df.pivot(index="set", columns="cohort", values="fdr")
        arr = [x for x in wide_z.columns if x not in RNASEQ]
        sig = (wide_q < 0.05)
        sgn = np.sign(wide_z)
        all_up = (sig & (sgn > 0)).all(axis=1)
        all_dn = (sig & (sgn < 0)).all(axis=1)
        arr_up = (sig[arr] & (sgn[arr] > 0)).all(axis=1)
        arr_dn = (sig[arr] & (sgn[arr] < 0)).all(axis=1)
        rs = [x for x in wide_z.columns if x in RNASEQ]
        flip = ((arr_up & (sig[rs] & (sgn[rs] < 0)).all(axis=1)) | (arr_dn & (sig[rs] & (sgn[rs] > 0)).all(axis=1)))
        summ = pd.DataFrame({"mean_z": wide_z.mean(axis=1), "n_cohorts_fdr05": sig.sum(axis=1),
                             "all_up": all_up, "all_down": all_dn, "arrays_up": arr_up, "arrays_down": arr_dn,
                             "platform_flip": flip}).join(wide_z.add_prefix("z_"))
        summ.sort_values("mean_z").round(3).to_csv(a.out / f"{c}_consistency.csv")
        rep[c] = {"n_sets_tested": int(len(wide_z)), "n_cohorts": int(wide_z.shape[1]),
                  "consistent_up_all": summ.index[all_up].tolist()[:40],
                  "consistent_down_all": summ.index[all_dn].tolist()[:40],
                  "n_consistent_up_all": int(all_up.sum()), "n_consistent_down_all": int(all_dn.sum()),
                  "n_arrays_only_up": int((arr_up & ~all_up).sum()), "n_arrays_only_down": int((arr_dn & ~all_dn).sum()),
                  "platform_flip": summ.index[flip].tolist()[:40], "n_platform_flip": int(flip.sum())}
        if c == "HALLMARK":
            rep[c]["z_table"] = wide_z.round(2).to_dict(orient="index")
    (a.out / "residual_go_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({c: {k: v for k, v in r.items() if k != "z_table"} for c, r in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
