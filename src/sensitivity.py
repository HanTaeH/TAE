"""Priority 3: sensitivity analysis of the Neff class projection.

Baseline = AD-only gene-wise z-scores on the measured Neff classifier genes, nearest class
centroid by Pearson r (as in validate_cohorts.py). Each variant changes one analysis choice and
is compared with the baseline on
  ari_vs_base   agreement of AD class labels with the baseline (ARI)
  frac_A/B/C    class proportions among AD samples
  neuron_rho    Spearman between C-minus-A score and neuron marker score (AD samples)
  stage_rho     Spearman between diagnosis order (Control < AsymAD < AD) and C-minus-A score
                (variants that change the reference also change the all-sample projection)

Variants
  age_sex       age and sex regressed out of every gene (within AD; all-sample: within all)
  spearman      Spearman instead of Pearson correlation to centroids
  ref_control   z-scores relative to control samples (mean/sd of controls) instead of AD
  subtype5      five-subtype centroids (A, B1, B2, C1, C2) collapsed to classes A, B, C
  signature     genome-wide Neff log-fold-change signatures (data file S1), top 1000 genes by
                |logFC| excluding the 271 classifier genes, collapsed to A, B, C
  gene_sub80    80% random subsets of classifier genes (200 draws; median reported)
  age_matched   stage trend restricted to controls >= 70 years (only where groups differ in age)

Usage: python sensitivity.py --data /mnt/project-files/d1-data/raw --out ../results/sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from validate_cohorts import GROUP_ORDER, cohorts

CLS = np.array(list("ABC"))


def spearman_corr(z: np.ndarray, cent: np.ndarray) -> np.ndarray:
    return corr_np(stats.rankdata(z, axis=0), stats.rankdata(cent, axis=0))


def regress_out(x: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """x genes x samples; covars samples x k (intercept added)."""
    D = np.column_stack([np.ones(covars.shape[0]), covars])
    beta, *_ = np.linalg.lstsq(D, x.T, rcond=None)
    return (x.T - D @ beta).T


def covariates(meta: pd.DataFrame) -> np.ndarray:
    age = meta["age"].astype(float)
    age = age.fillna(age.mean())
    sex = (meta["sex"].astype(str).str.lower() == "female").astype(float)
    cols = [age.values]
    if sex.nunique() > 1:
        cols.append(sex.values)
    return np.column_stack(cols)


def ref_z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = np.nanmean(ref, axis=1, keepdims=True)
    sd = np.nanstd(ref, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nan_to_num((x - mu) / sd)


def to_class(r: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse correlation to (sub)centroids into class labels and a C-minus-A score."""
    df = pd.DataFrame(r, columns=cols)
    lab = df.idxmax(axis=1).str[0].values
    best = {k: df[[c for c in cols if c[0] == k]].max(axis=1).values for k in "ABC"}
    return lab, best["C"] - best["A"]


def summarise(lab, ca, neuron, lab_base=None):
    out = {f"frac_{k}": round(float((lab == k).mean()), 3) for k in "ABC"}
    out["neuron_rho"] = round(float(stats.spearmanr(neuron, ca)[0]), 3)
    if lab_base is not None:
        out["ari_vs_base"] = round(float(adjusted_rand_score(lab_base, lab)), 3)
    return out


def stage(ca_all, meta):
    rho, p = stats.spearmanr(meta["group"].map(GROUP_ORDER).values, ca_all)
    return {"stage_rho": round(float(rho), 3), "stage_p": float(p)}


def run_cohort(name, expr, meta, cls, sub, logfc, rng):
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    cent5 = sub.loc[genes].sub(sub.loc[genes].mean(axis=1), axis=0).values
    ad = meta.index[meta["group"] == "AD"]
    ctrl = meta.index[meta["group"] == "Control"]

    def score(lst):
        lst = [g for g in lst if g in expr.index and g not in genes]
        return z_np(expr.loc[lst, ad].values).mean(0)
    neuron = score(NEURON_MARKERS)

    Xad = expr.loc[genes, ad].values
    Xall = expr.loc[genes, meta.index].values
    rows = []

    # baseline
    r = corr_np(z_np(Xad), cent)
    lab0, ca0 = CLS[r.argmax(1)], r[:, 2] - r[:, 0]
    ra = corr_np(z_np(Xall), cent)
    rows.append({"variant": "baseline", **summarise(lab0, ca0, neuron), "ari_vs_base": 1.0,
                 **stage(ra[:, 2] - ra[:, 0], meta)})

    # age + sex adjustment
    xa = regress_out(Xad, covariates(meta.loc[ad]))
    r = corr_np(z_np(xa), cent)
    xall = regress_out(Xall, covariates(meta))
    ra = corr_np(z_np(xall), cent)
    rows.append({"variant": "age_sex", **summarise(CLS[r.argmax(1)], r[:, 2] - r[:, 0], neuron, lab0),
                 **stage(ra[:, 2] - ra[:, 0], meta)})

    # spearman
    r = spearman_corr(z_np(Xad), cent)
    ra = spearman_corr(z_np(Xall), cent)
    rows.append({"variant": "spearman", **summarise(CLS[r.argmax(1)], r[:, 2] - r[:, 0], neuron, lab0),
                 **stage(ra[:, 2] - ra[:, 0], meta)})

    # control reference
    Xc = expr.loc[genes, ctrl].values
    r = corr_np(ref_z(Xad, Xc), cent)
    ra = corr_np(ref_z(Xall, Xc), cent)
    rows.append({"variant": "ref_control", **summarise(CLS[r.argmax(1)], r[:, 2] - r[:, 0], neuron, lab0),
                 **stage(ra[:, 2] - ra[:, 0], meta)})

    # five subtype centroids
    cols5 = list(sub.columns)
    lab, ca = to_class(corr_np(z_np(Xad), cent5), cols5)
    _, ca_all = to_class(corr_np(z_np(Xall), cent5), cols5)
    rows.append({"variant": "subtype5", **summarise(lab, ca, neuron, lab0), **stage(ca_all, meta)})

    # genome-wide signatures (independent of the 271 classifier genes)
    sig = logfc.loc[logfc.index.intersection(expr.index).difference(cls.index)]
    sig = sig.loc[sig.abs().max(axis=1).sort_values(ascending=False).index[:1000]]
    sc = sig.sub(sig.mean(axis=1), axis=0).values
    lab, ca = to_class(corr_np(z_np(expr.loc[sig.index, ad].values), sc), cols5)
    _, ca_all = to_class(corr_np(z_np(expr.loc[sig.index, meta.index].values), sc), cols5)
    rows.append({"variant": "signature", "n_genes": len(sig), **summarise(lab, ca, neuron, lab0),
                 **stage(ca_all, meta)})

    # 80% gene subsets
    sub_rows = []
    for _ in range(200):
        k = rng.choice(len(genes), size=int(0.8 * len(genes)), replace=False)
        r = corr_np(z_np(Xad[k]), cent[k])
        ra = corr_np(z_np(Xall[k]), cent[k])
        sub_rows.append({**summarise(CLS[r.argmax(1)], r[:, 2] - r[:, 0], neuron, lab0),
                         **stage(ra[:, 2] - ra[:, 0], meta)})
    s = pd.DataFrame(sub_rows)
    rows.append({"variant": "gene_sub80", **s.median(numeric_only=True).round(3).to_dict(),
                 "ari_q05": round(float(s["ari_vs_base"].quantile(.05)), 3),
                 "neuron_rho_q95": round(float(s["neuron_rho"].quantile(.95)), 3)})

    # age-matched stage trend
    if meta.groupby("group")["age"].mean().max() - meta.groupby("group")["age"].mean().min() > 5:
        keep = meta.index[(meta["group"] != "Control") | (meta["age"] >= 70)]
        m2 = meta.loc[keep]
        ra = corr_np(z_np(expr.loc[genes, keep].values), cent)
        rows.append({"variant": "age_matched", "n_control": int((m2["group"] == "Control").sum()),
                     "age_by_group": m2.groupby("group")["age"].mean().round(1).to_dict(),
                     **stage(ra[:, 2] - ra[:, 0], m2)})

    for row in rows:
        row["cohort"] = name
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sub, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    logfc = neff_subtypes.load_logfc(a.data / "neff2021")

    rows = []
    for name, e, m in cohorts(a.data):
        rows += run_cohort(name, e, m, cls, sub, logfc, rng)
        print(name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "sensitivity_long.csv", index=False)
    (a.out / "sensitivity_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    for col in ["ari_vs_base", "neuron_rho", "stage_rho", "frac_C"]:
        print(f"\n== {col}")
        print(df.pivot(index="variant", columns="cohort", values=col).round(2).to_string())


if __name__ == "__main__":
    main()
