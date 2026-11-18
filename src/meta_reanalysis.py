"""Reviewer revision: re-do the meta-analysis of the two primary endpoints.

Changes relative to forest_plot.py (reviewer 2, M5):
  - per-cohort standard error of Fisher-z from 2,000 bootstrap resamples (in addition to the
    Fieller sqrt(1.06/(n-3)) and Bonett-Wright sqrt((1 + r^2/2)/(n-3)) approximations);
  - random-effects pooling with REML tau^2 and the Hartung-Knapp-Sidik-Jonkman (HKSJ) CI (t, k-1 df);
  - 95% prediction interval (t, k-2 df);
  - endpoint 2 restricted to comparable endpoints (diagnosis group, 4 cohorts) and an age/sex-adjusted
    version (partial Spearman), with the original mixed 5-cohort pool kept for comparison.

Usage: python meta_reanalysis.py --data /mnt/project-files/d1-data/raw --res <results dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import neff_subtypes
from pilot_gse33000 import NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts

INDEPENDENT = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE125583_FG"]


def ca_score(z: pd.DataFrame, cent: pd.DataFrame) -> pd.Series:
    zc = (z - z.mean()) / z.std(ddof=0)
    cc = (cent - cent.mean()) / cent.std(ddof=0)
    r = zc.T.values @ cc.values / z.shape[0]
    return pd.Series(r[:, 2] - r[:, 0], index=z.columns)


def resid_rank(v: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    r = stats.rankdata(v)
    if cov is None:
        return r
    D = np.column_stack([np.ones(len(r)), cov])
    return r - D @ np.linalg.lstsq(D, r, rcond=None)[0]


def rho(x, y, cov=None) -> float:
    return float(np.corrcoef(resid_rank(x, cov), resid_rank(y, cov))[0, 1])


def boot_se(x, y, cov, rng, n_boot=2000) -> float:
    n = len(x)
    zs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        c = None if cov is None else cov[i]
        with np.errstate(all="ignore"):
            r = rho(x[i], y[i], c)
        if np.isfinite(r):
            zs.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))
    return float(np.std(zs, ddof=1))


def reml_tau2(z, v) -> float:
    def nll(t2):
        w = 1 / (v + t2)
        mu = np.sum(w * z) / w.sum()
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(w.sum()) + np.sum(w * (z - mu) ** 2))
    res = optimize.minimize_scalar(nll, bounds=(0, 10), method="bounded")
    return float(max(res.x, 0))


def pool(z, se) -> dict:
    z, v = np.asarray(z), np.asarray(se) ** 2
    k = len(z)
    w = 1 / v
    q = float(np.sum(w * (z - np.sum(w * z) / w.sum()) ** 2))
    t2 = reml_tau2(z, v)
    ws = 1 / (v + t2)
    mu = float(np.sum(ws * z) / ws.sum())
    # HKSJ variance
    var_hk = float(np.sum(ws * (z - mu) ** 2) / ((k - 1) * ws.sum()))
    se_hk = np.sqrt(max(var_hk, 1 / ws.sum()))  # truncated HKSJ (never narrower than standard RE)
    tq = stats.t.ppf(0.975, k - 1)
    p = float(2 * stats.t.sf(abs(mu / se_hk), k - 1))
    out = {"k": k, "rho": round(float(np.tanh(mu)), 3),
           "ci95_hksj": [round(float(np.tanh(mu - tq * se_hk)), 3), round(float(np.tanh(mu + tq * se_hk)), 3)],
           "p_hksj": p, "tau2_reml": round(t2, 4),
           "I2": round(max(0.0, (q - (k - 1)) / q), 3) if q > 0 else 0.0}
    if k >= 3:
        tp = stats.t.ppf(0.975, k - 2)
        s = np.sqrt(t2 + se_hk ** 2)
        out["pi95"] = [round(float(np.tanh(mu - tp * s)), 3), round(float(np.tanh(mu + tp * s)), 3)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True, help="results dir with braak/braak_per_sample.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        ad = m.index[m["group"] == "AD"]
        ca_ad = ca_score(zscore_rows(e.loc[genes, ad]), cent)
        mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
        neuron = zscore_rows(e.loc[mk, ad]).mean()
        ca_all = ca_score(zscore_rows(e.loc[genes, m.index]), cent)
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov = np.column_stack([age, sex])
        for ep, x, y, c in [("P1_neuron", ca_ad.values, neuron.values, None),
                            ("P2_stage", ca_all.values, stage, None),
                            ("P2_stage_agesex", ca_all.values, stage, cov)]:
            n = len(x)
            r = rho(x, y, c)
            rows.append({"endpoint": ep, "cohort": name, "n": n, "rho": r, "z": np.arctanh(r),
                         "se_fieller": np.sqrt(1.06 / (n - 3)),
                         "se_bonett": np.sqrt((1 + r ** 2 / 2) / (n - 3)),
                         "se_old_formula": 1.06 / np.sqrt(n - 3),
                         "se_boot": boot_se(x, y, c, rng, a.n_boot)})
        print(name, "done", flush=True)

    b = pd.read_csv(a.res / "braak/braak_per_sample.csv", index_col=0)
    cov = np.column_stack([b["age"].values, (b["sex"].str[0] == "m").astype(float).values])
    for ep, c in [("P2_stage_braak106241", None), ("P2_stage_agesex_braak106241", cov)]:
        x, y = b["ca_score"].values, b["braak"].values.astype(float)
        n = len(x)
        r = rho(x, y, c)
        rows.append({"endpoint": ep, "cohort": "GSE106241_ITC", "n": n, "rho": r, "z": np.arctanh(r),
                     "se_fieller": np.sqrt(1.06 / (n - 3)), "se_bonett": np.sqrt((1 + r ** 2 / 2) / (n - 3)),
                     "se_old_formula": 1.06 / np.sqrt(n - 3), "se_boot": boot_se(x, y, c, rng, a.n_boot)})

    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out / "per_cohort.csv", index=False)

    def sel(ep, extra=None):
        d = df[df.endpoint == ep]
        if extra is not None:
            d = pd.concat([d, df[df.endpoint == extra]])
        return d

    report = {}
    specs = {"P1_neuron (4 cohorts)": sel("P1_neuron"),
             "P2_stage diagnosis only (4 cohorts)": sel("P2_stage"),
             "P2_stage age/sex adjusted (4 cohorts)": sel("P2_stage_agesex"),
             "P2_stage original mix (4 diagnosis + GSE106241 Braak)": sel("P2_stage", "P2_stage_braak106241"),
             "P2_stage original mix, age/sex adjusted": sel("P2_stage_agesex", "P2_stage_agesex_braak106241")}
    for k, d in specs.items():
        report[k] = {se: pool(d["z"].values, d[se].values)
                     for se in ["se_boot", "se_bonett", "se_fieller", "se_old_formula"]}
        report[k]["per_cohort"] = d[["cohort", "n", "rho", "se_boot", "se_bonett"]].round(3).to_dict(orient="records")
    (a.out / "meta_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k, v in report.items():
        print(k, json.dumps(v["se_boot"]), flush=True)


if __name__ == "__main__":
    main()
