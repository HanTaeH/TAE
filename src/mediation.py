"""Formal mediation analysis: disease stage (X) -> cellular composition (M) -> C-minus-A score (Y).

All variables (and covariates) are rank-transformed and standardised, matching the Spearman analyses.
  a   : M ~ X + C              b, c': Y ~ X + M + C              c : Y ~ X + C
  indirect = a * b, proportion mediated = a * b / c
Percentile bootstrap (2000 resamples) for the indirect effect, c' and the proportion.
Cross-sectional data: 'mediated' means statistically accounted for, not a causal estimate.

Data sets (per-sample tables written by earlier scripts):
  GSE125583 fusiform (AD only, all samples)  X Braak, M neuron score | SEA-AD NNLS Exc, C age + sex
  GSE106241 inferior temporal (all donors)   X Braak, M neuron score | NNLS Exc, C age + sex + RNA quality
  MSBB GSE84422 (subject means, cortex)      X CDR | Braak, M neuron score, C age + sex
  SEA-AD MTG snRNA (84 donors, sorted)       X CPS, M L2/3 IT share of neurons | Exc share, C age + sex

Usage: python mediation.py --res ../results --seaad-meta /mnt/project-files/d1-data/seaad/donors/donor_meta.csv --n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pilot_gse33000 import SEED


def rz(v):
    r = stats.rankdata(v)
    return (r - r.mean()) / r.std()


def coefs(X, M, Y, C):
    D1 = np.column_stack([np.ones(len(X)), X, C])
    a = np.linalg.lstsq(D1, M, rcond=None)[0][1]
    c = np.linalg.lstsq(D1, Y, rcond=None)[0][1]
    D2 = np.column_stack([np.ones(len(X)), X, M, C])
    bb = np.linalg.lstsq(D2, Y, rcond=None)[0]
    return a, bb[2], c, bb[1]


def mediate(df, x, m, y, covs, n, rng):
    d = df[[x, m, y] + covs].dropna()
    X, M, Y = rz(d[x].values), rz(d[m].values), rz(d[y].values)
    C = np.column_stack([rz(d[c].values) for c in covs]) if covs else np.empty((len(d), 0))
    a, b, c, cp = coefs(X, M, Y, C)
    boot = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d))
        aa, bb, cc, cpp = coefs(X[i], M[i], Y[i], C[i])
        boot.append((aa * bb, cpp, aa * bb / cc if abs(cc) > 1e-6 else np.nan, cc))
    boot = np.array(boot)

    def ci(k):
        v = boot[:, k][~np.isnan(boot[:, k])]
        return [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]
    p_ind = float(min(1, 2 * min((boot[:, 0] <= 0).mean(), (boot[:, 0] >= 0).mean()) + 1 / n))
    return {"n": int(len(d)), "x": x, "m": m, "covariates": covs,
            "a_x_to_m": round(a, 3), "b_m_to_y": round(b, 3), "total_c": round(c, 3), "total_ci": ci(3),
            "direct_c_prime": round(cp, 3), "direct_ci": ci(1),
            "indirect_ab": round(a * b, 3), "indirect_ci": ci(0), "indirect_p_boot": round(p_ind, 4),
            "prop_mediated": round(a * b / c, 3) if abs(c) > 1e-6 else None, "prop_ci": ci(2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--seaad-meta", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = a.res / "mediation"
    out.mkdir(exist_ok=True)
    runs = []

    g = pd.read_csv(a.res / "braak_gse125583/braak_gse125583_per_sample.csv", index_col=0)
    g["female"] = (g["sex"] == "female").astype(float)
    for s in ["AD_only", "all_samples"]:
        d = g[g["set"] == s]
        for m in ["neuron_score", "nnls_Exc"]:
            runs.append({"dataset": f"GSE125583 {s}", **mediate(d, "braak", m, "ca_score", ["age", "female"], a.n, rng)})

    b = pd.read_csv(a.res / "braak/braak_per_sample.csv", index_col=0)
    b["female"] = (b["sex"] == "female").astype(float)
    for m in ["neuron_score", "nnls_Exc"]:
        runs.append({"dataset": "GSE106241 all donors",
                     **mediate(b, "braak", m, "ca_score", ["age", "female", "rna_quality"], a.n, rng)})
    b3 = b[b["braak"] >= 3]
    runs.append({"dataset": "GSE106241 Braak>=3 (all-donor scores)",
                 **mediate(b3, "braak", "neuron_score", "ca_score", ["age", "female", "rna_quality"], a.n, rng)})

    s = pd.read_csv(a.res / "msbb/msbb_subject_scores.csv", index_col=0)
    for x in ["cdr", "braak"]:
        runs.append({"dataset": "MSBB subjects (cortex)", **mediate(s, x, "neuron", "ca", ["age", "female"], a.n, rng)})
    sad = s[s["group"] == "AD"]
    runs.append({"dataset": "MSBB AD subjects (cortex)", **mediate(sad, "cdr", "neuron", "ca", ["age", "female"], a.n, rng)})

    p = pd.read_csv(a.res / "seaad_pseudobulk/seaad_pseudobulk_per_donor.csv", index_col=0)
    meta = pd.read_csv(a.seaad_meta).set_index("Donor ID")
    p["age"] = pd.to_numeric(meta.loc[p.index, "Age at Death"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    p["female"] = (meta.loc[p.index, "Sex"] == "Female").astype(float)
    for m in ["L2/3 IT_of_neurons", "Exc_of_neurons", "neuron_nuclei_frac"]:
        runs.append({"dataset": "SEA-AD MTG snRNA (sorted pseudobulk)",
                     **mediate(p, "CPS_Global", m, "ca_sorted", ["age", "female"], a.n, rng)})
    runs.append({"dataset": "SEA-AD MTG snRNA (composition-only pseudobulk)",
                 **mediate(p, "CPS_Global", "L2/3 IT_of_neurons", "ca_comp_only", ["age", "female"], a.n, rng)})

    (out / "mediation_report.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    flat = pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items() if k != "covariates"} for r in runs])
    flat.to_csv(out / "mediation_summary.csv", index=False)
    print(flat[["dataset", "x", "m", "n", "total_c", "direct_c_prime", "indirect_ab", "indirect_ci", "prop_mediated",
                "prop_ci"]].to_string())


if __name__ == "__main__":
    main()
