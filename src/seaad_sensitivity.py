"""Second-round review: sensitivity of the SEA-AD single-type / subclass pseudobulk results.

Reads the per-donor table and report of seaad_celltype_revision.py and the SEA-AD donor metadata.
  1. BH-FDR across all pseudobulk x outcome (CPS, Braak, dementia) tests, nominal and random-gene p.
  2. Braak leverage: Braak association within Braak 4-6 donors only, Braak given CPS, CPS given Braak.
  3. Low-Braak (0-3) vs Braak 4-6 donors on age, RIN, pH, PMI and neuronal nuclei fraction.
  4. L2/3 IT and L4 IT shares: mutual partial correlations with the sorted C-minus-A score and an
     ilr balance log(L4/L2/3).
  5. RIN vs sorted C-minus-A adjusted for CPS and neuronal nuclei fraction.

Usage: python seaad_sensitivity.py --res ../results --seaad-meta /mnt/project-files/d1-data/seaad/donors/donor_meta.csv \
         --out ../results/seaad_sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from residual_go import bh


def pr(df: pd.DataFrame, x: str, y: str, covs: list[str] | None = None) -> dict:
    covs = covs or []
    sub = df[[x, y] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    R = sub.rank()
    rx, ry = R[x].values, R[y].values
    if covs:
        C = np.c_[np.ones(len(R)), R[covs].values]
        rx = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
        ry = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return {"rho": round(float(r), 3), "p": float(f"{p:.3g}"), "n": int(len(sub))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--seaad-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.res / "seaad_celltype_revision/per_donor.csv", index_col=0)
    rep_in = json.loads((a.res / "seaad_celltype_revision/seaad_celltype_revision_report.json").read_text())
    meta = pd.read_csv(a.seaad_meta).set_index("Donor ID").loc[d.index]
    d["age"] = pd.to_numeric(meta["Age at Death"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    for c in ["RIN", "Brain pH", "PMI"]:
        d[c] = pd.to_numeric(meta[c], errors="coerce")
    rep = {}

    # 1. FDR across all tests
    rows = []
    for pb, v in rep_in.items():
        if not isinstance(v, dict) or "ca_vs_CPS" not in v:
            continue
        for o in ["CPS", "Braak", "dementia"]:
            rn = v.get("random_null", {}).get(o, {})
            rows.append({"pseudobulk": pb, "outcome": o, "rho": v[f"ca_vs_{o}"]["rho"], "p_nominal": v[f"ca_vs_{o}"]["p"],
                         "rho_age_sex": v[f"ca_vs_{o}_partial_age_sex"]["rho"],
                         "p_age_sex": v[f"ca_vs_{o}_partial_age_sex"]["p"],
                         "random_median": rn.get("random_median"), "p_random": rn.get("p")})
    t = pd.DataFrame(rows)
    t["fdr_nominal"] = bh(t["p_nominal"].values)
    t["fdr_age_sex"] = bh(t["p_age_sex"].values)
    t["fdr_random"] = bh(t["p_random"].values)
    t.round(4).to_csv(a.out / "fdr_all_pseudobulk_tests.csv", index=False)
    rep["fdr"] = {"n_tests": int(len(t)),
                  "fdr_nominal_lt_0.05": t.loc[t.fdr_nominal < 0.05, ["pseudobulk", "outcome", "rho", "fdr_nominal"]].round(3).to_dict("records"),
                  "fdr_random_lt_0.05": t.loc[t.fdr_random < 0.05, ["pseudobulk", "outcome", "rho", "fdr_random"]].round(3).to_dict("records")}

    # 2. Braak leverage
    cols = [c for c in d.columns if c.startswith("ca_")]
    hi = d[d["Braak"] >= 4]
    lev = {}
    for c in cols:
        lev[c] = {"Braak_all": pr(d, c, "Braak"), "Braak_given_CPS": pr(d, c, "Braak", ["CPS"]),
                  "CPS_given_Braak": pr(d, c, "CPS", ["Braak"]), "Braak_in_Braak4to6": pr(hi, c, "Braak"),
                  "CPS_in_Braak4to6": pr(hi, c, "CPS")}
    rep["braak_leverage"] = lev
    rep["braak_distribution"] = {str(int(k)): int(v) for k, v in d["Braak"].value_counts().sort_index().items()}
    rep["CPS_vs_Braak"] = pr(d, "CPS", "Braak")

    # 3. low- vs high-Braak donors
    lo = d["Braak"] <= 3
    cmp = {}
    for c in ["age", "RIN", "Brain pH", "PMI", "neuron_nuclei_frac", "ca_sorted"]:
        x, y = d.loc[lo, c].dropna(), d.loc[~lo, c].dropna()
        cmp[c] = {"median_braak0_3": round(float(x.median()), 3), "median_braak4_6": round(float(y.median()), 3),
                  "n": [int(len(x)), int(len(y))], "mannwhitney_p": float(f"{stats.mannwhitneyu(x, y).pvalue:.3g}")}
    rep["low_vs_high_braak"] = cmp

    # 4. L2/3 vs L4 IT
    l23, l4 = "L2/3 IT_of_Exc", "L4 IT_of_Exc"
    d["ilr_L4_vs_L23"] = np.sqrt(0.5) * np.log(d[l4] / d[l23])
    rep["laminar"] = {"L23_vs_L4_share": pr(d, l23, l4),
                      "ca_vs_L23_given_L4": pr(d, "ca_sorted", l23, [l4]),
                      "ca_vs_L4_given_L23": pr(d, "ca_sorted", l4, [l23]),
                      "ca_vs_ilr_L4_vs_L23": pr(d, "ca_sorted", "ilr_L4_vs_L23"),
                      "CPS_vs_ilr_L4_vs_L23": pr(d, "CPS", "ilr_L4_vs_L23"),
                      "ca_vs_CPS_given_ilr": pr(d, "ca_sorted", "CPS", ["ilr_L4_vs_L23"]),
                      "comp_only_vs_observed": pr(d, "ca_Exc_subclass_comp_only", "ca_sorted")}

    # 5. RIN
    rep["rin"] = {"ca_vs_RIN": pr(d, "ca_sorted", "RIN"),
                  "ca_vs_RIN_given_CPS": pr(d, "ca_sorted", "RIN", ["CPS"]),
                  "ca_vs_RIN_given_CPS_neuronfrac": pr(d, "ca_sorted", "RIN", ["CPS", "neuron_nuclei_frac"]),
                  "RIN_vs_CPS": pr(d, "RIN", "CPS"), "RIN_vs_neuronfrac": pr(d, "RIN", "neuron_nuclei_frac")}
    (a.out / "seaad_sensitivity_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "braak_leverage"}, indent=1))
    for c, v in lev.items():
        print(c, {k: (x["rho"], x["p"]) for k, x in v.items()})


if __name__ == "__main__":
    main()
