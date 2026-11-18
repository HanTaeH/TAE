"""Reviewer revision (R1 major 3, R2 M6): is the mediation result informative or forced by collinearity?

For every model of mediation.py, fits the reverse model with mediator and outcome swapped
(stage -> C-minus-A -> neuron score). If C-minus-A and the neuron score are near-duplicate
measurements of one construct, the reverse model will also look "fully mediated", so the forward
result cannot be read as a pathway. Also reports the mediator-outcome correlation and the
indirect effect (not only the proportion) for every model, including those with CIs spanning zero.

Usage: python mediation_revision.py --res <results> --seaad-meta <donor_meta.csv> --out <dir> --n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mediation import mediate
from pilot_gse33000 import SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--seaad-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sets = []
    gp = a.res / "gse125583/braak_gse125583_per_sample.csv"
    if not gp.exists():
        gp = a.res / "braak_gse125583/braak_gse125583_per_sample.csv"
    g = pd.read_csv(gp, index_col=0)
    g["female"] = (g["sex"] == "female").astype(float)
    for s in ["AD_only", "all_samples"]:
        for m in ["neuron_score", "nnls_Exc"]:
            sets.append((f"GSE125583 {s}", g[g["set"] == s], "braak", m, "ca_score", ["age", "female"]))
    b = pd.read_csv(a.res / "braak/braak_per_sample.csv", index_col=0)
    b["female"] = (b["sex"] == "female").astype(float)
    for m in ["neuron_score", "nnls_Exc"]:
        sets.append(("GSE106241 all donors", b, "braak", m, "ca_score", ["age", "female", "rna_quality"]))
    s = pd.read_csv(a.res / "msbb/msbb_subject_scores.csv", index_col=0)
    for x in ["cdr", "braak"]:
        sets.append(("MSBB subjects (cortex)", s, x, "neuron", "ca", ["age", "female"]))
    sets.append(("MSBB AD subjects (cortex)", s[s["group"] == "AD"], "cdr", "neuron", "ca", ["age", "female"]))
    p = pd.read_csv(a.res / "seaad_pseudobulk/seaad_pseudobulk_per_donor.csv", index_col=0)
    meta = pd.read_csv(a.seaad_meta).set_index("Donor ID")
    p["age"] = pd.to_numeric(meta.loc[p.index, "Age at Death"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    p["female"] = (meta.loc[p.index, "Sex"] == "Female").astype(float)
    for m in ["L2/3 IT_of_neurons", "neuron_nuclei_frac"]:
        sets.append(("SEA-AD MTG (sorted pseudobulk)", p, "CPS_Global", m, "ca_sorted", ["age", "female"]))

    rows = []
    for name, d, x, m, y, cov in sets:
        fwd = mediate(d, x, m, y, cov, a.n, rng)
        rev = mediate(d, x, y, m, cov, a.n, rng)
        dd = d[[m, y]].dropna()
        rows.append({"dataset": name, "x": x, "mediator": m, "outcome": y, "n": fwd["n"],
                     "rho_mediator_outcome": round(float(stats.spearmanr(dd[m], dd[y])[0]), 3),
                     "fwd_total": fwd["total_c"], "fwd_indirect": fwd["indirect_ab"], "fwd_indirect_ci": fwd["indirect_ci"],
                     "fwd_direct": fwd["direct_c_prime"], "fwd_direct_ci": fwd["direct_ci"],
                     "fwd_prop": fwd["prop_mediated"], "fwd_prop_ci": fwd["prop_ci"],
                     "fwd_indirect_p": fwd["indirect_p_boot"],
                     "rev_total": rev["total_c"], "rev_indirect": rev["indirect_ab"], "rev_indirect_ci": rev["indirect_ci"],
                     "rev_direct": rev["direct_c_prime"], "rev_prop": rev["prop_mediated"], "rev_prop_ci": rev["prop_ci"]})
        print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "mediation_forward_reverse.csv", index=False)
    (a.out / "mediation_forward_reverse.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
