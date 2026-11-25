"""SEA-AD within-cell-type key regulators and co-pathology (v11, review round 9, Reviewer 2 M4 and M11a).

(1) Key regulators nominated by Neff et al. (2021), grouped as they reported them:
      class C down-regulated neuronal: ATP6V1A, SCN2A, GABRB2, NAPB
      class A up-regulated neuronal:   GABRB2, SYT1, SCN2A
      class C up-regulated:            TLN1, MSN, IL6R, LRP10
      class B (oligodendrocyte/myelin): PLP1, ERMN, QKI
    For each gene and broad cell type (Exc, Inh, Astro, Oligo, Micro), log2(CPM + 1) of the donor's type
    pseudobulk (composition fixed by construction) is related to the all-nuclei C-A score, the CPS and
    Braak stage (Spearman, and partial for age and sex). A within-type association with the subtype score
    means the regulator differs between subtypes inside the cell type, not only through composition.
(2) Co-pathology: LATE-NC stage, Lewy body disease (any vs none) and APOE e4 carriage against the
    all-nuclei and fixed-composition C-A scores, age- and sex-adjusted, and with CPS added.
BH-FDR within each block.

Usage: python seaad_regulators_copath_v11.py --donors /mnt/project-files/d1-data/seaad/donors \
         --cps /mnt/project-files/d1-data/seaad/Global_and_Local_CPS.20260105.csv --res ../results --out ../results/seaad_v11
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from braak_gse106241 import partial_spearman
from seaad_celltype_revision import load_sub
from seaad_pseudobulk import TYPES, cpm

REG = {"C_down_neuronal": ["ATP6V1A", "SCN2A", "GABRB2", "NAPB"],
       "A_up_neuronal": ["GABRB2", "SYT1", "SCN2A"],
       "C_up": ["TLN1", "MSN", "IL6R", "LRP10"],
       "B_oligo": ["PLP1", "ERMN", "QKI"]}
def multipletests(p, method="fdr_bh"):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return None, out


BROAD = ["Exc", "Inh", "Astro", "Oligo", "Micro"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--cps", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    genes, meta, allsubs, U, N = load_sub(a.donors)
    pdn = pd.read_csv(a.res / "seaad_celltype_revision/per_donor.csv", index_col=0).reindex(meta.index)
    age = pd.to_numeric(meta["Age at Death"].astype(str).str.replace("90+", "90", regex=False), errors="coerce").values
    sex = (meta["Sex"] == "Male").astype(float).values
    cov = np.column_stack([age, sex])
    ys = {"CA_all_nuclei": pdn["ca_sorted"].values, "CPS": pdn["CPS"].values, "Braak": pdn["Braak"].values}
    gi = {g: i for i, g in enumerate(genes)}
    rows = []
    for t in BROAD:
        k = [allsubs.index(s) for s in TYPES[t]]
        prof = np.log2(cpm(U[:, k].sum(1)) + 1)
        for grp, gl in REG.items():
            for g in gl:
                if g not in gi:
                    continue
                x = prof[:, gi[g]]
                for yn, y in ys.items():
                    ok = ~np.isnan(y)
                    r, p, _ = partial_spearman(x[ok], y[ok], cov[ok])
                    rows.append({"group": grp, "gene": g, "cell_type": t, "vs": yn, "mean_log2cpm": round(float(x.mean()), 2),
                                 "rho_age_sex": round(float(r), 3), "p": float(f"{p:.3g}")})
    reg = pd.DataFrame(rows).drop_duplicates(["gene", "cell_type", "vs"])
    reg["q_bh"] = multipletests(reg["p"], method="fdr_bh")[1].round(4)
    reg.to_csv(a.out / "seaad_regulators_within_type.csv", index=False)
    late = meta["LATE"].astype(str)
    late_ord = late.map(lambda v: 0 if "Not" in v else (int(v.split()[-1]) if v.split()[-1].isdigit() else np.nan)).astype(float).values
    lbd = (~meta["Highest Lewy Body Disease"].astype(str).str.startswith(("Not Identified", "nan"))).astype(float).values
    e4 = meta["APOE Genotype"].astype(str).str.contains("4").astype(float).values
    cp = []
    for xn, x in {"LATE_stage": late_ord, "Lewy_body_any": lbd, "APOE_e4": e4}.items():
        for sn in ("ca_sorted", "ca_fixed_comp"):
            s = pdn[sn].values
            for spec, c in {"age_sex": cov, "age_sex_CPS": np.column_stack([cov, pdn["CPS"].values])}.items():
                ok = ~np.isnan(x) & ~np.isnan(s) & ~np.isnan(c).any(1)
                r, p, _ = partial_spearman(s[ok], x[ok], c[ok])
                cp.append({"pathology": xn, "score": sn, "adjust": spec, "n": int(ok.sum()),
                           "n_positive": int((x[ok] > 0).sum()), "rho": round(float(r), 3), "p": float(f"{p:.3g}")})
    cp = pd.DataFrame(cp)
    cp["q_bh"] = multipletests(cp["p"], method="fdr_bh")[1].round(4)
    cp.to_csv(a.out / "seaad_copathology.csv", index=False)
    summ = {"regulators_q_lt_0.05_vs_CA": reg[(reg.vs == "CA_all_nuclei") & (reg.q_bh < .05)][["gene", "cell_type", "rho_age_sex", "q_bh"]].to_dict("records"),
            "regulators_tests": int(len(reg)), "copathology": cp.to_dict("records"),
            "late_values": late.value_counts().to_dict(), "lbd_values": meta["Highest Lewy Body Disease"].astype(str).value_counts().to_dict()}
    (a.out / "seaad_v11_report.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1)[:4000])


if __name__ == "__main__":
    main()
