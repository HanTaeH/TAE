"""Reviewer revision (R1 major 4-5): technical and sampling explanations of the main C-minus-A axis.

1. RNA quality / agonal proxies against the *raw* C-minus-A score (not only the residual):
     MSBB GSE84422  PMI, pH (per region within AD and all samples; subject means)
     GSE106241      RNA quality category
     SEA-AD MTG     RIN, PMI, brain pH (sorted pseudobulk C-minus-A)
   and the stage endpoints re-tested after adjusting for these covariates.
2. White-matter / oligodendrocyte sampling: oligodendrocyte marker score vs C-minus-A, and the
   C-minus-A vs neuron association after adjusting for the oligodendrocyte score, in every cohort.
3. Cerebellum: cerebellum-appropriate neuron markers (granule: GABRA6, CBLN1, NEUROD1, ZIC1, PAX6,
   RELN; Purkinje: PCP2, CA8, ITPR1, CALB1, PVALB) instead of the cortical set.

Usage: python technical_confounders.py --data /mnt/project-files/d1-data/raw --res <results> \
         --seaad-dir /mnt/project-files/d1-data/seaad --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from braak_gse106241 import partial_spearman
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS
from validate_cohorts import GROUP_ORDER, cohorts

CEREBELLAR = ["GABRA6", "CBLN1", "NEUROD1", "ZIC1", "PAX6", "RELN", "PCP2", "CA8", "ITPR1", "CALB1", "PVALB"]


def fmt(t):
    return {"rho": round(float(t[0]), 3), "p": float(f"{t[1]:.3g}"), "n": int(t[2])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--seaad-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    # 1a. MSBB
    m = pd.read_csv(a.res / "msbb/msbb_per_sample.csv", index_col=0)
    m["pmi"] = pd.to_numeric(m["pmi"], errors="coerce")
    m["ph"] = pd.to_numeric(m["ph"], errors="coerce")
    msbb = {}
    for scope, d in [("all", m), ("AD", m[m["group"] == "AD"])]:
        per_region = {}
        for r, g in d.groupby("region"):
            g = g.dropna(subset=["pmi", "ph"])
            per_region[r] = {"ca_vs_pmi": partial_spearman(g["ca"].values, g["pmi"].values)[0],
                             "ca_vs_ph": partial_spearman(g["ca"].values, g["ph"].values)[0],
                             "neuron_vs_ph": partial_spearman(g["neuron"].values, g["ph"].values)[0]}
        pr = pd.DataFrame(per_region).T
        msbb[f"{scope}_per_region_median"] = pr.median().round(3).to_dict()
        msbb[f"{scope}_per_region_range"] = {c: [round(pr[c].min(), 3), round(pr[c].max(), 3)] for c in pr}
    subj = m.groupby("subject").agg(ca=("ca", "mean"), neuron=("neuron", "mean"), pmi=("pmi", "first"),
                                    ph=("ph", "first"), cdr=("cdr", "first"), braak=("braak", "first"),
                                    age=("age", "first"), female=("female", "first")).dropna()
    cov = subj[["age", "female"]].values
    cov2 = subj[["age", "female", "pmi", "ph"]].values
    msbb["subject_n"] = int(len(subj))
    msbb["subject_ca_vs_pmi"] = fmt(partial_spearman(subj["ca"].values, subj["pmi"].values))
    msbb["subject_ca_vs_ph"] = fmt(partial_spearman(subj["ca"].values, subj["ph"].values))
    for x in ["cdr", "braak"]:
        msbb[f"subject_ca_vs_{x}_age_sex"] = fmt(partial_spearman(subj["ca"].values, subj[x].values, cov))
        msbb[f"subject_ca_vs_{x}_age_sex_pmi_ph"] = fmt(partial_spearman(subj["ca"].values, subj[x].values, cov2))
    msbb["subject_ca_vs_neuron_age_sex_pmi_ph"] = fmt(partial_spearman(subj["ca"].values, subj["neuron"].values, cov2))
    rep["MSBB"] = msbb

    # 1b. GSE106241
    b = pd.read_csv(a.res / "braak/braak_per_sample.csv", index_col=0)
    b["female"] = (b["sex"] == "female").astype(float)
    rep["GSE106241"] = {
        "ca_vs_rna_quality": fmt(partial_spearman(b["ca_score"].values, b["rna_quality"].values)),
        "neuron_vs_rna_quality": fmt(partial_spearman(b["neuron_score"].values, b["rna_quality"].values)),
        "ca_vs_braak_age_sex": fmt(partial_spearman(b["ca_score"].values, b["braak"].values, b[["age", "female"]].values)),
        "ca_vs_braak_age_sex_rnaq": fmt(partial_spearman(b["ca_score"].values, b["braak"].values,
                                                         b[["age", "female", "rna_quality"]].values)),
        "ca_vs_neuron_rnaq": fmt(partial_spearman(b["ca_score"].values, b["neuron_score"].values,
                                                  b[["rna_quality"]].values))}

    # 1c. SEA-AD
    p = pd.read_csv(a.res / "seaad_pseudobulk/seaad_pseudobulk_per_donor.csv", index_col=0)
    meta = pd.read_csv(a.seaad_dir / "donors/donor_meta.csv").set_index("Donor ID").loc[p.index]
    tech = meta[["RIN", "PMI", "Brain pH"]].apply(pd.to_numeric, errors="coerce")
    age = pd.to_numeric(meta["Age at Death"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    fem = (meta["Sex"] == "Female").astype(float)
    sa = {"n_with_all": int(tech.notna().all(axis=1).sum())}
    for c in tech:
        ok = tech[c].notna().values
        sa[f"ca_vs_{c}"] = fmt(partial_spearman(p["ca_sorted"].values[ok], tech[c].values[ok]))
    ok = tech.notna().all(axis=1).values
    base = np.column_stack([age.values, fem.values])[ok]
    full = np.column_stack([base, tech.values[ok]])
    sa["ca_vs_CPS_age_sex"] = fmt(partial_spearman(p["ca_sorted"].values[ok], p["CPS_Global"].values[ok], base))
    sa["ca_vs_CPS_age_sex_RIN_PMI_pH"] = fmt(partial_spearman(p["ca_sorted"].values[ok], p["CPS_Global"].values[ok], full))
    rep["SEA-AD"] = sa

    # 2-3. oligodendrocyte score and cerebellar markers in every GEO cohort
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows = []
    for name, e, mm in cohorts(a.data):
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        ad = mm.index[mm["group"] == "AD"]
        r = corr_np(z_np(e.loc[genes, ad].values), cent)
        ca = r[:, 2] - r[:, 0]

        def score(lst):
            lst = [g for g in lst if g in e.index and g not in genes]
            return z_np(e.loc[lst, ad].values).mean(0), len(lst)
        neuron, _ = score(NEURON_MARKERS)
        oligo, n_ol = score(CELL_MARKERS["oligodendrocyte"])
        row = {"cohort": name, "n_AD": len(ad), "n_oligo_markers": n_ol,
               "ca_vs_oligo": partial_spearman(ca, oligo)[0],
               "neuron_vs_oligo": partial_spearman(neuron, oligo)[0],
               "ca_vs_neuron": partial_spearman(ca, neuron)[0],
               "ca_vs_neuron_given_oligo": partial_spearman(ca, neuron, oligo[:, None])[0],
               "ca_vs_oligo_given_neuron": partial_spearman(ca, oligo, neuron[:, None])[0]}
        if "Cerebellum" in name:
            cb, n_cb = score(CEREBELLAR)
            row.update({"n_cerebellar_markers": n_cb, "ca_vs_cerebellar_neuron": partial_spearman(ca, cb)[0],
                        "cortical_vs_cerebellar_neuron": partial_spearman(neuron, cb)[0]})
        rows.append(row)
        print(name, flush=True)
    rep["oligodendrocyte_and_cerebellum"] = pd.DataFrame(rows).round(3).to_dict(orient="records")
    (a.out / "technical_confounders_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
