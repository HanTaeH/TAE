"""Supplementary tables (one Excel workbook) from the result files.

S1 literature (32 studies), S2 per-cohort primary endpoints incl. donor-overlap sensitivity,
S3 random-gene null summary, S4 SEA-AD pseudobulk tests with FDR, S5 all mediation models,
S6 Hallmark residual-axis gene sets, S7 secondary-family tests with FDR, S8 MSBB neuronal-SPV
normalisation per region, S9 measurement-error (reliability) correction of the neuron-adjusted stage test, S10 random-gene null
for the neuron-adjusted stage test, S11 stage follow-ups (bootstrap CIs of within-cohort tests, covariate
specifications, reverse partial, glial covariates, GSE33000 age sensitivity), S12 specificity-matched null,
S13 NNLS-adjusted / alternative-region / AUC case-control analyses, S14 SEA-AD NeuN histology; added for v9:
S15 strength-matched null and kappa against random covariates, S16 further subtype schemes, S17 positive-control
simulation, S18 CSF proteomic subtypes projected on brain, S19 residual axis (de novo, MSBB, GTEx, SEA-AD),
S20 methods coding of the 29 de novo subtype studies.

Usage: python supplementary_tables.py --res ../results --out <file.xlsx>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FULL_TEXT = {"Milind2020", "Neff2021", "ZhengXu2021", "LiH2022_consensus", "Eteleeb2024", "Song2025", "Lee2026",
             "Lai2022", "Wang2024"}  # the last two were added from other sources
COMPOSITION_FULLTEXT = {
    "Milind2020": "not estimated",
    "Neff2021": "estimated (BRETIGEA); described per subtype; CDR/MMSE normalisation before subtyping",
    "ZhengXu2021": "not estimated (stated as a limitation)",
    "LiH2022_consensus": "not reported",
    "Lai2022": "immune cell scores only; no adjustment",
    "Eteleeb2024": "estimated by deconvolution; adjusted in differential expression",
    "Wang2024": "immune cell scores only (CIBERSORT); no adjustment",
    "Song2025": "not estimated",
    "Lee2026": "estimated; described per subtype",
}


def json_rows(obj, prefix=""):
    """Flatten nested JSON into (key, value) rows for a sheet."""
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows += json_rows(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list) and obj and all(isinstance(x, (dict, list)) for x in obj):
        for i, v in enumerate(obj):
            rows += json_rows(v, f"{prefix}[{i}]")
    else:
        rows.append({"key": prefix, "value": str(obj) if isinstance(obj, list) else obj})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R = a.res
    sheets = {}

    lit = pd.read_csv(R / "literature/included_features.csv")
    lit = lit.sort_values("year").groupby("study_id", as_index=False).agg(
        first_author=("first_author", "first"), year=("year", "max"), journal=("journal", "last"),
        role=("role", "first"), PMIDs=("PMID", lambda s: "; ".join(map(str, s))), title=("title", "last"),
        cohorts=("cohorts", lambda s: "; ".join(sorted({x for x in s.dropna()}))),
        composition_mentioned=("mentions_composition", "max"), null_control=("mentions_null_control", "max"),
        independent_validation=("mentions_validation", "max"))
    lai = pd.DataFrame([{"study_id": "Lai2022", "first_author": "Lai Y", "year": 2022, "journal": "Front Immunol",
                         "role": "de novo", "PMIDs": "", "title": "(added from other sources; see Methods)",
                         "cohorts": "GEO (3 series)", "composition_mentioned": True, "null_control": False,
                         "independent_validation": False},
                        {"study_id": "Wang2024", "first_author": "Wang W", "year": 2024, "journal": "Front Immunol",
                         "role": "de novo", "PMIDs": "", "title": "Identification of early Alzheimer's disease subclass and "
                         "signature genes based on PANoptosis genes (added from other sources)",
                         "cohorts": "GSE48350, GSE122063, GSE5281", "composition_mentioned": True, "null_control": False,
                         "independent_validation": False}])
    lit = pd.concat([lit, lai], ignore_index=True)
    lit["features_from"] = ["full text" if s in FULL_TEXT else "abstract" for s in lit.study_id]
    # automated keyword flags come from titles/abstracts; for the nine full-text studies the manual
    # reading below takes precedence (results/literature comparison table)
    lit = lit.rename(columns={"composition_mentioned": "composition_keyword_in_abstract"})
    lit["composition_fulltext"] = lit.study_id.map(COMPOSITION_FULLTEXT).fillna("not assessed (abstract only)")
    sheets["S1_literature"] = lit.sort_values(["year", "first_author"])

    sheets["S2_primary_per_cohort"] = pd.read_csv(R / "meta_revision/per_cohort.csv")
    mr = json.loads((R / "meta_revision/meta_report.json").read_text())
    sheets["S2b_pooled"] = pd.DataFrame([{"analysis": k, **{kk: (str(vv) if isinstance(vv, list) else vv)
                                                            for kk, vv in v["se_boot"].items()}}
                                         for k, v in mr.items() if "se_boot" in v])
    ss = json.loads((R / "singlesample/singlesample_report.json").read_text())["pooled_AD_ca_vs_neuron"]
    sheets["S2b_pooled"] = pd.concat([sheets["S2b_pooled"], pd.DataFrame([{
        "analysis": "P1_neuron, reference-free single-sample score (4 cohorts)", "rho": ss["rho"],
        "ci95_hksj": str(ss["ci"]), "p_hksj": ss["p"]}])], ignore_index=True)
    sheets["S3_random_gene_null"] = pd.read_csv(R / "null/null_summary.csv")
    sheets["S4_seaad_pseudobulk_tests"] = pd.read_csv(R / "seaad_sensitivity/fdr_all_pseudobulk_tests.csv")
    sheets["S5_mediation_all_models"] = pd.read_csv(R / "mediation_revision/mediation_forward_reverse.csv")
    h = pd.read_csv(R / "residual_go/HALLMARK_consistency.csv", index_col=0)
    sheets["S6_hallmark_residual_axis"] = h.sort_values("meta_z").reset_index()
    mt = pd.read_csv(R / "multiple_testing/all_tests.csv")
    sheets["S7_primary_secondary_tests"] = mt[mt.family.isin(["primary", "secondary"])].sort_values(["family", "p"])
    sheets["S8_msbb_neuron_spv_norm"] = pd.read_csv(R / "msbb/msbb_neuron_spv_normalisation.csv")
    def flat(v):
        return {kk: (str([round(float(x), 3) for x in vv]) if isinstance(vv, list) else vv) for kk, vv in v.items()}
    s9 = pd.read_csv(R / "stage_reliability/per_cohort.csv")
    for c in s9.columns:
        if s9[c].astype(str).str.startswith("[").any():
            s9[c] = s9[c].map(lambda x: str([round(float(t), 3) for t in str(x).strip("[]").split(",")]) if str(x).startswith("[") else x)
    na = json.loads((R / "stage_neuron_adjusted/stage_neuron_adjusted.json").read_text())
    rel = json.loads((R / "stage_reliability/stage_reliability.json").read_text())
    pooled9 = [{"cohort": "POOLED " + k, **flat(v["pooled"])} for k, v in na.items()]
    pooled9.append({"cohort": "POOLED ca_stage_agesex_neuron_disattenuated", **flat(rel["pooled_partial_disattenuated"])})
    sheets["S9_stage_reliability"] = pd.concat([pd.DataFrame(pooled9), s9], ignore_index=True)
    sheets["S10_stage_neuron_adj_null"] = pd.read_csv(R / "stage_neuron_adjusted/stage_neuron_adjusted_null.csv")
    fu = json.loads((R / "stage_followups/stage_followups.json").read_text())
    within = pd.DataFrame([{"test": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}}
                           for k, v in fu["within_cohort"].items()])
    pooled = pd.DataFrame([{"test": "pooled_" + k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}}
                           for k, v in fu["pooled"].items()])
    sheets["S11_stage_followups"] = pd.concat([within, pooled, pd.read_csv(R / "stage_followups/stage_followups_per_cohort.csv")
                                               .assign(test=lambda d: "per_cohort_" + d.spec)], ignore_index=True)
    sp = json.loads((R / "null_specificity/null_specificity_matched_report.json").read_text())
    sheets["S12_null_specificity_matched"] = pd.DataFrame([{"cohort": r["cohort"], "metric": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in r[k].items()}}
                                                          for r in sp for k in ["neuron_rho", "stage_agesex", "stage_agesex_neuron"]])
    spp = R / "null_specificity/null_specificity_pooled_report.json"
    if spp.exists():
        pp = json.loads(spp.read_text())["pooled"]
        sheets["S12_null_specificity_matched"] = pd.concat(
            [pd.DataFrame([{"cohort": "POOLED (4 cohorts)", "metric": k, **flat(v)} for k, v in pp.items()]),
             sheets["S12_null_specificity_matched"]], ignore_index=True)
    ex = json.loads((R / "stage_extra/stage_extra_report.json").read_text())
    sheets["S13_stage_extra"] = pd.concat([pd.DataFrame([{"spec": "POOLED " + k.replace("pooled_", ""), **flat(v)}
                                                         for k, v in ex.items() if k.startswith("pooled")]),
                                           pd.read_csv(R / "stage_extra/stage_extra_per_cohort.csv")], ignore_index=True)
    nb = json.loads((R / "seaad_neun/seaad_neun_report.json").read_text())["neun_density_beyond"]
    beyond = pd.DataFrame([{"subset": sub, "histology": "neun_density", "score": sc, "n": v["n"],
                            "rho_age_sex": r["rho_given_age_sex_CPS"], "p_age_sex": r["p"], "ci_age_sex": str(r["ci"]),
                            "note": "given age, sex and CPS"} for sub, v in nb.items() for sc, r in v.items()
                           if isinstance(r, dict)] +
                          [{"subset": sub, "histology": "neun_density", "score": sc, "n": v["n"],
                            "rho_age_sex": r["rho_given_age_sex_FANS_neuron_frac"], "p_age_sex": r["p_fans"],
                            "note": "given age, sex and FANS neuronal fraction"} for sub, v in nb.items()
                           for sc, r in v.items() if isinstance(r, dict)])
    sheets["S14_seaad_neun_histology"] = pd.concat([pd.read_csv(R / "seaad_neun/seaad_neun_associations.csv"), beyond],
                                                   ignore_index=True)

    import json as _json
    jl = lambda f: _json.loads((R / f).read_text())
    # S15 neuron association: strength-matched null, both tails (v10)
    up = jl("bound_upper_tail/bound_upper_tail.json")
    fu = jl("gse33000_null_followup/gse33000_null_followup.json")
    rows = []
    for c, d in up.items():
        rows.append({"cohort": c, "n_AD": d["n_ad"], "observed_rho": d["obs"],
                     "strength_null_median": d["strength"]["null_median"], "strength_p_more_coupled": d["strength"]["p_lower"],
                     "strength_p_less_coupled": d["strength"]["p_upper"],
                     "spec_null_median": d.get("spec", {}).get("null_median"), "spec_p_more_coupled": d.get("spec", {}).get("p_lower"),
                     "spec_p_less_coupled": d.get("spec", {}).get("p_upper"), "note": "500 sets each; p = (k+1)/(N+1)"})
    rows.append({"cohort": "GSE33000_PFC", "n_AD": fu["n_ad"], "observed_rho": fu["finer_match"]["neff"],
                 "note": f"finer match on top non-neuronal type: null median {fu['finer_match']['random_median']}, p more coupled {fu['finer_match']['p_empirical']}"})
    for n, d in fu["subsample"].items():
        rows.append({"cohort": f"GSE33000_PFC subsample n={n}", "n_AD": int(n), "observed_rho": d["obs_median"],
                     "spec_null_median": d["null_median"], "spec_p_more_coupled": d["p_median"],
                     "note": f"30 subsamples; median p; share p<0.05 = {d['share_p_lt_0.05']}"})
    sheets["S15_neuron_strength_null"] = pd.DataFrame(rows)
    sheets["S16_kappa_random_covariates"] = pd.read_csv(R / "msbb/msbb_normalisation_kappa_null.csv")
    # S17 positive control (tail: classifier less coupled than random = upper tail)
    rows = []
    for tag, f in [("200", "positive_control/positive_control_simulation.json")] + \
                  [(n, f"positive_control/positive_control_simulation_nAD{n}.json") for n in (52, 97, 219, 310)]:
        d = jl(f)
        for scen, byf in d["results"].items():
            for frac, r in byf.items():
                rows.append({"scenario": scen, "n_AD": int(tag), "signal_share_of_contrast": float(frac), "obs_rho_median": r["obs_median"],
                             "spec_null_median": r["spec"]["null_median"], "spec_detection_rate": r["spec"]["share_p_lt_0.05"],
                             "strength_null_median": r["strength"]["null_median"], "strength_detection_rate": r["strength"]["share_p_lt_0.05"],
                             "tail": "classifier less neuron-coupled than random" if scen == "within_AD_subtype" else
                                     "neuron-adjusted case-control association larger than random",
                             "genes": d["n_genes"], "reps": 20, "null_sets": 200})
    sheets["S17_positive_control"] = pd.DataFrame(rows)
    os_ = pd.read_csv(R / "other_schemes/other_schemes_clustered.csv")
    sheets["S18_further_schemes"] = os_[~os_.iloc[:, 0].astype(str).str.contains("Wang")]
    # S19 CSF subtypes
    rows = []
    pr, fo = jl("tijms_csf/tijms_csf_projection.json"), jl("tijms_followup/tijms_followup.json")
    for cp, cf in zip(pr["cohorts"], fo["cohorts"]):
        for st, d in cp["subtypes"].items():
            e = cf["subtypes"][st]
            rows.append({"cohort": cp["cohort"], "n_AD": cp["n_AD"], "genes_found": cp["genes_found"], "subtype": st,
                         "rho_neuron": d["rho_neuron"], "rho_neff_CA": d["rho_neff_CA"], "p_expr": d["p_expr"], "p_spec": d["p_spec"],
                         "genes_without_neff": cf["tijms_genes_used"], "neuron_strength_p": e["neuron_strength_p"],
                         "rho_resid_vs_neff_resid": e["rho_resid_vs_neff_resid_CA"], "resid_strength_p": e["resid_strength_p"],
                         "resid_spec_p": e["resid_spec_p"]})
    sheets["S19_csf_subtypes_brain"] = pd.DataFrame(rows)
    # S20 residual axis
    rows = []
    dn = jl("residual_denovo/residual_denovo.json")
    for c, d in dn["cohorts"].items():
        for k in ("unadjusted", "adjusted"):
            x = d[k]
            rows.append({"source": "de novo PCA", "cohort": c, "data": k, "n": d["n_AD"],
                         "var_PC1_5": str(x["var_pc1_5"]), "PC1_abs_rho_neuron": x["pc1_abs_rho_neuron"],
                         "silhouette": x["sigclust"]["silhouette"], "silhouette_null_median": x["sigclust"]["null_median"],
                         "sigclust_p": x["sigclust"]["p"]})
    for pair, d in dn["replication"].items():
        rows.append({"source": "loading replication", "cohort": pair, "data": "adjusted",
                     "note": f"best |r| per PC of first {d['best_abs_r_per_PC_of_first']}; matched PC {d['matched_PC_of_second']}; null p95 {d['null_p95']}"})
    gq, ga = jl("gtex_residual_axis_qc/gtex_residual_axis_qc.json"), jl("gtex_residual_axis/gtex_residual_axis.json")
    for t in ("cortex", "frontal_BA9"):
        rows.append({"source": "GTEx v8", "cohort": t, "n": gq[t]["n"],
                     "note": f"|r| template vs residual PC1-5 {ga[t]['abs_r_template_vs_residual_PC1_5']}; RIN rho {ga[t]['template']['rin_rho']}; "
                             f"R2 in-sample {gq[t]['r2_in_sample']}; R2 CV {gq[t]['r2_cv']}; gene-signature r {gq[t]['template_vs_gene_signature_r']}"})
    rows.append({"source": "GTEx v8", "cohort": "same donor cortex vs BA9", "n": gq["donor_cortex_vs_BA9"]["n_donors"],
                 "note": f"rho raw {gq['donor_cortex_vs_BA9']['rho_raw']}; after RIN and RNA-seq metrics {gq['donor_cortex_vs_BA9']['rho_after_RIN_seqQC']}; "
                         f"template vs gene length {gq['template_vs_gene_features_spearman']}"})
    sa = jl("seaad_residual_axis/seaad_residual_axis.json")
    for k, d in sa.items():
        if isinstance(d, dict) and "RIN" in d:
            rows.append({"source": "SEA-AD pseudobulk", "cohort": k, "n": sa["n_donors"],
                         "note": "; ".join(f"{v} {d[v]}" for v in ("abs_r_template_vs_PC1_5", "RIN", "pH", "PMI", "age", "Braak", "dementia") if v in d)})
    sheets["S20_residual_axis"] = pd.DataFrame(rows)
    # methods coding merged into S1
    mc = pd.read_csv(R / "literature/methods_coding_2026-11-25.csv")
    sheets["S1b_methods_coding"] = mc
    # within-AD plaque (S11 addition)
    pl = jl("msbb/msbb_plaque_within_ad.json")
    sheets["S11_stage_followups"] = pd.concat([sheets["S11_stage_followups"], pd.DataFrame(
        [{"spec": f"MSBB plaque density, {g}, {k}", **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in d[k].items()}, "n": d["n"]}
         for g, d in pl.items() for k in ("age_sex", "age_sex_neuron")])], ignore_index=True)

    titles = {
        "S1": "Included studies of the structured literature search (S1b: methods coding of all 32 studies, source full text or abstract)",
        "S2": "Per-cohort primary endpoints (S2b: pooled random-effects estimates, REML/HKSJ)",
        "S3": "Expression-matched random-gene null (1,000 sets) per cohort-region set",
        "S4": "SEA-AD pseudobulk x outcome tests with nominal and random-gene FDR",
        "S5": "Forward and reverse mediation models (descriptive)",
        "S6": "Hallmark gene sets on the composition-adjusted residual axis (GO sets in the code repository)",
        "S7": "Primary and secondary tests with within-family and global FDR",
        "S8": "MSBB neuronal SPV normalisation per region",
        "S9": "Neuron-adjusted case-control association: pooled rows (POOLED ...) and reliability correction per cohort",
        "S10": "Expression-matched random-gene null for the neuron-adjusted case-control association",
        "S11": "Follow-up stage analyses: bootstrap CIs, covariate specifications, reverse partial, glial covariates, age",
        "S12": "Random-gene null matched on cell-type specificity: pooled rows and per cohort (one-sided empirical p)",
        "S13": "NNLS-adjusted case-control association, alternative GSE118553 regions, GSE118553 AD vs control only, AUC with "
               "bootstrap 95% CIs (AUC compares AD with controls only; GSE118553 n = 83 excludes asymptomatic AD)",
        "S14": "SEA-AD histology (MTG grey matter) against pseudobulk C-A scores; subset neun_share_ge_5pct excludes 4 donors "
               "with NeuN+ < 5% of nuclei (threshold set after inspecting the data); rows with a note are NeuN density given CPS or FANS fraction",
        "S15": "Neuron association against strength-matched (post hoc) and specificity-matched nulls in both tails; GSE33000 finer match and subsamples",
        "S16": "MSBB kappa against labels before normalisation: neuronal SPV, CDR and 200 single random-gene covariates per region",
        "S17": "Positive-control simulation: detection rate by injected composition-independent signal (share of published C-A contrast), AD cohort size and null",
        "S18": "Further subtype schemes (Lai 2022, Ma 2023): PC1 axis vs neuron score, cell-score R2, label AUC, expression- and specificity-matched nulls",
        "S19": "Tijms 2024 CSF proteomic subtypes projected on brain bulk cohorts (supplementary analysis)",
        "S20": "Residual axis after five cell scores: de novo PCA, clustering, loading replication, GTEx v8 and SEA-AD tracing",
    }
    # v10: renumber in order of first citation in the manuscript (old -> new)
    REN = {"S7": "S1", "S1": "S2", "S3": "S3", "S12": "S4", "S15": "S5", "S17": "S6", "S2": "S7", "S16": "S8", "S8": "S9",
           "S13": "S10", "S9": "S11", "S11": "S12", "S10": "S13", "S5": "S14", "S4": "S15", "S14": "S16", "S18": "S17",
           "S19": "S18", "S6": "S19", "S20": "S20"}
    def ren(k):
        t = k.split("_")[0]
        base, suf = (t[:-1], t[-1]) if t[-1] == "b" else (t, "")
        return REN[base] + suf + k[len(t):]
    titles = {REN[k]: v for k, v in titles.items()}
    sheets = {ren(k): v for k, v in sheets.items()}
    sheets = dict(sorted(sheets.items(), key=lambda kv: (int(kv[0].split("_")[0].rstrip("b")[1:]), kv[0])))
    readme = pd.DataFrame([{"sheet": k, "table": k.split("_")[0], "title": titles.get(k.split("_")[0].rstrip("b"), "")} for k in sheets])
    readme = pd.concat([pd.DataFrame([{"sheet": "README", "table": "", "title":
                        "rho = Spearman or partial Spearman; ci95_hksj = Hartung-Knapp CI; pi95 = 95% prediction interval; "
                        "p_empirical = (k+1)/(N+1), one-sided; se_boot = bootstrap SE of Fisher z"}]), readme], ignore_index=True)
    sheets = {"README": readme, **sheets}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.out) as w:
        for k, d in sheets.items():
            d.to_excel(w, sheet_name=k[:31], index=False)
    print({k: len(v) for k, v in sheets.items()})


if __name__ == "__main__":
    main()
