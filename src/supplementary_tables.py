"""Supplementary tables (one Excel workbook) from the result files.

S1 literature (32 studies), S2 per-cohort primary endpoints incl. donor-overlap sensitivity,
S3 random-gene null summary, S4 SEA-AD pseudobulk tests with FDR, S5 all mediation models,
S6 Hallmark residual-axis gene sets, S7 secondary-family tests with FDR, S8 MSBB neuronal-SPV
normalisation per region, S9 measurement-error (reliability) correction of the neuron-adjusted stage test, S10 random-gene null
for the neuron-adjusted stage test, S11 stage follow-ups (bootstrap CIs of within-cohort tests, covariate
specifications, reverse partial, glial covariates, GSE33000 age sensitivity), S12 specificity-matched null,
S13 NNLS-adjusted / alternative-region / AUC case-control analyses, S14 SEA-AD NeuN histology.

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
    sheets["S3_random_gene_null"] = pd.read_csv(R / "null/null_summary.csv")
    sheets["S4_seaad_pseudobulk_tests"] = pd.read_csv(R / "seaad_sensitivity/fdr_all_pseudobulk_tests.csv")
    sheets["S5_mediation_all_models"] = pd.read_csv(R / "mediation_revision/mediation_forward_reverse.csv")
    h = pd.read_csv(R / "residual_go/HALLMARK_consistency.csv", index_col=0)
    sheets["S6_hallmark_residual_axis"] = h.sort_values("meta_z").reset_index()
    mt = pd.read_csv(R / "multiple_testing/all_tests.csv")
    sheets["S7_primary_secondary_tests"] = mt[mt.family.isin(["primary", "secondary"])].sort_values(["family", "p"])
    sheets["S8_msbb_neuron_spv_norm"] = pd.read_csv(R / "msbb/msbb_neuron_spv_normalisation.csv")
    sheets["S9_stage_reliability"] = pd.read_csv(R / "stage_reliability/per_cohort.csv")
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
    sheets["S13_stage_extra"] = pd.read_csv(R / "stage_extra/stage_extra_per_cohort.csv")
    sheets["S14_seaad_neun_histology"] = pd.read_csv(R / "seaad_neun/seaad_neun_associations.csv")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.out) as w:
        for k, d in sheets.items():
            d.to_excel(w, sheet_name=k[:31], index=False)
    print({k: len(v) for k, v in sheets.items()})


if __name__ == "__main__":
    main()
