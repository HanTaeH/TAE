"""Apply manual title/abstract screening decisions to the pre-screened PubMed export
and produce PRISMA counts plus an included-study feature table.

Usage: python lit_screen_decisions.py records_prescreened.csv out_dir
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

# study_id groups duplicate records (preprint + journal) into one study
INCLUDE = {
    # pmid: (study_id, role)
    "32492070": ("Milind2020", "de novo"),
    "33523961": ("Neff2021", "de novo"),
    "34014928": ("ZhengXu2021", "de novo"),
    "38687811": ("Eteleeb2024", "de novo"),
    "39394418": ("Wang2024_PANoptosis", "de novo"),
    "40654912": ("Song2025", "de novo"),
    "42056682": ("Song2025", "de novo"),
    "42620705": ("Lee2026", "de novo"),
    "36747803": ("Lee2026", "de novo"),
    "36399579": ("IturriaMedina2022", "de novo"),
    "40196602": ("IturriaMedina2025_translation", "application"),
    "40060468": ("Psych_traits_multiomic_subtypes", "application"),
    "42324246": ("Psych_traits_multiomic_subtypes", "application"),
    "42313214": ("Luleci2026_metabolic", "application"),
    "40181382": ("Ye2025_RNAmod_PCD", "de novo"),
    "37082508": ("Yang2023_SNF", "de novo"),
    "41313603": ("Mottaqi2025_PRISM-ML", "de novo"),
    "31763973": ("Hong2019_REO", "de novo"),
    "36204002": ("LiH2022_consensus", "de novo"),
    "37396662": ("Ma2023_m7G", "de novo"),
    "40074652": ("Hu2025_druggable", "de novo"),
    "40685632": ("Huang2025_pyroptosis", "de novo"),
    "42465724": ("Zhang2026_autolysosome_astro", "de novo"),
    "42774120": ("Gong2026_PPP_astro", "de novo"),
    "42568632": ("Liu2026_mitophagy_astro", "de novo"),
    "41757211": ("Nsangou2026_ROSMAP_DL", "de novo"),
    "40100818": ("Honda2025_female", "de novo"),
    "32225014": ("Dagan2020_personalized", "de novo"),
    "42230965": ("Jorg2026_ANG_female", "de novo"),
    "37736681": ("Zeng2023_lipid", "de novo"),
    "42178007": ("Mu2026_glycosylation", "de novo"),
    # recovered from rule-excluded records by a title keyword check (subtyp|cluster|subgroup|heterogen|...)
    "41950003": ("Scheidemantel2026_multiomics", "de novo"),
    "41040192": ("Scheidemantel2026_multiomics", "de novo"),
    "42146901": ("AlOuran2026_latent_factor", "de novo"),
}
RECOVERED = {"41950003", "41040192", "42146901"}
TITLE_FLAG = re.compile(r"subtyp|cluster|subgroup|heterogen|taxonom|stratif", re.I)

# notable exclusions with explicit reasons (all other manual-review records get the default)
EXCLUDE_REASON = {
    "35694866": "no brain transcriptome (genetics, miRNA, proteomics, blood)",
    "42446473": "subtypes defined on proteomics; transcriptome only for validation",
    "41818290": "diagnosis model, no patient subtyping",
    "40760184": "sorted microglia, not bulk tissue",
    "38343831": "sorted microglia, not bulk tissue (preprint)",
    "42953729": "laser-captured neurons, differential expression only",
    "42457956": "single-nucleus cell-state clusters, not patient subtypes",
    "40584839": "single-nucleus cell-state clusters, not patient subtypes (preprint)",
    "34924992": "clusters normal-ageing brains, not AD subtypes",
    "40166175": "epigenome-only subtypes",
    "40799738": "epigenome-only subtypes",
    "41733741": "epigenome-only subtypes",
    "40475406": "personality subtypes, not molecular",
    "39812331": "blood cells only",
    "42395416": "groups defined by TDP-43 pathology, not expression",
    "41975594": "review",
}
DEFAULT_REASON = "not patient-level AD subtyping from brain bulk transcriptome"

COMP = re.compile(r"deconvol|cell[- ]type (?:proportion|composition|fraction|abundance)|cellular composition|"
                  r"xCell|CIBERSORT|MuSiC|neuronal (?:loss|proportion)|cell proportion", re.I)
NULL = re.compile(r"random gene|permutation|null model|null distribution|random(?:ly)? (?:selected|sampled) genes", re.I)
EXT = re.compile(r"external|independent (?:cohort|dataset|validation)|validat|replicat", re.I)
RNASEQ = re.compile(r"RNA-seq|RNA seq|sequencing", re.I)
ARRAY = re.compile(r"microarray|array", re.I)
COHORT = re.compile(r"ROSMAP|ROS/MAP|Religious Orders|MSBB|Mount Sinai|Mayo|GSE\d+|AMP-AD|ANMerge|GTEx", re.I)


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(src, dtype={"PMID": str})
    d["stage"] = "excluded_pubtype"
    d.loc[~d.excl_pubtype & ~d.keep_for_manual, "stage"] = "excluded_rule"
    # manual review = rule-kept records plus rule-excluded records whose title has a subgrouping keyword
    title_flag = ~d.excl_pubtype & ~d.keep_for_manual & d.title.fillna("").str.contains(TITLE_FLAG)
    man = ~d.excl_pubtype & (d.keep_for_manual | title_flag)
    d.loc[man, "stage"] = "excluded_manual"
    inc = d.PMID.isin(INCLUDE)
    assert (inc <= (~d.excl_pubtype)).all()
    d.loc[inc, "stage"] = "included"
    d["study_id"] = d.PMID.map(lambda p: INCLUDE.get(p, ("", ""))[0])
    d["role"] = d.PMID.map(lambda p: INCLUDE.get(p, ("", ""))[1])
    d["reason"] = ""
    d.loc[d.stage == "excluded_manual", "reason"] = d.PMID.map(lambda p: EXCLUDE_REASON.get(p, DEFAULT_REASON))
    d.loc[d.stage == "excluded_rule", "reason"] = "rule: no subgrouping + brain molecular terms, or animal only"
    d.loc[d.stage == "excluded_pubtype", "reason"] = "publication type (review, editorial, comment, ...)"
    d[["PMID", "year", "first_author", "journal", "title", "stage", "study_id", "role", "reason"]].to_csv(
        out / "screening_decisions.csv", index=False)

    I = d[inc].copy()
    text = (I.title.fillna("") + " " + I.abstract.fillna(""))
    I["mentions_composition"] = text.str.contains(COMP)
    I["mentions_null_control"] = text.str.contains(NULL)
    I["mentions_validation"] = text.str.contains(EXT)
    I["rnaseq"] = text.str.contains(RNASEQ)
    I["array"] = text.str.contains(ARRAY)
    I["cohorts"] = text.map(lambda t: ", ".join(sorted({m.group(0).upper() for m in COHORT.finditer(t)})))
    I[["PMID", "study_id", "role", "year", "first_author", "journal", "title", "mentions_composition",
       "mentions_null_control", "mentions_validation", "rnaseq", "array", "cohorts"]].to_csv(
        out / "included_features.csv", index=False)

    n_found_known_missing = ["Lai2022 (Front Immunol; not retrieved, added from other sources)"]
    studies = I.groupby("study_id").agg(role=("role", "first")).reset_index()
    prisma = {
        "identified_search1": 366,
        "identified_search2": int(len(d)),
        "duplicates_removed": 366,
        "records_after_dedup": int(len(d)),
        "excluded_pubtype": int((d.stage == "excluded_pubtype").sum()),
        "screened_title_abstract": int((d.stage != "excluded_pubtype").sum()),
        "excluded_rule": int((d.stage == "excluded_rule").sum()),
        "manual_review": int(man.sum()),
        "manual_review_from_rule_kept": int((~d.excl_pubtype & d.keep_for_manual).sum()),
        "manual_review_from_title_check": int(title_flag.sum()),
        "recovered_by_title_check": int(d.PMID.isin(RECOVERED).sum()),
        "excluded_manual": int((d.stage == "excluded_manual").sum()),
        "included_records": int(inc.sum()),
        "included_studies": int(len(studies)),
        "included_de_novo": int((studies.role == "de novo").sum()),
        "included_application": int((studies.role == "application").sum()),
        "other_sources": n_found_known_missing,
        "included_studies_total": int(len(studies)) + 1,
        "study_level_flags": {
            k: int(I.groupby("study_id")[k].any().sum())
            for k in ["mentions_composition", "mentions_null_control", "mentions_validation"]
        },
    }
    (out / "prisma_counts.json").write_text(json.dumps(prisma, indent=2, ensure_ascii=False))
    print(json.dumps(prisma, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
