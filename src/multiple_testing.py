"""Pre-specified endpoints and false-discovery control across every test in the project.

Families
  primary       2 pre-specified endpoints, random-effects pooled over the independent cohorts
                (forest_summary.json): P1 C-minus-A vs neuron marker score (AD samples),
                P2 C-minus-A vs disease stage (all samples). Bonferroni, alpha 0.05 / 2.
  secondary     per-cohort versions of the primary claims and the main follow-up claims, listed
                in CONFIRMATORY (regex on 'report::path'); Benjamini-Hochberg within the family.
  exploratory   every other p-value found in results/*/*report.json; BH within the family.
A global BH over all tests is reported too. Tests whose expected result is a null (e.g. after
adjusting for the neuron score) are flagged 'adjusted_for_neuron' - for them a large p is the
finding and FDR does not apply; they are reported with the estimate.

Usage: python multiple_testing.py ../results
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INDEP = "(GSE33000_PFC|GSE132903_MTG|GSE118553_Temporal_Cortex|GSE125583_FG)"
CONFIRMATORY = [
    rf"^null::{INDEP}/(neuron_rho|stage_rho|r2_cell)/p_empirical$",
    r"^braak_gse125583::AD_only/(ca_vs_braak|random_null)/p$",
    r"^braak::(T1_all/ca_vs_braak|T5_random_null/all)/p$",
    r"^seaad_pseudobulk::sorted/(ca_vs_CPS|ca_vs_CPS_random_null)/p$",
    r"^msbb::S_subject_cortex/ca_vs_cdr/p$",
    # v11: the across-region test stays in the family, tested against its CDR-shuffle null (a null result)
    r"^deconv_validation_blocks::B_msbb_across_region_blocks/p_obs_more_negative_than_null$",
    # v11 (review round 9): the classifier-versus-cell-type-matched-null comparison, both tails
    rf"^bound_upper_tail::{INDEP}/(spec|strength)/p_(lower|upper)$",
    rf"^li2022::{INDEP}/null_max_abs_neuron_rho/p$",
    # second-round revision: secondary claims carried into the text
    rf"^null::{INDEP}/stage_rho_agesex/p_empirical$",
    r"^seaad_celltype_revision::(Exc_fixed_subclass_mix|type_Inh|sub_L4 IT|sub_L6 IT)/random_null/Braak/p$",
    r"^technical_confounders::MSBB/subject_ca_vs_cdr_age_sex_pmi_ph/p$",
]
EST_KEYS = ["rho", "obs", "neff", "neff_rho", "obs_rho", "r"]


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    n = int(ok.sum())
    if n == 0:
        return out
    o = np.argsort(p[ok])
    q = p[ok][o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty(n)
    tmp[o] = np.minimum(q, 1)
    out[ok] = tmp
    return out


def collect(res: Path) -> pd.DataFrame:
    rows = []
    # v11: every JSON result file (not only *report.json), except this step's own output and the literature counts
    # simulation outputs are power checks, not tests of the data, and are not counted
    skip = {"multiple_testing", "literature", "supplementary", "figures", "positive_control", "simulation",
            "simulation_revision", "seaad_v11"}
    files = [f for f in sorted(res.glob("*/*.json")) if f.parent.name not in skip]
    for f in dict.fromkeys(files):
        name = f.stem.replace("_report", "")
        name = {"braak_gse125583": "braak_gse125583"}.get(name, name)

        def walk(o, path):
            if isinstance(o, dict):
                est = next((o[k] for k in EST_KEYS if isinstance(o.get(k), (int, float))), np.nan)
                n = o.get("n") if isinstance(o.get("n"), (int, float)) else np.nan
                for k, v in o.items():
                    if (k == "p" or k.endswith("_p") or k.startswith("p_")) and isinstance(v, (int, float)) \
                            and not isinstance(v, bool):
                        rows.append({"report": name, "path": "/".join(path + [k]), "estimate": est, "n": n, "p": float(v)})
                    walk(v, path + [k])
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, path + [str(v.get("cohort", i)) if isinstance(v, dict) else str(i)])
        walk(json.loads(f.read_text()), [])
    nf = res / "stage_neuron_adjusted/stage_neuron_adjusted_null.csv"
    if nf.exists():  # random-gene null after neuron adjustment (one-sided empirical p per cohort)
        for _, r in pd.read_csv(nf).iterrows():
            rows.append({"report": "stage_neuron_adjusted_null", "path": f"{r.cohort}/p_empirical",
                         "estimate": r.neff_adjusted, "n": np.nan, "p": float(r.p_empirical)})
    for csv, rep in [("seaad_v11/seaad_regulators_within_type.csv", "seaad_regulators_within_type"),
                     ("seaad_v11/seaad_copathology.csv", "seaad_copathology")]:
        f = res / csv
        if f.exists():  # v11 tables with one test per row
            for _, r in pd.read_csv(f).iterrows():
                path = "/".join(str(r[c]) for c in r.index if c in ("gene", "cell_type", "vs", "pathology", "score", "adjust"))
                rows.append({"report": rep, "path": path + "/p", "estimate": r.get("rho_age_sex", r.get("rho")),
                             "n": r.get("n", np.nan), "p": float(r["p"])})
    df = pd.DataFrame(rows)
    # duplicates: the specificity-null lower tail is reported by both null_specificity_matched and
    # bound_upper_tail, and spike_in_real repeats both nulls; keep one copy of each test
    dup = ((df.report == "null_specificity_matched") & df.path.str.contains("neuron_rho")) | \
          ((df.report == "spike_in_real") & ~df.path.str.contains("strength_decile"))
    df = df[~dup]
    # v12 (review round 10, Reviewer 1 M5.2): list items that share a cohort label produced repeated
    # (report, path) keys. Exact repeats (same p) are one test; otherwise the keys are numbered.
    df = df.drop_duplicates(["report", "path", "p"])
    k = df.groupby(["report", "path"]).cumcount()
    many = df.duplicated(["report", "path"], keep=False)
    df.loc[many, "path"] = df.loc[many, "path"] + "#" + k[many].astype(str)
    return df


def main() -> None:
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
    df = collect(res)
    df["key"] = df["report"] + "::" + df["path"]
    df["family"] = np.where(df["key"].apply(lambda k: any(re.search(r, k) for r in CONFIRMATORY)),
                            "secondary", "exploratory")
    df["adjusted_for_neuron"] = df["path"].str.contains("neuron", case=False) & \
        df["path"].str.contains("partial|_adj", case=False)
    # primary family: REML + HKSJ pooled endpoints from meta_reanalysis.py (bootstrap SE);
    # P2 is the age/sex-adjusted stage endpoint
    mr = json.loads((res / "meta_revision/meta_report.json").read_text())
    pk = {"P1": "P1_neuron (4 cohorts)", "P2": "P2_stage age/sex adjusted (4 cohorts)"}
    prim = pd.DataFrame([{"report": "meta_revision", "path": f"{i}: {k}", "estimate": mr[k]["se_boot"]["rho"],
                          "n": mr[k]["se_boot"]["k"], "p": mr[k]["se_boot"]["p_hksj"], "family": "primary",
                          "adjusted_for_neuron": False, "key": f"meta_revision::{i}"} for i, k in pk.items()])
    df = df[df["report"] != "meta"]  # pooled endpoints are the primary family, not re-counted
    prim["q_family"] = np.minimum(prim["p"] * len(prim), 1)  # Bonferroni
    df["q_family"] = np.nan
    for fam in ["secondary", "exploratory"]:
        m = df["family"] == fam
        df.loc[m, "q_family"] = bh(df.loc[m, "p"].values)
    allt = pd.concat([prim, df], ignore_index=True)
    allt["q_global"] = bh(allt["p"].values)
    allt = allt[["family", "report", "path", "estimate", "n", "p", "q_family", "q_global", "adjusted_for_neuron"]]
    out = res / "multiple_testing"
    out.mkdir(exist_ok=True)
    allt.to_csv(out / "all_tests.csv", index=False)

    summ = {"n_tests": int(len(allt)), "by_family": {},
            "note": "n_p05 and n_q05 count only tests not flagged adjusted_for_neuron (for those a large p is the finding)"}
    for fam, d in allt.groupby("family"):
        disc = d[~d["adjusted_for_neuron"]]
        summ["by_family"][fam] = {"n": int(len(d)), "n_adjusted_for_neuron": int(d["adjusted_for_neuron"].sum()),
                                  "n_p05": int((disc["p"] < .05).sum()),
                                  "n_q05_family": int((disc["q_family"] < .05).sum()),
                                  "n_q05_global": int((disc["q_global"] < .05).sum())}
    summ["primary"] = prim[["path", "estimate", "n", "p", "q_family"]].to_dict(orient="records")
    conf = allt[allt["family"] == "secondary"].sort_values("p")
    summ["secondary"] = conf[["report", "path", "estimate", "p", "q_family"]].round(4).to_dict(orient="records")
    summ["secondary_failing_q05"] = conf.loc[conf["q_family"] >= .05, ["report", "path", "estimate", "p", "q_family"]] \
        .round(4).to_dict(orient="records")
    (out / "multiple_testing_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summ, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
