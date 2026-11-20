"""Follow-up checks on the stage analyses requested in the second internal review round.

1. Bootstrap 95% CIs (2,000 resamples) for the neuron-adjusted within-cohort tests:
   MSBB subject means (cortex; all 106 subjects and the 64 AD subjects) for CDR and Braak, and
   GSE125583 AD cases for Braak; covariates as in msbb_gse84422.py / braak_gse125583.py.
2. Diagnostic-group (case-control) association adjusted for age, sex and the neuron score:
   (a) neuron score entered as ranks, pooled with bootstrap SEs;
   (b) the reverse partial (neuron score vs group given age, sex and C-A), pooled;
   (c) the four glial marker scores added as covariates (per cohort);
   (d) GSE33000 age sensitivity: subjects aged >= 60 only, and age entered as a
       cubic polynomial.
Pooling uses REML/HKSJ as in meta_reanalysis.py.

Usage: python stage_followups.py --data /mnt/project-files/d1-data/raw --res ../results --out <dir>
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
from meta_reanalysis import INDEPENDENT, boot_se, ca_score, pool, rho
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts


def boot_ci(x, y, cov, rng, n=2000):
    vals = []
    m = len(x)
    for _ in range(n):
        i = rng.integers(0, m, m)
        with np.errstate(all="ignore"):
            r = partial_spearman(x[i], y[i], None if cov is None else cov[i])[0]
        if np.isfinite(r):
            vals.append(r)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def age_cubic(x):
    """Age as a cubic polynomial (centred and scaled) to allow non-linear age effects."""
    z = (x - x.mean()) / x.std()
    return np.column_stack([z, z ** 2, z ** 3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 31)
    rep = {"within_cohort": {}}

    # 1. within-cohort tests with bootstrap CIs
    s = pd.read_csv(a.res / "msbb/msbb_subject_scores.csv")
    for scope, d in [("MSBB_all_subjects", s), ("MSBB_AD_subjects", s[s.group == "AD"])]:
        cov = d[["age", "female"]].values.astype(float)
        covn = np.column_stack([cov, d["neuron"].values])
        for c in ["cdr", "braak"]:
            x, y = d["ca"].values, d[c].values.astype(float)
            r0, p0, n = partial_spearman(x, y, cov)
            r1, p1, _ = partial_spearman(x, y, covn)
            rep["within_cohort"][f"{scope}_{c}"] = {
                "n": n, "rho_agesex": round(r0, 3), "p_agesex": p0, "ci_agesex": boot_ci(x, y, cov, rng, a.n_boot),
                "rho_agesex_neuron": round(r1, 3), "p_agesex_neuron": p1,
                "ci_agesex_neuron": boot_ci(x, y, covn, rng, a.n_boot)}
    g = pd.read_csv(a.res / "braak_gse125583/braak_gse125583_per_sample.csv", index_col=0)
    g = g[g.set == "AD_only"].dropna(subset=["braak", "age"])
    cov = np.column_stack([g["age"].values, (g["sex"] == "male").astype(float).values])
    covn = np.column_stack([cov, g["neuron_score"].values])
    x, y = g["ca_score"].values, g["braak"].values
    r0, p0, n = partial_spearman(x, y, cov)
    r1, p1, _ = partial_spearman(x, y, covn)
    rep["within_cohort"]["GSE125583_AD_braak"] = {
        "n": n, "rho_agesex": round(r0, 3), "p_agesex": p0, "ci_agesex": boot_ci(x, y, cov, rng, a.n_boot),
        "rho_agesex_neuron": round(r1, 3), "p_agesex_neuron": p1,
        "ci_agesex_neuron": boot_ci(x, y, covn, rng, a.n_boot)}

    # 2. diagnostic-group association
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        ca = ca_score(zscore_rows(e.loc[genes, m.index]), cent).values

        def sc(lst):
            mk = [x for x in lst if x in e.index and x not in genes]
            return zscore_rows(e.loc[mk, m.index]).mean().values
        neu = sc(NEURON_MARKERS)
        glia = np.column_stack([sc(v) for v in CELL_MARKERS.values()])
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        base = np.column_stack([age, sex])
        neu_rank = stats.rankdata(neu)
        specs = {"neuron_as_ranks": (ca, np.column_stack([base, neu_rank])),
                 "reverse_neuron_given_ca": (neu, np.column_stack([base, ca])),
                 "neuron_plus_glia": (ca, np.column_stack([base, neu, glia]))}
        if name == "GSE33000_PFC":
            specs["age_cubic"] = (ca, np.column_stack([age_cubic(age), sex, neu]))
        for spec, (xv, cv) in specs.items():
            r = rho(xv, stage, cv)
            rows.append({"spec": spec, "cohort": name, "n": len(xv), "rho": round(r, 3), "z": np.arctanh(r),
                         "se_boot": boot_se(xv, stage, cv, rng, 500)})
        if name == "GSE33000_PFC":
            k = age >= 60
            r = rho(ca[k], stage[k], np.column_stack([base, neu])[k])
            rows.append({"spec": "age_ge_60", "cohort": name, "n": int(k.sum()), "rho": round(r, 3),
                         "z": np.arctanh(r), "se_boot": boot_se(ca[k], stage[k], np.column_stack([base, neu])[k],
                                                                 rng, 500)})
            # pooled estimate with GSE33000 replaced by the >=60 subset is computed below
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out / "stage_followups_per_cohort.csv", index=False)
    rep["pooled"] = {sp: pool(d.z.values, d.se_boot.values) for sp, d in df.groupby("spec")
                     if sp in {"neuron_as_ranks", "reverse_neuron_given_ca", "neuron_plus_glia"}}
    main_rows = pd.read_csv(a.res / "stage_neuron_adjusted/per_cohort.csv")
    main_rows = main_rows[main_rows.endpoint == "ca_stage_agesex_neuron"].set_index("cohort")
    for sp in ["age_ge_60", "age_cubic"]:
        alt = main_rows.copy()
        r = df[(df.spec == sp)].iloc[0]
        alt.loc["GSE33000_PFC", ["z", "se_boot"]] = [r.z, r.se_boot]
        rep["pooled"][f"GSE33000_{sp}"] = pool(alt.z.values, alt.se_boot.values)
    (a.out / "stage_followups.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
