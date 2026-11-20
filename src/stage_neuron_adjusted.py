"""Does the diagnostic-stage association of the C-A score survive adjustment for the neuron score?

Primary endpoint 2 (C-A score vs diagnostic stage, age/sex-adjusted partial Spearman, all samples) is
recomputed with the neuron marker score (computed on all samples of the cohort) as an extra covariate,
and the neuron score's own stage association is reported for comparison. Pooled with the same
REML/HKSJ random-effects model as meta_reanalysis.py.

Usage: python stage_neuron_adjusted.py --data /mnt/project-files/d1-data/raw --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from meta_reanalysis import INDEPENDENT, boot_se, ca_score, pool, rho
from pilot_gse33000 import NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 7)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        ca = ca_score(zscore_rows(e.loc[genes, m.index]), cent).values
        mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
        neuron = zscore_rows(e.loc[mk, m.index]).mean().values
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov = np.column_stack([age, sex])
        cov_n = np.column_stack([age, sex, neuron])
        for ep, x, c in [("ca_stage_agesex", ca, cov), ("ca_stage_agesex_neuron", ca, cov_n),
                         ("neuron_stage_agesex", neuron, cov)]:
            r = rho(x, stage, c)
            rows.append({"endpoint": ep, "cohort": name, "n": len(x), "rho": round(r, 3),
                         "z": np.arctanh(r), "se_boot": boot_se(x, stage, c, rng, a.n_boot)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "per_cohort.csv", index=False)
    rep = {ep: {"per_cohort": dict(zip(d.cohort, d.rho)), "pooled": pool(d.z.values, d.se_boot.values)}
           for ep, d in df.groupby("endpoint")}
    (a.out / "stage_neuron_adjusted.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
