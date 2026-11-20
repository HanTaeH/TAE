"""Random-gene null matched on cell-type specificity as well as expression.

The main null (null_models.py) matches random genes on mean-expression decile and SD tercile only. A
reviewer asked whether the classifier still beats random genes when these also carry the same mix of
neuron-enriched, glia-enriched and non-specific genes. Each classifier gene is replaced by a random gene
of the same SEA-AD specificity class (log2 neuron/glia ratio > 1, < -1, otherwise; genes absent from the
reference form their own class) and the same mean-expression decile; the random gene inherits the
classifier gene's centroid row. Metrics per independent cohort:
  - neuron_rho: Spearman of the C-A score with the neuron marker score in AD samples;
  - stage_agesex: age/sex partial Spearman with diagnostic group (all samples);
  - stage_agesex_neuron: the same, also adjusted for the neuron score.
One-sided empirical p (lower tail for neuron_rho, upper tail for the stage metrics), 500 sets.
Pooled across cohorts in null_specificity_pooled.py.

Usage: python null_specificity_matched.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from classifier_specificity import NEURONAL
from meta_reanalysis import INDEPENDENT, ca_score, rho
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 51)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    neu = ref[NEURONAL].max(axis=1)
    gli = ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1)
    ratio = np.log2((neu + 1) / (gli + 1))
    spec_all = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    rows, dist = [], []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index]
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        ad = (m["group"] == "AD").values
        mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
        Z_all = zscore_rows(e)
        Z_ad = zscore_rows(e.loc[:, ad])
        neuron_all = Z_all.loc[mk].mean().values
        neuron_ad = Z_ad.loc[mk].mean().values
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov, covn = np.column_stack([age, sex]), np.column_stack([age, sex, neuron_all])

        def metrics(gs):
            c_ad = ca_score(Z_ad.loc[gs].set_axis(genes), cent).values
            c_all = ca_score(Z_all.loc[gs].set_axis(genes), cent).values
            return {"neuron_rho": stats.spearmanr(c_ad, neuron_ad)[0],
                    "stage_agesex": rho(c_all, stage, cov), "stage_agesex_neuron": rho(c_all, stage, covn)}
        obs = metrics(list(genes))
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        spec = spec_all.reindex(e.index).fillna(3).astype(int)
        stratum = dec * 4 + spec
        pool = e.index.difference(genes).difference(list(marker))
        by = {k: pool[stratum.loc[pool].values == k] for k in stratum.unique()}
        gst = stratum.loc[genes].values
        null = []
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for k in np.unique(gst):
                idx = np.where(gst == k)[0]
                src = by.get(k, pool)
                if len(src) == 0:
                    src = pool
                pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
            null.append(metrics(list(pick)))
        null = pd.DataFrame(null)
        null["cohort"] = name
        dist.append(null)
        row = {"cohort": name, "n_genes": len(genes),
               "specificity_mix": {int(k): int((gst % 4 == k).sum()) for k in range(4)}}
        for k, v in obs.items():
            nv = null[k].values
            more = (nv <= v).sum() if k == "neuron_rho" else (nv >= v).sum()  # tails fixed by hypothesis
            row[k] = {"neff": round(float(v), 3), "random_median": round(float(np.median(nv)), 3),
                      "random_q05_q95": [round(float(np.quantile(nv, .05)), 3), round(float(np.quantile(nv, .95)), 3)],
                      "p_empirical": round(float((more + 1) / (a.n + 1)), 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.concat(dist).to_csv(a.out / "null_specificity_matched_distributions.csv", index=False)
    (a.out / "null_specificity_matched_report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
