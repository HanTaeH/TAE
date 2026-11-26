"""Li et al. (2022) hub-gene subgroups against cell-type-matched random genes (v12, field-wide framing).

other_scheme_li2022.py compared the subgroup scores with expression-matched random genes only. Here each
subgroup score (mean gene-wise z of its hub genes, AD samples of the four independent cohorts) is related
to the neuron score and compared with 1,000 random sets matched on expression decile and on SEA-AD
cell-type specificity class (as in other_schemes_clustered.py), two-sided on |rho|.

Usage: python li2022_specnull_v12.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out ../results/rq2_li2022
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from classifier_specificity import NEURONAL
from meta_reanalysis import INDEPENDENT
from other_scheme_li2022 import HUB
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1306)
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    allmark = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index].loc[:, (m["group"] == "AD").values]
        Z = zscore_rows(e)
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        strat = dec.astype(str) + "_" + spec.reindex(e.index).fillna(3).astype(int).astype(str)
        hub_all = {g for v in HUB.values() for g in v}
        neuron = Z.loc[[g for g in NEURON_MARKERS if g in Z.index and g not in hub_all]].mean().values
        pool = e.index.difference(list(hub_all | allmark))
        by = {k: pool[strat.loc[pool].values == k] for k in strat.unique()}
        for sg, gl in HUB.items():
            genes = [g for g in gl if g in Z.index]
            obs = float(stats.spearmanr(Z.loc[genes].mean().values, neuron)[0])
            nl = np.array([abs(stats.spearmanr(Z.loc[[rng.choice(by[k]) if len(by[k]) else rng.choice(pool)
                                                       for k in strat.loc[genes].values]].mean().values, neuron)[0])
                           for _ in range(a.n)])
            row = {"cohort": name, "n_AD": int(e.shape[1]), "subgroup": sg, "genes_found": f"{len(genes)}/{len(gl)}",
                   "spec_mix_glia_nonspec_neuron_absent": [int((strat.loc[genes].str[-1] == str(k)).sum()) for k in range(4)],
                   "rho_neuron": round(obs, 3), "spec_null_median_abs": round(float(np.median(nl)), 3),
                   "p_spec_ge": round(float(((nl >= abs(obs)).sum() + 1) / (a.n + 1)), 4),
                   "p_spec_le": round(float(((nl <= abs(obs)).sum() + 1) / (a.n + 1)), 4)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(a.out / "li2022_specnull_v12.csv", index=False)
    (a.out / "li2022_specnull_v12.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
