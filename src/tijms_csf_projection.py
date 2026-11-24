"""Living-patient CSF proteomic AD subtypes (Tijms et al., 2024, Nat Aging 4:33) projected onto bulk brain
(follow-up, 2026-11-24).

Supplementary Table 4 gives, per CSF protein, the mean level (scaled to controls) in each of five AD subtypes:
1 neuronal hyperplasticity, 2 innate immune activation, 3 RNA dysregulation, 4 choroid plexus dysfunction,
5 blood-brain barrier impairment. Proteins used for clustering ("Included for clustering" = 1) are mapped to
gene symbols; subtype centroids = the five subtype means, centred per gene across subtypes (as for Neff).
In the AD samples of each independent cohort, genes are z-scored and each sample is correlated (Pearson) with
each centroid, giving r(S1)..r(S5). Reported:
  - Spearman of each r(Sk) with the neuron marker score (markers in the protein set removed) and with the
    Neff C-A score;
  - nulls (300 sets): random genes inheriting the centroid rows, matched on expression decile, and on
    decile x SEA-AD specificity class; p two-sided on |rho|.
CSF proteins are not brain RNA: many are blood- or choroid-plexus-derived, so this tests only whether brain
samples vary along directions that separate living-patient subtypes.

Usage: python tijms_csf_projection.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out <dir>
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
from meta_reanalysis import INDEPENDENT, ca_score
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import cohorts

SUB = ["S1_hyperplasticity", "S2_innate_immune", "S3_RNA", "S4_choroid_plexus", "S5_BBB"]


def load_tijms(path: Path) -> pd.DataFrame:
    t = pd.read_excel(path, sheet_name="S table 4", header=None, skiprows=4)
    t = t[[3, 4, 5, 6, 7, 8, 22]]
    t.columns = ["gene"] + SUB + ["clustering"]
    t = t[t["clustering"] == 1].dropna(subset=["gene"])
    for c in SUB:
        t[c] = t[c].astype(str).str.extract(r"^\s*(-?[\d.]+)")[0].astype(float)
    t = t.groupby("gene")[SUB].mean()
    return t.sub(t.mean(axis=1), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 303)
    cent_all = load_tijms(a.data / "tijms2024" / "43587_2023_550_MOESM3_ESM.xlsx")
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index)
    allmark = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    out = {"n_clustering_proteins": int(len(cent_all)),
           "centroid_spec_mix_glia_nonspec_neuron_absent": [int((spec.reindex(cent_all.index).fillna(3) == k).sum()) for k in (0, 1, 2, 3)],
           "cohorts": []}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        e = e[m.index].loc[:, (m["group"] == "AD").values]
        Z = zscore_rows(e)
        genes = cent_all.index.intersection(Z.index)
        cent = cent_all.loc[genes].values
        mk = [g for g in NEURON_MARKERS if g in Z.index and g not in genes and g not in cls.index]
        neuron = Z.loc[mk].mean().values
        ng = cls.index.intersection(Z.index)
        neff_ca = ca_score(Z.loc[ng], cls.loc[ng].sub(cls.loc[ng].mean(axis=1), axis=0)).values
        r = corr_np(z_np(Z.loc[genes].values), cent)
        dec = pd.qcut(e.mean(axis=1).rank(method="first"), 10, labels=False)
        sp = spec.reindex(e.index).fillna(3).astype(int)
        pool = e.index.difference(genes).difference(list(allmark)).difference(cls.index)
        nulls = {}
        for kind, st in [("expr", dec.astype(str)), ("spec", dec.astype(str) + "_" + sp.astype(str))]:
            by = {k: pool[st.loc[pool].values == k] for k in st.loc[genes].unique()}
            gst = st.loc[genes].values
            draws = []
            for _ in range(a.n):
                pick = np.empty(len(genes), dtype=object)
                for k in np.unique(gst):
                    idx = np.where(gst == k)[0]
                    src = by[k] if len(by[k]) else pool
                    pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
                rr = corr_np(z_np(Z.loc[list(pick)].values), cent)
                draws.append([abs(stats.spearmanr(rr[:, j], neuron)[0]) for j in range(len(SUB))])
            nulls[kind] = np.array(draws)
        row = {"cohort": name, "n_AD": int(e.shape[1]), "genes_found": int(len(genes)), "subtypes": {}}
        for j, s in enumerate(SUB):
            rn = stats.spearmanr(r[:, j], neuron)[0]
            row["subtypes"][s] = {
                "rho_neuron": round(float(rn), 3), "rho_neff_CA": round(float(stats.spearmanr(r[:, j], neff_ca)[0]), 3),
                "null_expr_median_abs": round(float(np.median(nulls["expr"][:, j])), 3),
                "p_expr": round(float(((nulls["expr"][:, j] >= abs(rn)).sum() + 1) / (a.n + 1)), 4),
                "null_spec_median_abs": round(float(np.median(nulls["spec"][:, j])), 3),
                "p_spec": round(float(((nulls["spec"][:, j] >= abs(rn)).sum() + 1) / (a.n + 1)), 4)}
        out["cohorts"].append(row)
        print(json.dumps(row), flush=True)
    (a.out / "tijms_csf_projection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "cohorts"}))


if __name__ == "__main__":
    main()
