"""Reviewer revision (R2 M2): are the Neff classifier genes themselves cell-type marker genes?

Each of the 271 classifier genes is labelled by its expression across the 9 SEA-AD MTG neurotypical
cell types (CPM per nucleus, seaad_mtg_reference.csv.gz):
  log2 ratio = log2((max neuronal type + 1) / (max non-neuronal type + 1))
  neuron-enriched  ratio >  1      glia-enriched  ratio < -1      non-specific  |ratio| <= 1
  (sensitivity: threshold 0.5, and a stricter 'non-specific' band |ratio| <= 0.5)
The C-minus-A direction of each gene is the sign of its centred centroid (C minus A).

Then the C-minus-A score is recomputed in every cohort from (i) all genes, (ii) non-specific genes
only, (iii) all but neuron-enriched genes, (iv) all but cell-type-enriched genes at the 0.5 threshold,
and related to the neuron marker score (AD samples) and to diagnosis stage (all samples). Each
restricted set is compared with 1,000 random subsets of the same size drawn from the full classifier
(does removing marker-like genes weaken the axis more than removing random genes?).

Usage: python classifier_specificity.py --data /mnt/project-files/d1-data/raw \
         --ref /mnt/project-files/d1-data/seaad/seaad_mtg_reference.csv.gz --out ../results/classifier_specificity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import NEURON_MARKERS, SEED
from validate_cohorts import GROUP_ORDER, cohorts

NEURONAL = ["Exc", "Inh_SST", "Inh_PVALB", "Inh_other"]


def ca(x: np.ndarray, cent: np.ndarray) -> np.ndarray:
    r = corr_np(z_np(x), cent)
    return r[:, 2] - r[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]

    g = cls.index.intersection(ref.index)
    neu = ref.loc[g, NEURONAL].max(axis=1)
    gli = ref.loc[g, [c for c in ref.columns if c not in NEURONAL]].max(axis=1)
    ratio = np.log2((neu + 1) / (gli + 1))
    cent_all = cls.sub(cls.mean(axis=1), axis=0)
    direction = np.sign(cent_all["C"] - cent_all["A"]).map({1.0: "up_in_C", -1.0: "up_in_A", 0.0: "none"})
    lab = pd.Series("not_in_reference", index=cls.index)
    lab.loc[g] = np.where(ratio > 1, "neuron-enriched", np.where(ratio < -1, "glia-enriched", "non-specific"))
    tab = pd.crosstab(lab, direction.loc[lab.index])
    ann = pd.DataFrame({"class": lab, "direction": direction, "log2_neuron_vs_glia": ratio.reindex(cls.index)})
    ann.round(3).to_csv(a.out / "classifier_gene_specificity.csv")
    rep = {"table_specificity_by_direction": tab.to_dict(),
           "spearman_ratio_vs_C_minus_A_centroid": round(float(stats.spearmanr(
               ratio, (cent_all["C"] - cent_all["A"]).loc[g])[0]), 3)}
    print(tab)

    sets = {"all": ann.index,
            "non-specific only (|log2| <= 1)": ann.index[ann["class"] == "non-specific"],
            "non-specific strict (|log2| <= 0.5)": ann.index[ann["log2_neuron_vs_glia"].abs() <= 0.5],
            "without neuron-enriched (log2 > 1)": ann.index[ann["class"] != "neuron-enriched"],
            "without any enriched at 0.5": ann.index[ann["log2_neuron_vs_glia"].abs() <= 0.5]
                                            .union(ann.index[ann["class"] == "not_in_reference"])}
    rep["set_sizes"] = {k: int(len(v)) for k, v in sets.items()}
    rows = []
    for name, e, m in cohorts(a.data):
        ad = m.index[m["group"] == "AD"]
        stage = m["group"].map(GROUP_ORDER).values
        avail = cls.index.intersection(e.index)
        mk = [x for x in NEURON_MARKERS if x in e.index and x not in avail]
        neuron = z_np(e.loc[mk, ad].values).mean(0)
        full_ca = ca(e.loc[avail, ad].values, cent_all.loc[avail].values)
        for sname, gs in sets.items():
            gs = gs.intersection(e.index)
            c_ad = ca(e.loc[gs, ad].values, cent_all.loc[gs].values)
            c_all = ca(e.loc[gs, m.index].values, cent_all.loc[gs].values)
            row = {"cohort": name, "set": sname, "n_genes": len(gs),
                   "neuron_rho": stats.spearmanr(c_ad, neuron)[0],
                   "stage_rho": stats.spearmanr(c_all, stage)[0],
                   "rho_with_full_score": stats.spearmanr(c_ad, full_ca)[0]}
            if sname != "all" and len(gs) < len(avail):
                null = []
                for _ in range(a.n):
                    pick = rng.choice(avail, size=len(gs), replace=False)
                    null.append(stats.spearmanr(ca(e.loc[pick, ad].values, cent_all.loc[pick].values), neuron)[0])
                null = np.array(null)
                row["random_subset_neuron_rho_median"] = float(np.median(null))
                # one-sided: is the restricted set *less* neuron-coupled than random subsets of the same size?
                row["p_weaker_than_random_subset"] = float((np.sum(null >= row["neuron_rho"]) + 1) / (a.n + 1))
            rows.append(row)
        print(name, flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(a.out / "restricted_scores.csv", index=False)
    rep["restricted"] = df.to_dict(orient="records")
    (a.out / "classifier_specificity_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(df.pivot(index="cohort", columns="set", values="neuron_rho").round(3).to_string())


if __name__ == "__main__":
    main()
