"""Key regulators within cell type without circularity (v12, review round 10, Reviewer 1 minor 13).

All of Neff et al.'s key regulators tested in seaad_regulators_copath_v11.py are classifier genes, and each
type's nuclei contribute to the all-nuclei C-A score, so a within-type association could be partly
circular. Here the C-A score for each (gene, cell type) pair is recomputed from the all-nuclei pseudobulk
with (a) the tested gene removed from the classifier, and (b) also the tested type's nuclei removed from
the pseudobulk; the within-type log2(CPM + 1) of the gene is then related to it (age/sex partial Spearman),
next to the v11 value. BH-FDR over the (b) tests. Genes with mean within-type log2 CPM below 1 are flagged.

Usage: python seaad_regulators_loo_v12.py --donors /mnt/project-files/d1-data/seaad/donors --data /mnt/project-files/d1-data/raw \
         --res ../results --out ../results/seaad_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from braak_gse106241 import partial_spearman
from null_models import corr_np, z_np
from seaad_celltype_revision import load_sub
from seaad_pseudobulk import TYPES, cpm
from seaad_regulators_copath_v11 import BROAD, REG, multipletests


def ca_from(M, genes, cls, drop_gene):
    keep = M.mean(0) >= 1
    L = pd.DataFrame(np.log2(M[:, keep] + 1).T, index=genes[keep])
    L = L[~L.index.duplicated()]
    g = cls.index.intersection(L.index).drop(drop_gene, errors="ignore")
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    r = corr_np(z_np(L.loc[g].values), cent)
    return r[:, 2] - r[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes, meta, allsubs, U, N = load_sub(a.donors)
    age = pd.to_numeric(meta["Age at Death"].astype(str).str.replace("90+", "90", regex=False), errors="coerce").values
    sex = (meta["Sex"] == "Male").astype(float).values
    cov = np.column_stack([age, sex])
    tidx = {t: [allsubs.index(s) for s in v] for t, v in TYPES.items()}
    T = np.stack([U[:, tidx[t]].sum(1) for t in TYPES], 1)
    tl = list(TYPES)
    v11 = pd.read_csv(a.res / "seaad_v11/seaad_regulators_within_type.csv")
    v11 = v11[v11.vs == "CA_all_nuclei"].set_index(["gene", "cell_type"])["rho_age_sex"]
    gi = {g: i for i, g in enumerate(genes)}
    rows = []
    for t in BROAD:
        k = [allsubs.index(s) for s in TYPES[t]]
        prof = np.log2(cpm(U[:, k].sum(1)) + 1)
        others = [i for i, x in enumerate(tl) if x != t]
        for g in dict.fromkeys(x for v in REG.values() for x in v):
            if g not in gi:
                continue
            x = prof[:, gi[g]]
            ca_a = ca_from(cpm(T.sum(1)), genes, cls, g)
            ca_b = ca_from(cpm(T[:, others].sum(1)), genes, cls, g)
            ra, pa, _ = partial_spearman(x, ca_a, cov)
            rb, pb, _ = partial_spearman(x, ca_b, cov)
            rows.append({"gene": g, "cell_type": t, "mean_log2cpm": round(float(x.mean()), 2),
                         "low_expression": bool(x.mean() < 1), "rho_v11": v11.get((g, t), np.nan),
                         "rho_gene_removed": round(float(ra), 3), "p_gene_removed": float(f"{pa:.3g}"),
                         "rho_gene_and_type_removed": round(float(rb), 3), "p": float(f"{pb:.3g}")})
    d = pd.DataFrame(rows)
    d["q_bh"] = multipletests(d["p"].values)[1].round(4)
    d.to_csv(a.out / "seaad_regulators_loo_v12.csv", index=False)
    hit = d[d.q_bh < .05]
    summ = {"n_tests": int(len(d)), "n_q_lt_05": int(len(hit)),
            "hits": hit[["gene", "cell_type", "rho_v11", "rho_gene_and_type_removed", "q_bh", "low_expression"]].to_dict("records")}
    (a.out / "seaad_regulators_loo_v12.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
