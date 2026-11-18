"""Reviewer revision (R2 M8, M10).

A. Deconvolution accuracy: NNLS with the SEA-AD MTG neurotypical reference (as in deconv_seaad.py)
   applied to the 84 SEA-AD donor pseudobulks (sorted nuclei, no overlap with the reference donors),
   where the true type shares are known (UMI share and nucleus share per type).
   Reported per type: Pearson / Spearman r across donors, mean bias, and the same for the
   neuron total (Exc + Inh).
B. MSBB across-region analysis with subject-block resampling (17 regions come from the same subjects):
   - bootstrap over subjects (2,000) for the across-region rho between the neuron-CDR and the
     C-minus-A-CDR associations;
   - a null that shuffles CDR across subjects (same shuffle in every region): how strong is the
     across-region rho when CDR carries no information? (tests whether the result follows from
     C-minus-A ~ -neuron within region)

Usage: python deconv_validation_blocks.py --seaad-dir /mnt/project-files/d1-data/seaad \
         --res /mnt/project-files/d1-results --out ../results/deconv_validation_blocks --n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from deconv_seaad import est_nnls, signature_genes
from pilot_gse33000 import SEED
from seaad_celltype_revision import load_sub

MAP = {"Exc": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
       "Inh_SST": ["Sst", "Sst Chodl"], "Inh_PVALB": ["Pvalb", "Chandelier"],
       "Inh_other": ["Lamp5", "Lamp5 Lhx6", "Sncg", "Vip", "Pax6"],
       "Astro": ["Astrocyte"], "Oligo": ["Oligodendrocyte"], "OPC": ["OPC"],
       "Micro": ["Microglia-PVM", "Immune"],
       "Vascular": ["Endothelial", "VLMC", "VLMC & Perivascular", "Pericyte"]}


def part_a(seaad: Path) -> dict:
    ref = pd.read_csv(seaad / "seaad_mtg_reference.csv.gz", index_col=0)
    ref = ref[~ref.index.duplicated()]
    genes, meta, allsubs, U, N = load_sub(seaad / "donors")
    bulk = U.sum(1)                                              # donor x gene UMI
    bulk = bulk / bulk.sum(1, keepdims=True) * 1e6
    B = pd.DataFrame(bulk.T, index=genes)
    B = B[~B.index.duplicated()]
    sig = signature_genes(ref, set(), B.index)
    P = est_nnls(B.loc[sig].values, ref.loc[sig].values)
    est = pd.DataFrame(P, index=meta.index, columns=ref.columns)
    umi_t = pd.DataFrame({t: U[:, [allsubs.index(s) for s in v if s in allsubs]].sum((1, 2)) for t, v in MAP.items()},
                         index=meta.index)
    umi_t = umi_t.div(umi_t.sum(axis=1), axis=0)
    nuc_t = pd.DataFrame({t: N[[s for s in v if s in N.columns]].sum(axis=1) for t, v in MAP.items()})
    nuc_t = nuc_t.div(nuc_t.sum(axis=1), axis=0)
    out = {"n_donors": int(len(meta)), "n_signature_genes": len(sig), "per_type": {}}
    neu = ["Exc", "Inh_SST", "Inh_PVALB", "Inh_other"]
    est["Neuron_total"] = est[neu].sum(axis=1)
    umi_t["Neuron_total"] = umi_t[neu].sum(axis=1)
    nuc_t["Neuron_total"] = nuc_t[neu].sum(axis=1)
    for t in list(MAP) + ["Neuron_total"]:
        e, u, n = est[t].values, umi_t[t].values, nuc_t[t].values
        out["per_type"][t] = {"est_mean": round(float(e.mean()), 3), "true_umi_mean": round(float(u.mean()), 3),
                              "true_nuclei_mean": round(float(n.mean()), 3),
                              "pearson_vs_umi": round(float(stats.pearsonr(e, u)[0]), 3),
                              "spearman_vs_umi": round(float(stats.spearmanr(e, u)[0]), 3),
                              "spearman_vs_nuclei": round(float(stats.spearmanr(e, n)[0]), 3)}
    return out


def part_b(res: Path, n: int, rng) -> dict:
    m = pd.read_csv(res / "msbb/msbb_per_sample.csv", index_col=0)
    regions = sorted(m["region"].unique())
    subjects = m["subject"].unique()
    by = {r: g.set_index("subject") for r, g in m.groupby("region")}

    def across(cdr_map=None, subj_draw=None):
        nc, cc = [], []
        for r in regions:
            g = by[r]
            if subj_draw is not None:
                g = g.reindex([s for s in subj_draw if s in g.index])
            cdr = g["cdr"].values if cdr_map is None else g.index.map(cdr_map).values.astype(float)
            nc.append(stats.spearmanr(g["neuron"].values, cdr)[0])
            cc.append(stats.spearmanr(g["ca"].values, cdr)[0])
        return stats.spearmanr(nc, cc)[0]

    obs = across()
    boot = [across(subj_draw=rng.choice(subjects, len(subjects), replace=True)) for _ in range(n)]
    cdr_subj = m.groupby("subject")["cdr"].first()
    null = []
    for _ in range(n // 4):
        sh = pd.Series(rng.permutation(cdr_subj.values), index=cdr_subj.index)
        null.append(across(cdr_map=sh))
    null = np.array(null)
    return {"observed_rho": round(float(obs), 3),
            "subject_bootstrap_ci95": [round(float(np.nanquantile(boot, .025)), 3), round(float(np.nanquantile(boot, .975)), 3)],
            "n_boot": n,
            "cdr_shuffled_null_median": round(float(np.median(null)), 3),
            "cdr_shuffled_null_q05_q95": [round(float(np.quantile(null, .05)), 3), round(float(np.quantile(null, .95)), 3)],
            "p_obs_more_negative_than_null": round(float((np.sum(null <= obs) + 1) / (len(null) + 1)), 4),
            "n_null": int(len(null))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seaad-dir", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rep = {"B_msbb_across_region_blocks": part_b(a.res, a.n, rng)}
    print(json.dumps(rep, indent=1), flush=True)
    rep["A_nnls_validation_seaad"] = part_a(a.seaad_dir)
    (a.out / "deconv_validation_blocks_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["A_nnls_validation_seaad"], indent=1))


if __name__ == "__main__":
    main()
