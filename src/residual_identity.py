"""What is the signal left after cell-composition adjustment? (priority-2 E)

Residual C-minus-A score: Neff genes (AD samples, z-scored) residualised on the five marker scores
(neuron, astrocyte, microglia, oligodendrocyte, endothelial), then projected.

1 Gene sets  every measured gene is z-scored and residualised the same way, and correlated with the
  residual score; a set's score is the mean gene correlation, compared with 1000 random sets of
  equal size (z and empirical two-sided p). Sets: curated by gene name (oxidative phosphorylation,
  ribosome, immediate-early, heat shock, interferon, microglial activation / complement, reactive
  astrocyte, hypoxia) and SEA-AD MTG subclass-specific genes (24 subclasses, top 50 each).
  Neff genes and marker genes are removed from every set.
2 Technical  residual score vs PMI, pH (MSBB GSE84422, AD samples in each of 17 regions), RNA quality
  (GSE106241), age and sex (all cohorts).
3 Finer composition  GSE125583 (RNA-seq): Neff signal and C-minus-A R^2 after adjusting for marker
  scores, 9-type and 24-subclass SEA-AD NNLS proportions.

Usage: python residual_identity.py --data /mnt/project-files/d1-data/raw \
         --ref-dir /mnt/project-files/d1-data/seaad --out ../results/residual_identity --n 1000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_gse106241
import load_gse84422
import load_gse125583
import neff_subtypes
from deconv_seaad import clr, est_nnls, signature_genes
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from residual_structure import r2, residualise, signal
from validate_cohorts import cohorts

PATTERNS = {"oxphos": r"^(NDUF[ABSCV]\d|COX[4-8][A-C]?\d?|UQCR|ATP5|SDH[ABCD]$)", "ribosome": r"^RP[LS]\d+[AXY]?$"}
CURATED = {
    "immediate_early": ["FOS", "FOSB", "JUN", "JUNB", "EGR1", "EGR2", "EGR3", "ARC", "NR4A1", "NR4A2", "NR4A3",
                        "IER2", "IER3", "DUSP1", "ATF3", "NPAS4"],
    "heat_shock": ["HSPA1A", "HSPA1B", "HSPA4L", "HSPA6", "HSPB1", "HSPH1", "HSPD1", "HSPE1", "HSP90AA1", "DNAJB1", "BAG3",
                   "CRYAB", "SERPINH1"],
    "interferon": ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2", "IFI6", "IFI44L", "STAT1", "IRF7", "RSAD2", "IFITM3"],
    "microglia_activation_complement": ["HLA-DRA", "HLA-DRB1", "CD74", "C1QB", "C1QC", "C3", "CD14", "TREM2", "SPP1", "APOE",
                                        "LPL", "CD68", "ITGAX"],
    "reactive_astrocyte": ["VIM", "CD44", "SERPINA3", "CHI3L1", "OSMR", "S100B", "MAOB", "C4A", "C4B", "LCN2", "STEAP4"],
    "hypoxia": ["VEGFA", "ADM", "SLC2A1", "PGK1", "LDHA", "BNIP3", "CA9", "PDK1", "ANKRD37", "EGLN3"],
}
MARKERS = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}


def cell_scores(expr, idx, exclude):
    def sc(lst):
        lst = [g for g in lst if g in expr.index and g not in exclude]
        return z_np(expr.loc[lst, idx].values).mean(0)
    return np.column_stack([sc(NEURON_MARKERS)] + [sc(v) for v in CELL_MARKERS.values()])


def residual_ca(expr, idx, genes, cent, cells):
    z = z_np(expr.loc[genes, idx].values)
    r = corr_np(z_np(residualise(z, cells)), cent)
    return r[:, 2] - r[:, 0]


def subclass_sets(ref: pd.DataFrame, exclude: set) -> dict:
    ref = ref.loc[ref.index.difference(list(exclude))]
    out = {}
    for t in ref.columns:
        fold = (ref[t] + 1) / (ref.drop(columns=t).max(axis=1) + 1)
        ok = (ref[t] >= 5) & (fold >= 2)
        out[f"sub_{t}"] = fold[ok].sort_values(ascending=False).index[:50].tolist()
    return out


def set_scores(gc: pd.Series, sets: dict, n: int, rng) -> dict:
    res = {}
    allg = gc.index.values
    for k, v in sets.items():
        v = [g for g in v if g in gc.index]
        if len(v) < 5:
            continue
        obs = float(gc.loc[v].mean())
        null = np.array([gc.values[rng.choice(len(allg), len(v), replace=False)].mean() for _ in range(n)])
        res[k] = {"n_genes": len(v), "mean_r": round(obs, 3),
                  "z": round(float((obs - null.mean()) / null.std()), 2),
                  "p": round(float(min(1, 2 * min((null >= obs).mean(), (null <= obs).mean()) + 1 / n)), 4)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    sub_ref = pd.read_csv(a.ref_dir / "seaad_mtg_subclass_reference.csv.gz", index_col=0)
    sub_ref = sub_ref[~sub_ref.index.duplicated()]
    rep: dict = {"gene_sets": {}, "technical": {}}

    # 1 gene sets + age/sex, per cohort
    for name, expr, meta in cohorts(a.data):
        if "Cerebellum" in name:
            continue
        expr = expr.loc[expr.index.notna() & ~expr.index.duplicated()].dropna()
        ad = meta.index[meta["group"] == "AD"]
        genes = cls.index.intersection(expr.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
        cells = cell_scores(expr, ad, set(genes))
        rca = residual_ca(expr, ad, genes, cent, cells)
        keep = expr.index.difference(genes).difference(list(MARKERS))
        Z = residualise(z_np(expr.loc[keep, ad].values), cells)
        rz = (rca - rca.mean()) / rca.std()
        Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
        gc = pd.Series(Zs @ rz / len(rz), index=keep)
        sets = {k: [g for g in keep if re.match(p, g)] for k, p in PATTERNS.items()}
        sets.update(CURATED)
        sets.update(subclass_sets(sub_ref, set(genes) | MARKERS))
        rep["gene_sets"][name] = set_scores(gc, sets, a.n, rng)
        m = meta.loc[ad]
        rep["technical"][name] = {"age": round(float(stats.spearmanr(rca, m["age"].astype(float), nan_policy="omit")[0]), 3),
                                  "female_mwu_p": float(f"{stats.mannwhitneyu(rca[(m['sex'] == 'female').values], rca[(m['sex'] != 'female').values])[1]:.3g}")}
        gc.sort_values().to_csv(a.out / f"gene_corr_{name}.csv")
        print(name, "done", flush=True)

    # 2 technical: MSBB PMI / pH, GSE106241 RNA quality
    e, m = load_gse84422.load(a.data)
    genes = cls.index.intersection(e.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    rows = []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        cells = cell_scores(e, mm.index, set(genes))
        rca = residual_ca(e, mm.index, genes, cent, cells)
        z = z_np(e.loc[genes, mm.index].values)
        r = corr_np(z, cent)
        ca = r[:, 2] - r[:, 0]
        rows.append({"region": reg, "n": len(mm), **{f"{v}_raw": stats.spearmanr(ca, mm[v])[0] for v in ["pmi", "ph", "age"]},
                     **{f"{v}_resid": stats.spearmanr(rca, mm[v])[0] for v in ["pmi", "ph", "age"]}})
    t = pd.DataFrame(rows).set_index("region")
    t.round(3).to_csv(a.out / "msbb_technical_by_region.csv")
    rep["technical"]["MSBB_17_regions_median"] = t.median().round(3).to_dict()
    rep["technical"]["MSBB_regions_p05_resid_ph"] = int(sum(
        stats.spearmanr(residual_ca(e, mm.index, genes, cent, cell_scores(e, mm.index, set(genes))), mm["ph"])[1] < .05
        for _, mm in m[m["group"] == "AD"].groupby("region")))
    e, m = load_gse106241.load(a.data)
    genes = cls.index.intersection(e.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    cells = cell_scores(e, m.index, set(genes))
    rca = residual_ca(e, m.index, genes, cent, cells)
    rep["technical"]["GSE106241_rna_quality"] = {"rho": round(float(stats.spearmanr(rca, m["rna_quality"], nan_policy="omit")[0]), 3),
                                                 "p": float(f"{stats.spearmanr(rca, m['rna_quality'], nan_policy='omit')[1]:.3g}")}

    # 3 finer composition, GSE125583
    e, m = load_gse125583.load(a.data)
    ad = m.index[m["group"] == "AD"]
    genes = cls.index.intersection(e.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    lin = (2 ** e - 1).clip(lower=0)
    rank = e.mean(axis=1).rank(pct=True)
    present = rank.index[rank >= 0.5]
    ref9 = pd.read_csv(a.ref_dir / "seaad_mtg_reference.csv.gz", index_col=0)
    ref9 = ref9[~ref9.index.duplicated()]
    covs = {"markers5": cell_scores(e, ad, set(genes))}
    for nm, ref in [("nnls9", ref9), ("nnls24", sub_ref)]:
        sig = signature_genes(ref, set(cls.index), present)
        P = est_nnls(lin.loc[sig, ad].values, ref.loc[sig].values)
        covs[nm] = clr(P)
        rep.setdefault("finer_composition_n_sig", {})[nm] = len(sig)
    covs["markers5+nnls24"] = np.column_stack([covs["markers5"], covs["nnls24"]])
    z = z_np(e.loc[genes, ad].values)
    r = corr_np(z, cent)
    ca = r[:, 2] - r[:, 0]
    rep["finer_composition_GSE125583"] = {k: {"r2_ca": round(r2(ca, v), 3), "sig_raw": round(signal(z, cent, v)[0], 3),
                                              "sig_adj": round(signal(z, cent, v)[1], 3), "n_cov": int(v.shape[1])}
                                          for k, v in covs.items()}
    (a.out / "residual_identity_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    # compact table of gene-set z across cohorts
    tab = pd.DataFrame({c: {k: v["z"] for k, v in d.items()} for c, d in rep["gene_sets"].items()})
    tab.to_csv(a.out / "gene_set_z_by_cohort.csv")
    print(tab.round(1).to_string())
    print(json.dumps({k: rep[k] for k in ["technical", "finer_composition_GSE125583"]}, indent=1))


if __name__ == "__main__":
    main()
