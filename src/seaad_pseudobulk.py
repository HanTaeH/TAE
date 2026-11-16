"""Option 2: Neff classes on SEA-AD MTG donor pseudobulk (84 donors) with measured nucleus counts.

Input: per-donor subclass UMI sums from seaad_donor_aggregate.py, donor metadata, and the SEA-AD
continuous pseudo-progression score (CPS, MTG rows of Global_and_Local_CPS.20260105.csv).

Nuclei were FANS-sorted at a planned 70% NeuN+ / 30% NeuN-, so the neuron : non-neuron ratio of the
snRNA data is fixed by design; tissue neuron loss is invisible in it, but shifts within neurons
(e.g. L2/3 IT, Sst) and within glia are measured.

Pseudobulks (CPM, log2(CPM + 1), genes with mean CPM >= 1):
  sorted       sum of all nuclei as sequenced
  fixed_comp   per-type CPM profiles of the donor mixed with the same weights for every donor
               (simulation BASE proportions x RNA content), so only within-type expression varies
  comp_only    cohort-average type profiles mixed with the donor's own nucleus proportions
               (rescaled to the BASE neuron share), so only within-class composition varies
  per type     each broad type alone (Exc, Inh, Astro, Oligo, Micro): within-type state
For each: Neff projection across donors, C-minus-A and B scores vs CPS, Braak, Thal, CERAD, ADNC,
cognitive status (Spearman; partial for age and sex), plus a decile-matched random-gene null for
CPS (500 sets). Measured proportions (within neurons / within non-neurons) vs the scores.
Sensitivity: excluding neurotypical reference donors if any (the 2026 donor objects hold none, so the
84 donors do not overlap the 2022 reference used for deconvolution).

Usage: python seaad_pseudobulk.py --donors /mnt/project-files/d1-data/seaad/donors \
         --cps /mnt/project-files/d1-data/seaad/Global_and_Local_CPS.20260105.csv \
         --data /mnt/project-files/d1-data/raw --out ../results/seaad_pseudobulk --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from braak_gse106241 import partial_spearman
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

TYPES = {
    "Exc": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
    "Inh": ["Sst", "Sst Chodl", "Pvalb", "Chandelier", "Lamp5", "Lamp5 Lhx6", "Sncg", "Vip", "Pax6"],
    "Oligo": ["Oligodendrocyte"], "Astro": ["Astrocyte"], "OPC": ["OPC"],
    "Micro": ["Microglia-PVM", "Immune"], "Vascular": ["Endothelial", "VLMC", "VLMC & Perivascular", "Pericyte"],
}
BASE = {"Exc": .30, "Inh": .10, "Oligo": .30, "Astro": .15, "OPC": .06, "Micro": .06, "Vascular": .03}
RNA = {"Exc": 2, "Inh": 2}
NEURON = ["Exc", "Inh"]
CLS = np.array(list("ABC"))
ORD = {"Braak": {f"Braak {r}": i for i, r in enumerate(["0", "I", "II", "III", "IV", "V", "VI"])},
       "Thal": {f"Thal {i}": i for i in range(6)},
       "CERAD score": {"Absent": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3},
       "Overall AD neuropathological Change": {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3},
       "Cognitive Status": {"No dementia": 0, "Dementia": 1}}


def load(donor_dir: Path):
    genes = np.array((donor_dir / "genes.txt").read_text().split("\n"))
    meta = pd.read_csv(donor_dir / "donor_meta.csv").set_index("Donor ID")
    sub_of = {s: t for t, v in TYPES.items() for s in v}
    type_umi, sub_n = {}, {}
    for d in meta.index:
        z = np.load(donor_dir / f"{d}.npz", allow_pickle=True)
        subs = z["subclasses"].astype(str)
        unknown = set(subs) - set(sub_of)
        if unknown:
            raise ValueError(f"unmapped subclasses {unknown}")
        sub_n[d] = pd.Series(z["n_nuclei"], index=subs)
        tu = pd.DataFrame(z["umis"], index=subs).groupby([sub_of[s] for s in subs]).sum()
        type_umi[d] = tu.reindex(list(TYPES)).fillna(0)
    sub_n = pd.DataFrame(sub_n).T.fillna(0)
    return genes, meta, type_umi, sub_n


def cpm(m):
    s = m.sum(axis=-1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m, dtype=float), where=s > 0) * 1e6


def project(L: pd.DataFrame, cls: pd.DataFrame):
    genes = cls.index.intersection(L.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    r = corr_np(z_np(L.loc[genes].values), cent)
    return r[:, 2] - r[:, 0], r[:, 1] - (r[:, 0] + r[:, 2]) / 2, CLS[r.argmax(1)], genes, cent


def marker_score(L, lst, excl):
    lst = [g for g in lst if g in L.index and g not in excl]
    return z_np(L.loc[lst].values).mean(0)


def fmt(t):
    return {"rho": round(t[0], 3), "p": float(f"{t[1]:.3g}"), "n": t[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--cps", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    genes, meta, type_umi, sub_n = load(a.donors)
    donors = list(meta.index)
    cps = pd.read_csv(a.cps)
    cps = cps[cps["Brain Region"] == "MTG"].set_index("Donor ID")
    meta = meta.join(cps[["CPS_Global", "CPS_Local"]])
    for c, m in ORD.items():
        meta[c + "_num"] = meta[c].map(m)
    meta["female"] = (meta["Sex"] == "Female").astype(float)
    cov = meta[["Age at Death", "female"]].apply(pd.to_numeric, errors="coerce").replace({"90+": 90}).values.astype(float)

    T = np.stack([type_umi[d].values for d in donors])          # donor x type x gene (UMI)
    n_type = sub_n.T.groupby({s: t for t, v in TYPES.items() for s in v}).sum().T.reindex(columns=list(TYPES)).fillna(0)
    n_type = n_type.loc[donors]
    prof = cpm(T)                                                # per-type CPM
    w = np.array([BASE[t] * RNA.get(t, 1) for t in TYPES])
    w = w / w.sum()
    # fixed composition: missing types in a donor take the cohort-average profile
    avg = prof.mean(0)
    prof_f = np.where((n_type.values >= 10)[:, :, None], prof, avg[None])
    pb = {"sorted": cpm(T.sum(1)),
          "fixed_comp": cpm(np.einsum("t,dtg->dg", w, prof_f))}
    frac = n_type.div(n_type.sum(axis=1), axis=0).values
    neu = frac[:, :2].sum(1, keepdims=True)
    base_neu = sum(BASE[t] for t in NEURON)
    tissue_like = np.hstack([frac[:, :2] / neu * base_neu, frac[:, 2:] / (1 - neu) * (1 - base_neu)])
    rna = np.array([RNA.get(t, 1) for t in TYPES])
    pb["comp_only"] = cpm(np.einsum("dt,tg->dg", tissue_like * rna, avg))
    for t in ["Exc", "Inh", "Astro", "Oligo", "Micro"]:
        pb[f"type_{t}"] = prof[:, list(TYPES).index(t)]

    # measured proportions
    within = pd.DataFrame(index=donors)
    nn = n_type[NEURON].sum(axis=1)
    sub_frac_neu = sub_n.div(nn, axis=0)
    for s in ["L2/3 IT", "L4 IT", "Sst", "Pvalb", "Vip"]:
        within[f"{s}_of_neurons"] = sub_frac_neu.get(s, 0)
    within["Exc_of_neurons"] = n_type["Exc"] / nn
    glia = n_type.drop(columns=NEURON).sum(axis=1)
    for t in ["Astro", "Oligo", "Micro", "OPC"]:
        within[f"{t}_of_nonneurons"] = n_type[t] / glia
    within["neuron_nuclei_frac"] = nn / n_type.sum(axis=1)
    within["NeuN_FANS"] = pd.to_numeric(meta["NeuN positive fraction on FANS"], errors="coerce")

    outcomes = {"CPS": meta["CPS_Global"].values, "Braak": meta["Braak_num"].values,
                "Thal": meta["Thal_num"].values, "CERAD": meta["CERAD score_num"].values,
                "ADNC": meta["Overall AD neuropathological Change_num"].values,
                "dementia": meta["Cognitive Status_num"].values}
    ref_mask = meta["Neurotypical reference"].astype(str).str.lower().isin(["true", "1"]).values
    rep: dict = {"n_donors": len(donors), "n_reference_donors": int(ref_mask.sum()),
                 "nuclei_total": int(n_type.values.sum()),
                 "facs_plan_counts": meta["facs_population_plan"].value_counts().to_dict(),
                 "NeuN_FANS_unique": sorted(within["NeuN_FANS"].dropna().unique().round(3).tolist())[:20],
                 "neuron_nuclei_frac_range": [round(float(within["neuron_nuclei_frac"].min()), 3),
                                              round(float(within["neuron_nuclei_frac"].max()), 3)]}
    per = meta[["CPS_Global", "Braak", "Cognitive Status", "Neurotypical reference"]].copy().join(within)

    E_sorted = None
    for name, M in pb.items():
        keep = M.mean(0) >= 1
        L = pd.DataFrame(np.log2(M[:, keep] + 1).T, index=genes[keep], columns=donors)
        L = L[~L.index.duplicated()]
        if name == "sorted":
            E_sorted = L
        ca, b, lab, ng, cent = project(L, cls)
        neuron = marker_score(L, NEURON_MARKERS, set(ng))
        res = {"n_genes": int(len(L)), "n_neff_genes": int(len(ng)),
               "class_frac": {k: round(float((lab == k).mean()), 3) for k in "ABC"},
               "ca_vs_neuron_score": fmt(partial_spearman(ca, neuron))}
        for o, y in outcomes.items():
            res[f"ca_vs_{o}"] = fmt(partial_spearman(ca, y))
            res[f"ca_vs_{o}_partial_age_sex"] = fmt(partial_spearman(ca, y, cov))
            res[f"b_vs_{o}"] = fmt(partial_spearman(b, y))
        res["ca_vs_CPS_partial_neuron"] = fmt(partial_spearman(ca, outcomes["CPS"], np.column_stack([cov, neuron])))
        if ref_mask.any():
            res["ca_vs_CPS_excl_reference"] = fmt(partial_spearman(ca[~ref_mask], outcomes["CPS"][~ref_mask]))
        if name in ("sorted", "fixed_comp", "comp_only"):
            res["ca_vs_measured"] = {c: fmt(partial_spearman(ca, within[c].values)) for c in within.columns
                                     if within[c].nunique() > 2}
            res["class_vs_CPS_kruskal_p"] = float(stats.kruskal(*[outcomes["CPS"][lab == k] for k in "ABC"
                                                                  if (lab == k).sum() > 2]).pvalue)
            # random-gene null for CPS (decile x SD tercile, inherits centroid rows)
            dec = pd.qcut(L.mean(axis=1).rank(method="first"), 10, labels=False)
            terc = L.std(axis=1).groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
            strat = dec * 3 + terc
            markers = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
            pool = L.index.difference(ng).difference(list(markers))
            pool_by = {s: pool[strat.loc[pool].values == s] for s in range(30)}
            ns = strat.loc[ng].values
            obs = partial_spearman(ca, outcomes["CPS"])[0]
            null = []
            for _ in range(a.n):
                pick = np.empty(len(ng), dtype=object)
                for s in np.unique(ns):
                    idx = np.where(ns == s)[0]
                    pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=False)
                r = corr_np(z_np(L.loc[list(pick)].values), cent)
                null.append(partial_spearman(r[:, 2] - r[:, 0], outcomes["CPS"])[0])
            null = np.array(null)
            res["ca_vs_CPS_random_null"] = {"obs": round(obs, 3), "random_median": round(float(np.median(null)), 3),
                                            "random_q95": round(float(np.quantile(null, .95)), 3),
                                            "p": round(float((np.sum(null >= obs) + 1) / (a.n + 1)), 4)}
        rep[name] = res
        per[f"ca_{name}"] = ca
        per[f"class_{name}"] = lab
    # how the sorted score splits into within-type expression vs composition
    X = np.column_stack([np.ones(len(donors)), per["ca_fixed_comp"], per["ca_comp_only"]])
    y = per["ca_sorted"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rep["sorted_on_fixed_and_comp"] = {"r2": round(float(1 - np.var(y - X @ beta) / np.var(y)), 3),
                                       "r2_fixed_only": round(float(stats.pearsonr(y, X[:, 1])[0] ** 2), 3),
                                       "r2_comp_only": round(float(stats.pearsonr(y, X[:, 2])[0] ** 2), 3)}
    rep["measured_vs_CPS"] = {c: fmt(partial_spearman(within[c].values, outcomes["CPS"])) for c in within.columns
                              if within[c].nunique() > 2}
    (a.out / "seaad_pseudobulk_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    per.to_csv(a.out / "seaad_pseudobulk_per_donor.csv")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))

    names = ["sorted", "fixed_comp", "comp_only", "type_Exc", "type_Inh", "type_Astro", "type_Oligo", "type_Micro"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, nm in zip(axes[:2], ["sorted", "fixed_comp"]):
        ax.scatter(per["CPS_Global"], per[f"ca_{nm}"], c=np.where(ref_mask, "#999999", "#1f4e79"), s=18)
        r = rep[nm]["ca_vs_CPS"]
        ax.set_title(f"{nm}: C-minus-A vs CPS\nρ = {r['rho']}, p = {r['p']}", fontsize=10)
        ax.set_xlabel("SEA-AD continuous pseudo-progression score (MTG)")
        ax.set_ylabel("C-minus-A score")
    ax = axes[2]
    rh = [rep[nm]["ca_vs_CPS"]["rho"] for nm in names]
    ax.barh(names[::-1], rh[::-1], color="#1f4e79")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Spearman ρ, C-minus-A vs CPS")
    ax.set_title("Which pseudobulk carries the disease link?", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "seaad_pseudobulk_summary.png", dpi=150)


if __name__ == "__main__":
    main()
