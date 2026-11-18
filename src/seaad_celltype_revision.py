"""Reviewer revision: neuron number vs per-cell expression, and laminar sampling, in SEA-AD MTG.

Reviewer 1 (major 1-2) and reviewer 2 (M9): bulk marker scores cannot tell "fewer neurons" from
"lower expression per neuron"; the SEA-AD sorted pseudobulk is not at a fixed neuron share; L2/3 IT
and L4 IT shares move in opposite directions, which may indicate laminar dissection depth.

For 84 donors (per-donor subclass UMI sums from seaad_donor_aggregate.py):
  A. Pseudobulks that hold composition fixed and keep only within-cell-type expression:
       type_<broad> (Exc, Inh, Astro, Oligo, Micro) and sub_<subclass> (L2/3 IT, L4 IT, L5 IT, L6 IT)
     plus sorted / fixed_comp / comp_only as in seaad_pseudobulk.py.
     For each: C-minus-A vs CPS, Braak, dementia (Spearman, and partial for age + sex), and a random
     gene-set null (mean decile x SD tercile, 1,000 sets, centroid rows inherited) for CPS and Braak.
  B. Laminar sampling: share of each excitatory subclass among excitatory neurons vs the sorted
     C-minus-A score and vs CPS; superficial (L2/3 IT + L4 IT) : deep (L5/L6) ratio; partial
     C-minus-A ~ CPS after adjusting for neuron nuclei fraction, L2/3 IT share, L4 IT share,
     superficial:deep ratio, and all excitatory subclass shares (centred log-ratio).

Usage: python seaad_celltype_revision.py --donors /mnt/project-files/d1-data/seaad/donors \
         --cps /mnt/project-files/d1-data/seaad/Global_and_Local_CPS.20260105.csv \
         --data /mnt/project-files/d1-data/raw --out ../results/seaad_celltype_revision --n 1000
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
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED
from seaad_pseudobulk import BASE, NEURON, ORD, RNA, TYPES, cpm, fmt, project

EXC = TYPES["Exc"]
SUPERFICIAL = ["L2/3 IT", "L4 IT"]
DEEP = ["L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"]
SUBS = ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT"]


def load_sub(donor_dir: Path):
    genes = np.array((donor_dir / "genes.txt").read_text().split("\n"))
    meta = pd.read_csv(donor_dir / "donor_meta.csv").set_index("Donor ID")
    allsubs = [s for v in TYPES.values() for s in v]
    U = np.zeros((len(meta), len(allsubs), len(genes)), dtype=np.float64)
    N = pd.DataFrame(0.0, index=meta.index, columns=allsubs)
    for i, d in enumerate(meta.index):
        z = np.load(donor_dir / f"{d}.npz", allow_pickle=True)
        subs = z["subclasses"].astype(str)
        for j, s in enumerate(subs):
            k = allsubs.index(s)
            U[i, k] += z["umis"][j]
            N.loc[d, s] += z["n_nuclei"][j]
    return genes, meta, allsubs, U, N


def null_test(L, ng, cent, ys: dict, rng, n):
    dec = pd.qcut(L.mean(axis=1).rank(method="first"), 10, labels=False)
    terc = L.std(axis=1).groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    strat = dec * 3 + terc
    markers = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    pool = L.index.difference(ng).difference(list(markers))
    pool_by = {s: pool[strat.loc[pool].values == s] for s in range(30)}
    ns = strat.loc[ng].values
    r = corr_np(z_np(L.loc[ng].values), cent)
    ca = r[:, 2] - r[:, 0]
    obs = {k: partial_spearman(ca, y)[0] for k, y in ys.items()}
    null = {k: [] for k in ys}
    for _ in range(n):
        pick = np.empty(len(ng), dtype=object)
        for s in np.unique(ns):
            idx = np.where(ns == s)[0]
            pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=len(idx) > len(pool_by[s]))
        rr = corr_np(z_np(L.loc[list(pick)].values), cent)
        c = rr[:, 2] - rr[:, 0]
        for k, y in ys.items():
            null[k].append(partial_spearman(c, y)[0])
    out = {}
    for k in ys:
        v = np.array(null[k])
        out[k] = {"obs": round(obs[k], 3), "random_median": round(float(np.median(v)), 3),
                  "excess": round(obs[k] - float(np.median(v)), 3),
                  "p": round(float((np.sum(v >= obs[k]) + 1) / (n + 1)), 4)}
    return out


def clr(m: np.ndarray) -> np.ndarray:
    l = np.log(m + 0.5 / m.shape[1])
    return l - l.mean(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--cps", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")

    genes, meta, allsubs, U, N = load_sub(a.donors)
    donors = list(meta.index)
    cps = pd.read_csv(a.cps)
    cps = cps[cps["Brain Region"] == "MTG"].set_index("Donor ID")
    meta = meta.join(cps[["CPS_Global"]])
    for c, m in ORD.items():
        meta[c + "_num"] = meta[c].map(m)
    meta["female"] = (meta["Sex"] == "Female").astype(float)
    cov = meta[["Age at Death", "female"]].replace({"90+": 90}).apply(pd.to_numeric, errors="coerce").values
    ys = {"CPS": meta["CPS_Global"].values.astype(float), "Braak": meta["Braak_num"].values.astype(float),
          "dementia": meta["Cognitive Status_num"].values.astype(float)}

    sub_of = {s: t for t, v in TYPES.items() for s in v}
    tidx = {t: [allsubs.index(s) for s in v] for t, v in TYPES.items()}
    T = np.stack([U[:, tidx[t]].sum(1) for t in TYPES], 1)          # donor x type x gene
    n_type = pd.DataFrame({t: N[v].sum(axis=1) for t, v in TYPES.items()})

    pb = {"sorted": cpm(T.sum(1))}
    prof = cpm(T)
    avg = prof.mean(0)
    w = np.array([BASE[t] * RNA.get(t, 1) for t in TYPES]); w = w / w.sum()
    prof_f = np.where((n_type.values >= 10)[:, :, None], prof, avg[None])
    pb["fixed_comp"] = cpm(np.einsum("t,dtg->dg", w, prof_f))
    frac = n_type.div(n_type.sum(axis=1), axis=0).values
    neu = frac[:, :2].sum(1, keepdims=True)
    base_neu = sum(BASE[t] for t in NEURON)
    tissue_like = np.hstack([frac[:, :2] / neu * base_neu, frac[:, 2:] / (1 - neu) * (1 - base_neu)])
    rna = np.array([RNA.get(t, 1) for t in TYPES])
    pb["comp_only"] = cpm(np.einsum("dt,tg->dg", tissue_like * rna, avg))
    for t in ["Exc", "Inh", "Astro", "Oligo", "Micro"]:
        pb[f"type_{t}"] = prof[:, list(TYPES).index(t)]
    # within-Exc composition fixed: Exc profile rebuilt from each donor's own subclass profiles
    # mixed at the cohort-average subclass proportions -> only per-subclass expression varies
    exc_i = tidx["Exc"]
    sp = cpm(U[:, exc_i])                                             # donor x Exc subclass x gene
    sh = N[EXC].div(N[EXC].sum(axis=1), axis=0).values
    avg_sh = sh.mean(0)
    sp_f = np.where((N[EXC].values >= 10)[:, :, None], sp, sp.mean(0)[None])
    pb["Exc_fixed_subclass_mix"] = cpm(np.einsum("s,dsg->dg", avg_sh, sp_f))
    pb["Exc_subclass_comp_only"] = cpm(np.einsum("ds,sg->dg", sh, sp.mean(0)))
    for s in SUBS:
        k = allsubs.index(s)
        ok = N[s].values >= 20
        pb[f"sub_{s}"] = (cpm(U[:, k]), ok)

    rep = {"n_donors": len(donors), "n_random_sets": a.n}
    per = pd.DataFrame(index=donors)
    for name, M in pb.items():
        ok = np.ones(len(donors), bool)
        if isinstance(M, tuple):
            M, ok = M
        M = M[ok]
        keep = M.mean(0) >= 1
        L = pd.DataFrame(np.log2(M[:, keep] + 1).T, index=genes[keep], columns=np.array(donors)[ok])
        L = L[~L.index.duplicated()]
        ca, b, lab, ng, cent = project(L, cls)
        y_ok = {k: v[ok] for k, v in ys.items()}
        res = {"n_donors": int(ok.sum()), "n_neff_genes": int(len(ng)),
               "class_frac": {k: round(float((lab == k).mean()), 3) for k in "ABC"}}
        for o, y in y_ok.items():
            res[f"ca_vs_{o}"] = fmt(partial_spearman(ca, y))
            res[f"ca_vs_{o}_partial_age_sex"] = fmt(partial_spearman(ca, y, cov[ok]))
        res["random_null"] = null_test(L, ng, cent, {"CPS": y_ok["CPS"], "Braak": y_ok["Braak"]}, rng, a.n)
        rep[name] = res
        per.loc[np.array(donors)[ok], f"ca_{name}"] = ca
        print(name, json.dumps({k: res[k] for k in ["n_donors", "ca_vs_CPS", "ca_vs_Braak"]}),
              json.dumps(res["random_null"]), flush=True)

    # B. laminar sampling
    exc_share = N[EXC].div(N[EXC].sum(axis=1), axis=0)
    neuron_frac = N[[s for t in NEURON for s in TYPES[t]]].sum(axis=1) / N.sum(axis=1)
    sup_deep = np.log((N[SUPERFICIAL].sum(axis=1) + 1) / (N[DEEP].sum(axis=1) + 1))
    ca = per["ca_sorted"].values
    lam = {"share_of_Exc_vs_ca_sorted": {}, "share_of_Exc_vs_CPS": {}}
    for s in EXC:
        if exc_share[s].nunique() > 2:
            lam["share_of_Exc_vs_ca_sorted"][s] = fmt(partial_spearman(ca, exc_share[s].values))
            lam["share_of_Exc_vs_CPS"][s] = fmt(partial_spearman(exc_share[s].values, ys["CPS"]))
    lam["L23IT_vs_L4IT_share"] = fmt(partial_spearman(exc_share["L2/3 IT"].values, exc_share["L4 IT"].values))
    lam["superficial_deep_logratio_vs_ca_sorted"] = fmt(partial_spearman(ca, sup_deep.values))
    lam["superficial_deep_logratio_vs_CPS"] = fmt(partial_spearman(sup_deep.values, ys["CPS"]))
    lam["neuron_nuclei_frac_range"] = [round(float(neuron_frac.min()), 3), round(float(neuron_frac.max()), 3)]
    lam["neuron_nuclei_frac_vs_ca_sorted"] = fmt(partial_spearman(ca, neuron_frac.values))
    lam["neuron_nuclei_frac_vs_CPS"] = fmt(partial_spearman(neuron_frac.values, ys["CPS"]))
    adj = {"none": None, "age_sex": cov,
           "neuron_nuclei_frac": np.column_stack([cov, neuron_frac.values]),
           "L2/3 IT share": np.column_stack([cov, exc_share["L2/3 IT"].values]),
           "L4 IT share": np.column_stack([cov, exc_share["L4 IT"].values]),
           "superficial:deep ratio": np.column_stack([cov, sup_deep.values]),
           "L2/3 IT + L4 IT shares": np.column_stack([cov, exc_share[["L2/3 IT", "L4 IT"]].values]),
           "all Exc subclass shares (CLR)": np.column_stack([cov, clr(exc_share.values)[:, :-1]])}
    lam["ca_sorted_vs_CPS_partial"] = {k: fmt(partial_spearman(ca, ys["CPS"], v)) for k, v in adj.items()}
    lam["ca_sorted_vs_Braak_partial"] = {k: fmt(partial_spearman(ca, ys["Braak"], v)) for k, v in adj.items()}
    rep["laminar"] = lam
    per = per.join(exc_share.add_suffix("_of_Exc")).assign(neuron_nuclei_frac=neuron_frac,
                                                          sup_deep_logratio=sup_deep, CPS=ys["CPS"],
                                                          Braak=ys["Braak"])
    per.to_csv(a.out / "per_donor.csv")
    (a.out / "seaad_celltype_revision_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(lam, indent=1, default=str))


if __name__ == "__main__":
    main()
