"""Composition-free test of the Neff C-A axis within single SEA-AD subclasses, with a spike-in (v14).

Within one subclass (e.g. L2/3 IT neurons, astrocytes) the cell-type composition of a pseudobulk is fixed by
construction, so any C-A variation between donors is composition-independent (apart from supertype shifts
inside the subclass). Two questions, each against random genes that inherit the classifier genes' centroid
rows (matched on all-nuclei expression decile; classifier and marker genes excluded; the same gene identities
in every subclass, as for the classifier):

  T1 pathology : partial Spearman (age, sex) of the subclass C-A score with the continuous pseudo-progression
                 score (CPS) - two-sided empirical p;
  T2 concordance: mean pairwise Spearman correlation of donors' C-A scores across subclasses (all subclasses;
                 neuronal; non-neuronal) - a donor-level subtype that is not composition should place a donor
                 at the same end of the axis in many cell types. Shared technical factors (RIN, PMI, pH)
                 also produce concordance; random genes share them, so the test is against the null, and the
                 observed value is also reported after regressing RIN, PMI and pH out of each subclass score.

T1 is repeated after adding RIN, PMI and brain pH, and after adding the donor's supertype proportions within
the subclass (supertype composition is the composition left inside a subclass), to the covariates. Because all
subclasses come from the same donors, subclass p-values are not combined; instead the mean partial rho over
neuronal and over non-neuronal subclasses is compared with the same mean in the null (the same random genes in
every subclass, so the dependence is kept).

Share of variance = var(score with component - score without) / var(score with component).
Power (no existing signal assumed): for 100 random gene sets from the null, a CPS-aligned component
f * z(CPS)_j * (C - A)_g (C - A of the inherited centroid rows) is added in every subclass, with f calibrated per
subclass so that it explains the target share of that subclass's C-A variance; power is the share of sets whose
statistic then lies beyond the null (two-sided p < 0.05 per subclass; upper tail for the means).

Usage: python seaad_within_subclass_v14.py --donors <dir with npz + donor_meta.csv + genes.txt>
         --cps Global_and_Local_CPS.20260105.csv --data <raw> --out <dir> [--label MTG]
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from braak_gse106241 import partial_spearman
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

SUBCLASSES = ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "Sst", "Pvalb", "Vip", "Lamp5",
              "Astrocyte", "Oligodendrocyte", "OPC", "Microglia-PVM"]
NEURONAL = set(SUBCLASSES[:8])
MIN_NUC = 30
SHARES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
F_GRID = np.array([0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, .75, 1.0, 1.5, 2.0])


def load(donor_dir: Path):
    genes = np.array((donor_dir / "genes.txt").read_text().split("\n"))
    meta = pd.read_csv(donor_dir / "donor_meta.csv").set_index("Donor ID")
    umi = {s: {} for s in SUBCLASSES}
    sup = {s: {} for s in SUBCLASSES}
    alias = {"Astro": "Astrocyte", "Oligo": "Oligodendrocyte", "Micro-PVM": "Microglia-PVM", "Lamp5_Lhx6": "Lamp5 Lhx6"}
    tot = {}
    for d in meta.index:
        z = np.load(donor_dir / f"{d}.npz", allow_pickle=True)
        subs, n = z["subclasses"].astype(str), z["n_nuclei"]
        U = z["umis"].astype(float)
        tot[d] = U.sum(0)
        for i, s in enumerate(subs):
            if s in umi and n[i] >= MIN_NUC:
                umi[s][d] = U[i]
        st, sn = z["supertypes"].astype(str), z["supertype_n"]
        for t, k in zip(st, sn):
            m = re.match(r"^(.*?)_\d", t)
            parent = alias.get(m.group(1), m.group(1)) if m else t
            if parent in sup:
                sup[parent].setdefault(d, {})[t] = k
    sup = {s: pd.DataFrame(v).T.fillna(0) for s, v in sup.items()}
    sup = {s: v.div(v.sum(axis=1), axis=0) for s, v in sup.items() if len(v)}
    return genes, meta, umi, tot, sup


def logcpm(rows: dict, genes) -> pd.DataFrame:
    M = np.array(list(rows.values()))
    M = M / M.sum(1, keepdims=True) * 1e6
    keep = M.mean(0) >= 1
    return pd.DataFrame(np.log2(M[:, keep] + 1).T, index=genes[keep], columns=list(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--cps", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="MTG")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-power", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1515)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes, meta, umi, tot, sup = load(a.donors)
    cps = pd.read_csv(a.cps)
    cps = cps[cps["Brain Region"] == "MTG"].set_index("Donor ID")   # CPS_Global is donor-level
    meta = meta.join(cps[["CPS_Global"]])
    meta["age"] = pd.to_numeric(meta["Age at Death"].replace({"90+": 90}), errors="coerce")
    meta["female"] = (meta["Sex"] == "Female").astype(float)
    for c in ("RIN", "PMI", "Brain pH"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
        meta[c] = meta[c].fillna(meta[c].median())
    T = np.array(list(tot.values()))
    allcpm = pd.Series((T / T.sum(1, keepdims=True) * 1e6).mean(0), index=genes)
    allcpm = allcpm[~allcpm.index.duplicated()]
    marker = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    cg_all = [g for g in cls.index if g in allcpm.index and allcpm[g] >= 1]
    expressed = allcpm[allcpm >= 1]
    dec = pd.qcut(expressed.rank(method="first"), 10, labels=False)
    pool = expressed.index.difference(cg_all).difference(list(marker))
    by = {k: pool[dec.loc[pool].values == k] for k in range(10)}
    gst = dec.loc[cg_all].values

    def draw():
        pick = np.empty(len(cg_all), dtype=object)
        for k in np.unique(gst):
            idx = np.where(gst == k)[0]
            pick[idx] = rng.choice(by[k], len(idx), replace=False)
        return list(pick)
    L = {s: logcpm(umi[s], genes) for s in SUBCLASSES if len(umi[s]) >= 40}
    L = {s: x[~x.index.duplicated()] for s, x in L.items()}
    subs = list(L)
    neu_subs = [s for s in subs if s in NEURONAL]
    gli_subs = [s for s in subs if s not in NEURONAL]
    cols = {s: list(L[s].columns) for s in subs}
    cent_all = cls.loc[cg_all].sub(cls.loc[cg_all].mean(axis=1), axis=0)
    contrast_all = (cls.loc[cg_all, "C"] - cls.loc[cg_all, "A"])
    zc = meta["CPS_Global"]
    zc = ((zc - zc.mean()) / zc.std()).fillna(0.0)
    covs = {}
    for s in subs:
        m = meta.loc[cols[s]]
        base = m[["age", "female"]].values
        tech = m[["age", "female", "RIN", "PMI", "Brain pH"]].values
        sp = sup[s].reindex(cols[s]).fillna(0)
        sp = sp.loc[:, sp.mean() > 0.02]
        spv = sp.values[:, 1:] if sp.shape[1] > 1 else np.zeros((len(cols[s]), 0))   # drop one (sums to 1)
        covs[s] = {"age_sex": base, "tech": tech, "supertype": np.column_stack([base, spv]),
                   "n_supertypes": int(sp.shape[1])}

    def mats(gene_ids):
        out = {}
        for s in subs:
            use = [(g, cg_all[i]) for i, g in enumerate(gene_ids) if g in L[s].index]
            out[s] = (L[s].loc[[u for u, _ in use]].values, cent_all.loc[[c for _, c in use]].values,
                      contrast_all.loc[[c for _, c in use]].values)
        return out

    def scores(M, f=None):
        sc = {}
        for s, (X, cent, con) in M.items():
            if f is not None and f[s] > 0:
                X = X + f[s] * np.outer(con, zc.loc[cols[s]].values)
            r = corr_np(z_np(X), cent)
            sc[s] = r[:, 2] - r[:, 0]
        return sc

    def cps_stats(sc, adj="age_sex"):
        o = {s: partial_spearman(v, meta.loc[cols[s], "CPS_Global"].values, covs[s][adj])[0] for s, v in sc.items()}
        o["mean_neuronal"] = float(np.mean([o[s] for s in neu_subs]))
        o["mean_non_neuronal"] = float(np.mean([o[s] for s in gli_subs]))
        return o

    def concordance(sc):
        conc = {}
        for name, sel in (("all", subs), ("neuronal", neu_subs), ("non_neuronal", gli_subs)):
            rs = []
            for s1, s2 in combinations(sel, 2):
                a1, a2 = pd.Series(sc[s1], index=cols[s1]), pd.Series(sc[s2], index=cols[s2])
                j = a1.index.intersection(a2.index)
                rs.append(stats.spearmanr(a1[j], a2[j])[0])
            conc[name] = float(np.mean(rs))
        return conc
    ADJ = ("age_sex", "tech", "supertype")
    Mobs = mats(cg_all)
    sobs = scores(Mobs)
    obs = {k: cps_stats(sobs, k) for k in ADJ}
    obs_conc = concordance(sobs)
    null = {k: [] for k in ADJ}
    null_conc, null_sets = [], []
    for i in range(a.n_null):
        g = draw()
        M = mats(g)
        sc = scores(M)
        for k in ADJ:
            null[k].append(cps_stats(sc, k))
        null_conc.append(concordance(sc))
        if i < a.n_power:
            null_sets.append(M)
    null = {k: pd.DataFrame(v) for k, v in null.items()}
    null_conc = pd.DataFrame(null_conc)
    p2 = lambda nv, o: float(((np.abs(nv - np.median(nv)) >= abs(o - np.median(nv))).sum() + 1) / (len(nv) + 1))
    pu = lambda nv, o: float(((nv >= o).sum() + 1) / (len(nv) + 1))
    keys = subs + ["mean_neuronal", "mean_non_neuronal"]
    per = []
    for s in keys:
        row = {"subclass": s}
        if s in L:
            row.update({"n_donors": len(cols[s]), "n_classifier_genes": int(Mobs[s][0].shape[0]),
                        "n_supertypes_used": covs[s]["n_supertypes"]})
        for k in ADJ:
            nv = null[k][s].values
            row[k] = {"partial_rho": round(obs[k][s], 3), "null_median": round(float(np.median(nv)), 3),
                      "null_q025_q975": [round(float(np.quantile(nv, .025)), 3), round(float(np.quantile(nv, .975)), 3)],
                      "p_two_sided": round(p2(nv, obs[k][s]), 4), "p_upper": round(pu(nv, obs[k][s]), 4)}
        per.append(row)
        print(json.dumps(row), flush=True)
    conc = {k: {"observed": round(obs_conc[k], 3), "null_median": round(float(null_conc[k].median()), 3),
                "null_q05_q95": [round(float(null_conc[k].quantile(.05)), 3), round(float(null_conc[k].quantile(.95)), 3)],
                "p_upper": round(pu(null_conc[k].values, obs_conc[k]), 4),
                "p_lower": round(float(((null_conc[k].values <= obs_conc[k]).sum() + 1) / (len(null_conc) + 1)), 4)}
            for k in obs_conc}
    print(json.dumps(conc), flush=True)
    # power on random-gene baselines: calibrate f per subclass on the first 20 sets
    cal = {s: [] for s in subs}
    for f in F_GRID:
        sh = {s: [] for s in subs}
        for M in null_sets[:20]:
            sc, s0 = scores(M, {t: f for t in subs}), scores(M)
            for s in subs:   # share of C-A variance carried by the added component
                sh[s].append(float(np.var(sc[s] - s0[s]) / np.var(sc[s])) if f > 0 else 0.0)
        for s in subs:
            cal[s].append(max(0.0, float(np.median(sh[s]))))
    fmap = {v: {s: float(np.interp(v, np.maximum.accumulate(cal[s]) + np.arange(len(F_GRID)) * 1e-9, F_GRID)) for s in subs}
            for v in SHARES}
    prow = []
    for v in SHARES:
        det = {k: 0 for k in keys}
        for M in null_sets:
            st = cps_stats(scores(M, fmap[v]), "age_sex")
            for k in keys:
                nv = null["age_sex"][k].values
                det[k] += (p2(nv, st[k]) if k in subs else pu(nv, st[k])) < .05
        prow.append({"share": v, **{k: round(det[k] / len(null_sets), 3) for k in keys}})
        print(json.dumps(prow[-1]), flush=True)
    pd.DataFrame(prow).to_csv(a.out / f"within_subclass_power_{a.label}.csv", index=False)
    rep_ = {"label": a.label, "n_donors": int(len(meta)), "min_nuclei": MIN_NUC, "n_null": a.n_null,
            "n_power_sets": len(null_sets), "per_subclass_cps": per, "concordance": conc,
            "f_for_share": {str(k): {s: round(x, 4) for s, x in d.items()} for k, d in fmap.items()}, "power": prow}
    (a.out / f"within_subclass_v14_{a.label}.json").write_text(json.dumps(rep_, indent=2))


if __name__ == "__main__":
    main()
