"""Is the C-A score a person-level trait? Cross-region agreement in GSE84422 (v12, review round 10, Reviewer 2).

GSE84422 has up to 17 regions per subject. Within each region the C-A score (class C minus class A
centroid correlation) and the neuron marker score are computed as elsewhere (genes z-scored within region),
then standardised within region so that region means and spreads are removed. Per score:
  - ICC(1): one-way random-effects intraclass correlation over subjects (share of variance between people);
  - split-half: subject means over a random half of regions vs the other half (Spearman, 500 splits);
  - label agreement: share of a subject's regions carrying the subject's modal class, vs subjects
    shuffled within region (1,000 permutations).
Scores: C-A, neuron, and C-A residualised on the neuron score within region (the composition-independent
part). The C-A and residual ICCs are compared with those of 500 random 271-gene sets matched on expression
decile, or on expression decile and SEA-AD neuron/glia specificity class, that inherit the classifier
centroid rows (empirical upper-tail p). Subject means are also related to PMI and pH. Subject-level residual means are then
related to CDR, Braak and tangle sum (age/sex partial Spearman, AD and all subjects). Subject bootstrap 95% CIs.
Cortical regions only in a sensitivity run (caudate, putamen, hippocampus excluded).

Usage: python msbb_cross_region_v12.py --data /mnt/project-files/d1-data/raw --ref <seaad_mtg_reference.csv.gz> --out ../results/msbb_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import load_gse84422
import neff_subtypes
from meta_reanalysis import rho
from null_models import corr_np, z_np
from classifier_specificity import NEURONAL
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

NONCORTICAL = {"Caudate Nucleus", "Putamen", "Hippocampus"}
CLS = np.array(list("ABC"))


def icc1(v: pd.Series, subj: pd.Series) -> float:
    d = pd.DataFrame({"v": v.values, "s": subj.values}).dropna()
    g = d.groupby("s")["v"]
    k = g.size()
    d = d[d.s.isin(k[k >= 2].index)]
    g = d.groupby("s")["v"]
    k = g.size()
    n, N = len(k), len(d)
    grand = d.v.mean()
    msb = (k * (g.mean() - grand) ** 2).sum() / (n - 1)
    msw = ((d.v - d.s.map(g.mean())) ** 2).sum() / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def per_region(e, m, gi, cent, ni):
    out = []
    for reg, mm in m.groupby("region"):
        X = e[mm.index].values
        r = corr_np(z_np(X[gi]), cent)
        ca = r[:, 2] - r[:, 0]
        neu = z_np(X[ni]).mean(0)
        D = np.column_stack([np.ones(len(neu)), neu])
        res = ca - D @ np.linalg.lstsq(D, ca, rcond=None)[0]
        st = lambda v: (v - v.mean()) / v.std()
        out.append(pd.DataFrame({"subject": mm.subject.values, "region": reg, "ca": st(ca), "neuron": st(neu),
                                 "ca_resid": st(res), "label": CLS[r.argmax(1)]}, index=mm.index))
    return pd.concat(out)


def split_half(d, col, rng, n=500):
    regs = d.region.unique()
    out = []
    for _ in range(n):
        h = rng.permutation(regs)
        a = d[d.region.isin(h[: len(h) // 2])].groupby("subject")[col].mean()
        b = d[d.region.isin(h[len(h) // 2:])].groupby("subject")[col].mean()
        j = a.index.intersection(b.index)
        out.append(pd.Series(a[j]).rank().corr(pd.Series(b[j]).rank()))
    return float(np.nanmedian(out))


def label_agree(d):
    g = d.groupby("subject")["label"]
    k = g.size()
    keep = k[k >= 2].index
    return float(g.agg(lambda s: s.value_counts().iloc[0] / len(s))[keep].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1301)
    e, m = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    g = cls.index.intersection(e.index)
    gi = e.index.get_indexer(g)
    cent = cls.loc[g].sub(cls.loc[g].mean(axis=1), axis=0).values
    ni = e.index.get_indexer([x for x in NEURON_MARKERS if x in e.index and x not in g])
    marker = set(NEURON_MARKERS) | {x for v in CELL_MARKERS.values() for x in v}
    pool = e.index.difference(g).difference(list(marker))
    mean = e.mean(axis=1)
    dec = pd.qcut(mean.rank(method="first"), 10, labels=False)
    gdec = dec.loc[g].values
    ref = pd.read_csv(a.ref, index_col=0)
    ref = ref[~ref.index.duplicated()]
    ratio = np.log2((ref[NEURONAL].max(axis=1) + 1) / (ref[[c for c in ref.columns if c not in NEURONAL]].max(axis=1) + 1))
    spec = pd.Series(np.where(ratio > 1, 2, np.where(ratio < -1, 0, 1)), index=ref.index).reindex(e.index).fillna(3).astype(int)
    nulls = {"expression": dec, "specificity": dec * 4 + spec}
    subj_meta = m.drop_duplicates("subject").set_index("subject")
    report = {}
    for lab, mm in {"all_regions": m, "cortical": m[~m.region.isin(NONCORTICAL)]}.items():
        d = per_region(e, mm, gi, cent, ni)
        res = {"n_subjects": int(d.subject.nunique()), "n_samples": len(d), "n_regions": int(d.region.nunique())}
        for col in ["ca", "neuron", "ca_resid"]:
            boot = []
            subs = d.subject.unique()
            for _ in range(200):
                pick = rng.choice(subs, len(subs), replace=True)
                bd = pd.concat([d[d.subject == s].assign(subject=f"{s}_{i}") for i, s in enumerate(pick)])
                boot.append(icc1(bd[col], bd.subject))
            res[col] = {"icc1": round(icc1(d[col], d.subject), 3),
                        "icc1_ci95": [round(float(np.quantile(boot, q)), 3) for q in (.025, .975)],
                        "split_half_rho_median": round(split_half(d, col, rng), 3)}
        res["neuron_vs_ca_subject_means_rho"] = round(float(d.groupby("subject")[["ca", "neuron"]].mean().corr("spearman").iloc[0, 1]), 3)
        # label agreement vs subjects shuffled within region
        obs = label_agree(d)
        perm = []
        for _ in range(1000):
            dd = d.copy()
            dd["label"] = dd.groupby("region")["label"].transform(lambda s: rng.permutation(s.values))
            perm.append(label_agree(dd))
        res["label_modal_share"] = {"obs": round(obs, 3), "shuffled_median": round(float(np.median(perm)), 3),
                                    "p_upper": round(float(((np.array(perm) >= obs).sum() + 1) / 1001), 4)}
        # residual ICC against random gene sets (expression- or specificity-matched) with inherited centroid rows
        for nn, strat in nulls.items():
            gst = strat.loc[g].values
            by = {k: pool[strat.loc[pool].values == k] for k in np.unique(gst)}
            nul = []
            for _ in range(a.n):
                pick = np.empty(len(g), dtype=object)
                for k in np.unique(gst):
                    idx = np.where(gst == k)[0]
                    src = by[k] if len(by[k]) else pool
                    pick[idx] = rng.choice(src, len(idx), replace=len(idx) > len(src))
                dn = per_region(e, mm, e.index.get_indexer(list(pick)), cent, ni)
                nul.append((icc1(dn.ca, dn.subject), icc1(dn.ca_resid, dn.subject)))
            nul = np.array(nul)
            for j, col in enumerate(["ca", "ca_resid"]):
                o = res[col]["icc1"]
                res[col][f"{nn}_null"] = {"icc1_median": round(float(np.median(nul[:, j])), 3),
                                          "icc1_q05_q95": [round(float(np.quantile(nul[:, j], q)), 3) for q in (.05, .95)],
                                          "p_upper": round(float(((nul[:, j] >= o).sum() + 1) / (a.n + 1)), 4)}
        # subject-level means against clinical and pathology measures
        sm = d.groupby("subject")[["ca", "neuron", "ca_resid"]].mean().join(subj_meta[["age", "female", "cdr", "braak", "nft_sum", "group"]])
        sm = sm.drop(columns=[c for c in ["pmi", "ph"] if c in sm])
        clin = {}
        for grp, s in {"all": sm, "AD": sm[sm.group == "AD"]}.items():
            for y in ["cdr", "braak", "nft_sum"]:
                ss = s.dropna(subset=[y, "age", "female"])
                cov = ss[["age", "female"]].values.astype(float)
                for x in ["ca", "neuron", "ca_resid"]:
                    bs = []
                    for _ in range(1000):
                        b = rng.integers(0, len(ss), len(ss))
                        bs.append(rho(ss[x].values[b], ss[y].values[b].astype(float), cov[b]))
                    clin[f"{grp}_{y}_{x}"] = {"n": len(ss), "rho": round(rho(ss[x].values, ss[y].values.astype(float), cov), 3),
                                               "ci95": [round(float(np.nanquantile(bs, q)), 3) for q in (.025, .975)]}
        for y in ["pmi", "ph"]:
            ss = sm.join(subj_meta[[y]]).dropna(subset=[y])
            for x in ["ca", "neuron", "ca_resid"]:
                clin[f"all_{y}_{x}"] = {"n": len(ss), "rho": round(float(ss[x].rank().corr(ss[y].rank())), 3)}
        res["subject_level_association"] = clin
        report[lab] = res
        print(lab, json.dumps(res), flush=True)
    (a.out / "msbb_cross_region_v12.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
