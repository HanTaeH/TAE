"""Every pooled estimate that carries a conclusion, repeated without the 62 shared Banner donors (v12, review round 10, Reviewer 1 M6).

GSE132903 (MTG) and GSE125583 (fusiform gyrus) share 62 donors. The Methods state that every
meta-analysis was repeated without them; v11 did so only for the two primary endpoints. Here the
pooled estimates below are computed for three cohort sets: all samples, shared donors removed from
GSE132903, and shared donors removed from GSE125583 (random-effects pool of Fisher-z with bootstrap SEs,
meta_reanalysis.pool):
  P1                 : C-A vs neuron score, AD samples;
  P2_agesex          : C-A vs diagnostic group, age/sex partial;
  P2_neuron          : + neuron score; P2_neuron_rank: + neuron score as ranks;
  P2_neuron_glia     : + neuron and four glial scores; P2_nnls: + NNLS proportions (clr, deconv_seaad.py);
  reverse            : neuron score vs group given age, sex and C-A;
  semipartial        : case-control remainder on the scale of the unadjusted score (mean of Fisher-z).

Usage: python shared_donor_v12.py --data /mnt/project-files/d1-data/raw --res ../results --out ../results/shared_donor_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from deconv_seaad import clr
from meta_reanalysis import INDEPENDENT, banner_overlap, boot_se, ca_score, pool, resid_rank, rho
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from stage_extra_sensitivity import NNLS
from validate_cohorts import GROUP_ORDER, cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1305)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    P = pd.read_csv(a.res / "deconv/deconv_proportions.csv", index_col=0)
    ov = banner_overlap(a.data)
    drop = {"GSE132903_MTG": ov["GSE132903"], "GSE125583_FG": ov["GSE125583"]}
    rows = []
    for name, e0, m0 in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        for suf, keep in [("", m0.index)] + ([("_noshared", m0.index.difference(list(drop[name])))] if name in drop else []):
            e, m = e0[keep], m0.loc[keep]
            genes = cls.index.intersection(e.index)
            cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
            Z = zscore_rows(e)
            ad = (m["group"] == "AD").values
            Zad = zscore_rows(e.loc[:, ad])
            mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
            ca_all = ca_score(Z.loc[genes], cent).values
            ca_ad = ca_score(Zad.loc[genes], cent).values
            neu_all, neu_ad = Z.loc[mk].mean().values, Zad.loc[mk].mean().values
            glia = np.column_stack([Z.loc[[x for x in v if x in e.index and x not in genes]].mean().values
                                    for v in CELL_MARKERS.values()])
            stage = m["group"].map(GROUP_ORDER).values.astype(float)
            age = pd.to_numeric(m["age"], errors="coerce")
            age = age.fillna(age.median()).values
            sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
            cov = np.column_stack([age, sex])
            specs = {"P1": (ca_ad, neu_ad, None), "P2_agesex": (ca_all, stage, cov),
                     "P2_neuron": (ca_all, stage, np.column_stack([cov, neu_all])),
                     "P2_neuron_rank": (ca_all, stage, np.column_stack([cov, stats.rankdata(neu_all)])),
                     "P2_neuron_glia": (ca_all, stage, np.column_stack([cov, neu_all, glia])),
                     "reverse": (neu_all, stage, np.column_stack([cov, ca_all]))}
            pp = P.reindex(m.index)[NNLS]
            ok = pp.notna().all(1).values
            for sp, (x, y, c) in specs.items():
                r = rho(x, y, c)
                rows.append({"spec": sp, "cohort": name, "variant": suf, "n": len(x), "rho": r, "z": np.arctanh(r),
                             "se_boot": boot_se(x, y, c, rng, a.n_boot)})
            if ok.sum() > 20:
                C = np.column_stack([cov[ok], clr(pp.values[ok] / pp.values[ok].sum(1, keepdims=True))])
                r = rho(ca_all[ok], stage[ok], C)
                rows.append({"spec": "P2_nnls", "cohort": name, "variant": suf, "n": int(ok.sum()), "rho": r,
                             "z": np.arctanh(r), "se_boot": boot_se(ca_all[ok], stage[ok], C, rng, a.n_boot)})
            yr, xr = resid_rank(stage, cov), resid_rank(ca_all, cov)
            ex = resid_rank(ca_all, np.column_stack([cov, neu_all]))
            sr = float(np.cov(yr, ex)[0, 1] / (yr.std(ddof=1) * xr.std(ddof=1)))
            rows.append({"spec": "semipartial", "cohort": name, "variant": suf, "n": len(ca_all), "rho": sr,
                         "z": np.arctanh(sr), "se_boot": np.nan})
            print(name, suf, "done", flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out / "shared_donor_per_cohort.csv", index=False)
    rep = {"n_shared_donors": len(ov["GSE132903"])}
    for sp, d in df.groupby("spec"):
        sets = {"all": d[d.variant == ""]}
        for w in drop:
            sets[f"removed_from_{w}"] = pd.concat([d[(d.variant == "") & (d.cohort != w)], d[(d.variant == "_noshared") & (d.cohort == w)]])
        rep[sp] = {}
        for lab, s in sets.items():
            if sp == "semipartial":
                rep[sp][lab] = {"mean_fisher": round(float(np.tanh(s.z.mean())), 3)}
            else:
                rep[sp][lab] = pool(s.z.values, s.se_boot.values)
            rep[sp][lab]["per_cohort"] = dict(zip(s.cohort, s.rho.round(3)))
    (a.out / "shared_donor_v12.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: {l: (v[l].get("rho", v[l].get("mean_fisher"))) for l in v} for k, v in rep.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
