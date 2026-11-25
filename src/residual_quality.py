"""Is the replicated residual axis (ribosome / heat shock) a sample-quality axis? (follow-up, 2026-11-25)

residual_denovo.py found that, after removing five cell scores, the leading residual axis replicated
across the four independent cohorts and was dominated by ribosomal and heat-shock genes, unrelated to
Braak, age and sex. Those cohorts carry no RNA-quality data, so here:

Template: in each independent cohort (AD samples) the residual PCs 1..5 of the 2,280 shared genes are
computed as in residual_denovo.py; GSE125583 PC1 is the reference, the best-matching PC of each other
cohort is sign-aligned to it, and the four loading vectors are averaged (w).
Scores, in a new cohort after the same residualisation on five cell scores:
  template  correlation of each sample's residual profile with w
  ribosome  mean residual z of RP[LS] genes;  heat_shock  mean residual z of curated heat-shock genes
Tests
  MSBB GSE84422 (17 regions, AD samples): Spearman of each score with pH, PMI, age, Braak, CDR per region
    (median, regions with p < 0.05, and same sign); subject level: score averaged over a subject's regions
    vs pH, PMI (all AD subjects). Null for the template: w with gene weights shuffled (200 times), subject-level |rho|.
  GSE106241 (temporal cortex, 60 samples, RNA quality 1-4): same scores vs RNA quality and Braak.

Usage: python residual_quality.py --data /mnt/project-files/d1-data/raw --out <dir>
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
from meta_reanalysis import INDEPENDENT
from null_models import z_np
from pilot_gse33000 import SEED
from residual_denovo import NPC, loadings, pca
from residual_identity import CURATED, MARKERS, PATTERNS, cell_scores
from residual_structure import residualise
from validate_cohorts import cohorts

VARS = ["ph", "pmi", "age", "braak", "cdr"]


def resid(e: pd.DataFrame, cols, genes) -> tuple[np.ndarray, list]:
    g = [x for x in genes if x in e.index]
    cells = cell_scores(e, cols, set())
    return residualise(z_np(e.loc[g, cols].values), cells), g


def scores(R: np.ndarray, g: list, w: pd.Series) -> dict:
    gi = pd.Index(g)
    wv = w.reindex(gi).values
    ok = ~np.isnan(wv)
    Rz = z_np(R)
    tmpl = np.array([np.corrcoef(Rz[ok, j], wv[ok])[0, 1] for j in range(R.shape[1])])
    rib = [i for i, x in enumerate(g) if re.match(PATTERNS["ribosome"], x)]
    hs = [i for i, x in enumerate(g) if x in CURATED["heat_shock"]]
    return {"template": tmpl, "ribosome": Rz[rib].mean(0), "heat_shock": Rz[hs].mean(0)}


def sp(x, y):
    y = pd.to_numeric(pd.Series(y), errors="coerce").values
    ok = ~np.isnan(y)
    r = stats.spearmanr(x[ok], y[ok])
    return float(r[0]), float(r[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 606)
    data = {}
    for name, e, m in cohorts(a.data):
        if name in INDEPENDENT:
            e = e.loc[e.index.notna() & ~e.index.duplicated()].dropna()
            data[name] = e[m.index[m["group"] == "AD"]]
    shared = None
    for e in data.values():
        mu = e.mean(axis=1)
        ok = mu.index[mu.rank(pct=True) > 0.4]
        shared = ok if shared is None else shared.intersection(ok)
    shared = shared.difference(list(MARKERS)).sort_values()
    L = {}
    for name, e in data.items():
        R, _ = resid(e, e.columns, shared)
        S, _ = pca(R)
        L[name] = loadings(R, S[:, :NPC])
    ref = L["GSE125583_FG"][:, 0]
    aligned, match = [ref], {"GSE125583_FG": [1, 1.0]}
    for name, ld in L.items():
        if name == "GSE125583_FG":
            continue
        r = np.array([np.corrcoef(ref, ld[:, k])[0, 1] for k in range(NPC)])
        k = int(np.abs(r).argmax())
        aligned.append(ld[:, k] * np.sign(r[k]))
        match[name] = [k + 1, round(float(r[k]), 3)]
    w = pd.Series(np.mean(aligned, axis=0), index=shared)
    w.sort_values().to_csv(a.out / "residual_template_weights.csv")
    out = {"n_template_genes": int(len(shared)), "matched_pc_and_r_to_GSE125583_PC1": match,
           "template_top_positive": w.sort_values().index[-15:][::-1].tolist(),
           "template_top_negative": w.sort_values().index[:15].tolist()}

    # MSBB
    e, m = load_gse84422.load(a.data)
    e = e.loc[e.index.notna() & ~e.index.duplicated()]
    rows, per_sample = [], []
    for reg, mm in m[m["group"] == "AD"].groupby("region"):
        R, g = resid(e, mm.index, shared)
        sc = scores(R, g, w)
        row = {"region": reg, "n": len(mm)}
        for s, v in sc.items():
            for var in VARS:
                row[f"{s}_{var}_rho"], row[f"{s}_{var}_p"] = sp(v, mm[var].values)
        rows.append(row)
        per_sample.append(pd.DataFrame({"subject": mm["subject"].values, **sc}))
    t = pd.DataFrame(rows).set_index("region")
    t.round(4).to_csv(a.out / "msbb_by_region.csv")
    msbb = {"n_regions": int(len(t)), "genes_in_template_found": int(len([x for x in shared if x in e.index]))}
    for s in ("template", "ribosome", "heat_shock"):
        msbb[s] = {var: {"median_rho": round(float(t[f"{s}_{var}_rho"].median()), 3),
                         "regions_p05": int((t[f"{s}_{var}_p"] < .05).sum()),
                         "regions_same_sign_as_median": int((np.sign(t[f"{s}_{var}_rho"]) == np.sign(t[f"{s}_{var}_rho"].median())).sum())}
                   for var in VARS}
    ps = pd.concat(per_sample).groupby("subject").mean()
    subj = m[m["group"] == "AD"].groupby("subject")[["ph", "pmi", "age"]].first().loc[ps.index]
    msbb["subject_level_n"] = int(len(ps))
    msbb["subject_level"] = {s: {var: [round(x, 4) for x in sp(ps[s].values, subj[var].values)] for var in ("ph", "pmi", "age")}
                             for s in ("template", "ribosome", "heat_shock")}
    # null: shuffled template weights, subject-level |rho| with pH
    obs = abs(msbb["subject_level"]["template"]["ph"][0])
    nl = []
    regs = list(m[m["group"] == "AD"].groupby("region"))
    cache = {reg: resid(e, mm.index, shared) for reg, mm in regs}
    for _ in range(200):
        wp = pd.Series(rng.permutation(w.values), index=w.index)
        ps_ = []
        for reg, mm in regs:
            R, g = cache[reg]
            ps_.append(pd.DataFrame({"subject": mm["subject"].values, "t": scores(R, g, wp)["template"]}))
        q = pd.concat(ps_).groupby("subject")["t"].mean().loc[subj.index]
        nl.append(abs(sp(q.values, subj["ph"].values)[0]))
    nl = np.array(nl)
    msbb["template_ph_shuffled_null"] = {"median_abs": round(float(np.median(nl)), 3), "p95": round(float(np.percentile(nl, 95)), 3),
                                         "p": round(float(((nl >= obs).sum() + 1) / (len(nl) + 1)), 4)}
    out["MSBB_GSE84422"] = msbb
    print(json.dumps(msbb), flush=True)

    # GSE106241
    e, m = load_gse106241.load(a.data)
    e = e.loc[e.index.notna() & ~e.index.duplicated()]
    R, g = resid(e, m.index, shared)
    sc = scores(R, g, w)
    out["GSE106241"] = {"n": int(len(m)), **{s: {var: [round(x, 4) for x in sp(v, m[var].values)] for var in ("rna_quality", "braak", "age")}
                                              for s, v in sc.items()}}
    print(json.dumps(out["GSE106241"]), flush=True)
    (a.out / "residual_quality.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
