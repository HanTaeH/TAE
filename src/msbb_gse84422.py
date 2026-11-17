"""Option 3: MSBB clinical links and brain-region comparison (GSE84422, GPL96, 107 subjects x 17 regions).

Not an independent cohort (MSBB donors overlap Neff 2021's discovery data); used as a positive control
and for clinical / regional questions. C-minus-A score = r(C) - r(A).

Per region
  P  AD-only projection (definite + probable AD): class fractions (Neff regions BM10 frontal pole,
     BM22 STG, BM36 PHG, BM44 IFG compared with Neff's MSBB 31/36/33%), C-minus-A vs neuron score
  K  all-sample projection: C-minus-A vs CDR, Braak, plaque density, neuropathology category;
     partial (age, sex, PMI, pH); plus neuron score; AD-only CDR link; neuron score vs CDR
  N  random gene null for C-minus-A vs CDR (decile x SD tercile, inherits centroid rows)
Across regions
  R  do regions whose neuron score falls more with CDR show a stronger C-minus-A vs CDR link?
Per subject (n = 107)
  S  mean within-region C-minus-A (all-sample projection) vs CDR / Braak / plaque, partial for age
     and sex, then for the subject's mean neuron score
  T  is the class a trait of the person? correlation of C-minus-A between region pairs of the same
     subject, raw and after removing CDR, Braak, plaque, age, sex, PMI, pH and neuron score, vs a null with
     subjects shuffled within region

Usage: python msbb_gse84422.py --data /mnt/project-files/d1-data/raw --out ../results/msbb --n 200
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import load_gse84422
import neff_subtypes
from braak_gse106241 import partial_spearman
from null_models import corr_np, z_np
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED

CLS = np.array(list("ABC"))
NEFF_REGIONS = {"Frontal Pole": "BM10", "Superior Temporal Gyrus": "BM22",
                "Parahippocampal Gyrus": "BM36", "Inferior Frontal Gyrus": "BM44"}
NEFF_MSBB = {"A": round(47 / 151, 3), "B": round(54 / 151, 3), "C": round(50 / 151, 3)}
SUBCORTICAL = ["Caudate Nucleus", "Putamen"]
CLIN = ["cdr", "braak", "plaque", "np_num"]


def fmt(t):
    return {"rho": round(t[0], 3), "p": float(f"{t[1]:.3g}"), "n": t[2]}


def scores(x, cent):
    r = corr_np(z_np(x), cent)
    return r[:, 2] - r[:, 0], CLS[r.argmax(1)]


def residual(y, X):
    ok = ~(np.isnan(y) | np.isnan(X).any(1))
    out = np.full(len(y), np.nan)
    D = np.column_stack([np.ones(ok.sum()), X[ok]])
    out[ok] = y[ok] - D @ np.linalg.lstsq(D, y[ok], rcond=None)[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    expr, meta = load_gse84422.load(a.data)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(expr.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    markers = set(NEURON_MARKERS) | {g for v in CELL_MARKERS.values() for g in v}
    neu_genes = [g for g in NEURON_MARKERS if g in expr.index and g not in genes]

    rep: dict = {"n_samples": int(len(meta)), "n_subjects": int(meta["subject"].nunique()),
                 "n_genes": int(len(expr)), "neff_genes": int(len(genes)), "coverage": round(len(genes) / len(cls), 3),
                 "neuron_marker_genes": neu_genes, "neff_msbb_class_frac": NEFF_MSBB, "regions": {}}
    per = []
    for region, m in meta.groupby("region"):
        X = expr[m.index]
        cov = m[["age", "female", "pmi", "ph"]].values.astype(float)
        neuron = z_np(X.loc[neu_genes].values).mean(0)
        ad = (m["group"] == "AD").values
        ca_ad, lab_ad = scores(X.loc[genes].values[:, ad], cent)
        ca, lab = scores(X.loc[genes].values, cent)
        r: dict = {"n": int(len(m)), "n_AD": int(ad.sum()),
                   "AD_class_frac": {k: round(float((lab_ad == k).mean()), 3) for k in "ABC"},
                   "AD_ca_vs_neuron": fmt(partial_spearman(ca_ad, neuron[ad])),
                   "neuron_vs_cdr": fmt(partial_spearman(neuron, m["cdr"].values)),
                   "neuron_vs_cdr_partial": fmt(partial_spearman(neuron, m["cdr"].values, cov))}
        for c in CLIN:
            y = m[c].values
            r[f"ca_vs_{c}"] = fmt(partial_spearman(ca, y))
            r[f"ca_vs_{c}_partial"] = fmt(partial_spearman(ca, y, cov))
            r[f"ca_vs_{c}_partial_neuron"] = fmt(partial_spearman(ca, y, np.column_stack([cov, neuron])))
        r["AD_ca_vs_cdr"] = fmt(partial_spearman(ca_ad, m["cdr"].values[ad]))
        r["AD_ca_vs_cdr_partial_neuron"] = fmt(partial_spearman(ca_ad, m["cdr"].values[ad], neuron[ad, None]))
        # random-gene null for ca vs CDR
        dec = pd.qcut(X.mean(axis=1).rank(method="first"), 10, labels=False)
        terc = X.std(axis=1).groupby(dec).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        strat = dec * 3 + terc
        pool = X.index.difference(genes).difference(list(markers))
        pool_by = {s: pool[strat.loc[pool].values == s] for s in range(30)}
        ns = strat.loc[genes].values
        obs = r["ca_vs_cdr"]["rho"]
        null = []
        for _ in range(a.n):
            pick = np.empty(len(genes), dtype=object)
            for s in np.unique(ns):
                idx = np.where(ns == s)[0]
                pick[idx] = rng.choice(pool_by[s], size=len(idx), replace=False)
            null.append(partial_spearman(scores(X.loc[list(pick)].values, cent)[0], m["cdr"].values)[0])
        r["ca_vs_cdr_random_null"] = {"obs": obs, "random_median": round(float(np.median(null)), 3),
                                      "random_q95": round(float(np.quantile(null, .95)), 3),
                                      "p": round(float((np.sum(np.array(null) >= obs) + 1) / (a.n + 1)), 4)}
        rep["regions"][region] = r
        per.append(m.assign(ca=ca, neuron=neuron, cls=lab))
    per = pd.concat(per)

    # P: positive control in Neff's regions
    rep["P_neff_regions_AD_class_frac"] = {f"{k} ({v})": rep["regions"][k]["AD_class_frac"] for k, v in NEFF_REGIONS.items()}
    # R: across regions
    reg = pd.DataFrame({k: {"neuron_cdr": v["neuron_vs_cdr"]["rho"], "ca_cdr": v["ca_vs_cdr"]["rho"],
                            "ca_cdr_adj": v["ca_vs_cdr_partial_neuron"]["rho"], "ad_ca_neuron": v["AD_ca_vs_neuron"]["rho"],
                            "null_p": v["ca_vs_cdr_random_null"]["p"]} for k, v in rep["regions"].items()}).T
    reg.to_csv(a.out / "msbb_region_summary.csv")
    rep["R_across_regions"] = {
        "ca_cdr_vs_neuron_cdr": fmt(partial_spearman(reg["ca_cdr"].values, reg["neuron_cdr"].values)),
        "n_regions_ca_cdr_p05": int(sum(v["ca_vs_cdr"]["p"] < .05 for v in rep["regions"].values())),
        "n_regions_ca_cdr_null_p05": int((reg["null_p"] < .05).sum()),
        "n_regions_adj_p05": int(sum(v["ca_vs_cdr_partial_neuron"]["p"] < .05 for v in rep["regions"].values())),
        "median_ca_cdr": round(float(reg["ca_cdr"].median()), 3),
        "median_ca_cdr_adj": round(float(reg["ca_cdr_adj"].median()), 3),
        "median_AD_ca_neuron": round(float(reg["ad_ca_neuron"].median()), 3)}

    # S: subject level
    for scope, keep in {"all_regions": per.index, "cortex": per.index[~per["region"].isin(SUBCORTICAL)]}.items():
        p = per.loc[keep]
        subj = p.groupby("subject").agg(ca=("ca", "mean"), neuron=("neuron", "mean"), n_regions=("ca", "size"),
                                        **{c: (c, "first") for c in CLIN + ["age", "female", "group"]})
        cov = subj[["age", "female"]].values.astype(float)
        s = {"n_subjects": int(len(subj))}
        for c in CLIN:
            s[f"ca_vs_{c}"] = fmt(partial_spearman(subj["ca"].values, subj[c].values))
            s[f"ca_vs_{c}_partial"] = fmt(partial_spearman(subj["ca"].values, subj[c].values, cov))
            s[f"ca_vs_{c}_partial_neuron"] = fmt(partial_spearman(subj["ca"].values, subj[c].values,
                                                                  np.column_stack([cov, subj["neuron"].values])))
        s["ca_vs_neuron"] = fmt(partial_spearman(subj["ca"].values, subj["neuron"].values))
        adm = (subj["group"] == "AD").values
        s["AD_ca_vs_cdr"] = fmt(partial_spearman(subj["ca"].values[adm], subj["cdr"].values[adm]))
        s["AD_ca_vs_cdr_partial_neuron"] = fmt(partial_spearman(subj["ca"].values[adm], subj["cdr"].values[adm],
                                                                subj["neuron"].values[adm, None]))
        rep[f"S_subject_{scope}"] = s
        if scope == "cortex":
            subj.to_csv(a.out / "msbb_subject_scores.csv")
            subj_cortex = subj

    # T: person-level consistency between regions
    covs = ["cdr", "braak", "plaque", "age", "female", "pmi", "ph"]
    per["ca_resid"] = np.nan
    for region, m in per.groupby("region"):
        per.loc[m.index, "ca_resid"] = residual(m["ca"].values, m[covs + ["neuron"]].values.astype(float))
    wide_raw = per.pivot_table(index="subject", columns="region", values="ca")
    wide_res = per.pivot_table(index="subject", columns="region", values="ca_resid")

    def pair_median(w):
        rs = []
        for c1, c2 in itertools.combinations(w.columns, 2):
            d = w[[c1, c2]].dropna()
            if len(d) >= 20:
                rs.append(stats.spearmanr(d[c1], d[c2])[0])
        return float(np.median(rs)), len(rs)
    t = {}
    for nm, w in {"raw": wide_raw, "residual": wide_res}.items():
        obs, npairs = pair_median(w)
        null = []
        for _ in range(a.n):
            ws = w.copy()
            for c in ws.columns:
                v = ws[c].dropna()
                ws.loc[v.index, c] = rng.permutation(v.values)
            null.append(pair_median(ws)[0])
        t[nm] = {"median_pairwise_rho": round(obs, 3), "n_pairs": npairs, "null_median": round(float(np.median(null)), 3),
                 "null_q95": round(float(np.quantile(null, .95)), 3),
                 "p": round(float((np.sum(np.array(null) >= obs) + 1) / (a.n + 1)), 4)}
    # class agreement across regions (raw all-sample labels): fraction of subjects whose modal class covers >= 2/3 regions
    modal = per.groupby("subject")["cls"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    t["modal_class_share_median"] = round(float(modal.median()), 3)
    rep["T_person_consistency"] = t

    (a.out / "msbb_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    per.drop(columns=[c for c in per.columns if c.startswith("_")]).to_csv(a.out / "msbb_per_sample.csv")
    print(json.dumps({k: v for k, v in rep.items() if k != "regions"}, indent=1, ensure_ascii=False, default=str))
    print(reg.sort_values("neuron_cdr").to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    o = reg.sort_values("ca_cdr")
    y = np.arange(len(o))
    ax.barh(y - .2, o["ca_cdr"], height=.4, color="#1f4e79", label="raw")
    ax.barh(y + .2, o["ca_cdr_adj"], height=.4, color="#9db7d5", label="+ age, sex, PMI, pH, neuron score")
    ax.set_yticks(y)
    ax.set_yticklabels(o.index, fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Spearman ρ, C-minus-A vs CDR")
    ax.set_title("C-minus-A vs dementia rating by region", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.scatter(reg["neuron_cdr"], reg["ca_cdr"], color="#1f4e79")
    for k, row in reg.iterrows():
        ax.annotate(k.replace(" Cortex", "").replace(" Gyrus", " G."), (row["neuron_cdr"], row["ca_cdr"]), fontsize=6.5)
    rr = rep["R_across_regions"]["ca_cdr_vs_neuron_cdr"]
    ax.set_xlabel("Neuron score vs CDR (ρ): regional neuron loss with dementia")
    ax.set_ylabel("C-minus-A vs CDR (ρ)")
    ax.set_title(f"Across 17 regions: ρ = {rr['rho']}, p = {rr['p']}", fontsize=10)
    ax = axes[2]
    cd = sorted(subj_cortex["cdr"].dropna().unique())
    ax.boxplot([subj_cortex.loc[subj_cortex["cdr"] == c, "ca"] for c in cd], positions=range(len(cd)), widths=.6)
    ax.set_xticks(range(len(cd)))
    ax.set_xticklabels([str(c) for c in cd])
    s = rep["S_subject_cortex"]["ca_vs_cdr"]
    ax.set_xlabel("Clinical dementia rating")
    ax.set_ylabel("Subject mean C-minus-A (cortical regions)")
    ax.set_title(f"Per subject (n = {rep['S_subject_cortex']['n_subjects']}): ρ = {s['rho']}, p = {s['p']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "msbb_summary.png", dpi=150)


if __name__ == "__main__":
    main()
