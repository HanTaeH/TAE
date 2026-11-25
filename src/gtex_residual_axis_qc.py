"""What else explains the replicated residual axis? GTEx v8 brain cortex and BA9 (follow-up, 2026-11-25).

gtex_residual_axis.py found that RIN, ischaemic time, Hardy class, age and sex explain only 10-17% of the
axis. Three further angles:
 1 Sample level, all recorded variables. RNA-seq quality metrics from the sample attributes (intronic,
   exonic, intergenic, rRNA and mapping rates, expression efficiency, chimeric and split reads, base
   mismatch rate, total reads), autolysis score, RIN, ischaemic time, collection centre, expression batch,
   Hardy class, age, sex. In-sample R^2 and 5-fold cross-validated R^2 (ridge) for variable blocks and all.
 2 Donor vs sample. Donors with both tissues: Spearman of the axis score between cortex and BA9, raw and
   after removing RIN and the RNA-seq metrics from each. A donor-level cause (agonal state, systemic
   condition) makes the two agree; a sample-level cause (dissection, handling) does not.
 3 Gene level. Template weights vs (a) each gene's residual correlation with RIN, intronic rate, rRNA rate,
   ischaemic time, ventilator death, age (the gene-wise signature of each variable); (b) gene length,
   exonic length and exon number (GENCODE v26), which mark RNA degradation.

Usage: python gtex_residual_axis_qc.py --gtex /mnt/project-files/d1-data/raw/gtex_v8 \
         --weights ../results/residual_quality/residual_template_weights.csv --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gtex_residual_axis import TISSUES, load_counts, r2
from null_models import z_np
from residual_quality import resid, scores

SEQ = ["SMNTRART", "SMEXNCRT", "SMNTERRT", "SMRRNART", "SMMAPRT", "SMEXPEFF", "SMCHMPRS", "SMSPLTRD", "SME1MMRT", "SMRDTTL"]


def cv_r2(y, X, seed=0):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30)))
    p = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def dummies(s):
    s = pd.Series(s).astype(str)
    keep = s.map(s.value_counts()) >= 5
    return pd.get_dummies(s.where(keep, "other"), drop_first=True).values.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtex", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(a.weights, index_col=0).iloc[:, 0]
    sa = pd.read_csv(a.gtex / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t", index_col=0, low_memory=False)
    ph = pd.read_csv(a.gtex / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t", index_col=0)
    gl = pd.read_csv(a.gtex / "gene_lengths_gencode_v26.csv", index_col=0)
    out, donor_scores = {}, {}
    for tname, fn in TISSUES.items():
        e = load_counts(a.gtex / fn)
        s = sa.loc[e.columns]
        subj = pd.Index(["-".join(x.split("-")[:2]) for x in e.columns])
        m = s[["SMRIN", "SMTSISCH", "SMATSSCR"] + SEQ].copy()
        m["hardy"] = ph.reindex(subj)["DTHHRDY"].values
        m["age"] = ph.reindex(subj)["AGE"].str[:2].astype(float).values + 5
        m["sex"] = (ph.reindex(subj)["SEX"].values == 1).astype(float)
        R, g = resid(e, e.columns, w.index)
        sc = scores(R, g, w)["template"]
        res = {"n": int(e.shape[1]), "seq_metrics_available": int(m[SEQ].notna().all(axis=1).sum()),
               "autolysis_available": int(m["SMATSSCR"].notna().sum())}
        # 1 sample level
        ok = m[["SMRIN", "SMTSISCH", "hardy"] + SEQ].notna().all(axis=1).values
        mm, y = m[ok], sc[ok]
        blocks = {"death": np.column_stack([mm[["SMTSISCH", "age", "sex"]].values, dummies(mm["hardy"].astype(int))]),
                  "RIN": mm[["SMRIN"]].values,
                  "seq_qc": mm[SEQ].values,
                  "centre_batch": np.column_stack([dummies(s["SMCENTER"].values[ok]), dummies(s["SMGEBTCH"].values[ok])])}
        allX = np.column_stack(list(blocks.values()))
        res["sample_level_n"] = int(ok.sum())
        res["r2_in_sample"] = {k: round(r2(y, X), 3) for k, X in blocks.items()} | {"all": round(r2(y, allX), 3)}
        res["r2_cv"] = {k: round(cv_r2(y, X), 3) for k, X in blocks.items()} | {"all": round(cv_r2(y, allX), 3)}
        res["rho_seq_metrics"] = {c: round(float(stats.spearmanr(y, mm[c])[0]), 3) for c in SEQ + ["SMRIN"]}
        ok2 = m["SMATSSCR"].notna().values
        res["rho_autolysis"] = [round(float(x), 4) for x in stats.spearmanr(sc[ok2], m["SMATSSCR"].values[ok2])]
        # residual after RIN + seq metrics, for the donor comparison
        Xq = np.column_stack([np.ones(ok.sum()), mm[["SMRIN"] + SEQ].values])
        adj = pd.Series(np.nan, index=subj)
        adj[ok] = y - Xq @ np.linalg.lstsq(Xq, y, rcond=None)[0]
        donor_scores[tname] = (pd.Series(sc, index=subj), adj)
        # 3 gene level
        Rz = z_np(R)
        gene = {}
        ventil = np.where(np.isin(m["hardy"].values, [0]), 1.0, np.where(np.isin(m["hardy"].values, [1, 2]), 0.0, np.nan))
        for lab, v in {"RIN": m["SMRIN"].values, "intronic_rate": m["SMNTRART"].values, "rRNA_rate": m["SMRRNART"].values,
                       "ischaemic_time": m["SMTSISCH"].values, "ventilator_vs_fast": ventil, "age": m["age"].values}.items():
            okv = ~pd.isna(v)
            vv = stats.rankdata(np.asarray(v, float)[okv])
            gc = np.array([stats.pearsonr(stats.rankdata(Rz[i, okv]), vv)[0] for i in range(len(g))])
            gene[lab] = round(float(np.corrcoef(gc, w.reindex(g).values)[0, 1]), 3)
        res["template_vs_gene_signature_r"] = gene
        out[tname] = res
        print(tname, json.dumps(res), flush=True)
    # 2 donor vs sample
    (c_raw, c_adj), (b_raw, b_adj) = donor_scores["cortex"], donor_scores["frontal_BA9"]
    both = c_raw.index.intersection(b_raw.index)
    both = both[~both.duplicated()]
    rr = stats.spearmanr(c_raw.groupby(level=0).mean()[both], b_raw.groupby(level=0).mean()[both])
    ca, ba = c_adj.groupby(level=0).mean()[both], b_adj.groupby(level=0).mean()[both]
    okd = ca.notna() & ba.notna()
    ra = stats.spearmanr(ca[okd], ba[okd])
    out["donor_cortex_vs_BA9"] = {"n_donors": int(len(both)), "rho_raw": [round(float(rr[0]), 3), float(f"{rr[1]:.3g}")],
                                  "n_adj": int(okd.sum()), "rho_after_RIN_seqQC": [round(float(ra[0]), 3), float(f"{ra[1]:.3g}")]}
    # gene length
    gg = w.index.intersection(gl.index)
    out["template_vs_gene_features_spearman"] = {c: round(float(stats.spearmanr(w[gg], np.log10(gl.loc[gg, c].astype(float)))[0]), 3)
                                                 for c in ("gene_len", "exon_len", "n_exon")}
    out["n_genes_with_length"] = int(len(gg))
    print(json.dumps({k: out[k] for k in ("donor_cortex_vs_BA9", "template_vs_gene_features_spearman")}))
    (a.out / "gtex_residual_axis_qc.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
