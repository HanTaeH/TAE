"""Experiment 5d: do FinnGen R12 LoF burden signals concentrate in cell-type-specific genes?
Per trait, gene = best mask (largest CHISQ) among that gene's masks; universe = atlas genes with a
LoF test. For each cell type's top-10% specificity set: mean chi-square inside vs outside, one-sided
Mann-Whitney p, and count of genes with p < 0.001. Positive controls: height, hip/knee OA.
Usage: python -I lof_celltype_exp5.py <finngen_R12_lof.txt.gz> <atlas_cpm.tsv.gz> <out_dir>
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

lf, atl_f, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
TRAITS = ["M13_SPINSTENOSIS", "M13_SCIATICA", "M13_SPONDYLOSIS", "M13_SPONDYLOLISTHESIS", "M13_DORSALGIA",
          "M13_INTERVERTEB", "M13_LUMBAR_PROLAPSE", "M13_LOWBACKPAIN",
          "HEIGHT_IRN", "M13_ARTHROSIS_KNEE_PRIM_KNEESURG", "M13_ARTHTROSIS_COX_PRIM_HIPSURG", "F5_DEPRESSIO", "BMI_IRN"]
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
spec = atlas.div(atlas.sum(1), axis=0)
k = int(round(0.10 * len(spec)))
top = {c: set(spec[c].sort_values(ascending=False).index[:k]) for c in spec.columns}
parts = []
for ch in pd.read_csv(lf, sep="\t", usecols=["PHENO", "ID", "CHISQ", "LOG10P", "BETA"], chunksize=2_000_000):
    parts.append(ch[ch.PHENO.isin(TRAITS)])
d = pd.concat(parts)
d["gene"] = d.ID.str.split(".Mask").str[0]
d = d.sort_values("CHISQ", ascending=False).drop_duplicates(["PHENO", "gene"])
d = d[d.gene.isin(spec.index)]
rows = []
for t, x in d.groupby("PHENO"):
    for c, s in top.items():
        m = x.gene.isin(s)
        a, b = x.CHISQ[m], x.CHISQ[~m]
        rows.append(dict(trait=t, cell=c, n_in=int(m.sum()), n_out=int((~m).sum()), mean_chisq_in=a.mean(),
                         mean_chisq_out=b.mean(), ratio=a.mean() / b.mean(),
                         mw_p=mannwhitneyu(a, b, alternative="greater").pvalue,
                         n_p001_in=int((x.LOG10P[m] > 3).sum()), genes_p001=",".join(x.gene[m & (x.LOG10P > 3)])))
r = pd.DataFrame(rows)
r["fdr_in_trait"] = r.groupby("trait").mw_p.transform(lambda p: np.minimum(1, (p * len(p) / p.rank()).values))
r.to_csv(f"{out}/lof_celltype.tsv", sep="\t", index=False)
top_genes = d[d.LOG10P > 3].sort_values(["PHENO", "LOG10P"], ascending=[True, False])
top_genes.to_csv(f"{out}/lof_genes_p001.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
for t, x in r.groupby("trait"):
    print(t, "genes", x.n_in.iloc[0] + x.n_out.iloc[0])
    print(x.sort_values("mw_p").head(3)[["cell", "ratio", "mw_p", "fdr_in_trait", "genes_p001"]].round(4).to_string(index=False))
