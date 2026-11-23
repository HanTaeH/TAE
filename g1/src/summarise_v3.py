"""Atlas v3 LDSC-SEG summary + structure-vs-pain dissociation test.
 - long/wide tables; FDR over the 32 base sets (27 cell types + 5 pooled) within trait; empirical p vs 20 random sets
 - dissociation: for stenosis (STEN) vs dorsalgia (DORS) in the same cohort,
     D = [z(Fibro, STEN) - z(Fibro, DORS)] + [z(Neuron, DORS) - z(Neuron, STEN)],  Z_D = D / 2
   (Fibro = Fibro_pooled, Neuron = Brain_Neuron_pooled). Treating the four z as independent is conservative
   because sample overlap between the two traits makes their estimates positively correlated.
   Independent cohorts (FinnGen, MVP EUR, UKBB) are combined by Stouffer; EUR meta shown separately.
Usage: python -I summarise_v3.py <h2cts_dir> <out_dir>
"""
import sys, os, glob
import numpy as np
import pandas as pd
from scipy.stats import norm

h, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
d = pd.concat([pd.read_csv(f, sep="\t").assign(trait=os.path.basename(f).replace(".cell_type_results.txt", ""))
               for f in sorted(glob.glob(f"{h}/*.cell_type_results.txt"))], ignore_index=True)
d["z"] = d.Coefficient / d.Coefficient_std_error
rnd = d[d.Name.str.startswith("random")].groupby("trait").z.apply(np.array)
d["emp_p_vs_random"] = [(1 + (rnd[t] >= z).sum()) / (1 + len(rnd[t])) for t, z in zip(d.trait, d.z)]
base = ~d.Name.str.startswith("random") & ~d.Name.str.contains("__")


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r


d["fdr_base32"] = np.nan
for t, x in d[base].groupby("trait"):
    d.loc[x.index, "fdr_base32"] = bh(x.Coefficient_P_value)
d.to_csv(f"{out}/h2cts_v3_long.tsv", sep="\t", index=False)
d.pivot(index="Name", columns="trait", values="z").round(2).to_csv(f"{out}/h2cts_v3_z_wide.tsv", sep="\t")
d.pivot(index="Name", columns="trait", values="Coefficient_P_value").to_csv(f"{out}/h2cts_v3_p_wide.tsv", sep="\t")

Z = d.pivot(index="trait", columns="Name", values="z")
rows = []
for coh, s, p in [("EURmeta", "EURmeta.SPINSTENOSIS", "EURmeta.DORSALGIA"), ("FinnGen", "fg.SPINSTENOSIS", "fg.DORSALGIA"),
                  ("MVP_EUR", "MVP_EUR.SPINSTENOSIS", "MVP_EUR.DORSALGIA"), ("UKBB", "ukbb.SPINSTENOSIS", "ukbb.DORSALGIA")]:
    if s not in Z.index or p not in Z.index:
        continue
    for fib, neu in [("Fibro_pooled", "Brain_Neuron_pooled"), ("Fibro_pooled__w50", "Brain_Neuron_pooled__w50"),
                     ("Ligament_Fibro", "Brain_InhNeuron")]:
        D = (Z.loc[s, fib] - Z.loc[p, fib]) + (Z.loc[p, neu] - Z.loc[s, neu])
        rows.append(dict(cohort=coh, fibro_set=fib, neuron_set=neu, z_fib_sten=Z.loc[s, fib], z_fib_dors=Z.loc[p, fib],
                         z_neu_sten=Z.loc[s, neu], z_neu_dors=Z.loc[p, neu], Z_D=D / 2, p_D=norm.sf(D / 2)))
R = pd.DataFrame(rows)
for fib in R.fibro_set.unique():
    x = R[(R.fibro_set == fib) & R.cohort.isin(["FinnGen", "MVP_EUR", "UKBB"])]
    if len(x):
        zc = x.Z_D.sum() / np.sqrt(len(x))
        R = pd.concat([R, pd.DataFrame([dict(cohort="Stouffer(FG,MVP,UKBB)", fibro_set=fib, neuron_set=x.neuron_set.iloc[0],
                                             Z_D=zc, p_D=norm.sf(zc))])], ignore_index=True)
R.to_csv(f"{out}/dissociation_test.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False))
