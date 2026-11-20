"""Experiment 5 summary of LDSC-SEG runs (control traits, lumbar prolapse, BMI/height-adjusted stenosis):
z per set, p, FDR across the 27 main cell types within trait, empirical p vs the 20 random sets.
Usage: python -I summarise_exp5.py <out_dir> <h2cts_dir> [<h2cts_dir> ...]
"""
import sys, os, glob
import numpy as np
import pandas as pd

out = sys.argv[1]; dirs = sys.argv[2:]
d = pd.concat([pd.read_csv(f, sep="\t").assign(trait=os.path.basename(f).replace(".cell_type_results.txt", ""))
               for h in dirs for f in sorted(glob.glob(f"{h}/*.cell_type_results.txt"))], ignore_index=True)
d["z"] = d.Coefficient / d.Coefficient_std_error
rnd = d[d.Name.str.startswith("random")].groupby("trait").z.apply(np.array)
d["emp_p_vs_random"] = [(1 + (rnd[t] >= z).sum()) / (1 + len(rnd[t])) for t, z in zip(d.trait, d.z)]
main = ~d.Name.str.startswith("random") & ~d.Name.str.startswith("DRG_")


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r


d["fdr27"] = np.nan
for t, x in d[main].groupby("trait"):
    d.loc[x.index, "fdr27"] = bh(x.Coefficient_P_value)
d.to_csv(f"{out}/h2cts_exp5_long.tsv", sep="\t", index=False)
d.pivot(index="Name", columns="trait", values="z").round(2).to_csv(f"{out}/h2cts_exp5_z_wide.tsv", sep="\t")
pd.set_option("display.width", 250)
for t, x in d[main | d.Name.str.startswith("DRG_")].groupby("trait"):
    print(t, "random z mean/sd/max", np.round([rnd[t].mean(), rnd[t].std(), rnd[t].max()], 2))
    print(x.sort_values("Coefficient_P_value").head(5)[["Name", "z", "Coefficient_P_value", "fdr27", "emp_p_vs_random"]].round(4).to_string(index=False))
