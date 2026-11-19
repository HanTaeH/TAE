"""Collect ldsc --h2-cts results into long and wide tables (BH FDR within trait).
Usage: python summarise_exp3.py <h2cts_dir> <out_dir>"""
import sys, glob, os
import numpy as np
import pandas as pd
d_in, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
rows = []
for f in sorted(glob.glob(f"{d_in}/*.cell_type_results.txt")):
    d = pd.read_csv(f, sep="\t"); d["trait"] = os.path.basename(f).replace(".cell_type_results.txt", ""); rows.append(d)
d = pd.concat(rows, ignore_index=True)
d["z"] = d.Coefficient / d.Coefficient_std_error
d["FDR"] = np.nan
for t, g in d.groupby("trait"):
    p = g.Coefficient_P_value.to_numpy(); o = np.argsort(p); m = len(p)
    q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    d.loc[g.index, "FDR"] = np.minimum(q, 1)
d.to_csv(f"{out}/h2cts_long.tsv", sep="\t", index=False)
w = d.pivot(index="Name", columns="trait", values="Coefficient_P_value")
w.to_csv(f"{out}/h2cts_p_wide.tsv", sep="\t", float_format="%.4g")
pd.set_option("display.width", 300)
print((-np.log10(w)).round(1).to_string())
print(d[d.Coefficient_P_value < 0.01].sort_values("Coefficient_P_value")[["trait", "Name", "z", "Coefficient_P_value", "FDR"]].to_string(index=False))
