"""Atlas v2 gene sets: 27 cell types + all_genes (build_atlas_v2.py), pooled sets
(build_sets_pooled_exp5.py) and 20 random sets of the top-10% size from the v2 universe (seed 2026).
Usage: python -I combine_sets_v2.py <genesets.tsv> <pooled.tsv> <atlas_cpm.tsv.gz> <out.tsv>"""
import sys
import numpy as np
import pandas as pd
gs, po, atl, out = sys.argv[1:5]
a = pd.read_csv(gs, sep="\t").assign(window=100_000)
p = pd.read_csv(po, sep="\t")
u = pd.read_csv(atl, sep="\t", index_col=0).index.to_numpy()
k = int(round(0.10 * len(u))); rng = np.random.default_rng(2026)
r = pd.concat([pd.DataFrame(dict(set=f"random{i:02d}", SYMBOL=rng.choice(u, k, replace=False), window=100_000)) for i in range(20)])
allg = a[a.set == "all_genes"]
pd.concat([a[a.set != "all_genes"], p, r, allg]).to_csv(out, sep="\t", index=False)
print("sets", pd.concat([a, p, r]).set.nunique(), "k", k)
