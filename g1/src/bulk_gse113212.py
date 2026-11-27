"""Limitation 1B: bulk ligamentum flavum microarray GSE113212 (Agilent GPL17077), 4 hypertrophied (LSS, 75-83 y)
vs 4 non-hypertrophied (LDH, 19-24 y). Age and state are fully confounded, so this is a direction check only.
- probe -> gene symbol (GPL table), gene value = mean of probes; Welch t per gene (LSS - LDH)
- gene sets: top 10% of atlas genes by t (up) and bottom 10% (down), +/-100 kb, for LDSC-SEG
- direction check: mean t of the single-cell LSS-state, LDH-state and hypertrophy-up sets from experiment 7a
Usage: python -I bulk_gse113212.py <series_matrix.gz> <GPL.txt> <atlas_cpm.tsv.gz> <exp7_dir> <out_dir>
"""
import sys, os, gzip
import numpy as np, pandas as pd
from scipy import stats

smf, gplf, atl_f, e7, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
lines = gzip.open(smf, "rt").read().splitlines()
src = [l for l in lines if l.startswith("!Sample_source_name_ch1")][0].split("\t")[1:]
lss = np.array(["Stenosis" in s for s in src])
i0 = lines.index("!series_matrix_table_begin") + 1; i1 = lines.index("!series_matrix_table_end")
rows = [l.replace('"', "").split("\t") for l in lines[i0:i1]]
m = pd.DataFrame([r[1:] for r in rows[1:]], index=[r[0] for r in rows[1:]], columns=rows[0][1:]).apply(pd.to_numeric, errors="coerce")
gpl = pd.read_csv(gplf, sep="\t", comment="#", usecols=["ID", "GENE_SYMBOL", "CONTROL_TYPE"], dtype=str)
gpl = gpl[(gpl.CONTROL_TYPE == "FALSE") & gpl.GENE_SYMBOL.notna()]
m = m.join(gpl.set_index("ID")["GENE_SYMBOL"], how="inner").groupby("GENE_SYMBOL").mean()
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
m = m.reindex(atlas.index).dropna(how="all")
t, p = stats.ttest_ind(m.loc[:, lss], m.loc[:, ~lss], axis=1, equal_var=False, nan_policy="omit")
res = pd.DataFrame({"t": t, "p": p, "dmean": m.loc[:, lss].mean(1) - m.loc[:, ~lss].mean(1)}, index=m.index).dropna()
res.to_csv(f"{out}/gse113212_gene_t.tsv", sep="\t")
k = int(round(0.10 * len(atlas)))
up = list(res.t.sort_values(ascending=False).index[:k]); dn = list(res.t.sort_values().index[:k])
gs = [("LF_bulk_up", g, 100_000) for g in up] + [("LF_bulk_down", g, 100_000) for g in dn] + [("all_genes", g, 100_000) for g in atlas.index]
pd.DataFrame(gs, columns=["set", "SYMBOL", "window"]).to_csv(f"{out}/genesets_bulk.tsv", sep="\t", index=False)
# direction check against the single-cell sets of experiment 7a
chk = []
for f in [f"{e7}/genesets_exp7.tsv", f"{e7}/r2/genesets_exp7.tsv", f"{e7}/r4/genesets_exp7.tsv"]:
    if not os.path.exists(f): continue
    g = pd.read_csv(f, sep="\t")
    for s, d in g[g.set != "all_genes"].groupby("set"):
        tt = res.t.reindex(d.SYMBOL).dropna()
        u = stats.mannwhitneyu(tt, res.t.drop(tt.index), alternative="two-sided")
        chk.append(dict(set=s, n=len(tt), mean_t=tt.mean(), frac_t_pos=(tt > 0).mean(), mwu_p=u.pvalue,
                        overlap_bulk_up=len(set(d.SYMBOL) & set(up)) / k))
chk = pd.DataFrame(chk).drop_duplicates("set")
chk.to_csv(f"{out}/direction_check.tsv", sep="\t", index=False)
print("genes", len(res), "LSS n", lss.sum(), "LDH n", (~lss).sum())
print(chk.round(4).to_string(index=False))
for g in ["CRLF1", "MMP14", "BMP6", "SOX9", "NGF", "COL1A1", "TNC"]:
    if g in res.index: print(g, res.loc[g].round(3).to_dict())
print("genes nominal p<0.05:", int((res.p < 0.05).sum()), "of", len(res))
