"""Experiment 4b: are fine-mapped loci near cell-type-specific genes?
Gene universe = atlas genes (11,686) with GRCh38 coordinates. For each credible set:
 (a) nearest universe gene to the lead variant;
 (b) locus = any universe gene within +/-100 kb of the CS span.
CS within 500 kb of each other in one trait are merged into one locus for both tests.
Statistic per (trait, cell type): number of loci whose (a) gene / (b) any gene is in that cell type's
top-10% specificity set. Null: 20,000 random genome positions (uniform over gene-containing chromosome spans, CS widths
recycled). (a) binomial vs the null rate; (b) Poisson-binomial with each locus's null rate taken
from random windows holding a similar number of universe genes (GWAS loci sit in gene-dense regions).
Usage: python -I enrich_exp4.py <credible_set_genes.tsv> <atlas_cpm.tsv.gz> <v34_genes.gtf> <out_dir>
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.stats import binom

lf, atl_f, gtf, out = sys.argv[1:5]
loci = pd.read_csv(lf, sep="\t", dtype={"chr": str})
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
spec = atlas.div(atlas.sum(1), axis=0)
k = int(round(0.10 * len(spec)))
top = pd.DataFrame({c: spec.index.isin(spec[c].sort_values(ascending=False).index[:k]) for c in spec.columns},
                   index=spec.index)

g = pd.read_csv(gtf, sep="\t", header=None, usecols=[0, 2, 3, 4, 8])
g = g[g[2] == "gene"]
g["type"] = g[8].str.extract(r'gene_type "([^"]+)"')[0]
g["SYMBOL"] = g[8].str.extract(r'gene_name "([^"]+)"')[0]
g = g[(g.type == "protein_coding") & g.SYMBOL.isin(spec.index)].drop_duplicates("SYMBOL")
g = g.rename(columns={0: "chr", 3: "start", 4: "end"})[["chr", "start", "end", "SYMBOL"]]
g["chr"] = g.chr.str.replace("chr", "")
g = g[g.chr.isin([str(i) for i in range(1, 23)] + ["X"])]
print("universe genes with coordinates", len(g), "of", len(spec))
G = {c: (d.start.to_numpy(), d.end.to_numpy(), top.loc[d.SYMBOL].to_numpy()) for c, d in g.groupby("chr")}
cells = list(top.columns)
print("dropped CS on chromosomes without universe genes:", loci[~loci.chr.isin(list(G))].csid.tolist())
loci = loci[loci.chr.isin(list(G))]


def hits(c, lo, hi, lead):
    """returns (nearest-gene membership vector, any-gene-in-window membership vector)"""
    st, en, T = G[c]
    d = np.where(lead < st, st - lead, np.where(lead > en, lead - en, 0))
    near = T[int(np.argmin(d))]
    w = (en >= lo - 100_000) & (st <= hi + 100_000)
    anyw = T[w].any(0) if w.any() else np.zeros(T.shape[1], bool)
    return near, anyw, int(w.sum())


# null positions: uniform over [first gene start, last gene end] of each chromosome, length-weighted
spans = {c: (v[0].min(), v[1].max()) for c, v in G.items()}
L = np.array([spans[c][1] - spans[c][0] for c in spans]); chrs = list(spans)
rng = np.random.default_rng(2026)
NN = 20_000
cc = rng.choice(len(chrs), NN, p=L / L.sum())
nullN, nullW, nullK = [], [], []
widths = (loci.cs_end - loci.cs_start).to_numpy()
for i in range(NN):
    c = chrs[cc[i]]; a, b = spans[c]
    p = int(rng.integers(a, b)); w = int(widths[i % len(widths)])
    n_, w_, k_ = hits(c, p - w // 2, p + w // 2, p)
    nullN.append(n_); nullW.append(w_); nullK.append(k_)
nullN, nullW, nullK = np.array(nullN), np.array(nullW), np.array(nullK)
pN = nullN.mean(0)
KB = [0, 1, 2, 3, 4, 6, 9, 14, 10**6]   # bins of number of universe genes in the window


def pW_for(k):
    """null hit probability per cell type for a window holding k universe genes (gene-density matched)"""
    b = np.searchsorted(KB, k, side="right") - 1
    m = (nullK >= KB[b]) & (nullK < KB[b + 1])
    return nullW[m].mean(0)


def poibin_sf(x, p):
    """P(X >= x) for a sum of independent Bernoulli(p_i) (exact, by convolution)"""
    f = np.array([1.0])
    for q in p:
        f = np.convolve(f, [1 - q, q])
    return float(f[x:].sum())


res, gene_rows = [], []
for t, d in loci.groupby("trait"):
    d = d.sort_values(["chr", "cs_start"]).copy()
    # merge nearby CS into one locus for the window test
    grp, last = [], (None, -1e12); gi = -1
    for c, s in zip(d.chr, d.cs_start):
        if c != last[0] or s - last[1] > 500_000:
            gi += 1
        grp.append(gi); last = (c, s)
    d["locus"] = grp
    # nearest-gene test: one lead per merged locus (the CS with the smallest p), so that e.g. three CS at NGF count once
    N = np.array([hits(r.chr, r.cs_start, r.cs_end, int(r.lead.split(":")[1]))[0]
                  for r in d.loc[d.groupby("locus").lead_p.idxmin()].itertuples()])
    W, PW = [], []
    for _, x in d.groupby("locus"):
        hh = [hits(r.chr, r.cs_start, r.cs_end, int(r.lead.split(":")[1])) for r in x.itertuples()]
        W.append(np.any([h[1] for h in hh], 0)); PW.append(pW_for(max(h[2] for h in hh)))
    W, PW = np.array(W), np.array(PW)
    nl, nw = len(N), len(W)
    for j, c in enumerate(cells):
        on, ow = int(N[:, j].sum()), int(W[:, j].sum())
        res.append(dict(trait=t, cell=c, n_cs=nl, nearest_obs=on, nearest_exp=round(nl * pN[j], 2),
                        nearest_p=binom.sf(on - 1, nl, pN[j]), n_loci=nw, window_obs=ow,
                        window_exp=round(PW[:, j].sum(), 2), window_p=poibin_sf(ow, PW[:, j])))
    for r in d.itertuples():
        st, en, T = G[r.chr]
        sub = g[(g.chr == r.chr) & (g.end >= r.cs_start - 100_000) & (g.start <= r.cs_end + 100_000)]
        for s in sub.SYMBOL:
            gene_rows.append(dict(trait=t, csid=r.csid, lead_rsid=r.lead_rsid, gene=s, nearest=(s == r.nearest),
                                  **{f"top10_{c}": bool(top.loc[s, c]) for c in ["Ligament_Fibro", "Disc_Fibro", "Disc_Chondro", "Bone_Osteo_stromal", "Bone_Chondro", "DRG_Neuron"]},
                                  lig_spec=round(spec.loc[s, "Ligament_Fibro"], 3),
                                  lig_rank_pct=round(100 * spec.Ligament_Fibro.rank(ascending=False)[s] / len(spec), 1)))
res = pd.DataFrame(res)
res["nearest_fdr_in_trait"] = res.groupby("trait").nearest_p.transform(lambda p: np.minimum(1, p * len(p) / p.rank()))
res["window_fdr_in_trait"] = res.groupby("trait").window_p.transform(lambda p: np.minimum(1, p * len(p) / p.rank()))
res.to_csv(f"{out}/celltype_locus_enrichment.tsv", sep="\t", index=False)
pd.DataFrame(gene_rows).to_csv(f"{out}/locus_genes_specificity.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
for t, d in res.groupby("trait"):
    print(t); print(d.sort_values("window_p").head(6).round(4).to_string(index=False))
