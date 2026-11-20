"""Experiment 4a: FinnGen R12 SuSiE credible sets -> candidate genes.
For every credible set (CS, low_purity 0):
 - lead variant = highest PIP
 - nearest protein-coding gene to the lead (GENCODE v34, GRCh38, distance to gene body)
 - all protein-coding genes whose body lies within +/-100 kb of any CS variant (locus genes)
 - GTEx v10 SuSiE eQTL support: CLPP = sum over shared variants of PIP_gwas * PIP_eqtl,
   per (tissue, gene); kept if CLPP >= 0.01
Usage: python -I map_genes_exp4.py <fm_dir> <v34_genes.gtf> <gtex_dir> <out_dir>
"""
import sys, os, glob, re
import numpy as np
import pandas as pd

fm, gtf, gtex, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
TRAITS = ["M13_SPINSTENOSIS", "M13_SCIATICA", "M13_SPONDYLOSIS", "M13_SPONDYLOLISTHESIS", "M13_DORSALGIA"]
WIN = 100_000

g = pd.read_csv(gtf, sep="\t", header=None, usecols=[0, 2, 3, 4, 8])
g = g[g[2] == "gene"]
g["type"] = g[8].str.extract(r'gene_type "([^"]+)"')[0]
g["SYMBOL"] = g[8].str.extract(r'gene_name "([^"]+)"')[0]
g = g[g.type == "protein_coding"].rename(columns={0: "chr", 3: "start", 4: "end"})[["chr", "start", "end", "SYMBOL"]]
g["chr"] = g.chr.str.replace("chr", "")
g = g.drop_duplicates("SYMBOL")
G = {c: d.reset_index(drop=True) for c, d in g.groupby("chr")}


def dist(c, pos):
    d = G.get(str(c))
    if d is None:
        return None
    return np.where(pos < d.start, d.start - pos, np.where(pos > d.end, pos - d.end, 0)), d


cs = pd.concat([pd.read_csv(f"{fm}/{t}.cs.tsv", sep="\t") for t in TRAITS])
cs["chr"] = cs.chromosome.astype(str).str.replace("chr", "").replace({"23": "X"})
cs["csid"] = cs.trait + "|" + cs.region + "|" + cs.cs.astype(str)
cs["key"] = "chr" + cs.v.str.replace(":", "_")

loci = []
for cid, d in cs.groupby("csid", sort=False):
    lead = d.loc[d.prob.idxmax()]
    dd, gd = dist(lead.chr, lead.position)
    near = gd.SYMBOL[int(np.argmin(dd))]
    lo, hi = d.position.min() - WIN, d.position.max() + WIN
    lg = gd[(gd.end >= lo) & (gd.start <= hi)].SYMBOL.tolist()
    loci.append(dict(csid=cid, trait=lead.trait, chr=lead.chr, lead=lead.v, lead_rsid=lead.rsid, lead_pip=lead.prob,
                     lead_p=lead.p, lead_beta=lead.beta, cs_size=len(d), cs_start=d.position.min(), cs_end=d.position.max(),
                     nearest=near, nearest_dist=int(dd.min()), locus_genes=",".join(lg)))
loci = pd.DataFrame(loci)

# GTEx SuSiE eQTL overlap
pip = cs.set_index("key")
keys = set(cs.key)
rows = []
for f in sorted(glob.glob(f"{gtex}/*.parquet")):
    tis = os.path.basename(f).split(".v10")[0]
    e = pd.read_parquet(f, columns=["gene_name", "biotype", "variant_id", "pip", "cs_id"])
    e["key"] = e.variant_id.str.replace("_b38", "")
    e = e[e.key.isin(keys)]
    if e.empty:
        continue
    m = e.merge(cs[["key", "csid", "prob"]], on="key")
    m["clpp"] = m.pip * m.prob
    s = m.groupby(["csid", "gene_name", "biotype"]).clpp.sum().reset_index()
    s["tissue"] = tis
    rows.append(s[s.clpp >= 0.01])
eq = pd.concat(rows).sort_values("clpp", ascending=False)
eq.to_csv(f"{out}/eqtl_clpp.tsv", sep="\t", index=False)
best = (eq.groupby(["csid", "gene_name"]).agg(max_clpp=("clpp", "max"), n_tissues=("tissue", "nunique"),
        fibroblast=("tissue", lambda x: "Cells_Cultured_fibroblasts" in set(x))).reset_index())
top = best.sort_values("max_clpp", ascending=False).groupby("csid").head(3)
summ = top.groupby("csid").apply(lambda x: ";".join(f"{a}({b:.2f},{c}t{',FB' if f else ''})" for a, b, c, f in
                                  zip(x.gene_name, x.max_clpp, x.n_tissues, x.fibroblast)), include_groups=False)
loci["eqtl_genes"] = loci.csid.map(summ).fillna("")
loci.to_csv(f"{out}/credible_set_genes.tsv", sep="\t", index=False)
best.to_csv(f"{out}/eqtl_best_per_gene.tsv", sep="\t", index=False)
print(loci.groupby("trait").size())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(loci[loci.trait == "M13_SPINSTENOSIS"][["chr", "lead_rsid", "lead_p", "cs_size", "nearest", "nearest_dist", "eqtl_genes"]].to_string(index=False))
