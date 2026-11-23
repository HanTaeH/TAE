"""Experiment 5f: which fine-mapped candidate genes are drug targets (DGIdb 2026-10)?
Candidate genes per credible set: nearest protein-coding gene, genes with GTEx eQTL CLPP >= 0.05, and
locus genes (+/-100 kb) that are in a key cell type's top-10% set (atlas v3).
Output: one row per (trait, CS, gene) with approved drugs (name, interaction type), and a gene summary.
Usage: python -I drug_targets_exp5.py <interactions.tsv> <credible_set_genes.tsv> <eqtl_best_per_gene.tsv>
       <locus_genes_specificity.tsv> <atlas_cpm.tsv.gz> <out_dir>
"""
import sys, os
import pandas as pd

dg_f, cs_f, eq_f, lg_f, atl_f, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
dg = pd.read_csv(dg_f, sep="\t", comment="#", low_memory=False)
dg = dg[dg.drug_name.notna()]
cs = pd.read_csv(cs_f, sep="\t")
eq = pd.read_csv(eq_f, sep="\t")
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
spec = atlas.div(atlas.sum(1), axis=0)
k = int(round(0.10 * len(spec)))
KEY = ["Ligament_Fibro", "Disc_Fibro", "Disc_Chondro", "Disc_CEPlike", "Bone_Chondro", "Bone_Osteo_stromal",
       "Brain_ExcNeuron", "Brain_InhNeuron", "DRG_Neuron", "Cord_Neuron"]
top = {c: set(spec[c].sort_values(ascending=False).index[:k]) for c in KEY}

rows = []
for r in cs.itertuples():
    cand = {r.nearest: "nearest"}
    for g in eq[(eq.csid == r.csid) & (eq.max_clpp >= 0.05)].gene_name:
        cand[g] = cand.get(g, "") + "+eQTL" if g in cand else "eQTL"
    for g in str(r.locus_genes).split(","):
        cells = [c for c in KEY if g in top[c]]
        if cells:
            cand[g] = (cand.get(g, "") + "+" if g in cand else "") + "cell:" + "/".join(cells)
    for g, why in cand.items():
        d = dg[dg.gene_name == g]
        appr = d[d.drug_is_approved == True]
        rows.append(dict(trait=r.trait, csid=r.csid, lead=r.lead_rsid, lead_p=r.lead_p, gene=g, evidence=why,
                         n_drugs=d.drug_name.nunique(), n_approved=appr.drug_name.nunique(),
                         approved_drugs="; ".join(sorted(appr.drug_name.str.lower().unique())[:12]),
                         interaction_types="; ".join(sorted(set(";".join(appr.interaction_types.dropna()).split(";")) - {""}))))
R = pd.DataFrame(rows)
R.to_csv(f"{out}/drug_targets_by_locus.tsv", sep="\t", index=False)
G = (R.groupby("gene").agg(traits=("trait", lambda x: ",".join(sorted(set(t.replace("M13_", "") for t in x)))),
                           evidence=("evidence", lambda x: ";".join(sorted(set(x)))), best_p=("lead_p", "min"),
                           n_approved=("n_approved", "max"), approved_drugs=("approved_drugs", "first"),
                           interaction_types=("interaction_types", "first"))
     .reset_index().sort_values(["n_approved", "best_p"], ascending=[False, True]))
G.to_csv(f"{out}/drug_targets_genes.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print("candidate genes", len(G), "with approved drug", int((G.n_approved > 0).sum()))
print(G[G.n_approved > 0].head(40).to_string(index=False))
