"""Limitation 3: the stenosis-specific component (exp 2) has weak cell enrichment (fibro pooled z 1.19).
Pool all structural columns into one set (more power for a weak component) and, for contrast, all neurons.
Same rule as exp 5: mean of CPM-normalised member columns, then top 10% specificity, +/-100 kb.
Usage: python -I build_sets_lim3.py <atlas_cpm.tsv.gz> <out.tsv>
"""
import sys
import pandas as pd

atl_f, out = sys.argv[1:3]
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
POOL = {
    "Structural_pooled": ["Ligament_Fibro", "Disc_Fibro", "Disc_Chondro", "Disc_CEPlike", "Bone_Chondro", "Bone_Osteo_stromal"],
    "Fibro_pooled": ["Ligament_Fibro", "Disc_Fibro"],
    "Brain_Neuron_pooled": ["Brain_ExcNeuron", "Brain_InhNeuron"],
    "Neuron_all": ["Brain_ExcNeuron", "Brain_InhNeuron", "Cord_Neuron", "DRG_Neuron"],
}
k = int(round(0.10 * len(atlas)))
rows = []
for name, cols in POOL.items():
    a = atlas.drop(columns=cols).assign(**{name: atlas[cols].mean(1)})
    spec = a.div(a.sum(1), axis=0)
    rows += [(name, g, 100_000) for g in spec[name].sort_values(ascending=False).index[:k]]
rows += [("all_genes", g, 100_000) for g in atlas.index]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(out, sep="\t", index=False)
print({n: len(c) for n, c in POOL.items()}, "k", k)
