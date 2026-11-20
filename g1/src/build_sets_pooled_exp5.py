"""Experiment 5a follow-up: pooled-neuron gene sets.
With 27 columns, genes shared by all neuron types (brain excitatory/inhibitory, cord, DRG) have their
specificity split four ways and fall out of every neuron top-10% set. Here related columns are merged
(summed CPM, then re-normalised) before computing specificity, and the merged column's top 10% is kept.
Usage: python -I build_sets_pooled_exp5.py <atlas_cpm.tsv.gz> <out.tsv>
"""
import sys
import pandas as pd

atl_f, out = sys.argv[1:3]
atlas = pd.read_csv(atl_f, sep="\t", index_col=0)
POOL = {
    "Neuron_all": ["Brain_ExcNeuron", "Brain_InhNeuron", "Cord_Neuron", "DRG_Neuron"],
    "Brain_Neuron_pooled": ["Brain_ExcNeuron", "Brain_InhNeuron"],
    "CNS_all": [c for c in atlas.columns if c.startswith(("Brain_", "Cord_"))],
    "Fibro_pooled": ["Ligament_Fibro", "Disc_Fibro"],
    "Chondro_pooled": ["Disc_Chondro", "Disc_CEPlike", "Bone_Chondro"],
}
k = int(round(0.10 * len(atlas)))
rows = []
for name, cols in POOL.items():
    a = atlas.drop(columns=cols).assign(**{name: atlas[cols].mean(1)})   # mean of CPM-normalised columns
    spec = a.div(a.sum(1), axis=0)
    rows += [(name, g, 100_000) for g in spec[name].sort_values(ascending=False).index[:k]]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(out, sep="\t", index=False)
print({n: len(c) for n, c in POOL.items()})
