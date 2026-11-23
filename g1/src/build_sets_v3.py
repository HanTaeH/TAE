"""Atlas v3 gene sets in one file (set, SYMBOL, window):
 - 27 cell types, top 10% specificity, +/-100 kb
 - pooled sets (neurons, brain neurons, CNS, fibroblasts, chondrocytes), top 10%
 - sensitivity for key sets: top 5% and top 20% (+/-100 kb) and top 10% at +/-50 kb
 - 20 random sets of the top-10% size (seed 2026)
 - all_genes (+/-100 kb) and all_genes__w50 (+/-50 kb) controls
Usage: python -I build_sets_v3.py <atlas_cpm.tsv.gz> <out.tsv>
"""
import sys
import numpy as np
import pandas as pd

atl, out = sys.argv[1:3]
atlas = pd.read_csv(atl, sep="\t", index_col=0)
POOL = {
    "Neuron_all": ["Brain_ExcNeuron", "Brain_InhNeuron", "Cord_Neuron", "DRG_Neuron"],
    "Brain_Neuron_pooled": ["Brain_ExcNeuron", "Brain_InhNeuron"],
    "CNS_all": [c for c in atlas.columns if c.startswith(("Brain_", "Cord_"))],
    "Fibro_pooled": ["Ligament_Fibro", "Disc_Fibro"],
    "Chondro_pooled": ["Disc_Chondro", "Disc_CEPlike", "Bone_Chondro"],
}
KEY = ["Ligament_Fibro", "Disc_Fibro", "Fibro_pooled", "Chondro_pooled", "Disc_Chondro", "Bone_Chondro",
       "Brain_Neuron_pooled", "Brain_InhNeuron", "Brain_ExcNeuron", "DRG_Neuron", "Cord_Meningeal"]
n = len(atlas)


def spec_of(name):
    if name in POOL:
        a = atlas.drop(columns=POOL[name]).assign(**{name: atlas[POOL[name]].mean(1)})
    else:
        a = atlas
    return a.div(a.sum(1), axis=0)[name].sort_values(ascending=False)


rows = []
for c in list(atlas.columns) + list(POOL):
    s = spec_of(c)
    rows += [(c, g, 100_000) for g in s.index[: int(round(0.10 * n))]]
    if c in KEY:
        rows += [(f"{c}__top5", g, 100_000) for g in s.index[: int(round(0.05 * n))]]
        rows += [(f"{c}__top20", g, 100_000) for g in s.index[: int(round(0.20 * n))]]
        rows += [(f"{c}__w50", g, 50_000) for g in s.index[: int(round(0.10 * n))]]
rng = np.random.default_rng(2026); k = int(round(0.10 * n))
for i in range(20):
    rows += [(f"random{i:02d}", g, 100_000) for g in rng.choice(atlas.index.to_numpy(), k, replace=False)]
rows += [("all_genes__w50", g, 50_000) for g in atlas.index]
rows += [("all_genes", g, 100_000) for g in atlas.index]
pd.DataFrame(rows, columns=["set", "SYMBOL", "window"]).to_csv(out, sep="\t", index=False)
print("sets", len(dict.fromkeys(r[0] for r in rows)), "genes", n)
