"""Figure for SEA-AD reference deconvolution and composition-adjusted re-analysis."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res = Path(sys.argv[1] if len(sys.argv) > 1 else "../results/deconv")
P = pd.read_csv(res / "deconv_proportions.csv", index_col=0)
S = pd.read_csv(res / "deconv_adjustment_summary.csv")
order = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE118553_Frontal_Cortex",
         "GSE118553_Entorhinal_Cortex", "GSE118553_Cerebellum", "GSE125583_FG"]
short = ["33000\nPFC", "132903\nMTG", "118553\nTemporal", "118553\nFrontal", "118553\nEntorhinal", "118553\nCerebellum", "125583\nFusiform"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
ax = axes[0]
cats = [("Control", None, "#9e9e9e"), ("AD", "A", "#2e7d32"), ("AD", "B", "#f9a825"), ("AD", "C", "#c62828")]
w = 0.2
for i, (g, c, col) in enumerate(cats):
    vals = []
    for co in order:
        d = P[(P.cohort == co) & (P.group == g)]
        if c:
            d = d[d.neff_class == c]
        vals.append(d["nnls_Exc"].mean())
    ax.bar(np.arange(len(order)) + (i - 1.5) * w, vals, w, color=col, label=f"{g} class {c}" if c else g)
ax.set_xticks(range(len(order)), short, fontsize=7)
ax.set_ylabel("Excitatory neuron proportion (NNLS, SEA-AD MTG reference)")
ax.set_title("Estimated excitatory neurons by group and Neff class", fontsize=10)
ax.legend(fontsize=7)

ax = axes[1]
covs = [("markers5", "Marker scores", "#9e9e9e"), ("nnls", "SEA-AD NNLS", "#1f4e79"), ("svr", "SEA-AD nu-SVR", "#5b9bd5")]
for i, (k, lab, col) in enumerate(covs):
    d = S[S.covariates == k].set_index("cohort").reindex(order)
    ax.bar(np.arange(len(order)) + (i - 1) * 0.27, d["r2_ca"], 0.27, color=col, label=lab)
ax.set_xticks(range(len(order)), short, fontsize=7)
ax.set_ylim(0, 1)
ax.set_ylabel("R² of C-minus-A axis on composition")
ax.set_title("C-minus-A axis explained by cell composition", fontsize=10)
ax.legend(fontsize=7)

ax = axes[2]
for i, (k, lab, col) in enumerate(covs):
    d = S[S.covariates == k].set_index("cohort").reindex(order)
    ax.bar(np.arange(len(order)) + (i - 1) * 0.22, d["sig_adj"], 0.22, color=col, label=f"Neff, adjusted ({lab})")
d = S[S.covariates == "nnls"].set_index("cohort").reindex(order)
ax.scatter(np.arange(len(order)), d["sig_raw"], color="black", marker="_", s=400, label="Neff, unadjusted", zorder=3)
ax.scatter(np.arange(len(order)), d["sig_adj_rand"], color="red", marker="x", label="Random genes, adjusted (median)", zorder=3)
ax.set_xticks(range(len(order)), short, fontsize=7)
ax.set_ylabel("Mean best correlation to a Neff class centroid")
ax.set_title("Subtype signal left after composition adjustment", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(res / "deconv_summary.png", dpi=150)
