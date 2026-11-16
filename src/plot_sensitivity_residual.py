"""Figure for sensitivity (priority 3) and residual-structure results."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
sens = pd.read_csv(res / "sensitivity/sensitivity_long.csv")
resid = pd.read_csv(res / "residual/residual_summary.csv")
order = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE118553_Frontal_Cortex",
         "GSE118553_Entorhinal_Cortex", "GSE118553_Cerebellum"]
short = ["33000\nPFC", "132903\nMTG", "118553\nTemporal", "118553\nFrontal", "118553\nEntorhinal", "118553\nCerebellum"]
variants = ["baseline", "age_sex", "spearman", "subtype5", "signature", "gene_sub80", "ref_control"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
for ax, col, title, cmap, lim in [(axes[0], "ari_vs_base", "Class agreement with baseline (ARI)", "Greens", (0, 1)),
                                  (axes[1], "neuron_rho", "C-minus-A score vs neuron score (ρ, AD)", "Blues_r", (-1, 0))]:
    m = sens.pivot(index="variant", columns="cohort", values=col).reindex(index=variants, columns=order)
    im = ax.imshow(m.values, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(len(order)), short, fontsize=7)
    ax.set_yticks(range(len(variants)), variants, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)

ax = axes[2]
r = resid.set_index("cohort").reindex(order)
x = np.arange(len(order))
w = 0.2
ax.bar(x - 1.5 * w, r["sig_raw"], w, label="Neff genes, unadjusted", color="#1f4e79")
ax.bar(x - 0.5 * w, r["sig_adj"], w, label="Neff genes, cell-adjusted", color="#5b9bd5")
ax.bar(x + 0.5 * w, r["sig_adj_rand"], w, label="Random genes, adjusted (median)", color="#bbbbbb")
ax.bar(x + 1.5 * w, r["sig_adj_perm"], w, label="Permuted centroids, adjusted (median)", color="#e0e0e0")
ax.set_xticks(x, short, fontsize=7)
ax.set_ylabel("Mean best correlation to a Neff class centroid")
ax.set_title("Subtype signal left after cell adjustment", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(res / "sensitivity_residual_summary.png", dpi=150)
