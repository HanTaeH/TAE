"""Figure for the first-pass multi-cohort validation (reads validation_report.json)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
reps = json.loads((out / "validation_report.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
names = [r["cohort"].replace("GSE118553_", "118553\n").replace("_Cortex", "").replace("_", "\n") for r in reps]
x = np.arange(len(reps))
for i, k in enumerate("ABC"):
    ax[0].bar(x + (i - 1) * 0.27, [r["neuron_by_class"].get(k, np.nan) for r in reps], 0.27, label=k)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xticks(x, names, fontsize=8)
ax[0].set(title="Neuronal marker score by projected class (AD only)", ylabel="mean z")
ax[0].legend(title="class")
ax[1].bar(x - 0.2, [r["median_r2_cell_scores"] for r in reps], 0.4, label="variance explained by cell scores (median R²)")
ax[1].bar(x + 0.2, [r["adj_ari_vs_unadjusted"] for r in reps], 0.4, label="agreement after cell adjustment (ARI)")
ax[1].set_xticks(x, names, fontsize=8)
ax[1].set(title="Cell-composition dependence", ylim=(-0.05, 0.7))
ax[1].legend(fontsize=8)
groups = ["Control", "AsymAD", "AD"]
w = 0.27
for j, g in enumerate(groups):
    ax[2].bar(x + (j - 1) * w, [r["all_sample_class_by_group"].get(g, {}).get("A", np.nan) for r in reps], w, label=g)
ax[2].set_xticks(x, names, fontsize=8)
ax[2].set(title="Fraction projected to class A, by diagnosis", ylabel="fraction A")
ax[2].legend()
for i, r in enumerate(reps):
    p = r["stage_trend_spearman"]["p"]
    ax[2].text(i, 0.82, f"ρ={r['stage_trend_spearman']['rho']:.2f}\n{'*' if p < 0.05 else 'ns'}", ha="center", fontsize=7)
ax[2].set_ylim(0, 0.95)
fig.tight_layout()
fig.savefig(out / "validation_summary.png", dpi=150)
