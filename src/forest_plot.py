"""Forest plot of the main effects across cohorts / regions (Figure candidate).

Effects (Spearman ρ with Fisher-z 95% CI, se = 1.06 / sqrt(n - 3)):
  1. C-minus-A score vs neuron marker score (AD samples)            validation projection
  2. Diagnosis stage trend of C-minus-A score (Control<AsymAD<AD)  null-model run (observed values)
  3. Braak stage trend of C-minus-A score (GSE106241, all donors)
Random-effects (DerSimonian-Laird) pooled estimates use one entry per independent cohort:
GSE33000 PFC, GSE132903 MTG, GSE118553 temporal cortex (+ GSE106241 for stage). Cerebellum is shown
as the negative control and never pooled.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

res = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
proj = pd.read_csv(res / "validation/projection_all_cohorts.csv", index_col=0)
val = {r["cohort"]: r for r in json.loads((res / "validation/validation_report.json").read_text())}
null = pd.read_csv(res / "null/null_summary.csv").set_index("cohort")
braak = json.loads((res / "braak/braak_report.json").read_text())
order = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE118553_Frontal_Cortex",
         "GSE118553_Entorhinal_Cortex", "GSE118553_Cerebellum", "GSE125583_FG"]
label = {"GSE33000_PFC": "GSE33000 prefrontal", "GSE132903_MTG": "GSE132903 middle temporal",
         "GSE118553_Temporal_Cortex": "GSE118553 temporal", "GSE118553_Frontal_Cortex": "GSE118553 frontal",
         "GSE118553_Entorhinal_Cortex": "GSE118553 entorhinal", "GSE118553_Cerebellum": "GSE118553 cerebellum (neg. control)",
         "GSE106241_ITC": "GSE106241 inferior temporal (Braak)", "GSE125583_FG": "GSE125583 fusiform (RNA-seq)"}
INDEPENDENT = ["GSE33000_PFC", "GSE132903_MTG", "GSE118553_Temporal_Cortex", "GSE125583_FG"]


def ci(r, n):
    z, se = np.arctanh(r), 1.06 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), z, se


def pooled(rows):
    z = np.array([r["z"] for r in rows])
    v = np.array([r["se"] ** 2 for r in rows])
    w = 1 / v
    q = np.sum(w * (z - np.sum(w * z) / w.sum()) ** 2)
    tau2 = max(0, (q - (len(z) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    m, se = np.sum(ws * z) / ws.sum(), np.sqrt(1 / ws.sum())
    return {"name": "Pooled (random effects)", "r": np.tanh(m), "lo": np.tanh(m - 1.96 * se), "z_pooled": m,
            "se_pooled": se, "p": float(2 * stats.norm.sf(abs(m / se))),
            "hi": np.tanh(m + 1.96 * se), "pooled": True, "I2": max(0, (q - (len(z) - 1)) / q) if q > 0 else 0}


panels = {}
rows = []
for c in order:
    d = proj[proj.cohort == c]
    r = stats.spearmanr(d["r_C"] - d["r_A"], d["neuron"])[0]
    lo, hi, z, se = ci(r, len(d))
    rows.append({"name": label[c], "cohort": c, "r": r, "lo": lo, "hi": hi, "z": z, "se": se, "n": len(d)})
panels["C-minus-A score vs neuron marker score (AD)"] = rows

rows = []
for c in order:
    n = sum(val[c]["n"].values())
    r = null.loc[c, "stage_rho_neff"]
    lo, hi, z, se = ci(r, n)
    rows.append({"name": label[c], "cohort": c, "r": r, "lo": lo, "hi": hi, "z": z, "se": se, "n": n})
b = braak["T1_all"]["ca_vs_braak"]
lo, hi, z, se = ci(b["rho"], b["n"])
rows.append({"name": label["GSE106241_ITC"], "cohort": "GSE106241_ITC", "r": b["rho"], "lo": lo, "hi": hi, "z": z, "se": se, "n": b["n"]})
panels["C-minus-A score vs disease stage (all samples)"] = rows

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
summary = {}
for ax, (title, rows) in zip(axes, panels.items()):
    pool = pooled([r for r in rows if r["cohort"] in INDEPENDENT + ["GSE106241_ITC"]])
    allrows = rows + [pool]
    y = np.arange(len(allrows))[::-1]
    for yi, r in zip(y, allrows):
        col = "black" if r.get("pooled") else ("#888888" if "Cerebellum" in r.get("cohort", "") else "#1f4e79")
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col)
        ax.scatter(r["r"], yi, color=col, marker="D" if r.get("pooled") else "s", s=50 if r.get("pooled") else 30, zorder=3)
        ax.text(1.02, yi, f"{r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="grey", lw=.8, ls="--")
    ax.set_yticks(y, [r["name"] + (f" (n={r['n']})" if "n" in r else "") for r in allrows], fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (95% CI)")
    ax.set_title(title, fontsize=10)
    summary[title] = {"pooled_rho": round(float(pool["r"]), 3), "ci": [round(float(pool["lo"]), 3), round(float(pool["hi"]), 3)],
                      "I2": round(float(pool["I2"]), 3), "p": float(f"{pool['p']:.3g}"),
                      "pooled_cohorts": [r["cohort"] for r in rows if r["cohort"] in INDEPENDENT + ["GSE106241_ITC"]],
                      "rows": [{"cohort": r["cohort"], "rho": round(float(r["r"]), 3), "n": int(r["n"]),
                                "p": float(f"{2 * stats.norm.sf(abs(r['z'] / r['se'])):.3g}")} for r in rows]}
fig.tight_layout()
fig.savefig(res / "forest_plot.png", dpi=150, bbox_inches="tight")
(res / "forest_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
