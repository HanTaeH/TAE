"""Fig. 1b (v12): published subtype axes against the neuron score, with cell-type-matched random genes.

Rows: Neff 2021 C-A score; Li 2022 subgroups I-III (hub-gene scores); Lai 2022 and Ma 2023 (PC1 of the
scheme genes); Tijms 2024 CSF subtypes 1-5 projected on brain. x: |Spearman rho| with the neuron score in AD
samples of the four independent cohorts (filled marker); open marker: median of the specificity-matched
random sets for the same rows. Data from results already produced by run_all.sh.

Usage: python scheme_overview_v12.py --res ../results --out ../results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"  # as in main_figures.py

COH = {"GSE33000_PFC": ("GSE33000", "#2a78d6", "o"), "GSE132903_MTG": ("GSE132903", "#eb6834", "s"),
       "GSE118553_Temporal_Cortex": ("GSE118553", "#1baf7a", "^"), "GSE125583_FG": ("GSE125583", "#8a5cc2", "D")}


def rows(res: Path) -> pd.DataFrame:
    out = []
    for r in json.loads((res / "null_specificity/null_specificity_matched_report.json").read_text()):
        n = r["neuron_rho"]
        out.append(("Neff 2021  C−A", r["cohort"], abs(n["neff"]), abs(n["random_median"])))
    for r in json.loads((res / "rq2_li2022/li2022_specnull_v12.json").read_text()):
        out.append((f"Li 2022  subgroup {r['subgroup']}", r["cohort"], abs(r["rho_neuron"]), r["spec_null_median_abs"]))
    for r in json.loads((res / "other_schemes/other_schemes_clustered.json").read_text()):
        if r["scheme"].startswith("Wang"):
            continue
        lab = {"Lai2022_immune": "Lai 2022  immune", "Ma2023_m7G": "Ma 2023  m7G"}[r["scheme"]]
        out.append((lab, r["cohort"], r["abs_rho_axis_neuron"], r["null_spec"]["random_median"]))
    for c in json.loads((res / "tijms_csf/tijms_csf_projection.json").read_text())["cohorts"]:
        for st, d in c["subtypes"].items():
            k, nm = st.split("_", 1)
            out.append((f"Tijms 2024 CSF  {k[1:]} {nm.replace('_', ' ')}", c["cohort"], abs(d["rho_neuron"]), d["null_spec_median_abs"]))
    return pd.DataFrame(out, columns=["scheme", "cohort", "obs", "null"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = rows(a.res)
    d.to_csv(a.out / "fig1b_schemes_data.csv", index=False)
    order = list(dict.fromkeys(d.scheme))
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(order) + 0.9))
    for i, s in enumerate(order):
        ax.axhline(i, color=GRID, lw=0.5, zorder=0)
        for j, (c, (lab, col, mk)) in enumerate(COH.items()):
            q = d[(d.scheme == s) & (d.cohort == c)]
            if q.empty:
                continue
            y = i + (j - 1.5) * 0.15
            ax.plot([q.null.iloc[0], q.obs.iloc[0]], [y, y], color=col, lw=0.8, alpha=0.6, zorder=1)
            ax.scatter(q.null, [y], s=16, marker=mk, facecolor="white", edgecolor=col, lw=1, zorder=2)
            ax.scatter(q.obs, [y], s=22, marker=mk, color=col, edgecolor="white", lw=0.6, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("|ρ| with the neuron score, AD samples")
    for c, (lab, col, mk) in COH.items():
        ax.scatter([], [], s=22, marker=mk, color=col, label=lab)
    ax.scatter([], [], s=16, marker="o", facecolor="white", edgecolor=MUTED, label="cell-type-matched random median")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    ax.tick_params(axis="y", labelsize=7, colors=INK)
    fig.savefig(a.out / "fig1b_schemes.png", bbox_inches="tight")
    fig.savefig(a.out / "fig1b_schemes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
