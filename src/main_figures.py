"""Draft main figures (Fig 1-5) for the D1 manuscript from saved result tables.
Usage: python main_figures.py <d1-results dir> <out dir>
"""
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "savefig.dpi": 300})
SHORT = {"GSE33000_PFC": "GSE33000 PFC", "GSE132903_MTG": "GSE132903 MTG", "GSE118553_Temporal_Cortex": "GSE118553 TC",
         "GSE118553_Frontal_Cortex": "GSE118553 FC", "GSE118553_Entorhinal_Cortex": "GSE118553 EC",
         "GSE118553_Cerebellum": "GSE118553 CB", "GSE125583_FG": "GSE125583 FG (RNA-seq)", "GSE106241_ITC": "GSE106241 ITC"}


def tag(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, weight="bold", va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 1: design ----------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    def box(x, y, w, h, t, fc="#f5f4f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2)
    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    box(0.1, 3.3, 2.3, 1.4, "Neff 2021 subtypes\n271 marker genes\nA / B / C centroids", "#e7f1fb")
    box(0.1, 0.3, 2.3, 2.6, "Independent cohorts\nGSE33000 PFC\nGSE132903 MTG\nGSE118553 (4 regions)\nGSE125583 FG (RNA-seq)\nGSE106241 ITC (Braak)\nMSBB 17 regions (CDR)")
    box(3.0, 1.9, 2.0, 1.4, "Project each sample\nr(A), r(B), r(C)\nC−A axis score")
    box(5.6, 3.4, 2.0, 1.2, "Reproducibility\nvs random-gene null")
    box(5.6, 1.95, 2.0, 1.2, "Cell composition\nmarkers, NNLS (SEA-AD)")
    box(5.6, 0.5, 2.0, 1.2, "Disease stage\nBraak, CDR, mediation")
    box(8.1, 1.95, 1.8, 1.2, "SEA-AD 84 donors\npseudobulk +\nsimulation", "#e5f4e4")
    arr(2.4, 4.0, 3.0, 2.9); arr(2.4, 1.6, 3.0, 2.3)
    for y in (4.0, 2.55, 1.1): arr(5.0, 2.6, 5.6, y)
    arr(8.1, 2.55, 7.6, 2.55)
    save(fig, "Fig1_design")


# ---------- Fig 2: reproducibility ----------
def fig2():
    v = pd.read_csv(R / "validation/validation_summary.csv")
    nul = pd.read_csv(R / "null/null_distributions.csv"); ns = pd.read_csv(R / "null/null_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    rows = [("Neff 2021 (MSBB PHG)", 0.31, 0.36, 0.33)] + [(SHORT[c] + f" (n={n})", a, b, cc) for c, n, a, b, cc in
                                                           v[["cohort", "n_AD", "frac_A", "frac_B", "frac_C"]].values]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a, b, c) in zip(y, rows):
        left = 0
        for k, f in zip("ABC", (a, b, c)):
            ax.barh(yi, f - 0.004, left=left, color=COL[k], height=0.7); left += f
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 1); ax.set_xlabel("Fraction of AD samples")
    ax.get_yticklabels()[0].set_weight("bold")
    for k, x in zip("ABC", (0.17, 0.48, 0.82)):
        ax.text(x, y[0] + 0.62, k, color=INK, ha="center", fontsize=8, weight="bold")
    ax.set_title("Subtype proportions reproduce"); tag(ax, "a")
    ax = axs[1]
    order = list(v.cohort)
    for i, c in enumerate(order):
        d = nul[nul.cohort == c].neuron_rho.values
        ax.scatter(np.random.default_rng(0).normal(i, 0.07, len(d)), d, s=1.5, color="#b8b7b0", alpha=0.35, lw=0)
        obs = ns.set_index("cohort").loc[c, "neuron_rho_neff"]
        ax.scatter(i, obs, s=36, color=COL["C"], edgecolor="white", lw=1, zorder=3)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[c].split(" (")[0].replace("GSE", "") for c in order], rotation=45, ha="right")
    ax.set_ylabel("ρ (C−A score vs neuron score)")
    ax.scatter([], [], s=36, color=COL["C"], label="Neff genes"); ax.scatter([], [], s=8, color="#b8b7b0", label="1,000 random gene sets")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=7); ax.set_title("Neuron axis exceeds random-gene null"); tag(ax, "b")
    fig.tight_layout(); save(fig, "Fig2_reproducibility")


# ---------- Fig 3: neuron axis ----------
def fig3():
    p = pd.read_csv(R / "validation/projection_all_cohorts.csv"); p["ca"] = p.r_C - p.r_A
    fs = json.load(open(R / "forest_summary.json"))["C-minus-A score vs neuron marker score (AD)"]
    reg = pd.read_csv(R / "msbb/msbb_region_summary.csv", index_col=0).sort_values("ad_ca_neuron")
    fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    for j, c in enumerate(["GSE33000_PFC", "GSE132903_MTG", "GSE125583_FG"]):
        ax = fig.add_subplot(gs[0, j]); d = p[p.cohort == c]
        for k in "ABC":
            s = d[d.subtype == k]; ax.scatter(s.neuron, s.ca, s=7, color=COL[k], lw=0, alpha=0.8, label=k)
        rho = spearmanr(d.neuron, d.ca)[0]
        ax.text(0.97, 0.95, f"ρ = {rho:.2f}\nn = {len(d)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        ax.set_title(SHORT[c].replace(" (RNA-seq)", "\n(RNA-seq)") if j == 2 else SHORT[c]); ax.set_xlabel("Neuron marker score")
        if j == 0:
            ax.set_ylabel("C−A subtype score"); tag(ax, "a"); ax.legend(title="Subtype", fontsize=7, title_fontsize=7, loc="lower left", markerscale=1.6)
    ax = fig.add_subplot(gs[1, :2]); rows = fs["rows"]; yy = np.arange(len(rows) + 1)[::-1]
    for yi, r in zip(yy[:-1], rows):
        ax.scatter(r["rho"], yi, s=14 + r["n"] / 6, color=MUTED, zorder=3)
    lo, hi = fs["ci"]; ax.plot([lo, hi], [yy[-1]] * 2, color=INK, lw=2); ax.scatter(fs["pooled_rho"], yy[-1], marker="D", s=40, color=INK, zorder=3)
    ax.text(fs["pooled_rho"] + 0.04, yy[-1], f"{fs['pooled_rho']:.2f} [{lo:.2f}, {hi:.2f}]", va="center", fontsize=7)
    ax.set_yticks(yy); ax.set_yticklabels([f"{SHORT[r['cohort']]}  n={r['n']}" for r in rows] + ["Pooled (4 cohorts)"], fontsize=7)
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.set_xlim(-1.0, 0); ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("\u03c1 (C\u2212A score vs neuron score), AD samples"); ax.set_title("Meta-analysis (random effects)"); tag(ax, "b")
    ax = fig.add_subplot(gs[1, 2]); yv = np.arange(len(reg))
    ax.barh(yv, reg.ad_ca_neuron, color=COL["C"], height=0.65); ax.set_yticks(yv); ax.set_yticklabels(reg.index, fontsize=6.5)
    ax.set_xlim(-1, 0); ax.set_xlabel("ρ in AD, per region"); ax.set_title("MSBB, 17 regions"); tag(ax, "c")
    fig.subplots_adjust(hspace=0.55, wspace=0.9); save(fig, "Fig3_neuron_axis")


# ---------- Fig 4: stage and mediation ----------
def fig4():
    g = pd.read_csv(R / "gse125583/braak_gse125583_per_sample.csv", index_col=0); g = g[g.group == "AD"]
    m = pd.read_csv(R / "msbb/msbb_subject_scores.csv")
    reg = pd.read_csv(R / "msbb/msbb_region_summary.csv", index_col=0)
    med = pd.read_csv(R / "mediation/mediation_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axs[0, 0]; bs = sorted(g.braak.dropna().unique())
    data = [g[g.braak == b].ca_score.values for b in bs]
    ax.boxplot(data, positions=range(len(bs)), widths=0.55, showfliers=False, medianprops=dict(color=INK),
               boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i, 0.08, len(d)), d, s=5, color=COL["C"], alpha=0.6, lw=0)
    ax.set_xticks(range(len(bs))); ax.set_xticklabels([f"{int(b)}\n(n={len(d)})" for b, d in zip(bs, data)], fontsize=7)
    r = spearmanr(g.braak, g.ca_score, nan_policy="omit")
    ax.set_xlabel("Braak stage"); ax.set_ylabel("C−A subtype score")
    ax.set_title(f"Within AD: Braak (GSE125583), ρ = {r[0]:.2f}"); tag(ax, "a")
    ax = axs[0, 1]; cd = sorted(m.cdr.dropna().unique())
    data = [m[m.cdr == c].ca.values for c in cd]
    ax.boxplot(data, positions=range(len(cd)), widths=0.55, showfliers=False, medianprops=dict(color=INK),
               boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i, 0.08, len(d)), d, s=6, color=COL["C"], alpha=0.7, lw=0)
    r = spearmanr(m.cdr, m.ca, nan_policy="omit")
    ax.set_xticks(range(len(cd))); ax.set_xticklabels([f"{c:g}" for c in cd]); ax.set_xlabel("CDR")
    ax.set_ylabel("C−A score (subject mean, cortex)"); ax.set_title(f"MSBB subjects: CDR, ρ = {r[0]:.2f}"); tag(ax, "b")
    ax = axs[1, 0]
    rho = spearmanr(reg.neuron_cdr, reg.ca_cdr)[0]
    ax.scatter(reg.neuron_cdr, reg.ca_cdr, s=22, color=COL["C"], edgecolor="white", lw=0.8)
    for nm in ["Parahippocampal Gyrus", "Precentral Gyrus", "Caudate Nucleus", "Superior Temporal Gyrus"]:
        if nm in reg.index:
            ax.annotate(nm.replace(" Gyrus", ""), (reg.loc[nm, "neuron_cdr"], reg.loc[nm, "ca_cdr"]), fontsize=6.5,
                        xytext=(4, 2), textcoords="offset points", color=MUTED)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("ρ (neuron score vs CDR)"); ax.set_ylabel("ρ (C−A score vs CDR)")
    ax.set_title(f"Across 17 regions, ρ = {rho:.2f}"); tag(ax, "c")
    ax = axs[1, 1]
    sel = [(0, "GSE125583 AD, Braak → neuron"), (7, "MSBB, CDR → neuron"), (8, "MSBB, Braak → neuron"),
           (10, "SEA-AD, CPS → L2/3 IT"), (11, "SEA-AD, CPS → all Exc")]
    for yi, (i, lab) in enumerate(sel[::-1]):
        row = med.iloc[i]; lo, hi = ast.literal_eval(row.prop_ci)
        ax.plot([max(lo, -0.3), min(hi, 1.5)], [yi, yi], color=MUTED, lw=1.5)
        ax.scatter(row.prop_mediated, yi, s=30, color=COL["C"] if row.indirect_p_boot < 0.05 else "#b8b7b0", zorder=3)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([s[1] for s in sel[::-1]], fontsize=7)
    ax.axvline(0, color=GRID, lw=0.8); ax.axvline(1, color=GRID, lw=0.8, ls="--"); ax.set_xlim(-0.3, 1.5)
    ax.set_xlabel("Proportion mediated (95% bootstrap CI)"); ax.set_title("Stage → cell loss → subtype axis"); tag(ax, "d")
    fig.tight_layout(); save(fig, "Fig4_stage_mediation")


# ---------- Fig 5: mechanism ----------
def fig5():
    s = pd.read_csv(R / "seaad_pseudobulk/seaad_pseudobulk_per_donor.csv")
    nn = pd.read_csv(R / "msbb/neff_style_cdr_normalisation.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 0.9, 1]})
    ax = axs[0]
    for k in "ABC":
        d = s[s.class_sorted == k]; ax.scatter(d["L2/3 IT_of_neurons"], d.ca_sorted, s=10, color=COL[k], lw=0, label=k)
    r = spearmanr(s["L2/3 IT_of_neurons"], s.ca_sorted)[0]
    ax.set_xlabel("L2/3 IT fraction of neurons"); ax.set_ylabel("C−A score (pseudobulk)")
    ax.set_title(f"SEA-AD 84 donors, ρ = {r:.2f}"); ax.legend(fontsize=7, loc="upper right", markerscale=1.5); tag(ax, "a")
    ax = axs[1]
    rep = json.load(open(R / "simulation/simulation_report.json"))
    names = [("neuron_loss", "Neuron loss"), ("heterogeneous", "Mixed loss"), ("glia_only", "Glia change only"), ("noise_only", "Noise only")]
    sig = [rep[k]["signal_mean_best_r"] for k, _ in names]
    ax.barh(range(4)[::-1], sig, color=[COL["C"], COL["C"], "#b8b7b0", "#b8b7b0"], height=0.6)
    for yi, (k, _) in zip(range(4)[::-1], names):
        if k != "noise_only":
            ax.text(rep[k]["signal_mean_best_r"] + 0.01, yi, f"\u03c1 = {rep[k]['ca_vs_neuron_frac_rho']:.2f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(range(4)[::-1]); ax.set_yticklabels([n for _, n in names]); ax.set_xlim(0, 0.4)
    ax.set_xlabel("Subtype signal (mean best r)"); ax.set_title("Simulated bulk (SEA-AD)"); tag(ax, "b")
    ax = axs[2]
    for _, r in nn.iterrows():
        ax.plot([0, 1], [r.neuron_raw_vs_cdr, r.neuron_normalised_vs_cdr], color=GRID, lw=0.8)
    ax.scatter(np.zeros(len(nn)), nn.neuron_raw_vs_cdr, s=10, color=MUTED); ax.scatter(np.ones(len(nn)), nn.neuron_normalised_vs_cdr, s=10, color=MUTED)
    ax.scatter(np.full(len(nn), 2), nn.ca_normalised_vs_neuron_normalised, s=10, color=COL["C"])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Neuron\nvs CDR\n(raw)", "Neuron\nvs CDR\n(normalised)", "C\u2212A\nvs neuron\n(normalised)"], fontsize=6.5)
    ax.axhline(0, color=GRID, lw=0.8); ax.set_ylim(-1, 0.4); ax.set_ylabel("ρ per MSBB region")
    ax.set_title("Neff-style normalisation"); tag(ax, "c")
    save(fig, "Fig5_mechanism")


for f in (fig1, fig2, fig3, fig4, fig5):
    f(); print("ok", f.__name__)
