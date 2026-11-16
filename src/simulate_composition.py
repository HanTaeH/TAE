"""In silico experiment: can cell-composition differences alone produce the Neff AD subtypes?

Pseudo-bulk brain samples are built from the SEA-AD MTG single-nucleus reference (24 subclasses,
mean CPM per nucleus, neurotypical donors). The only thing that differs between samples is the
cell-type mixture (plus gene-level multiplicative noise). No disease-specific expression change is
put in. Neff class centroids are then projected exactly as on real cohorts (gene-wise z-scores,
nearest centroid by Pearson r).

Scenarios (N samples each)
  noise_only     proportions fixed near baseline (Dirichlet concentration 2000) + gene noise
  neuron_loss    neuron fraction lowered by l ~ U(0, 0.6) (excitatory x (1-l), inhibitory x (1-0.7 l));
                 freed share goes to astrocytes 50%, microglia 25%, oligodendrocytes 15%, rest 10%
  glia_only      neurons fixed; astrocyte / microglia / oligodendrocyte shares vary (log-normal)
  heterogeneous  all major types vary independently (Dirichlet concentration 30 around baseline)

Baseline major-type shares (cortical grey matter, approx.): Exc 0.30, Inh 0.10, Oligo 0.30,
Astro 0.15, OPC 0.06, Micro 0.06, Vascular 0.03; subclasses within a type follow SEA-AD nucleus
counts. Neuronal nuclei carry RNA_NEURON times more mRNA than glial nuclei (default 2).

Per scenario: class fractions; Spearman of C-minus-A score with the simulated neuron fraction;
mean best correlation to a Neff centroid (signal strength; real cohorts 0.34-0.60).
Decomposition: Neff class contrasts (C - A and B - mean(A, C), log CPM over the classifier genes) are
regressed on cell-type enrichment vectors log2(ref_type + 1) - log2(baseline mixture + 1).

Usage: python simulate_composition.py --data /mnt/project-files/d1-data/raw \
         --ref /mnt/project-files/d1-data/seaad --out ../results/simulation --n 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
from null_models import corr_np, z_np
from pilot_gse33000 import SEED

GROUPS = {
    "Exc": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
    "Inh": ["Sst", "Sst Chodl", "Pvalb", "Chandelier", "Lamp5", "Lamp5 Lhx6", "Sncg", "Vip", "Pax6"],
    "Astro": ["Astrocyte"], "Oligo": ["Oligodendrocyte"], "OPC": ["OPC"],
    "Micro": ["Microglia-PVM"], "Vascular": ["Endothelial", "VLMC"],
}
BASE = {"Exc": 0.30, "Inh": 0.10, "Oligo": 0.30, "Astro": 0.15, "OPC": 0.06, "Micro": 0.06, "Vascular": 0.03}
TYPES = list(BASE)
RNA_NEURON = 2.0
GENE_NOISE_SD = 0.25
CLS = np.array(list("ABC"))


def type_profiles(sub_ref: pd.DataFrame, ncell: pd.Series) -> pd.DataFrame:
    out = {}
    for t, lst in GROUPS.items():
        w = ncell[lst] / ncell[lst].sum()
        out[t] = (sub_ref[lst] * w.values).sum(axis=1)
    return pd.DataFrame(out)


def mix(props: np.ndarray, prof: np.ndarray, rng) -> np.ndarray:
    """props samples x types (nuclei shares) -> log2 CPM, genes x samples."""
    rna = np.array([RNA_NEURON if t in ("Exc", "Inh") else 1.0 for t in TYPES])
    w = props * rna
    w = w / w.sum(1, keepdims=True)
    x = prof @ w.T
    x = x * np.exp(rng.normal(0, GENE_NOISE_SD, x.shape))
    x = x / x.sum(0, keepdims=True) * 1e6
    return np.log2(x + 1)


def scenario_props(name: str, n: int, rng) -> np.ndarray:
    base = np.array([BASE[t] for t in TYPES])
    i = {t: k for k, t in enumerate(TYPES)}
    if name == "noise_only":
        return rng.dirichlet(base * 2000, n)
    if name == "heterogeneous":
        return rng.dirichlet(base * 30, n)
    P = np.tile(base, (n, 1))
    if name == "neuron_loss":
        l = rng.uniform(0, 0.6, n)
        lost = P[:, i["Exc"]] * l + P[:, i["Inh"]] * 0.7 * l
        P[:, i["Exc"]] *= 1 - l
        P[:, i["Inh"]] *= 1 - 0.7 * l
        for t, f in [("Astro", .5), ("Micro", .25), ("Oligo", .15), ("OPC", .05), ("Vascular", .05)]:
            P[:, i[t]] += lost * f
        P = np.array([rng.dirichlet(p * 300) for p in P])
    elif name == "glia_only":
        for t in ("Astro", "Micro", "Oligo"):
            P[:, i[t]] *= np.exp(rng.normal(0, 0.4, n))
        glia = [i[t] for t in TYPES if t not in ("Exc", "Inh")]
        P[:, glia] *= (1 - base[i["Exc"]] - base[i["Inh"]]) / P[:, glia].sum(1, keepdims=True)
    return P / P.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True, help="folder with seaad_mtg_subclass_reference.csv.gz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sub_ref = pd.read_csv(a.ref / "seaad_mtg_subclass_reference.csv.gz", index_col=0)
    sub_ref = sub_ref[~sub_ref.index.duplicated()]
    ncell = pd.read_csv(a.ref / "seaad_mtg_reference_ncells.csv", index_col=0).iloc[:, 0]
    prof_all = type_profiles(sub_ref, ncell)
    sub, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(prof_all.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    rep: dict = {"n_genes": int(len(genes)), "coverage": round(len(genes) / len(cls), 3),
                 "settings": {"baseline": BASE, "rna_neuron": RNA_NEURON, "gene_noise_sd": GENE_NOISE_SD}}

    sims = {}
    for sc in ["noise_only", "neuron_loss", "glia_only", "heterogeneous"]:
        P = scenario_props(sc, a.n, rng)
        X = mix(P, prof_all.values, rng)                      # all genes, so CPM is genome-wide
        Xg = pd.DataFrame(X, index=prof_all.index).loc[genes].values
        r = corr_np(z_np(Xg), cent)
        lab = CLS[r.argmax(1)]
        ca = r[:, 2] - r[:, 0]
        neuron = P[:, TYPES.index("Exc")] + P[:, TYPES.index("Inh")]
        res = {"class_frac": {k: round(float((lab == k).mean()), 3) for k in "ABC"},
               "signal_mean_best_r": round(float(r.max(1).mean()), 3),
               "ca_vs_neuron_frac_rho": round(float(stats.spearmanr(ca, neuron)[0]), 3),
               "neuron_frac_by_class": {k: round(float(neuron[lab == k].mean()), 3) for k in "ABC" if (lab == k).any()}}
        for t in ("Astro", "Micro", "Oligo"):
            res[f"ca_vs_{t}_rho"] = round(float(stats.spearmanr(ca, P[:, TYPES.index(t)])[0]), 3)
        rep[sc] = res
        sims[sc] = (ca, neuron, lab, r.max(1))
        print(sc, json.dumps(res), flush=True)

    # decomposition of Neff class contrasts into cell-type enrichment vectors
    raw = cls.loc[genes]
    contrasts = {"C_minus_A": (raw["C"] - raw["A"]).values,
                 "B_minus_AC": (raw["B"] - (raw["A"] + raw["C"]) / 2).values}
    base_mix = prof_all.loc[genes].values @ np.array([BASE[t] * (RNA_NEURON if t in ("Exc", "Inh") else 1) for t in TYPES])
    enrich = np.log2(prof_all.loc[genes].values + 1) - np.log2(base_mix + 1)[:, None]
    enrich = pd.DataFrame(enrich, index=genes, columns=TYPES)
    neuron_vec = np.log2(prof_all.loc[genes, ["Exc", "Inh"]].mean(axis=1).values + 1) - np.log2(base_mix + 1)
    dec = {}
    for k, y in contrasts.items():
        D = np.column_stack([np.ones(len(y)), enrich.values])
        beta, *_ = np.linalg.lstsq(D, y, rcond=None)
        r2 = 1 - ((y - D @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        dec[k] = {"r2_all_types": round(float(r2), 3),
                  "r_with_each_type": {t: round(float(stats.pearsonr(y, enrich[t])[0]), 3) for t in TYPES},
                  "r_with_neuronal_vector": round(float(stats.pearsonr(y, neuron_vec)[0]), 3)}
        # null: same regression with gene labels of y permuted
        null = []
        for _ in range(1000):
            yp = rng.permutation(y)
            bp, *_ = np.linalg.lstsq(D, yp, rcond=None)
            null.append(1 - ((yp - D @ bp) ** 2).sum() / ((yp - yp.mean()) ** 2).sum())
        dec[k]["r2_perm_p"] = round(float((np.sum(np.array(null) >= r2) + 1) / 1001), 4)
        dec[k]["r2_perm_median"] = round(float(np.median(null)), 3)
    rep["decomposition"] = dec
    print(json.dumps(dec, indent=1))
    (a.out / "simulation_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    names = ["noise_only", "neuron_loss", "glia_only", "heterogeneous"]
    bottom = np.zeros(len(names))
    colors = {"A": "#2e7d32", "B": "#f9a825", "C": "#c62828"}
    for k in "ABC":
        v = np.array([rep[s]["class_frac"][k] for s in names])
        ax.bar(names, v, bottom=bottom, color=colors[k], label=f"class {k}")
        bottom += v
    ax.set_ylabel("Fraction of simulated samples")
    ax.set_title("Neff classes assigned to composition-only pseudo-bulk", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax = axes[1]
    ca, neuron, lab, _ = sims["neuron_loss"]
    for k in "ABC":
        ax.scatter(neuron[lab == k], ca[lab == k], s=8, color=colors[k], label=f"class {k}")
    ax.set_xlabel("Simulated neuronal nuclei fraction")
    ax.set_ylabel("C-minus-A score")
    ax.set_title(f"Neuron-loss scenario: ρ = {rep['neuron_loss']['ca_vs_neuron_frac_rho']}", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[2]
    x = np.arange(len(TYPES))
    for j, (k, lab_) in enumerate([("C_minus_A", "Neff C − A"), ("B_minus_AC", "Neff B − mean(A, C)")]):
        ax.bar(x + (j - .5) * .38, [dec[k]["r_with_each_type"][t] for t in TYPES], .38, label=f"{lab_} (R² {dec[k]['r2_all_types']})")
    ax.set_xticks(x, TYPES, fontsize=8)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_ylabel("Correlation across classifier genes")
    ax.set_title("Neff class contrasts vs cell-type enrichment", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "simulation_summary.png", dpi=150)


if __name__ == "__main__":
    main()
