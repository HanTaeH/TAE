"""Reviewer revision (R1 major 8, R2 minor): simulation sensitivity and identifiability.

Builds on simulate_composition.py (same SEA-AD subclass reference, same projection).
  1. noise_only_exact     proportions exactly fixed (no Dirichlet jitter) - is the noise-only coupling
                          (rho -0.65 in the original run) produced by tiny composition jitter?
  2. percell_global       proportions fixed; total mRNA per neuron lowered by d ~ U(0, 0.6)
                          (all neuronal genes scaled together)
  3. percell_neuronal     proportions fixed; only neuron-enriched genes (log2 neuron/glia > 1 in the
                          reference) lowered by d ~ U(0, 0.6) inside neuronal profiles
  4. percell_synaptic_nonmarker  as 3 but sparing the 16 neuron marker genes, so the marker score
                          cannot see the change
  5. sensitivity of neuron_loss to RNA_NEURON (1, 1.5, 2, 3) and gene noise SD (0.1, 0.25, 0.5)
For each: class fractions, C-minus-A vs neuron fraction (or vs d), C-minus-A vs the marker-based
neuron score computed from the simulated bulk, and mean best-centroid correlation (signal).

Usage: python simulation_revision.py --data /mnt/project-files/d1-data/raw --ref /mnt/project-files/d1-data/seaad \
         --out ../results/simulation_revision --n 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import neff_subtypes
import simulate_composition as sc
from null_models import corr_np, z_np
from pilot_gse33000 import NEURON_MARKERS, SEED

CLS = np.array(list("ABC"))


def mix_custom(props, prof_by_sample, rna_neuron, noise_sd, rng):
    """props samples x types; prof_by_sample samples x genes x types."""
    rna = np.array([rna_neuron if t in ("Exc", "Inh") else 1.0 for t in sc.TYPES])
    w = props * rna
    w = w / w.sum(1, keepdims=True)
    x = np.einsum("sgt,st->gs", prof_by_sample, w)
    x = x * np.exp(rng.normal(0, noise_sd, x.shape))
    x = x / x.sum(0, keepdims=True) * 1e6
    return np.log2(x + 1)


def evaluate(X, genes_all, genes, cent, driver, neuron_frac):
    L = pd.DataFrame(X, index=genes_all)
    r = corr_np(z_np(L.loc[genes].values), cent)
    lab = CLS[r.argmax(1)]
    ca = r[:, 2] - r[:, 0]
    mk = [g for g in NEURON_MARKERS if g in L.index and g not in set(genes)]
    neuron_score = z_np(L.loc[mk].values).mean(0)
    out = {"class_frac": {k: round(float((lab == k).mean()), 3) for k in "ABC"},
           "signal_mean_best_r": round(float(r.max(1).mean()), 3),
           "ca_vs_driver_rho": round(float(stats.spearmanr(ca, driver)[0]), 3),
           "ca_vs_marker_neuron_score_rho": round(float(stats.spearmanr(ca, neuron_score)[0]), 3)}
    if neuron_frac is not None and np.std(neuron_frac) > 0:
        out["ca_vs_neuron_frac_rho"] = round(float(stats.spearmanr(ca, neuron_frac)[0]), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sub_ref = pd.read_csv(a.ref / "seaad_mtg_subclass_reference.csv.gz", index_col=0)
    sub_ref = sub_ref[~sub_ref.index.duplicated()]
    ncell = pd.read_csv(a.ref / "seaad_mtg_reference_ncells.csv", index_col=0).iloc[:, 0]
    prof = sc.type_profiles(sub_ref, ncell)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    genes = cls.index.intersection(prof.index)
    cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0).values
    ga = prof.index
    P0 = prof.values                                            # genes x types
    ni = [sc.TYPES.index("Exc"), sc.TYPES.index("Inh")]
    gi = [k for k in range(len(sc.TYPES)) if k not in ni]
    ratio = np.log2((P0[:, ni].max(1) + 1) / (P0[:, gi].max(1) + 1))
    neuro_genes = ratio > 1
    marker_mask = np.isin(ga, NEURON_MARKERS)
    base = np.array([sc.BASE[t] for t in sc.TYPES])
    n = a.n
    rep = {"n_genes_neuron_enriched": int(neuro_genes.sum()),
           "n_classifier_genes_neuron_enriched": int(neuro_genes[np.isin(ga, genes)].sum())}

    def tile_prof():
        return np.repeat(P0[None], n, 0)

    # 1. exact fixed proportions
    props = np.tile(base, (n, 1))
    X = mix_custom(props, tile_prof(), sc.RNA_NEURON, sc.GENE_NOISE_SD, rng)
    rep["noise_only_exact"] = evaluate(X, ga, genes, cent, rng.normal(size=n), None)
    rep["noise_only_exact"]["note"] = "driver is unrelated random number; neuron fraction constant"
    # 1b. original jitter, to compare
    props = rng.dirichlet(base * 2000, n)
    X = mix_custom(props, tile_prof(), sc.RNA_NEURON, sc.GENE_NOISE_SD, rng)
    nf = props[:, ni].sum(1)
    rep["noise_only_jitter"] = evaluate(X, ga, genes, cent, nf, nf)
    rep["noise_only_jitter"]["neuron_frac_sd"] = round(float(nf.std()), 4)

    # 2-4. per-cell downregulation at fixed composition
    for name, mask in [("percell_global", np.ones(len(ga), bool)),
                       ("percell_neuronal", neuro_genes),
                       ("percell_synaptic_nonmarker", neuro_genes & ~marker_mask)]:
        d = rng.uniform(0, 0.6, n)
        pb = tile_prof()
        for k in ni:
            pb[:, mask, k] *= (1 - d)[:, None]
        X = mix_custom(np.tile(base, (n, 1)), pb, sc.RNA_NEURON, sc.GENE_NOISE_SD, rng)
        rep[name] = evaluate(X, ga, genes, cent, -d, None)
        rep[name]["note"] = "driver = -d (less downregulation = higher), composition fixed"
    # reference: composition loss of matching size
    P = sc.scenario_props("neuron_loss", n, rng)
    X = mix_custom(P, tile_prof(), sc.RNA_NEURON, sc.GENE_NOISE_SD, rng)
    rep["neuron_loss_reference"] = evaluate(X, ga, genes, cent, P[:, ni].sum(1), P[:, ni].sum(1))

    # 5. sensitivity grid for neuron_loss
    grid = []
    for rn in [1.0, 1.5, 2.0, 3.0]:
        for sd in [0.1, 0.25, 0.5]:
            P = sc.scenario_props("neuron_loss", n, rng)
            X = mix_custom(P, tile_prof(), rn, sd, rng)
            e = evaluate(X, ga, genes, cent, P[:, ni].sum(1), P[:, ni].sum(1))
            grid.append({"rna_neuron": rn, "noise_sd": sd, **e["class_frac"],
                         "signal": e["signal_mean_best_r"], "ca_vs_neuron_frac": e["ca_vs_neuron_frac_rho"]})
    rep["neuron_loss_sensitivity"] = grid
    pd.DataFrame(grid).to_csv(a.out / "neuron_loss_sensitivity.csv", index=False)
    (a.out / "simulation_revision_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "neuron_loss_sensitivity"}, indent=1))
    print(pd.DataFrame(grid).to_string())


if __name__ == "__main__":
    main()
