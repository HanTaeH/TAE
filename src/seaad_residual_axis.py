"""Where does the replicated residual axis come from? SEA-AD MTG, 84 donors (follow-up, 2026-11-25).

SEA-AD records RIN, PMI, brain pH and fresh brain weight, and single-nucleus data let the axis be
located by cell type. Per-donor pseudobulks (as in seaad_celltype_revision.py, log2 CPM + 1):
  sorted, fixed_comp   whole-sample mixes; five marker cell scores regressed out as in the bulk cohorts
  type_Exc, type_Inh, type_Astro, type_Oligo, type_Micro   one broad type each (no composition), z only
Score = correlation of each donor's (residual) profile with the residual-axis weights.
Reported: presence (best |r| of the weights with PC1..5 loadings); Spearman with RIN, PMI, pH, brain
weight, age, Braak, dementia; donor-level correlation of the score between cell types (a shared
donor-level state would make the types agree).

Usage: python seaad_residual_axis.py --donors /mnt/project-files/d1-data/seaad/donors \
         --weights ../results/residual_quality/residual_template_weights.csv --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from null_models import z_np
from residual_denovo import NPC, loadings, pca
from residual_quality import resid
from seaad_celltype_revision import load_sub
from seaad_pseudobulk import BASE, ORD, RNA, TYPES, cpm

VARS = {"RIN": "RIN", "PMI": "PMI", "pH": "Brain pH", "brain_weight": "Fresh Brain Weight", "age": "Age at Death"}


def tmpl_score(X: np.ndarray, wv: np.ndarray) -> np.ndarray:
    Z = z_np(X)
    ok = ~np.isnan(wv) & ~np.isnan(Z).any(1)
    return np.array([np.corrcoef(Z[ok, j], wv[ok])[0, 1] for j in range(Z.shape[1])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(a.weights, index_col=0).iloc[:, 0]
    genes, meta, allsubs, U, N = load_sub(a.donors)
    tidx = {t: [allsubs.index(s) for s in v] for t, v in TYPES.items()}
    T = np.stack([U[:, tidx[t]].sum(1) for t in TYPES], 1)
    n_type = pd.DataFrame({t: N[v].sum(axis=1) for t, v in TYPES.items()})
    prof = cpm(T)
    avg = prof.mean(0)
    wt = np.array([BASE[t] * RNA.get(t, 1) for t in TYPES]); wt = wt / wt.sum()
    prof_f = np.where((n_type.values >= 10)[:, :, None], prof, avg[None])
    pb = {"sorted": cpm(T.sum(1)), "fixed_comp": cpm(np.einsum("t,dtg->dg", wt, prof_f))}
    for t in ["Exc", "Inh", "Astro", "Oligo", "Micro"]:
        pb[f"type_{t}"] = prof[:, list(TYPES).index(t)]
    y = {k: pd.to_numeric(meta[c], errors="coerce").values for k, c in VARS.items()}
    y["Braak"] = meta["Braak"].map(ORD["Braak"]).values.astype(float)
    y["dementia"] = (meta["Cognitive Status"] == "Dementia").astype(float).values
    out, sc_all = {"n_donors": int(len(meta))}, {}
    for name, M in pb.items():
        E = pd.DataFrame(np.log2(M + 1).T, index=genes).groupby(level=0).first()
        expressed = E.index[(E > 1).mean(axis=1) > 0.5]
        g = [x for x in w.index if x in expressed]
        if name in ("sorted", "fixed_comp"):
            R, g = resid(E.loc[expressed], E.columns, g)
        else:
            R = z_np(E.loc[g].values)
        S, _ = pca(R)
        ld = loadings(R, S[:, :NPC])
        wv = w.reindex(g).values
        sc = tmpl_score(R, wv)
        sc_all[name] = sc
        res = {"genes": len(g), "abs_r_template_vs_PC1_5": [round(float(abs(np.corrcoef(ld[:, k], wv)[0, 1])), 3) for k in range(NPC)]}
        for k, v in y.items():
            ok = ~np.isnan(v)
            r = stats.spearmanr(sc[ok], v[ok])
            res[k] = [round(float(r[0]), 3), float(f"{r[1]:.3g}")]
        out[name] = res
        print(name, json.dumps(res), flush=True)
    names = list(sc_all)
    out["between_type_rho"] = {f"{i}|{j}": round(float(stats.spearmanr(sc_all[i], sc_all[j])[0]), 3)
                               for ii, i in enumerate(names) for j in names[ii + 1:] if i.startswith("type") and j.startswith("type")}
    print(json.dumps(out["between_type_rho"]))
    pd.DataFrame(sc_all, index=meta.index).to_csv(a.out / "seaad_residual_axis_scores.csv")
    (a.out / "seaad_residual_axis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
