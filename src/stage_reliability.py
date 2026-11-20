"""Is the residual stage association after neuron adjustment (pooled rho 0.14) explained by measurement
error in the neuron marker score?

Adjusting for an error-prone covariate leaves part of the confounded association behind. For each
independent cohort (all samples, age/sex-residualised ranks as in meta_reanalysis.rho):
  1. reliability of the neuron marker score and of the C-A score from random split halves of their
     genes (Spearman-Brown, mean over splits);
  2. the partial correlation C-A ~ stage | neuron recomputed with correlations disattenuated for
     both reliabilities (stage treated as error-free), with a bootstrap CI;
  3. the same partial with the neuron score entered as ranks (functional-form sensitivity);
  4. an empirical check: the partial correlation when adjusting with half of the neuron markers
     (lower reliability) versus all of them, to show the direction of the bias.
Pooled with the same REML/HKSJ model as meta_reanalysis.py.

Usage: python stage_reliability.py --data /mnt/project-files/d1-data/raw --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from meta_reanalysis import INDEPENDENT, ca_score, pool, resid_rank
from pilot_gse33000 import NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts


def sb(r: float) -> float:
    return 2 * r / (1 + r)


def c(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def lin_resid(v: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Raw covariate values residualised on age/sex: matches meta_reanalysis.rho, which enters the
    neuron score on its own scale beside age and sex."""
    D = np.column_stack([np.ones(len(v)), cov])
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def partial(rxy, rxz, ryz) -> float:
    d = (1 - rxz ** 2) * (1 - ryz ** 2)
    return float((rxy - rxz * ryz) / np.sqrt(d)) if d > 0 else np.nan


def analyse(Zn, Zc, cent, stage, cov, rng, n_split):
    """Zn: neuron marker z (genes x samples); Zc: classifier gene z; returns dict of estimates."""
    neu = Zn.mean(0)
    ca = ca_score(pd.DataFrame(Zc), cent).values
    x, y, z = resid_rank(ca, cov), resid_rank(stage, cov), lin_resid(neu, cov)
    zr = resid_rank(neu, cov)
    rxy, rxz, ryz = c(x, y), c(x, z), c(y, z)
    rel_n, rel_c, half_p = [], [], []
    kn, kc = Zn.shape[0], Zc.shape[0]
    for _ in range(n_split):
        i = rng.permutation(kn)
        a, b = Zn[i[: kn // 2]].mean(0), Zn[i[kn // 2:]].mean(0)
        ra, rb = lin_resid(a, cov), lin_resid(b, cov)
        rel_n.append(sb(c(ra, rb)))
        half_p.append(partial(rxy, c(x, ra), c(y, ra)))
        j = rng.permutation(kc)
        ca_a = ca_score(pd.DataFrame(Zc[j[: kc // 2]]), cent.iloc[j[: kc // 2]]).values
        ca_b = ca_score(pd.DataFrame(Zc[j[kc // 2:]]), cent.iloc[j[kc // 2:]]).values
        rel_c.append(sb(c(resid_rank(ca_a, cov), resid_rank(ca_b, cov))))
    rn, rc = float(np.mean(rel_n)), float(np.mean(rel_c))
    rxz_s = rxz / np.sqrt(rn * rc)
    corr = partial(rxy / np.sqrt(rc), rxz_s, ryz / np.sqrt(rn)) if abs(rxz_s) < 1 else np.nan
    return {"rho_ca_stage": rxy, "rho_ca_neuron": rxz, "rho_neuron_stage": ryz,
            "partial_observed": partial(rxy, rxz, ryz),
            "partial_rank_neuron": partial(rxy, c(x, zr), c(y, zr)), "partial_half_markers": float(np.nanmean(half_p)),
            "rel_neuron": rn, "rel_ca": rc, "rho_ca_neuron_disattenuated": float(rxz_s),
            "partial_disattenuated": corr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-split", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 11)
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows, boots = [], {}
    for name, e, m in cohorts(a.data):
        if name not in INDEPENDENT:
            continue
        genes = cls.index.intersection(e.index)
        cent = cls.loc[genes].sub(cls.loc[genes].mean(axis=1), axis=0)
        Zc = zscore_rows(e.loc[genes, m.index]).values
        mk = [g for g in NEURON_MARKERS if g in e.index and g not in genes]
        Zn = zscore_rows(e.loc[mk, m.index]).values
        stage = m["group"].map(GROUP_ORDER).values.astype(float)
        age = pd.to_numeric(m["age"], errors="coerce")
        age = age.fillna(age.median()).values
        sex = (m["sex"].astype(str).str.lower().str[0] == "m").astype(float).values
        cov = np.column_stack([age, sex])
        est = analyse(Zn, Zc, cent, stage, cov, rng, a.n_split)
        bz = []
        n = len(stage)
        for _ in range(a.n_boot):
            i = rng.integers(0, n, n)
            with np.errstate(all="ignore"):
                b = analyse(Zn[:, i], Zc[:, i], cent, stage[i], cov[i], rng, 20)["partial_disattenuated"]
            bz.append(np.arctanh(np.clip(b, -0.9999, 0.9999)) if np.isfinite(b) else np.nan)
        bz = np.array(bz)
        est.update({"cohort": name, "n": n, "n_neuron_markers": len(mk),
                    "boot_frac_undefined": float(np.isnan(bz).mean()),
                    "se_boot_z": float(np.nanstd(bz, ddof=1)),
                    "ci95_boot": [float(np.tanh(np.nanpercentile(bz, 2.5))),
                                  float(np.tanh(np.nanpercentile(bz, 97.5)))]})
        rows.append(est)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in est.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out / "per_cohort.csv", index=False)
    ok = df.dropna(subset=["partial_disattenuated"])
    rep = {"per_cohort": df.round(3).set_index("cohort").to_dict(orient="index"),
           "pooled_partial_disattenuated": pool(np.arctanh(ok.partial_disattenuated.values), ok.se_boot_z.values)
           if len(ok) >= 2 else None,
           "note": "stage treated as error-free; reliabilities from split halves (Spearman-Brown)"}
    (a.out / "stage_reliability.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep["pooled_partial_disattenuated"], indent=2))


if __name__ == "__main__":
    main()
