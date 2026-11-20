"""Histological neuron density (SEA-AD quantitative neuropathology, MTG) against the subtype score.

The bulk analyses cannot tell fewer neurons from lower per-neuron expression. SEA-AD provides, for the
same 84 donors, NeuN immunohistochemistry of the middle temporal gyrus (grey matter and layers). We relate
  - histological measures: NeuN+ cells per area, NeuN+ share of haematoxylin-stained nuclei,
    percent GFAP+ area, Iba1+ cells per area (grey matter);
to
  - donor pseudobulk C-A scores (all nuclei, fixed composition, composition only, excitatory and
    inhibitory single-type pseudobulks), the FANS neuronal fraction of sequenced nuclei, CPS and Braak.
Spearman and age/sex partial Spearman with bootstrap 95% CIs (2,000 resamples). NeuN density is also
tested against the C-A score of each pseudobulk given CPS (does histological neuron number add to
progression?).

Usage: python seaad_neun_histology.py --res ../results --qnp <MTG csv> --seaad-meta <donor_meta.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from braak_gse106241 import partial_spearman
from pilot_gse33000 import SEED

HIST = {"neun_density": "number of NeuN positive cells per area",
        "neun_area_pct": "percent NeuN positive area",
        "nuclei_density": "number of Hematoxylin positive nuclei per area",
        "gfap_area_pct": "percent GFAP positive area",
        "iba1_density": "number of Iba1 positive cells per area",
        "at8_area_pct": "percent AT8 positive area",
        "abeta_area_pct": "percent 6e10 positive area"}
SCORES = ["ca_sorted", "ca_fixed_comp", "ca_comp_only", "ca_type_Exc", "ca_type_Inh", "ca_Exc_fixed_subclass_mix",
          "neuron_nuclei_frac", "CPS", "Braak"]


def boot(x, y, cov, rng, n):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        with np.errstate(all="ignore"):
            r = partial_spearman(x[i], y[i], None if cov is None else cov[i])[0]
        if np.isfinite(r):
            v.append(r)
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--qnp", type=Path, required=True)
    ap.add_argument("--seaad-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 41)
    q = pd.read_csv(a.qnp)
    g = q[q["analysis region"] == "Grey matter"].set_index("Donor ID")
    h = pd.DataFrame({k: pd.to_numeric(g[v], errors="coerce") for k, v in HIST.items()})
    h["neun_share_of_nuclei"] = h["neun_density"] / h["nuclei_density"]
    d = pd.read_csv(a.res / "seaad_celltype_revision/per_donor.csv", index_col=0)
    meta = pd.read_csv(a.seaad_meta).set_index("Donor ID").loc[d.index]
    d["age"] = pd.to_numeric(meta["Age at Death"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    d["male"] = (meta["Sex"].astype(str).str.lower().str[0] == "m").astype(float)
    d = d.join(h, how="inner")
    d.to_csv(a.out / "seaad_neun_per_donor.csv")
    cov = d[["age", "male"]].values.astype(float)
    rows = []
    for hv in ["neun_density", "neun_share_of_nuclei", "neun_area_pct", "gfap_area_pct", "iba1_density"]:
        for sc in SCORES:
            x, y = d[hv].values.astype(float), d[sc].values.astype(float)
            r, p, n = partial_spearman(x, y)
            ra, pa, _ = partial_spearman(x, y, cov)
            rows.append({"histology": hv, "score": sc, "n": n, "rho": round(r, 3), "p": p,
                         "rho_age_sex": round(ra, 3), "p_age_sex": pa, "ci_age_sex": str(boot(x, y, cov, rng, a.n_boot))})
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "seaad_neun_associations.csv", index=False)
    # does histological neuron density relate to the score beyond progression?
    extra = {}
    for sc in ["ca_sorted", "ca_fixed_comp", "ca_type_Exc", "ca_type_Inh"]:
        x, y = d["neun_density"].values.astype(float), d[sc].values.astype(float)
        c2 = np.column_stack([cov, d["CPS"].values])
        r, p, n = partial_spearman(x, y, c2)
        c3 = np.column_stack([cov, d["neuron_nuclei_frac"].values])
        r3, p3, _ = partial_spearman(x, y, c3)
        extra[sc] = {"rho_given_age_sex_CPS": round(r, 3), "p": p, "ci": boot(x, y, c2, rng, a.n_boot),
                     "rho_given_age_sex_FANS_neuron_frac": round(r3, 3), "p_fans": p3}
    rep = {"n_donors": int(len(d)),
           "neun_density_summary": d["neun_density"].describe().round(8).to_dict(),
           "associations": t.round(4).to_dict(orient="records"), "neun_density_beyond": extra}
    (a.out / "seaad_neun_report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(t[["histology", "score", "n", "rho", "p", "rho_age_sex", "p_age_sex", "ci_age_sex"]].to_string())
    print(json.dumps(extra, indent=2, default=float))


if __name__ == "__main__":
    main()
