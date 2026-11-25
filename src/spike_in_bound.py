"""One-sided 95% upper confidence limit on a composition-independent signal, by test inversion (v11).

Reads results/spike_in_real/spike_in_real.json. The cell-type-matched null (specificity- or strength-
matched random gene sets, projected identically on the same samples) gives the distribution of the
neuron coupling rho under f = 0. Under H_f the classifier's coupling is modelled as that distribution
shifted by delta_rho(f), the median shift produced by spiking f x (C - A contrast) x s into the real
classifier genes (spike_in_real.py). H_f is rejected when the observed rho lies below the 5th percentile
of the shifted null, i.e. when delta_rho(f) > rho_obs - q05(null). The UCL is the largest f not rejected
(linear interpolation on the f grid), also expressed as the share of C-A variance explained by the
added component. If rho_obs is already below q05(null), every f > 0 is rejected (UCL = 0).

Usage: python spike_in_bound.py --res ../results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((a.res / "spike_in_real/spike_in_real.json").read_text())
    out = {}
    for c, r in d.items():
        fs = np.array([s["f"] for s in r["spike"]])
        dr = np.maximum.accumulate(np.array([s["delta_rho"] for s in r["spike"]]))
        vs = np.array([s["var_share_median"] for s in r["spike"]])
        row = {"n_ad": r["n_ad"], "obs_rho": r["obs_rho"]}
        for k in ("spec", "strength_tertile", "strength_decile"):
            q05 = r["nulls"][k]["q05_q95"][0]
            margin = r["obs_rho"] - q05
            if margin <= 0:
                fu, vu = 0.0, 0.0
            else:
                fu = float(np.interp(margin, dr, fs))
                vu = float(np.interp(fu, fs, vs))
            row[k] = {"null_q05": q05, "margin": round(float(margin), 3), "ucl95_f": round(fu, 3),
                      "ucl95_var_share": round(vu, 3), "beyond_grid": bool(margin > dr[-1])}
        out[c] = row
        print(c, json.dumps(row))
    (a.res / "spike_in_real/spike_in_bound.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
