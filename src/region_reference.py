"""Priority-2 F: does the deconvolution reference region matter? Frontal cohorts re-adjusted with a
frontal (SEA-AD DFC) reference.

Two references are built the same way from the same six low-pathology SEA-AD donors (Braak 0-II,
ADNC not AD / low): MTG and DFC (dorsolateral frontal cortex). Type profile = summed raw UMIs of the
type's subclasses across the six donors, as CPM, in the 9 types used before. The frontal cohorts
(GSE33000 PFC, GSE118553 frontal) and, for contrast, the temporal ones are then run through the same
deconvolution / adjustment as deconv_seaad.py with each reference.

Usage: python region_reference.py --data /mnt/project-files/d1-data/raw --seaad /mnt/project-files/d1-data/seaad \
         --out ../results/region_reference --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from build_seaad_reference import GROUPS
from deconv_seaad import run_cohort
from pilot_gse33000 import SEED
from validate_cohorts import cohorts

DONORS = ["H21.33.003", "H21.33.004", "H21.33.018", "H21.33.019", "H21.33.028", "H21.33.043"]
EXTRA = {"Micro": ["Immune"], "Vascular": ["VLMC & Perivascular"]}
RUN = ["GSE33000_PFC", "GSE118553_Frontal_Cortex", "GSE132903_MTG", "GSE118553_Temporal_Cortex"]


def build(dirpath: Path) -> pd.DataFrame:
    genes = (dirpath / "genes.txt").read_text().split("\n")
    sub_of = {s: t for t, v in GROUPS.items() for s in v}
    for t, v in EXTRA.items():
        sub_of.update({s: t for s in v})
    tot = None
    for d in DONORS:
        z = np.load(dirpath / f"{d}.npz", allow_pickle=True)
        subs = z["subclasses"].astype(str)
        df = pd.DataFrame(z["umis"], index=subs, columns=genes).groupby([sub_of[s] for s in subs]).sum()
        tot = df if tot is None else tot.add(df, fill_value=0)
    tot = tot.reindex(list(GROUPS))
    cpm = tot.div(tot.sum(axis=1), axis=0) * 1e6
    out = cpm.T
    return out.groupby(level=0).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seaad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = {"MTG": build(a.seaad / "donors"), "DFC": build(a.seaad / "donors_dfc")}
    for k, r in refs.items():
        r.to_csv(a.out / f"ref6_{k}.csv.gz")
    common = refs["MTG"].index.intersection(refs["DFC"].index)
    lg = {k: np.log2(r.loc[common] + 1) for k, r in refs.items()}
    rep: dict = {"donors": DONORS, "type_profile_r_MTG_vs_DFC": {t: round(float(np.corrcoef(lg["MTG"][t], lg["DFC"][t])[0, 1]), 3)
                                                                 for t in refs["MTG"].columns}}
    _, cls = neff_subtypes.load_centroids(a.data / "neff2021")
    rows = []
    for name, e, m in cohorts(a.data):
        if name not in RUN:
            continue
        for k, ref in refs.items():
            rng = np.random.default_rng(SEED)
            r, tab = run_cohort(name, e, m, cls, ref, a.n, rng)
            for meth in ["nnls", "svr"]:
                d = r[f"adj_{meth}"]
                rows.append({"cohort": name, "reference": k, "method": meth, "n_sig": r["n_sig_genes"],
                             "exc_AD": r[f"{meth}_mean_by_group"].get("AD", {}).get("Exc"),
                             **{x: d[x] for x in ["r2_ca", "r2_b", "ari_adj", "sig_raw", "sig_adj", "sig_adj_rand_p"]}})
            print(name, k, "done", flush=True)
    flat = pd.DataFrame(rows)
    flat.to_csv(a.out / "region_reference_summary.csv", index=False)
    rep["summary"] = flat.to_dict(orient="records")
    (a.out / "region_reference_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep["type_profile_r_MTG_vs_DFC"]))
    print(flat.round(3).to_string())


if __name__ == "__main__":
    main()
