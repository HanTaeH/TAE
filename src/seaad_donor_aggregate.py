"""Download each SEA-AD MTG donor object (final nuclei, 2026-06-22), sum raw UMIs per subclass,
and keep donor metadata and nucleus counts per subclass / supertype. The donor file is deleted after
aggregation, so disk use stays below 1 GB.

Output (--out): <donor>.npz with umis (subclass x gene, float32), subclasses, n_nuclei, and
supertype counts; genes.txt; donor_meta.csv.
Usage: python seaad_donor_aggregate.py --out /mnt/project-files/d1-data/seaad/donors [--shard 0/4]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BUCKET = "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"
PREFIX = "MTG/RNAseq/donors_objects/"
META_COLS = ["Donor ID", "Age at Death", "Sex", "APOE Genotype", "Braak", "Thal", "CERAD score",
             "Overall AD neuropathological Change", "Cognitive Status", "Last MMSE Score", "LATE",
             "Highest Lewy Body Disease", "NeuN positive fraction on FANS", "facs_population_plan",
             "Neurotypical reference", "Severely Affected Donor", "RIN", "PMI", "Brain pH",
             "Fresh Brain Weight", "Years of education"]


def list_keys() -> list[str]:
    x = subprocess.run(["curl", "-s", "-m", "120", f"{BUCKET}?list-type=2&prefix={PREFIX}"],
                       capture_output=True, text=True).stdout
    return re.findall(r"<Key>([^<]*\.h5ad)</Key>", x)


def col(f, n):
    o = f["obs"][n]
    if isinstance(o, h5py.Group):
        return o["categories"][:].astype(str)[o["codes"][:]]
    v = o[:]
    return v.astype(str) if v.dtype.kind in "OS" else v


def aggregate(path: Path):
    with h5py.File(path) as f:
        sub = col(f, "Subclass")
        sup = col(f, "Supertype")
        meta = {}
        for c in META_COLS:
            try:
                v = col(f, c)
                meta[c] = v[0] if len(set(v.tolist())) == 1 else ";".join(sorted(set(map(str, v))))
            except KeyError:
                meta[c] = None
        meta["n_nuclei"] = len(sub)
        genes = f["var"][f["var"].attrs["_index"]][:].astype(str)
        g = f["layers/UMIs"]
        indptr, n_genes = g["indptr"][:], len(genes)
        subs = np.unique(sub)
        code = np.searchsorted(subs, sub)
        out = np.zeros((len(subs), n_genes))
        step = 5000
        for s in range(0, len(sub), step):
            e = min(s + step, len(sub))
            a, b = indptr[s], indptr[e]
            data, idx = g["data"][a:b], g["indices"][a:b]
            rows = np.repeat(code[s:e], np.diff(indptr[s:e + 1]))
            np.add.at(out, (rows, idx), data)
    su, sc = np.unique(sup, return_counts=True)
    return genes, subs, out.astype(np.float32), pd.Series(sub).value_counts().reindex(subs).values, su, sc, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/seaad_dl"))
    ap.add_argument("--shard", default="0/1", help="i/k: process every k-th donor from i (parallel downloads)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.tmp.mkdir(parents=True, exist_ok=True)
    si, sk = map(int, a.shard.split("/"))
    keys = list_keys()
    for i, key in list(enumerate(keys))[si::sk]:
        donor = Path(key).name.split("_")[0]
        dst = a.out / f"{donor}.npz"
        mf = a.out / f"{donor}.meta.csv"
        if dst.exists() and mf.exists():
            continue
        local = a.tmp / f"donor{si}.h5ad"
        t0 = time.time()
        for attempt in range(6):
            r = subprocess.run(["curl", "-s", "-f", "-C", "-", "-o", str(local), BUCKET + key.replace(" ", "%20")])
            if r.returncode in (0, 33):
                break
            time.sleep(10)
        genes, subs, umis, n, su, sc, meta = aggregate(local)
        np.savez_compressed(dst, umis=umis, subclasses=subs, n_nuclei=n, supertypes=su, supertype_n=sc)
        (a.out / "genes.txt").write_text("\n".join(genes))
        pd.DataFrame([meta]).to_csv(mf, index=False)
        local.unlink()
        print(f"{i + 1} {donor} nuclei={meta['n_nuclei']} {time.time() - t0:.0f}s", flush=True)
    metas = [a.out / f"{Path(k).name.split('_')[0]}.meta.csv" for k in keys]
    if all(m.exists() for m in metas):
        pd.concat([pd.read_csv(m) for m in metas]).to_csv(a.out / "donor_meta.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
