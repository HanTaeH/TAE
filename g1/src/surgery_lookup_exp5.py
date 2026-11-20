"""Experiment 5e: do fine-mapped spine-trait lead variants (FinnGen R12, GRCh38) carry over to spine
surgery / procedure GWAS in Pan-UKB EUR (OPCS4, GRCh37)?
For each credible set: variants of the CS ordered by PIP; first one whose rsID (from the FinnGen
sumstats via tabix) is found in 1000G EUR bim (GRCh37 position) and in the Pan-UKB file is used.
Statistics per (trait, procedure): sign concordance with the FinnGen risk allele (binomial, one-sided),
mean aligned z, and IVW slope of Pan-UKB beta on FinnGen beta.
Usage: python -I surgery_lookup_exp5.py <fm_dir> <1000G plink prefix> <panukb_manifest.tsv> <out_dir>
"""
import sys, os, zlib, urllib.request
import numpy as np
import pandas as pd
import pysam
from scipy.stats import binomtest, norm

fm, plink, man, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
TRAITS = ["M13_SPINSTENOSIS", "M13_SCIATICA", "M13_SPONDYLOSIS", "M13_SPONDYLOLISTHESIS", "M13_DORSALGIA"]
PROCS = ["V254", "V255", "V259", "V337", "V339", "A577", "V544", "V485"]
FG = "https://storage.googleapis.com/finngen-public-data-r12/summary_stats/release/finngen_R12_{}.gz"
PU = "https://pan-ukb-us-east-1.s3.amazonaws.com/"
bim = pd.concat([pd.read_csv(f"{plink}.{c}.bim", sep="\t", header=None, usecols=[0, 1, 3, 4, 5],
                             names=["chr", "rsid", "pos37", "a1", "a2"]) for c in range(1, 23)]).drop_duplicates("rsid").set_index("rsid")

sel = []
cache = f"{out}/surgery_lookup_variants.tsv"
for t in ([] if os.path.exists(cache) else TRAITS):
    cs = pd.read_csv(f"{fm}/{t}.cs.tsv", sep="\t")
    tb = pysam.TabixFile(FG.format(t), index=FG.format(t) + ".tbi")
    for (reg, k), d in cs.groupby(["region", "cs"]):
        for r in d.sort_values("prob", ascending=False).itertuples():
            c = str(r.chromosome).replace("chr", "")
            if c in ("X", "23"):
                break
            ch, pos, ref, alt = r.v.split(":")
            hit = [x.split("\t") for x in tb.fetch(c, int(pos) - 1, int(pos))]
            hit = [x for x in hit if x[2] == ref and x[3] == alt]
            if not hit or not hit[0][4].startswith("rs"):
                continue
            rs = hit[0][4].split(",")[0]
            if rs in bim.index:
                sel.append(dict(trait=t, csid=f"{reg}|{k}", rsid=rs, chr=c, pos37=int(bim.loc[rs, "pos37"]), ref=ref, alt=alt,
                                beta_fg=r.beta, p_fg=r.p, pip=r.prob))
                break
if os.path.exists(cache):
    sel = pd.read_csv(cache, sep="\t", dtype={"chr": str})
else:
    sel = pd.DataFrame(sel); sel.to_csv(cache, sep="\t", index=False)
print(sel.groupby("trait").size())

m = pd.read_csv(man, sep="\t", low_memory=False)
m = m[(m.phenocode.astype(str) == "41200") & m.coding.isin(PROCS)].set_index("coding")
rows = []
for p in PROCS:
    path = m.loc[p, "aws_path"].replace("s3://pan-ukb-us-east-1/", PU)
    tix = m.loc[p, "aws_path_tabix"].replace("s3://pan-ukb-us-east-1/", PU)
    tb = pysam.TabixFile(path, index=tix)
    # files differ: EUR-only (beta_EUR at col 6) or multi-ancestry (meta + per-ancestry columns)
    hdr = tb.header[0].lstrip("#").split("\t") if tb.header else None
    if hdr is None:
        raw = urllib.request.urlopen(urllib.request.Request(path, headers={"Range": "bytes=0-65535"})).read()
        hdr = zlib.decompressobj(31).decompress(raw).decode(errors="ignore").split("\n")[0].split("\t")
    ib, isd, ilc = hdr.index("beta_EUR"), hdr.index("se_EUR"), hdr.index("low_confidence_EUR")
    for r in sel.itertuples():
        for x in tb.fetch(r.chr, r.pos37 - 1, r.pos37):
            f = x.split("\t")
            if {f[2], f[3]} != {r.ref, r.alt}:
                continue
            if f[ib] == "NA" or f[isd] == "NA":
                continue
            b = float(f[ib]); se = float(f[isd])
            if f[3] != r.alt:          # Pan-UKB effect allele is alt; align to FinnGen alt
                b = -b
            rows.append(dict(proc=p, proc_desc=m.loc[p, "coding_description"], n_cases=m.loc[p, "n_cases_EUR"],
                             trait=r.trait, csid=r.csid, rsid=r.rsid, beta_fg=r.beta_fg, beta_pu=b, se_pu=se,
                             lowconf=f[ilc]))
L = pd.DataFrame(rows)
L.to_csv(f"{out}/surgery_lookup_long.tsv", sep="\t", index=False)
L = L[L.lowconf == "false"]
res = []
for (p, t), d in L.groupby(["proc", "trait"]):
    s = np.sign(d.beta_fg)
    z = s * d.beta_pu / d.se_pu
    conc = int((z > 0).sum())
    w = 1 / d.se_pu ** 2
    slope = (w * d.beta_fg * d.beta_pu).sum() / (w * d.beta_fg ** 2).sum()
    slope_se = np.sqrt(1 / (w * d.beta_fg ** 2).sum())
    res.append(dict(proc=p, proc_desc=d.proc_desc.iloc[0], n_cases=d.n_cases.iloc[0], trait=t, n_var=len(d),
                    concordant=conc, conc_p=binomtest(conc, len(d), 0.5, alternative="greater").pvalue,
                    mean_z=z.mean(), mean_z_p=norm.sf(z.sum() / np.sqrt(len(d))),
                    ivw_slope=slope, ivw_se=slope_se, ivw_p=norm.sf(slope / slope_se)))
R = pd.DataFrame(res)
R.to_csv(f"{out}/surgery_lookup_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(R.round(3).drop(columns="proc_desc").to_string(index=False))
