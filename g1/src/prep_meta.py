"""Stream a FinnGen R12 + MVP + UKBB meta-analysis file and split it into
per-cohort LDSC-ready tables restricted to an LD-score SNP list (rsID).

Usage: gsutil cat <meta.tsv.gz> | zcat | python prep_meta.py <ldscore_eur.gz> <ldscore_afr.gz> <out_prefix>
Writes <out_prefix>.<cohort>.tsv.gz for cohort in fg, ukbb, MVP_EUR, MVP_AFR, MVP_HIS
(columns SNP A1 A2 BETA SE P AF). Sample sizes are added later from the manifest.
"""
import sys
import pandas as pd

ld_eur, ld_afr, out = sys.argv[1:4]
keep = set(pd.read_csv(ld_eur, sep="\t", usecols=["SNP"])["SNP"]) | \
       set(pd.read_csv(ld_afr, sep="\t", usecols=["SNP"])["SNP"])
cohorts = {"fg": "fg_af_alt", "ukbb": "ukbb_af_alt", "MVP_EUR": "MVP_EUR_af_alt",
           "MVP_AFR": "MVP_AFR_af_alt", "MVP_HIS": "MVP_HIS_af_alt"}
buf = {c: [] for c in cohorts}
cols = ["#CHR", "POS", "REF", "ALT", "rsid"] + [f"{c}_{x}" for c in cohorts for x in ("beta", "sebeta", "pval")] + list(cohorts.values())
dtypes = {k: ("float64" if k.endswith("pval") else "float32") for k in cols[5:]}
reader = pd.read_csv(sys.stdin, sep="\t", chunksize=500_000, usecols=cols, dtype=dtypes | {"#CHR": str})
for ch in reader:
    ch = ch[ch["rsid"].isin(keep) & (ch["REF"].str.len() == 1) & (ch["ALT"].str.len() == 1)]
    mhc = (ch["#CHR"].astype(str) == "6") & ch["POS"].between(25_500_000, 34_000_000)
    ch = ch[~mhc]
    for c, af in cohorts.items():
        sub = ch[["rsid", "ALT", "REF", f"{c}_beta", f"{c}_sebeta", f"{c}_pval", af]].dropna()
        sub.columns = ["SNP", "A1", "A2", "BETA", "SE", "P", "AF"]
        buf[c].append(sub)
for c in list(buf):
    df = pd.concat(buf.pop(c)).drop_duplicates("SNP", keep=False)
    maf = df["AF"].where(df["AF"] <= 0.5, 1 - df["AF"])
    df = df[maf >= 0.01]
    df.to_csv(f"{out}.{c}.tsv.gz", sep="\t", index=False, compression="gzip")
    print(out, c, len(df), flush=True)
