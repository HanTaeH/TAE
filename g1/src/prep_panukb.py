"""Prepare a Pan-UKB flat-file GWAS (EUR columns, GRCh37) for LDSC, read from stdin.

Maps chr:pos to rsID with the Pan-UKB EUR LD-score SNP list; biallelic SNVs, MAF >= 1%
(controls), not low-confidence, non-MHC (chr6:25.5-34.0 Mb, GRCh37 approx.).
Usage: zcat file.tsv.bgz | python prep_panukb.py <ldscore.gz> <out.tsv.gz> <n_cases> <n_controls>
"""
import sys
import numpy as np
import pandas as pd

ldfile, out, ncase, nctrl = sys.argv[1:5]
ncase, nctrl = int(ncase), int(nctrl)
ld = pd.read_csv(ldfile, sep="\t", usecols=["CHR", "SNP", "BP"], dtype={"CHR": str})
ld["key"] = ld["CHR"] + ":" + ld["BP"].astype(str)
ld = ld.drop_duplicates("key", keep=False).set_index("key")["SNP"]

cols = ["chr", "pos", "ref", "alt", "af_controls_EUR", "beta_EUR", "se_EUR", "neglog10_pval_EUR", "low_confidence_EUR"]
parts = []
for ch in pd.read_csv(sys.stdin, sep="\t", usecols=cols, chunksize=2_000_000, dtype={"chr": str, "low_confidence_EUR": str}):
    ch = ch.dropna(subset=["beta_EUR", "se_EUR", "neglog10_pval_EUR"])
    ch = ch[ch["low_confidence_EUR"].str.lower() != "true"]
    ch = ch[(ch["ref"].str.len() == 1) & (ch["alt"].str.len() == 1)]
    maf = ch["af_controls_EUR"].where(ch["af_controls_EUR"] <= 0.5, 1 - ch["af_controls_EUR"])
    ch = ch[maf >= 0.01]
    ch = ch[~((ch["chr"] == "6") & ch["pos"].between(25_500_000, 34_000_000))]
    key = ch["chr"] + ":" + ch["pos"].astype(str)
    ch = ch.assign(SNP=key.map(ld)).dropna(subset=["SNP"])
    parts.append(ch)
df = pd.concat(parts).drop_duplicates("SNP", keep=False)
res = pd.DataFrame({"SNP": df["SNP"], "A1": df["alt"], "A2": df["ref"], "BETA": df["beta_EUR"],
                    "SE": df["se_EUR"], "P": np.power(10.0, -df["neglog10_pval_EUR"]), "N": ncase + nctrl})
res.to_csv(out, sep="\t", index=False, compression="gzip")
print(f"kept {len(res):,} SNPs; N={ncase+nctrl:,} (cases {ncase:,})")
