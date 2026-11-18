"""Prepare FinnGen R12 summary statistics for LDSC.

Keeps SNPs in the Pan-UKB EUR LD-score SNP list (HapMap3-like set, rsID),
biallelic, MAF >= 1%, non-MHC. Writes a slim table for munge_sumstats.py.
Usage: python prep_finngen.py <finngen.gz> <ldscore.gz> <out.tsv.gz> <n_cases> <n_controls>
"""
import sys
import pandas as pd

src, ldfile, out, ncase, nctrl = sys.argv[1:6]
ncase, nctrl = int(ncase), int(nctrl)

ld = pd.read_csv(ldfile, sep="\t", usecols=["SNP"])
keep = set(ld["SNP"])

cols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta", "af_alt"]
parts = []
for chunk in pd.read_csv(src, sep="\t", usecols=cols, chunksize=2_000_000,
                         dtype={"#chrom": str, "rsids": str}):
    chunk = chunk.dropna(subset=["rsids"])
    chunk = chunk[~chunk["rsids"].str.contains(",")]          # single rsID only
    chunk = chunk[chunk["rsids"].isin(keep)]
    chunk = chunk[(chunk["ref"].str.len() == 1) & (chunk["alt"].str.len() == 1)]
    maf = chunk["af_alt"].where(chunk["af_alt"] <= 0.5, 1 - chunk["af_alt"])
    chunk = chunk[maf >= 0.01]
    # drop extended MHC (GRCh38 chr6:25.5-34.0 Mb)
    mhc = (chunk["#chrom"] == "6") & chunk["pos"].between(25_500_000, 34_000_000)
    parts.append(chunk[~mhc])
df = pd.concat(parts)
df = df.drop_duplicates("rsids", keep=False)
res = pd.DataFrame({
    "SNP": df["rsids"], "A1": df["alt"], "A2": df["ref"],
    "BETA": df["beta"], "SE": df["sebeta"], "P": df["pval"],
    "N": ncase + nctrl,
})
res.to_csv(out, sep="\t", index=False, compression="gzip")
print(f"{src}: kept {len(res):,} SNPs; N={ncase+nctrl:,} (cases {ncase:,})")
