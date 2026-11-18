"""Drop rsIDs whose allele pair differs between munged .sumstats.gz files
(same rsID on different biallelic records), so LDSC --rg does not abort.
Usage: python harmonise_alleles.py <file1.sumstats.gz> [file2 ...]  (files rewritten in place)"""
import sys
import pandas as pd
fs = sys.argv[1:]
d = {f: pd.read_csv(f, sep="\t") for f in fs}
pairs = pd.concat([x.assign(k=[''.join(sorted(a + b)) for a, b in zip(x.A1, x.A2)])[["SNP", "k"]] for x in d.values()])
n = pairs.drop_duplicates().groupby("SNP").size()
bad = set(n.index[n > 1])
print("conflicting rsIDs dropped:", len(bad))
for f, x in d.items():
    x[~x.SNP.isin(bad)].to_csv(f, sep="\t", index=False, compression="gzip")
