"""Collect LDSC logs from the pilot into h2 and rg tables."""
import re, sys, glob, os
import pandas as pd
w, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows, h2 = [], {}
for log in sorted(glob.glob(f"{w}/rg_*.log")):
    txt = open(log).read()
    blk = txt.split("Summary of Genetic Correlation Results")[1].split("Analysis finished")[0]
    lines = [l.split() for l in blk.strip().splitlines()]
    hdr = lines[0]
    for l in lines[1:]:
        rows.append(dict(zip(hdr, l)))
    # phenotype 1 h2 + intercept
    m = re.search(r"Heritability of phenotype 1\n-+\nTotal Liability scale h2: (\S+) \((\S+)\)\nLambda GC: (\S+)\nMean Chi\^2: (\S+)\nIntercept: (\S+) \((\S+)\)\nRatio: (\S+) \((\S+)\)", txt)
    t = re.search(r"--rg (\S+?)\.sumstats", txt).group(1)
    h2[t] = dict(trait=t, h2_liab=m[1], se=m[2], lambda_gc=m[3], mean_chi2=m[4], intercept=m[5], intercept_se=m[6], ratio=m[7], ratio_se=m[8])
rg = pd.DataFrame(rows)
for c in ["p1", "p2"]:
    rg[c] = rg[c].str.replace(".sumstats.gz", "", regex=False)
for c in ["rg", "se", "z", "p"]:
    rg[c] = rg[c].astype(float)
rg["z_vs_1"] = (1 - rg["rg"]) / rg["se"]
rg.to_csv(f"{out}/rg_pilot.tsv", sep="\t", index=False)
# h2 for last trait comes from pairwise output
last = rg.iloc[-1]
h2.setdefault(last.p2, dict(trait=last.p2, h2_liab=last.h2_liab, se=last.h2_liab_se, intercept=last.h2_int, intercept_se=last.h2_int_se))
pd.DataFrame(h2.values()).to_csv(f"{out}/h2_pilot.tsv", sep="\t", index=False)
print(pd.DataFrame(h2.values()).to_string(index=False)); print(rg[["p1","p2","rg","se","z_vs_1","gcov_int"]].to_string(index=False))
