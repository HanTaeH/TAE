"""Collect experiment 1 LDSC logs into h2, within-cohort rg and cross-cohort rg tables.
Usage: python summarise_exp1.py <exp1_workdir> <out_dir>"""
import re, sys, glob, os
import pandas as pd
w, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

def parse(log):
    txt = open(log).read()
    blk = txt.split("Summary of Genetic Correlation Results")[1].split("Analysis finished")[0]
    ls = [l.split() for l in blk.strip().splitlines()]
    df = pd.DataFrame([dict(zip(ls[0], l)) for l in ls[1:]])
    for c in ["p1", "p2"]:
        df[c] = df[c].str.replace(".sumstats.gz", "", regex=False)
    return txt, df

h2, wr, cr = [], [], []
for log in sorted(glob.glob(f"{w}/within_*.log")):
    txt, df = parse(log)
    wr.append(df)
    m = re.search(r"Heritability of phenotype 1\n-+\nTotal (?:Liability|Observed) scale h2: (\S+) \((\S+)\)\nLambda GC: (\S+)\nMean Chi\^2: (\S+)\nIntercept: (\S+) \((\S+)\)\nRatio: (\S+) \((\S+)\)", txt)
    if m is None:  # ratio not reported when mean chi2 < 1
        m = re.search(r"Heritability of phenotype 1\n-+\nTotal (?:Liability|Observed) scale h2: (\S+) \((\S+)\)\nLambda GC: (\S+)\nMean Chi\^2: (\S+)\nIntercept: (\S+) \((\S+)\)", txt)
    g = list(m.groups()) + [None] * (8 - len(m.groups()))
    h2.append(dict(file=df.p1[0], h2_liab=g[0], se=g[1], lambda_gc=g[2], mean_chi2=g[3], intercept=g[4], intercept_se=g[5], ratio=g[6], ratio_se=g[7]))
    if len(df) == 1:  # last trait of the chain: take its h2 from the pairwise columns
        r = df.iloc[0]
        h2.append(dict(file=r.p2, h2_liab=r.h2_liab, se=r.h2_liab_se, intercept=r.h2_int, intercept_se=r.h2_int_se))
for log in sorted(glob.glob(f"{w}/cross_*.log")):
    cr.append(parse(log)[1])
h2 = pd.DataFrame(h2)
h2[["trait", "cohort"]] = h2.file.str.extract(r"(M13_[A-Z]+)\.(.+)")
wr = pd.concat(wr); cr = pd.concat(cr)
for d in (wr, cr):
    for c in ["rg", "se", "z", "p"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["z_vs_1"] = (1 - d["rg"]) / d["se"]
wr["cohort"] = wr.p1.str.split(".").str[1]
wr["t1"] = wr.p1.str.split(".").str[0]; wr["t2"] = wr.p2.str.split(".").str[0]
h2.drop(columns="file")[["trait", "cohort", "h2_liab", "se", "intercept", "intercept_se", "ratio", "ratio_se", "mean_chi2", "lambda_gc"]] \
  .sort_values(["trait", "cohort"]).to_csv(f"{out}/h2_by_cohort.tsv", sep="\t", index=False)
wr.to_csv(f"{out}/rg_within_cohort.tsv", sep="\t", index=False)
cr.to_csv(f"{out}/rg_cross_cohort.tsv", sep="\t", index=False)
short = {"M13_SPINSTENOSIS": "STEN", "M13_SCIATICA": "SCIA", "M13_SPONDYLOSIS": "SPDS", "M13_SPONDYLOLISTHESIS": "SPLT", "M13_DORSALGIA": "DORS"}
wr["pair"] = wr.t1.map(short) + "-" + wr.t2.map(short)
wr["cell"] = wr.rg.round(2).astype(str) + " (" + wr.se.round(2).astype(str) + ")"
piv = wr.pivot(index="pair", columns="cohort", values="cell")[["fg", "ukbb", "MVP_EUR", "MVP_AFR"]]
piv.to_csv(f"{out}/rg_within_wide.tsv", sep="\t")
print(h2[["trait", "cohort", "h2_liab", "se", "intercept", "intercept_se", "ratio"]].sort_values(["trait", "cohort"]).to_string(index=False))
print(piv.to_string())
print(cr[["p1", "p2", "rg", "se", "z_vs_1", "gcov_int"]].to_string(index=False))
