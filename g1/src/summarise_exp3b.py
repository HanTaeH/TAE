"""Experiment 3b summary: z-scores of robustness sets per trait, empirical p of each set against
the 20 random gene sets of the same size (fraction of random z >= observed, +1 smoothing),
and the joint (conditional) models.
Usage: python summarise_exp3b.py <h2cts_dir_3b> <cond_dir> <out_dir>"""
import sys, glob, os
import numpy as np
import pandas as pd
h2d, cond, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
d = pd.concat([pd.read_csv(f, sep="\t").assign(trait=os.path.basename(f).replace(".cell_type_results.txt", ""))
               for f in sorted(glob.glob(f"{h2d}/*.cell_type_results.txt"))])
d["z"] = d.Coefficient / d.Coefficient_std_error
rnd = d[d.Name.str.startswith("random")].groupby("trait").z.apply(np.array)
d["emp_p_vs_random"] = [(1 + (rnd[t] >= z).sum()) / (1 + len(rnd[t])) for t, z in zip(d.trait, d.z)]
d.to_csv(f"{out}/h2cts_3b_long.tsv", sep="\t", index=False)
d.pivot(index="Name", columns="trait", values="z").round(2).to_csv(f"{out}/h2cts_3b_z_wide.tsv", sep="\t")
# replication runs of the main 27 sets without FinnGen
rep = pd.concat([pd.read_csv(f, sep="\t").assign(trait=os.path.basename(f).replace(".cell_type_results.txt", ""))
                 for f in sorted(glob.glob(f"{cond}/*.cell_type_results.txt"))])
rep.to_csv(f"{out}/h2cts_main27_noFinnGen.tsv", sep="\t", index=False)
# joint models: annotation rows L2_2 (ligament) and L2_3 (comparison set)
rows = []
for f in sorted(glob.glob(f"{cond}/joint_*.results")):
    r = pd.read_csv(f, sep="\t").set_index("Category")
    name = os.path.basename(f)[6:-8]
    trait, rest = name.split(".SPINSTENOSIS_")
    other = rest.replace("Ligament_Fibro_", "")
    for cat, lab in [("L2_2", "Ligament_Fibro"), ("L2_3", other)]:
        rows.append(dict(trait=trait + ".SPINSTENOSIS", model=f"Ligament_Fibro + {other}", annotation=lab,
                         tau_z=r.loc[cat, "Coefficient_z-score"], enrichment=r.loc[cat, "Enrichment"]))
j = pd.DataFrame(rows); j.to_csv(f"{out}/joint_models.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(j.round(2).to_string(index=False))
