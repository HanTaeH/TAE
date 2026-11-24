"""Atlas v3 joint models: per (trait, set A, set B), conditional coefficient z and one-sided p of A and B.
Rows: L2_2 = first set, L2_3 = second set (L2_1 = all-genes control). Usage: python -I summarise_joint_v3.py <joint_dir> <out.tsv>"""
import sys, glob, os, re
import pandas as pd
from scipy.stats import norm
SETS = ["Ligament_Fibro", "Disc_Fibro", "Chondro_pooled", "Fibro_pooled", "Bone_Osteo_stromal", "Brain_Neuron_pooled"]
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/joint_*.results")):
    s = os.path.basename(f)[6:-8]
    a, b = [x for x in SETS if "_" + x in s][:2] if False else re.findall("|".join(SETS), s)[:2]
    t = s[: s.index("_" + a)]
    r = pd.read_csv(f, sep="\t").set_index("Category")
    za, zb = r.loc["L2_2", "Coefficient_z-score"], r.loc["L2_3", "Coefficient_z-score"]
    rows.append(dict(trait=t, set_a=a, set_b=b, z_a=za, p_a=norm.sf(za), z_b=zb, p_b=norm.sf(zb)))
R = pd.DataFrame(rows); R.to_csv(sys.argv[2], sep="\t", index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
