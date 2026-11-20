"""Is the stage association left after neuron adjustment (pooled rho 0.14) specific to the Neff
classifier? Random gene sets matched on expression decile x SD tercile (as null_models.py) are projected
with the same centroid weights, and their age/sex/neuron-adjusted stage association is compared with
the classifier's. Glial marker scores were also tried as extra covariates (reported in the text).

Usage: python stage_neuron_adjusted_null.py --data /mnt/project-files/d1-data/raw --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import neff_subtypes
from meta_reanalysis import INDEPENDENT, ca_score, rho
from pilot_gse33000 import CELL_MARKERS, NEURON_MARKERS, SEED, zscore_rows
from validate_cohorts import GROUP_ORDER, cohorts

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
_, cls = neff_subtypes.load_centroids(a.data / "neff2021")
rng = np.random.default_rng(SEED + 21)
N = a.n
rows = []
for name,e,m in cohorts(a.data):
    if name not in INDEPENDENT: continue
    e=e[m.index]
    g=cls.index.intersection(e.index); cent=cls.loc[g].sub(cls.loc[g].mean(axis=1),axis=0)
    mk=[x for x in NEURON_MARKERS if x in e.index and x not in g]
    neu=zscore_rows(e.loc[mk]).mean().values
    stage=m['group'].map(GROUP_ORDER).values.astype(float)
    age=pd.to_numeric(m['age'],errors='coerce'); age=age.fillna(age.median()).values
    sex=(m['sex'].astype(str).str.lower().str[0]=='m').astype(float).values
    cov=np.column_stack([age,sex,neu]); cov0=np.column_stack([age,sex])
    Z=zscore_rows(e)
    obs=rho(ca_score(Z.loc[g],cent).values,stage,cov)
    me=e.mean(1); dec=pd.qcut(me.rank(method='first'),10,labels=False)
    sd=e.std(1); ter=sd.groupby(dec).transform(lambda s: pd.qcut(s.rank(method='first'),3,labels=False))
    st=(dec*3+ter).astype(int)
    marker=set(NEURON_MARKERS)|{x for v in CELL_MARKERS.values() for x in v}
    poolg=e.index.difference(g).difference(list(marker))
    byst={k:poolg[st.loc[poolg].values==k] for k in st.unique()}
    nst=st.loc[g].values
    nl=[];nl0=[]
    for _ in range(N):
        pick=np.empty(len(g),dtype=object)
        for k in np.unique(nst):
            idx=np.where(nst==k)[0]; pick[idx]=rng.choice(byst[k],len(idx),replace=len(idx)>len(byst[k]))
        zz=Z.loc[list(pick)]; zz.index=g
        s=ca_score(zz,cent).values
        nl.append(rho(s,stage,cov)); nl0.append(rho(s,stage,cov0))
    nl=np.array(nl)
    rows.append({'cohort':name,'neff_adjusted':obs,'random_median':float(np.median(nl)),'random_q05':float(np.quantile(nl,.05)),'random_q95':float(np.quantile(nl,.95)),'p_empirical':float(((nl>=obs).sum()+1)/(N+1)),'random_median_agesex_only':float(np.median(nl0)),'n_random_sets':N})
    print(name, 'obs',round(obs,3),'null median',round(np.median(nl),3),'q05-95',np.round(np.quantile(nl,[.05,.95]),3),'p',round(((nl>=obs).sum()+1)/(N+1),3),'| null agesex median',round(np.median(nl0),3), flush=True)
pd.DataFrame(rows).round(4).to_csv(a.out / 'stage_neuron_adjusted_null.csv', index=False)
