#!/bin/bash
# Experiment 6: remove the depression genetic component (FinnGen F5_DEPRESSIO) from stenosis and dorsalgia
# in each cohort (Cholesky / GWAS-by-subtraction, adjust_multi.py), then LDSC-SEG on atlas v3 sets.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=/tmp/claude-0/venv/bin/python
W=/tmp/claude-0/g1work; O=$W/exp6; LD=$W/exp1/ld/EUR.l2
T=$W/v2/traits; T2=$W/v2/traits2
cd $O
declare -A F=( [DEP]=$T/fg.F5_DEPRESSIO.sumstats.gz
  [fgSTEN]=$T/fg.SPINSTENOSIS.sumstats.gz [fgDORS]=$T2/fg.DORSALGIA.sumstats.gz
  [mvpSTEN]=$T2/MVP_EUR.SPINSTENOSIS.sumstats.gz [mvpDORS]=$T2/MVP_EUR.DORSALGIA.sumstats.gz
  [ukbSTEN]=$T2/ukbb.SPINSTENOSIS.sumstats.gz [ukbDORS]=$T2/ukbb.DORSALGIA.sumstats.gz
  [metaSTEN]=$T/EURmeta.SPINSTENOSIS.sumstats.gz [metaDORS]=$T/EURmeta.DORSALGIA.sumstats.gz )
declare -A OUT=( [fgSTEN]=fg.SPINSTENOSIS [fgDORS]=fg.DORSALGIA [mvpSTEN]=MVP_EUR.SPINSTENOSIS [mvpDORS]=MVP_EUR.DORSALGIA
  [ukbSTEN]=ukbb.SPINSTENOSIS [ukbDORS]=ukbb.DORSALGIA [metaSTEN]=EURmeta.SPINSTENOSIS [metaDORS]=EURmeta.DORSALGIA )
[ -s S_dep.npz ] || $PY -I $HERE/mvldsc.py $LD.ldscore.gz $LD.M_5_50 S_dep $(for k in DEP fgSTEN fgDORS mvpSTEN mvpDORS ukbSTEN ukbDORS metaSTEN metaDORS; do echo $k=${F[$k]}; done)
mkdir -p traits
for k in "${!OUT[@]}"; do
  o=traits/${OUT[$k]}_adjDEP.sumstats.gz
  [ -s $o ] || $PY -I $HERE/adjust_multi.py S_dep.npz $k DEP ${F[$k]} ${F[DEP]} $LD.ldscore.gz $o
done
