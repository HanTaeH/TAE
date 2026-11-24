#!/bin/bash
# Experiment 7b: remove depression AND anxiety genetic components (FinnGen F5_DEPRESSIO, F5_ALLANXIOUS) from stenosis and dorsalgia
# in each cohort (Cholesky / GWAS-by-subtraction, adjust_multi.py), then LDSC-SEG on atlas v3 sets.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=/tmp/claude-0/venv/bin/python
W=/tmp/claude-0/g1work; O=$W/exp7/depanx; LD=$W/exp1/ld/EUR.l2
T=$W/v2/traits; T2=$W/v2/traits2
mkdir -p $O; cd $O
declare -A F=( [DEP]=$T/fg.F5_DEPRESSIO.sumstats.gz [ANX]=$W/exp5/traits/fg.F5_ALLANXIOUS.sumstats.gz
  [fgSTEN]=$T/fg.SPINSTENOSIS.sumstats.gz [fgDORS]=$T2/fg.DORSALGIA.sumstats.gz
  [mvpSTEN]=$T2/MVP_EUR.SPINSTENOSIS.sumstats.gz [mvpDORS]=$T2/MVP_EUR.DORSALGIA.sumstats.gz
  [ukbSTEN]=$T2/ukbb.SPINSTENOSIS.sumstats.gz [ukbDORS]=$T2/ukbb.DORSALGIA.sumstats.gz
  [metaSTEN]=$T/EURmeta.SPINSTENOSIS.sumstats.gz [metaDORS]=$T/EURmeta.DORSALGIA.sumstats.gz )
declare -A OUT=( [fgSTEN]=fg.SPINSTENOSIS [fgDORS]=fg.DORSALGIA [mvpSTEN]=MVP_EUR.SPINSTENOSIS [mvpDORS]=MVP_EUR.DORSALGIA
  [ukbSTEN]=ukbb.SPINSTENOSIS [ukbDORS]=ukbb.DORSALGIA [metaSTEN]=EURmeta.SPINSTENOSIS [metaDORS]=EURmeta.DORSALGIA )
[ -s S_depanx.npz ] || $PY -I $HERE/mvldsc.py $LD.ldscore.gz $LD.M_5_50 S_depanx $(for k in DEP ANX fgSTEN fgDORS mvpSTEN mvpDORS ukbSTEN ukbDORS metaSTEN metaDORS; do echo $k=${F[$k]}; done)
mkdir -p traits
for k in "${!OUT[@]}"; do
  o=traits/${OUT[$k]}_adjDEPANX.sumstats.gz
  [ -s $o ] || $PY -I $HERE/adjust_multi.py S_depanx.npz $k DEP,ANX ${F[$k]} ${F[DEP]},${F[ANX]} $LD.ldscore.gz $o
done
