#!/bin/bash
# Limitation 2: repeat experiments 6 / 7b with depression and anxiety GWAS from a second cohort (Pan-UKB EUR):
# DEP = self-reported depression (20002-1286), ANX = ICD-10 F41. GWAS-by-subtraction (adjust_multi.py),
# then LDSC-SEG on the atlas v3 sets (existing LD scores). Outputs *_adjUKBDEP.
# DEP+ANX joint adjustment was dropped: Pan-UKB DEP-ANX rg 0.99, so the two-covariate Cholesky step is
# ill-conditioned (coefficients -4 to -11 vs +6 to +16, share of h2 kept < 0 for some traits).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=/tmp/claude-0/venv/bin/python
W=/tmp/claude-0/g1work; O=$W/lim2; LD=$W/exp1/ld/EUR.l2
T=$W/v2/traits; T2=$W/v2/traits2
cd $O
declare -A F=( [DEP]=$O/traits/ukb.DEP.sumstats.gz [ANX]=$O/traits/ukb.ANX.sumstats.gz [FGDEP]=$T/fg.F5_DEPRESSIO.sumstats.gz
  [fgSTEN]=$T/fg.SPINSTENOSIS.sumstats.gz [fgDORS]=$T2/fg.DORSALGIA.sumstats.gz
  [mvpSTEN]=$T2/MVP_EUR.SPINSTENOSIS.sumstats.gz [mvpDORS]=$T2/MVP_EUR.DORSALGIA.sumstats.gz
  [ukbSTEN]=$T2/ukbb.SPINSTENOSIS.sumstats.gz [ukbDORS]=$T2/ukbb.DORSALGIA.sumstats.gz
  [metaSTEN]=$T/EURmeta.SPINSTENOSIS.sumstats.gz [metaDORS]=$T/EURmeta.DORSALGIA.sumstats.gz )
declare -A OUT=( [fgSTEN]=fg.SPINSTENOSIS [fgDORS]=fg.DORSALGIA [mvpSTEN]=MVP_EUR.SPINSTENOSIS [mvpDORS]=MVP_EUR.DORSALGIA
  [ukbSTEN]=ukbb.SPINSTENOSIS [ukbDORS]=ukbb.DORSALGIA [metaSTEN]=EURmeta.SPINSTENOSIS [metaDORS]=EURmeta.DORSALGIA )
[ -s S_ukbdepanx.npz ] || $PY -I $HERE/mvldsc.py $LD.ldscore.gz $LD.M_5_50 S_ukbdepanx $(for k in DEP ANX FGDEP fgSTEN fgDORS mvpSTEN mvpDORS ukbSTEN ukbDORS metaSTEN metaDORS; do echo $k=${F[$k]}; done)
mkdir -p adj
for k in "${!OUT[@]}"; do
  o=adj/${OUT[$k]}_adjUKBDEP.sumstats.gz
  [ -s $o ] || $PY -I $HERE/adjust_multi.py S_ukbdepanx.npz $k DEP ${F[$k]} ${F[DEP]} $LD.ldscore.gz $o
done
V=$W/v3
X=$V/sldsc GENESETS=$V/genesets_v3.tsv B=/tmp/claude-0/venv/bin GENES=$W/exp3/ref/genes_hg19.tsv HM3=$W/exp3/ref/hm3.snps \
  NPROC=${NPROC:-2} PATH=/tmp/claude-0/venv/bin:$PATH bash $HERE/run_sldsc_v3.sh adj/*_adjUKBDEP.sumstats.gz
echo done > adj.done
