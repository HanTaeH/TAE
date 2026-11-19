#!/usr/bin/env bash
# G1 experiment 2: multivariable LDSC -> genomic SEM factor models -> GWAS-by-subtraction
# (stenosis-specific and sciatica-specific components) -> LDSC validation.
# Needs the munged per-cohort sumstats and LD files from run_exp1.sh.
set -euo pipefail
E=${E:-/tmp/g1work/exp1}; W=${W:-/tmp/g1work/exp2}
HERE=$(cd "$(dirname "$0")" && pwd); B=${B:-$(dirname "$(command -v python)")}
mkdir -p "$W"; cd "$W"
C=(fg ukbb MVP_EUR MVP_AFR)
for c in "${C[@]}"; do
  a=EUR; [ $c = MVP_AFR ] && a=AFR
  python -I "$HERE/mvldsc.py" $E/ld/$a.l2.ldscore.gz $E/ld/$a.l2.M_5_50 S_$c \
    STEN=$E/M13_SPINSTENOSIS.$c.sumstats.gz SPDS=$E/M13_SPONDYLOSIS.$c.sumstats.gz \
    SPLT=$E/M13_SPONDYLOLISTHESIS.$c.sumstats.gz SCIA=$E/M13_SCIATICA.$c.sumstats.gz \
    DORS=$E/M13_DORSALGIA.$c.sumstats.gz
  # Cholesky subtraction: first-listed trait = shared factor only (its output equals that trait's own GWAS, discarded)
  python -I "$HERE/subtraction.py" S_$c.npz SCIA STEN $E/M13_SCIATICA.$c.sumstats.gz $E/M13_SPINSTENOSIS.$c.sumstats.gz $E/ld/$a.l2.ldscore.gz sub_${c}_SCIAfirst
  python -I "$HERE/subtraction.py" S_$c.npz STEN SCIA $E/M13_SPINSTENOSIS.$c.sumstats.gz $E/M13_SCIATICA.$c.sumstats.gz $E/ld/$a.l2.ldscore.gz sub_${c}_STENfirst
  rm -f sub_${c}_*.shared.sumstats.gz
done
python -I "$HERE/gsem.py" sem_fits.tsv S_fg.npz S_ukbb.npz S_MVP_EUR.npz S_MVP_AFR.npz
run() { "$B/python" -W ignore "$B/ldsc.py" --rg "$1" --ref-ld $E/ld/$2 --w-ld $E/ld/$2 --out "$3" > /dev/null; }
for c in "${C[@]}"; do
  a=EUR; [ $c = MVP_AFR ] && a=AFR
  t="$E/M13_SPONDYLOSIS.$c.sumstats.gz,$E/M13_DORSALGIA.$c.sumstats.gz,$E/M13_SPONDYLOLISTHESIS.$c.sumstats.gz"
  run sub_${c}_SCIAfirst.STEN_specific.sumstats.gz,$E/M13_SCIATICA.$c.sumstats.gz,$E/M13_SPINSTENOSIS.$c.sumstats.gz,sub_${c}_STENfirst.SCIA_specific.sumstats.gz,$t $a val_${c}_STENspec
  run sub_${c}_STENfirst.SCIA_specific.sumstats.gz,$E/M13_SPINSTENOSIS.$c.sumstats.gz,$E/M13_SCIATICA.$c.sumstats.gz,$t $a val_${c}_SCIAspec
done
for s in SCIAfirst.STEN_specific STENfirst.SCIA_specific; do
  run sub_fg_$s.sumstats.gz,sub_ukbb_$s.sumstats.gz,sub_MVP_EUR_$s.sumstats.gz EUR cross_fg_$s
  run sub_ukbb_$s.sumstats.gz,sub_MVP_EUR_$s.sumstats.gz EUR cross_ukbb_$s
done
