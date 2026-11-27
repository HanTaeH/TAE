#!/bin/bash
# Limitation 2: depression / anxiety GWAS from a second cohort (Pan-UKB EUR, public) -> LDSC sumstats.
# DEP = self-reported depression (20002-1286); ANX = ICD-10 F41 other anxiety disorders.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/tmp/claude-0/venv/bin
W=/tmp/claude-0/g1work/lim2; REF=/mnt/project-files/g1-data/ref; S3=https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files
mkdir -p $W/traits; cd $W
while read -r tag file nc nt; do
  [ -s traits/ukb.$tag.sumstats.gz ] && continue
  curl -s --retry 4 $S3/$file | zcat | $B/python -I $HERE/prep_panukb.py $REF/UKBB.EUR.rsid.l2.ldscore.gz $tag.prep.tsv.gz $nc $nt
  zcat $tag.prep.tsv.gz > $tag.prep.tsv
  $B/python -W ignore $B/munge_sumstats.py --sumstats $tag.prep.tsv --out traits/ukb.$tag --signed-sumstats BETA,0 --a1 A1 --a2 A2 > /dev/null
  rm -f $tag.prep.tsv
done <<LIST
DEP categorical-20002-both_sexes-1286.tsv.bgz 24071 396402
ANX icd10-F41-both_sexes.tsv.bgz 9006 411525
LIST
echo done > prep.done
