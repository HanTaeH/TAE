#!/usr/bin/env bash
# Compute LD scores for a list of chromosomes (skip finished ones), 2 at a time.
# Usage: l2_chunk.sh <annot_prefix> <hm3_snps> "<chr list>"
set -euo pipefail
P=$1; HM3=$2; CHRS=$3; R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; B=${B:-$(dirname "$(command -v python)")}
PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
for c in $CHRS; do [ -f "$P.$c.l2.M_5_50" ] || echo $c; done | xargs -P ${NPROC:-2} -I{} sh -c "\"$B/python\" -W ignore \"$B/ldsc.py\" --l2 --bfile $PL.{} --ld-wind-cm 1 --annot $P.{}.annot.gz --thin-annot --out $P.{} --print-snps $HM3 > /dev/null"
