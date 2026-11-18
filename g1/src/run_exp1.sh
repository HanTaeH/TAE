#!/usr/bin/env bash
# G1 experiment 1: per-cohort LDSC h2 and rg for five spine endpoints from the public
# FinnGen R12 + MVP + UKBB meta (gs://finngen-public-data-r12/meta_analysis/mvp_ukbb).
# Inputs: <trait>.<cohort>.tsv.gz from prep_meta.py, the meta manifest, Pan-UKB LD scores.
# EUR-ancestry cohorts (fg, ukbb, MVP_EUR) use EUR LD scores; MVP_AFR uses AFR LD scores.
# MVP_HIS (admixed, no matching LD reference) is not analysed with LDSC.
set -euo pipefail
M=${M:-/tmp/g1work/meta}; MAN=${MAN:-/tmp/g1work/FinnGen_R12_MVP_UKBB_manifest.tsv}
REF=${REF:-/mnt/project-files/g1-data/ref}; W=${W:-/tmp/g1work/exp1}
HERE=$(cd "$(dirname "$0")" && pwd); B=${B:-$(dirname "$(command -v python)")}
mkdir -p "$W/ld"; cd "$W"
for a in EUR AFR; do
  [ -f ld/$a.l2.ldscore.gz ] || cp "$REF/UKBB.$a.rsid.l2.ldscore.gz" ld/$a.l2.ldscore.gz
  [ -f ld/$a.l2.M_5_50 ] || cp "$REF/UKBB.$a.l2.M_5_50" ld/$a.l2.M_5_50
done
T=(M13_SPINSTENOSIS M13_SCIATICA M13_SPONDYLOSIS M13_SPONDYLOLISTHESIS M13_DORSALGIA)
C=(fg ukbb MVP_EUR MVP_AFR)
col() { awk -F'\t' -v p="$1" -v c="$2" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $1==p{print $h[c]}' "$MAN"; }
declare -A PREV
rm -f ns.tsv.tmp
for c in "${C[@]}"; do
  for p in "${T[@]}"; do
    case $c in fg) nc=$(col $p fg_n_cases); nt=$(col $p fg_n_controls);;
               ukbb) nc=$(col $p ukbb_n_cases); nt=$(col $p ukbb_n_controls);;
               MVP_EUR) nc=$(col $p MVP_EUR_n_cases); nt=$(col $p MVP_EUR_n_controls);;
               MVP_AFR) nc=$(col $p MVP_AFR_n_cases); nt=$(col $p MVP_AFR_n_controls);; esac
    PREV[$p.$c]=$(python -c "print(round($nc/($nc+$nt),4))")
    echo -e "$p\t$c\t$nc\t$nt" >> ns.tsv.tmp
    [ -f $p.$c.sumstats.gz ] && continue
    zcat "$M/$p.$c.tsv.gz" > tmp.tsv
    "$B/python" -W ignore "$B/munge_sumstats.py" --sumstats tmp.tsv --out $p.$c \
      --signed-sumstats BETA,0 --a1 A1 --a2 A2 --N-cas $nc --N-con $nt --ignore AF > /dev/null
  done
done
# a few rsIDs map to different biallelic records across files; drop them everywhere
python -I "$HERE/harmonise_alleles.py" *.sumstats.gz
for c in "${C[@]}"; do
  a=EUR; [ $c = MVP_AFR ] && a=AFR
  # within-cohort rg: each trait against all later traits (sample prevalence as population prevalence)
  for i in 0 1 2 3; do
    lst=""; pv=""
    for j in $(seq $i 4); do lst="$lst,${T[$j]}.$c.sumstats.gz"; pv="$pv,${PREV[${T[$j]}.$c]}"; done
    "$B/python" -W ignore "$B/ldsc.py" --rg "${lst#,}" --ref-ld ld/$a --w-ld ld/$a \
      --samp-prev "${pv#,}" --pop-prev "${pv#,}" --out within_${c}_${T[$i]} > /dev/null
  done
done
sort -u ns.tsv.tmp > ns.tsv; rm ns.tsv.tmp
# cross-cohort rg for the same trait among EUR cohorts (replication)
for p in "${T[@]}"; do
  "$B/python" -W ignore "$B/ldsc.py" --rg $p.fg.sumstats.gz,$p.ukbb.sumstats.gz,$p.MVP_EUR.sumstats.gz \
    --ref-ld ld/EUR --w-ld ld/EUR --out cross_${p}_fg > /dev/null
  "$B/python" -W ignore "$B/ldsc.py" --rg $p.ukbb.sumstats.gz,$p.MVP_EUR.sumstats.gz \
    --ref-ld ld/EUR --w-ld ld/EUR --out cross_${p}_ukbb > /dev/null
done
