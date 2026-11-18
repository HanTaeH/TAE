#!/usr/bin/env bash
# G1 experiment 0: LDSC h2 + rg for four FinnGen R12 spine endpoints.
# Needs: venv with ldsc==2.0.1 (PyPI), pandas==2.2.3, numpy==2.1.3, and two one-line
# patches to that package (see g1/README.md). Data: gs://finngen-public-data-r12,
# Pan-UKB EUR LD scores (pan-ukb-us-east-1/ld_release).
set -euo pipefail
RAW=${RAW:-/mnt/project-files/g1-data/raw/finngen_r12}
REF=${REF:-/mnt/project-files/g1-data/ref}
W=${W:-/tmp/g1work}; B=${B:-$(dirname "$(command -v python)")}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/ld"
cp "$REF/UKBB.EUR.rsid.l2.ldscore.gz" "$W/ld/eur.l2.ldscore.gz"
cp "$REF/UKBB.EUR.l2.M_5_50" "$W/ld/eur.l2.M_5_50"; cp "$REF/UKBB.EUR.l2.M" "$W/ld/eur.l2.M"
T=(M13_SPINSTENOSIS M13_INTERVERTEB M13_SCIATICA M13_LOWBACKPAIN)
declare -A PREV
for p in "${T[@]}"; do
  read -r nc nt < <(awk -F'\t' -v p="$p" '$1==p{print $4, $5}' "$RAW/finngen_R12_manifest.tsv")
  PREV[$p]=$(python -c "print(round($nc/($nc+$nt),4))")
  python -I "$HERE/prep_finngen.py" "$RAW/finngen_R12_$p.gz" "$REF/UKBB.EUR.rsid.l2.ldscore.gz" "$W/$p.prep.tsv.gz" "$nc" "$nt"
  zcat "$W/$p.prep.tsv.gz" > "$W/$p.prep.tsv"
  "$B/python" "$B/munge_sumstats.py" --sumstats "$W/$p.prep.tsv" --out "$W/$p" --signed-sumstats BETA,0 --a1 A1 --a2 A2
done
cd "$W"
for i in 0 1 2; do
  lst=""; pv=""
  for j in $(seq $i 3); do lst="$lst,${T[$j]}.sumstats.gz"; pv="$pv,${PREV[${T[$j]}]}"; done
  # population prevalence = FinnGen sample prevalence (registry-based cohort)
  "$B/python" -W ignore "$B/ldsc.py" --rg "${lst#,}" --ref-ld ld/eur --w-ld ld/eur \
     --samp-prev "${pv#,}" --pop-prev "${pv#,}" --out "rg_${T[$i]}"
done
