#!/usr/bin/env bash
# G1 experiment 3b: robustness sets (top 5/20%, 50 kb window, DRG subtypes, random null sets).
# Reuses make_annot.py / split_ldscores.py; the main all-genes control (100 kb) comes from
# experiment 3 ($C3/all_genes.), the 50 kb sets use their own all_genes__w50 control.
set -euo pipefail
G=${G:?genesets_3b.tsv}; T=${T:?traits dir}; X=${X:?work dir}; C3=${C3:?exp3 cts dir}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; GENES=${GENES:?}; HM3=${HM3:?}
HERE=$(cd "$(dirname "$0")" && pwd); B=${B:-$(dirname "$(command -v python)")}
PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
mkdir -p "$X/annot" "$X/cts" "$X/h2cts"
[ -f "$X/annot/sets.22.annot.gz" ] || python -I "$HERE/make_annot.py" "$G" "$GENES" "$PL" "$X/annot/sets"
for c in $(seq 1 22); do [ -f "$X/annot/sets.$c.l2.M_5_50" ] || echo $c; done | xargs -P ${NPROC:-3} -I{} sh -c "\"$B/python\" -W ignore \"$B/ldsc.py\" --l2 --bfile $PL.{} --ld-wind-cm 1 --annot $X/annot/sets.{}.annot.gz --thin-annot --out $X/annot/sets.{} --print-snps $HM3 > /dev/null"
[ -f "$X/cts/random19.22.l2.M_5_50" ] || python -I "$HERE/split_ldscores.py" "$X/annot/sets" "$X/cts"
: > "$X/sets.ldcts"
while read -r s; do
  case $s in all_genes__w50) continue;; *__w50) ctl=$X/cts/all_genes__w50.;; *) ctl=$C3/all_genes.;; esac
  echo -e "$s\t$X/cts/$s.,$ctl" >> "$X/sets.ldcts"
done < "$X/annot/sets.names"
for f in "$T"/*.sumstats.gz; do t=$(basename "$f" .sumstats.gz); [ -f "$X/h2cts/$t.cell_type_results.txt" ] || echo "$f"; done | \
  xargs -P ${NPROC_H2:-2} -I{} sh -c "t=\$(basename {} .sumstats.gz); \"$B/python\" -W ignore \"$B/ldsc.py\" --h2-cts {} --ref-ld-chr $R/1000G_Phase3_baselineLD_v2.2/baselineLD. --w-ld-chr $R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --ref-ld-chr-cts $X/sets.ldcts --out $X/h2cts/\$t > /dev/null"
