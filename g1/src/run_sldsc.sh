#!/usr/bin/env bash
# G1 experiment 3: LDSC-SEG cell-type enrichment.
# 1) thin annotations from genesets.tsv (+/-100 kb), 2) LD scores for all sets at once per
# chromosome (1000G EUR, HapMap3 print SNPs), 3) split into one LD-score prefix per set,
# 4) ldsc --h2-cts with baselineLD v2.2 + all-genes control, per trait.
# Needs the PyPI ldsc 2.0.1 venv with the patches listed in g1/README.md.
set -euo pipefail
A=${A:?atlas dir with genesets.tsv}; T=${T:?traits dir with *.sumstats.gz}; X=${X:?work dir}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; GENES=${GENES:?genes_hg19.tsv}; HM3=${HM3:?hm3 snp list}
HERE=$(cd "$(dirname "$0")" && pwd); B=${B:-$(dirname "$(command -v python)")}
PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
mkdir -p "$X/annot" "$X/cts" "$X/h2cts"
if [ ! -f "$X/annot/done" ]; then
  [ -f "$X/annot/sets.22.annot.gz" ] || python -I "$HERE/make_annot.py" "$A/genesets.tsv" "$GENES" "$PL" "$X/annot/sets"
  for c in $(seq 1 22); do [ -f "$X/annot/sets.$c.l2.M_5_50" ] || echo $c; done | xargs -P ${NPROC:-4} -I{} sh -c "\"$B/python\" -W ignore \"$B/ldsc.py\" --l2 --bfile $PL.{} --ld-wind-cm 1 --annot $X/annot/sets.{}.annot.gz --thin-annot --out $X/annot/sets.{} --print-snps $HM3 > /dev/null"
  touch "$X/annot/done"
fi
python -I "$HERE/split_ldscores.py" "$X/annot/sets" "$X/cts"
: > "$X/sets.ldcts"
while read -r s; do [ "$s" = all_genes ] && continue; echo -e "$s\t$X/cts/$s.,$X/cts/all_genes." >> "$X/sets.ldcts"; done < "$X/annot/sets.names"
for f in "$T"/*.sumstats.gz; do t=$(basename "$f" .sumstats.gz); [ -f "$X/h2cts/$t.cell_type_results.txt" ] || echo "$f"; done | \
  xargs -P ${NPROC:-4} -I{} sh -c "t=\$(basename {} .sumstats.gz); \"$B/python\" -W ignore \"$B/ldsc.py\" --h2-cts {} --ref-ld-chr $R/1000G_Phase3_baselineLD_v2.2/baselineLD. --w-ld-chr $R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --ref-ld-chr-cts $X/sets.ldcts --out $X/h2cts/\$t > /dev/null"
