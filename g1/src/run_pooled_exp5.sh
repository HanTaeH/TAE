#!/usr/bin/env bash
# G1 experiment 5a follow-up: pooled cell-type sets (neurons pooled, fibroblasts pooled, chondrocytes pooled)
# -> LD scores -> LDSC-SEG for spine traits and control traits. Resumable.
set -euo pipefail
X=${X:?work dir}; E3=${E3:?exp3 work dir}; B=${B:?venv bin}; GENES=${GENES:?genes_hg19.tsv}; HM3=${HM3:?hm3 snps}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; HERE=$(cd "$(dirname "$0")" && pwd)
PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
mkdir -p "$X/annot" "$X/cts" "$X/h2cts"
[ -f "$X/genesets.tsv" ] || "$B/python" -I "$HERE/build_sets_pooled_exp5.py" "$E3/atlas/atlas_cpm.tsv.gz" "$X/genesets.tsv"
[ -f "$X/annot/sets.22.annot.gz" ] || "$B/python" -I "$HERE/make_annot.py" "$X/genesets.tsv" "$GENES" "$PL" "$X/annot/sets"
NPROC=${NPROC_L2:-3} B=$B R=$R bash "$HERE/l2_chunk.sh" "$X/annot/sets" "$HM3" "$(seq 1 22)"
"$B/python" -I "$HERE/split_ldscores.py" "$X/annot/sets" "$X/cts"
: > "$X/sets.ldcts"
while read -r s; do echo -e "$s\t$X/cts/$s.,$E3/sldsc/cts/all_genes." >> "$X/sets.ldcts"; done < "$X/annot/sets.names"
shift 0
for f in "$@"; do t=$(basename "$f" .sumstats.gz); [ -f "$X/h2cts/$t.cell_type_results.txt" ] || echo "$f"; done | \
  xargs -P ${NPROC:-2} -I{} sh -c "t=\$(basename {} .sumstats.gz); \"$B/python\" -W ignore \"$B/ldsc.py\" --h2-cts {} --ref-ld-chr $R/1000G_Phase3_baselineLD_v2.2/baselineLD. --w-ld-chr $R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --ref-ld-chr-cts $X/sets.ldcts --out $X/h2cts/\$t > /dev/null"
touch "$X/done"
