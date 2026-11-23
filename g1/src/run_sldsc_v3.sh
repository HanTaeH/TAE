#!/usr/bin/env bash
# G1 atlas v3: annotations + LD scores for all v3 sets (resumable), then LDSC-SEG per trait.
# 50 kb sets use the 50 kb all-genes control, everything else the 100 kb control.
# Usage: X=<work> GENESETS=<build_sets_v3 output> B=<venv bin> GENES=... HM3=... run_sldsc_v3.sh <trait.sumstats.gz> ...
set -euo pipefail
X=${X:?}; GS=${GENESETS:?}; B=${B:?}; GENES=${GENES:?}; HM3=${HM3:?}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; HERE=$(cd "$(dirname "$0")" && pwd)
PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
mkdir -p "$X/annot" "$X/cts" "$X/h2cts"
[ -f "$X/annot/sets.22.annot.gz" ] || "$B/python" -I "$HERE/make_annot.py" "$GS" "$GENES" "$PL" "$X/annot/sets"
NPROC=${NPROC_L2:-2} B=$B R=$R bash "$HERE/l2_chunk.sh" "$X/annot/sets" "$HM3" "$(seq 1 22)"
[ -f "$X/cts/all_genes.22.l2.ldscore.gz" ] || "$B/python" -I "$HERE/split_ldscores.py" "$X/annot/sets" "$X/cts"
: > "$X/sets.ldcts"
while read -r s; do
  case $s in all_genes|all_genes__w50) continue;; *__w50) c=all_genes__w50;; *) c=all_genes;; esac
  echo -e "$s\t$X/cts/$s.,$X/cts/$c." >> "$X/sets.ldcts"
done < "$X/annot/sets.names"
for f in "$@"; do t=$(basename "$f" .sumstats.gz); [ -f "$X/h2cts/$t.cell_type_results.txt" ] || echo "$f"; done | \
  xargs -P ${NPROC:-2} -I{} sh -c "t=\$(basename {} .sumstats.gz); \"$B/python\" -W ignore \"$B/ldsc.py\" --h2-cts {} --ref-ld-chr $R/1000G_Phase3_baselineLD_v2.2/baselineLD. --w-ld-chr $R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --ref-ld-chr-cts $X/sets.ldcts --out $X/h2cts/\$t > /dev/null"
