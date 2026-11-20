#!/usr/bin/env bash
# G1 experiment 5a/5b/5c: LDSC-SEG for control traits, BMI/height-adjusted stenosis and lumbar prolapse,
# reusing the experiment-3 (27 cell types) and 3b (DRG subtypes, 20 random sets) LD scores.
set -euo pipefail
T=${T:?dir with *.sumstats.gz}; X=${X:?out dir}; E3=${E3:?exp3 work dir}; B=${B:?venv bin}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}
mkdir -p "$X/h2cts"
{ cat "$E3/sldsc/sets.ldcts"; grep -E "^(DRG_PEP|DRG_NP_prurito|DRG_Abeta_proprio|DRG_LTMR|random[0-9]+)\s" "$E3/sldsc3b/sets.ldcts"; } > "$X/sets.ldcts"
for f in "$T"/*.sumstats.gz; do t=$(basename "$f" .sumstats.gz); [ -f "$X/h2cts/$t.cell_type_results.txt" ] || echo "$f"; done | \
  xargs -P ${NPROC:-2} -I{} sh -c "t=\$(basename {} .sumstats.gz); \"$B/python\" -W ignore \"$B/ldsc.py\" --h2-cts {} --ref-ld-chr $R/1000G_Phase3_baselineLD_v2.2/baselineLD. --w-ld-chr $R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --ref-ld-chr-cts $X/sets.ldcts --out $X/h2cts/\$t > /dev/null"
