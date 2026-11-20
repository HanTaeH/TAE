#!/usr/bin/env bash
# G1 experiment 5 preparation: control / confounder FinnGen R12 GWAS -> LDSC sumstats, plus FinnGen LoF.
# Positive controls for the atlas: hip/knee OA with joint replacement (chondrocyte), depression, migraine
# (brain neuron), any pain; confounders: height, BMI; extra structural trait: lumbar disc prolapse.
set -euo pipefail
W=${W:?work dir}; REF=${REF:-/mnt/project-files/g1-data/ref}; B=${B:?venv bin}; PN=${PN:?finngen_R12_pheno_n.tsv}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/raw" "$W/traits" "$W/lof"
T=(M13_ARTHROSIS_KNEE_PRIM_KNEESURG M13_ARTHTROSIS_COX_PRIM_HIPSURG F5_DEPRESSIO G6_MIGRAINE PAIN HEIGHT_IRN BMI_IRN M13_LUMBAR_PROLAPSE)
[ -f "$W/lof/finngen_R12_lof.txt.gz" ] || gsutil -q cp gs://finngen-public-data-r12/lof/data/finngen_R12_lof.txt.gz gs://finngen-public-data-r12/lof/data/finngen_R12_lof_variants.txt gs://finngen-public-data-r12/lof/data/finngen_R12_sig.txt "$W/lof/"
for p in "${T[@]}"; do
  [ -f "$W/traits/fg.$p.sumstats.gz" ] && continue
  [ -f "$W/raw/$p.gz" ] || gsutil -q cp gs://finngen-public-data-r12/summary_stats/release/finngen_R12_$p.gz "$W/raw/$p.gz"
  read -r nc nt < <(awk -F'\t' -v p="$p" '$2==p{print $4, ($5==""?0:$5)}' "$PN")
  "$B/python" -I "$HERE/prep_finngen.py" "$W/raw/$p.gz" "$REF/UKBB.EUR.rsid.l2.ldscore.gz" "$W/$p.prep.tsv.gz" "$nc" "$nt"
  zcat "$W/$p.prep.tsv.gz" > "$W/$p.prep.tsv"
  "$B/python" -W ignore "$B/munge_sumstats.py" --sumstats "$W/$p.prep.tsv" --out "$W/traits/fg.$p" --signed-sumstats BETA,0 --a1 A1 --a2 A2 > /dev/null
  rm -f "$W/$p.prep.tsv" "$W/raw/$p.gz"
done
echo done > "$W/prep.done"
