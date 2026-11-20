#!/usr/bin/env bash
# Experiment 3b extras: (1) main 27-type LDSC-SEG for stenosis replication sets without FinnGen
# (UKBB, MVP EUR, UKBB+MVP meta); (2) joint models: ligament fibroblast and disc fibroblast /
# disc chondrocyte annotations fitted together (BL: baselineLD with chr6 MHC SNPs dropped so SNP rows match the set LD scores) (baselineLD + all-genes control), --h2 --overlap-annot.
set -euo pipefail
C3=${C3:?exp3 cts dir}; L=${L:?exp3 sets.ldcts}; T=${T:?traits3b dir}; X=${X:?out dir}
R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}; B=${B:-$(dirname "$(command -v python)")}
mkdir -p "$X"
BL=${BL:-$R/1000G_Phase3_baselineLD_v2.2/baselineLD.}; WW=$R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
for t in ukbb.SPINSTENOSIS MVP_EUR.SPINSTENOSIS noFGmeta.SPINSTENOSIS; do
  [ -f "$X/$t.cell_type_results.txt" ] || "$B/python" -W ignore "$B/ldsc.py" --h2-cts "$T/$t.sumstats.gz" \
    --ref-ld-chr $BL --w-ld-chr $WW --ref-ld-chr-cts "$L" --out "$X/$t" > /dev/null
done
for t in EURmeta.SPINSTENOSIS fg.SPINSTENOSIS noFGmeta.SPINSTENOSIS; do
  for pair in "Ligament_Fibro Disc_Fibro" "Ligament_Fibro Disc_Chondro" "Ligament_Fibro Bone_Osteo_stromal"; do
    set -- $pair; o="$X/joint_${t}_$1_$2"
    [ -f "$o.results" ] || "$B/python" -W ignore "$B/ldsc.py" --h2 "$T/$t.sumstats.gz" \
      --ref-ld-chr $BL,$C3/all_genes.,$C3/$1.,$C3/$2. --w-ld-chr $WW --overlap-annot --print-coefficients \
      --frqfile-chr $R/1000G_Phase3_frq/1000G.EUR.QC. --out "$o" > /dev/null
  done
done
