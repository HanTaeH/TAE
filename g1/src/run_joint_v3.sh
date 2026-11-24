#!/usr/bin/env bash
# Atlas v3 joint models: two cell-type annotations fitted together with baselineLD (MHC dropped) and the
# all-genes control (--h2 --overlap-annot), so each tau is conditional on the other set.
# Usage: A=<thin annot prefix> C=<v3 cts dir> BL=<baselineLD noMHC prefix> X=<out dir> B=<venv bin> run_joint_v3.sh <trait.sumstats.gz> ...
set -euo pipefail
C=${C:?}; BL=${BL:?}; X=${X:?}; B=${B:?}; R=${R:-/mnt/project-files/g1-data/ref/ldsc_1000G}
WW=$R/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.; mkdir -p "$X"
A=${A:?thin annot prefix from run_sldsc_v3.sh, e.g. <v3>/sldsc/annot/sets}; PL=$R/1000G_EUR_Phase3_plink/1000G.EUR.QC
[ -f "$C/Brain_Neuron_pooled.22.annot.gz" ] || "$B/python" -I "$(dirname "$0")/write_full_annot.py" "$A" "$PL" "$C" all_genes Ligament_Fibro Disc_Fibro Chondro_pooled Fibro_pooled Bone_Osteo_stromal Brain_Neuron_pooled
PAIRS="Ligament_Fibro:Disc_Fibro Ligament_Fibro:Chondro_pooled Fibro_pooled:Chondro_pooled Fibro_pooled:Bone_Osteo_stromal Fibro_pooled:Brain_Neuron_pooled"
for f in "$@"; do t=$(basename "$f" .sumstats.gz)
  for p in $PAIRS; do a=${p%:*}; b=${p#*:}; o="$X/joint_${t}_${a}_${b}"
    [ -f "$o.results" ] || echo "$f $a $b $o"; done
done | xargs -P ${NPROC:-2} -L1 sh -c "\"$B/python\" -W ignore \"$B/ldsc.py\" --h2 \$0 --ref-ld-chr $BL,$C/all_genes.,$C/\$1.,$C/\$2. \
  --w-ld-chr $WW --overlap-annot --print-coefficients --frqfile-chr $R/1000G_Phase3_frq/1000G.EUR.QC. --out \$3 > /dev/null"
