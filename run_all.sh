#!/usr/bin/env bash
# Reproduce every analysis and figure from raw downloaded files.
# DATA: folder with GSE33000/, GSE132903/, GSE118553/, GSE106241/, GSE125583/ (series matrix + raw/), GPL10558/, GPL24170/, hgnc/, neff2021/
# SEAAD_H5AD: SEA-AD MTG reference (Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad, ~6.4 GB,
#             https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/RNAseq/); only needed once
set -euo pipefail
DATA=${DATA:-/mnt/project-files/d1-data/raw}
REF_DIR=${REF_DIR:-/mnt/project-files/d1-data/seaad}
OUT=${OUT:-$(pwd)/results}
cd "$(dirname "$0")/src"

python3 pilot_gse33000.py      --data "$DATA" --out "$OUT/pilot"
python3 validate_cohorts.py    --data "$DATA" --out "$OUT/validation"
python3 plot_validation.py     "$OUT/validation"
python3 null_models.py         --data "$DATA" --out "$OUT/null" --n 1000
python3 sensitivity.py         --data "$DATA" --out "$OUT/sensitivity"
python3 residual_structure.py  --data "$DATA" --out "$OUT/residual" --n 500
python3 plot_sensitivity_residual.py "$OUT"
if [ ! -f "$REF_DIR/seaad_mtg_reference.csv.gz" ]; then
  python3 build_seaad_reference.py --h5ad "${SEAAD_H5AD:?set SEAAD_H5AD}" --out "$REF_DIR"
fi
python3 deconv_seaad.py        --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/deconv" --n 200
python3 plot_deconv.py         "$OUT/deconv"
python3 braak_gse106241.py     --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/braak" --n 500
python3 braak_gse125583.py     --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/braak_gse125583" --n 500
python3 other_scheme_li2022.py --data "$DATA" --out "$OUT/rq2_li2022" --n 500
python3 simulate_composition.py --data "$DATA" --ref "$REF_DIR" --out "$OUT/simulation" --n 400
if [ ! -f "$REF_DIR/donors/donor_meta.csv" ]; then  # downloads 84 donor objects (~33 GB, one at a time)
  for i in 0 1 2 3; do python3 seaad_donor_aggregate.py --out "$REF_DIR/donors" --shard $i/4 & done; wait
fi
python3 seaad_pseudobulk.py     --donors "$REF_DIR/donors" --cps "$REF_DIR/Global_and_Local_CPS.20260105.csv" --data "$DATA" --out "$OUT/seaad_pseudobulk" --n 500
python3 msbb_gse84422.py       --data "$DATA" --out "$OUT/msbb" --n 200
python3 forest_plot.py         "$OUT"
python3 neff_style_normalisation.py --data "$DATA" --out "$OUT/msbb"
python3 msbb_neuron_normalisation.py --data "$DATA" --out "$OUT/msbb"
python3 mediation.py           --res "$OUT" --seaad-meta "$REF_DIR/donors/donor_meta.csv" --n 2000
python3 singlesample.py        --data "$DATA" --out "$OUT/singlesample" --n 200
python3 marker_split.py        --data "$DATA" --out "$OUT/marker_split" --n 500
python3 residual_identity.py   --data "$DATA" --ref-dir "$REF_DIR" --out "$OUT/residual_identity" --n 1000
if [ ! -f "$REF_DIR/donors_dfc/donor_meta.csv" ]; then
  python3 seaad_donor_aggregate.py --out "$REF_DIR/donors_dfc" --prefix PFC/RNAseq/donors_objects/ \
    --donors H21.33.003 H21.33.004 H21.33.018 H21.33.019 H21.33.028 H21.33.043
fi
python3 region_reference.py    --data "$DATA" --seaad "$REF_DIR" --out "$OUT/region_reference" --n 200
# --- reviewer revision (2026-11-18) ---
python3 meta_reanalysis.py      --data "$DATA" --res "$OUT" --out "$OUT/meta_revision"
python3 classifier_specificity.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/classifier_specificity"
python3 seaad_celltype_revision.py --donors "$REF_DIR/donors" --cps "$REF_DIR/Global_and_Local_CPS.20260105.csv" --data "$DATA" --out "$OUT/seaad_celltype_revision" --n 1000
python3 mediation_revision.py   --res "$OUT" --seaad-meta "$REF_DIR/donors/donor_meta.csv" --out "$OUT/mediation_revision" --n 2000
python3 technical_confounders.py --data "$DATA" --res "$OUT" --seaad-dir "$REF_DIR" --out "$OUT/technical_confounders"
python3 simulation_revision.py  --data "$DATA" --ref "$REF_DIR" --out "$OUT/simulation_revision" --n 400
python3 deconv_validation_blocks.py --seaad-dir "$REF_DIR" --res "$OUT" --out "$OUT/deconv_validation_blocks" --n 2000
python3 residual_go.py         --data "$DATA" --gmt-dir "${GMT_DIR:-/mnt/project-files/d1-data/msigdb}" --out "$OUT/residual_go"
python3 residual_go.py         --data "$DATA" --gmt-dir "${GMT_DIR:-/mnt/project-files/d1-data/msigdb}" --out "$OUT/residual_go_nooverlap" --drop-banner-overlap
python3 seaad_sensitivity.py    --res "$OUT" --seaad-meta "$REF_DIR/donors/donor_meta.csv" --out "$OUT/seaad_sensitivity"
python3 stage_neuron_adjusted.py --data "$DATA" --out "$OUT/stage_neuron_adjusted"
python3 stage_neuron_adjusted_null.py --data "$DATA" --out "$OUT/stage_neuron_adjusted"
python3 stage_followups.py --data "$DATA" --res "$OUT" --out "$OUT/stage_followups"
python3 stage_extra_sensitivity.py --data "$DATA" --res "$OUT" --out "$OUT/stage_extra"
python3 null_specificity_matched.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/null_specificity"
python3 null_specificity_pooled.py --res "$OUT"
python3 gse33000_null_followup.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/gse33000_null_followup"
python3 msbb_normalisation_kappa_null.py --data "$DATA" --out "$OUT/msbb"
python3 other_schemes_clustered.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/other_schemes"
python3 positive_control_simulation.py --data "$DATA" --ref "$REF_DIR" --out "$OUT/positive_control"
python3 tijms_csf_projection.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/tijms_csf"
python3 tijms_followup.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --out "$OUT/tijms_followup"
python3 residual_denovo.py --data "$DATA" --ref-dir "$REF_DIR" --out "$OUT/residual_denovo"
python3 residual_quality.py --data "$DATA" --out "$OUT/residual_quality"
python3 gtex_residual_axis.py --gtex "$DATA/gtex_v8" --weights "$OUT/residual_quality/residual_template_weights.csv" --out "$OUT/gtex_residual_axis"
python3 gtex_residual_axis_qc.py --gtex "$DATA/gtex_v8" --weights "$OUT/residual_quality/residual_template_weights.csv" --out "$OUT/gtex_residual_axis_qc"
python3 seaad_residual_axis.py --donors "$REF_DIR/donors" --weights "$OUT/residual_quality/residual_template_weights.csv" --out "$OUT/seaad_residual_axis"
python3 seaad_neun_histology.py --res "$OUT" --qnp "$REF_DIR/quant_neuropath/MTG_all-values_aibs-ids_20250922.csv" --seaad-meta "$REF_DIR/donors/donor_meta.csv" --out "$OUT/seaad_neun"
python3 stage_reliability.py --data "$DATA" --out "$OUT/stage_reliability"
python3 multiple_testing.py    "$OUT"   # last: covers every report above
python3 supplementary_tables.py --res "$OUT" --out "$OUT/supplementary/Supplementary_Tables_S1-S14.xlsx"
# v10 (2026-11-25): upper-tail bound, positive control at real AD sizes, within-AD plaque
python3 bound_upper_tail.py --data "$DATA" --ref "$REF_DIR/seaad_mtg_reference.csv.gz" --res "$OUT" --out "$OUT/bound_upper_tail"
for n in 52 97 219 310; do
  python3 positive_control_simulation.py --data "$DATA" --ref "$REF_DIR" --out "$OUT/positive_control" --n-ad $n \
    --scen within_AD_subtype --fracs 0 0.02 0.05 0.1 0.25 --tag _nAD$n
done
python3 msbb_plaque_within_ad.py --res "$OUT" --out "$OUT/msbb"
python3 supplementary_tables.py --res "$OUT" --out "$OUT/supplementary/Supplementary_Tables_v10.xlsx"
