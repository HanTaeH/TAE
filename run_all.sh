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
python3 forest_plot.py         "$OUT"
