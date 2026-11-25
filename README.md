# Neuronal signal and AD transcriptomic subtypes (D1)

Analysis code for a re-analysis of the Neff et al. (2021) Alzheimer's disease transcriptomic subtypes by
centroid projection onto public cohorts, with cell-type-matched random gene sets, a spike-in positive
control and single-nucleus follow-ups. `run_all.sh` reproduces every result, table and figure.

## Data (not included in the repository)

All inputs are public. Place them as follows (`DATA`, default `/mnt/project-files/d1-data/raw`):

| Folder | Content | Source |
| --- | --- | --- |
| `GSE33000/`, `GSE132903/`, `GSE118553/`, `GSE106241/`, `GSE125583/`, `GSE84422/` | series matrix files (and supplementary raw tables where the series matrix lacks gene symbols) | GEO |
| `GPL10558/`, `GPL24170/`, `GPL96/` | platform full tables | GEO |
| `hgnc/` | HGNC complete set (symbol mapping) | genenames.org |
| `bretigea/` | markers_df_human_brain.rda and R sources of BRETIGEA 1.0.3 | github.com/cran/BRETIGEA |
| `neff2021/` | Data files S1–S3 of Neff et al. 2021 (S2 = class centroids of the 271 classifier genes) | Sci Adv |
| `tijms2024/` | Supplementary Table 4 of Tijms et al. 2024 | Nat Aging |
| `gtex_v8/` | gene read counts for brain cortex and frontal cortex (BA9), sample and subject annotations | gs://adult-gtex (open access) |

SEA-AD inputs (`REF_DIR`, default `/mnt/project-files/d1-data/seaad`): the MTG reference h5ad
(`sea-ad-single-cell-profiling` bucket; set `SEAAD_H5AD` on first run), per-donor objects (downloaded by
`seaad_donor_aggregate.py`), `Global_and_Local_CPS.20260105.csv`, and quantitative neuropathology
`quant_neuropath/MTG_all-values_aibs-ids_20250922.csv` (`sea-ad-quantitative-neuropathology` bucket).
MSigDB Hallmark and GO Biological Process GMT files go in `GMT_DIR` (default `d1-data/msigdb`; record the
version you download). GENCODE v26 gene lengths are read from the GTEx annotation.

Controlled-access data (MSBB RNA-seq, ROSMAP) are not used.

## Run

```
pip install -r requirements.txt
DATA=... REF_DIR=... OUT=$(pwd)/results MS=<manuscript folder> bash run_all.sh
```

Seeds are fixed (`SEED + k` in each script). The last steps are `multiple_testing.py` (FDR over every
result file), `manuscript_tokens.py` (supplementary tables numbered by first citation in the manuscript
source) and `supplementary_tables.py`. Each script's docstring states its question, method and inputs.

## Layout

- `src/` analysis scripts (one question per script; v11 additions: `spike_in_real.py`, `spike_in_bound.py`,
  `msbb_neuropath_v11.py`, `msbb_kappa_v11.py`, `seaad_regulators_copath_v11.py`, `cohort_followups_v11.py`)
- `results/` outputs written by the scripts (superseded outputs are not part of the deposited results)
