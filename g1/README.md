# G1: structural vs neural cell types in the genetics of spine disease

Second project. Data are public summary statistics (FinnGen R12, Pan-UKB) – no access approval.
Plan: https://claude.ai/artifact/3hF3hQj3b9UhA7wp8UtadZ

## Experiment 0 (pilot)
`src/run_pilot.sh` → LDSC h2 and rg for spinal stenosis, intervertebral disc disorders,
sciatica and low back pain (FinnGen R12, Pan-UKB EUR LD scores, ~1.07 M SNPs, MHC removed).
`src/summarise_pilot.py <workdir> <outdir>` → `h2_pilot.tsv`, `rg_pilot.tsv`.

### Environment
Python 3.13 venv: `pip install ldsc==2.0.1 pandas==2.2.3 numpy==2.1.3`.
The PyPI `ldsc` port needs two fixes in `ldscore/sumstats.py`:
`all((i is not None for it in args.pop_prev))` → `it is not None`, and
`traceback.format_exc(ex)` → `traceback.format_exc()` (also in `ldsc.py`).
munge_sumstats.py cannot read gzip input in this port, so the prep table is unzipped first.

For `--l2` (experiment 3) three more fixes are needed:
`ldscore/ldscore.py`: `np.array(slice.decode(self._bedcode), ...)` → `np.array(list(slice.decode(self._bedcode)), ...)`;
`ldsc.py`: `df.iloc[df.SNP.isin(print_snps.SNP),:]` → `df.loc[df.SNP.isin(print_snps.SNP).values,:]`,
and the `.M` / `.M_5_50` files opened with `'wb'` → `'w'`.
`--h2-cts` needs ~4 GB RAM per process; more than three in parallel on 15 GB gets killed.

## Experiment 1
`src/prep_meta.py` (stream the FinnGen+MVP+UKBB meta file, split by cohort; run one file at a time),
`src/run_exp1.sh`, `src/harmonise_alleles.py`, `src/summarise_exp1.py` → per-cohort h2 and rg.

## Experiment 2
`src/run_exp2.sh`: `mvldsc.py` (multivariable LDSC: S, V, intercepts), `gsem.py` (genomic SEM CFA,
DWLS), `subtraction.py` (GWAS-by-subtraction, stenosis- and sciatica-specific components).

## Experiment 3
Single-cell: second venv with `scanpy leidenalg harmonypy`. `src/sc_process.py <GSE> <raw> <out>` per
dataset (QC, Harmony, Leiden, canonical-marker tables); clusters were labelled by hand from those
tables → `src/cluster_labels_manual.tsv`. `src/build_atlas.py` → 27-column atlas (+ SEA-AD MTG brain
types) and top-10% specificity gene sets. Gene coordinates (GRCh37) come from the Broad hg19 cDNA
headers + GENCODE v34 symbols (gs://gcp-public-data--broad-references). `src/meta_z.py` builds
EUR (FG+UKBB+MVP) meta Z files. `src/run_sldsc.sh` (`make_annot.py`, `split_ldscores.py`, LDSC-SEG
`--h2-cts`, baselineLD v2.2 + all-genes control), `src/summarise_exp3.py`.
