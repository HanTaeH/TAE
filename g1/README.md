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
