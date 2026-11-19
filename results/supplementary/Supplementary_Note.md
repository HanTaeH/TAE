# Supplementary Note

## S1. Literature search (PubMed, run 2026-11-17)

Search 2 (broad; 773 records):

```
("Alzheimer Disease"[MeSH] OR alzheimer*[tiab]) AND (subtyp*[tiab] OR "molecular subgroup*"[tiab] OR stratif*[tiab] OR heterogeneity[tiab] OR cluster*[tiab] OR taxonomy[tiab]) AND (transcriptom*[tiab] OR "gene expression"[tiab] OR "RNA-seq"[tiab] OR microarray*[tiab] OR multi-omic*[tiab]) AND (brain[tiab] OR cortex[tiab] OR hippocamp*[tiab] OR postmortem[tiab] OR "post-mortem"[tiab] OR ROSMAP[tiab] OR MSBB[tiab] OR "Mayo"[tiab]) AND ("2015"[dp] : "2026"[dp])
```

Search 1 (narrow; 366 records, all contained in Search 2) was the same query without heterogeneity, cluster*, taxonomy and the cohort names.

Flow: 773 records; 139 excluded by publication type; 634 screened on title and abstract; 374 excluded by keyword rule; 260 reviewed manually (203 passing the rule plus 57 rule-excluded records whose titles mentioned subtypes or clusters); 226 excluded on review; 34 records from 30 studies included (27 de novo subtyping, 3 applications). Two further studies were not retrieved by the search and were added from other sources: Lai et al. 2022 (Front Immunol) and Wang et al. 2024 (Front Immunol 15:1462003). Total: 32 studies (Supplementary Table S1). Features were extracted from the full text for 9 studies and from the abstract for 23. A second reviewer re-screened a random sample; agreement after reconciling the inclusion rule was 30/30 (κ = 1.00; κ = 0.39 before reconciliation).

## S2. Shared donors between GSE132903 and GSE125583

Both series come from the Banner Sun Health Research Institute. Matching the case-id suffix of the GEO sample titles identified 62 donors present in both (37 AD, 25 control). Each primary endpoint was recomputed after removing these donors from one series at a time; the bootstrap used a separate random stream so the main estimates are unchanged.

| Analysis | Pooled C−A vs neuron score (P1) | Age/sex-adjusted stage association (P2) |
|---|---|---|
| All samples | −0.924 (95% CI −0.94 to −0.90) | 0.426 (95% CI 0.34–0.51) |
| Shared donors removed from GSE132903 | −0.926 | 0.421 (95% CI 0.33–0.50) |
| Shared donors removed from GSE125583 | −0.924 | 0.405 (95% CI 0.28–0.52) |

The residual-axis gene-set test with the shared donors removed gave meta-z values correlated 0.995 with the main analysis; 15 Hallmark and 338 GO sets were significant, 317 of which were also significant in the main analysis.

## S3. MSBB cross-region test against a CDR-shuffle null

Across the 17 MSBB regions, the region-level association of the C−A score with CDR was strongly related to that region's neuron-score association with CDR (ρ = −0.868, p = 6.4 × 10⁻⁶). To test whether this simply reflects the shared CDR vector, CDR was shuffled across subjects within region (500 permutations). The null median was −0.853 and the permutation p was 0.44, so the across-region relation is not evidence beyond what shared clinical labels produce. This test was therefore removed from the secondary family and is reported only here.

## S4. Curated gene lists on the residual axis

Gene-level correlations with the residual axis (C−A after removing the neuron score) were summarised for 8 curated programmes and 20 SEA-AD subclass marker sets, as z-scores against size-matched random sets (table below; columns are cohorts and regions).

| Set | GSE33000 PFC | GSE132903 MTG | GSE118553 TC | GSE118553 FC | GSE118553 EC | GSE125583 FG |
|---|---|---|---|---|---|---|
| Oxidative phosphorylation | −2.11 | −7.50 | −12.15 | −4.85 | −5.49 | 5.09 |
| Ribosome | 0.04 | −6.24 | −5.35 | 1.80 | −3.94 | 7.91 |
| Reactive astrocyte | 1.79 | 1.10 | 1.88 | 1.98 | 2.31 | 1.60 |
| Hypoxia | −1.49 | −1.48 | −1.74 | −2.19 | −0.85 | −0.88 |
| Microglia activation / complement | 0.31 | 0.37 | 0.55 | 2.41 | 4.51 | 1.54 |
| Immediate early | 2.15 | 0.89 | 0.31 | −0.39 | 2.06 | 1.61 |

Only reactive-astrocyte genes had the same sign in every cohort, and none of the 20 subclass marker sets was consistent across cohorts. Oxidative phosphorylation and ribosome genes reversed sign in the RNA-seq cohort (GSE125583), consistent with platform-specific technical structure rather than a shared biological programme. The full table is in `results/residual_identity/gene_set_z_by_cohort.csv`; the Hallmark analysis with the corrected variance-inflation estimate is Supplementary Table S6.
