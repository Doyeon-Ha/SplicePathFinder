# Data and model availability

The repository contains source code, configurations and synthetic tests. It does
not bundle trained checkpoints, frozen benchmark splits, derived variant tables,
per-variant predictions, conservation sidecars or probability caches. There is no
checkpoint or benchmark-asset download endpoint in this distribution.

## Public upstream sources

| Input | Version used | Upstream |
| --- | --- | --- |
| Gene annotation and matching Genome Reference Consortium Human Build 38 (GRCh38) sequence | GENCODE v50 | [GENCODE human release 50](https://www.gencodegenes.org/human/release_50.html) |
| Clinical variant classifications | ClinVar, 2026-06-28, at least two review stars; pathogenic/likely pathogenic and benign/likely benign | [ClinVar downloads](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/) |
| Clinical breast-cancer-gene benchmark | BRCA Exchange expert-reviewed and full classification sets | [BRCA Exchange](https://brcaexchange.org/) |
| Evolutionary conservation | University of California, Santa Cruz (UCSC) GRCh38 100-way phastCons and phyloP tracks | [UCSC downloads](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/) |

The v50 training recipe expects the uncompressed annotation at
`gencode/gencode.v50.annotation.gtf` and indexed sequence at
`genome/gencode_v50/GRCh38.primary_assembly.genome.fa`. The data builder also
requires `paralog_info/grch38_paralog_flags.tsv`, the fixed gene-level paralog
flags used in the experiment; that derived input is not bundled.

## Restricted source

SpliceVarDB (Splice Variant Database) provides the experimentally tested
splice-altering versus normal labels used for epoch selection and functional
benchmarking. Obtain access directly from
[SpliceVarDB](https://compbio.ccia.org.au/splicevardb/) under its applicable terms.
This repository redistributes neither its variant table nor predictions keyed to
that table. The loader and selection code operate on a user-supplied copy.

## Required derived assets

An inference variant table must preserve genomic chromosome, 1-based position,
reference and alternate alleles, label when evaluating, transcript strand,
anchor and within-window variant offset. Inspect the anchored-table builders and
cache reader for the exact HDF5 (Hierarchical Data Format version 5) schema.

SPF (SplicePathFinder) fitting additionally needs the five frozen gene-component
folds, reported-gene selection/eligibility masks and a six-column conservation
array aligned to the cache row order. The reference cohort and GENCODE v50
Matched Annotation from NCBI and EMBL-EBI (MANE) Select cohort have different
anchor assignments and strata; keep the matching artifacts together.

Third-party model weights and reference annotations must be obtained from their
publishers. Their licenses and dataset terms apply independently of this code.
