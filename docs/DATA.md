# Data and model availability

Paths below are relative to the repository root. HDF5 means Hierarchical Data
Format version 5; VCF means Variant Call Format. Keep input checksums and row order
with each run. Source code, synthetic tests and the frozen gene-level paralog
lookup are distributed here. Trained checkpoints, probability caches and
variant-level benchmark assets have no public download endpoint in this release.

## SpliceVarDB: obtain permission from its authors

SpliceVarDB (Splice Variant Database) supplies experimentally measured
splice-altering versus normal labels for checkpoint selection and functional
evaluation. We used a commercially licensed copy purchased by our company.

**This repository does not redistribute that copy or predictions keyed to its
variants. Contact the SpliceVarDB authors via [splicevardb.org](https://splicevardb.org/)
to obtain the data and appropriate permission.** The
[official project contact page](https://github.com/CCICB/SpliceVarDB#contact) is an
additional contact route. Use `process_input/splicevardb_loader.py` with your
licensed input. Synthetic test records are artificial examples.

## GENCODE and the reference genome

The main model uses GENCODE v50, Genome Reference Consortium Human Build 38
(GRCh38), and the primary-assembly sequence from the same release.

```bash
mkdir -p gencode genome/gencode_v50 data/conservation data/clinvar_archive paralog_info
curl --fail --location \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz \
  --output gencode/gencode.v50.annotation.gtf.gz
curl --fail --location \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/GRCh38.primary_assembly.genome.fa.gz \
  --output genome/gencode_v50/GRCh38.primary_assembly.genome.fa.gz
gzip -dk gencode/gencode.v50.annotation.gtf.gz
gzip -dk genome/gencode_v50/GRCh38.primary_assembly.genome.fa.gz
cp resources/grch38_paralog_flags.tsv paralog_info/grch38_paralog_flags.tsv
```

The reference epoch-8 experiments use
[GENCODE v49 annotation and sequence](https://www.gencodegenes.org/human/release_49.html).
Select the matching paths in the reference recipe and pass `--ref-fasta`
explicitly when building a cache. The data builder creates the canonical
transcript table; v50 selection prioritizes Matched Annotation from NCBI and
EMBL-EBI (MANE) Select, MANE Plus Clinical, then Ensembl canonical.

`resources/grch38_paralog_flags.tsv` is the exact gene-name/flag lookup used for
the experiment, including the held-out annotation filter. Its upstream derivation
and annotation release were not recorded; treat this supplied file as a frozen
experimental input, not a current homology annotation. Its SHA-256 checksum is
`acd450d69c3750bfdb5155257020c7bb7a634806b807779c0d21992472b98d96`.

## ClinVar and historical classifications

The benchmark name `clinvar_20260628` records the acquisition date. The file's
header is `##fileDate=2026-06-27`; the corresponding official file is
[clinvar_20260627.vcf.gz](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260627.vcf.gz).
The compressed input used here matches the publisher's MD5 checksum
`27166aac6887cb7ac10829de69587993`. Preserve the local benchmark filename:

```bash
curl --fail --location \
  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260627.vcf.gz \
  --output data/clinvar_archive/clinvar_20260628.vcf.gz
printf '27166aac6887cb7ac10829de69587993  data/clinvar_archive/clinvar_20260628.vcf.gz\n' | md5sum --check
```

Filtering retains pathogenic/likely pathogenic and benign/likely benign variants
with at least two review stars. `process_input/clinvar_vcf_to_variant_table.py`
converts the VCF. Reported-gene anchoring additionally needs row-aligned preferred
and submitted gene metadata, obtained from `variant_summary.txt.gz` and
`submission_summary.txt.gz`. Download these through the
[ClinVar tab-delimited archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/)
and preserve the snapshot used for a run; the top-level filenames are mutable.
The exact summary snapshots and derived gene sidecars are not released here.

Temporal controls use the dated VCFs
[2018-12-17](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20181217.vcf.gz)
and [2021-06-26](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20210626.vcf.gz),
plus the submission summary. Save them under `data/clinvar_archive/` with their
original names. `scripts/build_clinvar_precirc_masks.py` builds temporal masks.
These restrict test-label eras; the trained heads use the other folds' all-era
labels, so this is not a complete removal of label-era overlap during fitting.

## BRCA Exchange

Download a dated release from the official
[BRCA Exchange releases page](https://brcaexchange.org/releases). Extract
`output/release/built_with_change_types.tsv`. The compressed archive's simplified
`variants_output.tsv` is not the input expected by this converter.

The frozen input used for the reported results has SHA-256
`61a6e400cbefee640547803393de58277f829be4306aa1ba7597a679a912e665`.
Its portal release identifier was not preserved with the local file; use the
checksum to identify a matching archived release. A current release can support
a new evaluation but does not establish reproduction of the paper's counts.

```bash
.venv/bin/python -m process_input.brca_exchange_to_variant_table \
  --tsv data/brca_exchange/output/release/built_with_change_types.tsv \
  --subset expert --gtf gencode/gencode.v49.annotation.gtf \
  --max-indel 50 --out-h5 data/brca_expert.h5
```

Run again with `--subset all` and a separate output. The expert subset uses
expert pathogenicity; the all subset uses combined classifications, excluding
uncertain or conflicting labels. Anchor the resulting tables before inference.
The retained BRCA Exchange variant tables use their recorded reference anchors;
the v50 SPF head and case anchors are specified separately in the reproduction
map. For external breast cancer gene 1/2 (BRCA1/BRCA2) evaluation use fold 3 only
when its training and validation gene exclusions have been verified.

## Evolutionary conservation

Download the two University of California, Santa Cruz (UCSC) GRCh38 tracks:

```bash
curl --fail --location \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw \
  --output data/conservation/hg38.phastCons100way.bw
curl --fail --location \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw \
  --output data/conservation/hg38.phyloP100way.bw
.venv/bin/python -m process_input.build_conservation_scalar_sidecar \
  --cache outputs/cache.h5 --cons-dir data/conservation \
  --workers 8 --out outputs/conservation_scalars.npz
```

The six columns are phastCons position/defined-position mean/maximum, then
phyloP position/defined-position mean/maximum. The window contains the variant
and 2,500 bases on each side (5,001 bases away from chromosome edges). phyloP is
clipped to [-10,10] and divided by 10. Undefined values become zero for position
and maximum; means count defined positions only. Preserve this order and cache
row alignment. Raw affected-interval conservation controls use
`scripts/eval_conservation_raw_clinvar_coord.py` and
`scripts/build_cons_interval_scalars.py`.

## External models and required derived inputs

Obtain comparison code, weights and annotations from their publishers:
[SpliceAI](https://github.com/Illumina/SpliceAI),
[Pangolin](https://github.com/tkzeng/Pangolin),
[SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer),
and [SpliceSelectNet](https://github.com/yuna728/SSNet).
Their environments and terms apply separately. The SplicePath trunk uses the
[HyenaDNA medium configuration and tokenizer](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf)
with random initialization in the paper's training recipe.

Numerical reproduction needs the matching trained backbone/head states, frozen
connected-component folds, anchor eligibility masks, submitted-gene sidecars and
row-aligned input tables. `scripts/build_cc_gene_5fold.py` generates folds;
`scripts/build_v50mane_reportedgene_anchors.py` generates anchor masks;
`scripts/build_clinvar_v50mane_strata.py` generates distance strata.
A prepared cache alone does not supply these inputs. The
[result-to-code map](PAPER_RESULTS.md) states which assets each analysis consumes.
