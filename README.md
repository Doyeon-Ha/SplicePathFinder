# SplicePathFinder

Source code for the Research in Computational Molecular Biology (RECOMB) 2027
submission. SplicePath predicts nucleotide-level splice-site probabilities;
SplicePathFinder (SPF) learns variant pathogenicity from frozen reference and
alternate probability tracks and six conservation features.

SplicePath combines a two-stage Mamba-2 dynamic chunker, a bidirectional HyenaDNA
trunk, a short-convolution branch and auxiliary exon/intron supervision. SPF uses
a radial basis function expansion, four causal Mamba-2 blocks, max pooling and
conservation features fused after pooling. Training uses plain PyTorch.

The code covers the numerical results in the manuscript and supplementary
material: backbone and head fitting, ablations, native-tool adapters, variant
scoring, conservation controls, confidence intervals and figure rendering. The
[result-to-code map](docs/PAPER_RESULTS.md) identifies each reproduction path.

## Installation

Run commands from this repository's root. The locked environment targets Linux,
Python 3.12 and NVIDIA graphics processing units (GPUs), with PyTorch's Compute
Unified Device Architecture (CUDA) 12.8 build. Mamba-2 and causal convolution extensions require compatible CUDA build
tools when suitable wheels are unavailable. Install [uv](https://docs.astral.sh/uv/)
and then:

```bash
git clone https://github.com/Doyeon-Ha/SplicePathFinder.git
cd SplicePathFinder
uv sync --frozen --inexact
.venv/bin/python -c 'import torch, mamba_ssm, causal_conv1d; print(torch.__version__)'
```

`pyproject.toml` and `uv.lock` define the environment. Third-party benchmark
models, including the official SpliceAI TensorFlow distribution, need their own
environments and weights.

## Data downloads and access

We used a **commercially licensed copy of SpliceVarDB (Splice Variant Database)
purchased by our company**. It supplies functional splice-altering variant labels
for epoch selection and evaluation. We do not redistribute the data or
variant-linked outputs. **Contact the SpliceVarDB authors through the
[official website](https://splicevardb.org/) to obtain the data and appropriate
permission**; their [project contact page](https://github.com/CCICB/SpliceVarDB#contact)
provides an additional route.

| Public input | Download | Required local input |
| --- | --- | --- |
| GENCODE v50 annotation | [Gene Transfer Format file](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz) | `gencode/gencode.v50.annotation.gtf` after decompression |
| Genome Reference Consortium Human Build 38 (GRCh38) reference | [v50 primary-assembly FASTA](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/GRCh38.primary_assembly.genome.fa.gz) | `genome/gencode_v50/GRCh38.primary_assembly.genome.fa` after decompression |
| Annotation for epoch-8 reference experiments | [GENCODE v49](https://www.gencodegenes.org/human/release_49.html) | `gencode/gencode.v49.annotation.gtf` |
| ClinVar | [2026-06-27 release](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260627.vcf.gz) | Acquired on 2026-06-28, locally named `clinvar_20260628.vcf.gz`; checksum in [DATA.md](docs/DATA.md) |
| Historical ClinVar labels | [2018-12-17](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20181217.vcf.gz), [2021-06-26](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20210626.vcf.gz), [submission summaries](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/) | Temporal-control inputs |
| BRCA Exchange benchmark | [BRCA Exchange data releases](https://brcaexchange.org/releases) | Frozen release's `output/release/built_with_change_types.tsv`; expert and all-classification subsets |
| phastCons 100-way conservation | [GRCh38 bigWig](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw) | `data/conservation/hg38.phastCons100way.bw` |
| phyloP 100-way conservation | [GRCh38 bigWig](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw) | `data/conservation/hg38.phyloP100way.bw` |
| Frozen gene-level paralog flags | [Included lookup](resources/grch38_paralog_flags.tsv) | Copy to `paralog_info/grch38_paralog_flags.tsv`; provenance limits in [DATA.md](docs/DATA.md) |

The [data guide](docs/DATA.md) gives preparation commands, external model links,
version checks and required derived assets. Exact numerical reproduction also
needs row-aligned gene metadata, folds, eligibility
masks and the matching trained checkpoints. These assets have no public download
endpoint in this source release; rebuilding inputs from a newer upstream release
changes the benchmark.

## Hardware resources

CPU means central processing unit; RAM is host random-access memory; VRAM is
GPU device memory. The table records reference job allocations and provisioning
guidance, **not measured minimum hardware requirements**. Host RAM is shared by
all GPUs in a job. GPU memory is shown per device.

| Stage | GPU | VRAM per GPU | Allocated CPUs | Host RAM |
| --- | --- | --- | --- | --- |
| Genome/annotation extraction | None | — | 8 | 96 GiB reference request |
| Full training-window construction | None | — | 8 | Provision 256–400 GiB; an in-memory reference build peaked near 170 GB |
| SplicePath backbone training | 4 × NVIDIA A100 | 80 GB | 32 total | 320 GiB reference request |
| SplicePath inference / probability-cache generation | 1 × A100 or RTX A6000 | 80 GB or 48 GB, respectively | 4 | 48 GiB reference request |
| SPF head training on frozen tracks | 1 × A100 | 80 GB | 12 | 64 GiB reference request |
| Fully connected/conservation probes and statistical aggregation | None required | — | 8 | 32 GiB reference request |

The 8,750-position, batch-64 reference head recipe records approximately
**16.9 GiB peak allocated VRAM**. This is a reference measurement, not a measured
minimum for every head or the backbone. Whole-dataset construction is the largest
host-memory step; sharded inference does not require its allocation.

Release validation measured **3.41 GiB peak allocated / 4.08 GiB reserved** for a
single backbone forward on 16 synthetic 15,000-base sequences using an RTX A6000.
This excludes cache construction and head training; it is not a full-workflow
memory requirement.

Keep the backbone training recipe at **4 GPUs × micro-batch 16 × accumulation 2
= effective batch 128**, inference forward batch at **16**, and head training
batch at **64**. Preserve sequence order, batch composition and final partial
batches. The environment uses Python 3.12, PyTorch 2.11.0 with Compute Unified
Device Architecture (CUDA) 12.8, Mamba-2 2.3.2.post1 and causal-conv1d 1.6.1.
The manuscript's conservation-free head control ran on an NVIDIA B300 using
PyTorch's CUDA 13.0 build, with 12 CPUs and 64 GiB host RAM allocated. The observed
offset between A100 and B300 is about 0.004 in area under
the receiver operating characteristic curve. Reproducing small differences
requires the matching device and software environment.

## Check the implementation

The included tests use synthetic inputs. They cover transcript selection,
splice-site coordinates, insertion/deletion alignment, chunk geometry, cache
readout and the learned head. GPU-dependent tests skip when CUDA is unavailable.

```bash
.venv/bin/python -m pytest -q test
```

These tests check the implementation; they do not recompute the manuscript's
benchmark performance. See [the validation record](docs/VALIDATION.md) for the
checks performed on this distribution.

## Score variants

Full inference requires a trained backbone checkpoint, its training configuration,
a reference genome and an anchored variant table in Hierarchical Data Format
version 5 (HDF5). Download locations and required inputs are described in
[data availability](docs/DATA.md). Checkpoints, benchmark tables, frozen splits,
conservation sidecars and probability caches are not bundled in this source
distribution.

Given those inputs, generate the four-chunk probability cache and read delta max:

```bash
.venv/bin/python -m process_input.build_multichunk_pernt_cache \
  --variant-h5 data/variants.h5 \
  --checkpoint checkpoints/splicepath/epoch_010_model.pt \
  --train-config config/train/splicepath_v50.yaml \
  --ref-fasta genome/gencode_v50/GRCh38.primary_assembly.genome.fa \
  --out-h5 outputs/cache.h5 \
  --batch-size 16 --stride 1250 \
  --total-length 15000 --target-length 5000 \
  --window-left 4375 --window-right 4375 \
  --merge select --no-mask-canonical

.venv/bin/python -m scripts.deltamax_from_multichunk_cache \
  --cache outputs/cache.h5 --source-h5 data/variants.h5 \
  --out outputs/deltamax.npz --no-mask-canonical
```

The backbone forward batch is fixed at **16**. Preserve input order, batch
composition and partial batches for numerical reproduction. The merged track
contains 8,750 positions with transcript-oriented offsets `[-4375,4375)` and the
variant at index 4,375. Each position receives the reference/alternate pair from
the chunk with the largest acceptor or donor change. Uncovered positions have
zero probabilities and zero coverage. Indels use zero-padded alignment.

To score a separate dataset with a fitted SPF head, supply row-aligned conservation
features and an appropriate held-out fold:

```bash
.venv/bin/python -m scripts.score_spf_external \
  --cache outputs/cache.h5 --scalars data/conservation_scalars.npz \
  --cvdir checkpoints/splicepathfinder --folds 3 \
  --batch-size 64 --source-h5 data/variants.h5 \
  --out outputs/spf.npz --require-complete
```

Fold 3 is appropriate only for datasets whose genes were excluded from that
fold's training and validation sets. ClinVar evaluation uses pooled out-of-fold
predictions, with each row scored by its own held-out fold.

## Training and manuscript reproduction

[The reproduction guide](docs/REPRODUCIBILITY.md) maps preparation, training,
inference and aggregation to the implementation. It distinguishes the GENCODE
v50 epoch-10 model from epoch-8 reference experiments. A checkpoint's saved recipe
and input cohort determine which results it can reproduce.

The v50 backbone recipe is `config/train/splicepath_v50.yaml`; the corresponding
SPF recipe is `config/train/splicepathfinder_v50.yaml`. Checkpoints and training
logs are written to the configured output directories.

## Source provenance and third-party code

`release_manifest.json` records the frozen research source commit, per-file source
hashes and adaptations made for this distribution. No research Git history is
included. See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled code and
its licenses. Dataset and external model terms apply independently.

## License

No project-wide license has been specified. Public access to this source does
not grant a general permission to reuse, modify or redistribute the project's
original code. Bundled third-party code retains the licenses listed in
[third-party notices](THIRD_PARTY_NOTICES.md).
