# SplicePathFinder

Source code for the Research in Computational Molecular Biology (RECOMB) 2027
submission. SplicePath predicts nucleotide-level splice-site probabilities;
SplicePathFinder (SPF) learns variant pathogenicity from frozen reference and
alternate probability tracks and six conservation features.

SplicePath combines a two-stage Mamba-2 dynamic chunker, a bidirectional HyenaDNA
trunk, a short-convolution branch and auxiliary exon/intron supervision. SPF uses
a radial basis function expansion, four causal Mamba-2 blocks, max pooling and
conservation features fused after pooling. Training uses plain PyTorch.

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
SPF recipe is `config/train/splicepathfinder_v50.yaml`. Tracking and notifications
are disabled in the distributed configurations.

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
