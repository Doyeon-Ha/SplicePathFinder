# Reproduction guide

This source distribution separates the SplicePath splice-site backbone from the
SplicePathFinder (SPF) pathogenicity head. Reproducing numerical results requires
the matching checkpoint, input sequences, variant row order, frozen fold files,
eligibility masks and conservation sidecars. The supplied cohort/gene metadata
and preparation workflow define the reference inputs; trained weights and large
probability caches are obtained or computed separately.

## Experiment identity

| Experiment | Backbone and inputs | Configuration |
| --- | --- | --- |
| Main v50 model | GENCODE v50 representative transcripts, real genomic flanks, stop-codon exons included, selected epoch 10 | `config/train/splicepath_v50.yaml` |
| v50 SPF | Frozen v50 epoch-10 tracks; reported-gene anchors; six conservation scalars | `config/train/splicepathfinder_v50.yaml` |
| Epoch-8 reference | GENCODE v49 training inputs, real flanks, stop-codon-exon windows excluded, H-Net routing | `config/train/train_med_ac_exonintron_allgenes_realflank_nostopexon_hnet.yaml` |
| Epoch-8 reference SPF | Corresponding epoch-8 frozen tracks | `config/train/train_spf_sp_nostop_hnet_ep8.yaml` |

GENCODE is the reference gene annotation project. Representative v50 transcripts
follow Matched Annotation from NCBI and EMBL-EBI (MANE) Select, MANE Plus Clinical,
then Ensembl canonical. The ClinVar v50 benchmark uses MANE Select assignments.
Reference tables using a different cohort or backbone require their own inputs;
their predictions cannot be substituted by matching a genomic variant alone.

## Data preparation

For the current ClinVar benchmark, follow the complete
[ClinVar preparation and evaluation workflow](CLINVAR_WORKFLOW.md). It provides
frozen cohort/gene metadata, source-regeneration commands, explicit paths and
row-identity checks. The commands below build genome-annotation training inputs
for the backbone.

Populate the paths in `config/build_data/h5_data_regions_s1250_realflank_v50mane.yaml`
with the inputs in [DATA.md](DATA.md). From the repository root:

```bash
.venv/bin/python -m process_input.canonical_dataset_builder \
  --config-path config/build_data/h5_data_regions_s1250_realflank_v50mane.yaml
for split in train test; do
  .venv/bin/python -m process_input.h5_datafile_builder \
    --config-path config/build_data/h5_data_regions_s1250_realflank_v50mane.yaml \
    "$split" all
  .venv/bin/python -m process_input.h5_dataset_builder \
    --config-path config/build_data/h5_data_regions_s1250_realflank_v50mane.yaml \
    "$split" all
done
.venv/bin/python -m scripts.build_all_chroms_h5 \
  --variant v50mane --test-paralog all
```

The builders' split and paralog selectors must include the training genes required
by the chosen experiment; inspect `--help` before a full build. The all-genes
backbone trains on the train and test chromosome partitions together. The
annotation-only twin excludes chromosomes 1, 3, 5, 7 and 9 from both training and
epoch selection, and is evaluated on their non-paralog genes.

`process_input/clinvar_vcf_to_variant_table.py` converts Variant Call Format (VCF)
input to a variant table. `scripts/build_v50mane_reportedgene_anchors.py` resolves
reported genes and writes training eligibility masks;
`scripts/build_clinvar_v50mane_strata.py` derives transcript-relative distance
strata. Their explicit input paths and submitted-gene metadata preparation are
documented in the [ClinVar workflow](CLINVAR_WORKFLOW.md).

`scripts/build_cc_gene_5fold.py` constructs the frozen connected-component gene
folds: every set of gene tokens sharing a variant is kept together. For fold
`k`, test is `k`, validation is `(k+1)%5` and training is the remaining three
folds. Use the recorded eligibility mask to prevent anchor-gene leakage.

## Backbone training

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 -m train.train \
  --config-path config/train/splicepath_v50.yaml
```

The recipe uses seed 42, a 20-epoch cosine schedule with 10% warmup, four devices,
16 sequences per device and two gradient-accumulation steps: effective batch 128.
Mixed precision is `bfloat16`. Shortening the schedule changes learning rates and
does not reproduce an early checkpoint from the recorded run. Training selects
the epoch by SpliceVarDB validation after excluding exact ClinVar variant matches;
the selected v50 checkpoint is epoch 10. Source data access is required for this
selection step.

Preserve the run's saved configuration when resuming. Preserve device count,
micro-batch size, effective batch size, seed, shuffling and partial-batch handling.

## Frozen tracks and pathogenicity-head training

Generate the cache using the command in the main README. Backbone inference uses
batch 16, four target-covering 15,000-base chunks, stride 1,250 and an unmasked
8,750-position track. Reference and alternate channels are acceptor, donor,
neither; the head sees six probability channels plus coverage. Conservation adds
phastCons and phyloP at the variant, their window means and window maxima over
plus/minus 2,500 bases, including the variant. Generate the row-aligned six-column
sidecar with `process_input/build_conservation_scalar_sidecar.py`. Means exclude
undefined positions; phyloP is clipped to [-10,10] and divided by 10. The exact
column order and download commands are specified in [DATA.md](DATA.md).

Use the [five-fold fitting and evaluation commands](CLINVAR_WORKFLOW.md#6-fit-all-five-heads-and-evaluate-those-exact-outputs)
to connect the prepared cache, scalars, folds and eligibility mask to
`outputs/spf/fold0..4/best.pt` and pooled held-out predictions.

SPF uses batch 64 on one device, full precision, seed 42, AdamW, learning rate
0.0003, weight decay 0.05, positive-class weight 3.2, at most 50 epochs and
validation early stopping with patience 8. Adjust the fold index and output
directory together. Pool predictions only from each variant's held-out test fold.
`scripts/eval_drcons_cv5_oof.py` implements this out-of-fold (OOF) evaluation using
this checkout and explicit `--cache`, `--cvdir`, `--split-dir`, `--split-tag`,
`--coord`, `--variants-h5`, `--scalars` and `--out` paths. Each row is scored
exactly once by its held-out model.

## Analysis entry points

The [result-to-code map](PAPER_RESULTS.md) identifies the matching model,
cohort and inputs for every reported result family.

| Analysis | Source |
| --- | --- |
| Label-free delta max from cached tracks | `scripts/deltamax_from_multichunk_cache.py` |
| Cache metric aggregation | `scripts/aggregate_deltamax_cache_k4.py` |
| Sorted and position-fixed fully connected controls | `scripts/fc_probe_5fold_fulltrain.py` |
| ClinVar v50 shared-cohort tables | `scripts/assemble_v50mane_universe_tables.py` |
| Gene-clustered bootstrap | `scripts/bootstrap_spf_vs_fcp.py` |
| Shared v50 annotation inputs and evaluation | `scripts/prepare_v50_annotation.py`, `scripts/score_v50_annotation.py`, `scripts/aggregate_v50_annotation.py` |

Analysis scripts retain experiment-specific filenames under `data/eval/`. Complete
table reproduction requires those assets, including third-party predictions.
External tools use their native scoring methods and environments. A native
no-call receives zero; a missing computation for our model is not a native
no-call. Average precision is the reported precision-recall metric.

Full training and complete manuscript benchmark reproduction have not been rerun
as part of preparing this source distribution. The release validation record
reports implementation checks and inference parity using the trained checkpoint.
