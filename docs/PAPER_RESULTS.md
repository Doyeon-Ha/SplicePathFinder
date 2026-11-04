# Manuscript result-to-code map

Scope: the main manuscript and supplementary material at manuscript commit
`6867ec4c0d3971cc136043fa2ac5491bc5ce7a88`. This map covers reported numerical
results. Columns printed as pending require no additional model implementation
in this release. Table numbers may change; manuscript labels identify the result.

SplicePathFinder is abbreviated SPF; splice-altering variant is SAV;
fully connected is FC; out-of-fold is OOF. MANE denotes Matched Annotation from
NCBI and EMBL-EBI. HDF5 denotes Hierarchical Data Format version 5.

| Result | Implementation / entry point | Matching inputs and model |
| --- | --- | --- |
| Architecture and chunk construction | `model/model.py`, `model/chunker.py`, `process_input/build_multichunk_pernt_cache.py` | Bidirectional HyenaDNA, two H-Net stages, Mamba-2 encoder/decoder and short-convolution branch |
| Main cross-task annotation result | `scripts/prepare_v50_annotation.py`, `scripts/score_v50_annotation.py`, `scripts/aggregate_v50_annotation.py` | Annotation-only v50 twin, epoch 8; held-out chromosomes 1, 3, 5, 7, 9; non-paralog genes; identical labels across tools |
| Main ClinVar fine strata (`tab:clinvar-cv5-region-fine`) | `scripts/build_clinvar_v50mane_strata.py`, `scripts/assemble_v50mane_universe_tables.py` | 107,426-variant v50 MANE splice-region universe; v50 epoch-10 backbone and matching OOF head scores |
| Main learned-head fitting and OOF scoring | `train/train_sav_classifier.py`, `scripts/eval_drcons_cv5_oof.py` | Connected-component gene folds; reported-gene eligibility; six conservation features; no train/validation/test gene overlap |
| Main conservation and head ablation (`tab:ablation-summary`, `tab:deltareader-ablation-short`) | `scripts/fc_probe_5fold_fulltrain.py`, `scripts/lucy_cons_only_probe_cc.py`, `train/train_sav_classifier.py` | Sorted FC, FC plus conservation at input, six-feature logistic regression, SPF without conservation and SPF; retain each result's fitted cohort |
| Main hard-stratum precision-recall plot | `scripts/render_recomb2027_figures.py --only-pr` | Row-aligned `analysis_frame.npz`, `clinvar_metrics.json`, `spf_clinical_cutoffs.json`; current v50 scores and masks |
| BRCA Exchange (`tab:brca`) | `process_input/brca_exchange_to_variant_table.py`, `scripts/score_spf_external.py`, `scripts/brca_v50mane_spf_metrics.py` | Frozen expert/all tables; recorded reference anchors; v50 reported-gene SPF fold 3, with training and validation exclusion verified |
| Retinoblastoma 1 (RB1) and BRCA1-associated RING domain 1 (BARD1) cases | `scripts/prepare_paper_cases.py`, `scripts/score_paper_cases.py`, `scripts/render_recomb2027_figures.py --only-cases` | ClinVar variation IDs 13096 and 2854936; reported-gene v50 MANE anchors; fixed-batch cache; held-out head and batch-64 controls |
| Functional SAV holdout | `scripts/build_svdb_validation_h5.py`, `scripts/refresh_recomb_svdb_holdout.py` | Licensed SpliceVarDB; epoch-8 reference model; exact ClinVar-overlap complement of the epoch-selection rows |
| Supplementary backbone ablations | `train/train.py`, `scripts/assemble_v50mane_universe_tables.py` | v50 no-region epoch 2; no-bidirectionality/no-hybrid epoch 10; no-region epoch-10 trajectory is a distinct arm |
| Supplementary reference backbone arms | Same training and cache implementation, epoch-8 reference configuration | Reference no-region epoch 4, causal epoch 8, no-hybrid epoch 7; retain the reference cohort |
| Supplementary head and base-swap grids | `train/train_sav_classifier.py`, `process_input/build_pangolin_pernt_cache.py`, `process_input/build_spliceai_official_pernt_cache.py`, table assembler/renderer | Matching frozen native-tool tracks and fitted head states; actual input channel count comes from each cache |
| Supplementary raw conservation controls | `scripts/eval_conservation_raw_clinvar_coord.py`, `scripts/build_cons_interval_scalars.py` | Anchor / affected-reference-interval mean / maximum; normalized phyloP and phastCons |
| Supplementary temporal controls | `scripts/build_clinvar_precirc_masks.py`, `scripts/v50mane_universe_followups.py` | Historical 2018/2021 ClinVar labels, current gene components and frozen model predictions |
| Supplementary operating points and intervals | `scripts/v50mane_universe_followups.py`, `scripts/spf_clinical_cutoffs.py` | Thresholds 0.5, 0.75, 0.9; seed 42; 2,000 gene-component resamples |
| Supplementary paired uncertainty, including epoch-12 comparison | `scripts/v50mane_universe_followups.py`, `scripts/bootstrap_spf_vs_fcp.py` | Retained epoch-12 and corresponding reference predictions; paired rows and gene-component IDs |
| Supplementary SAV bootstrap | `scripts/refresh_recomb_svdb_holdout.py` | Licensed holdout rows; fixed fitted models; seed 42, 2,000 component resamples |
| Supplementary forward-only efficiency | `scripts/bench_recomb_current.py` | Epoch-8 reference backbone/head; four chunks; forward batch 16; 68 variants with the first four excluded; head timing batch 16; NVIDIA A100 80 GB |

DNA language-model comparison columns without reported values do not require
Evo 2, Caduceus, NeoBERT, Qwen or separate pretraining implementations. The shipped
HyenaDNA implementation is the trunk used by SplicePath. Shared alignment,
chunking, checkpoint and dataset helpers remain because the retained entry points
use them.

## Model and ablation identity

The main v50 backbone uses `config/train/splicepath_v50.yaml`. Keep its 20-epoch
schedule when selecting epoch 10. The annotation twin uses the held-out training
input and epoch 8. Supplementary epoch-8 reference experiments use
`config/train/train_med_ac_exonintron_allgenes_realflank_nostopexon_hnet.yaml`;
retained epoch-12 head comparisons use their corresponding configuration.

Backbone ablation overrides are `model.aux_heads.enable_exon=false`,
`model.hyena_bidirectional=false`, and `model.use_hybrid_backbone=false`,
respectively. Keep all other recipe values and use separate run/output names.
For the conservation-free head use `feature.scalar_npz=null` with the same
cache, folds, eligibility and batch size. The head infers its scalar feature
count from the dataset. The paper's two head variants were trained on different
GPU architectures; their difference includes that measured environment effect.

## Run aggregation from the recorded inputs

The analysis entry points resolve their recorded relative paths below
`data/eval/`. Populate those paths with the matching prepared inputs; their
constants identify the required artifact names. Outputs are written to an
explicit directory:

```bash
mkdir -p outputs/manuscript
.venv/bin/python -m scripts.assemble_v50mane_universe_tables \
  --out-dir outputs/manuscript_metrics
.venv/bin/python -m scripts.v50mane_universe_followups \
  --out-dir outputs/manuscript_metrics
.venv/bin/python -m scripts.render_v50mane_universe_tables \
  --metrics outputs/manuscript_metrics/clinvar_metrics_v50mane_universe.json \
  --manuscript outputs/manuscript
.venv/bin/python -m scripts.render_v50mane_universe_followups \
  --dir outputs/manuscript_metrics --manuscript outputs/manuscript
```

Create `outputs/manuscript` before running the formatters. The assembler requires
full-release native scores, raw conservation readouts, fold predictions, MANE
strata and backbone ablation delta-max arrays. These input predictions are
computed by the scoring paths above; the assembler does not fit a model or
produce missing predictions. Its `pending` output distinguishes unavailable arms
from scored variants. Use the v50 head keys (`SPF_v50`, `SPF_nocons_v50`) for the
main head comparison; retained reference-score keys identify separate experiments.

The plot renderer consumes saved, row-aligned plotting inputs. Its integrated
precision-recall values are checked against the supplied metric JSON. Case
preparation writes `variants.h5` and identity metadata; run the standard cache and
SpliceAI scorers before plotting. `score_paper_cases.py` preserves the case-specific
MANE/transcript checks and 62 reference-control rows used for the batch-64 head
forward; these require the same frozen clinical inputs and held-out head.

The efficiency command exposes its reference genome, variant table, backbone,
training configuration and head as arguments. It times network forwards with
CUDA events, excluding loading, encoding, alignment, merging and disk I/O.
It does not measure end-to-end throughput or fresh training time.

## Availability and validation boundary

The code is sufficient to define the retained computations. Exact runs also
require the matching licensed source and frozen derived inputs described in
[DATA.md](DATA.md). The public release currently has no download endpoint for
trained weights or saved plotting frames. Frozen ClinVar cohort/gene metadata
are supplied in `resources/`; the [ClinVar workflow](CLINVAR_WORKFLOW.md)
regenerates folds, anchor masks, strata and conservation arrays. The BRCA Exchange snapshot is identified by checksum; its
release identifier remains unrecorded. These availability limits affect
end-to-end reproduction even when the implementation tests pass.

[VALIDATION.md](VALIDATION.md) records which checks were actually run. Full model
training and the complete manuscript benchmark are separate from source and
checkpoint-parity validation.
