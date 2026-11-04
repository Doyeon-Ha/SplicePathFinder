# ClinVar preparation, fitting and held-out evaluation

Run from the repository root in the locked environment. This workflow uses the
current GENCODE v50 MANE (Matched Annotation from NCBI and EMBL-EBI) model and
five connected-component gene folds. HDF5 means Hierarchical Data Format version
5; VCF means Variant Call Format; OOF means out-of-fold; GPU means graphics
processing unit.

The supplied `resources/clinvar20260628_rows.tsv.gz` fixes the paper cohort's
378,543 variant identities, order and reference annotation anchors. The supplied
`resources/clinvar20260628_genes.tsv.gz` fixes the submitted/preferred gene table.
Both contain public ClinVar/GENCODE metadata. Checksums and field provenance are
in `resources/clinvar20260628_metadata.json`. They contain variant alleles and
identities, without model input windows, probability tracks, model weights or
SpliceVarDB (Splice Variant Database) data.
The model's evaluation universe contains 107,426 of these cohort rows.

Each compressed tab-separated table has a header. `key` is
`chrom:pos:ref:alt`, with no `chr` prefix, a 1-based Genome Reference Consortium
Human Build 38 (GRCh38) position and uppercase
alleles. The cohort table has `key`, binary `label`, `gene_symbol`, diagnostic
`legacy_splice`, and `v50_row`, `v50_gene`, `v50_strand`, `v50_offset`,
`v50_chunk_idx`, `v50_local_idx`; `v50_row=-1` denotes an absent reference anchor.
The gene table has `key`, `submitted_genes`, `preferred_gene`; multiple submitted
genes use `|`. All complete-cohort files preserve the cohort table's row order.

## 1. Download public inputs and prepare identity tables

Follow [DATA.md](DATA.md) for the dated ClinVar VCF, GENCODE annotation/reference,
and both conservation tracks. The exact paralog lookup is supplied at
`resources/grch38_paralog_flags.tsv`. The following commands generate the v50
canonical transcript table and the raw VCF table:

```bash
mkdir -p outputs/clinvar paralog_info
cp resources/grch38_paralog_flags.tsv paralog_info/grch38_paralog_flags.tsv
.venv/bin/python -m process_input.canonical_dataset_builder \
  --config-path config/build_data/h5_data_v50mane_anchor.yaml
.venv/bin/python -m process_input.clinvar_vcf_to_variant_table \
  --vcf data/clinvar_archive/clinvar_20260628.vcf.gz \
  --gtf gencode/gencode.v50.annotation.gtf --min-star 2 \
  --out-h5 outputs/clinvar/raw.h5
.venv/bin/python -m scripts.prepare_clinvar_release \
  --input-h5 outputs/clinvar/raw.h5 \
  --cohort-manifest resources/clinvar20260628_rows.tsv.gz \
  --out-dir outputs/clinvar
cp resources/clinvar20260628_genes.tsv.gz outputs/clinvar/submitted_genes.tsv.gz
```

`release.h5` preserves the paper's cohort order. `v50_anchors.h5` preserves the
378,002-row v50 reference-anchor order. These are small metadata tables; the cache
builder obtains deoxyribonucleic acid (DNA) sequences from the reference genome.
The cohort selector checks every
variant against the raw VCF-derived table and rejects changed labels, gene
annotations, missing identities or duplicates. It does not select variants by
model scores. The `legacy_splice` field preserves the reference diagnostic mask;
the current MANE evaluation mask is generated in step 3.

To regenerate submitted-gene metadata, obtain `variant_summary.txt.gz` and
`submission_summary.txt.gz` from the
[ClinVar tab-delimited archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/)
and run:

```bash
.venv/bin/python -m scripts.build_clinvar_gene_metadata \
  --variant-h5 outputs/clinvar/release.h5 \
  --variant-summary data/clinvar_archive/variant_summary.txt.gz \
  --submission-summary data/clinvar_archive/submission_summary.txt.gz \
  --out outputs/clinvar/submitted_genes.tsv.gz
```

The generator joins GRCh38 coordinates to VariationID, then to all matching
submissions, including submissions that do not contribute to the aggregate
classification. It retains missing submitted genes and uses the first matching
preferred Name. It writes `key`, `submitted_genes`, `preferred_gene` in cohort
order and records source hashes. Compare the regenerated table with the supplied
frozen table before claiming paper reproduction; newer summary snapshots can
change genes even when the dated VCF is unchanged.
The source-summary 256-bit Secure Hash Algorithm (SHA-256) hashes are recorded under `gene_source_sha256` in the
metadata JSON (JavaScript Object Notation) file.

## 2. Generate the frozen gene-component folds

```bash
.venv/bin/python -m scripts.build_cc_gene_5fold \
  --variant-h5 outputs/clinvar/release.h5 \
  --out-dir outputs/clinvar/folds --tag cc_seed42
```

Genes sharing a variant belong to one connected component. The split uses seed
42; test is fold `k`, validation is `(k+1)%5`, and the other three folds train.
Each file contains the three index arrays and a digest of cohort identity/order.
No prediction cache or trained model is needed to generate the folds.

## 3. Generate reported-gene anchors, eligibility and MANE strata

```bash
.venv/bin/python -m scripts.build_v50mane_reportedgene_anchors \
  --release-h5 outputs/clinvar/release.h5 \
  --v50-h5 outputs/clinvar/v50_anchors.h5 \
  --mane-table data/canonical_dataset_v50_mane_regions.txt \
  --submitted-genes outputs/clinvar/submitted_genes.tsv.gz \
  --fold-npz outputs/clinvar/folds/gene_grouped_5fold_cc_seed42.npz \
  --out outputs/clinvar/anchors
.venv/bin/python -m scripts.build_clinvar_v50mane_strata \
  --release-h5 outputs/clinvar/release.h5 \
  --mane-table data/canonical_dataset_v50_mane_regions.txt \
  --anchor-selection outputs/clinvar/anchors/anchor_selection_v50mane.npz \
  --out outputs/clinvar/coord.npz
```

The anchor builder checks every gene-table key against the cohort, chooses the
reported-gene anchor without model scores, and excludes anchor genes spanning
multiple component folds from fitting. It writes `train_mask`, anchor identities
and `alt_anchors.h5` for changed anchors. The strata file contains labels, splice
region membership, intron/exon distances and a cohort-order digest.

## 4. Score reference and alternative anchors

A trained v50 epoch-10 backbone is required. There is currently no plan to
publicly release its weights. Use a checkpoint produced by
[the backbone training recipe](REPRODUCIBILITY.md#backbone-training). Set the local
checkpoint variable below. Training writes a timestamped run directory under
`checkpoints/`; replace `YYYYMMDD_HHMMSS` with that run's timestamp. Run these GPU
commands using your compute scheduler.

```bash
BACKBONE=checkpoints/splicepath-v50mane-allgenes-realflank-hnet-20ep_YYYYMMDD_HHMMSS/epoch_010_model.pt
build_parts() {
  local input_h5="$1" output_dir="$2" count first part
  mkdir -p "$output_dir"
  count=$(.venv/bin/python -c 'import h5py,sys; f=h5py.File(sys.argv[1]); print(len(f["label"]))' "$input_h5")
  for ((first=0; first<count; first+=20000)); do
    printf -v part '%05d' "$((first/20000))"
    .venv/bin/python -m process_input.build_multichunk_pernt_cache \
      --variant-h5 "$input_h5" --checkpoint "$BACKBONE" \
      --train-config config/train/splicepath_v50.yaml \
      --ref-fasta genome/gencode_v50/GRCh38.primary_assembly.genome.fa \
      --out-h5 "$output_dir/part_$part.h5" \
      --start "$first" --end "$((first+20000))" \
      --batch-size 16 --block-size 500 --stride 1250 \
      --total-length 15000 --target-length 5000 \
      --window-left 4375 --window-right 4375 --merge select --no-mask-canonical
  done
}
build_parts outputs/clinvar/v50_anchors.h5 outputs/clinvar/base_parts
build_parts outputs/clinvar/anchors/alt_anchors.h5 outputs/clinvar/alt_parts
.venv/bin/python -m scripts.assemble_v50mane_reportedgene_cache \
  --release-h5 outputs/clinvar/release.h5 \
  --v50-h5 outputs/clinvar/v50_anchors.h5 \
  --anchor-selection outputs/clinvar/anchors/anchor_selection_v50mane.npz \
  --alt-h5 outputs/clinvar/anchors/alt_anchors.h5 \
  --coord outputs/clinvar/coord.npz \
  --parts-dir outputs/clinvar/base_parts --alt-dir outputs/clinvar/alt_parts \
  --out-h5 outputs/clinvar/cache.h5
```

Preserve row order, shard boundaries, forward batch 16, block size 500 and partial
batches. The assembler checks shard order and geometry, copies reference or
changed-anchor tracks according to the saved selection, and restores absent v50
rows as zero tracks and coverage. Its complete cache is aligned to `release.h5`
and contains variant identities, labels and the fitting mask.

## 5. Generate conservation features and verify alignment

```bash
.venv/bin/python -m process_input.build_conservation_scalar_sidecar \
  --cache outputs/clinvar/cache.h5 --variant-h5 outputs/clinvar/release.h5 \
  --cons-dir data/conservation --workers 8 \
  --out outputs/clinvar/scalars.npz
.venv/bin/python -m scripts.verify_clinvar_inputs \
  --cache outputs/clinvar/cache.h5 --variants-h5 outputs/clinvar/release.h5 \
  --coord outputs/clinvar/coord.npz --scalars outputs/clinvar/scalars.npz \
  --split-dir outputs/clinvar/folds --split-tag cc_seed42 \
  --anchor-selection outputs/clinvar/anchors/anchor_selection_v50mane.npz
```

The scalar columns are phastCons position/defined-position mean/maximum, followed
by phyloP position/defined-position mean/maximum over the variant ±2,500 bases.
phyloP is clipped to [-10,10] and divided by 10. The verifier checks identities,
order, labels, six finite scalar columns, and exactly one test-fold assignment
per cohort row. Do not repair a failed identity check by resizing arrays.

## 6. Fit all five heads and evaluate those exact outputs

```bash
for fold in 0 1 2 3 4; do
  .venv/bin/python -m train.train_sav_classifier \
    --config-name splicepathfinder_v50 \
    cache_h5=outputs/clinvar/cache.h5 \
    feature.scalar_npz=outputs/clinvar/scalars.npz \
    split.freeze_npz=outputs/clinvar/folds/fold${fold}_split_cc_seed42.npz \
    split.splice_mask_npz=outputs/clinvar/anchors/anchor_selection_v50mane.npz \
    split.splice_mask_key=train_mask \
    train.out_dir=outputs/spf/fold${fold}
done
.venv/bin/python -m scripts.eval_drcons_cv5_oof \
  --cache outputs/clinvar/cache.h5 \
  --cvdir outputs/spf \
  --split-dir outputs/clinvar/folds --split-tag cc_seed42 \
  --coord outputs/clinvar/coord.npz \
  --variants-h5 outputs/clinvar/release.h5 \
  --scalars outputs/clinvar/scalars.npz \
  --batch-size 64 --out outputs/spf/oof.json
```

Each trained head is read from `outputs/spf/fold{k}/best.pt`. The evaluator uses
this checkout's implementation, explicit paths and the dynamic cohort size. It
uses each row's held-out model once, preserves full precision and partial
batches, and writes `oof.json` plus row-aligned `oof_preds.npz`.
`strata.coord_restricted_overall` is the current MANE universe; `overall` also
includes cohort rows outside that universe. These metrics should not be confused.
For the conservation-free control, fit with `feature.enabled=false` and
`feature.scalar_npz=null`, use a separate head directory, and omit `--scalars`
when evaluating. Its hardware/environment must match the intended comparison.
