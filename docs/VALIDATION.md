# Source distribution validation

Validation date: 2026-11-04. Source lineage and file checksums are recorded in
`release_manifest.json`; the result scope is defined in [PAPER_RESULTS.md](PAPER_RESULTS.md).
Reference checks ran from isolated release checkouts in the existing Python 3.12
research environment, using an NVIDIA RTX A6000 graphics processing unit (GPU).
Installation, trained-model inference and the full test suite were also checked
in an empty Python virtual environment.
Full-input preparation checks used separate central processing unit (CPU) jobs.
MANE means Matched Annotation from NCBI and EMBL-EBI.

| Check | Result |
| --- | --- |
| Public-workflow tests on CPU | 7 passed, 1 GPU-dependent case skipped |
| Full suite with GPU available in the fresh environment | 233 passed, no skipped tests |
| Command-line help | 30 entry points passed, including both training stages, input preparation, evaluation, controls, statistics, rendering and timing |
| Python source | All 120 Python files parsed; undefined-name checks passed |
| Dependency lock | `uv lock --check --offline` passed with the selected Python 3.12 interpreter |
| Fresh Python installation | Zero packages before installation; 103 installed packages passed `uv pip check`; five training, preparation and scoring commands passed their help checks |
| Fresh trained-model inference | All 16 public ClinVar variants produced finite reference/alternate tracks of shape `(16, 8750, 3)`, six conservation features and finite classification scores; input identities were preserved |
| GENCODE v50 transcript preparation | The 18,539-transcript table regenerated from the original annotation and supplied paralog flags was byte-identical to the paper's reference |
| Dated ClinVar Variant Call Format (VCF) conversion | All 384,073 raw identities, labels and gene symbols matched the reference; selecting the 378,543-row paper cohort reproduced its identities, labels and gene symbols |
| Frozen reference anchors | All 378,002 identities, source genes, strands, offsets and chunk/local indices matched |
| Submitted/preferred genes | All 378,543 regenerated rows matched the supplied reference gene table |
| Gene-component folds | All train, validation and test indices matched the five frozen reference folds |
| Reported-gene anchors and fitting masks | Every saved selection array matched the reference, including fold-spanning exclusions |
| MANE distance strata | Every reference field array matched, preserving the paper's 107,426-row evaluation universe |
| Long-allele identity comparison | A 10,000-row comparison containing a 10,000-base allele passed with 128 MiB of additional virtual address space allowed |
| Current-model held-out evaluation | Five temporary heads predicted all 330 synthetic rows exactly once, using full batches of 64 and final batches of 2 with explicitly supplied scalar and input paths |
| Trained v50 epoch-10 backbone | All 471 state entries matched the reference distribution; 16 synthetic 15,000-base sequences produced exactly identical full probability arrays, maximum absolute difference 0 |
| Frozen ClinVar table aggregation | All values and pending-column statuses matched the manuscript JSON across 12 strata and 33 available score columns |
| Credential scan | Gitleaks: no findings across the release's reachable history |
| Targeted confidentiality review | No operational private server address, Slack webhook, MLflow endpoint or private key found; dependency-version strings were reviewed as non-operational matches |

The GPU tests include the learned head, the portable five-fold evaluation command,
and gradient flow through every H-Net router. Workflow tests also check changed
anchor cache replacement, absent-anchor zero filling, all six scalar values and
rejection of row-order mismatches. Temporary synthetic head weights and test
inputs are not distributed. The synthetic backbone forward comparison used the same device, trained state,
input order, batch 16 and software environment in both checkouts. Peak allocated
and reserved device memory were 3.41 and 4.08 GiB, respectively, for that forward.
The comparison checks inference parity; it is not a training-memory measurement.

The fresh installation used Python 3.12.11, uv 0.11.28, the CUDA (Compute Unified
Device Architecture) 12.8.93 development toolkit and GNU Compiler Collection
(GCC) 13.4.0 with its G++ compiler. The job had
one RTX A6000 with 48 GB device memory, eight CPU cores and 64 GiB host memory
allocated. These are validation allocations, not minimum requirements. A
separate package cache held dependencies and CUDA extension wheels built from
the locked source packages for this validation. The input pipeline's Pydantic
dependency is locked at 2.11.7. User-site packages were disabled, and the checked
runtime modules loaded from the new virtual environment.

The 16-variant functional check used privately held v50 epoch-10 backbone
weights and one trained fold-3 SplicePathFinder head. The backbone forward batch
was 16; the head batch argument was 64, with a final partial batch of 16. Each variant used
four anchored chunks and an 8,750-position merged track. This check validates
execution and row identity; it does not estimate benchmark performance. The
weights and generated probability caches are not distributed.

The ClinVar check re-aggregated frozen predictions. It did not regenerate those
predictions or refit their models. SpliceVarDB (Splice Variant Database) data and
variant-linked outputs were excluded from publication. Scans inspect repository
content and credential patterns; public Git authorship metadata is retained.

Full backbone/head training and the complete manuscript benchmark were not rerun
for this source release. In particular, case
and external-tool workflows still require their matching frozen inputs and
publisher environments. Cross-device numerical equality is not established by
the single-device parity test. Torch and Mamba dependency deprecation warnings
were non-fatal in the passing test run.
