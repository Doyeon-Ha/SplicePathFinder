# Source distribution validation

Validation date: 2026-11-04. Source lineage and file checksums are recorded in
`release_manifest.json`; the result scope is defined in [PAPER_RESULTS.md](PAPER_RESULTS.md).
Tests ran from isolated release checkouts in the existing Python 3.12 research
environment, using an NVIDIA RTX A6000 graphics processing unit (GPU).
Full-input preparation checks used separate central processing unit (CPU) jobs.
MANE means Matched Annotation from NCBI and EMBL-EBI.

| Check | Result |
| --- | --- |
| Public-workflow tests on CPU | 7 passed, 1 GPU-dependent case skipped |
| Full suite with GPU available | 233 passed, no skipped tests |
| Command-line help | 30 entry points passed, including both training stages, input preparation, evaluation, controls, statistics, rendering and timing |
| Python source | All 120 Python files parsed; undefined-name checks passed |
| Dependency lock | `uv lock --check --offline` passed with the selected Python 3.12 interpreter |
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

The ClinVar check re-aggregated frozen predictions. It did not regenerate those
predictions or refit their models. SpliceVarDB (Splice Variant Database) data and
variant-linked outputs were excluded from publication. Scans inspect repository
content and credential patterns; public Git authorship metadata is retained.

Fresh environment installation, full backbone/head training and the complete
manuscript benchmark were not rerun for this source release. In particular, case
and external-tool workflows still require their matching frozen inputs and
publisher environments. Cross-device numerical equality is not established by
the single-device parity test. Torch and Mamba dependency deprecation warnings
were non-fatal in the passing test run.
