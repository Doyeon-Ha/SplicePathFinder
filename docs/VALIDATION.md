# Source distribution validation

Validation date: 2026-11-04. Source lineage and file checksums are recorded in
`release_manifest.json`; the result scope is defined in [PAPER_RESULTS.md](PAPER_RESULTS.md).
Tests ran from isolated release checkouts in the existing Python 3.12 research
environment, using an NVIDIA RTX A6000 graphics processing unit (GPU).

| Check | Result |
| --- | --- |
| Central processing unit tests | 217 passed, 8 GPU-dependent tests skipped |
| Full suite with GPU available | 225 passed, no skipped tests |
| Command-line help | 22 entry points passed, including both training stages, data/cache builders, controls, statistics, rendering and timing |
| Python source | All 114 Python files parsed; undefined-name checks passed |
| Dependency lock | `uv lock --check --offline` passed with the selected Python 3.12 interpreter |
| Trained v50 epoch-10 backbone | All 471 state entries matched the reference distribution; 16 synthetic 15,000-base sequences produced exactly identical full probability arrays, maximum absolute difference 0 |
| Frozen ClinVar table aggregation | All values and pending-column statuses matched the manuscript JSON across 12 strata and 33 available score columns |
| Credential scan | Gitleaks: no findings across the release's reachable history |
| Targeted confidentiality review | No operational private server address, Slack webhook, MLflow endpoint or private key found; dependency-version strings were reviewed as non-operational matches |

The GPU tests include the learned head and gradient flow through every H-Net
router. The synthetic forward comparison used the same device, trained state,
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
