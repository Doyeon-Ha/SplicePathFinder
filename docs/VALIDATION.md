# Source distribution validation

Validation date: 2026-11-03. Research source commit:
`cc4fbb0b71008a224b81f2b76a614ebeb9247c4a`.
The GENCODE v50 data-builder revision is recorded separately in
`release_manifest.json`.

| Check | Result |
| --- | --- |
| Included synthetic tests | 214 passed, 8 skipped, 4 warnings |
| Command-line entry points | Nine help invocations passed: transcript selection, intermediate/final dataset builders, probability cache, delta max, external SPF scoring, all-chromosome merge and both training stages |
| Locked dependencies | `uv lock --check --offline` passed |
| Python syntax | All distributed Python files parsed |
| Restricted storage paths and common credential patterns | No matches |
| Gitleaks source scan, default rules | No findings |

The eight skipped tests need a CUDA-capable graphics processing unit (GPU).
Tests ran using the existing Python 3.12 research environment with imports rooted
in the isolated release tree. Source scans include the final SSNet path-default
adaptation. Documentation and provenance metadata accompany the checked source.

Fresh environment installation, GPU model inference, full training and manuscript
benchmark reproduction were not performed for this source-distribution check.
Passing synthetic tests does not establish numerical reproduction of the paper.
