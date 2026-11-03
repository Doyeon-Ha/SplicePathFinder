"""Public API for the SpliceVarDB H5 build pipeline modules.

Downstream callers should import names from :mod:`process_input` rather
than reaching into submodules directly. Other pipeline artifacts (e.g.
SpliceAI H5 builders) live alongside in this package and expose their
own public surface via their own modules.
"""

from process_input.gene_strand_lookup import (
    load_gene_strand_from_gtf,
    resolve_strand,
)
from process_input.liftover_runner import (
    LiftoverResult,
    LiftoverStats,
    run_liftover,
)
from process_input.splicevardb_config import (
    LoggingConfig,
    PathsConfig,
    PipelineConfig,
)
from process_input.splicevardb_exceptions import (
    InvalidVariantRecordError,
    LiftoverThresholdError,
    RefAlleleMismatchError,
)
from process_input.splicevardb_h5_builder import (
    SpliceVarDBH5Payload,
    build_h5_payload,
    build_splicevardb_h5,
)
from process_input.splicevardb_loader import (
    SpliceVarDBVariantRecord,
    load_splicevardb_excel,
)
from process_input.variant_sequence_extractor import (
    VariantSequences,
    encode_variant_sequence,
    extract_variant_sequences,
)

__all__ = [
    "InvalidVariantRecordError",
    "LiftoverResult",
    "LiftoverStats",
    "LiftoverThresholdError",
    "LoggingConfig",
    "PathsConfig",
    "PipelineConfig",
    "RefAlleleMismatchError",
    "SpliceVarDBH5Payload",
    "SpliceVarDBVariantRecord",
    "VariantSequences",
    "build_h5_payload",
    "build_splicevardb_h5",
    "encode_variant_sequence",
    "extract_variant_sequences",
    "load_gene_strand_from_gtf",
    "load_splicevardb_excel",
    "resolve_strand",
    "run_liftover",
]
