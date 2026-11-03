"""Custom exceptions for the SpliceVarDB H5 build pipeline.

These exceptions signal pipeline-specific failure modes that should be
distinguishable from generic built-in errors so that CLI exit codes and
log messages can be mapped deterministically. See
``.specify/specs/004-splicevardb-h5-build/contracts/cli.md`` for the
exit-code contract.
"""


class InvalidVariantRecordError(ValueError):
    """Raised when the input Excel is missing expected columns or sheet.

    Attributes:
        missing_columns (list[str]): Column names that were expected but
            absent in the source Excel. Empty when the cause is a missing
            sheet rather than columns.
    """

    def __init__(self, message: str, missing_columns: list[str] | None = None) -> None:
        """Initialize the exception.

        Args:
            message (str): Human readable error detail.
            missing_columns (list[str] | None): Optional list of missing
                column names; defaults to an empty list.
        """
        super().__init__(message)
        self.missing_columns: list[str] = (
            list(missing_columns) if missing_columns else []
        )


class LiftoverThresholdError(RuntimeError):
    """Raised when liftover success rate falls below the configured gate.

    The CLI maps this exception to exit code ``2`` so that downstream CI
    steps can distinguish a data-quality gate failure from generic errors.

    Attributes:
        actual_rate (float): Observed liftover success rate in ``[0, 1]``.
        threshold (float): Configured minimum success rate.
        examples (list[str]): Up to ten human-readable failure examples
            (e.g. ``"chr1:12345 A>G (chain miss)"``).
    """

    def __init__(
        self,
        actual_rate: float,
        threshold: float,
        examples: list[str] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            actual_rate (float): Observed liftover success rate.
            threshold (float): Configured threshold.
            examples (list[str] | None): Optional failure examples for
                logging; defaults to an empty list.
        """
        message = (
            f"Liftover success rate {actual_rate:.4f} below threshold "
            f"{threshold:.4f}; aborting (use --force to override)"
        )
        super().__init__(message)
        self.actual_rate: float = actual_rate
        self.threshold: float = threshold
        self.examples: list[str] = list(examples) if examples else []


class RefAlleleMismatchError(ValueError):
    """Raised when the GRCh38 reference base conflicts with the record's ref.

    This is intended for strict validation paths. The default liftover
    pipeline drops mismatched records and counts them rather than raising,
    so this exception is reserved for cases where callers explicitly opt
    into fail-fast behavior.

    Attributes:
        chrom (str): UCSC-style chromosome name.
        pos (int): 1-based GRCh38 coordinate.
        expected (str): Expected reference allele from the input record.
        observed (str): Reference base(s) observed at the coordinate.
    """

    def __init__(self, chrom: str, pos: int, expected: str, observed: str) -> None:
        """Initialize the exception.

        Args:
            chrom (str): UCSC-style chromosome name.
            pos (int): 1-based GRCh38 coordinate of the variant anchor.
            expected (str): Expected reference allele.
            observed (str): Observed base(s) at the coordinate.
        """
        message = (
            f"Ref allele mismatch at {chrom}:{pos}: "
            f"expected={expected!r}, observed={observed!r}"
        )
        super().__init__(message)
        self.chrom: str = chrom
        self.pos: int = pos
        self.expected: str = expected
        self.observed: str = observed
