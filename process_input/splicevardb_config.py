"""Pydantic configuration models for the SpliceVarDB H5 build pipeline.

Configuration is loaded from a YAML file via ``OmegaConf`` and then
validated through these models so that every downstream stage can rely
on well-formed values and attribute-style access. Keeping the config
definitions in their own module lets tests exercise validation without
importing the full builder / loader chain.
"""

from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem inputs and outputs referenced by the pipeline.

    Attributes:
        excel (str): Path to the SpliceVarDB supplementary Excel file.
        ref_genome (str): Path to the GRCh38 reference FASTA.
        chain_hg19_to_hg38 (str): Path to the UCSC liftover chain file.
        gene_annotation_gtf (str): Path to a GENCODE annotation GTF
            (e.g. ``gencode.v49.annotation.gtf`` or its ``.gz``
            counterpart) used to map each ``gene_symbol`` to its
            ``+``/``-`` strand.
        output (str): Destination path for the final H5 artifact.
    """

    excel: str
    ref_genome: str
    chain_hg19_to_hg38: str
    gene_annotation_gtf: str
    output: str


class LoggingConfig(BaseModel):
    """Standard ``logging`` configuration options.

    Attributes:
        level (str): Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        format (str): ``logging.Formatter`` format string.
        date_format (str): ``strftime`` date format string.
    """

    level: str = "INFO"
    format: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


class PipelineConfig(BaseModel):
    """Top-level configuration for the SpliceVarDB H5 build pipeline.

    Attributes:
        target_length (int): Size of the variant-centric target window in
            base pairs. Must be positive and even so that the window can
            be centered symmetrically on the variant anchor.
        flank_length (int): Number of flanking base pairs added on each
            side of the target region. Must be non-negative.
        strand_aware (bool): When true, sequences for minus-strand
            variants are reverse-complemented (including the alt allele).
        min_liftover_success_rate (float): Minimum acceptable fraction of
            records that must be successfully lifted from hg19 to hg38.
            Must lie in ``[0.0, 1.0]``.
        paths (PathsConfig): Filesystem paths used by the pipeline.
        logging (LoggingConfig): Logging configuration.
    """

    target_length: int = Field(gt=0)
    flank_length: int = Field(ge=0)
    strand_aware: bool = True
    min_liftover_success_rate: float = Field(ge=0.0, le=1.0, default=0.95)
    paths: PathsConfig
    logging: LoggingConfig = LoggingConfig()

    @field_validator("target_length")
    @classmethod
    def _target_length_must_be_even(cls, value: int) -> int:
        """Validate that ``target_length`` is even.

        Args:
            value (int): Candidate ``target_length`` value.

        Returns:
            int: The same value if validation passes.

        Raises:
            ValueError: If ``value`` is not divisible by 2.
        """
        if value % 2 != 0:
            raise ValueError(
                f"target_length must be even for centered window, got {value}"
            )
        return value

    @property
    def total_length(self) -> int:
        """Return the final sequence length ``target + 2 * flank``.

        Returns:
            int: ``target_length + 2 * flank_length``.
        """
        return self.target_length + 2 * self.flank_length
