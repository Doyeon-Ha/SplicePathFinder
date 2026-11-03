"""Convert SpliceVarDB variant coordinates from GRCh37 (hg19) to GRCh38.

The runner consumes :class:`SpliceVarDBVariantRecord` objects produced by
``splicevardb_loader`` and emits :class:`LiftoverResult` objects whose
positions reference GRCh38 and whose reference alleles have been
validated against the GRCh38 FASTA. Unmappable records and records whose
ref allele disagrees with GRCh38 are dropped and counted in
:class:`LiftoverStats`. When the overall success rate falls below the
configured threshold the runner raises :class:`LiftoverThresholdError`
unless ``force=True`` is passed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from process_input.splicevardb_config import PipelineConfig
from process_input.splicevardb_exceptions import LiftoverThresholdError
from process_input.splicevardb_loader import SpliceVarDBVariantRecord

if TYPE_CHECKING:
    from pyfaidx import Fasta

logger = logging.getLogger(__name__)

MAX_FAILURE_EXAMPLES: int = 10


class _ChainQuery(Protocol):
    """Structural protocol for objects exposing ``query(chrom, pos)``."""

    def query(self, chrom: str, pos: int) -> list[tuple[str, int, str]]:
        """Return hits for ``chrom:pos`` (0-based) as ``(chrom, pos, strand)``."""
        ...


@dataclass(frozen=True, slots=True)
class LiftoverResult:
    """Variant record whose coordinates are expressed in GRCh38 space.

    Attributes:
        chrom (str): UCSC-style chromosome on GRCh38.
        pos_hg38 (int): 1-based GRCh38 coordinate.
        ref (str): Reference allele (unchanged from input).
        alt (str): Alternate allele (unchanged from input).
        gene_symbol (str): Gene symbol.
        gene_location (str): Location within the gene.
        gene_region (str): Region classification.
        strand (str): Transcription strand, either ``"+"`` or ``"-"``.
        label (int): Binary SAV/nonSAV label.
    """

    chrom: str
    pos_hg38: int
    ref: str
    alt: str
    gene_symbol: str
    gene_location: str
    gene_region: str
    strand: Literal["+", "-"]
    label: int


@dataclass(slots=True)
class LiftoverStats:
    """Aggregate statistics describing a liftover pass.

    Attributes:
        total_input (int): Number of input records.
        lifted (int): Number of records lifted and ref-validated.
        lift_failed (int): Records with no chain hit.
        ref_mismatch (int): Records whose ref disagreed with GRCh38.
        success_rate (float): ``lifted / total_input`` in ``[0, 1]``.
        failure_examples (list[str]): Up to ten human-readable examples.
    """

    total_input: int = 0
    lifted: int = 0
    lift_failed: int = 0
    ref_mismatch: int = 0
    success_rate: float = 0.0
    failure_examples: list[str] = field(default_factory=list)


def _check_ref_allele(fasta: "Fasta", chrom: str, pos_hg38: int, ref: str) -> bool:
    """Return whether ``ref`` matches the GRCh38 sequence at ``chrom:pos``.

    Args:
        fasta (Fasta): pyfaidx Fasta handle for GRCh38.
        chrom (str): UCSC-style chromosome.
        pos_hg38 (int): 1-based anchor position.
        ref (str): Expected reference allele.

    Returns:
        bool: ``True`` when the observed base(s) equal ``ref``
        (case-insensitive).
    """
    start = pos_hg38 - 1
    end = start + len(ref)
    observed = str(fasta[chrom][start:end])
    return observed.upper() == ref.upper()


def _record_failure_example(
    stats: LiftoverStats, record: SpliceVarDBVariantRecord, reason: str
) -> None:
    """Append a formatted failure example to ``stats`` up to the cap.

    Args:
        stats (LiftoverStats): Stats accumulator to update.
        record (SpliceVarDBVariantRecord): Record that failed.
        reason (str): Short failure reason (e.g. ``"chain miss"``).
    """
    if len(stats.failure_examples) >= MAX_FAILURE_EXAMPLES:
        return
    example = f"{record.chrom}:{record.pos_hg19} {record.ref}>{record.alt} ({reason})"
    stats.failure_examples.append(example)


def run_liftover(
    records: list[SpliceVarDBVariantRecord],
    cfg: PipelineConfig,
    *,
    chain: _ChainQuery,
    fasta: "Fasta",
    force: bool,
) -> tuple[list[LiftoverResult], LiftoverStats]:
    """Lift variant coordinates to GRCh38 and validate against the FASTA.

    Args:
        records (list[SpliceVarDBVariantRecord]): Input records keyed on
            hg19 coordinates.
        cfg (PipelineConfig): Pipeline configuration; consulted for
            ``min_liftover_success_rate``.
        chain (_ChainQuery): Object exposing ``query(chrom, pos)`` in the
            style of :class:`liftover.ChainFile`.
        fasta (Fasta): pyfaidx Fasta handle for GRCh38.
        force (bool): When ``True``, skip the success-rate gate and emit
            a warning log instead of raising.

    Returns:
        tuple[list[LiftoverResult], LiftoverStats]: Successfully lifted
        records and the aggregate statistics.

    Raises:
        LiftoverThresholdError: When ``force`` is ``False`` and the
            observed success rate falls below
            ``cfg.min_liftover_success_rate``.
    """
    stats = LiftoverStats(total_input=len(records))
    results: list[LiftoverResult] = []

    for record in records:
        hits = chain.query(record.chrom, record.pos_hg19 - 1)
        if not hits:
            stats.lift_failed += 1
            _record_failure_example(stats, record, "chain miss")
            continue
        if len(hits) > 1:
            logger.warning(
                "Multi-hit liftover for %s:%d (%d hits); using first hit",
                record.chrom,
                record.pos_hg19,
                len(hits),
            )
        new_chrom, new_pos_0based, _ = hits[0]
        pos_hg38 = new_pos_0based + 1
        if not _check_ref_allele(fasta, new_chrom, pos_hg38, record.ref):
            stats.ref_mismatch += 1
            _record_failure_example(stats, record, "ref mismatch")
            continue

        results.append(
            LiftoverResult(
                chrom=new_chrom,
                pos_hg38=pos_hg38,
                ref=record.ref,
                alt=record.alt,
                gene_symbol=record.gene_symbol,
                gene_location=record.gene_location,
                gene_region=record.gene_region,
                strand=record.strand,
                label=record.label,
            )
        )
        stats.lifted += 1

    stats.success_rate = stats.lifted / stats.total_input if stats.total_input else 0.0

    logger.info(
        "Liftover completed: lifted=%d, failed=%d, ref_mismatch=%d, rate=%.4f",
        stats.lifted,
        stats.lift_failed,
        stats.ref_mismatch,
        stats.success_rate,
    )

    if stats.success_rate < cfg.min_liftover_success_rate:
        if not force:
            logger.error(
                "Liftover success rate %.4f below threshold %.4f; "
                "aborting (use --force to override)",
                stats.success_rate,
                cfg.min_liftover_success_rate,
            )
            raise LiftoverThresholdError(
                actual_rate=stats.success_rate,
                threshold=cfg.min_liftover_success_rate,
                examples=stats.failure_examples,
            )
        logger.warning(
            "--force specified; proceeding despite low liftover rate %.4f "
            "(threshold %.4f, forced=true)",
            stats.success_rate,
            cfg.min_liftover_success_rate,
        )

    return results, stats


def open_chain_file(path: str) -> Any:
    """Load a UCSC liftover chain file.

    Thin wrapper around :class:`liftover.ChainFile` so that callers in
    this repository import through a single entry point.

    Args:
        path (str): Absolute path to the ``hg19ToHg38.over.chain.gz`` file.

    Returns:
        liftover.ChainFile: Opened chain file ready for
        :meth:`~liftover.ChainFile.query`.
    """
    from liftover import ChainFile

    return ChainFile(path, "hg19", "hg38")
