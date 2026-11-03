"""Load and normalize variant records from the SpliceVarDB Excel file.

The loader is the first stage of the SpliceVarDB H5 build pipeline. It
parses the supplementary Excel file, renames the external column labels
(``CHROM``/``POS``/``REF``/``ALT``/``Gene_Symbol``/``Gene_Location``/
``Gene_Region``/``Outcome``) to internal field names, normalizes
chromosome names to UCSC style, normalizes SAV/nonSAV labels to binary
integers, filters out records with null or malformed mandatory fields,
and resolves duplicate ``(chrom, pos, ref, alt)`` records (collapsing
label-matching duplicates and dropping label-conflicting ones). Strand
is injected from an external ``gene_symbol → strand`` lookup because the
Excel itself does not carry strand information.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from process_input.gene_strand_lookup import resolve_strand
from process_input.splicevardb_config import PipelineConfig
from process_input.splicevardb_exceptions import InvalidVariantRecordError

logger = logging.getLogger(__name__)

EXCEL_SKIPROWS: int = 3

# Mapping from Excel column headers to the internal loader field names.
COLUMN_ALIAS: dict[str, str] = {
    "CHROM": "chrom",
    "POS": "pos",
    "REF": "ref",
    "ALT": "alt",
    "Gene_Symbol": "gene_symbol",
    "Gene_Location": "gene_location",
    "Gene_Region": "gene_region",
    "Outcome": "label",
}

REQUIRED_EXCEL_COLUMNS: tuple[str, ...] = tuple(COLUMN_ALIAS.keys())
INTERNAL_COLUMNS: tuple[str, ...] = tuple(COLUMN_ALIAS.values())

UCSC_CANONICAL_PATTERN = re.compile(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M)$")
ACGT_ONLY_PATTERN = re.compile(r"^[ACGT]+$")
VALID_STRANDS: tuple[str, ...] = ("+", "-")
MAX_CONFLICT_EXAMPLES: int = 10


@dataclass(frozen=True, slots=True)
class SpliceVarDBVariantRecord:
    """Raw variant record parsed from the SpliceVarDB supplementary Excel.

    Attributes:
        chrom (str): UCSC-style chromosome name (e.g. ``"chr1"``).
        pos_hg19 (int): 1-based GRCh37 genomic coordinate.
        ref (str): Reference allele (uppercase A/C/G/T).
        alt (str): Alternate allele (uppercase A/C/G/T).
        gene_symbol (str): Gene symbol as reported by SpliceVarDB (may be
            a comma-separated composite such as ``"ATM,C11orf65"``).
        gene_location (str): Location within the gene (e.g. ``"intronic"``).
        gene_region (str): Region classification (e.g. ``"acceptor_region"``).
        strand (str): Transcription strand, either ``"+"`` or ``"-"``,
            injected from the gene-strand lookup.
        label (int): Binary label; ``1`` for SAV, ``0`` for nonSAV.
    """

    chrom: str
    pos_hg19: int
    ref: str
    alt: str
    gene_symbol: str
    gene_location: str
    gene_region: str
    strand: Literal["+", "-"]
    label: int


def _normalize_chromosome(raw: Any) -> str | None:
    """Normalize an input chromosome value to UCSC style.

    Args:
        raw (Any): Raw chromosome value (string or integer) from the Excel.

    Returns:
        str | None: UCSC-style chromosome name (``"chr1"`` ... ``"chrM"``)
        or ``None`` if the value is null, empty, or non-canonical.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int,)):
        text = str(raw)
    else:
        text = str(raw).strip()
    if not text:
        return None
    if not text.startswith("chr"):
        if text.upper() == "MT":
            text = "chrM"
        else:
            text = f"chr{text}"
    if UCSC_CANONICAL_PATTERN.match(text):
        return text
    return None


def _normalize_label(raw: Any) -> int | None:
    """Normalize a SAV/nonSAV label to binary integer form.

    Args:
        raw (Any): Raw ``Outcome`` value (string ``"SAV"``/``"nonSAV"`` or
            integer ``1``/``0``) from the Excel.

    Returns:
        int | None: ``1`` for SAV, ``0`` for nonSAV, or ``None`` when the
        value is null or unrecognized.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int,)):
        return 1 if raw == 1 else 0 if raw == 0 else None
    text = str(raw).strip()
    if text.upper() == "SAV":
        return 1
    if text.lower() in ("nonsav", "non-sav", "non_sav"):
        return 0
    if text in ("1", "0"):
        return int(text)
    return None


def _is_blank(value: Any) -> bool:
    """Return ``True`` when ``value`` is null, empty, or whitespace-only.

    Args:
        value (Any): Any value parsed from the Excel sheet.

    Returns:
        bool: Whether the value should be treated as missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_pos(value: Any) -> int | None:
    """Parse the ``POS`` column into a positive integer.

    Args:
        value (Any): Raw ``POS`` value from the Excel.

    Returns:
        int | None: Positive 1-based coordinate, or ``None`` when invalid.
    """
    if _is_blank(value):
        return None
    try:
        pos = int(float(value))
    except (TypeError, ValueError):
        return None
    return pos if pos > 0 else None


def _resolve_duplicates(
    records: list[SpliceVarDBVariantRecord],
) -> list[SpliceVarDBVariantRecord]:
    """Collapse label-match duplicates and drop label-conflict duplicates.

    Args:
        records (list[SpliceVarDBVariantRecord]): Candidate records after
            null filtering and normalization.

    Returns:
        list[SpliceVarDBVariantRecord]: Records with duplicates resolved.
    """
    grouped: dict[tuple[str, int, str, str], list[SpliceVarDBVariantRecord]] = (
        defaultdict(list)
    )
    for record in records:
        grouped[(record.chrom, record.pos_hg19, record.ref, record.alt)].append(record)

    resolved: list[SpliceVarDBVariantRecord] = []
    collapse_count = 0
    conflict_count = 0
    conflict_examples: list[str] = []

    for key, group in grouped.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        labels = {record.label for record in group}
        if len(labels) == 1:
            resolved.append(group[0])
            collapse_count += len(group) - 1
            continue
        conflict_count += len(group)
        if len(conflict_examples) < MAX_CONFLICT_EXAMPLES:
            chrom, pos, ref, alt = key
            conflict_examples.append(f"{chrom}:{pos} {ref}>{alt}")

    if collapse_count:
        logger.info("Deduplicated %d identical records", collapse_count)
    if conflict_count:
        logger.warning(
            "Dropped %d records with conflicting labels; examples: %s",
            conflict_count,
            conflict_examples,
        )

    return resolved


def _ensure_columns(df: pd.DataFrame) -> None:
    """Raise when the DataFrame is missing any required Excel column.

    Args:
        df (pd.DataFrame): DataFrame loaded from the Excel file.

    Raises:
        InvalidVariantRecordError: If any required column is absent.
    """
    missing = [column for column in REQUIRED_EXCEL_COLUMNS if column not in df.columns]
    if missing:
        raise InvalidVariantRecordError(
            f"Excel is missing required columns: {missing}",
            missing_columns=missing,
        )


def load_splicevardb_excel(
    cfg: PipelineConfig,
    gene_strand: dict[str, str],
) -> list[SpliceVarDBVariantRecord]:
    """Load and normalize variant records from the SpliceVarDB Excel file.

    The first sheet is read via ``pandas.read_excel`` with
    ``skiprows=3`` and ``engine="openpyxl"``. Excel-side column headers
    (``CHROM``/``POS``/...) are aliased to internal field names defined
    in :data:`COLUMN_ALIAS`. Records are filtered for null mandatory
    fields (``chrom``, ``pos``, ``ref``, ``alt``), non-canonical
    chromosome values, invalid alleles, unrecognized labels, and
    unresolvable strand lookups. Duplicate records sharing the same
    ``(chrom, pos, ref, alt)`` key are collapsed when labels agree and
    dropped entirely when they conflict.

    Args:
        cfg (PipelineConfig): Pipeline configuration; only
            ``cfg.paths.excel`` is consulted.
        gene_strand (dict[str, str]): Mapping of gene symbol to ``"+"``
            or ``"-"`` produced by
            :func:`process_input.gene_strand_lookup.load_gene_strand_table`.

    Returns:
        list[SpliceVarDBVariantRecord]: Normalized variant records.

    Raises:
        InvalidVariantRecordError: If the Excel is missing required
            columns.
    """
    excel_path = cfg.paths.excel
    df = pd.read_excel(
        excel_path,
        sheet_name=0,
        engine="openpyxl",
        skiprows=EXCEL_SKIPROWS,
    )
    _ensure_columns(df)
    df = df.rename(columns=COLUMN_ALIAS)

    total_rows = len(df)
    null_dropped = 0
    chrom_dropped = 0
    allele_dropped = 0
    strand_dropped = 0
    label_dropped = 0

    records: list[SpliceVarDBVariantRecord] = []
    for row in df[list(INTERNAL_COLUMNS)].itertuples(index=False):
        row_dict: dict[str, Any] = row._asdict()

        if any(_is_blank(row_dict[col]) for col in ("chrom", "pos", "ref", "alt")):
            null_dropped += 1
            continue

        chrom = _normalize_chromosome(row_dict["chrom"])
        if chrom is None:
            chrom_dropped += 1
            continue

        pos = _parse_pos(row_dict["pos"])
        if pos is None:
            null_dropped += 1
            continue

        ref = str(row_dict["ref"]).strip().upper()
        alt = str(row_dict["alt"]).strip().upper()
        if not ACGT_ONLY_PATTERN.match(ref) or not ACGT_ONLY_PATTERN.match(alt):
            allele_dropped += 1
            continue

        label = _normalize_label(row_dict["label"])
        if label is None:
            label_dropped += 1
            continue

        gene_symbol_raw = row_dict["gene_symbol"]
        gene_location_raw = row_dict["gene_location"]
        gene_region_raw = row_dict["gene_region"]
        if any(_is_blank(v) for v in (gene_symbol_raw, gene_location_raw)):
            null_dropped += 1
            continue

        gene_symbol = str(gene_symbol_raw).strip()
        gene_location = str(gene_location_raw).strip()
        gene_region = "" if _is_blank(gene_region_raw) else str(gene_region_raw).strip()

        strand = resolve_strand(gene_symbol, gene_strand)
        if strand not in VALID_STRANDS:
            strand_dropped += 1
            continue

        records.append(
            SpliceVarDBVariantRecord(
                chrom=chrom,
                pos_hg19=pos,
                ref=ref,
                alt=alt,
                gene_symbol=gene_symbol,
                gene_location=gene_location,
                gene_region=gene_region,
                strand=strand,  # type: ignore[arg-type]
                label=label,
            )
        )

    logger.info("Loaded %d variant records from %s", total_rows, excel_path)
    if null_dropped:
        logger.warning("Dropped %d records with null fields", null_dropped)
    if chrom_dropped:
        logger.warning(
            "Dropped %d records with non-canonical chromosomes", chrom_dropped
        )
    if allele_dropped:
        logger.warning(
            "Dropped %d records with non-ACGT ref/alt alleles", allele_dropped
        )
    if strand_dropped:
        logger.warning(
            "Dropped %d records whose gene_symbol could not be resolved to a strand",
            strand_dropped,
        )
    if label_dropped:
        logger.warning(
            "Dropped %d records with unrecognized label values", label_dropped
        )

    resolved = _resolve_duplicates(records)
    logger.info(
        "Loader produced %d records after filters and deduplication", len(resolved)
    )
    return resolved
