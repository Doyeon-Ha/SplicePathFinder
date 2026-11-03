"""Build a ``gene_symbol → strand`` lookup from a GENCODE GTF annotation.

The SpliceVarDB supplementary Excel does not carry a ``strand`` column,
so the loader derives strand per-variant from an external source. We
stream the GENCODE v49 annotation GTF (``gencode.v49.annotation.gtf``),
extract only the ``gene`` feature rows, and map the ``gene_name``
attribute to the strand column. The parser supports both plain ``.gtf``
and gzipped ``.gtf.gz`` inputs.

Genes whose annotation reports both strands (rare, but possible for
overlapping loci) are resolved by majority vote; ties fall back to the
first strand seen. Comma-separated composite ``gene_symbol`` values from
SpliceVarDB (e.g. ``"ATM,C11orf65"``) are resolved by trying each
component in order.
"""

from __future__ import annotations

import gzip
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO, Iterator

logger = logging.getLogger(__name__)

GENE_FEATURE: str = "gene"
_GENE_NAME_PATTERN = re.compile(r'gene_name "([^"]+)"')


def _open_text(path: Path) -> IO[str]:
    """Open a GTF file transparently for plain or gzip content.

    Args:
        path (Path): Filesystem path to a ``.gtf`` or ``.gtf.gz`` file.

    Returns:
        IO[str]: Text-mode file handle.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _iter_gene_rows(handle: IO[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(gene_name, strand)`` pairs from GTF ``gene`` rows.

    Args:
        handle (IO[str]): Open text-mode GTF handle.

    Yields:
        tuple[str, str]: ``(gene_name, strand)`` for each ``gene`` feature
        row with a parseable ``gene_name`` attribute and ``+``/``-`` strand.
    """
    for line in handle:
        if not line or line[0] == "#":
            continue
        # Fast pre-filter: avoid splitting every line.
        if "\tgene\t" not in line:
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9 or fields[2] != GENE_FEATURE:
            continue
        strand = fields[6]
        if strand not in ("+", "-"):
            continue
        match = _GENE_NAME_PATTERN.search(fields[8])
        if not match:
            continue
        yield match.group(1), strand


def load_gene_strand_from_gtf(path: str | Path) -> dict[str, str]:
    """Load a ``gene_name → strand`` dictionary from a GENCODE GTF file.

    Only rows whose 3rd column is ``gene`` are considered. Memory stays
    bounded: the file is streamed line by line and only per-gene strand
    counters are retained.

    Args:
        path (str | Path): Filesystem path to the GENCODE annotation
            GTF (``.gtf`` or ``.gtf.gz``).

    Returns:
        dict[str, str]: Mapping of gene symbol to the dominant strand.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When no gene features are recognized.
    """
    gtf_path = Path(path)
    if not gtf_path.is_file():
        raise FileNotFoundError(f"Gene annotation GTF not found: {path}")

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    with _open_text(gtf_path) as handle:
        for gene_name, strand in _iter_gene_rows(handle):
            counters[gene_name][strand] += 1

    if not counters:
        raise ValueError(
            f"No usable 'gene' features found in {path}; "
            "expected GENCODE GTF with gene_name attributes."
        )

    lookup: dict[str, str] = {}
    conflict_count = 0
    for gene_name, counts in counters.items():
        dominant, _ = counts.most_common(1)[0]
        if len(counts) > 1:
            conflict_count += 1
        lookup[gene_name] = dominant

    logger.info(
        "Loaded gene-strand map from GTF: %d genes, %d multi-strand conflicts",
        len(lookup),
        conflict_count,
    )
    return lookup


def resolve_strand(gene_symbol: str, lookup: dict[str, str]) -> str | None:
    """Resolve the strand for a SpliceVarDB ``gene_symbol`` value.

    SpliceVarDB sometimes uses comma-separated composite symbols (e.g.
    ``"ATM,C11orf65"``) when a variant lies in overlapping annotations.
    This helper splits on commas and returns the first resolvable strand.

    Args:
        gene_symbol (str): Raw gene symbol string from the Excel.
        lookup (dict[str, str]): Mapping produced by
            :func:`load_gene_strand_from_gtf`.

    Returns:
        str | None: ``"+"`` or ``"-"`` when resolvable; ``None`` when no
        component of ``gene_symbol`` is present in ``lookup``.
    """
    if not gene_symbol:
        return None
    for component in gene_symbol.split(","):
        key = component.strip()
        if not key:
            continue
        strand = lookup.get(key)
        if strand in ("+", "-"):
            return strand
    return None
