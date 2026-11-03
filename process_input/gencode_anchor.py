"""Gencode-anchored sequence generation for SpliceVarDB H5 build.

Implements the ``gencode_anchored`` ``build_mode`` defined in
``.specify/specs/004-splicevardb-h5-build/spec.md`` (FR-012a..FR-012g):
locate the canonical transcript whose datapoint block target region
contains a variant, then produce ``X_ref``/``X_alt`` by copying the
gencode training H5's row and swapping a single base for the alt allele.

The module is intentionally focused on the deterministic
``(chrom, pos) → (chunk_idx, local_idx, offset)`` mapping. Caller code
(``splicevardb_h5_builder``) is responsible for higher-level
orchestration, logging, and SpliceVarDB record bookkeeping.

Encoding convention (matches ``process_input.h5_utils``):
``N=0, A=1, C=2, G=3, T=4``. Reverse complement on encoded values is
``(5 - x) % 5``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 100
SEQ_ENCODE = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
RC_ENCODE = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}


@dataclass(frozen=True, slots=True)
class GencodeGeneEntry:
    """Compact descriptor of one canonical transcript.

    Attributes:
        gene_name (str): Gene symbol from canonical splice table column 1.
        chrom (str): UCSC-style chromosome (e.g. ``"chr2"``).
        strand (str): ``"+"`` or ``"-"``.
        tx_start (int): 1-based inclusive transcript start (col 5).
        tx_end (int): 1-based inclusive transcript end (col 6).
        chunk_idx (int): Index of the gencode H5 ``X{chunk_idx}`` chunk
            that holds this transcript's blocks.
        local_offset (int): Row index within ``X{chunk_idx}`` where this
            transcript's first block begins (sum of preceding genes'
            block counts in the same chunk).
        n_blocks (int): Number of datapoint blocks for this transcript
            (``ceil(tx_len / sl)``).
    """

    gene_name: str
    chrom: str
    strand: str
    tx_start: int
    tx_end: int
    chunk_idx: int
    local_offset: int
    n_blocks: int

    @property
    def tx_len(self) -> int:
        """Return transcript length in bases.

        Returns:
            int: ``tx_end - tx_start + 1``.
        """
        return self.tx_end - self.tx_start + 1


@dataclass(frozen=True, slots=True)
class AnchorMatch:
    """Result of mapping a variant onto a gencode canonical transcript.

    Attributes:
        gene (GencodeGeneEntry): Selected canonical transcript.
        chunk_idx (int): Gencode H5 chunk index.
        local_idx (int): Row index inside the chunk for the matching
            datapoint block.
        variant_offset (int): 0-based column index of the variant within
            ``X_ref[local_idx, :]`` / ``X_alt[local_idx, :]``.
    """

    gene: GencodeGeneEntry
    chunk_idx: int
    local_idx: int
    variant_offset: int


def load_canonical_table(
    canonical_path: str,
    *,
    chrom_filter: set[str] | None,
    sl: int,
    chunk_size: int = CHUNK_SIZE,
) -> list[GencodeGeneEntry]:
    """Replicate the gencode H5 chunk layout from a canonical splice table.

    Re-implements the iteration order used by
    :func:`process_input.h5_dataset_builder.build_dataset` so we can map
    each gene to the H5 chunk and row offset where its datapoints live.

    Args:
        canonical_path (str): Path to ``canonical_dataset_*.txt``
            (13-column TSV).
        chrom_filter (set[str] | None): UCSC-style chromosomes to keep.
            ``None`` keeps every chromosome.
        sl (int): Sequence length per datapoint block (e.g. ``5000``).
        chunk_size (int): Number of genes per H5 chunk
            (matches ``cfg.chunk_size`` in ``h5_data.yaml``).

    Returns:
        list[GencodeGeneEntry]: One entry per canonical transcript that
        survives ``chrom_filter``, in the order it appears in the table.
    """
    raw: list[tuple[str, str, str, int, int, int]] = []
    with open(canonical_path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            chrom = cols[2]
            if chrom_filter is not None and chrom not in chrom_filter:
                continue
            tx_start = int(cols[4])
            tx_end = int(cols[5])
            tx_len = tx_end - tx_start + 1
            n_blocks = max(1, ceil(tx_len / sl))
            raw.append(
                (cols[0], chrom, cols[3], tx_start, tx_end, n_blocks)
            )

    n_genes = len(raw)
    # ``h5_dataset_builder.build_dataset`` uses ``n_genes // chunk_size``
    # for ``num_chunks`` (or 1 when zero) and the **last chunk holds the
    # remainder**. Replicate that here exactly.
    num_chunks = max(1, n_genes // chunk_size)
    entries: list[GencodeGeneEntry] = []
    accepted_idx = 0
    for chunk_idx in range(num_chunks):
        if (chunk_idx + 1) == num_chunks:
            cur_size = n_genes - chunk_idx * chunk_size
        else:
            cur_size = chunk_size
        for j in range(cur_size):
            gene_name, chrom, strand, tx_start, tx_end, n_blocks = (
                raw[accepted_idx]
            )
            entries.append(
                GencodeGeneEntry(
                    gene_name=gene_name,
                    chrom=chrom,
                    strand=strand,
                    tx_start=tx_start,
                    tx_end=tx_end,
                    chunk_idx=chunk_idx,
                    local_offset=0,
                    n_blocks=n_blocks,
                )
            )
            accepted_idx += 1

    # Second pass: compute local_offset per chunk.
    by_chunk: dict[int, int] = {}
    fixed: list[GencodeGeneEntry] = []
    for entry in entries:
        offset = by_chunk.get(entry.chunk_idx, 0)
        fixed.append(
            GencodeGeneEntry(
                gene_name=entry.gene_name,
                chrom=entry.chrom,
                strand=entry.strand,
                tx_start=entry.tx_start,
                tx_end=entry.tx_end,
                chunk_idx=entry.chunk_idx,
                local_offset=offset,
                n_blocks=entry.n_blocks,
            )
        )
        by_chunk[entry.chunk_idx] = offset + entry.n_blocks

    return fixed


def verify_chunk_sizes(
    entries: list[GencodeGeneEntry],
    h5_path: str,
) -> None:
    """Sanity-check that derived chunk sizes match the gencode H5.

    Args:
        entries (list[GencodeGeneEntry]): Output of
            :func:`load_canonical_table`.
        h5_path (str): Path to the gencode training H5.

    Raises:
        ValueError: When derived chunk sizes do not match the on-disk
            sizes — indicates the wrong canonical table or filter.
    """
    derived: dict[int, int] = {}
    for entry in entries:
        derived[entry.chunk_idx] = (
            derived.get(entry.chunk_idx, 0) + entry.n_blocks
        )

    with h5py.File(h5_path, "r") as f:
        x_keys = sorted(
            (k for k in f.keys() if k.startswith("X") and k[1:].isdigit()),
            key=lambda k: int(k[1:]),
        )
        h5_sizes = {
            int(k[1:]): int(f[k].shape[0]) for k in x_keys
        }

    if derived != h5_sizes:
        diffs = [
            (i, derived.get(i, 0), h5_sizes.get(i, 0))
            for i in sorted(set(derived) | set(h5_sizes))
            if derived.get(i, 0) != h5_sizes.get(i, 0)
        ]
        raise ValueError(
            "Derived chunk sizes do not match gencode H5;"
            f" first diffs (chunk, derived, h5)={diffs[:5]}"
        )


def build_genome_index(
    entries: list[GencodeGeneEntry],
) -> dict[str, list[GencodeGeneEntry]]:
    """Group canonical transcripts by chromosome for fast lookup.

    Args:
        entries (list[GencodeGeneEntry]): Output of
            :func:`load_canonical_table`.

    Returns:
        dict[str, list[GencodeGeneEntry]]: ``{chrom: [genes…]}`` with
            transcripts sorted by ``tx_start``.
    """
    by_chrom: dict[str, list[GencodeGeneEntry]] = {}
    for entry in entries:
        by_chrom.setdefault(entry.chrom, []).append(entry)
    for chrom, genes in by_chrom.items():
        genes.sort(key=lambda g: (g.tx_start, g.tx_end))
    return by_chrom


def find_anchor(
    *,
    chrom: str,
    pos: int,
    sl: int,
    cl_max: int,
    genome_index: dict[str, list[GencodeGeneEntry]],
    gene_symbol_hint: str | None = None,
) -> AnchorMatch | None:
    """Locate the canonical transcript whose target region covers ``pos``.

    Implements the FR-012a..FR-012b mapping: a variant maps to at most
    one canonical transcript. When two transcripts overlap on the same
    base, prefer the longer transcript and break ties by gene_name
    ascending.

    Args:
        chrom (str): UCSC-style chromosome.
        pos (int): 1-based GRCh38 coordinate.
        sl (int): Sequence length per block (target window size).
        cl_max (int): Maximum context length (CL).
        genome_index (dict[str, list[GencodeGeneEntry]]): Output of
            :func:`build_genome_index`.
        gene_symbol_hint (str | None): When provided, prefer transcripts
            whose ``gene_name`` matches the SpliceVarDB ``gene_symbol``
            entry (a comma-separated composite is tolerated). Used as a
            soft tie-breaker before the deterministic length/name rules.

    Returns:
        AnchorMatch | None: Anchor mapping, or ``None`` when no canonical
        transcript covers the variant.
    """
    genes = genome_index.get(chrom)
    if not genes:
        return None

    candidates: list[GencodeGeneEntry] = []
    for entry in genes:
        if pos < entry.tx_start or pos > entry.tx_end:
            continue
        # Ensure variant is inside a real (non-padding) target block.
        # Last block may extend past tx_end virtually but we only accept
        # variants within tx range, which is equivalent.
        candidates.append(entry)

    if not candidates:
        return None

    if gene_symbol_hint:
        hint_set = {
            piece.strip()
            for piece in gene_symbol_hint.split(",")
            if piece.strip()
        }
        hinted = [c for c in candidates if c.gene_name in hint_set]
        if hinted:
            candidates = hinted

    # Deterministic tiebreak: longest transcript first, then gene_name asc.
    candidates.sort(
        key=lambda g: (-g.tx_len, g.gene_name)
    )
    chosen = candidates[0]

    if chosen.strand == "+":
        p = pos - chosen.tx_start
    else:
        p = chosen.tx_end - pos

    block_idx_in_gene = p // sl
    if block_idx_in_gene >= chosen.n_blocks:
        return None
    offset = p % sl + cl_max // 2

    local_idx = chosen.local_offset + block_idx_in_gene
    return AnchorMatch(
        gene=chosen,
        chunk_idx=chosen.chunk_idx,
        local_idx=local_idx,
        variant_offset=offset,
    )


def fetch_x_ref_row(
    h5_handles: dict[int, h5py.File],
    h5_path: Path,
    chunk_idx: int,
    local_idx: int,
) -> np.ndarray:
    """Lazy-open ``X{chunk_idx}`` and return one row.

    Args:
        h5_handles (dict[int, h5py.File]): Cache of open H5 handles.
        h5_path (Path): Gencode training H5 path.
        chunk_idx (int): Chunk index to read.
        local_idx (int): Row to read within the chunk.

    Returns:
        np.ndarray: int8 row of length ``sl + cl_max``.
    """
    handle = h5_handles.get(chunk_idx)
    if handle is None:
        # Re-use a single handle for all chunks; lazy open keeps the
        # caller agnostic to file lifetime.
        handle = h5py.File(h5_path, "r")
        h5_handles[chunk_idx] = handle
    return np.asarray(handle[f"X{chunk_idx}"][local_idx], dtype=np.int8)


def encode_alt_allele(allele: str, strand: str) -> int:
    """Return the integer encoding of an alt allele on the given strand.

    Args:
        allele (str): Single uppercase base ``A``/``C``/``G``/``T``.
        strand (str): ``"+"`` or ``"-"``. For ``"-"`` the base is reverse
            complemented so it lines up with the gencode H5's RC layout.

    Returns:
        int: Integer in ``{1, 2, 3, 4}``.

    Raises:
        ValueError: For non-SNV alleles.
    """
    if len(allele) != 1 or allele not in SEQ_ENCODE:
        raise ValueError(f"Non-SNV alt allele: {allele!r}")
    code = SEQ_ENCODE[allele]
    if strand == "-":
        code = RC_ENCODE[code]
    return code


def encode_ref_allele(allele: str, strand: str) -> int:
    """Return the integer encoding of a ref allele on the given strand.

    Args:
        allele (str): Single uppercase base.
        strand (str): ``"+"`` or ``"-"``.

    Returns:
        int: Integer in ``{1, 2, 3, 4}``.

    Raises:
        ValueError: For non-SNV alleles.
    """
    return encode_alt_allele(allele, strand)
