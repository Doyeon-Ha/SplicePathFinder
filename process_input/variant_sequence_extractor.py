"""Extract variant-centric ref/alt sequences from the GRCh38 reference.

The extractor is the third stage of the SpliceVarDB H5 build pipeline.
Given a :class:`LiftoverResult` and a FASTA handle, it returns integer
encoded reference and alternate sequences, each of length
``target_length + 2 * flank_length``. The alternate sequence is built by
replacing the ``len(ref)`` anchor bases with the ``alt`` allele and
symmetrically padding or trimming the result so that its length matches
the reference window. When ``strand_aware`` is enabled, the sequences
for minus-strand variants (ref and alt alike) are reverse complemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from process_input.liftover_runner import LiftoverResult

if TYPE_CHECKING:
    from pyfaidx import Fasta

VARIANT_DNA_ENCODE_MAP: dict[str, int] = {
    "N": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}

_ENCODE_LOOKUP: np.ndarray = np.zeros(128, dtype=np.int8)
for _char, _val in VARIANT_DNA_ENCODE_MAP.items():
    _ENCODE_LOOKUP[ord(_char)] = _val
    _ENCODE_LOOKUP[ord(_char.lower())] = _val

_COMPLEMENT_TABLE: dict[int, str] = {
    ord("A"): "T",
    ord("C"): "G",
    ord("G"): "C",
    ord("T"): "A",
    ord("N"): "N",
    ord("a"): "T",
    ord("c"): "G",
    ord("g"): "C",
    ord("t"): "A",
    ord("n"): "N",
}


@dataclass(frozen=True, slots=True)
class VariantSequences:
    """Integer-encoded reference and alternate sequences for a variant.

    Attributes:
        x_ref (np.ndarray): Reference sequence, shape ``(total_length,)``
            with ``dtype=int8`` and values in ``{0, 1, 2, 3, 4}``.
        x_alt (np.ndarray): Alternate sequence, same shape and dtype as
            ``x_ref`` with the variant positions substituted.
    """

    x_ref: np.ndarray
    x_alt: np.ndarray


def encode_variant_sequence(sequence: str) -> np.ndarray:
    """Encode a DNA sequence to an int8 array using the variant mapping.

    ``N`` maps to ``0``, ``A`` to ``1``, ``C`` to ``2``, ``G`` to ``3``,
    and ``T`` to ``4``. Any other character (IUPAC ambiguous codes
    R/Y/S/W/K/M/B/D/H/V and unknowns) maps to ``0`` (``N``).

    Args:
        sequence (str): Arbitrary DNA sequence.

    Returns:
        np.ndarray: Encoded sequence with ``dtype=int8`` and shape
        ``(len(sequence),)``.
    """
    if not sequence:
        return np.zeros(0, dtype=np.int8)
    raw = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    # Characters outside ASCII (replaced by '?') fall into the zero bucket
    # thanks to the pre-populated lookup table.
    clipped = np.minimum(raw, 127)
    return _ENCODE_LOOKUP[clipped]


def _reverse_complement(sequence: str) -> str:
    """Return the reverse complement of ``sequence``.

    Bases outside ``A/C/G/T/N`` (e.g. IUPAC ambiguous codes) are mapped
    to ``N`` in the result so that downstream encoding remains in domain.

    Args:
        sequence (str): Input sequence.

    Returns:
        str: Reverse-complemented sequence of the same length.
    """
    complemented = "".join(_COMPLEMENT_TABLE.get(ord(base), "N") for base in sequence)
    return complemented[::-1]


def _extract_padded_window(
    fasta: "Fasta", chrom: str, start_0based: int, end_0based: int
) -> str:
    """Return a padded slice of ``chrom`` from ``fasta``.

    Out-of-range coordinates on the left or right of the contig are
    substituted with ``N`` so that the returned string always has length
    ``end_0based - start_0based``.

    Args:
        fasta (Fasta): pyfaidx FASTA handle.
        chrom (str): UCSC-style chromosome name.
        start_0based (int): 0-based window start (inclusive).
        end_0based (int): 0-based window end (exclusive).

    Returns:
        str: Padded sequence of length ``end_0based - start_0based``.
    """
    chrom_len = len(fasta[chrom])
    left_pad = max(0, -start_0based)
    right_pad = max(0, end_0based - chrom_len)
    slice_start = max(0, start_0based)
    slice_end = min(chrom_len, end_0based)
    body = str(fasta[chrom][slice_start:slice_end])
    return "N" * left_pad + body + "N" * right_pad


def _center_symmetric_fit(sequence: str, total_length: int, anchor_offset: int) -> str:
    """Pad or trim ``sequence`` symmetrically around ``anchor_offset``.

    When the sequence is shorter than ``total_length``, ``N`` bases are
    added on both sides. Any length deficit that is odd assigns the extra
    base to the right side so that anchor bases remain centered. When the
    sequence is longer than ``total_length``, the same rule applies to
    truncation.

    Args:
        sequence (str): Candidate sequence.
        total_length (int): Desired final length.
        anchor_offset (int): Unused but retained for future symmetry
            anchoring strategies that key off the anchor position.

    Returns:
        str: Sequence of length ``total_length``.
    """
    _ = anchor_offset  # reserved for future anchor-aware strategies
    delta = total_length - len(sequence)
    if delta == 0:
        return sequence
    if delta > 0:
        left = delta // 2
        right = delta - left
        return "N" * left + sequence + "N" * right
    # delta < 0 -> symmetric trim.
    trim = -delta
    left = trim // 2
    right = trim - left
    return sequence[left : len(sequence) - right]


def _compute_window_bounds(
    pos_hg38: int, target_length: int, flank_length: int
) -> tuple[int, int, int]:
    """Compute the 0-based window start/end and the anchor offset.

    The window is centered so that ``pos_hg38`` (1-based) is the
    right-most base of the left half, mirroring the spec's
    ``target_length // 2`` + flank computation.

    Args:
        pos_hg38 (int): 1-based anchor coordinate.
        target_length (int): Target region length in base pairs.
        flank_length (int): Flank length in base pairs.

    Returns:
        tuple[int, int, int]: ``(start_0based, end_0based, anchor_offset)``.
    """
    total_length = target_length + 2 * flank_length
    anchor_offset = flank_length + target_length // 2
    center_0based = pos_hg38 - 1
    start_0based = center_0based - anchor_offset
    end_0based = start_0based + total_length
    return start_0based, end_0based, anchor_offset


def extract_variant_sequences(
    fasta: "Fasta",
    record: LiftoverResult,
    target_length: int,
    flank_length: int,
    strand_aware: bool,
) -> VariantSequences:
    """Return integer-encoded ref/alt sequences centered on the variant.

    For SNV variants the alternate sequence differs from the reference at
    a single anchor position. For indels and MNVs the ``len(ref)`` anchor
    bases of the reference window are replaced by the ``alt`` allele and
    the resulting sequence is symmetrically N-padded or trimmed so that
    its length matches ``target_length + 2 * flank_length``.

    Args:
        fasta (Fasta): pyfaidx FASTA handle for GRCh38.
        record (LiftoverResult): Variant record with hg38 coordinates.
        target_length (int): Target region length in base pairs.
        flank_length (int): Flank length in base pairs on each side.
        strand_aware (bool): When ``True``, minus-strand records have
            both ref and alt sequences reverse complemented.

    Returns:
        VariantSequences: Encoded ref and alt arrays of shape
        ``(target_length + 2 * flank_length,)`` with ``dtype=int8``.
    """
    total_length = target_length + 2 * flank_length
    start_0based, end_0based, anchor_offset = _compute_window_bounds(
        record.pos_hg38, target_length, flank_length
    )

    ref_window = _extract_padded_window(fasta, record.chrom, start_0based, end_0based)
    # Construct alt by replacing ref-length bases at anchor with record.alt.
    alt_raw = (
        ref_window[:anchor_offset]
        + record.alt
        + ref_window[anchor_offset + len(record.ref) :]
    )
    alt_window = _center_symmetric_fit(alt_raw, total_length, anchor_offset)

    if strand_aware and record.strand == "-":
        ref_window = _reverse_complement(ref_window)
        alt_window = _reverse_complement(alt_window)

    x_ref = encode_variant_sequence(ref_window)
    x_alt = encode_variant_sequence(alt_window)

    if x_ref.shape[0] != total_length:
        # Defensive: pad/trim to enforce contract even on malformed input.
        x_ref = np.resize(x_ref, (total_length,)).astype(np.int8)
    if x_alt.shape[0] != total_length:
        x_alt = np.resize(x_alt, (total_length,)).astype(np.int8)

    return VariantSequences(x_ref=x_ref, x_alt=x_alt)
