"""Per-position conservation feature track from phyloP/phastCons bigWigs.

Reads UCSC 100-way conservation scores over a variant-anchored window and
returns a 2-channel track (ch0 phyloP normalised, ch1 phastCons) aligned to
the model's window frame — i.e. reversed on the minus strand so it lines up
position-for-position with the reverse-complemented sequence the model sees
(mirrors :func:`process_input.intron_feature.intron_feature_for_chunk`).

Both scores are per-base genomic values (strand-agnostic in value); only the
array orientation depends on strand. Missing bigWig values (assembly gaps) are
filled with 0.
"""

from __future__ import annotations

import numpy as np

# Number of conservation feature channels emitted per position.
CONSERVATION_CHANNELS = 2

# phyloP clip bound: values are clipped to [-C, C] then divided by C -> [-1, 1].
DEFAULT_PHYLOP_CLIP = 10.0


class ConservationReader:
    """Reads phyloP + phastCons windows from two bigWig files."""

    def __init__(
        self,
        phylop_bw: str,
        phastcons_bw: str,
        phylop_clip: float = DEFAULT_PHYLOP_CLIP,
    ) -> None:
        """Open the two bigWig files.

        Args:
            phylop_bw (str): Path to ``hg38.phyloP100way.bw``.
            phastcons_bw (str): Path to ``hg38.phastCons100way.bw``.
            phylop_clip (float): Clip bound C for phyloP normalisation
                (value -> ``clip(v, -C, C) / C``). phastCons is already [0, 1].
        """
        import pyBigWig

        self._phylop = pyBigWig.open(phylop_bw)
        self._phastcons = pyBigWig.open(phastcons_bw)
        self.phylop_clip = float(phylop_clip)
        # bigWig chrom names (e.g. "chr1"); used to resolve H5 chrom naming.
        self._phylop_chroms = set(self._phylop.chroms().keys())
        self._phastcons_chroms = set(self._phastcons.chroms().keys())

    def _resolve_chrom(self, chrom: str, available: set[str]) -> str | None:
        """Match an H5 chrom label to the bigWig's chrom naming.

        Args:
            chrom (str): H5 chrom label (``"chr1"`` or ``"1"``).
            available (set[str]): chrom names present in the bigWig.

        Returns:
            str | None: matching bigWig chrom name, or None if absent.
        """
        if chrom in available:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        return alt if alt in available else None

    def _values(
        self, handle: object, chrom: str, start_0b: int, length: int, avail: set[str]
    ) -> np.ndarray:
        """Read ``length`` per-base values from ``start_0b`` (0-based), NaN->0.

        Args:
            handle (object): open pyBigWig handle.
            chrom (str): H5 chrom label.
            start_0b (int): 0-based genomic start.
            length (int): window length in bp.
            avail (set[str]): bigWig chrom names for resolution.

        Returns:
            np.ndarray: float32 array of shape ``(length,)`` (0 where missing
                or out of contig bounds).
        """
        out = np.zeros(length, dtype=np.float32)
        bw_chrom = self._resolve_chrom(chrom, avail)
        if bw_chrom is None:
            return out
        chrom_len = int(handle.chroms()[bw_chrom])
        lo = max(0, start_0b)
        hi = min(chrom_len, start_0b + length)
        if hi <= lo:
            return out
        vals = handle.values(bw_chrom, lo, hi, numpy=True)
        vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        out[lo - start_0b : hi - start_0b] = vals
        return out

    def window(
        self, chrom: str, start_0b: int, length: int, strand: str
    ) -> np.ndarray:
        """Build the 2-channel conservation track for a window.

        Args:
            chrom (str): H5 chrom label.
            start_0b (int): 0-based + strand genomic start of the window.
            length (int): window length in bp (== model window length).
            strand (str): ``'+'`` or ``'-'``. Minus reverses the array to align
                with the model's reverse-complemented input.

        Returns:
            np.ndarray: float32 array of shape ``(length, 2)`` — ch0 phyloP
                (clipped/normalised), ch1 phastCons.
        """
        pv = self._values(self._phylop, chrom, start_0b, length, self._phylop_chroms)
        cv = self._values(
            self._phastcons, chrom, start_0b, length, self._phastcons_chroms
        )
        pv = np.clip(pv, -self.phylop_clip, self.phylop_clip) / self.phylop_clip
        feat = np.stack([pv, cv], axis=-1).astype(np.float32)
        if strand == "-":
            feat = feat[::-1].copy()
        return feat

    def close(self) -> None:
        """Close both bigWig handles."""
        self._phylop.close()
        self._phastcons.close()
