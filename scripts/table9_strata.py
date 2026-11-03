"""Canonical Table 9 (manuscript tab:clinvar-region-fine) distance-strata masks —
single source of truth so every foundation-model / DeltaReader 5-fold CV eval uses
identical definitions.

Strata are set by nucleotide position from the nearest GENCODE v49 internal splice
site (intron nt k = k-th intronic base, 1-based; exon nt k = k-th exonic base,
1-based, nt1 = boundary exonic base) and, where noted, by donor vs acceptor side of
that nearest site. The strata OVERLAP by design (canonical ⊂ essential; deep pooled
⊃ deep acceptor), so membership is returned as one boolean mask per Table-9 row,
not a single categorical label.

Inputs are the per-variant coordinate fields from coord_splice_fields_seed42.npz
(row-aligned to cache.h5): intron_nt / exon_nt (int; -1 where n/a), akind
(0=donor, 1=acceptor, -1=none). Their union tiles the foldset splice-region subset
(exon nt 1-3 ∪ all intron nt>=1).
"""
from __future__ import annotations
import numpy as np

# ordered to match the Table 9 rows
TABLE9_STRATA = [
    "canonical", "essential",
    "splice_region_pooled", "splice_region_donor", "splice_region_acceptor",
    "acc_branch_le40", "proximal_mid_intronic",
    "deep_intronic_pooled", "deep_intronic_acceptor",
]


def table9_masks(intron_nt, exon_nt, akind):
    """Return {stratum_name: bool ndarray} for the 9 Table-9 distance strata."""
    intron_nt = np.asarray(intron_nt); exon_nt = np.asarray(exon_nt); akind = np.asarray(akind)
    acc = akind == 1
    don = akind == 0
    in_ = lambda lo, hi: (intron_nt >= lo) & (intron_nt <= hi)
    ex = lambda lo, hi: (exon_nt >= lo) & (exon_nt <= hi)

    canonical = in_(1, 2)                                   # GT/AG dinucleotide
    essential = ex(1, 2) | in_(1, 2)                        # exon 1-2 + intron 1-2
    sr_band = ex(2, 3) | in_(3, 8)                          # dinucleotide removed
    return {
        "canonical": canonical,
        "essential": essential,
        "splice_region_pooled": sr_band,
        "splice_region_donor": sr_band & don,
        "splice_region_acceptor": sr_band & acc,
        "acc_branch_le40": acc & in_(1, 40),
        "proximal_mid_intronic": (don & in_(9, 99)) | (acc & in_(41, 99)),
        "deep_intronic_pooled": intron_nt >= 100,
        "deep_intronic_acceptor": acc & (intron_nt >= 100),
    }
