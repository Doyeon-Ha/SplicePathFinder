"""Annotation-referenced delta: coordinate mapping and the per-channel mask."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from canonical_site_track import (  # noqa: E402
    annotation_probs,
    apply_canonical_mask,
    site_indices,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_splicevardb_sav_multichunk import (  # noqa: E402
    _compute_per_bp_deltas,
    _mask_sequence_outside_transcript,
)

_L = 60


def _gene() -> dict:
    """Return a gene whose exonic anchors sit at known genomic positions."""
    return {
        "chrom": "chr1", "strand": "+",
        "acc_exonic": np.array([1010, 1030]),
        "don_exonic": np.array([1020, 1040]),
        "acc_intronic": np.array([1009, 1029]),
        "don_intronic": np.array([1021, 1041]),
    }


class TestSiteIndices:
    """Genomic sites land on the right chunk-local index in both frames."""

    def test_plus_strand_is_a_plain_offset(self) -> None:
        acc, don = site_indices(_gene(), gstart=1000, length=_L, minus=False)
        assert acc.tolist() == [10, 30]
        assert don.tolist() == [20, 40]

    def test_minus_strand_is_mirrored(self) -> None:
        acc, don = site_indices(_gene(), gstart=1000, length=_L, minus=True)
        assert acc.tolist() == [_L - 1 - 30, _L - 1 - 10]
        assert don.tolist() == [_L - 1 - 40, _L - 1 - 20]

    def test_sites_outside_the_chunk_are_dropped(self) -> None:
        acc, don = site_indices(_gene(), gstart=1035, length=10, minus=False)
        assert acc.tolist() == []
        assert don.tolist() == [5]


class TestAnnotationProbs:
    """The stand-in reference readout is a clean one-hot."""

    def test_one_at_sites_zero_elsewhere(self) -> None:
        p = annotation_probs(np.array([10]), np.array([20]), _L)
        assert p[10, 0] == 1.0 and p[10, 1] == 0.0
        assert p[20, 1] == 1.0 and p[20, 0] == 0.0
        assert p[5].tolist() == [0.0, 0.0, 1.0]
        assert np.allclose(p.sum(axis=1), 1.0)


class TestCanonicalMask:
    """The rule is per channel, and a no-op once the reference is annotation."""

    def test_acceptor_site_still_allows_a_donor_gain(self) -> None:
        acc, don = np.array([10]), np.array([20])
        tracks = np.ones((_L, 4), dtype=np.float32)
        out = apply_canonical_mask(tracks, acc, don)
        # at the annotated acceptor: acc gain killed, acc loss kept,
        # donor gain SURVIVES because no donor is annotated there.
        assert out[10, 0] == 0.0
        assert out[10, 1] == 1.0
        assert out[10, 2] == 1.0
        assert out[10, 3] == 0.0
        # mirrored at the annotated donor
        assert out[20, 2] == 0.0 and out[20, 3] == 1.0
        assert out[20, 0] == 1.0 and out[20, 1] == 0.0

    def test_unannotated_position_keeps_gains_only(self) -> None:
        out = apply_canonical_mask(np.ones((_L, 4), np.float32),
                                   np.array([10]), np.array([20]))
        assert out[5].tolist() == [1.0, 0.0, 1.0, 0.0]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mask_is_a_noop_against_an_annotation_reference(self, seed) -> None:
        """With p_ref = annotation the rule already holds by construction.

        This is the invariant that keeps the two arms honest: if the explicit
        mask ever changes the annotation arm's score, the track and the mask
        have drifted onto different coordinates.
        """
        rng = np.random.default_rng(seed)
        acc = np.array([7, 33])
        don = np.array([18, 44])
        p_ref = annotation_probs(acc, don, _L)
        p_alt = rng.random((_L, 3)).astype(np.float32)
        p_alt /= p_alt.sum(axis=1, keepdims=True)
        tracks = _compute_per_bp_deltas(p_ref, p_alt)
        assert np.array_equal(tracks, apply_canonical_mask(tracks, acc, don))


class TestTranscriptBoundaryMask:
    """전사체 밖 N 패딩은 두 strand의 model 좌표에서 동일하게 동작한다."""

    @pytest.mark.parametrize("strand", ["+", "-"])
    def test_masks_only_positions_outside_inclusive_bounds(self, strand) -> None:
        seq = np.arange(1, 11, dtype=np.int8)
        out, n = _mask_sequence_outside_transcript(
            seq, pos_hg38=105, clp_model=5, strand=strand,
            tx_start=101, tx_end=107,
        )
        assert n == 4
        if strand == "+":
            assert out.tolist() == [0, 0, 3, 4, 5, 6, 7, 8, 0, 0]
        else:
            assert out.tolist() == [0, 0, 0, 4, 5, 6, 7, 8, 9, 0]
