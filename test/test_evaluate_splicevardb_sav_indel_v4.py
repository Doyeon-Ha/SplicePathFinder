"""Unit tests for the v4 per-position aligned indel SAV evaluator.

These tests exercise the scoring logic on synthetic probability arrays
(no model / FASTA needed). They assert the diff-then-max behaviour that
distinguishes v4 from the v3 max-then-diff rule, the zero-compare for
deleted/inserted bases on both strands, and SNV-equivalence.
"""

import numpy as np
import pytest

from scripts.evaluate_splicevardb_sav_indel_v3 import _score_indel_v3
from scripts.evaluate_splicevardb_sav_indel_v4 import (
    _align_indel_probs,
    _aligned_ref_indices,
    _indel_record_from_h5,
    _score_indel_v4,
)
from scripts.evaluate_splicevardb_sav_multichunk import _compute_per_bp_deltas

TOTAL_LENGTH = 40
CLP = 20
DELTA_WINDOW = 4
ACC = 0
DON = 1


def _make_rec(l_ref: int, l_alt: int, strand: str) -> dict:
    """Build a variant record matching ``_build_indel_chunk`` coords."""
    if strand == "+":
        var_start_ref = CLP
        var_start_alt = CLP
    else:
        var_start_ref = TOTAL_LENGTH - CLP - l_ref
        var_start_alt = TOTAL_LENGTH - CLP - l_alt
    return {
        "var_start_ref": var_start_ref,
        "var_end_ref": var_start_ref + l_ref,
        "var_start_alt": var_start_alt,
        "var_end_alt": var_start_alt + l_alt,
        "l_ref": l_ref,
        "l_alt": l_alt,
        "strand": strand,
    }


def _zeros() -> np.ndarray:
    """Return a zeroed ``(TOTAL_LENGTH, 3)`` probability array."""
    return np.zeros((TOTAL_LENGTH, 3), dtype=np.float32)


@pytest.mark.parametrize(
    ("strand", "l_ref", "l_alt", "expected"),
    [
        ("+", 3, 1, (20, 23, 20, 21)),
        ("-", 3, 1, (18, 21, 18, 19)),
        ("-", 1, 3, (20, 21, 20, 23)),
    ],
)
def test_h5_record_uses_model_space_variant_offset(
    strand: str, l_ref: int, l_alt: int, expected: tuple[int, ...]
) -> None:
    """Stored H5 chunks already use model coordinates, including on minus."""
    rec = _indel_record_from_h5(
        np.zeros(TOTAL_LENGTH, dtype=np.int8),
        np.zeros(TOTAL_LENGTH, dtype=np.int8),
        variant_offset=CLP,
        ref="A" * l_ref,
        alt="T" * l_alt,
        strand=strand,
        total_length=TOTAL_LENGTH,
    )
    got = (
        rec["var_start_ref"], rec["var_end_ref"],
        rec["var_start_alt"], rec["var_end_alt"],
    )
    assert got == expected
    assert rec["strand"] == "+"  # H5 model-array alignment is left anchored.
    assert rec["source_strand"] == strand


def test_diff_then_max_captures_offset_peaks() -> None:
    """v4 captures a loss and a gain that v3 max-then-diff cancels out.

    Ref has a strong acceptor at one flank position and alt has an
    equally strong acceptor at a *different* aligned position. The
    per-position rule sees a 0.9 loss and a 0.9 gain; the max-then-diff
    rule subtracts the two equal peaks and reports ~0.
    """
    rec = _make_rec(l_ref=2, l_alt=1, strand="+")  # deletion
    prob_ref = _zeros()
    prob_alt = _zeros()
    prob_ref[17, ACC] = 0.9
    prob_alt[18, ACC] = 0.9

    v4 = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert v4["delta_max"] == pytest.approx(0.9, abs=1e-6)
    assert v4["delta_acc_loss"] == pytest.approx(0.9, abs=1e-6)
    assert v4["delta_acc_gain"] == pytest.approx(0.9, abs=1e-6)

    v3 = _score_indel_v3(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert v3["delta_max"] < 0.05


def test_deletion_zero_compare_loss() -> None:
    """A deleted ref base with high acceptor prob becomes a loss."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="+")
    prob_ref = _zeros()
    prob_alt = _zeros()
    # Extra (deleted) ref bases sit at the tail: indices 21, 22.
    prob_ref[22, ACC] = 0.8

    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["delta_acc_loss"] == pytest.approx(0.8, abs=1e-6)
    assert scores["delta_max"] == pytest.approx(0.8, abs=1e-6)


def test_insertion_zero_compare_gain() -> None:
    """An inserted alt base with high donor prob becomes a gain."""
    rec = _make_rec(l_ref=1, l_alt=3, strand="+")
    prob_ref = _zeros()
    prob_alt = _zeros()
    # Extra (inserted) alt bases sit at the tail: indices 21, 22.
    prob_alt[21, DON] = 0.7

    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["delta_don_gain"] == pytest.approx(0.7, abs=1e-6)
    assert scores["delta_max"] == pytest.approx(0.7, abs=1e-6)


def test_minus_strand_deletion_head_zero_compare() -> None:
    """On the - strand the deleted bases sit at the head and zero-compare."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="-")
    # var_start_ref = 40 - 20 - 3 = 17; var_end_ref = 20; overlap = 1.
    # Extra ref head bases: indices 17, 18.
    prob_ref = _zeros()
    prob_alt = _zeros()
    prob_ref[17, ACC] = 0.6

    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["delta_acc_loss"] == pytest.approx(0.6, abs=1e-6)


def test_snv_reduces_to_per_position_diff() -> None:
    """l_ref == l_alt == 1 reduces to the SNV per-position diff-then-max."""
    rec = _make_rec(l_ref=1, l_alt=1, strand="+")
    prob_ref = _zeros()
    prob_alt = _zeros()
    prob_ref[19, ACC] = 0.5
    prob_alt[19, ACC] = 0.2
    prob_alt[21, DON] = 0.4

    aligned_ref, aligned_alt = _align_indel_probs(
        prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH,
    )
    expected = _compute_per_bp_deltas(aligned_ref, aligned_alt).max(axis=0)
    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["delta_acc_loss"] == pytest.approx(float(expected[1]), abs=1e-6)
    assert scores["delta_don_gain"] == pytest.approx(float(expected[2]), abs=1e-6)
    assert scores["delta_max"] == pytest.approx(0.4, abs=1e-6)


def test_alignment_pairs_have_equal_length() -> None:
    """Aligned ref/alt arrays always have matching row counts."""
    for strand in ("+", "-"):
        for l_ref, l_alt in ((1, 1), (3, 1), (1, 3), (2, 2)):
            rec = _make_rec(l_ref=l_ref, l_alt=l_alt, strand=strand)
            aligned_ref, aligned_alt = _align_indel_probs(
                _zeros(), _zeros(), rec, DELTA_WINDOW, TOTAL_LENGTH,
            )
            assert aligned_ref.shape == aligned_alt.shape
            assert aligned_ref.shape[1] == 3


def test_aligned_ref_indices_match_the_track_rows() -> None:
    """One reference index per aligned row, on both strands and both lengths."""
    for strand in ("+", "-"):
        for l_ref, l_alt in ((1, 1), (3, 1), (1, 3), (2, 2)):
            rec = _make_rec(l_ref=l_ref, l_alt=l_alt, strand=strand)
            aligned_ref, _ = _align_indel_probs(
                _zeros(), _zeros(), rec, DELTA_WINDOW, TOTAL_LENGTH,
            )
            rows = _aligned_ref_indices(rec, DELTA_WINDOW, TOTAL_LENGTH)
            assert rows.shape[0] == aligned_ref.shape[0]


def test_inserted_rows_have_no_reference_position() -> None:
    """Inserted bases exist in neither allele's reference, so they map to -1."""
    rec = _make_rec(l_ref=1, l_alt=3, strand="+")
    rows = _aligned_ref_indices(rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert (rows == -1).sum() == 2
    # Every other row is a real reference position, strictly increasing.
    kept = rows[rows >= 0]
    assert np.all(np.diff(kept) > 0)


def test_deleted_rows_keep_their_reference_position() -> None:
    """A deleted base still exists in the reference, so it can be a site."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="+")
    rows = _aligned_ref_indices(rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert -1 not in rows
    assert list(rows[DELTA_WINDOW:DELTA_WINDOW + 3]) == [CLP, CLP + 1, CLP + 2]


def test_mask_keeps_loss_at_an_annotated_deleted_donor() -> None:
    """A deletion over an annotated donor keeps donor loss and drops donor gain."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="+")
    prob_ref, prob_alt = _zeros(), _zeros()
    prob_ref[CLP + 1, DON] = 0.9          # annotated donor, deleted by the variant
    prob_alt[CLP + 1, DON] = 0.0
    prob_alt[CLP - 2, DON] = 0.7          # cryptic donor gain in the upstream flank

    sites = (np.array([], dtype=np.int64), np.array([CLP + 1], dtype=np.int64))
    scores = _score_indel_v4(
        prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH, mask_sites=sites,
    )
    assert scores["delta_don_loss"] == pytest.approx(0.9, abs=1e-6)
    assert scores["delta_don_gain"] == pytest.approx(0.7, abs=1e-6)


def test_mask_drops_loss_where_no_site_is_annotated() -> None:
    """Loss away from an annotated site of that type is zeroed; gain survives."""
    rec = _make_rec(l_ref=1, l_alt=3, strand="+")
    prob_ref, prob_alt = _zeros(), _zeros()
    prob_ref[CLP - 3, ACC] = 0.8          # unannotated -- the loss here is noise
    prob_alt[CLP - 2, ACC] = 0.6          # unannotated -- a real gain

    empty = np.array([], dtype=np.int64)
    scores = _score_indel_v4(
        prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH,
        mask_sites=(empty, empty),
    )
    assert scores["delta_acc_loss"] == pytest.approx(0.0, abs=1e-6)
    assert scores["delta_acc_gain"] == pytest.approx(0.6, abs=1e-6)


def test_mask_is_a_no_op_when_no_sites_are_passed() -> None:
    """Without mask_sites the scorer is byte-identical to the unmasked path."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="-")
    prob_ref, prob_alt = _zeros(), _zeros()
    prob_ref[17, ACC] = 0.5
    prob_alt[19, DON] = 0.4
    a = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    b = _score_indel_v4(
        prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH, mask_sites=None,
    )
    assert a == b


def test_peak_offset_points_at_the_genomic_position() -> None:
    """On the plus strand the offset is the peak's distance from the variant."""
    rec = _make_rec(l_ref=1, l_alt=1, strand="+")
    prob_ref, prob_alt = _zeros(), _zeros()
    prob_alt[CLP + 2, DON] = 0.7          # donor gain two bases downstream
    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["offset_don_gain"] == 2
    assert scores["offset_delta_max"] == 2


def test_peak_offset_is_genomic_on_the_minus_strand() -> None:
    """A minus-strand peak downstream in model space is upstream genomically."""
    rec = _make_rec(l_ref=1, l_alt=1, strand="-")
    prob_ref, prob_alt = _zeros(), _zeros()
    # Model index CLP is the variant; +2 in model space runs genomically left.
    prob_alt[rec["var_start_ref"] + 2, DON] = 0.7
    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["offset_don_gain"] == -2


def test_minus_strand_deletion_anchors_on_the_vcf_position() -> None:
    """The VCF position is the last model-space base of a minus-strand ref span."""
    rec = _make_rec(l_ref=3, l_alt=1, strand="-")
    prob_ref, prob_alt = _zeros(), _zeros()
    # The base at the VCF position itself: model index var_end_ref - 1.
    prob_ref[rec["var_end_ref"] - 1, DON] = 0.9
    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["offset_don_loss"] == 0


def test_flat_direction_reports_no_peak() -> None:
    """A direction with no signal has no position to report."""
    from scripts.evaluate_splicevardb_sav_multichunk import _NO_PEAK

    rec = _make_rec(l_ref=1, l_alt=1, strand="+")
    prob_ref, prob_alt = _zeros(), _zeros()
    prob_alt[CLP + 1, ACC] = 0.5
    scores = _score_indel_v4(prob_ref, prob_alt, rec, DELTA_WINDOW, TOTAL_LENGTH)
    assert scores["offset_acc_gain"] == 1
    assert scores["offset_don_gain"] == _NO_PEAK
    assert scores["offset_don_loss"] == _NO_PEAK
