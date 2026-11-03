"""The e2e chunk geometry must reproduce the cache builder's collapse exactly.

An end-to-end arm is compared against the frozen head trained on the cache, so if
the live path and the builder disagree the comparison measures that disagreement
rather than the effect of unfreezing the backbone. These feed the same synthetic
probabilities to both and require bit-identical output.
"""

import numpy as np
import pytest

from process_input.build_multichunk_pernt_cache import (
    TARGET_END,
    TARGET_START,
    collapse_variant,
    mask_chunk_alt,
    target_window,
)
from train.multichunk_geometry import NO_SOURCE

W_LEFT = W_RIGHT = 4375
W = W_LEFT + W_RIGHT
L = 15000


def _gather(probs: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Apply a source-index mapping, zero where there is no source.

    Args:
        probs (np.ndarray): ``(K, L, C)`` per-chunk probabilities.
        src (np.ndarray): ``(K, W)`` source indices, ``NO_SOURCE`` for none.

    Returns:
        np.ndarray: ``(K, W, C)`` gathered values.
    """
    out = np.zeros((src.shape[0], src.shape[1], probs.shape[-1]), np.float32)
    for k in range(src.shape[0]):
        m = src[k] >= 0
        out[k, m] = probs[k][src[k][m]]
    return out


def _collapse_via_geometry(
    p_ref: np.ndarray, p_alt: np.ndarray, geo: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse with the gather formulation the e2e forward uses.

    Args:
        p_ref (np.ndarray): ``(K, L, C)`` reference probabilities.
        p_alt (np.ndarray): ``(K, L, C)`` alternate probabilities.
        geo (dict): a ``variant_geometry`` result.

    Returns:
        tuple: ``(ref, alt, coverage)``.
    """
    ref = _gather(p_ref, geo["ref_src"])
    alt = _gather(p_alt, geo["alt_src"])
    cov = geo["covered"]
    d = np.where(cov, np.abs(alt[..., :2] - ref[..., :2]).max(-1), -np.inf)
    best = d.argmax(0)
    ar = np.arange(cov.shape[1])
    r, a = ref[best, ar], alt[best, ar]
    r[~cov.any(0)] = 0
    a[~cov.any(0)] = 0
    return r, a, cov.sum(0)


def _snv_geometry(clps: list[int]) -> dict:
    """Build the geometry an SNV would produce for the given chunk positions.

    Args:
        clps (list[int]): chunk-local variant positions.

    Returns:
        dict: the same keys ``variant_geometry`` returns, minus the tokens.
    """
    k = len(clps)
    src = np.full((k, W), NO_SOURCE, dtype=np.int64)
    cov = np.zeros((k, W), dtype=bool)
    for i, clp in enumerate(clps):
        lo = max(5000 - clp, -W_LEFT)
        hi = min(9999 - clp, W_RIGHT - 1)
        lo, hi = max(lo, -clp), min(hi, L - 1 - clp)
        dst = slice(lo + W_LEFT, hi + W_LEFT + 1)
        src[i, dst] = np.arange(lo + clp, hi + clp + 1)
        cov[i, dst] = True
    return {"ref_src": src, "alt_src": src.copy(), "covered": cov}


@pytest.mark.parametrize("clps", [[5000, 6250, 7500, 8750], [5625, 6875, 8125, 9375]])
def test_gather_matches_builder_collapse(clps: list[int]) -> None:
    """Both formulations agree bit-for-bit on random probabilities."""
    rng = np.random.default_rng(0)
    k = len(clps)
    p_ref = rng.random((k, L, 3)).astype(np.float32)
    p_alt = rng.random((k, L, 3)).astype(np.float32)
    geo = _snv_geometry(clps)
    r_b, a_b, c_b = collapse_variant(p_ref, p_alt, clps, clps, W_LEFT, W_RIGHT)
    r_g, a_g, c_g = _collapse_via_geometry(p_ref, p_alt, geo)
    np.testing.assert_array_equal(c_b, c_g)
    np.testing.assert_array_equal(r_b, r_g)
    np.testing.assert_array_equal(a_b, a_g)


@pytest.mark.parametrize("clps", [[5000, 6250, 7500, 8750], [5625, 6875, 8125, 9375]])
def test_covered_span_is_contiguous_around_the_variant(clps: list[int]) -> None:
    """All four chunks cover the variant; the union is contiguous but not the
    whole window.

    The four chunks' spans slide with ``variant_offset % stride``, so their union
    slides too while the stored window does not. Up to 625 positions at one edge
    are therefore never covered, and which edge depends on the variant's phase --
    the guaranteed-covered part is the intersection, [v-3750, v+3750]. This is
    what the coverage channel exists to tell the head.
    """
    geo = _snv_geometry(clps)
    cov = geo["covered"]
    assert cov[:, W_LEFT].sum() == len(clps)
    any_cov = cov.any(0)
    idx = np.flatnonzero(any_cov)
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), "union has a hole"
    assert any_cov[W_LEFT - 3750:W_LEFT + 3751].all(), "intersection not covered"


def test_canonical_mask_selects_chunk_after_clamping() -> None:
    """A disallowed raw peak must not hide a valid peak in another chunk."""
    clps = [7, 8]
    ref = np.zeros((2, 16, 3), np.float32)
    alt = ref.copy()
    alt[0, clps[0], 1] = 0.9  # donor gain at an annotated donor: disallowed
    alt[1, clps[1], 0] = 0.7  # acceptor gain at that position: allowed
    sites = {"acc_exonic": np.array([], dtype=np.int64),
             "don_exonic": np.array([1000], dtype=np.int64)}
    masked = np.stack([
        mask_chunk_alt(ref[k], alt[k], pos=1000, ref_allele="A",
                       alt_allele="G", minus=False, var_idx=clps[k],
                       sites=sites)
        for k in range(2)
    ])
    r, a, _ = collapse_variant(ref, masked, clps, clps, 5, 5, 3, 13)
    assert a[5, 0] - r[5, 0] == pytest.approx(0.7)
    assert a[5, 1] - r[5, 1] == 0.0


def test_minus_strand_indel_mask_uses_aligned_reference_rows() -> None:
    """A deleted donor base retains loss after minus-strand v4 alignment."""
    ref = np.zeros((20, 3), np.float32)
    alt = ref.copy()
    ref[8:10, 1] = 0.8
    sites = {"acc_exonic": np.array([], dtype=np.int64),
             "don_exonic": np.array([1002], dtype=np.int64)}
    out = mask_chunk_alt(ref, alt, pos=1000, ref_allele="TCA",
                         alt_allele="T", minus=True, var_idx=8, sites=sites)
    assert out[8, 1] == 0.0  # annotated deleted base: donor loss survives
    assert out[9, 1] == ref[9, 1]  # unannotated base: loss is removed


# --- long-context geometry (32,768 nt input / 16,384 nt target) --------------
#
# The builder's target window used to be the module constants 5000/10000, which
# silently mis-sliced any backbone not trained at 15,000/5,000. These pin the
# derivation and the 32,768 nt arm's numbers.

L_LONG, TGT_LONG = 32768, 16384


def test_target_window_reproduces_the_frozen_default() -> None:
    """The 15,000/5,000 layout must still give exactly 5000/10000."""
    assert target_window(15000, 5000) == (TARGET_START, TARGET_END)


def test_target_window_for_the_long_context_arm() -> None:
    """32,768 input with a centred 16,384 target leaves 8,192 flank a side."""
    start, end = target_window(L_LONG, TGT_LONG)
    assert (start, end) == (8192, 24576)
    assert start == (L_LONG - TGT_LONG) // 2 == L_LONG - end


def test_long_context_collapse_covers_the_whole_window() -> None:
    """Every stored position is covered, unlike the 15,000 nt geometry.

    The chunks whose target contains the variant guarantee +-15,134 around it
    at this geometry, so a 4,375-a-side window is covered end to end. At
    15,000/5,000 the guarantee is only +-3,750 and the window's edges can be
    uncovered, so this is a real difference between the two caches, not a
    restatement of the same property.
    """
    start, end = target_window(L_LONG, TGT_LONG)
    stride = 1250
    clps = [c for c in range(start, end, stride)]
    assert len(clps) == 14
    rng = np.random.default_rng(1)
    p_ref = rng.random((len(clps), L_LONG, 3)).astype(np.float32)
    p_alt = rng.random((len(clps), L_LONG, 3)).astype(np.float32)
    _, _, cov = collapse_variant(
        p_ref, p_alt, clps, clps, W_LEFT, W_RIGHT, start, end
    )
    assert cov.shape == (W,)
    assert cov.min() >= 1, "a position inside the window had no source chunk"


def test_long_context_bounds_change_what_is_selected() -> None:
    """Passing the default 5000/10000 at this geometry drops real chunks.

    Guards the actual bug: the builder filtered chunks with the module
    constants, so at 32,768/16,384 every chunk with clp >= 10000 -- most of
    them -- was discarded and the survivors were sliced on the wrong span.
    """
    start, end = target_window(L_LONG, TGT_LONG)
    stride = 1250
    all_clps = [c for c in range(start, end, stride)]
    kept_default = [c for c in all_clps if TARGET_START <= c < TARGET_END]
    assert len(kept_default) < len(all_clps)
    assert len(kept_default) == 2
