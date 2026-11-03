"""Per-variant K=4 chunk geometry, shared by the cache builder and the e2e path.

``build_multichunk_pernt_cache.py`` collapses four chunks' supervised targets onto
a variant-centred genomic axis. The end-to-end trainer has to do the same thing to
a live backbone, so the geometry lives here rather than being written twice: an
e2e arm and the frozen head it is compared against must see byte-identical inputs
or the comparison measures the discrepancy instead of the freeze.

What a collapse needs, per chunk, is which position of that chunk's probability
track supplies each output position -- a fixed integer mapping that depends only
on the variant and the chunk, never on the model. This builds that mapping so the
runtime op is a gather.

For an indel the mapping is whatever ``_align_indel_probs`` does, and it is
recovered by running that function itself on a probe track carrying its own
indices. Reimplementing the alignment here would be a second source of truth for
the one part of the pipeline that has already caused a wrong answer once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_splicevardb_sav_indel import (  # noqa: E402
    _build_indel_chunk,
)
from scripts.evaluate_splicevardb_sav_indel_v4 import (  # noqa: E402
    _align_indel_probs,
)
from scripts.evaluate_splicevardb_sav_multichunk import (  # noqa: E402
    _aligned_chunk_positions,
    _generate_chunks_for_variant,
)

TARGET_START, TARGET_END = 5000, 10000
NO_SOURCE = -1


def chunk_positions(variant_offset: int, total_length: int, stride: int) -> list[int]:
    """Return the chunk-local variant positions that place it in the target.

    Args:
        variant_offset (int): the variant's position in the canonical chunk.
        total_length (int): chunk length.
        stride (int): training stride.

    Returns:
        list[int]: the K chunk-local positions, ascending.
    """
    return [
        c for c in _aligned_chunk_positions(
            variant_offset=variant_offset, total_length=total_length, stride=stride)
        if TARGET_START <= c < TARGET_END
    ]


def _indel_source_indices(
    rec: dict, total_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recover which chunk position each aligned row of an indel track came from.

    Runs the canonical alignment on a probe track whose value at position ``i`` is
    ``i + 1``, so a zero-compared row -- one where the allele has no base to pair
    against -- comes back as 0 and is distinguishable from position 0.

    Args:
        rec (dict): the ``_build_indel_chunk`` record, with ``l_ref``/``l_alt``/
            ``strand`` filled in.
        total_length (int): chunk length.

    Returns:
        tuple[np.ndarray, np.ndarray]: ref and alt source indices, ``NO_SOURCE``
            where the row is zero-compared.
    """
    probe = (np.arange(total_length, dtype=np.float32) + 1.0)[:, None].repeat(3, 1)
    a_ref, a_alt = _align_indel_probs(
        probe, probe, rec, delta_window=total_length // 2,
        total_length=total_length)
    return (a_ref[:, 0].astype(np.int64) - 1, a_alt[:, 0].astype(np.int64) - 1)


def variant_geometry(
    fasta: object,
    variant: dict,
    *,
    total_length: int = 15000,
    stride: int = 1250,
    window_left: int = 4375,
    window_right: int = 4375,
) -> dict | None:
    """Build one variant's chunk tokens and per-chunk source-index mapping.

    Args:
        fasta (object): an open pyfaidx ``Fasta``.
        variant (dict): a row from ``_load_variant_table``.
        total_length (int): chunk length.
        stride (int): training stride.
        window_left (int): output positions before the variant.
        window_right (int): output positions from the variant onward.

    Returns:
        dict | None: ``x_ref``/``x_alt`` of shape ``(K, total_length)`` int8,
            ``ref_src``/``alt_src`` of shape ``(K, W)`` int64 (``NO_SOURCE`` where
            the row is zero-compared or outside the chunk's target) and
            ``covered`` of shape ``(K, W)`` bool, or None if no chunk could be
            built.
    """
    w = window_left + window_right
    clps = chunk_positions(variant["variant_offset"], total_length, stride)
    is_snv = len(variant["ref"]) == 1 and len(variant["alt"]) == 1
    xs_ref, xs_alt, srcs = [], [], []

    for clp in clps:
        fwd = total_length - 1 - clp if variant["strand"] == "-" else clp
        if is_snv:
            recs = _generate_chunks_for_variant(
                fasta=fasta, chrom=variant["chrom"], pos_hg38=variant["pos"],
                ref=variant["ref"], alt=variant["alt"], strand=variant["strand"],
                total_length=total_length, chunk_local_positions=[fwd])
            r = recs[0]
            if r["status"] != "ok":
                continue
            # A raw chunk needs no alignment: output position w maps to the
            # variant-relative offset w - window_left away from clp.
            var_idx = r["clp_model"]
            ref_src = alt_src = None
        else:
            rec = _build_indel_chunk(
                fasta=fasta, chrom=variant["chrom"], pos_hg38=variant["pos"],
                ref=variant["ref"], alt=variant["alt"], strand=variant["strand"],
                total_length=total_length, clp=fwd)
            if rec is None:
                continue
            r = dict(rec)
            r["l_ref"], r["l_alt"] = len(variant["ref"]), len(variant["alt"])
            r["strand"] = variant["strand"]
            ref_src, alt_src = _indel_source_indices(r, total_length)
            # The alignment re-bases the track on the variant, which now sits at
            # the number of upstream rows it emitted.
            var_idx = min(
                total_length // 2, r["var_start_ref"], r["var_start_alt"])

        # This chunk only speaks for its supervised target, in coordinates
        # relative to the variant.
        lo = max(TARGET_START - clp, -window_left)
        hi = min(TARGET_END - 1 - clp, window_right - 1)
        n_track = total_length if ref_src is None else len(ref_src)
        lo = max(lo, -var_idx)
        hi = min(hi, n_track - 1 - var_idx)
        if lo > hi:
            continue
        rs = np.full(w, NO_SOURCE, dtype=np.int64)
        as_ = np.full(w, NO_SOURCE, dtype=np.int64)
        cov = np.zeros(w, dtype=bool)
        track = np.arange(lo + var_idx, hi + var_idx + 1)
        dst = slice(lo + window_left, hi + window_left + 1)
        rs[dst] = track if ref_src is None else ref_src[track]
        as_[dst] = track if alt_src is None else alt_src[track]
        # Coverage is the span this chunk speaks for, NOT where a source index
        # exists. A v4 zero-compare row -- a base one allele does not have -- is a
        # covered position whose value is legitimately zero, and treating it as
        # uncovered both undercounts coverage and drops the position out of the
        # per-position chunk selection.
        cov[dst] = True
        xs_ref.append(r["x_ref"])
        xs_alt.append(r["x_alt"])
        srcs.append((rs, as_, cov))

    if not srcs:
        return None
    return {
        "x_ref": np.stack(xs_ref).astype(np.int8),
        "x_alt": np.stack(xs_alt).astype(np.int8),
        "ref_src": np.stack([s[0] for s in srcs]),
        "alt_src": np.stack([s[1] for s in srcs]),
        "covered": np.stack([s[2] for s in srcs]),
    }
