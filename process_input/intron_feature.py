"""Per-base intron-structure feature track for the hybrid CNN head.

Builds a 2-channel per-position feature track, aligned to the model's
**input window** (length ``sl + cl_max``, the same frame as the encoded
sequence ``X`` produced by :func:`process_input.h5_utils.create_datapoints`),
so it can be concatenated to the one-hot input of the hybrid convolutional
head (``model.model._convert_tokens_to_onehot`` -> ``hybrid_head``).

Channels (both derived only from the gene's canonical splice-site
coordinates -- ``JN_START`` donors + ``JN_END`` acceptors -- so there is no
variant/label leakage):

    ch0  depth:   clipped, normalised distance to the nearest canonical
                  splice site. 0.0 exactly at a splice site, 1.0 at/beyond
                  ``DEPTH_CAP`` bp away. Encodes "how deep" a position is.
    ch1  regime:  normalised log length of the inter-site segment the
                  position falls in (the local intron/exon length).
                  Encodes the exon-definition vs intron-definition regime,
                  which long introns switch.

The windowing mirrors the ``X`` branch of
:func:`process_input.h5_utils.reformat_data` exactly, so window ``i`` row
``half_cl + m`` aligns with splice label ``Y[i][m]`` (verified in
``test/test_intron_feature.py``).
"""

import re

import numpy as np

from process_input.h5_utils import ceil_div

# Number of feature channels emitted per position.
INTRON_FEATURE_CHANNELS = 2

# Distance (bp) at and beyond which the depth channel saturates to 1.0.
DEPTH_CAP = 5000.0

# log10 length normaliser for the segment-length channel (1e6 bp -> 1.0).
SEGLEN_LOG_NORM = 6.0


def _parse_site_csv(csv: str) -> list[int]:
    """Parse a trailing-comma CSV of 1-based positions into a list of ints."""
    return [int(x) for x in re.split(",", csv) if x]


def load_gene_sites_lookup(datafile_path: str) -> dict[str, np.ndarray]:
    """Build a ``gene_name -> sorted splice-site positions`` lookup.

    Reads the per-gene canonical donor (``JN_START``) + acceptor (``JN_END``)
    coordinates from a datafile (the same source the training sidecar uses), so
    inference-time feature tracks are built from the identical site set. Keyed
    by the datafile ``NAME`` field (gene symbol), matching the eval H5's
    ``source_gene_name``.

    Args:
        datafile_path (str): Path to ``datafile_*.h5`` with ``NAME`` /
            ``JN_START`` / ``JN_END``.

    Returns:
        dict[str, np.ndarray]: gene name -> int64 array of sorted sites.
    """
    import h5py

    with h5py.File(datafile_path, "r") as f:
        names = f["NAME"][:].astype(str)
        js = f["JN_START"][:]
        je = f["JN_END"][:]
        lut: dict[str, np.ndarray] = {}
        for i, nm in enumerate(names):
            s = js[i][0] if js.ndim == 2 else js[i]
            e = je[i][0] if je.ndim == 2 else je[i]
            s = s.decode() if isinstance(s, bytes) else s
            e = e.decode() if isinstance(e, bytes) else e
            sites = sorted(set(_parse_site_csv(s)) | set(_parse_site_csv(e)))
            lut[nm] = np.asarray(sites, dtype=np.int64)
    return lut


def _feature_from_sites(
    positions: np.ndarray, sites_arr: np.ndarray
) -> np.ndarray:
    """Compute the 2-channel feature for genomic ``positions`` given ``sites``.

    Shared core used by both the training-window builder and the
    inference-window builder so the feature definition is identical. Assumes
    ``sites_arr`` is non-empty and sorted ascending.

    Args:
        positions (np.ndarray): 1-based genomic positions (ascending), int64.
        sites_arr (np.ndarray): sorted canonical splice-site positions, int64.

    Returns:
        np.ndarray: float32 array of shape ``(len(positions), 2)`` —
            ch0 depth, ch1 regime.
    """
    feat = np.zeros((len(positions), INTRON_FEATURE_CHANNELS), dtype=np.float32)

    ins = np.searchsorted(sites_arr, positions)
    left_idx = np.clip(ins - 1, 0, len(sites_arr) - 1)
    right_idx = np.clip(ins, 0, len(sites_arr) - 1)
    d_left = np.abs(positions - sites_arr[left_idx])
    d_right = np.abs(positions - sites_arr[right_idx])
    dist = np.minimum(d_left, d_right).astype(np.float32)
    feat[:, 0] = np.minimum(dist, DEPTH_CAP) / DEPTH_CAP

    lo = sites_arr[left_idx].astype(np.float64)
    hi = sites_arr[right_idx].astype(np.float64)
    seg_len = np.where(ins == right_idx, hi - lo, DEPTH_CAP)  # ins==0 or ==len
    inside = (positions >= sites_arr[0]) & (positions <= sites_arr[-1])
    seg_len = np.where(inside, np.maximum(seg_len, 1.0), DEPTH_CAP)
    feat[:, 1] = (np.log10(seg_len + 1.0) / SEGLEN_LOG_NORM).astype(np.float32)

    return feat


def intron_feature_for_chunk(
    chunk_start_0b: int,
    length: int,
    strand: str,
    sites_arr: np.ndarray,
) -> np.ndarray:
    """Build the feature track for an inference chunk window.

    Mirrors a single training window: the chunk spans 0-based genomic
    ``[chunk_start_0b, chunk_start_0b + length)`` on the **+ strand**; on the
    minus strand the array is reversed to align with the reverse-complemented
    sequence the model receives (matching ``create_datapoints`` / the eval
    chunk generator's RC).

    Args:
        chunk_start_0b (int): 0-based + strand genomic start of the chunk.
        length (int): chunk length in bp (== ``sl + cl_max``).
        strand (str): ``'+'`` or ``'-'``.
        sites_arr (np.ndarray): sorted canonical splice-site positions of the
            variant's anchored gene (1-based), int64. May be empty.

    Returns:
        np.ndarray: float32 array of shape ``(length, INTRON_FEATURE_CHANNELS)``.
    """
    positions = np.arange(
        chunk_start_0b + 1, chunk_start_0b + 1 + length, dtype=np.int64
    )
    if len(sites_arr) == 0:
        feat = np.zeros((length, INTRON_FEATURE_CHANNELS), dtype=np.float32)
        feat[:, 0] = 1.0
    else:
        feat = _feature_from_sites(positions, sites_arr)
    if strand == "-":
        feat = feat[::-1].copy()
    return feat


def build_intron_feature_array(
    tx_start: int,
    tx_end: int,
    strand: str,
    jn_start_csv: str,
    jn_end_csv: str,
    cl_max: int,
) -> np.ndarray:
    """Build the per-base intron-structure feature array for one gene.

    The array spans ``[tx_start - half_cl, tx_end + half_cl]`` (length
    ``tx_len + cl_max``), matching the encoded-sequence array ``X0`` built by
    :func:`create_datapoints` before windowing, and is returned in transcript
    5'->3' order (genomic order is reversed on the minus strand) so it aligns
    with the reverse-complemented sequence.

    Args:
        tx_start (int): Transcription start (1-based, inclusive).
        tx_end (int): Transcription end (1-based, inclusive).
        strand (str): ``'+'`` or ``'-'``.
        jn_start_csv (str): Trailing-comma CSV of donor (exon-end) positions.
        jn_end_csv (str): Trailing-comma CSV of acceptor (exon-start) positions.
        cl_max (int): Total flank context (``half_cl`` on each side).

    Returns:
        np.ndarray: Float32 array of shape ``(tx_len + cl_max,
            INTRON_FEATURE_CHANNELS)``.
    """
    half_cl = cl_max // 2
    span_start = tx_start - half_cl
    span_end = tx_end + half_cl  # inclusive
    span_len = span_end - span_start + 1  # == tx_len + cl_max

    sites = sorted(
        set(_parse_site_csv(jn_start_csv)) | set(_parse_site_csv(jn_end_csv))
    )

    feat = np.zeros((span_len, INTRON_FEATURE_CHANNELS), dtype=np.float32)

    if not sites:
        # No annotated junctions: depth saturates everywhere, segment
        # length undefined -> leave regime at 0 (log of the cap).
        feat[:, 0] = 1.0
        if strand == "-":
            feat = feat[::-1].copy()
        return feat

    sites_arr = np.asarray(sites, dtype=np.int64)
    positions = np.arange(span_start, span_end + 1, dtype=np.int64)
    feat = _feature_from_sites(positions, sites_arr)

    if strand == "-":
        feat = feat[::-1].copy()

    return feat


def window_input_array(
    f0: np.ndarray,
    cl_max: int,
    sl: int,
    stride: int | None = None,
) -> np.ndarray:
    """Window an input-space per-base array into ``(num_points, sl+cl_max, k)``.

    Mirrors the ``X`` branch of :func:`reformat_data`: ``num_points`` and the
    block start offsets are identical, so the returned windows align
    position-for-position with the encoded-sequence windows ``X``.

    Args:
        f0 (np.ndarray): Per-base array of shape ``(tx_len + cl_max, k)`` in
            transcript 5'->3' order (output of
            :func:`build_intron_feature_array`).
        cl_max (int): Total flank context.
        sl (int): Target region length per block.
        stride (int | None): Block stride; defaults to ``sl``.

    Returns:
        np.ndarray: Float32 array of shape
            ``(num_points, sl + cl_max, k)``.
    """
    if stride is None:
        stride = sl
    if not (1 <= stride <= sl):
        raise ValueError(f"stride must be in [1, sl={sl}], got {stride}")

    k = f0.shape[1]
    tx_len = f0.shape[0] - cl_max
    if tx_len <= 0:
        return np.zeros((0, sl + cl_max, k), dtype=np.float32)

    num_points = ceil_div(max(tx_len - sl, 0), stride) + 1
    f0p = np.pad(f0, [(0, sl), (0, 0)], "constant", constant_values=0.0)

    out = np.zeros((num_points, sl + cl_max, k), dtype=np.float32)
    for i in range(num_points):
        start = stride * i
        out[i] = f0p[start : start + sl + cl_max]

    return out


def build_intron_feature_windows(
    tx_start: int,
    tx_end: int,
    strand: str,
    jn_start_csv: str,
    jn_end_csv: str,
    cl_max: int,
    sl: int,
    stride: int | None = None,
) -> np.ndarray:
    """Build intron-feature windows for one gene, aligned to its ``X`` windows.

    Convenience wrapper combining :func:`build_intron_feature_array` and
    :func:`window_input_array`.

    Returns:
        np.ndarray: Float32 array of shape ``(num_points, sl + cl_max,
            INTRON_FEATURE_CHANNELS)``.
    """
    f0 = build_intron_feature_array(
        tx_start, tx_end, strand, jn_start_csv, jn_end_csv, cl_max
    )
    return window_input_array(f0, cl_max, sl, stride)
