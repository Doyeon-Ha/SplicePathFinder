"""Utility functions for H5 splice data creation.

Provides data transformation functions for converting genomic
sequences and splice annotations into fixed-size integer-encoded
blocks for model training.

Sequence encoding: 0=N(pad), 1=A, 2=C, 3=G, 4=T.
Splice label encoding: 0=no_splice, 1=acceptor, 2=donor, -1=padding.
Region label encoding: 0=intron, 1=5'UTR, 2=CDS_phase0,
    3=CDS_phase1, 4=CDS_phase2, 5=3'UTR, 6=ncRNA_exon, -1=padding.
Exon-position label encoding: 0=first_exon, 1=internal_exon,
    2=last_exon, -1=intron/single-exon/padding.

Ported from the original SpliceAI utils.py with parameterized
CL_max/SL and one-hot encoding removed in favour of compact
integer storage.
"""

import re
from math import ceil

import numpy as np

SEQ_ENCODE = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
"""Sequence integer encoding. 0=N(pad), 1=A, 2=C, 3=G, 4=T."""

SEQ_DECODE = np.array(["N", "A", "C", "G", "T"])
"""Reverse map: integer -> nucleotide character."""

REGION_INTRON = 0
REGION_UTR5 = 1
REGION_CDS_PHASE0 = 2
REGION_CDS_PHASE1 = 3
REGION_CDS_PHASE2 = 4
REGION_UTR3 = 5
REGION_NCRNA_EXON = 6
REGION_NUM_CLASSES = 7
REGION_IGNORE = -1

EXON_FIRST = 0
EXON_INTERNAL = 1
EXON_LAST = 2
EXON_NUM_CLASSES = 3
EXON_IGNORE = -1

USAGE_NUM_TISSUES = 4
"""Number of tissues with quantified splice-site usage (Pangolin SSU)."""

USAGE_IGNORE = -1.0
"""Usage label for positions excluded from the usage loss (y_true < 0)."""


def _parse_csv_ints(s: str) -> list[int]:
    """Parse a comma-separated integer string with optional trailing comma.

    Args:
        s (str): Comma-separated integer string, possibly with a
            trailing comma (e.g. ``"100,200,300,"`` or ``","``).

    Returns:
        list[int]: Parsed integers; empty list when the string
            contains no numeric tokens.
    """
    if not s:
        return []
    return [int(tok) for tok in s.split(",") if tok.strip() != ""]


def build_region_label_array(
    tx_start: int,
    tx_end: int,
    exon_starts_1: list[int],
    exon_ends_1: list[int],
    cds_starts_1: list[int],
    cds_ends_1: list[int],
    cds_frames: list[int],
    strand: str,
) -> np.ndarray:
    """Create a per-base region annotation label array.

    Args:
        tx_start (int): Transcription start position (1-based, inclusive).
        tx_end (int): Transcription end position (1-based, inclusive).
        exon_starts_1 (list[int]): Per-exon start positions, genomic
            ascending, 1-based inclusive.
        exon_ends_1 (list[int]): Per-exon end positions, genomic
            ascending, 1-based inclusive.
        cds_starts_1 (list[int]): Per-CDS-exon start positions, genomic
            ascending, 1-based inclusive.  Empty list means non-coding.
        cds_ends_1 (list[int]): Per-CDS-exon end positions, genomic
            ascending, 1-based inclusive.
        cds_frames (list[int]): GFF3 phase (0/1/2) per CDS exon, aligned
            with ``cds_starts_1`` (genomic ascending order).
        strand (str): ``'+'`` or ``'-'``.

    Returns:
        np.ndarray: int8 array of length ``tx_end - tx_start + 1``
            in genomic coordinates.  Values follow the region label
            encoding documented in this module's docstring.
    """
    tx_len = tx_end - tx_start + 1
    labels = np.full(tx_len, REGION_INTRON, dtype=np.int8)

    is_coding = len(cds_starts_1) > 0
    cds_min = min(cds_starts_1) if is_coding else None
    cds_max = max(cds_ends_1) if is_coding else None

    for ex_s, ex_e in zip(exon_starts_1, exon_ends_1):
        s = max(ex_s, tx_start)
        e = min(ex_e, tx_end)
        if s > e:
            continue
        i0 = s - tx_start
        i1 = e - tx_start + 1
        if not is_coding:
            labels[i0:i1] = REGION_NCRNA_EXON
            continue

        # Default exon bases to UTR; CDS bases overwrite below.
        if strand == "+":
            for pos in range(s, e + 1):
                idx = pos - tx_start
                if pos < cds_min:
                    labels[idx] = REGION_UTR5
                elif pos > cds_max:
                    labels[idx] = REGION_UTR3
                else:
                    labels[idx] = REGION_UTR5
        else:
            for pos in range(s, e + 1):
                idx = pos - tx_start
                if pos < cds_min:
                    labels[idx] = REGION_UTR3
                elif pos > cds_max:
                    labels[idx] = REGION_UTR5
                else:
                    labels[idx] = REGION_UTR3

    if not is_coding:
        return labels

    # CDS phase labels override UTR placeholders inside CDS spans.
    # The label encodes the per-base codon position (0/1/2) in 5'→3'
    # order.  Given GFF3 phase ``p`` (number of bases into the CDS
    # feature where the next codon begins) and 5'→3' offset ``k``
    # within the truncated span, codon_position(k, p) = (k + (3 - p))
    # mod 3 -- i.e. ``p`` bases of the previous exon have already
    # consumed codon positions ``3 - p``..``2``.
    for cs, ce, frame in zip(cds_starts_1, cds_ends_1, cds_frames):
        s = max(cs, tx_start)
        e = min(ce, tx_end)
        if s > e:
            continue
        # Bases of the original CDS feature that fall before the
        # truncated span shift the effective phase forward.
        if strand == "+":
            head_skip = s - cs
            cds_len = e - s + 1
            k_53 = np.arange(cds_len, dtype=np.int64) + head_skip
        else:
            head_skip = ce - e
            cds_len = e - s + 1
            k_genomic = np.arange(cds_len, dtype=np.int64)
            k_53 = ((cds_len - 1) - k_genomic) + head_skip
        phases = (k_53 + (3 - frame)) % 3
        i0 = s - tx_start
        i1 = e - tx_start + 1
        labels[i0:i1] = (REGION_CDS_PHASE0 + phases).astype(np.int8)

    return labels


def build_exon_position_label_array(
    tx_start: int,
    tx_end: int,
    exon_starts_1: list[int],
    exon_ends_1: list[int],
    strand: str,
) -> np.ndarray:
    """Create a per-base exon-position label array.

    Args:
        tx_start (int): Transcription start position (1-based, inclusive).
        tx_end (int): Transcription end position (1-based, inclusive).
        exon_starts_1 (list[int]): Per-exon start positions, genomic
            ascending, 1-based inclusive.
        exon_ends_1 (list[int]): Per-exon end positions, genomic
            ascending, 1-based inclusive.
        strand (str): ``'+'`` or ``'-'``.

    Returns:
        np.ndarray: int8 array of length ``tx_end - tx_start + 1``.
            Intron and single-exon transcripts return ``EXON_IGNORE``.
    """
    tx_len = tx_end - tx_start + 1
    labels = np.full(tx_len, EXON_IGNORE, dtype=np.int8)

    n_exons = len(exon_starts_1)
    if n_exons < 2:
        return labels

    if strand == "-":
        first_idx = n_exons - 1
        last_idx = 0
    else:
        first_idx = 0
        last_idx = n_exons - 1

    for k, (ex_s, ex_e) in enumerate(zip(exon_starts_1, exon_ends_1)):
        s = max(ex_s, tx_start)
        e = min(ex_e, tx_end)
        if s > e:
            continue
        i0 = s - tx_start
        i1 = e - tx_start + 1
        if k == first_idx:
            labels[i0:i1] = EXON_FIRST
        elif k == last_idx:
            labels[i0:i1] = EXON_LAST
        else:
            labels[i0:i1] = EXON_INTERNAL

    return labels


def ceil_div(x: int, y: int) -> int:
    """Ceiling integer division.

    Args:
        x (int): Numerator.
        y (int): Denominator.

    Returns:
        int: Ceiling of x / y.
    """
    return int(ceil(float(x) / y))


def build_usage_label_array(
    tx_start: int,
    tx_end: int,
    strand: str,
    usage_by_pos: dict[int, np.ndarray],
    num_tissues: int = USAGE_NUM_TISSUES,
) -> np.ndarray:
    """Create a per-base splice-site usage label array for a transcript.

    Places each measured splice-site usage vector at its genomic position
    and leaves every other base masked with ``USAGE_IGNORE``. The array is
    returned in transcript 5'->3' order (genomic order reversed on the minus
    strand) so it aligns with the sequence and splice labels produced by
    :func:`create_datapoints` before windowing.

    Args:
        tx_start (int): Transcription start position (1-based, inclusive).
        tx_end (int): Transcription end position (1-based, inclusive).
        strand (str): Strand of the gene, ``'+'`` or ``'-'``.
        usage_by_pos (dict[int, np.ndarray]): Maps a 1-based genomic position
            to its (num_tissues,) usage vector (e.g. from the Pangolin SSU
            table). Positions outside ``[tx_start, tx_end]`` are ignored.
        num_tissues (int): Number of usage channels. Defaults to
            ``USAGE_NUM_TISSUES`` (4).

    Returns:
        np.ndarray: Float32 array of shape ``(tx_end - tx_start + 1,
            num_tissues)``; masked positions hold ``USAGE_IGNORE``.
    """
    tx_len = tx_end - tx_start + 1
    usage = np.full((tx_len, num_tissues), USAGE_IGNORE, dtype=np.float32)

    for pos, vec in usage_by_pos.items():
        if tx_start <= pos <= tx_end:
            usage[pos - tx_start] = vec

    if strand == "-":
        usage = usage[::-1].copy()

    return usage


def window_genomic_array(
    arr: np.ndarray,
    cl_max: int,
    sl: int,
    pad_value: float,
    stride: int | None = None,
) -> np.ndarray:
    """Slice a transcript-length label array into fixed-size target blocks.

    Uses the same block geometry as :func:`reformat_data` (output blocks of
    length ``sl`` stepped by ``stride``), so the returned windows align
    position-for-position with that function's ``Yd``. Supports an optional
    trailing channel axis for multi-channel labels such as tissue usage.

    Args:
        arr (np.ndarray): Transcript-ordered label array of shape ``(T,)`` or
            ``(T, C)``.
        cl_max (int): Maximum context length (unused for output sizing; kept
            for signature parity with :func:`reformat_data`).
        sl (int): Output block length.
        pad_value (float): Value used to pad the trailing block.
        stride (int | None): Step between consecutive block starts. Defaults
            to ``sl`` (non-overlapping). Must satisfy ``1 <= stride <= sl``.

    Returns:
        np.ndarray: Array of shape ``(num_points, sl)`` or
            ``(num_points, sl, C)`` matching the dtype of ``arr``.

    Raises:
        ValueError: If ``stride`` is outside ``[1, sl]``.
    """
    if stride is None:
        stride = sl
    if not (1 <= stride <= sl):
        raise ValueError(f"stride must be in [1, sl={sl}], got {stride}")

    tx_len = arr.shape[0]
    num_points = 0 if tx_len == 0 else ceil_div(max(tx_len - sl, 0), stride) + 1

    pad_width = [(0, sl)] + [(0, 0)] * (arr.ndim - 1)
    padded = np.pad(arr, pad_width, "constant", constant_values=pad_value)

    out_shape = (num_points, sl) + arr.shape[1:]
    windows = np.full(out_shape, pad_value, dtype=arr.dtype)
    for i in range(num_points):
        start = stride * i
        windows[i] = padded[start : start + sl]

    return windows


def create_datapoints(
    seq: str,
    strand: str,
    tx_start: str,
    tx_end: str,
    jn_start: list[str],
    jn_end: list[str],
    cl_max: int,
    sl: int,
    exon_start: str | None = None,
    exon_end: str | None = None,
    cds_start: str | None = None,
    cds_end: str | None = None,
    cds_frame: str | None = None,
    stride: int | None = None,
    flank_fill: str = "N",
    extra_flank: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert gene annotation into integer-encoded datapoints.

    Encodes DNA sequence to integers (A=1,C=2,G=3,T=4,N=0),
    handles reverse complement for minus strand, creates splice
    junction labels, region-annotation labels, and exon-position
    labels, then reformats them into fixed-size blocks.

    Args:
        seq (str): DNA sequence (length = tx_len + cl_max + 1).
        strand (str): Strand direction ('+' or '-').
        tx_start (str): Transcription start position (1-based).
        tx_end (str): Transcription end position (1-based).
        jn_start (list[str]): Comma-separated exon-end positions
            per annotation set (the 0th entry is consumed).
        jn_end (list[str]): Comma-separated exon-start positions
            per annotation set (the 0th entry is consumed).
        cl_max (int): Maximum context length.
        sl (int): Sequence length per output block.
        exon_start (str | None): Comma-separated per-exon start
            positions (1-based, genomic asc, trailing comma).
            None or empty disables region/exon label generation
            and the corresponding outputs are filled with the
            ignore index.
        exon_end (str | None): Comma-separated per-exon end
            positions (1-based, genomic asc, trailing comma).
        cds_start (str | None): Comma-separated per-CDS-exon
            starts (1-based, genomic asc).  Empty/None marks
            the transcript as non-coding.
        cds_end (str | None): Comma-separated per-CDS-exon ends.
        cds_frame (str | None): Comma-separated per-CDS-exon
            phases (0/1/2), aligned with cds_start.
        stride (int | None): Step between consecutive output
            block starts; ``None`` defaults to ``sl``
            (non-overlapping blocks).  Use ``stride < sl`` to emit
            overlapping training windows that cover every
            transcript position at multiple within-target offsets.
        flank_fill (str): ``"N"`` replaces the ``cl_max // 2`` flanking
            bases on each side of the transcript with N (the SpliceAI
            convention); ``"real"`` keeps the genome sequence, matching
            inference on the genome FASTA.
        extra_flank (int): Bases ``seq`` carries beyond the
            ``cl_max // 2`` flank on each side. The transcript-downstream
            ones replace the zero padding at the tail of the last block
            with genome sequence; the upstream ones are dropped. Requires
            ``flank_fill="real"``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X of shape (num_points, SL+CL_max) int8
              (0=N,1=A,2=C,3=G,4=T),
            Y of shape (num_points, SL) int8
              (0=neither,1=acceptor,2=donor,-1=pad),
            Y_region of shape (num_points, SL) int8 with values
              defined in this module's docstring,
            Y_exon of shape (num_points, SL) int8 with values
              defined in this module's docstring.
    """
    half_cl = cl_max // 2
    if flank_fill not in ("N", "real"):
        raise ValueError(f"flank_fill must be 'N' or 'real', got {flank_fill!r}")
    if extra_flank:
        if flank_fill != "real":
            raise ValueError("extra_flank requires flank_fill='real'")
        # Keep the transcript-downstream extension only: genomic right on '+',
        # genomic left on '-' (the reverse complement below moves it to the
        # tail).
        seq = seq[extra_flank:] if strand == "+" else seq[:-extra_flank]
    if flank_fill == "N":
        seq = "N" * half_cl + seq[half_cl:-half_cl] + "N" * half_cl

    seq = re.sub(r"[^ACGTN]", "N", seq.upper())
    seq_encoded = np.fromiter(
        (SEQ_ENCODE[b] for b in seq),
        dtype=np.int8,
        count=len(seq),
    )

    tx_s = int(tx_start)
    tx_e = int(tx_end)
    tx_len = tx_e - tx_s + 1

    jn_s = [list(map(int, re.split(",", x)[:-1])) for x in jn_start]
    jn_e = [list(map(int, re.split(",", x)[:-1])) for x in jn_end]

    has_region_inputs = (
        exon_start is not None
        and exon_end is not None
        and _parse_csv_ints(exon_start)
    )

    if has_region_inputs:
        exon_starts_1 = _parse_csv_ints(exon_start)
        exon_ends_1 = _parse_csv_ints(exon_end)
        cds_starts_1 = _parse_csv_ints(cds_start) if cds_start else []
        cds_ends_1 = _parse_csv_ints(cds_end) if cds_end else []
        cds_frames = _parse_csv_ints(cds_frame) if cds_frame else []

        Y_region_genomic = build_region_label_array(
            tx_s,
            tx_e,
            exon_starts_1,
            exon_ends_1,
            cds_starts_1,
            cds_ends_1,
            cds_frames,
            strand,
        )
        Y_exon_genomic = build_exon_position_label_array(
            tx_s,
            tx_e,
            exon_starts_1,
            exon_ends_1,
            strand,
        )
    else:
        Y_region_genomic = np.full(tx_len, REGION_IGNORE, dtype=np.int8)
        Y_exon_genomic = np.full(tx_len, EXON_IGNORE, dtype=np.int8)

    if strand == "+":
        X0 = seq_encoded
        Y0 = -np.ones(tx_len)

        if len(jn_s[0]) > 0:
            Y0 = np.zeros(tx_len)
            for c in jn_s[0]:
                if tx_s <= c <= tx_e:
                    Y0[c - tx_s] = 2
            for c in jn_e[0]:
                if tx_s <= c <= tx_e:
                    Y0[c - tx_s] = 1

        Y_region = Y_region_genomic
        Y_exon = Y_exon_genomic

    elif strand == "-":
        X0 = (5 - seq_encoded[::-1]) % 5
        Y0 = -np.ones(tx_len)

        if len(jn_s[0]) > 0:
            Y0 = np.zeros(tx_len)
            for c in jn_e[0]:
                if tx_s <= c <= tx_e:
                    Y0[tx_e - c] = 2
            for c in jn_s[0]:
                if tx_s <= c <= tx_e:
                    Y0[tx_e - c] = 1

        # Reverse genomic-order labels into transcript 5'->3' order
        # so they align with the reverse-complemented X0.
        Y_region = Y_region_genomic[::-1].copy()
        Y_exon = Y_exon_genomic[::-1].copy()

    Xd, Yd, Y_region_d, Y_exon_d = reformat_data(
        X0,
        Y0,
        cl_max,
        sl,
        Y_region0=Y_region,
        Y_exon0=Y_exon,
        stride=stride,
    )

    return (
        Xd.astype("int8"),
        Yd.astype("int8"),
        Y_region_d.astype("int8"),
        Y_exon_d.astype("int8"),
    )


def reformat_data(
    X0: np.ndarray,
    Y0: np.ndarray,
    cl_max: int,
    sl: int,
    Y_region0: np.ndarray | None = None,
    Y_exon0: np.ndarray | None = None,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reformat data into fixed-size blocks.

    Breaks sequence and labels into blocks where each input block
    has length ``SL + CL_max`` and each output block has length
    ``SL``.  ``CL_max / 2`` context nucleotides are provided on
    each side of the ``SL`` region of interest.  All label arrays
    are sliced identically and padded with their ignore index at
    the trailing block.

    When ``stride < sl`` consecutive blocks overlap by
    ``sl - stride`` positions, so each genomic position appears in
    multiple training windows at different within-target offsets.
    This is required when the variant-centred ``2W``-bp window
    around any position must lie entirely inside *some* training
    target (covered when ``stride <= sl - 2W``).

    Args:
        X0 (np.ndarray): Integer-encoded full sequence.
        Y0 (np.ndarray): Splice label array for the transcript.
        cl_max (int): Maximum context length.
        sl (int): Sequence length per block.
        Y_region0 (np.ndarray | None): Region annotation label
            array.  Defaults to a full-ignore array of length
            ``len(Y0)``.
        Y_exon0 (np.ndarray | None): Exon-position label array.
            Defaults to a full-ignore array of length ``len(Y0)``.
        stride (int | None): Step between consecutive block
            starts.  Defaults to ``sl`` (non-overlapping blocks).
            Must satisfy ``1 <= stride <= sl``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Xd of shape (num_points, SL+CL_max),
            Yd of shape (num_points, SL),
            Y_region_d of shape (num_points, SL),
            Y_exon_d of shape (num_points, SL).

    Raises:
        ValueError: If ``stride`` is outside ``[1, sl]``.
    """
    if stride is None:
        stride = sl
    if not (1 <= stride <= sl):
        raise ValueError(
            f"stride must be in [1, sl={sl}], got {stride}"
        )

    if Y_region0 is None:
        Y_region0 = np.full(len(Y0), REGION_IGNORE, dtype=np.int8)
    if Y_exon0 is None:
        Y_exon0 = np.full(len(Y0), EXON_IGNORE, dtype=np.int8)

    tx_len = len(Y0)
    if tx_len == 0:
        num_points = 0
    else:
        num_points = ceil_div(max(tx_len - sl, 0), stride) + 1

    Xd = np.zeros((num_points, sl + cl_max))
    Yd = -np.ones((num_points, sl))
    Y_region_d = np.full((num_points, sl), REGION_IGNORE, dtype=np.int8)
    Y_exon_d = np.full((num_points, sl), EXON_IGNORE, dtype=np.int8)

    # Pad by `sl` so the final block's slices never read past the
    # array end. The maximum required tail padding is
    # ``stride * (num_points - 1) + sl - tx_len`` which is bounded
    # above by ``stride - 1 <= sl - 1``, so ``sl`` is always safe.
    X0 = np.pad(X0, [0, sl], "constant", constant_values=0)
    Y0 = np.pad(Y0, [0, sl], "constant", constant_values=-1)
    Y_region0 = np.pad(
        Y_region0, [0, sl], "constant", constant_values=REGION_IGNORE
    )
    Y_exon0 = np.pad(
        Y_exon0, [0, sl], "constant", constant_values=EXON_IGNORE
    )

    for i in range(num_points):
        start = stride * i
        Xd[i] = X0[start : start + sl + cl_max]
        Yd[i] = Y0[start : start + sl]
        Y_region_d[i] = Y_region0[start : start + sl]
        Y_exon_d[i] = Y_exon0[start : start + sl]

    return Xd, Yd, Y_region_d, Y_exon_d


def clip_datapoints(
    X: np.ndarray,
    Y: np.ndarray,
    cl: int,
    cl_max: int,
    n_gpus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip datapoints for context length and GPU alignment.

    Ensures the number of datapoints is a multiple of n_gpus
    and clips input to SL+CL when CL < CL_max.

    Args:
        X (np.ndarray): Input sequences,
            shape (N, SL+CL_max).
        Y (np.ndarray): Labels, shape (N, SL).
        cl (int): Desired context length.
        cl_max (int): Maximum context length used in data.
        n_gpus (int): Number of GPUs for alignment.

    Returns:
        tuple[np.ndarray, np.ndarray]: Clipped (X, Y).
    """
    rem = X.shape[0] % n_gpus
    clip = (cl_max - cl) // 2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], Y[:-rem]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], Y
    elif rem != 0 and clip == 0:
        return X[:-rem], Y[:-rem]
    else:
        return X, Y
