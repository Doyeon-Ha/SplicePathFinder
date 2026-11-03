"""Multi-chunk SpliceVarDB SAV evaluation (TTA across shifted windows).

For each variant, generate ``K`` stride-aligned 15000-bp windows whose
chunk-local variant positions are ``{clp_0, clp_0 + stride, clp_0 +
2*stride, ...}``, with ``clp_0 = variant_offset % stride`` taken from
the input H5's ``variant_offset`` field. This guarantees that one of
the K windows is the original gencode-anchored chunk that gave the
canonical single-chunk baseline, and the others are stride-shifted
versions on the same offset grid the model saw during training.

For each chunk we run the model on ref/alt, compute per-bp 4-direction
deltas (acc_gain, acc_loss, don_gain, don_loss), and aggregate across
chunks via ``max`` at each chunk-local position within the variant's
+-``delta_window``. The variant score is ``max`` over the window.

Two modes:

- ``conservative`` — only use bp scores from each chunk's target region
  (the supervised window of the trained model).
- ``liberal`` — use bp scores from all 15000 bp output (including
  untrained flank positions).

The script also reports a ``baseline`` score per variant that uses
**only** the chunk with ``clp == variant_offset`` — i.e. the original
gencode-anchored chunk. By the max-aggregation property the multichunk
score is always >= baseline; the gap measures the TTA contribution.

Defaults (2026-08-13)
---------------------
``--stride 1250`` and ``--chunk-selection target`` (K=4) are the project
protocol and are now the argparse defaults. Both used to default to something
else and both caused incidents:

* stride defaulted to 2500 (K=6), which samples half the training grid, so a
  run that inherited it produced numbers comparable to nothing in the registry.
* chunk-selection defaulted to ``all`` (K=12), forwarding the eight chunks that
  place the variant in the unsupervised flank.

K=4 keeps only the chunks whose supervised 5,000 nt target contains the
variant. Those four jointly cover the whole +-2,500 bp delta window, so no
scored position is lost, and it is 3x cheaper. Registry rows scored this way
take ``scoring=v4_aligned_k4``; the pre-existing ``v4_aligned`` rows are K=12
and the two must not be mixed in one comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.evaluate_splicevardb_sav import (  # noqa: E402
    _FIXED_BATCH_SIZE,
    _hyenadna_forward_probs,
    _load_hyenadna_model,
    _resolve_probability_mode,
)
from model.hybrid_splice import HybridSpliceModel, ModelConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_site_track import (  # noqa: E402
    ANNOTATION,
    annotation_probs,
    apply_canonical_mask,
    load_canonical_sites,
    load_sites_by_accession,
    load_variant_transcripts,
    resolve_sites,
    site_indices,
)

logger = logging.getLogger(__name__)

# torch sizes its intra-op pool from the machine's core count, not the cgroup
# Slurm gave us, so several concurrent tasks oversubscribe the node badly.
# Honour the allocation when it is known.
_slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if _slurm_cpus:
    torch.set_num_threads(int(_slurm_cpus))

_NT_TO_INT: dict[str, int] = {
    "N": 0, "A": 1, "C": 2, "G": 3, "T": 4,
    "n": 0, "a": 1, "c": 2, "g": 3, "t": 4,
}
_INT_TO_NT_STR: str = "NACGT"
_COMPLEMENT: dict[str, str] = {
    "A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
    "a": "t", "t": "a", "c": "g", "g": "c", "n": "n",
}

_FEATURE_KEYS: tuple[str, ...] = (
    "delta_max",
    "delta_acc_max",
    "delta_don_max",
    "delta_acc_gain",
    "delta_acc_loss",
    "delta_don_gain",
    "delta_don_loss",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed args.
    """
    p = argparse.ArgumentParser(
        description="Multi-chunk SAV evaluation (shifted-window TTA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument(
        "--model-type",
        choices=("hyenadna", "hybrid_splice", "dlm_qwen"),
        default="hyenadna",
        help=(
            "Checkpoint flavour. 'hyenadna' (default) is the project's "
            "backbone family and needs --train-config; 'hybrid_splice' is the "
            "CPU-servable dilated-CNN + Transformer model, which carries its "
            "own config inside the checkpoint; 'dlm_qwen' is the frozen "
            "DNA-language-model -> Qwen stack, which carries its constructor "
            "arguments inside the checkpoint. Everything downstream -- chunk "
            "grid, alignment, delta_max aggregation, modes -- is shared, so "
            "all flavours are scored by an identical methodology."
        ),
    )
    p.add_argument(
        "--variant-h5",
        required=True,
        type=str,
        help="Gencode-anchored SpliceVarDB H5 with chrom/pos/ref/alt/"
             "strand/label/variant_offset (the canonical eval H5).",
    )
    p.add_argument(
        "--genome-fasta",
        type=str,
        default="./genome/GRCh38.fa",
    )
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--delta-window", type=int, default=2500)
    p.add_argument(
        "--extra-mass-windows", type=int, nargs="*", default=[],
        help="Additional delta_mass window half-widths (bp) to emit alongside "
             "the default; additive, does not change existing outputs. Saved as "
             "multichunk_delta_mass_w<W> in the npz.",
    )
    p.add_argument("--stride", type=int, default=1250)
    p.add_argument(
        "--chunk-selection", choices=["all", "target"], default="target",
        dest="chunk_selection",
        help="all: every stride-aligned position across the 15kbp chunk "
             "(K=12 at stride 1250). target: only positions placing the "
             "variant inside the supervised target region (K=4), which cuts "
             "the forward cost 3x. The four target-region chunks jointly "
             "cover the whole +-delta_window, so coverage is preserved.",
    )
    p.add_argument("--total-length", type=int, default=15000)
    p.add_argument(
        "--train-config", type=str, default=None,
        help="Optional explicit training config YAML.",
    )
    p.add_argument(
        "--hyena-filter-max-seq-len", type=int, default=None,
        help="Evaluation-only Hyena implicit-filter position-grid override. "
             "The biological input, tokenizer, target, and chunk grid retain "
             "their checkpoint geometry; no nucleotide padding is added.",
    )
    p.add_argument(
        "--probability-mode", type=str,
        choices=["softmax", "sigmoid", "auto"], default="auto",
    )
    p.add_argument(
        "--ref-source", choices=["model", "annotation"], default="model",
        help="What stands in for the reference readout. 'model' (default) is "
             "the usual model(ref) pass. 'annotation' replaces it with a 0/1 "
             "GENCODE v49 canonical splice-site track, so the score becomes "
             "model(alt) against the annotation instead of against the "
             "model's own reference prediction.",
    )
    p.add_argument(
        "--mask-canonical", action=argparse.BooleanOptionalAction, default=False,
        help="Per channel, allow a loss only where the reference annotates a "
             "site of that type and a gain only where it does not. An "
             "annotated acceptor can still carry a donor gain. Enabled only "
             "when explicitly requested; use the masked scoring tag. "
             "A no-op under --ref-source annotation, "
             "where the track already implies the rule.",
    )
    p.add_argument(
        "--mask-transcript-boundaries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace sequence positions outside the anchoring transcript's "
             "[TX_START, TX_END] with N before the forward pass. This "
             "reproduces the training H5 input convention; the default keeps "
             "the current genome-FASTA inference convention.",
    )
    p.add_argument(
        "--annotation-path", default=ANNOTATION,
        help="GENCODE v49 canonical annotation table (SpliceAI 7-column TSV).",
    )
    p.add_argument(
        "--row-start", type=int, default=0,
        help="Score only SNVs [row-start, row-end) of the table, so a very "
             "large universe can be walked in bounded pieces on one GPU.",
    )
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument(
        "--transcript-annotation", default=None,
        help="Accession-keyed annotation (NAME = RefSeq accession). With "
             "--variant-transcript-csv, the canonical mask follows the "
             "transcript each variant was interpreted against instead of the "
             "gene's representative one.",
    )
    p.add_argument(
        "--variant-transcript-csv", default=None,
        help="CSV with cpra_grch38,transcript columns.",
    )
    p.add_argument(
        "--target-region-length", type=int, default=None,
        help="Override supervised target length. Default uses the value"
             " from the loaded model's train config.",
    )
    p.add_argument(
        "--anchor-fallback", choices=["error", "center"], default="error",
        dest="anchor_fallback",
        help="What to do with variants that carry no gencode anchor. 'error' "
             "(default) refuses, since the chunk grid is what makes the TTA "
             "on-distribution. 'center' scores them on a variant-centred "
             "chunk (offset=total_length//2, left-biased on the even split) "
             "and records the count in the summary, so a mixed run is "
             "visible rather than silent.",
    )
    p.add_argument(
        "--block-size", type=int, default=2000,
        help="Variants scored per persisted block. Each block writes its own "
             "npz under <output-dir>/_partial as it finishes, so a timeout "
             "costs one block and a rerun resumes from the last completed "
             "one. Also caps peak RSS: a whole 17k-SNV shard holds ~23 GB of "
             "probabilities at once, one block holds ~2.5 GB.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the inherited fixed eval batch size. Lower to"
             " ease GPU pressure when sharing the device with training.",
    )
    return p.parse_args()


def _reverse_complement(seq: str) -> str:
    """Return the reverse-complement of ``seq`` (Ns preserved).

    Args:
        seq (str): DNA string.

    Returns:
        str: RC of ``seq``.
    """
    return "".join(_COMPLEMENT.get(b, "N") for b in seq[::-1])


def _encode_seq(seq: str) -> np.ndarray:
    """Encode ``seq`` to int8 with ``N=0, A=1, C=2, G=3, T=4``.

    Args:
        seq (str): DNA string.

    Returns:
        np.ndarray: int8 array of length ``len(seq)``.
    """
    return np.array(
        [_NT_TO_INT.get(b, 0) for b in seq], dtype=np.int8,
    )


def _decode_seq(arr: np.ndarray) -> str:
    """Decode an int8 nucleotide array back to a DNA string.

    Args:
        arr (np.ndarray): int8 array with values in ``{0,1,2,3,4}``.

    Returns:
        str: DNA string.
    """
    arr_clipped = np.clip(arr.astype(np.int64), 0, 4)
    return "".join(_INT_TO_NT_STR[i] for i in arr_clipped.tolist())


def _slice_window(
    fasta: Fasta,
    chrom: str,
    start_0based: int,
    end_0based: int,
) -> str:
    """Return a padded slice of ``chrom`` from FASTA.

    Args:
        fasta (Fasta): pyfaidx FASTA handle.
        chrom (str): chromosome key (FASTA naming).
        start_0based (int): inclusive start.
        end_0based (int): exclusive end.

    Returns:
        str: uppercase DNA string of length ``end - start``, padded with
            ``N`` at out-of-bound edges.
    """
    chrom_len = len(fasta[chrom])
    left_pad = max(0, -start_0based)
    right_pad = max(0, end_0based - chrom_len)
    slice_start = max(0, start_0based)
    slice_end = min(chrom_len, end_0based)
    body = str(fasta[chrom][slice_start:slice_end]).upper()
    return "N" * left_pad + body + "N" * right_pad


def _resolve_fasta_chrom(fasta: Fasta, chrom: str) -> str | None:
    """Return the FASTA's chromosome key matching ``chrom``.

    Args:
        fasta (Fasta): pyfaidx FASTA handle.
        chrom (str): chromosome string from the variant table.

    Returns:
        str | None: matching key or ``None`` if not found.
    """
    if chrom in fasta:
        return chrom
    bare = chrom[3:] if chrom.startswith("chr") else chrom
    if bare in fasta:
        return bare
    if f"chr{bare}" in fasta:
        return f"chr{bare}"
    return None


def _generate_chunks_for_variant(
    fasta: Fasta,
    chrom: str,
    pos_hg38: int,
    ref: str,
    alt: str,
    strand: str,
    total_length: int,
    chunk_local_positions: list[int],
) -> list[dict]:
    """Generate per-chunk ref/alt encoded sequences for one variant.

    Args:
        fasta (Fasta): pyfaidx FASTA handle.
        chrom (str): variant chromosome (e.g. ``chr1``).
        pos_hg38 (int): 1-based variant position on the + strand of hg38.
        ref (str): reference allele (single base for SNV).
        alt (str): alternate allele.
        strand (str): ``+`` or ``-`` (gene orientation).
        total_length (int): chunk length.
        chunk_local_positions (list[int]): forward-strand variant
            positions inside each chunk.

    Returns:
        list[dict]: per-chunk records with keys ``clp_model`` (variant's
            chunk-local position as seen by the model after RC),
            ``x_ref``, ``x_alt`` (int8 arrays), and a ``status`` string.
    """
    fa_chrom = _resolve_fasta_chrom(fasta, chrom)
    if fa_chrom is None:
        return [
            {"clp_model": clp, "x_ref": None, "x_alt": None,
             "status": f"chrom_not_found:{chrom}"}
            for clp in chunk_local_positions
        ]

    records: list[dict] = []
    for clp in chunk_local_positions:
        chunk_start_0b = (pos_hg38 - 1) - clp
        chunk_end_0b = chunk_start_0b + total_length
        ref_seq = _slice_window(fasta, fa_chrom, chunk_start_0b, chunk_end_0b)
        if len(ref_seq) != total_length:
            records.append(
                {"clp_model": clp, "x_ref": None, "x_alt": None,
                 "status": f"length_mismatch:{len(ref_seq)}"}
            )
            continue
        ref_at_var = ref_seq[clp]
        if ref_at_var != ref.upper():
            records.append(
                {"clp_model": clp, "x_ref": None, "x_alt": None,
                 "status": f"ref_mismatch:{ref_at_var}!={ref}"}
            )
            continue
        alt_seq = ref_seq[:clp] + alt.upper() + ref_seq[clp + len(ref):]
        if len(alt_seq) != total_length:
            records.append(
                {"clp_model": clp, "x_ref": None, "x_alt": None,
                 "status": f"alt_length_mismatch:{len(alt_seq)}"}
            )
            continue
        if strand == "-":
            ref_final = _reverse_complement(ref_seq)
            alt_final = _reverse_complement(alt_seq)
            clp_model = total_length - 1 - clp
        else:
            ref_final = ref_seq
            alt_final = alt_seq
            clp_model = clp
        records.append({
            "clp_model": clp_model,
            "x_ref": _encode_seq(ref_final),
            "x_alt": _encode_seq(alt_final),
            "status": "ok",
        })
    return records


def _mask_sequence_outside_transcript(
    seq: np.ndarray,
    *,
    pos_hg38: int,
    clp_model: int,
    strand: str,
    tx_start: int,
    tx_end: int,
) -> tuple[np.ndarray, int]:
    """Replace model-space positions outside a transcript with N tokens.

    Args:
        seq (np.ndarray): Encoded sequence in model (gene-reading) orientation.
        pos_hg38 (int): 1-based genomic variant position.
        clp_model (int): Variant index in ``seq`` after reverse complementing.
        strand (str): Anchoring transcript strand.
        tx_start (int): 0-based boundary before the first transcript base.
        tx_end (int): 1-based inclusive transcript end.

    Returns:
        tuple[np.ndarray, int]: Masked copy and number of positions replaced.
    """
    j = np.arange(len(seq), dtype=np.int64)
    if strand == "-":
        genomic = int(pos_hg38) - (j - int(clp_model))
    else:
        genomic = int(pos_hg38) + (j - int(clp_model))
    # The SpliceAI-style annotation stores TX_START as a 0-based boundary and
    # TX_END as a 1-based inclusive coordinate. The training H5 therefore
    # masks genomic coordinate == TX_START as well (byte-exactly verified
    # against X_ref), while retaining coordinate == TX_END.
    outside = (genomic <= int(tx_start)) | (genomic > int(tx_end))
    out = np.asarray(seq, dtype=np.int8).copy()
    out[outside] = 0
    return out, int(outside.sum())


def _load_variant_table(h5_path: str) -> list[dict]:
    """Read variant records from a gencode-anchored SpliceVarDB H5.

    Args:
        h5_path (str): path to the H5 file.

    Returns:
        list[dict]: one entry per variant with ``chrom``, ``pos``,
            ``ref``, ``alt``, ``strand``, ``label``, and (optional)
            ``variant_offset`` keys.

    Raises:
        KeyError: if required fields are missing.
    """
    def _decode(x: object) -> str:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

    with h5py.File(h5_path, "r") as f:
        chrom = f["chrom"][:]
        pos = f["pos"][:].astype(np.int64)
        ref = f["ref"][:]
        alt = f["alt"][:]
        strand = f["strand"][:]
        # Prefer the canonical-anchor strand (set by the gencode_anchored
        # builder) because that is the orientation the model was trained
        # on. The raw ``strand`` field carries the SpliceVarDB metadata
        # which can disagree with the canonical annotation (e.g. RHD
        # variants where SpliceVarDB says ``-`` but canonical v49 says
        # ``+``). Using the wrong strand reverse-complements the chunk
        # away from training distribution.
        source_strand = (
            f["source_strand"][:] if "source_strand" in f else None
        )
        label = f["label"][:].astype(np.int64)
        variant_offset = (
            f["variant_offset"][:].astype(np.int64)
            if "variant_offset" in f else None
        )
        # Anchored gene name → used to look up canonical splice sites for the
        # optional nt-embedding feature track (intron structure / conservation).
        source_gene_name = (
            f["source_gene_name"][:] if "source_gene_name" in f else None
        )
        # Fine-grained location stratum (e.g. exonic, acceptor_deep_intronic);
        # used to optionally restrict the eval to specific strata.
        gene_region = f["gene_region"][:] if "gene_region" in f else None
    records: list[dict] = []
    for i in range(len(pos)):
        if source_strand is not None:
            strand_to_use = _decode(source_strand[i])
        else:
            strand_to_use = _decode(strand[i])
        rec = {
            "idx": i,
            "chrom": _decode(chrom[i]),
            "pos": int(pos[i]),
            "ref": _decode(ref[i]).upper(),
            "alt": _decode(alt[i]).upper(),
            "strand": strand_to_use,
            "label": int(label[i]),
        }
        if variant_offset is not None:
            rec["variant_offset"] = int(variant_offset[i])
        if source_gene_name is not None:
            rec["source_gene_name"] = _decode(source_gene_name[i])
        if gene_region is not None:
            rec["gene_region"] = _decode(gene_region[i])
        records.append(rec)
    return records


def _aligned_chunk_positions(
    variant_offset: int,
    total_length: int,
    stride: int,
) -> list[int]:
    """Return K stride-aligned variant chunk-local positions in model space.

    Builds the residue class ``{clp_0, clp_0 + stride, ...}`` where
    ``clp_0 = variant_offset % stride``, truncated to ``[0, total_length)``.
    One element equals ``variant_offset`` exactly — the original
    gencode-anchored chunk.

    Args:
        variant_offset (int): canonical chunk-local variant position
            in model space (from the gencode-anchored H5).
        total_length (int): chunk length.
        stride (int): stride between adjacent chunks (1250 = the
            training grid; see the module docstring).

    Returns:
        list[int]: K positions in ascending order.
    """
    clp_0 = int(variant_offset) % int(stride)
    positions: list[int] = []
    cp = clp_0
    while cp < total_length:
        positions.append(cp)
        cp += stride
    return positions


def _compute_per_bp_deltas(
    p_ref: np.ndarray, p_alt: np.ndarray,
) -> np.ndarray:
    """Compute the four per-position directional delta tracks.

    Args:
        p_ref (np.ndarray): ref probs ``(L, 3)`` channel order
            ``(acc, don, nei)``.
        p_alt (np.ndarray): alt probs ``(L, 3)``.

    Returns:
        np.ndarray: ``(L, 4)`` non-negative tracks in order
            ``(acc_gain, acc_loss, don_gain, don_loss)``.
    """
    diff = p_alt - p_ref  # (L, 3)
    acc_gain = np.maximum(diff[:, 0], 0.0)
    acc_loss = np.maximum(-diff[:, 0], 0.0)
    don_gain = np.maximum(diff[:, 1], 0.0)
    don_loss = np.maximum(-diff[:, 1], 0.0)
    return np.stack([acc_gain, acc_loss, don_gain, don_loss], axis=-1)


# A direction whose delta track is zero at every scored position has no peak
# to report. Kept far outside any real offset so it cannot be read as one.
_NO_PEAK: int = -2_000_000_000

_OFFSET_KEYS: tuple[str, ...] = (
    "offset_acc_gain",
    "offset_acc_loss",
    "offset_don_gain",
    "offset_don_loss",
)


def _EMPTY_SCORES(mass_windows: tuple[int, ...]) -> dict[str, float]:
    """Scores for a variant with no usable chunk: zero everywhere, no peak.

    Args:
        mass_windows (tuple[int, ...]): extra delta_mass half-widths.

    Returns:
        dict[str, float]: the feature, mass and offset keys at their nulls.
    """
    return {
        **{k: 0.0 for k in _FEATURE_KEYS}, "delta_mass": 0.0,
        **{f"delta_mass_w{w}": 0.0 for w in mass_windows},
        **{k: _NO_PEAK for k in _OFFSET_KEYS}, "offset_delta_max": _NO_PEAK,
    }


def _aggregate_variant_features(
    per_chunk_tracks: list[np.ndarray],
    chunk_local_var_positions: list[int],
    total_length: int,
    target_start: int,
    target_end: int,
    delta_window: int,
    mode: str,
    mass_windows: tuple[int, ...] = (),
) -> dict[str, float]:
    """Compute variant-level scores by max across chunks/positions.

    For each feature, score = max over (chunk, valid_position,
    relevant_channel-direction) of the per-bp delta value, where:

    - ``valid_position`` means ``offset = chunk_local_pos -
      chunk_local_var_pos`` is in ``[-delta_window, delta_window]``,
      AND in ``conservative`` mode that ``chunk_local_pos`` is also in
      the supervised target region ``[target_start, target_end)``.
    - ``relevant_channel-direction`` depends on the feature:
      ``delta_max`` uses all four, per-channel ``_max`` uses both
      directions in that channel, and direction-specific features
      use just one of the four tracks.

    Args:
        per_chunk_tracks (list[np.ndarray]): per chunk ``(L, 4)``
            non-negative track arrays.
        chunk_local_var_positions (list[int]): variant position in
            chunk-local model coords for each chunk.
        total_length (int): chunk length (e.g. 15000).
        target_start (int): inclusive supervised target start.
        target_end (int): exclusive supervised target end.
        delta_window (int): half-width of the score window.
        mode (str): ``conservative`` or ``liberal``.

    Returns:
        dict[str, float]: feature -> scalar score. Returns all-zero if
            no chunk yields a valid window position.
    """
    _mw = tuple(int(w) for w in mass_windows)
    if not per_chunk_tracks:
        return _EMPTY_SCORES(_mw)

    best_per_dir = np.zeros(4, dtype=np.float32)
    # Where each direction peaked, as an offset from the variant in model
    # (gene-reading) coordinates. _NO_PEAK marks a direction whose track is
    # flat zero everywhere, which has no position to report.
    best_off = np.full(4, _NO_PEAK, dtype=np.int32)
    # delta_mass = Σ over the ±delta_window window of (|Δacc| + |Δdon|) per bp,
    # i.e. the L1 mass of the per-bp deltas (all 4 non-negative tracks summed).
    # Across chunks we take the max window-mass (the anchoring that captures the
    # most perturbation), mirroring how delta_max maxes across chunks.
    best_mass = 0.0
    best_mass_w = {w: 0.0 for w in _mw}  # additive: extra delta_mass windows
    saw_any = False

    for tracks, clp_var in zip(per_chunk_tracks, chunk_local_var_positions):
        if mode in ("target_only", "target_strict") and not (
            target_start <= clp_var < target_end
        ):
            # Drop chunks that place the variant in a flank. Every surviving
            # contribution then comes from a position the model was supervised
            # on. Strictly a subset of the `liberal` chunk set, so by the
            # max-aggregation property scores can only be <= liberal.
            continue
        win_start = max(0, clp_var - delta_window)
        win_end = min(total_length, clp_var + delta_window + 1)
        if mode in ("conservative", "target_strict"):
            win_start = max(win_start, target_start)
            win_end = min(win_end, target_end)
        if win_end <= win_start:
            continue
        sliced = tracks[win_start:win_end, :]  # (window_len, 4)
        chunk_max = sliced.max(axis=0)  # (4,)
        # Record the position before the running max absorbs the value.
        improved = chunk_max > best_per_dir
        chunk_off = sliced.argmax(axis=0) + win_start - clp_var
        best_off[improved] = chunk_off[improved]
        best_per_dir = np.maximum(best_per_dir, chunk_max)
        best_mass = max(best_mass, float(sliced.sum()))
        for w in _mw:  # same max-over-chunks mass at each extra window half-width
            ws = max(0, clp_var - w)
            we = min(total_length, clp_var + w + 1)
            if mode == "conservative":
                ws = max(ws, target_start)
                we = min(we, target_end)
            if we > ws:
                best_mass_w[w] = max(best_mass_w[w], float(tracks[ws:we, :].sum()))
        saw_any = True

    if not saw_any:
        return _EMPTY_SCORES(_mw)

    acc_gain, acc_loss, don_gain, don_loss = best_per_dir.tolist()
    return {
        "delta_max": float(max(acc_gain, acc_loss, don_gain, don_loss)),
        "delta_acc_max": float(max(acc_gain, acc_loss)),
        "delta_don_max": float(max(don_gain, don_loss)),
        "delta_acc_gain": float(acc_gain),
        "delta_acc_loss": float(acc_loss),
        "delta_don_gain": float(don_gain),
        "delta_don_loss": float(don_loss),
        "delta_mass": best_mass,
        **{f"delta_mass_w{w}": best_mass_w[w] for w in _mw},
        **{k: int(o) for k, o in zip(_OFFSET_KEYS, best_off.tolist())},
        "offset_delta_max": int(best_off[int(np.argmax(best_per_dir))]),
    }


def _load_hybrid_splice_model(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, int]:
    """Load a ``HybridSpliceModel`` checkpoint for full-length scoring.

    ``output_length`` is forced to ``None`` so the forward returns all
    ``input_length`` positions. The canonical ``liberal`` mode reads the flank,
    exactly as it does for the HyenaDNA models, so cropping to the supervised
    target here would silently change the readout.

    Args:
        checkpoint_path (str): Path to a checkpoint written by
            ``train/train_hybrid_splice.py``.
        device (torch.device): Compute device.

    Returns:
        tuple[torch.nn.Module, int]: The model in eval mode on ``device``, and
        the supervised target length it was TRAINED with (needed for the
        conservative-mode window clamp, and not recoverable afterwards because
        ``output_length`` is overridden here).
    """
    from dataclasses import fields

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = dict(payload["model_config"])
    known = {f.name for f in fields(ModelConfig)}
    unknown = sorted(set(settings) - known)
    if unknown:
        logger.warning("checkpoint carries unknown ModelConfig keys: %s", unknown)
    settings = {k: v for k, v in settings.items() if k in known}
    # asdict() turned the tuple fields into lists on the way in.
    for key in ("dilations", "class_order"):
        if settings.get(key) is not None:
            settings[key] = tuple(settings[key])
    trained_target_length = int(
        settings.get("output_length") or settings["input_length"]
    )
    settings["output_length"] = None

    model = HybridSpliceModel(ModelConfig(**settings))
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    logger.info(
        "Loaded hybrid_splice checkpoint (epoch=%s, %d parameters): %s",
        payload.get("epoch"), model.num_parameters(), model.summary(),
    )
    if tuple(model.config.class_order) != ("acceptor", "donor", "neither"):
        raise ValueError(
            "scorer assumes (acceptor, donor, neither) channel order, got "
            f"{model.config.class_order}"
        )
    return model, trained_target_length


@torch.inference_mode()
def _hybrid_splice_forward_probs(
    model: torch.nn.Module, tokens_int8: np.ndarray, device: torch.device
) -> np.ndarray:
    """Score one batch of int8 token rows.

    Args:
        model (torch.nn.Module): Loaded ``HybridSpliceModel``.
        tokens_int8 (np.ndarray): ``(B, L)`` tokens, 0=N, 1=A, 2=C, 3=G, 4=T --
            the same encoding this module already uses (``_NT_TO_INT``) and the
            same one the training H5 stores.
        device (torch.device): Compute device.

    Returns:
        np.ndarray: ``(B, L, 3)`` probabilities in (acceptor, donor, neither)
        order.
    """
    tokens = torch.from_numpy(np.ascontiguousarray(tokens_int8)).to(device)
    channels = model.config.in_channels
    one_hot = torch.zeros(
        tokens.shape[0], channels, tokens.shape[1],
        device=device, dtype=torch.float32,
    )
    index = tokens.long().clamp_(0, channels - 1).unsqueeze(1)
    one_hot.scatter_(1, index, 1.0)
    output = model(one_hot)
    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.softmax(logits.float(), dim=1)
    return probs.permute(0, 2, 1).cpu().numpy()


def _load_dlm_qwen_model(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, int]:
    """Load a ``DlmQwenSplice`` checkpoint for full-length scoring.

    The checkpoint holds only the trained tensors -- the frozen DLM backbone is
    reloaded from its HuggingFace cache -- so the module is constructed first
    from the ``args`` the trainer stored, then filled in with
    ``strict=False``. Gradient checkpointing is off: it trades compute for
    activation memory, which is the wrong trade under ``inference_mode``.

    Args:
        checkpoint_path (str): Path to a checkpoint written by
            ``train/train_dlm_qwen.py``.
        device (torch.device): Compute device.

    Returns:
        tuple[torch.nn.Module, int]: The model in eval mode on ``device``, and
        the supervised target length it was trained with (the conservative-mode
        window clamp needs it, and the full-length readout hides it).
    """
    from model.dlm_qwen_splice import DlmQwenSplice

    state = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    train_args = dict(state["args"])
    evo2_path = train_args.get("evo2_path")
    model = DlmQwenSplice(
        dlm=train_args["dlm"],
        dlm_paths={"evo2": evo2_path} if evo2_path else None,
        qwen_path=train_args["qwen_path"],
        strands=train_args["strands"],
        gradient_checkpointing=False,
        qwen_dtype=torch.bfloat16,
    )
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # The frozen DLM and (in later runs) the unused token embedding are absent
    # from the checkpoint by design; anything else missing means the trained
    # weights did not land, which would score noise without ever erroring.
    unfilled = [
        k for k in missing
        if not k.startswith("dlms.") and k != "qwen.embed_tokens.weight"
    ]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} trained tensors missing from {checkpoint_path}, "
            f"e.g. {unfilled[:5]}"
        )
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} unexpected tensors in {checkpoint_path}, "
            f"e.g. {unexpected[:5]}"
        )
    model.eval().requires_grad_(False).to(device)
    logger.info(
        "Loaded dlm_qwen checkpoint (epoch=%s, step=%s, dlm=%s, strands=%s, "
        "qwen=%s, feature_dim=%d); %d frozen-backbone tensors reloaded from "
        "the HuggingFace cache.",
        state.get("epoch"), state.get("step"), train_args["dlm"],
        train_args["strands"], train_args["qwen_path"], model.feature_dim,
        len(missing),
    )
    return model, int(model.target_len)


@torch.inference_mode()
def _dlm_qwen_forward_probs(
    model: torch.nn.Module, tokens_int8: np.ndarray, device: torch.device
) -> np.ndarray:
    """Score one batch of int8 token rows with a ``DlmQwenSplice``.

    The model consumes nucleotide codes directly, so there is no one-hot and no
    tokenizer between this module's encoding and the backbone. bfloat16
    autocast matches how the stack was trained.

    Args:
        model (torch.nn.Module): Loaded ``DlmQwenSplice``.
        tokens_int8 (np.ndarray): ``(B, L)`` tokens, 0=N, 1=A, 2=C, 3=G, 4=T.
        device (torch.device): Compute device.

    Returns:
        np.ndarray: ``(B, L, 3)`` probabilities in (acceptor, donor, neither)
        order -- all ``L`` positions, so both aggregation modes are available.
    """
    codes = torch.from_numpy(np.ascontiguousarray(tokens_int8)).to(device)
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = model(codes.long(), full_length=True)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def _is_dlm_qwen(model: torch.nn.Module) -> bool:
    """Whether ``model`` is a ``DlmQwenSplice``.

    The class is imported lazily by ``_load_dlm_qwen_model`` -- it drags in the
    DNA-language-model backbones, which the other flavours do not need -- so
    check the already-imported module rather than importing it to ask.

    Args:
        model (torch.nn.Module): Loaded model.

    Returns:
        bool: True for a ``DlmQwenSplice``.
    """
    module = sys.modules.get("model.dlm_qwen_splice")
    return module is not None and isinstance(model, module.DlmQwenSplice)


def _batched_forward(
    model: torch.nn.Module,
    cfg: OmegaConf,
    seqs_int8: list[np.ndarray],
    device: torch.device,
    probability_mode: str,
    total_length: int,
    batch_size: int = _FIXED_BATCH_SIZE,
) -> np.ndarray:
    """Run model on a list of int8 sequences and return ``(N, L, 3)`` probs.

    Args:
        model (torch.nn.Module): loaded model.
        cfg (OmegaConf): training config used for tokenizer settings.
        seqs_int8 (list[np.ndarray]): list of length-``total_length``
            int8 arrays.
        device (torch.device): compute device.
        probability_mode (str): ``softmax`` or ``sigmoid``.
        total_length (int): chunk length; pass to disable center crop.
        batch_size (int): inference batch size.

    Returns:
        np.ndarray: probs ``(N, total_length, 3)`` in
            ``(acceptor, donor, neither)`` channel order.
    """
    n = len(seqs_int8)
    out = np.zeros((n, total_length, 3), dtype=np.float32)
    if _is_dlm_qwen(model):
        # Nucleotide codes go straight in: no one-hot, no tokenizer.
        for i in tqdm(
            range(0, n, batch_size), desc="Inference", leave=False,
        ):
            chunk = np.stack(seqs_int8[i : i + batch_size])
            out[i : i + len(chunk)] = _dlm_qwen_forward_probs(
                model, chunk, device
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out
    if isinstance(model, HybridSpliceModel):
        # No tokenizer: the model consumes a one-hot of the very same int8
        # encoding, so the verification dance below does not apply.
        for i in tqdm(
            range(0, n, batch_size), desc="Inference", leave=False,
        ):
            chunk = np.stack(seqs_int8[i : i + batch_size])
            out[i : i + len(chunk)] = _hybrid_splice_forward_probs(
                model, chunk, device
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out
    # HyenaDNA's tokenizer is a pure-Python CharacterTokenizer: decoding int8
    # to a 15,000-char string and re-splitting it costs ~480k interpreter-level
    # operations per batch to compute what is a fixed 5-entry remap. Build the
    # table once (_INT_TO_NT_STR indexes it) and index the array directly.
    lut = np.array(
        [model.tokenizer.convert_tokens_to_ids(c) for c in _INT_TO_NT_STR],
        dtype=np.int64,
    )
    verified = False
    for i in tqdm(
        range(0, n, batch_size),
        desc="Inference", leave=False,
    ):
        chunk = seqs_int8[i : i + batch_size]
        ids_np = lut[np.clip(np.stack(chunk).astype(np.int64), 0, 4)]
        if not verified:
            # Prove identity against the tokenizer once rather than assume it;
            # a silent mismatch here would corrupt every score downstream.
            ref = model.tokenizer(
                [_decode_seq(s) for s in chunk],
                return_tensors="pt", padding="max_length", truncation=True,
                max_length=int(cfg.model.tokenizer_max_length),
                return_offsets_mapping=False, add_special_tokens=False,
            )["input_ids"].numpy()
            if ref.shape != ids_np.shape or not np.array_equal(ref, ids_np):
                raise RuntimeError(
                    "Lookup-table tokenisation does not match the tokenizer "
                    f"(lut shapes {ids_np.shape} vs {ref.shape}). Refusing to "
                    "score with an unverified encoding."
                )
            logger.info(
                "Tokenisation lookup table verified against the tokenizer "
                "on the first batch (%s).", ids_np.shape,
            )
            verified = True
        probs = _hyenadna_forward_probs(
            model=model,
            seqs=None,
            cfg=cfg,
            device=device,
            probability_mode=probability_mode,
            target_length=total_length,
            input_ids=torch.from_numpy(ids_np),
        )
        out[i : i + len(chunk)] = probs.astype(np.float32)
    # Release cached allocations once the block is done, so a long run does
    # not crowd a co-located training process off the GPU. This used to run
    # every batch, which forces a device sync and drops the caching
    # allocator's arena thousands of times per shard -- the courtesy is
    # worth one call per block, not one per batch.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _evaluate(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run the multi-chunk SAV eval and persist artifacts.

    Args:
        args (argparse.Namespace): parsed CLI args.

    Returns:
        dict[str, object]: summary dict with per-mode metrics and
            counts. Side effects: writes features.npz and features.json
            for each mode under ``args.output_dir``.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading variant table from %s", args.variant_h5)
    variants = _load_variant_table(args.variant_h5)
    snvs = [v for v in variants if len(v["ref"]) == 1 and len(v["alt"]) == 1]
    n_indel = len(variants) - len(snvs)
    n_all_snvs = len(snvs)
    if args.row_start or args.row_end is not None:
        snvs = snvs[args.row_start:args.row_end]
        logger.info("Row slice [%d, %s) -> %d of %d SNVs",
                    args.row_start, args.row_end, len(snvs), n_all_snvs)
    logger.info("Variants: total=%d snv=%d indel=%d (indels skipped)",
                len(variants), len(snvs), n_indel)

    logger.info("Loading model from %s", args.checkpoint)
    trained_target_len: int | None = None
    if args.hyena_filter_max_seq_len is not None and args.model_type != "hyenadna":
        raise ValueError(
            "--hyena-filter-max-seq-len is only valid with --model-type hyenadna"
        )
    if args.model_type == "dlm_qwen":
        model, trained_target_len = _load_dlm_qwen_model(args.checkpoint, device)
        cfg = None
        prob_mode = "softmax"
        loss_type = "dlm_qwen_focal"
        from model.dlm_qwen_splice import CONTEXT_LEN as _dlm_context_len

        if int(args.total_length) != int(_dlm_context_len):
            raise ValueError(
                f"--total-length {args.total_length} does not match the "
                f"dlm_qwen training context {_dlm_context_len}; the chunk grid "
                "and the model would disagree about coordinates"
            )
    elif args.model_type == "hybrid_splice":
        model, trained_target_len = _load_hybrid_splice_model(
            args.checkpoint, device,
        )
        cfg = None
        prob_mode = "softmax"
        loss_type = "hybrid_splice_multitask"
        if int(args.total_length) != int(model.config.input_length):
            raise ValueError(
                f"--total-length {args.total_length} does not match the "
                f"model's input_length {model.config.input_length}; the "
                "chunk grid and the model would disagree about coordinates"
            )
    else:
        model, cfg = _load_hyenadna_model(
            checkpoint_path=args.checkpoint,
            train_config_path=args.train_config,
            device=device,
            hyena_filter_max_seq_len=args.hyena_filter_max_seq_len,
        )
        prob_mode = _resolve_probability_mode(cfg, args.probability_mode)
        loss_type = OmegaConf.select(
            cfg, "training.loss_function.type", default=None,
        )
    logger.info("loss_type=%s probability_mode=%s", loss_type, prob_mode)

    # Canonical sites are needed both to stand in for the reference readout
    # and to gate the four directions; load once, reuse for every chunk.
    canonical_genes: dict[str, dict] = {}
    if (args.ref_source == "annotation" or args.mask_canonical
            or args.mask_transcript_boundaries):
        canonical_genes = load_canonical_sites(args.annotation_path)
        covered = sum(
            1 for v in snvs if v.get("source_gene_name") in canonical_genes
        )
        logger.info(
            "Canonical annotation: %d genes; %d/%d variants have their "
            "anchoring gene in it", len(canonical_genes), covered, len(snvs),
        )
        if covered < len(snvs):
            raise RuntimeError(
                f"{len(snvs) - covered} of {len(snvs)} variants have no "
                f"anchoring gene in {args.annotation_path}. Those rows would be "
                "scored WITHOUT the canonical rule while the rest are scored "
                "with it, and nothing downstream could tell the two apart. "
                "Fix the variant table's source_gene_name, or pass "
                "--no-mask-canonical and register the run as unmasked."
            )

    tx_sites: dict[str, dict] = {}
    var_tx: dict[str, str] = {}
    site_source = {"transcript": 0, "gene": 0}
    if canonical_genes and args.transcript_annotation and args.variant_transcript_csv:
        tx_sites = load_sites_by_accession(args.transcript_annotation)
        var_tx = load_variant_transcripts(args.variant_transcript_csv)
        logger.info(
            "Transcript-keyed sites: %d accessions, %d variant->transcript rows",
            len(tx_sites), len(var_tx),
        )

    if args.target_region_length is not None:
        target_len = int(args.target_region_length)
    elif trained_target_len is not None:
        target_len = trained_target_len
    else:
        target_len = int(
            OmegaConf.select(cfg, "data.sequence_layout.target_region_length",
                             default=None)
            or OmegaConf.select(cfg, "data.target_region_length",
                                default=None)
            or 5000
        )
    total_len = int(args.total_length)
    flank_len = (total_len - target_len) // 2
    # --total-length is CLI-only and defaults to 15000, while target_len comes
    # from the checkpoint. A backbone trained at another geometry therefore
    # scores against the wrong chunk length unless --total-length is passed:
    # with the 32,768/16,384 arm the default gives flank_len = -692 and a
    # target window running off both ends of the chunk, which produces numbers
    # rather than an error. Refuse instead.
    if flank_len < 0:
        raise ValueError(
            f"--total-length {total_len} is shorter than the checkpoint's "
            f"target region {target_len}; pass the geometry this checkpoint "
            f"was trained at (e.g. --total-length 32768 for a 16,384 target)."
        )
    target_start = flank_len
    target_end = flank_len + target_len
    logger.info(
        "Model layout: total=%d target=%d flank=%d (target window=[%d, %d))",
        total_len, target_len, flank_len, target_start, target_end,
    )

    # Per-variant chunk positions: aligned to each variant's
    # variant_offset so one chunk = the gencode-anchored baseline.
    n_fallback = 0
    missing = [v for v in snvs if "variant_offset" not in v]
    if missing:
        if args.anchor_fallback == "error":
            raise ValueError(
                f"{len(missing)} of {len(snvs)} variants lack 'variant_offset' "
                f"— pass a gencode-anchored H5 so the multichunk grid can align "
                f"to the canonical chunk, or use --anchor-fallback center to "
                f"score them on a variant-centred chunk instead."
            )
        # Centre the variant, breaking the tie leftward: total_length is even,
        # so there is no exact centre. offset = total_length // 2 gives the
        # left flank one more base than the right, and lands the variant at
        # the midpoint of the supervised target region.
        centred = int(total_len // 2)
        for v in missing:
            v["variant_offset"] = centred
            v["anchor_fallback"] = True
        n_fallback = len(missing)
        logger.warning(
            "%d of %d variants had no gencode anchor; scored on a "
            "variant-centred chunk (offset=%d, left-biased). These are OFF the "
            "training grid -- the anchored chunk that the training H5 used is "
            "not among their K, so their scores are not directly comparable to "
            "anchored ones.",
            n_fallback, len(snvs), centred,
        )
    logger.info(
        "Generating K stride-aligned chunks per variant (stride=%d, "
        "total_length=%d); each variant's grid passes through its "
        "own variant_offset.",
        args.stride, total_len,
    )

    fasta = Fasta(args.genome_fasta, sequence_always_upper=True)

    batch_size = (
        int(args.batch_size) if args.batch_size is not None
        else _FIXED_BATCH_SIZE
    )
    modes = ("conservative", "liberal", "target_only", "target_strict")
    extra_mw = tuple(int(w) for w in args.extra_mass_windows)

    def _process_block(lo: int, hi: int) -> tuple[dict, dict]:
        """Slice, forward and aggregate the variants ``snvs[lo:hi]``.

        Scoring a whole shard in one pass holds every sequence's output in
        RAM at once -- for a 17k-SNV shard at K=4 that is ~23 GB, which is
        why the job needs a 96 GB reservation and why several of them on one
        node contend. Working a block at a time keeps peak RSS near 2 GB and,
        more importantly, bounds what a timeout destroys to a single block.

        Args:
            lo (int): first variant index in this block.
            hi (int): one past the last variant index in this block.

        Returns:
            tuple[dict, dict]: the block's numeric arrays keyed
                ``"<mode>|<feature>"``, and a small bookkeeping dict holding
                skip counts and the missing-baseline count.

        Raises:
            RuntimeError: If no valid chunk could be generated for the block.
        """
        block = snvs[lo:hi]
        all_seqs: list[np.ndarray] = []
        # (var_i, chunk_i, clp_model, ref_idx, alt_idx, is_baseline)
        variant_chunk_index: list[tuple[int, int, int, int, int, bool]] = []
        skips: dict[str, int] = {}
        n_transcript_masked_tokens = 0
        clps_per_variant: list[list[int]] = []
        for v in block:
            # variant_offset is in model space (post-RC for `-` strand).
            # Build the residue class there, then translate to forward
            # space so the genome-slicing function picks the right windows.
            clps_model = _aligned_chunk_positions(
                variant_offset=v["variant_offset"],
                total_length=total_len, stride=args.stride,
            )
            if args.chunk_selection == "target":
                # Drop chunks that place the variant outside the supervised
                # target BEFORE forwarding them -- this is where the 3x
                # saving comes from. Filtering only at aggregation time still
                # pays for all K=12 forwards. The kept chunks jointly cover
                # the whole +-delta_window, so no scored position is lost.
                kept = [c for c in clps_model if target_start <= c < target_end]
                if kept:
                    clps_model = kept
            clps_per_variant.append(clps_model)
            if v["strand"] == "-":
                clps_forward = [total_len - 1 - c for c in clps_model]
            else:
                clps_forward = clps_model
            recs = _generate_chunks_for_variant(
                fasta=fasta, chrom=v["chrom"], pos_hg38=v["pos"],
                ref=v["ref"], alt=v["alt"], strand=v["strand"],
                total_length=total_len,
                chunk_local_positions=clps_forward,
            )
            var_i = len(clps_per_variant) - 1
            for chunk_i, r in enumerate(recs):
                if r["status"] != "ok":
                    tag = r["status"].split(":", 1)[0]
                    skips[tag] = skips.get(tag, 0) + 1
                    continue
                if args.mask_transcript_boundaries:
                    gene = canonical_genes[v["source_gene_name"]]
                    r["x_ref"], n_masked = _mask_sequence_outside_transcript(
                        r["x_ref"], pos_hg38=v["pos"],
                        clp_model=r["clp_model"], strand=v["strand"],
                        tx_start=gene["tx_start"], tx_end=gene["tx_end"],
                    )
                    r["x_alt"], _ = _mask_sequence_outside_transcript(
                        r["x_alt"], pos_hg38=v["pos"],
                        clp_model=r["clp_model"], strand=v["strand"],
                        tx_start=gene["tx_start"], tx_end=gene["tx_end"],
                    )
                    n_transcript_masked_tokens += n_masked
                ref_idx = len(all_seqs)
                all_seqs.append(r["x_ref"])
                alt_idx = len(all_seqs)
                all_seqs.append(r["x_alt"])
                variant_chunk_index.append(
                    (var_i, chunk_i, r["clp_model"], ref_idx, alt_idx,
                     r["clp_model"] == v["variant_offset"])
                )

        if not all_seqs:
            raise RuntimeError(
                f"No valid chunks generated for variants [{lo}, {hi})."
            )
        probs_all = _batched_forward(
            model=model, cfg=cfg, seqs_int8=all_seqs, device=device,
            probability_mode=prob_mode, total_length=total_len,
            batch_size=batch_size,
        )

        nb = len(block)
        per_variant: dict[int, dict] = {
            i: {"tracks": [], "clp_models": [], "baseline_idx": None}
            for i in range(nb)
        }
        for var_i, _ci, clp_model, ref_idx, alt_idx, is_base in variant_chunk_index:
            entry = per_variant[var_i]
            v = block[var_i]
            acc_idx = don_idx = None
            if canonical_genes:
                gene = None
                if v.get("source_gene_name") in canonical_genes:
                    gene, from_tx = resolve_sites(
                        canonical_genes, tx_sites, var_tx,
                        chrom=v["chrom"], pos=v["pos"],
                        ref=v["ref"], alt=v["alt"],
                        gene=v["source_gene_name"],
                    )
                    site_source["transcript" if from_tx else "gene"] += 1
                if gene is None:
                    skips["gene_not_annotated"] = (
                        skips.get("gene_not_annotated", 0) + 1
                    )
                else:
                    minus = v["strand"] == "-"
                    # clp_model is in model space; the genome slice started at
                    # the forward-space position, so undo the mirror first.
                    clp_fwd = (total_len - 1 - clp_model) if minus else clp_model
                    acc_idx, don_idx = site_indices(
                        gene, gstart=v["pos"] - clp_fwd,
                        length=total_len, minus=minus,
                    )
            if acc_idx is not None and args.ref_source == "annotation":
                p_ref = annotation_probs(acc_idx, don_idx, total_len)
            else:
                p_ref = probs_all[ref_idx]
            tracks = _compute_per_bp_deltas(p_ref, probs_all[alt_idx])
            if acc_idx is not None and args.mask_canonical:
                tracks = apply_canonical_mask(tracks, acc_idx, don_idx)
            entry["tracks"].append(tracks)
            entry["clp_models"].append(clp_model)
            if is_base:
                entry["baseline_idx"] = len(entry["tracks"]) - 1
        # The per-chunk delta tracks are all that is needed from here on;
        # release the raw probabilities before aggregating.
        del probs_all, all_seqs

        out: dict[str, np.ndarray] = {
            "labels": np.array([v["label"] for v in block], dtype=np.int64),
            "row_index": np.array([v["idx"] for v in block], dtype=np.int64),
            "k_per_variant": np.array(
                [len(c) for c in clps_per_variant], dtype=np.int32
            ),
            "n_chunks_used": np.array(
                [len(per_variant[i]["tracks"]) for i in range(nb)],
                dtype=np.int32,
            ),
        }
        n_missing_baseline = 0
        for mode in modes:
            feats = {k: np.zeros(nb, dtype=np.float32) for k in _FEATURE_KEYS}
            bfeats = {k: np.zeros(nb, dtype=np.float32) for k in _FEATURE_KEYS}
            mass = np.zeros(nb, dtype=np.float32)
            bmass = np.zeros(nb, dtype=np.float32)
            mass_w = {w: np.zeros(nb, dtype=np.float32) for w in extra_mw}
            offs = {k: np.full(nb, _NO_PEAK, dtype=np.int64)
                    for k in (*_OFFSET_KEYS, "offset_delta_max")}
            for i in range(nb):
                entry = per_variant[i]
                scores = _aggregate_variant_features(
                    per_chunk_tracks=entry["tracks"],
                    chunk_local_var_positions=entry["clp_models"],
                    total_length=total_len,
                    target_start=target_start, target_end=target_end,
                    delta_window=args.delta_window, mode=mode,
                    mass_windows=extra_mw,
                )
                for k in _FEATURE_KEYS:
                    feats[k][i] = scores[k]
                mass[i] = scores["delta_mass"]
                for w in extra_mw:
                    mass_w[w][i] = scores[f"delta_mass_w{w}"]
                # Report the peak as a GENOMIC offset, so the caller reads the
                # position off as pos + offset without redoing strand logic.
                # The aggregator works in model (gene-reading) coordinates,
                # which run backwards on the minus strand.
                flip = -1 if block[i]["strand"] == "-" else 1
                for k in (*_OFFSET_KEYS, "offset_delta_max"):
                    o = scores[k]
                    offs[k][i] = o if o == _NO_PEAK else flip * o

                # Single-chunk baseline = the gencode-anchored chunk (clp ==
                # variant_offset). This reproduces the canonical per-variant
                # score that the standard evaluate_splicevardb_sav reports.
                bidx = entry["baseline_idx"]
                if bidx is None:
                    if mode == modes[0]:
                        n_missing_baseline += 1
                    continue
                bscores = _aggregate_variant_features(
                    per_chunk_tracks=[entry["tracks"][bidx]],
                    chunk_local_var_positions=[entry["clp_models"][bidx]],
                    total_length=total_len,
                    target_start=target_start, target_end=target_end,
                    delta_window=args.delta_window, mode=mode,
                )
                for k in _FEATURE_KEYS:
                    bfeats[k][i] = bscores[k]
                bmass[i] = bscores["delta_mass"]
            for k in _FEATURE_KEYS:
                out[f"{mode}|{k}"] = feats[k]
                out[f"{mode}|baseline_{k}"] = bfeats[k]
            out[f"{mode}|delta_mass"] = mass
            out[f"{mode}|baseline_delta_mass"] = bmass
            for k in (*_OFFSET_KEYS, "offset_delta_max"):
                out[f"{mode}|{k}"] = offs[k]
            for w in extra_mw:
                out[f"{mode}|delta_mass_w{w}"] = mass_w[w]
        return out, {"skip_counts": skips,
                     "n_transcript_masked_tokens": n_transcript_masked_tokens,
                     "n_missing_baseline": n_missing_baseline}

    block_size = max(int(args.block_size), 1)
    n_blocks = (len(snvs) + block_size - 1) // block_size
    partial_dir = out_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Scoring %d SNVs in %d block(s) of %d. Each block is persisted to %s "
        "as it finishes, so a timeout costs one block and a rerun resumes.",
        len(snvs), n_blocks, block_size, partial_dir,
    )

    blocks: list[dict] = []
    skip_counts: dict[str, int] = {}
    n_missing_baseline = 0
    n_transcript_masked_tokens = 0
    for bi in range(n_blocks):
        lo = bi * block_size
        hi = min(lo + block_size, len(snvs))
        npz_path_b = partial_dir / f"block_{bi:04d}.npz"
        json_path_b = partial_dir / f"block_{bi:04d}.json"
        if npz_path_b.exists() and json_path_b.exists():
            logger.info("[block %d/%d] resuming from %s",
                        bi + 1, n_blocks, npz_path_b.name)
            with np.load(npz_path_b) as z:
                payload = {k: z[k] for k in z.files}
            meta = json.loads(json_path_b.read_text())
        else:
            logger.info("[block %d/%d] variants [%d, %d)",
                        bi + 1, n_blocks, lo, hi)
            payload, meta = _process_block(lo, hi)
            # Write under a temp name and rename: being killed inside
            # np.savez would otherwise leave a truncated block that the
            # resume path above would happily trust.
            # NB the temp name must itself end in .npz: np.savez appends the
            # extension when it is absent, which would leave the payload at
            # <name>.npz.tmp.npz and make the rename below fail.
            tmp_path = npz_path_b.with_suffix(".tmp.npz")
            np.savez(tmp_path, **payload)
            tmp_path.replace(npz_path_b)
            json_path_b.write_text(json.dumps(meta, indent=2))
        blocks.append(payload)
        for tag, cnt in meta["skip_counts"].items():
            skip_counts[tag] = skip_counts.get(tag, 0) + int(cnt)
        n_missing_baseline += int(meta["n_missing_baseline"])
        n_transcript_masked_tokens += int(
            meta.get("n_transcript_masked_tokens", 0)
        )

    def _cat(key: str) -> np.ndarray:
        """Concatenate one per-block array back into variant order.

        Args:
            key (str): array name inside each block payload.

        Returns:
            np.ndarray: the concatenated array over all blocks.
        """
        return np.concatenate([b[key] for b in blocks])

    labels = _cat("labels")
    n_chunks_used = _cat("n_chunks_used")
    k_per_variant_arr = _cat("k_per_variant")
    n = int(labels.shape[0])
    if n != len(snvs):
        raise RuntimeError(
            f"Block reassembly produced {n} rows but the shard holds "
            f"{len(snvs)} SNVs — stale blocks in {partial_dir}?"
        )
    if n_missing_baseline:
        logger.warning(
            "%d variants had no chunk at clp == variant_offset "
            "(baseline NaN-equivalent for those rows).",
            n_missing_baseline,
        )

    summaries: dict[str, dict] = {}
    for mode in modes:
        logger.info("Aggregating mode=%s", mode)
        features = {k: _cat(f"{mode}|{k}") for k in _FEATURE_KEYS}
        baseline_features = {
            k: _cat(f"{mode}|baseline_{k}") for k in _FEATURE_KEYS
        }
        mass = _cat(f"{mode}|delta_mass")
        baseline_mass = _cat(f"{mode}|baseline_delta_mass")
        mass_w = {w: _cat(f"{mode}|delta_mass_w{w}") for w in extra_mw}

        metrics: dict[str, float] = {"n": int(n),
                                     "positive_count": int(labels.sum()),
                                     "negative_count": int((labels == 0).sum())}
        for k in _FEATURE_KEYS:
            metrics[f"auroc_{k}"] = float(roc_auc_score(labels, features[k]))
            metrics[f"auprc_{k}"] = float(average_precision_score(labels, features[k]))
            metrics[f"baseline_auroc_{k}"] = float(
                roc_auc_score(labels, baseline_features[k])
            )
            metrics[f"baseline_auprc_{k}"] = float(
                average_precision_score(labels, baseline_features[k])
            )
        metrics["auroc_delta_mass"] = float(roc_auc_score(labels, mass))
        metrics["auprc_delta_mass"] = float(average_precision_score(labels, mass))
        for w in extra_mw:
            metrics[f"auroc_delta_mass_w{w}"] = float(roc_auc_score(labels, mass_w[w]))
            metrics[f"auprc_delta_mass_w{w}"] = float(average_precision_score(labels, mass_w[w]))

        npz_path = out_dir / f"features_{mode}.npz"
        np.savez(
            npz_path, labels=labels, n_chunks_used=n_chunks_used,
            **{f"multichunk_{k}": features[k] for k in _FEATURE_KEYS},
            **{f"baseline_{k}": baseline_features[k] for k in _FEATURE_KEYS},
            multichunk_delta_mass=mass, baseline_delta_mass=baseline_mass,
            **{k: _cat(f"{mode}|{k}")
               for k in (*_OFFSET_KEYS, "offset_delta_max")},
            **{f"multichunk_delta_mass_w{w}": mass_w[w] for w in extra_mw},
        )
        logger.info("Saved per-variant features: %s", npz_path)

        summary = {
            "mode": mode,
            "checkpoint": args.checkpoint,
            "variant_h5": args.variant_h5,
            "genome_fasta": args.genome_fasta,
            "delta_window_bp": args.delta_window,
            "stride": args.stride,
            "chunk_selection": args.chunk_selection,
            "mask_canonical": args.mask_canonical,
            "annotation_path": args.annotation_path if args.mask_canonical else None,
            "transcript_annotation": args.transcript_annotation,
            # Counted per (variant, chunk), so K times the variant count.
            "mask_site_source": dict(site_source),
            "mask_transcript_boundaries": args.mask_transcript_boundaries,
            "n_transcript_masked_tokens": n_transcript_masked_tokens,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "total_length": total_len,
            "target_length": target_len,
            "flank_length": flank_len,
            "hyena_filter_max_seq_len": args.hyena_filter_max_seq_len,
            "filter_grid_intervention": (
                "position_basis_and_exponential_modulation"
                if args.hyena_filter_max_seq_len is not None else None
            ),
            "chunk_alignment": "per_variant_residue_class_of_variant_offset",
            "anchor_fallback": args.anchor_fallback,
            "n_anchor_fallback": n_fallback,
            "k_per_variant_min": int(k_per_variant_arr.min()) if n else 0,
            "k_per_variant_max": int(k_per_variant_arr.max()) if n else 0,
            "k_per_variant_mean": float(k_per_variant_arr.mean()) if n else 0.0,
            "n_snv": len(snvs),
            "n_snv_all": n_all_snvs,
            "row_start": args.row_start,
            "row_end": args.row_end,
            "n_indel_skipped": n_indel,
            "skip_counts": skip_counts,
            "metrics": metrics,
            "loss_type": loss_type,
            "probability_mode": prob_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        json_path = out_dir / f"features_{mode}.json"
        json_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Saved: %s", json_path)

        logger.info("=" * 60)
        logger.info("Mode=%s summary", mode)
        logger.info("  n=%d (pos=%d, neg=%d)",
                    n, int(labels.sum()), int((labels == 0).sum()))
        for k in _FEATURE_KEYS:
            logger.info(
                "  %s: multichunk auroc=%.4f (baseline=%.4f) | "
                "auprc=%.4f (baseline=%.4f)",
                k,
                metrics[f"auroc_{k}"], metrics[f"baseline_auroc_{k}"],
                metrics[f"auprc_{k}"], metrics[f"baseline_auprc_{k}"],
            )
        summaries[mode] = summary
    return summaries


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    _evaluate(args)


if __name__ == "__main__":
    main()
