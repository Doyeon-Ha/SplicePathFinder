"""Indel SAV evaluator v4 — per-position aligned delta (SNV-consistent).

Successor to ``evaluate_splicevardb_sav_indel_v3.py``. v3 scored indels
with a *max-then-diff* rule (``max(prob_alt[window]) - max(prob_ref[window])``
per channel), which compares the ref and alt probability peaks even when
they sit at different positions. The SNV multi-chunk evaluator instead
uses the SpliceAI-style *diff-then-max* rule: a per-position difference
``prob_alt - prob_ref`` split into four non-negative directional tracks,
then the max over positions within a window. v4 brings indels onto that
same definition so SNV and indel ``delta_max`` are computed identically.

The only indel-specific step is aligning the variable-length ref and alt
into a shared coordinate frame before the per-position diff:

* **Flanks** outside the variant region are aligned 1:1 from the variant
  edge outward (upstream and downstream), so a frame shift introduced by
  the indel is captured as genuine per-position change.
* **Variant overlap** (``min(L_ref, L_alt)`` bases at the anchored edge)
  is aligned 1:1.
* **Extra bases** with no counterpart (the ``|L_ref - L_alt|`` deleted or
  inserted bases) are compared against an implicit zero, so a deletion
  charges ``prob_ref`` as loss and an insertion charges ``prob_alt`` as
  gain — the same zero-compare intuition introduced in v3.

Anchoring follows the left-anchored VCF representation used by
``_build_indel_chunk`` (extra bases at the tail on ``+`` strand, at the
head on ``-`` strand after reverse-complementing).

Output schema matches v3 (``features_liberal.npz`` with the same 7
``multichunk_*`` keys + a ``baseline_*`` mirror) so downstream selection,
ensemble and post-processing scripts work without changes.
"""

from __future__ import annotations

import argparse
import h5py
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.evaluate_splicevardb_sav import (  # noqa: E402
    _load_hyenadna_model,
    _resolve_probability_mode,
)
from scripts.canonical_site_track import (  # noqa: E402
    ANNOTATION,
    apply_canonical_mask,
    load_canonical_sites,
    load_sites_by_accession,
    load_variant_transcripts,
    resolve_sites,
    site_indices,
)
from scripts.evaluate_splicevardb_sav_indel_v3 import (  # noqa: E402
    _FEATURE_KEYS,
    _batched_forward,
    _build_indel_chunk,
)
from scripts.evaluate_splicevardb_sav_multichunk import _NO_PEAK, _OFFSET_KEYS, _compute_per_bp_deltas, _load_variant_table

logger = logging.getLogger(__name__)

_NUM_CHANNELS: int = 3


def _mask_chunk_gstart(v: dict, total_length: int, training_h5: bool) -> int:
    """Genomic position of forward chunk index zero for the mask."""
    clp = int(v.get("variant_offset", total_length // 2))
    # Genome slicing receives clp in forward space; H5 sequences store it in
    # model space, which is mirrored for minus-strand genes.
    if training_h5 and v["strand"] == "-":
        clp = total_length - 1 - clp
    return int(v["pos"]) - clp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--model-type",
        choices=("hyenadna",),
        default="hyenadna",
        help=(
            "Checkpoint flavour; see the same flag on "
            "evaluate_splicevardb_sav_multichunk.py. The indel alignment and "
            "aggregation are shared, so every flavour is scored identically."
        ),
    )
    p.add_argument("--variant-h5", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--genome-fasta", default="./genome/GRCh38.fa"
    )
    p.add_argument("--train-config", default=None)
    p.add_argument("--delta-window", type=int, default=2500)
    p.add_argument(
        "--extra-mass-windows", type=int, nargs="*", default=[],
        help="Additional delta_mass window half-widths (bp) emitted as "
             "multichunk_delta_mass_w<W>; additive, existing outputs unchanged.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--probability-mode", default="auto",
        choices=["auto", "softmax", "sigmoid"]
    )
    p.add_argument("--target-region-length", type=int, default=None)
    p.add_argument(
        "--row-start", type=int, default=0,
        help="Score only indels [row-start, row-end) of the table. This path "
             "writes once at the end, so on a very large table a slice is what "
             "bounds what a timeout can destroy; the SNV path has its own "
             "per-block checkpointing and needs no slicing.",
    )
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument(
        "--mask-canonical", action=argparse.BooleanOptionalAction, default=False,
        help="Apply the channel-wise canonical rule: count a loss only where "
             "that site type is annotated and a gain only where it is not. "
             "Off by default here, unlike the SNV evaluator, because this "
             "script is shared with running arms registered as unmasked.",
    )
    p.add_argument(
        "--annotation-path", default=ANNOTATION,
        help="GENCODE v49 canonical annotation table (SpliceAI 7-column TSV).",
    )
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
        "--mask-transcript-boundaries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the input H5's training-built X_ref/X_alt sequences, which "
             "are N-padded outside transcript boundaries, instead of slicing "
             "the real genomic flanks from FASTA.",
    )
    return p.parse_args()


def _align_indel_probs(
    prob_ref: np.ndarray,
    prob_alt: np.ndarray,
    rec: dict,
    delta_window: int,
    total_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ref/alt probabilities into a shared per-position frame.

    Walks the variant region plus ``delta_window`` flanking positions on
    each side and emits one aligned ``(prob_ref_row, prob_alt_row)`` pair
    per shared position. Positions present in only one allele (the
    deleted or inserted bases) pair the real probability against a zero
    vector, realising the v3 zero-compare within the per-position frame.

    Args:
        prob_ref (np.ndarray): ref probs of shape ``(L, 3)`` in channel
            order ``(acc, don, nei)``.
        prob_alt (np.ndarray): alt probs of shape ``(L, 3)``.
        rec (dict): variant record from ``_build_indel_chunk`` with
            ``var_start_ref`` / ``var_end_ref`` / ``var_start_alt`` /
            ``var_end_alt`` / ``l_ref`` / ``l_alt`` / ``strand``.
        delta_window (int): number of flanking positions to include on
            each side of the variant region.
        total_length (int): chunk length (e.g. 15000).

    Returns:
        tuple[np.ndarray, np.ndarray]: aligned ref and alt arrays, each of
            shape ``(W, 3)`` where ``W`` is the number of shared
            positions. Both are empty ``(0, 3)`` if no position is valid.
    """
    var_start_ref = rec["var_start_ref"]
    var_end_ref = rec["var_end_ref"]
    var_start_alt = rec["var_start_alt"]
    var_end_alt = rec["var_end_alt"]
    l_ref = rec["l_ref"]
    l_alt = rec["l_alt"]
    strand = rec["strand"]
    overlap = min(l_ref, l_alt)
    zero = np.zeros(_NUM_CHANNELS, dtype=np.float32)

    ref_rows: list[np.ndarray] = []
    alt_rows: list[np.ndarray] = []

    # Upstream flank: aligned 1:1 from the variant start outward.
    for offset in range(delta_window, 0, -1):
        idx_ref = var_start_ref - offset
        idx_alt = var_start_alt - offset
        if idx_ref < 0 or idx_alt < 0:
            continue
        ref_rows.append(prob_ref[idx_ref])
        alt_rows.append(prob_alt[idx_alt])

    # Variant region: overlap aligned 1:1, extra bases zero-compared.
    # The left-anchored representation puts extra bases at the tail on
    # the + strand and at the head on the - strand.
    # Bounds-guarded access: positions past the model window edge (a large
    # insertion/deletion can push var_end_alt/ref beyond total_length) have no
    # prediction and contribute a null (zero) probability -- keeps every variant
    # scored instead of an IndexError.
    def _at(arr, idx):
        return arr[idx] if 0 <= idx < total_length else zero

    if strand == "+":
        for j in range(overlap):
            ref_rows.append(_at(prob_ref, var_start_ref + j))
            alt_rows.append(_at(prob_alt, var_start_alt + j))
        for j in range(var_start_ref + overlap, var_end_ref):
            ref_rows.append(_at(prob_ref, j))
            alt_rows.append(zero)
        for j in range(var_start_alt + overlap, var_end_alt):
            ref_rows.append(zero)
            alt_rows.append(_at(prob_alt, j))
    else:
        for j in range(var_start_ref, var_end_ref - overlap):
            ref_rows.append(_at(prob_ref, j))
            alt_rows.append(zero)
        for j in range(var_start_alt, var_end_alt - overlap):
            ref_rows.append(zero)
            alt_rows.append(_at(prob_alt, j))
        for j in range(overlap):
            ref_rows.append(_at(prob_ref, var_end_ref - overlap + j))
            alt_rows.append(_at(prob_alt, var_end_alt - overlap + j))

    # Downstream flank: aligned 1:1 from the variant end outward.
    for offset in range(delta_window):
        idx_ref = var_end_ref + offset
        idx_alt = var_end_alt + offset
        if idx_ref >= total_length or idx_alt >= total_length:
            break
        ref_rows.append(prob_ref[idx_ref])
        alt_rows.append(prob_alt[idx_alt])

    if not ref_rows:
        empty = np.zeros((0, _NUM_CHANNELS), dtype=np.float32)
        return empty, empty.copy()
    return (
        np.asarray(ref_rows, dtype=np.float32),
        np.asarray(alt_rows, dtype=np.float32),
    )


def _aligned_ref_indices(
    rec: dict,
    delta_window: int,
    total_length: int,
) -> np.ndarray:
    """Model-space reference index behind each row of the aligned axis.

    The v4 axis is not the genomic axis: it splices deletion rows (ref
    observed, alt null) and insertion rows (ref null, alt observed) into the
    variant span, and on the ``-`` strand puts them ahead of the anchor. A
    constant offset therefore shifts the canonical mask exactly where the
    alignment diverges. A deleted base still exists in the reference and can
    be an annotated site; an inserted base exists in neither, so it has no
    reference position at all.

    The loop structure mirrors :func:`_align_indel_probs` row for row,
    including its ``continue``/``break`` edges, so row ``i`` of the returned
    array describes row ``i`` of the aligned tracks.

    Args:
        rec (dict): variant record from ``_build_indel_chunk``.
        delta_window (int): number of flanking positions per side.
        total_length (int): chunk length.

    Returns:
        np.ndarray: per-row model-space ref index, ``-1`` where none exists
            (inserted bases, and positions past the chunk edge).
    """
    var_start_ref = rec["var_start_ref"]
    var_end_ref = rec["var_end_ref"]
    var_start_alt = rec["var_start_alt"]
    var_end_alt = rec["var_end_alt"]
    overlap = min(rec["l_ref"], rec["l_alt"])
    idx: list[int] = []

    def _keep(i: int) -> int:
        return i if 0 <= i < total_length else -1

    for offset in range(delta_window, 0, -1):
        idx_ref = var_start_ref - offset
        idx_alt = var_start_alt - offset
        if idx_ref < 0 or idx_alt < 0:
            continue
        idx.append(idx_ref)

    if rec["strand"] == "+":
        idx.extend(_keep(var_start_ref + j) for j in range(overlap))
        idx.extend(_keep(j) for j in range(var_start_ref + overlap, var_end_ref))
        idx.extend(-1 for _ in range(var_start_alt + overlap, var_end_alt))
    else:
        idx.extend(_keep(j) for j in range(var_start_ref, var_end_ref - overlap))
        idx.extend(-1 for _ in range(var_start_alt, var_end_alt - overlap))
        idx.extend(_keep(var_end_ref - overlap + j) for j in range(overlap))

    for offset in range(delta_window):
        idx_ref = var_end_ref + offset
        idx_alt = var_end_alt + offset
        if idx_ref >= total_length or idx_alt >= total_length:
            break
        idx.append(idx_ref)

    return np.asarray(idx, dtype=np.int64)


def _score_indel_v4(
    prob_ref: np.ndarray,
    prob_alt: np.ndarray,
    rec: dict,
    delta_window: int,
    total_length: int,
    mass_windows: tuple[int, ...] = (),
    mask_sites: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute per-position aligned directional scores for one indel.

    Aligns ref/alt via :func:`_align_indel_probs`, applies the shared
    SNV per-position diff (:func:`_compute_per_bp_deltas`) to obtain the
    four non-negative tracks, then takes the max over positions — the
    same diff-then-max definition as the SNV multi-chunk evaluator.

    Args:
        prob_ref (np.ndarray): ref probs of shape ``(L, 3)`` in channel
            order ``(acc, don, nei)``.
        prob_alt (np.ndarray): alt probs of shape ``(L, 3)``.
        rec (dict): variant record from ``_build_indel_chunk``.
        delta_window (int): number of flanking positions per side.
        total_length (int): chunk length (e.g. 15000).
        mask_sites (tuple[np.ndarray, np.ndarray] | None): annotated acceptor
            and donor indices in model space. When given, the channel-wise
            canonical rule is applied to the aligned tracks.

    Returns:
        dict[str, float]: the same 7 feature keys as the SNV evaluator.
    """
    def _masked(tracks: np.ndarray, window: int) -> np.ndarray:
        if mask_sites is None or not tracks.shape[0]:
            return tracks
        rows = _aligned_ref_indices(rec, window, total_length)
        if rows.shape[0] != tracks.shape[0]:
            raise RuntimeError(
                f"aligned row count {tracks.shape[0]} does not match the "
                f"{rows.shape[0]} reference indices derived for the same "
                "window; the mask would be applied to the wrong rows"
            )
        acc_sites, don_sites = mask_sites
        return apply_canonical_mask(
            tracks,
            np.flatnonzero(np.isin(rows, acc_sites)),
            np.flatnonzero(np.isin(rows, don_sites)),
        )

    aligned_ref, aligned_alt = _align_indel_probs(
        prob_ref,
        prob_alt,
        rec,
        delta_window,
        total_length,
    )
    tracks = _masked(
        _compute_per_bp_deltas(aligned_ref, aligned_alt), delta_window
    )  # (W, 4)
    if tracks.shape[0]:
        best = tracks.max(axis=0)
        # delta_mass = Σ over the aligned window of (|Δacc| + |Δdon|) per bp
        # (all 4 non-negative tracks summed) — the L1 mass of the deltas.
        mass = float(tracks.sum())
    else:
        best = np.zeros(4, dtype=np.float32)
        mass = 0.0
    acc_gain, acc_loss, don_gain, don_loss = (float(v) for v in best)
    # Where each direction peaked. The aligned axis is not the genomic axis, so
    # the row index is turned back into a reference position before it becomes
    # an offset; inserted rows have no reference position and report none.
    offsets = [_NO_PEAK] * 4
    if tracks.shape[0]:
        rows = _aligned_ref_indices(rec, delta_window, total_length)
        minus = rec["strand"] == "-"
        # The VCF position is the genomically leftmost reference base, which on
        # the minus strand is the LAST model-space base of the ref span, not
        # the first. Anchoring on var_start_ref there would shift every
        # multi-base variant's reported position by len(ref) - 1.
        anchor = rec["var_end_ref"] - 1 if minus else rec["var_start_ref"]
        flip = -1 if minus else 1
        for d in range(4):
            if best[d] <= 0:
                continue
            ref_i = rows[int(tracks[:, d].argmax())]
            if ref_i >= 0:
                offsets[d] = flip * int(ref_i - anchor)
    out = {
        "delta_max": max(acc_gain, acc_loss, don_gain, don_loss),
        "delta_acc_max": max(acc_gain, acc_loss),
        "delta_don_max": max(don_gain, don_loss),
        "delta_acc_gain": acc_gain,
        "delta_acc_loss": acc_loss,
        "delta_don_gain": don_gain,
        "delta_don_loss": don_loss,
        "delta_mass": mass,
        **dict(zip(_OFFSET_KEYS, offsets)),
        "offset_delta_max": offsets[int(np.argmax(best))],
    }
    # Additive: delta_mass at extra window half-widths. A smaller window is just
    # a re-alignment with fewer flanking positions (no backbone rerun).
    for w in mass_windows:
        w = int(w)
        aref_w, aalt_w = _align_indel_probs(prob_ref, prob_alt, rec, w, total_length)
        tw = _masked(_compute_per_bp_deltas(aref_w, aalt_w), w)
        out[f"delta_mass_w{w}"] = float(tw.sum()) if tw.shape[0] else 0.0
    return out


def _indel_record_from_h5(
    x_ref: np.ndarray,
    x_alt: np.ndarray,
    *,
    variant_offset: int,
    ref: str,
    alt: str,
    strand: str,
    total_length: int,
) -> dict:
    """Build v4 alignment metadata for training-built H5 sequences.

    ``variant_offset`` is already in model (gene-reading) coordinates. The H5
    builder patches a minus-strand alternate at the reference span's *start*
    (``voff - len(ref) + 1``), so variable-length alleles are left-aligned in
    the stored model array on both strands. V4 must follow those stored array
    coordinates rather than re-derive a right-anchored span from the VCF.
    """
    xr = np.asarray(x_ref, dtype=np.int8)
    xa = np.asarray(x_alt, dtype=np.int8)
    if len(xr) != total_length or len(xa) != total_length:
        raise ValueError(
            f"H5 sequence length mismatch: ref={len(xr)} alt={len(xa)} "
            f"expected={total_length}"
        )
    clp = int(variant_offset)
    l_ref, l_alt = len(ref), len(alt)
    if strand == "-":
        var_start_ref = clp - l_ref + 1
        var_end_ref = var_start_ref + l_ref
        var_start_alt = var_start_ref
        var_end_alt = var_start_alt + l_alt
        # `_align_indel_probs` uses this field only to decide whether
        # unmatched allele bases live at the span head or tail. The H5 patch
        # is left-aligned in model-array space, hence tail semantics here.
        alignment_strand = "+"
    else:
        var_start_ref, var_end_ref = clp, clp + l_ref
        var_start_alt, var_end_alt = clp, clp + l_alt
        alignment_strand = "+"
    return {
        "x_ref": xr,
        "x_alt": xa,
        "var_start_ref": int(var_start_ref),
        "var_end_ref": int(var_end_ref),
        "var_start_alt": int(var_start_alt),
        "var_end_alt": int(var_end_alt),
        "l_ref": int(l_ref),
        "l_alt": int(l_alt),
        "strand": alignment_strand,
        "source_strand": strand,
    }


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the v4 indel SAV evaluator and persist artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading variant table from %s", args.variant_h5)
    variants = _load_variant_table(args.variant_h5)
    # "indel" = any non-SNV variant (len(ref)>1 or len(alt)>1), including
    # same-length MNVs: they need the same alignment-aware scoring and are NOT
    # scored by the SNV (len==1) path, so there is no cross-partition double
    # counting.
    indels = [v for v in variants if not (len(v["ref"]) == 1 and len(v["alt"]) == 1)]
    n_all_indels = len(indels)
    if args.row_start or args.row_end is not None:
        indels = indels[args.row_start:args.row_end]
        logger.info("Row slice [%d, %s) -> %d of %d indels",
                    args.row_start, args.row_end, len(indels), n_all_indels)
    logger.info("Variants: total=%d non-SNV(indel/MNV)=%d", len(variants), len(indels))
    if not indels:
        logger.warning("No indels — nothing to do")
        return {"mode": "indel_v4", "n_indel": 0}

    canonical_genes: dict[str, dict] = {}
    if args.mask_canonical:
        canonical_genes = load_canonical_sites(args.annotation_path)
        covered = sum(
            1 for v in indels if v.get("source_gene_name") in canonical_genes
        )
        logger.info(
            "Canonical annotation: %d genes; %d/%d indels have their anchoring "
            "gene in it", len(canonical_genes), covered, len(indels),
        )
        if covered < len(indels):
            raise RuntimeError(
                f"{len(indels) - covered} of {len(indels)} indels have no "
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

    logger.info("Loading model from %s", args.checkpoint)
    model, cfg = _load_hyenadna_model(
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
        device=device,
    )
    prob_mode = _resolve_probability_mode(cfg, args.probability_mode)
    seq_length = int(OmegaConf.select(cfg, "model.seq_length", default=15000))
    total_length = seq_length
    fasta = Fasta(args.genome_fasta, sequence_always_upper=True, as_raw=True)

    chunks: list[dict | None] = []
    if args.mask_transcript_boundaries:
        with h5py.File(args.variant_h5, "r") as f:
            if "X_ref" not in f or "X_alt" not in f:
                raise RuntimeError(
                    "--mask-transcript-boundaries requires X_ref and X_alt "
                    "datasets built with the training N-padding convention"
                )
            for v in tqdm(indels, desc="Loading training-built indel chunks"):
                source_i = int(v["idx"])
                chunks.append(_indel_record_from_h5(
                    f["X_ref"][source_i], f["X_alt"][source_i],
                    variant_offset=int(v.get("variant_offset",
                                             total_length // 2)),
                    ref=v["ref"], alt=v["alt"], strand=v["strand"],
                    total_length=total_length,
                ))
    else:
        for v in tqdm(indels, desc="Slicing indel chunks"):
            clp = int(v.get("variant_offset", total_length // 2))
            rec = _build_indel_chunk(
                fasta=fasta,
                chrom=v["chrom"],
                pos_hg38=v["pos"],
                ref=v["ref"],
                alt=v["alt"],
                strand=v["strand"],
                total_length=total_length,
                clp=clp,
            )
            chunks.append(rec)
    valid_idx = [i for i, c in enumerate(chunks) if c is not None]
    refs = [chunks[i]["x_ref"] for i in valid_idx]
    alts = [chunks[i]["x_alt"] for i in valid_idx]
    logger.info("Forward pass on %d ref + %d alt sequences", len(refs), len(alts))
    probs_ref = _batched_forward(
        model,
        cfg,
        refs,
        device,
        prob_mode,
        total_length,
        args.batch_size,
    )
    probs_alt = _batched_forward(
        model,
        cfg,
        alts,
        device,
        prob_mode,
        total_length,
        args.batch_size,
    )

    features: dict[str, list[float]] = {k: [] for k in _FEATURE_KEYS}
    mass_vals: list[float] = []  # delta_mass (L1) — not in _FEATURE_KEYS
    extra_mw = tuple(int(w) for w in args.extra_mass_windows)
    mass_w_vals: dict[int, list[float]] = {w: [] for w in extra_mw}
    labels: list[int] = []
    offs: dict[str, list[int]] = {k: [] for k in (*_OFFSET_KEYS, "offset_delta_max")}
    for j, i_var in enumerate(valid_idx):
        rec = chunks[i_var]
        mask_sites = None
        if canonical_genes:
            v = indels[i_var]
            minus = v["strand"] == "-"
            rec_sites, from_tx = resolve_sites(
                canonical_genes, tx_sites, var_tx,
                chrom=v["chrom"], pos=v["pos"], ref=v["ref"], alt=v["alt"],
                gene=v["source_gene_name"],
            )
            site_source["transcript" if from_tx else "gene"] += 1
            mask_sites = site_indices(
                rec_sites,
                gstart=_mask_chunk_gstart(v, total_length, args.mask_transcript_boundaries),
                length=total_length,
                minus=minus,
            )
        scores = _score_indel_v4(
            probs_ref[j],
            probs_alt[j],
            rec,
            args.delta_window,
            total_length,
            mass_windows=extra_mw,
            mask_sites=mask_sites,
        )
        for k in _FEATURE_KEYS:
            features[k].append(scores[k])
        mass_vals.append(scores["delta_mass"])
        for w in extra_mw:
            mass_w_vals[w].append(scores[f"delta_mass_w{w}"])
        for k in (*_OFFSET_KEYS, "offset_delta_max"):
            offs[k].append(int(scores[k]))
        labels.append(indels[i_var]["label"])

    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(labels, dtype=np.int64),
        "n_chunks_used": np.full(len(labels), 1, dtype=np.int32),
    }
    for k in _FEATURE_KEYS:
        col = np.asarray(features[k], dtype=np.float32)
        arrays[f"multichunk_{k}"] = col
        arrays[f"baseline_{k}"] = col.copy()
    mass_col = np.asarray(mass_vals, dtype=np.float32)
    arrays["multichunk_delta_mass"] = mass_col
    arrays["baseline_delta_mass"] = mass_col.copy()
    for w in extra_mw:
        arrays[f"multichunk_delta_mass_w{w}"] = np.asarray(mass_w_vals[w], dtype=np.float32)
    for k in (*_OFFSET_KEYS, "offset_delta_max"):
        arrays[k] = np.asarray(offs[k], dtype=np.int64)
    np.savez(out_dir / "features_liberal.npz", **arrays)

    label_arr = arrays["labels"]
    summary = {
        "mode": "indel_v4",
        "checkpoint": args.checkpoint,
        "variant_h5": args.variant_h5,
        "delta_window_bp": args.delta_window,
        "stride": "n/a (single-chunk)",
        "total_length": total_length,
        "mask_canonical": args.mask_canonical,
        "annotation_path": args.annotation_path if args.mask_canonical else None,
        "transcript_annotation": args.transcript_annotation,
        "mask_site_source": dict(site_source),
        "mask_transcript_boundaries": args.mask_transcript_boundaries,
        "sequence_source": (
            "training_h5_x_ref_x_alt"
            if args.mask_transcript_boundaries else "genome_fasta"
        ),
        "n_total": len(variants),
        "n_indel": len(indels),
        "n_indel_all": n_all_indels,
        "row_start": args.row_start,
        "row_end": args.row_end,
        "n_indel_valid": len(valid_idx),
        "label_pos": int((label_arr == 1).sum()),
        "label_neg": int((label_arr == 0).sum()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if label_arr.size and label_arr.max() != label_arr.min():
        summary["auprc_delta_max"] = float(
            average_precision_score(label_arr, arrays["multichunk_delta_max"])
        )
        summary["auroc_delta_max"] = float(
            roc_auc_score(label_arr, arrays["multichunk_delta_max"])
        )
    with open(out_dir / "features_liberal.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %d indel features (v4) to %s", len(label_arr), out_dir)
    return summary


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    _evaluate(args)


if __name__ == "__main__":
    main()
