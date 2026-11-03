"""Per-nt cache from K=4 anchored chunks for the head and delta max.

Only each chunk's supervised target contributes. At each position the chunk
with the largest acceptor/donor probability change supplies both ref and alt;
``coverage`` records how many target regions contributed. Optional canonical
masking clamps impossible directions in every SNV and indel chunk *before*
this selection, so a masked delta max is exact rather than a lower bound.

Window
------
A chunk whose variant sits at chunk-local position ``clp`` contributes the
genomic span ``[v-clp+5000, v-clp+9999]``. The four clps are congruent mod
the stride, so the union is 8,750 long but slides with
``variant_offset % stride``. The fixed head/cache window is
``[v-4375, v+4375)``; coverage is zero where that fixed window extends
beyond the four target regions.

Pairing
-------
ref and alt are NOT maxed independently. Taking each channel's max separately
lets a position's ref come from one chunk and its alt from another, so the
implied delta exists in no chunk at all. Instead one chunk is chosen per
position -- the one with the largest 4-direction |delta| there, matching what
delta_max would pick -- and both its ref and its alt are carried over.
``--merge mean`` instead averages ref and alt over every chunk covering the
position, so each position's delta is the chunks' mean delta.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from scripts.canonical_site_track import (
    ANNOTATION,
    load_canonical_sites,
    load_sites_by_accession,
    load_variant_transcripts,
    resolve_sites,
)
from scripts.deltamax_from_multichunk_cache import row_to_ref_genomic

logger = logging.getLogger(__name__)

# Supervised target of a chunk, in chunk-local coordinates: the centred
# ``target_length`` of a ``total_length`` input. These module values are the
# 15,000 nt / 5,000 nt layout every cache built so far used; a backbone trained
# at another geometry passes its own via ``--total-length`` / ``--target-length``
# (32,768 / 16,384 for the long-context arm) and they are threaded through
# rather than read from here.
TARGET_START, TARGET_END = 5000, 10000


def target_window(total_length: int, target_length: int) -> tuple[int, int]:
    """Chunk-local bounds of the supervised target.

    Args:
        total_length (int): Model input length.
        target_length (int): Supervised target length, centred in the input.

    Returns:
        tuple[int, int]: ``(start, end)``, end exclusive.
    """
    start = (int(total_length) - int(target_length)) // 2
    return start, start + int(target_length)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-h5", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--model-type",
        choices=("hyenadna", "dlm_qwen"),
        default="hyenadna",
        help="Checkpoint flavour; see the same flag on "
             "scripts/evaluate_splicevardb_sav_multichunk.py. The chunk grid, "
             "pairing and window are shared, so both are cached identically.",
    )
    p.add_argument("--out-h5", required=True)
    p.add_argument("--train-config", default=None)
    p.add_argument("--ref-fasta", default="./genome/GRCh38.fa")
    p.add_argument("--stride", type=int, default=1250)
    p.add_argument("--total-length", type=int, default=15000)
    p.add_argument("--target-length", type=int, default=5000,
                   help="supervised target length, centred in --total-length. "
                        "Sets which chunks are kept (variant inside the target) "
                        "and each chunk's span on the genomic axis. Must match "
                        "the checkpoint's training geometry: 5000 for the "
                        "15,000 nt backbones, 16384 for the 32,768 nt arm.")
    p.add_argument("--window-left", type=int, default=4375)
    p.add_argument("--window-right", type=int, default=4375,
                   help="positions from the variant onward, "
                        "inclusive; 4375 gives 8750 total with "
                        "the extra base on the left")
    p.add_argument("--mask-canonical", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Clamp impossible acceptor/donor directions in all "
                        "SNV and indel chunks before position-wise merging")
    p.add_argument("--annotation-path", default=ANNOTATION)
    p.add_argument("--transcript-annotation", default=None)
    p.add_argument("--variant-transcript-csv", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=500,
                   help="variants per persisted block")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument(
        "--chunker-boundary-mode", action="append", default=[], metavar="STAGE=MODE",
        help="inference-only override of a dynamic-chunker stage's boundary rule: "
             "learned (the checkpoint's), fixed (a boundary every "
             "round(1/target_chunk_rate) positions; hnet stages then skip the "
             "dechunker EMA) or learned_nosmooth (hnet: learned boundaries "
             "without the EMA); repeat per stage",
    )
    p.add_argument("--merge", choices=("select", "mean"), default="select",
                   help="per-position merge of the chunks' targets: the chunk "
                        "with the largest change (select) or their mean (mean)")
    p.add_argument("--chunk-summary-out", default=None,
                   help="also write each chunk's delta max and four directional "
                        "peaks per variant to this .npz (see chunk_summaries and "
                        "chunk_direction_peaks); the cache is unchanged")
    return p.parse_args()


def apply_chunker_boundary_modes(model: object, specs: list[str]) -> list[str]:
    """Set the boundary rule of the chunker stages named in ``specs``.

    Args:
        model (object): Loaded model with a ``hierarchical_chunker``.
        specs (list[str]): ``STAGE=MODE`` items. MODE is ``learned`` (the
            checkpoint's rule), ``fixed`` (a boundary every
            ``round(1 / target_chunk_rate)`` positions; an ``hnet`` stage then
            dechunks without its EMA), or ``learned_nosmooth`` (``hnet`` only:
            the learned boundaries, dechunked without the EMA).

    Returns:
        list[str]: The applied items, normalised as ``STAGE=MODE``.

    Raises:
        ValueError: On a malformed item, an unknown mode, a stage the model
            does not have, or a model without a dynamic chunker.
    """
    if not specs:
        return []
    chunker = getattr(model, "hierarchical_chunker", None)
    if chunker is None:
        raise ValueError("--chunker-boundary-mode needs a model with a dynamic chunker")
    applied = []
    for spec in specs:
        stage, sep, mode = spec.partition("=")
        if (
            not sep
            or not stage.isdigit()
            or mode not in ("learned", "fixed", "learned_nosmooth")
        ):
            raise ValueError(
                "--chunker-boundary-mode expects STAGE=learned|fixed|"
                f"learned_nosmooth, got {spec!r}"
            )
        idx = int(stage)
        if idx >= len(chunker.chunkers):
            raise ValueError(
                f"stage {idx} requested, the chunker has {len(chunker.chunkers)}"
            )
        stage_chunker = chunker.chunkers[idx]
        if mode == "learned_nosmooth":
            if getattr(stage_chunker, "chunker_style", "legacy") != "hnet":
                raise ValueError(
                    f"stage {idx}: learned_nosmooth needs an hnet-style chunker"
                )
            stage_chunker.boundary_mode = "learned"
            stage_chunker.hnet_smoothing = False
        else:
            stage_chunker.boundary_mode = mode
        applied.append(f"{idx}={mode}")
        logger.info("chunker stage %d boundary rule at inference: %s", idx, mode)
    return applied


def mask_chunk_alt(
    ref: np.ndarray,
    alt: np.ndarray,
    *,
    pos: int,
    ref_allele: str,
    alt_allele: str,
    minus: bool,
    var_idx: int,
    sites: dict,
) -> np.ndarray:
    """Clamp impossible directions in one raw or v4-aligned chunk."""
    genomic = row_to_ref_genomic(
        pos, ref_allele, alt_allele, minus, len(ref), var_idx,
    )
    out = alt.copy()
    for channel, key in enumerate(("acc_exonic", "don_exonic")):
        annotated = np.isin(genomic, sites[key])
        change = out[:, channel] - ref[:, channel]
        impossible = (annotated & (change > 0)) | (~annotated & (change < 0))
        out[impossible, channel] = ref[impossible, channel]
    return out


def collapse_variant(
    ref_by_chunk: np.ndarray,
    alt_by_chunk: np.ndarray,
    clps: list[int],
    var_indices: list[int],
    window_left: int = 4375,
    window_right: int = 4375,
    target_start: int = TARGET_START,
    target_end: int = TARGET_END,
    merge: str = "select",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse one variant's chunks onto the genomic axis.

    Args:
        ref_by_chunk (np.ndarray): ``(K, total_length, C)`` reference probs.
        alt_by_chunk (np.ndarray): ``(K, total_length, C)`` alternate probs.
        clps (list[int]): each chunk's variant position, chunk-local. Sets
            which slice of the chunk is its supervised target.
        var_indices (list[int]): index of the variant within each track. Equal
            to ``clp`` for a raw chunk; for a v4-aligned indel track it is the
            number of upstream rows, since alignment re-bases the track on the
            variant.
        window_left (int): positions kept before the variant.
        window_right (int): positions from the variant onward, inclusive.
        target_start (int): chunk-local start of the supervised target.
        target_end (int): chunk-local end of the supervised target, exclusive.
        merge (str): ``select`` carries the ref and alt of the chunk with the
            largest change at each position; ``mean`` averages ref and alt over
            every chunk whose target covers the position, so the change is the
            chunks' mean change.

    Returns:
        tuple: ``(ref, alt, coverage)`` of shapes ``(W, C)``, ``(W, C)`` and
            ``(W,)`` where ``W = window_left + window_right``.
    """
    if merge not in ("select", "mean"):
        raise ValueError(f"unknown merge {merge!r}")
    w = window_left + window_right
    n_ch = ref_by_chunk.shape[-1]
    ref = np.zeros((w, n_ch), dtype=np.float32)
    alt = np.zeros((w, n_ch), dtype=np.float32)
    cov = np.zeros(w, dtype=np.int8)
    best = np.full(w, -np.inf, dtype=np.float32)

    for k, (clp, var_idx) in enumerate(zip(clps, var_indices)):
        # This chunk's supervised target, in coordinates relative to the
        # variant. Identical for SNVs and indels -- what differs is only where
        # the variant sits inside the track: at clp for a raw chunk, at the
        # upstream-row count for a v4-aligned indel track.
        lo_rel, hi_rel = target_start - clp, target_end - 1 - clp
        lo = max(lo_rel, -window_left)
        hi = min(hi_rel, window_right - 1)
        if lo > hi:
            continue
        n_track = ref_by_chunk.shape[1]
        lo = max(lo, -var_idx)
        hi = min(hi, n_track - 1 - var_idx)
        if lo > hi:
            continue
        src = slice(lo + var_idx, hi + var_idx + 1)
        dst = slice(lo + window_left, hi + window_left + 1)
        r = ref_by_chunk[k, src]
        a = alt_by_chunk[k, src]
        cov[dst] += 1
        if merge == "mean":
            ref[dst] += r
            alt[dst] += a
            continue
        # Pick per position by the same quantity delta_max ranks on: the
        # largest one-sided change in either direction. Acceptor and donor
        # only -- the third channel is `neither`, which sits near 1.0 almost
        # everywhere, so a v4 zero-compare row (a deleted or inserted base
        # paired against a null probability) reads as a ~1.0 change there and
        # would win every position of every indel. delta_max scores the same
        # four directions off these two channels.
        d = np.abs(a[..., :2] - r[..., :2]).max(axis=-1)
        take = d > best[dst]
        idx = np.arange(dst.start, dst.stop)[take]
        ref[idx] = r[take]
        alt[idx] = a[take]
        best[dst] = np.maximum(best[dst], d)
    if merge == "mean":
        covered = cov > 0
        ref[covered] /= cov[covered, None]
        alt[covered] /= cov[covered, None]
    return ref, alt, cov


def chunk_summaries(
    ref_by_chunk: np.ndarray,
    alt_by_chunk: np.ndarray,
    clps: list[int],
    var_indices: list[int],
    window_left: int = 4375,
    window_right: int = 4375,
    target_start: int = TARGET_START,
    target_end: int = TARGET_END,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each chunk's delta max over its own supervised target.

    The per-position quantity is the one ``collapse_variant`` selects on, the
    larger of |delta acceptor| and |delta donor|. ``window`` clips each target
    to the cache window as the merge does, so its maximum over chunks is the
    merged track's delta max. ``target`` keeps the whole supervised target --
    the delta max of a best-chunk track, built from one chunk only.

    Args:
        ref_by_chunk (np.ndarray): ``(K, total_length, C)`` reference probs.
        alt_by_chunk (np.ndarray): ``(K, total_length, C)`` alternate probs.
        clps (list[int]): each chunk's variant position, chunk-local.
        var_indices (list[int]): index of the variant within each track.
        window_left (int): positions kept before the variant.
        window_right (int): positions from the variant onward, inclusive.
        target_start (int): chunk-local start of the supervised target.
        target_end (int): chunk-local end of the supervised target, exclusive.

    Returns:
        tuple: ``(dmax_window, dmax_target, argmax_rel)``, each ``(K,)``;
            ``argmax_rel`` is the target maximum's position relative to the
            variant. NaN and the int32 minimum mark a chunk with no positions.
    """
    k = len(clps)
    dmax_window = np.full(k, np.nan, dtype=np.float32)
    dmax_target = np.full(k, np.nan, dtype=np.float32)
    argmax_rel = np.full(k, np.iinfo(np.int32).min, dtype=np.int32)
    n_track = ref_by_chunk.shape[1]
    for i, (clp, var_idx) in enumerate(zip(clps, var_indices)):
        lo = max(target_start - clp, -var_idx)
        hi = min(target_end - 1 - clp, n_track - 1 - var_idx)
        if lo > hi:
            continue
        src = slice(lo + var_idx, hi + var_idx + 1)
        d = np.abs(alt_by_chunk[i, src, :2] - ref_by_chunk[i, src, :2]).max(axis=-1)
        j = int(np.argmax(d))
        dmax_target[i] = d[j]
        argmax_rel[i] = lo + j
        w_lo, w_hi = max(lo, -window_left), min(hi, window_right - 1)
        if w_lo <= w_hi:
            dmax_window[i] = d[w_lo - lo:w_hi - lo + 1].max()
    return dmax_window, dmax_target, argmax_rel


def chunk_direction_peaks(
    ref_by_chunk: np.ndarray,
    alt_by_chunk: np.ndarray,
    clps: list[int],
    var_indices: list[int],
    target_start: int = TARGET_START,
    target_end: int = TARGET_END,
) -> tuple[np.ndarray, np.ndarray]:
    """Each chunk's four directional peaks over its own supervised target.

    A consequence rule pairs a loss with the gain that compensates it, so a
    best-chunk track needs every direction's peak and position from the one
    chosen chunk. ``chunk_summaries`` keeps only the larger channel; this keeps
    acceptor gain, acceptor loss, donor gain and donor loss separately, in the
    order ``deltamax_from_multichunk_cache.DIRECTIONS`` uses.

    Args:
        ref_by_chunk (np.ndarray): ``(K, total_length, C)`` reference probs.
        alt_by_chunk (np.ndarray): ``(K, total_length, C)`` alternate probs.
        clps (list[int]): each chunk's variant position, chunk-local.
        var_indices (list[int]): index of the variant within each track.
        target_start (int): chunk-local start of the supervised target.
        target_end (int): chunk-local end of the supervised target, exclusive.

    Returns:
        tuple: ``(peak, rel)``, each ``(K, 4)``; ``rel`` is each peak's position
            relative to the variant. NaN and the int32 minimum mark a chunk
            with no positions.
    """
    k = len(clps)
    peak = np.full((k, 4), np.nan, dtype=np.float32)
    rel = np.full((k, 4), np.iinfo(np.int32).min, dtype=np.int32)
    n_track = ref_by_chunk.shape[1]
    for i, (clp, var_idx) in enumerate(zip(clps, var_indices)):
        lo = max(target_start - clp, -var_idx)
        hi = min(target_end - 1 - clp, n_track - 1 - var_idx)
        if lo > hi:
            continue
        src = slice(lo + var_idx, hi + var_idx + 1)
        d = alt_by_chunk[i, src, :2] - ref_by_chunk[i, src, :2]
        stacked = np.stack([
            np.clip(d[:, 0], 0, None),    # acceptor gain
            np.clip(-d[:, 0], 0, None),   # acceptor loss
            np.clip(d[:, 1], 0, None),    # donor gain
            np.clip(-d[:, 1], 0, None),   # donor loss
        ], axis=1)
        j = stacked.argmax(axis=0)
        peak[i] = stacked[j, np.arange(4)]
        rel[i] = lo + j
    return peak, rel


def main() -> None:
    """Build the multichunk cache for a variant table."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    args = parse_args()
    tgt_start, tgt_end = target_window(args.total_length, args.target_length)
    logger.info(
        "geometry: total_length=%d target=[%d, %d) (%d nt) stride=%d",
        args.total_length, tgt_start, tgt_end, args.target_length, args.stride,
    )
    import torch
    from pyfaidx import Fasta

    from evaluate.evaluate_splicevardb_sav import _load_hyenadna_model
    from scripts.evaluate_splicevardb_sav_indel import _build_indel_chunk
    from scripts.evaluate_splicevardb_sav_indel_v4 import _align_indel_probs
    from scripts.evaluate_splicevardb_sav_multichunk import (
        _aligned_chunk_positions,
        _batched_forward,
        _generate_chunks_for_variant,
        _load_dlm_qwen_model,
        _load_variant_table,
        _resolve_probability_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = _load_variant_table(args.variant_h5)
    end = len(variants) if args.end < 0 else min(args.end, len(variants))
    variants = variants[args.start:end]
    logger.info("variants %d..%d (%d)", args.start, end, len(variants))
    genes: dict[str, dict] = {}
    by_tx: dict[str, dict] = {}
    var_tx: dict[str, str] = {}
    if args.mask_canonical:
        if bool(args.transcript_annotation) != bool(args.variant_transcript_csv):
            raise ValueError("transcript annotation and variant CSV must be paired")
        genes = load_canonical_sites(args.annotation_path)
        missing = sorted({v.get("source_gene_name", "") for v in variants
                          if v.get("source_gene_name", "") not in genes})
        if missing:
            raise ValueError(
                f"{len(missing)} anchoring genes lack canonical sites: {missing[:5]}"
            )
        if args.transcript_annotation:
            by_tx = load_sites_by_accession(args.transcript_annotation)
            var_tx = load_variant_transcripts(args.variant_transcript_csv)

    if args.model_type == "dlm_qwen":
        model, _ = _load_dlm_qwen_model(args.checkpoint, device)
        cfg = None
        prob_mode = "softmax"
    else:
        model, cfg = _load_hyenadna_model(
            checkpoint_path=args.checkpoint,
            train_config_path=args.train_config, device=device,
        )
        prob_mode = _resolve_probability_mode(cfg, "auto")
    boundary_modes = apply_chunker_boundary_modes(model, args.chunker_boundary_mode)
    fasta = Fasta(args.ref_fasta,
                  sequence_always_upper=True)
    w = args.window_left + args.window_right
    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("window=%d (-%d..+%d), out=%s",
                w, args.window_left, args.window_right - 1, out_path)
    logger.info("ref/alt come from the SAME chunk at each position; maxing "
                "them independently would pair values from different chunks.")

    n = len(variants)
    n_ch = 3
    summary = None
    if args.chunk_summary_out:
        k_max = -(-args.target_length // args.stride)
        summary = {
            "clp": np.full((n, k_max), -1, dtype=np.int32),
            "dmax_window": np.full((n, k_max), np.nan, dtype=np.float32),
            "dmax_target": np.full((n, k_max), np.nan, dtype=np.float32),
            "argmax_rel": np.full((n, k_max), np.iinfo(np.int32).min, dtype=np.int32),
            "dir_peak_target": np.full((n, k_max, 4), np.nan, dtype=np.float32),
            "dir_rel_target": np.full((n, k_max, 4), np.iinfo(np.int32).min,
                                      dtype=np.int32),
        }
    with h5py.File(out_path, "w") as out:
        d_ref = out.create_dataset("ref_probs", shape=(n, w, n_ch),
                                   dtype=np.float16, chunks=(1, w, n_ch))
        d_alt = out.create_dataset("alt_probs", shape=(n, w, n_ch),
                                   dtype=np.float16, chunks=(1, w, n_ch))
        d_cov = out.create_dataset("coverage", shape=(n, w),
                                   dtype=np.int8, chunks=(1, w))
        d_k = out.create_dataset("n_chunks_used", shape=(n,), dtype=np.uint8)
        out.create_dataset("labels",
                           data=np.array([v["label"] for v in variants],
                                         dtype=np.int8))
        out.create_dataset("row_index",
                           data=np.array([v["idx"] for v in variants],
                                         dtype=np.int64))
        out.attrs["window_left"] = args.window_left
        out.attrs["window_right"] = args.window_right
        out.attrs["stride"] = args.stride
        out.attrs["total_length"] = args.total_length
        out.attrs["target_length"] = args.target_length
        out.attrs["pairing"] = (
            "per-position mean over covering chunks, ref+alt together"
            if args.merge == "mean"
            else "per-position argmax|alt-ref|, ref+alt together"
        )
        out.attrs["mask_canonical"] = bool(args.mask_canonical)
        out.attrs["mask_application"] = "premerge" if args.mask_canonical else "none"
        out.attrs["annotation_path"] = args.annotation_path if args.mask_canonical else ""
        out.attrs["chunker_boundary_modes"] = ",".join(boundary_modes)
        out.attrs["batch_size"] = args.batch_size
        n_trunc = 0

        for lo in range(0, n, args.block_size):
            hi = min(lo + args.block_size, n)
            block = variants[lo:hi]
            seqs: list[np.ndarray] = []
            index: list[tuple[int, int, int]] = []   # (i_local, k, clp)
            for i_local, v in enumerate(block):
                clps = [
                    c for c in _aligned_chunk_positions(
                        variant_offset=v["variant_offset"],
                        total_length=args.total_length, stride=args.stride)
                    if tgt_start <= c < tgt_end
                ]
                if max(len(v["ref"]), len(v["alt"])) > args.window_right:
                    n_trunc += 1
                is_snv = len(v["ref"]) == 1 and len(v["alt"]) == 1
                if is_snv:
                    fwd = ([args.total_length - 1 - c for c in clps]
                           if v["strand"] == "-" else clps)
                    recs = _generate_chunks_for_variant(
                        fasta=fasta, chrom=v["chrom"], pos_hg38=v["pos"],
                        ref=v["ref"], alt=v["alt"], strand=v["strand"],
                        total_length=args.total_length,
                        chunk_local_positions=fwd,
                    )
                    for r in recs:
                        if r["status"] != "ok":
                            continue
                        index.append((i_local, len(seqs), r["clp_model"], None))
                        seqs.append(r["x_ref"])
                        seqs.append(r["x_alt"])
                    continue
                # Indels need _build_indel_chunk: _generate_chunks_for_variant
                # compares ref_seq[clp], a single character, against the whole
                # ref allele, so every multi-base allele is rejected as a
                # ref_mismatch. The rec is carried through so the probabilities
                # can be put back on ref coordinates after the forward.
                for c in clps:
                    fwd_c = (args.total_length - 1 - c
                             if v["strand"] == "-" else c)
                    rec = _build_indel_chunk(
                        fasta=fasta, chrom=v["chrom"], pos_hg38=v["pos"],
                        ref=v["ref"], alt=v["alt"], strand=v["strand"],
                        total_length=args.total_length, clp=fwd_c,
                    )
                    if rec is None:
                        continue
                    rec = dict(rec)
                    rec["l_ref"] = len(v["ref"])
                    rec["l_alt"] = len(v["alt"])
                    rec["strand"] = v["strand"]
                    index.append((i_local, len(seqs), c, rec))
                    seqs.append(rec["x_ref"])
                    seqs.append(rec["x_alt"])

            probs = _batched_forward(
                model=model, cfg=cfg, seqs_int8=seqs, device=device,
                probability_mode=prob_mode, total_length=args.total_length,
                batch_size=args.batch_size,
            )
            sites_for_block = [
                resolve_sites(
                    genes, by_tx, var_tx, chrom=v["chrom"], pos=v["pos"],
                    ref=v["ref"], alt=v["alt"], gene=v["source_gene_name"],
                )[0] for v in block
            ] if args.mask_canonical else []
            per: dict[int, list] = {}
            for i_local, base, clp, rec in index:
                p_ref, p_alt = probs[base], probs[base + 1]
                if rec is not None:
                    # Put alt back on ref coordinates. Without this the whole
                    # downstream of an indel is compared against shifted
                    # sequence: measured on the current single-anchor cache,
                    # |delta| downstream of an indel runs 20-30x its upstream
                    # level, against 1.3x for SNVs. Extra bases on either side
                    # are zero-compared, matching v4.
                    p_ref, p_alt = _align_indel_probs(
                        p_ref, p_alt, rec,
                        delta_window=args.total_length // 2,
                        total_length=args.total_length,
                    )
                var_idx = clp if rec is None else min(
                    args.total_length // 2,
                    rec["var_start_ref"], rec["var_start_alt"],
                )
                if args.mask_canonical:
                    v = block[i_local]
                    p_alt = mask_chunk_alt(
                        p_ref, p_alt, pos=v["pos"], ref_allele=v["ref"],
                        alt_allele=v["alt"], minus=v["strand"] == "-",
                        var_idx=var_idx, sites=sites_for_block[i_local],
                    )
                per.setdefault(i_local, []).append(
                    (p_ref, p_alt, clp, var_idx))
            for i_local in range(len(block)):
                items = per.get(i_local, [])
                d_k[lo + i_local] = len(items)
                if not items:
                    continue
                n_min = min(t[0].shape[0] for t in items)
                r_stack = np.stack([t[0][:n_min] for t in items])
                a_stack = np.stack([t[1][:n_min] for t in items])
                clps = [t[2] for t in items]
                var_idx = [t[3] for t in items]
                r, a, cov = collapse_variant(
                    r_stack, a_stack, clps, var_idx,
                    args.window_left, args.window_right,
                    tgt_start, tgt_end, merge=args.merge,
                )
                d_ref[lo + i_local] = r.astype(np.float16)
                d_alt[lo + i_local] = a.astype(np.float16)
                d_cov[lo + i_local] = cov
                if summary is not None:
                    win, tgt, arg = chunk_summaries(
                        r_stack, a_stack, clps, var_idx,
                        args.window_left, args.window_right, tgt_start, tgt_end,
                    )
                    k = len(clps)
                    summary["clp"][lo + i_local, :k] = clps
                    summary["dmax_window"][lo + i_local, :k] = win
                    summary["dmax_target"][lo + i_local, :k] = tgt
                    summary["argmax_rel"][lo + i_local, :k] = arg
                    peak, rel = chunk_direction_peaks(
                        r_stack, a_stack, clps, var_idx, tgt_start, tgt_end,
                    )
                    summary["dir_peak_target"][lo + i_local, :k] = peak
                    summary["dir_rel_target"][lo + i_local, :k] = rel
            logger.info("block %d..%d done", lo, hi)

        out.attrs["n_truncated_indel"] = n_trunc
        logger.info("%d variants had an indel wider than the right window; "
                    "their tail is zero-filled, as the v4 scoring path also "
                    "does, and coverage marks it", n_trunc)

    if summary is not None:
        np.savez(
            args.chunk_summary_out,
            row_index=np.array([v["idx"] for v in variants], dtype=np.int64),
            labels=np.array([v["label"] for v in variants], dtype=np.int8),
            target_start=tgt_start, target_end=tgt_end,
            window_left=args.window_left, window_right=args.window_right,
            **summary,
        )
        logger.info("per-chunk summaries -> %s", args.chunk_summary_out)


if __name__ == "__main__":
    main()
