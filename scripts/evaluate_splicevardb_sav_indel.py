"""Single-chunk indel SAV evaluator using window-aggregated splice prob change.

For each indel variant (deletions, insertions, MNVs of unequal length):

1. Re-fetch the 15000-bp genomic chunk centered on the variant
   (variant at chunk-local position ``variant_offset``, typically 7500).
2. Build a length-15000 alt sequence: ``ref_seq[:clp] + alt + ref_seq[clp + L_ref:]``,
   then pad on the right with additional genome bases (for deletions) or
   truncate on the right (for insertions) to maintain length 15000.
3. Reverse-complement both for negative-strand variants.
4. Run the model on both, getting per-bp probabilities ``(L, 3)`` in
   channel order ``(acceptor, donor, neither)``.
5. Score each direction by the change in *max splice probability* in a
   window around the variant region. This is robust to the ref/alt
   length mismatch that breaks per-bp delta comparison for indels.

Output schema matches the SNV multichunk evaluator
(``features_liberal.npz`` with keys ``multichunk_delta_max``, etc.) so
the indel scores can be concatenated with SNV scores for ensemble
construction without code changes downstream.

Skips SNVs by design — they are scored by the per-bp-delta multichunk
evaluator. Run this only on H5s that contain indels (e.g., the raw
``splicevardb_train_with_offset.h5`` or the raw test H5).

Usage::

    python -m scripts.evaluate_splicevardb_sav_indel \\
        --checkpoint <path> --variant-h5 <h5> --output-dir <dir> \\
        --delta-window 2500 --batch-size 16
"""

from __future__ import annotations

import argparse
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
    _hyenadna_forward_probs,
    _load_hyenadna_model,
    _resolve_probability_mode,
)
from scripts.evaluate_splicevardb_sav_multichunk import (  # noqa: E402
    _decode_seq,
    _encode_seq,
    _load_variant_table,
    _resolve_fasta_chrom,
    _reverse_complement,
    _slice_window,
)

logger = logging.getLogger(__name__)

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
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variant-h5", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--genome-fasta",
                   default="./genome/GRCh38.fa")
    p.add_argument("--train-config", default=None)
    p.add_argument("--delta-window", type=int, default=2500,
                   help="Half-width (bp) around the variant region used "
                        "for the window-aggregated splice-prob score.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probability-mode", default=None,
                   choices=[None, "softmax", "sigmoid"])
    p.add_argument("--target-region-length", type=int, default=None)
    return p.parse_args()


def _build_indel_chunk(
    fasta: Fasta,
    chrom: str,
    pos_hg38: int,
    ref: str,
    alt: str,
    strand: str,
    total_length: int,
    clp: int,
) -> dict | None:
    """Return ``{x_ref, x_alt, var_start_model, var_end_ref_model, var_end_alt_model}``.

    Args:
        fasta (Fasta): pyfaidx FASTA handle.
        chrom (str): variant chromosome (e.g. ``chr1``).
        pos_hg38 (int): 1-based variant position on the + strand of hg38.
        ref (str): reference allele (any length).
        alt (str): alternate allele (any length).
        strand (str): ``+`` or ``-`` (gene orientation).
        total_length (int): chunk length (15000).
        clp (int): variant chunk-local position in input (genomic-+ frame).

    Returns:
        dict | None: encoded sequences and variant span coordinates in
            model space, or ``None`` if the genome window could not be
            built (ref mismatch / chromosome missing / length mismatch).
    """
    fa_chrom = _resolve_fasta_chrom(fasta, chrom)
    if fa_chrom is None:
        return None

    l_ref = len(ref)
    l_alt = len(alt)
    chunk_start_0b = (pos_hg38 - 1) - clp
    ref_seq = _slice_window(fasta, fa_chrom, chunk_start_0b, chunk_start_0b + total_length)
    if len(ref_seq) != total_length:
        return None
    if ref_seq[clp:clp + l_ref].upper() != ref.upper():
        return None

    raw_alt_seq = ref_seq[:clp] + alt.upper() + ref_seq[clp + l_ref:]
    if l_alt > l_ref:
        alt_seq = raw_alt_seq[:total_length]
    elif l_alt < l_ref:
        extra_needed = l_ref - l_alt
        extra = _slice_window(
            fasta,
            fa_chrom,
            chunk_start_0b + total_length,
            chunk_start_0b + total_length + extra_needed,
        )
        alt_seq = raw_alt_seq + extra
    else:
        alt_seq = raw_alt_seq
    if len(alt_seq) != total_length:
        return None

    if strand == "-":
        ref_final = _reverse_complement(ref_seq)
        alt_final = _reverse_complement(alt_seq)
        var_start_ref = total_length - clp - l_ref
        var_end_ref = total_length - clp
        var_start_alt = total_length - clp - l_alt
        var_end_alt = total_length - clp
    else:
        ref_final = ref_seq
        alt_final = alt_seq
        var_start_ref = clp
        var_end_ref = clp + l_ref
        var_start_alt = clp
        var_end_alt = clp + l_alt

    return {
        "x_ref": _encode_seq(ref_final),
        "x_alt": _encode_seq(alt_final),
        "var_start_ref": int(var_start_ref),
        "var_end_ref": int(var_end_ref),
        "var_start_alt": int(var_start_alt),
        "var_end_alt": int(var_end_alt),
    }


def _score_indel(
    prob_ref: np.ndarray,
    prob_alt: np.ndarray,
    var_start_ref: int,
    var_end_ref: int,
    var_start_alt: int,
    var_end_alt: int,
    delta_window: int,
    total_length: int,
) -> dict[str, float]:
    """Compute Option-A directional scores via max splice prob in window.

    Args:
        prob_ref (np.ndarray): ref probs ``(L, 3)`` in channel order
            ``(acc, don, nei)``.
        prob_alt (np.ndarray): alt probs ``(L, 3)``.
        var_start_ref (int): inclusive start of variant region in ref model space.
        var_end_ref (int): exclusive end of variant region in ref model space.
        var_start_alt (int): inclusive start of variant region in alt model space.
        var_end_alt (int): exclusive end of variant region in alt model space.
        delta_window (int): half-width of the scoring window.
        total_length (int): chunk length (e.g. 15000).

    Returns:
        dict[str, float]: same 7 feature keys as the SNV evaluator.
    """
    win_ref_start = max(0, var_start_ref - delta_window)
    win_ref_end = min(total_length, var_end_ref + delta_window)
    win_alt_start = max(0, var_start_alt - delta_window)
    win_alt_end = min(total_length, var_end_alt + delta_window)

    max_acc_ref = float(prob_ref[win_ref_start:win_ref_end, 0].max())
    max_don_ref = float(prob_ref[win_ref_start:win_ref_end, 1].max())
    max_acc_alt = float(prob_alt[win_alt_start:win_alt_end, 0].max())
    max_don_alt = float(prob_alt[win_alt_start:win_alt_end, 1].max())

    acc_gain = max(0.0, max_acc_alt - max_acc_ref)
    acc_loss = max(0.0, max_acc_ref - max_acc_alt)
    don_gain = max(0.0, max_don_alt - max_don_ref)
    don_loss = max(0.0, max_don_ref - max_don_alt)

    return {
        "delta_max": max(acc_gain, acc_loss, don_gain, don_loss),
        "delta_acc_max": max(acc_gain, acc_loss),
        "delta_don_max": max(don_gain, don_loss),
        "delta_acc_gain": acc_gain,
        "delta_acc_loss": acc_loss,
        "delta_don_gain": don_gain,
        "delta_don_loss": don_loss,
    }


def _batched_forward(
    model: torch.nn.Module,
    cfg: OmegaConf,
    seqs_int8: list[np.ndarray],
    device: torch.device,
    probability_mode: str,
    total_length: int,
    batch_size: int,
) -> np.ndarray:
    """Run model on a list of int8 sequences and return ``(N, L, 3)`` probs."""
    n = len(seqs_int8)
    out = np.zeros((n, total_length, 3), dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="Inference", leave=False):
        chunk = seqs_int8[i : i + batch_size]
        seqs_str = [_decode_seq(s) for s in chunk]
        probs = _hyenadna_forward_probs(
            model=model,
            seqs=seqs_str,
            cfg=cfg,
            device=device,
            probability_mode=probability_mode,
            target_length=total_length,
        )
        out[i : i + len(chunk)] = probs.astype(np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the indel-only SAV eval and persist artifacts.

    Args:
        args (argparse.Namespace): parsed CLI args.

    Returns:
        dict[str, object]: summary dict with per-variant counts and metrics.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading variant table from %s", args.variant_h5)
    variants = _load_variant_table(args.variant_h5)
    indels = [v for v in variants if not (len(v["ref"]) == 1 and len(v["alt"]) == 1)]
    n_snv = len(variants) - len(indels)
    logger.info(
        "Variants: total=%d snv=%d (skipped) indel=%d (evaluating)",
        len(variants),
        n_snv,
        len(indels),
    )

    if not indels:
        logger.warning("No indels in H5 — nothing to do.")
        return {"mode": "indel", "n_indel": 0}

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

    logger.info("Building indel chunks (n=%d)", len(indels))
    chunks: list[dict] = []
    skipped: list[tuple[int, str]] = []
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
        if rec is None:
            skipped.append((v["idx"], "chunk_build_failed"))
            chunks.append(None)
        else:
            chunks.append(rec)

    valid_idx = [i for i, c in enumerate(chunks) if c is not None]
    logger.info(
        "Indel chunks: built=%d failed=%d",
        len(valid_idx),
        len(chunks) - len(valid_idx),
    )

    refs = [chunks[i]["x_ref"] for i in valid_idx]
    alts = [chunks[i]["x_alt"] for i in valid_idx]
    logger.info("Running forward pass on %d ref + %d alt sequences", len(refs), len(alts))
    probs_ref = _batched_forward(
        model, cfg, refs, device, prob_mode, total_length, args.batch_size,
    )
    probs_alt = _batched_forward(
        model, cfg, alts, device, prob_mode, total_length, args.batch_size,
    )

    features: dict[str, list[float]] = {k: [] for k in _FEATURE_KEYS}
    labels: list[int] = []
    n_chunks_used: list[int] = []
    for j, i_var in enumerate(valid_idx):
        rec = chunks[i_var]
        scores = _score_indel(
            probs_ref[j],
            probs_alt[j],
            rec["var_start_ref"],
            rec["var_end_ref"],
            rec["var_start_alt"],
            rec["var_end_alt"],
            args.delta_window,
            total_length,
        )
        for k in _FEATURE_KEYS:
            features[k].append(scores[k])
        labels.append(indels[i_var]["label"])
        n_chunks_used.append(1)

    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(labels, dtype=np.int64),
        "n_chunks_used": np.asarray(n_chunks_used, dtype=np.int32),
    }
    for k in _FEATURE_KEYS:
        col = np.asarray(features[k], dtype=np.float32)
        arrays[f"multichunk_{k}"] = col
        arrays[f"baseline_{k}"] = col.copy()

    np.savez(out_dir / "features_liberal.npz", **arrays)

    label_arr = arrays["labels"]
    summary = {
        "mode": "indel",
        "checkpoint": args.checkpoint,
        "variant_h5": args.variant_h5,
        "genome_fasta": args.genome_fasta,
        "delta_window_bp": args.delta_window,
        "stride": "n/a (single-chunk)",
        "total_length": total_length,
        "n_total": len(variants),
        "n_snv_skipped": n_snv,
        "n_indel": len(indels),
        "n_indel_valid": len(valid_idx),
        "n_indel_failed": len(chunks) - len(valid_idx),
        "skipped_reasons": skipped[:20],
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
    logger.info("Wrote %d indel features to %s", len(label_arr), out_dir)
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
