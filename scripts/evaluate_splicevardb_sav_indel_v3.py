"""Indel SAV evaluator v3 — zero-compare for missing/inserted positions.

Extension of ``evaluate_splicevardb_sav_indel.py`` (Option A) with one
additional contribution per the user's proposal (2026-06-01):

* **Deletion** (``L_ref > L_alt``): the ``L_diff = L_ref - L_alt`` ref
  bases that no longer exist in alt contribute their per-position
  splice probabilities as candidate ``loss`` deltas
  (``|prob_ref[i] - 0| = prob_ref[i]``).
* **Insertion** (``L_alt > L_ref``): the ``L_diff`` alt bases that have
  no counterpart in ref contribute as candidate ``gain`` deltas
  (``|prob_alt[i] - 0| = prob_alt[i]``).

Per direction (acc, don), the final track is the max of:
  (a) Option A window-aggregated change ``max(prob_*[window]) - max(prob_*[window])``
  (b) Per-position prob value at the missing/inserted bases

Then ``delta_max`` is the max across the four directional tracks.
This honours the user's intuition: bases that explicitly disappear or
appear should be compared against the implicit zero of the missing
counterpart, not only via the window aggregate.

Output schema matches the legacy indel evaluator
(``features_liberal.npz`` with the same 7 ``multichunk_*`` keys + a
``baseline_*`` mirror) so downstream selection scripts work without
changes.
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
    p.add_argument("--delta-window", type=int, default=2500)
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
    """Return encoded ``x_ref`` / ``x_alt`` plus variant span coords.

    Identical to v2 ``_build_indel_chunk`` — both ref and alt are
    length ``total_length`` after right-side genome padding (deletion)
    or right-side truncation (insertion). For ``-`` strand the chunk is
    reverse-complemented so the variant region in model coords is at
    ``[total_length - clp - L_*, total_length - clp)``.
    """
    fa_chrom = _resolve_fasta_chrom(fasta, chrom)
    if fa_chrom is None:
        return None
    l_ref = len(ref)
    l_alt = len(alt)
    chunk_start_0b = (pos_hg38 - 1) - clp
    ref_seq = _slice_window(
        fasta, fa_chrom, chunk_start_0b, chunk_start_0b + total_length
    )
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
        "l_ref": int(l_ref),
        "l_alt": int(l_alt),
        "strand": strand,
    }


def _score_indel_v3(
    prob_ref: np.ndarray,
    prob_alt: np.ndarray,
    rec: dict,
    delta_window: int,
    total_length: int,
) -> dict[str, float]:
    """Compute v3 directional scores (Option A + missing/inserted contribution).

    For each splice channel (acceptor=0, donor=1):

    * ``window_change`` = ``max(prob_alt[win_alt]) - max(prob_ref[win_ref])``
      where the windows extend ``delta_window`` bp around each variant
      region. Positive values become gain, negative become loss.
    * ``missing_contribution`` (deletion only): for each ref position
      that has no alt counterpart (the last ``L_diff = L_ref - L_alt``
      bases of the ref variant region in gene-reading direction),
      take the max splice prob at those positions. This adds to ``loss``
      because alt effectively has 0 there.
    * ``inserted_contribution`` (insertion only): for each alt position
      that has no ref counterpart (the last ``L_diff = L_alt - L_ref``
      bases of the alt variant region), take the max splice prob at
      those positions. This adds to ``gain``.

    The final per-direction score is the max of the window-aggregated
    component and the missing/inserted contribution.

    Args:
        prob_ref: ref probs ``(L, 3)`` in channel order ``(acc, don, nei)``.
        prob_alt: alt probs ``(L, 3)``.
        rec: dict from ``_build_indel_chunk`` (has variant span +
            ``l_ref`` / ``l_alt`` / ``strand``).
        delta_window: half-width (bp) of the window for Option A.
        total_length: chunk length (e.g. 15000).

    Returns:
        dict[str, float]: same 7 feature keys as the SNV evaluator.
    """
    var_start_ref = rec["var_start_ref"]
    var_end_ref = rec["var_end_ref"]
    var_start_alt = rec["var_start_alt"]
    var_end_alt = rec["var_end_alt"]
    l_ref = rec["l_ref"]
    l_alt = rec["l_alt"]
    strand = rec["strand"]

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

    if l_ref > l_alt:
        l_diff = l_ref - l_alt
        if strand == "+":
            miss_start = var_start_ref + l_alt
            miss_end = var_end_ref
        else:
            miss_start = var_start_ref
            miss_end = var_start_ref + l_diff
        if miss_end > miss_start:
            miss_acc = float(prob_ref[miss_start:miss_end, 0].max())
            miss_don = float(prob_ref[miss_start:miss_end, 1].max())
            acc_loss = max(acc_loss, miss_acc)
            don_loss = max(don_loss, miss_don)

    elif l_alt > l_ref:
        l_diff = l_alt - l_ref
        if strand == "+":
            ins_start = var_start_alt + l_ref
            ins_end = var_end_alt
        else:
            ins_start = var_start_alt
            ins_end = var_start_alt + l_diff
        if ins_end > ins_start:
            ins_acc = float(prob_alt[ins_start:ins_end, 0].max())
            ins_don = float(prob_alt[ins_start:ins_end, 1].max())
            acc_gain = max(acc_gain, ins_acc)
            don_gain = max(don_gain, ins_don)

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
    """Run model on int8 sequences and return ``(N, L, 3)`` probabilities."""
    n = len(seqs_int8)
    out = np.zeros((n, total_length, 3), dtype=np.float32)
    # The CPU-servable model has no tokenizer: it consumes a one-hot of the
    # same int8 encoding directly. Imported lazily so this module keeps no
    # import-time dependency on the multichunk scorer.
    for i in tqdm(range(0, n, batch_size), desc="Inference", leave=False):
        chunk = seqs_int8[i:i + batch_size]
        seqs_str = [_decode_seq(s) for s in chunk]
        probs = _hyenadna_forward_probs(
            model=model,
            seqs=seqs_str,
            cfg=cfg,
            device=device,
            probability_mode=probability_mode,
            target_length=total_length,
        )
        out[i:i + len(chunk)] = probs.astype(np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the v3 indel SAV evaluator and persist artifacts."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading variant table from %s", args.variant_h5)
    variants = _load_variant_table(args.variant_h5)
    indels = [v for v in variants if not (len(v["ref"]) == 1 and len(v["alt"]) == 1)]
    logger.info("Variants: total=%d indel=%d", len(variants), len(indels))
    if not indels:
        logger.warning("No indels — nothing to do")
        return {"mode": "indel_v3", "n_indel": 0}

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
        model, cfg, refs, device, prob_mode, total_length, args.batch_size
    )
    probs_alt = _batched_forward(
        model, cfg, alts, device, prob_mode, total_length, args.batch_size
    )

    features: dict[str, list[float]] = {k: [] for k in _FEATURE_KEYS}
    labels: list[int] = []
    for j, i_var in enumerate(valid_idx):
        rec = chunks[i_var]
        scores = _score_indel_v3(
            probs_ref[j], probs_alt[j], rec, args.delta_window, total_length,
        )
        for k in _FEATURE_KEYS:
            features[k].append(scores[k])
        labels.append(indels[i_var]["label"])

    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(labels, dtype=np.int64),
        "n_chunks_used": np.full(len(labels), 1, dtype=np.int32),
    }
    for k in _FEATURE_KEYS:
        col = np.asarray(features[k], dtype=np.float32)
        arrays[f"multichunk_{k}"] = col
        arrays[f"baseline_{k}"] = col.copy()
    np.savez(out_dir / "features_liberal.npz", **arrays)

    label_arr = arrays["labels"]
    summary = {
        "mode": "indel_v3",
        "checkpoint": args.checkpoint,
        "variant_h5": args.variant_h5,
        "delta_window_bp": args.delta_window,
        "stride": "n/a (single-chunk)",
        "total_length": total_length,
        "n_total": len(variants),
        "n_indel": len(indels),
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
    logger.info("Wrote %d indel features (v3) to %s", len(label_arr), out_dir)
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
