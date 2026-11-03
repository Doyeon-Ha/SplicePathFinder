"""SpliceVarDB SAV (Splice-Altering Variant) prediction evaluation.

For each variant in a SpliceVarDB H5 file (``X_ref``, ``X_alt``, ``label``):

1. Run the model on ``X_ref`` and ``X_alt`` to obtain per-position
   acceptor/donor probabilities over the target window.
2. Compute a SpliceAI-style delta score:

       delta = max over [-W, +W] window around the variant of
               max(|P_alt(acc) - P_ref(acc)|,
                   |P_alt(don) - P_ref(don)|)

3. Compare the per-variant delta score against the binary label
   (1=SAV, 0=nonSAV) using AUROC and AUPRC.

Three backends are supported, dispatched by ``--backend``:

- ``hyenadna`` — IssaSplice-style HyenaDNA / Caduceus checkpoints.
  Uses the same ``train_config`` resolution as ``evaluate.evaluate``.
- ``spliceai`` — Plain SpliceAI checkpoints (``model.spliceai.SpliceAI``).
  Uses one-hot encoding via :func:`scripts.spliceai_pipeline.seq_to_onehot`.
- ``spliceai-v1`` — SpliceAI checkpoints from the v1 training pipeline
  (``input_conv`` / ``residual_blocks.*`` / ``output_conv`` state-dict
  layout, e.g. ``spliceai-original-h5-v1``). Uses :class:`model.spliceai_v1.SpliceAIV1`.
"""

import argparse
import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

DECODE_TABLE = np.array(["N", "A", "C", "G", "T"])

# Eval-time batch size is fixed to keep delta scores reproducible.
# Mamba2 / SSM kernels under bf16 autocast produce batch-shape-dependent
# numerics, so changing this constant between runs is enough to shift
# AUROC / AUPRC by ~0.002. Edit the constant if hardware forces a change,
# but do not expose it as a CLI flag.
_FIXED_BATCH_SIZE: int = 16

_DEFAULT_DELTA_WINDOW: int = 50

_LOSS_TYPE_TO_PROB_MODE = {
    "multi_class_focal_loss": "softmax",
    "weighted_ce": "softmax",
    "weighted_bce": "sigmoid",
    "multi_label_focal_loss": "sigmoid",
}
_DEFAULT_PROB_MODE = "softmax"


# ── CLI ─────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SAV evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="SpliceVarDB SAV evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hyenadna", "spliceai", "spliceai-v1"],
        required=True,
        help="Inference backend.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--test-h5-path",
        type=str,
        required=True,
        help="Path to SpliceVarDB H5 file (X_ref/X_alt/label).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to write per-variant scores + summary metrics.",
    )
    # ``--batch-size`` intentionally omitted: see ``_FIXED_BATCH_SIZE``.
    # Mamba2/SSM kernels are batch-shape-sensitive under bf16, so we pin
    # the eval-time batch size for reproducibility.
    parser.add_argument(
        "--delta-window",
        type=int,
        default=_DEFAULT_DELTA_WINDOW,
        help="Half-width (bp) of the window around variant.",
    )
    parser.add_argument(
        "--probability-mode",
        type=str,
        choices=["softmax", "sigmoid", "auto"],
        default="auto",
        help="Probability normalization (hyenadna only).",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Explicit training config YAML (hyenadna).",
    )
    parser.add_argument(
        "--spliceai-context-length",
        type=int,
        default=10000,
        help="SpliceAI context length (spliceai only).",
    )
    parser.add_argument(
        "--spliceai-target-length",
        type=int,
        default=5000,
        help="SpliceAI target region length (spliceai only).",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default=None,
        help=(
            "Optional NPZ path to dump per-variant features"
            " (delta_max, gain/loss per channel, etc.) and"
            " labels for downstream meta-classifier fitting."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help=(
            "Aggregator size for per-window features."
            " 1 = max (SpliceAI standard);"
            " 3/5 = mean of top-k positions (denoised)."
        ),
    )
    # NOTE: HyenaDNA forward always returns the full ``seq_length``
    # output (target + flank). No center-crop is applied so the
    # delta window can reach into the flank when the user picks a
    # large enough ``--delta-window``. Model is only supervised on
    # the target labels, but its flank predictions still carry
    # SAV signal and excluding them throws information away.

    return parser.parse_args()


# ── H5 loader ───────────────────────────────────────────


def _decode_int_array(x: np.ndarray) -> str:
    """Decode an int8 nucleotide array to an ACGTN string.

    Args:
        x (np.ndarray): Array with values in ``{0,1,2,3,4}``.

    Returns:
        str: DNA string of the same length.
    """
    return "".join(DECODE_TABLE[x.astype(np.int64)])


def _load_h5_metadata(h5_path: str) -> dict[str, int]:
    """Read shape and length attributes from a SpliceVarDB H5.

    Args:
        h5_path (str): Path to the SpliceVarDB H5.

    Returns:
        dict[str, int]: ``{n, total_length, target_length, flank_length,
            has_variant_offset}``.

    Raises:
        ValueError: If required attributes are missing.
    """
    with h5py.File(h5_path, "r") as f:
        n = int(f["X_ref"].shape[0])
        total = int(f["X_ref"].shape[1])
        target = int(f.attrs.get("target_length", total))
        flank = int(f.attrs.get("flank_length", (total - target) // 2))
        has_variant_offset = "variant_offset" in f

    if total - 2 * flank != target:
        raise ValueError(
            f"Inconsistent H5 attrs: total={total},"
            f" flank={flank}, target={target}"
        )

    return {
        "n": n,
        "total_length": total,
        "target_length": target,
        "flank_length": flank,
        "has_variant_offset": has_variant_offset,
    }


def _crop_seq_centered(seq: str, length: int) -> str:
    """Return the centered ``length``-bp slice of ``seq``.

    No-op when ``len(seq) == length``. Drops symmetric tails when
    ``len(seq) > length`` so the target window stays in the same
    relative position the model expects.

    Args:
        seq (str): Decoded DNA string.
        length (int): Desired output length.

    Returns:
        str: ``seq`` cropped to ``length``.

    Raises:
        ValueError: When ``len(seq) < length`` (we never pad here).
    """
    if len(seq) == length:
        return seq
    if len(seq) < length:
        raise ValueError(
            f"Cannot crop seq of len {len(seq)} to {length}; padding"
            " would shift the variant offset."
        )
    start = (len(seq) - length) // 2
    return seq[start : start + length]


# ── HyenaDNA backend ────────────────────────────────────


def _resolve_probability_mode(
    train_cfg: DictConfig,
    cli_mode: str,
) -> str:
    """Resolve probability mode (softmax / sigmoid).

    Args:
        train_cfg (DictConfig): Training config.
        cli_mode (str): CLI value (``"auto"`` to derive).

    Returns:
        str: ``"softmax"`` or ``"sigmoid"``.
    """
    if cli_mode != "auto":
        # Anything that is neither "auto" nor a real mode used to be returned
        # verbatim, and the only downstream test is ``== "softmax"`` -- so a
        # None default silently selected sigmoid. evaluate_splicevardb_sav_indel_v4
        # defaulted to None, which put every registry indel number in sigmoid
        # space while the SNV path used softmax. Fail loudly instead.
        if cli_mode not in ("softmax", "sigmoid"):
            raise ValueError(
                f"probability_mode must be 'auto', 'softmax' or 'sigmoid'; "
                f"got {cli_mode!r}"
            )
        return cli_mode

    loss_type = OmegaConf.select(
        train_cfg,
        "training.loss_function.type",
        default=None,
    )
    if loss_type is None:
        return _DEFAULT_PROB_MODE
    return _LOSS_TYPE_TO_PROB_MODE.get(
        str(loss_type), _DEFAULT_PROB_MODE
    )


def _load_hyenadna_model(
    *,
    checkpoint_path: str,
    train_config_path: str | None,
    device: torch.device,
    hyena_filter_max_seq_len: int | None = None,
) -> tuple["torch.nn.Module", DictConfig]:
    """Load an IssaSplice HyenaDNA-style model from checkpoint.

    Args:
        checkpoint_path (str): Checkpoint .pt path.
        train_config_path (str | None): Explicit train config.
        device (torch.device): Compute device.
        hyena_filter_max_seq_len (int | None): Optional evaluation-only
            override for the Hyena implicit-filter position grid. This leaves
            ``model.seq_length`` and tokenisation geometry unchanged.

    Returns:
        tuple[torch.nn.Module, DictConfig]: ``(model, cfg)`` where
            ``cfg`` is the merged config used to construct the model.
    """
    from evaluate.evaluate import (
        _build_merged_config,
        _load_train_config,
    )
    from train.trainer import IssaSpliceTrainer

    train_cfg = _load_train_config(
        checkpoint_path,
        use_h5=True,
        model_type="hyenadna",
        train_config_path=train_config_path,
    )

    eval_cfg = OmegaConf.create(
        {
            "evaluation": {
                "checkpoint_path": checkpoint_path,
                "test_h5_path": None,
                "thresholds": None,
                "batch_size": 8,
                "num_workers": 0,
                "use_histogram": False,
                "strand_filter": "both",
            }
        }
    )
    cfg = _build_merged_config(
        eval_cfg,
        train_cfg,
        checkpoint_path=checkpoint_path,
    )
    if hyena_filter_max_seq_len is not None:
        hyena_filter_max_seq_len = int(hyena_filter_max_seq_len)
        if hyena_filter_max_seq_len < 1:
            raise ValueError(
                "hyena_filter_max_seq_len must be positive; got "
                f"{hyena_filter_max_seq_len}."
            )
        cfg.model.hyena_filter_max_seq_len = hyena_filter_max_seq_len

    cfg_model_path = Path("config/issasplice/model.yaml")
    if cfg_model_path.exists():
        cfg_model = OmegaConf.load(str(cfg_model_path))
    else:
        cfg_model = OmegaConf.create({})
    del cfg_model

    trainer = IssaSpliceTrainer(cfg)
    model = trainer.init_model(verbose=True).to(device)
    model.load_model_from_checkpoint(checkpoint_path)
    model.eval()

    if hyena_filter_max_seq_len is not None:
        actual = int(model.model.config.max_seq_len)
        if actual != hyena_filter_max_seq_len:
            raise RuntimeError(
                "Hyena filter-grid override was not applied: requested "
                f"{hyena_filter_max_seq_len}, constructed {actual}."
            )
        logger.info(
            "Hyena filter-grid intervention: max_seq_len=%d while "
            "model.seq_length=%d and tokenizer_max_length=%d",
            actual,
            int(cfg.model.seq_length),
            int(cfg.model.tokenizer_max_length),
        )

    return model, cfg


def _hyenadna_forward_probs(
    *,
    model: "torch.nn.Module",
    seqs: list[str],
    cfg: DictConfig,
    device: torch.device,
    probability_mode: str,
    target_length: int,
    input_ids: "torch.Tensor | None" = None,
) -> np.ndarray:
    """Forward HyenaDNA model and return per-position probs.

    Args:
        model (torch.nn.Module): IssaSplice model in eval mode.
        seqs (list[str]): DNA strings of length seq_length.
        cfg (DictConfig): Merged eval config.
        device (torch.device): Compute device.
        probability_mode (str): ``softmax`` or ``sigmoid``.
        target_length (int): Center-cropped output length.
        input_ids (torch.Tensor | None): Pre-tokenised ids of shape
            ``(B, seq_len)``. When given, ``seqs`` is ignored and the
            tokenizer is skipped -- callers scoring many batches should
            build these with a lookup table rather than pay for
            HyenaDNA's pure-Python CharacterTokenizer per batch.

    Returns:
        np.ndarray: Probabilities of shape (B, target_length, 3)
            with channel order (acceptor, donor, neither).
    """
    if input_ids is None:
        max_len = int(cfg.model.tokenizer_max_length)
        enc = model.tokenizer(
            seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
    else:
        # Caller already produced ids (see _batched_forward's lookup table).
        # HyenaDNA ships a pure-Python CharacterTokenizer, so tokenizing here
        # costs ~15,000 dict lookups per sequence to reproduce a fixed 5-entry
        # remap of an array the caller already holds.
        input_ids = input_ids.to(device)

    sig = inspect.signature(model.forward)
    fwd_kwargs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": None,
    }
    if "strand" in sig.parameters:
        fwd_kwargs["strand"] = None
    # Every caller of this function is pure inference and takes the numpy
    # return, so there is no reason to build an autograd graph over a
    # 15,000-position forward. inference_mode is stronger than no_grad --
    # it also skips version counting and view tracking -- and changes no
    # arithmetic.
    with torch.inference_mode():
        outputs = model(**fwd_kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            cls_logits = outputs[0]
        else:
            cls_logits = outputs[0]

        cls_logits = cls_logits.float()

        # NOTE: HyenaDNA output always covers the full ``seq_length``
        # (target + flank). We no longer center-crop to
        # ``target_length`` — keeping the flank lets delta scoring
        # probe flank-region predictions when the user picks a wide
        # ``--delta-window``. Model is only supervised on target, but
        # empirically the flank predictions still carry small but
        # non-zero SAV signal (verified on 2stageDC SpliceVarDB).
        # The ``target_length`` argument is retained for API parity
        # with the SpliceAI backend.
        del target_length

        if probability_mode == "softmax":
            probs = torch.softmax(cls_logits, dim=-1)
        else:
            probs = torch.sigmoid(cls_logits)

        return probs.detach().cpu().numpy()


# ── SpliceAI backend ────────────────────────────────────


def _load_spliceai_model(
    *,
    checkpoint_path: str,
    context_length: int,
    target_length: int,
    device: torch.device,
) -> tuple["torch.nn.Module", int, int]:
    """Load a SpliceAI checkpoint.

    Args:
        checkpoint_path (str): Checkpoint .pt path.
        context_length (int): Default context length if missing.
        target_length (int): Target region length.
        device (torch.device): Compute device.

    Returns:
        tuple[torch.nn.Module, int, int]: ``(model,
            resolved_context_length, target_length)``.
    """
    from scripts.spliceai_pipeline import (
        CONTEXT_LENGTH_TO_MODEL,
        create_model,
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cl = int(ckpt.get("context_length", context_length))
    else:
        state_dict = ckpt
        cl = context_length

    if cl not in CONTEXT_LENGTH_TO_MODEL:
        raise ValueError(f"Invalid SpliceAI context length: {cl}")

    cfg = OmegaConf.create(
        {
            "model": {
                "spliceai_context_length": cl,
                "target_region_length": target_length,
            }
        }
    )
    model, _ = create_model(cfg)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, cl, target_length


def _spliceai_forward_probs(
    *,
    model: "torch.nn.Module",
    seqs: list[str],
    device: torch.device,
    target_length: int,
) -> np.ndarray:
    """Forward SpliceAI model and return per-position probs.

    Args:
        model (torch.nn.Module): SpliceAI model in eval mode.
        seqs (list[str]): DNA strings of length seq_length.
        device (torch.device): Compute device.
        target_length (int): Output target length.

    Returns:
        np.ndarray: Probabilities of shape (B, target_length, 3)
            with channel order (acceptor, donor, neither).
    """
    from scripts.spliceai_pipeline import seq_to_onehot

    onehots = torch.stack([seq_to_onehot(s) for s in seqs], dim=0)
    inputs = onehots.to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)

    probs = torch.softmax(logits.float(), dim=-1)

    out_len = probs.shape[1]
    if out_len > target_length:
        start = (out_len - target_length) // 2
        probs = probs[:, start : start + target_length, :]

    return probs.detach().cpu().numpy()


# ── SpliceAI v1 backend ─────────────────────────────────


# H5 token convention: 0=N, 1=A, 2=C, 3=G, 4=T (see DECODE_TABLE).
# SpliceAIV1 token convention: 0=A, 1=C, 2=G, 3=T (N gets clamped → A).
# Remap: ``v1_id = clip(h5_id - 1, 0, 3)``.
_H5_TO_V1_LUT: np.ndarray = np.clip(
    np.arange(5, dtype=np.int64) - 1, 0, 3
)


def _load_spliceai_v1_model(
    *,
    checkpoint_path: str,
    seq_length: int,
    context_length: int,
    target_length: int,
    device: torch.device,
) -> "torch.nn.Module":
    """Load a SpliceAIV1 checkpoint (``spliceai-original-h5-v1`` family).

    Args:
        checkpoint_path (str): Checkpoint .pt path. May be a raw
            ``state_dict`` or a dict with a ``model_state_dict`` key.
        seq_length (int): Input sequence length (e.g., 15000).
        context_length (int): SpliceAI context length preset
            (selects dilation rates). Must be one of {80, 400, 2000, 10000}.
        target_length (int): Target region length. Unused by the model
            itself (the v1 forward returns logits over the full sequence)
            but kept for parity with the spliceai loader signature.
        device (torch.device): Compute device.

    Returns:
        torch.nn.Module: Model in ``eval()`` mode on ``device``.

    Raises:
        RuntimeError: If state-dict keys do not match the SpliceAIV1 layout.
    """
    from model.spliceai_v1 import SpliceAIV1

    cfg = OmegaConf.create(
        {
            "model": {
                "seq_length": seq_length,
                "n_classification_classes": 3,
                "spliceai_context_length": context_length,
                "spliceai_channels": 32,
                "spliceai_kernel_size": 11,
                "spliceai_blocks_per_dilation": 4,
            }
        }
    )
    model = SpliceAIV1(cfg)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _spliceai_v1_forward_probs(
    *,
    model: "torch.nn.Module",
    x_ids_h5: np.ndarray,
    device: torch.device,
    target_length: int,
) -> np.ndarray:
    """Forward SpliceAIV1 model and return per-position probs.

    Args:
        model (torch.nn.Module): SpliceAIV1 in eval mode.
        x_ids_h5 (np.ndarray): H5 token IDs of shape (B, seq_length) with
            values in ``{0..4}`` under the ``N=0, A=1, C=2, G=3, T=4``
            convention. Internally remapped to SpliceAIV1's
            ``A=0, C=1, G=2, T=3`` (N → A fallback).
        device (torch.device): Compute device.
        target_length (int): Target region length. Logits are cropped to
            the centered window of this length before softmax.

    Returns:
        np.ndarray: Probabilities of shape (B, target_length, 3) with
            channel order ``(acceptor, donor, neither)`` matching the v1
            training-pipeline label stack.
    """
    ids_v1 = _H5_TO_V1_LUT[x_ids_h5.astype(np.int64)]
    inputs = torch.from_numpy(ids_v1).to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(inputs)

    out_len = logits.shape[1]
    if out_len > target_length:
        start = (out_len - target_length) // 2
        logits = logits[:, start : start + target_length, :]

    probs = torch.softmax(logits.float(), dim=-1)
    return probs.detach().cpu().numpy()


# ── Delta scoring & metrics ─────────────────────────────


def _topk_mean(x: np.ndarray, k: int, axis: int = 1) -> np.ndarray:
    """Compute the mean of the top-k values along an axis.

    For ``k == 1`` this collapses to ``np.max(axis)``. For ``k`` larger
    than the axis length, the result is the full mean.

    Args:
        x (np.ndarray): Input array.
        k (int): Number of largest values to average.
        axis (int): Axis along which to aggregate.

    Returns:
        np.ndarray: Aggregated array with the chosen axis collapsed.
    """
    if k <= 1:
        return x.max(axis=axis)
    n = x.shape[axis]
    if k >= n:
        return x.mean(axis=axis)
    pivot = n - k
    partitioned = np.partition(x, pivot, axis=axis)
    top = np.take(partitioned, range(pivot, n), axis=axis)
    return top.mean(axis=axis)


def _compute_delta_scores(
    p_ref: np.ndarray,
    p_alt: np.ndarray,
    variant_offset: int | np.ndarray,
    half_window: int,
    top_k: int = 1,
) -> dict[str, np.ndarray]:
    """Compute SAV delta scores from per-position probs.

    The four directional features (acc/don × gain/loss) match the
    standard SpliceAI delta score representation: ``gain`` measures
    how much the alt sequence increases the splice probability,
    ``loss`` measures how much it decreases it. When ``top_k > 1``,
    the per-window aggregator becomes the mean of the top-k positions
    rather than the single max — this denoises the score against
    isolated outlier positions.

    Args:
        p_ref (np.ndarray): Ref probs (B, L, 3) — (acc, don, nei).
        p_alt (np.ndarray): Alt probs (B, L, 3) — (acc, don, nei).
        variant_offset (int): Variant index within the target window.
        half_window (int): Half-width of the score window in bp.
        top_k (int): Aggregator size. ``1`` reproduces the SpliceAI-
            standard ``max``; ``3``/``5`` average the top positions.

    Returns:
        dict[str, np.ndarray]: Per-variant scores keyed by name:
            ``delta_max`` — max over (|ΔA|, |ΔD|) in the window;
            ``delta_acc_max``, ``delta_don_max`` — per-channel
            absolute deltas; ``delta_acc_gain``, ``delta_acc_loss``,
            ``delta_don_gain``, ``delta_don_loss`` — directional
            deltas (clamped at 0).
    """
    target_length = p_ref.shape[1]
    batch_size = p_ref.shape[0]

    if isinstance(variant_offset, (int, np.integer)):
        offsets = np.full(batch_size, int(variant_offset), dtype=np.int64)
    else:
        offsets = np.asarray(variant_offset, dtype=np.int64)
        if offsets.shape[0] != batch_size:
            raise ValueError(
                f"variant_offset length {offsets.shape[0]} does not"
                f" match batch size {batch_size}"
            )

    diff_signed_full = p_alt - p_ref
    diff_abs_full = np.abs(diff_signed_full)
    pos_signed_full = np.maximum(diff_signed_full, 0.0)
    neg_signed_full = np.maximum(-diff_signed_full, 0.0)

    if np.all(offsets == offsets[0]):
        # Fast path: shared window across the batch.
        v = int(offsets[0])
        start = max(0, v - half_window)
        end = min(target_length, v + half_window + 1)
        delta_acc = _topk_mean(diff_abs_full[:, start:end, 0], top_k, axis=1)
        delta_don = _topk_mean(diff_abs_full[:, start:end, 1], top_k, axis=1)
        gain_acc = _topk_mean(pos_signed_full[:, start:end, 0], top_k, axis=1)
        loss_acc = _topk_mean(neg_signed_full[:, start:end, 0], top_k, axis=1)
        gain_don = _topk_mean(pos_signed_full[:, start:end, 1], top_k, axis=1)
        loss_don = _topk_mean(neg_signed_full[:, start:end, 1], top_k, axis=1)
    else:
        delta_acc = np.empty(batch_size, dtype=np.float32)
        delta_don = np.empty(batch_size, dtype=np.float32)
        gain_acc = np.empty(batch_size, dtype=np.float32)
        loss_acc = np.empty(batch_size, dtype=np.float32)
        gain_don = np.empty(batch_size, dtype=np.float32)
        loss_don = np.empty(batch_size, dtype=np.float32)
        for i in range(batch_size):
            v = int(offsets[i])
            start = max(0, v - half_window)
            end = min(target_length, v + half_window + 1)
            delta_acc[i] = _topk_mean(
                diff_abs_full[i : i + 1, start:end, 0], top_k, axis=1
            )[0]
            delta_don[i] = _topk_mean(
                diff_abs_full[i : i + 1, start:end, 1], top_k, axis=1
            )[0]
            gain_acc[i] = _topk_mean(
                pos_signed_full[i : i + 1, start:end, 0], top_k, axis=1
            )[0]
            loss_acc[i] = _topk_mean(
                neg_signed_full[i : i + 1, start:end, 0], top_k, axis=1
            )[0]
            gain_don[i] = _topk_mean(
                pos_signed_full[i : i + 1, start:end, 1], top_k, axis=1
            )[0]
            loss_don[i] = _topk_mean(
                neg_signed_full[i : i + 1, start:end, 1], top_k, axis=1
            )[0]
    delta_max = np.maximum(delta_acc, delta_don)

    return {
        "delta_max": delta_max,
        "delta_acc_max": delta_acc,
        "delta_don_max": delta_don,
        "delta_acc_gain": gain_acc,
        "delta_acc_loss": loss_acc,
        "delta_don_gain": gain_don,
        "delta_don_loss": loss_don,
    }


def _compute_summary_metrics(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute AUROC and AUPRC over per-variant scores.

    Args:
        labels (np.ndarray): Binary labels of shape (N,).
        scores (dict[str, np.ndarray]): Score arrays per name.

    Returns:
        dict[str, float]: ``{auroc_<name>, auprc_<name>}``
            for each score, plus ``positive_count``,
            ``negative_count``.
    """
    metrics: dict[str, float] = {
        "n": int(labels.shape[0]),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
    }
    for name, sc in scores.items():
        metrics[f"auroc_{name}"] = float(roc_auc_score(labels, sc))
        metrics[f"auprc_{name}"] = float(
            average_precision_score(labels, sc)
        )

    return metrics


# ── Main loop ───────────────────────────────────────────


def _run_inference(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Run inference + delta scoring across the SpliceVarDB H5.

    Args:
        args (argparse.Namespace): CLI args.
        device (torch.device): Compute device.

    Returns:
        dict[str, object]: Result dict ready for JSON dump.
    """
    meta = _load_h5_metadata(args.test_h5_path)
    target_length = meta["target_length"]
    flank_length = meta["flank_length"]
    total_length = meta["total_length"]
    has_variant_offset = bool(meta.get("has_variant_offset", False))

    if has_variant_offset:
        with h5py.File(args.test_h5_path, "r") as f:
            variant_offsets_h5 = f["variant_offset"][:].astype(np.int64)
        logger.info(
            "H5 meta: n=%d total=%d target=%d flank=%d (per-record"
            " variant_offset present, range=[%d, %d])",
            meta["n"],
            total_length,
            target_length,
            flank_length,
            int(variant_offsets_h5.min()),
            int(variant_offsets_h5.max()),
        )
    else:
        # Fall back to per-record diff between X_ref and X_alt for SNV.
        # H5 files (e.g. splicevardb_block_test) without an explicit
        # variant_offset dataset still encode the variant position
        # implicitly: the single base where X_ref and X_alt differ.
        # For indel/MNV the diff count is large; we fall back to the
        # window center for those rows and warn.
        center = total_length // 2
        variant_offsets_h5 = np.full(meta["n"], center, dtype=np.int64)
        n_diff_one = n_diff_other = 0
        with h5py.File(args.test_h5_path, "r") as f:
            x_ref_full = f["X_ref"][:]
            x_alt_full = f["X_alt"][:]
        diffs_per_row = (x_ref_full != x_alt_full)
        diff_counts = diffs_per_row.sum(axis=1)
        for i in range(meta["n"]):
            if diff_counts[i] == 1:
                variant_offsets_h5[i] = int(
                    np.argmax(diffs_per_row[i])
                )
                n_diff_one += 1
            else:
                n_diff_other += 1
        logger.info(
            "H5 meta: n=%d total=%d target=%d flank=%d offset=diff"
            "(SNV=%d, fallback_center=%d)",
            meta["n"],
            total_length,
            target_length,
            flank_length,
            n_diff_one,
            n_diff_other,
        )
        if n_diff_one:
            logger.info(
                "Derived variant_offset range: [%d, %d] (median=%d)",
                int(variant_offsets_h5.min()),
                int(variant_offsets_h5.max()),
                int(np.median(variant_offsets_h5)),
            )

    if args.backend == "hyenadna":
        model, cfg = _load_hyenadna_model(
            checkpoint_path=args.checkpoint,
            train_config_path=args.train_config,
            device=device,
        )
        loss_type = OmegaConf.select(
            cfg, "training.loss_function.type", default=None
        )
        prob_mode = _resolve_probability_mode(
            cfg, args.probability_mode
        )
        logger.info(
            "Loss type=%s, probability_mode=%s",
            loss_type,
            prob_mode,
        )
        backend_meta = {
            "loss_type": loss_type,
            "probability_mode": prob_mode,
        }
    elif args.backend == "spliceai-v1":
        v1_seq_length = (
            int(args.spliceai_target_length)
            + int(args.spliceai_context_length)
        )
        model = _load_spliceai_v1_model(
            checkpoint_path=args.checkpoint,
            seq_length=v1_seq_length,
            context_length=int(args.spliceai_context_length),
            target_length=int(args.spliceai_target_length),
            device=device,
        )
        if args.spliceai_target_length != target_length:
            logger.warning(
                "SpliceAI-v1 target_length=%d != H5 target_length=%d;"
                " using SpliceAI-v1 value.",
                args.spliceai_target_length,
                target_length,
            )
            target_length = int(args.spliceai_target_length)
        backend_meta = {
            "context_length": int(args.spliceai_context_length),
            "target_region_length": target_length,
            "model_class": "SpliceAIV1",
        }
    else:
        model, ctx_len, target_len_resolved = _load_spliceai_model(
            checkpoint_path=args.checkpoint,
            context_length=args.spliceai_context_length,
            target_length=args.spliceai_target_length,
            device=device,
        )
        if target_len_resolved != target_length:
            logger.warning(
                "SpliceAI target_length=%d != H5 target_length=%d;"
                " using SpliceAI value.",
                target_len_resolved,
                target_length,
            )
            target_length = target_len_resolved
        backend_meta = {
            "context_length": ctx_len,
            "target_region_length": target_length,
        }

    feature_keys = (
        "delta_max",
        "delta_acc_max",
        "delta_don_max",
        "delta_acc_gain",
        "delta_acc_loss",
        "delta_don_gain",
        "delta_don_loss",
    )
    feature_buffers: dict[str, list[np.ndarray]] = {
        k: [] for k in feature_keys
    }
    labels_all: list[np.ndarray] = []

    if args.backend == "hyenadna":
        model_seq_length = int(cfg.model.seq_length)
    else:
        model_seq_length = (
            int(args.spliceai_target_length)
            + int(args.spliceai_context_length)
        )
    crop_offset = (total_length - model_seq_length) // 2
    if crop_offset < 0:
        raise ValueError(
            f"H5 total_length={total_length} is shorter than model"
            f" seq_length={model_seq_length}; cannot crop without padding."
        )
    if crop_offset > 0:
        logger.info(
            "Cropping H5 input %d bp -> model %d bp (drop %d bp from each side)",
            total_length,
            model_seq_length,
            crop_offset,
        )

    # Variant offset inside the model's output array.
    # - HyenaDNA: output spans the full ``seq_length`` = target +
    #   flank; variant position in that output is exactly
    #   ``variant_offset_h5`` (input coords).
    # - SpliceAI: model output is already target-only; variant
    #   position in the (B, target_length, _) array is
    #   ``variant_offset_h5 - flank_length``.
    if args.backend == "hyenadna":
        variant_offsets_in_target = variant_offsets_h5.astype(np.int64)
        logger.info(
            "HyenaDNA full-output mode: variant offsets in [%d, %d]"
            " of %d-bp output (target+flank, no center-crop)",
            int(variant_offsets_in_target.min()),
            int(variant_offsets_in_target.max()),
            model_seq_length,
        )
    else:
        variant_offsets_in_target = (
            variant_offsets_h5 - flank_length
        ).astype(np.int64)

    n = meta["n"]
    bs = _FIXED_BATCH_SIZE
    logger.info("Fixed eval batch_size=%d (reproducibility lock)", bs)

    with torch.no_grad():
        with h5py.File(args.test_h5_path, "r") as f:
            for start in tqdm(
                range(0, n, bs),
                desc="SAV inference",
            ):
                end = min(start + bs, n)
                x_ref = f["X_ref"][start:end]
                x_alt = f["X_alt"][start:end]
                lbl = f["label"][start:end]

                seqs_ref = [_decode_int_array(r) for r in x_ref]
                seqs_alt = [_decode_int_array(a) for a in x_alt]
                if crop_offset > 0:
                    seqs_ref = [
                        _crop_seq_centered(s, model_seq_length)
                        for s in seqs_ref
                    ]
                    seqs_alt = [
                        _crop_seq_centered(s, model_seq_length)
                        for s in seqs_alt
                    ]

                if args.backend == "hyenadna":
                    p_ref = _hyenadna_forward_probs(
                        model=model,
                        seqs=seqs_ref,
                        cfg=cfg,
                        device=device,
                        probability_mode=backend_meta[
                            "probability_mode"
                        ],
                        target_length=target_length,
                    )
                    p_alt = _hyenadna_forward_probs(
                        model=model,
                        seqs=seqs_alt,
                        cfg=cfg,
                        device=device,
                        probability_mode=backend_meta[
                            "probability_mode"
                        ],
                        target_length=target_length,
                    )
                elif args.backend == "spliceai-v1":
                    if crop_offset > 0:
                        x_ref_v1 = x_ref[
                            :, crop_offset : crop_offset + model_seq_length
                        ]
                        x_alt_v1 = x_alt[
                            :, crop_offset : crop_offset + model_seq_length
                        ]
                    else:
                        x_ref_v1 = x_ref
                        x_alt_v1 = x_alt
                    p_ref = _spliceai_v1_forward_probs(
                        model=model,
                        x_ids_h5=x_ref_v1,
                        device=device,
                        target_length=target_length,
                    )
                    p_alt = _spliceai_v1_forward_probs(
                        model=model,
                        x_ids_h5=x_alt_v1,
                        device=device,
                        target_length=target_length,
                    )
                else:
                    p_ref = _spliceai_forward_probs(
                        model=model,
                        seqs=seqs_ref,
                        device=device,
                        target_length=target_length,
                    )
                    p_alt = _spliceai_forward_probs(
                        model=model,
                        seqs=seqs_alt,
                        device=device,
                        target_length=target_length,
                    )

                deltas = _compute_delta_scores(
                    p_ref=p_ref,
                    p_alt=p_alt,
                    variant_offset=variant_offsets_in_target[start:end],
                    half_window=int(args.delta_window),
                    top_k=int(args.top_k),
                )
                for k in feature_keys:
                    feature_buffers[k].append(deltas[k])
                labels_all.append(lbl.astype(np.int64))

    features = {
        k: np.concatenate(feature_buffers[k]) for k in feature_keys
    }
    labels = np.concatenate(labels_all)

    metrics = _compute_summary_metrics(labels, features)

    if args.features_output:
        feat_path = Path(args.features_output)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            feat_path,
            labels=labels,
            **features,
        )
        logger.info("Saved per-variant features: %s", feat_path)

    if has_variant_offset:
        offset_summary: int | dict[str, int] = {
            "min": int(variant_offsets_in_target.min()),
            "max": int(variant_offsets_in_target.max()),
            "mean": int(variant_offsets_in_target.mean()),
        }
    else:
        offset_summary = int(variant_offsets_in_target[0])

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "test_h5_path": str(args.test_h5_path),
        "backend": args.backend,
        "delta_window_bp": int(args.delta_window),
        "variant_offset_in_target": offset_summary,
        "model_seq_length": int(model_seq_length),
        "h5_total_length": int(total_length),
        "target_length": int(target_length),
        "h5_meta": meta,
        "backend_meta": backend_meta,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return result


# ── Entry point ─────────────────────────────────────────


def main() -> None:
    """CLI entrypoint for SpliceVarDB SAV evaluation."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")

    # GPU is allocated by the launcher (Slurm srun); use the visible device.
    device = torch.device("cuda")

    result = _run_inference(args=args, device=device)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    metrics = result["metrics"]
    logger.info("=" * 60)
    logger.info("SAV evaluation summary")
    logger.info("  checkpoint: %s", args.checkpoint)
    logger.info("  n=%d (pos=%d, neg=%d)", metrics["n"],
                metrics["positive_count"],
                metrics["negative_count"])
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
    logger.info("Saved: %s", out_path)

    return


if __name__ == "__main__":
    main()
