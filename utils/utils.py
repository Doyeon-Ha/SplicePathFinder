"""
Utility functions package for the splicing project.
"""

import logging
import math
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from utils.distributed_utils import is_master_process


@dataclass(frozen=True)
class BatchPlan:
    """Resolved per-step batch sizing.

    Attributes:
        micro_batch_size (int): Per-GPU micro batch (``training.batch_size``).
        num_gpus (int): GPUs participating in the all-reduce. Taken from the
            ``WORLD_SIZE`` environment variable set by the launcher (torchrun /
            Slurm ``srun``), defaulting to 1 when unset.
        grad_accum_steps (int): Gradient accumulation steps to use.
        effective_batch_size (int): Effective batch per optimizer step
            (``micro_batch_size * num_gpus * grad_accum_steps``).
        source (str): How the plan was resolved. One of:
            - ``"effective"``: derived from ``training.effective_batch_size``.
            - ``"manual"``: read from ``training.gradient_accumulation_steps``
              with default 1 when absent.
    """

    micro_batch_size: int
    num_gpus: int
    grad_accum_steps: int
    effective_batch_size: int
    source: str


def resolve_batch_plan(cfg: DictConfig) -> BatchPlan:
    """Resolve batch sizing from a cfg, preferring ``effective_batch_size``.

    Precedence:
      1. ``training.effective_batch_size`` set → derive
         ``grad_accum_steps = effective / (micro * num_gpus)``. Must divide
         evenly. If ``training.gradient_accumulation_steps`` is also set,
         it must match the derived value (else ``ValueError``).
      2. ``training.effective_batch_size`` null → read
         ``training.gradient_accumulation_steps`` (default 1) and compute
         the effective batch from it.

    ``num_gpus`` is read from the ``WORLD_SIZE`` environment variable set by
    the launcher (torchrun / Slurm ``srun``), defaulting to 1 when unset.
    This lets a single ``effective_batch_size`` hold across any GPU count
    chosen at submission time: the derived grad-accum absorbs the difference,
    so doubling the allocated GPUs halves the grad-accum steps automatically.

    Args:
        cfg (DictConfig): Top-level training config (must expose
            ``training.batch_size``).

    Returns:
        BatchPlan: Resolved plan with grad-accum steps and provenance.

    Raises:
        ValueError: ``effective_batch_size`` not divisible by
            ``micro_batch * num_gpus``, or conflict between
            ``effective_batch_size`` and an explicit
            ``gradient_accumulation_steps``.
    """
    micro_batch = int(OmegaConf.select(cfg, "training.batch_size"))
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))

    effective_bs = OmegaConf.select(
        cfg, "training.effective_batch_size", default=None
    )
    grad_accum_user = OmegaConf.select(
        cfg, "training.gradient_accumulation_steps", default=None
    )

    if effective_bs is None:
        grad_accum = int(grad_accum_user) if grad_accum_user is not None else 1
        return BatchPlan(
            micro_batch_size=micro_batch,
            num_gpus=num_gpus,
            grad_accum_steps=grad_accum,
            effective_batch_size=micro_batch * num_gpus * grad_accum,
            source="manual",
        )

    effective_bs = int(effective_bs)
    base = micro_batch * num_gpus
    if base <= 0 or effective_bs % base != 0:
        raise ValueError(
            f"training.effective_batch_size ({effective_bs}) is not divisible "
            f"by training.batch_size ({micro_batch}) * num_gpus ({num_gpus}) = "
            f"{base}. Adjust one of them so the division is exact."
        )
    derived_accum = effective_bs // base
    if grad_accum_user is not None and int(grad_accum_user) != derived_accum:
        raise ValueError(
            f"training.effective_batch_size={effective_bs} implies "
            f"gradient_accumulation_steps={derived_accum}, but "
            f"training.gradient_accumulation_steps={grad_accum_user} is also "
            "set. Remove one of the two keys or align them."
        )
    return BatchPlan(
        micro_batch_size=micro_batch,
        num_gpus=num_gpus,
        grad_accum_steps=derived_accum,
        effective_batch_size=effective_bs,
        source="effective",
    )


def log_model_timing_summary(logger: logging.Logger, actual_model: Any) -> None:
    """
    Log final model-level timing summary if the model has timing logging enabled.

    Args:
        logger (logging.Logger): Logger instance used for output.
        actual_model (Any): Model instance that may provide timing summary via
            `get_timing_summary()` and flag `enable_timing_log`.
    """
    if not hasattr(actual_model, "enable_timing_log") or not getattr(
        actual_model, "enable_timing_log", False
    ):
        return

    logger.info("Final Timing Statistics:")
    timing_summary: Dict[str, Any] = actual_model.get_timing_summary()
    if "error" in timing_summary:
        logger.info("  No timing data available")
        return

    average_times: Dict[str, float] = timing_summary.get("average_times", {})
    time_percentages: Dict[str, float] = timing_summary.get("time_percentages", {})

    logger.info(
        "  Total Forward Passes: %s", timing_summary.get("total_forward_passes", 0)
    )
    logger.info(
        "  Average Backbone Time: %.4fs (%.1f%%)",
        average_times.get("backbone", 0.0),
        time_percentages.get("backbone", 0.0),
    )
    logger.info(
        "  Average Classification Head Time: %.4fs (%.1f%%)",
        average_times.get("classification_head", 0.0),
        time_percentages.get("classification_head", 0.0),
    )
    logger.info("  Average Total Time: %.4fs", average_times.get("total", 0.0))

    head_time: float = average_times.get("classification_head", 0.0)
    if head_time > 0:
        ratio: float = average_times.get("backbone", 0.0) / head_time
        logger.info(
            "  Speed Ratio: Backbone is %.1fx slower than Classification Head", ratio
        )


def center_indices(orig_len: int, tgt_len: int) -> tuple[int, int]:
    """
    Center-crop helper

    Args:
        orig_len (int): Original length
        tgt_len (int): Target length

    Returns:
        tuple[int, int]: (start, end)
    """
    start: int = max(0, (orig_len - tgt_len) // 2)
    end: int = start + tgt_len
    return start, end


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a NumPy array safely.

    Args:
        tensor (torch.Tensor): Input tensor to convert.

    Returns:
        np.ndarray: Numpy array detached from graph and moved to CPU.

    Notes:
        BFloat16 tensors are converted to float32 before numpy conversion
        since NumPy does not support BFloat16 dtype.
    """
    tensor = tensor.detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.cpu().numpy()


def compute_keep_mask_and_stats(
    ref_seqs: list[str],
    original_labels: torch.Tensor,
    *,
    seq_length: int,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute boolean keep mask for a batch and aggregated filtering statistics.

    Filters include:
    - Non-ACGT token presence in any sequence (case-insensitive)
    - Reference length mismatch against the provided seq_length
    - For multi-class focal loss only: any position with sum of labels > 1

    Args:
        ref_seqs (list[str]): Reference DNA sequences for the batch.
        original_labels (torch.Tensor): Labels tensor; shapes like [B, 3, L] or [3, L].
        seq_length (int): Expected fixed reference sequence length.
        loss_type (str): Loss type string to determine multi-class behavior.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]: A tuple of (keep_mask, stats) where
            keep_mask is [B] bool tensor marking samples to keep, and stats contains
            'length_mismatch_total', 'overlap_sequences_total',
            'overlap_positions_total', and 'unexpected_tokens'.
    """
    batch_size = len(ref_seqs)
    keep_mask = torch.ones(batch_size, dtype=torch.bool)

    allowed_tokens = {"A", "C", "G", "T", "N"}
    unexpected_tokens: set[str] = set()

    # Non-ACGT filter
    for sample_index, sequence in enumerate(ref_seqs):
        if not keep_mask[sample_index]:
            continue
        for character in sequence:
            if character.upper() not in allowed_tokens:
                unexpected_tokens.add(character)
                keep_mask[sample_index] = False
                break

    # Length filter
    length_mismatch_total: int = 0
    for sample_index, sequence in enumerate(ref_seqs):
        if not keep_mask[sample_index]:
            continue
        if len(sequence) != seq_length:
            keep_mask[sample_index] = False
            length_mismatch_total += 1

    # Overlap filter for multi-class focal loss
    overlap_sequences_total: int = 0
    overlap_positions_total: int = 0
    if loss_type == "multi_class_focal_loss":
        labels_for_filter = original_labels
        if labels_for_filter.dim() == 2 and labels_for_filter.size(0) == 3:
            labels_for_filter = labels_for_filter.unsqueeze(0)
        if labels_for_filter.dim() == 3 and labels_for_filter.size(1) == 3:
            per_pos_sum = labels_for_filter.sum(dim=1)
            overlap_mask_per_pos = per_pos_sum > 1.0 + 1e-6
            overlap_positions_total = int(overlap_mask_per_pos.sum().item())
            overlap_per_seq = overlap_mask_per_pos.any(dim=1)
            overlap_sequences_total = int(overlap_per_seq.sum().item())
            keep_mask &= ~overlap_per_seq.cpu()

    stats: dict[str, Any] = {
        "length_mismatch_total": length_mismatch_total,
        "overlap_sequences_total": overlap_sequences_total,
        "overlap_positions_total": overlap_positions_total,
        "unexpected_tokens": unexpected_tokens,
    }

    return keep_mask, stats


def reduce_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes by summing and averaging by world size.

    Args:
        tensor (torch.Tensor): Tensor to reduce.

    Returns:
        torch.Tensor: Reduced tensor (averaged).
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def init_weights_factory(linear_layer_init_method: str) -> Callable[[nn.Module], None]:
    """
    Create a weight initialization function for use with nn.Module.apply().

    Args:
        linear_layer_init_method (str): Initialization method for linear layers.
            One of "xavier", "kaiming", or "normal".

    Returns:
        Callable[[nn.Module], None]: Function that initializes weights for a module.
    """

    def init_weights(module: nn.Module) -> None:
        """
        Initialize weights for a module.

        Args:
            module (nn.Module): Module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            if linear_layer_init_method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif linear_layer_init_method == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif linear_layer_init_method == "normal":
                nn.init.normal_(module.weight, std=0.02)

            _reset_bias(module)

        elif isinstance(module, nn.Conv1d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            _reset_bias(module)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.MultiheadAttention):
            # Special initialization for attention
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.Embedding):
            # Use normal distribution with std=1/sqrt(embedding_dim)
            embedding_dim = module.embedding_dim
            nn.init.normal_(module.weight, mean=0, std=1 / math.sqrt(embedding_dim))

        return

    return init_weights


def _reset_bias(target: nn.Module) -> None:
    """Reset bias parameter to zero if present.
    Args:
        target (nn.Module): Module that may include a bias tensor.
    """
    bias_param = getattr(target, "bias", None)
    if bias_param is not None:
        nn.init.constant_(bias_param, 0)
    return


def archive_code_snapshot(
    output_dir: str,
    directories: list[str] | None = None,
    *,
    root_dir: str | None = None,
    archive_name_prefix: str = "code_snapshot",
    logger: logging.Logger | None = None,
) -> str:
    """
    Create a compressed archive (tar.gz) of selected source directories and
    save it into the specified output directory.

    Args:
        output_dir (str): Directory where the archive file will be written.
        directories (list[str] | None): Project-relative directories to include.
            Defaults to ["config", "model", "train", "utils"].
        root_dir (str | None): Project root to resolve relative directories.
            When running under Hydra, this should typically be HYDRA_ORIG_CWD.
            If None, uses environment HYDRA_ORIG_CWD or the current working dir.
        archive_name_prefix (str): Prefix for the archive filename.
        logger (logging.Logger | None): Optional logger for messages.

    Returns:
        str: Absolute path to the created archive file.
    """
    # Resolve project root robustly when Hydra is changing cwd
    effective_root: str = (
        root_dir
        if isinstance(root_dir, str) and len(root_dir) > 0
        else os.environ.get("HYDRA_ORIG_CWD", os.getcwd())
    )

    include_dirs: list[str] = (
        directories
        if directories is not None
        else ["config", "model", "train", "utils"]
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f"{archive_name_prefix}_{timestamp}.tar.gz"
    archive_path = os.path.abspath(os.path.join(output_dir, archive_filename))

    # Create archive
    with tarfile.open(archive_path, mode="w:gz") as tar:
        for relative_dir in include_dirs:
            abs_dir = os.path.abspath(os.path.join(effective_root, relative_dir))
            if not os.path.exists(abs_dir):
                if logger is not None:
                    logger.warning("Archive skip: directory not found: %s", abs_dir)
                continue
            # Store paths relative to project root inside the archive for portability
            arcname = os.path.relpath(abs_dir, start=effective_root)
            tar.add(abs_dir, arcname=arcname)

    if logger is not None:
        logger.info("Code snapshot archived to: %s", archive_path)

    return archive_path


def log_only_master_process(logger: logging.Logger, level: str, message: str) -> None:
    """
    Log message only if current process is master process in distributed training.

    Args:
        logger (logging.Logger): Logger instance to use
        level (str): Log level ('info', 'warning', 'error', 'debug')
        message (str): Message to log
    """
    if is_master_process():
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)


def normalize_checkpoint_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize checkpoint keys by removing legacy prefixes and filtering keys.

    - Remove the 'hyena.' prefix if present.
    - Drop AF-feature related keys that are no longer used.

    Args:
        state_dict (dict[str, Any]): Input state dictionary.

    Returns:
        dict[str, Any]: Normalized state dictionary.
    """
    normalized: dict[str, Any] = {}
    for original_key, tensor_value in state_dict.items():
        adjusted_key: str = original_key

        # 1) Drop common top-level prefixes produced by various training wrappers
        #    - 'model.'  : typical Lightning/Engine wrapper
        #    - 'module.' : DDP wrappers
        #    - 'hyena.'  : legacy base_model_prefix
        adjusted_key = adjusted_key.removeprefix("module.")

        # 2) Some checkpoints insert an extra 'layer' node inside submodules.
        #    Map these to the current module names expected by our implementation.
        #    Examples:
        #      - mixer.layer.in_proj.*   -> mixer.in_proj.*
        #      - mixer.layer.out_proj.*  -> mixer.out_proj.*
        #      - mixer.layer.short_filter.* -> mixer.short_filter.*
        #      - mixer.layer.filter_fn.* -> mixer.filter_fn.*
        #      - mlp.layer.fc1.*         -> mlp.fc1.*
        #      - mlp.layer.fc2.*         -> mlp.fc2.*
        if "classification_head_nt" in adjusted_key:
            adjusted_key = adjusted_key.replace(
                "classification_head_nt", "classification_head"
            )
        if "regression_head_nt" in adjusted_key:
            adjusted_key = adjusted_key.replace("regression_head_nt", "regression_head")

        adjusted_key = adjusted_key.replace(".mixer.layer.", ".mixer.")
        adjusted_key = adjusted_key.replace(".mlp.layer.", ".mlp.")

        normalized[adjusted_key] = tensor_value
    return normalized


def select_matching_state_dict(
    model: nn.Module, state_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Select only parameters whose keys exist in the provided model.

    Args:
        model (nn.Module): Target model instance.
        state_dict (dict[str, Any]): Candidate parameters.

    Returns:
        dict[str, Any]: Filtered parameters mapping matching the model.
    """
    model_state: dict[str, Any] = model.state_dict()
    filtered: dict[str, Any] = {}
    for key, tensor_value in state_dict.items():
        if key in model_state:
            model_tensor = model_state[key]
            # Load only when shapes are compatible to avoid RuntimeError
            try:
                if getattr(model_tensor, "shape", None) == getattr(
                    tensor_value, "shape", None
                ):
                    filtered[key] = tensor_value
                else:
                    print(
                        "Checkpoint key shape mismatch: "
                        f"{key} "
                        f"(ckpt {tuple(getattr(tensor_value, 'shape', ()))}, "
                        f"model {tuple(getattr(model_tensor, 'shape', ()))})"
                    )
            except Exception:
                # Best-effort fallback: compare number of elements when shape access fails
                try:
                    if model_tensor.numel() == tensor_value.numel():
                        filtered[key] = tensor_value
                    else:
                        print(f"Checkpoint key incompatible (numel): {key}")
                except Exception:
                    print(f"Checkpoint key incompatible: {key}")
        else:
            print(f"Checkpoint key not found in model: {key}")
    return filtered
