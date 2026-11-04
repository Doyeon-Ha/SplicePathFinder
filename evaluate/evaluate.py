"""Standalone evaluation script for IssaSplice models.

Loads a trained checkpoint, runs validation on a test set
(SpliceAI H5 or LMDB-based), and reports metrics.

Training-time settings (model architecture, label
preprocessing, loss type, etc.) are automatically loaded
from the checkpoint directory — either from
``train_config.yaml`` (new checkpoints) or by extracting
the appropriate YAML from the source-code snapshot
``tar.gz`` (legacy checkpoints).

Supports single-GPU and multi-GPU (DDP via torchrun)
evaluation.

Usage:
    Single-GPU::

        uv run -m evaluate.evaluate \\
            --checkpoint /path/to/best_model.pt \\
            --test-h5-path /path/to/test.h5

    Multi-GPU (DDP)::

        torchrun --nproc_per_node=N \\
            -m evaluate.evaluate \\
            --checkpoint /path/to/best_model.pt

    With explicit thresholds::

        uv run -m evaluate.evaluate \\
            --checkpoint /path/to/best_model.pt \\
            --thresholds 0.5 0.5 0.5
"""

import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import (
    DistributedSampler,
)

from evaluate.evaluator import Evaluator
from model.checkpoint_metadata import (
    UNSAFE_OVERRIDE_PREFIXES,
    CheckpointMetadata,
    UnsafeOverrideError,
    load_checkpoint_metadata,
)
from utils.distributed_utils import is_master_process

logger = logging.getLogger(__name__)

_LOSS_TYPE_TO_PROB_MODE = {
    "multi_class_focal_loss": "softmax",
}
_DEFAULT_PROB_MODE = "sigmoid"


# ── CLI ─────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Evaluate IssaSplice model checkpoint"),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=("config/evaluate/evaluate_hyenadna.yaml"),
        help="Path to evaluation YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.pt)."
            " Overrides evaluation.checkpoint_path"
            " in config."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        metavar=("ACCEPTOR", "DONOR", "NEITHER"),
        default=None,
        help=(
            "Per-class thresholds"
            " (acceptor donor neither)."
            " Overrides evaluation.thresholds"
            " in config."
        ),
    )
    parser.add_argument(
        "--test-h5-path",
        type=str,
        default=None,
        help=(
            "Path to a SpliceAI-format H5 test file."
            " Overrides evaluation.test_h5_path"
            " in config."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hyenadna",
        help=(
            "Model type hint used only for legacy"
            " checkpoints that lack train_config.yaml."
            " Determines which train YAML to extract"
            " from the source snapshot."
        ),
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help=(
            "Explicit path to a training config YAML."
            " Bypasses automatic discovery from the"
            " checkpoint directory. Useful when the"
            " training script did not save a config"
            " alongside checkpoints (e.g. SpliceAI)."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Also write the final test metrics, the checkpoint and the"
            " test H5 to this JSON file (rank 0)."
        ),
    )

    return parser.parse_args()


# ── Resolve helpers ─────────────────────────────────────


def _resolve_thresholds(
    cfg: DictConfig,
    cli_thresholds: list[float] | None,
) -> tuple[float, float, float] | None:
    """Resolve thresholds from CLI args or config.

    Args:
        cfg (DictConfig): Evaluation config.
        cli_thresholds (list[float] | None):
            CLI-provided thresholds.

    Returns:
        tuple[float, float, float] | None: Resolved
            thresholds or None.
    """
    if cli_thresholds is not None:
        return tuple(cli_thresholds)

    cfg_thresholds = OmegaConf.select(cfg, "evaluation.thresholds", default=None)
    if cfg_thresholds is not None:
        return tuple(cfg_thresholds)

    return None


def _resolve_test_h5_path(
    cfg: DictConfig,
    cli_h5_path: str | None,
) -> str | None:
    """Resolve H5 test path from CLI args or config.

    Args:
        cfg (DictConfig): Evaluation config.
        cli_h5_path (str | None): CLI-provided H5 path.

    Returns:
        str | None: Resolved H5 path or None.
    """
    if cli_h5_path is not None:
        return cli_h5_path

    return OmegaConf.select(cfg, "evaluation.test_h5_path", default=None)


def _resolve_checkpoint_path(
    cfg: DictConfig,
    cli_checkpoint: str | None,
) -> str:
    """Resolve checkpoint path from CLI or config.

    Args:
        cfg (DictConfig): Evaluation config.
        cli_checkpoint (str | None):
            CLI-provided checkpoint path.

    Returns:
        str: Resolved checkpoint path.

    Raises:
        ValueError: If no checkpoint path is specified.
    """
    path = cli_checkpoint or OmegaConf.select(
        cfg,
        "evaluation.checkpoint_path",
        default=None,
    )
    if not path:
        raise ValueError(
            "Checkpoint path must be specified via"
            " --checkpoint or"
            " evaluation.checkpoint_path in config."
        )

    return path


# ── Training config loading ─────────────────────────────


def _load_train_config(
    checkpoint_path: str,
    *,
    use_h5: bool,
    model_type: str = "hyenadna",
    train_config_path: str | None = None,
    ignore_embed_metadata: bool = False,
) -> DictConfig:
    """Load training config from checkpoint directory.

    Resolution order:

    0a. Explicit *train_config_path* (if provided).
    0b. ``metadata.training_config_snapshot`` embedded inside the ``.pt``
        file itself (new in feature 010 — single-file portability).
    1. ``{ckpt_dir}/train_config.yaml`` (new format,
       saved by ``train/train.py`` during training).
    2. ``{ckpt_dir}/config/train/`` directory (copied
       alongside source snapshot in some checkpoints).
    3. Latest ``source_snapshot_*.tar.gz`` in the
       checkpoint directory (legacy fallback).

    For (2) and (3), the specific YAML is chosen by
    *model_type* and *use_h5*:
    ``train_{model_type}_h5.yaml`` when *use_h5* is True,
    ``train_{model_type}.yaml`` otherwise.

    Args:
        checkpoint_path (str): Path to the ``.pt`` file.
        use_h5 (bool): Whether evaluation uses H5 data.
        model_type (str): Model type string for legacy
            config file name resolution.
        train_config_path (str | None): Explicit path to
            a training config YAML. Bypasses automatic
            discovery when provided.
        ignore_embed_metadata (bool): When ``True``, malformed
            embed metadata is silently dropped and the function
            falls through to the sidecar chain. Default
            ``False`` (fail-loud, FR-016).

    Returns:
        DictConfig: Loaded training configuration.

    Raises:
        FileNotFoundError: If no training config can be
            found in the checkpoint directory.
        MalformedCheckpointMetadataError: When embed metadata is
            malformed and *ignore_embed_metadata* is ``False``.
        IncompatibleCheckpointError: When embed metadata's schema_version
            is not supported by this evaluation code.
        UnknownModelClassIdError: When embed metadata's model_class_id
            is not in the registry.
    """
    # ── 0a. Explicit path ──────────────────────────────
    if train_config_path is not None:
        p = Path(train_config_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Explicit train config not found: {train_config_path}"
            )
        logger.info("Loading explicit train config: %s", p)
        return OmegaConf.load(str(p))

    # ── 0b. Embed metadata inside the .pt ─────────────
    metadata = load_checkpoint_metadata(
        checkpoint_path,
        ignore_embed_metadata=ignore_embed_metadata,
    )
    if metadata is not None:
        logger.info(
            "Loading train config from embed metadata"
            " (model_class_id=%s, schema_version=%d, created_at=%s)",
            metadata.model_class_id,
            metadata.schema_version,
            metadata.created_at,
        )
        return OmegaConf.create(metadata.training_config_snapshot)

    logger.info(
        "No 'metadata' key found in checkpoint — falling back to"
        " sidecar resolution chain (train_config.yaml → config/ → tarball)."
    )

    ckpt_dir = Path(checkpoint_path).parent

    # ── 1. New-format: train_config.yaml ────────────
    direct = ckpt_dir / "train_config.yaml"
    if direct.exists():
        logger.info("Loading train config from sidecar yaml: %s", direct)
        return OmegaConf.load(str(direct))

    # Determine legacy config filename
    suffix = "_h5" if use_h5 else ""
    legacy_name = f"train_{model_type}{suffix}.yaml"

    # ── 2. Copied config/ directory ─────────────────
    copied = ckpt_dir / "config" / "train" / legacy_name
    if copied.exists():
        logger.info("Loading legacy train config: %s", copied)
        return OmegaConf.load(str(copied))

    # ── 3. Source snapshot tar.gz ───────────────────
    pattern = str(ckpt_dir / "source_snapshot_*.tar.gz")
    archives = sorted(glob.glob(pattern))
    if not archives:
        raise FileNotFoundError(
            "No training config found in checkpoint"
            f" directory {ckpt_dir}. Looked for:"
            f" train_config.yaml,"
            f" config/train/{legacy_name},"
            " source_snapshot_*.tar.gz"
        )

    tar_path = archives[-1]
    member_name = f"config/train/{legacy_name}"
    logger.info(
        "Extracting %s from %s",
        member_name,
        Path(tar_path).name,
    )

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            member = tar.getmember(member_name)
            src = tar.extractfile(member)
            if src is None:
                raise FileNotFoundError(f"{member_name} not readable in {tar_path}")
            return OmegaConf.load(src)
    except KeyError:
        raise FileNotFoundError(
            f"{member_name} not found inside"
            f" {tar_path}. Available members can be"
            " listed with: tar -tzf <archive>"
        )


def _derive_probability_mode(
    train_cfg: DictConfig,
) -> str:
    """Derive probability mode from loss function type.

    Args:
        train_cfg (DictConfig): Training config.

    Returns:
        str: ``"softmax"`` or ``"sigmoid"``.
    """
    loss_type = OmegaConf.select(
        train_cfg,
        "training.loss_function.type",
        default=None,
    )
    if loss_type is None:
        return _DEFAULT_PROB_MODE
    return _LOSS_TYPE_TO_PROB_MODE.get(str(loss_type), _DEFAULT_PROB_MODE)


_LEGACY_MODEL_KEY_RENAMES = {
    "nt_hidden_state_layer_index": ("hidden_state_layer_index"),
}

_LEGACY_MODEL_KEY_ALIASES = {
    "hidden_size": [
        "base_backbone_output_dim",
        "local_feature_output_dim",
    ],
}


def _patch_legacy_model_keys(
    cfg: DictConfig,
) -> None:
    """Rename legacy model keys to current names.

    Handles two kinds of legacy gaps:

    1. **Renames** – old key replaced by a new name
       (``_LEGACY_MODEL_KEY_RENAMES``).
    2. **Aliases** – one old key that should be copied to
       multiple new keys
       (``_LEGACY_MODEL_KEY_ALIASES``).

    Mutates *cfg* in-place.

    Args:
        cfg (DictConfig): Merged config with a ``model``
            section.
    """
    model = cfg.model

    for old, new in _LEGACY_MODEL_KEY_RENAMES.items():
        old_val = OmegaConf.select(model, old, default=None)
        new_val = OmegaConf.select(model, new, default=None)
        if old_val is not None and new_val is None:
            OmegaConf.update(model, new, old_val)
            logger.info(
                "Patched legacy key model.%s -> %s",
                old,
                new,
            )

    for old, targets in _LEGACY_MODEL_KEY_ALIASES.items():
        old_val = OmegaConf.select(model, old, default=None)
        if old_val is None:
            continue
        for tgt in targets:
            tgt_val = OmegaConf.select(model, tgt, default=None)
            if tgt_val is None:
                OmegaConf.update(model, tgt, old_val)
                logger.info(
                    "Patched legacy alias model.%s -> %s",
                    old,
                    tgt,
                )

    return


def _detect_unsafe_overrides(
    metadata: CheckpointMetadata,
    eval_cfg: DictConfig,
) -> None:
    """Reject user overrides that would mutate weight semantics.

    Walks each unsafe-prefixed subtree (``model``, ``data``,
    ``tokenizer``) present in *eval_cfg* and compares with the matching
    subtree inside ``metadata.training_config_snapshot``. Any value
    mismatch — including a key the user added that wasn't in the
    snapshot — surfaces as :class:`UnsafeOverrideError` (FR-011).

    This intentionally runs **before** :func:`_build_merged_config`
    because that function gives precedence to the training cfg snapshot
    and would otherwise silently overwrite user overrides, making
    post-merge diffs useless.

    Subtrees absent from the user's *eval_cfg* are skipped (no
    override). The check is therefore opt-in by the user adding
    ``model:``, ``data:``, or ``tokenizer:`` blocks to their evaluation
    YAML / CLI.

    Args:
        metadata (CheckpointMetadata): Embed metadata block from the
            checkpoint.
        eval_cfg (DictConfig): The user-supplied evaluation cfg, before
            merge with the training cfg snapshot.

    Raises:
        UnsafeOverrideError: When an unsafe-prefixed key in *eval_cfg*
            differs from the value recorded in the metadata snapshot.
    """
    snapshot = metadata.training_config_snapshot

    for prefix in ("model", "data", "tokenizer"):
        # Defensive: only enforce prefixes that the spec marks unsafe.
        if f"{prefix}." not in UNSAFE_OVERRIDE_PREFIXES:
            continue

        eval_sub_node = OmegaConf.select(eval_cfg, prefix, default=None)
        if eval_sub_node is None:
            continue
        eval_sub = OmegaConf.to_container(eval_sub_node, resolve=True)
        if not isinstance(eval_sub, dict):
            continue

        snap_sub = snapshot.get(prefix)
        snap_sub_dict: dict[str, Any] = snap_sub if isinstance(snap_sub, dict) else {}

        for key, user_val in eval_sub.items():
            snap_val = snap_sub_dict.get(key)
            if user_val != snap_val:
                raise UnsafeOverrideError(
                    f"Override of {prefix}.{key!r} is not allowed at "
                    "evaluation time (would invalidate weight semantics). "
                    f"metadata={snap_val!r}, user={user_val!r}. "
                    "Either remove the override from your evaluate yaml/CLI "
                    "or use a checkpoint trained with the desired settings."
                )


_INFO_LOGGED_SAFE_KEYS: set[str] = set()


def _log_safe_override_usage(cfg: DictConfig) -> None:
    """Log INFO for each ``evaluation.*`` key actually present in cfg.

    The training cfg snapshot does not contain ``evaluation.*`` keys
    (different Hydra config group), so a value-vs-metadata comparison is
    not meaningful here (FR-009). We instead surface the values that
    will drive evaluation so users can confirm what got picked up from
    YAML defaults vs CLI overrides.

    Args:
        cfg (DictConfig): The merged config used by the evaluator.
    """
    if not is_master_process():
        return
    for key in (
        "evaluation.batch_size",
        "evaluation.thresholds",
        "evaluation.test_h5_path",
        "evaluation.probability_mode",
        "evaluation.num_workers",
    ):
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            continue
        if key in _INFO_LOGGED_SAFE_KEYS:
            continue
        logger.info("safe override applied: %s=%r", key, value)
        _INFO_LOGGED_SAFE_KEYS.add(key)


def _build_merged_config(
    eval_cfg: DictConfig,
    train_cfg: DictConfig,
    *,
    checkpoint_path: str,
) -> DictConfig:
    """Merge training config into evaluation config.

    Extracts ``training`` and ``model`` sections (and
    ``logging`` when present) from *train_cfg* and merges
    them into *eval_cfg*.  Sets
    ``model.initial_weights_path`` and derives
    ``evaluation.probability_mode``.

    Args:
        eval_cfg (DictConfig): User evaluation config
            (evaluation + mlflow sections).
        train_cfg (DictConfig): Loaded training config.
        checkpoint_path (str): Path to checkpoint ``.pt``.

    Returns:
        DictConfig: Merged config ready for Evaluator.
    """
    sections = {
        "training": OmegaConf.to_container(
            train_cfg.training,
            resolve=True,
        ),
        "model": OmegaConf.to_container(
            train_cfg.model,
            resolve=True,
        ),
    }
    logging_cfg = OmegaConf.select(train_cfg, "logging", default=None)
    if logging_cfg is not None:
        sections["logging"] = OmegaConf.to_container(
            logging_cfg,
            resolve=True,
        )

    merged = OmegaConf.merge(
        eval_cfg,
        OmegaConf.create(sections),
    )

    merged.model.initial_weights_path = checkpoint_path

    _patch_legacy_model_keys(merged)

    prob_mode = OmegaConf.select(
        eval_cfg,
        "evaluation.probability_mode",
        default=None,
    )
    if prob_mode is None:
        merged.evaluation.probability_mode = _derive_probability_mode(train_cfg)

    return merged


# ── DDP ─────────────────────────────────────────────────


def _init_ddp() -> tuple[int, int]:
    """Initialise DDP from the launcher's environment.

    The launcher (Slurm ``srun`` / torchrun) owns GPU allocation and exports
    ``CUDA_VISIBLE_DEVICES``; this only binds each rank to its ``local_rank``
    within the visible set and initialises the NCCL process group.

    Returns:
        tuple[int, int]: (local_rank, world_size).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # % device_count keeps the index valid under Slurm per-task GPU isolation.
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    backend = "nccl"
    if not dist.is_nccl_available():
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        rank=global_rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )

    return local_rank, world_size


# ── DataLoader ──────────────────────────────────────────


def _build_h5_test_loader(
    h5_path: str,
    batch_size: int,
    num_workers: int,
    world_size: int,
    local_rank: int,
) -> DataLoader:
    """Create a DataLoader from a SpliceAI H5 file.

    Args:
        h5_path (str): Path to the H5 file.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        world_size (int): Total number of processes.
        local_rank (int): Local process rank.

    Returns:
        DataLoader: Configured data loader.
    """
    from train.dataset import SplicingH5Dataset

    dataset = SplicingH5Dataset(h5_path)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


# ── Main evaluate ───────────────────────────────────────


_THRESHOLD_FREE_KEYS = {
    "topk_accuracy_acceptor",
    "topk_accuracy_donor",
    "topk_k_acceptor",
    "topk_k_donor",
    "auc_pr_acceptor",
    "auc_pr_donor",
    "auc_roc_acceptor",
    "auc_roc_donor",
    "mcc_acceptor",
    "mcc_donor",
}


def evaluate(
    cfg: DictConfig,
    cfg_model: DictConfig,
    *,
    checkpoint_path: str,
    thresholds: (tuple[float, float, float] | None) = None,
    test_h5_path: str | None = None,
    model_type: str = "hyenadna",
    train_config_path: str | None = None,
) -> dict[str, float]:
    """Run model evaluation on a test set.

    Training-time settings are auto-loaded from the
    checkpoint directory and merged into *cfg*.

    Args:
        cfg (DictConfig): Evaluation config
            (evaluation + mlflow sections).
        cfg_model (DictConfig): Legacy model config
            (config/issasplice/model.yaml).
        checkpoint_path (str): Path to the model
            checkpoint.
        thresholds (tuple[float, float, float] | None):
            Optional per-class thresholds
            (acceptor, donor, neither).
        test_h5_path (str | None): Path to SpliceAI H5
            test file. When None, LMDB-based evaluation
            is not yet supported in standalone mode.
        model_type (str): Model type hint for legacy
            checkpoint config resolution.
        train_config_path (str | None): Explicit path
            to a training config YAML. Bypasses automatic
            discovery from the checkpoint directory.

    Returns:
        dict[str, float]: Evaluation metrics.
    """
    use_h5 = test_h5_path is not None

    ignore_embed_metadata = bool(
        OmegaConf.select(
            cfg,
            "evaluation.ignore_embed_metadata",
            default=False,
        )
    )
    # Pre-load metadata once so we can both (a) build train_cfg from the
    # snapshot and (b) diff the merged cfg against the snapshot to detect
    # unsafe overrides (FR-011) — the second pass would otherwise need a
    # redundant torch.load.
    metadata = load_checkpoint_metadata(
        checkpoint_path,
        ignore_embed_metadata=ignore_embed_metadata,
    )

    # FR-011: detect unsafe overrides BEFORE merging the training cfg
    # snapshot in. _build_merged_config gives precedence to train_cfg's
    # model section, so a post-merge diff would silently miss user
    # overrides — we must compare the user's eval_cfg against the
    # metadata snapshot instead.
    if metadata is not None:
        _detect_unsafe_overrides(metadata, cfg)

    train_cfg = _load_train_config(
        checkpoint_path,
        use_h5=use_h5,
        model_type=model_type,
        train_config_path=train_config_path,
        ignore_embed_metadata=ignore_embed_metadata,
    )
    cfg = _build_merged_config(
        cfg,
        train_cfg,
        checkpoint_path=checkpoint_path,
    )

    # FR-009: log safe override usage (evaluation.* keys live outside
    # the training cfg snapshot, so direct value comparison with
    # metadata is not meaningful).
    _log_safe_override_usage(cfg)

    if is_master_process():
        logger.info(f"probability_mode: {cfg.evaluation.probability_mode}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        local_rank, world_size = _init_ddp()
        if is_master_process():
            logger.info(
                f"DDP initialized: local_rank={local_rank}, world_size={world_size}"
            )
    else:
        if is_master_process():
            logger.info("Single GPU evaluation mode")

    evaluator = Evaluator(
        cfg,
        cfg_model,
        local_rank=local_rank,
        world_size=world_size,
    )

    if is_master_process():
        logger.info("Evaluate-only mode started")

    if test_h5_path is not None:
        if is_master_process():
            logger.info(f"Loading H5 test set: {test_h5_path}")

        eval_bs = cfg.evaluation.batch_size
        eval_workers = cfg.evaluation.num_workers
        test_loader = _build_h5_test_loader(
            h5_path=test_h5_path,
            batch_size=eval_bs,
            num_workers=eval_workers,
            world_size=world_size,
            local_rank=local_rank,
        )
        n_samples = len(test_loader.dataset)
        sample_info = f"{n_samples} samples from H5"
    else:
        raise NotImplementedError(
            "Standalone LMDB evaluation is not yet supported. Provide --test-h5-path."
        )

    if is_master_process():
        logger.info(f"Using {sample_info} for evaluation")

    if thresholds is not None and is_master_process():
        thr_a, thr_d, thr_n = thresholds
        logger.info(
            "Using class thresholds: "
            f"acceptor={thr_a:.4f}, "
            f"donor={thr_d:.4f}, "
            f"neither={thr_n:.4f}"
        )

    test_metrics = evaluator.evaluate(
        test_loader,
        thresholds=thresholds,
    )

    if is_master_process():
        if thresholds is None:
            display_metrics = {
                k: v for k, v in test_metrics.items() if k in _THRESHOLD_FREE_KEYS
            }
        else:
            display_metrics = test_metrics

        formatted = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in display_metrics.items()
        )
        print(f"Final test metrics:\n  {formatted}")


    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    return test_metrics


def main() -> None:
    """CLI entrypoint for model evaluation.

    Single-GPU::

        uv run -m evaluate.evaluate \\
            --checkpoint /path/to/best_model.pt \\
            --test-h5-path /path/to/test.h5

    Multi-GPU (DDP)::

        torchrun --nproc_per_node=N \\
            -m evaluate.evaluate \\
            --checkpoint /path/to/best_model.pt \\
            --test-h5-path /path/to/test.h5
    """
    args = parse_args()

    config_path = Path(to_absolute_path(args.config))
    if not config_path.exists():
        print(
            f"Config file not found: {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    cfg_model_path = Path(to_absolute_path("config/issasplice/model.yaml"))
    cfg_model = OmegaConf.load(cfg_model_path)

    checkpoint_path = _resolve_checkpoint_path(cfg, args.checkpoint)
    thresholds = _resolve_thresholds(cfg, args.thresholds)
    test_h5_path = _resolve_test_h5_path(cfg, args.test_h5_path)

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )

    test_metrics = evaluate(
        cfg,
        cfg_model,
        checkpoint_path=checkpoint_path,
        thresholds=thresholds,
        test_h5_path=test_h5_path,
        model_type=args.model_type,
        train_config_path=args.train_config,
    )
    if args.metrics_json and int(os.environ.get("RANK", "0")) == 0:
        payload = {
            "metrics": {
                k: v.item() if hasattr(v, "item") else v
                for k, v in test_metrics.items()
            },
            "checkpoint": str(checkpoint_path),
            "test_h5_path": str(test_h5_path),
        }
        Path(args.metrics_json).write_text(json.dumps(payload, indent=2) + "\n")

    return


if __name__ == "__main__":
    main()
