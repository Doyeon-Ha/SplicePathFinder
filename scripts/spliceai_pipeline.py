"""SpliceAI H5 training and evaluation pipeline.

Standalone pipeline for SpliceAI models using H5 datasets.
Supports single/multi-GPU (DDP) training, validation with AUPRC
metrics, checkpoint saving/resuming, MLflow experiment tracking,
LR scheduling, mixed precision, early stopping, and standalone
evaluation of trained checkpoints.
"""

import argparse
import json
import logging
import os
import shutil
import socket
from datetime import datetime

import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.spliceai import SpliceAI
from train.dataset import SplicingH5Dataset
from train.online_metrics_calculator import OnlineMetricsCalculator
from utils.distributed_utils import (
    cleanup_distributed,
    is_master_process,
    set_up_ddp,
)
from utils.slack_notifier import SlackNotifier
from utils.utils import (
    archive_code_snapshot,
    reduce_across_processes,
    safe_tensor_to_numpy,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_CONTEXT_LENGTHS = {80, 400, 2000, 10000}

CONTEXT_LENGTH_TO_MODEL = {
    80: "80nt",
    400: "400nt",
    2000: "2k",
    10000: "10k",
}

DNA_ENCODE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def seq_to_onehot(seq: str) -> torch.Tensor:
    """Convert DNA sequence string to one-hot tensor.

    Args:
        seq (str): DNA sequence string (A/C/G/T).

    Returns:
        torch.Tensor: One-hot tensor of shape ``(4, L)``.
    """
    seq_len = len(seq)
    onehot = torch.zeros(4, seq_len, dtype=torch.float32)
    for i, char in enumerate(seq):
        if char in DNA_ENCODE_MAP:
            onehot[DNA_ENCODE_MAP[char], i] = 1.0
    return onehot


def spliceai_collate_fn(
    batch: list[dict],
) -> dict[str, torch.Tensor]:
    """Collate SpliceAI H5 samples into batches.

    Converts DNA sequences to one-hot encoding and labels
    to class indices. Padded positions (all-zero labels) get
    index ``-1`` for ``CrossEntropyLoss(ignore_index=-1)``.

    Args:
        batch (list[dict]): Samples from ``SplicingH5Dataset``.

    Returns:
        dict[str, torch.Tensor]: ``inputs`` ``[B, 4, L]``
            and ``targets`` ``[B, L]``.
    """
    inputs = []
    targets = []

    for sample in batch:
        ref_seq = str(sample["ref"])
        labels = sample["labels"]

        onehot = seq_to_onehot(ref_seq)
        inputs.append(onehot)

        if isinstance(labels, torch.Tensor):
            target_idx = torch.argmax(labels, dim=0)
            padding_mask = labels.sum(dim=0) == 0
            target_idx[padding_mask] = -1
            targets.append(target_idx)

    return {
        "inputs": torch.stack(inputs),
        "targets": torch.stack(targets),
    }


def create_model(cfg: DictConfig) -> tuple[nn.Module, int]:
    """Create SpliceAI model from config.

    Args:
        cfg (DictConfig): Config with ``model`` section.

    Returns:
        tuple[nn.Module, int]: Model and context_length.

    Raises:
        ValueError: If ``spliceai_context_length`` is invalid.
    """
    cl = cfg.model.spliceai_context_length
    if cl not in VALID_CONTEXT_LENGTHS:
        raise ValueError(
            f"Invalid spliceai_context_length={cl}. "
            f"Valid: {sorted(VALID_CONTEXT_LENGTHS)}"
        )

    variant = CONTEXT_LENGTH_TO_MODEL[cl]
    model = SpliceAI.from_preconfigured(variant)
    seq_len = cfg.model.target_region_length + cl

    if is_master_process():
        logger.info(f"SpliceAI model: {variant}, context={cl}, seq_length={seq_len}")

    return model, cl


def create_dataloaders(
    cfg: DictConfig,
    world_size: int,
    rank: int,
) -> tuple[
    DataLoader,
    DataLoader | None,
    DataLoader | None,
    DistributedSampler | None,
]:
    """Create train/val/test data loaders from H5 files.

    If ``val_h5_path`` is set, a separate val dataset is loaded.
    Otherwise the train dataset is split by ``train_split`` /
    ``val_split``. ``test_h5_path`` loads an independent test set.

    Args:
        cfg (DictConfig): Config with ``data`` section.
        world_size (int): Number of distributed processes.
        rank (int): Current process rank.

    Returns:
        tuple: ``(train_loader, val_loader, test_loader,
            train_sampler)``. Loaders may be ``None``.
    """
    is_distributed = world_size > 1
    batch_size = cfg.training.batch_size
    num_workers = OmegaConf.select(cfg, "training.num_workers", default=4)

    train_dataset = SplicingH5Dataset(cfg.data.train_h5_path)

    val_h5 = OmegaConf.select(cfg, "data.val_h5_path")
    test_h5 = OmegaConf.select(cfg, "data.test_h5_path")

    val_dataset = None
    test_dataset = None

    if val_h5:
        val_dataset = SplicingH5Dataset(val_h5)
        if is_master_process():
            logger.info(
                f"Train: {len(train_dataset)} samples, "
                f"Val (separate H5): {len(val_dataset)}"
            )
    else:
        ts = cfg.data.train_split
        total = len(train_dataset)
        train_size = int(total * ts)
        val_size = total - train_size
        if val_size > 0:
            g = torch.Generator().manual_seed(cfg.data.random_seed)
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=g,
            )
        if is_master_process():
            val_len = len(val_dataset) if val_dataset else 0
            logger.info(f"Train: {len(train_dataset)}, Val (split): {val_len}")

    if test_h5:
        test_dataset = SplicingH5Dataset(test_h5)
        if is_master_process():
            logger.info(f"Test (separate H5): {len(test_dataset)} samples")

    train_sampler = None
    val_sampler = None
    test_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        if test_dataset is not None:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

    pw = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=spliceai_collate_fn,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
        drop_last=(not is_distributed),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=spliceai_collate_fn,
            pin_memory=True,
            persistent_workers=pw,
            prefetch_factor=2 if pw else None,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=spliceai_collate_fn,
            pin_memory=True,
            persistent_workers=pw,
            prefetch_factor=2 if pw else None,
        )

    return train_loader, val_loader, test_loader, train_sampler


def create_scheduler(
    optimizer: optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
) -> optim.lr_scheduler.LRScheduler | None:
    """Create LR scheduler with optional warmup.

    Args:
        optimizer (optim.Optimizer): Optimizer instance.
        cfg (DictConfig): Config with ``training.scheduler``
            and ``training.warmup_proportion``.
        total_steps (int): Total training steps.

    Returns:
        optim.lr_scheduler.LRScheduler | None: Scheduler or
            ``None`` if not configured.
    """
    sched_type = OmegaConf.select(cfg, "training.scheduler.type")
    if not sched_type:
        return None

    warmup_prop = OmegaConf.select(cfg, "training.warmup_proportion", default=0.0)
    warmup_steps = int(total_steps * warmup_prop)
    main_steps = max(total_steps - warmup_steps, 1)

    if sched_type == "cosine":
        main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_steps)
    elif sched_type == "linear":
        main_sched = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=main_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    if warmup_steps > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_steps],
        )
        return scheduler

    return main_sched


def save_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_model_epoch: int,
    is_best: bool,
    is_distributed: bool,
    context_length: int,
) -> None:
    """Save training checkpoint (master only).

    Args:
        checkpoint_dir (str): Directory for checkpoint files.
        model (nn.Module): Model (may be DDP-wrapped).
        optimizer (optim.Optimizer): Optimizer.
        scheduler (LRScheduler | None): Scheduler.
        epoch (int): Current epoch.
        global_step (int): Global training step.
        best_val_loss (float): Best validation loss so far.
        best_model_epoch (int): Epoch of best model.
        is_best (bool): Whether current is best model.
        is_distributed (bool): Whether DDP is active.
        context_length (int): Model context length for
            auto-detection during evaluation.
    """
    if not is_master_process():
        return

    sd = model.module.state_dict() if is_distributed else (model.state_dict())

    state = {
        "model_state_dict": sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_model_epoch": best_model_epoch,
        "context_length": context_length,
    }

    recent = os.path.join(checkpoint_dir, "most_recent_model.pt")
    torch.save(state, recent)

    epoch_file = os.path.join(
        checkpoint_dir,
        f"epoch_{epoch:03d}_model.pt",
    )
    torch.save(state, epoch_file)

    if is_best:
        best = os.path.join(checkpoint_dir, "best_model.pt")
        shutil.copy2(recent, best)
        logger.info(
            f"New best model saved (epoch {epoch}, val_loss={best_val_loss:.6f})"
        )

    return


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    is_distributed: bool,
) -> dict:
    """Load checkpoint and restore training state.

    Args:
        path (str): Path to checkpoint ``.pt`` file.
        model (nn.Module): Model (may be DDP-wrapped).
        optimizer (optim.Optimizer): Optimizer.
        scheduler (LRScheduler | None): Scheduler.
        device (torch.device): Target device.
        is_distributed (bool): Whether DDP is active.

    Returns:
        dict: Loaded checkpoint dict with ``epoch``,
            ``global_step``, ``best_val_loss``,
            ``best_model_epoch``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    target = model.module if is_distributed else model
    target.load_state_dict(ckpt["model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if is_master_process():
        logger.info(f"Resumed from epoch {ckpt['epoch']}, step {ckpt['global_step']}")

    return ckpt


def _get_amp_dtype(
    cfg: DictConfig,
) -> torch.dtype | None:
    """Resolve AMP dtype from config.

    Args:
        cfg (DictConfig): Config with ``training.amp_dtype``.

    Returns:
        torch.dtype | None: ``torch.bfloat16``,
            ``torch.float16``, or ``None``.
    """
    amp_str = OmegaConf.select(cfg, "training.amp_dtype")
    if not amp_str:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    dtype = mapping.get(str(amp_str).lower())
    if dtype is None:
        raise ValueError(f"Invalid amp_dtype: {amp_str}. Valid: bf16, fp16, null")
    return dtype


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    cfg: DictConfig,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    context_length: int,
) -> tuple[float, int]:
    """Train one epoch with AMP, grad accum, and clipping.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.CrossEntropyLoss): Loss function.
        device (torch.device): Compute device.
        cfg (DictConfig): Full config.
        scheduler (LRScheduler | None): LR scheduler.
        scaler (torch.amp.GradScaler | None): AMP scaler.
        epoch (int): Current epoch number.
        global_step (int): Current global step.
        context_length (int): Model context length.

    Returns:
        tuple[float, int]: ``(avg_loss, updated_global_step)``
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_dtype = _get_amp_dtype(cfg)

    accum = OmegaConf.select(
        cfg,
        "training.gradient_accumulation_steps",
        default=1,
    )
    clip_norm = OmegaConf.select(cfg, "training.gradient_clipping.max_norm")
    log_interval = OmegaConf.select(cfg, "logging.log_interval", default=10)
    offset = context_length // 2
    trl = cfg.model.target_region_length

    optimizer.zero_grad()
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}",
        disable=not is_master_process(),
    )

    for step, batch in enumerate(pbar):
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        if targets.shape[1] > trl:
            targets = targets[:, offset : offset + trl]

        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(inputs)
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
        else:
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        loss_scaled = loss / accum

        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step + 1) % accum == 0:
            if scaler:
                if clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()
            global_step += 1

        total_loss += loss.item()
        num_batches += 1

        if is_master_process() and (step + 1) % log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{cur_lr:.2e}",
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    context_length: int,
    target_region_length: int,
    amp_dtype: torch.dtype | None,
    topk_fraction: float,
    world_size: int,
) -> dict[str, float]:
    """Run validation and compute classification metrics.

    Computes loss, per-class AUPRC, precision, recall, and
    top-k accuracy for acceptor/donor classes.

    Args:
        model (nn.Module): Model in eval mode.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.CrossEntropyLoss): Loss function.
        device (torch.device): Compute device.
        context_length (int): Model context length.
        target_region_length (int): Target region length.
        amp_dtype (torch.dtype | None): AMP dtype.
        topk_fraction (float): Top-k fraction for accuracy.
        world_size (int): DDP world size.

    Returns:
        dict[str, float]: Validation metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    offset = context_length // 2
    trl = target_region_length

    calc = OnlineMetricsCalculator(
        logger=logger,
        num_classes=2,
        use_histogram=False,
    )

    all_acceptor_scores = []
    all_acceptor_labels = []
    all_donor_scores = []
    all_donor_labels = []

    for batch in dataloader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        if targets.shape[1] > trl:
            targets = targets[:, offset : offset + trl]

        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(inputs)
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
        else:
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        total_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(logits.float(), dim=-1)

        valid_mask = targets != -1
        for i in range(inputs.size(0)):
            m = valid_mask[i]
            if m.sum() == 0:
                continue

            p_np = safe_tensor_to_numpy(probs[i][m][:, :2])
            t_np = safe_tensor_to_numpy(targets[i][m])

            lbl = np.zeros((p_np.shape[0], 2), dtype=np.float32)
            lbl[t_np == 0, 0] = 1.0  # acceptor
            lbl[t_np == 1, 1] = 1.0  # donor

            calc.update_auc_data(
                labels=lbl[np.newaxis],
                probs=p_np[np.newaxis],
            )

            pred_binary = np.zeros_like(lbl)
            pred_binary[p_np[:, 0] > 0.5, 0] = 1.0
            pred_binary[p_np[:, 1] > 0.5, 1] = 1.0
            calc.update(
                preds=pred_binary[np.newaxis],
                labels=lbl[np.newaxis],
            )

            all_acceptor_scores.append(p_np[:, 0])
            all_acceptor_labels.append(lbl[:, 0])
            all_donor_scores.append(p_np[:, 1])
            all_donor_labels.append(lbl[:, 1])

    avg_loss = total_loss / max(num_batches, 1)

    if world_size > 1:
        loss_t = torch.tensor([avg_loss], device=device)
        avg_loss = reduce_across_processes(loss_t).item()

    metrics: dict[str, float] = {"loss": avg_loss}

    if is_master_process() and num_batches > 0:
        auc_metrics = calc.compute_metrics()
        metrics.update(auc_metrics)

        for cls_name, scores_list, labels_list in [
            ("acceptor", all_acceptor_scores, all_acceptor_labels),
            ("donor", all_donor_scores, all_donor_labels),
        ]:
            if not scores_list:
                continue
            sc = np.concatenate(scores_list)
            lb = np.concatenate(labels_list)
            pos_count = int(lb.sum())
            if pos_count == 0:
                continue
            k = max(1, int(pos_count * topk_fraction))
            topk_idx = np.argpartition(sc, -k)[-k:]
            topk_acc = float(lb[topk_idx].mean())
            metrics[f"topk_accuracy_{cls_name}"] = topk_acc

    return metrics


def setup_mlflow(
    cfg: DictConfig,
    run_name: str,
    context_length: int,
    world_size: int,
) -> bool:
    """Initialize MLflow tracking and log hyperparameters.

    Args:
        cfg (DictConfig): Full configuration.
        run_name (str): Run name for MLflow.
        context_length (int): Model context length.
        world_size (int): Number of GPUs.

    Returns:
        bool: Whether MLflow is active.
    """
    mlflow_cfg = OmegaConf.select(cfg, "mlflow")
    if not mlflow_cfg or not mlflow_cfg.enabled:
        return False

    uri = OmegaConf.select(mlflow_cfg, "tracking_uri")
    if not uri:
        uri = os.environ.get("MLFLOW_URI")
    if uri:
        mlflow.set_tracking_uri(uri)

    exp_id = OmegaConf.select(mlflow_cfg, "experiment_id")
    if exp_id is not None:
        mlflow.set_experiment(experiment_id=str(exp_id))
    else:
        mlflow.set_experiment(experiment_name=mlflow_cfg.experiment_name)

    tags_raw = OmegaConf.select(mlflow_cfg, "tags", default={})
    tags = (
        OmegaConf.to_container(OmegaConf.create(tags_raw), resolve=True)
        if tags_raw
        else {}
    )
    tags["run_name"] = run_name
    tags["hostname"] = socket.gethostname()
    tags["timestamp"] = datetime.now().isoformat()
    tags["gpu_count"] = str(world_size)
    tags["model_type"] = "spliceai"

    display_name = OmegaConf.select(mlflow_cfg, "run_name")
    if not display_name:
        display_name = run_name

    mlflow.start_run(run_name=display_name, tags=tags)

    flat = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    truncated = {k: str(v)[:250] for k, v in flat.items()}
    mlflow.log_params(truncated)

    variant = CONTEXT_LENGTH_TO_MODEL[context_length]
    seq_len = cfg.model.target_region_length + context_length
    mlflow.log_params(
        {
            "spliceai_variant": variant,
            "spliceai_context_length": context_length,
            "spliceai_seq_length": seq_len,
            "spliceai_target_region": (cfg.model.target_region_length),
        }
    )

    return True


def _flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict:
    """Flatten a nested dict for MLflow param logging.

    Args:
        d (dict): Nested dictionary.
        parent_key (str): Prefix for keys.
        sep (str): Separator between levels.

    Returns:
        dict: Flattened dictionary.
    """
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _check_early_stopping(
    monitor_value: float,
    best_value: float,
    wait: int,
    cfg_es: DictConfig,
) -> tuple[bool, float, int, bool]:
    """Check early stopping condition.

    Args:
        monitor_value (float): Current metric value.
        best_value (float): Best metric value so far.
        wait (int): Epochs without improvement.
        cfg_es (DictConfig): Early stopping config.

    Returns:
        tuple: ``(should_stop, new_best, new_wait,
            improved)``.
    """
    mode = cfg_es.mode
    delta = OmegaConf.select(cfg_es, "min_delta", default=0.0)
    patience = cfg_es.patience

    if mode == "min":
        improved = monitor_value < (best_value - delta)
    else:
        improved = monitor_value > (best_value + delta)

    if improved:
        return False, monitor_value, 0, True

    wait += 1
    should_stop = wait >= patience
    return should_stop, best_value, wait, False


def train(cfg: DictConfig, resume_path: str | None = None) -> None:
    """Orchestrate full SpliceAI training pipeline.

    Handles single/multi-GPU training, validation, checkpoint
    saving, MLflow logging, early stopping, and test evaluation.

    Args:
        cfg (DictConfig): Full configuration.
        resume_path (str | None): Path to checkpoint for
            resuming training.
    """
    is_resuming = resume_path is not None

    if not is_resuming:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OmegaConf.update(
            cfg,
            "training.run_name",
            f"{cfg.training.run_name}_{timestamp}",
        )

    run_name = cfg.training.run_name

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    local_rank = 0

    if is_ddp:
        local_rank, world_size = set_up_ddp(cfg)
        # % device_count keeps the index valid under Slurm per-task GPU isolation.
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        # Single-process run: GPU is allocated by the launcher (Slurm srun),
        # which sets CUDA_VISIBLE_DEVICES; use the first visible device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_master_process():
        logger.info(f"Run: {run_name}")
        logger.info(f"Device: {device}, GPUs: {world_size}")

    checkpoint_dir = os.path.join(cfg.training.checkpoint_dir, run_name)

    if not is_resuming:
        if is_master_process():
            if os.path.exists(checkpoint_dir):
                raise FileExistsError(f"Checkpoint dir exists: {checkpoint_dir}")
            os.makedirs(checkpoint_dir)
        if is_ddp:
            dist.barrier()

    model, context_length = create_model(cfg)
    model = model.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    lr = cfg.training.learning_rate
    wd = OmegaConf.select(cfg, "training.weight_decay", default=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    cw_raw = OmegaConf.select(cfg, "loss.class_weights")
    if cw_raw is None:
        cw_raw = OmegaConf.select(cfg, "training.class_weights")

    if cw_raw is not None:
        class_weights = torch.tensor(list(cw_raw), dtype=torch.float32).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
    ) = create_dataloaders(cfg, world_size, local_rank)

    steps_per_epoch = len(train_loader)
    accum = OmegaConf.select(
        cfg,
        "training.gradient_accumulation_steps",
        default=1,
    )
    optim_steps_per_epoch = max(steps_per_epoch // accum, 1)
    total_steps = optim_steps_per_epoch * cfg.training.epochs

    scheduler = create_scheduler(optimizer, cfg, total_steps)

    amp_dtype = _get_amp_dtype(cfg)
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_model_epoch = 0

    if is_resuming:
        ckpt = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            device,
            is_ddp,
        )
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_model_epoch = ckpt.get("best_model_epoch", 0)

    use_mlflow = False
    if is_master_process():
        use_mlflow = setup_mlflow(cfg, run_name, context_length, world_size)
        archive_code_snapshot(
            checkpoint_dir,
            directories=[
                "config",
                "model",
                "train",
                "utils",
                "scripts",
            ],
        )

    slack_notifier = SlackNotifier.from_config(cfg, run_name=run_name)
    slack_notifier.on_train_start(
        {
            "model": f"spliceai_{CONTEXT_LENGTH_TO_MODEL[context_length]}",
            "context_length": context_length,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "world_size": world_size,
            "train_h5_path": str(OmegaConf.select(cfg, "data.train_h5_path") or "n/a"),
            "start_epoch": start_epoch + 1,
            "checkpoint_dir": checkpoint_dir,
        }
    )

    es_enabled = OmegaConf.select(
        cfg,
        "training.evaluation.early_stopping.enabled",
        default=False,
    )
    es_wait = 0
    es_best = best_val_loss
    topk_frac = OmegaConf.select(
        cfg,
        "training.evaluation.topk_fraction",
        default=0.5,
    )

    if is_master_process():
        logger.info(
            f"Training: epochs={cfg.training.epochs}, "
            f"batch={cfg.training.batch_size}, "
            f"lr={lr}, amp={amp_dtype}"
        )
        logger.info("Starting training...")

    for epoch in range(start_epoch, cfg.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            context_length=context_length,
        )

        if is_master_process():
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{cfg.training.epochs}"
                f" — train_loss={avg_loss:.6f}"
                f", lr={cur_lr:.2e}"
            )

        val_metrics = {}
        is_best = False

        if val_loader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                context_length=context_length,
                target_region_length=(cfg.model.target_region_length),
                amp_dtype=amp_dtype,
                topk_fraction=topk_frac,
                world_size=world_size,
            )

            if is_master_process():
                vl = val_metrics.get("loss", float("inf"))
                ap = val_metrics.get("auc_pr_acceptor", 0.0)
                dp = val_metrics.get("auc_pr_donor", 0.0)
                logger.info(
                    f"  val_loss={vl:.6f}, auc_pr_acc={ap:.4f}, auc_pr_don={dp:.4f}"
                )

                if vl < best_val_loss:
                    best_val_loss = vl
                    best_model_epoch = epoch
                    is_best = True

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_model_epoch=best_model_epoch,
            is_best=is_best,
            is_distributed=is_ddp,
            context_length=context_length,
        )

        if use_mlflow and is_master_process():
            step_num = (epoch + 1) * optim_steps_per_epoch
            mlflow.log_metrics(
                {
                    "train/loss": avg_loss,
                    "train/learning_rate": (optimizer.param_groups[0]["lr"]),
                },
                step=step_num,
            )
            if val_metrics:
                mlflow.log_metrics(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=step_num,
                )

        slack_notifier.on_epoch_end(
            epoch=epoch,
            total_epochs=cfg.training.epochs,
            train_loss=float(avg_loss),
            val_metrics={k: float(v) for k, v in val_metrics.items()}
            if val_metrics
            else None,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if is_best:
            slack_notifier.on_best_model(
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

        early_stop = False
        if es_enabled and val_loader is not None and is_master_process():
            vl = val_metrics.get("loss", float("inf"))
            cfg_es = cfg.training.evaluation.early_stopping
            (
                early_stop,
                es_best,
                es_wait,
                _improved,
            ) = _check_early_stopping(vl, es_best, es_wait, cfg_es)
            if early_stop:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (patience={cfg_es.patience})"
                )

        if is_ddp:
            stop_t = torch.tensor([int(early_stop)], device=device)
            dist.broadcast(stop_t, src=0)
            early_stop = bool(stop_t.item())

        if early_stop:
            break

    if test_loader is not None:
        if is_master_process():
            logger.info("Running final test evaluation...")

        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            load_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                device,
                is_ddp,
            )

        test_metrics = validate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            context_length=context_length,
            target_region_length=(cfg.model.target_region_length),
            amp_dtype=amp_dtype,
            topk_fraction=topk_frac,
            world_size=world_size,
        )

        if is_master_process():
            logger.info(f"Test metrics: {test_metrics}")
            if use_mlflow:
                mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})

    if use_mlflow and is_master_process():
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            mlflow.log_artifact(best_path, "checkpoints")
        config_path = OmegaConf.select(cfg, "_config_path_")
        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path, "config")
        mlflow.end_run()

    end_status = "early_stopped" if early_stop else "completed"
    end_summary: dict[str, object] = {
        "epochs_run": (epoch + 1) if "epoch" in locals() else start_epoch,
        "total_epochs": cfg.training.epochs,
        "best_val_loss": f"{best_val_loss:.6f}",
        "best_model_epoch": best_model_epoch + 1,
        "checkpoint_dir": checkpoint_dir,
    }
    if "test_metrics" in locals():
        for key in ("loss", "auc_pr_acceptor", "auc_pr_donor"):
            if key in test_metrics:
                end_summary[f"test_{key}"] = f"{float(test_metrics[key]):.4f}"
    slack_notifier.on_train_end(end_status, end_summary)

    if is_ddp:
        cleanup_distributed()

    if is_master_process():
        logger.info("Training complete.")

    return


def _print_eval_summary(
    metrics: dict[str, float],
    checkpoint_path: str,
    test_h5_path: str,
    model_variant: str,
    context_length: int,
) -> None:
    """Print evaluation results to console.

    Only prints on the master process. Outputs a formatted
    header with model info followed by sorted metric values.

    Args:
        metrics (dict[str, float]): Evaluation metrics.
        checkpoint_path (str): Checkpoint file path.
        test_h5_path (str): Test H5 data path.
        model_variant (str): Model variant name.
        context_length (int): Model context length.
    """
    if not is_master_process():
        return

    seq_len = 5000 + context_length
    logger.info("━━━ SpliceAI Evaluation ━━━")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test data:  {test_h5_path}")
    logger.info(
        f"Model:      spliceai_{model_variant} "
        f"(context={context_length}, seq={seq_len})"
    )
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("")
    logger.info("Evaluation metrics:")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key:30s} {metrics[key]:.6f}")

    return


def _save_eval_metrics(
    metrics: dict[str, float],
    checkpoint_path: str,
    test_h5_path: str,
    context_length: int,
    model_variant: str,
    amp_dtype: str | None,
    world_size: int,
    num_test_samples: int,
) -> str:
    """Save evaluation metrics to JSON file.

    Writes a JSON file with metadata and metrics to the same
    directory as the checkpoint. File is named with a timestamp
    suffix to avoid collisions across evaluations.

    Args:
        metrics (dict[str, float]): Evaluation metrics.
        checkpoint_path (str): Checkpoint file path.
        test_h5_path (str): Test H5 data path.
        context_length (int): Model context length.
        model_variant (str): Model variant name.
        amp_dtype (str | None): AMP dtype string.
        world_size (int): Number of GPUs used.
        num_test_samples (int): Number of test samples.

    Returns:
        str: Path to the saved JSON file.
    """
    if not is_master_process():
        return ""

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"eval_metrics_{ts}.json"
    out_path = os.path.join(ckpt_dir, fname)

    payload = {
        "metadata": {
            "checkpoint_path": str(checkpoint_path),
            "test_h5_path": str(test_h5_path),
            "context_length": context_length,
            "model_variant": f"spliceai_{model_variant}",
            "timestamp": datetime.now().isoformat(),
            "amp_dtype": amp_dtype,
            "num_gpus": world_size,
            "num_test_samples": num_test_samples,
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Results saved to: {out_path}")

    return out_path


def evaluate(
    cfg: DictConfig,
    checkpoint_path: str,
    test_h5_path: str,
) -> None:
    """Orchestrate standalone evaluation of a checkpoint.

    Loads a trained SpliceAI checkpoint, creates the model,
    prepares a test DataLoader, calls the existing
    ``validate()`` function, then outputs results to console,
    JSON, and optionally MLflow.

    Args:
        cfg (DictConfig): Full configuration.
        checkpoint_path (str): Path to ``.pt`` checkpoint.
        test_h5_path (str): Path to test H5 file.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    local_rank = 0

    if is_ddp:
        local_rank, world_size = set_up_ddp(cfg)
        # % device_count keeps the index valid under Slurm per-task GPU isolation.
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        # Single-process run: GPU is allocated by the launcher (Slurm srun),
        # which sets CUDA_VISIBLE_DEVICES; use the first visible device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cl = ckpt.get(
            "context_length",
            cfg.model.spliceai_context_length,
        )
    else:
        state_dict = ckpt
        cl = cfg.model.spliceai_context_length

    variant = CONTEXT_LENGTH_TO_MODEL[cl]

    OmegaConf.update(cfg, "model.spliceai_context_length", cl)

    model, context_length = create_model(cfg)
    model.load_state_dict(state_dict)
    model = model.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    test_dataset = SplicingH5Dataset(test_h5_path)
    num_test_samples = len(test_dataset)

    batch_size = cfg.evaluation.batch_size
    num_workers = cfg.evaluation.num_workers

    test_sampler = None
    if is_ddp:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=False,
        )

    pw = num_workers > 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=spliceai_collate_fn,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )

    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=-1)

    amp_dtype = _get_amp_dtype(cfg)
    topk_frac = cfg.evaluation.topk_fraction

    metrics = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        context_length=context_length,
        target_region_length=cfg.model.target_region_length,
        amp_dtype=amp_dtype,
        topk_fraction=topk_frac,
        world_size=world_size,
    )

    amp_str = OmegaConf.select(cfg, "training.amp_dtype")

    _print_eval_summary(
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        test_h5_path=test_h5_path,
        model_variant=variant,
        context_length=context_length,
    )

    json_path = _save_eval_metrics(
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        test_h5_path=test_h5_path,
        context_length=context_length,
        model_variant=variant,
        amp_dtype=amp_str,
        world_size=world_size,
        num_test_samples=num_test_samples,
    )

    mlflow_cfg = OmegaConf.select(cfg, "mlflow")
    if mlflow_cfg and mlflow_cfg.enabled and is_master_process():
        run_name = f"eval_spliceai_{variant}_{context_length}"
        setup_mlflow(cfg, run_name, context_length, world_size)
        mlflow.log_metrics(
            {f"eval/{k}": v for k, v in metrics.items()},
            step=0,
        )
        if json_path and os.path.exists(json_path):
            mlflow.log_artifact(json_path, "eval_results")
        mlflow.end_run()

    if is_ddp:
        cleanup_distributed()

    if is_master_process():
        logger.info("Evaluation complete.")

    return


def main() -> None:
    """CLI entry point for SpliceAI H5 pipeline.

    Dispatches to ``train()`` or ``evaluate()`` based on
    the required ``--mode`` argument.
    """
    parser = argparse.ArgumentParser(
        description="SpliceAI H5 pipeline",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate"],
        help="Pipeline mode: train or evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming (train)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path override (evaluate)",
    )
    parser.add_argument(
        "--test-h5-path",
        type=str,
        default=None,
        help="Test H5 path override (evaluate)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "_config_path_", args.config)

    if args.mode == "train":
        try:
            train(cfg, resume_path=args.resume)
        except (RuntimeError, ValueError, FileNotFoundError, KeyboardInterrupt) as exc:
            run_name = OmegaConf.select(cfg, "training.run_name", default="spliceai")
            SlackNotifier.from_config(cfg, run_name=str(run_name)).on_train_error(exc)
            raise
    elif args.mode == "evaluate":
        if args.checkpoint:
            OmegaConf.update(
                cfg,
                "evaluation.checkpoint_path",
                args.checkpoint,
            )
        if args.test_h5_path:
            OmegaConf.update(
                cfg,
                "evaluation.test_h5_path",
                args.test_h5_path,
            )
        evaluate(
            cfg,
            checkpoint_path=cfg.evaluation.checkpoint_path,
            test_h5_path=cfg.evaluation.test_h5_path,
        )

    return


if __name__ == "__main__":
    main()
