"""
Training script for splicing site and splicing site strength prediction model.
"""

import argparse
import gc
import heapq
import json
import logging
import math
import os
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist  # noqa: F401
import torch.nn as nn
import torch.optim as optim
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.optimization import get_scheduler

from model.checkpoint_metadata import build_checkpoint_metadata
from process_input.conservation import DEFAULT_PHYLOP_CLIP
from train.loss import (
    MultiClassFocalLoss,
    MultiLabelFocalLoss,
    WeightedBCELoss,
    WeightedCELoss,
    WeightedMSELoss,
)
from train.online_metrics_calculator import OnlineMetricsCalculator
from train.svdb_validation import (
    SvdbValidationConfig,
    replay_early_stopping,
    score_checkpoint,
    visible_gpus,
)
from train.trainer import IssaSpliceTrainer
from utils.distributed_engine import DistributedEngine, create_engine
from utils.distributed_utils import cleanup_distributed, set_up_ddp
from utils.utils import (
    archive_code_snapshot,
    center_indices,
    compute_keep_mask_and_stats,
    is_master_process,
    log_model_timing_summary,
    log_only_master_process,
    normalize_checkpoint_keys,
    reduce_across_processes,
    safe_tensor_to_numpy,
)




class ScoreLabelPairs:
    """Device-movable container for (score, label) pairs.
    This class is used to keep gathered top-k (score, label) pairs on CPU to
    avoid GPU OOM, while still allowing optional transfer to GPU when needed.
    """

    def __init__(self, pairs: torch.Tensor) -> None:
        """Initialize container.
        Args:
            pairs (torch.Tensor): Tensor of shape [N, 2] where pairs[:, 0] are
                scores and pairs[:, 1] are binary labels (0.0/1.0).
        """
        self.pairs = pairs

        return

    def to(self, device: torch.device) -> "ScoreLabelPairs":
        """Move stored pairs tensor to a target device.
        Args:
            device (torch.device): Target device.
        Returns:
            ScoreLabelPairs: New container with the pairs tensor on `device`.
        """
        return ScoreLabelPairs(self.pairs.to(device))


class GradientClipMonitor:
    """Monitor gradient clipping statistics"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.clip_history = []
        self.step_count = 0
        self.total_clips = 0
        self.max_grad_norm_seen = 0

    def log_clipping(
        self,
        grad_norm_before: float,
        grad_norm_after: float,
        step: int,
    ) -> None:
        """Log gradient clipping event"""
        self.step_count += 1
        self.max_grad_norm_seen = max(self.max_grad_norm_seen, grad_norm_before)

        if grad_norm_before > grad_norm_after:
            self.total_clips += 1
            clip_ratio = grad_norm_after / grad_norm_before

            self.clip_history.append(
                {
                    "step": step,
                    "before": grad_norm_before,
                    "after": grad_norm_after,
                    "ratio": clip_ratio,
                }
            )

            # Log significant clipping events
            # if step % 100 == 0 or grad_norm_before > grad_norm_after * 2:
            if step % 100 == 0:
                self.logger.info(
                    f"✂️ Gradient clipped at step {step}: "
                    f"{grad_norm_before:.4f} → {grad_norm_after:.4f} "
                    f"(reduced to {clip_ratio:.1%})"
                )

    def get_summary(self) -> dict:
        """Get clipping statistics summary"""
        if self.step_count == 0:
            return {}

        clip_rate = self.total_clips / self.step_count
        avg_clip_ratio = (
            sum(h["ratio"] for h in self.clip_history) / len(self.clip_history)
            if self.clip_history
            else 1.0
        )

        return {
            "clip_rate": clip_rate,
            "total_clips": self.total_clips,
            "max_grad_norm": self.max_grad_norm_seen,
            "avg_clip_ratio": avg_clip_ratio,
        }



def _numpy_rng_to_portable(state: tuple) -> dict:
    """Convert a numpy RNG state to plain Python types.

    Args:
        state (tuple): the 5-tuple from ``np.random.get_state()``.

    Returns:
        dict: the same state with the key array as a list of ints.
    """
    name, keys, pos, has_gauss, cached = state
    return {"name": name, "keys": [int(k) for k in keys], "pos": int(pos),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}


def _portable_to_numpy_rng(state: dict) -> tuple:
    """Rebuild a numpy RNG state from :func:`_numpy_rng_to_portable`.

    Args:
        state (dict): portable form, or a raw tuple from an older checkpoint.

    Returns:
        tuple: the 5-tuple ``np.random.set_state`` expects.
    """
    if isinstance(state, tuple):
        return state
    return (state["name"], np.array(state["keys"], dtype=np.uint32),
            state["pos"], state["has_gauss"], state["cached_gaussian"])


class SplicingTrainer:
    """Trainer for splicing site prediction."""

    def __init__(
        self,
        cfg_train: DictConfig,
        cfg_model: DictConfig,
        local_rank: int,
        world_size: int,
        *,
        skip_data_loading: bool = False,
        checkpoint_path: str | None = None,
    ):
        """Initialize trainer with config.

        Args:
            cfg_train (DictConfig): Training config.
            cfg_model (DictConfig): Model config.
            local_rank (int): Local rank for DDP.
            world_size (int): World size for DDP.
            skip_data_loading (bool): When True,
                skip dataset creation.
            checkpoint_path (str | None): Path to a
                checkpoint to resume training from.
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self._checkpoint_path = checkpoint_path

        self.rank = local_rank

        # Store config as instance variable for access in all methods
        self.cfg = cfg_train
        self.cfg_model = cfg_model

        model_options = OmegaConf.select(self.cfg, "model.model_options", default=None)
        model_index = OmegaConf.select(self.cfg, "model.model_index", default=0)
        if model_options and len(model_options) > 0:
            self.model_name = model_options[model_index]
        else:
            self.model_name = self.cfg.model.model_type

        # Initialize logger first (only on master process for distributed training)
        if is_master_process():
            self.logger = logging.getLogger(self.__class__.__name__)
            log_only_master_process(
                self.logger, "info", "🔧 Initializing SplicingTrainer..."
            )
        else:
            # Create a dummy logger for non-master processes
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.WARNING)  # Only show warnings/errors

        log_only_master_process(self.logger, "info", "📄 Training config loaded")
        log_only_master_process(self.logger, "info", "📄 Model config loaded")

        # Create timestamp and run name for this training run
        self.training_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine run name with timestamp suffix
        config_run_name = self.cfg.training.run_name
        if config_run_name:
            self.run_name = f"{config_run_name}_{self.training_start_time}"
        else:
            self.run_name = f"run_{self.training_start_time}"
        log_only_master_process(
            self.logger,
            "info",
            f"🏷️  Run name: {self.run_name}",
        )

        # Create checkpoint directory with run name (only on master process)
        if is_master_process():
            base_checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
            self.checkpoint_dir = base_checkpoint_dir / self.run_name
            if self.checkpoint_dir.exists():
                raise FileExistsError(
                    f"Checkpoint directory already exists: {self.checkpoint_dir}"
                )
            self.checkpoint_dir.mkdir(parents=True)
            self.logger.info(f"Checkpoint directory created: {self.checkpoint_dir}")
            # Save resolved training config for reproducibility
            train_cfg_path = self.checkpoint_dir / "train_config.yaml"
            OmegaConf.save(self.cfg, str(train_cfg_path))
            self.logger.info(f"Training config saved: {train_cfg_path}")

            # Archive source snapshot into checkpoint directory
            try:
                project_root = os.environ.get("HYDRA_ORIG_CWD", os.getcwd())
                archive_code_snapshot(
                    output_dir=str(self.checkpoint_dir),
                    directories=["config", "model", "train", "utils"],
                    root_dir=project_root,
                    archive_name_prefix="source_snapshot",
                    logger=self.logger,
                )
            except Exception as e:
                self.logger.warning("⚠️  Failed to archive code snapshot: %s", str(e))
        else:
            # Non-master processes still need checkpoint_dir for potential access
            base_checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
            self.checkpoint_dir = base_checkpoint_dir / self.run_name

        # Setup logging (only on master process)
        if is_master_process():
            log_only_master_process(self.logger, "info", "📝 Setting up logging...")
            self._setup_logging()

            log_only_master_process(
                self.logger, "info", f"Training started at: {self.training_start_time}"
            )
            log_only_master_process(
                self.logger,
                "info",
                f"Checkpoints will be saved to: {self.checkpoint_dir}",
            )

        # Initialize trainer
        log_only_master_process(self.logger, "info", "🎯 Initializing base trainer...")
        self.trainer = IssaSpliceTrainer(self.cfg)

        # Create data loaders with distributed support
        if skip_data_loading:
            log_only_master_process(
                self.logger,
                "info",
                "📊 Skipping data loader creation (external test loader expected)",
            )
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
        else:
            log_only_master_process(
                self.logger,
                "info",
                "📊 Creating data loaders...",
            )
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
            ) = self._create_distributed_dataloaders()

        # Set device for distributed training
        log_only_master_process(self.logger, "info", "🔍 Determining compute device...")

        # GPU selection is owned by the launcher (Slurm srun / torchrun), which
        # sets CUDA_VISIBLE_DEVICES. Each DDP rank binds to its local_rank
        # within the visible set; a single-process run takes the first visible
        # device. We never read a GPU index from config.
        if self.is_distributed:
            # % device_count keeps the index valid under Slurm per-task GPU
            # isolation, where a worker may see only its single allocated GPU.
            local_device = self.local_rank % torch.cuda.device_count()
            self.device = torch.device(f"cuda:{local_device}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        log_only_master_process(self.logger, "info", f"🖥️  Using device: {self.device}")
        if self.is_distributed:
            log_only_master_process(
                self.logger,
                "info",
                f"🌐 Distributed training: local_rank "
                f"{self.local_rank}/{self.world_size}",
            )
            log_only_master_process(
                self.logger, "info", f"🎯 GPU device: {self.device}"
            )

            # Log global rank if available
            global_rank = int(os.environ.get("RANK", "0"))
            log_only_master_process(
                self.logger, "info", f"🎯 Global rank: {global_rank}"
            )
        else:
            device_type = "CPU" if str(self.device) == "cpu" else "Single GPU"
            log_only_master_process(
                self.logger, "info", f"🖥️  {device_type} training mode"
            )
            log_only_master_process(self.logger, "info", f"🎯 Device: {self.device}")

        log_only_master_process(
            self.logger,
            "info",
            "🤖 Loading and initializing model...",
        )
        self.model = self.trainer.init_model(verbose=is_master_process()).to(
            self.device
        )

        # Note: MLM pretrained backbone is now loaded during model initialization
        # in IssaSplice._initialize_hyenadna_model() when mlm_pretrained_path is set

        # Optionally load initial weights from checkpoint if provided
        # (This will override backbone weights if specified)
        init_ckpt_path = self.cfg.model.initial_weights_path
        if init_ckpt_path:
            if not os.path.exists(init_ckpt_path):
                raise FileNotFoundError(
                    f"Initial weights checkpoint not found: {init_ckpt_path}"
                )
            log_only_master_process(
                self.logger,
                "info",
                f"🔁 Loading initial model weights from: {init_ckpt_path}",
            )
            self.model.load_model_from_checkpoint(init_ckpt_path)
            log_only_master_process(
                self.logger, "info", "✅ Initial weights loaded successfully"
            )

        # Store model type as instance variable for consistent use across methods
        self.model_type = self.cfg.model.model_type.lower()

        # Mixed-precision (AMP) configuration.
        # ``training.mixed_precision`` ∈ {"fp32", "bf16", "fp16"}. When
        # set to a half-precision mode, the forward pass is wrapped in
        # ``torch.autocast(device_type="cuda", dtype=...)``. dnaHNet paper
        # reports BF16 Mixed for its best runs (Table 3). FP16 would need
        # a GradScaler (not added here); BF16 does not. Default "fp32"
        # preserves backward compatibility for existing configs.
        mixed_precision_raw = OmegaConf.select(
            self.cfg, "training.mixed_precision", default="fp32"
        )
        mixed_precision = str(mixed_precision_raw).lower()
        _MP_MAP: dict[str, torch.dtype | None] = {
            "fp32": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        if mixed_precision not in _MP_MAP:
            raise ValueError(
                f"Unknown training.mixed_precision: {mixed_precision_raw!r}. "
                f"Supported: {list(_MP_MAP)}."
            )
        self.amp_dtype: torch.dtype | None = _MP_MAP[mixed_precision]
        self.amp_enabled: bool = (
            self.amp_dtype is not None and self.device.type == "cuda"
        )
        if self.amp_enabled:
            log_only_master_process(
                self.logger,
                "info",
                f"🎯 Mixed precision: {mixed_precision} "
                f"(autocast dtype={self.amp_dtype})",
            )

        # Display model information (only on master process)
        if is_master_process():
            log_only_master_process(
                self.logger, "info", f"✅ Model loaded: {self.model_type}"
            )

            log_only_master_process(
                self.logger, "info", "🧬 Hyenadna backbone detected"
            )
            log_only_master_process(
                self.logger, "info", f"   - HF model: {self.model_name}"
            )

            log_only_master_process(
                self.logger, "info", f"🎭 Model dtype: {self.model.weight_dtype}"
            )
            log_only_master_process(self.logger, "info", f"📍 Device: {self.device}")
            log_only_master_process(
                self.logger, "info", f"🔢 Backbone dim: {self.model.backbone_dim}"
            )

            # Print detailed model information
            self.model.print_model_info()

        # Wrap distributed execution after the model and optimizer are built.
        engine_local_rank = self.local_rank if self.is_distributed else None
        self.engine: DistributedEngine = create_engine(
            self.cfg, local_rank=engine_local_rank
        )
        log_only_master_process(
            self.logger,
            "info",
            f"🌐 Distributed engine selected: {type(self.engine).__name__} "
            f"(distributed={self.is_distributed}).",
        )

        # Toggle for per-batch token/char distribution logging (debug only)
        self.enable_distribution_logging: bool = OmegaConf.select(
            self.cfg, "logging.enable_distribution_logging", default=False
        )
        if self.enable_distribution_logging:
            log_only_master_process(
                self.logger, "info", "🔎 Distribution logging is ENABLED (debug mode)"
            )
        else:
            log_only_master_process(
                self.logger, "info", "🔕 Distribution logging is DISABLED"
            )

        test_eval_cfg = OmegaConf.select(
            self.cfg, "training.evaluation.test_evaluation", default=None
        )
        self._test_eval_enabled = bool(
            test_eval_cfg
            and bool(getattr(test_eval_cfg, "enabled", False))
            and self.test_loader is not None
            and len(self.test_loader) > 0
        )
        self._test_eval_frequency = (
            str(getattr(test_eval_cfg, "frequency", "epoch")).lower()
            if test_eval_cfg
            else "epoch"
        )
        self._test_eval_interval = max(1, int(getattr(test_eval_cfg, "interval", 1)))
        self._test_eval_sample_proportion = float(
            getattr(test_eval_cfg, "sample_proportion", 1.0)
        )
        self._test_eval_sample_proportion = min(
            max(self._test_eval_sample_proportion, 0.0),
            1.0,
        )
        self._test_eval_evaluate_at_end = bool(
            getattr(test_eval_cfg, "evaluate_at_end", True)
        )
        self._last_test_eval_step = 0
        self._last_test_eval_epoch = 0

        # Cache sampled test loader for consistent evaluation across calls
        self._cached_sampled_test_loader: DataLoader | None = None
        if self._test_eval_enabled and self._test_eval_sample_proportion < 1.0:
            fixed_seed = int(self.cfg.data.random_seed)
            self._cached_sampled_test_loader = self._get_sampled_test_loader(
                self.test_loader, self._test_eval_sample_proportion, seed=fixed_seed
            )
            log_only_master_process(
                self.logger,
                "info",
                (
                    f"🧪 Created cached sampled test loader with fixed seed "
                    f"({fixed_seed}) for consistent evaluation"
                ),
            )

        if self._test_eval_enabled:
            log_only_master_process(
                self.logger,
                "info",
                (
                    "🧪 Periodic test evaluation enabled "
                    f"(frequency={self._test_eval_frequency}, "
                    f"interval={self._test_eval_interval}, "
                    f"sample_proportion={self._test_eval_sample_proportion:.1%}, "
                    f"evaluate_at_end={self._test_eval_evaluate_at_end})"
                ),
            )
        elif test_eval_cfg and bool(getattr(test_eval_cfg, "enabled", False)):
            log_only_master_process(
                self.logger,
                "info",
                "🧪 Periodic test evaluation requested but test loader is empty; "
                "skipping.",
            )

        # Loss function setup
        log_only_master_process(self.logger, "info", "⚖️  Setting up loss function...")

        # Get loss function configuration
        loss_config = self.cfg.training.loss_function
        self.loss_type = loss_config.type
        class_weights = torch.tensor(
            self.cfg.training.class_weights,
            dtype=torch.float32,
        ).to(self.device)

        if self.loss_type == "multi_label_focal_loss":
            multi_label_focal_config = loss_config.multi_label_focal_loss
            gamma_cfg = multi_label_focal_config.gamma
            # Accept scalar or per-class list for gamma
            if isinstance(gamma_cfg, (list, tuple)) or (
                "ListConfig" in str(type(gamma_cfg))
            ):
                try:
                    gamma_value = [float(g) for g in list(gamma_cfg)]
                except Exception:
                    gamma_value = [float(g) for g in gamma_cfg]
            else:
                try:
                    gamma_value = float(gamma_cfg)
                except Exception:
                    gamma_value = 2.0

            use_negative_sampling = getattr(
                self.cfg.training.loss_function, "use_negative_sampling", False
            )

            self.criterion = MultiLabelFocalLoss(
                alpha=class_weights,
                gamma=gamma_value,
                use_negative_sampling=use_negative_sampling,
                negative_sample_rate=getattr(
                    self.cfg.training.loss_function, "negative_sample_rate", 0.05
                ),
            )
            log_only_master_process(
                self.logger,
                "info",
                f"📊 Loss function: MultiLabelFocalLoss with negative sampling: "
                f"{use_negative_sampling}",
            )
            log_only_master_process(
                self.logger, "info", f"📊 Alpha weights: {class_weights.tolist()}"
            )
            if isinstance(gamma_value, list):
                log_only_master_process(
                    self.logger, "info", f"📊 Focal gamma per-class: {gamma_value}"
                )
            else:
                log_only_master_process(
                    self.logger, "info", f"📊 Focal gamma: {gamma_value}"
                )

        elif self.loss_type == "weighted_bce":
            self.criterion = WeightedBCELoss(pos_weights=class_weights)
            log_only_master_process(
                self.logger, "info", "📊 Loss function: WeightedBCELoss"
            )
            if class_weights is not None:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"📊 Positive weights: {class_weights.tolist()}",
                )
            else:
                log_only_master_process(
                    self.logger,
                    "info",
                    "📊 Auto-calculating positive weights from class imbalance",
                )
        elif self.loss_type == "weighted_ce":
            # Get optional label_smoothing from config
            label_smoothing = float(getattr(loss_config, "label_smoothing", 0.0))
            self.criterion = WeightedCELoss(
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
            log_only_master_process(
                self.logger, "info", "📊 Loss function: WeightedCELoss"
            )
            log_only_master_process(
                self.logger,
                "info",
                f"📊 Class weights: {class_weights.tolist()}",
            )
            if label_smoothing > 0:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"📊 Label smoothing: {label_smoothing}",
                )

        elif self.loss_type == "multi_class_focal_loss":
            # Safe access with default gamma
            try:
                gamma = float(loss_config.multi_class_focal_loss.gamma)
            except (AttributeError, ValueError, TypeError):
                gamma = 2.0
            self.criterion = MultiClassFocalLoss(alpha=class_weights, gamma=gamma)
            log_only_master_process(
                self.logger, "info", "📊 Loss function: MultiClassFocalLoss"
            )
            log_only_master_process(
                self.logger, "info", f"📊 Alpha weights: {class_weights.tolist()}"
            )
            log_only_master_process(self.logger, "info", f"📊 Focal gamma: {gamma}")

        else:
            raise ValueError(f"Unknown loss function type: {self.loss_type}")

        log_only_master_process(
            self.logger, "info", "🔄 Using float32 for all loss calculations"
        )

        # Positive sample scaling / positive-only training removed

        # Regression (PSI) was removed in the v2 config revamp; the
        # ``training.regression`` block is therefore optional in the YAML.
        # Fall back to a benign default when absent so the criterion can be
        # instantiated for backward compatibility with older configs.
        regression_pos_weight = float(
            OmegaConf.select(self.cfg, "training.regression.pos_weight", default=1.0)
        )
        self.regression_criterion = WeightedMSELoss(pos_weight=regression_pos_weight)
        # Toggle classification head usage via config
        self.is_classification_enabled: bool = bool(
            OmegaConf.select(self.cfg, "training.classification.enabled", default=True)
        )
        # Regression (PSI) was removed in the v2 config revamp; the classification
        # head is now the only supported task.
        self.is_regression_enabled: bool = False
        if not self.is_classification_enabled:
            raise ValueError(
                "training.classification.enabled must be True; the classification "
                "head is the only supported task."
            )
        log_only_master_process(
            self.logger,
            "info",
            f"🎯 Classification head enabled: {self.is_classification_enabled}",
        )
        self.loss_weight_classification: float = float(
            OmegaConf.select(
                self.cfg, "training.loss_weights.classification", default=1.0
            )
        )
        self.loss_weight_regression: float = 0.0

        # Auxiliary region / exon prediction heads (per-base CE with -1 ignore).
        self.is_aux_region_enabled: bool = bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_region", default=False)
        )
        self.is_aux_exon_enabled: bool = bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_exon", default=False)
        )
        # Quantitative splice-site usage regression head (Pangolin SSU, 4
        # tissues). Trained with masked BCE: usage in [0, 1] is treated as a
        # Bernoulli target, positions with no measurement (label < 0) are
        # excluded, mirroring Pangolin's loss.
        self.is_aux_usage_enabled: bool = bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_usage", default=False)
        )
        # Per-tissue spliced/not classification head (Pangolin-faithful, 2-class,
        # cryptic-aware): the head scored for SAV in the multi-species SSU work.
        self.is_aux_spliced_enabled: bool = bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_spliced", default=False)
        )
        # Per-base phyloP conservation regression head. Trained with masked
        # MSE on clip(phyloP, -C, C) / C; positions without a phyloP value
        # (assembly gaps, window padding) arrive as NaN and are excluded.
        self.is_aux_phylop_enabled: bool = bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_phylop", default=False)
        )
        self.loss_weight_aux_region: float = float(
            OmegaConf.select(self.cfg, "training.loss_weights.aux_region", default=0.1)
        )
        self.loss_weight_aux_exon: float = float(
            OmegaConf.select(self.cfg, "training.loss_weights.aux_exon", default=0.1)
        )
        self.loss_weight_aux_usage: float = float(
            OmegaConf.select(self.cfg, "training.loss_weights.aux_usage", default=0.1)
        )
        self.loss_weight_aux_spliced: float = float(
            OmegaConf.select(self.cfg, "training.loss_weights.aux_spliced", default=0.0)
        )
        self.loss_weight_aux_phylop: float = float(
            OmegaConf.select(self.cfg, "training.loss_weights.aux_phylop", default=0.1)
        )
        # Binary exon/intron auxiliary target derived on-the-fly from the
        # region labels (Y_REGION: 0=intron -> 0, 1..6=exon subtypes -> 1,
        # -1=pad kept as ignore) instead of the default Y_EXON exon-position
        # labels. Set exon_num_classes=2 when enabling this. Exon bases are
        # rare (~4% of positions); either an inverse-frequency class weight
        # (plain CE) or focal loss keeps the head from collapsing to the
        # all-intron prediction so the aux gradient shapes the shared trunk.
        self.aux_exon_binary_from_region: bool = bool(
            OmegaConf.select(
                self.cfg, "training.aux_exon_binary_from_region", default=False
            )
        )
        # When True the exon head uses focal loss (mirroring the main 3-class
        # head) instead of plain cross-entropy. The (1-p)^gamma modulation
        # already down-weights easy intron positions, so `aux_exon_class_weights`
        # (the focal alpha) should be milder than the CE inverse-frequency value
        # -- e.g. [1.0, 4.0] to match the main head's rare-class alpha of 4.
        self.aux_exon_use_focal: bool = bool(
            OmegaConf.select(self.cfg, "training.aux_exon_use_focal", default=False)
        )
        aux_exon_focal_gamma = float(
            OmegaConf.select(self.cfg, "training.aux_exon_focal_gamma", default=2.0)
        )
        aux_exon_class_weights = OmegaConf.select(
            self.cfg, "training.aux_exon_class_weights", default=None
        )
        if self.is_aux_region_enabled or self.is_aux_exon_enabled:
            self.aux_region_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            weight = None
            if aux_exon_class_weights is not None:
                weight = torch.tensor(
                    list(aux_exon_class_weights),
                    dtype=torch.float32,
                    device=self.device,
                )
            if self.aux_exon_use_focal:
                # ignore_index=-1 is handled by explicit masking in
                # _compute_single_aux_loss (MultiClassFocalLoss has no
                # ignore_index support).
                self.aux_exon_criterion = MultiClassFocalLoss(
                    alpha=weight, gamma=aux_exon_focal_gamma
                )
            elif weight is not None:
                self.aux_exon_criterion = nn.CrossEntropyLoss(
                    weight=weight, ignore_index=-1
                )
            else:
                self.aux_exon_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.aux_region_criterion = None
            self.aux_exon_criterion = None
        self.aux_usage_criterion = (
            nn.BCEWithLogitsLoss(reduction="none")
            if self.is_aux_usage_enabled
            else None
        )
        self.aux_spliced_criterion = (
            nn.CrossEntropyLoss(ignore_index=-1)
            if self.is_aux_spliced_enabled
            else None
        )

        # Dynamic chunker (FR-024 / DD-018): boundary regularization loss.
        # Active iff `model.tokenizer_type == "dynamic_chunker"`. Reads
        # `target_chunk_rate` and `boundary_loss_weight` from the
        # `model.dynamic_chunker` config block written by the IssaSplice
        # init code path. Baseline tokenizers leave `is_dynamic_chunker_enabled`
        # False and `boundary_loss_weight = 0`, so total loss is unchanged.
        self.is_dynamic_chunker_enabled: bool = (
            OmegaConf.select(
                self.cfg, "model.tokenizer_type", default="single_nucleotide"
            )
            == "dynamic_chunker"
        )
        # Per-stage target chunk rates for the boundary regularizer. Multi-
        # stage hierarchies (F5) define one entry per outer-to-inner stage
        # via ``model.dynamic_chunker.stages[i].target_chunk_rate``;
        # single-stage configs collapse to a length-1 list derived from
        # the flat ``model.dynamic_chunker.target_chunk_rate``. The legacy
        # scalar ``dynamic_chunker_target_rate`` is kept as the
        # outermost stage's rate for log lines and unchanged downstream
        # validation paths.
        self.dynamic_chunker_target_rates: list[float] = (
            self._resolve_stage_target_rates()
        )
        self.dynamic_chunker_target_rate: float = self.dynamic_chunker_target_rates[0]
        self.loss_weight_boundary: float = float(
            OmegaConf.select(
                self.cfg,
                "model.dynamic_chunker.boundary_loss_weight",
                default=0.0,
            )
        )
        # Boundary loss formulation. ``l1`` (default, DD-018) is the
        # legacy ``|F - target|`` regularizer kept for checkpoint
        # reproducibility; ``hnet_ratio`` (DD-020) is the H-Net paper
        # eq. 10 ratio loss with both F (hard) and G (soft) signals.
        self.boundary_loss_type: str = str(
            OmegaConf.select(
                self.cfg,
                "model.dynamic_chunker.boundary_loss_type",
                default="l1",
            )
        ).lower()
        if self.boundary_loss_type not in ("l1", "hnet_ratio"):
            raise ValueError(
                "model.dynamic_chunker.boundary_loss_type must be one of "
                f"{{'l1', 'hnet_ratio'}}; got {self.boundary_loss_type!r}."
            )
        if (
            self.is_dynamic_chunker_enabled
            and self.boundary_loss_type == "hnet_ratio"
            and any(rate >= 1.0 for rate in self.dynamic_chunker_target_rates)
        ):
            raise ValueError(
                "boundary_loss_type='hnet_ratio' requires every stage's "
                "target_chunk_rate < 1.0 (so that N = 1 / target_chunk_rate > 1); "
                f"got per-stage rates {self.dynamic_chunker_target_rates}."
            )
        chunker_cfg = OmegaConf.select(self.cfg, "model.dynamic_chunker", default=None)
        if self.is_dynamic_chunker_enabled and chunker_cfg is not None:
            flat_style = chunker_cfg.get("chunker_style", None) or "legacy"
            stage_styles = [
                (stage.get("chunker_style", None) or flat_style)
                for stage in (chunker_cfg.get("stages", None) or [chunker_cfg])
            ]
            # The hnet style reports F without gradient (as H-Net), so an F-only
            # loss would leave the rate unregularized.
            if "hnet" in stage_styles and self.boundary_loss_type != "hnet_ratio":
                raise ValueError(
                    "chunker_style='hnet' needs boundary_loss_type='hnet_ratio'; "
                    f"got {self.boundary_loss_type!r}."
                )

        log_only_master_process(
            self.logger,
            "info",
            f"🧮 Loss weights — classification: "
            f"{self.loss_weight_classification:.3f} | "
            f"regression: {self.loss_weight_regression:.3f} | "
            f"aux_region: {self.loss_weight_aux_region:.3f} "
            f"(enabled={self.is_aux_region_enabled}) | "
            f"aux_exon: {self.loss_weight_aux_exon:.3f} "
            f"(enabled={self.is_aux_exon_enabled}) | "
            f"aux_phylop: {self.loss_weight_aux_phylop:.3f} "
            f"(enabled={self.is_aux_phylop_enabled}) | "
            f"boundary: {self.loss_weight_boundary:.3f} "
            f"(enabled={self.is_dynamic_chunker_enabled}, "
            f"target={self.dynamic_chunker_target_rate:.3f}, "
            f"type={self.boundary_loss_type})",
        )

        # Cache for tuned thresholds computed on validation to reuse on test
        self.tuned_thresholds: list[float] | None = None

        # Training-only setup: optimizer, scheduler,
        # gradient clipping, checkpointing, early stopping.
        # Skipped when skip_data_loading is True
        # (evaluate-only with external test loader).
        if not skip_data_loading:
            self._best_val_loss: float = float("inf")
            self._best_model_epoch: int = 0
            self.best_model_path = self.checkpoint_dir / "best_model.pt"

            early_cfg = OmegaConf.select(
                self.cfg,
                "training.evaluation.early_stopping",
                default=None,
            )
            self.early_stopping_enabled = bool(
                early_cfg and bool(getattr(early_cfg, "enabled", False))
            )

            if self.early_stopping_enabled:
                self.early_stopping_patience = int(early_cfg.patience)
                self.early_stopping_min_delta = float(early_cfg.min_delta)
                self.early_stopping_mode = str(early_cfg.mode).lower()
                self.early_stopping_monitor = str(early_cfg.monitor)
                self.early_stopping_restore_best = bool(early_cfg.restore_best_weights)
                self._early_stopping_best_value = (
                    float("inf") if self.early_stopping_mode == "min" else float("-inf")
                )
                self._early_stopping_wait = 0
                self._early_stop_triggered = False
                log_only_master_process(
                    self.logger,
                    "info",
                    (
                        "⏹️  Early stopping enabled "
                        f"(monitor={self.early_stopping_monitor}, "
                        f"patience={self.early_stopping_patience}, "
                        f"mode={self.early_stopping_mode}, "
                        f"min_delta={self.early_stopping_min_delta})"
                    ),
                )
            else:
                self.early_stopping_patience = 0
                self.early_stopping_min_delta = 0.0
                self.early_stopping_mode = "min"
                self.early_stopping_monitor = "loss"
                self.early_stopping_restore_best = False
                self._early_stopping_best_value = float("inf")
                self._early_stopping_wait = 0
                self._early_stop_triggered = False

            # SpliceVarDB metric (train/svdb_validation.py): after each epoch
            # checkpoint, the default delta max readout scored on every GPU of
            # the job; early stopping can monitor ``svdb_auroc``.
            svdb_cfg = OmegaConf.select(
                self.cfg, "training.evaluation.svdb_validation", default=None
            )
            self.svdb_validation: SvdbValidationConfig | None = None
            # While the master scores, the other ranks wait on this CPU (gloo)
            # group. Waiting in an NCCL collective instead keeps a kernel
            # spinning on each GPU the scorers share, which slowed them 2-4x
            # and coincided with a CUDA launch failure in that wait.
            self._svdb_wait_group: dist.ProcessGroup | None = None
            if svdb_cfg is not None and bool(svdb_cfg.get("enabled", False)):
                gpus = visible_gpus()
                cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
                self.svdb_validation = SvdbValidationConfig(
                    variant_h5=str(svdb_cfg.variant_h5),
                    work_dir=str(self.checkpoint_dir / "svdb_validation"),
                    gpus=gpus,
                    threads_per_shard=max(1, cpus // max(len(gpus), 1)),
                    batch_size=int(svdb_cfg.get("batch_size", 16)),
                )
                if self.is_distributed:
                    timeout = OmegaConf.select(
                        self.cfg, "training.distributed.timeout_minutes", default=30
                    )
                    self._svdb_wait_group = dist.new_group(
                        backend="gloo", timeout=timedelta(minutes=int(timeout))
                    )
                log_only_master_process(
                    self.logger,
                    "info",
                    f"🧪 SpliceVarDB validation after every epoch: "
                    f"{self.svdb_validation.variant_h5} on GPUs {gpus}",
                )

            # Optimizer
            log_only_master_process(self.logger, "info", "🚀 Setting up optimizer...")
            head_name_tokens = (
                "classification_head",
                "hybrid_head",
                "aux_region_head",
                "aux_exon_head",
                "aux_phylop_head",
            )
            # H-Net learning-rate modulation (paper hint, F4 + F5): outer
            # stages process longer sequences than the chunked main
            # backbone, so giving them the same per-step LR yields an
            # effectively higher gradient step. The paper sidesteps this
            # by *down*-scaling outer-stage encoder/decoder LR; here we
            # make it configurable per-stage via
            # ``model.dynamic_chunker.stages[i].{encoder,dechunker}_lr_scale``
            # (multi-stage F5) or via the flat
            # ``model.dynamic_chunker.{encoder,dechunker}_lr_scale``
            # (single-stage F4 backward-compat). Defaults to 1.0
            # (no scaling).
            base_backbone_lr = float(self.cfg.training.learning_rate_backbone)
            head_lr = float(self.cfg.training.learning_rate_classification_head)
            encoder_lr_scales = self._resolve_stage_lr_scales("encoder_lr_scale")
            dechunker_lr_scales = self._resolve_stage_lr_scales("dechunker_lr_scale")

            backbone_params: list = []
            head_params: list = []
            stage_encoder_params: list[list] = [[] for _ in encoder_lr_scales]
            stage_dechunker_params: list[list] = [[] for _ in dechunker_lr_scales]
            for name, param in self.model.named_parameters():
                stage_idx = self._match_stage_param(name)
                if stage_idx is not None:
                    role, idx = stage_idx
                    if role == "encoder":
                        stage_encoder_params[idx].append(param)
                    else:
                        stage_dechunker_params[idx].append(param)
                elif any(token in name for token in head_name_tokens):
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            param_groups: list[dict] = [
                {"params": backbone_params, "lr": base_backbone_lr},
                {"params": head_params, "lr": head_lr},
            ]
            stage_lr_log_parts: list[str] = []
            for idx, params in enumerate(stage_encoder_params):
                if params:
                    scale = encoder_lr_scales[idx]
                    stage_lr = base_backbone_lr * scale
                    param_groups.append({"params": params, "lr": stage_lr})
                    stage_lr_log_parts.append(
                        f"stage{idx}.encoder x{scale:.3f}->{stage_lr:.2e}"
                    )
            for idx, params in enumerate(stage_dechunker_params):
                if params:
                    scale = dechunker_lr_scales[idx]
                    stage_lr = base_backbone_lr * scale
                    param_groups.append({"params": params, "lr": stage_lr})
                    stage_lr_log_parts.append(
                        f"stage{idx}.dechunker x{scale:.3f}->{stage_lr:.2e}"
                    )

            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.cfg.training.weight_decay,
            )
            log_only_master_process(
                self.logger,
                "info",
                "⚡ Optimizer: AdamW with learning rate "
                + f"{base_backbone_lr} for backbone and "
                + f"{head_lr} for prediction heads "
                + "(classification / hybrid / aux region / aux exon)"
                + (" | " + ", ".join(stage_lr_log_parts) if stage_lr_log_parts else ""),
            )

            # Gradient accumulation configuration. Prefer
            # ``training.effective_batch_size`` when set so users do not need
            # to juggle gradient_accumulation_steps when batch_size or
            # num_gpus changes.
            from utils.utils import resolve_batch_plan

            batch_plan = resolve_batch_plan(self.cfg)
            self.grad_accumulation_steps: int = batch_plan.grad_accum_steps
            log_only_master_process(
                self.logger,
                "info",
                "🧮 Batch plan ("
                f"source={batch_plan.source}): effective="
                f"{batch_plan.effective_batch_size} = micro="
                f"{batch_plan.micro_batch_size} × gpus={batch_plan.num_gpus} × "
                f"grad_accum={batch_plan.grad_accum_steps}",
            )

            # Learning rate scheduler (optimizer steps count)
            log_only_master_process(
                self.logger,
                "info",
                "📈 Setting up learning rate scheduler...",
            )
            self.steps_per_epoch_optim: int = int(
                math.ceil(len(self.train_loader) / max(1, self.grad_accumulation_steps))
            )
            self.num_training_steps = int(
                self.steps_per_epoch_optim * self.cfg.training.epochs
            )
            self.num_warmup_steps = int(
                self.num_training_steps * self.cfg.training.warmup_proportion
            )

            self.scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_training_steps=self.num_training_steps,
                num_warmup_steps=self.num_warmup_steps,
            )
            log_only_master_process(
                self.logger,
                "info",
                f"📊 Scheduler: cosine with "
                f"{self.num_training_steps:,} total steps, "
                f"{self.num_warmup_steps:,} warmup steps",
            )

            # Wrap the model with PyTorch distributed data parallel.
            self.model, self.optimizer, self.scheduler = self.engine.wrap(
                self.model, self.optimizer, self.scheduler
            )
            log_only_master_process(
                self.logger,
                "info",
                f"✅ Engine wrap complete (engine={type(self.engine).__name__}).",
            )

            # Gradient clipping
            self.gradient_clipping_enabled: bool = OmegaConf.select(
                self.cfg,
                "training.gradient_clipping.enabled",
                default=True,
            )
            log_only_master_process(
                self.logger,
                "info",
                "✂️ Gradient clipping is "
                f"{'ENABLED' if self.gradient_clipping_enabled else 'DISABLED'}",
            )
            self.gradient_clip_monitor = (
                GradientClipMonitor(self.logger)
                if self.gradient_clipping_enabled
                else None
            )

            self.gradient_clip_min: float = float(
                OmegaConf.select(
                    self.cfg,
                    "training.gradient_clipping.min_clip",
                    default=1.0,
                )
            )
            self.gradient_clip_max: float = float(
                OmegaConf.select(
                    self.cfg,
                    "training.gradient_clipping.max_clip",
                    default=10.0,
                )
            )
            if self.gradient_clip_max < self.gradient_clip_min:
                self.gradient_clip_min, self.gradient_clip_max = (
                    self.gradient_clip_max,
                    self.gradient_clip_min,
                )
            log_only_master_process(
                self.logger,
                "info",
                f"✂️ Gradient clipping range: "
                f"min={self.gradient_clip_min:.2f}, "
                f"max={self.gradient_clip_max:.2f}",
            )

            # Checkpoint tracking
            self.last_checkpoint_step = 0
            self._accumulated_microbatches: int = 0

            self.checkpoint_save_strategy = OmegaConf.select(
                self.cfg,
                "training.checkpoint_save_strategy",
                default="step",
            )
            self.checkpoint_save_steps = OmegaConf.select(
                self.cfg,
                "training.checkpoint_save_steps",
                default=100,
            )
            self.checkpoint_save_epochs = OmegaConf.select(
                self.cfg,
                "training.checkpoint_save_epochs",
                default=1,
            )

            log_only_master_process(
                self.logger,
                "info",
                f"📋 Checkpoint strategy: {self.checkpoint_save_strategy}",
            )
            if self.checkpoint_save_strategy == "step":
                log_only_master_process(
                    self.logger,
                    "info",
                    "💾 Will save checkpoints every "
                    f"{self.checkpoint_save_steps} steps",
                )
            elif self.checkpoint_save_strategy == "epoch":
                log_only_master_process(
                    self.logger,
                    "info",
                    "💾 Will save checkpoints every "
                    f"{int(self.checkpoint_save_epochs)} epochs",
                )
            else:
                log_only_master_process(
                    self.logger,
                    "info",
                    "📅 Unknown checkpoint strategy; no step-based autosave",
                )

        # Initialize experiment tracking (MLflow)
        if is_master_process():

            log_only_master_process(
                self.logger, "info", "✅ SplicingTrainer initialization complete!"
            )
            if not skip_data_loading:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"🎉 Ready to train for {self.cfg.training.epochs} epochs",
                )
            else:
                log_only_master_process(
                    self.logger,
                    "info",
                    "🧪 Ready for evaluate-only mode",
                )
            log_only_master_process(self.logger, "info", "=" * 60)


        # Soft labels were removed in the v2 config revamp; the trainer now
        # always operates on hard one-hot labels.
        self.use_soft_labels = False

        # Hard label configuration for classification (regression is removed)
        self.uses_hard_labels_classification: bool = bool(
            OmegaConf.select(
                self.cfg, "training.classification.use_hard_labels", default=False
            )
        )
        self.uses_hard_labels_regression: bool = False
        log_only_master_process(
            self.logger,
            "info",
            f"🎯 Classification hard labels: {self.uses_hard_labels_classification}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"🧪 Regression hard labels: {self.uses_hard_labels_regression}",
        )

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{self.training_start_time}.log"

        # Configure logging to output to file and console
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
            force=True,  # Force reconfiguration even if logging already exists
        )

        self.logger.info(f"Logging configured to save to {log_file}")

    def _create_distributed_dataloaders(self):
        """Create data loaders with distributed sampler support for all splits."""
        # Get the original datasets
        datasets = self.trainer.create_datasets()
        train_dataset, val_dataset, test_dataset = datasets

        # Create distributed samplers for all splits when distributed training
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        # Get batch size
        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        # Create data loaders for all processes
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        log_only_master_process(self.logger, "info", "📊 Created data loaders:")
        log_only_master_process(
            self.logger, "info", f"   - Train samples: {len(train_dataset)}"
        )
        log_only_master_process(
            self.logger, "info", f"   - Val samples: {len(val_dataset)}"
        )
        log_only_master_process(
            self.logger, "info", f"   - Test samples: {len(test_dataset)}"
        )

        if self.is_distributed:
            log_only_master_process(
                self.logger,
                "info",
                f"   - Samples per GPU (approx): "
                f"Train={len(train_dataset) // self.world_size}, "
                f"Val={len(val_dataset) // self.world_size}, "
                f"Test={len(test_dataset) // self.world_size}",
            )

        return train_loader, val_loader, test_loader

    def _calculate_compile_speedup(self) -> None:
        """
        Calculate and log the speedup achieved by torch.compile.
        """
        if not hasattr(self, "compile_performance") or not self.compile_performance.get(
            "enabled", False
        ):
            return

        perf = self.compile_performance

        # Need at least 10 samples each for reliable comparison
        if len(perf["pre_compile_times"]) < 10 or len(perf["post_compile_times"]) < 10:
            return

        if perf["speedup_calculated"]:
            return

        # Calculate average times
        avg_pre = sum(perf["pre_compile_times"]) / len(perf["pre_compile_times"])
        avg_post = sum(perf["post_compile_times"]) / len(perf["post_compile_times"])

        # Calculate speedup
        speedup = avg_pre / avg_post if avg_post > 0 else 1.0

        # Log results
        self.logger.info("🚀 torch.compile Performance Analysis:")
        self.logger.info(f"   Compilation time: {perf['compilation_time']:.2f}s")
        self.logger.info(f"   Pre-compile avg: {avg_pre:.4f}s")
        self.logger.info(f"   Post-compile avg: {avg_post:.4f}s")
        self.logger.info(f"   Speedup: {speedup:.2f}x")

        if speedup > 1.1:
            self.logger.info("   ✅ Significant speedup achieved!")
        elif speedup > 1.0:
            self.logger.info("   ⚡ Modest speedup achieved")
        else:
            self.logger.warning(
                "   ⚠️  No speedup detected - consider disabling torch.compile"
            )

        perf["speedup_calculated"] = True






    def _save_checkpoint(
        self,
        step: int,
        global_step: int,
        epoch: int,
        train_metrics: dict[str, float],
        checkpoint_type: str = "step",
    ) -> None:
        """
        Save checkpoint at specific step.

        Args:
            step (int): Current step within epoch
            global_step (int): Global step across all epochs
            epoch (int): Current epoch
            train_metrics (dict[str, float]): Training metrics
            checkpoint_type (str): Type of checkpoint ("step", "recent")
        """
        if not is_master_process():
            return

        actual_model = self._get_actual_model()
        state = {
            "model_state_dict": (actual_model.state_dict()),
            "optimizer_state_dict": (self.optimizer.state_dict()),
            "scheduler_state_dict": (self.scheduler.state_dict()),
            "epoch": int(epoch),
            "global_step": int(global_step),
            # Batches of THIS epoch already consumed. Epoch checkpoints store
            # -1 to mean "epoch finished, resume at the next one"; step
            # checkpoints store the batch index so a resume can fast-forward
            # into the middle of an epoch instead of replaying it.
            "step_in_epoch": (-1 if checkpoint_type == "epoch" else int(step)),
            # Without these a resume silently draws different dropout masks,
            # different chunker boundary noise and a different augmentation
            # stream, so a "resumed" run is not the run that was interrupted.
            "rng_state": {
                "python": random.getstate(),
                # np.random.get_state() carries a numpy ndarray, which pickles
                # numpy._core.multiarray._reconstruct and makes the whole
                # checkpoint unloadable under torch.load's weights_only=True
                # default -- the evaluation path loads that way. Store the
                # MT19937 key as a plain list so checkpoints stay loadable by
                # every consumer, not just the resume path.
                "numpy": _numpy_rng_to_portable(np.random.get_state()),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
            # DistributedSampler shards by rank, so a checkpoint is only
            # replayable at the same world_size. Recorded so a mismatched
            # resume can say so instead of quietly reading other batches.
            "world_size": int(self.world_size),
            "best_val_loss": self._best_val_loss,
            "best_model_epoch": (self._best_model_epoch),
            "metadata": build_checkpoint_metadata(
                self.cfg,
                run_name=self.run_name,
                training_start_time=self.training_start_time,
                mlflow_run_id=None,
            ),
        }

        recent_path = self.checkpoint_dir / "most_recent_model.pt"
        torch.save(state, recent_path)

        if checkpoint_type == "epoch":
            epoch_index = int(epoch) + 1
            ckpt_path = self.checkpoint_dir / f"epoch_{epoch_index:03d}_model.pt"
            torch.save(state, ckpt_path)
            self.logger.info(f"💾 Epoch checkpoint saved: {ckpt_path}")
        else:
            step_path = self.checkpoint_dir / f"step_{global_step:06d}_model.pt"
            torch.save(state, step_path)
            self.logger.info(f"💾 Step checkpoint saved: {step_path}")

        # Save training metadata separately (YAML) to avoid non-weight pickling
        try:
            metadata = {
                "step": int(step),
                "global_step": int(global_step),
                "epoch": int(epoch),
                "train_metrics": train_metrics,
                # resolve=False: parity with sidecar `train_config.yaml`,
                # tolerates dangling cfg interpolations.
                "config": OmegaConf.to_container(self.cfg, resolve=False),
                "training_start_time": self.training_start_time,
                "run_name": self.run_name,
            }

            if checkpoint_type == "epoch":
                epoch_index = int(epoch) + 1
                step_meta_filename = f"epoch_{epoch_index:03d}_metadata.yaml"
            else:
                step_meta_filename = f"step_{global_step:06d}_metadata.yaml"
            step_meta_path = self.checkpoint_dir / step_meta_filename
            with open(step_meta_path, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save checkpoint metadata YAML: {e}")

    def _should_save_checkpoint(self, global_step: int) -> bool:
        """
        Determine if checkpoint should be saved.

        Args:
            global_step (int): Global step across all epochs

        Returns:
            bool: Whether to save checkpoint at this step
        """
        if self.checkpoint_save_strategy != "step":
            return False

        # Check if this is the step right after warm-up completion
        if global_step == self.num_warmup_steps + 1:
            return True

        # Check if it's time for regular checkpoint (every N steps)
        steps_since_last = global_step - self.last_checkpoint_step
        return steps_since_last >= self.checkpoint_save_steps


    def _get_actual_model(self):
        """
        Get the actual model, handling DDP wrapper.

        Returns:
            The actual model (unwrapped from DDP if necessary)
        """
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_model_dtype(self) -> torch.dtype:
        """
        Get torch_dtype from model, handling DDP wrapper.
        Always returns float32 as we use fixed float32 dtype.

        Returns:
            torch.dtype: Always torch.float32
        """
        return torch.float32

    def _encode_sequences(
        self, sequences: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode DNA sequences using the model's tokenizer.

        Args:
            sequences (list[str]): List of DNA sequences to encode.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                (input_ids, attention_mask)
        """
        # Get actual model (unwrap DDP if needed) to access tokenizer
        actual_model = self._get_actual_model()

        # Tokenize sequences
        # "transformer" (self-attention backbone) reuses the HyenaDNA HF
        # tokenizer verbatim, so it shares the HyenaDNA tokenization path
        # (attention_mask=None; the chunker handles padding downstream and
        # the backbone gets its own key-padding mask from the chunker).
        # "caduceus" is here for the same reason as "transformer": it ships no
        # tokenizer and borrows the HyenaDNA one, whose ids its own
        # complement_map already implies (checked in _assert_caduceus_token_ids).
        for sequence_index, sequence in enumerate(sequences):
            if len(sequence) != self.cfg.model.seq_length:
                self.logger.info(
                    f"Sequence {sequence_index} has length {len(sequence)} "
                    f"(expected {self.cfg.model.seq_length})"
                )
        encoded = actual_model.tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.model.tokenizer_max_length,
            return_offsets_mapping=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = None

        return input_ids, attention_mask

    def _log_token_id_distribution(
        self, input_ids: torch.Tensor, phase: str, batch_index: int
    ) -> None:
        """Log distribution of token IDs for a batch after encoding.

        Args:
            input_ids (torch.Tensor): Token IDs tensor of shape [batch, seq_len]
            phase (str): Phase name (e.g., "train", "val", "test")
            batch_index (int): Batch index within the current epoch/phase
        """
        if not is_master_process():
            return

        if input_ids is None or input_ids.numel() == 0:
            return

        # Flatten and count occurrences of each token id
        flat_ids = input_ids.detach().view(-1).to(torch.long)
        max_id = int(flat_ids.max().item())
        counts = torch.bincount(flat_ids, minlength=max_id + 1)

        present_indices = torch.nonzero(counts).squeeze(1).tolist()
        # Build compact distribution dict for readability
        distribution = {int(idx): int(counts[idx].item()) for idx in present_indices}

        # Basic stats to aid quick inspection
        unique_tokens = len(present_indices)
        message_prefix = (
            f"🧮 Token ID distribution ({phase}) batch={batch_index} | "
            f"unique={unique_tokens}, min_id={int(flat_ids.min().item())}, "
            f"max_id={max_id}: "
        )
        # Ensure line length stays reasonable; split if needed
        self.logger.info(message_prefix + f"{distribution}")

    def _log_ref_char_distribution(
        self, sequences: list[str], phase: str, batch_index: int
    ) -> None:
        """Log distribution of characters in reference sequences per batch.

        Counts characters exactly as they appear, distinguishing uppercase and
        lowercase (e.g., 'A' vs 'a'). Also logs any non-ACGTN- characters.

        Args:
            sequences (list[str]): List of reference sequences (post-clipping)
            phase (str): Phase name (e.g., "train", "val", "test")
            batch_index (int): Batch index within the current epoch/phase
        """
        if not is_master_process():
            return

        if not sequences:
            return

        from collections import Counter

        char_counter: Counter[str] = Counter()
        total_chars: int = 0

        for seq in sequences:
            if not seq:
                continue
            char_counter.update(seq)
            total_chars += len(seq)

        unique_chars = len(char_counter)

        header = (
            f"🔡 Ref char distribution ({phase}) batch={batch_index} | "
            f"unique={unique_chars}, total={total_chars}"
        )
        self.logger.info(header)

        # Convert Counter to a regular dict sorted by frequency (desc), then key
        sorted_items = sorted(char_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        distribution = {k: int(v) for k, v in sorted_items}
        self.logger.info(f"{distribution}")

    def _compute_clip_threshold_from_progress(
        self,
        global_step: int,
        total_steps: int,
        min_clip: float = 1.0,
        max_clip: float = 10.0,
    ) -> float:
        """
        Linearly increase gradient clipping threshold from min_clip to max_clip
        over the course of training.

        Args:
            global_step (int): Current 0-based global training step.
            total_steps (int): Total number of training steps across all epochs.
            min_clip (float): Starting clipping threshold.
            max_clip (float): Final clipping threshold at the end of training.

        Returns:
            float: Clipping threshold for the current step.
        """
        if total_steps <= 1:
            return float(max_clip)
        # Clamp progress to [0, 1]
        denom: int = max(total_steps - 1, 1)
        progress: float = max(0.0, min(1.0, float(global_step) / float(denom)))
        return float(min_clip + (max_clip - min_clip) * progress)

    def _apply_gradient_clipping(self, batch_index: int) -> None:
        """
        Apply progressive gradient clipping and log clipping statistics.

        Args:
            batch_index (int): Index of the current batch within the epoch.

        Returns:
            None: No return value.
        """
        current_global_step: int = (
            len(self.train_loader) * getattr(self, "_current_epoch", 0) + batch_index
        )
        total_steps: int = int(self.num_training_steps)
        clip_value: float = self._compute_clip_threshold_from_progress(
            current_global_step,
            total_steps,
            self.gradient_clip_min,
            self.gradient_clip_max,
        )

        total_norm_before: float = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter_norm = parameter.grad.data.norm(2)
                total_norm_before += parameter_norm.item() ** 2
        total_norm_before = total_norm_before**0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)

        total_norm_after: float = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter_norm = parameter.grad.data.norm(2)
                total_norm_after += parameter_norm.item() ** 2
        total_norm_after = total_norm_after**0.5

        if self.gradient_clip_monitor is not None:
            global_step_for_log = (
                len(self.train_loader) * self._current_epoch + batch_index
            )
            self.gradient_clip_monitor.log_clipping(
                total_norm_before, total_norm_after, global_step_for_log
            )
        return

    def _encode_and_maybe_log(
        self, ref_seqs: list[str], phase: str, batch_index: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode DNA sequences to input IDs and optionally log token/reference
        distributions for debugging.

        Args:
            ref_seqs (list[str]): List of reference sequences for the batch.
            phase (str): Current phase (e.g., "train", "val").
            batch_index (int): Batch index used for conditional logging.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                input_ids and optional attention_mask
        """
        ref_input_ids, attention_mask = self._encode_sequences(ref_seqs)
        if self.enable_distribution_logging:
            self._log_token_id_distribution(
                input_ids=ref_input_ids, phase=phase, batch_index=batch_index
            )
            self._log_ref_char_distribution(
                sequences=ref_seqs, phase=phase, batch_index=batch_index
            )
        return ref_input_ids, attention_mask

    def _convert_to_hard_labels_classification(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert soft/continuous labels to hard labels by keeping only the center
        position of consecutive positive regions.

        For acceptor (channel 0): keeps center of consecutive positives
        For donor (channel 1): keeps center of consecutive positives
        For neither (channel 2): no change (complement of acceptor and donor)

        Window sizes are read from config.training.classification.

        Args:
            labels (torch.Tensor): Labels tensor with shape [B, 3, L].

        Returns:
            torch.Tensor: Hard labels with same shape [B, 3, L].
        """
        hard_labels = labels.clone()

        # Get window sizes from config
        acceptor_window = self.cfg.training.classification.acceptor_window_size
        donor_window = self.cfg.training.classification.donor_window_size

        # Process acceptor channel (index 0)
        acceptor_channel = hard_labels[:, 0, :]  # [B, L]
        acceptor_hard = self._find_center_of_nonzero_regions(
            acceptor_channel, window_size=acceptor_window, keep_original_value=False
        )
        hard_labels[:, 0, :] = acceptor_hard

        # Process donor channel (index 1)
        donor_channel = hard_labels[:, 1, :]  # [B, L]
        donor_hard = self._find_center_of_nonzero_regions(
            donor_channel, window_size=donor_window, keep_original_value=False
        )
        hard_labels[:, 1, :] = donor_hard

        # Neither channel (index 2) is complement: both acceptor and donor are 0
        hard_labels[:, 2, :] = (
            (hard_labels[:, 0, :] == 0) & (hard_labels[:, 1, :] == 0)
        ).float()

        return hard_labels

    def _find_center_of_nonzero_regions(
        self,
        channel: torch.Tensor,
        window_size: int,
        keep_original_value: bool = False,
    ) -> torch.Tensor:
        """
        Find centers of consecutive non-zero regions and keep only centers.
        Optimized vectorized implementation for GPU batch processing.

        Args:
            channel (torch.Tensor): Single channel tensor [B, L].
            window_size (int): Expected size of consecutive non-zero region.
            keep_original_value (bool): If True, keep original value at center.
                If False, assign 1.0 at center (for classification).

        Returns:
            torch.Tensor: Hard label tensor [B, L] with only centers marked.
        """
        batch_size, seq_len = channel.shape
        device = channel.device

        # Create nonzero mask for entire batch [B, L] - vectorized
        nonzero_mask = (channel != 0.0).float()

        # Find transitions for entire batch using diff - vectorized
        # Prepend zeros to detect starts at position 0
        padded_mask = torch.cat(
            [torch.zeros(batch_size, 1, device=device), nonzero_mask], dim=1
        )
        diff = torch.diff(padded_mask, dim=1)  # [B, L]

        # Find region starts (0->1 transitions) - vectorized
        is_start = diff == 1.0  # [B, L]

        # Get all start positions across batch
        start_batch_idx, start_pos = torch.where(is_start)

        if len(start_batch_idx) == 0:
            return torch.zeros_like(channel)

        # Calculate center positions directly: start + window_size // 2 - vectorized
        center_offset = window_size // 2
        center_positions = start_pos + center_offset

        # Validate centers are within bounds - vectorized
        valid_mask = center_positions < seq_len
        valid_batch_idx = start_batch_idx[valid_mask]
        valid_center_pos = center_positions[valid_mask]

        # Extract ends for validation by checking window_size positions ahead
        # end_pos = start_pos + window_size for each region
        expected_end_pos = start_pos + window_size

        # Validate region lengths by checking if expected_end_pos is valid
        # and if all positions from start to end are non-zero - vectorized
        valid_end_mask = expected_end_pos <= seq_len

        if not valid_end_mask.all():
            # Find first invalid region
            invalid_idx = (~valid_end_mask).nonzero(as_tuple=False)[0].item()
            b_idx = start_batch_idx[invalid_idx].item()
            s_pos = start_pos[invalid_idx].item()
            e_pos = expected_end_pos[invalid_idx].item()
            raise ValueError(
                f"Region would extend beyond sequence boundary: "
                f"batch {b_idx}, start {s_pos}, expected end {e_pos}, "
                f"seq_len {seq_len}"
            )

        # Validate that all positions in each window are non-zero
        # Create index tensors for all positions in all windows - vectorized
        # This checks window_size positions starting from each start_pos
        window_offsets = torch.arange(window_size, device=device).unsqueeze(
            0
        )  # [1, window_size]
        all_positions = start_pos.unsqueeze(1) + window_offsets  # [N, window_size]

        # Gather values at all positions - vectorized
        # Expand batch indices to match all positions
        expanded_batch_idx = start_batch_idx.unsqueeze(1).expand(
            -1, window_size
        )  # [N, window_size]

        # Check if all positions in window are nonzero using advanced indexing
        window_values = nonzero_mask[
            expanded_batch_idx, all_positions
        ]  # [N, window_size]
        all_nonzero = (window_values != 0).all(dim=1)  # [N]

        if not all_nonzero.all():
            # Find first invalid region
            invalid_idx = (~all_nonzero).nonzero(as_tuple=False)[0].item()
            b_idx = start_batch_idx[invalid_idx].item()
            s_pos = start_pos[invalid_idx].item()
            e_pos = expected_end_pos[invalid_idx].item()

            # Count actual non-zero length
            window_vals = window_values[invalid_idx]
            if window_vals.any():
                # Find first zero in window
                first_zero = (window_vals == 0).int().argmax()
                actual_length = first_zero.item()
            else:
                actual_length = 0

            raise ValueError(
                f"Region length {actual_length} does not match expected "
                f"window_size {window_size} at batch {b_idx}, "
                f"positions [{s_pos}:{e_pos}]"
            )

        # Assign values at centers using advanced indexing - fully vectorized
        hard_channel = torch.zeros_like(channel)
        if keep_original_value:
            hard_channel[valid_batch_idx, valid_center_pos] = channel[
                valid_batch_idx, valid_center_pos
            ]
        else:
            hard_channel[valid_batch_idx, valid_center_pos] = 1.0

        return hard_channel

    def _convert_to_hard_labels_regression(
        self, psi_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert continuous PSI values to hard labels by keeping only the center
        position of consecutive non-zero regions.

        For acceptor PSI (channel 0): keeps center of consecutive non-zeros
        For donor PSI (channel 1): keeps center of consecutive non-zeros

        Window sizes are read from config.training.regression.

        Args:
            psi_targets (torch.Tensor): PSI targets with shape [B, L, 2].

        Returns:
            torch.Tensor: Hard PSI targets with same shape [B, L, 2].
        """
        hard_psi = torch.zeros_like(psi_targets)

        # Get window sizes from config
        acceptor_window = self.cfg.training.regression.acceptor_window_size
        donor_window = self.cfg.training.regression.donor_window_size

        # Process acceptor PSI (channel 0)
        acceptor_psi = psi_targets[:, :, 0]  # [B, L]
        acceptor_hard = self._find_center_of_nonzero_regions(
            acceptor_psi, window_size=acceptor_window, keep_original_value=True
        )
        hard_psi[:, :, 0] = acceptor_hard

        # Process donor PSI (channel 1)
        donor_psi = psi_targets[:, :, 1]  # [B, L]
        donor_hard = self._find_center_of_nonzero_regions(
            donor_psi, window_size=donor_window, keep_original_value=True
        )
        hard_psi[:, :, 1] = donor_hard

        return hard_psi

    def _forward_logits(
        self,
        ref_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Run model forward once and return both classification and regression outputs.

        Args:
            ref_input_ids (torch.Tensor): Token IDs for input sequences.
            attention_mask (torch.Tensor | None): Attention mask aligned with
                `ref_input_ids` for models that require it (e.g., nucleotide-
                transformer). Shape [batch_size, seq_len].
            features (torch.Tensor | None): Optional per-position feature track
                of shape [batch_size, seq_len, k] injected into the model's nt
                embeddings (intron-structure / conservation). None disables it.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
                ``(cls_logits, reg_outputs, aux)`` where ``aux`` may carry
                ``"region"`` and ``"exon"`` per-position logits when the
                corresponding heads are enabled, otherwise empty.
        """
        # During evaluation with DDP, avoid using the DDP wrapper to prevent
        # parameter/buffer broadcasts that require all ranks to participate.
        model_to_use = (
            self.model
            if (self.model.training or not self.is_distributed)
            else self._get_actual_model()
        )

        with torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype if self.amp_enabled else torch.float32,
            enabled=self.amp_enabled,
        ):
            cls_logits, aux_outputs = model_to_use(
                input_ids=ref_input_ids,
                attention_mask=attention_mask,
                features=features,
            )
        # Regression head was removed; reg_outputs always None for
        # backwards-compatible call sites that still expect a 3-tuple.
        return cls_logits, None, aux_outputs

    def _normalize_label_shape(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Ensure labels tensor has shape [B, C, seq_len] with C == 3.

        Args:
            labels (torch.Tensor): Labels tensor with possible shapes
                [B, 3, L] or [3, L].

        Returns:
            torch.Tensor: Labels with shape [B, 3, L].

        Raises:
            ValueError: If labels have an unexpected shape.
        """
        if len(labels.shape) == 3 and labels.shape[1] == 3:
            return labels
        if len(labels.shape) == 2 and labels.shape[0] == 3:
            return labels.unsqueeze(0)
        raise ValueError(f"Unexpected labels shape: {labels.shape}")

    def _align_logits_and_labels(
        self, logits: torch.Tensor, labels: torch.Tensor, *, phase: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align logits [B, L, C] and labels [B, 3, L] by center-cropping to
        the minimum common sequence length. Returns possibly cropped tensors.

        Args:
            logits (torch.Tensor): Classification logits of shape [B, L, C].
            labels (torch.Tensor): Labels of shape [B, 3, L].
            phase (str): Phase name for logging context.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Aligned (logits, labels).
        """
        logits_len: int = int(logits.shape[1])
        labels_len: int = int(labels.shape[2])
        if logits_len == labels_len:
            return logits, labels

        target_len: int = min(logits_len, labels_len)

        # Crop logits [B, L, C]
        s_lg, e_lg = center_indices(logits_len, target_len)
        logits = logits[:, s_lg:e_lg, :]
        # Crop labels [B, 3, L]
        s_lb, e_lb = center_indices(labels_len, target_len)
        labels = labels[:, :, s_lb:e_lb]
        return logits, labels

    def _align_regression_and_targets(
        self,
        reg_outputs: torch.Tensor | None,
        psi_targets: torch.Tensor | None,
        *,
        ref_len: int,
        phase: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Align regression outputs [B, L, 2] and targets [B, L, 2] to ref_len
        by center-cropping. Returns possibly cropped tensors.

        Args:
            reg_outputs (torch.Tensor | None): Regression outputs.
            psi_targets (torch.Tensor | None): Regression targets.
            ref_len (int): Reference sequence length to align to.
            phase (str): Phase name for logging context.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: Aligned tensors.
        """
        if reg_outputs is None or psi_targets is None:
            return reg_outputs, psi_targets

        out_len: int = int(reg_outputs.shape[1])
        tgt_len: int = int(psi_targets.shape[1])
        target_len: int = min(ref_len, out_len, tgt_len)
        if out_len == target_len and tgt_len == target_len:
            return reg_outputs, psi_targets

        s_ro, e_ro = center_indices(out_len, target_len)
        reg_outputs = reg_outputs[:, s_ro:e_ro, :]
        s_pt, e_pt = center_indices(tgt_len, target_len)
        psi_targets = psi_targets[:, s_pt:e_pt, :]
        return reg_outputs, psi_targets

    def _compute_aux_losses(
        self,
        aux_outputs: dict[str, torch.Tensor],
        batch: dict,
        keep_mask: torch.Tensor,
        ref_seq_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute auxiliary region/exon CE losses.

        Center-crops auxiliary logits to the labels' sequence length and
        applies cross-entropy with ``ignore_index=-1`` so that padding and
        single-exon-transcript bases are excluded from the loss.

        Args:
            aux_outputs (dict[str, torch.Tensor]): Dict from the model
                forward containing ``"region"`` and/or ``"exon"`` logits
                with shape ``[B, L, C]``.
            batch (dict): Batch dict.  Region/exon labels are read from
                ``batch["region_labels"]`` and ``batch["exon_labels"]``
                when present (long tensors of shape ``[B, L]`` with -1 for
                ignore).
            keep_mask (torch.Tensor): Boolean mask aligned with the batch
                dimension that selects samples passed through the model.
            ref_seq_len (int): Reference sequence length to crop labels to.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: Region and exon
                loss tensors (each None when the head is disabled, the
                corresponding label key is missing, or every label position
                is the ignore index).
        """
        aux_region_loss: torch.Tensor | None = None
        aux_exon_loss: torch.Tensor | None = None

        if (
            self.is_aux_region_enabled
            and "region" in aux_outputs
            and batch.get("region_labels") is not None
        ):
            aux_region_loss = self._compute_single_aux_loss(
                logits=aux_outputs["region"],
                labels=batch["region_labels"][keep_mask].to(self.device).long(),
                criterion=self.aux_region_criterion,
                ref_seq_len=ref_seq_len,
            )

        if self.is_aux_exon_enabled and "exon" in aux_outputs:
            exon_labels: torch.Tensor | None = None
            if (
                self.aux_exon_binary_from_region
                and batch.get("region_labels") is not None
            ):
                # Binary exon(1)/intron(0) from region labels; keep -1 ignore.
                region = batch["region_labels"][keep_mask].to(self.device).long()
                exon_labels = torch.where(
                    region < 0, region, (region >= 1).long()
                )
            elif batch.get("exon_labels") is not None:
                exon_labels = batch["exon_labels"][keep_mask].to(self.device).long()

            if exon_labels is not None:
                aux_exon_loss = self._compute_single_aux_loss(
                    logits=aux_outputs["exon"],
                    labels=exon_labels,
                    criterion=self.aux_exon_criterion,
                    ref_seq_len=ref_seq_len,
                )

        return aux_region_loss, aux_exon_loss

    def _compute_single_aux_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        ref_seq_len: int,
    ) -> torch.Tensor | None:
        """Center-crop logits/labels to a common length and run CE.

        Args:
            logits (torch.Tensor): Per-position logits, shape ``[B, L, C]``.
            labels (torch.Tensor): Per-position class indices, shape
                ``[B, L]`` with -1 for ignore.
            criterion (nn.Module): Cross-entropy criterion with
                ``ignore_index=-1``.
            ref_seq_len (int): Reference (classification head) sequence
                length used as a target crop length when consistent.

        Returns:
            torch.Tensor | None: Scalar loss tensor, or ``None`` when no
                non-ignore positions remain after cropping.
        """
        logits_len = int(logits.shape[1])
        labels_len = int(labels.shape[1])
        target_len = min(logits_len, labels_len, ref_seq_len)
        if target_len <= 0:
            return None

        if logits_len != target_len:
            s, e = center_indices(logits_len, target_len)
            logits = logits[:, s:e, :]
        if labels_len != target_len:
            s, e = center_indices(labels_len, target_len)
            labels = labels[:, s:e]

        # Mask out ignore (-1) positions explicitly so the criterion need not
        # support ignore_index (MultiClassFocalLoss does not). Numerically
        # identical to CrossEntropyLoss(ignore_index=-1) for the CE path since
        # both reduce by mean over the retained positions.
        c = logits.shape[-1]
        logits_flat = logits.reshape(-1, c)
        labels_flat = labels.reshape(-1)
        valid = labels_flat != -1
        if not bool(valid.any()):
            return None
        return criterion(logits_flat[valid], labels_flat[valid])

    @staticmethod
    def masked_usage_bce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor | None:
        """Compute BCE on measured splice-site usage positions only.

        Treats each per-tissue usage value in ``[0, 1]`` as a Bernoulli
        target (Pangolin's formulation) and averages the elementwise BCE over
        positions where a measurement exists (``label >= 0``); masked
        positions (``USAGE_IGNORE == -1``) are excluded.

        Args:
            logits (torch.Tensor): Usage logits of shape ``(..., num_tissues)``.
            labels (torch.Tensor): Usage targets of the same shape; entries
                ``< 0`` are treated as unmeasured and ignored.
            criterion (nn.Module): ``BCEWithLogitsLoss(reduction="none")``.

        Returns:
            torch.Tensor | None: Scalar mean BCE over measured positions, or
                ``None`` when no position is measured.
        """
        mask = labels >= 0.0
        if not bool(mask.any()):
            return None
        per_element = criterion(logits, labels.clamp(min=0.0))
        return per_element[mask].mean()

    def _compute_usage_loss(
        self,
        aux_outputs: dict[str, torch.Tensor],
        batch: dict,
        keep_mask: torch.Tensor,
        ref_seq_len: int,
    ) -> torch.Tensor | None:
        """Compute the masked-BCE splice-site usage loss for a batch.

        Center-crops the usage logits and labels to a common length (matching
        the auxiliary-head convention) before applying :meth:`masked_usage_bce`.

        Args:
            aux_outputs (dict[str, torch.Tensor]): Model forward outputs;
                ``"usage"`` holds logits of shape ``[B, L, num_tissues]``.
            batch (dict): Batch dict; ``"usage_labels"`` holds targets of
                shape ``[B, L, num_tissues]`` with -1 for unmeasured bases.
            keep_mask (torch.Tensor): Boolean mask over the batch dimension
                selecting samples passed through the model.
            ref_seq_len (int): Reference (classification head) sequence length.

        Returns:
            torch.Tensor | None: Scalar loss tensor, or ``None`` when the head
                is disabled, labels are absent, or no position is measured.
        """
        if not (
            self.is_aux_usage_enabled
            and "usage" in aux_outputs
            and batch.get("usage_labels") is not None
        ):
            return None

        logits = aux_outputs["usage"]
        labels = batch["usage_labels"][keep_mask].to(self.device).float()

        logits_len = int(logits.shape[1])
        labels_len = int(labels.shape[1])
        target_len = min(logits_len, labels_len, ref_seq_len)
        if target_len <= 0:
            return None

        if logits_len != target_len:
            s, e = center_indices(logits_len, target_len)
            logits = logits[:, s:e, :]
        if labels_len != target_len:
            s, e = center_indices(labels_len, target_len)
            labels = labels[:, s:e, :]

        return self.masked_usage_bce(logits, labels, self.aux_usage_criterion)

    def _compute_spliced_loss(
        self,
        aux_outputs: dict[str, torch.Tensor],
        batch: dict,
        keep_mask: torch.Tensor,
        ref_seq_len: int,
    ) -> torch.Tensor | None:
        """Per-tissue spliced/not cross-entropy over labelled positions.

        Mirrors :meth:`_compute_usage_loss` for the cryptic-aware 2-class
        spliced head (Pangolin-faithful). Logits are ``[B, L, T, 2]``; labels
        ``[B, L, T]`` with -1 for unmeasured positions (ignored via
        ``ignore_index=-1``). Returns ``None`` when the head is disabled,
        labels are absent, or no position is measured.

        Args:
            aux_outputs (dict[str, torch.Tensor]): Model outputs; ``"spliced"``
                holds logits of shape ``[B, L, num_tissues, 2]``.
            batch (dict): Batch dict; ``"spliced_labels"`` holds targets of
                shape ``[B, L, num_tissues]`` with -1 for unmeasured bases.
            keep_mask (torch.Tensor): Boolean mask over the batch dimension.
            ref_seq_len (int): Reference (classification head) sequence length.

        Returns:
            torch.Tensor | None: Scalar cross-entropy, or None.
        """
        if not (
            self.is_aux_spliced_enabled
            and "spliced" in aux_outputs
            and batch.get("spliced_labels") is not None
        ):
            return None

        logits = aux_outputs["spliced"]
        labels = batch["spliced_labels"][keep_mask].to(self.device).long()

        logits_len = int(logits.shape[1])
        labels_len = int(labels.shape[1])
        target_len = min(logits_len, labels_len, ref_seq_len)
        if target_len <= 0:
            return None
        if logits_len != target_len:
            s, e = center_indices(logits_len, target_len)
            logits = logits[:, s:e]
        if labels_len != target_len:
            s, e = center_indices(labels_len, target_len)
            labels = labels[:, s:e]

        if not bool((labels >= 0).any()):
            return None
        num_cls = int(logits.shape[-1])
        return self.aux_spliced_criterion(
            logits.reshape(-1, num_cls), labels.reshape(-1)
        )

    @staticmethod
    def masked_phylop_mse(
        preds: torch.Tensor,
        phylop: torch.Tensor,
        clip: float = DEFAULT_PHYLOP_CLIP,
    ) -> torch.Tensor | None:
        """MSE between predictions and scaled phyloP over positions with a value.

        The target is ``clip(phyloP, -clip, clip) / clip``, the same [-1, 1]
        scaling the conservation input track uses. Raw phyloP100way spans
        roughly [-20, 10] with most bases near 0, so without the clip a handful
        of strongly accelerated bases would dominate the squared error.

        Args:
            preds (torch.Tensor): Predictions, shape ``[B, L]``.
            phylop (torch.Tensor): Raw phyloP, shape ``[B, L]``; NaN marks
                positions without a value (assembly gap or window padding).
            clip (float): Clip bound C for the target scaling.

        Returns:
            torch.Tensor | None: Scalar mean squared error over valid
                positions, or ``None`` when no position has a value.
        """
        valid = torch.isfinite(phylop)
        if not bool(valid.any()):
            return None
        target = phylop[valid].clamp(-clip, clip) / clip
        return nn.functional.mse_loss(preds[valid], target)

    def _compute_phylop_loss(
        self,
        aux_outputs: dict[str, torch.Tensor],
        batch: dict,
        keep_mask: torch.Tensor,
        ref_seq_len: int,
    ) -> torch.Tensor | None:
        """Compute the masked-MSE phyloP regression loss for a batch.

        Center-crops predictions and targets to a common length (matching the
        auxiliary-head convention) before applying :meth:`masked_phylop_mse`.

        Args:
            aux_outputs (dict[str, torch.Tensor]): Model forward outputs;
                ``"phylop"`` holds predictions of shape ``[B, L]``.
            batch (dict): Batch dict; ``"phylop_labels"`` holds raw phyloP of
                shape ``[B, L]`` with NaN where no value exists.
            keep_mask (torch.Tensor): Boolean mask over the batch dimension
                selecting samples passed through the model.
            ref_seq_len (int): Reference (classification head) sequence length.

        Returns:
            torch.Tensor | None: Scalar loss tensor, or ``None`` when the head
                is disabled, labels are absent, or no position has a value.
        """
        if not (
            self.is_aux_phylop_enabled
            and "phylop" in aux_outputs
            and batch.get("phylop_labels") is not None
        ):
            return None

        preds = aux_outputs["phylop"]
        labels = batch["phylop_labels"][keep_mask].to(self.device).float()

        preds_len = int(preds.shape[1])
        labels_len = int(labels.shape[1])
        target_len = min(preds_len, labels_len, ref_seq_len)
        if target_len <= 0:
            return None
        if preds_len != target_len:
            s, e = center_indices(preds_len, target_len)
            preds = preds[:, s:e]
        if labels_len != target_len:
            s, e = center_indices(labels_len, target_len)
            labels = labels[:, s:e]

        return self.masked_phylop_mse(preds, labels)

    def _compute_masked_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        labels_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute loss with optional element-wise mask. Flattens time/batch dims.

        Args:
            logits (torch.Tensor): Logits with shape [B, L, C].
            labels (torch.Tensor): Labels with shape [B, 3, L].
            labels_dtype (torch.dtype): Target dtype for labels
                (e.g., torch.long or torch.float).

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        num_classes: int = logits.shape[-1]
        logits_flat = logits.reshape(-1, num_classes).float()
        labels_flat = labels.transpose(1, 2).contiguous().view(-1, num_classes)

        # Multiclass focal loss branch supports hard indices or soft labels
        if getattr(self, "loss_type", "") == "multi_class_focal_loss":
            # If labels are provided as one-hot/soft [N, C], pass directly when float
            if labels_flat.dtype in (torch.float16, torch.float32, torch.float64):
                targets_for_loss = labels_flat.float()
            else:
                # One-hot ints -> indices
                targets_for_loss = torch.argmax(labels_flat, dim=-1).long()
            loss = self.criterion(logits_flat, targets_for_loss)
            return loss

        # Default: multi-label losses expect float multi-hot
        if labels_dtype is torch.long:
            labels_flat = labels_flat.long()
        else:
            labels_flat = labels_flat.float()
        loss = self.criterion(logits_flat, labels_flat)
        return loss

    def _resolve_stage_lr_scales(self, key: str) -> list[float]:
        """Resolve per-stage LR scale list from the config.

        Per-stage entries (``model.dynamic_chunker.stages[i].<key>``)
        take precedence; missing per-stage entries inherit the flat
        ``model.dynamic_chunker.<key>`` default; if neither is set the
        scale is ``1.0`` (no modulation, backward-compat).

        Args:
            key (str): Either ``"encoder_lr_scale"`` or
                ``"dechunker_lr_scale"``.

        Returns:
            list[float]: One scale per stage (length ``num_stages``,
                always ``>=1``).
        """
        flat_scale = float(
            OmegaConf.select(
                self.cfg,
                f"model.dynamic_chunker.{key}",
                default=1.0,
            )
        )
        stages_cfg = OmegaConf.select(
            self.cfg,
            "model.dynamic_chunker.stages",
            default=None,
        )
        if not stages_cfg:
            return [flat_scale]
        scales: list[float] = []
        for stage_cfg in stages_cfg:
            stage_scale = OmegaConf.select(stage_cfg, key, default=None)
            scales.append(flat_scale if stage_scale is None else float(stage_scale))
        return scales

    def _match_stage_param(self, name: str) -> tuple[str, int] | None:
        """Match a parameter name to ``(role, stage_index)`` for a chunker stage.

        Recognized name patterns:

        - ``hierarchical_chunker.chunkers.<i>.encoder.*``  → ``("encoder", i)``
        - ``hierarchical_chunker.chunkers.<i>.router_*``   → ``("encoder", i)``
          (the ``hnet`` router takes its stage's rate, as in H-Net)
        - ``hierarchical_chunker.dechunkers.<i>.*``        → ``("dechunker", i)``

        Args:
            name (str): Parameter name from ``named_parameters()``.

        Returns:
            tuple[str, int] | None: Role + stage index, or ``None`` if
                the name is not a chunker-stage parameter.
        """
        prefix_chunkers = "hierarchical_chunker.chunkers."
        prefix_dechunkers = "hierarchical_chunker.dechunkers."
        if name.startswith(prefix_chunkers):
            tail = name[len(prefix_chunkers) :]
            stage_str, _, rest = tail.partition(".")
            if stage_str.isdigit() and rest.startswith(("encoder.", "router_")):
                return ("encoder", int(stage_str))
            return None
        if name.startswith(prefix_dechunkers):
            tail = name[len(prefix_dechunkers) :]
            stage_str, _, _ = tail.partition(".")
            if stage_str.isdigit():
                return ("dechunker", int(stage_str))
            return None
        return None

    def _resolve_stage_target_rates(self) -> list[float]:
        """Resolve per-stage target chunk rates from the config.

        Returns the outer-to-inner list. Reads
        ``model.dynamic_chunker.stages`` first (multi-stage F5);
        falls back to a length-1 list built from the flat
        ``model.dynamic_chunker.target_chunk_rate`` (legacy single-stage).

        Returns:
            list[float]: Per-stage target chunk rates. Always non-empty
                (at minimum ``[flat_rate]``).
        """
        stages_cfg = OmegaConf.select(
            self.cfg,
            "model.dynamic_chunker.stages",
            default=None,
        )
        flat_rate = float(
            OmegaConf.select(
                self.cfg,
                "model.dynamic_chunker.target_chunk_rate",
                default=0.2,
            )
        )
        if not stages_cfg:
            return [flat_rate]
        rates: list[float] = []
        for stage_cfg in stages_cfg:
            rate = OmegaConf.select(stage_cfg, "target_chunk_rate", default=None)
            rates.append(flat_rate if rate is None else float(rate))
        return rates

    def _compute_boundary_loss(
        self,
        mean_chunk_rate: torch.Tensor,
        mean_boundary_prob: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the dynamic-chunker boundary regularization loss.

        Inputs are stacked per-stage tensors of shape ``[num_stages]``
        (single-stage hierarchies surface a 1-element tensor). The loss
        is computed independently per stage with its own target chunk
        rate, then summed; multi-stage hierarchies thus regularize each
        compression level toward its own target.

        Dispatches on ``self.boundary_loss_type``:

        - ``l1`` (default, DD-018): ``Σ_s |F_s - target_s|`` regularizer.
          Bit-identical to the pre-DD-020 / pre-F5 behaviour for
          single-stage configs (the sum over a 1-element tensor is the
          tensor itself).
        - ``hnet_ratio`` (DD-020, H-Net paper eq. 10):
          ``L_s = N_s/(N_s-1) × ((N_s-1)·F_s·G_s + (1 - F_s)·(1 - G_s))``
          where ``N_s = 1 / target_s``; total = ``Σ_s L_s``. Drives F_s
          and G_s jointly toward 1/N_s and pushes G_s away from 0.5
          (confident routing) at every stage.

        Args:
            mean_chunk_rate (torch.Tensor): Per-stage F of shape
                ``[num_stages]`` (mean of the hard boundary mask, with
                straight-through Gumbel gradient during training).
            mean_boundary_prob (torch.Tensor | None): Per-stage G of
                shape ``[num_stages]`` (mean of ``sigmoid(logits)``).
                Required for ``hnet_ratio``; ignored for ``l1``.

        Returns:
            torch.Tensor: Scalar loss tensor with gradient flowing
                back into every stage's boundary predictor.

        Raises:
            ValueError: If ``hnet_ratio`` is selected but
                ``mean_boundary_prob`` is None or has the wrong shape,
                or if any per-stage target slipped past init validation.
        """
        f_tensor = mean_chunk_rate.float().reshape(-1)
        targets = torch.tensor(
            self.dynamic_chunker_target_rates,
            dtype=f_tensor.dtype,
            device=f_tensor.device,
        )
        if targets.shape[0] != f_tensor.shape[0]:
            raise ValueError(
                "Per-stage target rate count "
                f"({targets.shape[0]}) does not match the model-surfaced "
                f"per-stage F count ({f_tensor.shape[0]}). Check that "
                "model.dynamic_chunker.stages matches the IssaSplice "
                "hierarchical_chunker."
            )
        if self.boundary_loss_type == "l1":
            return (f_tensor - targets).abs().sum()
        # hnet_ratio
        if mean_boundary_prob is None:
            raise ValueError(
                "boundary_loss_type='hnet_ratio' requires the model to "
                "surface 'mean_boundary_prob' in aux_outputs. Update "
                "model/chunker.py to the DD-020 ChunkerOutputs API."
            )
        if torch.any(targets >= 1.0):
            raise ValueError(
                "hnet_ratio loss is undefined at target_chunk_rate>=1.0 for any "
                f"stage; got per-stage rates {self.dynamic_chunker_target_rates}."
            )
        g_tensor = mean_boundary_prob.float().reshape(-1)
        n_inv = targets  # 1 / N_s
        n_minus_1 = (1.0 - n_inv) / n_inv  # N_s - 1
        scale = 1.0 / (1.0 - n_inv)  # N_s / (N_s - 1)
        per_stage_loss = scale * (
            n_minus_1 * f_tensor * g_tensor + (1.0 - f_tensor) * (1.0 - g_tensor)
        )
        return per_stage_loss.sum()

    def _run_train_batch(
        self, batch_index: int, batch: dict[str, Any]
    ) -> tuple[float, float | None, float | None, bool] | None:
        """
        Run a single training batch: filter samples, encode, forward, compute loss,
        backward, optionally clip gradients, and perform optimizer/scheduler steps.

        Args:
            batch_index (int): Index of the batch within the current epoch.
            batch (dict[str, Any]): A batch from the dataloader containing at least
                `ref` and `labels`/`soft_labels`.

        Returns:
            tuple[float, float | None, float | None, bool] | None:
                (
                    total_loss_value,
                    classification_loss_value,
                    regression_loss_value,
                    optimizer_step_performed,
                )
                for the batch, or None if the batch was skipped. Classification or
                regression losses are None when the corresponding heads are disabled.
        """
        # Get reference sequences and labels, and filter out unsuitable samples
        ref_seqs = batch["ref"]
        labels_key = "soft_labels" if self.use_soft_labels else "labels"
        # Be robust: fall back to 'labels' if 'soft_labels' absent
        original_labels = batch.get(labels_key, batch.get("labels"))
        if original_labels is None:
            log_only_master_process(
                self.logger,
                "info",
                f"⚠️  Missing labels in batch (expected '{labels_key}' or 'labels').",
            )
            log_only_master_process(
                self.logger,
                "info",
                f"Skipping batch {batch_index} due to missing labels.",
            )
            return None

        # Build keep_mask and gather stats for logging
        keep_mask, keep_stats = compute_keep_mask_and_stats(
            ref_seqs=ref_seqs,
            original_labels=original_labels,
            seq_length=int(self.cfg.model.seq_length),
            loss_type=getattr(self, "loss_type", ""),
        )
        batch_size = len(ref_seqs)

        # Optional: log filtered count for visibility
        dropped = int((~keep_mask).sum().item())
        if dropped > 0:
            log_only_master_process(
                self.logger,
                "info",
                f"🧹 Dropped {dropped}/{batch_size} samples due to invalid "
                "nucleotides (allowed: A,C,G,T)",
            )
            length_mismatch_total = int(keep_stats.get("length_mismatch_total", 0))
            if length_mismatch_total > 0:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"📏 Length-mismatch sequences dropped: {length_mismatch_total} "
                    f"(expected length: {int(self.cfg.model.seq_length)})",
                )
            overlap_sequences_total = int(keep_stats.get("overlap_sequences_total", 0))
            overlap_positions_total = int(keep_stats.get("overlap_positions_total", 0))
            if overlap_sequences_total > 0:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"🔁 Overlap sequences dropped (multiclass): "
                    f"{overlap_sequences_total}, overlap positions in batch: "
                    f"{overlap_positions_total}",
                )
            unexpected_tokens = keep_stats.get("unexpected_tokens", set())
            log_only_master_process(
                self.logger, "info", f"Unexpected tokens: {unexpected_tokens}"
            )

        # If all samples in the batch are filtered out, skip to the next batch
        if not torch.any(keep_mask):
            log_only_master_process(
                self.logger,
                "info",
                f"All samples in batch {batch_index} were filtered out.",
            )
            return None

        # Apply the filter mask to all relevant batch components
        ref_seqs = [seq for i, seq in enumerate(ref_seqs) if keep_mask[i]]
        labels = original_labels[keep_mask].to(self.device)
        features = batch.get("features")
        if features is not None:
            features = features[keep_mask].to(self.device)

        # positions는 클리핑에 사용하지 않음

        # cfg.model.seq_length 고정 길이 데이터: 클리핑/패딩/마스크 불필요
        # 그대로 사용 (검증만 수행)
        if not isinstance(ref_seqs, list):
            ref_seqs = list(ref_seqs)
        labels = labels.to(self.device)

        # Clear gradients only at the start of accumulation
        if getattr(self, "_accumulated_microbatches", 0) == 0:
            self.engine.zero_grad()

        # Encode sequences to input_ids and optional logging
        ref_input_ids, attention_mask = self._encode_and_maybe_log(
            ref_seqs=ref_seqs, phase="train", batch_index=batch_index
        )

        if hasattr(self, "_first_forward_pass") and self._first_forward_pass:
            log_only_master_process(
                self.logger, "info", "🔥 First forward pass - compiling model..."
            )

        cls_logits, reg_outputs, aux_outputs = self._forward_logits(
            ref_input_ids, attention_mask, features=features
        )

        # Log compilation for first forward pass
        if hasattr(self, "_first_forward_pass") and self._first_forward_pass:
            log_only_master_process(
                self.logger, "info", "⚡ Model compilation completed"
            )
            log_only_master_process(
                self.logger, "info", "   Subsequent forward passes will be faster"
            )
            self._first_forward_pass = False

        # Compute classification loss (if enabled)
        _, seq_len, _ = cls_logits.shape
        if seq_len == 0:
            log_only_master_process(
                self.logger,
                "warning",
                "⚠️  Zero-length outputs after model forward. Skipping batch.",
            )
            return None

        classification_loss_tensor: torch.Tensor | None = None
        if self.is_classification_enabled:
            labels = self._normalize_label_shape(labels)
            # Align length between logits and labels via center-crop
            cls_logits, labels = self._align_logits_and_labels(
                cls_logits, labels, phase="train"
            )
            _, seq_len, _ = cls_logits.shape

            # Apply hard label conversion if enabled
            if self.uses_hard_labels_classification:
                labels = self._convert_to_hard_labels_classification(labels)

            classification_loss_tensor = self._compute_masked_loss(
                logits=cls_logits, labels=labels, labels_dtype=torch.long
            )

        # Optionally compute regression loss if PSI targets are present in batch
        regression_loss_tensor: torch.Tensor | None = None
        donor_psi_full = batch.get("donor_psi")
        acceptor_psi_full = batch.get("acceptor_psi")
        if (
            self.is_regression_enabled
            and donor_psi_full is not None
            and acceptor_psi_full is not None
        ):
            try:
                donor_psi = donor_psi_full[keep_mask].to(self.device).float()
                acceptor_psi = acceptor_psi_full[keep_mask].to(self.device).float()

                # Ensure target shape matches logits: [B, L, 2] -> [acceptor, donor]
                psi_targets = torch.stack([acceptor_psi, donor_psi], dim=-1)

                # Align regression outputs and targets to classification length
                if reg_outputs is None:
                    raise RuntimeError("Regression outputs are None from model forward")
                reg_outputs, psi_targets = self._align_regression_and_targets(
                    reg_outputs, psi_targets, ref_len=seq_len, phase="train"
                )

                # Apply hard label conversion if enabled for regression
                if self.uses_hard_labels_regression:
                    psi_targets = self._convert_to_hard_labels_regression(psi_targets)

                regression_loss_tensor = self.regression_criterion(
                    reg_outputs.float(), psi_targets
                )
            except Exception as e:
                # Fallback to classification-only loss if PSI path fails
                log_only_master_process(
                    self.logger,
                    "warning",
                    f"PSI regression loss skipped due to error: {e}",
                )
                regression_loss_tensor = None

        # Auxiliary region / exon losses (per-base CE with -1 ignore index).
        aux_region_loss_tensor, aux_exon_loss_tensor = self._compute_aux_losses(
            aux_outputs=aux_outputs,
            batch=batch,
            keep_mask=keep_mask,
            ref_seq_len=seq_len,
        )

        # Splice-site usage regression loss (masked BCE over measured bases).
        aux_usage_loss_tensor = self._compute_usage_loss(
            aux_outputs=aux_outputs,
            batch=batch,
            keep_mask=keep_mask,
            ref_seq_len=seq_len,
        )

        # Per-tissue spliced/not classification loss (Pangolin-faithful, 2-class).
        aux_spliced_loss_tensor = self._compute_spliced_loss(
            aux_outputs=aux_outputs,
            batch=batch,
            keep_mask=keep_mask,
            ref_seq_len=seq_len,
        )

        # Per-base phyloP conservation regression loss (masked MSE).
        aux_phylop_loss_tensor = self._compute_phylop_loss(
            aux_outputs=aux_outputs,
            batch=batch,
            keep_mask=keep_mask,
            ref_seq_len=seq_len,
        )

        # Dynamic chunker boundary regularization loss
        # (FR-024, DD-018 / DD-020). Two formulations are selectable
        # via ``model.dynamic_chunker.boundary_loss_type``:
        #   - ``l1`` (default, legacy): ``|F - target_chunk_rate|``;
        #     bit-identical to the pre-DD-020 behaviour, kept so prior
        #     checkpoints can be reproduced.
        #   - ``hnet_ratio`` (paper eq. 10): MoE-style ratio loss using
        #     both F (hard mask mean) and G (soft prob mean) so the
        #     network is pushed toward confident 0/1 boundary decisions
        #     at the configured compression ratio N = 1 / target_rate.
        # ``aux_outputs`` carries the scalars only when the model uses
        # ``tokenizer_type=dynamic_chunker``; baseline paths skip this term.
        boundary_loss_tensor: torch.Tensor | None = None
        mean_chunk_rate_tensor = aux_outputs.get("mean_chunk_rate")
        mean_boundary_prob_tensor = aux_outputs.get("mean_boundary_prob")
        if self.is_dynamic_chunker_enabled and mean_chunk_rate_tensor is not None:
            boundary_loss_tensor = self._compute_boundary_loss(
                mean_chunk_rate=mean_chunk_rate_tensor,
                mean_boundary_prob=mean_boundary_prob_tensor,
            )
            # Track per-batch mean chunk rate for epoch-level MLflow
            # logging. Using getattr/setattr pattern to avoid extending
            # the train-batch return tuple (DD-018 minimum-invasive).
            # F5 multi-stage: log the *effective* chunk rate (∏_s F_s)
            # so the metric remains a single scalar comparable across
            # single-stage and multi-stage runs.
            rate_vec = mean_chunk_rate_tensor.detach().reshape(-1).float()
            buf: list[float] = getattr(self, "_chunker_rate_buf", [])
            buf.append(float(torch.prod(rate_vec).item()))
            self._chunker_rate_buf = buf
            # Per-stage chunk rate (F_s): rows == per-batch, cols == stage.
            stage_rate_buf: list[list[float]] = getattr(
                self, "_chunker_stage_rate_buf", []
            )
            stage_rate_buf.append([float(v) for v in rate_vec.tolist()])
            self._chunker_stage_rate_buf = stage_rate_buf
            # Boundary regularization loss (DD-018 / DD-020). Detach to keep
            # the diagnostic buffer out of the autograd graph; the loss used
            # for `.backward()` is the un-detached tensor above.
            bnd_loss_buf: list[float] = getattr(self, "_chunker_boundary_loss_buf", [])
            bnd_loss_buf.append(float(boundary_loss_tensor.detach().item()))
            self._chunker_boundary_loss_buf = bnd_loss_buf
            # Per-stage mean boundary probability (G_s, soft mask mean).
            # Only emitted by chunkers that surface it (DD-020); legacy
            # paths leave the aux key absent and we skip this buffer.
            if mean_boundary_prob_tensor is not None:
                prob_vec = mean_boundary_prob_tensor.detach().reshape(-1).float()
                stage_prob_buf: list[list[float]] = getattr(
                    self, "_chunker_stage_prob_buf", []
                )
                stage_prob_buf.append([float(v) for v in prob_vec.tolist()])
                self._chunker_stage_prob_buf = stage_prob_buf

        # Combine losses with weights
        total_loss_tensor = torch.tensor(0.0, device=self.device)
        if classification_loss_tensor is not None:
            total_loss_tensor += (
                self.loss_weight_classification * classification_loss_tensor
            )
        if regression_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_regression * regression_loss_tensor
        if aux_region_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_aux_region * aux_region_loss_tensor
        if aux_exon_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_aux_exon * aux_exon_loss_tensor
        if aux_usage_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_aux_usage * aux_usage_loss_tensor
            # Per-batch usage loss for epoch-level logging (minimum-invasive
            # buffer pattern; avoids extending the train-batch return tuple).
            usage_buf: list[float] = getattr(self, "_aux_usage_loss_buf", [])
            usage_buf.append(float(aux_usage_loss_tensor.detach().item()))
            self._aux_usage_loss_buf = usage_buf
        if aux_spliced_loss_tensor is not None:
            total_loss_tensor += (
                self.loss_weight_aux_spliced * aux_spliced_loss_tensor
            )
        if aux_phylop_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_aux_phylop * aux_phylop_loss_tensor
            # Same buffer pattern as the usage loss; train_epoch averages it
            # into the epoch metrics.
            phylop_buf: list[float] = getattr(self, "_aux_phylop_loss_buf", [])
            phylop_buf.append(float(aux_phylop_loss_tensor.detach().item()))
            self._aux_phylop_loss_buf = phylop_buf
        if boundary_loss_tensor is not None:
            total_loss_tensor += self.loss_weight_boundary * boundary_loss_tensor

        loss_value = float(total_loss_tensor.item())
        classification_loss_value: float | None = (
            float(classification_loss_tensor.item())
            if classification_loss_tensor is not None
            else None
        )
        regression_loss_value: float | None = (
            float(regression_loss_tensor.item())
            if regression_loss_tensor is not None
            else None
        )
        aux_region_loss_value: float | None = (
            float(aux_region_loss_tensor.item())
            if aux_region_loss_tensor is not None
            else None
        )
        aux_exon_loss_value: float | None = (
            float(aux_exon_loss_tensor.item())
            if aux_exon_loss_tensor is not None
            else None
        )
        # Scale loss for gradient accumulation and backpropagate via the
        # PyTorch distributed engine.
        loss_scale: float = 1.0 / max(1, self.grad_accumulation_steps)
        self.engine.backward(total_loss_tensor * loss_scale)

        # Increase accumulation counter and perform optimizer step when due
        self._accumulated_microbatches = (
            getattr(self, "_accumulated_microbatches", 0) + 1
        )
        is_accumulation_boundary = (
            self._accumulated_microbatches % max(1, self.grad_accumulation_steps)
        ) == 0
        is_last_microbatch = batch_index == len(self.train_loader) - 1
        optimizer_step_performed = False

        if is_accumulation_boundary or is_last_microbatch:
            if self.gradient_clipping_enabled:
                self._apply_gradient_clipping(batch_index)
            self.engine.step()
            self._accumulated_microbatches = 0
            optimizer_step_performed = True

        return (
            loss_value,
            classification_loss_value,
            regression_loss_value,
            aux_region_loss_value,
            aux_exon_loss_value,
            optimizer_step_performed,
        )

    def train_epoch(self, skip_batches: int = 0) -> dict[str, float]:
        """Train for one epoch (no step-wise validation).

        Args:
            skip_batches (int): Batches to fast-forward past before training,
                used by a mid-epoch resume so the interrupted epoch continues
                from where it stopped rather than replaying from its start.

        Returns:
            dict[str, float]: Aggregated training metrics for the epoch.
        """
        self.logger.info("🏃 Starting training epoch...")
        self.model.train()

        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0.0
        total_aux_region_loss = 0.0
        total_aux_exon_loss = 0.0
        classification_count = 0
        regression_count = 0
        aux_region_count = 0
        aux_exon_count = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=(
                f"Training | Epoch {self._current_epoch + 1}/{self.cfg.training.epochs}"
            ),
            disable=not is_master_process(),
        )
        total_batches = len(self.train_loader)
        self.logger.info(
            f"📊 Training epoch: {total_batches} batches, "
            f"{len(self.train_loader.dataset)} samples"
        )
        if skip_batches:
            self.logger.info(
                "⏩ Mid-epoch resume: skipping %d batches already consumed "
                "before the checkpoint", skip_batches,
            )
        for batch_idx, batch in enumerate(progress_bar):
            # Fast-forward past the batches the interrupted run already
            # trained on. Iterating and discarding (rather than slicing the
            # loader) keeps the sampler's sequence intact, so the batches that
            # follow are the ones the original run would have seen next.
            if batch_idx < skip_batches:
                continue
            batch_result = self._run_train_batch(batch_idx, batch)
            if batch_result is None:
                # Skip batches that were filtered or had missing labels
                continue

            (
                loss_item,
                cls_loss_item,
                reg_loss_item,
                aux_region_loss_item,
                aux_exon_loss_item,
                optimizer_step_performed,
            ) = batch_result

            micro_step = batch_idx + 1
            optim_steps_completed = math.floor(
                micro_step / max(1, self.grad_accumulation_steps)
            )
            global_step = (
                self.steps_per_epoch_optim * getattr(self, "_current_epoch", 0)
            ) + optim_steps_completed

            current_lr = self.scheduler.get_last_lr()[0]
            total_loss += loss_item
            if cls_loss_item is not None:
                total_classification_loss += cls_loss_item
                classification_count += 1
            if reg_loss_item is not None:
                total_regression_loss += reg_loss_item
                regression_count += 1
            if aux_region_loss_item is not None:
                total_aux_region_loss += aux_region_loss_item
                aux_region_count += 1
            if aux_exon_loss_item is not None:
                total_aux_exon_loss += aux_exon_loss_item
                aux_exon_count += 1

            # Progress bar postfix with classification loss
            running_loss = total_loss / (batch_idx + 1)
            postfix_lr = f"{current_lr:.6f}"

            postfix_dict = {
                "loss": f"{running_loss:.4f}",
                "lr": postfix_lr,
            }
            progress_bar.set_postfix(postfix_dict)

            # Log metrics to MLflow periodically
            if is_master_process():
                # Log learning rate periodically (every 100 micro-steps)
                if micro_step % 100 == 0:
                    # Build log message with available losses
                    log_parts = [f"Step {micro_step}: Total = {float(loss_item):.4f}"]
                    if self.is_classification_enabled and classification_count > 0:
                        avg_cls_loss_log = total_classification_loss / max(
                            classification_count, 1
                        )
                        cls_val = float(cls_loss_item) if cls_loss_item else 0.0
                        log_parts.append(f"Cls = {cls_val:.4f}")
                        log_parts.append(f"Avg Cls = {avg_cls_loss_log:.4f}")
                    log_parts.append(f"LR = {current_lr:.6f}")
                    self.logger.info(", ".join(log_parts))
                    # Optionally log regression loss when available
                    if regression_count > 0 and reg_loss_item is not None:
                        avg_reg_loss_log = total_regression_loss / regression_count
                        self.logger.info(
                            f"Step {micro_step}: Reg = {float(reg_loss_item):.4f}, "
                            f"Avg Reg = {avg_reg_loss_log:.4f}"
                        )
                    if aux_region_count > 0 and aux_region_loss_item is not None:
                        avg_aux_region_log = total_aux_region_loss / aux_region_count
                        self.logger.info(
                            f"Step {micro_step}: AuxRegion = "
                            f"{float(aux_region_loss_item):.4f}, "
                            f"Avg AuxRegion = {avg_aux_region_log:.4f}"
                        )
                    if aux_exon_count > 0 and aux_exon_loss_item is not None:
                        avg_aux_exon_log = total_aux_exon_loss / aux_exon_count
                        self.logger.info(
                            f"Step {micro_step}: AuxExon = "
                            f"{float(aux_exon_loss_item):.4f}, "
                            f"Avg AuxExon = {avg_aux_exon_log:.4f}"
                        )
                    phylop_log_buf = getattr(self, "_aux_phylop_loss_buf", [])
                    if phylop_log_buf:
                        self.logger.info(
                            f"Step {micro_step}: AuxPhyloP = "
                            f"{phylop_log_buf[-1]:.4f}, "
                            f"Avg AuxPhyloP = "
                            f"{sum(phylop_log_buf) / len(phylop_log_buf):.4f}"
                        )

                # Check if we should save checkpoint (no validation)
                # Use micro_step for interval checks; save at optimizer cadence
                if self._should_save_checkpoint(global_step) and (
                    (micro_step % max(1, self.grad_accumulation_steps)) == 0
                ):
                    # Determine checkpoint reason for logging
                    if global_step == self.num_warmup_steps + 1:
                        self.logger.info(
                            f"🔥 Saving checkpoint at global step {global_step} "
                            "(warm-up completed)..."
                        )
                    else:
                        self.logger.info(
                            f"💾 Saving checkpoint at global step {global_step} "
                            "(regular interval)..."
                        )

                    # Create temporary train metrics for checkpoint saving
                    temp_train_metrics = {
                        "loss": total_loss / micro_step if micro_step > 0 else 0.0,
                        "accuracy": 0.0,  # Placeholder, will be calculated at epoch end
                    }

                    # Save checkpoint
                    self._save_checkpoint(
                        step=optim_steps_completed,
                        global_step=global_step,
                        epoch=getattr(self, "_current_epoch", 0),
                        train_metrics=temp_train_metrics,
                    )

                    # Update last checkpoint step
                    self.last_checkpoint_step = global_step

                    self.logger.info(f"✅ Checkpoint saved at step {global_step}")

            if (
                self._test_eval_enabled
                and self._test_eval_frequency == "step"
                and len(self.test_loader) > 0
            ):
                steps_since_last_eval = global_step - self._last_test_eval_step
                if (
                    optimizer_step_performed
                    and self._test_eval_interval > 0
                    and steps_since_last_eval >= self._test_eval_interval
                ):
                    log_only_master_process(
                        self.logger,
                        "info",
                        f"🧪 Running periodic test evaluation at step {global_step}",
                    )
                    self._run_periodic_test_eval(
                        trigger="step", global_step=global_step
                    )
                    self._last_test_eval_step = global_step

        if self.gradient_clip_monitor is not None and is_master_process():
            summary = self.gradient_clip_monitor.get_summary()
            if summary:
                self.logger.info(
                    f"✂️ Gradient clipping summary for epoch {self._current_epoch + 1}:"
                )
                self.logger.info(f"  - Clipping rate: {summary['clip_rate']:.1%}")
                self.logger.info(f"  - Total clips: {summary['total_clips']}")
                self.logger.info(
                    f"  - Max gradient norm: {summary['max_grad_norm']:.4f}"
                )
                self.logger.info(f"  - Avg clip ratio: {summary['avg_clip_ratio']:.1%}")

        # Handle distributed training - aggregate metrics across processes
        if self.is_distributed:
            # Reduce loss across processes
            loss_tensor = torch.tensor(total_loss, device=self.device)
            reduced_loss = reduce_across_processes(loss_tensor)
            total_loss = reduced_loss.item()

        batch_avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = (
            total_classification_loss / max(classification_count, 1)
            if classification_count > 0
            else 0.0
        )
        avg_regression_loss = (
            total_regression_loss / max(regression_count, 1)
            if regression_count > 0
            else 0.0
        )
        avg_aux_region_loss = (
            total_aux_region_loss / max(aux_region_count, 1)
            if aux_region_count > 0
            else 0.0
        )
        avg_aux_exon_loss = (
            total_aux_exon_loss / max(aux_exon_count, 1) if aux_exon_count > 0 else 0.0
        )
        # Dynamic chunker: average chunk rate over the epoch (FR-024 / SC-006).
        # Buffer is populated in `_run_train_batch` only when the chunker is
        # active; otherwise stays empty and the metric is reported as None.
        chunker_rate_buf: list[float] = getattr(self, "_chunker_rate_buf", [])
        avg_mean_chunk_rate: float | None = (
            sum(chunker_rate_buf) / len(chunker_rate_buf) if chunker_rate_buf else None
        )
        # Per-stage means and boundary diagnostics (DD-018 / DD-020). All
        # buffers are aligned: one element per training micro-batch where
        # the chunker emitted aux tensors. Average across batches per stage.
        stage_rate_buf: list[list[float]] = getattr(
            self, "_chunker_stage_rate_buf", []
        )
        per_stage_chunk_rates: list[float] = []
        if stage_rate_buf:
            num_stages = len(stage_rate_buf[0])
            per_stage_chunk_rates = [
                sum(row[s] for row in stage_rate_buf) / len(stage_rate_buf)
                for s in range(num_stages)
            ]
        boundary_loss_buf: list[float] = getattr(self, "_chunker_boundary_loss_buf", [])
        avg_boundary_loss: float | None = (
            sum(boundary_loss_buf) / len(boundary_loss_buf)
            if boundary_loss_buf
            else None
        )
        stage_prob_buf: list[list[float]] = getattr(
            self, "_chunker_stage_prob_buf", []
        )
        per_stage_boundary_probs: list[float] = []
        if stage_prob_buf:
            num_prob_stages = len(stage_prob_buf[0])
            per_stage_boundary_probs = [
                sum(row[s] for row in stage_prob_buf) / len(stage_prob_buf)
                for s in range(num_prob_stages)
            ]
        avg_mean_boundary_prob: float | None = (
            sum(per_stage_boundary_probs) / len(per_stage_boundary_probs)
            if per_stage_boundary_probs
            else None
        )
        phylop_loss_buf: list[float] = getattr(self, "_aux_phylop_loss_buf", [])
        avg_aux_phylop_loss: float | None = (
            sum(phylop_loss_buf) / len(phylop_loss_buf) if phylop_loss_buf else None
        )
        # Reset buffers for next epoch.
        self._chunker_rate_buf = []
        self._chunker_stage_rate_buf = []
        self._chunker_boundary_loss_buf = []
        self._chunker_stage_prob_buf = []
        self._aux_phylop_loss_buf = []

        log_only_master_process(
            self.logger,
            "info",
            f"✅ Training epoch completed - Total Avg. Loss: {batch_avg_loss:.4f}",
        )
        if self.is_classification_enabled and classification_count > 0:
            log_only_master_process(
                self.logger,
                "info",
                f"  📊 Classification Loss: {avg_classification_loss:.4f} "
                f"({classification_count} batches)",
            )
        if self.is_regression_enabled and regression_count > 0:
            log_only_master_process(
                self.logger,
                "info",
                f"  🧪 Regression Loss: {avg_regression_loss:.4f} "
                f"({regression_count} batches)",
            )
        if self.is_aux_region_enabled and aux_region_count > 0:
            log_only_master_process(
                self.logger,
                "info",
                f"  🧬 Aux Region Loss: {avg_aux_region_loss:.4f} "
                f"({aux_region_count} batches)",
            )
        if self.is_aux_exon_enabled and aux_exon_count > 0:
            log_only_master_process(
                self.logger,
                "info",
                f"  🧬 Aux Exon Loss: {avg_aux_exon_loss:.4f} "
                f"({aux_exon_count} batches)",
            )
        if avg_aux_phylop_loss is not None:
            log_only_master_process(
                self.logger,
                "info",
                f"  🧬 Aux PhyloP Loss: {avg_aux_phylop_loss:.4f} "
                f"({len(phylop_loss_buf)} batches)",
            )
        if avg_mean_chunk_rate is not None:
            gap = abs(avg_mean_chunk_rate - self.dynamic_chunker_target_rate)
            log_only_master_process(
                self.logger,
                "info",
                f"  🧩 Dynamic chunker mean rate: "
                f"{avg_mean_chunk_rate:.4f} "
                f"(target={self.dynamic_chunker_target_rate:.4f}, "
                f"gap={gap:.4f}, "
                f"batches={len(chunker_rate_buf)})",
            )

        epoch_metrics: dict[str, float] = {
            "loss": batch_avg_loss,
            "classification_loss": avg_classification_loss,
            "classification_count": classification_count,
            "regression_loss": avg_regression_loss,
            "regression_count": regression_count,
            "aux_region_loss": avg_aux_region_loss,
            "aux_region_count": aux_region_count,
            "aux_exon_loss": avg_aux_exon_loss,
            "aux_exon_count": aux_exon_count,
        }
        if avg_mean_chunk_rate is not None:
            epoch_metrics["mean_chunk_rate"] = avg_mean_chunk_rate
        if avg_boundary_loss is not None:
            epoch_metrics["boundary_loss"] = avg_boundary_loss
        if avg_mean_boundary_prob is not None:
            epoch_metrics["mean_boundary_prob"] = avg_mean_boundary_prob
        if avg_aux_phylop_loss is not None:
            epoch_metrics["aux_phylop_loss"] = avg_aux_phylop_loss
        # Per-stage diagnostics (single-stage runs collapse to one entry,
        # equivalent to the scalar mean but kept for symmetry with multi-
        # stage hierarchies where stage-wise compression matters).
        for stage_idx, rate in enumerate(per_stage_chunk_rates):
            epoch_metrics[f"mean_chunk_rate_stage{stage_idx}"] = rate
        for stage_idx, prob in enumerate(per_stage_boundary_probs):
            epoch_metrics[f"mean_boundary_prob_stage{stage_idx}"] = prob
        return epoch_metrics

    def _run_eval_batch(
        self,
        batch_index: int,
        batch: dict[str, Any],
        name: str,
    ) -> tuple[
        float,
        float | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        float,
        int,
        float,
        int,
        float,
        float,
    ]:
        """
        Run a single evaluation batch: encode, forward, compute losses, and
        return tensors for distributed metrics aggregation.

        Args:
            batch_index (int): Index of the batch within the evaluation loop.
            batch (dict[str, Any]): Batch containing `ref`, and `labels` or
                `soft_labels`, optionally `pos`.
            name (str): Phase name (e.g., "val" or "test") for logging context.

        Returns:
            tuple:
                (
                    classification_loss_value,
                    regression_loss_value (or None if PSI absent),
                    probabilities [B, L, C],
                    labels_for_metrics [B, L, C],
                    soft_labels_for_metrics [B, L, C] or None,
                    nonzero_acceptor_sse,
                    nonzero_acceptor_count,
                    nonzero_donor_sse,
                    nonzero_donor_count,
                    nonzero_acceptor_mae_sum,
                    nonzero_donor_mae_sum,
                )
        """
        ref_seqs = batch["ref"]
        labels_key = "soft_labels" if self.use_soft_labels else "labels"
        original_labels = batch.get(labels_key, batch.get("labels"))
        if original_labels is None:
            raise KeyError(
                (
                    f"Missing labels in batch (expected '{labels_key}{{}}' or 'labels')"
                ).format("")
            )

        # keep_mask filtering (same as training)
        keep_mask, _ = compute_keep_mask_and_stats(
            ref_seqs=ref_seqs,
            original_labels=original_labels,
            seq_length=int(self.cfg.model.seq_length),
            loss_type=getattr(self, "loss_type", ""),
        )
        batch_size = len(ref_seqs)

        # If all samples in the batch are filtered out, skip this batch
        if not torch.any(keep_mask):
            log_only_master_process(
                self.logger,
                "info",
                f"All samples in eval batch {batch_index} were filtered out.",
            )
            return (
                0.0,
                None,
                torch.empty(0),
                torch.empty(0),
                None,
                0.0,
                0,
                0.0,
                0,
                0.0,
                0.0,
            )

        # Apply keep_mask to sequences and labels
        ref_seqs = [seq for i, seq in enumerate(ref_seqs) if keep_mask[i]]
        labels = original_labels[keep_mask].to(self.device)
        features = batch.get("features")
        if features is not None:
            features = features[keep_mask].to(self.device)

        # cfg.model.seq_length 고정 길이 데이터: 클리핑/패딩/마스크 불필요
        if not isinstance(ref_seqs, list):
            ref_seqs = list(ref_seqs)

        # Encode sequences to input_ids and optional logging
        ref_input_ids, attention_mask = self._encode_and_maybe_log(
            ref_seqs=ref_seqs, phase=name, batch_index=batch_index
        )

        # Forward pass once; get both classification and regression outputs
        cls_logits, reg_outputs, aux_outputs = self._forward_logits(
            ref_input_ids, attention_mask, features=features
        )

        # Compute classification loss (if enabled)
        classification_loss_value: float = 0.0
        probabilities: torch.Tensor | None = None
        labels_for_metrics: torch.Tensor | None = None
        soft_labels_for_metrics: torch.Tensor | None = None

        if self.is_classification_enabled:
            # Handle labels shape - ensure it's (batch_size, 3, seq_len)
            labels = self._normalize_label_shape(labels)

            # Check and ensure sequence lengths match (first batch only for logging)
            batch_size, seq_len, num_classes = cls_logits.shape
            original_label_seq_len = labels.shape[2]
            if is_master_process() and batch_index == 0:
                self.logger.info(f"🔍 Sequence length check ({name} validation):")
                msg_shape = (
                    f"  - Logits shape: {cls_logits.shape} (B={batch_size}, "
                    f"seq_len={seq_len}, n_classes={num_classes})"
                )
                self.logger.info(msg_shape)
                self.logger.info(
                    f"  - Labels shape: {labels.shape} "
                    f"(original_seq_len={original_label_seq_len})"
                )

            # Align logits and labels via center-crop for metrics consistency
            cls_logits, labels = self._align_logits_and_labels(
                cls_logits, labels, phase=name
            )

            # Apply hard label conversion if enabled
            if self.uses_hard_labels_classification:
                labels = self._convert_to_hard_labels_classification(labels)

            # Prepare soft labels for top-k metrics when available
            try:
                soft_labels_full = batch.get("soft_labels")
                if soft_labels_full is not None:
                    soft_labels = soft_labels_full[keep_mask].to(self.device)
                    # Align to classification logits length using same helper
                    _, soft_labels = self._align_logits_and_labels(
                        cls_logits, soft_labels, phase=name
                    )
                    soft_labels_for_metrics = soft_labels.transpose(1, 2).contiguous()
            except Exception as e:
                log_only_master_process(
                    self.logger,
                    "warning",
                    f"Soft labels alignment skipped due to error: {e}",
                )

            # Calculate classification loss (multi-label)
            loss = self._compute_masked_loss(
                logits=cls_logits,
                labels=labels,
                labels_dtype=torch.float32,
            )
            classification_loss_value = float(loss.item())

            # Compute probabilities and labels for metrics
            if getattr(self, "loss_type", "") == "multi_class_focal_loss":
                probabilities = torch.softmax(cls_logits, dim=-1)
            else:
                probabilities = torch.sigmoid(cls_logits)
            labels_for_metrics = labels.transpose(1, 2).contiguous()  # [B, L, C]

        # Optionally compute regression loss if PSI targets are present
        regression_loss_value: float | None = None
        nonzero_acceptor_sse: float = 0.0
        nonzero_acceptor_count: int = 0
        nonzero_donor_sse: float = 0.0
        nonzero_donor_count: int = 0
        nonzero_acceptor_mae_sum: float = 0.0
        nonzero_donor_mae_sum: float = 0.0
        donor_psi_full = batch.get("donor_psi")
        acceptor_psi_full = batch.get("acceptor_psi")
        if (
            self.is_regression_enabled
            and donor_psi_full is not None
            and acceptor_psi_full is not None
        ):
            try:
                # Apply keep_mask to PSI targets to match filtered batch
                donor_psi = donor_psi_full[keep_mask].to(self.device).float()
                acceptor_psi = acceptor_psi_full[keep_mask].to(self.device).float()

                # Targets: [B, L, 2] with channel order [acceptor, donor]
                psi_targets = torch.stack([acceptor_psi, donor_psi], dim=-1)

                # Align regression outputs and targets to classification length
                if reg_outputs is None:
                    raise RuntimeError("Regression outputs are None from model forward")
                reg_outputs, psi_targets = self._align_regression_and_targets(
                    reg_outputs, psi_targets, ref_len=cls_logits.shape[1], phase=name
                )

                # Apply hard label conversion if enabled for regression
                if self.uses_hard_labels_regression:
                    psi_targets = self._convert_to_hard_labels_regression(psi_targets)

                regression_loss = self.regression_criterion(
                    reg_outputs.float(), psi_targets
                )
                regression_loss_value = float(regression_loss.item())

                # Compute SSE/MAE and counts for non-zero acceptor/donor positions
                errors_sq = (reg_outputs.float() - psi_targets) ** 2
                errors_abs = torch.abs(reg_outputs.float() - psi_targets)
                acc_mask = psi_targets[..., 0] != 0
                don_mask = psi_targets[..., 1] != 0

                if acc_mask.any():
                    nonzero_acceptor_sse = errors_sq[..., 0][acc_mask].sum().item()
                    nonzero_acceptor_mae_sum = errors_abs[..., 0][acc_mask].sum().item()
                    nonzero_acceptor_count = int(acc_mask.sum().item())
                if don_mask.any():
                    nonzero_donor_sse = errors_sq[..., 1][don_mask].sum().item()
                    nonzero_donor_mae_sum = errors_abs[..., 1][don_mask].sum().item()
                    nonzero_donor_count = int(don_mask.sum().item())
            except Exception as e:
                log_only_master_process(
                    self.logger,
                    "warning",
                    f"PSI regression loss (eval) skipped due to error: {e}",
                )

        # Auxiliary region/exon validation losses (no backward).
        aux_region_loss_value: float | None = None
        aux_exon_loss_value: float | None = None
        if self.is_aux_region_enabled or self.is_aux_exon_enabled:
            ref_seq_len = (
                int(cls_logits.shape[1])
                if self.is_classification_enabled
                else (
                    int(aux_outputs["region"].shape[1])
                    if "region" in aux_outputs
                    else int(aux_outputs.get("exon", torch.empty(0, 0)).shape[1])
                )
            )
            if ref_seq_len > 0:
                with torch.no_grad():
                    aux_region_t, aux_exon_t = self._compute_aux_losses(
                        aux_outputs=aux_outputs,
                        batch=batch,
                        keep_mask=keep_mask,
                        ref_seq_len=ref_seq_len,
                    )
                if aux_region_t is not None:
                    aux_region_loss_value = float(aux_region_t.item())
                if aux_exon_t is not None:
                    aux_exon_loss_value = float(aux_exon_t.item())

        return (
            classification_loss_value,
            regression_loss_value,
            probabilities,
            labels_for_metrics,
            soft_labels_for_metrics,
            nonzero_acceptor_sse,
            nonzero_acceptor_count,
            nonzero_donor_sse,
            nonzero_donor_count,
            nonzero_acceptor_mae_sum,
            nonzero_donor_mae_sum,
            aux_region_loss_value,
            aux_exon_loss_value,
        )

    def _gather_tensors_to_master(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Gather tensors from all GPUs to master process only.

        This method collects tensors from all processes in distributed training
        and concatenates them along dimension 0 on the master process only.
        Other processes receive None. Handles variable-sized tensors across
        different GPUs.

        Args:
            tensor (torch.Tensor): Local tensor to gather from this process.

        Returns:
            torch.Tensor | None: Concatenated tensor on master process,
                None on other processes.
        """
        if not self.is_distributed:
            return tensor

        # Ensure tensor is on GPU for NCCL backend
        tensor = tensor.to(self.device)

        # Get local tensor size
        local_size = torch.tensor(
            [tensor.shape[0]], device=self.device, dtype=torch.long
        )

        # Gather sizes from all processes (all processes need this for padding)
        all_sizes = [
            torch.zeros(1, dtype=torch.long, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_sizes, local_size)

        # Determine max size for padding
        max_size = max(s.item() for s in all_sizes)

        # Pad local tensor to max size if needed
        if tensor.shape[0] < max_size:
            pad_size = max_size - tensor.shape[0]
            padding_shape = (pad_size,) + tensor.shape[1:]
            padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=self.device)
            tensor_padded = torch.cat([tensor, padding], dim=0)
        else:
            tensor_padded = tensor

        # Gather to master only (rank 0)
        if is_master_process():
            gathered_tensors = [
                torch.zeros_like(tensor_padded) for _ in range(self.world_size)
            ]
        else:
            gathered_tensors = None

        dist.gather(tensor_padded, gather_list=gathered_tensors, dst=0)

        # Only master process concatenates and returns
        if is_master_process() and gathered_tensors is not None:
            # Move gathered tensors to CPU immediately to free GPU memory
            cpu_tensors = []
            for t in gathered_tensors:
                cpu_tensors.append(t.cpu())

            # Explicitly delete GPU tensors
            del gathered_tensors
            torch.cuda.empty_cache()
            result_tensors = []
            for i, gathered in enumerate(cpu_tensors):
                actual_size = int(all_sizes[i].item())
                result_tensors.append(gathered[:actual_size])
            return torch.cat(result_tensors, dim=0)

        return None

    def _select_local_topk_pairs_cpu(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        *,
        k: int,
        chunk_size: int,
    ) -> ScoreLabelPairs:
        """Select local top-k (score, label) pairs on CPU in a streaming manner.
        Args:
            scores (torch.Tensor): Score tensor (any shape) on the current device.
                Expected values are probabilities in [0, 1].
            labels (torch.Tensor): Label tensor with the same shape as `scores`.
                Labels can be hard (0/1) or soft in [0, 1].
            k (int): Number of top scores to keep.
            chunk_size (int): Number of flattened elements to process per chunk.
        Returns:
            ScoreLabelPairs: CPU container with pairs tensor of shape [N, 2] where
                N = min(k, scores.numel()).
        """
        k_int = int(k)
        if k_int <= 0 or scores.numel() == 0:
            empty = torch.empty((0, 2), dtype=torch.float32, device="cpu")
            return ScoreLabelPairs(empty)

        scores_flat = scores.reshape(-1)
        labels_flat = labels.reshape(-1)
        numel = int(scores_flat.numel())
        k_eff = min(k_int, numel)

        top_scores = torch.empty((0,), dtype=torch.float32, device="cpu")
        top_labels = torch.empty((0,), dtype=torch.float32, device="cpu")

        for start in range(0, numel, int(chunk_size)):
            end = min(start + int(chunk_size), numel)
            chunk_scores = scores_flat[start:end].detach().float().cpu()
            chunk_labels = labels_flat[start:end].detach().cpu()
            chunk_pos = (chunk_labels > 0.5).to(dtype=torch.float32)

            k_chunk = min(k_eff, int(chunk_scores.numel()))
            if k_chunk <= 0:
                continue

            cand_scores, cand_idx = torch.topk(
                chunk_scores, k=k_chunk, largest=True, sorted=False
            )
            cand_labels = chunk_pos[cand_idx]

            if top_scores.numel() == 0:
                top_scores = cand_scores
                top_labels = cand_labels
                continue

            merged_scores = torch.cat([top_scores, cand_scores], dim=0)
            merged_labels = torch.cat([top_labels, cand_labels], dim=0)
            if merged_scores.numel() > k_eff:
                new_scores, new_idx = torch.topk(
                    merged_scores, k=k_eff, largest=True, sorted=False
                )
                top_scores = new_scores
                top_labels = merged_labels[new_idx]
            else:
                top_scores = merged_scores
                top_labels = merged_labels

        pairs_cpu = torch.stack([top_scores, top_labels], dim=1)
        return ScoreLabelPairs(pairs_cpu)

    def _gather_score_label_pairs_to_master(
        self, local_pairs: ScoreLabelPairs
    ) -> ScoreLabelPairs | None:
        """Gather (score, label) pairs to master process only.
        Args:
            local_pairs (ScoreLabelPairs): Local pairs container.
        Returns:
            ScoreLabelPairs | None: Gathered pairs on CPU for master, None otherwise.
        """
        if not self.is_distributed:
            return local_pairs.to(torch.device("cpu"))

        gathered = self._gather_tensors_to_master(local_pairs.pairs.to(self.device))
        if is_master_process() and gathered is not None:
            return ScoreLabelPairs(gathered.detach().cpu())

        return None

    def _compute_topk_accuracy_from_pairs_cpu(
        self,
        gathered_pairs: ScoreLabelPairs,
        *,
        k: int,
    ) -> float:
        """Compute top-k accuracy from gathered (score, label) pairs using a heap.
        Args:
            gathered_pairs (ScoreLabelPairs): Pairs container (expected on CPU).
            k (int): Global k for top-k accuracy (number of positive labels).
        Returns:
            float: Top-k accuracy in [0, 1].
        """
        k_int = int(k)
        if k_int <= 0:
            return 0.0

        pairs = gathered_pairs.pairs
        if pairs.numel() == 0:
            return 0.0
        if pairs.device.type != "cpu":
            pairs = pairs.cpu()

        heap: list[tuple[float, int]] = []
        n_rows = int(pairs.shape[0])
        for i in range(n_rows):
            score = float(pairs[i, 0].item())
            label = 1 if float(pairs[i, 1].item()) > 0.5 else 0
            if len(heap) < k_int:
                heapq.heappush(heap, (score, label))
                continue
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, label))

        hits = int(sum(lbl for _, lbl in heap))
        return float(hits / k_int)

    def validate(
        self,
        loader: DataLoader,
        name: str = "val",
        *,
        thresholds: tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        """
        Validate model on given loader.

        Args:
            loader (DataLoader): DataLoader for the evaluation split.
            name (str): Phase name for logging context (e.g., "val", "test").
            thresholds (tuple[float, float, float] | None): Optional explicit per-
                class thresholds in the order (acceptor, donor, neither). When
                provided, these override both dynamic thresholds and the default 0.5
                thresholding scheme.

        Returns:
            dict[str, float]: Aggregated evaluation metrics for the given split.
        """
        self.logger.info(f"🔍 Starting {name} validation...")
        self.model.eval()
        # cuFFT plan cache + allocator fragmentation accumulate across training
        # epochs and can trigger CUFFT_INTERNAL_ERROR inside HyenaDNA fftconv.
        # Reset both before validation to keep long runs stable.
        if torch.cuda.is_available():
            torch.backends.cuda.cufft_plan_cache.clear()
            torch.cuda.empty_cache()
        total_loss = 0

        # Only initialize metrics calculator if classification is enabled
        metrics_calculator = None
        if self.is_classification_enabled:
            metrics_calculator = OnlineMetricsCalculator(
                logger=self.logger,
                num_classes=3,
                use_histogram=self.cfg.training.evaluation.use_histogram,
            )

        total_batches = len(loader)
        self.logger.info(
            f"📊 Validation ({name}): {total_batches} batches, "
            f"{len(loader.dataset)} samples"
        )

        all_probs_local = []
        all_labels_local = []
        all_soft_labels_local = []

        with torch.no_grad():
            total_regression_loss = 0.0
            regression_batches = 0

            # Aggregators for non-zero PSI MSE and MAE
            nz_acc_sse_total = 0.0
            nz_acc_count_total = 0
            nz_don_sse_total = 0.0
            nz_don_count_total = 0
            nz_acc_mae_total = 0.0
            nz_don_mae_total = 0.0
            total_aux_region_loss = 0.0
            aux_region_batches = 0
            total_aux_exon_loss = 0.0
            aux_exon_batches = 0

            for batch_idx, batch in enumerate(
                tqdm(loader, desc=f"Evaluating {name}", disable=False)
            ):
                (
                    loss_value,
                    regression_loss_value,
                    probs_local,
                    labels_local,
                    soft_labels_local,
                    nonzero_acceptor_sse,
                    nonzero_acceptor_count,
                    nonzero_donor_sse,
                    nonzero_donor_count,
                    nonzero_acceptor_mae_sum,
                    nonzero_donor_mae_sum,
                    aux_region_loss_value,
                    aux_exon_loss_value,
                ) = self._run_eval_batch(
                    batch_index=batch_idx,
                    batch=batch,
                    name=name,
                )
                total_loss += loss_value

                if regression_loss_value is not None:
                    total_regression_loss += regression_loss_value
                    regression_batches += 1

                if aux_region_loss_value is not None:
                    total_aux_region_loss += aux_region_loss_value
                    aux_region_batches += 1
                if aux_exon_loss_value is not None:
                    total_aux_exon_loss += aux_exon_loss_value
                    aux_exon_batches += 1

                # Accumulate non-zero PSI SSE/MAE/counts
                nz_acc_sse_total += nonzero_acceptor_sse
                nz_acc_count_total += nonzero_acceptor_count
                nz_don_sse_total += nonzero_donor_sse
                nz_don_count_total += nonzero_donor_count
                nz_acc_mae_total += nonzero_acceptor_mae_sum
                nz_don_mae_total += nonzero_donor_mae_sum

                # Only accumulate classification results if classification is enabled
                if self.is_classification_enabled:
                    if probs_local is not None:
                        all_probs_local.append(probs_local.detach().cpu())
                    if labels_local is not None:
                        all_labels_local.append(labels_local.detach().cpu())
                    if soft_labels_local is not None and soft_labels_local.numel() > 0:
                        all_soft_labels_local.append(soft_labels_local.detach().cpu())

        # Distributed: aggregate scalar metrics across all processes
        if self.is_distributed:
            # Aggregate loss and regression metrics
            scalars_to_reduce = torch.tensor(
                [
                    total_loss,
                    total_regression_loss,
                    float(regression_batches),
                    nz_acc_sse_total,
                    float(nz_acc_count_total),
                    nz_don_sse_total,
                    float(nz_don_count_total),
                    nz_acc_mae_total,
                    nz_don_mae_total,
                    total_aux_region_loss,
                    float(aux_region_batches),
                    total_aux_exon_loss,
                    float(aux_exon_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(scalars_to_reduce, op=dist.ReduceOp.SUM)

            total_loss = scalars_to_reduce[0].item()
            total_regression_loss = scalars_to_reduce[1].item()
            regression_batches = int(scalars_to_reduce[2].item())
            nz_acc_sse_total = scalars_to_reduce[3].item()
            nz_acc_count_total = int(scalars_to_reduce[4].item())
            nz_don_sse_total = scalars_to_reduce[5].item()
            nz_don_count_total = int(scalars_to_reduce[6].item())
            nz_acc_mae_total = scalars_to_reduce[7].item()
            nz_don_mae_total = scalars_to_reduce[8].item()
            total_aux_region_loss = scalars_to_reduce[9].item()
            aux_region_batches = int(scalars_to_reduce[10].item())
            total_aux_exon_loss = scalars_to_reduce[11].item()
            aux_exon_batches = int(scalars_to_reduce[12].item())

        # Initialize default metrics dict
        metrics: dict[str, float] = {}

        # Only compute classification metrics if classification is enabled
        if self.is_classification_enabled and len(all_probs_local) > 0:
            all_probs_concatenated = torch.cat(all_probs_local, dim=0)
            all_labels_concatenated = torch.cat(all_labels_local, dim=0)

            # Release list memory
            all_probs_local = []
            all_labels_local = []
            all_soft_labels_concatenated: torch.Tensor | None = None
            if len(all_soft_labels_local) > 0:
                all_soft_labels_concatenated = torch.cat(all_soft_labels_local, dim=0)
                all_soft_labels_local = []

            # Distributed: gather probs/labels to master process only
            # Only gather acceptor/donor (first 2 classes) for AUPRC and top-k
            all_probs_acc_don = None
            all_labels_acc_don = None
            all_soft_labels_acc_don = None
            # Compute dataset-level top-k accuracy (acceptor/donor only).
            # In distributed mode, the final computation is done on master using
            # only per-rank top-k candidates to avoid OOM.
            topk_accuracy_acceptor: float = 0.0
            topk_accuracy_donor: float = 0.0
            topk_k_acceptor: int = 0
            topk_k_donor: int = 0
            if self.is_distributed:
                # Extract acceptor/donor only (indices 0, 1) for memory efficiency
                probs_acc_don_local = all_probs_concatenated[..., :2].contiguous()
                labels_acc_don_local = all_labels_concatenated[..., :2].contiguous()
                soft_labels_acc_don_local = (
                    all_soft_labels_concatenated[..., :2].contiguous()
                    if all_soft_labels_concatenated is not None
                    else None
                )

                # -----------------------------------------------------------
                # [DDP Top-k] Gather only per-rank candidates (OOM-safe)
                # -----------------------------------------------------------
                labels_for_topk_local = (
                    soft_labels_acc_don_local
                    if self.use_soft_labels and soft_labels_acc_don_local is not None
                    else labels_acc_don_local
                )

                acc_pos_local = (labels_for_topk_local[..., 0] > 0.5).sum()
                don_pos_local = (labels_for_topk_local[..., 1] > 0.5).sum()
                pos_counts = torch.stack([acc_pos_local, don_pos_local]).to(
                    device=self.device, dtype=torch.long
                )
                dist.all_reduce(pos_counts, op=dist.ReduceOp.SUM)
                topk_k_acceptor = int(pos_counts[0].item())
                topk_k_donor = int(pos_counts[1].item())

                topk_chunk_size = 1_000_000
                if topk_k_acceptor > 0:
                    acc_pairs_local = self._select_local_topk_pairs_cpu(
                        scores=probs_acc_don_local[..., 0],
                        labels=labels_for_topk_local[..., 0],
                        k=topk_k_acceptor,
                        chunk_size=topk_chunk_size,
                    )
                    acc_pairs_gathered = self._gather_score_label_pairs_to_master(
                        acc_pairs_local
                    )
                    if is_master_process() and acc_pairs_gathered is not None:
                        topk_accuracy_acceptor = (
                            self._compute_topk_accuracy_from_pairs_cpu(
                                acc_pairs_gathered,
                                k=topk_k_acceptor,
                            )
                        )

                if topk_k_donor > 0:
                    don_pairs_local = self._select_local_topk_pairs_cpu(
                        scores=probs_acc_don_local[..., 1],
                        labels=labels_for_topk_local[..., 1],
                        k=topk_k_donor,
                        chunk_size=topk_chunk_size,
                    )
                    don_pairs_gathered = self._gather_score_label_pairs_to_master(
                        don_pairs_local
                    )
                    if is_master_process() and don_pairs_gathered is not None:
                        topk_accuracy_donor = (
                            self._compute_topk_accuracy_from_pairs_cpu(
                                don_pairs_gathered,
                                k=topk_k_donor,
                            )
                        )

                # Gather to master only (saves memory on non-master GPUs)
                probs_acc_don_gathered = self._gather_tensors_to_master(
                    probs_acc_don_local
                )
                labels_acc_don_gathered = self._gather_tensors_to_master(
                    labels_acc_don_local
                )
                soft_labels_acc_don_gathered: torch.Tensor | None = None
                if soft_labels_acc_don_local is not None:
                    soft_labels_acc_don_gathered = self._gather_tensors_to_master(
                        soft_labels_acc_don_local
                    )

                # Non-master processes: skip detailed classification metrics
                if is_master_process():
                    all_probs_acc_don = probs_acc_don_gathered
                    all_labels_acc_don = labels_acc_don_gathered
                    all_soft_labels_acc_don = soft_labels_acc_don_gathered
            else:
                # Non-distributed: use local data directly
                all_probs_acc_don = all_probs_concatenated[..., :2]
                all_labels_acc_don = all_labels_concatenated[..., :2]
                all_soft_labels_acc_don = (
                    all_soft_labels_concatenated[..., :2]
                    if all_soft_labels_concatenated is not None
                    else None
                )

            # Skip classification metrics computation for non-master in distributed
            if self.is_distributed and not is_master_process():
                # Non-master: skip to loss computation below
                pass
            elif all_probs_acc_don is not None and all_labels_acc_don is not None:
                if self.use_soft_labels and all_soft_labels_acc_don is not None:
                    labels_for_metrics = all_soft_labels_acc_don
                else:
                    labels_for_metrics = all_labels_acc_don

                if not self.is_distributed:
                    # Non-distributed: compute top-k directly from full tensors.
                    try:
                        acc_scores = all_probs_acc_don[..., 0].reshape(-1)
                        don_scores = all_probs_acc_don[..., 1].reshape(-1)

                        acc_labels_flat = labels_for_metrics[..., 0].reshape(-1)
                        don_labels_flat = labels_for_metrics[..., 1].reshape(-1)

                        acc_pos_mask = acc_labels_flat > 0.5
                        don_pos_mask = don_labels_flat > 0.5

                        topk_k_acceptor = int(acc_pos_mask.sum().item())
                        topk_k_donor = int(don_pos_mask.sum().item())

                        if topk_k_acceptor > 0:
                            acc_topk_idx = torch.topk(
                                acc_scores, topk_k_acceptor, largest=True
                            ).indices
                            acc_hits = acc_pos_mask[acc_topk_idx].float().sum().item()
                            topk_accuracy_acceptor = float(acc_hits / topk_k_acceptor)
                        if topk_k_donor > 0:
                            don_topk_idx = torch.topk(
                                don_scores, topk_k_donor, largest=True
                            ).indices
                            don_hits = don_pos_mask[don_topk_idx].float().sum().item()
                            topk_accuracy_donor = float(don_hits / topk_k_donor)
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️  Failed to compute top-k accuracy for {name}: {e}"
                        )

                # Store for AUC calculation (use same labels_for_metrics)
                probs_numpy = safe_tensor_to_numpy(all_probs_acc_don)
                labels_numpy = safe_tensor_to_numpy(labels_for_metrics)
                metrics_calculator.update_auc_data(
                    labels=labels_numpy, probs=probs_numpy
                )

            # -----------------------------------------------------------
            # [Logic] Threshold & Preds Selection
            # -----------------------------------------------------------
            # Thresholding and additional metrics (master process only in distributed)
            # Non-master processes skip this block as they don't have gathered data
            if not (self.is_distributed and not is_master_process()):
                # Thresholding for acceptor/donor (2 classes)
                # 1) Explicit thresholds override everything
                if thresholds is not None:
                    try:
                        thr_list = [float(x) for x in list(thresholds)[:2]]
                    except Exception:
                        thr_list = [
                            float(thresholds[0]),
                            float(thresholds[1]),
                        ]
                    log_only_master_process(
                        self.logger,
                        "info",
                        "🔧 Using explicit thresholds(override): ",
                        f"acceptor={thr_list[0]:.4f}, donor={thr_list[1]:.4f}",
                    )
                    preds_tensor = torch.zeros_like(all_probs_acc_don)
                    for class_idx in range(2):
                        thr = thr_list[class_idx] if class_idx < len(thr_list) else 0.5
                        preds_tensor[..., class_idx] = (
                            all_probs_acc_don[..., class_idx] >= float(thr)
                        ).float()
                else:
                    # Check if using multi-class focal loss (uses argmax, not threshold)
                    is_multiclass = (
                        getattr(self, "loss_type", "") == "multi_class_focal_loss"
                    )

                    if is_multiclass:
                        # Multi-class focal loss: use argmax (no threshold needed)
                        pred_indices = torch.argmax(all_probs_acc_don, dim=-1)
                        preds_full = torch.nn.functional.one_hot(
                            pred_indices,
                            num_classes=all_probs_acc_don.shape[-1],
                        ).float()
                        preds_tensor = preds_full[..., :2]
                    # 2) Reuse tuned thresholds computed during validation for test
                    if (
                        name == "test"
                        and getattr(self, "tuned_thresholds", None) is not None
                    ):
                        thr_list = [float(x) for x in self.tuned_thresholds[:2]]
                        if is_master_process():
                            self.logger.info(
                                "🔁 Using cached tuned thresholds for test: "
                                f"acceptor={thr_list[0]:.4f}, donor={thr_list[1]:.4f}"
                            )
                        preds_tensor = torch.zeros_like(all_probs_acc_don)
                        for class_idx in range(2):
                            thr = (
                                thr_list[class_idx]
                                if class_idx < len(thr_list)
                                else 0.5
                            )
                            preds_tensor[..., class_idx] = (
                                all_probs_acc_don[..., class_idx] >= float(thr)
                            ).float()
                    else:
                        # 3) Dynamic thresholds: compute only during validation
                        threshold_cfg = self.cfg.training.evaluation.dynamic_threshold
                        if threshold_cfg.enabled and name == "val":
                            tuned_thresholds = (
                                metrics_calculator.compute_optimal_thresholds(
                                    objective=threshold_cfg.objective,
                                    target_precision=list(
                                        threshold_cfg.target_precision
                                    )[:2],
                                    target_recall=list(threshold_cfg.target_recall)[:2],
                                )
                            )
                            # Cache as python list[float] (only 2 classes)
                            try:
                                self.tuned_thresholds = [
                                    float(t) for t in list(tuned_thresholds)[:2]
                                ]
                            except Exception:
                                self.tuned_thresholds = [
                                    float(tuned_thresholds[0]),
                                    float(tuned_thresholds[1]),
                                ]
                            log_only_master_process(
                                self.logger,
                                "info",
                                f"💾 Cached tuned thresholds from validation: "
                                f"acceptor={self.tuned_thresholds[0]:.4f}, "
                                f"donor={self.tuned_thresholds[1]:.4f}",
                            )
                            preds_tensor = torch.zeros_like(all_probs_acc_don)
                            for class_idx in range(2):
                                threshold = self.tuned_thresholds[class_idx]
                                preds_tensor[..., class_idx] = (
                                    all_probs_acc_don[..., class_idx] >= threshold
                                ).float()
                        else:
                            # 4)[Fallback] Default thresholding
                            if (
                                self.cfg.model.get("loss_type", "")
                                == "multi_class_focal_loss"
                            ):
                                # Multi-class: Argmax
                                pred_indices = torch.argmax(all_probs_acc_don, dim=-1)
                                preds_full = torch.nn.functional.one_hot(
                                    pred_indices,
                                    num_classes=all_probs_acc_don.shape[-1],
                                ).float()
                                preds_tensor = preds_full[..., :2]
                            else:
                                # 4)[Fallback] Default thresholding (multi-label: 0.5)
                                preds_tensor = (all_probs_acc_don >= 0.5).float()

                preds_numpy = safe_tensor_to_numpy(preds_tensor)
                metrics_calculator.update(preds=preds_numpy, labels=labels_numpy)

                # Release CPU memory for large tensors after use
                del preds_tensor
                del preds_numpy
                del labels_numpy
                del probs_numpy
                if "all_probs_acc_don" in locals():
                    del all_probs_acc_don
                if "all_labels_acc_don" in locals():
                    del all_labels_acc_don
                if "all_soft_labels_acc_don" in locals():
                    del all_soft_labels_acc_don
                if "all_probs_concatenated" in locals():
                    del all_probs_concatenated
                if "all_labels_concatenated" in locals():
                    del all_labels_concatenated
                if "all_soft_labels_concatenated" in locals():
                    del all_soft_labels_concatenated
                gc.collect()

                # -----------------------------------------------------------
                # [Metrics]
                # -----------------------------------------------------------
                # Calculate metrics
                metrics = metrics_calculator.compute_metrics()

                # Attach top-k accuracy metrics
                metrics["topk_accuracy_acceptor"] = topk_accuracy_acceptor
                metrics["topk_accuracy_donor"] = topk_accuracy_donor
                # Mean across acceptor/donor — preferred early-stopping monitor
                # because it is positive-only and robust to class imbalance.
                metrics["topk_accuracy_mean"] = 0.5 * (
                    topk_accuracy_acceptor + topk_accuracy_donor
                )
                metrics["topk_k_acceptor"] = float(topk_k_acceptor)
                metrics["topk_k_donor"] = float(topk_k_donor)

                # Attach tuned thresholds for MLflow logging (validation only)
                if name == "val":
                    try:
                        threshold_cfg = self.cfg.training.evaluation.dynamic_threshold
                        if threshold_cfg.enabled and (
                            getattr(self, "tuned_thresholds", None) is not None
                        ):
                            try:
                                thr_acc = float(self.tuned_thresholds[0])
                                thr_don = float(self.tuned_thresholds[1])
                                metrics["threshold_acceptor"] = thr_acc
                                metrics["threshold_donor"] = thr_don
                            except (IndexError, ValueError, TypeError):
                                pass
                    except AttributeError:
                        pass

                #  Optional detailed logging for validation phase (master only)
                if (
                    name == "val"
                    and self.is_classification_enabled
                    and is_master_process()
                ):
                    acc_topk = metrics.get("topk_accuracy_acceptor", 0.0)
                    don_topk = metrics.get("topk_accuracy_donor", 0.0)
                    acc_p = metrics.get("precision_acceptor", 0.0)
                    don_p = metrics.get("precision_donor", 0.0)
                    acc_r = metrics.get("recall_acceptor", 0.0)
                    don_r = metrics.get("recall_donor", 0.0)
                    acc_auc = metrics.get("auc_pr_acceptor", 0.0)
                    don_auc = metrics.get("auc_pr_donor", 0.0)

                    log_lines = []
                    log_lines.append(f"\n{'=' * 60}")
                    report_line = (
                        "🔍 Validation Metrics Report "
                        f"({metrics_calculator.total_samples} samples)"
                    )
                    log_lines.append(report_line)
                    log_lines.append(f"{'-' * 60}")
                    # Header: Metric | Acceptor | Donor
                    log_lines.append(
                        f"{'Metric':<25} | {'Acceptor':<12} | {'Donor':<12}"
                    )
                    log_lines.append(f"{'-' * 60}")

                    # 1. Default Performance (Default/Argmax)
                    log_lines.append(
                        f"{'Top-k Accuracy':<25} | {acc_topk:<12.4f} | "
                        + f"{don_topk:<12.4f}"
                    )
                    log_lines.append(
                        f"{'Precision':<25} | {acc_p:<12.4f} | {don_p:<12.4f}"
                    )
                    log_lines.append(
                        f"{'Recall':<25} | {acc_r:<12.4f} | {don_r:<12.4f}"
                    )
                    log_lines.append(
                        f"{'AUC-PR':<25} | {acc_auc:<12.4f} | {don_auc:<12.4f}"
                    )
                    log_lines.append(f"{'=' * 60}")

                    self.logger.info("\n".join(log_lines))

        # Compute final loss and regression metrics
        # In distributed mode, total_loss is already summed across all GPUs,
        # so divide by total number of batches across all processes
        if self.is_distributed:
            total_batches_all = len(loader) * self.world_size
        else:
            total_batches_all = len(loader)
        if total_batches_all > 0:
            metrics["loss"] = total_loss / total_batches_all
            if regression_batches > 0:
                metrics["regression_loss"] = total_regression_loss / regression_batches
            # Non-zero PSI MSEs and MAEs
            if nz_acc_count_total > 0:
                metrics["regression_mse_acceptor_nonzero"] = (
                    nz_acc_sse_total / nz_acc_count_total
                )
                metrics["regression_mae_acceptor_nonzero"] = (
                    nz_acc_mae_total / nz_acc_count_total
                )
            if nz_don_count_total > 0:
                metrics["regression_mse_donor_nonzero"] = (
                    nz_don_sse_total / nz_don_count_total
                )
                metrics["regression_mae_donor_nonzero"] = (
                    nz_don_mae_total / nz_don_count_total
                )
            if aux_region_batches > 0:
                metrics["aux_region_loss"] = total_aux_region_loss / aux_region_batches
            if aux_exon_batches > 0:
                metrics["aux_exon_loss"] = total_aux_exon_loss / aux_exon_batches
        else:
            metrics["loss"] = 0.0

        # Build log message based on enabled heads
        log_msg_parts = [f"✅ Validation ({name}) completed"]
        log_msg_parts.append(f"Loss: {metrics.get('loss', 0.0):.4f}")
        if self.is_classification_enabled and "accuracy" in metrics:
            log_msg_parts.append(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        if self.is_regression_enabled and "regression_loss" in metrics:
            log_msg_parts.append(
                f"Regression Loss: {metrics.get('regression_loss', 0.0):.4f}"
            )
        if self.is_aux_region_enabled and "aux_region_loss" in metrics:
            log_msg_parts.append(
                f"Aux Region Loss: {metrics.get('aux_region_loss', 0.0):.4f}"
            )
        if self.is_aux_exon_enabled and "aux_exon_loss" in metrics:
            log_msg_parts.append(
                f"Aux Exon Loss: {metrics.get('aux_exon_loss', 0.0):.4f}"
            )
        self.logger.info(" - ".join(log_msg_parts))

        # Log detailed regression metrics if available
        if self.is_regression_enabled and is_master_process():
            if "regression_mae_acceptor_nonzero" in metrics:
                self.logger.info(
                    f"  🧪 Acceptor MAE (non-zero): "
                    f"{metrics['regression_mae_acceptor_nonzero']:.4f}"
                )
            if "regression_mae_donor_nonzero" in metrics:
                self.logger.info(
                    f"  🧪 Donor MAE (non-zero): "
                    f"{metrics['regression_mae_donor_nonzero']:.4f}"
                )
            if "regression_mse_acceptor_nonzero" in metrics:
                self.logger.info(
                    f"  🧪 Acceptor MSE (non-zero): "
                    f"{metrics['regression_mse_acceptor_nonzero']:.4f}"
                )
            if "regression_mse_donor_nonzero" in metrics:
                self.logger.info(
                    f"  🧪 Donor MSE (non-zero): "
                    f"{metrics['regression_mse_donor_nonzero']:.4f}"
                )

        # Release metrics calculator memory
        if metrics_calculator is not None:
            del metrics_calculator
        gc.collect()

        return metrics

    def _get_sampled_test_loader(
        self, base_loader: DataLoader, proportion: float, *, seed: int | None = None
    ) -> DataLoader:
        """
        Build a sampled DataLoader for periodic test evaluation.

        Args:
            base_loader (DataLoader): Original test loader.
            proportion (float): Sampling ratio in [0, 1].
            seed (int | None): Optional seed for reproducibility.

        Returns:
            DataLoader: Loader serving the sampled subset.
        """
        proportion = min(max(proportion, 0.0), 1.0)
        dataset = base_loader.dataset
        total_size = len(dataset)
        if proportion >= 0.999 or total_size <= 1:
            return base_loader

        sample_size = max(1, int(total_size * proportion))
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else 42)
        indices = (
            torch.randperm(total_size, generator=generator)[:sample_size].cpu().tolist()
        )
        subset = Subset(dataset, indices)

        if self.is_distributed:
            sampler = DistributedSampler(
                subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = None

        return DataLoader(
            subset,
            batch_size=base_loader.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=base_loader.num_workers,
            pin_memory=getattr(base_loader, "pin_memory", False),
            drop_last=False,
        )

    def _run_periodic_test_eval(
        self,
        *,
        trigger: str,
        global_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """
        Execute periodic test evaluation with optional sampling.

        Uses cached sampled test loader to ensure consistent samples across
        evaluations within a single training run.

        Args:
            trigger (str): "step" or "epoch" for logging context.
            global_step (int | None): Current optimizer step when trigger == "step".
            epoch (int | None): Epoch index (1-based) when trigger == "epoch".
        """
        if len(self.test_loader) == 0:
            return

        # Use cached sampled loader for consistent evaluation
        if self._cached_sampled_test_loader is not None:
            sampled_loader = self._cached_sampled_test_loader
        else:
            # Full test set evaluation
            sampled_loader = self.test_loader

        if self.is_distributed:
            dist.barrier()

        test_metrics = self.validate(sampled_loader, "test")

        if self.is_distributed:
            dist.barrier()

        if is_master_process():
            context = (
                f"step {global_step}"
                if trigger == "step" and global_step is not None
                else f"epoch {epoch}"
            )
            sample_info = (
                f"{self._test_eval_sample_proportion:.1%} of test set"
                if sampled_loader is not self.test_loader
                else "full test set"
            )
            formatted = ", ".join(
                (
                    f"{key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"{key}: {value}"
                )
                for key, value in test_metrics.items()
            )
            log_only_master_process(
                self.logger,
                "info",
                f"🧪 Test metrics ({context}, {sample_info}):\n  {formatted}",
            )

    def _run_svdb_validation(self, epoch: int) -> dict[str, float]:
        """Score this epoch's checkpoint on SpliceVarDB.

        The scoring processes share the job's GPUs with the training ranks, so
        every rank first releases its cached blocks. The master scores and
        records the result while the other ranks wait for it on a CPU (gloo)
        barrier, which leaves no NCCL kernel spinning on the scorers' GPUs; all
        ranks then return together. A scoring failure is logged and leaves the
        epoch unscored -- raising here would leave the other ranks waiting
        until the process-group timeout.

        Args:
            epoch (int): 0-based epoch just finished.

        Returns:
            dict[str, float]: ``svdb_*`` metrics on the master when scoring
                succeeded, ``{}`` otherwise.
        """
        torch.cuda.empty_cache()
        if self.is_distributed:
            dist.barrier()
        if not is_master_process():
            self._wait_for_svdb_scoring()
            return {}
        try:
            checkpoint = self.checkpoint_dir / f"epoch_{epoch + 1:03d}_model.pt"
            if not checkpoint.exists():
                raise RuntimeError(
                    f"{checkpoint} is missing; SpliceVarDB validation needs an "
                    "epoch checkpoint every epoch (checkpoint_save_epochs: 1)"
                )
            try:
                result = score_checkpoint(
                    self.svdb_validation, str(checkpoint), epoch + 1
                )
            except Exception:
                self.logger.exception(
                    f"❌ SpliceVarDB validation failed for epoch {epoch + 1}; "
                    "the epoch stays unscored"
                )
                return {}

            history_path = Path(self.svdb_validation.work_dir) / "history.json"
            records = (
                json.loads(history_path.read_text())["records"]
                if history_path.exists()
                else []
            )
            records = [r for r in records if r["epoch"] != epoch + 1]
            records.append({"epoch": epoch + 1, **result})
            best = max(records, key=lambda r: r["svdb_auroc"])
            history_path.write_text(
                json.dumps({"records": records, "best_epoch": best["epoch"]}, indent=2)
            )
            self.logger.info(
                f"🧪 Epoch {epoch + 1} SpliceVarDB AUROC {result['svdb_auroc']:.4f} "
                f"(n={int(result['svdb_n'])}, {result['svdb_seconds'] / 60:.1f} min); "
                f"best so far epoch {best['epoch']} ({best['svdb_auroc']:.4f})"
            )
            return result
        finally:
            self._wait_for_svdb_scoring()

    def _wait_for_svdb_scoring(self) -> None:
        """Meet the other ranks on the CPU group once the master has scored."""
        if self._svdb_wait_group is not None:
            dist.barrier(group=self._svdb_wait_group)

    def _restore_svdb_history(self, start_epoch: int) -> None:
        """Carry a resumed run's SpliceVarDB history and early-stopping state.

        Args:
            start_epoch (int): Number of epochs the resumed run completed.
        """
        src = Path(self._checkpoint_path).parent / "svdb_validation" / "history.json"
        if not src.exists():
            return
        saved = json.loads(src.read_text())["records"]
        records = sorted(
            (r for r in saved if r["epoch"] <= start_epoch), key=lambda r: r["epoch"]
        )
        if is_master_process() and records:
            dst = Path(self.svdb_validation.work_dir) / "history.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            best = max(records, key=lambda r: r["svdb_auroc"])
            dst.write_text(
                json.dumps({"records": records, "best_epoch": best["epoch"]}, indent=2)
            )
        if self.early_stopping_enabled and self.early_stopping_monitor == "svdb_auroc":
            self._early_stopping_best_value, self._early_stopping_wait = (
                replay_early_stopping(
                    [r["svdb_auroc"] for r in records],
                    self.early_stopping_mode,
                    self.early_stopping_min_delta,
                )
            )

    def _handle_early_stopping(self, val_metrics: dict[str, float], epoch: int) -> bool:
        """
        Update early stopping state based on validation metrics.

        Args:
            val_metrics (dict[str, float]): Validation metrics.
            epoch (int): Zero-based epoch index.

        Returns:
            bool: True when training should stop.
        """
        if not self.early_stopping_enabled:
            return False

        monitor_value = val_metrics.get(self.early_stopping_monitor)
        if monitor_value is None:
            log_only_master_process(
                self.logger,
                "warning",
                (
                    f"Early stopping monitor '{self.early_stopping_monitor}' "
                    "not found in validation metrics."
                ),
            )
            return False

        improved = False
        if self.early_stopping_mode == "min":
            improved = (
                monitor_value
                < self._early_stopping_best_value - self.early_stopping_min_delta
            )
        else:
            improved = (
                monitor_value
                > self._early_stopping_best_value + self.early_stopping_min_delta
            )

        if improved:
            self._early_stopping_best_value = monitor_value
            self._early_stopping_wait = 0
            if self.early_stopping_restore_best:
                self._save_best_model_state("early_stopping")
            log_only_master_process(
                self.logger,
                "info",
                (
                    f"✅ Early stopping monitor improved to {monitor_value:.6f} "
                    f"(epoch {epoch + 1})"
                ),
            )
            return False

        self._early_stopping_wait += 1
        log_only_master_process(
            self.logger,
            "info",
            (
                f"⏳ Early stopping patience "
                f"{self._early_stopping_wait}/{self.early_stopping_patience}"
            ),
        )

        if self._early_stopping_wait >= self.early_stopping_patience:
            self._early_stop_triggered = True
            log_only_master_process(
                self.logger,
                "info",
                (
                    "🛑 Early stopping triggered — "
                    f"best {self.early_stopping_monitor}: "
                    f"{self._early_stopping_best_value:.6f}"
                ),
            )
            return True

        return False

    def _reached_stop_epoch(self, epoch: int) -> bool:
        """
        Return True once the run has finished ``training.stop_after_epoch``.

        The cosine schedule is still built from ``training.epochs``, so a run
        cut here follows the same learning-rate path as a longer run stopped
        by hand after that epoch.

        Args:
            epoch (int): Zero-based epoch index.

        Returns:
            bool: True when training should stop after this epoch.
        """
        stop_after = OmegaConf.select(
            self.cfg, "training.stop_after_epoch", default=None
        )
        return stop_after is not None and epoch + 1 >= int(stop_after)

    def train(self) -> None:
        """Execute the main training loop."""
        self.logger.info("🚀 Starting training process...")
        self.logger.info("📊 Training configuration:")
        self.logger.info(f"  - Epochs: {self.cfg.training.epochs}")
        self.logger.info(f"  - Batch size: {self.cfg.training.batch_size}")
        self.logger.info(
            f"  - Learning rate backbone: {self.cfg.training.learning_rate_backbone}"
        )
        self.logger.info(
            "  - Learning rate classification head: "
            f"{self.cfg.training.learning_rate_classification_head}"
        )
        self.logger.info(f"  - Device: {self.device}")

        start_epoch = 0
        skip_batches = 0
        if self._checkpoint_path is not None:
            start_epoch, skip_batches = self._load_training_state(
                self._checkpoint_path
            )
            self._restore_rng_state()
            if self.svdb_validation is not None:
                self._restore_svdb_history(start_epoch)


        for epoch in range(start_epoch, self.cfg.training.epochs):
            self._current_epoch = epoch
            stop_training = False

            if self.is_distributed:
                if hasattr(
                    self.train_loader.sampler,
                    "set_epoch",
                ):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(
                    self.val_loader.sampler,
                    "set_epoch",
                ):
                    self.val_loader.sampler.set_epoch(epoch)
                if hasattr(
                    self.test_loader.sampler,
                    "set_epoch",
                ):
                    self.test_loader.sampler.set_epoch(epoch)

            log_only_master_process(
                self.logger,
                "info",
                f"\n🎯 Starting Epoch {epoch + 1}/{self.cfg.training.epochs}",
            )

            # Train (all processes)
            self.logger.info("🏃 Starting training phase...")
            train_metrics_epoch = self.train_epoch(
                skip_batches=skip_batches
            )
            # Only the first epoch after a mid-epoch resume is partial.
            skip_batches = 0
            self.logger.info(f"📊 Training completed for epoch {epoch + 1}")

            # Epoch-based checkpoint saving (master only)
            if is_master_process() and self.checkpoint_save_strategy == "epoch":
                save_every = int(getattr(self, "checkpoint_save_epochs", 1))
                if save_every > 0 and ((epoch + 1) % save_every == 0):
                    epoch_step = (epoch + 1) * len(self.train_loader)
                    self.logger.info(
                        f"💾 Saving checkpoint at end of epoch {epoch + 1} "
                        f"(epoch-based)..."
                    )
                    self._save_checkpoint(
                        step=len(self.train_loader),
                        global_step=epoch_step,
                        epoch=epoch,
                        train_metrics=(
                            train_metrics_epoch
                            if isinstance(train_metrics_epoch, dict)
                            else {}
                        ),
                        checkpoint_type="epoch",
                    )
                    self.last_checkpoint_step = epoch_step

            # Validation (all processes participate)
            log_only_master_process(
                self.logger,
                "info",
                f"🔍 Running validation after epoch {epoch + 1}",
            )
            val_metrics = self.validate(self.val_loader, "val")
            if self.svdb_validation is not None:
                val_metrics.update(self._run_svdb_validation(epoch))

            # Log metrics to MLflow (master process only)

            log_only_master_process(
                self.logger,
                "info",
                f"📊 Epoch {epoch + 1} validation - Loss: "
                f"{val_metrics.get('loss', 0):.4f}",
            )

            # Early stopping and checkpoint logic (master process checks)
            is_new_best = False
            if is_master_process():
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._best_model_epoch = epoch + 1
                    self._save_best_model_state("validation")
                    is_new_best = True
                stop_training = self._handle_early_stopping(val_metrics, epoch)
                if self._reached_stop_epoch(epoch):
                    stop_training = True

            current_lr = self.scheduler.get_last_lr()[0]

            stop_tensor = torch.tensor(
                1 if (stop_training and is_master_process()) else 0,
                device=self.device,
            )
            if self.is_distributed:
                dist.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                break

            if (
                self._test_eval_enabled
                and self._test_eval_frequency == "epoch"
                and len(self.test_loader) > 0
                and (epoch + 1 - self._last_test_eval_epoch) >= self._test_eval_interval
            ):
                log_only_master_process(
                    self.logger,
                    "info",
                    f"🧪 Running periodic test evaluation after epoch {epoch + 1}",
                )
                self._run_periodic_test_eval(trigger="epoch", epoch=epoch + 1)
                self._last_test_eval_epoch = epoch + 1

            # Log learning rate (all processes)
            current_lr = self.scheduler.get_last_lr()[0]
            self.logger.info((f"📈 Current learning rate: {current_lr:.6f}"))
            log_only_master_process(
                self.logger, "info", f"✅ Epoch {epoch + 1} completed"
            )

        # A terminal checkpoint, whatever the save strategy. The epoch-boundary
        # save above is guarded by strategy == "epoch", so a step-strategy run
        # finishes with only step_NNNNNN files and no checkpoint that names the
        # epoch it completed -- and its last save can be up to
        # checkpoint_save_steps batches short of the end. Downstream work keys
        # off "the model at epoch N", so the final state is always written here
        # under the epoch name.
        if is_master_process():
            final_epoch_index = int(self._current_epoch)
            final_path = (
                self.checkpoint_dir / f"epoch_{final_epoch_index + 1:03d}_model.pt"
            )
            if self.checkpoint_save_strategy == "epoch" and final_path.exists():
                log_only_master_process(
                    self.logger, "info",
                    f"💾 Final checkpoint already written by the epoch strategy: "
                    f"{final_path.name}",
                )
            else:
                log_only_master_process(
                    self.logger, "info",
                    f"💾 Saving final checkpoint at end of training "
                    f"(epoch {final_epoch_index + 1})...",
                )
                self._save_checkpoint(
                    step=len(self.train_loader),
                    global_step=(final_epoch_index + 1) * len(self.train_loader),
                    epoch=final_epoch_index,
                    train_metrics=(
                        train_metrics_epoch
                        if isinstance(train_metrics_epoch, dict) else {}
                    ),
                    checkpoint_type="epoch",
                )

        log_only_master_process(
            self.logger, "info", f"\n🎉 Training complete! {epoch + 1} epochs finished"
        )
        log_only_master_process(
            self.logger,
            "info",
            f"🏆 Best validation loss: {self._best_val_loss:.4f} "
            f"(epoch {self._best_model_epoch})",
        )

        if self.early_stopping_restore_best and self.best_model_path.exists():
            self._restore_best_model_state()

        run_final_test = False
        if len(self.test_loader) > 0:
            if self._test_eval_enabled:
                run_final_test = self._test_eval_evaluate_at_end
            else:
                run_final_test = True

        if run_final_test:
            log_only_master_process(
                self.logger, "info", "🧪 Starting final test evaluation..."
            )
            log_only_master_process(self.logger, "info", "Evaluating on test set...")
            test_metrics = self.validate(self.test_loader, "test")
            formatted_test_metrics = ", ".join(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in test_metrics.items()
            )
            log_only_master_process(
                self.logger,
                "info",
                f"📊 Final test metrics:\n  {formatted_test_metrics}",
            )
        elif len(self.test_loader) == 0:
            self.logger.info(
                "Test loader is empty, skipping test evaluation and terminating..."
            )
        else:
            log_only_master_process(
                self.logger,
                "info",
                "🧪 Final test evaluation skipped per configuration "
                "(evaluate_at_end=False).",
            )

        # Print final timing statistics if enabled
        actual_model = self._get_actual_model()
        log_model_timing_summary(self.logger, actual_model)

        # Print log file locations
        log_dir = self.checkpoint_dir / "logs"
        log_file = log_dir / f"training_{self.training_start_time}.log"

        # Determine checkpoint files
        most_recent_file = self.checkpoint_dir / "most_recent_model.pt"

        self.logger.info("Files saved to:")
        self.logger.info(f"  Checkpoints: {self.checkpoint_dir}")
        self.logger.info(f"  Logs: {log_file}")
        self.logger.info(f"  Most recent model: {most_recent_file}")

        # Final logging
        self.logger.info("📁 Training files saved to:")
        self.logger.info(f"  Checkpoints: {self.checkpoint_dir}")
        self.logger.info(f"  Logs: {log_file}")
        self.logger.info(f"  Most recent model: {most_recent_file}")
        self.logger.info("🎉 Training session completed successfully!")

        # Final torch.compile performance summary already logged during training
        if hasattr(self, "compile_performance") and self.compile_performance.get(
            "enabled", False
        ):
            log_only_master_process(
                self.logger,
                "info",
                "🚀 torch.compile performance analysis completed during training",
            )

        # Log final test metrics and cleanup experiment tracking

        stopped_early = "stop_tensor" in locals() and bool(stop_tensor.item())
        end_status = "early_stopped" if stopped_early else "completed"
        end_summary: dict[str, Any] = {
            "epochs_run": (epoch + 1) if "epoch" in locals() else start_epoch,
            "total_epochs": self.cfg.training.epochs,
            "best_val_loss": f"{self._best_val_loss:.6f}",
            "best_model_epoch": self._best_model_epoch,
            "checkpoint_dir": str(self.checkpoint_dir),
        }
        if run_final_test and "test_metrics" in locals():
            for key in ("loss", "auc_pr_acceptor", "auc_pr_donor"):
                if key in test_metrics:
                    end_summary[f"test_{key}"] = f"{float(test_metrics[key]):.4f}"

    def _save_best_model_state(self, reason: str) -> None:
        """Persist full training state as best checkpoint.

        Args:
            reason (str): Context string for logging.
        """
        if not is_master_process():
            return
        actual_model = self._get_actual_model()
        try:
            state = {
                "model_state_dict": (actual_model.state_dict()),
                "optimizer_state_dict": (self.optimizer.state_dict()),
                "scheduler_state_dict": (self.scheduler.state_dict()),
                "epoch": getattr(self, "_current_epoch", 0),
                "global_step": getattr(self, "last_checkpoint_step", 0),
                "best_val_loss": (self._best_val_loss),
                "best_model_epoch": (self._best_model_epoch),
                "metadata": build_checkpoint_metadata(
                    self.cfg,
                    run_name=self.run_name,
                    training_start_time=self.training_start_time,
                    mlflow_run_id=None,
                ),
            }
            torch.save(state, self.best_model_path)
            log_only_master_process(
                self.logger,
                "info",
                f"💾 Best model updated ({reason}) → {self.best_model_path}",
            )
        except Exception as exc:
            log_only_master_process(
                self.logger,
                "warning",
                f"⚠️  Failed to save best model snapshot: {exc}",
            )

    def _load_training_state(self, checkpoint_path: str) -> tuple[int, int]:
        """Restore full training state from checkpoint.

        Loads model weights, optimizer state,
        scheduler state, and best-model trackers.

        Args:
            checkpoint_path (str): Path to a .pt
                checkpoint saved by
                ``_save_checkpoint`` or
                ``_save_best_model_state``.

        Returns:
            tuple[int, int]: ``(start_epoch, skip_batches)``. ``skip_batches``
                is 0 for an epoch-boundary checkpoint and the number of
                already-consumed batches for a mid-epoch one.
        """
        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        actual_model = self._get_actual_model()
        actual_model.load_state_dict(normalize_checkpoint_keys(state_dict))

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "best_val_loss" in ckpt:
            self._best_val_loss = ckpt["best_val_loss"]
        if "best_model_epoch" in ckpt:
            self._best_model_epoch = ckpt["best_model_epoch"]

        saved_epoch = int(ckpt.get("epoch", 0))
        # step_in_epoch == -1 (or absent, for checkpoints written before this
        # existed) means the epoch completed, so resume at the next one.
        # Otherwise re-enter the SAME epoch and skip the batches already seen.
        step_in_epoch = int(ckpt.get("step_in_epoch", -1))
        if step_in_epoch < 0:
            start_epoch, skip_batches = saved_epoch + 1, 0
        else:
            start_epoch, skip_batches = saved_epoch, step_in_epoch + 1

        # DistributedSampler shards by rank: the same checkpoint replayed at a
        # different world_size reads a different batch sequence entirely. That
        # is silent unless flagged -- it is what made a single-GPU replay of a
        # 4-GPU divergence inconclusive.
        saved_ws = ckpt.get("world_size")
        if saved_ws is not None and int(saved_ws) != int(self.world_size):
            log_only_master_process(
                self.logger, "warning",
                f"⚠️  Checkpoint was written at world_size={saved_ws} but this "
                f"run has world_size={self.world_size}. DistributedSampler "
                f"shards by rank, so the batch sequence will NOT match and "
                f"mid-epoch resume cannot be exact.",
            )

        self._resume_rng_state = ckpt.get("rng_state")
        if self._resume_rng_state is None and skip_batches:
            log_only_master_process(
                self.logger, "warning",
                "⚠️  Mid-epoch resume without a saved RNG state: dropout and "
                "chunker noise will diverge from the interrupted run.",
            )

        if skip_batches:
            log_only_master_process(
                self.logger, "info",
                f"🔄 Resumed mid-epoch (epoch {saved_epoch + 1}, "
                f"skipping {skip_batches} batches already seen)",
            )
        else:
            log_only_master_process(
                self.logger, "info",
                f"🔄 Resumed from checkpoint"
                f" (epoch {saved_epoch + 1},"
                f" resuming at {start_epoch + 1})",
            )
        return start_epoch, skip_batches

    def _restore_rng_state(self) -> None:
        """Restore the RNG streams captured in the resumed checkpoint.

        Model weights and optimizer state alone do not make a resume
        equivalent to an uninterrupted run: dropout masks, the dynamic
        chunker's boundary noise and any sampling all advance their own
        generators. Without this the resumed run diverges from the one it
        claims to continue, which is exactly what makes "resume and see if it
        breaks again" an unreliable diagnostic.
        """
        rng = getattr(self, "_resume_rng_state", None)
        if not rng:
            return
        try:
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(_portable_to_numpy_rng(rng["numpy"]))
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            cuda_states = rng.get("torch_cuda")
            if cuda_states is not None and torch.cuda.is_available():
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                else:
                    log_only_master_process(
                        self.logger, "warning",
                        f"⚠️  Checkpoint holds {len(cuda_states)} CUDA RNG "
                        f"states but this run has {torch.cuda.device_count()} "
                        f"devices; leaving CUDA RNG unrestored.",
                    )
            log_only_master_process(
                self.logger, "info", "🎲 RNG state restored from checkpoint",
            )
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            log_only_master_process(
                self.logger, "warning",
                f"⚠️  Could not restore RNG state ({type(exc).__name__}: "
                f"{exc}); the resume will not be bit-exact.",
            )

    def _restore_best_model_state(self) -> None:
        """Restore best model weights from disk."""
        if not self.best_model_path.exists():
            return
        # Pin barriers to this rank's CUDA device. A bare dist.barrier() lets
        # NCCL infer the device from the current context, which on single-host
        # multi-GPU setups can deadlock post-training (observed: rank 0 spins
        # in user space while rank 1 sleeps on futex).
        barrier_device_ids = [self.local_rank] if self.device.type == "cuda" else None
        if self.is_distributed:
            dist.barrier(device_ids=barrier_device_ids)
        ckpt = torch.load(
            self.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(ckpt, dict) and ("model_state_dict" in ckpt):
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        state_dict = normalize_checkpoint_keys(state_dict)
        actual_model = self._get_actual_model()
        actual_model.load_state_dict(state_dict)
        if self.is_distributed:
            dist.barrier(device_ids=barrier_device_ids)
        log_only_master_process(
            self.logger,
            "info",
            f"🔄 Restored best model weights from {self.best_model_path}",
        )


def train(
    cfg_train: DictConfig,
    cfg_model: DictConfig,
    *,
    checkpoint_path: str | None = None,
) -> None:
    """Execute distributed training via torchrun.

    Args:
        cfg_train (DictConfig): Training config.
        cfg_model (DictConfig): Model config.
        checkpoint_path (str | None): Path to a
            checkpoint to resume training from.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        local_rank, world_size = set_up_ddp(cfg_train)
        print("🌐 Distributed training with torchrun")
        print(f"🎯 Local rank: {local_rank}, World size: {world_size}")
        print(f"🎯 GPU device: cuda:{local_rank}")
    else:
        print("🖥️  Single GPU training")

    trainer = SplicingTrainer(
        cfg_train,
        cfg_model,
        local_rank=local_rank,
        world_size=world_size,
        checkpoint_path=checkpoint_path,
    )

    try:
        trainer.train()
    except (RuntimeError, ValueError, FileNotFoundError, KeyboardInterrupt) as exc:
        raise
    finally:
        # Cleanup distributed training
        if world_size > 1:
            cleanup_distributed()


def main() -> None:
    """Execute training via torchrun or single-GPU.

    For model evaluation, use ``uv run -m evaluate.evaluate``
    instead.
    """
    # Mitigate cuFFT plan-cache / allocator fragmentation that surfaces as
    # CUFFT_INTERNAL_ERROR after many epochs of HyenaDNA fftconv calls.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(
        description=("Train splicing site prediction model"),
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config-preset",
        type=str,
        choices=[
            "hyenadna",
            "hyenadna_h5",
            "spliceai",
            "mha",
            "nucleotide",
        ],
        default=None,
        help=("Select preset config file under config/train/"),
    )
    config_group.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a custom YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=("Path to a training checkpoint (.pt) to resume from"),
    )
    args = parser.parse_args()

    if args.config_path is not None:
        config_path = to_absolute_path(args.config_path)
        cfg_train = OmegaConf.load(config_path)
    else:
        preset_map = {
            "nucleotide": ("config/train/train_nucleotide.yaml"),
            "hyenadna": ("config/train/train_hyenadna.yaml"),
            "hyenadna_h5": ("config/train/train_hyenadna_h5.yaml"),
        }
        preset_path = to_absolute_path(preset_map[args.config_preset])
        cfg_train = OmegaConf.load(preset_path)

    cfg_model = OmegaConf.load("config/issasplice/model.yaml")

    train(
        cfg_train,
        cfg_model,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
