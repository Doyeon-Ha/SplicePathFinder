"""Distributed training engine abstraction (feature 009 — Phase 7).

This module introduces a thin :class:`DistributedEngine` ABC plus two
implementations — :class:`DDPEngine` (PyTorch
``DistributedDataParallel``) and :class:`DeepSpeedEngine` (lazy-imported
DeepSpeed). The factory :func:`create_engine` selects the implementation
from a Hydra ``training.distributed.engine`` value.

The goal is to let the trainer call a single set of methods
(``wrap`` / ``backward`` / ``step`` / ``zero_grad`` /
``save_checkpoint`` / ``load_checkpoint`` / ``unwrap``) and have either
DDP or DeepSpeed dispatch correctly underneath. ``train/train.py`` is
wired through this ABC for the model wrap, gradient backward, and
optimizer/scheduler step paths. Checkpoint persistence still uses the
existing rich-metadata ``torch.save`` flow for the DDP path; the
DeepSpeed ZeRO-aware checkpoint format is tracked as a follow-up
enhancement.

Reference: data-model.md entity 3, contracts/config-schema.md C-003 /
C-004, research.md R-003 / R-004.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

LOGGER = logging.getLogger(__name__)


SchedulerLike = Any  # torch LR schedulers do not share a public ABC.


class DistributedEngine(ABC):
    """Abstract trainer-side wrapper over DDP / DeepSpeed.

    Implementations isolate engine-specific calls (DDP wrapping vs
    ``deepspeed.initialize``, ``loss.backward()`` vs
    ``model_engine.backward(loss)``, etc.) so the trainer code stays
    engine-agnostic.

    Attributes:
        cfg (DictConfig): Hydra config slice used to construct the engine.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Store the Hydra config for the concrete engine.

        Args:
            cfg (DictConfig): Hydra config root (must contain
                ``training.distributed.*``).
        """
        self.cfg = cfg

    @abstractmethod
    def wrap(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: SchedulerLike | None,
    ) -> tuple[nn.Module, torch.optim.Optimizer | None, SchedulerLike | None]:
        """Wrap model / optimizer / scheduler for distributed execution.

        Args:
            model (nn.Module): The base model.
            optimizer (torch.optim.Optimizer | None): Pre-built optimizer.
            scheduler (SchedulerLike | None): Pre-built LR scheduler.

        Returns:
            tuple[nn.Module, torch.optim.Optimizer | None, SchedulerLike | None]:
                Wrapped triple (model is replaced with the engine wrapper).
        """

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Run the backward pass on the engine.

        Args:
            loss (torch.Tensor): Scalar loss tensor.
        """

    @abstractmethod
    def step(self) -> None:
        """Perform an optimizer step plus scheduler step (if applicable)."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Reset gradients on the underlying optimizer / engine."""

    @abstractmethod
    def save_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Persist trainable state to ``path``.

        Args:
            path (str): Filesystem path or directory used by the engine.
            tag (str | None): Optional tag (DeepSpeed multi-tag layout).
        """

    @abstractmethod
    def load_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Restore trainable state from ``path``.

        Args:
            path (str): Filesystem path or directory.
            tag (str | None): Optional tag matching ``save_checkpoint``.
        """

    @abstractmethod
    def unwrap(self, model: nn.Module) -> nn.Module:
        """Return the underlying ``nn.Module`` (e.g. for tokenizer access).

        Args:
            model (nn.Module): The wrapped model previously returned by
                :meth:`wrap`.

        Returns:
            nn.Module: The underlying user-defined module.
        """

    def is_master(self) -> bool:
        """Return True on rank 0 (or when not distributed).

        Returns:
            bool: True when this process is the master / single-process.
        """
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0


class DDPEngine(DistributedEngine):
    """PyTorch ``DistributedDataParallel`` engine.

    Mirrors the existing trainer's DDP semantics so the engine can act
    as a drop-in replacement once ``train.train`` is refactored.
    Optimizer / scheduler step semantics use the standard PyTorch flow:
    ``loss.backward(); optimizer.step(); scheduler.step()``.

    Attributes:
        local_rank (int | None): Local CUDA device id when distributed.
        wrapped_model (nn.Module | None): The DDP-wrapped model after
            :meth:`wrap`.
        optimizer (torch.optim.Optimizer | None): The bound optimizer.
        scheduler (SchedulerLike | None): The bound scheduler.
    """

    def __init__(self, cfg: DictConfig, local_rank: int | None = None) -> None:
        """Initialize the DDP engine bookkeeping.

        Args:
            cfg (DictConfig): Hydra config root.
            local_rank (int | None): Local CUDA device id (rank within
                node). Defaults to None (single-process mode).
        """
        super().__init__(cfg)
        self.local_rank = local_rank
        self.wrapped_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: SchedulerLike | None = None

    def wrap(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: SchedulerLike | None,
    ) -> tuple[nn.Module, torch.optim.Optimizer | None, SchedulerLike | None]:
        """Wrap with ``DistributedDataParallel`` when distributed.

        When :func:`torch.distributed.is_initialized` returns False the
        model is returned unchanged (single-GPU / CPU path).

        Args:
            model (nn.Module): Base model already moved to the right device.
            optimizer (torch.optim.Optimizer | None): Pre-built optimizer.
            scheduler (SchedulerLike | None): Pre-built scheduler.

        Returns:
            tuple[nn.Module, torch.optim.Optimizer | None, SchedulerLike | None]:
                (DDP-wrapped or unchanged model, optimizer, scheduler).
        """
        find_unused = bool(
            OmegaConf.select(
                self.cfg,
                "training.distributed.find_unused_parameters",
                default=False,
            )
        )
        is_distributed = (
            dist.is_available()
            and dist.is_initialized()
            and self.local_rank is not None
        )
        if is_distributed:
            wrapped = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused,
            )
            LOGGER.info(
                "DDPEngine wrapped model on local_rank=%s "
                "(find_unused_parameters=%s).",
                self.local_rank,
                find_unused,
            )
        else:
            wrapped = model
            LOGGER.info("DDPEngine: distributed not initialized; pass-through wrap.")

        self.wrapped_model = wrapped
        self.optimizer = optimizer
        self.scheduler = scheduler
        return wrapped, optimizer, scheduler

    def backward(self, loss: torch.Tensor) -> None:
        """Run ``loss.backward()`` (autograd handles DDP all-reduce).

        Args:
            loss (torch.Tensor): Scalar loss tensor.
        """
        loss.backward()

    def step(self) -> None:
        """Optimizer step + scheduler step (if bound).

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.optimizer is None:
            raise RuntimeError("DDPEngine.step called before wrap().")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        """Reset optimizer gradients.

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.optimizer is None:
            raise RuntimeError("DDPEngine.zero_grad called before wrap().")
        self.optimizer.zero_grad()

    def save_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Save model state_dict via ``torch.save``.

        Mirrors the existing trainer's checkpoint format: a dict with
        ``model_state_dict`` plus ``optimizer_state_dict`` /
        ``scheduler_state_dict`` when bound. ``tag`` is currently ignored
        (kept for ABC parity with the DeepSpeed engine).

        Args:
            path (str): Output filesystem path (.pt / .pth).
            tag (str | None): Unused for DDP. Defaults to None.
        """
        if not self.is_master():
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        underlying = self.unwrap(self.wrapped_model) if self.wrapped_model else None
        state: dict[str, Any] = {}
        if underlying is not None:
            state["model_state_dict"] = underlying.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)
        LOGGER.info("DDPEngine checkpoint saved → %s", path)

    def load_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Load checkpoint produced by :meth:`save_checkpoint`.

        Args:
            path (str): Path to the .pt / .pth file.
            tag (str | None): Unused for DDP. Defaults to None.

        Raises:
            FileNotFoundError: Checkpoint path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"DDPEngine checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        underlying = self.unwrap(self.wrapped_model) if self.wrapped_model else None
        if underlying is not None and "model_state_dict" in ckpt:
            underlying.load_state_dict(ckpt["model_state_dict"], strict=False)
        if self.optimizer is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        LOGGER.info("DDPEngine checkpoint loaded ← %s", path)

    def unwrap(self, model: nn.Module) -> nn.Module:
        """Return ``model.module`` when DDP-wrapped, else ``model``.

        Args:
            model (nn.Module): Possibly DDP-wrapped model.

        Returns:
            nn.Module: The underlying user model.
        """
        return model.module if isinstance(model, DDP) else model


class DeepSpeedEngine(DistributedEngine):
    """DeepSpeed engine wrapper (lazy import).

    DeepSpeed is an opt-in dependency installed via
    ``uv sync --group deepspeed``. Importing it inside the constructor
    keeps the default (DDP) install footprint small and prevents import
    errors on environments without the extra group.

    Attributes:
        config_path (str): Path to the DeepSpeed JSON config.
        zero_stage_override (int | None): Optional override for
            ``zero_optimization.stage``.
        model_engine (Any): The ``model_engine`` returned by
            ``deepspeed.initialize`` after :meth:`wrap`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        config_path: str,
        zero_stage_override: int | None = None,
    ) -> None:
        """Validate inputs and lazy-import DeepSpeed.

        Args:
            cfg (DictConfig): Hydra config root.
            config_path (str): Path to a DeepSpeed JSON config file.
            zero_stage_override (int | None): If provided, overrides
                ``zero_optimization.stage`` from the JSON config.

        Raises:
            FileNotFoundError: ``config_path`` does not exist.
            ImportError: DeepSpeed is not installed in the current env.
        """
        super().__init__(cfg)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"DeepSpeed config not found: {config_path}"
            )
        try:
            self._deepspeed = importlib.import_module("deepspeed")
        except ImportError as exc:
            raise ImportError(
                "DeepSpeed is not installed. Run `uv sync --group deepspeed` "
                "to install."
            ) from exc

        self.config_path = config_path
        self.zero_stage_override = zero_stage_override
        self.model_engine: Any = None

    def wrap(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: SchedulerLike | None,
    ) -> tuple[Any, torch.optim.Optimizer | None, SchedulerLike | None]:
        """Initialize DeepSpeed ``model_engine`` and bind it.

        The Hydra config slice is rendered into a JSON-compatible dict
        via :func:`build_deepspeed_config` so YAML keys
        (``training.batch_size``, ``training.gradient_accumulation_steps``,
        ``training.gradient_clipping``, ``training.weight_dtype``)
        override the JSON values per data-model.md entity 4.

        Args:
            model (nn.Module): Base model.
            optimizer (torch.optim.Optimizer | None): Pre-built optimizer.
            scheduler (SchedulerLike | None): Pre-built scheduler.

        Returns:
            tuple[Any, torch.optim.Optimizer | None, SchedulerLike | None]:
                (model_engine, optimizer, scheduler).
        """
        ds_config = build_deepspeed_config(
            self.cfg,
            self.config_path,
            zero_stage_override=self.zero_stage_override,
        )
        model_engine, opt, _, sched = self._deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )
        self.model_engine = model_engine
        return model_engine, opt, sched

    def backward(self, loss: torch.Tensor) -> None:
        """Delegate to ``model_engine.backward``.

        Args:
            loss (torch.Tensor): Scalar loss tensor.

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.model_engine is None:
            raise RuntimeError("DeepSpeedEngine.backward called before wrap().")
        self.model_engine.backward(loss)

    def step(self) -> None:
        """Delegate to ``model_engine.step`` (handles optim + scheduler).

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.model_engine is None:
            raise RuntimeError("DeepSpeedEngine.step called before wrap().")
        self.model_engine.step()

    def zero_grad(self) -> None:
        """Delegate to ``model_engine.zero_grad``.

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.model_engine is None:
            raise RuntimeError("DeepSpeedEngine.zero_grad called before wrap().")
        self.model_engine.zero_grad()

    def save_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Delegate to ``model_engine.save_checkpoint``.

        Args:
            path (str): Output directory (DeepSpeed treats this as a dir).
            tag (str | None): Optional checkpoint tag.

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.model_engine is None:
            raise RuntimeError("DeepSpeedEngine.save_checkpoint called before wrap().")
        self.model_engine.save_checkpoint(path, tag=tag)

    def load_checkpoint(self, path: str, tag: str | None = None) -> None:
        """Delegate to ``model_engine.load_checkpoint``.

        Args:
            path (str): Checkpoint directory.
            tag (str | None): Optional checkpoint tag.

        Raises:
            RuntimeError: ``wrap`` has not been called yet.
        """
        if self.model_engine is None:
            raise RuntimeError("DeepSpeedEngine.load_checkpoint called before wrap().")
        self.model_engine.load_checkpoint(path, tag=tag)

    def unwrap(self, model: nn.Module) -> nn.Module:
        """Return the user ``nn.Module`` underneath the model_engine.

        DeepSpeed exposes the wrapped model via ``model_engine.module``.

        Args:
            model (nn.Module): The DeepSpeed engine instance returned by
                :meth:`wrap`. Accepts a plain ``nn.Module`` for symmetry.

        Returns:
            nn.Module: Underlying user model.
        """
        return getattr(model, "module", model)


def build_deepspeed_config(
    cfg: DictConfig,
    config_path: str,
    zero_stage_override: int | None = None,
) -> dict[str, Any]:
    """Render the DeepSpeed JSON with Hydra YAML overrides applied.

    Sync rules (data-model.md entity 4):

    - ``training.batch_size`` → ``train_micro_batch_size_per_gpu``
    - ``training.gradient_accumulation_steps`` →
      ``gradient_accumulation_steps``
    - ``training.gradient_clipping`` (legacy bool dict) or float →
      ``gradient_clipping``
    - ``model.init_weight_dtype`` → ``bf16.enabled`` /
      ``fp16.enabled`` (mutually exclusive)
    - ``zero_stage_override`` (when provided) → ``zero_optimization.stage``

    Args:
        cfg (DictConfig): Hydra config root.
        config_path (str): Path to a DeepSpeed JSON file.
        zero_stage_override (int | None): Optional ZeRO stage override.

    Returns:
        dict[str, Any]: The merged DeepSpeed config dict ready to pass
            to ``deepspeed.initialize(config=...)``.

    Raises:
        FileNotFoundError: ``config_path`` does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")

    with open(config_path) as fh:
        ds_config: dict[str, Any] = json.load(fh)
    # Strip any documentation-only keys (we use _comment in the default).
    ds_config.pop("_comment", None)

    from utils.utils import resolve_batch_plan

    batch_plan = resolve_batch_plan(cfg)
    ds_config["train_micro_batch_size_per_gpu"] = batch_plan.micro_batch_size
    ds_config["gradient_accumulation_steps"] = batch_plan.grad_accum_steps

    grad_clip_value = _resolve_gradient_clip(cfg)
    if grad_clip_value is not None:
        ds_config["gradient_clipping"] = float(grad_clip_value)

    weight_dtype = OmegaConf.select(
        cfg, "model.init_weight_dtype", default=None
    )
    if weight_dtype == "bfloat16":
        ds_config["bf16"] = {"enabled": True}
        ds_config["fp16"] = {"enabled": False}
    elif weight_dtype == "float16":
        ds_config["bf16"] = {"enabled": False}
        ds_config["fp16"] = {"enabled": True}
    elif weight_dtype == "float32":
        ds_config["bf16"] = {"enabled": False}
        ds_config["fp16"] = {"enabled": False}

    if zero_stage_override is not None:
        ds_config.setdefault("zero_optimization", {})
        ds_config["zero_optimization"]["stage"] = int(zero_stage_override)

    return ds_config


def _resolve_gradient_clip(cfg: DictConfig) -> float | None:
    """Resolve the YAML ``training.gradient_clipping`` value to a float.

    The repo uses two shapes for this key:

    - ``training.gradient_clipping`` as a float (newer SpliceAI configs).
    - ``training.gradient_clipping`` as a dict
      ``{enabled: bool, min_clip: float, max_clip: float}`` (HyenaDNA /
      NeoBERT configs).

    Returns:
        float | None: Effective max clip value, or None when disabled or
            unset.
    """
    raw = OmegaConf.select(cfg, "training.gradient_clipping", default=None)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    enabled = OmegaConf.select(raw, "enabled", default=True)
    if not enabled:
        return None
    return float(OmegaConf.select(raw, "max_clip", default=1.0))


def create_engine(
    cfg: DictConfig,
    local_rank: int | None = None,
) -> DistributedEngine:
    """Build the engine selected by ``training.distributed.engine``.

    Defaults to DDP when the key is missing (preserves backward
    compatibility for HyenaDNA / NT / SpliceAI YAMLs that predate
    feature 009).

    Args:
        cfg (DictConfig): Hydra config root.
        local_rank (int | None): Local CUDA device id (rank within
            node). Required for the DDP engine when distributed.

    Returns:
        DistributedEngine: The engine instance ready for ``wrap``.

    Raises:
        ValueError: ``training.distributed.engine`` is not in
            ``{ddp, deepspeed}``.
    """
    engine_name = str(
        OmegaConf.select(cfg, "training.distributed.engine", default="ddp")
    ).lower()

    if engine_name == "ddp":
        return DDPEngine(cfg, local_rank=local_rank)

    if engine_name == "deepspeed":
        config_path = str(
            OmegaConf.select(
                cfg,
                "training.distributed.deepspeed.config_path",
                default="config/deepspeed/zero2_bf16.json",
            )
        )
        zero_override = OmegaConf.select(
            cfg,
            "training.distributed.deepspeed.zero_stage_override",
            default=None,
        )
        zero_override_int = int(zero_override) if zero_override is not None else None
        return DeepSpeedEngine(
            cfg,
            config_path=config_path,
            zero_stage_override=zero_override_int,
        )

    raise ValueError(
        f"Unsupported training.distributed.engine={engine_name!r}; "
        "use 'ddp' or 'deepspeed'."
    )
