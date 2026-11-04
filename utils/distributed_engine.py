"""PyTorch distributed-data-parallel training engine."""

from __future__ import annotations

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
    """Trainer-side interface for PyTorch distributed execution."""

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
            tag (str | None): Optional checkpoint tag.
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
        (unused for this engine).

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








def create_engine(cfg: DictConfig, local_rank: int | None = None) -> DistributedEngine:
    """Use PyTorch distributed data parallel for the recorded recipe."""
    return DDPEngine(cfg, local_rank=local_rank)
