"""Unit tests for ``utils.utils.resolve_batch_plan``.

Covers the precedence rules around ``training.effective_batch_size`` and
``training.gradient_accumulation_steps`` (auto-derive vs manual fallback,
divisibility check, conflict detection), plus the ``WORLD_SIZE``-driven
``num_gpus`` that lets one effective batch hold across any GPU count.
"""

import pytest
from omegaconf import OmegaConf

from utils.utils import BatchPlan, resolve_batch_plan


def _make_cfg(
    *,
    batch_size: int,
    effective_batch_size: int | None = None,
    grad_accum: int | None = None,
) -> OmegaConf:
    """Build a minimal cfg exposing only the keys the helper reads."""
    cfg = {"training": {"batch_size": batch_size}}
    if effective_batch_size is not None:
        cfg["training"]["effective_batch_size"] = effective_batch_size
    if grad_accum is not None:
        cfg["training"]["gradient_accumulation_steps"] = grad_accum
    return OmegaConf.create(cfg)


class TestResolveBatchPlan:
    """Behavioural tests for batch plan resolution."""

    def test_effective_batch_derives_grad_accum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """effective_batch_size set → grad_accum auto-derived from WORLD_SIZE."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg = _make_cfg(batch_size=8, effective_batch_size=128)
        plan = resolve_batch_plan(cfg)
        assert plan == BatchPlan(
            micro_batch_size=8,
            num_gpus=2,
            grad_accum_steps=8,
            effective_batch_size=128,
            source="effective",
        )

    def test_effective_batch_constant_across_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same effective_batch_size holds across GPU counts via grad-accum."""
        cfg = _make_cfg(batch_size=8, effective_batch_size=128)
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert resolve_batch_plan(cfg).grad_accum_steps == 16
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert resolve_batch_plan(cfg).grad_accum_steps == 4
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_batch_plan(cfg).grad_accum_steps == 2

    def test_manual_grad_accum_when_effective_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """effective_batch_size null → manual GA path scaled by WORLD_SIZE."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg = _make_cfg(batch_size=4, grad_accum=8)
        plan = resolve_batch_plan(cfg)
        assert plan.grad_accum_steps == 8
        assert plan.effective_batch_size == 4 * 2 * 8
        assert plan.source == "manual"

    def test_manual_default_grad_accum_is_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither key set → grad_accum defaults to 1."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        cfg = _make_cfg(batch_size=4)
        plan = resolve_batch_plan(cfg)
        assert plan.grad_accum_steps == 1
        assert plan.source == "manual"

    def test_single_gpu_when_world_size_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WORLD_SIZE unset (non-distributed launch) collapses num_gpus to 1."""
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        cfg = _make_cfg(batch_size=16, effective_batch_size=64)
        plan = resolve_batch_plan(cfg)
        assert plan.num_gpus == 1
        assert plan.grad_accum_steps == 4

    def test_non_divisible_effective_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """effective_batch_size that does not divide micro*num_gpus raises."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg = _make_cfg(batch_size=8, effective_batch_size=100)
        with pytest.raises(ValueError, match="not divisible"):
            resolve_batch_plan(cfg)

    def test_conflict_between_effective_and_manual_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting both keys with mismatched values raises."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg = _make_cfg(
            batch_size=8,
            effective_batch_size=128,
            grad_accum=2,  # would imply effective=32, conflicts with 128
        )
        with pytest.raises(ValueError, match="implies"):
            resolve_batch_plan(cfg)

    def test_consistent_effective_and_manual_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting both keys with consistent values is accepted."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg = _make_cfg(
            batch_size=8,
            effective_batch_size=128,
            grad_accum=8,  # matches derived
        )
        plan = resolve_batch_plan(cfg)
        assert plan.grad_accum_steps == 8
        assert plan.source == "effective"
