"""RbfMamba2SavClassifier 단위 테스트 (spec 015 US2)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.sav_classifier import RbfExpansion, RbfMamba2SavClassifier

_HAS_CUDA = torch.cuda.is_available()


def test_rbf_expansion() -> None:
    rbf = RbfExpansion(num_centers=16, trainable=False)
    x = torch.rand(2, 5, 6)
    out = rbf(x)
    assert out.shape == (2, 5, 6 * 16)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_rbf_position_invariant() -> None:
    rbf = RbfExpansion(num_centers=8)
    x = torch.full((1, 3, 1), 0.5)
    out = rbf(x)
    assert torch.allclose(out[0, 0], out[0, 1])
    assert torch.allclose(out[0, 1], out[0, 2])


def test_rbf_fixed_kernel_has_no_params() -> None:
    rbf = RbfExpansion(num_centers=16, trainable=False, trainable_gamma=False)
    assert len(list(rbf.parameters())) == 0


def test_rbf_trainable_centers() -> None:
    rbf = RbfExpansion(num_centers=16, trainable=True)
    names = [n for n, _ in rbf.named_parameters()]
    assert "centers" in names


@pytest.mark.skipif(not _HAS_CUDA, reason="Mamba2는 CUDA 필요")
def test_mamba2_forward_shape() -> None:
    model = RbfMamba2SavClassifier(
        d_model=64, n_layers=2, d_state=32, headdim=32
    ).cuda()
    channels = torch.rand(2, 257, 6, device="cuda")
    logits = model(channels)
    assert logits.shape == (2,)


@pytest.mark.skipif(not _HAS_CUDA, reason="full 모델 구성에 CUDA 필요")
def test_mean_pool_readout() -> None:
    model = RbfMamba2SavClassifier(
        d_model=64, n_layers=1, d_state=32, headdim=32
    ).cuda()
    hidden = torch.rand(3, 41, 64, device="cuda")
    out = model.pool_and_head(hidden)
    assert out.shape == (3,)
