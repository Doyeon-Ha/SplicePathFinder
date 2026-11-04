"""Tests for ``chunker_style='hnet'`` (H-Net boundary learning).

The pieces are checked against goombalab/hnet (``dc.py`` / ``hnet.py``):
cosine router with ``p_0 = 1``, boundary where ``p > 0.5``, F without
gradient, the EMA recurrence, the straight-through confidence product, and
padding that never becomes a boundary. CPU tests cover the functions; the
CUDA tests run the Mamba-2 kernel path and check, on a tiny two-stage
IssaSplice, that the task loss reaches the router -- the ``legacy`` style's
router gets no task gradient at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from model.chunker import (
    Dechunker,
    DynamicChunker,
    hnet_boundary_mask,
    hnet_boundary_prob,
    hnet_ema,
    hnet_ema_reference,
    hnet_rate_stats,
    hnet_ste,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _identity(dim: int) -> nn.Linear:
    """Bias-free identity projection, as H-Net initializes the router."""
    layer = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dim))
    return layer


class TestRouter:
    """Cosine routing probability and the 0.5 decision."""

    def test_prob_is_half_one_minus_cosine_with_first_position_forced(self) -> None:
        """Same, orthogonal and opposite neighbours give 0, 0.5 and 1."""
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        hidden = torch.stack([a, a, b, -b]).unsqueeze(0)  # (1, 4, 2)
        prob = hnet_boundary_prob(hidden, _identity(2), _identity(2))
        assert prob.dtype == torch.float32
        torch.testing.assert_close(prob, torch.tensor([[1.0, 0.0, 0.5, 1.0]]))

    def test_prob_is_float32_for_bf16_input(self) -> None:
        """The router runs in fp32 whatever the hidden dtype."""
        hidden = torch.randn(2, 5, 4).to(torch.bfloat16)
        prob = hnet_boundary_prob(hidden, _identity(4), _identity(4))
        assert prob.dtype == torch.float32
        assert prob.shape == (2, 5)

    def test_tie_is_not_a_boundary_and_padding_never_is(self) -> None:
        """argmax([1-p, p]) leaves a tie at 0.5 unselected; padding is excluded."""
        prob = torch.tensor([[1.0, 0.5, 0.51, 0.9]])
        valid = torch.tensor([[True, True, True, False]])
        assert hnet_boundary_mask(prob, None).tolist() == [[True, False, True, True]]
        assert hnet_boundary_mask(prob, valid).tolist() == [[True, False, True, False]]


class TestRateStats:
    """F without gradient, G with gradient, both over real positions."""

    def test_f_has_no_gradient_and_g_does(self) -> None:
        """The ratio loss reaches the router through G only (H-Net eq. 10)."""
        prob = torch.tensor([[1.0, 0.2, 0.8, 0.9]], requires_grad=True)
        mask = prob.detach() > 0.5
        rate, mean_prob = hnet_rate_stats(mask, prob, None)
        assert not rate.requires_grad
        assert mean_prob.requires_grad

    def test_padding_is_left_out_of_both_means(self) -> None:
        """A padded position counts in neither F nor G."""
        prob = torch.tensor([[1.0, 0.2, 0.8, 0.9]])
        mask = torch.tensor([[True, False, True, True]])
        valid = torch.tensor([[True, True, True, False]])
        rate, mean_prob = hnet_rate_stats(mask, prob, valid)
        assert rate.item() == pytest.approx(2 / 3)
        assert mean_prob.item() == pytest.approx((1.0 + 0.2 + 0.8) / 3)


class TestSmoothingAndConfidence:
    """EMA recurrence and the straight-through confidence factor."""

    def test_ema_follows_the_recurrence(self) -> None:
        """z_k = p_k x_k + (1 - p_k) z_{k-1} from z_{-1} = 0."""
        x = torch.tensor([[[2.0], [4.0], [8.0]]])
        p = torch.tensor([[1.0, 0.5, 0.25]])
        z1 = 0.5 * 4.0 + 0.5 * 2.0
        expected = torch.tensor([[[2.0], [z1], [0.25 * 8.0 + 0.75 * z1]]])
        torch.testing.assert_close(hnet_ema_reference(x, p), expected)

    def test_cpu_path_is_the_reference(self) -> None:
        """Off CUDA the public EMA is the sequential loop."""
        x = torch.randn(2, 7, 4)
        p = torch.rand(2, 7) * 0.9 + 0.05
        torch.testing.assert_close(hnet_ema(x, p), hnet_ema_reference(x, p))

    def test_ste_is_one_forward_and_identity_backward(self) -> None:
        """The product leaves values unchanged and sends x back to c."""
        confidence = torch.tensor([0.3, 0.9], requires_grad=True)
        x = torch.tensor([2.0, 5.0])
        out = x * hnet_ste(confidence)
        torch.testing.assert_close(out, x)
        out.sum().backward()
        torch.testing.assert_close(confidence.grad, x)


class TestConstruction:
    """Configuration guards and H-Net initialization."""

    def test_hnet_needs_an_ssm_encoder(self) -> None:
        """The router reads the encoder hidden state, which conv1d lacks."""
        with pytest.raises(ValueError, match="hnet"):
            DynamicChunker(hidden_dim=32, encoder_type="conv1d", chunker_style="hnet")

    def test_unknown_style_is_rejected(self) -> None:
        """A typo fails at build time."""
        with pytest.raises(ValueError, match="chunker_style"):
            DynamicChunker(hidden_dim=32, chunker_style="h-net")

    def test_hnet_dechunker_needs_the_decoder_stack(self) -> None:
        """Broadcast has no decoder layers to play H-Net's decoder."""
        with pytest.raises(ValueError, match="hnet"):
            Dechunker(dechunker_type="broadcast", hidden_dim=64, chunker_style="hnet")

    def test_hnet_modules_start_as_in_hnet(self) -> None:
        """Identity router, no logit heads, zero residual weight."""
        pytest.importorskip("mamba_ssm")
        chunker = DynamicChunker(
            hidden_dim=64,
            encoder_type="mamba2",
            target_chunk_rate=0.5,
            chunker_style="hnet",
        )
        eye = torch.eye(64)
        torch.testing.assert_close(chunker.router_q.weight, eye)
        torch.testing.assert_close(chunker.router_k.weight, eye)
        assert chunker.router_q.bias is None and chunker.router_k.bias is None
        assert not hasattr(chunker, "proj")
        assert chunker.encoder.proj is None
        assert float(chunker.encoder.residual_proj.weight.abs().sum()) == 0.0

    def test_router_takes_its_stage_learning_rate(self) -> None:
        """Router parameters join the stage's encoder group."""
        from train.train import SplicingTrainer

        match = SplicingTrainer._match_stage_param
        name = "hierarchical_chunker.chunkers.1.router_q.weight"
        assert match(None, name) == ("encoder", 1)
        assert match(None, "hierarchical_chunker.chunkers.0.proj.weight") is None


class TestLegacyUnchanged:
    """The default style ignores the new arguments."""

    def test_legacy_chunker_ignores_valid_mask(self) -> None:
        """Passing a validity mask changes nothing for ``legacy``."""
        torch.manual_seed(0)
        chunker = DynamicChunker(hidden_dim=16, target_chunk_rate=0.5).eval()
        x = torch.randn(2, 12, 16)
        valid = torch.ones(2, 12, dtype=torch.bool)
        valid[0, 8:] = False
        with torch.no_grad():
            plain = chunker(x)
            masked = chunker(x, valid_mask=valid)
        assert torch.equal(plain.boundary_indices, masked.boundary_indices)
        assert torch.equal(plain.chunk_seq, masked.chunk_seq)
        assert plain.boundary_prob is None and plain.selected_probs is None


def _tiny_hnet_model() -> Any:
    """Two-stage H-Net with a small local HyenaDNA trunk and character tokens."""
    from train.trainer import IssaSpliceTrainer
    from model.hyenadna.src.configuration_hyena import HyenaConfig

    full_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "config/train/splicepath_v50.yaml"
    )
    cfg = full_cfg.model
    cfg.seq_length = cfg.tokenizer_max_length = 128
    cfg.init_weight_dtype = "float32"
    cfg.base_backbone_output_dim = cfg.hidden_size = 64
    cfg.local_feature_output_dim = 256
    stage = {
        "encoder_type": "mamba2",
        "mamba2_num_layers": 2,
        "dechunker_type": "mamba2",
        "dechunker_mamba2_num_layers": 2,
        "encoder_transformer_num_layers": 1,
        "encoder_transformer_num_heads": 4,
        "dechunker_transformer_num_layers": 1,
        "dechunker_transformer_num_heads": 4,
        "chunker_style": "hnet",
    }
    cfg.dynamic_chunker = {
        "target_chunk_rate": 0.5,
        "enable_encoder_residual": True,
        "stages": [
            dict(stage, target_chunk_rate=0.3333),
            dict(stage, target_chunk_rate=0.5),
        ],
    }
    torch.manual_seed(0)
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "[PAD]": 5},
        pad_token_id=5,
    )
    tiny_config = HyenaConfig(
        d_model=64, n_layer=2, d_inner=128, vocab_size=16, max_seq_len=128
    )
    with (
        patch("model.model.AutoConfig.from_pretrained", return_value=tiny_config),
        patch("model.model.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        model = IssaSpliceTrainer(full_cfg).init_model(verbose=False).cuda()
    input_ids = torch.randint(0, 4, (2, 128))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return model, input_ids.cuda(), attention_mask.cuda()


@_CUDA
class TestCudaPath:
    """Kernel EMA and the end-to-end gradient path."""

    def test_kernel_ema_matches_the_reference(self) -> None:
        """H-Net's SSD-kernel EMA equals the recurrence up to bf16 rounding."""
        pytest.importorskip("mamba_ssm")
        torch.manual_seed(0)
        x = torch.randn(2, 300, 64, device="cuda")
        p = torch.rand(2, 300, device="cuda") * 0.9 + 0.05
        torch.testing.assert_close(
            hnet_ema(x, p).float(),
            hnet_ema_reference(x, p).float(),
            rtol=3e-2,
            atol=3e-2,
        )

    def test_task_loss_reaches_every_router(self) -> None:
        """Backward of the output alone gives every stage's router a gradient."""
        model, input_ids, attention_mask = _tiny_hnet_model()
        model.train()
        logits, aux = model(input_ids, attention_mask)
        logits.float().pow(2).mean().backward()
        for chunker in model.hierarchical_chunker.chunkers:
            for layer in (chunker.router_q, chunker.router_k):
                assert layer.weight.grad is not None
                assert float(layer.weight.grad.abs().sum()) > 0.0
        assert not aux["mean_chunk_rate"].requires_grad
        assert aux["mean_boundary_prob"].requires_grad

    @pytest.mark.parametrize("path", ["ema", "ste"])
    def test_each_dechunk_path_alone_reaches_the_router(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The EMA and the confidence product each carry the task gradient."""
        import model.chunker as chunker_module

        if path == "ema":  # cut the STE path
            monkeypatch.setattr(
                chunker_module, "hnet_ste", lambda c: torch.ones_like(c)
            )
        else:  # cut the EMA path
            ema = chunker_module.hnet_ema
            monkeypatch.setattr(
                chunker_module, "hnet_ema", lambda h, p, **kw: ema(h, p.detach(), **kw)
            )
        model, input_ids, attention_mask = _tiny_hnet_model()
        model.train()
        logits, _ = model(input_ids, attention_mask)
        logits.float().pow(2).mean().backward()
        for chunker in model.hierarchical_chunker.chunkers:
            for layer in (chunker.router_q, chunker.router_k):
                assert layer.weight.grad is not None
                assert float(layer.weight.grad.abs().sum()) > 0.0

    def test_fixed_and_unsmoothed_readouts_skip_the_ema(self) -> None:
        """Forced or unsmoothed boundaries reach the dechunker with p = 1."""
        model, _, _ = _tiny_hnet_model()
        model.eval()
        chunker = model.hierarchical_chunker.chunkers[1]
        torch.manual_seed(2)
        x = torch.randn(2, 40, 64, device="cuda")
        with torch.no_grad():
            learned = chunker(x)
            chunker.boundary_mode = "fixed"
            fixed = chunker(x)
            chunker.boundary_mode = "learned"
            chunker.hnet_smoothing = False
            unsmoothed = chunker(x)
        for out in (fixed, unsmoothed):
            at = torch.gather(out.boundary_prob, 1, out.boundary_indices)
            assert bool((at[out.token_mask] == 1.0).all())
        assert torch.equal(unsmoothed.boundary_indices, learned.boundary_indices)
        assert fixed.boundary_indices[0, :5].tolist() == [0, 2, 4, 6, 8]

    def test_inner_stage_never_selects_padding(self) -> None:
        """With a padded tail, boundaries and the rates stay on real positions."""
        model, _, _ = _tiny_hnet_model()
        model.eval()
        chunker = model.hierarchical_chunker.chunkers[1]
        torch.manual_seed(1)
        x = torch.randn(2, 40, 64, device="cuda")
        valid = torch.ones(2, 40, dtype=torch.bool, device="cuda")
        valid[0, 25:] = False
        x[0, 25:] = 0.0
        with torch.no_grad():
            out = chunker(x, None, valid_mask=valid)
        chosen = out.boundary_indices[0][out.token_mask[0]]
        assert int(chosen.max()) < 25
        assert out.boundary_prob is not None and out.selected_probs is not None
