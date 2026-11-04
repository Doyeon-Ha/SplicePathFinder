"""Dynamic chunker tokenizer for HyenaDNA-based IssaSplice models.

This module implements an H-Net style learnable variable-length tokenizer
(``DynamicChunker``) and a 1 bp resolution restoration module
(``Dechunker``) that lets a HyenaDNA backbone process variable-length
chunk sequences while keeping the existing per-nucleotide classification
head unchanged.

The chunker reads input nucleotide embeddings, predicts a per-position
boundary probability with a configurable encoder followed by a shared
linear logit head, applies straight-through Gumbel-sigmoid for
differentiable hard selection during training and a deterministic
threshold during inference, and outputs the chunk-level sequence via
boundary-position selection (each chunk token is the input embedding at
the position of the chunk's starting boundary).

Two boundary-feature encoders are supported (selected via
``encoder_type``):

- ``conv1d`` (default, no extra dependencies): 1D convolution + GELU.
  Local context bounded by ``kernel_size``. Lightweight and stable.
- ``mamba2`` (H-Net Hwang et al. 2025): stacked Mamba-2 SSM blocks
  giving O(L) long-range context. Requires the optional
  ``mamba_ssm`` + ``causal-conv1d`` packages (lazy-imported).


Position 0 is always forced as a boundary so that the chunk count K is at
least 1 for every sample (DD-014). The boundary-predictor logit at
position 0 is detached so that the forced boundary contributes no
gradient through the predictor.

The dechunker restores chunk-level hidden states ``[B, K, D]`` back to
nucleotide-level ``[B, L, D]``. Four modes are supported:

- ``broadcast`` (DD-012, default v1): each nucleotide receives the
  hidden state of the chunk it belongs to (literal copy, 0 params).
- ``linear_interpolation`` (DD-012a, 2026-05-07): smooth blend between
  adjacent chunks, ``alpha = (l - boundary_k) / chunk_size`` (0 params).
- ``position_offset`` (DD-012a, 2026-05-07): broadcast plus a learnable
  per-within-chunk-offset bias (~ ``max_within_chunk_offset × D``
  params, zero-initialized so behaviour starts identical to broadcast).
- ``windowed_attention`` (DD-012b, 2026-05-07): multi-head
  cross-attention from per-position queries (``nt_emb``) to a local
  window of ``2 * attention_window_size + 1`` chunks. ``Q / K / V / O``
  linear projections (``4 × D²`` params) plus standard scaled
  dot-product attention. Memory stays linear in L (scores tensor
  ``[B, H, L, 2W+1]``) — 65 MB at the training-scale envelope vs.
  28.8 GB for full attention.
- ``mamba2`` (DD-012c, 2026-05-08, H-Net Hwang et al. 2025):
  broadcast plus stacked Mamba-2 SSM blocks (with residual links) at
  nt resolution. SSD algorithm gives O(L) long-range refinement.
  Within-chunk positions naturally differentiate via SSM state
  propagation. Requires the optional ``mamba_ssm`` + ``causal-conv1d``
  packages.


The ``attention`` keyword (full O(B*H*L*K) cross-attention) is rejected
at init with a redirect to ``windowed_attention``; full attention is
memory-prohibitive at training scale.

``chunker_style='hnet'`` replaces the boundary learning with H-Net's own
(Hwang, Wang & Gu 2025; goombalab/hnet ``dc.py`` / ``hnet.py``): a cosine
router on the encoder output (``p_t = (1 - cos(q(h_{t-1}), k(h_t))) / 2``,
``p_0 = 1``, boundary where ``p_t > 0.5`` in training and inference), chunks
gathered from the encoder output, an EMA over the chunk sequence weighted by
``p`` before broadcasting back, the decision confidence multiplied in through a
straight-through estimator, a zero-initialized fp32 residual projection, a
ratio loss whose F carries no gradient, and padding masks for inner stages.
The EMA and the confidence product are what let the task loss reach the
router; the ``legacy`` style uses boundaries only as integer indices, so its
router is trained by the ratio loss alone.

The chunker forward returns a ``ChunkerOutputs`` dataclass that
exposes both ``mean_chunk_rate`` (F, hard mask mean) and
``mean_boundary_prob`` (G, soft probability mean). G is required by
the H-Net paper-faithful ``hnet_ratio`` boundary loss (eq. 10); the
default ``l1`` boundary loss only consumes F and stays bit-identical
to the pre-DD-020 behaviour for checkpoint reproducibility.

References:
    - Spec: ``.specify/specs/001-hyenadna-h5-training/spec.md``
      (FR-021 ~ FR-029)
    - Plan: DD-010 (boundary-position selection),
      DD-011 (predictor input source),
      DD-012 / DD-012a (dechunker modes),
      DD-014 (position-0 forced boundary),
      DD-020 (H-Net ratio loss option, 2026-05-08)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_GUMBEL_NOISE_EPS: float = 1e-9

_CHUNKER_STYLES: tuple[str, ...] = ("legacy", "hnet")
# H-Net DeChunkLayer clamps p to [1e-4, 1 - 1e-4] before the EMA and runs the
# EMA through the Mamba-2 SSD kernel with these defaults (goombalab/hnet dc.py).
_HNET_P_CLAMP: float = 1e-4
_HNET_EMA_HEADDIM: int = 32
_HNET_EMA_BLOCK_SIZE: int = 256


class _OnesSTE(torch.autograd.Function):
    """H-Net's straight-through estimator: ones forward, identity backward."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        """Return ones shaped like ``x``."""
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass the incoming gradient to ``x`` unchanged."""
        return grad_output


def hnet_ste(confidence: torch.Tensor) -> torch.Tensor:
    """Ones that route their gradient into ``confidence`` (H-Net ``ste_func``).

    Args:
        confidence (torch.Tensor): Decision confidence ``c_t``.

    Returns:
        torch.Tensor: Ones of the same shape, differentiable w.r.t. ``c_t``.
    """
    return _OnesSTE.apply(confidence)


def hnet_boundary_prob(
    hidden: torch.Tensor, q_proj: nn.Linear, k_proj: nn.Linear
) -> torch.Tensor:
    """H-Net routing probability from the dissimilarity of adjacent positions.

    ``p_t = clamp((1 - cos(q(h_{t-1}), k(h_t))) / 2, 0, 1)`` for ``t >= 1`` and
    ``p_0 = 1`` (goombalab/hnet ``RoutingModule.forward``). Computed in fp32
    outside autocast so that the 0.5 decision is not shifted by bf16 rounding.

    Args:
        hidden (torch.Tensor): ``(B, L, D)`` encoder output.
        q_proj (nn.Linear): Projection applied to position ``t - 1``.
        k_proj (nn.Linear): Projection applied to position ``t``.

    Returns:
        torch.Tensor: ``(B, L)`` float32 boundary probabilities.
    """
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h = hidden.float()
        q = F.normalize(q_proj(h[:, :-1]), dim=-1)
        k = F.normalize(k_proj(h[:, 1:]), dim=-1)
        prob = ((1.0 - (q * k).sum(dim=-1)) / 2.0).clamp(0.0, 1.0)
    return F.pad(prob, (1, 0), value=1.0)


def hnet_boundary_mask(
    prob: torch.Tensor, valid_mask: torch.Tensor | None
) -> torch.Tensor:
    """Boundaries where ``prob > 0.5``, never on padding.

    H-Net takes ``argmax([1 - p, p])``, so a tie at exactly 0.5 is not a
    boundary.

    Args:
        prob (torch.Tensor): ``(B, L)`` boundary probabilities.
        valid_mask (torch.Tensor | None): ``(B, L)`` bool, True on real
            positions; None when every position is real.

    Returns:
        torch.Tensor: ``(B, L)`` bool boundary mask.
    """
    mask = prob > 0.5
    return mask if valid_mask is None else mask & valid_mask


def hnet_rate_stats(
    boundary_mask: torch.Tensor,
    prob: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ratio-loss statistics over real positions.

    F is the fraction actually selected and carries no gradient; G is the
    mean probability and is the ratio loss's only gradient path (H-Net
    paper, eq. 10).

    Args:
        boundary_mask (torch.Tensor): ``(B, L)`` bool.
        prob (torch.Tensor): ``(B, L)`` boundary probabilities.
        valid_mask (torch.Tensor | None): ``(B, L)`` bool or None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Scalars ``(F, G)``.
    """
    if valid_mask is None:
        return boundary_mask.to(prob.dtype).mean().detach(), prob.mean()
    weight = valid_mask.to(prob.dtype)
    count = weight.sum().clamp(min=1.0)
    rate = (boundary_mask.to(prob.dtype) * weight).sum() / count
    return rate.detach(), (prob * weight).sum() / count


def hnet_ema_reference(
    chunk_hidden: torch.Tensor, p_chunks: torch.Tensor
) -> torch.Tensor:
    """EMA over the chunk axis: ``z_k = p_k x_k + (1 - p_k) z_{k-1}``, ``z_{-1} = 0``.

    Sequential reference for CPU and for checking the kernel path.

    Args:
        chunk_hidden (torch.Tensor): ``(B, K, D)``.
        p_chunks (torch.Tensor): ``(B, K)`` boundary probability of each
            chunk's first position.

    Returns:
        torch.Tensor: ``(B, K, D)`` in ``chunk_hidden``'s dtype.
    """
    x = chunk_hidden.float()
    p = p_chunks.float().unsqueeze(-1)
    state = torch.zeros_like(x[:, 0])
    smoothed = []
    for k in range(x.shape[1]):
        state = p[:, k] * x[:, k] + (1.0 - p[:, k]) * state
        smoothed.append(state)
    return torch.stack(smoothed, dim=1).to(chunk_hidden.dtype)


def hnet_ema(
    chunk_hidden: torch.Tensor,
    p_chunks: torch.Tensor,
    headdim: int = _HNET_EMA_HEADDIM,
    block_size: int = _HNET_EMA_BLOCK_SIZE,
) -> torch.Tensor:
    """H-Net ``DeChunkLayer`` EMA over the chunk axis.

    On CUDA this is H-Net's own computation: the Mamba-2 SSD kernel with
    ``A = -1``, ``dt = log(1 / (1 - p))``, ``B = p``, ``C = 1`` and
    ``x = hidden / dt`` in bf16, which unrolls to the recurrence of
    :func:`hnet_ema_reference`. Elsewhere the reference loop runs.

    Args:
        chunk_hidden (torch.Tensor): ``(B, K, D)``.
        p_chunks (torch.Tensor): ``(B, K)`` probabilities, already clamped
            away from 0 and 1.
        headdim (int): SSD head dim; must divide ``D``.
        block_size (int): SSD chunk size.

    Returns:
        torch.Tensor: ``(B, K, D)`` in ``chunk_hidden``'s dtype.
    """
    if not chunk_hidden.is_cuda:
        return hnet_ema_reference(chunk_hidden, p_chunks)
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    batch, length, dim = chunk_hidden.shape
    nheads = dim // headdim
    dtype = torch.bfloat16
    dt = torch.log(1 / (1 - p_chunks)).to(dtype)
    x = (chunk_hidden / dt[..., None]).to(dtype)
    a = -torch.ones((nheads,), device=chunk_hidden.device, dtype=torch.float32)
    b = p_chunks.to(dtype)
    c = torch.ones_like(b)
    out = mamba_chunk_scan_combined(
        x.reshape(batch, length, nheads, headdim),
        dt.unsqueeze(-1).expand(batch, length, nheads).contiguous(),
        a,
        b[:, :, None, None],
        c[:, :, None, None],
        chunk_size=block_size,
        seq_idx=None,
    )
    return out.reshape(batch, length, dim).to(chunk_hidden.dtype)


@dataclass
class ChunkerOutputs:
    """Outputs from a single ``DynamicChunker`` forward pass.

    Returned as a named-key bag (per team convention for >=5 fields)
    rather than a tuple so consumers can address fields explicitly and
    new signals can be added without breaking unpack sites.

    Attributes:
        chunk_seq (torch.Tensor): Chunk-level sequence of shape
            ``(batch_size, k_max, hidden_dim)`` with padding zeroed.
        boundary_indices (torch.Tensor): int64 of shape
            ``(batch_size, k_max)`` — nucleotide positions of boundaries.
        token_mask (torch.Tensor): bool of shape
            ``(batch_size, k_max)`` — True for valid chunks.
        mean_chunk_rate (torch.Tensor): Scalar F = mean of the hard
            boundary mask across (batch x positions). Carries straight-
            through gradient via Gumbel-sigmoid soft-path during
            training; deterministic at inference. Used by both ``l1``
            and ``hnet_ratio`` boundary losses.
        mean_boundary_prob (torch.Tensor): Scalar G = mean of
            ``sigmoid(logits)`` across (batch x positions), with the
            position-0 contribution detached (consistent with F's
            position-0 detach per DD-014). Used as the differentiable
            signal in the H-Net ``hnet_ratio`` loss (paper eq. 10).
        encoder_residual (torch.Tensor | None): Separation-of-two-
            streams residual ``(batch_size, seq_len, hidden_dim)``
            (H-Net paper). The encoder's post-Mamba-2 hidden state is
            projected through ``Linear(D, D)`` and bypasses the chunker
            / main backbone, getting summed into the dechunker output
            before its own Mamba-2 refinement. ``None`` when the
            encoder doesn't produce one — e.g. ``conv1d`` encoder, or
            mamba2 with ``enable_residual_projection=False``.
        boundary_prob (torch.Tensor | None): ``(batch_size, seq_len)``
            float32 boundary probability ``p`` of the ``hnet`` style; the
            dechunker's EMA reads it at the boundaries. None for ``legacy``.
        selected_probs (torch.Tensor | None): ``(batch_size, seq_len)``
            decision confidence ``c_t`` (``p_t`` on boundaries, ``1 - p_t``
            elsewhere) of the ``hnet`` style. None for ``legacy``.
    """

    chunk_seq: torch.Tensor
    boundary_indices: torch.Tensor
    token_mask: torch.Tensor
    mean_chunk_rate: torch.Tensor
    mean_boundary_prob: torch.Tensor
    encoder_residual: torch.Tensor | None = None
    boundary_prob: torch.Tensor | None = None
    selected_probs: torch.Tensor | None = None


class _Conv1dBoundaryEncoder(nn.Module):
    """Conv1d-based boundary feature encoder for ``DynamicChunker``.

    Maps input nucleotide embedding ``[B, L, D]`` to per-position
    boundary features ``[B, L, H]`` via a single 1D convolution + GELU.
    Translation-invariant local context capture; weight-shared and
    parameter-efficient (``K × D × H`` params).
    """

    def __init__(
        self,
        hidden_dim: int,
        boundary_predictor_kernel_size: int,
        boundary_predictor_hidden_dim: int,
    ) -> None:
        """Initialize the Conv1d boundary encoder.

        Args:
            hidden_dim (int): Input nt embedding dim D.
            boundary_predictor_kernel_size (int): 1D conv kernel
                size (must be odd for symmetric padding).
            boundary_predictor_hidden_dim (int): Output channel dim H.
        """
        super().__init__()
        if boundary_predictor_kernel_size % 2 == 0:
            raise ValueError(
                "boundary_predictor_kernel_size must be odd; got "
                f"{boundary_predictor_kernel_size}"
            )
        padding = boundary_predictor_kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=int(hidden_dim),
            out_channels=int(boundary_predictor_hidden_dim),
            kernel_size=int(boundary_predictor_kernel_size),
            padding=padding,
        )
        self.activation = nn.GELU()

    def forward(self, nt_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Conv1d encoder.

        Args:
            nt_emb (torch.Tensor): ``(B, L, D)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - ``boundary_features``: ``(B, L, H)`` for the boundary
                  predictor.
                - ``encoder_residual``: always ``None`` for the conv1d
                  encoder. The residual stream (separation of two
                  streams, H-Net paper) is currently mamba2-only —
                  conv1d has no D-preserving hidden state by
                  construction (its output is already the H-dim
                  boundary feature).
        """
        x = nt_emb.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)
        return x.transpose(1, 2), None


def _build_ssm_layers(
    ssm_variant: str,
    num_layers: int,
    hidden_dim: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
) -> nn.ModuleList:
    """Build the SSM stack shared by the boundary encoder and the dechunker.

    Args:
        ssm_variant (str): ``"mamba2"``.
        num_layers (int): Number of blocks in the stack.
        hidden_dim (int): Model dim D of each block.
        d_state (int): SSM state dim.
        d_conv (int): Depthwise conv width. ``mamba2`` only.
        expand (int): Inner expansion factor.
        headdim (int): SSD head dim; must divide ``expand * hidden_dim``.

    Returns:
        nn.ModuleList: ``num_layers`` blocks, each preserving shape.

    Raises:
        ImportError: If the optional ``mamba_ssm`` package (or a version
            new enough to carry ``Mamba3``) is not installed.
    """
    kwargs = {
        "d_model": int(hidden_dim),
        "d_state": int(d_state),
        "expand": int(expand),
        "headdim": int(headdim),
    }
    if ssm_variant == "mamba2":
        try:
            from mamba_ssm import Mamba2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ssm_variant='mamba2' requires the optional "
                "mamba_ssm package (and causal-conv1d). Install with:\n"
                "    uv pip install mamba-ssm causal-conv1d\n"
                "Note: requires CUDA toolkit; build may take several "
                "minutes. See https://github.com/state-spaces/mamba "
                "for environment notes."
            ) from exc
        return nn.ModuleList(
            [Mamba2(d_conv=int(d_conv), **kwargs) for _ in range(int(num_layers))]
        )
    raise ValueError(f"Unsupported SSM variant: {ssm_variant}")


class _Mamba2BoundaryEncoder(nn.Module):
    """SSM based boundary feature encoder for ``DynamicChunker``.

    Stacks ``num_layers`` SSM blocks (with residual
    connections) on the input nt embedding, then projects to the
    per-position boundary feature dim H. Mamba-2's linear-time
    state-space duality (SSD) gives the chunker O(L) long-range
    context — a strict capability gain over the local Conv1d encoder
    whose receptive field is bounded by ``kernel_size``.

    Following H-Net (Hwang et al., 2025), 1–2 lightweight Mamba-2
    layers are typically sufficient for the chunker.

    Requires the optional ``mamba_ssm`` (and ``causal-conv1d``)
    packages. Import is deferred to ``__init__`` so the rest of the
    chunker module loads cleanly even when these are absent.

    Attributes:
        layers (nn.ModuleList): Mamba-2 blocks, each preserving shape.
        proj (nn.Linear): Final projection from D to H.
        activation (nn.GELU): Non-linearity before projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        boundary_predictor_hidden_dim: int,
        num_layers: int = 2,
        ssm_variant: str = "mamba2",
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        enable_residual_projection: bool = True,
        transformer_num_layers: int = 0,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.1,
        boundary_feature_head: bool = True,
        hnet_residual: bool = False,
    ) -> None:
        """Initialize the Mamba-2 boundary encoder.

        Args:
            hidden_dim (int): Input nt embedding dim D (also Mamba2's
                internal ``d_model``).
            boundary_predictor_hidden_dim (int): Output dim H of the
                feature projection (consumed by the shared logit head).
            num_layers (int): Number of stacked Mamba-2 blocks.
                Defaults to 2 (matches H-Net chunker depth).
            ssm_variant (str): ``"mamba2"``.
                Mamba-3 has no depthwise conv, so ``d_conv`` is ignored
                for that variant.
            d_state (int): SSM state dim. Lower is cheaper. Defaults
                to 64 (Mamba-2 paper default is 128).
            d_conv (int): Local conv kernel inside each Mamba-2 block.
                Defaults to 4.
            expand (int): Channel expansion factor inside Mamba-2.
                Defaults to 2 (paper default).
            headdim (int): SSM head dim. Mamba-2 requires ``d_inner``
                (= ``expand * hidden_dim``) to be divisible by this.
                Defaults to 64 (paper default; works for production
                ``hidden_dim=256`` with ``expand=2`` → 8 heads). Tests
                with smaller ``hidden_dim`` should override to 8 or 16.
            enable_residual_projection (bool): When True, the encoder
                produces a separate ``encoder_residual`` stream
                (``Linear(D, D)`` projection of the post-Mamba-2 hidden
                state) returned alongside the boundary features.
                Implements H-Net's "separation of two streams"
                (boundary stream + residual stream) — the residual
                bypasses the chunker / main backbone and is added back
                at the dechunker side. Defaults to True
                (paper-faithful).
            transformer_num_layers (int): Number of Transformer encoder
                layers appended after the Mamba-2 stack. Implements the
                dnaHNet ``M{num_layers}T{transformer_num_layers}`` hybrid
                template (4 Mamba + 1 Transformer per stage in the
                paper). 0 (default) = pure Mamba (backward-compat).
            transformer_num_heads (int): Number of attention heads when
                ``transformer_num_layers > 0``. Must divide
                ``hidden_dim``. Defaults to 4.
            transformer_dropout (float): Dropout for the Transformer
                layers. Defaults to 0.1.
            boundary_feature_head (bool): When False, no ``proj`` is built
                and ``forward`` returns the normalized encoder hidden state
                as the boundary features (the ``hnet`` router reads it).
            hnet_residual (bool): When True, ``residual_proj`` gets H-Net's
                treatment: zero-initialized weight and an fp32 computation
                outside autocast.

        Raises:
            ImportError: If ``mamba_ssm`` is not installed. The message
                points to the install command.
            ValueError: If ``num_layers < 1`` or if
                ``expand * hidden_dim`` is not divisible by ``headdim``.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
        d_inner = int(expand) * int(hidden_dim)
        if d_inner % int(headdim) != 0:
            raise ValueError(
                f"{ssm_variant} requires expand*hidden_dim ({d_inner}) to be "
                f"divisible by headdim ({headdim}). Reduce headdim or "
                f"increase hidden_dim/expand."
            )

        self.ssm_variant: str = str(ssm_variant)
        self.layers = _build_ssm_layers(
            ssm_variant=self.ssm_variant,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        # Pre-norm RMSNorm interleaved between Mamba-2 layers
        # (H-Net paper Section 3.2). One RMSNorm per layer + a final
        # output norm, matching the canonical SSM stack template.
        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(int(hidden_dim)) for _ in range(int(num_layers))]
        )
        self.output_norm = nn.RMSNorm(int(hidden_dim))
        self.activation = nn.GELU()
        self.proj: nn.Linear | None = (
            nn.Linear(int(hidden_dim), int(boundary_predictor_hidden_dim))
            if boundary_feature_head
            else None
        )
        # Separation-of-two-streams residual projection (H-Net): the
        # encoder's post-Mamba-2 hidden state is projected through
        # ``Linear(D, D)`` to form the encoder_residual, which bypasses
        # the chunker/main backbone and is summed into the dechunker
        # output before its own Mamba-2 layers.
        self.residual_proj: nn.Linear | None = (
            nn.Linear(int(hidden_dim), int(hidden_dim))
            if enable_residual_projection
            else None
        )
        self.hnet_residual: bool = bool(hnet_residual)
        if self.hnet_residual and self.residual_proj is not None:
            nn.init.zeros_(self.residual_proj.weight)
        # dnaHNet ``M{N}T{M}`` hybrid: optional Transformer layers stacked
        # after the Mamba-2 block. PyTorch ``TransformerEncoderLayer``
        # with ``norm_first=True`` uses its own pre-norm + residual
        # internally, so no extra wrapping needed.
        # ``batch_first=True`` keeps the (B, L, D) layout consistent with
        # the Mamba stack. SDPA (used under the hood) gives
        # memory-efficient attention on CUDA in bf16/fp16.
        self.transformer_layers: nn.ModuleList | None = None
        if int(transformer_num_layers) > 0:
            if int(hidden_dim) % int(transformer_num_heads) != 0:
                raise ValueError(
                    f"transformer_num_heads ({transformer_num_heads}) must "
                    f"divide hidden_dim ({hidden_dim})."
                )
            self.transformer_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=int(hidden_dim),
                        nhead=int(transformer_num_heads),
                        dim_feedforward=4 * int(hidden_dim),
                        dropout=float(transformer_dropout),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(int(transformer_num_layers))
                ]
            )


    def forward(
        self,
        nt_emb: torch.Tensor,
        symbols: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Mamba-2 encoder.

        Args:
            nt_emb (torch.Tensor): ``(B, L, D)``.
            symbols (torch.Tensor | None): ``(B, L)`` int64 nt symbols
                (A, C, G, T, N = 0..4); reserved for input compatibility.
            pad_mask (torch.Tensor | None): ``(B, L)`` bool, True on
                padding; passed to the Transformer layers as
                ``src_key_padding_mask``. The SSM blocks are causal, so
                trailing padding never reaches a real position there.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - ``boundary_features``: ``(B, L, H)`` for the boundary
                  predictor.
                - ``encoder_residual``: ``(B, L, D)`` projected hidden
                  state for the decoder-side residual stream when
                  ``enable_residual_projection=True``; otherwise None.

        """
        x = nt_emb
        # Pre-norm template: x = x + Layer(RMSNorm(x)). Residual is on
        # the unnormalized stream, so signal magnitude does not collapse
        # with depth — a known stability property of pre-norm transformers
        # adapted to SSM stacks (H-Net paper, Mamba-2 paper).
        for idx, (layer, norm) in enumerate(
            zip(self.layers, self.layer_norms, strict=True)
        ):
            x = x + layer(norm(x))
        # dnaHNet hybrid: Transformer layers follow the Mamba stack
        # (M-then-T order from the paper's ``M{N}T{M}`` layout). Each
        # TransformerEncoderLayer does its own pre-norm + residual.
        if self.transformer_layers is not None:
            for tlayer in self.transformer_layers:
                x = tlayer(x, src_key_padding_mask=pad_mask)
        encoder_hidden = self.output_norm(x)
        boundary_features = (
            self.proj(self.activation(encoder_hidden))
            if self.proj is not None
            else encoder_hidden
        )
        if self.residual_proj is None:
            encoder_residual = None
        elif self.hnet_residual:
            with torch.autocast(device_type=encoder_hidden.device.type, enabled=False):
                encoder_residual = self.residual_proj(encoder_hidden.float())
        else:
            encoder_residual = self.residual_proj(encoder_hidden)
        return boundary_features, encoder_residual


class DynamicChunker(nn.Module):
    """Learnable variable-length tokenizer using boundary-position selection.

    The chunker predicts a per-position boundary probability from input
    nucleotide embeddings using a 1D convolution + MLP head. At training
    time it samples a hard boundary mask via straight-through
    Gumbel-sigmoid; at inference time it applies a deterministic threshold
    on ``sigmoid(logits)``. The chunk-level token at index ``k`` is the
    input embedding at the position of the ``k``-th boundary
    (boundary-position selection); no additional aggregation parameters
    are introduced.

    Position 0 is always forced as a boundary so that ``K >= 1`` is
    guaranteed for every sample. The forced boundary at position 0
    contributes no gradient to the boundary predictor (the logit at that
    position is detached before sampling).

    Attributes:
        target_chunk_rate (float): Target ratio of chunk count to input
            length. Used by the trainer to compute the boundary
            regularization loss.
        gumbel_temperature (float): Temperature for Gumbel-sigmoid
            sampling.
        inference_threshold (float): Threshold applied to
            ``sigmoid(logits)`` at inference time.
        conv (nn.Conv1d): 1D convolution that produces local context
            features from the nucleotide embedding.
        proj (nn.Linear): Linear head that projects conv features to a
            single boundary logit per position.
    """

    _SUPPORTED_ENCODER_TYPES: tuple[str, ...] = ("conv1d", "mamba2")

    def __init__(
        self,
        hidden_dim: int,
        boundary_predictor_kernel_size: int = 7,
        boundary_predictor_hidden_dim: int = 64,
        target_chunk_rate: float = 0.2,
        gumbel_temperature: float = 1.0,
        inference_threshold: float = 0.5,
        boundary_decision: str = "gumbel",
        encoder_type: str = "conv1d",
        mamba2_num_layers: int = 2,
        mamba2_d_state: int = 64,
        mamba2_d_conv: int = 4,
        mamba2_expand: int = 2,
        mamba2_headdim: int = 64,
        enable_encoder_residual: bool = True,
        transformer_num_layers: int = 0,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.1,
        boundary_mode: str = "learned",
        chunker_style: str = "legacy",
    ) -> None:
        """Initialize the dynamic chunker.

        Args:
            hidden_dim (int): Dimensionality of input nucleotide
                embedding (D in shape (B, L, D)).
            boundary_predictor_kernel_size (int): 1D conv kernel size for
                the ``conv1d`` encoder. Must be odd so that
                ``padding = kernel_size // 2`` preserves sequence
                length. Ignored when ``encoder_type='mamba2'``.
            boundary_predictor_hidden_dim (int): Hidden dim H of the
                boundary feature, consumed by the shared logit head.
            target_chunk_rate (float): Target ratio of chunks to input
                length. Must be in ``(0, 1]``.
            gumbel_temperature (float): Temperature for Gumbel-sigmoid.
            boundary_decision (str): Training-time rule -- "gumbel" (legacy
                stochastic straight-through) or "deterministic_ste" (H-Net's
                threshold, identical to the inference rule).
            inference_threshold (float): Threshold for inference-time
                hard selection.
            encoder_type (str): Boundary feature encoder. One of
                ``conv1d`` (default, lightweight local context) or
                ``mamba2`` (H-Net style, O(L) long-range SSM).
            mamba2_num_layers (int): Number of Mamba-2 layers when
                ``encoder_type='mamba2'``. Defaults to 2.
            mamba2_d_state (int): Mamba-2 SSM state dim. Defaults to 64.
            mamba2_d_conv (int): Mamba-2 internal conv kernel size.
                Defaults to 4.
            mamba2_expand (int): Mamba-2 channel expansion factor.
                Defaults to 2.
            mamba2_headdim (int): Mamba-2 SSD head dim. Must divide
                ``mamba2_expand * hidden_dim``. Defaults to 64.
            enable_encoder_residual (bool): When True and
                ``encoder_type='mamba2'``, the encoder also produces a
                projected residual stream (``Linear(D, D)``) that
                bypasses the chunker and is summed into the dechunker
                output. Implements H-Net's "separation of two streams"
                (paper Section 3.3). Ignored for ``conv1d`` encoder.
                Defaults to True.
            transformer_num_layers (int): Number of Transformer encoder
                layers appended after the Mamba-2 stack in the boundary
                encoder. dnaHNet paper uses ``M4T1`` (4 Mamba + 1
                Transformer) per stage. 0 (default) = pure Mamba
                (backward-compat). Ignored for ``conv1d`` encoder.
            transformer_num_heads (int): Attention heads for the
                Transformer layers. Defaults to 4.
            transformer_dropout (float): Dropout for the Transformer
                layers. Defaults to 0.1.
            boundary_mode (str): Boundary selection strategy. One of
                ``learned`` (default, backward-compat): boundaries are
                predicted by the encoder + logit head and sampled via
                straight-through Gumbel-sigmoid (train) / threshold
                (inference). ``fixed``: boundaries are placed on a
                deterministic uniform grid at stride
                ``round(1 / target_chunk_rate)`` regardless of content —
                the H-Net ``fixbound`` control (Hwang et al. 2025 §5.1 /
                LDARNet Table 3). The encoder still runs (so the residual
                stream and FLOP budget match the learned arm), but its
                boundary logits are ignored; pair with
                ``boundary_loss_weight=0`` since there is no routing to
                regularize. Enables the learned-vs-fixed ablation that
                isolates the contribution of *learned* routing at matched
                compute.
            chunker_style (str): ``legacy`` (default) or ``hnet``. ``hnet``
                routes with H-Net's cosine router on the encoder output
                (boundary where ``p > 0.5`` in training and inference),
                gathers chunks from the encoder output, never places a
                boundary on padding, and reports F without gradient; the
                matching ``Dechunker`` adds the EMA and the confidence
                product. ``gumbel_temperature``, ``inference_threshold``,
                ``boundary_decision`` and ``boundary_predictor_hidden_dim``
                are unused. Requires a ``mamba2`` encoder with
                ``enable_encoder_residual=True``.

        Raises:
            ValueError: If ``boundary_predictor_kernel_size`` is even
                (conv1d only), if ``target_chunk_rate`` is outside
                ``(0, 1]``, if ``encoder_type`` is not supported, or if
                ``boundary_mode`` is not one of ``{learned, fixed}``.
            ImportError: If ``encoder_type='mamba2'`` but the optional
                ``mamba_ssm`` package is not installed.
        """
        super().__init__()

        if encoder_type not in self._SUPPORTED_ENCODER_TYPES:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type!r}. Supported: "
                f"{self._SUPPORTED_ENCODER_TYPES}."
            )
        if boundary_mode not in ("learned", "fixed"):
            raise ValueError(
                f"Unknown boundary_mode: {boundary_mode!r}. Supported: "
                "{'learned', 'fixed'}."
            )
        if not 0.0 < target_chunk_rate <= 1.0:
            raise ValueError(
                f"target_chunk_rate must be in (0, 1]; got {target_chunk_rate}"
            )

        self.target_chunk_rate: float = float(target_chunk_rate)
        self.gumbel_temperature: float = float(gumbel_temperature)
        self.inference_threshold: float = float(inference_threshold)
        # "gumbel" (legacy default, keeps existing checkpoints reproducible) or
        # "deterministic_ste" (H-Net's own rule; train and inference agree).
        self.boundary_decision: str = str(boundary_decision)
        if self.boundary_decision not in ("gumbel", "deterministic_ste"):
            raise ValueError(
                "boundary_decision must be one of "
                f"{{'gumbel', 'deterministic_ste'}}; got {boundary_decision!r}."
            )
        self.encoder_type: str = encoder_type
        self.boundary_mode: str = boundary_mode
        if chunker_style not in _CHUNKER_STYLES:
            raise ValueError(
                f"Unknown chunker_style: {chunker_style!r}. Supported: "
                f"{_CHUNKER_STYLES}."
            )
        self.chunker_style: str = chunker_style
        # Inference-only switch for ``hnet``: False hands the dechunker p = 1 at
        # every boundary, so its EMA passes each chunk through unsmoothed.
        self.hnet_smoothing: bool = True
        hnet = chunker_style == "hnet"
        if hnet and encoder_type == "conv1d":
            raise ValueError(
                "chunker_style='hnet' routes on the encoder hidden state; use "
                "encoder_type 'mamba2'."
            )
        if hnet and not enable_encoder_residual:
            raise ValueError(
                "chunker_style='hnet' needs the encoder residual (H-Net "
                "residual_proj); set enable_encoder_residual=True."
            )

        if encoder_type == "conv1d":
            self.encoder: nn.Module = _Conv1dBoundaryEncoder(
                hidden_dim=hidden_dim,
                boundary_predictor_kernel_size=(boundary_predictor_kernel_size),
                boundary_predictor_hidden_dim=(boundary_predictor_hidden_dim),
            )
        else:  # mamba2
            self.encoder = _Mamba2BoundaryEncoder(
                hidden_dim=hidden_dim,
                boundary_predictor_hidden_dim=(boundary_predictor_hidden_dim),
                num_layers=mamba2_num_layers,
                ssm_variant=encoder_type,
                d_state=mamba2_d_state,
                d_conv=mamba2_d_conv,
                expand=mamba2_expand,
                headdim=mamba2_headdim,
                enable_residual_projection=enable_encoder_residual,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_dropout=transformer_dropout,
                boundary_feature_head=not hnet,
                hnet_residual=hnet,
            )
        if hnet:
            # H-Net RoutingModule: bias-free q / k projections initialized to
            # the identity, so at step 0 p_t is the plain cosine
            # dissimilarity of adjacent encoder states.
            self.router_q = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
            self.router_k = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
            with torch.no_grad():
                self.router_q.weight.copy_(torch.eye(int(hidden_dim)))
                self.router_k.weight.copy_(torch.eye(int(hidden_dim)))
        else:
            self.proj = nn.Linear(int(boundary_predictor_hidden_dim), 1)

    def predict_logits(self, nt_emb: torch.Tensor) -> torch.Tensor:
        """Predict per-position boundary logits.

        Args:
            nt_emb (torch.Tensor): Input nucleotide embedding of shape
                ``(batch_size, seq_len, hidden_dim)``.

        Returns:
            torch.Tensor: Boundary logits of shape
                ``(batch_size, seq_len)``.

        Raises:
            NotImplementedError: For ``chunker_style='hnet'``, which has
                probabilities, not logits.
        """
        if self.chunker_style == "hnet":
            raise NotImplementedError(
                "chunker_style='hnet' has no logits; read "
                "ChunkerOutputs.boundary_prob from forward()."
            )
        # Encoder returns (boundary_features, encoder_residual). For
        # logit-only callers (legacy / debug), the residual stream is
        # discarded — full ``forward()`` surfaces it via ChunkerOutputs.
        boundary_features, _ = self.encoder(nt_emb)
        logits = self.proj(boundary_features).squeeze(-1)
        return logits

    def forward(
        self,
        nt_emb: torch.Tensor,
        symbols: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> ChunkerOutputs:
        """Run the dynamic chunker forward pass.

        Args:
            nt_emb (torch.Tensor): Input nucleotide embedding of shape
                ``(batch_size, seq_len, hidden_dim)``.
            symbols (torch.Tensor | None): ``(batch_size, seq_len)`` int64 nt
                symbols for an encoder with an N-gram embedding; None
                otherwise.
            valid_mask (torch.Tensor | None): ``(batch_size, seq_len)`` bool,
                True on real positions of a padded inner-stage input. Read by
                the ``hnet`` style only; ``legacy`` ignores it.

        Returns:
            ChunkerOutputs: Named-key bag with fields ``chunk_seq``,
                ``boundary_indices``, ``token_mask``, ``mean_chunk_rate``
                (F, hard mask average), ``mean_boundary_prob``
                (G, soft probability average), and ``encoder_residual``
                (Linear-projected encoder hidden, separation-of-two-
                streams stream that bypasses the chunker / main backbone
                and is summed at the dechunker side; ``None`` when the
                encoder does not produce one — e.g. ``conv1d`` encoder
                or ``enable_encoder_residual=False``). See
                ``ChunkerOutputs`` for shape and gradient details.
        """
        if self.chunker_style == "hnet":
            return self._forward_hnet(nt_emb, symbols, valid_mask)
        # Run the encoder once; it returns both the boundary feature
        # bag (for logits) and the separation-of-two-streams residual
        # tensor (for the dechunker side).
        boundary_features, encoder_residual = (
            self.encoder(nt_emb) if symbols is None else self.encoder(nt_emb, symbols)
        )
        logits = self.proj(boundary_features).squeeze(-1)

        # Detach logit at position 0 so the forced boundary carries no
        # gradient through the boundary predictor (DD-014).
        position_0_mask = torch.zeros_like(logits, dtype=torch.bool)
        position_0_mask[:, 0] = True
        logits_safe = torch.where(position_0_mask, logits.detach(), logits)

        if self.boundary_mode == "fixed":
            # Deterministic uniform grid (H-Net fixbound control). Content-
            # independent: the encoder still ran above (residual stream +
            # FLOP parity with the learned arm) but its logits do not drive
            # boundary placement here.
            boundary_mask_float = self._fixed_grid_mask(logits_safe)
        elif self.training:
            if self.boundary_decision == "deterministic_ste":
                boundary_mask_float = self._threshold_ste(logits_safe)
            else:
                boundary_mask_float = self._sample_gumbel_sigmoid(logits_safe)
        else:
            boundary_mask_float = self._threshold(logits_safe)

        # Force position 0 to be a boundary; gradient is already detached
        # at this position via ``logits_safe`` above.
        boundary_mask_float = torch.where(
            position_0_mask,
            torch.ones_like(boundary_mask_float),
            boundary_mask_float,
        )
        boundary_mask = boundary_mask_float.bool()

        boundary_indices, token_mask = self._extract_boundary_indices(boundary_mask)
        chunk_seq = self._gather_chunk_tokens(nt_emb, boundary_indices, token_mask)

        # F = mean hard boundary rate over all positions. The forced
        # position-0 contribution is a constant 1/L with detached
        # gradient (via ``logits_safe``), so gradient w.r.t. predictor
        # params reflects only the data-driven decisions.
        mean_chunk_rate = boundary_mask_float.mean()

        # G = mean soft boundary probability sigmoid(logits) over all
        # positions. Required as the differentiable signal of the H-Net
        # ratio loss (paper eq. 10). Using ``logits_safe`` keeps the
        # position-0 detach contract identical to F.
        mean_boundary_prob = torch.sigmoid(logits_safe).mean()

        return ChunkerOutputs(
            chunk_seq=chunk_seq,
            boundary_indices=boundary_indices,
            token_mask=token_mask,
            mean_chunk_rate=mean_chunk_rate,
            mean_boundary_prob=mean_boundary_prob,
            encoder_residual=encoder_residual,
        )

    def _forward_hnet(
        self,
        nt_emb: torch.Tensor,
        symbols: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> ChunkerOutputs:
        """H-Net routing and chunking (``chunker_style='hnet'``).

        Follows goombalab/hnet ``HNet.forward`` up to the main network:
        encoder, residual projection, ``RoutingModule``, ``ChunkLayer``.

        Args:
            nt_emb (torch.Tensor): ``(B, L, D)`` stage input.
            symbols (torch.Tensor | None): N-gram symbols or None.
            valid_mask (torch.Tensor | None): ``(B, L)`` bool or None.

        Returns:
            ChunkerOutputs: With ``boundary_prob`` and ``selected_probs``.
        """
        pad_mask = None if valid_mask is None else ~valid_mask
        encoder_hidden, encoder_residual = (
            self.encoder(nt_emb, pad_mask=pad_mask)
            if symbols is None
            else self.encoder(nt_emb, symbols, pad_mask=pad_mask)
        )
        prob = hnet_boundary_prob(encoder_hidden, self.router_q, self.router_k)
        if self.boundary_mode == "fixed":
            boundary_mask = self._fixed_grid_mask(prob).bool()
            if valid_mask is not None:
                boundary_mask = boundary_mask & valid_mask
        else:
            boundary_mask = hnet_boundary_mask(prob, valid_mask)
        boundary_indices, token_mask = self._extract_boundary_indices(boundary_mask)
        chunk_seq = self._gather_chunk_tokens(
            encoder_hidden, boundary_indices, token_mask
        )
        mean_chunk_rate, mean_boundary_prob = hnet_rate_stats(
            boundary_mask, prob, valid_mask
        )
        # A fixed-grid boundary was not chosen by the router, and its p is
        # often below 0.5 -- outside anything the EMA saw in training, where
        # it would mostly carry the previous chunk forward. Fixed-grid and
        # unsmoothed readouts therefore enter the dechunker with p = 1 at
        # every boundary (no smoothing), so the two differ only in placement.
        if self.boundary_mode == "fixed" or not self.hnet_smoothing:
            prob_out = torch.where(boundary_mask, torch.ones_like(prob), prob)
        else:
            prob_out = prob
        return ChunkerOutputs(
            chunk_seq=chunk_seq,
            boundary_indices=boundary_indices,
            token_mask=token_mask,
            mean_chunk_rate=mean_chunk_rate,
            mean_boundary_prob=mean_boundary_prob,
            encoder_residual=encoder_residual,
            boundary_prob=prob_out,
            selected_probs=torch.where(boundary_mask, prob_out, 1.0 - prob_out),
        )

    def _sample_gumbel_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        """Draw straight-through Gumbel-sigmoid samples.

        Args:
            logits (torch.Tensor): Per-position logits of shape
                ``(batch_size, seq_len)``.

        Returns:
            torch.Tensor: Hard 0/1 mask of shape
                ``(batch_size, seq_len)`` with straight-through gradient
                via the soft sigmoid.
        """
        u = torch.rand_like(logits).clamp(
            min=_GUMBEL_NOISE_EPS, max=1.0 - _GUMBEL_NOISE_EPS
        )
        gumbel_noise = torch.log(u) - torch.log(1.0 - u)
        soft = torch.sigmoid((logits + gumbel_noise) / self.gumbel_temperature)
        hard = (soft > 0.5).float()
        return hard + (soft - soft.detach())

    def _threshold_ste(self, logits: torch.Tensor) -> torch.Tensor:
        """Deterministic thresholding with a straight-through gradient.

        This is H-Net's own rule (Hwang et al. 2025): the boundary decision is
        the same hard threshold at training and at inference, and gradients
        reach the predictor through the soft probability.

        Why it matters: under Gumbel sampling a CONSTANT logit ``c`` already
        satisfies any target rate, because the injected logistic noise supplies
        the per-position variation (``F = G = sigmoid(c)``). The ratio loss is
        then fully satisfied by a predictor that ignores its input. Under this
        deterministic rule a constant logit can only yield ``F in {0, 1}``, so
        hitting a fractional target rate *requires* position-dependent logits.

        Args:
            logits (torch.Tensor): Per-position logits of shape
                ``(batch_size, seq_len)``.

        Returns:
            torch.Tensor: Hard 0/1 mask of shape ``(batch_size, seq_len)``
                carrying a straight-through gradient via the soft sigmoid.
        """
        soft = torch.sigmoid(logits)
        hard = (soft > self.inference_threshold).float()
        return hard + (soft - soft.detach())

    def _threshold(self, logits: torch.Tensor) -> torch.Tensor:
        """Deterministic hard thresholding for inference.

        Args:
            logits (torch.Tensor): Per-position logits of shape
                ``(batch_size, seq_len)``.

        Returns:
            torch.Tensor: Hard 0/1 mask of shape
                ``(batch_size, seq_len)``.
        """
        return (torch.sigmoid(logits) > self.inference_threshold).float()

    def _fixed_grid_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Content-independent uniform boundary grid (H-Net fixbound).

        Places a boundary every ``stride = round(1 / target_chunk_rate)``
        positions (stride >= 1), matching the learned arm's target chunk
        rate so the compressed sequence length — and thus the main
        backbone's FLOP budget — is identical. Position 0 is always a
        boundary (``arange % stride == 0`` includes index 0), guaranteeing
        ``K >= 1`` for every sample. The grid does not depend on
        ``logits`` (only its shape/device/dtype), so no gradient flows to
        the boundary predictor; pair with ``boundary_loss_weight=0``.

        Args:
            logits (torch.Tensor): Per-position logits of shape
                ``(batch_size, seq_len)`` (used only for shape / device /
                dtype).

        Returns:
            torch.Tensor: Hard 0/1 mask of shape
                ``(batch_size, seq_len)``.
        """
        seq_len = logits.shape[1]
        stride = max(1, round(1.0 / self.target_chunk_rate))
        positions = torch.arange(seq_len, device=logits.device)
        row = (positions % stride == 0).to(logits.dtype)
        return row.unsqueeze(0).expand_as(logits)

    @staticmethod
    def _extract_boundary_indices(
        boundary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract padded boundary positions from a boolean mask.

        For each row, the True positions are collected in ascending
        order and packed into a tensor of width ``k_max`` (the maximum
        count across the batch). Trailing slots are filled with 0 and
        the accompanying boolean ``token_mask`` marks valid positions.

        Args:
            boundary_mask (torch.Tensor): Boolean tensor of shape
                ``(batch_size, seq_len)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``boundary_indices``: int64 of shape
                  ``(batch_size, k_max)``.
                - ``token_mask``: bool of shape
                  ``(batch_size, k_max)``.
        """
        batch_size, seq_len = boundary_mask.shape
        device = boundary_mask.device

        counts = boundary_mask.sum(dim=1).long()
        # ``k_max`` is at least 1 because position 0 is forced True for
        # every sample by the caller. The defensive branch is kept for
        # robustness against unexpected callers passing all-False masks.
        if batch_size == 0 or int(counts.max().item()) == 0:
            return (
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
            )
        k_max = int(counts.max().item())

        # Vectorised scatter: assign each True position a contiguous rank
        # within its row, then scatter the absolute position to that
        # column. Invalid (False) positions scatter to a dump column at
        # index ``k_max`` which is then discarded.
        rank = boundary_mask.long().cumsum(dim=1) - 1
        safe_rank = torch.where(
            boundary_mask,
            rank,
            torch.full_like(rank, k_max),
        )
        positions = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        padded = torch.zeros((batch_size, k_max + 1), dtype=torch.long, device=device)
        padded.scatter_(dim=1, index=safe_rank, src=positions)
        boundary_indices = padded[:, :k_max]

        arange_k = torch.arange(k_max, device=device).unsqueeze(0)
        token_mask = arange_k < counts.unsqueeze(1)

        return boundary_indices, token_mask

    @staticmethod
    def _gather_chunk_tokens(
        nt_emb: torch.Tensor,
        boundary_indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather chunk tokens via boundary-position selection.

        Args:
            nt_emb (torch.Tensor): Nucleotide embeddings of shape
                ``(batch_size, seq_len, hidden_dim)``.
            boundary_indices (torch.Tensor): int64 of shape
                ``(batch_size, k_max)``.
            token_mask (torch.Tensor): bool of shape
                ``(batch_size, k_max)``.

        Returns:
            torch.Tensor: Chunk-level tokens of shape
                ``(batch_size, k_max, hidden_dim)`` with padding
                positions zeroed out.
        """
        hidden_dim = nt_emb.size(-1)
        index = boundary_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        chunk_seq = torch.gather(nt_emb, dim=1, index=index)
        chunk_seq = chunk_seq * token_mask.unsqueeze(-1).to(chunk_seq.dtype)
        return chunk_seq


class Dechunker(nn.Module):
    """Restore chunk-level hidden states to nucleotide-level resolution.

    Three restoration modes are supported:

    - ``broadcast`` (default, v1): Each nucleotide position receives the
      hidden state of the chunk it falls within (literal copy). No
      learnable parameters. All positions within the same chunk share an
      identical backbone summary — adequate when the downstream path
      (hybrid backbone, position embedding, classification head) can
      differentiate within a chunk, but limiting for pure 1 bp annotation
      tasks.

    - ``linear_interpolation`` (v2): Each nucleotide position blends the
      current chunk's hidden state with the next chunk's hidden state by
      a relative-position weight ``alpha = (l - boundary_k) /
      (boundary_{k+1} - boundary_k)`` ∈ [0, 1). Yields a smooth gradient
      along the sequence and a unique feature per nt position. No
      learnable parameters. The last chunk degenerates to the broadcast
      output (no successor to interpolate toward).

    - ``position_offset`` (v2): Adds a learnable per-position offset
      embedding to the broadcast output. The offset is keyed by
      ``min(l - boundary_k, max_within_chunk_offset - 1)``. Tiny
      parameter cost (``max_within_chunk_offset × hidden_dim``) while
      directly giving the head distinct features at every nt position.

    - ``windowed_attention`` (v3, 2026-05-07): Multi-head cross-attention
      from per-position queries (drawn from the input nucleotide
      embedding) to a local window of chunk-level keys/values. Each
      nucleotide attends to its containing chunk plus
      ``±attention_window_size`` neighboring chunks (window size
      ``2W + 1``). Out-of-range or padded chunks are masked. Memory
      stays linear in sequence length: scores tensor is
      ``[B, H, L, 2W+1]`` rather than the prohibitive ``[B, H, L, K]``
      of full attention. Most expressive of the four modes for 1 bp
      annotation tasks where each position needs a distinct, refined
      representation.

    Attributes:
        dechunker_type (str): The selected restoration mode.
        max_within_chunk_offset (int | None): For ``position_offset``,
            the maximum within-chunk index that gets a unique embedding.
            Larger offsets clamp to ``max_within_chunk_offset - 1``.
        position_offset_embedding (nn.Embedding | None): Learnable bias
            for the ``position_offset`` mode (else ``None``).
        attention_window_size (int | None): For ``windowed_attention``,
            half-window of chunks (each side).
        attention_num_heads (int | None): For ``windowed_attention``,
            number of attention heads.
        q_proj / k_proj / v_proj / o_proj (nn.Linear | None): Linear
            projections used by ``windowed_attention``.
    """

    _SUPPORTED_TYPES: tuple[str, ...] = (
        "broadcast",
        "linear_interpolation",
        "position_offset",
        "windowed_attention",
        "mamba2",
    )

    def __init__(
        self,
        dechunker_type: str = "broadcast",
        hidden_dim: int | None = None,
        max_within_chunk_offset: int = 32,
        attention_window_size: int = 8,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
        mamba2_num_layers: int = 4,
        mamba2_d_state: int = 64,
        mamba2_d_conv: int = 4,
        mamba2_expand: int = 2,
        mamba2_headdim: int = 64,
        transformer_num_layers: int = 0,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.1,
        chunker_style: str = "legacy",
    ) -> None:
        """Initialize the dechunker.

        Args:
            dechunker_type (str): Restoration mode. One of
                ``broadcast``, ``linear_interpolation``,
                ``position_offset``, ``windowed_attention``.
            hidden_dim (int | None): Hidden dimensionality D of the
                chunk hidden states; required for ``position_offset``
                and ``windowed_attention``.
            max_within_chunk_offset (int): For ``position_offset``,
                the maximum unique within-chunk position that gets its
                own learnable bias. Defaults to 32 (covers chunks up to
                32 nt; larger offsets share the last bin's embedding).
            attention_window_size (int): For ``windowed_attention``,
                half-window of chunks (each side). Total window is
                ``2 * attention_window_size + 1``. Defaults to 8
                (window of 17 chunks). Must be ``>= 0``.
            attention_num_heads (int): For ``windowed_attention``,
                number of attention heads. ``hidden_dim`` must be
                divisible by this. Defaults to 4.
            attention_dropout (float): For ``windowed_attention``,
                dropout probability applied to attention weights during
                training. Defaults to 0.0 (no dropout).
            mamba2_num_layers (int): For ``mamba2`` dechunker, number
                of stacked Mamba-2 SSM blocks applied to the broadcast
                output. Defaults to 4 (matches H-Net paper depth).
            mamba2_d_state (int): For ``mamba2`` dechunker, SSM state
                dim per Mamba-2 block. Defaults to 64.
            mamba2_d_conv (int): For ``mamba2`` dechunker, internal
                conv kernel size in each Mamba-2 block. Defaults to 4.
            mamba2_expand (int): For ``mamba2`` dechunker, channel
                expansion factor inside Mamba-2. Defaults to 2.
            mamba2_headdim (int): For ``mamba2`` dechunker, SSD head
                dim. Must divide ``mamba2_expand * hidden_dim``.
                Defaults to 64.
            transformer_num_layers (int): For ``mamba2`` dechunker,
                number of Transformer encoder layers stacked after the
                Mamba-2 refinement. dnaHNet paper uses ``M4T1`` per
                stage. 0 (default) = pure Mamba (backward-compat).
            transformer_num_heads (int): For ``mamba2`` dechunker,
                Transformer attention heads. Must divide ``hidden_dim``.
                Defaults to 4.
            transformer_dropout (float): For ``mamba2`` dechunker,
                Transformer dropout probability. Defaults to 0.1.
            chunker_style (str): ``legacy`` (default) or ``hnet``. ``hnet``
                restores like H-Net's ``DeChunkLayer`` plus ``residual_func``:
                an EMA over the chunk sequence weighted by the boundary
                probability, broadcast back, multiplied by the decision
                confidence through a straight-through estimator, plus the
                encoder residual; the SSM / Transformer stack then plays
                H-Net's decoder, with padding masked in the Transformer
                layers. Requires ``mamba2``.

        Raises:
            ValueError: If ``dechunker_type`` is unknown, or if a mode
                is selected without its required arguments, or if
                ``hidden_dim`` is not divisible by
                ``attention_num_heads``, or if dropout is out of range.
            ImportError: If ``dechunker_type='mamba2'`` but the optional
                ``mamba_ssm`` package is not installed.
        """
        super().__init__()
        if dechunker_type == "attention":
            raise NotImplementedError(
                "dechunker_type='attention' (full O(B*H*L*K) cross-"
                "attention) is memory-prohibitive at training scale. "
                "Use 'windowed_attention' instead — each nucleotide "
                "position attends to ±attention_window_size neighboring "
                "chunks for linear scaling. See plan.md DD-012a."
            )
        if dechunker_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown dechunker_type: {dechunker_type!r}. "
                f"Supported: {self._SUPPORTED_TYPES}."
            )

        self.dechunker_type: str = dechunker_type
        if chunker_style not in _CHUNKER_STYLES:
            raise ValueError(
                f"Unknown chunker_style: {chunker_style!r}. Supported: "
                f"{_CHUNKER_STYLES}."
            )
        if chunker_style == "hnet":
            if dechunker_type not in ("mamba2",):
                raise ValueError(
                    "chunker_style='hnet' needs dechunker_type 'mamba2' or "
                    f"(H-Net's decoder stack); got {dechunker_type!r}."
                )
            if hidden_dim is None or int(hidden_dim) % _HNET_EMA_HEADDIM != 0:
                raise ValueError(
                    f"chunker_style='hnet' needs hidden_dim divisible by "
                    f"{_HNET_EMA_HEADDIM} (EMA kernel head dim); got {hidden_dim}."
                )
        self.chunker_style: str = chunker_style
        self.max_within_chunk_offset: int | None = None
        self.position_offset_embedding: nn.Embedding | None = None
        self.attention_window_size: int | None = None
        self.attention_num_heads: int | None = None
        self.attention_head_dim: int | None = None
        self.attention_dropout: float = 0.0
        self.q_proj: nn.Linear | None = None
        self.k_proj: nn.Linear | None = None
        self.v_proj: nn.Linear | None = None
        self.o_proj: nn.Linear | None = None
        self.mamba2_layers: nn.ModuleList | None = None
        self.mamba2_layer_norms: nn.ModuleList | None = None
        self.mamba2_output_norm: nn.RMSNorm | None = None
        self.mamba2_transformer_layers: nn.ModuleList | None = None

        if dechunker_type == "position_offset":
            if hidden_dim is None:
                raise ValueError(
                    "hidden_dim is required when dechunker_type='position_offset'."
                )
            if max_within_chunk_offset < 1:
                raise ValueError(
                    "max_within_chunk_offset must be >= 1; got "
                    f"{max_within_chunk_offset}."
                )
            self.max_within_chunk_offset = int(max_within_chunk_offset)
            self.position_offset_embedding = nn.Embedding(
                num_embeddings=self.max_within_chunk_offset,
                embedding_dim=int(hidden_dim),
            )
            # Initialize to zero so the layer starts as a no-op
            # (== broadcast) before training adapts it.
            nn.init.zeros_(self.position_offset_embedding.weight)

        if dechunker_type == "windowed_attention":
            if hidden_dim is None:
                raise ValueError(
                    "hidden_dim is required when dechunker_type='windowed_attention'."
                )
            if attention_window_size < 0:
                raise ValueError(
                    f"attention_window_size must be >= 0; got {attention_window_size}."
                )
            if attention_num_heads < 1:
                raise ValueError(
                    f"attention_num_heads must be >= 1; got {attention_num_heads}."
                )
            if int(hidden_dim) % int(attention_num_heads) != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by "
                    f"attention_num_heads ({attention_num_heads})."
                )
            if not 0.0 <= attention_dropout < 1.0:
                raise ValueError(
                    f"attention_dropout must be in [0, 1); got {attention_dropout}."
                )
            self.attention_window_size = int(attention_window_size)
            self.attention_num_heads = int(attention_num_heads)
            self.attention_head_dim = int(hidden_dim) // int(attention_num_heads)
            self.attention_dropout = float(attention_dropout)
            self.q_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.k_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.v_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.o_proj = nn.Linear(int(hidden_dim), int(hidden_dim))

        if dechunker_type in ("mamba2",):
            if hidden_dim is None:
                raise ValueError(
                    f"hidden_dim is required when dechunker_type='{dechunker_type}'."
                )
            if mamba2_num_layers < 1:
                raise ValueError(
                    f"mamba2_num_layers must be >= 1; got {mamba2_num_layers}."
                )
            d_inner = int(mamba2_expand) * int(hidden_dim)
            if d_inner % int(mamba2_headdim) != 0:
                raise ValueError(
                    f"{dechunker_type} requires expand*hidden_dim ({d_inner}) to be "
                    f"divisible by mamba2_headdim ({mamba2_headdim}). Reduce "
                    f"mamba2_headdim or increase hidden_dim/mamba2_expand."
                )
            # The ``mamba2_*`` attribute names predate the Mamba-3 option
            # and are kept so existing checkpoints still load by key.
            self.mamba2_layers = _build_ssm_layers(
                ssm_variant=dechunker_type,
                num_layers=mamba2_num_layers,
                hidden_dim=hidden_dim,
                d_state=mamba2_d_state,
                d_conv=mamba2_d_conv,
                expand=mamba2_expand,
                headdim=mamba2_headdim,
            )
            # Pre-norm RMSNorm interleaved between Mamba-2 layers
            # (H-Net paper Section 3.2). One RMSNorm per layer + a
            # final output norm before returning to nt-level caller.
            self.mamba2_layer_norms = nn.ModuleList(
                [nn.RMSNorm(int(hidden_dim)) for _ in range(int(mamba2_num_layers))]
            )
            self.mamba2_output_norm = nn.RMSNorm(int(hidden_dim))
            # dnaHNet hybrid: optional Transformer layers after the
            # Mamba-2 dechunker stack. Each TransformerEncoderLayer with
            # ``norm_first=True`` carries its own pre-norm + residual.
            if int(transformer_num_layers) > 0:
                if int(hidden_dim) % int(transformer_num_heads) != 0:
                    raise ValueError(
                        f"transformer_num_heads ({transformer_num_heads}) "
                        f"must divide hidden_dim ({hidden_dim})."
                    )
                self.mamba2_transformer_layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=int(hidden_dim),
                            nhead=int(transformer_num_heads),
                            dim_feedforward=4 * int(hidden_dim),
                            dropout=float(transformer_dropout),
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(int(transformer_num_layers))
                    ]
                )

    def forward(
        self,
        chunk_hidden: torch.Tensor,
        boundary_indices: torch.Tensor,
        token_mask: torch.Tensor,
        output_length: int,
        nt_emb: torch.Tensor | None = None,
        encoder_residual: torch.Tensor | None = None,
        boundary_prob: torch.Tensor | None = None,
        selected_probs: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore chunk hidden states to nucleotide-level positions.

        Args:
            chunk_hidden (torch.Tensor): Chunk-level hidden states of
                shape ``(batch_size, k_max, hidden_dim)``.
            boundary_indices (torch.Tensor): Boundary positions in
                nucleotide space, shape ``(batch_size, k_max)`` int64.
            token_mask (torch.Tensor): bool of shape
                ``(batch_size, k_max)`` — True for valid chunks.
            output_length (int): Target nucleotide-level length L.
            nt_emb (torch.Tensor | None): Per-position nucleotide
                embedding of shape ``(batch_size, output_length,
                hidden_dim)``, used as the query source for
                ``windowed_attention``. Required when
                ``dechunker_type == 'windowed_attention'``; ignored
                otherwise.
            encoder_residual (torch.Tensor | None): H-Net "separation
                of two streams" residual stream of shape
                ``(batch_size, output_length, hidden_dim)`` produced by
                the chunker's encoder ``residual_proj`` (see
                ``ChunkerOutputs.encoder_residual``). When provided and
                ``dechunker_type == 'mamba2'``, summed into the
                broadcast output before the dechunker's own Mamba-2
                refinement layers, letting nt-level encoder
                representations bypass the chunker / main backbone.
                Other dechunker modes ignore it for now.
            boundary_prob (torch.Tensor | None): ``(batch_size,
                output_length)`` boundary probability of this stage's
                chunker; required for ``chunker_style='hnet'``.
            selected_probs (torch.Tensor | None): ``(batch_size,
                output_length)`` decision confidence; required for ``hnet``.
            valid_mask (torch.Tensor | None): ``(batch_size,
                output_length)`` bool, True on real positions of a padded
                level; ``hnet`` masks the rest in its Transformer layers.

        Returns:
            torch.Tensor: Nucleotide-level hidden states of shape
                ``(batch_size, output_length, hidden_dim)``.

        Raises:
            ValueError: If ``windowed_attention`` is selected but
                ``nt_emb`` is None or has the wrong shape.
        """
        chunk_idx, sorted_boundaries = self._compute_assignment(
            boundary_indices, token_mask, output_length, chunk_hidden.shape[1]
        )
        if self.chunker_style == "hnet":
            if boundary_prob is None or selected_probs is None:
                raise ValueError(
                    "chunker_style='hnet' needs boundary_prob and "
                    "selected_probs from the paired chunker."
                )
            return self._hnet(
                chunk_hidden,
                boundary_indices,
                chunk_idx,
                boundary_prob,
                selected_probs,
                encoder_residual,
                valid_mask,
            )
        if self.dechunker_type == "broadcast":
            return self._broadcast(chunk_hidden, chunk_idx)
        if self.dechunker_type == "linear_interpolation":
            return self._linear_interpolation(
                chunk_hidden, chunk_idx, sorted_boundaries, output_length
            )
        if self.dechunker_type == "position_offset":
            return self._position_offset(
                chunk_hidden, chunk_idx, sorted_boundaries, output_length
            )
        if self.dechunker_type in ("mamba2",):
            return self._mamba2(
                chunk_hidden, chunk_idx, encoder_residual=encoder_residual
            )
        # windowed_attention
        if nt_emb is None:
            raise ValueError(
                "nt_emb is required when dechunker_type='windowed_attention'."
            )
        if nt_emb.shape[:2] != (chunk_hidden.shape[0], output_length):
            raise ValueError(
                "nt_emb must have shape (batch_size, output_length, "
                f"hidden_dim); got {tuple(nt_emb.shape)} with expected "
                f"first two dims ({chunk_hidden.shape[0]}, "
                f"{output_length})."
            )
        return self._windowed_attention(
            chunk_hidden=chunk_hidden,
            chunk_idx=chunk_idx,
            token_mask=token_mask,
            nt_emb=nt_emb,
        )

    @staticmethod
    def _compute_assignment(
        boundary_indices: torch.Tensor,
        token_mask: torch.Tensor,
        output_length: int,
        k_max: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-position chunk assignment via searchsorted.

        Args:
            boundary_indices (torch.Tensor): (B, K_max) int64.
            token_mask (torch.Tensor): (B, K_max) bool.
            output_length (int): Target nucleotide-level length L.
            k_max (int): Max number of chunks (``chunk_hidden.shape[1]``).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``chunk_idx``: (B, L) int64, the chunk index assigned
                  to each nt position, clamped to [0, K_max - 1].
                - ``sorted_boundaries``: (B, K_max) int64 with invalid
                  chunks pushed past ``output_length`` so searchsorted
                  never picks them.
        """
        batch_size = boundary_indices.shape[0]
        device = boundary_indices.device

        sentinel = output_length + 1
        masked_boundaries = torch.where(
            token_mask,
            boundary_indices,
            torch.full_like(boundary_indices, sentinel),
        )
        sorted_boundaries, _ = torch.sort(masked_boundaries, dim=1)

        positions = (
            torch.arange(output_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, output_length)
            .contiguous()
        )
        chunk_idx = (
            torch.searchsorted(sorted_boundaries, positions, right=True) - 1
        ).clamp(min=0, max=k_max - 1)
        return chunk_idx, sorted_boundaries

    @staticmethod
    def _broadcast(
        chunk_hidden: torch.Tensor,
        chunk_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Copy each chunk's hidden state to all member positions.

        Args:
            chunk_hidden (torch.Tensor): (B, K_max, D).
            chunk_idx (torch.Tensor): (B, L) int64.

        Returns:
            torch.Tensor: (B, L, D) with within-chunk positions sharing
                identical features.
        """
        hidden_dim = chunk_hidden.size(-1)
        index = chunk_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return torch.gather(chunk_hidden, dim=1, index=index)

    @staticmethod
    def _linear_interpolation(
        chunk_hidden: torch.Tensor,
        chunk_idx: torch.Tensor,
        sorted_boundaries: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        """Blend each position with its current and next chunk hidden.

        For a position l in chunk k, the output is
        ``(1 - alpha) * h_k + alpha * h_{k+1}`` where
        ``alpha = (l - b_k) / (b_{k+1} - b_k)`` and ``b_*`` are the
        sorted boundary positions. The last chunk falls back to broadcast
        (no successor to interpolate toward, ``alpha = 0``).

        Args:
            chunk_hidden (torch.Tensor): (B, K_max, D).
            chunk_idx (torch.Tensor): (B, L) int64.
            sorted_boundaries (torch.Tensor): (B, K_max) int64 with
                invalid entries pushed past ``output_length``.
            output_length (int): L.

        Returns:
            torch.Tensor: (B, L, D).
        """
        batch_size, k_max, hidden_dim = chunk_hidden.shape
        device = chunk_hidden.device

        next_chunk_idx = (chunk_idx + 1).clamp(max=k_max - 1)
        chunk_start = torch.gather(sorted_boundaries, dim=1, index=chunk_idx)
        next_start_raw = torch.gather(sorted_boundaries, dim=1, index=next_chunk_idx)
        # When next chunk is invalid (sentinel) or equal to current
        # chunk (last valid chunk), use the sequence end as the upper
        # bound so alpha stays in [0, 1) and degenerates cleanly.
        next_start = torch.where(
            next_start_raw > output_length,
            torch.full_like(next_start_raw, output_length),
            next_start_raw,
        )

        positions = (
            torch.arange(output_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, output_length)
        )
        chunk_size = (next_start - chunk_start).clamp(min=1)
        alpha = ((positions - chunk_start).float() / chunk_size.float()).clamp(0.0, 1.0)
        # Last chunk: chunk_idx == next_chunk_idx -> alpha forced to 0
        # so we copy h_curr (broadcast fallback).
        no_next = chunk_idx == next_chunk_idx
        alpha = torch.where(no_next, torch.zeros_like(alpha), alpha)

        index_curr = chunk_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        index_next = next_chunk_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        h_curr = torch.gather(chunk_hidden, dim=1, index=index_curr)
        h_next = torch.gather(chunk_hidden, dim=1, index=index_next)

        alpha_exp = alpha.unsqueeze(-1).to(h_curr.dtype)
        return (1.0 - alpha_exp) * h_curr + alpha_exp * h_next

    def _position_offset(
        self,
        chunk_hidden: torch.Tensor,
        chunk_idx: torch.Tensor,
        sorted_boundaries: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        """Broadcast plus a learned within-chunk position bias.

        Args:
            chunk_hidden (torch.Tensor): (B, K_max, D).
            chunk_idx (torch.Tensor): (B, L) int64.
            sorted_boundaries (torch.Tensor): (B, K_max) int64.
            output_length (int): L.

        Returns:
            torch.Tensor: (B, L, D) — broadcast + learned offset.
        """
        if self.position_offset_embedding is None:
            raise RuntimeError(
                "position_offset_embedding was not initialized; this "
                "indicates a corrupted Dechunker state."
            )
        device = chunk_hidden.device
        batch_size = chunk_hidden.shape[0]

        broadcast_out = self._broadcast(chunk_hidden, chunk_idx)
        # Within-chunk offset = current position - chunk start, clamped
        # to the embedding's range.
        chunk_start = torch.gather(sorted_boundaries, dim=1, index=chunk_idx)
        positions = (
            torch.arange(output_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, output_length)
        )
        within_chunk_offset = (positions - chunk_start).clamp(
            min=0, max=int(self.max_within_chunk_offset) - 1
        )
        offset_emb = self.position_offset_embedding(within_chunk_offset)
        return broadcast_out + offset_emb.to(broadcast_out.dtype)

    def _mamba2(
        self,
        chunk_hidden: torch.Tensor,
        chunk_idx: torch.Tensor,
        encoder_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Broadcast plus stacked Mamba-2 SSM refinement at nt resolution.

        Following H-Net (Hwang et al., 2025), the dechunker first
        broadcasts each chunk's hidden state to all member positions,
        then applies a stack of Mamba-2 SSM blocks (with residual
        connections) at the full nucleotide resolution. The Mamba-2
        layers learn to refine the broadcasted features into per-nt
        outputs using the SSD algorithm's O(L) long-range context.

        When ``encoder_residual`` is provided (H-Net "separation of
        two streams"), it is summed into the broadcast output BEFORE
        the Mamba-2 refinement, so the nt-level encoder representations
        re-enter the decoder side without going through the chunker /
        main backbone bottleneck.

        Args:
            chunk_hidden (torch.Tensor): ``(B, K_max, D)``.
            chunk_idx (torch.Tensor): ``(B, L)`` int64.
            encoder_residual (torch.Tensor | None): ``(B, L, D)``
                encoder residual stream from
                ``ChunkerOutputs.encoder_residual``. When None (no
                residual stream from encoder), behaviour is unchanged.

        Returns:
            torch.Tensor: ``(B, L, D)``.
        """
        if self.mamba2_layers is None or self.mamba2_layer_norms is None:
            raise RuntimeError(
                "mamba2_layers were not initialized; this indicates a "
                "corrupted Dechunker state."
            )
        # Step 1: broadcast chunk hidden states to nt-level positions.
        x = self._broadcast(chunk_hidden, chunk_idx)
        # Step 1b: H-Net separation-of-two-streams residual injection.
        # Encoder residual carries nt-level features that bypassed the
        # chunker; summing here lets the dechunker recover fine-grained
        # local information that was lost during chunk compression.
        if encoder_residual is not None:
            x = x + encoder_residual
        return self._refine(x, pad_mask=None)

    def _refine(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        """Run the SSM stack, the optional Transformer layers and the output norm.

        Args:
            x (torch.Tensor): ``(B, L, D)`` restored hidden states.
            pad_mask (torch.Tensor | None): ``(B, L)`` bool, True on padding;
                passed to the Transformer layers as ``src_key_padding_mask``.

        Returns:
            torch.Tensor: ``(B, L, D)``.
        """
        # Stacked Mamba-2 SSM refinement using pre-norm template (RMSNorm
        # interleaved per H-Net Section 3.2): x = x + Layer(RMSNorm(x)).
        # One RMSNorm per layer + final output norm.
        for layer, norm in zip(
            self.mamba2_layers, self.mamba2_layer_norms, strict=True
        ):
            x = x + layer(norm(x))
        # Optional Transformer hybrid (dnaHNet M{N}T{M}). Self-attention
        # layers come after the SSM stack — paper layout order.
        if self.mamba2_transformer_layers is not None:
            for tlayer in self.mamba2_transformer_layers:
                x = tlayer(x, src_key_padding_mask=pad_mask)
        return self.mamba2_output_norm(x)

    def _hnet(
        self,
        chunk_hidden: torch.Tensor,
        boundary_indices: torch.Tensor,
        chunk_idx: torch.Tensor,
        boundary_prob: torch.Tensor,
        selected_probs: torch.Tensor,
        encoder_residual: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """H-Net ``DeChunkLayer`` and ``residual_func``, then the decoder stack.

        Args:
            chunk_hidden (torch.Tensor): ``(B, K_max, D)`` inner output.
            boundary_indices (torch.Tensor): ``(B, K_max)`` int64, ascending.
            chunk_idx (torch.Tensor): ``(B, L)`` chunk of each position.
            boundary_prob (torch.Tensor): ``(B, L)`` boundary probability.
            selected_probs (torch.Tensor): ``(B, L)`` decision confidence.
            encoder_residual (torch.Tensor | None): ``(B, L, D)`` float32.
            valid_mask (torch.Tensor | None): ``(B, L)`` bool or None.

        Returns:
            torch.Tensor: ``(B, L, D)``.
        """
        if self.mamba2_layers is None or self.mamba2_layer_norms is None:
            raise RuntimeError(
                "mamba2_layers were not initialized; this indicates a "
                "corrupted Dechunker state."
            )
        prob = boundary_prob.float().clamp(_HNET_P_CLAMP, 1.0 - _HNET_P_CLAMP)
        # Padded chunk slots point at position 0; they sit after the real
        # chunks, so the causal EMA never carries them into a real one.
        p_chunks = torch.gather(prob, dim=1, index=boundary_indices)
        smoothed = hnet_ema(chunk_hidden, p_chunks)
        x = self._broadcast(smoothed, chunk_idx)
        gate = hnet_ste(selected_probs.float()).unsqueeze(-1)
        if encoder_residual is None:
            x = (x.float() * gate).to(x.dtype)
        else:
            x = (x.to(encoder_residual.dtype) * gate + encoder_residual).to(x.dtype)
        return self._refine(x, pad_mask=None if valid_mask is None else ~valid_mask)

    def _windowed_attention(
        self,
        chunk_hidden: torch.Tensor,
        chunk_idx: torch.Tensor,
        token_mask: torch.Tensor,
        nt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head cross-attention from per-position queries to a
        local window of chunk-level keys / values.

        Each nucleotide position l attends to chunk indices in the
        range ``[chunk_idx[l] - W, chunk_idx[l] + W]`` clamped to
        ``[0, K - 1]``. Out-of-range chunks (those that would fall
        outside the valid range before clamping) and padded chunks
        (``token_mask == False``) are masked out of the softmax. The
        position's own containing chunk is always inside the window
        (``W >= 0`` guaranteed at init), so at least one valid target
        exists per position and softmax never collapses to NaN.

        Memory: scores tensor is ``[B, H, L, 2W + 1]`` rather than the
        ``[B, H, L, K]`` of full attention. With W=8, B=32, L=15000,
        H=4, bf16 → 65 MB; full attention with K=7500 would be 28.8 GB.

        Args:
            chunk_hidden (torch.Tensor): ``(B, K_max, D)``.
            chunk_idx (torch.Tensor): ``(B, L)`` int64, the chunk
                each position belongs to (clamped to ``[0, K_max - 1]``
                by ``_compute_assignment``).
            token_mask (torch.Tensor): ``(B, K_max)`` bool.
            nt_emb (torch.Tensor): ``(B, L, D)`` per-position queries.

        Returns:
            torch.Tensor: ``(B, L, D)``.
        """
        if (
            self.q_proj is None
            or self.k_proj is None
            or self.v_proj is None
            or self.o_proj is None
            or self.attention_window_size is None
            or self.attention_num_heads is None
            or self.attention_head_dim is None
        ):
            raise RuntimeError(
                "windowed_attention projections were not initialized; "
                "this indicates a corrupted Dechunker state."
            )
        batch_size, k_max, hidden_dim = chunk_hidden.shape
        seq_len = nt_emb.shape[1]
        num_heads = int(self.attention_num_heads)
        head_dim = int(self.attention_head_dim)
        window_half = int(self.attention_window_size)
        window_size = 2 * window_half + 1
        device = chunk_hidden.device

        # 1. Project Q, K, V once at chunk-level (cheaper than projecting
        # the gathered window which would scale by ``window_size``).
        q = (
            self.q_proj(nt_emb)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )  # [B, H, L, Hd]
        k = (
            self.k_proj(chunk_hidden)
            .view(batch_size, k_max, num_heads, head_dim)
            .transpose(1, 2)
        )  # [B, H, K, Hd]
        v = (
            self.v_proj(chunk_hidden)
            .view(batch_size, k_max, num_heads, head_dim)
            .transpose(1, 2)
        )  # [B, H, K, Hd]

        # 2. Compute window indices for each position. Positions outside
        # ``[0, K_max - 1]`` are clamped (so gather succeeds) and tagged
        # in ``in_range`` for masking.
        offsets = torch.arange(-window_half, window_half + 1, device=device)  # [W_size]
        win_idx_raw = chunk_idx.unsqueeze(-1) + offsets  # [B, L, W_size]
        in_range = (win_idx_raw >= 0) & (win_idx_raw < k_max)
        win_idx = win_idx_raw.clamp(min=0, max=k_max - 1)

        # 3. Gather window-level K and V from per-head projections.
        # K/V have shape [B, H, K, Hd]; we want per-position windows
        # [B, H, L, W_size, Hd]. Flatten ``L * W_size`` for a single
        # gather call.
        flat_idx = win_idx.view(batch_size, seq_len * window_size)  # [B, L*W]
        flat_idx_h = flat_idx.unsqueeze(1).expand(-1, num_heads, -1)
        flat_idx_h_d = flat_idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_win = torch.gather(k, dim=2, index=flat_idx_h_d).view(
            batch_size, num_heads, seq_len, window_size, head_dim
        )
        v_win = torch.gather(v, dim=2, index=flat_idx_h_d).view(
            batch_size, num_heads, seq_len, window_size, head_dim
        )

        # 4. Build the attention mask: invalid window slots are
        # out-of-range or correspond to padded chunks.
        chunk_valid = torch.gather(token_mask.long(), dim=1, index=flat_idx).bool()
        chunk_valid = chunk_valid.view(batch_size, seq_len, window_size)
        valid_mask = in_range & chunk_valid  # [B, L, W_size]

        # 5. Scaled dot-product attention scores per (B, H, L, W_size).
        # ``q.unsqueeze(-2) * k_win`` broadcasts ``q`` across the window
        # dimension; sum over the head_dim axis.
        scores = (q.unsqueeze(-2) * k_win).sum(dim=-1) / (head_dim**0.5)
        # Mask before softmax. ``valid_mask`` broadcasts across heads.
        scores = scores.masked_fill(
            ~valid_mask.unsqueeze(1),
            float("-inf"),
        )

        attn = torch.softmax(scores, dim=-1)
        if self.attention_dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.attention_dropout)

        # 6. Weighted sum of V, then project back to D.
        out = (attn.unsqueeze(-1) * v_win).sum(dim=-2)  # [B, H, L, Hd]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.o_proj(out)


@dataclass
class HierarchicalChunkerOutputs:
    """Outputs from a multi-stage ``HierarchicalChunker.encode`` pass.

    Carries the deepest chunk sequence (fed to the main backbone) plus the
    per-stage state needed to unwind the hierarchy in ``decode``. Stages
    are stored outer-to-inner: ``[0]`` runs at nt resolution, ``[-1]``
    feeds the main backbone.

    Attributes:
        deepest_chunk_seq (torch.Tensor): Innermost chunk sequence of
            shape ``(batch_size, k_inner, hidden_dim)`` to feed to the
            main backbone via ``inputs_embeds``.
        stage_outputs (list[ChunkerOutputs]): Per-stage chunker outputs,
            outer-to-inner. Stage ``i`` holds the boundary indices /
            token mask / mean rates / encoder residual at level ``i``.
        level_inputs (list[torch.Tensor]): Per-level input embeddings,
            outer-to-inner. ``level_inputs[i]`` is the input to stage
            ``i``'s chunker (and equivalently the target output level of
            stage ``i``'s dechunker). ``level_inputs[0]`` is the original
            nt embedding ``(B, L, D)``; ``level_inputs[i]`` for ``i > 0``
            is ``stage_outputs[i-1].chunk_seq``.
        level_masks (list[torch.Tensor | None]): Per-level validity,
            outer-to-inner, aligned with ``level_inputs``. ``[0]`` is None
            (the nt level is never padded); ``[i]`` for ``i > 0`` is
            ``stage_outputs[i-1].token_mask``.
    """

    deepest_chunk_seq: torch.Tensor
    stage_outputs: list[ChunkerOutputs]
    level_inputs: list[torch.Tensor]
    level_masks: list[torch.Tensor | None] | None = None


class HierarchicalChunker(nn.Module):
    """Multi-stage nested dynamic chunker (H-Net hierarchical compression).

    Wraps ``N`` ``(DynamicChunker, Dechunker)`` pairs into a U-Net style
    hierarchy. ``encode`` chunks outer-to-inner (each stage compresses
    the previous level's output further); the deepest chunk sequence is
    handed to the main backbone; ``decode`` unwinds inner-to-outer,
    each stage's dechunker mapping level ``i+1`` hidden states back to
    level ``i`` while consuming that stage's encoder residual stream
    (separation-of-two-streams) at level ``i``.

    Single-stage (``N=1``) is a valid configuration and is the
    backward-compat path used when a flat ``DynamicChunker`` config is
    provided. ``N>=2`` activates the H-Net hierarchical architecture
    (paper Section 3) — typical compression ratios are products of
    per-stage ``target_chunk_rate`` (e.g. ``[0.5, 0.4]`` → 0.2).

    Attributes:
        chunkers (nn.ModuleList): ``DynamicChunker`` per stage,
            outer-to-inner.
        dechunkers (nn.ModuleList): ``Dechunker`` per stage, paired by
            index with ``chunkers``.
        target_chunk_rates (list[float]): Per-stage
            ``target_chunk_rate`` exposed for the boundary regularizer.
        num_stages (int): ``len(chunkers)``.
    """

    def __init__(
        self,
        chunkers: list[DynamicChunker],
        dechunkers: list[Dechunker],
    ) -> None:
        """Initialize from pre-constructed chunker / dechunker lists.

        Args:
            chunkers (list[DynamicChunker]): Per-stage chunkers,
                outer-to-inner. Must be non-empty.
            dechunkers (list[Dechunker]): Per-stage dechunkers, paired
                by index with ``chunkers``. Must have the same length.

        Raises:
            ValueError: If the lists are empty or have mismatched lengths.
        """
        super().__init__()
        if len(chunkers) == 0:
            raise ValueError("HierarchicalChunker requires at least one stage.")
        if len(chunkers) != len(dechunkers):
            raise ValueError(
                "chunkers and dechunkers must have the same length; got "
                f"{len(chunkers)} vs {len(dechunkers)}."
            )
        self.chunkers: nn.ModuleList = nn.ModuleList(chunkers)
        self.dechunkers: nn.ModuleList = nn.ModuleList(dechunkers)
        self.target_chunk_rates: list[float] = [
            float(c.target_chunk_rate) for c in chunkers
        ]

    @property
    def num_stages(self) -> int:
        """Number of nested stages (depth of the hierarchy)."""
        return len(self.chunkers)

    def encode(
        self, nt_emb: torch.Tensor, symbols: torch.Tensor | None = None
    ) -> HierarchicalChunkerOutputs:
        """Run the chunker stack outer-to-inner.

        Args:
            nt_emb (torch.Tensor): Nucleotide-level embedding of shape
                ``(batch_size, seq_len, hidden_dim)``.
            symbols (torch.Tensor | None): ``(batch_size, seq_len)`` int64 nt
                symbols for a stage-0 encoder with an N-gram embedding. Inner
                stages run on chunks and never receive them.

        Returns:
            HierarchicalChunkerOutputs: Bag with ``deepest_chunk_seq``
                (to feed the main backbone), per-stage ``stage_outputs``,
                and ``level_inputs`` (one entry per stage; needed by
                ``decode`` for windowed-attention queries and for
                ``output_length`` per stage).
        """
        level_inputs: list[torch.Tensor] = [nt_emb]
        level_masks: list[torch.Tensor | None] = [None]
        stage_outputs: list[ChunkerOutputs] = []
        current = nt_emb
        for stage_idx, chunker in enumerate(self.chunkers):
            stage_out = chunker(
                current,
                symbols if stage_idx == 0 else None,
                valid_mask=level_masks[-1],
            )
            stage_outputs.append(stage_out)
            current = stage_out.chunk_seq
            level_inputs.append(current)
            level_masks.append(stage_out.token_mask)
        # ``level_inputs`` has length ``num_stages + 1``; we only need
        # the first ``num_stages`` entries (per-stage *input* level) for
        # decode. The trailing entry equals ``deepest_chunk_seq``.
        return HierarchicalChunkerOutputs(
            deepest_chunk_seq=current,
            stage_outputs=stage_outputs,
            level_inputs=level_inputs[:-1],
            level_masks=level_masks[:-1],
        )

    def decode(
        self,
        deepest_hidden: torch.Tensor,
        stage_outputs: list[ChunkerOutputs],
        level_inputs: list[torch.Tensor],
        nt_output_length: int,
        level_masks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Unwind the chunker stack inner-to-outer.

        Args:
            deepest_hidden (torch.Tensor): Backbone output at the deepest
                level, shape ``(batch_size, k_inner, hidden_dim)``.
            stage_outputs (list[ChunkerOutputs]): Per-stage chunker
                outputs from ``encode``.
            level_inputs (list[torch.Tensor]): Per-stage input
                embeddings from ``encode``, used as ``nt_emb`` queries
                by ``windowed_attention`` dechunkers.
            nt_output_length (int): Nucleotide-level target length L.
                Used as ``output_length`` for the outermost dechunker;
                inner dechunkers derive their target from
                ``level_inputs[i].shape[1]``.
            level_masks (list[torch.Tensor | None] | None): Per-level
                validity from ``encode``; the ``hnet`` dechunkers mask
                padding with it. None treats every level as unpadded.

        Returns:
            torch.Tensor: Nucleotide-level hidden states of shape
                ``(batch_size, nt_output_length, hidden_dim)``.
        """
        hidden = deepest_hidden
        for stage_idx in reversed(range(self.num_stages)):
            stage_out = stage_outputs[stage_idx]
            stage_input = level_inputs[stage_idx]
            output_length = (
                nt_output_length if stage_idx == 0 else int(stage_input.shape[1])
            )
            hidden = self.dechunkers[stage_idx](
                chunk_hidden=hidden,
                boundary_indices=stage_out.boundary_indices,
                token_mask=stage_out.token_mask,
                output_length=output_length,
                nt_emb=stage_input,
                encoder_residual=stage_out.encoder_residual,
                boundary_prob=stage_out.boundary_prob,
                selected_probs=stage_out.selected_probs,
                valid_mask=level_masks[stage_idx] if level_masks else None,
            )
        return hidden
