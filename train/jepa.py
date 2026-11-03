"""Building blocks for JEPA-DNA style pretraining.

Implements the components introduced in Larey et al., "JEPA-DNA: Grounding
Genomic Foundation Models through Joint-Embedding Predictive Architectures"
(arXiv:2602.17162):

* ``SpanMaskingCollator`` — span-based masking (20-40%, 1-3 spans).
* ``JEPAPredictor`` — light 3-layer Transformer that predicts the target
  aggregation latent from the context-encoded sequence (with re-masking).
* ``aggregate_hidden_states`` — pooling helper for cls/eos/mean tokens.
* ``jepa_cosine_loss`` — cosine prediction loss.
* ``vicreg_variance_loss`` / ``vicreg_covariance_loss`` — collapse prevention.
* ``compute_ema_tau`` — cosine EMA-momentum schedule.
* ``ema_update`` — in-place EMA copy of a parameter group.

All modules are framework-agnostic w.r.t. backbone — the context encoder
only needs to expose ``last_hidden_state``-style tensors of shape
``(batch_size, seq_len, d_model)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

AGGREGATION_MODES = ("cls", "eos", "mean")


# ============================ Span masking collator ============================


@dataclass
class SpanMaskingCollator:
    """Collator that applies span-based masking for JEPA-DNA pretraining.

    Samples ``num_spans`` ~ Uniform[span_min, span_max] contiguous spans
    per sequence and aggregates them until the overall masking ratio falls
    within ``[mask_ratio_min, mask_ratio_max]``. Masked positions are
    replaced by ``tokenizer.mask_token_id`` and the original tokens are
    written into ``labels`` (``-100`` elsewhere) so that the standard
    MLM cross-entropy can ignore unmasked positions.

    Optionally prepends/appends an aggregation token so that the model
    has a stable position for the JEPA-DNA [CLS]/[EOS] readout. The same
    aggregation token is added to both the masked input and the unmasked
    target sequence (the target encoder never sees mask tokens).

    Attributes:
        tokenizer (Any): HuggingFace tokenizer (provides mask/cls/eos ids).
        mask_ratio_min (float): Lower bound of aggregate mask ratio
            (paper uses 0.20).
        mask_ratio_max (float): Upper bound of aggregate mask ratio
            (paper uses 0.40).
        span_min (int): Minimum number of contiguous spans per sequence.
        span_max (int): Maximum number of contiguous spans per sequence.
        aggregation (str): One of ``{"cls", "eos", "mean"}``. When
            ``"cls"``/``"eos"`` the corresponding special token is
            prepended/appended to every sequence.
        seed (int | None): Optional generator seed (per-step random state).
    """

    tokenizer: Any
    mask_ratio_min: float = 0.20
    mask_ratio_max: float = 0.40
    span_min: int = 1
    span_max: int = 3
    aggregation: str = "eos"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration and resolve special token ids."""
        if self.aggregation not in AGGREGATION_MODES:
            raise ValueError(
                f"aggregation must be one of {AGGREGATION_MODES}, "
                f"got '{self.aggregation}'"
            )
        if not 0.0 < self.mask_ratio_min <= self.mask_ratio_max < 1.0:
            raise ValueError(
                "Require 0 < mask_ratio_min <= mask_ratio_max < 1; "
                f"got [{self.mask_ratio_min}, {self.mask_ratio_max}]."
            )
        if self.span_min < 1 or self.span_max < self.span_min:
            raise ValueError(
                f"Require 1 <= span_min <= span_max; got "
                f"[{self.span_min}, {self.span_max}]."
            )
        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must define mask_token_id for JEPA.")

        self._cls_id: int | None = self.tokenizer.cls_token_id
        self._eos_id: int | None = (
            self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        )
        self._mask_id: int = self.tokenizer.mask_token_id

        if self.aggregation == "cls" and self._cls_id is None:
            raise ValueError(
                "Tokenizer has no cls_token_id; cannot use aggregation='cls'."
            )
        if self.aggregation == "eos" and self._eos_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id; cannot use aggregation='eos'."
            )

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Apply span masking and aggregation-token augmentation.

        Args:
            batch (list[dict[str, torch.Tensor]]): Raw items from the
                upstream dataset; each item must define ``"input_ids"``
                of shape ``(seq_len,)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys
                ``input_ids`` (B, L) — masked sequence,
                ``labels`` (B, L) — original ids at masked positions,
                    ``-100`` elsewhere,
                ``unmasked_input_ids`` (B, L) — clean sequence for target
                    encoder,
                ``mask`` (B, L) bool — masked positions.
        """
        gen = (
            torch.Generator()
            if self.seed is None
            else torch.Generator().manual_seed(self.seed)
        )

        unmasked = torch.stack([item["input_ids"] for item in batch], dim=0)
        if self.aggregation in ("cls", "eos"):
            unmasked = self._add_aggregation_token(unmasked)

        batch_size, seq_len = unmasked.shape
        masked = unmasked.clone()
        labels = torch.full_like(unmasked, fill_value=-100)
        mask = torch.zeros_like(unmasked, dtype=torch.bool)

        for batch_idx in range(batch_size):
            mask[batch_idx] = self._sample_span_mask(seq_len, gen)
            # Protect aggregation token slot from masking.
            mask[batch_idx] = self._unmask_aggregation_slot(mask[batch_idx])

        labels[mask] = unmasked[mask]
        masked[mask] = self._mask_id

        return {
            "input_ids": masked,
            "labels": labels,
            "unmasked_input_ids": unmasked,
            "mask": mask,
        }

    def _add_aggregation_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prepend [CLS] (``aggregation='cls'``) or append [EOS] otherwise.

        Args:
            input_ids (torch.Tensor): Input token ids of shape (B, L).

        Returns:
            torch.Tensor: Augmented ids of shape (B, L+1).
        """
        batch_size = input_ids.size(0)
        if self.aggregation == "cls":
            cls_col = torch.full(
                (batch_size, 1), fill_value=self._cls_id, dtype=input_ids.dtype
            )
            return torch.cat([cls_col, input_ids], dim=1)
        eos_col = torch.full(
            (batch_size, 1), fill_value=self._eos_id, dtype=input_ids.dtype
        )
        return torch.cat([input_ids, eos_col], dim=1)

    def _sample_span_mask(
        self, seq_len: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample a span mask whose total ratio lies in the configured band.

        Args:
            seq_len (int): Length of the sequence to mask.
            generator (torch.Generator): RNG controlling sampling.

        Returns:
            torch.Tensor: Boolean tensor of shape (seq_len,) with True at
            masked positions.
        """
        target_ratio = self.mask_ratio_min + (
            self.mask_ratio_max - self.mask_ratio_min
        ) * torch.rand((), generator=generator).item()
        target_mask_tokens = max(1, int(target_ratio * seq_len))

        num_spans = int(
            torch.randint(
                self.span_min, self.span_max + 1, (1,), generator=generator
            ).item()
        )
        per_span = max(1, target_mask_tokens // num_spans)

        mask = torch.zeros(seq_len, dtype=torch.bool)
        placed = 0
        attempts = 0
        max_attempts = num_spans * 20
        while placed < num_spans and attempts < max_attempts:
            attempts += 1
            if per_span >= seq_len:
                start = 0
                length = seq_len
            else:
                start = int(
                    torch.randint(
                        0, seq_len - per_span + 1, (1,), generator=generator
                    ).item()
                )
                length = per_span
            mask[start : start + length] = True
            placed += 1

        return mask

    def _unmask_aggregation_slot(self, mask_row: torch.Tensor) -> torch.Tensor:
        """Ensure the aggregation token position is never masked.

        Args:
            mask_row (torch.Tensor): Boolean mask of shape (seq_len,).

        Returns:
            torch.Tensor: ``mask_row`` with the aggregation slot cleared.
        """
        if self.aggregation == "cls":
            mask_row[0] = False
        elif self.aggregation == "eos":
            mask_row[-1] = False
        return mask_row


# ============================ Predictor head ============================


class JEPAPredictor(nn.Module):
    """Light Transformer predictor mapping context tokens to target latent.

    Following JEPA-DNA §4.1 (Training & Implementation Details), this is
    a pre-norm Transformer encoder operating in a reduced latent space,
    with sinusoidal positional embeddings and an input/output projection
    bridging the backbone hidden size and the predictor hidden size.

    Attributes:
        input_proj (nn.Linear): Projects backbone hidden states into the
            predictor latent space.
        positional_embedding (torch.Tensor): Frozen sinusoidal embeddings
            of shape ``(1, max_seq_len, predictor_dim)``.
        transformer (nn.TransformerEncoder): Stack of pre-norm encoder
            layers (default 3 layers, 3 heads).
        output_proj (nn.Linear): Projects the aggregated predictor output
            back into the backbone hidden space.
        mask_embedding (nn.Parameter): Learnable ``[MASK]`` embedding used
            for re-masking (replaces context hidden states at masked
            positions, see paper §3.2).
        aggregation (str): How to aggregate predictor outputs — ``cls``,
            ``eos``, or ``mean``.
    """

    def __init__(
        self,
        d_model: int,
        predictor_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 3,
        max_seq_len: int = 32768,
        dropout: float = 0.0,
        aggregation: str = "eos",
    ) -> None:
        """Build the predictor.

        Args:
            d_model (int): Backbone hidden size.
            predictor_dim (int): Predictor latent size (paper default 384).
            num_layers (int): Number of encoder layers (paper default 3).
            num_heads (int): Attention heads (paper default 3).
            max_seq_len (int): Upper bound on supported sequence length
                for the (frozen) sinusoidal positional embeddings.
            dropout (float): Dropout probability inside the encoder.
            aggregation (str): One of ``{"cls", "eos", "mean"}``.
        """
        super().__init__()
        if aggregation not in AGGREGATION_MODES:
            raise ValueError(
                f"aggregation must be one of {AGGREGATION_MODES}, "
                f"got '{aggregation}'"
            )

        self.aggregation = aggregation
        self.input_proj = nn.Linear(d_model, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        positional = _build_sinusoidal_positional_embedding(max_seq_len, predictor_dim)
        self.register_buffer("positional_embedding", positional, persistent=False)

    def forward(
        self,
        context_hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the target aggregation latent from masked context tokens.

        Args:
            context_hidden_states (torch.Tensor): Hidden states from the
                context encoder of shape ``(batch_size, seq_len, d_model)``.
            mask (torch.Tensor): Boolean tensor of shape
                ``(batch_size, seq_len)`` marking masked positions; these
                are re-masked with ``self.mask_embedding`` before being
                fed into the predictor (paper §3.2).

        Returns:
            torch.Tensor: Predicted aggregation latent of shape
            ``(batch_size, d_model)``.
        """
        batch_size, seq_len, _ = context_hidden_states.shape

        rebuilt = context_hidden_states.clone()
        if mask is not None:
            rebuilt = torch.where(
                mask.unsqueeze(-1),
                self.mask_embedding.to(rebuilt.dtype).expand_as(rebuilt),
                rebuilt,
            )

        latent = self.input_proj(rebuilt)
        if seq_len > self.positional_embedding.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds predictor positional "
                f"embedding capacity {self.positional_embedding.size(1)}."
            )
        latent = latent + self.positional_embedding[:, :seq_len, :].to(latent.dtype)

        encoded = self.transformer(latent)
        pooled = aggregate_hidden_states(encoded, self.aggregation)
        return self.output_proj(pooled)


def _build_sinusoidal_positional_embedding(
    max_seq_len: int, dim: int
) -> torch.Tensor:
    """Build a frozen sinusoidal positional embedding matrix.

    Args:
        max_seq_len (int): Maximum supported sequence length.
        dim (int): Embedding dimension (must be even).

    Returns:
        torch.Tensor: Tensor of shape ``(1, max_seq_len, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal embedding requires even dim; got {dim}.")
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(max_seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# ============================ Aggregation ============================


def aggregate_hidden_states(
    hidden_states: torch.Tensor, aggregation: str
) -> torch.Tensor:
    """Reduce a sequence of hidden states to a single vector per example.

    Args:
        hidden_states (torch.Tensor): Tensor of shape
            ``(batch_size, seq_len, d_model)``.
        aggregation (str): One of ``{"cls", "eos", "mean"}``.

    Returns:
        torch.Tensor: Pooled tensor of shape ``(batch_size, d_model)``.

    Raises:
        ValueError: If ``aggregation`` is not supported.
    """
    if aggregation == "cls":
        return hidden_states[:, 0, :]
    if aggregation == "eos":
        return hidden_states[:, -1, :]
    if aggregation == "mean":
        return hidden_states.mean(dim=1)
    raise ValueError(
        f"aggregation must be one of {AGGREGATION_MODES}, got '{aggregation}'"
    )


# ============================ Losses ============================


def jepa_cosine_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Cosine prediction loss between predictor output and target latent.

    Implements paper Eq. (2):
        ``L_jepa = 1 - cos(P_phi(h_cls), z_target)``.

    Args:
        predicted (torch.Tensor): Predictor output of shape
            ``(batch_size, d_model)``.
        target (torch.Tensor): Target encoder latent of shape
            ``(batch_size, d_model)``. Gradients should be stopped before
            calling this function.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    sim = F.cosine_similarity(predicted, target, dim=-1)
    return (1.0 - sim).mean()


def vicreg_variance_loss(
    embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> torch.Tensor:
    """Hinge variance term from VICReg (paper Eq. 3).

    Encourages every embedding dimension to maintain standard deviation
    of at least ``gamma`` across the batch, preventing informational
    collapse.

    Args:
        embeddings (torch.Tensor): Tensor of shape ``(N, d)`` (batch
            dimension first). When supplying both context and predictor
            aggregations, stack them along ``dim=0`` so ``N = 2 * B``.
        gamma (float): Variance threshold (paper default 1.0).
        eps (float): Numerical stability constant inside ``sqrt``.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    std = torch.sqrt(embeddings.var(dim=0, unbiased=True) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty from VICReg (paper Eq. 4 & 5).

    Args:
        embeddings (torch.Tensor): Tensor of shape ``(N, d)``.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n_samples, dim = embeddings.shape
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = centered.t().mm(centered) / max(n_samples - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / dim


# ============================ EMA target encoder ============================


@torch.no_grad()
def ema_update(
    target_params: Iterable[nn.Parameter],
    source_params: Iterable[nn.Parameter],
    tau: float,
) -> None:
    """Update target parameters in-place via Exponential Moving Average.

    ``target = tau * target + (1 - tau) * source``. Following the paper,
    ``tau`` is scheduled from 0.996 toward 1.0 over training (see
    :func:`compute_ema_tau`).

    Args:
        target_params (Iterable[nn.Parameter]): Parameters of the target
            encoder (updated in place).
        source_params (Iterable[nn.Parameter]): Parameters of the trainable
            context encoder.
        tau (float): Momentum coefficient in ``[0, 1]``.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be within [0, 1]; got {tau}.")
    for target, source in zip(target_params, source_params, strict=True):
        target.data.mul_(tau).add_(source.data, alpha=1.0 - tau)


def compute_ema_tau(
    step: int,
    total_steps: int,
    tau_start: float = 0.996,
    tau_end: float = 1.0,
) -> float:
    """Cosine schedule for the EMA momentum coefficient.

    Args:
        step (int): Current optimizer step (0-indexed).
        total_steps (int): Total number of optimizer steps in training.
        tau_start (float): Initial momentum (paper default 0.996).
        tau_end (float): Final momentum (paper default 1.0).

    Returns:
        float: Interpolated ``tau`` value at ``step``.

    Raises:
        ValueError: If ``total_steps`` is non-positive.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive; got {total_steps}.")
    progress = min(max(step / float(total_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return tau_start + (tau_end - tau_start) * cosine
