"""HyenaDNA backbone wrapped with the JEPA-DNA pretraining objective.

This module exposes :class:`HyenaDNAForJEPA`, a thin wrapper around
:class:`HyenaDNAForMaskedLM` that adds:

* an EMA-tracked target encoder (structural duplicate of the context
  backbone),
* a light Transformer predictor that maps re-masked context tokens to
  the target aggregation latent (paper §3.2–§3.3),
* an interface compatible with HuggingFace Trainer's ``compute_loss`` so
  that the existing distributed launcher in
  :mod:`train.train_mlm` can be reused.

The wrapper does not implement the EMA update itself — that is driven
externally by a training-loop callback (see ``train.train_jepa``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.jepa import (
    JEPAPredictor,
    aggregate_hidden_states,
    jepa_cosine_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)

from .modeling_hyena import HyenaDNAForMaskedLM


@dataclass
class JEPAOutput:
    """Container holding the JEPA-DNA pretraining outputs.

    Attributes:
        loss (torch.Tensor): Composite training loss (scalar). ``None``
            when no labels are supplied.
        mlm_loss (torch.Tensor): MLM cross-entropy on masked positions.
        jepa_loss (torch.Tensor): Cosine prediction loss in latent space.
        variance_loss (torch.Tensor): VICReg variance regularizer.
        covariance_loss (torch.Tensor): VICReg covariance regularizer.
        mlm_logits (torch.Tensor): Logits from the LM head, shape
            ``(batch_size, seq_len, vocab_size)``.
        predicted_latent (torch.Tensor): Predictor output, shape
            ``(batch_size, d_model)``.
        target_latent (torch.Tensor): Aggregated target encoder latent,
            shape ``(batch_size, d_model)`` (detached from the graph).
    """

    loss: torch.Tensor | None
    mlm_loss: torch.Tensor
    jepa_loss: torch.Tensor
    variance_loss: torch.Tensor
    covariance_loss: torch.Tensor
    mlm_logits: torch.Tensor
    predicted_latent: torch.Tensor
    target_latent: torch.Tensor


class HyenaDNAForJEPA(nn.Module):
    """JEPA-DNA pretraining wrapper around a HyenaDNA backbone.

    Attributes:
        context (HyenaDNAForMaskedLM): Trainable context encoder; the
            MLM head is reused for the token-level recovery loss.
        target (HyenaDNAForMaskedLM): EMA-tracked structural duplicate of
            ``context``; weights are frozen w.r.t. gradient descent and
            updated externally via :func:`train.jepa.ema_update`.
        predictor (JEPAPredictor): Light Transformer predictor producing
            the predicted aggregation latent from the re-masked context
            hidden states.
        aggregation (str): One of ``{"cls", "eos", "mean"}``.
        lambda_mlm (float): Weight of the MLM loss term (paper λ₁).
        lambda_jepa (float): Weight of the JEPA cosine loss term (λ₂).
        lambda_var (float): Weight of the VICReg variance term (λ₃,
            paper default 25.0).
        lambda_cov (float): Weight of the VICReg covariance term (λ₄,
            paper default 0.5).
        var_gamma (float): Variance hinge threshold (paper default 1.0).
    """

    def __init__(
        self,
        context_encoder: HyenaDNAForMaskedLM,
        predictor_dim: int = 384,
        predictor_layers: int = 3,
        predictor_heads: int = 3,
        predictor_dropout: float = 0.0,
        max_seq_len: int = 32768,
        aggregation: str = "eos",
        lambda_mlm: float = 1.0,
        lambda_jepa: float = 1.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 0.5,
        var_gamma: float = 1.0,
    ) -> None:
        """Wire up context / target / predictor and store loss weights.

        Args:
            context_encoder (HyenaDNAForMaskedLM): Backbone+MLM head to
                use as the trainable context encoder. Weights should
                already be initialized / transferred prior to wrapping.
            predictor_dim (int): Predictor latent size.
            predictor_layers (int): Predictor depth.
            predictor_heads (int): Number of attention heads in predictor.
            predictor_dropout (float): Dropout inside the predictor.
            max_seq_len (int): Sequence length budget for the predictor's
                positional embeddings.
            aggregation (str): Aggregation token strategy (see
                :func:`train.jepa.aggregate_hidden_states`).
            lambda_mlm (float): MLM loss weight.
            lambda_jepa (float): JEPA cosine loss weight.
            lambda_var (float): Variance regularizer weight.
            lambda_cov (float): Covariance regularizer weight.
            var_gamma (float): Variance hinge threshold.
        """
        super().__init__()
        self.context = context_encoder
        self.target = copy.deepcopy(context_encoder)
        for param in self.target.parameters():
            param.requires_grad_(False)

        d_model = self.context.config.d_model
        self.predictor = JEPAPredictor(
            d_model=d_model,
            predictor_dim=predictor_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            max_seq_len=max_seq_len,
            dropout=predictor_dropout,
            aggregation=aggregation,
        )

        self.aggregation = aggregation
        self.lambda_mlm = lambda_mlm
        self.lambda_jepa = lambda_jepa
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_gamma = var_gamma

    @property
    def config(self) -> Any:
        """Expose the context encoder's config (for HF Trainer compat)."""
        return self.context.config

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Enable gradient checkpointing on the context backbone only.

        The target encoder runs under ``no_grad`` so checkpointing is
        unnecessary; the predictor is small enough to skip as well.

        Args:
            **kwargs: Forwarded to the underlying call (e.g. ``use_reentrant``).
        """
        if hasattr(self.context, "gradient_checkpointing_enable"):
            self.context.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the context backbone."""
        if hasattr(self.context, "gradient_checkpointing_disable"):
            self.context.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        unmasked_input_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **_: Any,
    ) -> JEPAOutput:
        """Run the JEPA-DNA pretraining forward pass.

        Args:
            input_ids (torch.Tensor): Masked context tokens of shape
                ``(batch_size, seq_len)``.
            labels (torch.Tensor | None): Original ids at masked
                positions (``-100`` elsewhere), shape ``(B, L)``. When
                ``None`` only the MLM logits are returned and no loss
                is computed.
            unmasked_input_ids (torch.Tensor | None): Clean tokens for
                the target encoder, shape ``(B, L)``. Required during
                training; ignored at inference.
            mask (torch.Tensor | None): Boolean mask of shape ``(B, L)``
                identifying masked positions (used for re-masking).
                Required during training.
            **_: Ignored, present for HF Trainer kwarg compatibility.

        Returns:
            JEPAOutput: Composite loss, per-term losses, MLM logits and
            the predicted/target aggregation latents.
        """
        backbone_outputs = self.context.hyena(
            input_ids=input_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        context_hidden = backbone_outputs.last_hidden_state
        mlm_logits = self.context.lm_head(context_hidden).float()

        if labels is None or unmasked_input_ids is None or mask is None:
            zero = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            return JEPAOutput(
                loss=None,
                mlm_loss=zero,
                jepa_loss=zero,
                variance_loss=zero,
                covariance_loss=zero,
                mlm_logits=mlm_logits,
                predicted_latent=zero,
                target_latent=zero,
            )

        predicted_latent = self.predictor(context_hidden, mask)

        with torch.no_grad():
            target_backbone_outputs = self.target.hyena(
                input_ids=unmasked_input_ids,
                output_hidden_states=False,
                return_dict=True,
            )
            target_hidden = target_backbone_outputs.last_hidden_state
            target_latent = aggregate_hidden_states(target_hidden, self.aggregation)
            target_latent = target_latent.detach()

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.context.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        jepa_loss = jepa_cosine_loss(predicted_latent, target_latent)

        variance_inputs = torch.cat(
            [predicted_latent.float(), target_latent.float()], dim=0
        )
        variance_loss = vicreg_variance_loss(variance_inputs, gamma=self.var_gamma)
        covariance_loss = vicreg_covariance_loss(variance_inputs)

        total_loss = (
            self.lambda_mlm * mlm_loss
            + self.lambda_jepa * jepa_loss
            + self.lambda_var * variance_loss
            + self.lambda_cov * covariance_loss
        )

        return JEPAOutput(
            loss=total_loss,
            mlm_loss=mlm_loss.detach(),
            jepa_loss=jepa_loss.detach(),
            variance_loss=variance_loss.detach(),
            covariance_loss=covariance_loss.detach(),
            mlm_logits=mlm_logits,
            predicted_latent=predicted_latent.detach(),
            target_latent=target_latent.detach(),
        )

    def context_parameters(self) -> list[nn.Parameter]:
        """Return the parameters of the trainable context encoder."""
        return list(self.context.parameters())

    def target_parameters(self) -> list[nn.Parameter]:
        """Return the parameters of the EMA-tracked target encoder."""
        return list(self.target.parameters())
