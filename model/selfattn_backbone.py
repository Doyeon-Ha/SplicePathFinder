"""Self-attention (RoPE Transformer) backbone — drop-in for ``HyenaDNAModel``.

This module provides ``SelfAttentionBackbone``, a bidirectional Transformer
encoder that mirrors the public surface of the project-local
``HyenaDNAModel`` so it can replace the long-range main backbone in
``IssaSplice`` **without any change to the chunker / dechunker / head /
data / training path**. It is intended for the "swap only the mixer"
ablation: the dnaHNet dynamic chunker compresses a 15000 bp input to
~2500 chunk tokens, and this backbone runs full O(K^2) self-attention over
that compressed sequence (tractable at K~2500) in place of HyenaDNA's
implicit long-convolution.

Interface parity with ``HyenaDNAModel`` (see
``model/hyenadna/src/modeling_hyena.py``):

- ``self.backbone.embeddings.word_embeddings`` (``nn.Embedding``) — the
  nt-embedding lookup that ``IssaSplice._resolve_nt_embedding_layer``
  hands to the chunker (DD-015).
- ``self.backbone.layers`` (``nn.ModuleList``) — the transformer block
  list that ``IssaSplice._get_backbone_layers`` / the freeze policy walk.
- ``self.backbone.ln_f`` — final LayerNorm.
- ``forward(input_ids, inputs_embeds=None, output_hidden_states=None,
  return_dict=None, attention_mask=None)`` returning a
  ``BaseModelOutputWithNoAttention`` whose ``hidden_states`` is the
  per-layer list (input embedding, each block output, post-ln_f) so that
  ``all_hidden_states[hidden_state_layer_index]`` behaves identically.

Positional encoding: Rotary Position Embedding (RoPE) applied to Q/K
inside each attention layer, keyed by **chunk index** (0..K-1). This
matches the information HyenaDNA had over the same compressed sequence
(chunk ordering only, not genomic coordinates) — the fair mixer-only
ablation. Injecting true genomic nt positions is a deliberate follow-up
variant, not implemented here.

Padding: the chunker zero-pads shorter samples to ``K_max``; the boolean
``attention_mask`` (True = valid chunk) is forwarded from the deepest
stage's ``token_mask`` so padded chunks do not participate in attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


def _build_rope_cache(
    position_ids: torch.Tensor,
    head_dim: int,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample RoPE ``cos`` / ``sin`` tables from position ids.

    Positions are arbitrary (not necessarily contiguous): for the
    chunk-index arm they are ``0..K-1``; for the offset arm they are each
    chunk's within-window nt offset (``boundary_indices``), so the actual
    genomic spacing between chunks enters the rotation. Because RoPE
    attention depends only on the position *difference*
    ``R(p_i)^T R(p_j) = R(p_j - p_i)``, a single-window offset carries the
    same signal as a whole-genome absolute coordinate (the window start
    cancels) — so within-window offsets are sufficient.

    Args:
        position_ids (torch.Tensor): Positions of shape ``(B, L)`` (any
            integer/float dtype; cast to float32 internally).
        head_dim (int): Per-head dimension; must be even.
        base (float): RoPE frequency base (theta).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` each of shape
            ``(B, L, head_dim)`` in float32.
    """
    device = position_ids.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )  # (head_dim/2,)
    positions = position_ids.to(torch.float32)  # (B, L)
    freqs = positions[..., None] * inv_freq  # (B, L, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, L, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by half for RoPE (``[-x2, x1]``)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query / key tensors of shape ``(B, H, L, head_dim)``.

    Args:
        q (torch.Tensor): Queries ``(B, H, L, head_dim)``.
        k (torch.Tensor): Keys ``(B, H, L, head_dim)``.
        cos (torch.Tensor): Per-sample cosine table ``(B, L, head_dim)``.
        sin (torch.Tensor): Per-sample sine table ``(B, L, head_dim)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated ``(q, k)``.
    """
    # (B, L, head_dim) -> (B, 1, L, head_dim) to broadcast over heads.
    cos = cos[:, None, :, :].to(q.dtype)
    sin = sin[:, None, :, :].to(q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class _RoPESelfAttention(nn.Module):
    """Multi-head bidirectional self-attention with RoPE, using SDPA.

    RoPE is applied to Q/K before the scaled dot-product so relative
    chunk distance is encoded directly in the attention scores. A
    boolean key-padding mask removes padded chunks.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        """Initialize the RoPE self-attention block.

        Args:
            d_model (int): Model dimension D.
            n_head (int): Number of attention heads; must divide D and
                yield an even head dim (RoPE requirement).
            dropout (float): Attention dropout probability.
            rope_base (float): RoPE frequency base (theta).

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_head`` or
                the resulting head dim is odd.
        """
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})."
            )
        self.head_dim = d_model // n_head
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head dim; got {self.head_dim} "
                f"(d_model={d_model}, n_head={n_head})."
            )
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run bidirectional RoPE self-attention.

        Args:
            x (torch.Tensor): Input ``(B, L, D)``.
            key_padding_mask (torch.Tensor | None): Boolean ``(B, L)``
                where True marks a valid (non-padded) chunk. ``None``
                disables masking.
            position_ids (torch.Tensor | None): RoPE positions ``(B, L)``.
                ``None`` falls back to chunk-index positions ``0..L-1``.

        Returns:
            torch.Tensor: Output ``(B, L, D)``.
        """
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
            )
        cos, sin = _build_rope_cache(position_ids, self.head_dim, self.rope_base)
        q, k = _apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA bool mask: True = participate. Broadcast the per-key
            # validity over heads and query positions -> (B, 1, 1, L).
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class _RoPEEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder block: RoPE attention + GELU FFN."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
        layer_norm_eps: float,
    ) -> None:
        """Initialize the encoder layer.

        Args:
            d_model (int): Model dimension.
            n_head (int): Attention heads.
            dim_feedforward (int): FFN hidden dimension.
            dropout (float): Dropout for attention and FFN.
            rope_base (float): RoPE theta.
            layer_norm_eps (float): LayerNorm epsilon.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = _RoPESelfAttention(d_model, n_head, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-norm residual block forward.

        Args:
            x (torch.Tensor): Input ``(B, L, D)``.
            key_padding_mask (torch.Tensor | None): Boolean ``(B, L)``,
                True = valid chunk.
            position_ids (torch.Tensor | None): RoPE positions ``(B, L)``.

        Returns:
            torch.Tensor: Output ``(B, L, D)``.
        """
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask, position_ids))
        x = x + self.ffn(self.norm2(x))
        return x


class _Embeddings(nn.Module):
    """Thin wrapper exposing ``word_embeddings`` (HyenaDNA parity)."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class _SelfAttnInner(nn.Module):
    """Inner encoder stack, mirroring ``HyenaLMBackbone`` structure."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
        layer_norm_eps: float,
        padding_idx: int | None,
    ) -> None:
        super().__init__()
        self.embeddings = _Embeddings(vocab_size, d_model, padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                _RoPEEncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope_base=rope_base,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder stack (HyenaLMBackbone-compatible signature).

        Args:
            input_ids (torch.Tensor | None): Token ids ``(B, L)`` (used
                only when ``inputs_embeds`` is None).
            inputs_embeds (torch.Tensor | None): Precomputed embeddings
                ``(B, L, D)`` — the chunk sequence in the chunker path.
            output_hidden_states (bool): Collect per-layer hidden states.
            attention_mask (torch.Tensor | None): Boolean ``(B, L)``,
                True = valid chunk (key-padding mask).
            position_ids (torch.Tensor | None): RoPE positions ``(B, L)``.
                ``None`` = chunk-index positions ``0..L-1``.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: ``(last_hidden,
                all_hidden_states)``. ``all_hidden_states`` is empty when
                ``output_hidden_states`` is False.
        """
        all_hidden_states: list[torch.Tensor] = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                key_padding_mask=attention_mask,
                position_ids=position_ids,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states.to(dtype=self.ln_f.weight.dtype))
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class SelfAttentionBackbone(nn.Module):
    """Bidirectional RoPE Transformer, drop-in for ``HyenaDNAModel``.

    See the module docstring for the interface-parity contract. Instances
    expose ``self.backbone`` (an ``_SelfAttnInner``) so downstream
    embedding-resolution, layer-access, and freeze code paths in
    ``IssaSplice`` work unchanged.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        padding_idx: int | None = None,
    ) -> None:
        """Initialize the self-attention backbone.

        Args:
            vocab_size (int): Token vocabulary size for ``word_embeddings``.
            d_model (int): Model / hidden dimension D.
            n_layer (int): Number of transformer encoder layers.
            n_head (int): Attention heads per layer.
            dim_feedforward (int): FFN hidden dimension.
            dropout (float): Dropout probability. Defaults to 0.1.
            rope_base (float): RoPE frequency base (theta). Defaults to
                10000.0.
            layer_norm_eps (float): LayerNorm epsilon. Defaults to 1e-5.
            padding_idx (int | None): Padding token id for the embedding.
        """
        super().__init__()
        self.backbone = _SelfAttnInner(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            rope_base=rope_base,
            layer_norm_eps=layer_norm_eps,
            padding_idx=padding_idx,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the nt embedding layer (HF-style hook)."""
        return self.backbone.embeddings.word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithNoAttention:
        """Forward pass matching ``HyenaDNAModel.forward``.

        Args:
            input_ids (torch.Tensor | None): Token ids ``(B, L)``.
            inputs_embeds (torch.Tensor | None): Precomputed embeddings
                ``(B, L, D)`` (chunk sequence in the chunker path).
            output_hidden_states (bool | None): Return per-layer hidden
                states. Defaults to True when None (IssaSplice always
                requests them).
            return_dict (bool | None): Return a ``BaseModelOutput``.
                Defaults to True.
            attention_mask (torch.Tensor | None): Boolean ``(B, L)``,
                True = valid chunk (key-padding mask). Extra vs. the
                HyenaDNA signature; IssaSplice passes the deepest stage's
                ``token_mask`` here.
            position_ids (torch.Tensor | None): RoPE positions ``(B, L)``.
                ``None`` = chunk-index (``0..L-1``); IssaSplice passes the
                composed within-window nt offsets for the offset-RoPE arm.

        Returns:
            BaseModelOutputWithNoAttention: ``last_hidden_state`` and the
                per-layer ``hidden_states`` list.
        """
        output_hidden_states = (
            True if output_hidden_states is None else output_hidden_states
        )
        last_hidden, all_hidden = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return BaseModelOutputWithNoAttention(
            last_hidden_state=last_hidden,
            hidden_states=all_hidden if output_hidden_states else None,
        )
