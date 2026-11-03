"""Multi-head attention for Evo2.

Causal self-attention with RoPE. Three backends dispatchable via the standard
HF attn_implementation kwarg:

  * "eager"              -- pure F.softmax(QK^T)V; returns weights on demand
  * "sdpa"               -- F.scaled_dot_product_attention; no weights
  * "flash_attention_2"  -- flash_attn.flash_attn_qkvpacked_func; no weights

Constructor signature is a strict subset of vortex's MHA so the
AttentionBlock instantiation site only changes the attn_implementation kwarg.
KV cache (used by Evo2 generation) goes through an SDPA path that handles
arbitrary sequence lengths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import RotaryEmbedding


def _flash_attn_required():
    try:
        from flash_attn import flash_attn_qkvpacked_func  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "attn_implementation='flash_attention_2' requires the flash-attn "
            "package. Install with `pip install flash-attn --no-build-isolation`."
        ) from exc


def _update_kv_cache(kv: torch.Tensor, inference_params, layer_idx: int) -> torch.Tensor:
    num_heads, head_dim = kv.shape[-2:]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    if layer_idx not in inference_params.key_value_memory_dict:
        capacity = min(
            inference_params.max_seqlen,
            max(16, 1 << (sequence_end - 1).bit_length()),
        )
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            capacity,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
        if sequence_end > kv_cache.shape[1]:
            capacity = min(
                inference_params.max_seqlen,
                max(sequence_end, kv_cache.shape[1] * 2),
            )
            if capacity < sequence_end:
                raise ValueError(
                    f"Cache length {sequence_end} exceeds maximum "
                    f"{inference_params.max_seqlen}."
                )
            grown = torch.empty(
                inference_params.max_batch_size,
                capacity,
                2,
                num_heads,
                head_dim,
                dtype=kv.dtype,
                device=kv.device,
            )
            grown[:, :kv_cache.shape[1]].copy_(kv_cache)
            kv_cache = grown
            inference_params.key_value_memory_dict[layer_idx] = kv_cache
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention with backend-dispatch."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_kv: int | None = None,
        cross_attn: bool = False,
        qkv_proj_bias: bool = True,
        out_proj_bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        layer_idx: int | None = None,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        rotary_emb_scale_base: float | None = None,
        rotary_emb_interleaved: bool = False,
        use_flash_attn: bool = False,  # legacy kwarg kept for ctor compat
        attn_implementation: str = "eager",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if cross_attn:
            raise NotImplementedError("Cross-attention is not supported.")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        if self.embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_heads % self.num_heads_kv != 0:
            raise ValueError("num_heads must be divisible by num_heads_kv")
        self.head_dim = self.embed_dim // num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.attn_implementation = attn_implementation
        self.dropout_p = dropout
        self.cross_attn = cross_attn

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                scale_base=rotary_emb_scale_base,
                device=device,
            )

        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim,
            dtype=dtype, device=device,
        )

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.Wqkv(x)
        if self.num_heads_kv == self.num_heads:
            return qkv.view(*qkv.shape[:-1], 3, self.num_heads, self.head_dim)
        q = qkv[..., : self.num_heads * self.head_dim]
        kv = qkv[..., self.num_heads * self.head_dim:]
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        kv = kv.view(*kv.shape[:-1], 2, self.num_heads_kv, self.head_dim)
        return q, kv

    def _forward_eager(
        self,
        qkv: torch.Tensor,
        output_attentions: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Bit-identical port of flash_attn.modules.mha.SelfAttention.forward.

        Key choices that must match vortex (which delegates to this class
        when use_flash_attn=False):

          * einsum (not matmul) for QK^T and PV: bf16 matmul vs bf16 einsum
            use different accumulator orders, ~3e-2 per attention layer.
          * -10000.0 causal mask (not -inf): vortex uses a finite mask value.
          * softmax(scores, dim=-1, dtype=v.dtype) (not F.softmax(.float())):
            PyTorch promotes to fp32 internally then casts to v.dtype.
            Explicit .float() then .to(v.dtype) cast rounds differently.
          * scale applied to K (k * scale), not Q. Same math, different
            bf16 multiply target.
        """
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D)
        softmax_scale = (
            self.softmax_scale if self.softmax_scale is not None
            else 1.0 / math.sqrt(self.head_dim)
        )
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        T = q.shape[1]
        if self.causal:
            causal_mask = torch.triu(
                torch.full((T, T), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(),
                -10000.0,
            )
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_weights = (
            torch.softmax(scores.float(), dim=-1)
            if output_attentions
            else None
        )
        if self.training and self.dropout_p > 0:
            attention = F.dropout(attention, p=self.dropout_p)
        out = torch.einsum("bhts,bshd->bthd", attention, v)
        return out, (attention_weights if output_attentions else None)

    def _forward_sdpa(
        self,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scale = self.softmax_scale if self.softmax_scale is not None else None
        sdpa_mask = None
        is_causal = self.causal
        if attention_mask is not None:
            query_length = q.shape[-2]
            key_length = k.shape[-2]
            allowed = attention_mask[:, None, None, :].bool()
            if self.causal:
                causal = (
                    torch.arange(key_length, device=q.device)[None, :]
                    <= torch.arange(query_length, device=q.device)[:, None]
                )
                allowed = allowed & causal[None, None, :, :]
            sdpa_mask = allowed
            is_causal = False
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=scale,
        )
        return out.permute(0, 2, 1, 3)

    def _forward_flash(
        self,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None and not attention_mask.bool().all():
            return self._forward_sdpa(qkv, attention_mask)
        _flash_attn_required()
        from flash_attn import flash_attn_qkvpacked_func
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return out

    def _forward_with_cache(
        self,
        qkv: torch.Tensor,
        inference_params,
    ) -> torch.Tensor:
        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(
                qkv,
                seqlen_offset=inference_params.seqlen_offset,
                max_seqlen=inference_params.seqlen_offset + qkv.shape[1],
            )
        q, k, v = qkv.unbind(dim=2)
        kv = torch.stack((k, v), dim=2)
        kv = _update_kv_cache(kv, inference_params, self.layer_idx)
        k_full, v_full = kv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k_full = k_full.permute(0, 2, 1, 3)
        v_full = v_full.permute(0, 2, 1, 3)
        scale = self.softmax_scale if self.softmax_scale is not None else None
        is_causal = self.causal and q.shape[-2] == k_full.shape[-2]
        out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=is_causal, scale=scale,
        )
        return out.permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        inference_params=None,
        output_attentions: bool = False,
        **_unused,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.num_heads_kv != self.num_heads:
            raise NotImplementedError("GQA is not exercised by Evo2.")

        qkv = self._project_qkv(x)
        attention_mask = _unused.get("padding_mask")

        if inference_params is not None:
            out_btd = self._forward_with_cache(qkv, inference_params)
            attn_weights = None
        else:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv, seqlen_offset=0, max_seqlen=qkv.shape[1])

            backend = self.attn_implementation
            if output_attentions and backend != "eager":
                backend = "eager"

            if backend == "eager":
                out_btd, attn_weights = self._forward_eager(
                    qkv,
                    output_attentions=output_attentions,
                    attention_mask=attention_mask,
                )
            elif backend == "sdpa":
                out_btd = self._forward_sdpa(qkv, attention_mask)
                attn_weights = None
            elif backend == "flash_attention_2":
                out_btd = self._forward_flash(qkv, attention_mask)
                attn_weights = None
            else:
                raise ValueError(f"Unknown attn_implementation: {backend!r}")

        B, T, H, D = out_btd.shape
        out_flat = out_btd.reshape(B, T, H * D)
        return self.out_proj(out_flat), attn_weights
