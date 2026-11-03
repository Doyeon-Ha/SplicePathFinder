"""Rotary embeddings for Evo2.

Mirrors models/evo/hf/rotary.py (same RoPE math): flash_attn fast path when
available, pure-PyTorch fallback otherwise. LinearlyScaledRotaryEmbedding
implements linear position-index interpolation (`t = t / scaling_factor`)
used by all the 1M / 262K context variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn


try:
    from flash_attn.layers.rotary import RotaryEmbedding as _FlashRotaryEmbedding
    _HAS_FLASH_ROTARY = True
except ImportError:
    _FlashRotaryEmbedding = None
    _HAS_FLASH_ROTARY = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def restore_rotary_inv_freq(
    rotary_emb: nn.Module,
    *,
    device=None,
    invalidate_cache: bool = True,
) -> None:
    """Rebuild ``inv_freq`` from RoPE metadata in fp32."""
    if not hasattr(rotary_emb, "inv_freq"):
        return
    device = rotary_emb.inv_freq.device if device is None else device
    if hasattr(rotary_emb, "_compute_inv_freq"):
        inv_freq = rotary_emb._compute_inv_freq(device=device)
    else:
        inv_freq = 1.0 / (
            float(rotary_emb.base)
            ** (
                torch.arange(
                    0,
                    int(rotary_emb.dim),
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / int(rotary_emb.dim)
            )
        )
    rotary_emb.inv_freq = inv_freq.to(dtype=torch.float32)
    if invalidate_cache:
        rotary_emb._seq_len_cached = 0
        for name in (
            "_cos_cached",
            "_sin_cached",
            "_cos_k_cached",
            "_sin_k_cached",
        ):
            if hasattr(rotary_emb, name):
                setattr(rotary_emb, name, None)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Compute the multiply in fp32 then cast back to x.dtype to match
    # flash_attn's Triton kernel bit-exactly. Doing the multiply in bf16
    # directly compounds rounding error of ~3e-2 per layer.
    rot_dim = cos.shape[-1] * 2
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    orig_dtype = x.dtype
    cos_full = torch.cat((cos, cos), dim=-1).float()
    sin_full = torch.cat((sin, sin), dim=-1).float()
    x_rot_f = x_rot.float()
    rotated = (x_rot_f * cos_full) + (_rotate_half(x_rot_f) * sin_full)
    rotated = rotated.to(orig_dtype)
    return torch.cat((rotated, x_pass), dim=-1)


class _PureRotaryEmbedding(nn.Module):
    """Pure-PyTorch fallback RoPE. Mirrors flash_attn.layers.rotary.RotaryEmbedding's surface."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        pos_idx_in_fp32: bool = True,
        device=None,
    ):
        super().__init__()
        if interleaved:
            raise NotImplementedError("Interleaved RoPE is not implemented.")
        if scale_base is not None:
            raise NotImplementedError("xPos scale_base is not implemented.")
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        inv_freq = self._compute_inv_freq(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale = None  # xPos slot kept for swap_mha_rope compatibility

        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _compute_inv_freq(self, device=None) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                inv_freq = self._compute_inv_freq(device=device)
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        seqlen_offset: int | torch.Tensor = 0,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if isinstance(seqlen_offset, torch.Tensor):
            seqlen_offset = int(seqlen_offset.max().item())
        T = qkv.shape[1]
        seqlen = max_seqlen if max_seqlen is not None else (T + seqlen_offset)
        self._update_cos_sin_cache(seqlen, device=qkv.device, dtype=qkv.dtype)

        cos = self._cos_cached[seqlen_offset : seqlen_offset + T]
        sin = self._sin_cached[seqlen_offset : seqlen_offset + T]
        q, k, v = qkv.unbind(dim=2)
        cos_b = cos[None, :, None, :]
        sin_b = sin[None, :, None, :]
        q = _apply_rotary(q, cos_b, sin_b)
        k = _apply_rotary(k, cos_b, sin_b)
        return torch.stack((q, k, v), dim=2)


if _HAS_FLASH_ROTARY:
    class RotaryEmbedding(_FlashRotaryEmbedding):
        """Flash RoPE with exact fp32 frequency reconstruction."""

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            needs_update = (
                seqlen > self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            )
            if needs_update and self.pos_idx_in_fp32:
                restore_rotary_inv_freq(
                    self,
                    device=device,
                    invalidate_cache=False,
                )
            return super()._update_cos_sin_cache(
                seqlen,
                device=device,
                dtype=dtype,
            )
else:
    RotaryEmbedding = _PureRotaryEmbedding


class LinearlyScaledRotaryEmbedding(RotaryEmbedding):
    """RoPE with linear interpolation of position indices.

    Used for evo2_7b (1M), evo2_7b_262k, evo2_20b, evo2_40b. Positions are
    divided by ``scaling_factor`` before cos/sin tables are computed.
    """

    def __init__(self, dim: int, scaling_factor: float = 1.0, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self._linear_scaling_factor = float(scaling_factor)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen <= self._seq_len_cached
            and self._cos_cached is not None
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
            and not (self.training and self._cos_cached.is_inference())
        ):
            return

        self._seq_len_cached = seqlen
        if self.pos_idx_in_fp32:
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            t = t / self._linear_scaling_factor
            inv_freq = self._compute_inv_freq(device=device)
        else:
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            t = t / self._linear_scaling_factor
            inv_freq = self.inv_freq

        freqs = torch.outer(t, inv_freq)
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
        else:
            from einops import rearrange
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)


def swap_mha_rope(mha, new_rope=LinearlyScaledRotaryEmbedding, kwargs_new_rope=None):
    """Replace ``mha.rotary_emb`` with a freshly-constructed scaled RoPE."""
    weight_attr = "Wq" if getattr(mha, "cross_attn", False) else "Wqkv"
    weight = getattr(mha, weight_attr).weight
    dtype = weight.dtype
    kwargs_old_rope = dict(
        dim=mha.rotary_emb.dim,
        base=mha.rotary_emb.base,
        interleaved=mha.rotary_emb.interleaved,
        scale_base=mha.rotary_emb.scale_base,
        pos_idx_in_fp32=mha.rotary_emb.pos_idx_in_fp32,
        device=mha.rotary_emb.inv_freq.device,
    )
    del mha.rotary_emb
    kwargs_new_rope = kwargs_new_rope or {"scaling_factor": 1.0}
    scaled = new_rope(**kwargs_new_rope, **kwargs_old_rope).to(dtype)
    restore_rotary_inv_freq(scaled)
    mha.rotary_emb = scaled
    return mha
