"""StripedHyena2 block definitions for Evo2.

Block types:

  * AttentionBlock       -- standard pre-norm transformer block with MHA + RoPE.
  * ParallelGatedConvBlock -- pre-norm hyena block; the inner `filter` is a
                              HyenaCascade configured per block-type idx:
                                hcl: IIR via log_poles + residues
                                hcm: FIR cascade, fir_inner_filter_length=128
                                hcs: FIR cascade, fir_inner_filter_length=7

Inside ParallelGatedConvBlock, `projections` is a TELinear (3x hidden_size
output) -- the FP8-capable input projection. For 7B variants without TE,
the pure-PyTorch fallback in layers.py is used.

The `interleave` config flag (True for all Evo2 variants) reorders the
channel dim after the outer FIR before the channel-split / cascade.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MHA
from .engine import HyenaInferenceEngine
from .layers import ParallelGatedMLP, RMSNorm, TELinear
from .rotary import restore_rotary_inv_freq, swap_mha_rope


def _interleave(z_pre: torch.Tensor) -> torch.Tensor:
    """Deinterleave a [x1_0,x2_0,v_0,x1_1,x2_1,v_1,...] channel layout into
    blocks of [x1_all | x2_all | v_all]. Matches vortex.model.utils.interleave."""
    if len(z_pre.shape) == 3:  # (B, C, L) non-cached path
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v = z_pre[:, 2::3, :]
        return torch.cat([x1, x2, v], dim=1)
    x1 = z_pre[..., 0::3]
    x2 = z_pre[..., 1::3]
    v = z_pre[..., 2::3]
    return torch.cat([x1, x2, v], dim=-1)


def _column_split_step(z_pre, num_heads, head_size):
    """Per-step variant of column_split for sequential_forward (cache step)."""
    x = z_pre.reshape(z_pre.shape[0], num_heads, 3 * head_size)
    x2 = x[..., :head_size].reshape(z_pre.shape[0], -1)
    x1 = x[..., head_size : 2 * head_size].reshape(z_pre.shape[0], -1)
    v = x[..., 2 * head_size :].reshape(z_pre.shape[0], -1)
    return x2, x1, v


# =============================================================================
# Attention block
# =============================================================================


class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pre_norm = RMSNorm(config)
        self.post_norm = RMSNorm(config)
        self.proj_groups = config.get("proj_groups", 1)
        dtype = self._resolve_dtype(config.get("attn_block_dtype", "bfloat16"))
        mlp_dtype = self._resolve_dtype(config.get("mlp_dtype", "bfloat16"))
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        attn_impl = getattr(config, "_attn_implementation", "eager")

        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            qkv_proj_bias=config.get("qkv_proj_bias", False),
            rotary_emb_base=config.get("rotary_emb_base", 1000000),
            causal=True,
            layer_idx=layer_idx,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            attn_implementation=attn_impl,
        ).to(dtype=dtype)

        if config.get("use_interpolated_rotary_pos_emb", False):
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={
                    "scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)
                },
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        restore_rotary_inv_freq(self.inner_mha_cls.rotary_emb)

        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

    @staticmethod
    def _resolve_dtype(name) -> torch.dtype:
        if isinstance(name, torch.dtype):
            return name
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(name, torch.bfloat16)

    def forward(
        self,
        u: torch.Tensor,
        inference_params=None,
        padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(padding_mask, torch.Tensor):
            # Zero attended values at pad positions (no qkv bias here).
            u = u * padding_mask[..., None]

        attn_out, attn_weights = self.inner_mha_cls(
            self.pre_norm(u),
            inference_params=inference_params,
            padding_mask=padding_mask,
            output_attentions=output_attentions,
        )
        u = attn_out + u

        if isinstance(padding_mask, torch.Tensor):
            u = u * padding_mask[..., None]
        u = self.mlp(self.post_norm(u)) + u
        return u, attn_weights


# =============================================================================
# Hyena cascade (hcl / hcm / hcs)
# =============================================================================


class HyenaCascade(nn.Module):
    """The inner mixer of a hyena block.

    Two execution modes selected by `fir_inner_filter_length`:
      * None        -> hcl: outer FIR -> channel split -> parallel_iir
                              (modal-form long convolution)
      * int (e.g. 7 or 128) -> hcm/hcs: outer FIR -> interleave -> inner FIR
                              cascade via parallel_fir(gate=True)
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        hyena_filter_groups: int,
        fir_inner_filter_length: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.column_split_hyena = config.get("column_split_hyena", False)
        self.hyena_flip_x1x2 = config.get("hyena_flip_x1x2", False)

        assert self.hidden_size % self.num_filters == 0
        assert self.num_filters <= self.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = (
            self.hidden_size // self.num_attention_heads
        )

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * config.hidden_size, 1, config.short_filter_length)
        )
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size))
            if config.short_filter_bias else None
        )

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx,
            hyena_flip_x1x2=config.get("hyena_flip_x1x2", False),
        )
        self.fir_fn = F.conv1d
        self.fir_inner_fn = F.conv1d
        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert not self.use_flashfft, "long_fir_threshold incompatible with flashfft"

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups

        if self.fir_inner_filter_length:
            # hcm / hcs: explicit FIR filter `h` of shape (groups, 1, inner_len)
            self.h = nn.Parameter(
                torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length)
            )
            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))
            else:
                self.D = None
        else:
            # hcl: modal-form IIR, log_poles in (num_systems, state_size, 1)
            log_poles = torch.randn(self.num_systems, self.state_size, 1, dtype=torch.float32)
            self.log_poles = nn.Parameter(log_poles)
            self.residues = nn.Parameter(
                torch.randn(self.num_systems, self.state_size, dtype=torch.float32)
            )
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            self.h = None

        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (
            inference_params is not None
            and self.layer_idx in inference_params.fir_state_dict
        ):
            return self.sequential_forward(u, inference_params)
        return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )
        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn, u,
            self.short_filter_weight, self.short_filter_bias,
            L, dims=dims, gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask, dim_last=True,
        )
        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = _interleave(z_pre)

        if self.h is None:
            # hcl path: compute IIR impulse response then convolve via FFT
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h

        D = self.D
        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        if self.fir_inner_filter_length is not None:
            # hcm / hcs: inner FIR cascade
            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn, z_pre, h, D, L,
                dims=dims, gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
            )
            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            # hcl: parallel IIR via FFT
            y = self.engine.parallel_iir(
                z_pre, h, D, L, t=self.t,
                poles=self.log_poles, residues=self.residues,
                dims=dims, inference_params=inference_params,
                layer_idx=self.layer_idx,
                prefill_style=self.config.get("prefill_style", "fft"),
                use_flashfft=self.use_flashfft,
                fftconv_fn=self.fftconv_fn,
                column_split_hyena=self.column_split_hyena,
                long_fir_threshold=self.long_fir_threshold,
                padding_mask=padding_mask,
            )

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if len(u.shape) > 2:
            u = u[:, -1]
        # Track input dtype so we can cast back at the end. step_iir produces
        # fp32 output (log_poles / residues are fp32 for stability) which would
        # otherwise dtype-mismatch the downstream bf16 Linear (out_filter_dense).
        input_dtype = u.dtype

        z_pre, fir_state = self.engine.step_fir(
            u, inference_params.fir_state_dict[self.layer_idx],
            weight=self.short_filter_weight, bias=self.short_filter_bias,
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            # For a single step, dim-1 length is C and `_interleave` operates
            # along the channel dim; reshape to (B, C, 1) so the dim==3 branch
            # applies, then squeeze the last dim back.
            z_pre = _interleave(z_pre.unsqueeze(-1)).squeeze(-1)

        if self.column_split_hyena:
            x2, x1, v = _column_split_step(
                z_pre, self.num_attention_heads, self.hidden_size_per_attention_head
            )
        else:
            x2, x1, v = z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
            else:
                h = self.h
            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h, bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2, x1, v, self.D,
                self.residues, self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(input_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        if self.t is None:
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues = self.residues.to(filter_dtype)
        log_poles = self.log_poles.to(filter_dtype)
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # B, D, L
        return h, filter_dtype, log_poles, residues


# =============================================================================
# Hyena block (wraps HyenaCascade with norms / projections / mlp)
# =============================================================================


class ParallelGatedConvBlock(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        hyena_filter_groups: int | None = None,
        fir_inner_filter_length: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups if hyena_filter_groups is not None else config.hidden_size
        )
        dtype = AttentionBlock._resolve_dtype(config.get("hyena_block_dtype", "bfloat16"))
        mlp_dtype = AttentionBlock._resolve_dtype(config.get("mlp_dtype", "bfloat16"))
        self.pre_norm = RMSNorm(config).to(dtype=dtype)
        self.post_norm = RMSNorm(config).to(dtype=dtype)
        self.filter = HyenaCascade(
            config, layer_idx,
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        ).to(dtype=dtype)

        self.projections = TELinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.qkv_proj_bias,
            init_method=torch.nn.init.xavier_uniform_,
            use_fp8=config.get("use_fp8_input_projections", False),
        )

        self.out_filter_dense = nn.Linear(
            config.hidden_size, config.hidden_size,
            bias=config.hyena_out_proj_bias,
        ).to(dtype)
        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
        """Right-pad along seq dim to a multiple of `multiple` when FP8 input
        projections are enabled. TE's FP8 path requires the product of all
        dims except the last to be divisible by 8 (we use 16 to be safe).
        No-op when FP8 is off."""
        if not self.config.get("use_fp8_input_projections", False):
            return x
        seq_len = x.size(1)
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def _proj_norm(self, x):
        original_seq_len = x.size(1)
        normalized = self.pre_norm(x)
        normalized = self._pad_to_multiple(normalized)
        with torch.cuda.device(x.device) if x.is_cuda else _nullctx():
            projected = self.projections(normalized)
        if isinstance(projected, tuple):
            projected = projected[0]
        # Slice back to original seq length if padding was added.
        if projected.size(1) > original_seq_len:
            projected = projected[:, :original_seq_len, :]
        return projected

    def forward(
        self,
        u: torch.Tensor,
        inference_params=None,
        padding_mask=None,
        output_attentions: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        z = self._proj_norm(u)

        if isinstance(padding_mask, torch.Tensor):
            z = z * padding_mask[..., None]

        z, inference_params = self.filter(
            z, inference_params=inference_params, padding_mask=padding_mask,
        )
        z_in = self.out_filter_dense(z) + u

        if isinstance(padding_mask, torch.Tensor):
            z_in = z_in * padding_mask[..., None]

        y = self.mlp(self.post_norm(z_in)) + z_in
        # Hyena blocks have no attention matrix.
        return y, None


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# =============================================================================
# Block dispatch
# =============================================================================


def get_block(config, layer_idx: int):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    if layer_idx in config.hcl_layer_idxs:
        return ParallelGatedConvBlock(
            config, layer_idx,
            hyena_filter_groups=config.hcl_filter_groups,
            fir_inner_filter_length=None,
        )
    if layer_idx in config.hcm_layer_idxs:
        return ParallelGatedConvBlock(
            config, layer_idx,
            hyena_filter_groups=config.hcm_filter_groups,
            fir_inner_filter_length=config.hcm_filter_length,
        )
    if layer_idx in config.hcs_layer_idxs:
        return ParallelGatedConvBlock(
            config, layer_idx,
            hyena_filter_groups=config.hcs_filter_groups,
            fir_inner_filter_length=config.hcs_filter_length,
        )
    raise NotImplementedError(f"layer_idx {layer_idx} not in any block-type idxs")


def block_type_for_idx(config, layer_idx: int) -> str:
    if layer_idx in config.attn_layer_idxs: return "mha"
    if layer_idx in config.hcl_layer_idxs:  return "hcl"
    if layer_idx in config.hcm_layer_idxs:  return "hcm"
    if layer_idx in config.hcs_layer_idxs:  return "hcs"
    raise ValueError(f"block idx {layer_idx} not classified")
