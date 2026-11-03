"""Basic layers for Evo2 (StripedHyena2).

  * RMSNorm                -- pre/post norm in each block
  * ParallelGatedMLP       -- GLU(act(l1(x)) * l2(x)) -> l3(...); evo2_style
                              activations replace `act` with Identity for layer > 0
  * VocabParallelEmbedding -- single-process variant of vortex's
                              VocabParallelEmbedding so checkpoint keys match
  * TELinear               -- TransformerEngine FP8-capable Linear, with a
                              pure-PyTorch fallback when TE is not installed.
                              Used for the input QKV-like projections of
                              Hyena cascade blocks; required for 1B/20B/40B.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    from transformer_engine.pytorch import Linear as _TELinearBase
    from transformer_engine.common.recipe import Format, DelayedScaling
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False


def grab_first_if_tuple(x):
    if x.__class__.__name__ == "tuple":
        return x[0]
    return x


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.hidden_size = config.hidden_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)
        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func
            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        return self.scale * y


class ParallelGatedMLP(nn.Module):
    """GLU MLP. With evo2_style_activations=True, layer_idx > 0 uses Identity
    in place of the gating activation; layer 0 keeps gelu."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError(f"Unknown mlp_activation: {self.act_type}")

        if self.layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * config.model_parallel_size
        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        if config.get("inner_mlp_size", None) is not None:
            inner_size = config.inner_mlp_size

        self.l1 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, config.hidden_size, bias=False)

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class VocabParallelEmbedding(nn.Embedding):
    """Single-process variant of vortex's VocabParallelEmbedding.

    Drops distributed sharding so this minimal port runs on a single device
    (HF/accelerate handles cross-device placement via device_map="auto").
    """

    def __init__(self, config):
        vocab_size = config.vocab_size
        padding_idx = config.get("padding_idx", None)
        super().__init__(vocab_size, embedding_dim=config.hidden_size, padding_idx=padding_idx)

    def embed(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def unembed(self, u: Tensor) -> Tensor:
        return u @ self.weight.T


if HAS_TE:

    _TE_WORKSPACE_FIXUP_APPLIED = False

    def fixup_te_workspace():
        """Patch TE's Linear module to use per-device cuBLAS workspaces.

        Vortex helper, ported verbatim. Without this, TE's Linear uses a single
        workspace tensor allocated on whatever CUDA device was current at first
        call, then fails ("cuBLAS Error: the function failed to launch on the
        GPU") when called from layers on other devices -- which happens with
        device_map="auto" sharding. Idempotent; safe to call multiple times.
        No-op when TE is not installed (the import guard above gates it).
        """
        global _TE_WORKSPACE_FIXUP_APPLIED
        if _TE_WORKSPACE_FIXUP_APPLIED:
            return
        from functools import lru_cache

        @lru_cache
        def te_cublas_get_workspace_per_device(device):
            import transformer_engine.pytorch.module.base as tebase
            with torch.cuda.device(device):
                tebase._cublas_workspace = None  # force get_workspace() to reallocate
                return tebase.get_workspace()

        def get_workspace():
            return te_cublas_get_workspace_per_device(torch.cuda.current_device())

        import transformer_engine.pytorch.module.linear as telinear
        telinear.get_workspace = get_workspace
        _TE_WORKSPACE_FIXUP_APPLIED = True

    def set_format_recipe():
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
        return fp8_format, fp8_recipe

    class TELinear(_TELinearBase):
        """Wrapper for Transformer-Engine's Linear, matching vortex's signature.

        Returns (out, None) so callers can grab_first_if_tuple, matching vortex.
        """

        def __init__(
            self,
            input_size: int,
            output_size: int,
            init_method: Callable | None = None,
            bias: bool = True,
            skip_bias_add: bool = False,
            use_fp8: bool = False,
            **kwargs,
        ):
            params_dtype = torch.bfloat16
            self.te_return_bias = skip_bias_add and bias
            self.use_fp8_input_projections = use_fp8
            if use_fp8:
                self.fp8_format, self.fp8_recipe = set_format_recipe()

            super().__init__(
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=False,
                fuse_wgrad_accumulation=False,
                tp_group=None,
                tp_size=1,
                init_method=init_method,
                params_dtype=params_dtype,
                parallel_mode=None,
                bias=bias,
                return_bias=self.te_return_bias,
                **kwargs,
            )

        def forward(self, x):
            if self.use_fp8_input_projections:
                with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                    out = super().forward(x)
            else:
                out = super().forward(x)

            if self.te_return_bias:
                return out
            return out, None

else:

    def fixup_te_workspace():
        """No-op when TransformerEngine is not installed."""
        return

    class TELinear(nn.Module):
        """Pure-PyTorch fallback for TELinear (no FP8). Used by 7B variants
        when TransformerEngine isn't installed.

        Parameters are registered with names that match TE's state_dict layout
        (`weight`, `bias`), so checkpoints saved from either path are
        cross-loadable.
        """

        def __init__(
            self,
            input_size: int,
            output_size: int,
            init_method: Callable | None = None,
            bias: bool = True,
            skip_bias_add: bool = False,
            use_fp8: bool = False,
            **kwargs,
        ):
            super().__init__()
            if use_fp8:
                raise RuntimeError(
                    "FP8 requires Transformer Engine, which is not installed. "
                    "Install it with: pip install transformer_engine>=2.3.0"
                )

            self.te_return_bias = skip_bias_add and bias
            self.use_fp8_input_projections = False
            self.in_features = input_size
            self.out_features = output_size
            self.has_bias = bias

            self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=torch.bfloat16))
            if bias:
                self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.bfloat16))
            else:
                self.register_parameter("bias", None)

            if init_method is not None:
                init_method(self.weight)
            else:
                nn.init.xavier_uniform_(self.weight)

        def forward(self, x):
            out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
            if self.te_return_bias:
                return out, self.bias
            return out, None
