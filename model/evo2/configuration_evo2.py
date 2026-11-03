"""Configuration for Evo2 (StripedHyena2)."""

from __future__ import annotations

import json
from typing import List

from transformers import PretrainedConfig


class Evo2Config(PretrainedConfig):
    """Evo2 config.

    Defaults match evo2_7b_base (32-layer, 4096-hidden). Per-variant overrides
    are written into the config.json of each repo by convert_checkpoint.py.

    Block dispatch is driven by four index lists:
      attn_layer_idxs : transformer (MHA + RoPE) blocks
      hcl_layer_idxs  : Hyena Cascade Long  (IIR via log_poles + residues)
      hcm_layer_idxs  : Hyena Cascade Medium (FIR, fir_inner_filter_length=128)
      hcs_layer_idxs  : Hyena Cascade Short  (FIR, fir_inner_filter_length=7)
    Their disjoint union must equal range(num_layers).
    """

    model_type = "evo2"

    def __init__(
        self,
        # Architecture core
        vocab_size: int = 512,
        hidden_size: int = 4096,
        num_filters: int = 4096,
        inner_mlp_size: int = 11008,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        # Block dispatch
        attn_layer_idxs: List[int] | None = None,
        hcl_layer_idxs: List[int] | None = None,
        hcm_layer_idxs: List[int] | None = None,
        hcs_layer_idxs: List[int] | None = None,
        # Filter geometry
        hcm_filter_length: int = 128,
        hcs_filter_length: int = 7,
        hcl_filter_groups: int = 4096,
        hcm_filter_groups: int = 256,
        hcs_filter_groups: int = 256,
        hyena_filter_groups: int = 1,
        short_filter_length: int = 3,
        short_filter_bias: bool = False,
        state_size: int = 16,
        # Channel-split conventions
        column_split: bool = True,
        column_split_hyena: bool = False,
        interleave: bool = True,
        hyena_flip_x1x2: bool = False,
        # Norms / activations
        eps: float = 1e-6,
        final_norm: bool = True,
        mlp_activation: str = "gelu",
        evo2_style_activations: bool = True,
        # Linear biases
        mha_out_proj_bias: bool = True,
        hyena_out_proj_bias: bool = True,
        qkv_proj_bias: bool = False,
        # MLP geometry
        inner_size_multiple_of: int = 16,
        make_vocab_size_divisible_by: int = 8,
        # Embeddings
        tie_embeddings: bool = True,
        # Rotary / sequence
        max_seqlen: int = 8192,
        max_batch_size: int = 1,
        rotary_emb_base: float = 10000.0,
        use_interpolated_rotary_pos_emb: bool = False,
        rotary_emb_scaling_factor: float = 1.0,
        # Inference engine
        prefill_style: str = "fft",
        inference_mode: bool = True,
        # Projection precision (TE FP8 - required for 1b / 20b / 40b)
        use_fp8_input_projections: bool = False,
        # Backend toggles
        use_cache: bool = True,
        # Vortex used this flag at runtime. Our HF port ignores it entirely
        # (attention dispatch is driven by ``config._attn_implementation``,
        # the standard HF mechanism). Default is False here so the config
        # reflects what actually runs: SDPA by default unless the user passes
        # ``attn_implementation="flash_attention_2"`` to from_pretrained.
        use_flash_attn: bool = False,
        use_flash_rmsnorm: bool = False,
        use_flash_depthwise: bool = False,
        use_flashfft: bool = False,
        # Per-block dtypes (cast at module init)
        attn_block_dtype: str = "bfloat16",
        hyena_block_dtype: str = "bfloat16",
        mlp_dtype: str = "bfloat16",
        # Multi-tensor parallel knobs (kept for ParallelGatedMLP._compute_inner_size)
        model_parallel_size: int = 1,
        pipe_parallel_size: int = 1,
        # GQA
        proj_groups: int = 1,
        smeared_gqa: bool = False,
        **kwargs,
    ):
        if attn_layer_idxs is None:
            attn_layer_idxs = [3, 10, 17, 24, 31]
        if hcl_layer_idxs is None:
            hcl_layer_idxs = [2, 6, 9, 13, 16, 20, 23, 27, 30]
        if hcm_layer_idxs is None:
            hcm_layer_idxs = [1, 5, 8, 12, 15, 19, 22, 26, 29]
        if hcs_layer_idxs is None:
            hcs_layer_idxs = [0, 4, 7, 11, 14, 18, 21, 25, 28]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inner_mlp_size = inner_mlp_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads

        self.attn_layer_idxs = attn_layer_idxs
        self.hcl_layer_idxs = hcl_layer_idxs
        self.hcm_layer_idxs = hcm_layer_idxs
        self.hcs_layer_idxs = hcs_layer_idxs

        self.hcm_filter_length = hcm_filter_length
        self.hcs_filter_length = hcs_filter_length
        self.hcl_filter_groups = hcl_filter_groups
        self.hcm_filter_groups = hcm_filter_groups
        self.hcs_filter_groups = hcs_filter_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.short_filter_length = short_filter_length
        self.short_filter_bias = short_filter_bias
        self.state_size = state_size

        self.column_split = column_split
        self.column_split_hyena = column_split_hyena
        self.interleave = interleave
        self.hyena_flip_x1x2 = hyena_flip_x1x2

        self.eps = eps
        self.final_norm = final_norm
        self.mlp_activation = mlp_activation
        self.evo2_style_activations = evo2_style_activations

        self.mha_out_proj_bias = mha_out_proj_bias
        self.hyena_out_proj_bias = hyena_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias

        self.inner_size_multiple_of = inner_size_multiple_of
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by

        self.tie_embeddings = tie_embeddings

        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.rotary_emb_base = rotary_emb_base
        self.use_interpolated_rotary_pos_emb = use_interpolated_rotary_pos_emb
        self.rotary_emb_scaling_factor = rotary_emb_scaling_factor

        self.prefill_style = prefill_style
        self.inference_mode = inference_mode

        self.use_fp8_input_projections = use_fp8_input_projections

        self.use_cache = use_cache
        self.use_flash_attn = use_flash_attn
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flashfft = use_flashfft

        self.attn_block_dtype = attn_block_dtype
        self.hyena_block_dtype = hyena_block_dtype
        self.mlp_dtype = mlp_dtype

        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size

        self.proj_groups = proj_groups
        self.smeared_gqa = smeared_gqa

        super().__init__(**kwargs)

    # HF generation helpers expect `num_hidden_layers`.
    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    # Internal blocks were originally written against a dotdict and call
    # `config.get(key, default)` extensively; provide a dict-like getter.
    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_original_config(cls, config_path: str, **kwargs) -> "Evo2Config":
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config, **kwargs)
