"""HuggingFace wrappers for Evo2 (StripedHyena2).

Two top-level model classes:

  * Evo2Model        -- bare backbone returning BaseModelOutputWithPast
                        (no LM head); the post-RMSNorm hidden state is the
                        last_hidden_state.
  * Evo2ForCausalLM  -- with the tied LM head + GenerationMixin so
                        model.generate() works out of the box.

Evo2Cache uses optimized recurrent MHA/Hyena state for unpadded prompts and
automatically uses exact full recomputation for padded batches.

Per-block dtype: bfloat16 for everything except the modal-form filter
parameters (`log_poles`, `residues`) which must stay fp32 for stability.
This is enforced both at convert time (the safetensors store these as fp32)
and automatically after checkpoint loading.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

from .cache import Evo2Cache
from .configuration_evo2 import Evo2Config
from .hyena import (
    AttentionBlock,
    ParallelGatedConvBlock,
    block_type_for_idx,
    get_block,
)
from .layers import RMSNorm, VocabParallelEmbedding
# Bundle the tokenizer file via trust_remote_code:
from .tokenization_evo2 import ByteTokenizer  # noqa: F401
# Force HF's trust_remote_code loader to copy these transitive deps into its
# dynamic-module cache (it only walks top-level `from .X import Y` patterns
# of modeling_evo2.py, not `from . import X`, so we use the explicit form).
from .attention import MHA as _MHA  # noqa: F401
from .engine import HyenaInferenceEngine as _HyenaInferenceEngine  # noqa: F401
from .rotary import (  # noqa: F401
    RotaryEmbedding as _RotaryEmbedding,
    restore_rotary_inv_freq,
)

logger = logging.get_logger(__name__)


# =============================================================================
# Backbone
# =============================================================================


class StripedHyena2(nn.Module):
    """Pure backbone: token embedding -> N blocks -> final RMSNorm."""

    def __init__(self, config: Evo2Config):
        super().__init__()
        # Patch TE's cuBLAS workspace handling so multi-GPU device_map="auto"
        # works (TE otherwise allocates a single workspace on the first device
        # it sees, then fails on layers placed on other devices). No-op when
        # TE isn't installed.
        from .layers import fixup_te_workspace
        fixup_te_workspace()

        self.config = config
        self.embedding_layer = VocabParallelEmbedding(config)
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None

        if config.get("use_flashfft", False):
            import importlib
            FlashFFTConv = importlib.import_module("flashfftconv").FlashFFTConv
            self.flash_fft = FlashFFTConv(2 * config.max_seqlen, dtype=torch.bfloat16)
        else:
            self.flash_fft = None

        self.blocks = nn.ModuleList(
            get_block(config, i) for i in range(config.num_layers)
        )
        # Wire fftconv_fn into the hcl filters when flashfft is on.
        if self.flash_fft is not None:
            for block in self.blocks:
                if isinstance(block, ParallelGatedConvBlock) and block.filter.h is None:
                    block.filter.fftconv_fn = self.flash_fft

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Evo2Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Run the backbone using the standard HuggingFace base-model API."""
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # The bare backbone defaults to no cache because embedding extraction
        # does not benefit from autoregressive state.
        use_cache = use_cache if use_cache is not None else False
        if use_cache and self.training:
            use_cache = False
        if attention_mask is not None:
            attention_mask = attention_mask.ne(0)

        suffix_only = False
        requested_length = input_ids.shape[1]
        if use_cache:
            if not isinstance(past_key_values, Evo2Cache):
                past_key_values = self.initialize_inference_params(
                    max_batch_size=input_ids.shape[0],
                )
            has_padding = (
                attention_mask is not None
                and not attention_mask.bool().all()
            )
            if past_key_values.mode is None:
                past_key_values.mode = (
                    "recompute" if has_padding else "recurrent"
                )
            if past_key_values.mode == "recurrent" and has_padding:
                raise ValueError(
                    "A recurrent Evo2 cache cannot switch to padded inputs. "
                    "Reset the cache and prefill the padded batch again."
                )
            if past_key_values.mode == "recurrent":
                cached_length = past_key_values.get_seq_length()
                if cached_length > 0 and input_ids.shape[1] > cached_length:
                    input_ids = input_ids[:, cached_length:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, cached_length:]
            else:
                input_ids, attention_mask, suffix_only = past_key_values.prepare_inputs(
                    input_ids,
                    attention_mask,
                )

        last_hidden, past_kv, hidden_states, attentions = self._forward_features(
            input_ids,
            inference_params_dict=(
                past_key_values
                if use_cache and past_key_values.mode == "recurrent"
                else None
            ),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if use_cache:
            past_kv = past_key_values
            if past_key_values.mode == "recurrent":
                past_key_values.advance(input_ids.shape[1])
            else:
                past_key_values.update_history(input_ids, attention_mask)
            if suffix_only:
                last_hidden = last_hidden[:, -requested_length:]
                if output_hidden_states:
                    hidden_states = [
                        state[:, -requested_length:]
                        for state in hidden_states
                    ]
                if output_attentions:
                    attentions = [
                        (
                            attention[..., -requested_length:, :]
                            if attention is not None
                            else None
                        )
                        for attention in attentions
                    ]

        if not return_dict:
            outputs = (last_hidden,)
            if use_cache:
                outputs += (past_kv,)
            if output_hidden_states:
                outputs += (tuple(hidden_states),)
            if output_attentions:
                outputs += (tuple(attentions),)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden,
            past_key_values=past_kv if use_cache else None,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )

    def _forward_features(
        self,
        input_ids: torch.LongTensor,
        inference_params_dict: Optional[Evo2Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        x = self.embedding_layer.embed(input_ids)

        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[Optional[torch.Tensor]] = []
        if output_hidden_states:
            all_hidden_states.append(x)

        if inference_params_dict is not None:
            x, params_out = self._stateful_forward(
                x, inference_params_dict,
                all_hidden_states=all_hidden_states,
                all_attentions=all_attentions,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            x, params_out = self._stateless_forward(
                x, padding_mask=attention_mask,
                all_hidden_states=all_hidden_states,
                all_attentions=all_attentions,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

        if self.norm is not None:
            x = self.norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        return x, params_out, all_hidden_states, all_attentions

    def get_input_embeddings(self):
        return self.embedding_layer

    def set_input_embeddings(self, value):
        self.embedding_layer = value

    def _stateful_forward(
        self, x, cache: Evo2Cache,
        all_hidden_states, all_attentions,
        output_hidden_states, output_attentions,
    ):
        for block_idx, block in enumerate(self.blocks):
            block_name = block_type_for_idx(self.config, block_idx)
            inference_params = cache.by_block_name(block_name)
            x, attn = block(
                x, inference_params=inference_params,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                all_hidden_states.append(x)
            if output_attentions:
                all_attentions.append(attn)
        return x, cache

    def _stateless_forward(
        self, x, padding_mask,
        all_hidden_states, all_attentions,
        output_hidden_states, output_attentions,
    ):
        if isinstance(padding_mask, torch.Tensor):
            x = x * padding_mask[..., None]
        for block in self.blocks:
            x, attn = block(
                x, inference_params=None, padding_mask=padding_mask,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                all_hidden_states.append(x)
            if output_attentions:
                all_attentions.append(attn)
        return x, None

    def initialize_inference_params(self, max_batch_size: int = 1) -> Evo2Cache:
        return Evo2Cache(
            max_seqlen=self.config.get("max_seqlen", 8192),
            max_batch_size=max_batch_size,
            short_filter_length=self.config.short_filter_length,
            hcm_filter_length=self.config.hcm_filter_length,
            hcs_filter_length=self.config.hcs_filter_length,
            state_size=self.config.state_size,
        )

    def to_dtype_except_poles_residues(self, dtype):
        """Cast params to bf16, restore fp32 invariants:

          * log_poles / residues -- bf16 collapses the modal-form IIR (the
            poles get rounded too aggressively, the recurrence blows up).
            Mirrors vortex's ``to_bfloat16_except_pr_lc(to_float32=True)``
            first-pass behaviour.

          * rotary_emb.inv_freq -- HF ``from_pretrained(dtype=bf16)`` casts
            ALL buffers to bf16. ``inv_freq = 1 / base^(2i/dim)`` loses ~7
            bits of mantissa in bf16, which shifts cos/sin tables by ~5e-2
            per cell at position 64+, causing each attention layer to add
            ~4e-2 of noise on Q/K. We recompute inv_freq in fp32 here so
            the cos/sin builder uses the precise stored value.
        """
        for k, p in self.named_parameters():
            if "log_poles" not in k and "residues" not in k:
                p.data = p.data.to(dtype)
        self.restore_precision_invariants()

    def to_bfloat16_except_poles_residues(self):
        self.to_dtype_except_poles_residues(torch.bfloat16)

    def restore_precision_invariants(self):
        """Restore exact RoPE frequencies and fp32 modal-filter parameters."""
        for k, p in self.named_parameters():
            if (
                ("log_poles" in k or "residues" in k)
                and p.dtype != torch.float32
            ):
                p.data = p.data.to(torch.float32)
        for module in self.modules():
            if hasattr(module, "inv_freq"):
                restore_rotary_inv_freq(module)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.restore_precision_invariants()
        return result


# =============================================================================
# HF wrappers
# =============================================================================


class Evo2PreTrainedModel(PreTrainedModel):
    config_class = Evo2Config
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq", r"\.t$", r"_extra_state"]
    _keys_to_ignore_on_load_unexpected = [
        r"fftconv",
        r"twiddle_factors",
        r"_extra_state$",
        r"\.filter\.t$",
        r"rotary_emb\.inv_freq",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    # Evo2 cache is a custom Evo2Cache (not a DynamicCache); tell HF not to wrap it.
    _supports_cache_class = False
    # log_poles / residues parameterize a modal long-range filter; bf16 collapses
    # them. HF will keep these in fp32 even with dtype=bf16 at load time.
    _keep_in_fp32_modules = ["log_poles", "residues"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requested_dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if requested_dtype == torch.float16:
            raise ValueError(
                "Evo2 does not support float16 inference because its modal "
                "filters are numerically unstable in FP16. Use bfloat16 "
                "(recommended) or float32."
            )
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = torch.bfloat16
            requested_dtype = torch.bfloat16
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if isinstance(requested_dtype, torch.dtype):
            model.backbone.to_dtype_except_poles_residues(requested_dtype)
        else:
            model.backbone.restore_precision_invariants()
        return loaded


class Evo2Model(Evo2PreTrainedModel):
    """Bare backbone returning BaseModelOutputWithPast."""

    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.backbone = StripedHyena2(config)
        self.config = config
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues()

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def set_input_embeddings(self, value):
        self.backbone.embedding_layer = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Evo2ForCausalLM(Evo2PreTrainedModel, GenerationMixin):
    """LM head wrapper. Tied to backbone.embedding_layer (Evo2 ties weights)."""

    def __init__(self, config: Evo2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = StripedHyena2(config)
        self.config = config

        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )
        self.vocab_size = vocab_size
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues()

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def set_input_embeddings(self, value):
        self.backbone.embedding_layer = value

    def get_output_embeddings(self):
        return self.backbone.embedding_layer

    def set_output_embeddings(self, value):
        self.backbone.embedding_layer = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and labels is not None:
            logger.warning_once("use_cache=True is incompatible with loss computation; disabling.")
            use_cache = False
        inputs = input_ids

        backbone_outputs = self.backbone(
            input_ids=inputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden = backbone_outputs.last_hidden_state
        past_kv = backbone_outputs.past_key_values
        hidden_states = backbone_outputs.hidden_states
        attentions = backbone_outputs.attentions

        logits = last_hidden @ self.backbone.embedding_layer.weight.T

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            outputs = (logits,)
            if use_cache: outputs += (past_kv,)
            if output_hidden_states: outputs += (hidden_states,)
            if output_attentions: outputs += (attentions,)
            if loss is not None: outputs = (loss,) + outputs
            return outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv if use_cache else None,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, past_key_values=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", self.config.use_cache),
        }
