# -*- coding: utf-8 -*-
"""
HyenaDNA custom code port to Hugging Face Hub.
This module provides HyenaDNA model implementations with support for:
- Causal (unidirectional) mode for autoregressive tasks
- Bidirectional mode for masked language modeling (MLM)
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .configuration_hyena import HyenaConfig


def fftconv(
    u: torch.Tensor,
    k: torch.Tensor,
    D: torch.Tensor,
    bidirectional: bool = False,
    k_rev: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply convolution through the Fourier domain (Convolution Theorem).

    Two mechanisms exist for non-causal context, and they are NOT equivalent:

    ``k_rev`` (preferred) supplies a second, anti-causal kernel that is folded
    in as ``k_f + conj(k_rev_f)``. Conjugation in the frequency domain is
    time reversal, so the anti-causal taps land on the negative-time half of
    the circular buffer. This yields a genuinely two-sided filter: output
    position ``n`` depends on inputs both before and after ``n``.

    ``bidirectional`` (legacy, kept for checkpoint compatibility) pads the
    input symmetrically and leaves ``fft_size`` at ``2 * seqlen``. Because the
    padded signal then exactly fills the FFT window, the linear result of
    length ``2 * seqlen + len(k) - 1`` no longer fits and wraps around. With a
    causal kernel this branch computes a causal filter *delayed* by
    ``seqlen // 2`` plus wrap-around aliasing at the front -- it does not
    create future dependence. See ``fftconv_ref`` in HazyResearch/hyena-dna,
    from which this behaviour is inherited verbatim.

    Args:
        u (torch.Tensor): Input tensor of shape (B, D, L).
        k (torch.Tensor): Causal kernel tensor.
        D (torch.Tensor): Bias/skip connection tensor.
        bidirectional (bool): Use the legacy symmetric-padding branch.
        k_rev (torch.Tensor | None): Anti-causal kernel. When given, the
            symmetric padding is skipped -- the two mechanisms are
            alternatives, not complements.

    Returns:
        torch.Tensor: Output tensor of same shape as input.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev.to(torch.float32), n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()

    if bidirectional and k_rev is None:
        # Pad 1/2 before and 1/2 after the input for bidirectional convolution
        padded_length = seqlen + 2 * (seqlen // 2)
        pad_before = padded_length // 2 - (seqlen // 2)
        pad_after = padded_length - seqlen - pad_before
        padded_u = F.pad(u, (pad_before, pad_after), mode="constant", value=0)
        u_f = torch.fft.rfft(padded_u.to(dtype=torch.float32), n=fft_size)
    else:
        u_f = torch.fft.rfft(u.to(dtype=torch.float32), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class HyenaSin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""

    def __init__(self, config):
        super().__init__()
        self.freq = (
            nn.Parameter(config.activation_freq * torch.ones(1, config.filter_order))
            if config.train_freq
            else config.activation_freq * torch.ones(1, config.filter_order)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class HyenaPositionalEmbedding(nn.Module):
    def __init__(self, config):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = config.max_seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if config.emb_dim > 1:
            bands = (config.emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / self.seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]

        z = torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], dim=-1)

        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class HyenaExponentialModulation(nn.Module):
    """The window function applied to the output of the (MLP) filter function."""

    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulate: bool = True,
        shift: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    """Implicit long filter with modulation and bidirectional support."""

    def __init__(self, config, **kwargs):
        """
        Initialize the Hyena filter.
        Args:
            config (HyenaConfig): Model configuration with attributes:
                - d_model: number of channels in the input
                - emb_dim: dimension of the positional encoding
                - filter_order: width of the FFN
                - num_inner_mlps: number of inner linear layers
                - bidirectional: whether to use bidirectional convolution
        """
        super().__init__()

        self.d_model = config.d_model * (config.hyena_order - 1)
        self.use_bias = config.use_bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(config.hyena_filter_dropout)
        self.bidirectional = getattr(config, "bidirectional", False)

        act = HyenaSin(config)
        self.emb_dim = config.emb_dim
        assert self.emb_dim % 2 != 0 and self.emb_dim >= 3, (
            "emb_dim must be odd and >= 3 (time, sine and cosine)"
        )
        self.seq_len = config.max_seq_len

        self.pos_emb = HyenaPositionalEmbedding(config)

        self.implicit_filter = nn.Sequential(
            nn.Linear(self.emb_dim, config.filter_order),
            act,
        )
        for i in range(config.num_inner_mlps):
            self.implicit_filter.append(
                nn.Linear(config.filter_order, config.filter_order)
            )
            self.implicit_filter.append(act)

        self.implicit_filter.append(
            nn.Linear(config.filter_order, config.d_model, bias=False)
        )

        self.modulation = HyenaExponentialModulation(config.d_model)

        # ``bidirectional_mode`` selects HOW non-causal context is obtained:
        #   "padding" (default) -- legacy symmetric zero-padding. Preserved as
        #       the default so existing checkpoints keep bit-identical
        #       behaviour, but note it does not actually produce future
        #       dependence (see ``fftconv``).
        #   "k_rev" -- a second, anti-causal implicit filter folded in as
        #       conj(k_rev) in the frequency domain. This is the mechanism
        #       ``fftconv_ref`` upstream exposes but never feeds.
        self.bidirectional_mode = str(
            getattr(config, "bidirectional_mode", "padding")
        )
        if self.bidirectional_mode not in ("padding", "k_rev"):
            raise ValueError(
                "bidirectional_mode must be one of {'padding', 'k_rev'}; "
                f"got {self.bidirectional_mode!r}."
            )

        self.implicit_filter_rev = None
        if self.bidirectional and self.bidirectional_mode == "k_rev":
            rev = nn.Sequential(
                nn.Linear(self.emb_dim, config.filter_order),
                HyenaSin(config),
            )
            for _ in range(config.num_inner_mlps):
                rev.append(nn.Linear(config.filter_order, config.filter_order))
                rev.append(HyenaSin(config))
            rev.append(nn.Linear(config.filter_order, config.d_model, bias=False))
            self.implicit_filter_rev = rev
            self.modulation_rev = HyenaExponentialModulation(config.d_model)

        self.normalized = False

    def filter(self, L, *args, **kwargs):
        """
        Generate the implicit filter for sequence length L.
        Args:
            L (int): Sequence length.
        Returns:
            torch.Tensor: Filter tensor.
        """
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z.to(dtype=self.implicit_filter[0].weight.dtype))
        h = self.modulation(t, h)
        return h

    def filter_rev(self, L: int) -> "torch.Tensor | None":
        """Generate the anti-causal filter for sequence length L.

        Args:
            L (int): Sequence length.

        Returns:
            torch.Tensor | None: Reverse filter, or None when the anti-causal
                branch is disabled.
        """
        if self.implicit_filter_rev is None:
            return None
        z, t = self.pos_emb(L)
        dtype = self.implicit_filter_rev[0].weight.dtype
        h = self.implicit_filter_rev(z.to(dtype=dtype))
        return self.modulation_rev(t, h)

    def forward(
        self, x, L, k=None, bias=None, k_rev: "torch.Tensor | None" = None,
        *args, **kwargs,
    ):
        """
        Apply the filter to input.
        Args:
            x (torch.Tensor): Input tensor.
            L (int): Sequence length.
            k (torch.Tensor | None): Optional pre-computed kernel.
            bias (torch.Tensor | None): Optional bias.
            k_rev (torch.Tensor | None): Pre-shaped anti-causal kernel for this
                order slice. Supplied by ``HyenaOperator``; derived here only
                when the filter is called standalone.
        Returns:
            torch.Tensor: Filtered output.
        """
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        if k_rev is None and self.implicit_filter_rev is not None:
            # Standalone call (no HyenaOperator to pre-shape the kernels).
            k_rev = self.filter_rev(L)
            k_rev = k_rev[0] if type(k_rev) is tuple else k_rev

        y = fftconv(x, k, bias, bidirectional=self.bidirectional, k_rev=k_rev)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        config,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        self.d_model = config.d_model
        self.l_max = config.max_seq_len
        self.order = config.hyena_order
        inner_width = config.d_model * (self.order + 1)
        self.dropout = nn.Dropout(config.hyena_dropout)
        self.in_proj = nn.Linear(self.d_model, inner_width)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            config.short_filter_order,
            padding=2,
            groups=inner_width,
        )
        self.filter_fn = HyenaFilter(config)

    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u).transpose(1, 2)

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
        bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

        # The anti-causal companion kernel needs the identical reshape so that
        # each order slice lines up with its causal counterpart.
        k_rev = self.filter_fn.filter_rev(l_filter)
        if k_rev is not None:
            k_rev = k_rev[0].transpose(0, 1).reshape(
                self.order - 1, self.d_model, l_filter
            )

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(
                v,
                l_filter,
                k=k[o],
                bias=bias[o],
                k_rev=None if k_rev is None else k_rev[o],
            )

        y = (v * x[0]).transpose(1, 2)

        y = self.out_proj(y)
        return y


class HyenaMlp(nn.Module):
    def __init__(self, config):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        super().__init__()
        in_features = config.d_model
        hidden_features = config.d_inner
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.d_model)

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y, approximate="tanh")
        y = self.fc2(y)
        return y


class HyenaBlock(nn.Module):
    def __init__(self, config):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.mixer = HyenaOperator(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.mlp = HyenaMlp(config)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        residual = hidden_states
        residual = residual.to(torch.float32)
        hyena_normed = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        hidden_states = self.mixer(hyena_normed)
        # Tested above here and all is equivalent. That means the mixer is fine!!!
        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454


class HyenaEmbeddings(nn.Module):
    def __init__(self, config, padding_idx=None):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
            the project up to embed_dim
        """
        super().__init__()
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )
        self.word_embeddings = nn.Embedding(
            vocab_size, config.d_model, padding_idx=padding_idx
        )

    def forward(self, input_ids):
        """
        input_ids: (batch, seqlen)
        """
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class HyenaLMBackbone(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = HyenaEmbeddings(config)
        self.dropout = nn.Dropout(config.embed_dropout)

        self.layers = nn.ModuleList([HyenaBlock(config) for i in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states
                )
            else:
                hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states.to(dtype=self.ln_f.weight.dtype))
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


class HyenaDNAPreTrainedModel(PreTrainedModel):
    config_class = HyenaConfig
    base_model_prefix = "hyena"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HyenaBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [
        r"freq"
    ]  # Shared tensors that safetensors merges

    def _init_weights(self, module, initializer_range=0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in self.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=initializer_range / math.sqrt(2 * self.config.num_layers),
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=initializer_range / math.sqrt(2 * self.config.num_layers),
                )


class HyenaDNAModel(HyenaDNAPreTrainedModel):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.backbone = HyenaLMBackbone(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, input_ids, inputs_embeds=None, output_hidden_states=None, return_dict=None
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class HyenaDNAForCausalLM(HyenaDNAPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.hyena = HyenaDNAModel(config)
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.hyena.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        self.hyena.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder) -> None:
        self.hyena = decoder

    def get_decoder(self):
        return self.hyena

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.hyena(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class HyenaDNAForSequenceClassification(HyenaDNAPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.hyena = HyenaDNAModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.hyena.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        self.hyena.backbone.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.hyena(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
        )


class HyenaDNAForMaskedLM(HyenaDNAPreTrainedModel):
    """HyenaDNA model for Masked Language Modeling (MLM)."""

    def __init__(self, config, **kwargs):
        """
        Initialize the MLM model.
        Args:
            config (HyenaConfig): Model configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config, **kwargs)
        self.hyena = HyenaDNAModel(config)
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.hyena.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        """Set input embeddings."""
        self.hyena.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        """Get output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings) -> None:
        """Set output embeddings (LM head)."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        """
        Forward pass for MLM.
        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (B, L).
            inputs_embeds (torch.FloatTensor | None): Optional input embeddings.
            labels (torch.LongTensor | None): Labels for MLM loss.
                Tokens with -100 are ignored in loss computation.
            output_hidden_states (bool | None): Whether to output hidden states.
            return_dict (bool | None): Whether to return a dict or tuple.
            **kwargs: Additional arguments (ignored).
        Returns:
            Union[Tuple, MaskedLMOutput]: Model outputs with loss and logits.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.hyena(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # MLM loss: no shift needed (unlike causal LM)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
