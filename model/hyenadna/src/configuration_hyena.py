"""
HyenaDNA configuration for Hugging Face.
This module defines the configuration class for HyenaDNA models,
including support for bidirectional mode.
"""

import json

from transformers import PretrainedConfig


class HyenaConfig(PretrainedConfig):
    """
    Configuration class for HyenaDNA models.
    Supports both causal (unidirectional) and bidirectional modes.
    """

    model_type = "hyenadna"

    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 256,
        d_inner: int = None,
        use_bias: bool = True,
        train_freq: bool = True,
        max_seq_len: int = 1024,
        emb_dim: int = 3,
        n_layer: int = 12,
        num_inner_mlps: int = 2,
        hyena_order: int = 2,
        short_filter_order: int = 3,
        filter_order: int = 64,
        activation_freq: int = 1,
        embed_dropout: float = 0.1,
        hyena_dropout: float = 0.0,
        hyena_filter_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        pad_vocab_size_multiple: int = 8,
        bidirectional: bool = False,
        bidirectional_mode: str = "padding",
        **kwargs,
    ):
        """
        Initialize the HyenaConfig.
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Hidden dimension size.
            d_inner (int): Inner dimension for MLP (default: 4 * d_model).
            use_bias (bool): Whether to use bias in linear layers.
            train_freq (bool): Whether to train frequency parameters.
            max_seq_len (int): Maximum sequence length.
            emb_dim (int): Embedding dimension for positional encoding.
            n_layer (int): Number of Hyena blocks.
            num_inner_mlps (int): Number of inner MLP layers in filter.
            hyena_order (int): Order of the Hyena recurrence.
            short_filter_order (int): Kernel size for short convolution.
            filter_order (int): Width of the FFN for implicit filter.
            activation_freq (int): Frequency for sin activation.
            embed_dropout (float): Dropout rate for embeddings.
            hyena_dropout (float): Dropout rate for Hyena operator.
            hyena_filter_dropout (float): Dropout rate for Hyena filter.
            layer_norm_epsilon (float): Epsilon for layer normalization.
            initializer_range (float): Standard deviation for weight init.
            pad_vocab_size_multiple (int): Pad vocab size to this multiple.
            bidirectional (bool): Whether to use bidirectional convolution.
            bidirectional_mode (str): How non-causal context is obtained --
                "padding" (legacy symmetric zero-padding, default) or
                "k_rev" (anti-causal companion kernel).
            **kwargs: Additional arguments for PretrainedConfig.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        if d_inner is None:
            self.d_inner = 4 * d_model
        else:
            self.d_inner = d_inner
        self.use_bias = use_bias
        self.train_freq = train_freq
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.n_layer = n_layer
        self.hyena_order = hyena_order
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.activation_freq = activation_freq
        self.num_inner_mlps = num_inner_mlps
        self.embed_dropout = embed_dropout
        self.hyena_dropout = hyena_dropout
        self.hyena_filter_dropout = hyena_filter_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.bidirectional = bidirectional
        self.bidirectional_mode = bidirectional_mode
        super().__init__(**kwargs)

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path, "r") as f:
            config = json.load(f)

        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        d_inner = config["d_inner"]
        max_seq_len = config["layer"]["l_max"]
        emb_dim = config["layer"]["emb_dim"]
        filter_order = config["layer"]["filter_order"]
        if "local_order" in config["layer"]:
            short_filter_order = config["layer"]["local_order"]
        elif "short_filter_order" in config["layer"]:
            short_filter_order = config["layer"]["short_filter_order"]
        else:
            short_filter_order = 3
        n_layer = config["n_layer"]
        activation_freq = config["layer"]["w"]
        embed_dropout = config["embed_dropout"]
        pad_vocab_size_multiple = config["pad_vocab_size_multiple"]
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            d_inner=d_inner,
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            filter_order=filter_order,
            short_filter_order=short_filter_order,
            n_layer=n_layer,
            activation_freq=activation_freq,
            embed_dropout=embed_dropout,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            tie_word_embeddings=False,
            **kwargs,
        )
