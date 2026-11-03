"""CPU-servable hybrid splice-site annotation model.

A nucleotide-resolution dilated CNN branch (local motifs: donor/acceptor
consensus, branch point, polypyrimidine tract) runs in parallel with a
downsampled Transformer branch (coarse, content-dependent long-range context).
The two are combined by gated residual fusion and read out as per-position
3-class logits.

Only standard PyTorch ops are used -- no Mamba/selective scan, Triton, custom
CUDA extensions, FlashAttention package, Hyena kernels, or full-resolution
attention over 15,000 positions -- so the graph traces cleanly for TorchScript /
ONNX Runtime and runs on CPU.

I/O matches the project's existing chunk layout (see
``config/train/train_med_ac_ep6_reference.yaml`` and
``data/h5_cl10000_sl5000_s1250_regions``): a 15,000-nt input is
5,000-nt left flank + 5,000-nt supervised target + 5,000-nt right flank, and the
model predicts the central 5,000 positions.

Note on class order: this module follows the REPO convention
``(acceptor, donor, neither)``, matching ``class_weights: [4.0, 4.0, 1.0]`` in
the canonical training recipe and the existing evaluation scripts. The Illumina
SpliceAI distribution uses ``(neither, acceptor, donor)`` instead; set
``ModelConfig.class_order`` if you need that ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Repo-wide class ordering for the 3-class splice head.
CLASS_ORDER: tuple[str, str, str] = ("acceptor", "donor", "neither")

#: Illumina SpliceAI ordering, kept here so conversions are explicit.
SPLICEAI_CLASS_ORDER: tuple[str, str, str] = ("neither", "acceptor", "donor")

_NORMALIZATIONS = ("batch", "group", "layer")
_POSITIONAL_ENCODINGS = ("sinusoidal", "learned", "relative", "none")
_FUSIONS = ("gated", "concat", "film", "add")
_GLOBAL_BRANCH_TYPES = ("transformer", "se")
_LOCAL_BRANCH_TYPES = ("dilated", "unet")


@dataclass
class ModelConfig:
    """Configuration for :class:`HybridSpliceModel`.

    Every contested design choice is a switch here so the open questions can be
    settled by experiment rather than by assertion. See ``ABLATION_PRESETS`` for
    the arms of the planned comparison.
    """

    # --- I/O ---------------------------------------------------------------
    in_channels: int = 5
    """5 matches the repo's token encoding directly: 0=N/pad, 1=A, 2=C, 3=G,
    4=T (see process_input/h5_dataset_builder.py), so a one-hot of the stored
    int8 token is the model input as-is. Use 4 if N is dropped to an all-zero
    column instead."""
    input_length: int = 15000
    output_length: Optional[int] = 5000
    """Central crop length. ``None`` returns all ``input_length`` positions."""
    num_classes: int = 3
    class_order: tuple[str, ...] = CLASS_ORDER

    # --- local (nucleotide-resolution) CNN branch --------------------------
    local_branch_type: str = "dilated"
    """'dilated' runs every block at full 15,000-nt resolution. 'unet' runs a
    few blocks there and the rest at 1/4 and 1/16 resolution with skip
    connections, so the deep blocks cover far more sequence per unit of
    compute -- aimed at the deep-intronic strata, where the dilated stack
    trails the large backbones by ~0.05 AUROC."""
    local_channels: int = 32
    kernel_size: int = 11
    dilations: tuple[int, ...] = (1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    groups: int = 1
    """1 = dense conv, ``local_channels`` = depthwise, anything between =
    grouped.

    Default is dense on measured evidence (job 12235, AMD EPYC 7763, batch 16,
    16 threads): grouping buys FLOPs but not wall-clock. Cutting FLOPs 6.1x via
    depthwise made the model only 1.18x faster, and groups=8/64 were SLOWER
    than groups=4 despite fewer FLOPs -- the stack is memory-bound, and small
    per-group GEMMs block poorly. Dense sustained 124-166 GFLOP/s against
    32-72 for grouped. Compare at fixed wall-clock, not fixed FLOPs."""

    unet_blocks_per_level: tuple[int, ...] = (3, 4, 7)
    """Residual blocks at each resolution, finest first. Within a level the
    dilations are (1, 1, 2, 4, ...), so level 2's 7 blocks reach 641 tokens =
    ~10,256 nt -- matching the dilated stack's receptive field at 1/16 the
    positions."""
    unet_channel_mult: tuple[int, ...] = (1, 2, 2)
    """Channel width per level as a multiple of ``local_channels``."""
    unet_stride: int = 4
    """Resolution reduction between consecutive levels."""

    # --- global (downsampled) branch ---------------------------------------
    global_dim: int = 96
    downsample_factor: int = 16
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    ffn_dim: int = 192
    positional_encoding: str = "sinusoidal"
    relative_max_distance: int = 128
    """Clip distance for ``positional_encoding='relative'`` (in coarse tokens)."""

    # --- shared ------------------------------------------------------------
    dropout: float = 0.1
    normalization: str = "batch"
    norm_num_groups: int = 8

    # --- ablation switches -------------------------------------------------
    use_local_branch: bool = True
    use_global_branch: bool = True
    global_branch_type: str = "transformer"
    fusion: str = "gated"
    gate_bias_init: float = 0.0
    """Bias init for the fusion gate. A positive value (e.g. 1.0 -> sigmoid
    0.73) keeps the global path alive early; useful if the gate collapses."""
    refine_kernel_size: int = 3
    aux_region_head: bool = False
    """Binary exon/intron auxiliary head (the SplicePath recipe). Free at
    inference -- drop the head -- and costs one 1x1 conv at training time."""

    def __post_init__(self) -> None:
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_NORMALIZATIONS}, got "
                f"{self.normalization!r}"
            )
        if self.positional_encoding not in _POSITIONAL_ENCODINGS:
            raise ValueError(
                f"positional_encoding must be one of {_POSITIONAL_ENCODINGS}, "
                f"got {self.positional_encoding!r}"
            )
        if self.fusion not in _FUSIONS:
            raise ValueError(
                f"fusion must be one of {_FUSIONS}, got {self.fusion!r}"
            )
        if self.global_branch_type not in _GLOBAL_BRANCH_TYPES:
            raise ValueError(
                f"global_branch_type must be one of {_GLOBAL_BRANCH_TYPES}, "
                f"got {self.global_branch_type!r}"
            )
        if self.local_branch_type not in _LOCAL_BRANCH_TYPES:
            raise ValueError(
                f"local_branch_type must be one of {_LOCAL_BRANCH_TYPES}, got "
                f"{self.local_branch_type!r}"
            )
        if len(self.unet_blocks_per_level) != len(self.unet_channel_mult):
            raise ValueError(
                "unet_blocks_per_level and unet_channel_mult must have the "
                "same length (one entry per resolution level)"
            )
        if not (self.use_local_branch or self.use_global_branch):
            raise ValueError("at least one of the two branches must be enabled")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")
        if self.refine_kernel_size % 2 == 0:
            raise ValueError("refine_kernel_size must be odd to preserve length")
        if self.local_channels % self.groups != 0:
            raise ValueError(
                f"local_channels ({self.local_channels}) must be divisible by "
                f"groups ({self.groups})"
            )
        if self.global_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"global_dim ({self.global_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.output_length is not None:
            if self.output_length > self.input_length:
                raise ValueError("output_length must not exceed input_length")
            if (self.input_length - self.output_length) % 2 != 0:
                raise ValueError(
                    "input_length - output_length must be even so the target "
                    "region is exactly centred"
                )
        if len(self.class_order) != self.num_classes:
            raise ValueError("class_order length must equal num_classes")

    @property
    def local_receptive_field(self) -> int:
        """Receptive field of the local CNN branch, in nucleotides."""
        return local_receptive_field(self.kernel_size, self.dilations)


def local_receptive_field(kernel_size: int, dilations: tuple[int, ...]) -> int:
    """Receptive field of a stack of one dilated conv per residual block.

    Args:
        kernel_size (int): Spatial kernel size, shared by every block.
        dilations (tuple[int, ...]): Dilation of each block.

    Returns:
        int: Receptive field in input positions.
    """
    return 1 + (kernel_size - 1) * sum(dilations)


def _make_norm(kind: str, channels: int, num_groups: int) -> nn.Module:
    """Build a normalization layer for ``[B, C, L]`` tensors."""
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "group":
        return nn.GroupNorm(min(num_groups, channels), channels)
    # "layer": one group -> normalize jointly over (C, L), the conv analogue of
    # LayerNorm. Sequence-length dependent, unlike GroupNorm/BatchNorm.
    return nn.GroupNorm(1, channels)


def _factor_downsample(factor: int) -> tuple[int, int]:
    """Split a total downsampling factor into two roughly balanced strides."""
    if factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {factor}")
    best = 1
    for stride in range(1, math.isqrt(factor) + 1):
        if factor % stride == 0:
            best = stride
    return best, factor // best


class ResidualDilatedConvBlock(nn.Module):
    """Pre-activation residual block with a single dilated spatial conv.

    Layout (one spatial conv per block, as specified)::

        x -> Norm -> ReLU -> dilated Conv1d -> Norm -> ReLU -> 1x1 Conv1d
          -> Dropout -> (+ x)

    The 1x1 conv carries channel mixing, which is what makes ``groups > 1``
    (and especially depthwise, ``groups == channels``) viable in the spatial
    conv.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        groups: int,
        dropout: float,
        normalization: str,
        norm_num_groups: int,
    ) -> None:
        super().__init__()
        self.norm1 = _make_norm(normalization, channels, norm_num_groups)
        self.act1 = nn.ReLU(inplace=True)
        # bias=False: norm2 immediately follows and absorbs any constant shift.
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.norm2 = _make_norm(normalization, channels, norm_num_groups)
        self.act2 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x (torch.Tensor): Input of shape ``[B, C, L]``.

        Returns:
            torch.Tensor: Output of shape ``[B, C, L]``.
        """
        h = self.act1(self.norm1(x))
        h = self.conv(h)
        h = self.act2(self.norm2(h))
        h = self.pointwise(h)
        return x + self.dropout(h)


class LocalCNNBranch(nn.Module):
    """Stack of dilated residual blocks at full nucleotide resolution."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedConvBlock(
                    channels=config.local_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    groups=config.groups,
                    dropout=config.dropout,
                    normalization=config.normalization,
                    norm_num_groups=config.norm_num_groups,
                )
                for dilation in config.dilations
            ]
        )
        # A pre-activation stack needs a trailing norm+activation before readout.
        self.out_norm = _make_norm(
            config.normalization, config.local_channels, config.norm_num_groups
        )
        self.out_act = nn.ReLU(inplace=True)
        self.receptive_field = local_receptive_field(
            config.kernel_size, config.dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dilated CNN stack.

        Args:
            x (torch.Tensor): Stem features of shape ``[B, C, L]``.

        Returns:
            torch.Tensor: Local features of shape ``[B, C, L]``.
        """
        for block in self.blocks:
            x = block(x)
        return self.out_act(self.out_norm(x))


def unet_level_dilations(num_blocks: int) -> tuple[int, ...]:
    """Dilation schedule inside one U-Net level: ``(1, 1, 2, 4, 8, ...)``."""
    if num_blocks <= 0:
        return ()
    return (1,) + tuple(2 ** i for i in range(num_blocks - 1))


def unet_receptive_field(
    kernel_size: int, blocks_per_level: tuple[int, ...], stride: int
) -> int:
    """Receptive field of the U-Net encoder, in input nucleotides.

    Each level contributes its own field measured in that level's positions,
    scaled by how much the input has been decimated by the time it is reached.

    Args:
        kernel_size (int): Spatial kernel, shared by every block.
        blocks_per_level (tuple[int, ...]): Blocks at each level, finest first.
        stride (int): Decimation between consecutive levels.

    Returns:
        int: Receptive field in input positions.
    """
    field = 1
    scale = 1
    for level, num_blocks in enumerate(blocks_per_level):
        if level:
            scale *= stride
        span = (kernel_size - 1) * sum(unet_level_dilations(num_blocks))
        field += span * scale
    return field


class UNetLocalBranch(nn.Module):
    """Multi-resolution encoder-decoder alternative to the dilated stack.

    Splice motifs need nucleotide resolution; exon/intron architecture does
    not. Running every dilated block at the full 15,000 positions pays for
    nucleotide precision on context that cannot use it. Here a few blocks run
    at full resolution, then the sequence is decimated twice, so the deep
    blocks cover far more sequence per unit of compute. Skip connections carry
    the nucleotide-precise features back to the output, which stays at
    ``local_channels`` so the fusion and head are unchanged.

    This is the CPU-servable analogue of SplicePath's learned dynamic chunker:
    fixed-stride pooling instead of a learned boundary predictor.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stride = config.unet_stride
        channels = [config.local_channels * m for m in config.unet_channel_mult]
        self.channels = channels
        blocks = config.unet_blocks_per_level

        def _stack(width: int, num_blocks: int) -> nn.ModuleList:
            return nn.ModuleList(
                ResidualDilatedConvBlock(
                    channels=width,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    groups=config.groups if width % config.groups == 0 else 1,
                    dropout=config.dropout,
                    normalization=config.normalization,
                    norm_num_groups=config.norm_num_groups,
                )
                for dilation in unet_level_dilations(num_blocks)
            )

        self.encoders = nn.ModuleList(
            [_stack(channels[i], blocks[i]) for i in range(len(channels))]
        )
        kernel = 2 * self.stride - 1
        self.downs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i], channels[i + 1], kernel_size=kernel,
                    stride=self.stride, padding=(kernel - 1) // 2,
                )
                for i in range(len(channels) - 1)
            ]
        )
        # Decoder: project the coarser level onto the finer one, add the skip,
        # then one block to blend them. Upsampling is linear interpolation to
        # the encoder's recorded length, so lengths are restored exactly rather
        # than inferred from a stride formula.
        self.ups = nn.ModuleList(
            [
                nn.Conv1d(channels[i + 1], channels[i], kernel_size=1)
                for i in range(len(channels) - 1)
            ]
        )
        self.decoders = nn.ModuleList(
            [_stack(channels[i], 1) for i in range(len(channels) - 1)]
        )
        self.out_norm = _make_norm(
            config.normalization, channels[0], config.norm_num_groups
        )
        self.out_act = nn.ReLU(inplace=True)
        self.receptive_field = unet_receptive_field(
            config.kernel_size, blocks, self.stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode down the resolution ladder, then decode back up.

        Args:
            x (torch.Tensor): Stem features of shape ``[B, C, L]``.

        Returns:
            torch.Tensor: Local features of shape ``[B, C, L]``.
        """
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoders):
            for block in blocks:
                x = block(x)
            if level < len(self.downs):
                skips.append(x)
                x = F.relu(self.downs[level](x), inplace=True)

        for level in reversed(range(len(self.ups))):
            skip = skips[level]
            x = F.interpolate(
                x, size=skip.shape[-1], mode="linear", align_corners=False
            )
            x = self.ups[level](x) + skip
            for block in self.decoders[level]:
                x = block(x)

        return self.out_act(self.out_norm(x))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal absolute positional encoding.

    Caveat for this project: absolute position is measured from the start of an
    arbitrary 15 kb window. Under anchored multi-chunk scoring (K=4) the same
    variant sits at a different absolute offset in every chunk, so an absolutely
    position-aware model can give inconsistent per-chunk scores. Prefer
    ``positional_encoding='relative'`` unless an ablation says otherwise.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the encoding to ``[B, L, D]`` tokens."""
        return x + self.encoding[: x.shape[1]].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional encoding. Same caveat as the sinusoidal one."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the encoding to ``[B, L, D]`` tokens."""
        return x + self.encoding[: x.shape[1]].unsqueeze(0)


class RelativePositionBias(nn.Module):
    """Clipped relative-position bias added to the attention logits.

    Translation-equivariant, so a variant scored at different offsets within a
    chunk gets consistent coarse context.
    """

    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.embedding = nn.Embedding(2 * max_distance + 1, num_heads)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        """Build the bias.

        Args:
            length (int): Number of coarse tokens.
            device (torch.device): Device for the index tensor.

        Returns:
            torch.Tensor: Bias of shape ``[1, num_heads, length, length]``,
            broadcastable over the batch.
        """
        positions = torch.arange(length, device=device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        index = relative.clamp(-self.max_distance, self.max_distance)
        index = index + self.max_distance
        bias = self.embedding(index)  # [L, L, H]
        return bias.permute(2, 0, 1).unsqueeze(0)


class _SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with an optional additive bias.

    Written out rather than using ``nn.TransformerEncoder`` so that (a) the
    nested-tensor fast path cannot interfere with tracing/export and (b) a
    broadcastable ``[1, H, L, L]`` relative bias can be used without expanding
    it to ``[B * H, L, L]``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Attend over the token axis.

        Args:
            x (torch.Tensor): Tokens of shape ``[B, L, D]``.
            bias (Optional[torch.Tensor]): Additive attention bias broadcastable
                to ``[B, H, L, L]``, or ``None``.

        Returns:
            torch.Tensor: Output of shape ``[B, L, D]``.
        """
        batch, length, d_model = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, d_model)
        return self.proj(attended)


class _EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (no causal mask -- both sides used)."""

    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply attention and feed-forward sublayers."""
        x = x + self.dropout(self.attention(self.norm1(x), bias))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GlobalTransformerBranch(nn.Module):
    """Strided-conv downsampling, a small bidirectional Transformer, upsampling.

    Two cheap efficiency choices worth knowing about:

    * The ``global_dim -> local_channels`` 1x1 projection is applied BEFORE the
      upsampling. Linear interpolation and a 1x1 conv are both linear maps on
      disjoint axes, so they commute exactly (the bias survives because
      interpolation weights sum to one) -- this is the same result for ~1/16th
      of the projection cost.
    * The input is right-padded to a multiple of ``downsample_factor`` so each
      coarse token corresponds to exactly ``downsample_factor`` input
      positions; the output is cropped back. Linear interpolation still places
      token centres half a cell off the conv's true centres, which is
      immaterial for coarse context but matters if you ever read positions
      directly off this branch.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        stride1, stride2 = _factor_downsample(config.downsample_factor)
        self.downsample_factor = config.downsample_factor

        kernel1 = 4 * stride1 - 1 if stride1 > 1 else 1
        kernel2 = stride2 + 1 if stride2 > 1 else 1
        self.downsample = nn.Sequential(
            nn.Conv1d(
                config.local_channels,
                config.global_dim,
                kernel_size=kernel1,
                stride=stride1,
                padding=(kernel1 - 1) // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                config.global_dim,
                config.global_dim,
                kernel_size=kernel2,
                stride=stride2,
                padding=(kernel2 - 1) // 2,
            ),
            nn.ReLU(inplace=True),
        )

        max_tokens = math.ceil(config.input_length / config.downsample_factor) + 1
        encoding = config.positional_encoding
        self.positional_encoding: Optional[nn.Module] = None
        self.relative_bias: Optional[RelativePositionBias] = None
        if encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                config.global_dim, max_tokens
            )
        elif encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(
                config.global_dim, max_tokens
            )
        elif encoding == "relative":
            self.relative_bias = RelativePositionBias(
                config.num_attention_heads, config.relative_max_distance
            )

        self.layers = nn.ModuleList(
            [
                _EncoderLayer(
                    config.global_dim,
                    config.num_attention_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.num_transformer_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.global_dim)
        self.project = nn.Conv1d(config.global_dim, config.local_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce coarse context at full resolution.

        Args:
            x (torch.Tensor): Stem features of shape ``[B, C, L]``.

        Returns:
            torch.Tensor: Context features of shape ``[B, C, L]``.
        """
        length = x.shape[-1]
        padded = math.ceil(length / self.downsample_factor) * self.downsample_factor
        if padded != length:
            x = F.pad(x, (0, padded - length))

        tokens = self.downsample(x).transpose(1, 2)  # [B, L_g, D]
        if self.positional_encoding is not None:
            tokens = self.positional_encoding(tokens)
        bias = (
            self.relative_bias(tokens.shape[1], tokens.device)
            if self.relative_bias is not None
            else None
        )
        for layer in self.layers:
            tokens = layer(tokens, bias)
        tokens = self.final_norm(tokens)

        coarse = self.project(tokens.transpose(1, 2))  # [B, C, L_g]
        upsampled = F.interpolate(
            coarse, size=padded, mode="linear", align_corners=False
        )
        return upsampled[..., :length]


class GlobalSEBranch(nn.Module):
    """Squeeze-and-excitation style global conditioning (ablation arm B).

    Collapses the whole sequence to one vector and broadcasts it back, i.e. the
    cheapest possible form of "global context". Useful as the floor that the
    Transformer branch has to beat.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.local_channels, config.global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.global_dim, config.local_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over length, transform, broadcast back to ``[B, C, L]``."""
        pooled = self.mlp(x.mean(dim=-1))
        return pooled.unsqueeze(-1).expand(-1, -1, x.shape[-1])


class GatedFusion(nn.Module):
    """Combine local and global features, then refine.

    ``fusion='gated'`` (default) lets the global branch modulate the local
    prediction::

        gate  = sigmoid(conv1x1([local ; global]))
        fused = local + gate * global

    Alternatives kept for the fusion ablation: ``concat`` (1x1 conv over the
    concatenation), ``film`` (feature-wise affine modulation driven by the
    global branch) and ``add`` (no learned mixing at all).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = config.local_channels
        self.mode = config.fusion
        self.gate_conv: Optional[nn.Conv1d] = None
        self.merge: Optional[nn.Conv1d] = None
        self.film: Optional[nn.Conv1d] = None

        if self.mode == "gated":
            self.gate_conv = nn.Conv1d(2 * channels, channels, kernel_size=1)
            nn.init.constant_(self.gate_conv.bias, config.gate_bias_init)
        elif self.mode == "concat":
            self.merge = nn.Conv1d(2 * channels, channels, kernel_size=1)
        elif self.mode == "film":
            self.film = nn.Conv1d(channels, 2 * channels, kernel_size=1)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        self.refine = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.refine_kernel_size,
            padding=config.refine_kernel_size // 2,
        )
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)

    @property
    def context_margin(self) -> int:
        """Extra positions each side needs for the refine conv to be exact."""
        return self.refine.kernel_size[0] // 2

    def forward(
        self, local: torch.Tensor, global_feature: torch.Tensor
    ) -> torch.Tensor:
        """Fuse the two branches.

        Args:
            local (torch.Tensor): Local features ``[B, C, L]``.
            global_feature (torch.Tensor): Context features ``[B, C, L]``.

        Returns:
            torch.Tensor: Fused features ``[B, C, L]``.
        """
        if self.mode == "gated":
            assert self.gate_conv is not None
            gate = torch.sigmoid(
                self.gate_conv(torch.cat([local, global_feature], dim=1))
            )
            fused = local + gate * global_feature
        elif self.mode == "concat":
            assert self.merge is not None
            fused = self.merge(torch.cat([local, global_feature], dim=1))
        elif self.mode == "film":
            assert self.film is not None
            scale, shift = self.film(global_feature).chunk(2, dim=1)
            fused = local * (1.0 + scale) + shift
        else:  # "add"
            fused = local + global_feature

        return fused + self.dropout(self.act(self.refine(fused)))

    def gate_statistics(
        self, local: torch.Tensor, global_feature: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Diagnostics for whether the global branch is actually being used.

        A gate that collapses towards zero makes the ``CNN + Transformer`` arms
        indistinguishable from ``CNN only`` for optimisation reasons rather than
        modelling ones -- log these during training to tell the two apart.

        Args:
            local (torch.Tensor): Local features ``[B, C, L]``.
            global_feature (torch.Tensor): Context features ``[B, C, L]``.

        Returns:
            dict[str, torch.Tensor]: ``gate_mean``, ``gate_std`` and
            ``contribution_ratio`` (norm of the injected global signal relative
            to the local signal). Empty for fusion modes without a gate.
        """
        if self.mode != "gated":
            return {}
        assert self.gate_conv is not None
        gate = torch.sigmoid(
            self.gate_conv(torch.cat([local, global_feature], dim=1))
        )
        return {
            "gate_mean": gate.mean().detach(),
            "gate_std": gate.std().detach(),
            "contribution_ratio": (
                (gate * global_feature).norm() / local.norm().clamp_min(1e-6)
            ).detach(),
        }


class HybridSpliceModel(nn.Module):
    """Dilated CNN + downsampled Transformer splice-site annotator.

    Args:
        config (ModelConfig): Model configuration. Defaults to the recommended
            reference configuration.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.stem = nn.Conv1d(cfg.in_channels, cfg.local_channels, kernel_size=1)

        self.local_branch: Optional[nn.Module] = None
        if cfg.use_local_branch:
            self.local_branch = (
                UNetLocalBranch(cfg) if cfg.local_branch_type == "unet"
                else LocalCNNBranch(cfg)
            )
        self.global_branch: Optional[nn.Module] = None
        if cfg.use_global_branch:
            self.global_branch = (
                GlobalTransformerBranch(cfg)
                if cfg.global_branch_type == "transformer"
                else GlobalSEBranch(cfg)
            )

        self.fusion = (
            GatedFusion(cfg)
            if (cfg.use_local_branch and cfg.use_global_branch)
            else None
        )
        self.head = nn.Conv1d(cfg.local_channels, cfg.num_classes, kernel_size=1)
        self.aux_head = (
            nn.Conv1d(cfg.local_channels, 2, kernel_size=1)
            if cfg.aux_region_head
            else None
        )

    # -- introspection ------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters.

        Args:
            trainable_only (bool): Count only parameters with ``requires_grad``.

        Returns:
            int: Number of parameters.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def receptive_field(self) -> dict[str, int]:
        """Receptive field of each branch, in input nucleotides.

        Returns:
            dict[str, int]: ``local`` (dilated CNN stack) and ``global``
            (attention is all-to-all over coarse tokens, so the global branch
            sees the entire input window).
        """
        cfg = self.config
        local = 0
        if cfg.use_local_branch:
            local = (
                self.local_branch.receptive_field
                if self.local_branch is not None
                else cfg.local_receptive_field
            )
        return {
            "local": local,
            "global": cfg.input_length if cfg.use_global_branch else 0,
        }

    def summary(self) -> str:
        """One-paragraph description of the instantiated model."""
        cfg = self.config
        fields = self.receptive_field()
        branches = []
        if cfg.use_local_branch:
            if cfg.local_branch_type == "unet":
                branches.append(
                    f"local(UNet {cfg.unet_blocks_per_level} lvl, "
                    f"C={cfg.local_channels}x{cfg.unet_channel_mult}, "
                    f"stride={cfg.unet_stride}, k={cfg.kernel_size}, "
                    f"RF={fields['local']})"
                )
            else:
                branches.append(
                    f"local(CNN x{len(cfg.dilations)}, C={cfg.local_channels}, "
                    f"k={cfg.kernel_size}, groups={cfg.groups}, "
                    f"RF={fields['local']})"
                )
        if cfg.use_global_branch:
            if cfg.global_branch_type == "transformer":
                branches.append(
                    f"global(down x{cfg.downsample_factor}, "
                    f"d={cfg.global_dim}, L={cfg.num_transformer_layers}, "
                    f"heads={cfg.num_attention_heads}, "
                    f"pe={cfg.positional_encoding})"
                )
            else:
                branches.append("global(SE pooling)")
        fusion = cfg.fusion if self.fusion is not None else "n/a (single branch)"
        return (
            f"HybridSpliceModel: {' + '.join(branches)} -> fusion={fusion} -> "
            f"{cfg.num_classes}-class head {cfg.class_order}\n"
            f"  in {cfg.in_channels}x{cfg.input_length} -> out "
            f"{cfg.num_classes}x{cfg.output_length or cfg.input_length}\n"
            f"  parameters: {self.num_parameters():,}"
        )

    # -- forward ------------------------------------------------------------

    def _target_bounds(self) -> tuple[int, int]:
        """Half-open ``[start, end)`` of the supervised target region."""
        cfg = self.config
        if cfg.output_length is None or cfg.output_length == cfg.input_length:
            return 0, cfg.input_length
        start = (cfg.input_length - cfg.output_length) // 2
        return start, start + cfg.output_length

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict per-position splice-site logits.

        Args:
            x (torch.Tensor): One-hot input of shape
                ``[B, in_channels, input_length]``.

        Returns:
            torch.Tensor: Raw logits (no softmax) of shape
            ``[B, num_classes, output_length]``. If ``aux_region_head`` is
            enabled, returns ``(logits, region_logits)`` where the second
            tensor holds binary intron/exon logits of shape ``[B, 2, L_out]``.
        """
        cfg = self.config
        if x.dim() != 3:
            raise ValueError(f"expected a 3-D tensor [B, C, L], got {tuple(x.shape)}")
        if x.shape[1] != cfg.in_channels:
            raise ValueError(
                f"expected {cfg.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[2] != cfg.input_length:
            raise ValueError(
                f"expected input length {cfg.input_length}, got {x.shape[2]}"
            )

        features = self.stem(x)

        if self.fusion is not None:
            assert self.local_branch is not None and self.global_branch is not None
            local = self.local_branch(features)
            context = self.global_branch(features)
            # Crop before fusion: the fusion + head only ever contribute to the
            # target region, so paying for all 15,000 positions is waste. The
            # refine conv is the only length-coupled op, so keeping a margin of
            # its half-width makes this exactly equal to fusing at full length
            # and cropping afterwards.
            start, end = self._target_bounds()
            margin = self.fusion.context_margin
            lo = max(0, start - margin)
            hi = min(cfg.input_length, end + margin)
            fused = self.fusion(local[..., lo:hi], context[..., lo:hi])
            fused = fused[..., start - lo : (start - lo) + (end - start)]
        else:
            branch = self.local_branch or self.global_branch
            assert branch is not None
            start, end = self._target_bounds()
            fused = branch(features)[..., start:end]

        logits = self.head(fused)
        if self.aux_head is not None:
            return logits, self.aux_head(fused)
        return logits


#: Configuration overrides for the planned ablation ladder.
#:
#: ``A``-``E`` are the arms named in the design spec; ``F`` adds the exon/intron
#: auxiliary head that defines SplicePath in this project (free at inference).
ABLATION_PRESETS: dict[str, dict] = {
    "A_cnn_only": {"use_global_branch": False},
    "B_cnn_se": {"global_branch_type": "se", "fusion": "gated"},
    "C_cnn_transformer_concat": {"fusion": "concat"},
    "D_cnn_transformer_gated": {},
    "E_transformer_only": {"use_local_branch": False},
    "F_gated_plus_region_head": {"aux_region_head": True},
    # Arms below are one-knob twins of F, added after F's ClinVar strata showed
    # the gap to the large backbones widening with distance from the splice
    # site (deep-intronic -0.056 AUROC, proximal/mid -0.052) while F WON at the
    # site itself. All three probe long-range context.
    "G_no_global_region_head": {"aux_region_head": True, "use_global_branch": False},
    "H_relative_pe_region_head": {
        "aux_region_head": True, "positional_encoding": "relative",
    },
    "I_unet_region_head": {
        "aux_region_head": True, "local_branch_type": "unet",
        "use_global_branch": False,
    },
    # G/H/I all lost to F: the global branch is working (G), relative position
    # bias hurts (H), and decimating the local branch destroys the deep-intronic
    # signal even at a matched receptive field (I). So the long-range gap is a
    # capacity problem, not an architecture one -- and capacity is far cheaper in
    # the global branch, which holds 63% of the parameters but only 15% of the
    # FLOPs. J spends it there and nowhere else.
    "J_capacity_region_head": {
        "aux_region_head": True,
        "global_dim": 192,
        "ffn_dim": 384,
        "num_transformer_layers": 4,
        "num_attention_heads": 8,
        "downsample_factor": 8,
    },
}


def build_model(
    preset: str = "D_cnn_transformer_gated", **overrides
) -> HybridSpliceModel:
    """Instantiate a model from an ablation preset plus keyword overrides.

    Args:
        preset (str): Key into ``ABLATION_PRESETS``.
        **overrides: Extra :class:`ModelConfig` fields, applied after the preset.

    Returns:
        HybridSpliceModel: The configured model.
    """
    if preset not in ABLATION_PRESETS:
        raise KeyError(
            f"unknown preset {preset!r}; choose from {sorted(ABLATION_PRESETS)}"
        )
    settings = dict(ABLATION_PRESETS[preset])
    settings.update(overrides)
    return HybridSpliceModel(ModelConfig(**settings))
