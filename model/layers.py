"""
Basic layer implementations for the HyenaDNA model
"""

from typing import Literal

import torch
import torch.nn as nn


class Conv1DResidualBlock(nn.Module):
    """Dilated 1D convolution block with residual connection.

    Uses a pre-norm layout (GN -> ReLU -> Conv -> GN -> ReLU
    -> Dropout -> Conv) under ``conv_path``. The shortcut is a
    1x1 Conv when channel count changes, otherwise Identity.
    Layer naming and index layout match the legacy checkpoint
    format so checkpoints can be loaded without key remapping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Conv1DResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            dilation (int): Dilation factor for the convolution.
            dropout (float): Dropout rate.
        """
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        # Index layout must stay stable for checkpoint compatibility:
        # [0] GN(in)  [1] ReLU  [2] Conv(in->out, dilated)
        # [3] GN(out) [4] ReLU  [5] Dropout
        # [6] Conv(out->out)
        self.conv_path = nn.Sequential(
            nn.GroupNorm(
                min(32, in_channels), in_channels
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.GroupNorm(
                min(32, out_channels), out_channels
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch, channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch, out_channels, seq_len).
        """
        return self.conv_path(x) + self.shortcut(x)


class DeepClassificationHead(nn.Module):
    """Deep classification head with optional residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        activation: str = "gelu",
    ):
        """
        Deep classification head with optional residual connections.

        Args:
            input_dim (int): Input dimension.
            hidden_dims (list[int]): List of hidden dimensions.
            output_dim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            use_residual (bool): Whether to use residual connections.
            activation (str): Activation function.
        """
        super().__init__()
        self.use_residual = use_residual

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Residual projections if dimensions don't match
        if use_residual:
            self.residual_projections = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.residual_projections.append(nn.Linear(prev_dim, hidden_dim))
                else:
                    self.residual_projections.append(nn.Identity())
                prev_dim = hidden_dim

    def _get_activation(
        self, activation: Literal["relu", "gelu", "silu", "mish"]
    ) -> nn.Module:
        """
        Get the activation function.

        Args:
            activation (str): Activation function.

        Returns:
            nn.Module: Activation function.
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }
        return activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the deep classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        layer_idx = 0

        for i in range(0, len(self.layers), 4):  # 4 components per block
            # Apply block
            for j in range(4):
                x = self.layers[i + j](x)

            # Add residual if enabled
            if self.use_residual and layer_idx < len(self.residual_projections):
                residual = self.residual_projections[layer_idx](residual)
                x = x + residual
                residual = x
                layer_idx += 1

        return self.output_layer(x)


class AttentionClassificationHead(nn.Module):
    """Self-attention based classification head for splice site prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
        window_size: int = 15,
    ):
        """
        Self-attention based classification head for splice site prediction.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            num_heads (int): Number of heads.
            window_size (int): Window size.
        """
        super().__init__()
        self.window_size = window_size

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # Position encoding for local context
        self.pos_encoding = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, input_dim) * 0.02
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        _, seq_len, _ = x.shape

        # Apply local self-attention with sliding window
        output = torch.zeros_like(x)

        for i in range(seq_len):
            # Define window boundaries
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            window_actual_size = end - start

            # Extract window
            window = x[:, start:end, :]

            # Add position encoding (truncate if needed)
            pos_enc = self.pos_encoding[:, :window_actual_size, :]
            window = window + pos_enc

            # Apply attention
            attn_out, _ = self.attention(window, window, window)

            # Get the center position (or closest to center)
            center_idx = min(i - start, window_actual_size - 1)
            output[:, i, :] = attn_out[:, center_idx, :]

        # Apply FFN for classification
        return self.ffn(output)


class ConvolutionalHead(nn.Module):
    """Multi-scale 1D CNN head (Inception-like) for genomic patterns."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_sizes: list[int] = [3, 5, 7],
        dropout_rate: float = 0.1,
    ):
        """
        Multi-scale 1D CNN head for genomic patterns.
        Captures both short motifs (e.g., splice sites) and wider context
        simultaneously.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            kernel_sizes (list[int]): List of kernel sizes.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        num_kernels = len(kernel_sizes)

        base_channels = hidden_dim // num_kernels

        for i, k in enumerate(kernel_sizes):
            out_channels = (
                base_channels
                if i < num_kernels - 1
                else (hidden_dim - base_channels * (num_kernels - 1))
            )

            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_dim,
                        out_channels,
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )

        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),  # Multi-scale Information Fusion
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        # Concatenate along channel dimension
        # (batch, hidden_dim, seq_len)
        x = torch.cat(conv_outputs, dim=1)

        # (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)

        # Final classification
        return self.combine(x)
