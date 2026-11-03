"""SpliceAI v1 PyTorch implementation matching ``spliceai-original-h5-v1``.

Vendored from the January 2026 training-pipeline snapshot at
``./checkpoints/spliceai-original-h5-v1/source_snapshot_20260118_010110.tar.gz``
(``model/spliceai.py`` therein). The class is renamed to :class:`SpliceAIV1`
to avoid clashing with :class:`model.spliceai.SpliceAI` (the factory class in
the current repo). State-dict keys (``input_conv``, ``residual_blocks.*``,
``output_conv``) match the v1 checkpoints byte-for-byte.

Used by ``evaluate.evaluate_splicevardb_sav`` under ``--backend spliceai-v1``.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SpliceAIV1ResidualBlock(nn.Module):
    """Residual block with dilated convolutions (BN -> ReLU -> Conv twice).

    Args:
        channels (int): Number of input/output channels.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation rate for both convolutions.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 11,
        dilation: int = 1,
    ) -> None:
        """Initialize residual block.

        Args:
            channels (int): Number of input/output channels.
            kernel_size (int): Convolution kernel size.
            dilation (int): Dilation rate.
        """
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x (torch.Tensor): Input of shape (B, C, L).

        Returns:
            torch.Tensor: Output of shape (B, C, L).
        """
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class SpliceAIV1(nn.Module):
    """SpliceAI v1 model with configurable context length and channels.

    Token-ID input convention (matches v1 training pipeline): ``0=A, 1=C,
    2=G, 3=T``. Values outside ``[0, 3]`` are clamped to ``[0, 3]`` (so an
    H5 ``N`` token must be remapped externally; the eval script handles this).

    Attributes:
        seq_length (int): Input sequence length (e.g., 15000).
        n_channels (int): Hidden conv channels (default: 32).
        n_classes (int): Output classes (default: 3 — acceptor, donor, neither).
        kernel_size (int): Conv kernel size (default: 11).
        n_blocks (int): Residual blocks per dilation rate (default: 4).
        context_length (int): Context length preset (e.g., 10000).
        dilation_rates (list[int]): Dilation rates derived from context length.
        input_conv (nn.Conv1d): 4 -> n_channels projection.
        residual_blocks (nn.ModuleList): Stacked residual blocks.
        output_conv (nn.Conv1d): n_channels -> n_classes projection.
    """

    DILATION_RATES_10K = [1, 4, 10, 25, 57, 133, 310, 723, 1685, 3927, 9157]
    DILATION_RATES_2K = [1, 4, 10, 25, 57, 133, 310]
    DILATION_RATES_400 = [1, 4, 10, 25, 57]
    DILATION_RATES_80 = [1, 4, 10]

    def __init__(self, cfg: DictConfig | dict) -> None:
        """Initialize SpliceAIV1 from a config.

        Args:
            cfg (DictConfig | dict): Must include ``model.seq_length`` and
                ``model.n_classification_classes``. Optional fields with
                defaults: ``spliceai_channels`` (32), ``spliceai_kernel_size``
                (11), ``spliceai_blocks_per_dilation`` (4),
                ``spliceai_context_length`` (10000).
        """
        super().__init__()

        if isinstance(cfg, dict):
            cfg = DictConfig(cfg)
        self.cfg = cfg

        self.seq_length = cfg.model.seq_length
        self.n_channels = getattr(cfg.model, "spliceai_channels", 32)
        self.n_classes = cfg.model.n_classification_classes
        self.kernel_size = getattr(cfg.model, "spliceai_kernel_size", 11)
        self.n_blocks = getattr(cfg.model, "spliceai_blocks_per_dilation", 4)
        self.context_length = getattr(cfg.model, "spliceai_context_length", 10000)

        self.dilation_rates = self._get_dilation_rates()
        self._build_model()

    def _get_dilation_rates(self) -> list[int]:
        """Select dilation rates based on context length.

        Returns:
            list[int]: Dilation rates for residual stack.
        """
        if self.context_length <= 80:
            return self.DILATION_RATES_80
        if self.context_length <= 400:
            return self.DILATION_RATES_400
        if self.context_length <= 2000:
            return self.DILATION_RATES_2K
        return self.DILATION_RATES_10K

    def _build_model(self) -> None:
        """Build input conv, residual stack, and output conv."""
        self.input_conv = nn.Conv1d(4, self.n_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList()
        for dilation in self.dilation_rates:
            for _ in range(self.n_blocks):
                self.residual_blocks.append(
                    SpliceAIV1ResidualBlock(
                        channels=self.n_channels,
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                    )
                )

        self.output_conv = nn.Conv1d(
            self.n_channels, self.n_classes, kernel_size=1
        )

    def _one_hot_encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One-hot encode token IDs (0=A, 1=C, 2=G, 3=T) to (B, 4, L).

        Args:
            input_ids (torch.Tensor): Token IDs of shape (B, L), values in
                ``[0, 3]``. Values outside this range are clamped.

        Returns:
            torch.Tensor: One-hot tensor of shape (B, 4, L) in float32.
        """
        batch_size, seq_len = input_ids.shape
        ids_clamped = input_ids.clamp(0, 3)
        one_hot = torch.zeros(batch_size, 4, seq_len, device=input_ids.device)
        one_hot.scatter_(1, ids_clamped.unsqueeze(1), 1.0)
        return one_hot

    def forward(
        self,
        input_ids: torch.Tensor,
        strand: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (B, L).
            strand (torch.Tensor | None): Unused in SpliceAI.
            attention_mask (torch.Tensor | None): Unused in SpliceAI.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - Classification logits of shape (B, L, n_classes).
                - ``None`` placeholder (no regression head).
        """
        x = self._one_hot_encode(input_ids)
        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        logits = self.output_conv(x)
        logits = logits.transpose(1, 2)
        return logits, None
