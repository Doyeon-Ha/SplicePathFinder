import json
import math
import re
from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    """Residual block wrapper.

    Args:
        fn (nn.Module): The function to apply before adding the residual.
    """

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.fn(x)


def ResidualBlock(
    in_channels: int, out_channels: int, kernel_size: int, dilation: int
) -> Residual:
    """Create a residual block for SpliceAI.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation rate.

    Returns:
        Residual: A residual block module.
    """
    return Residual(
        nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
        )
    )


class SpliceAI_80nt(nn.Module):
    """SpliceAI model with 80nt context."""

    S = 80
    TARGET_LEN = 5000

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding="same")
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            nn.Conv1d(32, 32, 1, dilation=1, padding="same"),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = detour + self.block1(x)
        x = self.conv_last(x)

        offset = self.S // 2
        return rearrange(
            x[..., offset : self.TARGET_LEN + offset],
            "b c l -> b l c",
        )


class SpliceAI_400nt(nn.Module):
    """SpliceAI model with 400nt context."""

    S = 400
    TARGET_LEN = 5000

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding="same")
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            nn.Conv1d(32, 32, 1, dilation=1, padding="same"),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x) + detour
        x = self.conv_last(x)

        offset = self.S // 2
        return rearrange(
            x[..., offset : self.TARGET_LEN + offset],
            "b c l -> b l c",
        )


class SpliceAI_2k(nn.Module):
    """SpliceAI model with 2k context."""

    S = 2000
    TARGET_LEN = 5000

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding="same")
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
        )

        self.res_conv3 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            nn.Conv1d(32, 32, 1, dilation=1, padding="same"),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x)
        detour += self.res_conv3(x)

        x = self.block3(x) + detour
        x = self.conv_last(x)

        offset = self.S // 2
        return rearrange(
            x[..., offset : self.TARGET_LEN + offset],
            "b c l -> b l c",
        )


class SpliceAI_10k(nn.Module):
    """SpliceAI model with 10k context."""

    S = 10000
    TARGET_LEN = 5000

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding="same")
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
        )

        self.res_conv3 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
        )

        self.res_conv4 = nn.Conv1d(32, 32, 1, dilation=1, padding="same")

        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x)
        detour += self.res_conv3(x)

        x = self.block3(x)
        detour += self.res_conv4(x)

        x = self.block4(x) + detour
        x = self.conv_last(x)

        offset = self.S // 2
        return rearrange(
            x[..., offset : self.TARGET_LEN + offset],
            "b c l -> b l c",
        )


SPLICEAI_10K_KERNEL_SIZES = (11,) * 8 + (21,) * 4 + (41,) * 4
"""``W`` of the released 10k model (``train_model.py``)."""

SPLICEAI_10K_DILATIONS = (1,) * 4 + (4,) * 4 + (10,) * 4 + (25,) * 4
"""``AR`` of the released 10k model (``train_model.py``)."""


class _IlluminaResidualUnit(nn.Module):
    """Pre-activation residual unit of the released ``spliceai.py``.

    BatchNorm -> ReLU -> dilated conv -> BatchNorm -> ReLU -> dilated conv,
    added to the input.

    Args:
        channels (int): Number of convolution kernels (``L``).
        kernel_size (int): Convolution window (``W``).
        dilation (int): Atrous rate (``AR``).
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        eps, momentum = SpliceAIIllumina.BN_EPS, SpliceAIIllumina.BN_MOMENTUM
        self.bn1 = nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch, channels, length).

        Returns:
            torch.Tensor: Output of the same shape.
        """
        h = self.conv1(torch.relu(self.bn1(x)))
        h = self.conv2(torch.relu(self.bn2(h)))
        return x + h


class SpliceAIIllumina(nn.Module):
    """SpliceAI as ``SpliceAI(L, W, AR)`` builds it in Illumina's training code.

    Follows ``spliceai.py`` of the BaseSpace ``SpliceAI_train_code`` release
    (Jaganathan et al. 2019; Keras 2.0.5): a 1x1 input conv, a 1x1 skip conv on
    its output, residual units with a 1x1 skip conv after every fourth unit and
    after the last, cropping of ``CL / 2`` per side, and a 1x1 output conv.
    BatchNorm uses Keras' epsilon 1e-3 and momentum 0.99; biases start at zero
    and kernels glorot-uniform as Keras 2.0.5 computes it, whose
    ``_compute_fans`` takes ``shape[:2] = (k, in)`` of a ``(k, in, out)``
    Conv1D kernel as the receptive field: both fans carry an extra factor of
    ``in``, so the limit is ``sqrt(in)`` times smaller than PyTorch's xavier
    (5.7x for 32 channels). With xavier limits the first steps saturate the
    softmax and training stalls. Logits come in the released channel order
    (neither, acceptor, donor), so the released ``spliceai{1..5}.h5`` weights
    load with :meth:`load_keras_h5`. :class:`SpliceAI_10k` has no skip conv
    after its last group, so those weights do not fit it.

    Args:
        channels (int): Number of convolution kernels (``L``).
        kernel_sizes (Sequence[int]): Window of each residual unit (``W``).
        dilations (Sequence[int]): Atrous rate of each residual unit (``AR``).

    Raises:
        ValueError: If ``kernel_sizes`` and ``dilations`` differ in length.
    """

    BN_EPS = 1e-3
    BN_MOMENTUM = 0.01  # PyTorch's update weight; Keras momentum 0.99
    CHANNEL_ORDER = ("neither", "acceptor", "donor")

    def __init__(
        self,
        channels: int = 32,
        kernel_sizes: Sequence[int] = SPLICEAI_10K_KERNEL_SIZES,
        dilations: Sequence[int] = SPLICEAI_10K_DILATIONS,
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")
        n_units = len(kernel_sizes)
        self.context = 2 * sum(d * (k - 1) for k, d in zip(kernel_sizes, dilations))
        self.skip_after = tuple(
            i for i in range(n_units) if (i + 1) % 4 == 0 or (i + 1) == n_units
        )

        self.conv_in = nn.Conv1d(4, channels, 1)
        self.skip_in = nn.Conv1d(channels, channels, 1)
        self.units = nn.ModuleList(
            _IlluminaResidualUnit(channels, k, d)
            for k, d in zip(kernel_sizes, dilations)
        )
        self.skips = nn.ModuleList(
            nn.Conv1d(channels, channels, 1) for _ in self.skip_after
        )
        self.conv_out = nn.Conv1d(channels, 3, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                k, c_in, c_out = m.kernel_size[0], m.in_channels, m.out_channels
                limit = math.sqrt(6.0 / (k * c_in * (c_in + c_out)))
                nn.init.uniform_(m.weight, -limit, limit)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): One-hot input of shape (batch, 4, length) in
                A, C, G, T order; N is all zeros.

        Returns:
            torch.Tensor: Logits of shape (batch, length - context, 3).
        """
        x = self.conv_in(x)
        skip = self.skip_in(x)
        skips = iter(self.skips)
        for i, unit in enumerate(self.units):
            x = unit(x)
            if i in self.skip_after:
                skip = skip + next(skips)(x)
        half = self.context // 2
        skip = skip[..., half : skip.shape[-1] - half]
        return rearrange(self.conv_out(skip), "b c l -> b l c")

    def _layers_in_keras_order(self) -> tuple[list[nn.Conv1d], list[nn.BatchNorm1d]]:
        """Convs and BatchNorms in the order ``SpliceAI(L, W, AR)`` creates them."""
        convs: list[nn.Conv1d] = [self.conv_in, self.skip_in]
        skips = iter(self.skips)
        for i, unit in enumerate(self.units):
            convs += [unit.conv1, unit.conv2]
            if i in self.skip_after:
                convs.append(next(skips))
        convs.append(self.conv_out)
        bns = [bn for unit in self.units for bn in (unit.bn1, unit.bn2)]
        return convs, bns

    def load_keras_h5(self, path: str) -> None:
        """Load weights saved by the released Keras model (``spliceai{1..5}.h5``).

        Keras numbers auto-named layers per class in creation order, so the
        ``conv1d_*`` and ``batch_normalization_*`` layers sorted by that number
        line up with :meth:`_layers_in_keras_order`. Each layer's kernel size,
        dilation, BatchNorm epsilon and momentum are checked against the saved
        model config first.

        Args:
            path (str): Keras HDF5 model file.

        Raises:
            ValueError: If the file's layers do not match this model.
        """
        import h5py

        convs, bns = self._layers_in_keras_order()
        with h5py.File(path, "r") as f:
            config = f.attrs["model_config"]
            if isinstance(config, bytes):
                config = config.decode()
            config = json.loads(config)
            layer_config = {
                layer["name"]: layer["config"] for layer in config["config"]["layers"]
            }
            weights = f["model_weights"]
            names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in weights.attrs["layer_names"]
            ]

            def numbered(prefix: str) -> list[str]:
                found = [n for n in names if re.fullmatch(rf"{prefix}_\d+", n)]
                return sorted(found, key=lambda n: int(n.rsplit("_", 1)[1]))

            conv_names = numbered("conv1d")
            bn_names = numbered("batch_normalization")
            if len(conv_names) != len(convs) or len(bn_names) != len(bns):
                raise ValueError(
                    f"{path}: {len(conv_names)} convs and {len(bn_names)} BatchNorms, "
                    f"model has {len(convs)} and {len(bns)}"
                )

            def arrays(name: str) -> dict[str, torch.Tensor]:
                group = weights[name]
                out = {}
                for w in group.attrs["weight_names"]:
                    w = w.decode() if isinstance(w, bytes) else str(w)
                    key = w.rsplit("/", 1)[1].split(":")[0]  # ".../kernel:0" -> kernel
                    out[key] = torch.from_numpy(group[w][()])
                return out

            for name, conv in zip(conv_names, convs):
                cfg = layer_config[name]
                if (
                    tuple(cfg["kernel_size"]) != conv.kernel_size
                    or tuple(cfg["dilation_rate"]) != conv.dilation
                    or cfg["filters"] != conv.out_channels
                ):
                    raise ValueError(f"{path}: {name} does not match {conv}")
                w = arrays(name)
                conv.weight.data.copy_(w["kernel"].permute(2, 1, 0))
                conv.bias.data.copy_(w["bias"])
            for name, bn in zip(bn_names, bns):
                cfg = layer_config[name]
                momentum = 1 - cfg["momentum"]
                if cfg["epsilon"] != bn.eps or abs(momentum - bn.momentum) > 1e-9:
                    raise ValueError(f"{path}: {name} does not match {bn}")
                w = arrays(name)
                bn.weight.data.copy_(w["gamma"])
                bn.bias.data.copy_(w["beta"])
                bn.running_mean.copy_(w["moving_mean"])
                bn.running_var.copy_(w["moving_variance"])


class SpliceAI:
    """Factory class for SpliceAI models."""

    @staticmethod
    def from_preconfigured(model_name: str) -> nn.Module:
        """Create a SpliceAI model from a preconfigured name.

        Args:
            model_name (str): The name of the model to create.

        Returns:
            nn.Module: The created model.

        Raises:
            ValueError: If the model name is unknown.
        """
        if model_name == "80nt":
            return SpliceAI_80nt()
        elif model_name == "400nt":
            return SpliceAI_400nt()
        elif model_name == "2k":
            return SpliceAI_2k()
        elif model_name == "10k":
            return SpliceAI_10k()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
