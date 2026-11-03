"""Mamba-3 trunk, as a drop-in for the HyenaDNA backbone.

``Mamba3Backbone(bidirectional=True)`` stacks :class:`BiMamba3Block`, two
independent scans per block over the sequence and its reverse;
``bidirectional=False`` stacks :class:`Mamba3Block`, one causal scan.

Motivation
----------
The HyenaDNA trunk's ``bidirectional=True`` path does not actually see the
future (issue #106): symmetric zero-padding only delays a causal kernel, and
what non-causality it has comes from circular wrap-around aliasing. Mamba-3
sidesteps that question entirely -- bidirectionality here is by construction.

The causal variant exists to ask whether that construction earns its keep. The
bidirectional arm did not beat the padding-bidirectional HyenaDNA baseline
(ONBOARDING 4.31), so the arm that removes the second scan tests whether a
bidirectional trunk is doing anything for this task at all.

Mamba-3 (Mamba-3: Improved Sequence Modeling using State Space Principles,
ICLR 2026, arXiv 2603.15569) targets inference efficiency via trapezoidal
discretization, a MIMO formulation and data-dependent rotary embeddings. Our
speed benchmark (ONBOARDING 4.20) measured this stack as ~94% backbone-bound,
so trunk throughput passes through to end-to-end variant scoring almost
one-for-one.

Sizing
------
The HyenaDNA trunk is 8 layers totalling 7,268,992 parameters. At
``d_model=256, d_state=64, expand=2, headdim=64`` a block is 1,006,368
parameters -- two Mamba-3 scans, an RMSNorm, the 2*d_model -> d_model merge
projection and the LayerScale gain -- so eight of them come to 8,050,944,
**10.8% above** the HyenaDNA budget. A win by this arm therefore cannot be
attributed to architecture alone without noting the capacity difference; the
7-layer variant (7,044,576, 3.1% below) is the parameter-matched alternative.

A causal block drops one scan and the merge projection, so it is roughly half
of that. ``n_layer=16, bidirectional=False`` therefore lands near the HyenaDNA
budget again AND runs the same total number of scans as the 8-layer
bidirectional trunk, which is what keeps a directionality comparison from
turning into a capacity or a throughput comparison.

Stability
---------
Each block is near-identity at initialisation and stays bounded with depth:

* the two directions are merged by concatenation and a learned projection
  rather than summed, so a block contributes exactly ONE branch to the
  residual stream, and the mixture is learned rather than forced to be equal;
* ``LayerScale`` -- a per-channel gain initialised at 1e-4 -- gates that
  branch, so depth is opened up only as far as training needs;
* residual-path projections are divided by ``sqrt(n_layer)`` at construction,
  the GPT-2 scheme that ``mamba_ssm.models.mixer_seq_simple._init_weights``
  applies and that a hand-assembled trunk does not receive for free.

Measured on real batches, the first run of this trunk without those three
grew its residual stream 83x across 8 blocks at epoch 1 and 1484x by epoch 4,
with a single block amplifying its input 282x, and went NaN partway through
epoch 5.

Requires ``mamba_ssm >= 2.3.0`` for ``Mamba3``. The canonical venv is pinned to
2.2.6.post3 (CLAUDE.md venv rules), so jobs using this backbone put a 2.3.x
build first on ``PYTHONPATH`` rather than upgrading the shared environment.
"""

import math

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


class BiMamba3Block(nn.Module):
    """Pre-norm residual block running one forward and one reverse scan.

    Mamba is causal by construction, so bidirectionality is obtained the same
    way the SAV head does it (``model/sav_classifier.py``): a second, weight-
    independent scan consumes the reversed sequence and its output is flipped
    back before the two are summed.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        layer_scale_init: float = 1e-4,
    ) -> None:
        """Build the block.

        Args:
            d_model (int): Channel width, must match the trunk width.
            d_state (int): SSM state size per head.
            d_conv (int): Depthwise causal conv width inside each scan.
            expand (int): Inner expansion factor.
            headdim (int): Head dimension.
            layer_scale_init (float): Initial per-channel residual gain. Small
                so each block starts near-identity.

        Raises:
            ImportError: If the installed ``mamba_ssm`` predates ``Mamba3``.
        """
        super().__init__()
        try:
            from mamba_ssm import Mamba3
        except ImportError as exc:
            raise ImportError(
                "BiMamba3Block requires mamba_ssm >= 2.3.0 (for Mamba3). The "
                "canonical venv pins 2.2.6.post3; put a 2.3.x build first on "
                "PYTHONPATH instead of upgrading the shared environment."
            ) from exc

        self.norm = nn.RMSNorm(d_model)
        kwargs = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim,
        )
        self.mamba = Mamba3(**kwargs)
        self.mamba_rev = Mamba3(**kwargs)
        # Merge the two directions by concatenation and a learned projection
        # rather than summing them. Summing forces equal weight and injects two
        # full-magnitude branches into the residual stream per layer; the
        # projection lets the model choose the mixture and leaves exactly one
        # residual branch per block.
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)
        # LayerScale. Measured on the diverged run, a block amplified its input
        # 282x by epoch 4 and the residual stream grew 1484x across 8 blocks,
        # accelerating with training until it went NaN in epoch 5. A per-channel
        # gain starting near zero makes each block near-identity at init and
        # opens only as far as training needs.
        self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both directions and add them into the residual stream.

        Args:
            x (torch.Tensor): ``(batch_size, seq_len, d_model)``.

        Returns:
            torch.Tensor: Same shape as ``x``.
        """
        normed = self.norm(x)
        out = self.mamba(normed)
        rev = torch.flip(self.mamba_rev(torch.flip(normed, dims=[1])), dims=[1])
        merged = self.merge(torch.cat([out, rev], dim=-1))
        return x + self.layer_scale * merged


class Mamba3Block(nn.Module):
    """Pre-norm residual block running a single causal scan.

    The unidirectional twin of :class:`BiMamba3Block`: same operator, same
    normalisation and same LayerScale gate, with the reverse scan and the merge
    projection removed. Mamba is causal by construction, so a block built this
    way sees only ``[0, t]`` at position ``t``; the hybrid head's dilated
    convolution branch still reads both sides locally, so the arm removes
    long-range future context from the trunk, not all of it.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        layer_scale_init: float = 1e-4,
    ) -> None:
        """Build the block.

        Args:
            d_model (int): Channel width, must match the trunk width.
            d_state (int): SSM state size per head.
            d_conv (int): Depthwise causal conv width.
            expand (int): Inner expansion factor.
            headdim (int): Head dimension.
            layer_scale_init (float): Initial per-channel residual gain. Small
                so each block starts near-identity.

        Raises:
            ImportError: If the installed ``mamba_ssm`` predates ``Mamba3``.
        """
        super().__init__()
        try:
            from mamba_ssm import Mamba3
        except ImportError as exc:
            raise ImportError(
                "Mamba3Block requires mamba_ssm >= 2.3.0 (for Mamba3). The "
                "canonical venv pins 2.2.6.post3; put a 2.3.x build first on "
                "PYTHONPATH instead of upgrading the shared environment."
            ) from exc

        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba3(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim,
        )
        # Same LayerScale gate as the bidirectional block. It is what actually
        # bounds the residual stream at initialisation, so keeping it identical
        # is what makes the two arms comparable at step 0.
        self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the causal scan and add it into the residual stream.

        Args:
            x (torch.Tensor): ``(batch_size, seq_len, d_model)``.

        Returns:
            torch.Tensor: Same shape as ``x``.
        """
        return x + self.layer_scale * self.mamba(self.norm(x))


class Mamba3Backbone(nn.Module):
    """Stack of Mamba-3 blocks exposing the HyenaDNA trunk's interface.

    ``bidirectional`` selects the block: :class:`BiMamba3Block` (two scans) or
    :class:`Mamba3Block` (one causal scan).

    ``IssaSplice`` calls its trunk with ``inputs_embeds`` (the dynamic chunker
    feeds chunk-level embeddings) and reads
    ``outputs["hidden_states"][hidden_state_layer_index]``, so this returns the
    per-layer tuple rather than only the final state.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int = 8,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        padding_idx: int | None = None,
        layer_scale_init: float = 1e-4,
        max_residual_absmax: float = 1e4,
        bidirectional: bool = True,
    ) -> None:
        """Build the trunk.

        Args:
            vocab_size (int): Token vocabulary size for the embedding table.
            d_model (int): Channel width.
            n_layer (int): Number of blocks.
            d_state (int): SSM state size per head.
            d_conv (int): Depthwise conv width.
            expand (int): Inner expansion factor.
            headdim (int): Head dimension.
            padding_idx (int | None): Padding token id, if any.
            layer_scale_init (float): Initial per-channel residual gain per
                block.
            max_residual_absmax (float): Abort training if the residual stream
                exceeds this magnitude. The first version of this trunk reached
                7,528 by epoch 4 while still reporting a healthy loss, then
                spent five further epochs and ~28 GPU-hours emitting NaN. A
                ceiling turns that into a fast, diagnosable failure. Set to 0
                to disable.
            bidirectional (bool): ``True`` stacks :class:`BiMamba3Block` (a
                forward and a reverse scan, merged); ``False`` stacks
                :class:`Mamba3Block`, one causal scan per block. A causal block
                is roughly half the parameters and half the scan work, so an
                arm that means to test directionality rather than capacity
                doubles ``n_layer`` when it turns this off.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.max_residual_absmax = float(max_residual_absmax)
        self.bidirectional = bool(bidirectional)
        self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        block_cls = BiMamba3Block if self.bidirectional else Mamba3Block
        self.layers = nn.ModuleList(
            block_cls(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, headdim=headdim,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(n_layer)
        )
        self.norm_f = nn.RMSNorm(d_model)
        self._apply_residual_rescale(n_layer)

    def _apply_residual_rescale(self, n_layer: int) -> None:
        """Scale residual-path projections by 1/sqrt(N), as the library does.

        ``mamba_ssm.models.mixer_seq_simple._init_weights`` applies the GPT-2
        scheme to every weight feeding a residual add, so the accumulated
        variance does not grow with depth. This trunk is assembled by hand and
        so never received it, which is one of the reasons the residual stream
        grew 1484x across 8 blocks before diverging.

        Both block types contribute exactly one branch to the residual stream,
        so N = n_layer either way.

        Every projection feeding that branch is scaled: for the bidirectional
        block that is the merge projection and both scans' ``out_proj``, for
        the causal block the single ``out_proj``. The bidirectional block
        therefore ends up scaled twice over (merge after out_proj, 1/N in
        total) where the causal block is scaled once. That asymmetry is
        deliberate: the trained bidirectional arm's initialisation is
        provenance and is left byte-identical, and the causal block gets the
        textbook GPT-2 scheme. Either way ``layer_scale`` (1e-4) is what
        actually bounds the branch at step 0.

        Args:
            n_layer (int): Number of blocks in the trunk.
        """
        scale = math.sqrt(n_layer)
        with torch.no_grad():
            for block in self.layers:
                merge = getattr(block, "merge", None)
                if merge is not None:
                    merge.weight.div_(scale)
                scans = [block.mamba]
                if hasattr(block, "mamba_rev"):
                    scans.append(block.mamba_rev)
                for scan in scans:
                    out_proj = getattr(scan, "out_proj", None)
                    if out_proj is not None:
                        out_proj.weight.div_(scale)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table.

        Returns:
            nn.Embedding: The embedding module.
        """
        return self.embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutputWithNoAttention:
        """Run the trunk.

        Args:
            input_ids (torch.Tensor | None): ``(batch_size, seq_len)`` token
                ids. Ignored when ``inputs_embeds`` is given.
            inputs_embeds (torch.Tensor | None): ``(batch_size, seq_len,
                d_model)`` precomputed embeddings, as supplied by the chunker.
            output_hidden_states (bool | None): Accepted for interface parity;
                per-layer states are always returned because ``IssaSplice``
                indexes into them.
            return_dict (bool | None): Accepted for interface parity.

        Returns:
            BaseModelOutputWithNoAttention: ``last_hidden_state`` plus the
                per-layer ``hidden_states`` tuple (embeddings first, final
                normalised state last).

        Raises:
            ValueError: If neither ``input_ids`` nor ``inputs_embeds`` is given.
        """
        del output_hidden_states, return_dict

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "Mamba3Backbone requires either input_ids or inputs_embeds."
                )
            inputs_embeds = self.embeddings(input_ids)

        hidden = inputs_embeds
        all_hidden: list[torch.Tensor] = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            all_hidden.append(hidden)

        # One reduction on the final state in the common case; the per-block
        # walk only runs when the ceiling is already breached.
        if self.max_residual_absmax > 0 and self.training:
            final_absmax = float(hidden.detach().abs().max())
            if (not math.isfinite(final_absmax)
                    or final_absmax > self.max_residual_absmax):
                trace = ", ".join(
                    f"b{i}={float(h.detach().abs().max()):.3g}"
                    for i, h in enumerate(all_hidden)
                )
                raise RuntimeError(
                    f"Mamba3Backbone residual stream reached "
                    f"{final_absmax:.4g} (ceiling {self.max_residual_absmax:g}). "
                    f"Per-block absmax: {trace}. Growth with depth means the "
                    f"LayerScale gates have opened too far or the residual "
                    f"rescale is not in effect; a non-finite value means the "
                    f"blow-up already happened. Stopping here rather than "
                    f"training on into NaN."
                )

        hidden = self.norm_f(hidden)
        all_hidden[-1] = hidden

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden,
            hidden_states=tuple(all_hidden),
        )
