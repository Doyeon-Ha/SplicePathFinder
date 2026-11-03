"""Two-pass bidirectional HyenaDNA trunk (bi-HyenaDNA).

Motivation
----------
HyenaDNA ships two ways of asking a causal long convolution for future context,
and neither uses the released weights the way they were trained:

* ``bidirectional_mode="padding"`` (the ``SplicePath`` / ``SpliceBoa`` default)
  only *delays* a causal kernel -- what little non-causality it has comes from
  circular wrap-around aliasing, not from seeing the future
  (``test/test_hyena_bidirectionality.py``).
* ``bidirectional_mode="k_rev"`` folds in a second, anti-causal implicit filter.
  That is genuinely two-sided, but the anti-causal filter has no pretrained
  counterpart, so it must be learned from scratch.

This module takes the route the Mamba line already takes here
(:class:`model.mamba3_backbone.BiMamba3Block`, ``model/sav_classifier.py``):
leave the operator strictly causal and read the sequence twice, forward and
reversed. Every parameter then has a pretrained counterpart, so both directions
can warm-start from the released causal checkpoint.

Structure
---------
Bidirectionality is per block, not per trunk: each of the ``n_layer`` Hyena
blocks gets a weight-independent reverse twin, and the two directions are merged
before the next block, so depth-``i+1`` filters already see two-sided context::

    h_fwd = block_fwd(x)
    h_rev = flip(block_rev(flip(x)))
    x     = merge([h_fwd ; h_rev])          # Linear(2*d_model -> d_model)

``merge`` starts at ``[0.5*I | 0.5*I]``, so at initialisation a block returns the
mean of its two directions. With both twins warm-started from the same
pretrained block that makes the whole trunk a well-scaled perturbation of the
pretrained model at step 0 -- a random projection there would discard the warm
start on the first forward pass. The mixture is learned from that point on.
``HyenaBlock`` closes its own residual path, so the mean of two blocks is again
"input plus a bounded update"; the trunk needs no LayerScale gate for stability,
unlike the from-scratch Mamba-3 trunk.

Sizing
------
Doubling the blocks and adding the merge takes the medium trunk (``d_model=256``,
``n_layer=8``) from 6.55M to ~14.1M parameters. That is only affordable next to
``med_ei_ac`` (15.46M) because this arm also drops the ~7.8M dynamic chunker;
the two changes roughly cancel, which is what keeps the comparison honest on
parameter count.

Assumptions
-----------
Reversal is ``torch.flip`` over the length axis, matching the Mamba blocks here.
It is a plain reversal, not a reverse complement, so the reverse pass reads a
strand that does not occur in nature; the trunk is asked to learn that frame.

Batches must be constant-length and left-aligned (our 15,000nt windows are).
Padding at the tail of a sequence would move to the *head* under the flip and
shift the causal frame of the reverse pass, which this module does not correct
for.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

from .hyenadna.src.configuration_hyena import HyenaConfig
from .hyenadna.src.modeling_hyena import (
    HyenaBlock,
    HyenaDNAPreTrainedModel,
    HyenaEmbeddings,
)


class BiHyenaBlock(nn.Module):
    """One Hyena block read forward and reversed, then merged.

    The two directions have independent weights (the bi-Mamba convention in
    this repository); only the merge projection is new relative to the
    pretrained architecture.
    """

    def __init__(self, config: HyenaConfig) -> None:
        """Build the block.

        Args:
            config (HyenaConfig): Model configuration. Consumed by both
                ``HyenaBlock`` twins; ``config.bidirectional`` is expected to
                be ``False`` here, because the whole point of this trunk is to
                keep the operator causal and get the second direction from the
                reversed pass.
        """
        super().__init__()
        self.fwd = HyenaBlock(config)
        self.rev = HyenaBlock(config)
        self.merge = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.reset_merge_to_mean()

    def reset_merge_to_mean(self) -> None:
        """Set ``merge`` to ``[0.5*I | 0.5*I]`` (average the two directions).

        Called again after :meth:`transformers.PreTrainedModel.post_init`, which
        would otherwise overwrite the projection with ``normal(0, 0.02)`` like
        any other ``nn.Linear`` and throw away the warm start.
        """
        d_model = self.merge.out_features
        with torch.no_grad():
            self.merge.weight.zero_()
            eye = torch.eye(d_model, dtype=self.merge.weight.dtype)
            self.merge.weight[:, :d_model].copy_(0.5 * eye)
            self.merge.weight[:, d_model:].copy_(0.5 * eye)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run both directions and merge them.

        Args:
            hidden_states (torch.Tensor): ``(batch_size, seq_len, d_model)``.

        Returns:
            torch.Tensor: Same shape as ``hidden_states``.
        """
        out_fwd = self.fwd(hidden_states)
        flipped = torch.flip(hidden_states, dims=[1])
        out_rev = torch.flip(self.rev(flipped), dims=[1])
        merged = torch.cat([out_fwd, out_rev], dim=-1)
        return self.merge(merged.to(dtype=self.merge.weight.dtype))


class BiHyenaLMBackbone(nn.Module):
    """Stack of :class:`BiHyenaBlock`, mirroring ``HyenaLMBackbone``.

    Same call signature and same ``all_hidden_states`` contract (embeddings
    first, final normalised state last), so ``IssaSplice`` can index into it
    with ``hidden_state_layer_index`` exactly as for the causal trunk.
    """

    def __init__(self, config: HyenaConfig) -> None:
        """Build the trunk.

        Args:
            config (HyenaConfig): Model configuration.
        """
        super().__init__()
        self.embeddings = HyenaEmbeddings(config)
        self.dropout = nn.Dropout(config.embed_dropout)
        self.layers = nn.ModuleList(
            [BiHyenaBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def reset_merge_to_mean(self) -> None:
        """Re-apply the mean initialisation to every block's merge projection."""
        for layer in self.layers:
            layer.reset_merge_to_mean()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the trunk.

        Args:
            input_ids (torch.Tensor | None): ``(batch_size, seq_len)`` token
                ids. Ignored when ``inputs_embeds`` is given.
            inputs_embeds (torch.Tensor | None): ``(batch_size, seq_len,
                d_model)`` precomputed embeddings, for callers that bypass the
                token-embedding table (the dynamic chunker and the one-hot
                front end both feed the trunk this way).
            output_hidden_states (bool): Collect the per-layer states.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Final normalised state and
                the per-layer states (empty when ``output_hidden_states`` is
                ``False``).

        Raises:
            ValueError: If neither ``input_ids`` nor ``inputs_embeds`` is given.
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "BiHyenaLMBackbone requires either input_ids or inputs_embeds."
                )
            hidden_states = self.embeddings(input_ids)

        all_hidden_states: list[torch.Tensor] = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states.to(dtype=self.ln_f.weight.dtype))
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


class BiHyenaDNAModel(HyenaDNAPreTrainedModel):
    """``HyenaDNAModel`` with the two-pass bidirectional trunk.

    Subclasses the upstream base class so the from-scratch arm gets the same
    weight-initialisation scheme (``normal(0, initializer_range)`` plus the
    GPT-2 residual rescale) as the causal and padding-bidirectional baselines --
    an init difference would otherwise confound the twin comparison.
    """

    def __init__(self, config: HyenaConfig, **kwargs: object) -> None:
        """Build the model.

        Args:
            config (HyenaConfig): Model configuration.
            **kwargs (object): Forwarded to the base class.
        """
        super().__init__(config, **kwargs)
        self.backbone = BiHyenaLMBackbone(config)
        self.config = config
        self.post_init()
        # post_init re-initialises every nn.Linear, merge projections included.
        self.backbone.reset_merge_to_mean()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutputWithNoAttention:
        """Run the trunk and return it in the HuggingFace output container.

        Args:
            input_ids (torch.Tensor | None): ``(batch_size, seq_len)`` token ids.
            inputs_embeds (torch.Tensor | None): Precomputed embeddings.
            output_hidden_states (bool | None): Collect per-layer states.
            return_dict (bool | None): Accepted for interface parity; this
                trunk always returns the output container, matching how
                ``IssaSplice`` reads ``outputs["hidden_states"]``.

        Returns:
            BaseModelOutputWithNoAttention: ``last_hidden_state`` plus the
                per-layer ``hidden_states`` tuple when requested.
        """
        del return_dict
        output_hidden_states = bool(output_hidden_states)
        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def load_causal_pretrained_weights(
    model: BiHyenaDNAModel,
    causal_state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Warm-start both direction stacks from a causal HyenaDNA checkpoint.

    Every ``backbone.layers.<i>.<rest>`` tensor of the causal checkpoint is
    copied into both ``backbone.layers.<i>.fwd.<rest>`` and
    ``...rev.<rest>``; the embeddings and ``ln_f`` are copied verbatim. The
    merge projections have no pretrained counterpart and keep their mean
    initialisation.

    Args:
        model (BiHyenaDNAModel): Target model, modified in place.
        causal_state_dict (dict[str, torch.Tensor]): ``state_dict`` of a causal
            ``HyenaDNAModel`` (as released by LongSafari).

    Returns:
        dict[str, int]: ``{"copied", "merge_kept", "unused_source"}`` counts,
            for logging.

    Raises:
        ValueError: If any target tensor other than a merge projection is left
            uninitialised, or if a shape disagrees. Silently falling back to
            random weights would look like a successful warm start and cost a
            full training run to notice.
    """
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    unused_source = 0

    for key, tensor in causal_state_dict.items():
        if key.startswith("backbone.layers."):
            tail = key[len("backbone.layers.") :]
            index, _, rest = tail.partition(".")
            targets = [
                f"backbone.layers.{index}.fwd.{rest}",
                f"backbone.layers.{index}.rev.{rest}",
            ]
        else:
            targets = [key]

        hit = False
        for target_key in targets:
            if target_key not in target:
                continue
            if target[target_key].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch warm-starting {target_key}: checkpoint "
                    f"{tuple(tensor.shape)} vs model "
                    f"{tuple(target[target_key].shape)}."
                )
            mapped[target_key] = tensor
            hit = True
        if not hit:
            unused_source += 1

    missing = [
        key for key in target if key not in mapped and not key.endswith(".merge.weight")
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} bi-HyenaDNA tensors found no counterpart in the "
            f"causal checkpoint, e.g. {missing[:5]}. Refusing a partial warm "
            f"start -- it is indistinguishable from a successful one in the logs."
        )

    model.load_state_dict(mapped, strict=False)
    return {
        "copied": len(mapped),
        "merge_kept": len(target) - len(mapped),
        "unused_source": unused_source,
    }
