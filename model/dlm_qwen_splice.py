"""Frozen DNA-language-model representation -> Qwen transformer -> splice-site head.

A 15,000 nt window is encoded once by one or more frozen DNA language models
(DLMs), projected into the Qwen hidden size and run through a fully fine-tuned
Qwen transformer body. Acceptor/donor/neither logits are read out over the centre
target region (5,000 nt), matching the SpliceAI-format H5 layout used everywhere
else in this repo (``X`` = 15,000 nt, ``Y`` = centre 5,000 nt).

Two DLM backbones are supported and may be combined:

``evo2``
    Evo 2 StripedHyena-2, 7B, single-nucleotide, 1M context -- but **causal**, so
    a position's representation carries no downstream sequence.
``caduceus``
    Caduceus BiMamba with reverse-complement parameter sharing, ~8M, 131k
    context -- **bidirectional**, so each position sees both directions in one
    pass. Output width is ``d_model * 2`` because RCPS concatenates the forward
    and (position-aligned) reverse-complement channels.

Class order is ``[acceptor, donor, neither]`` to match ``SplicingH5Dataset``.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from model.caduceus import Caduceus
from model.evo2 import Evo2Config, Evo2Model

CONTEXT_LEN = 15000
TARGET_LEN = 5000

DLM_DEFAULT_PATHS = {
    "evo2": "Taykhoom/Evo2-7B-262K",
    "caduceus": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
}

# H5 int8 codes (0=N, 1=A, 2=C, 3=G, 4=T) -> each DLM's token ids.
#   evo2     : byte tokenizer, ids are ``ord(letter)`` clamped to [0, 511].
#   caduceus : char vocab [BOS, UNK, SEP, PAD, CLS, MASK, N, A, C, G, T].
DLM_TOKEN_LUTS = {
    "evo2": (78, 65, 67, 71, 84),
    "caduceus": (6, 7, 8, 9, 10),
}

# Complement within the H5 int8 code space: N->N, A->T, C->G, G->C, T->A.
COMPLEMENT_LUT = (0, 4, 3, 2, 1)


class FrozenDlm(nn.Module):
    """One frozen DNA language model plus its nucleotide-code lookup table."""

    def __init__(self, name: str, path: str | None = None) -> None:
        """Load the backbone in bfloat16 and freeze it.

        Args:
            name (str): ``"evo2"`` or ``"caduceus"``.
            path (str | None): Checkpoint path or HuggingFace id. Defaults to
                the entry in ``DLM_DEFAULT_PATHS``.
        """
        super().__init__()
        if name not in DLM_TOKEN_LUTS:
            raise ValueError(
                f"unknown DLM {name!r}; expected one of {sorted(DLM_TOKEN_LUTS)}"
            )
        self.name = name
        path = path or DLM_DEFAULT_PATHS[name]

        if name == "evo2":
            # FP8 input projections are a TransformerEngine *compute* path, not a
            # storage format: the weights are bfloat16 either way and the
            # pure-PyTorch fallback registers identical parameter names, so
            # disabling FP8 is the same math at higher precision. It also drops
            # the FP8 kernel's pad-to-multiple-of-16 requirement on the sequence.
            config = Evo2Config.from_pretrained(
                path, use_fp8_input_projections=False
            )
            self.model = Evo2Model.from_pretrained(
                path, config=config, dtype=torch.bfloat16
            )
            self.dim = config.hidden_size
            self.bidirectional = False
        else:
            # Vendored rather than trust_remote_code: see model/caduceus/README.md.
            self.model = Caduceus.from_pretrained(path, dtype=torch.bfloat16)
            config = self.model.config
            # RCPS concatenates the forward and reverse-complement channels.
            self.dim = config.d_model * (2 if getattr(config, "rcps", False) else 1)
            self.bidirectional = True

        self.model.requires_grad_(False)
        self.model.eval()
        self.register_buffer(
            "lut",
            torch.tensor(DLM_TOKEN_LUTS[name], dtype=torch.long),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Encode int8 nucleotide codes.

        Args:
            codes (torch.Tensor): shape ``(batch, seq_len)``, values in 0..4.

        Returns:
            torch.Tensor: shape ``(batch, seq_len, self.dim)``.
        """
        ids = self.lut[codes.long()]
        if self.name == "evo2":
            out = self.model(input_ids=ids, use_cache=False, return_dict=True)
        else:
            out = self.model(input_ids=ids, return_dict=True)
        return out.last_hidden_state


def _find_text_body(model: nn.Module) -> nn.Module:
    """Return the decoder stack that consumes ``inputs_embeds``.

    Qwen3.5 checkpoints are multimodal wrappers, so the text transformer sits a
    couple of attributes down. Walk the known layouts rather than hard-coding
    one, and fail loudly with what was actually found.

    Args:
        model (nn.Module): Model loaded by ``AutoModel.from_pretrained``.

    Returns:
        nn.Module: Submodule exposing ``layers`` and accepting ``inputs_embeds``.
    """
    for path in (
        (),
        ("language_model",),
        ("model",),
        ("model", "language_model"),
        ("language_model", "model"),
        ("model", "text_model"),
    ):
        node: nn.Module | None = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers"):
            return node
    raise AttributeError(
        "could not locate the Qwen text body; top-level children = "
        f"{[n for n, _ in model.named_children()]}"
    )


class DlmQwenSplice(nn.Module):
    """Frozen DLM(s) -> projection -> Qwen body -> per-nt 3-class head."""

    def __init__(
        self,
        dlm: str = "evo2",
        dlm_paths: dict[str, str] | None = None,
        qwen_path: str = "Qwen/Qwen3.5-4B",
        strands: str = "fwd",
        target_len: int = TARGET_LEN,
        gradient_checkpointing: bool = True,
        qwen_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the stack.

        Args:
            dlm (str): One or more DLM names joined by ``+`` (e.g. ``"evo2"``,
                ``"caduceus"``, ``"evo2+caduceus"``). Their per-position outputs
                are concatenated.
            dlm_paths (dict[str, str] | None): Per-name checkpoint overrides.
            qwen_path (str): HuggingFace id or local path of the Qwen model.
            strands (str): ``"fwd"`` encodes the given strand only. ``"both"``
                additionally encodes the reverse complement and concatenates the
                position-aligned result -- but only for **causal** DLMs, since a
                bidirectional one already carries downstream context and a second
                pass would just duplicate it.
            target_len (int): Width of the centre region that is scored.
            gradient_checkpointing (bool): Enable checkpointing on the Qwen body.
            qwen_dtype (torch.dtype): Storage dtype for the trainable Qwen body.
                Defaults to float32 so AdamW keeps full-precision moments;
                compute still runs in bfloat16 under autocast.
        """
        super().__init__()
        if strands not in ("fwd", "both"):
            raise ValueError(f"strands must be 'fwd' or 'both', got {strands!r}")
        self.strands = strands
        self.target_len = target_len

        paths = dlm_paths or {}
        self.dlms = nn.ModuleList(
            [FrozenDlm(name, paths.get(name)) for name in dlm.split("+")]
        )
        self.dlm_names = [d.name for d in self.dlms]

        qwen = AutoModel.from_pretrained(qwen_path, dtype=qwen_dtype)
        # Enable on the wrapper first -- it always exposes the helper, whereas the
        # inner text body may not once it is detached from the multimodal shell.
        if gradient_checkpointing:
            # Non-reentrant checkpointing is the variant that composes with DDP.
            qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.qwen = _find_text_body(qwen)
        self.qwen.gradient_checkpointing = gradient_checkpointing
        # inputs_embeds bypasses the token embedding, so its parameters can never
        # receive a gradient. Freeze them: keeping them trainable both breaks the
        # optimiser's expectations and wastes ~16 bytes/param of AdamW state on a
        # matrix that is meaningless for DNA input anyway.
        embed = getattr(self.qwen, "embed_tokens", None)
        if embed is not None:
            embed.requires_grad_(False)
        qwen_cfg = AutoConfig.from_pretrained(qwen_path)
        qwen_dim = getattr(qwen_cfg, "text_config", qwen_cfg).hidden_size

        in_dim = sum(d.dim * (2 if self._doubles(d) else 1) for d in self.dlms)
        self.feature_dim = in_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, qwen_dim),
        )
        self.head = nn.Linear(qwen_dim, 3)

        self.register_buffer(
            "_complement",
            torch.tensor(COMPLEMENT_LUT, dtype=torch.long),
            persistent=False,
        )

    def _doubles(self, dlm: FrozenDlm) -> bool:
        """Whether this DLM gets a second, reverse-complement pass.

        Args:
            dlm (FrozenDlm): Backbone to check.

        Returns:
            bool: True when ``strands="both"`` and the backbone is causal.
        """
        return self.strands == "both" and not dlm.bidirectional

    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        """Concatenate the per-position features of every configured DLM.

        Args:
            codes (torch.Tensor): shape ``(batch, seq_len)``, values in 0..4.

        Returns:
            torch.Tensor: shape ``(batch, seq_len, feature_dim)``.
        """
        rc: torch.Tensor | None = None
        parts: list[torch.Tensor] = []
        for dlm in self.dlms:
            parts.append(dlm(codes))
            if self._doubles(dlm):
                if rc is None:
                    rc = self._complement[codes.long()].flip(dims=[1])
                parts.append(dlm(rc).flip(dims=[1]))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def forward(
        self, codes: torch.Tensor, full_length: bool = False
    ) -> torch.Tensor:
        """Score the centre target region, or every input position.

        Args:
            codes (torch.Tensor): shape ``(batch, 15000)``, values in 0..4
                (0=N, 1=A, 2=C, 3=G, 4=T).
            full_length (bool): Read the head out over all ``codes.shape[1]``
                positions instead of the centre ``target_len``. Training always
                uses the default; the SAV scorers need it because the project's
                canonical ``liberal`` aggregation mode reads the flank, so
                cropping here would silently change the readout. Flank
                positions are extrapolation -- the model was never supervised
                there -- which is exactly what ``conservative`` mode excludes.

        Returns:
            torch.Tensor: logits of shape ``(batch, target_len, 3)``, or
                ``(batch, seq_len, 3)`` when ``full_length``, ordered
                ``[acceptor, donor, neither]``.
        """
        hidden = self.proj(self.encode(codes).float())
        hidden = self.qwen(inputs_embeds=hidden, use_cache=False, return_dict=True)
        hidden = hidden.last_hidden_state

        if not full_length:
            start = (hidden.shape[1] - self.target_len) // 2
            hidden = hidden[:, start : start + self.target_len, :]
        return self.head(hidden).float()

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return every parameter that receives gradients.

        Returns:
            list[nn.Parameter]: Qwen body, projection and head parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr: float, head_lr: float) -> list[dict]:
        """Split trainable parameters into pretrained-body and fresh-head groups.

        The projection and head are randomly initialised while the Qwen body
        carries pretrained weights, so they want different learning rates --
        the same split the project's reference recipe makes between backbone
        and heads.

        Args:
            lr (float): Learning rate for the pretrained Qwen body.
            head_lr (float): Learning rate for the projection and output head.

        Returns:
            list[dict]: Optimiser parameter groups.
        """
        fresh = list(self.proj.parameters()) + list(self.head.parameters())
        fresh_ids = {id(p) for p in fresh}
        body = [
            p for p in self.parameters() if p.requires_grad and id(p) not in fresh_ids
        ]
        return [
            {"params": body, "lr": lr, "name": "qwen_body"},
            {"params": fresh, "lr": head_lr, "name": "proj_head"},
        ]
