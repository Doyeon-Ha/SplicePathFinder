"""
IssaSplice model class
Model for predicting splicing effects using various backbone models.
"""

import logging
import os
import time
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from model.chunker import Dechunker, DynamicChunker, HierarchicalChunker
from model.hyenadna.src import (
    HyenaDNAModel,
    OverlappingKmerTokenizer,
)
from model.layers import (
    AttentionClassificationHead,
    Conv1DResidualBlock,
    ConvolutionalHead,
    DeepClassificationHead,
)
from utils.utils import (
    init_weights_factory,
    log_only_master_process,
    normalize_checkpoint_keys,
    select_matching_state_dict,
)


def _resolve_mlm_fallback_config_source(
    model_cfg: DictConfig | dict,
) -> str | None:
    """Derive the MLM fallback config source from `model_options[model_index]`.

    Replaces the previously cfg-defined interpolation
    ``mlm_fallback_config_source: ${model.model_options[${model.model_index}]}``,
    which OmegaConf 2.3 grammar cannot evaluate (the inner
    ``${model.model_index}`` returns an ``int`` and the grammar rejects
    non-string values as path keys inside ``[...]`` indexing). The
    interpolation always meant "the same HF id as the selected model" —
    this helper reproduces that semantics from plain Python.

    Args:
        model_cfg (DictConfig | dict): The model section of the training
            config.

    Returns:
        str | None: ``model_options[model_index]`` when both keys are
            present and the index is in range; ``None`` otherwise.
    """
    options = OmegaConf.select(model_cfg, "model_options", default=None)
    index = OmegaConf.select(model_cfg, "model_index", default=None)
    if not options or index is None:
        return None
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(options):
        return None
    return str(options[idx])


class IssaSplice(nn.Module):
    """
    SplicePath implementation, retaining the checkpoint-compatible class name.
    """

    def __init__(self, cfg: DictConfig | dict) -> None:
        """
        Initialize the IssaSplice model from a configuration object.

        Args:
            cfg (DictConfig | dict): Configuration for model initialization. Expected
                keys include (but are not limited to):
                - model_name | (model_options, model_index)
                - device
                - n_classes, backbone_output_dim, dropout_rate
                - download, freeze_backbone, freeze_policy, partial_freeze_layers
                - weight_dtype, backbone_type | model_type
                - classification_head_type, seq_length, etc.
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Normalize config to DictConfig for dot-access and .get support
        self.config = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

        # Pull commonly used, top-level attributes with sensible defaults
        self.model_name = self.config.model_name

        self.dropout_rate = self.config.dropout_rate
        self.freeze_backbone = self.config.freeze_backbone
        self.freeze_policy = self.config.freeze_policy
        self.partial_freeze_layers = self.config.partial_freeze_layers
        self.weight_dtype = self.config.init_weight_dtype
        self.torch_dtype = self._get_torch_dtype(self.weight_dtype)
        # Autocast is always enabled regardless of weight dtype
        # Autocast dtype follows torch_dtype, or auto-selects based on GPU capability
        self.autocast_dtype = self._determine_autocast_dtype()
        self.backbone_type = self.config.model_type
        self.hidden_state_layer_index = getattr(
            self.config, "hidden_state_layer_index", -1
        )
        self.seq_length = self.config.seq_length
        self.base_backbone_output_dim = self.config.base_backbone_output_dim
        self.local_feature_output_dim = self.config.local_feature_output_dim
        self.backbone_output_dim = self.base_backbone_output_dim
        self.use_hybrid_backbone = getattr(self.config, "use_hybrid_backbone", False)

        # Replace the backbone's learned token-embedding lookup with a learned
        # linear projection of the fixed one-hot nt encoding -- the same
        # encoding the hybrid conv head reads. With the dynamic chunker the
        # projection produces the nt embedding the chunker consumes; without
        # it, the backbone's ``inputs_embeds``. Created in
        # :meth:`_init_onehot_input_proj`; stays ``None`` when off.
        self.backbone_onehot_input = bool(
            OmegaConf.select(self.config, "backbone_onehot_input", default=False)
        )
        self.onehot_input_proj: nn.Module | None = None

        # Channels of that one-hot encoding. 5 (default) = A, C, G, T, N.
        # 4 = A, C, G, T, with N encoded as all zeros like the special tokens.
        self.onehot_channels = int(
            OmegaConf.select(self.config, "onehot_channels", default=5)
        )
        if self.onehot_channels not in (4, 5):
            raise ValueError(
                f"model.onehot_channels must be 4 or 5; got {self.onehot_channels}."
            )

        # What the hybrid dilated-conv (local context) branch reads.
        # "onehot" (default) keeps the fixed 5-channel A,C,G,T,N encoding every
        # existing checkpoint was trained with. "embedding" feeds it the
        # backbone's own nt embedding instead, so the local and global branches
        # share one learned table -- weight tying across the two context
        # scales, and one lookup serves both.
        self.hybrid_head_input = str(
            OmegaConf.select(self.config, "hybrid_head_input", default="onehot")
        )
        if self.hybrid_head_input not in ("onehot", "embedding"):
            raise ValueError(
                "model.hybrid_head_input must be 'onehot' or 'embedding'; got "
                f"{self.hybrid_head_input!r}."
            )

        self.backbone_additional_dim = self.config.get("backbone_additional_dim", 0)

        # Optional per-position input feature track (e.g. intron structure,
        # conservation) added to the nt embeddings *before* the dynamic chunker
        # + backbone, so the features flow through the full long-range trunk
        # (not just a local head). Created in
        # :meth:`_init_dynamic_chunker_if_enabled` where the embedding dim is
        # known; stays ``None`` when ``nt_feature_channels == 0``.
        self.nt_feature_channels = int(
            OmegaConf.select(self.config, "nt_feature_channels", default=0)
        )
        self.feature_proj: nn.Module | None = None


        # 1. Initialize Backbone
        self.model = None
        self._initialize_model()

        if hasattr(self, "tokenizer"):
            self._create_token_to_onehot_map()

        # 2. Weight Initialization Logic
        self._init_weights = init_weights_factory(self.config.linear_layer_init_method)

        if self.use_hybrid_backbone:
            self.hybrid_head = None
            self._initialize_hybrid_head()
            self.backbone_output_dim += self.local_feature_output_dim
            self.local_norm = nn.LayerNorm(self.local_feature_output_dim)

        if self.backbone_onehot_input:
            self._init_onehot_input_proj()

        # The chunker resolves this too; embedding-input mode needs it without a
        # chunker, so resolve it here when it is not already set.
        if self.hybrid_head_input == "embedding" and not hasattr(
            self, "_nt_embedding_layer"
        ):
            self._nt_embedding_layer = IssaSplice._resolve_nt_embedding_layer(
                self.model
            )

        # hidden_size: backbone + hybrid dimension only,
        # used for head hidden layer sizing (matches legacy
        # checkpoint architecture).
        self.hidden_size = self.backbone_output_dim

        # 4. Initialize Additional Features (Position, etc.)
        if self.backbone_additional_dim > 0:
            self.additional_pos_features = nn.Parameter(
                torch.randn(1, self.seq_length, self.backbone_additional_dim)
            )
            self.backbone_output_dim += self.backbone_additional_dim
        else:
            self.additional_pos_features = None

        # 5. Initialize Heads
        self.classification_head = None
        self.aux_region_head: nn.Module | None = None
        self.aux_exon_head: nn.Module | None = None
        self.aux_usage_head: nn.Module | None = None
        self.aux_spliced_head: nn.Module | None = None
        self.aux_phylop_head: nn.Module | None = None
        aux_cfg = self.config.get("aux_heads", None)
        self.aux_region_enabled = bool(
            aux_cfg is not None
            and OmegaConf.select(self.config, "aux_heads.enable_region", default=False)
        )
        self.aux_exon_enabled = bool(
            aux_cfg is not None
            and OmegaConf.select(self.config, "aux_heads.enable_exon", default=False)
        )
        # Quantitative splice-site usage regression head (4 tissues), trained
        # on Pangolin's SSU targets for splice-altering-variant prediction.
        self.aux_usage_enabled = bool(
            aux_cfg is not None
            and OmegaConf.select(self.config, "aux_heads.enable_usage", default=False)
        )
        # Per-tissue spliced/not classification head (Pangolin-faithful, 2-class,
        # cryptic-aware). This is the head scored for SAV in the multi-species
        # SSU reproduction (4 tissues x 2 classes).
        self.aux_spliced_enabled = bool(
            aux_cfg is not None
            and OmegaConf.select(self.config, "aux_heads.enable_spliced", default=False)
        )
        # Per-base phyloP conservation regression head (one scalar per
        # position), trained on the UCSC phyloP100way track as a third task
        # next to the 3-class splice head and the exon/intron head.
        self.aux_phylop_enabled = bool(
            aux_cfg is not None
            and OmegaConf.select(self.config, "aux_heads.enable_phylop", default=False)
        )
        self.aux_region_num_classes = int(
            OmegaConf.select(self.config, "aux_heads.region_num_classes", default=7)
        )
        self.aux_exon_num_classes = int(
            OmegaConf.select(self.config, "aux_heads.exon_num_classes", default=3)
        )
        self.aux_usage_num_classes = int(
            OmegaConf.select(self.config, "aux_heads.usage_num_classes", default=4)
        )
        self.aux_spliced_num_tissues = int(
            OmegaConf.select(self.config, "aux_heads.spliced_num_tissues", default=4)
        )
        self._initialize_fc_layers(input_dim=self.backbone_output_dim)

        # 7. Finalize (Dtype, Freeze, Logging)
        self._apply_dtype_conversion()
        self.apply_freeze_policy(self.freeze_policy, self.partial_freeze_layers)

        # 8. Initialize Dynamic Chunker (after freeze so chunker is excluded
        # Multi-stage support (F5): ``hierarchical_chunker`` wraps N
        # ``(DynamicChunker, Dechunker)`` pairs (N>=1). When the config's
        # ``model.dynamic_chunker.stages`` list is non-empty, each entry
        # defines one outer-to-inner stage; otherwise the flat config
        # builds a single-stage hierarchy (legacy behaviour). Legacy
        # ``self.chunker`` / ``self.dechunker`` aliases point to stage 0
        # for backward-compat with tests and external code.
        #    from freeze policy by construction; DD-013).
        self.hierarchical_chunker: HierarchicalChunker | None = None
        self._init_dynamic_chunker_if_enabled()

        # Initialize timing and logging
        self.enable_timing_log = self.config.enable_timing_log
        self.timing_stats = {
            "backbone_times": [],
            "classification_head_times": [],
            "total_forward_count": 0,
            "log_interval": self.config.timing_log_interval,
        }

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Convert string dtype to torch dtype.

        Args:
            dtype_str (str): String representation of dtype.
            Supported values:
                - "float32": Full precision (default).
                - "bfloat16": Brain floating point (requires Ampere+ GPU).

        Returns:
            torch.dtype: Corresponding torch dtype.

        Raises:
            ValueError: If dtype_str is not supported.
        """
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype_str not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype: {dtype_str}. "
                f"Supported dtypes: {list(dtype_mapping.keys())}"
            )

        return dtype_mapping[dtype_str]

    def _determine_autocast_dtype(self) -> torch.dtype:
        """
        Determine the best dtype for autocast based on configured dtype and GPU.
        If torch_dtype is bfloat16, uses bfloat16 for autocast.
        If torch_dtype is float32, selects bfloat16 on Ampere+ GPUs (compute
        capability 8.0+) for better numerical stability, or float16 on older GPUs.

        Returns:
            torch.dtype: The dtype to use for autocast (bfloat16 or float16).
        """
        # If configured dtype is bfloat16, use it for autocast
        if self.torch_dtype == torch.bfloat16:
            log_only_master_process(
                self.logger,
                "info",
                "Using bfloat16 for autocast (configured dtype)",
            )
            return torch.bfloat16

        # For float32, auto-select based on GPU capability
        if not torch.cuda.is_available():
            log_only_master_process(
                self.logger,
                "warning",
                "CUDA not available, autocast will use float16 on CPU",
            )
            return torch.float16

        try:
            device_capability = torch.cuda.get_device_capability()
            # bfloat16 requires compute capability 8.0+ (Ampere architecture)
            if device_capability[0] >= 8:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"GPU compute capability {device_capability[0]}."
                    f"{device_capability[1]} detected - using bfloat16 for autocast",
                )
                return torch.bfloat16
            else:
                log_only_master_process(
                    self.logger,
                    "info",
                    f"GPU compute capability {device_capability[0]}."
                    f"{device_capability[1]} detected - using float16 for autocast",
                )
                return torch.float16
        except Exception as e:
            log_only_master_process(
                self.logger,
                "warning",
                f"Failed to detect GPU capability: {e}, defaulting to float16",
            )
            return torch.float16

    def _initialize_model(self) -> None:
        """Construct the paper's HyenaDNA backbone."""
        self._initialize_hyenadna_model()


    def _initialize_hyenadna_model(self) -> None:
        """Initialize the randomly initialized HyenaDNA trunk and tokenizer."""
        config_source = _resolve_mlm_fallback_config_source(self.config)
        if not config_source:
            raise ValueError("The paper recipe must specify model_options and model_index.")
        self._initialize_hyenadna_bidirectional_random(config_source)
        self.backbone_dim = int(self.backbone_output_dim)
        self.layer_norm = nn.LayerNorm(self.backbone_output_dim)
        self.hyenadna_pad_token_id = getattr(self.config, "pad_token_id", None)
        self.tokenizer = self._create_hyenadna_tokenizer()

    def _create_hyenadna_tokenizer(self) -> AutoTokenizer:
        """Use the checkpoint recipe's single-nucleotide tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
            model_max_length=self.config.tokenizer_max_length,
        )






    def _init_dynamic_chunker_if_enabled(self) -> None:
        """Instantiate DynamicChunker + Dechunker for dynamic-chunker tokenizer.

        Reads ``self.config.tokenizer_type`` and ``self.config.dynamic_chunker.*``
        and wires up the chunker / dechunker submodules when
        ``tokenizer_type == "dynamic_chunker"``. Validates that the active
        backbone supports the chunker path. Emits a WARNING when
        ``use_hybrid_backbone`` is co-enabled (DD-017). For other tokenizer
        types this is a no-op so baseline paths see no behavioral change
        (FR-027).

        Raises:
            ValueError: If ``tokenizer_type`` is not one of
                ``{single_nucleotide, overlapping_kmer, dynamic_chunker}``,
                or if ``tokenizer_type='dynamic_chunker'`` is combined with
                an unsupported backbone.
        """
        tokenizer_type = getattr(self.config, "tokenizer_type", "single_nucleotide")
        valid_types = (
            "single_nucleotide",
            "overlapping_kmer",
            "dynamic_chunker",
        )
        if tokenizer_type not in valid_types:
            raise ValueError(
                f"Unsupported tokenizer_type: {tokenizer_type!r}. "
                f"Supported: {valid_types}."
            )
        if tokenizer_type != "dynamic_chunker":
            return


        if self.use_hybrid_backbone:
            log_only_master_process(
                self.logger,
                "warning",
                "tokenizer_type='dynamic_chunker' + use_hybrid_backbone=true "
                "are both enabled. Hybrid path runs at nt-resolution while "
                "the chunker reduces backbone-token count; the two signals "
                "may interfere. For cleaner ablation set "
                "use_hybrid_backbone=false.",
            )

        chunker_cfg = getattr(self.config, "dynamic_chunker", None)
        if chunker_cfg is None:
            raise ValueError(
                "model.dynamic_chunker config block is required when "
                "tokenizer_type='dynamic_chunker'. See spec FR-024."
            )

        hidden_dim = int(self.base_backbone_output_dim)
        # Class-qualified call so SimpleNamespace stubs in tests can
        # invoke this method without binding ``self``.
        chunkers, dechunkers = IssaSplice._build_chunker_stages(chunker_cfg, hidden_dim)
        self.hierarchical_chunker = HierarchicalChunker(
            chunkers=chunkers,
            dechunkers=dechunkers,
        )
        # Use class-qualified call so tests can invoke this method on a
        # ``SimpleNamespace`` stub without requiring a bound IssaSplice
        # instance (the staticmethod has no implicit-self anyway).
        self._nt_embedding_layer: nn.Embedding = IssaSplice._resolve_nt_embedding_layer(
            self.model
        )

        # Project the optional per-position feature track into the nt-embedding
        # space and add it to the embeddings feeding the chunker + backbone.
        # Zero-initialised so a warm-started model starts as an exact no-op and
        # learns to use the features gradually (minimal trunk perturbation).
        # ``getattr`` fallback keeps SimpleNamespace test stubs (which invoke
        # this method without running ``__init__``) working.
        if getattr(self, "nt_feature_channels", 0) > 0:
            d_model = self._nt_embedding_layer.embedding_dim
            self.feature_proj = nn.Linear(self.nt_feature_channels, d_model)
            nn.init.zeros_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)


        num_stages = self.hierarchical_chunker.num_stages
        per_stage_summary = ", ".join(
            f"stage{i}(rate={chunkers[i].target_chunk_rate:.3f},"
            f"enc={chunkers[i].encoder_type},"
            f"dec={dechunkers[i].dechunker_type})"
            for i in range(num_stages)
        )
        effective_rate = 1.0
        for chunker in chunkers:
            effective_rate *= chunker.target_chunk_rate
        log_only_master_process(
            self.logger,
            "info",
            f"🧬 Dynamic chunker enabled: num_stages={num_stages}, "
            f"effective_chunk_rate={effective_rate:.4f}, "
            f"per_stage=[{per_stage_summary}].",
        )

    @staticmethod
    def _build_chunker_stages(
        chunker_cfg: object,
        hidden_dim: int,
    ) -> tuple[list[DynamicChunker], list[Dechunker]]:
        """Build per-stage chunker / dechunker pairs from a config block.

        Reads ``chunker_cfg.stages`` (a list of stage configs); when the
        list is missing / empty, falls back to building a single-stage
        hierarchy from the flat ``chunker_cfg`` fields (legacy
        backward-compat). Each stage entry inherits any omitted field
        from the flat ``chunker_cfg`` defaults.

        Args:
            chunker_cfg (object): OmegaConf or attribute-style config
                block from ``model.dynamic_chunker``.
            hidden_dim (int): Backbone hidden dimensionality D.

        Returns:
            tuple[list[DynamicChunker], list[Dechunker]]: Paired
                outer-to-inner lists of chunkers and dechunkers.
        """
        stages_cfg = getattr(chunker_cfg, "stages", None)
        # Treat ``null``, missing, or empty list as legacy single-stage.
        if not stages_cfg:
            stages_cfg = [chunker_cfg]
        chunkers: list[DynamicChunker] = []
        dechunkers: list[Dechunker] = []
        for stage_cfg in stages_cfg:
            chunkers.append(
                IssaSplice._build_single_chunker(stage_cfg, chunker_cfg, hidden_dim)
            )
            dechunkers.append(
                IssaSplice._build_single_dechunker(stage_cfg, chunker_cfg, hidden_dim)
            )
        return chunkers, dechunkers

    @staticmethod
    def _stage_value(
        stage_cfg: object,
        flat_cfg: object,
        key: str,
        default: object,
    ) -> object:
        """Return the per-stage override falling back to flat default.

        Stage configs partially specify fields; anything missing inherits
        from the flat ``model.dynamic_chunker`` block; anything still
        missing falls back to the supplied built-in default.

        Args:
            stage_cfg (object): Per-stage config (one entry of
                ``model.dynamic_chunker.stages``). For single-stage this
                is the flat block itself.
            flat_cfg (object): Flat ``model.dynamic_chunker`` block used
                as the inheritance source.
            key (str): Attribute name to read.
            default (object): Built-in fallback when neither stage nor
                flat config defines the key.

        Returns:
            object: The resolved value.
        """
        sentinel = object()
        if stage_cfg is not flat_cfg:
            value = getattr(stage_cfg, key, sentinel)
            if value is not sentinel and value is not None:
                return value
        return getattr(flat_cfg, key, default)

    @staticmethod
    def _build_single_chunker(
        stage_cfg: object,
        flat_cfg: object,
        hidden_dim: int,
    ) -> DynamicChunker:
        """Construct one ``DynamicChunker`` from a stage config.

        Args:
            stage_cfg (object): Per-stage config.
            flat_cfg (object): Flat ``model.dynamic_chunker`` block used
                as inheritance source.
            hidden_dim (int): Backbone hidden dimensionality D.

        Returns:
            DynamicChunker: The instantiated chunker.
        """
        get = IssaSplice._stage_value
        return DynamicChunker(
            hidden_dim=hidden_dim,
            boundary_predictor_kernel_size=int(
                get(stage_cfg, flat_cfg, "boundary_predictor_kernel_size", 7)
            ),
            boundary_predictor_hidden_dim=int(
                get(stage_cfg, flat_cfg, "boundary_predictor_hidden_dim", 64)
            ),
            target_chunk_rate=float(get(stage_cfg, flat_cfg, "target_chunk_rate", 0.2)),
            gumbel_temperature=float(
                get(stage_cfg, flat_cfg, "gumbel_temperature", 1.0)
            ),
            inference_threshold=float(
                get(stage_cfg, flat_cfg, "inference_threshold", 0.5)
            ),
            boundary_decision=str(
                get(stage_cfg, flat_cfg, "boundary_decision", "gumbel")
            ),
            encoder_type=str(get(stage_cfg, flat_cfg, "encoder_type", "conv1d")),
            mamba2_num_layers=int(get(stage_cfg, flat_cfg, "mamba2_num_layers", 2)),
            mamba2_d_state=int(get(stage_cfg, flat_cfg, "mamba2_d_state", 64)),
            mamba2_d_conv=int(get(stage_cfg, flat_cfg, "mamba2_d_conv", 4)),
            mamba2_expand=int(get(stage_cfg, flat_cfg, "mamba2_expand", 2)),
            enable_encoder_residual=bool(
                get(stage_cfg, flat_cfg, "enable_encoder_residual", True)
            ),
            transformer_num_layers=int(
                get(stage_cfg, flat_cfg, "encoder_transformer_num_layers", 0)
            ),
            transformer_num_heads=int(
                get(stage_cfg, flat_cfg, "encoder_transformer_num_heads", 4)
            ),
            transformer_dropout=float(
                get(stage_cfg, flat_cfg, "encoder_transformer_dropout", 0.1)
            ),
            boundary_mode=str(get(stage_cfg, flat_cfg, "boundary_mode", "learned")),
            chunker_style=str(get(stage_cfg, flat_cfg, "chunker_style", "legacy")),
        )

    @staticmethod
    def _build_single_dechunker(
        stage_cfg: object,
        flat_cfg: object,
        hidden_dim: int,
    ) -> Dechunker:
        """Construct one ``Dechunker`` from a stage config.

        Args:
            stage_cfg (object): Per-stage config.
            flat_cfg (object): Flat ``model.dynamic_chunker`` block used
                as inheritance source.
            hidden_dim (int): Backbone hidden dimensionality D.

        Returns:
            Dechunker: The instantiated dechunker.
        """
        get = IssaSplice._stage_value
        return Dechunker(
            dechunker_type=str(get(stage_cfg, flat_cfg, "dechunker_type", "broadcast")),
            hidden_dim=hidden_dim,
            max_within_chunk_offset=int(
                get(stage_cfg, flat_cfg, "max_within_chunk_offset", 32)
            ),
            attention_window_size=int(
                get(stage_cfg, flat_cfg, "attention_window_size", 8)
            ),
            attention_num_heads=int(get(stage_cfg, flat_cfg, "attention_num_heads", 4)),
            attention_dropout=float(get(stage_cfg, flat_cfg, "attention_dropout", 0.0)),
            mamba2_num_layers=int(
                get(stage_cfg, flat_cfg, "dechunker_mamba2_num_layers", 4)
            ),
            mamba2_d_state=int(
                get(stage_cfg, flat_cfg, "dechunker_mamba2_d_state", 64)
            ),
            mamba2_d_conv=int(get(stage_cfg, flat_cfg, "dechunker_mamba2_d_conv", 4)),
            mamba2_expand=int(get(stage_cfg, flat_cfg, "dechunker_mamba2_expand", 2)),
            transformer_num_layers=int(
                get(stage_cfg, flat_cfg, "dechunker_transformer_num_layers", 0)
            ),
            transformer_num_heads=int(
                get(stage_cfg, flat_cfg, "dechunker_transformer_num_heads", 4)
            ),
            transformer_dropout=float(
                get(stage_cfg, flat_cfg, "dechunker_transformer_dropout", 0.1)
            ),
            chunker_style=str(get(stage_cfg, flat_cfg, "chunker_style", "legacy")),
        )

    @staticmethod
    def _resolve_nt_embedding_layer(backbone: nn.Module) -> nn.Embedding:
        """Locate the backbone's nucleotide embedding layer for chunker input.

        DD-015 specifies that the chunker reads the backbone's nt
        embedding directly (weight-shared, lookup is cheap). HuggingFace
        ``PreTrainedModel.get_input_embeddings()`` is the standard hook,
        but the project-local ``HyenaDNAModel`` does not override it —
        instead the embedding is exposed at
        ``backbone.embeddings.word_embeddings``. The project-local
        ``NeoBERTModel`` exposes the embedding at ``backbone.encoder``.
        The HF ``ModernBertModel`` exposes the embedding at
        ``backbone.embeddings.tok_embeddings``. Try the HF API first and
        fall back to the documented backbone-specific paths.

        Args:
            backbone (nn.Module): Backbone model (``self.model``).

        Returns:
            nn.Embedding: The shared nt embedding layer.

        Raises:
            RuntimeError: If no path resolves to an ``nn.Embedding``.
        """
        embedding: nn.Module | None
        try:
            embedding = backbone.get_input_embeddings()
        except (NotImplementedError, AttributeError):
            embedding = None
        if not isinstance(embedding, nn.Embedding):
            embedding = getattr(
                getattr(getattr(backbone, "backbone", None), "embeddings", None),
                "word_embeddings",
                None,
            )
        if not isinstance(embedding, nn.Embedding):
            embedding = getattr(backbone, "encoder", None)
        if not isinstance(embedding, nn.Embedding):
            embedding = getattr(
                getattr(backbone, "embeddings", None),
                "tok_embeddings",
                None,
            )
        if not isinstance(embedding, nn.Embedding):
            raise RuntimeError(
                "Could not locate nt embedding layer for dynamic chunker. "
                "Tried backbone.get_input_embeddings(), "
                "backbone.backbone.embeddings.word_embeddings (HyenaDNA), "
                "backbone.encoder (NeoBERT), and "
                "backbone.embeddings.tok_embeddings (ModernBERT). "
                f"Got: {type(embedding).__name__}"
            )
        return embedding


    def _initialize_hyenadna_bidirectional_random(self, config_source: str) -> None:
        """Initialize bidirectional HyenaDNA with random weights.

        Args:
            config_source (str): HuggingFace model ID or
                local path for architecture config.
        """
        log_only_master_process(
            self.logger,
            "info",
            "🧬 Initializing bidirectional HyenaDNA "
            f"from config: {config_source} "
            "(random weights)",
        )

        config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        # Bidirectional by default (SpliceBoa/-region backbone). The
        # `hyena_bidirectional=false` knob flips HyenaDNA to causal (left-pad
        # only, no non-causal pass) for the bidirectional-vs-causal ablation
        # twin, while keeping the same random-init medium config — so the twin
        # differs in exactly this one knob (not also warm-start vs random init,
        # which the separate AutoModel.from_pretrained causal path would incur).
        config.bidirectional = bool(getattr(self.config, "hyena_bidirectional", True))
        # "padding" keeps the legacy (upstream) operator; "k_rev" enables the
        # anti-causal companion kernel that actually delivers future context.
        config.bidirectional_mode = getattr(
            self.config, "bidirectional_mode", "padding"
        )
        config.use_flash_fft = getattr(self.config, "use_flash_fft", False)
        # Evaluation-only mechanism probes may change the implicit-filter time
        # grid without changing the nucleotide input geometry.  Keep this
        # separate from ``seq_length``: changing ``seq_length`` would also
        # alter tokenisation, the dynamic chunker, dechunker, and hybrid head,
        # so it would no longer isolate the Hyena filter grid.
        filter_max_seq_len = OmegaConf.select(
            self.config, "hyena_filter_max_seq_len", default=None
        )
        if filter_max_seq_len is None:
            filter_max_seq_len = self.config.seq_length
        filter_max_seq_len = int(filter_max_seq_len)
        if filter_max_seq_len < 1:
            raise ValueError(
                "model.hyena_filter_max_seq_len must be positive; got "
                f"{filter_max_seq_len}."
            )
        config.max_seq_len = filter_max_seq_len

        tokenizer_type = getattr(
            self.config,
            "tokenizer_type",
            "single_nucleotide",
        )
        if tokenizer_type == "overlapping_kmer":
            kmer_size = self.config.kmer_size
            kmer_vocab = OverlappingKmerTokenizer(k=kmer_size).vocab_size
            config.vocab_size = kmer_vocab
            log_only_master_process(
                self.logger,
                "info",
                f"🔤 Overriding vocab_size for "
                f"{kmer_size}-mer tokenizer: "
                f"{config.vocab_size}",
            )

        self.model = HyenaDNAModel(config)

        log_only_master_process(
            self.logger,
            "info",
            "✅ Bidirectional HyenaDNA initialized "
            f"(d_model={config.d_model}, "
            f"n_layer={config.n_layer}, "
            f"vocab_size={config.vocab_size}, "
            f"max_seq_len={config.max_seq_len}, "
            f"bidirectional={config.bidirectional}, "
            f"use_flash_fft={config.use_flash_fft})",
        )

        return








    def _create_token_to_onehot_map(self) -> None:
        """Build token-ID to one-hot channel index map.

        Maps each token to a nucleotide channel using the
        first character of the token string:
          A->0, C->1, G->2, T->3, N->4.
        Works for both single-nucleotide and k-mer tokens
        (overlapping k-mer position i starts at nucleotide
        i, so its first character is that nucleotide).
        Special tokens (e.g. [PAD]) map to -1 (masked out).
        """
        vocab = self.tokenizer.get_vocab()
        vocab_size = len(vocab)

        base_to_idx = {
            "A": 0,
            "a": 0,
            "C": 1,
            "c": 1,
            "G": 2,
            "g": 2,
            "T": 3,
            "t": 3,
            "N": 4,
            "n": 4,
        }

        token_to_onehot_idx = torch.full((vocab_size,), -1, dtype=torch.long)

        for token, token_id in vocab.items():
            if token.startswith("["):
                continue
            first_char = token[0]
            if first_char in base_to_idx:
                token_to_onehot_idx[token_id] = base_to_idx[first_char]

        self.register_buffer("token_to_onehot_idx", token_to_onehot_idx)

        return

    def _convert_tokens_to_onehot(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to ``model.onehot_channels``-channel one-hot.

        Channels: A=0, C=1, G=2, T=3, and N=4 when there are five.
        Special / padding tokens produce all-zero vectors, and so does N when
        there are four.

        Args:
            input_ids (torch.Tensor): [B, L] token IDs.

        Returns:
            torch.Tensor: [B, C, L] one-hot tensor, ``C = onehot_channels``.
        """
        B, L = input_ids.shape
        target_dtype = torch.float32
        num_channels = self.onehot_channels

        onehot_idx = self.token_to_onehot_idx[input_ids]

        valid = (onehot_idx >= 0) & (onehot_idx < num_channels)
        onehot_idx = onehot_idx.masked_fill(~valid, 0)

        one_hot = torch.zeros(
            B,
            L,
            num_channels,
            device=input_ids.device,
            dtype=target_dtype,
        )
        one_hot.scatter_(2, onehot_idx.unsqueeze(-1), 1.0)
        one_hot = one_hot * valid.unsqueeze(-1).to(dtype=target_dtype)

        return one_hot.transpose(1, 2)

    def _initialize_hybrid_head(self) -> None:
        """Initialize the hybrid CNN head.

        Reads the ``model.onehot_channels``-channel one-hot nt encoding by
        default, or the backbone's own nt embedding when
        ``model.hybrid_head_input='embedding'`` -- in which case the stem
        widens to ``base_backbone_output_dim`` input channels and the local
        branch shares the trunk's embedding table.

        Raises:
            ValueError: If embedding input is combined with
                ``backbone_onehot_input``, which leaves the backbone's
                embedding table bypassed and therefore trained only through
                this conv branch.
        """
        if self.hybrid_head_input == "embedding" and self.backbone_onehot_input:
            raise ValueError(
                "hybrid_head_input='embedding' with backbone_onehot_input=true "
                "is incoherent: the one-hot front end bypasses the backbone's "
                "embedding table, so the table the conv branch would read is "
                "not the one the trunk uses. Pick one input representation."
            )
        stem_in_channels = (
            int(self.base_backbone_output_dim)
            if self.hybrid_head_input == "embedding"
            else self.onehot_channels
        )
        self.hybrid_head = nn.Sequential(
            nn.Conv1d(stem_in_channels, 64, kernel_size=1),
            Conv1DResidualBlock(
                64, 64, kernel_size=3, dilation=1, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                64, 64, kernel_size=3, dilation=2, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                64, 64, kernel_size=3, dilation=4, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                64, 64, kernel_size=3, dilation=8, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                64, 64, kernel_size=3, dilation=16, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                64, 128, kernel_size=3, dilation=32, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                128, 128, kernel_size=3, dilation=64, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                128, 128, kernel_size=3, dilation=128, dropout=self.dropout_rate
            ),
            Conv1DResidualBlock(
                128, 256, kernel_size=3, dilation=1, dropout=self.dropout_rate
            ),
            nn.GroupNorm(32, self.local_feature_output_dim),
        )

        # Gating Projection Layer
        # Decide which information (hidden_size) from Hyena and which from CNN to keep.
        gate_input_dim = self.base_backbone_output_dim
        if self.backbone_additional_dim is not None:
            gate_input_dim += self.backbone_additional_dim

        self.gate_proj = nn.Linear(gate_input_dim, self.local_feature_output_dim)

        self.gate_proj.apply(self._init_weights)

        return

    def _init_onehot_input_proj(self) -> None:
        """Build the one-hot -> d_model projection that replaces the nt lookup.

        Projects the fixed ``model.onehot_channels``-channel one-hot nt
        encoding to ``d_model``. With the dynamic chunker the result is the nt
        embedding the chunker consumes; without it, the backbone's
        ``inputs_embeds``. The projection is the only learned mapping from nt
        identity to ``d_model``, so the backbone's own ``word_embeddings``
        table is left unused.

        Each input is one channel or all zeros, so column ``c`` of the weight
        is the embedding of channel ``c``: a bias-free projection initialised
        like an embedding row (normal, std 0.02 -- the backbone
        ``initializer_range``) is the token lookup restricted to the nucleotide
        rows, with the same initial distribution. With four channels N, like
        the special tokens, maps to the zero vector.
        """
        self.onehot_input_proj = nn.Linear(
            self.onehot_channels, self.base_backbone_output_dim, bias=False
        )
        nn.init.normal_(self.onehot_input_proj.weight, std=0.02)
        log_only_master_process(
            self.logger,
            "info",
            f"🧬 One-hot nt input enabled: Linear({self.onehot_channels} -> "
            f"{self.base_backbone_output_dim}, bias=False) projects the "
            f"{self.onehot_channels}-channel one-hot nt encoding, bypassing the "
            "learned token embedding lookup.",
        )
        return

    def _project_onehot(self, onehot: torch.Tensor) -> torch.Tensor:
        """Project the ``[B, C, L]`` one-hot to ``[B, L, d_model]``.

        The input is exactly one-hot (all-zero for special tokens), so the
        projection selects weight columns. Computing it as that selection keeps
        the result in the weight's fp32: a matmul would run in bf16 under
        autocast and round to TF32 outside it, while the token lookup this
        replaces returns its fp32 rows unchanged.

        Args:
            onehot (torch.Tensor): ``[B, C, L]`` from
                :meth:`_convert_tokens_to_onehot`.

        Returns:
            torch.Tensor: ``[B, L, d_model]`` nt embedding.
        """
        weight = self.onehot_input_proj.weight  # [d_model, C]
        columns = torch.cat([weight.t(), weight.new_zeros(1, weight.shape[0])])
        channel = torch.where(
            onehot.amax(dim=1) > 0, onehot.argmax(dim=1), weight.shape[1]
        )  # [B, L]; special tokens select the trailing zero row
        return columns[channel]

    def _initialize_fc_layers(self, input_dim: int) -> None:
        """
        Initialize FC layers for position-wise 3-class classification
        (acceptor / donor / neither).
        """
        classification_head_type = self.config.classification_head_type


        # 3-class classification: acceptor / donor / neither.
        output_dim_cls = 3

        self.classification_head = self._make_head(
            head_type=classification_head_type,
            input_dim=input_dim,
            output_dim=output_dim_cls,
        )

        if self.aux_region_enabled:
            self.aux_region_head = self._make_head(
                head_type=classification_head_type,
                input_dim=input_dim,
                output_dim=self.aux_region_num_classes,
            )

        if self.aux_exon_enabled:
            self.aux_exon_head = self._make_head(
                head_type=classification_head_type,
                input_dim=input_dim,
                output_dim=self.aux_exon_num_classes,
            )

        if self.aux_usage_enabled:
            self.aux_usage_head = self._make_head(
                head_type=classification_head_type,
                input_dim=input_dim,
                output_dim=self.aux_usage_num_classes,
            )
        if self.aux_spliced_enabled:
            self.aux_spliced_head = self._make_head(
                head_type=classification_head_type,
                input_dim=input_dim,
                output_dim=self.aux_spliced_num_tissues * 2,
            )
        if self.aux_phylop_enabled:
            self.aux_phylop_head = self._make_head(
                head_type=classification_head_type,
                input_dim=input_dim,
                output_dim=1,
            )

        self.classification_head.apply(self._init_weights)
        if self.aux_region_head is not None:
            self.aux_region_head.apply(self._init_weights)
        if self.aux_exon_head is not None:
            self.aux_exon_head.apply(self._init_weights)
        if self.aux_usage_head is not None:
            self.aux_usage_head.apply(self._init_weights)
        if self.aux_spliced_head is not None:
            self.aux_spliced_head.apply(self._init_weights)
        if self.aux_phylop_head is not None:
            self.aux_phylop_head.apply(self._init_weights)
        log_only_master_process(
            self.logger,
            "info",
            f"Classification head initialized. "
            f"Classification head type: {classification_head_type}, "
            f"output_dim={output_dim_cls}",
        )

    def _make_head(self, head_type: str, input_dim: int, output_dim: int) -> nn.Module:
        """
        Build a prediction head module by type.

        Args:
            head_type (str): Head architecture type. One of
                {"standard", "deep", "attention", "convolution"}.
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.

        Returns:
            nn.Module: Configured head module.

        Raises:
            ValueError: If unsupported head type is provided.
        """
        if head_type == "standard":
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size, output_dim),
            )
        elif head_type == "deep":
            return DeepClassificationHead(
                input_dim=input_dim,
                hidden_dims=[
                    self.hidden_size,
                    self.hidden_size // 2,
                ],
                output_dim=output_dim,
                dropout_rate=self.dropout_rate,
                use_residual=True,
            )
        elif head_type == "attention":
            return AttentionClassificationHead(
                input_dim=input_dim,
                hidden_dim=self.hidden_size,
                output_dim=output_dim,
                dropout_rate=self.dropout_rate,
                num_heads=4,
            )
        elif head_type == "convolution":
            return ConvolutionalHead(
                input_dim=input_dim,
                hidden_dim=self.hidden_size,
                output_dim=output_dim,
                kernel_sizes=[3, 5, 7],
                dropout_rate=self.dropout_rate,
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def _apply_dtype_conversion(self) -> None:
        """
        Apply dtype conversion to the model (without device assignment).

        Autocast is always enabled and handles mixed precision during forward pass.
        Model weights are converted to the configured dtype (float32 or bfloat16).
        """
        log_only_master_process(
            self.logger,
            "info",
            f"Model dtype: {self.weight_dtype}, autocast dtype: {self.autocast_dtype}",
        )

        if self.weight_dtype == "float32":
            # Default dtype, no conversion needed
            log_only_master_process(
                self.logger, "info", f"Using default precision: {self.weight_dtype}"
            )
            return

        try:
            self.model = self.model.to(dtype=self.torch_dtype)
            log_only_master_process(
                self.logger,
                "info",
                f"Backbone model converted to {self.weight_dtype}",
            )

            self.classification_head = self.classification_head.to(
                dtype=self.torch_dtype
            )
            log_only_master_process(
                self.logger,
                "info",
                f"Classification head converted to {self.weight_dtype}",
            )

            if self.aux_region_head is not None:
                self.aux_region_head = self.aux_region_head.to(dtype=self.torch_dtype)
            if self.aux_exon_head is not None:
                self.aux_exon_head = self.aux_exon_head.to(dtype=self.torch_dtype)
            if self.aux_spliced_head is not None:
                self.aux_spliced_head = self.aux_spliced_head.to(dtype=self.torch_dtype)
            if self.aux_phylop_head is not None:
                self.aux_phylop_head = self.aux_phylop_head.to(dtype=self.torch_dtype)

            # Convert layer normalization
            if hasattr(self, "layer_norm"):
                self.layer_norm = self.layer_norm.to(dtype=self.torch_dtype)
                log_only_master_process(
                    self.logger,
                    "info",
                    f"Layer norm converted to {self.weight_dtype}",
                )

            # Convert additional position features
            if self.additional_pos_features is not None:
                self.additional_pos_features = nn.Parameter(
                    self.additional_pos_features.data.to(dtype=self.torch_dtype)
                )
                log_only_master_process(
                    self.logger,
                    "info",
                    f"Additional pos features converted to {self.weight_dtype}",
                )

            # Convert hybrid backbone components
            if self.use_hybrid_backbone:
                self.hybrid_head = self.hybrid_head.to(dtype=self.torch_dtype)
                self.local_norm = self.local_norm.to(dtype=self.torch_dtype)
                self.gate_proj = self.gate_proj.to(dtype=self.torch_dtype)
                log_only_master_process(
                    self.logger,
                    "info",
                    f"Hybrid backbone components converted to {self.weight_dtype}",
                )

            # Check if GPU supports the requested dtype
            if self.weight_dtype == "bfloat16" and torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability()
                if (
                    device_capability[0] < 8
                ):  # bfloat16 requires compute capability 8.0+
                    warning_msg = (
                        f"Warning: GPU (compute capability "
                        f"{device_capability[0]}.{device_capability[1]}) "
                        "may not efficiently support bfloat16. "
                        "Consider using float16 instead."
                    )
                    log_only_master_process(self.logger, "info", warning_msg)
                else:
                    log_only_master_process(
                        self.logger,
                        "info",
                        f"GPU supports {self.weight_dtype} - optimization enabled",
                    )

        except Exception as e:
            log_only_master_process(
                self.logger,
                "info",
                f"Warning: Failed to convert model to {self.weight_dtype}: {e}",
            )
            raise

    def enable_timing_logging(
        self, enable: bool = True, log_interval: int = 100
    ) -> None:
        """
        Enable or disable timing logging.

        Args:
            enable (bool): Whether to enable timing logging.
            log_interval (int): Interval for logging (every N forward passes).
        """
        self.enable_timing_log = enable
        self.timing_stats["log_interval"] = log_interval

        if enable:
            log_only_master_process(
                self.logger,
                "info",
                (
                    f"Timing logging enabled - will log every {log_interval} "
                    "forward passes"
                ),
            )
        else:
            log_only_master_process(self.logger, "info", "Timing logging disabled")

        return

    def save_model_checkpoint(
        self,
        save_path: str,
        additional_info: dict[str, Any] | None = None,
        full_cfg: DictConfig | None = None,
    ) -> None:
        """
        Save model to checkpoint (PEFT-aware: LoRA/IA3 adapters).

        Args:
            save_path (str): Path to save the checkpoint file (.pt/.pth).
            additional_info (dict[str, Any] | None): Extra info to merge into ckpt.
            full_cfg (DictConfig | None): Full training Hydra config. When
                provided, a self-describing ``metadata`` block is embedded
                via ``build_checkpoint_metadata`` (feature 010). When
                ``None``, no metadata block is added — caller is
                responsible for portability.

        Raises:
            Exception: If saving fails.
        """
        # Imported here to avoid circular dependency between model.model
        # and model.checkpoint_metadata at module-load time.
        from model.checkpoint_metadata import build_checkpoint_metadata

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)


            checkpoint = self._create_standard_checkpoint()

            # Add common metadata
            checkpoint["model_config"] = self.config
            if full_cfg is not None:
                checkpoint["metadata"] = build_checkpoint_metadata(full_cfg)
            if additional_info:
                checkpoint.update(additional_info)

            torch.save(checkpoint, save_path)
            log_only_master_process(
                self.logger, "info", f"Model saved to '{save_path}'"
            )
        except Exception as e:
            log_only_master_process(self.logger, "error", f"Error saving model: {e}")
            raise



    def _create_standard_checkpoint(self) -> dict[str, Any]:
        """Create checkpoint dict for standard model."""
        return {
            "is_peft_model": False,
            "model_state_dict": self.model.state_dict(),
        }



    def load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load the complete model state from a paper checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file not found.
            ValueError: If invalid adapter path.
            Exception: If loading fails.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # weights_only=False: these are our own training checkpoints, and
        # since 387d680 they carry RNG state for mid-epoch resume. That
        # includes a numpy ndarray, which the PyTorch 2.6 default of
        # weights_only=True refuses -- so every checkpoint written after
        # that commit was unloadable here while resume kept working,
        # because _load_training_state already passes False.
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)

        self._load_standard_checkpoint(ckpt)
        return


    def _load_standard_checkpoint(self, ckpt: dict) -> None:
        """Load standard (non-PEFT) checkpoint.

        Args:
            ckpt (dict): Checkpoint dictionary.

        Raises:
            ValueError: If state dictionary is not a dictionary.
            Exception: If loading fails.
        """
        # Extract state dict from various formats
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict", ckpt)

        # Normalize keys
        try:
            normalized_state_dict = normalize_checkpoint_keys(state_dict)
        except Exception:
            # Fallback: only strip DDP prefix
            normalized_state_dict = {
                k.removeprefix("module."): v for k, v in state_dict.items()
            }

        # Load matching parameters only
        filtered_state = select_matching_state_dict(self, normalized_state_dict)
        load_result = self.load_state_dict(filtered_state, strict=False)

        # Log loading summary
        self._log_load_summary(filtered_state, load_result)

        return

    def _log_load_summary(
        self, filtered_state: dict, load_result: tuple[list[str], list[str]]
    ) -> None:
        """Log checkpoint loading statistics.

        Args:
            filtered_state (dict): Filtered state dictionary.
            load_result (tuple[list[str], list[str]]): Load result tuple containing
                missing and unexpected keys.
        """
        try:
            num_all = len(list(self.state_dict().keys()))
            num_loaded = len(list(filtered_state.keys()))
            num_missing = len(getattr(load_result, "missing_keys", []))
            num_unexpected = len(getattr(load_result, "unexpected_keys", []))

            log_only_master_process(
                self.logger,
                "info",
                f"Checkpoint load summary: loaded {num_loaded}/{num_all} parameters "
                f"(missing={num_missing}, unexpected={num_unexpected})",
            )
        except Exception:
            log_only_master_process(
                self.logger,
                "info",
                "Error occurred while logging checkpoint load summary",
            )
            raise
        return

    def print_model_info(self) -> None:
        """
        Print detailed model information including dtype and device.
        """
        log_only_master_process(self.logger, "info", "\n=== Model Information ===")
        log_only_master_process(
            self.logger, "info", f"Model Type: {self.backbone_type.lower()}"
        )
        log_only_master_process(self.logger, "info", f"Model Name: {self.model_name}")
        log_only_master_process(self.logger, "info", f"Device: {self.device}")
        log_only_master_process(
            self.logger, "info", f"Weight Dtype: {self.weight_dtype}"
        )
        log_only_master_process(
            self.logger, "info", f"Model Loaded: {self.model is not None}"
        )

        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            log_only_master_process(
                self.logger,
                "info",
                f"Backbone Parameters: {total_params:,} total, "
                f"{trainable_params:,} trainable",
            )
            # Check actual dtype of model parameters
            if hasattr(self.model, "parameters"):
                param_dtypes = set(p.dtype for p in self.model.parameters())
                log_only_master_process(
                    self.logger, "info", f"Actual Parameter Dtypes: {param_dtypes}"
                )

        backbone_param_count = 0
        head_param_count = 0
        for name, param in self.named_parameters():
            if "classification_head" in name:
                head_param_count += param.numel()
            else:
                backbone_param_count += param.numel()

        log_only_master_process(
            self.logger,
            "info",
            f"Backbone params: {backbone_param_count:,}, "
            f"Classification head params: {head_param_count:,}",
        )

        log_only_master_process(
            self.logger, "info", f"Freeze Backbone: {self.freeze_backbone}"
        )
        log_only_master_process(
            self.logger, "info", f"Freeze Policy: {self.freeze_policy}"
        )
        log_only_master_process(self.logger, "info", "=" * 30)

        return

    def to(self, device: torch.device) -> Self:
        """
        Override PyTorch's to() method to handle both backbone and classification head.

        This ensures consistent device/dtype assignment for all model components.
        Supports method chaining by returning self.

        Args:
            device (torch.device): Device to move the model to.

        Returns:
            Self: Returns self to enable method chaining

        Example:
            >>> model = IssaSplice(...)
            >>> model = model.to('cuda').to(torch.float16)  # Method chaining
        """
        # Call parent's to() method for classification_head, chromosome_embedding,
        # and other nn.Module components
        super().to(device)

        self.device = device

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Perform forward pass of the model

        Uses torch.autocast for automatic mixed precision regardless of the
        configured weight dtype. Autocast automatically selects optimal dtypes:
        - MatMul/Conv operations: reduced precision (float16/bfloat16)
        - Reductions/Softmax/LayerNorm: float32 for numerical stability

        Args:
            input_ids (torch.Tensor): Input sequence tensor (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor (batch_size, seq_len).
            features (torch.Tensor | None): Optional per-position feature track
                of shape ``(batch_size, seq_len, k)`` added to the nt
                embeddings; ``None`` disables it.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                ``(cls_logits [B, L, 3], aux)`` where ``aux`` carries
                optional auxiliary head outputs keyed by ``"region"``
                (logits ``[B, L, region_num_classes]``) and ``"exon"``
                (logits ``[B, L, exon_num_classes]``). ``aux`` is empty
                when the corresponding heads are disabled.

        Raises:
            RuntimeError: If model is not initialized.
        """
        # Autocast is always applied for mixed precision training
        return self._forward_with_autocast(input_ids, attention_mask, features=features)

    def _forward_with_autocast(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass with automatic mixed precision (autocast).
        Wraps the forward implementation with torch.autocast for automatic
        dtype selection per layer type. Autocast is always enabled regardless
        of the configured weight dtype.
        Args:
            input_ids (torch.Tensor): Input sequence tensor (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor (batch_size, seq_len).
            features (torch.Tensor | None): Optional per-position feature track
                of shape ``(batch_size, seq_len, k)`` added to the nt
                embeddings; ``None`` disables it.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                Classification logits and a dict of auxiliary head logits,
                all in float32.
        """
        device_type = "cuda" if input_ids.is_cuda else "cpu"

        with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
            cls_head, aux_outputs = self._forward_impl(
                input_ids, attention_mask, features=features
            )

        # Cast outputs to float32 for stable loss computation
        cls_head = cls_head.float()
        aux_outputs = {k: v.float() for k, v in aux_outputs.items()}

        return cls_head, aux_outputs

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run encoder→backbone→dechunker→post-process up to the head input.

        Identical to the body of :meth:`_forward_impl` except that the
        final classification / aux heads are *not* applied. This is the
        entry point that pretraining wrappers (MLM, JEPA) use to plug a
        different head onto the same feature trunk.

        Args:
            input_ids (torch.Tensor): Input token ids of shape
                ``(batch_size, seq_len)``.
            attention_mask (torch.Tensor | None): Optional attention mask
                of shape ``(batch_size, seq_len)``.
            features (torch.Tensor | None): Optional per-position feature track
                of shape ``(batch_size, seq_len, k)`` added to the nt
                embeddings; ``None`` disables it.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                ``(hidden_states, aux_outputs)`` where ``hidden_states``
                has shape ``(batch_size, seq_len, backbone_output_dim)``
                and ``aux_outputs`` carries dynamic-chunker statistics
                (``mean_chunk_rate`` and ``mean_boundary_prob``) used by
                the boundary regularization loss.
        """
        seq_len = input_ids.shape[1]

        # 2. Backbone Forward (dynamic chunker branch or standard branch)
        mean_chunk_rate_per_stage: list[torch.Tensor] = []
        mean_boundary_prob_per_stage: list[torch.Tensor] = []
        hierarchical_outputs = None
        # Set when the nt embedding is computed here rather than inside the
        # trunk, so the hybrid conv branch can reuse that one lookup.
        shared_nt_emb: torch.Tensor | None = None
        # Set when the trunk input is built from the one-hot encoding, so the
        # hybrid conv branch reads the same [B, C, L] tensor.
        onehot: torch.Tensor | None = None
        if self.hierarchical_chunker is not None:
            # Dynamic chunker path (DD-015 ~ DD-019, F5 multi-stage).
            # Step A: nt embedding via backbone's input embedding lookup
            # (DD-015). One cheap embedding lookup, weight shared. The
            # project-local ``HyenaDNAModel`` does not implement the HF
            # ``get_input_embeddings()`` hook and instead exposes the
            # embedding at ``backbone.embeddings.word_embeddings``; the
            # fallback handles that case. With one-hot input the projection
            # of the one-hot encoding takes the lookup's place.
            if self.backbone_onehot_input:
                onehot = self._convert_tokens_to_onehot(input_ids)  # [B, C, L]
                nt_emb = self._project_onehot(onehot)
            else:
                nt_emb = self._nt_embedding_layer(input_ids)
            # Inject the optional per-position feature track here (before the
            # chunker) so it flows through the full long-range trunk. zero-init
            # feature_proj makes this an exact no-op at warm-start.
            if features is not None:
                if self.feature_proj is None:
                    raise ValueError(
                        "features were passed but model.nt_feature_channels == 0 "
                        "(no feature_proj). Set nt_feature_channels in the config."
                    )
                nt_emb = nt_emb + self.feature_proj(features.to(nt_emb.dtype))
            # The N-gram embedding reads nt symbols A, C, G, T, N = 0..4; special
            # tokens, which one-hot to zeros, read as N.
            symbols = None
            encoder_symbols = symbols
            # Step B: Hierarchical chunker -> outer-to-inner encode pass.
            # For single-stage (N=1) this is identical to the legacy
            # path; for N>=2 each stage compresses the previous level's
            # output further.
            shared_nt_emb = nt_emb
            hierarchical_outputs = self.hierarchical_chunker.encode(
                nt_emb, encoder_symbols
            )
            for stage_out in hierarchical_outputs.stage_outputs:
                mean_chunk_rate_per_stage.append(stage_out.mean_chunk_rate)
                mean_boundary_prob_per_stage.append(stage_out.mean_boundary_prob)
            # Step C: backbone with inputs_embeds (DD-016) at the deepest
            # (most compressed) level. Output accuracy is preserved by
            # Dechunker filtering padding chunks out of the broadcast.
            backbone_kwargs: dict[str, Any] = dict(
                input_ids=None,
                inputs_embeds=hierarchical_outputs.deepest_chunk_seq,
                output_hidden_states=True,
                return_dict=True,
            )
            # The self-attention backbone accepts a key-padding mask so
            # zero-padded chunks (K < K_max samples) do not participate in
            # attention. HyenaDNA's long-conv path ignores padding by
            # construction, so no mask is threaded for it. token_mask is
            # True for valid chunks at the deepest stage.
            outputs = self.model(**backbone_kwargs)
        elif self.backbone_onehot_input:
            # One-hot backbone-input ablation: project the one-hot nt
            # encoding into d_model and feed it as inputs_embeds, bypassing
            # the learned token-embedding lookup (mirrors the hybrid conv front
            # end). attention_mask is omitted to match the chunker path; the
            # supported backbones derive a full mask from inputs_embeds.
            onehot = self._convert_tokens_to_onehot(input_ids)  # [B, C, L]
            inputs_embeds = self._project_onehot(onehot)
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
        elif self.hybrid_head_input == "embedding":
            # Weight-tied path: do the embedding lookup once here and hand the
            # same tensor to the trunk (as inputs_embeds) and to the local conv
            # branch below, so both context scales read one learned table.
            # attention_mask is omitted as in the chunker path -- the HyenaDNA
            # long-conv trunk ignores padding by construction, and the
            # splicing input is fixed-length anyway.
            shared_nt_emb = self._nt_embedding_layer(input_ids)
            outputs = self.model(
                input_ids=None,
                inputs_embeds=shared_nt_emb,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            # Standard backbone path.
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=input_ids.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

        # 3. Extract Hidden States
        all_hidden_states = outputs["hidden_states"]
        hidden_states = all_hidden_states[self.hidden_state_layer_index]

        # 3-A. Dechunker: restore chunk-level [B, K_inner, D] -> nt-level
        # [B, L, D]. For multi-stage hierarchies, decode runs inner-to-
        # outer with each stage's encoder residual added back at its
        # corresponding level (separation-of-two-streams). For
        # single-stage this collapses to the legacy DD-012 / DD-012a path.
        if self.hierarchical_chunker is not None and hierarchical_outputs is not None:
            hidden_states = self.hierarchical_chunker.decode(
                deepest_hidden=hidden_states,
                stage_outputs=hierarchical_outputs.stage_outputs,
                level_inputs=hierarchical_outputs.level_inputs,
                nt_output_length=seq_len,
                level_masks=hierarchical_outputs.level_masks,
            )

        # 4. Post-Process Hidden States (LayerNorm)
        # HyenaDNA / NeoBERT (feature 009) / ModernBERT (feature 011) all use
        # the same post-process path: no CLS prepend, learned LayerNorm adapter.
        hidden_states = self.layer_norm(hidden_states)  # (B, L, BackboneDim)

        # 5. Add Additional Position Features
        if self.additional_pos_features is not None:
            actual_len = hidden_states.shape[1]
            pos_feat = self.additional_pos_features
            if pos_feat.shape[1] < actual_len:
                raise ValueError(
                    f"additional_pos_features length "
                    f"({pos_feat.shape[1]}) is shorter than the "
                    f"actual sequence length ({actual_len}); "
                    "increase the position feature window."
                )
            if pos_feat.shape[1] > actual_len:
                start = (pos_feat.shape[1] - actual_len) // 2
                pos_feat = pos_feat[:, start : start + actual_len, :]
            additional_features = pos_feat.expand(hidden_states.shape[0], -1, -1)
            hidden_states = torch.cat(
                [hidden_states, additional_features],
                dim=2,
            )

        # 6. Apply Hybrid Backbone
        if self.use_hybrid_backbone:
            if self.hybrid_head_input == "embedding":
                # Same tensor the trunk consumed: [B, L, D] -> [B, D, L].
                nt_emb_local = (
                    shared_nt_emb
                    if shared_nt_emb is not None
                    else self._nt_embedding_layer(input_ids)
                )
                local_input = nt_emb_local.transpose(1, 2)
            else:
                local_input = (
                    onehot
                    if onehot is not None
                    else self._convert_tokens_to_onehot(input_ids)
                )
            local_feat = self.hybrid_head(
                local_input.to(dtype=self.hybrid_head[0].weight.dtype)
            )  # [B, local_feature_output_dim, L]
            local_feat = local_feat.transpose(1, 2)  # [B, L, local_feature_output_dim]
            local_feat = self.local_norm(local_feat)
            # hidden_states: [B, L, Hyena_Dim] -> gate: [B, L, local_feature_output_dim]
            gate = torch.sigmoid(self.gate_proj(hidden_states))
            gated_local_feat = local_feat * gate

            # Original Hyena information + validated (Gated) Local information
            hidden_states = torch.cat([hidden_states, gated_local_feat], dim=-1)
            # [B, L, hyena_dim + local_feature_output_dim]

        # Dynamic chunker: surface stacked per-stage F (hard mask mean)
        # and G (soft probability mean) tensors for the trainer's
        # boundary regularization loss. Shape is ``[num_stages]`` so
        # single-stage (the legacy default) yields a 1-element tensor —
        # the trainer's loss handles arbitrary lengths and applies the
        # per-stage ``target_chunk_rate``. ``l1`` consumes F only;
        # ``hnet_ratio`` (DD-020) consumes both. Baseline paths leave
        # the keys absent and the trainer treats boundary_loss = 0.
        aux_outputs: dict[str, torch.Tensor] = {}
        if mean_chunk_rate_per_stage:
            aux_outputs["mean_chunk_rate"] = torch.stack(mean_chunk_rate_per_stage)
            aux_outputs["mean_boundary_prob"] = torch.stack(
                mean_boundary_prob_per_stage
            )

        return hidden_states, aux_outputs

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Core supervised forward pass: trunk features → classification heads.

        Delegates the encoder/dechunker/post-process trunk to
        :meth:`forward_features` so pretraining wrappers can reuse the
        same trunk with a different head.

        Args:
            input_ids (torch.Tensor): Input sequence tensor (batch_size, seq_len).
            attention_mask (torch.Tensor | None): Attention mask tensor
                (batch_size, seq_len).
            features (torch.Tensor | None): Optional per-position feature track
                of shape ``(batch_size, seq_len, k)`` added to the nt
                embeddings; ``None`` disables it.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                ``(cls_logits, aux)`` where ``aux`` may carry ``"region"``
                and ``"exon"`` per-position logits when the corresponding
                heads are enabled, plus chunker statistics propagated
                from :meth:`forward_features`.
        """
        batch_size = input_ids.shape[0]

        if self.enable_timing_log:
            backbone_start = time.perf_counter()

        hidden_states, aux_outputs = self.forward_features(
            input_ids, attention_mask, features=features
        )

        if self.enable_timing_log:
            backbone_end = time.perf_counter()
            backbone_time = backbone_end - backbone_start
            head_start = time.perf_counter()

        # 8. Heads
        cls_head = self.classification_head(hidden_states)
        cls_head = cls_head.view(batch_size, -1, 3)

        if self.aux_region_head is not None:
            region_logits = self.aux_region_head(hidden_states)
            aux_outputs["region"] = region_logits.view(
                batch_size, -1, self.aux_region_num_classes
            )
        if self.aux_exon_head is not None:
            exon_logits = self.aux_exon_head(hidden_states)
            aux_outputs["exon"] = exon_logits.view(
                batch_size, -1, self.aux_exon_num_classes
            )
        if self.aux_usage_head is not None:
            usage_logits = self.aux_usage_head(hidden_states)
            aux_outputs["usage"] = usage_logits.view(
                batch_size, -1, self.aux_usage_num_classes
            )
        if self.aux_spliced_head is not None:
            spliced_logits = self.aux_spliced_head(hidden_states)
            aux_outputs["spliced"] = spliced_logits.view(
                batch_size, -1, self.aux_spliced_num_tissues, 2
            )
        if self.aux_phylop_head is not None:
            phylop_pred = self.aux_phylop_head(hidden_states)
            aux_outputs["phylop"] = phylop_pred.view(batch_size, -1)

        if self.enable_timing_log:
            head_end = time.perf_counter()
            head_time = head_end - head_start
            self._update_timing_stats(backbone_time, head_time)

        return cls_head, aux_outputs

    def _update_timing_stats(self, backbone_time: float, head_time: float) -> None:
        """
        Update timing statistics and log periodically.

        Args:
            backbone_time (float): Time spent in backbone forward pass.
            head_time (float): Time spent in classification head forward pass.
        """
        # Add to timing statistics
        self.timing_stats["backbone_times"].append(backbone_time)
        self.timing_stats["classification_head_times"].append(head_time)
        self.timing_stats["total_forward_count"] += 1

        # Log periodically
        if (
            self.timing_stats["total_forward_count"] % self.timing_stats["log_interval"]
            == 0
        ):
            self._log_timing_stats()

    def _log_timing_stats(self) -> None:
        """
        Log timing statistics showing the time distribution between backbone
        and classification head.
        """
        backbone_times = self.timing_stats["backbone_times"]
        head_times = self.timing_stats["classification_head_times"]

        if not backbone_times or not head_times:
            return

        # Calculate statistics
        total_backbone_time = sum(backbone_times)
        total_head_time = sum(head_times)
        total_time = total_backbone_time + total_head_time

        avg_backbone_time = total_backbone_time / len(backbone_times)
        avg_head_time = total_head_time / len(head_times)
        avg_total_time = avg_backbone_time + avg_head_time

        # Calculate percentages
        backbone_percent = (
            (total_backbone_time / total_time) * 100 if total_time > 0 else 0
        )
        head_percent = (total_head_time / total_time) * 100 if total_time > 0 else 0

        # Recent performance (last 10 forward passes)
        recent_count = min(10, len(backbone_times))
        recent_backbone_avg = sum(backbone_times[-recent_count:]) / recent_count
        recent_head_avg = sum(head_times[-recent_count:]) / recent_count
        recent_total_avg = recent_backbone_avg + recent_head_avg
        recent_backbone_percent = (
            (recent_backbone_avg / recent_total_avg) * 100
            if recent_total_avg > 0
            else 0
        )
        recent_head_percent = (
            (recent_head_avg / recent_total_avg) * 100 if recent_total_avg > 0 else 0
        )

        # Log timing information
        stats_header = (
            f"\n=== Forward Pass Timing Statistics "
            f"(After {self.timing_stats['total_forward_count']} forward passes) ==="
        )
        log_only_master_process(self.logger, "info", stats_header)
        log_only_master_process(self.logger, "info", f"Overall Average Times:")
        log_only_master_process(
            self.logger,
            "info",
            (
                f"  Backbone Model:        {avg_backbone_time:.4f}s "
                f"({backbone_percent:.1f}%)"
            ),
        )
        log_only_master_process(
            self.logger,
            "info",
            (f"  Classification Head:   {avg_head_time:.4f}s ({head_percent:.1f}%)"),
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Total Forward Pass:    {avg_total_time:.4f}s (100.0%)",
        )

        log_only_master_process(
            self.logger,
            "info",
            f"Recent Performance (last {recent_count} forward passes):",
        )
        log_only_master_process(
            self.logger,
            "info",
            (
                f"  Backbone Model:        {recent_backbone_avg:.4f}s "
                f"({recent_backbone_percent:.1f}%)"
            ),
        )
        log_only_master_process(
            self.logger,
            "info",
            (
                f"  Classification Head:   {recent_head_avg:.4f}s "
                f"({recent_head_percent:.1f}%)"
            ),
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Total Forward Pass:    {recent_total_avg:.4f}s (100.0%)",
        )

        # Calculate speed ratios
        if avg_head_time > 0:
            speed_ratio = avg_backbone_time / avg_head_time
            log_only_master_process(
                self.logger,
                "info",
                (
                    f"Speed Ratio: Backbone is {speed_ratio:.1f}x slower than "
                    "Classification Head"
                ),
            )

        log_only_master_process(self.logger, "info", "=" * 70)

    def get_timing_summary(self) -> dict[str, Any]:
        """
        Get a summary of timing statistics.

        Returns:
            dict[str, Any]: Dictionary containing timing summary.
        """
        backbone_times = self.timing_stats["backbone_times"]
        head_times = self.timing_stats["classification_head_times"]

        if not backbone_times or not head_times:
            return {"error": "No timing data available"}

        total_backbone_time = sum(backbone_times)
        total_head_time = sum(head_times)
        total_time = total_backbone_time + total_head_time

        return {
            "total_forward_passes": self.timing_stats["total_forward_count"],
            "average_times": {
                "backbone": total_backbone_time / len(backbone_times),
                "classification_head": total_head_time / len(head_times),
                "total": (total_backbone_time + total_head_time) / len(backbone_times),
            },
            "time_percentages": {
                "backbone": (total_backbone_time / total_time) * 100
                if total_time > 0
                else 0,
                "classification_head": (total_head_time / total_time) * 100
                if total_time > 0
                else 0,
            },
            "total_cumulative_time": {
                "backbone": total_backbone_time,
                "classification_head": total_head_time,
                "total": total_time,
            },
        }

    def apply_freeze_policy(
        self, policy: str, partial_layers: int | None = None
    ) -> None:
        """
        Apply freezing policy to the backbone model.

        Args:
            policy (str): Freezing policy ('full', 'partial', 'none').
            partial_layers (int | None): Number of layers to freeze for partial policy.
        """
        if not self.freeze_backbone:
            log_only_master_process(
                self.logger, "info", f"self.freeze_backbone is False, skipping freeze"
            )
            return

        if policy == "full":
            self._freeze_backbone_full()
        elif policy == "partial":
            if partial_layers is None:
                log_only_master_process(
                    self.logger,
                    "info",
                    "Warning: partial_layers not specified for partial freeze, using "
                    + "full freeze",
                )
                self._freeze_backbone_full()
            else:
                self._freeze_backbone_partial(partial_layers)
        elif policy == "none":
            self.unfreeze_backbone()
        else:
            log_only_master_process(
                self.logger,
                "info",
                f"Warning: Unknown freeze policy '{policy}', skipping freeze",
            )

    def print_freeze_status(self) -> None:
        """
        Print detailed freeze status information.
        """
        status = self.get_freeze_status()

        if "error" in status:
            log_only_master_process(self.logger, "info", f"Error: {status['error']}")
            return

        log_only_master_process(self.logger, "info", "\n=== Model Freeze Status ===")
        log_only_master_process(
            self.logger, "info", f"Freeze Policy: {status['freeze_policy']}"
        )
        log_only_master_process(
            self.logger, "info", f"Freeze Backbone: {status['freeze_backbone']}"
        )

        log_only_master_process(self.logger, "info", f"\nBackbone Model:")
        log_only_master_process(
            self.logger,
            "info",
            f"  Total Parameters: {status['backbone']['total_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Frozen Parameters: {status['backbone']['frozen_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Trainable Parameters: {status['backbone']['trainable_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Freeze Ratio: {status['backbone']['freeze_ratio']:.1%}",
        )

        log_only_master_process(self.logger, "info", f"\nClassification Head:")
        log_only_master_process(
            self.logger,
            "info",
            f"  Total Parameters: {status['classification_head']['total_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Frozen Parameters: {status['classification_head']['frozen_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Trainable Parameters: "
            f"{status['classification_head']['trainable_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Freeze Ratio: {status['classification_head']['freeze_ratio']:.1%}",
        )

        log_only_master_process(self.logger, "info", f"\nOverall:")
        log_only_master_process(
            self.logger,
            "info",
            f"  Total Parameters: {status['overall']['total_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Frozen Parameters: {status['overall']['frozen_params']}",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"  Trainable Parameters: {status['overall']['trainable_params']}",
        )
        freeze_ratio = (
            status["overall"]["frozen_params"] / status["overall"]["total_params"]
        )
        log_only_master_process(
            self.logger, "info", f"  Overall Freeze Ratio: {freeze_ratio:.1%}"
        )

    def get_freeze_status(self) -> dict[str, Any]:
        """
        Get detailed information about freeze status.

        Returns:
            dict[str, Any]: Dictionary containing freeze status information.
        """
        if self.model is None:
            return {"error": "Model not initialized"}

        backbone_params = list(self.model.parameters())
        head_params = (
            list(self.classification_head.parameters())
            if hasattr(self, "classification_head")
            else []
        )

        backbone_frozen = sum(1 for p in backbone_params if not p.requires_grad)
        backbone_trainable = sum(1 for p in backbone_params if p.requires_grad)
        head_frozen = sum(1 for p in head_params if not p.requires_grad)
        head_trainable = sum(1 for p in head_params if p.requires_grad)

        total_backbone_params = len(backbone_params)
        total_head_params = len(head_params)

        return {
            "backbone": {
                "total_params": total_backbone_params,
                "frozen_params": backbone_frozen,
                "trainable_params": backbone_trainable,
                "freeze_ratio": backbone_frozen / total_backbone_params
                if total_backbone_params > 0
                else 0,
            },
            "classification_head": {
                "total_params": total_head_params,
                "frozen_params": head_frozen,
                "trainable_params": head_trainable,
                "freeze_ratio": head_frozen / total_head_params
                if total_head_params > 0
                else 0,
            },
            "overall": {
                "total_params": total_backbone_params + total_head_params,
                "frozen_params": backbone_frozen + head_frozen,
                "trainable_params": backbone_trainable + head_trainable,
            },
            "freeze_policy": self.freeze_policy,
            "freeze_backbone": self.freeze_backbone,
        }

    def toggle_freeze_backbone(self) -> None:
        """
        Toggle freeze state of the backbone model.
        """
        if self.model is None:
            log_only_master_process(
                self.logger, "info", "Warning: Backbone model is not initialized."
            )
            return

        # Check current state
        frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.model.parameters())

        if frozen_params == total_params:
            # Currently fully frozen, unfreeze
            self.unfreeze_backbone()
            self.freeze_backbone = False
        else:
            # Currently not fully frozen, freeze
            self._freeze_backbone_full()
            self.freeze_backbone = True

    def _freeze_backbone_full(self) -> None:
        """
        Freeze all parameters in the backbone model.
        """
        if self.model is None:
            log_only_master_process(
                self.logger, "info", "Warning: Backbone model is not initialized."
            )
            return

        for param in self.model.parameters():
            param.requires_grad = False

        log_only_master_process(
            self.logger,
            "info",
            "Backbone model fully frozen - all parameters set to requires_grad=False",
        )

    def _freeze_backbone_partial(self, num_layers: int) -> None:
        """Freeze the bottom ``num_layers`` layers of the backbone.

        Resolves the per-backbone transformer block list and freezes
        indices ``[0, num_layers)`` while leaving the remaining blocks
        trainable. Currently supports HyenaDNA (``backbone.layers``) and
        NeoBERT (``transformer_encoder``); other backbones get a no-op
        warning since their layer layout is not yet wired in.

        Args:
            num_layers (int): Number of bottom layers to freeze.

        Raises:
            ValueError: ``num_layers`` exceeds the backbone's transformer
                block count (FR-011 / contracts C-008 message).
        """
        if self.model is None:
            log_only_master_process(
                self.logger, "info", "Warning: Backbone model is not initialized."
            )
            return

        layers = self._get_backbone_layers()
        if layers is None:
            log_only_master_process(
                self.logger,
                "info",
                "Warning: backbone has no recognised transformer block list; "
                "partial freeze skipped.",
            )
            return

        total_layers = len(layers)
        if num_layers > total_layers:
            raise ValueError(
                f"partial_freeze_layers ({num_layers}) > num_hidden_layers "
                f"({total_layers})."
            )

        frozen_count = 0
        total_params = 0
        for i, layer in enumerate(layers):
            requires_grad = i >= num_layers
            for param in layer.parameters():
                param.requires_grad = requires_grad
                if not requires_grad:
                    frozen_count += 1
                total_params += 1

        log_only_master_process(
            self.logger,
            "info",
            f"Backbone partially frozen - {frozen_count}/{total_params} "
            "parameters frozen",
        )
        log_only_master_process(
            self.logger,
            "info",
            f"Frozen layers: 0-{num_layers - 1}, Trainable layers: {num_layers}+",
        )

        return

    def _get_backbone_layers(self) -> "nn.ModuleList | list[nn.Module] | None":
        """Return the backbone's transformer block list, or None if unknown.

        - HyenaDNA: ``self.model.backbone.layers``
        - NeoBERT: ``self.model.transformer_encoder``
        - ModernBERT: ``self.model.layers`` (HF ``ModernBertModel``)
        - Other backbones currently fall through to None.

        Returns:
            nn.ModuleList | list[nn.Module] | None: The transformer block
                list when the backbone exposes one, else None.
        """
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layers"):
            return self.model.backbone.layers
        if hasattr(self.model, "transformer_encoder"):
            return self.model.transformer_encoder
        backbone_lc = self.backbone_type.lower()
        if backbone_lc == "modernbert" and hasattr(self.model, "layers"):
            return self.model.layers
        return None

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters in the backbone model.
        """
        if self.model is None:
            log_only_master_process(
                self.logger, "info", "Warning: Backbone model is not initialized."
            )
            return

        for param in self.model.parameters():
            param.requires_grad = True

        log_only_master_process(
            self.logger,
            "info",
            "Backbone model unfrozen - all parameters set to requires_grad=True",
        )

        return
