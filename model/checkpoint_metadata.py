"""Self-describing checkpoint metadata schema, registry, and helpers.

This module defines the data contract for the ``"metadata"`` block that
every ``.pt`` checkpoint produced by the training pipeline embeds, plus
helpers for building the block (write side), loading and validating it
(evaluation side), and reconstructing a model class instance from the
embedded training-config snapshot.

The schema is *Hybrid*: top-level routing fields (``schema_version``,
``model_class_id``, ``created_at``, ...) live alongside a single
``training_config_snapshot`` dict that mirrors the Hydra/OmegaConf
training configuration. The snapshot keeps interpolation strings
unresolved (``${a.b}`` form) — same as the sidecar ``train_config.yaml``
written via ``OmegaConf.save`` — so dangling/recursive interpolations do
not break checkpoint creation; they re-resolve in the evaluation cfg
context on load. See `.specify/specs/010-checkpoint-metadata/
data-model.md` for the full schema reference.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from utils.git import get_git_sha

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Schema version & supported set ──────────────────────────────────────

CHECKPOINT_METADATA_SCHEMA_VERSION: int = 1
"""Writer-side schema version. Bump on breaking changes only."""

SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
"""Reader-side accepted schema versions. Mismatch raises
``IncompatibleCheckpointError`` (FR-008)."""


# ── Override classification ─────────────────────────────────────────────

SAFE_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "evaluation.batch_size",
        "evaluation.thresholds",
        "evaluation.test_data_path",
        "evaluation.test_h5_path",
        "evaluation.probability_mode",
        "evaluation.num_workers",
        "evaluation.ignore_embed_metadata",
    }
)
"""Dot-path keys that may be overridden at evaluation time without
invalidating weight semantics. They live entirely in the evaluation Hydra
config group, so they have no counterpart inside the training-config
snapshot embedded in the metadata."""

UNSAFE_OVERRIDE_PREFIXES: tuple[str, ...] = (
    "model_class_id",
    "model.",
    "data.",
    "tokenizer.",
)
"""Dot-path prefixes whose override at evaluation time would silently
re-shape weights or change tokenization. ``validate_user_overrides``
rejects any user override matching one of these (FR-011)."""


# ── Model class registry (lazy) ─────────────────────────────────────────

MODEL_CLASS_IDS: tuple[str, ...] = ("IssaSplice",)
"""All ``model_class_id`` values known to this codebase. The matching
class is resolved lazily by ``get_model_class`` to avoid a circular
import between ``model.checkpoint_metadata`` and ``model.model``."""

_MODEL_CLASS_CACHE: dict[str, type] = {}


def get_model_class(model_class_id: str) -> type:
    """Resolve *model_class_id* to a concrete ``nn.Module`` subclass.

    Lazy import is used so this module can be imported from ``model.model``
    without circularity. Result is cached on first lookup.

    Args:
        model_class_id (str): Registry key. Must appear in
            :data:`MODEL_CLASS_IDS`.

    Returns:
        type: Model class object (``IssaSplice`` for now).

    Raises:
        UnknownModelClassIdError: When *model_class_id* is unknown.
    """
    if model_class_id in _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[model_class_id]

    if model_class_id == "IssaSplice":
        from model.model import IssaSplice

        _MODEL_CLASS_CACHE[model_class_id] = IssaSplice
        return IssaSplice

    raise UnknownModelClassIdError(
        f"Unknown model_class_id: {model_class_id!r}. "
        f"Known candidates: {MODEL_CLASS_IDS}"
    )


# ── Exceptions ──────────────────────────────────────────────────────────


class CheckpointMetadataError(Exception):
    """Base exception for all checkpoint-metadata-related load errors."""


class MalformedCheckpointMetadataError(CheckpointMetadataError):
    """A ``"metadata"`` key exists but its content is structurally invalid.

    Raised when required fields are missing, types are wrong, or the
    dataclass reconstruction fails. Suppressed and converted to ``None``
    only when the caller passes ``ignore_embed_metadata=True`` (FR-016).
    """


class IncompatibleCheckpointError(CheckpointMetadataError):
    """Embedded ``schema_version`` not in :data:`SUPPORTED_SCHEMA_VERSIONS` (FR-008)."""


class UnknownModelClassIdError(CheckpointMetadataError):
    """Embedded ``model_class_id`` is not a key of
    :data:`MODEL_CLASS_IDS` (FR-010)."""


class UnsafeOverrideError(CheckpointMetadataError):
    """User attempted to override a weight-affecting field at evaluation
    time (matches :data:`UNSAFE_OVERRIDE_PREFIXES`, FR-011)."""


# ── Dataclass schema ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointMetadata:
    """Self-describing metadata block embedded inside ``.pt`` files.

    Hybrid schema: top-level routing fields stay independent of cfg
    structure, while ``training_config_snapshot`` carries the entire
    resolved Hydra training config (bit-identical to the sidecar
    ``train_config.yaml``).

    Attributes:
        schema_version (int): Block schema version. Must satisfy
            ``schema_version in SUPPORTED_SCHEMA_VERSIONS`` for evaluation
            to accept the checkpoint.
        model_class_id (str): Registry key naming the model class to
            instantiate (currently ``"IssaSplice"``).
        created_at (str): ISO-8601 timestamp recorded at save time.
        training_config_snapshot (dict[str, Any]): Result of
            ``OmegaConf.to_container(cfg, resolve=True)`` over the full
            training config. Contains every ``cfg.model.*`` (model
            architecture and tokenizer config), ``cfg.data.*``
            (``context_length``, ``sequence_layout.*``, dataset format),
            ``cfg.training.*``, and ``cfg.logging.*`` key from the run.
        code_git_sha (str | None): Git SHA at save time, suffixed with
            ``"-dirty"`` when uncommitted changes were present. ``None``
            when extraction failed (FR-015).
        training_run_id (str | None): MLflow run id, or ``None`` when
            tracking was disabled.
        framework_versions (dict[str, str]): ``{"torch": "...", "cuda":
            "...", "python": "..."}``-style diagnostic block. Empty dict
            allowed when collection fails.
    """

    schema_version: int
    model_class_id: str
    created_at: str
    training_config_snapshot: dict[str, Any]
    code_git_sha: str | None = None
    training_run_id: str | None = None
    framework_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field types eagerly so malformed inputs fail fast."""
        if not isinstance(self.schema_version, int):
            raise TypeError(
                f"schema_version must be int, got {type(self.schema_version).__name__}"
            )
        if not isinstance(self.model_class_id, str) or not self.model_class_id:
            raise ValueError("model_class_id must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty ISO-8601 string")
        if not isinstance(self.training_config_snapshot, dict):
            raise TypeError(
                "training_config_snapshot must be dict, "
                f"got {type(self.training_config_snapshot).__name__}"
            )


# ── Build side helpers ──────────────────────────────────────────────────


def _collect_framework_versions() -> dict[str, str]:
    """Capture torch / cuda / python versions for diagnostics.

    Returns:
        dict[str, str]: Mapping of framework name to version string.
            Returns an empty dict on any failure (best-effort only).
    """
    versions: dict[str, str] = {}
    try:
        versions["python"] = sys.version.split()[0]
        versions["platform"] = platform.platform()
        # Cast to plain str — torch exposes TorchVersion objects whose
        # pickling is rejected under torch.load(weights_only=True) in
        # PyTorch >= 2.6.
        versions["torch"] = str(torch.__version__)
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version is not None:
            versions["cuda"] = str(cuda_version)
    except (AttributeError, OSError):
        pass
    return versions


def build_checkpoint_metadata(
    cfg: DictConfig,
    *,
    run_name: str | None = None,
    training_start_time: str | None = None,
    mlflow_run_id: str | None = None,
    model_class_id: str = "IssaSplice",
) -> dict[str, Any]:
    """Build the embed-metadata dict for a ``.pt`` checkpoint.

    Implementation is essentially a one-liner cfg snapshot plus
    bookkeeping fields. Caller (``train.py`` / ``model.py``) inserts the
    returned dict under the ``"metadata"`` key of the state dict before
    a single ``torch.save`` call (FR-014).

    Args:
        cfg (DictConfig): Training config. Snapshotted via
            ``OmegaConf.to_container(cfg, resolve=False)`` to preserve
            interpolation strings (``${a.b}``) verbatim — this matches
            the behavior of sidecar ``train_config.yaml`` written via
            ``OmegaConf.save`` and prevents dangling-interpolation errors
            (e.g., ``${model.model_index}`` referring to a key absent
            at save time) from breaking checkpoint creation. flank/target
            lengths, tokenizer settings, model architecture, and all
            other ``cfg.*`` keys are auto-preserved with no per-subtree
            mapping. Interpolations re-resolve on the evaluation side
            when ``OmegaConf.create(snapshot)`` is followed by attribute
            access in the evaluation cfg context.
        run_name (str | None): Training run name (currently unused inside
            the metadata block, accepted for caller convenience and future
            extension).
        training_start_time (str | None): Training start ISO-8601 string
            (currently unused, accepted for parity with sidecar metadata).
        mlflow_run_id (str | None): MLflow run id when tracking is on.
        model_class_id (str): Registry key for the model class. Defaults
            to ``"IssaSplice"``.

    Returns:
        dict[str, Any]: ``dataclasses.asdict(CheckpointMetadata(...))``
        result, ready to embed verbatim into a ``torch.save`` payload.
    """
    # run_name and training_start_time are accepted for forward-compat /
    # caller-side parity; they are intentionally not yet stored as
    # dedicated fields because the snapshot already preserves run_name
    # via cfg.logging.* and training start time is recorded in the sidecar
    # metadata yaml separately.
    del run_name, training_start_time  # noqa: F841

    snapshot = _serialize_cfg_to_dict(cfg)

    metadata = CheckpointMetadata(
        schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
        model_class_id=model_class_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        training_config_snapshot=snapshot,
        code_git_sha=get_git_sha(),
        training_run_id=mlflow_run_id,
        framework_versions=_collect_framework_versions(),
    )
    return asdict(metadata)


def _serialize_cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Serialize a Hydra cfg to a plain dict for embedding.

    Strategy:

    1. Try ``OmegaConf.to_container(cfg, resolve=False)`` — this matches
       what ``OmegaConf.save`` writes to ``train_config.yaml`` (the
       sidecar) and tolerates dangling interpolations because nothing is
       resolved at save time.
    2. If that still fails (extremely unusual — only on container
       structural issues), retry with ``throw_on_missing=False`` to
       skip ``MISSING`` markers.
    3. Validate the result is a top-level dict.

    Args:
        cfg (DictConfig): Training config to serialize.

    Returns:
        dict[str, Any]: Plain dict suitable for ``torch.save`` and
        ``OmegaConf.create`` round-trip.

    Raises:
        TypeError: When *cfg* does not serialize to a top-level dict.
    """
    try:
        snapshot = OmegaConf.to_container(cfg, resolve=False)
    except OmegaConfBaseException as exc:
        logger.warning(
            "OmegaConf.to_container(resolve=False) raised %s — retrying "
            "with throw_on_missing=False as a defensive fallback.",
            type(exc).__name__,
        )
        snapshot = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)

    if not isinstance(snapshot, dict):
        raise TypeError(
            f"cfg must serialize to a top-level dict, got {type(snapshot).__name__}"
        )
    return snapshot


# ── Load side helpers ───────────────────────────────────────────────────


_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "model_class_id",
    "created_at",
    "training_config_snapshot",
)


def _validate_metadata_dict(meta_dict: dict[str, Any]) -> CheckpointMetadata:
    """Validate a raw metadata dict and return a typed CheckpointMetadata.

    Args:
        meta_dict (dict[str, Any]): Raw dict pulled from the ``.pt``'s
            ``"metadata"`` key.

    Returns:
        CheckpointMetadata: Typed dataclass instance.

    Raises:
        IncompatibleCheckpointError: When schema_version is unsupported.
        MalformedCheckpointMetadataError: When required fields are
            missing or types are wrong.
        UnknownModelClassIdError: When model_class_id is unknown.
    """
    schema_version = meta_dict.get("schema_version")
    if not isinstance(schema_version, int):
        raise MalformedCheckpointMetadataError(
            f"schema_version missing or not an int: got {type(schema_version).__name__}"
        )
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise IncompatibleCheckpointError(
            f"Unsupported schema_version {schema_version}. "
            f"This evaluation code supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}. "
            "Upgrade the codebase or migrate the checkpoint."
        )

    missing = [k for k in _REQUIRED_FIELDS if k not in meta_dict]
    if missing:
        raise MalformedCheckpointMetadataError(
            f"Required metadata fields missing: {missing}. "
            f"Required: {list(_REQUIRED_FIELDS)}"
        )

    snapshot = meta_dict.get("training_config_snapshot")
    if not isinstance(snapshot, dict):
        raise MalformedCheckpointMetadataError(
            f"training_config_snapshot must be dict, got {type(snapshot).__name__}"
        )

    model_class_id = meta_dict["model_class_id"]
    if model_class_id not in MODEL_CLASS_IDS:
        raise UnknownModelClassIdError(
            f"Unknown model_class_id: {model_class_id!r}. "
            f"Known candidates: {MODEL_CLASS_IDS}"
        )

    try:
        return CheckpointMetadata(
            schema_version=schema_version,
            model_class_id=model_class_id,
            created_at=meta_dict["created_at"],
            training_config_snapshot=snapshot,
            code_git_sha=meta_dict.get("code_git_sha"),
            training_run_id=meta_dict.get("training_run_id"),
            framework_versions=meta_dict.get("framework_versions") or {},
        )
    except (TypeError, ValueError) as exc:
        raise MalformedCheckpointMetadataError(
            f"Failed to reconstruct CheckpointMetadata: {exc}"
        ) from exc


def load_checkpoint_metadata(
    checkpoint_path: str,
    *,
    ignore_embed_metadata: bool = False,
) -> CheckpointMetadata | None:
    """Read and validate the ``"metadata"`` block of a ``.pt`` file.

    Args:
        checkpoint_path (str): Path to the torch ``.pt`` artifact.
        ignore_embed_metadata (bool): When ``True``, malformed metadata
            yields ``None`` (caller should fall back to sidecar yaml)
            rather than raising. Default ``False`` (fail-loud, FR-016).

    Returns:
        CheckpointMetadata | None: Typed metadata when the block is
        present and valid; ``None`` when the ``"metadata"`` key is absent
        (legacy checkpoint — caller falls back to sidecar chain), or when
        ``ignore_embed_metadata=True`` and the block is malformed.

    Raises:
        IncompatibleCheckpointError: schema_version not supported
            (always raised, regardless of ``ignore_embed_metadata``).
        MalformedCheckpointMetadataError: malformed block and
            ``ignore_embed_metadata=False``.
        UnknownModelClassIdError: model_class_id not in registry
            (always raised).
    """
    # weights_only=False is required because our metadata block contains
    # non-tensor primitives (str, dict, int) that are rejected under the
    # PyTorch >= 2.6 default weights_only=True. We trust our own
    # checkpoints; a stricter policy can be layered on at load sites that
    # accept untrusted checkpoints.
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta_dict = raw.get("metadata") if isinstance(raw, dict) else None

    if meta_dict is None:
        return None

    if not isinstance(meta_dict, dict):
        if ignore_embed_metadata:
            logger.warning(
                "Embed metadata is not a dict (got %s); ignore_embed_metadata"
                " is True so falling back to sidecar resolution.",
                type(meta_dict).__name__,
            )
            return None
        raise MalformedCheckpointMetadataError(
            f"'metadata' key must be a dict, got {type(meta_dict).__name__}"
        )

    try:
        return _validate_metadata_dict(meta_dict)
    except MalformedCheckpointMetadataError:
        if ignore_embed_metadata:
            logger.warning(
                "Embed metadata is malformed but ignore_embed_metadata is"
                " True — falling back to sidecar resolution."
            )
            return None
        raise


_MAX_FALLBACK_ATTEMPTS: int = 30
"""Upper bound on iterations of the try-instantiate fallback loop in
:func:`reconstruct_model_from_metadata`. Each iteration plugs one missing
key with ``None``; runaway behavior would indicate a wider schema drift
that warrants explicit migration rather than dynamic patching."""


def reconstruct_model_from_metadata(
    metadata: CheckpointMetadata,
    *,
    user_overrides: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a model class from a CheckpointMetadata block.

    Pipeline (see plan.md DD-005 / data-model.md §6):

    1. Resolve ``model_class_id`` to a concrete class via
       :func:`get_model_class`.
    2. Reject any unsafe user override via :func:`validate_user_overrides`.
    3. Reconstruct a DictConfig from ``training_config_snapshot`` and
       attempt ``model_cls(cfg.model)`` directly.
    4. On ``ConfigAttributeError`` / ``AttributeError`` raised by the
       constructor accessing a missing cfg key, identify the missing key,
       fill it with ``None`` (so ``getattr(cfg, key, default)`` paths fall
       back), emit a WARNING, and retry. Bounded by
       :data:`_MAX_FALLBACK_ATTEMPTS` (FR-007).
    5. Unknown keys present in the snapshot but unused by the constructor
       are silently ignored by OmegaConf (FR-006). After a successful
       reconstruction, snapshot keys not transferred to the instance are
       logged as INFO for diagnostics.

    Args:
        metadata (CheckpointMetadata): Typed metadata block.
        user_overrides (dict[str, Any] | None): Optional dot-path → value
            mapping to be validated. Only used here for unsafe-prefix
            rejection; safe overrides are applied by the caller against
            the evaluation Hydra config.

    Returns:
        Any: An ``nn.Module`` instance produced by the model class
        constructor.

    Raises:
        UnsafeOverrideError: When a user override targets an unsafe
            prefix.
        CheckpointMetadataError: When fallback iteration cannot converge
            (more than :data:`_MAX_FALLBACK_ATTEMPTS` distinct missing
            keys) or the same key is requested twice in a row.
    """
    model_cls = get_model_class(metadata.model_class_id)

    if user_overrides:
        validate_user_overrides(metadata, user_overrides)

    snapshot = metadata.training_config_snapshot
    raw_model_cfg = snapshot.get("model")
    if not isinstance(raw_model_cfg, dict):
        raise MalformedCheckpointMetadataError(
            "training_config_snapshot.model must be dict, "
            f"got {type(raw_model_cfg).__name__}"
        )

    # Mutable working copy — we patch it on each fallback iteration.
    working_model_cfg: dict[str, Any] = dict(raw_model_cfg)
    filled_keys: list[str] = []
    instance: Any | None = None
    last_error: Exception | None = None

    for _attempt in range(_MAX_FALLBACK_ATTEMPTS):
        try_cfg = OmegaConf.create(working_model_cfg)
        try:
            instance = model_cls(try_cfg)
            break
        except (AttributeError, KeyError) as exc:
            last_error = exc
            missing_key = _parse_missing_key(exc, working_model_cfg)
            if missing_key is None or missing_key in filled_keys:
                # Either we cannot identify the missing key from the
                # error message, or we already filled it once and the
                # constructor still rejects — give up and re-raise.
                raise
            working_model_cfg[missing_key] = None
            filled_keys.append(missing_key)
            logger.warning(
                "model.%s missing from embed metadata snapshot — "
                "filled with None default (FR-007). Class evolved since "
                "training? Consider re-training to capture the new arg.",
                missing_key,
            )
    else:
        raise CheckpointMetadataError(
            f"Failed to instantiate {model_cls.__name__} after "
            f"{_MAX_FALLBACK_ATTEMPTS} fallback iterations. "
            f"Filled keys: {filled_keys}. Last error: {last_error!r}"
        )

    _log_unused_snapshot_keys(raw_model_cfg, instance)
    return instance


def _parse_missing_key(exc: BaseException, working_cfg: dict[str, Any]) -> str | None:
    """Best-effort extraction of a missing model-cfg key from an exception.

    OmegaConf raises ``ConfigAttributeError`` (subclass of
    ``AttributeError``) with a message like
    ``"Missing key X\\n    full_key: model.X\\n    object_type=dict"``.
    Plain ``AttributeError`` from ``self.config.X`` carries the attribute
    name in ``exc.args[0]`` like ``"'DictConfig' object has no attribute 'X'"``.

    Args:
        exc (BaseException): Caught exception.
        working_cfg (dict[str, Any]): Current model cfg working copy —
            used to skip keys that already exist (sanity guard).

    Returns:
        str | None: The missing top-level key, or ``None`` when the
        message cannot be parsed.
    """
    msg = str(exc)
    candidates: list[str] = []

    # OmegaConf style: "Missing key X" or "full_key: model.X"
    for prefix in ("full_key: model.", "Missing key "):
        if prefix in msg:
            tail = msg.split(prefix, 1)[1]
            # Cut at first whitespace / newline / closing punctuation
            for sep in ("\n", " ", ",", ".", "'", '"', ")"):
                if sep in tail:
                    tail = tail.split(sep, 1)[0]
            if tail:
                candidates.append(tail)

    # Plain AttributeError: extract quoted attribute name
    if "has no attribute" in msg:
        # message has form "...has no attribute 'X'..."
        parts = msg.split("has no attribute", 1)[1]
        if "'" in parts:
            candidates.append(parts.split("'", 2)[1])

    for cand in candidates:
        if cand and cand not in working_cfg:
            return cand
    return None


def _log_unused_snapshot_keys(
    snapshot_model_cfg: dict[str, Any], instance: Any
) -> None:
    """Emit one INFO listing keys present in snapshot.model but missing
    from ``instance.__dict__`` (best-effort FR-006 diagnostic).

    This is a heuristic — some keys may be consumed without becoming
    attributes (e.g., used only inside ``__init__`` to construct another
    object). The log line names that uncertainty.
    """
    try:
        instance_attrs = set(vars(instance).keys())
    except TypeError:
        return
    snapshot_keys = set(snapshot_model_cfg.keys())
    unused = sorted(snapshot_keys - instance_attrs)
    if unused:
        logger.info(
            "Snapshot model.* keys present but not transferred to "
            "%s.__dict__: %s. They were silently ignored by the "
            "constructor (FR-006) — may indicate obsolete training cfg.",
            type(instance).__name__,
            unused,
        )


def validate_user_overrides(
    metadata: CheckpointMetadata,
    user_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Split user overrides into safe / unsafe and reject unsafe ones.

    Any dot-path key matching a prefix in :data:`UNSAFE_OVERRIDE_PREFIXES`
    raises :class:`UnsafeOverrideError`. Safe overrides are returned as
    a dict for the caller to apply against the evaluation config.

    Args:
        metadata (CheckpointMetadata): Reference metadata used to fetch
            the conflicting metadata-side value for the error message.
        user_overrides (dict[str, Any]): Mapping of dot-path strings to
            user-supplied values (e.g., ``{"model.seq_length": 4096}``).

    Returns:
        dict[str, Any]: Subset of *user_overrides* whose keys are in
        :data:`SAFE_OVERRIDE_KEYS`.

    Raises:
        UnsafeOverrideError: When any key matches an unsafe prefix.
    """
    safe: dict[str, Any] = {}
    for key, value in user_overrides.items():
        if _is_unsafe_key(key):
            meta_value = _select_dotpath(metadata.training_config_snapshot, key)
            raise UnsafeOverrideError(
                f"Override of {key!r} is not allowed at evaluation time "
                f"(would invalidate weight semantics). "
                f"metadata={meta_value!r}, user={value!r}. "
                "Either remove the override or use a checkpoint trained "
                "with the desired settings."
            )
        if key in SAFE_OVERRIDE_KEYS:
            safe[key] = value
    return safe


def _is_unsafe_key(key: str) -> bool:
    """Return ``True`` when *key* matches any unsafe prefix.

    ``"model_class_id"`` is treated as an exact match; the other entries
    in :data:`UNSAFE_OVERRIDE_PREFIXES` are dot-prefix matches.

    Args:
        key (str): Candidate dot-path override key.

    Returns:
        bool: Whether *key* is in the unsafe set.
    """
    for prefix in UNSAFE_OVERRIDE_PREFIXES:
        if prefix.endswith("."):
            if key.startswith(prefix):
                return True
        elif key == prefix:
            return True
    return False


def _select_dotpath(snapshot: dict[str, Any], dotpath: str) -> Any:
    """Walk a dot-path through *snapshot* and return the value or ``None``.

    Args:
        snapshot (dict[str, Any]): Source dict (typically
            ``metadata.training_config_snapshot``).
        dotpath (str): Dot-separated key path, e.g.
            ``"model.tokenizer_type"``.

    Returns:
        Any: Resolved value; ``None`` when any segment is missing or the
        traversal hits a non-dict before consuming the path.
    """
    cursor: Any = snapshot
    for segment in dotpath.split("."):
        if not isinstance(cursor, dict) or segment not in cursor:
            return None
        cursor = cursor[segment]
    return cursor
