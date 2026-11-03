"""Orchestrate SpliceVarDB Excel → H5 build and expose the CLI.

This module ties together the loader, liftover runner, and variant
sequence extractor. It writes the final :file:`splicevardb_test.h5`
artifact described in ``contracts/h5_schema.md`` and exposes a CLI
entrypoint (``python -m process_input.splicevardb_h5_builder``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import OmegaConf
from pyfaidx import Fasta

from process_input.gene_strand_lookup import load_gene_strand_from_gtf
from process_input.liftover_runner import (
    LiftoverResult,
    LiftoverStats,
    open_chain_file,
    run_liftover,
)
from process_input.splicevardb_config import PipelineConfig
from process_input.splicevardb_exceptions import (
    InvalidVariantRecordError,
    LiftoverThresholdError,
)
from process_input.splicevardb_loader import load_splicevardb_excel
from process_input.variant_sequence_extractor import (
    VariantSequences,
    extract_variant_sequences,
)

logger = logging.getLogger(__name__)

EXIT_OK: int = 0
EXIT_GENERAL_FAILURE: int = 1
EXIT_LIFTOVER_GATE: int = 2


@dataclass(slots=True)
class SpliceVarDBH5Payload:
    """Column-stacked arrays ready to write as H5 datasets.

    Attributes:
        x_ref (np.ndarray): shape ``(N, total_length)`` int8.
        x_alt (np.ndarray): shape ``(N, total_length)`` int8.
        label (np.ndarray): shape ``(N,)`` int8.
        chrom (np.ndarray): shape ``(N,)`` object (str).
        pos (np.ndarray): shape ``(N,)`` uint64.
        ref (np.ndarray): shape ``(N,)`` object.
        alt (np.ndarray): shape ``(N,)`` object.
        gene_symbol (np.ndarray): shape ``(N,)`` object.
        gene_location (np.ndarray): shape ``(N,)`` object.
        gene_region (np.ndarray): shape ``(N,)`` object.
        strand (np.ndarray): shape ``(N,)`` object.
    """

    x_ref: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int8))
    x_alt: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int8))
    label: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int8))
    chrom: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))
    pos: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.uint64))
    ref: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))
    alt: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))
    gene_symbol: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=object)
    )
    gene_location: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=object)
    )
    gene_region: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=object)
    )
    strand: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))


def setup_logging(cfg: PipelineConfig) -> None:
    """Configure the root logger per ``cfg.logging``.

    Args:
        cfg (PipelineConfig): Pipeline configuration.
    """
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=cfg.logging.format,
        datefmt=cfg.logging.date_format,
    )


def load_config(path: str) -> PipelineConfig:
    """Load and validate a YAML config file.

    Args:
        path (str): Absolute path to the YAML configuration.

    Returns:
        PipelineConfig: Validated pipeline configuration.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping, got {type(payload)!r}")
    # Drop Hydra section if present (not part of PipelineConfig schema).
    payload.pop("hydra", None)
    payload.pop("base_path", None)
    return PipelineConfig.model_validate(payload)


def build_h5_payload(
    results: list[LiftoverResult], sequences: list[VariantSequences]
) -> SpliceVarDBH5Payload:
    """Stack per-record arrays into a :class:`SpliceVarDBH5Payload`.

    Args:
        results (list[LiftoverResult]): Records matching ``sequences``
            row-for-row.
        sequences (list[VariantSequences]): Encoded ref/alt sequences.

    Returns:
        SpliceVarDBH5Payload: Stacked arrays ready for H5 writing.

    Raises:
        ValueError: If the two input lists have different lengths.
    """
    if len(results) != len(sequences):
        raise ValueError(
            f"results and sequences length mismatch: {len(results)} vs {len(sequences)}"
        )
    n = len(results)
    if n == 0:
        return SpliceVarDBH5Payload()

    total_length = sequences[0].x_ref.shape[0]
    x_ref = np.stack([seq.x_ref for seq in sequences]).astype(np.int8)
    x_alt = np.stack([seq.x_alt for seq in sequences]).astype(np.int8)
    label = np.array([record.label for record in results], dtype=np.int8)
    chrom = np.array([record.chrom for record in results], dtype=object)
    pos = np.array([record.pos_hg38 for record in results], dtype=np.uint64)
    ref = np.array([record.ref for record in results], dtype=object)
    alt = np.array([record.alt for record in results], dtype=object)
    gene_symbol = np.array([record.gene_symbol for record in results], dtype=object)
    gene_location = np.array([record.gene_location for record in results], dtype=object)
    gene_region = np.array([record.gene_region for record in results], dtype=object)
    strand = np.array([record.strand for record in results], dtype=object)

    assert x_ref.shape == (n, total_length)
    assert x_alt.shape == (n, total_length)
    return SpliceVarDBH5Payload(
        x_ref=x_ref,
        x_alt=x_alt,
        label=label,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        gene_symbol=gene_symbol,
        gene_location=gene_location,
        gene_region=gene_region,
        strand=strand,
    )


def write_h5(payload: SpliceVarDBH5Payload, path: Path, attrs: dict[str, Any]) -> None:
    """Write ``payload`` to an HDF5 file at ``path`` with ``attrs`` metadata.

    Args:
        payload (SpliceVarDBH5Payload): Stacked arrays from
            :func:`build_h5_payload`.
        path (Path): Output file path. Parent directory is created if
            absent.
        attrs (dict[str, Any]): File-level attributes per
            ``contracts/h5_schema.md``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    total_length = (
        payload.x_ref.shape[1] if payload.x_ref.ndim == 2 else payload.x_ref.shape[0]
    )
    chunks = (1, total_length) if total_length > 0 else None

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "X_ref",
            data=payload.x_ref,
            dtype="int8",
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        h5.create_dataset(
            "X_alt",
            data=payload.x_alt,
            dtype="int8",
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        h5.create_dataset("label", data=payload.label, dtype="int8")
        h5.create_dataset(
            "chrom", data=payload.chrom.astype(object), dtype=string_dtype
        )
        h5.create_dataset("pos", data=payload.pos, dtype="uint64")
        h5.create_dataset("ref", data=payload.ref.astype(object), dtype=string_dtype)
        h5.create_dataset("alt", data=payload.alt.astype(object), dtype=string_dtype)
        h5.create_dataset(
            "gene_symbol",
            data=payload.gene_symbol.astype(object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "gene_location",
            data=payload.gene_location.astype(object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "gene_region",
            data=payload.gene_region.astype(object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "strand", data=payload.strand.astype(object), dtype=string_dtype
        )

        for key, value in attrs.items():
            h5.attrs[key] = value

    logger.info(
        "Wrote H5 to %s with datasets=%s, attrs=%s",
        path,
        [
            "X_ref",
            "X_alt",
            "label",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene_symbol",
            "gene_location",
            "gene_region",
            "strand",
        ],
        list(attrs.keys()),
    )


def _build_attrs(
    cfg: PipelineConfig, stats: LiftoverStats, *, forced: bool
) -> dict[str, Any]:
    """Return the file-level attribute dict per ``contracts/h5_schema.md``.

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        stats (LiftoverStats): Liftover statistics from
            :func:`~process_input.liftover_runner.run_liftover`.
        forced (bool): Whether the user passed ``--force``.

    Returns:
        dict[str, Any]: Mapping of attribute name to value.
    """
    return {
        "target_length": int(cfg.target_length),
        "flank_length": int(cfg.flank_length),
        "total_length": int(cfg.total_length),
        "genome_build": "GRCh38",
        "source_coordinates": "GRCh37",
        "liftover": "hg19 → hg38",
        "strand_aware": bool(cfg.strand_aware),
        "source": "SpliceVarDB",
        "min_liftover_success_rate": float(cfg.min_liftover_success_rate),
        "actual_liftover_success_rate": float(stats.success_rate),
        "forced": bool(forced),
        "build_timestamp_utc": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _preflight_paths(cfg: PipelineConfig) -> None:
    """Verify that every required input file exists before the build starts.

    Failing here keeps the error surface small and actionable: operators
    learn about missing NAS files in a single short message rather than
    after an expensive GTF scan or an opaque ``liftover.ChainFile``
    failure.

    Args:
        cfg (PipelineConfig): Validated pipeline configuration.

    Raises:
        FileNotFoundError: When any of ``excel``, ``ref_genome``,
            ``chain_hg19_to_hg38``, or ``gene_annotation_gtf`` paths do
            not point to an existing file. The message lists every
            missing path so the operator can fix them in one pass.
    """
    required = {
        "excel": cfg.paths.excel,
        "ref_genome": cfg.paths.ref_genome,
        "chain_hg19_to_hg38": cfg.paths.chain_hg19_to_hg38,
        "gene_annotation_gtf": cfg.paths.gene_annotation_gtf,
    }
    missing = [
        f"{key}={path}" for key, path in required.items() if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Preflight failed; the following configured paths do not exist: "
            + "; ".join(missing)
        )


def build_splicevardb_h5(
    cfg: PipelineConfig,
    *,
    force: bool,
    output_override: str | None = None,
) -> Path | None:
    """Run the full loader → liftover → extractor → H5 pipeline.

    Args:
        cfg (PipelineConfig): Validated pipeline configuration.
        force (bool): Bypass the liftover success-rate gate.
        output_override (str | None): Optional CLI override for the
            output path. When ``None``, ``cfg.paths.output`` is used.

    Returns:
        Path | None: Absolute path to the created H5 file, or ``None``
        when no records survived the pipeline (no file is written).

    Raises:
        FileNotFoundError: When any required input file is missing (see
            :func:`_preflight_paths`).
    """
    _preflight_paths(cfg)
    gene_strand = load_gene_strand_from_gtf(cfg.paths.gene_annotation_gtf)
    records = load_splicevardb_excel(cfg, gene_strand=gene_strand)
    if not records:
        logger.warning("No variant records after loading; skipping H5 creation")
        return None

    chain = open_chain_file(cfg.paths.chain_hg19_to_hg38)
    fasta = Fasta(cfg.paths.ref_genome, sequence_always_upper=True)

    results, stats = run_liftover(records, cfg, chain=chain, fasta=fasta, force=force)
    if not results:
        logger.warning("No records survived liftover; skipping H5 creation")
        return None

    sequences: list[VariantSequences] = []
    for record in results:
        sequences.append(
            extract_variant_sequences(
                fasta,
                record,
                target_length=cfg.target_length,
                flank_length=cfg.flank_length,
                strand_aware=cfg.strand_aware,
            )
        )
    logger.info(
        "Extracted sequences for %d variants (total_length=%d)",
        len(sequences),
        cfg.total_length,
    )

    payload = build_h5_payload(results, sequences)
    attrs = _build_attrs(cfg, stats, forced=force)

    output_path = Path(output_override) if output_override else Path(cfg.paths.output)
    write_h5(payload, output_path, attrs)
    logger.info("Build complete. rows=%d", len(results))
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments per ``contracts/cli.md``.

    Args:
        argv (list[str] | None): Argument list to parse. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m process_input.splicevardb_h5_builder",
        description="Build SpliceVarDB variant-centric H5 file from Excel + GRCh38.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for the output H5 path (default: cfg.paths.output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the liftover success-rate gate and proceed even if below threshold",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code per ``contracts/cli.md``.

    Args:
        argv (list[str] | None): Optional argument list for testing.

    Returns:
        int: ``0`` on success, ``1`` on generic failure, ``2`` on
        liftover-rate gate failure.
    """
    args = parse_args(argv)
    try:
        cfg = load_config(args.config_path)
    except (FileNotFoundError, ValueError) as exc:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        )
        logger.error("Failed to load config %s: %s", args.config_path, exc)
        return EXIT_GENERAL_FAILURE

    setup_logging(cfg)

    try:
        build_splicevardb_h5(cfg, force=args.force, output_override=args.output)
    except LiftoverThresholdError as exc:
        logger.error("%s", exc)
        return EXIT_LIFTOVER_GATE
    except InvalidVariantRecordError as exc:
        logger.error("Invalid Excel input: %s", exc)
        return EXIT_GENERAL_FAILURE
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.exception("Build failed: %s", exc)
        return EXIT_GENERAL_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
