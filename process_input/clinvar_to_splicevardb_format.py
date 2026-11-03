"""Convert ClinVar 2-star all-chrom H5 to a SpliceVarDB-shaped H5.

The downstream gencode-anchored builder consumes a fixed set of
columns (``X_ref``, ``X_alt``, ``label``, ``chrom``, ``pos``,
``ref``, ``alt``, ``gene_symbol``, ``gene_location``, ``gene_region``,
``strand``). ClinVar's H5 schema is a superset on some axes and a
subset on others — this preprocessor:

1. Filters to SNVs (``len(ref)==len(alt)==1``).
2. Keeps only Pathogenic / Likely pathogenic / Pathogenic-Likely
   pathogenic and the matching Benign tier; drops anything else
   (VUS, "not provided", etc.).
3. Maps pathogenicity → binary ``label`` (1=Pathogenic, 0=Benign).
4. Adds empty placeholder columns (``gene_location``,
   ``gene_region``, ``strand``) so the gencode-anchored builder can
   read it without modification. Strand is filled from a GENCODE
   gene-symbol lookup so the builder's strand_aware logic stays
   meaningful for later stages.
5. Writes a single output H5 mirroring the SpliceVarDB layout.

Run via::

    python -m process_input.clinvar_to_splicevardb_format \\
        --input  /NAS/.../clinvar_2star_allchr_h5/clinvar.h5 \\
        --output /NAS/.../clinvar_2star_allchr_h5/clinvar_pb_snv.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from process_input.gene_strand_lookup import load_gene_strand_from_gtf

logger = logging.getLogger(__name__)

PATHOGENIC_TIER: frozenset[str] = frozenset(
    {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
)
BENIGN_TIER: frozenset[str] = frozenset(
    {"Benign", "Likely benign", "Benign/Likely benign"}
)


def _decode(value: object) -> str:
    """Coerce an HDF5 vlen-string element to a Python str.

    Args:
        value (object): Raw element read from a string dataset.

    Returns:
        str: UTF-8 decoded value, or ``str(value)`` for non-bytes.
    """
    return value.decode() if isinstance(value, bytes) else str(value)


def _resolve_strand(
    gene_symbol_raw: str,
    gene_strand: dict[str, str],
) -> str:
    """Look up strand for a possibly composite gene symbol.

    ClinVar's ``gene_symbol`` may carry multiple gene names separated
    by ``";"`` or ``","``. We try each component and return the first
    successful hit. Empty strand when nothing matches.

    Args:
        gene_symbol_raw (str): ClinVar gene_symbol string.
        gene_strand (dict[str, str]): GENCODE-derived
            ``gene_name → strand`` mapping.

    Returns:
        str: ``"+"``, ``"-"``, or ``""`` when unresolved.
    """
    if not gene_symbol_raw:
        return ""
    for sep in (";", ","):
        gene_symbol_raw = gene_symbol_raw.replace(sep, "|")
    for piece in gene_symbol_raw.split("|"):
        piece = piece.strip()
        if not piece:
            continue
        s = gene_strand.get(piece)
        if s in {"+", "-"}:
            return s
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv (list[str] | None): Argument list (defaults to
            ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Filter and reformat a ClinVar H5 into the"
            " SpliceVarDB-style input layout consumed by the"
            " gencode-anchored builder."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="ClinVar H5 (e.g. clinvar_2star_allchr_h5/clinvar.h5).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination SpliceVarDB-shaped H5 path.",
    )
    parser.add_argument(
        "--gene-annotation-gtf",
        default=(
            "./gencode/"
            "gencode.v49.annotation.gtf"
        ),
        help="GENCODE GTF used for gene_symbol → strand lookup.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        help="Python logging level (default INFO).",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure the project-standard log format.

    Args:
        level (str): Log level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def reformat_clinvar(
    *,
    input_path: str,
    output_path: str,
    gene_annotation_gtf: str,
) -> Path:
    """Filter ClinVar to SNV {Path,Benign} and write SpliceVarDB shape.

    Args:
        input_path (str): ClinVar H5 path.
        output_path (str): Destination H5 path.
        gene_annotation_gtf (str): Path to GENCODE GTF used for the
            strand lookup table.

    Returns:
        Path: Absolute path to the written H5.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading gene strand table from %s", gene_annotation_gtf)
    gene_strand = load_gene_strand_from_gtf(gene_annotation_gtf)
    logger.info("Loaded %d gene → strand entries", len(gene_strand))

    logger.info("Reading ClinVar H5: %s", input_path)
    with h5py.File(input_path, "r") as f:
        n_total = f["X_ref"].shape[0]
        seq_length = int(f["X_ref"].shape[1])
        in_attrs = dict(f.attrs)
        target_length = int(in_attrs.get("target_length", seq_length))
        flank_length = int(in_attrs.get("flank_length", 0))

        chroms_raw = f["chrom"][:]
        poses_raw = f["pos"][:]
        refs_raw = f["ref"][:]
        alts_raw = f["alt"][:]
        gene_symbols_raw = f["gene_symbol"][:]
        path_raw = f["pathogenicity"][:]

        # SNV + label filter without materializing X first (X is huge).
        is_pathogenic = np.zeros(n_total, dtype=bool)
        is_benign = np.zeros(n_total, dtype=bool)
        is_snv = np.zeros(n_total, dtype=bool)
        for i in range(n_total):
            ref_b = refs_raw[i]
            alt_b = alts_raw[i]
            if len(ref_b) != 1 or len(alt_b) != 1:
                continue
            is_snv[i] = True
            p = _decode(path_raw[i])
            if p in PATHOGENIC_TIER:
                is_pathogenic[i] = True
            elif p in BENIGN_TIER:
                is_benign[i] = True

        keep = is_snv & (is_pathogenic | is_benign)
        keep_idx = np.where(keep)[0]
        n_kept = int(keep.sum())
        logger.info(
            "Filtered: total=%d, snv=%d, pathogenic=%d, benign=%d, kept=%d",
            n_total,
            int(is_snv.sum()),
            int(is_pathogenic.sum()),
            int(is_benign.sum()),
            n_kept,
        )

        if n_kept == 0:
            raise ValueError("No records survived filter; nothing to write")

        x_ref = f["X_ref"][keep_idx]
        x_alt = f["X_alt"][keep_idx]
        chroms = np.array(
            [_decode(chroms_raw[i]) for i in keep_idx], dtype=object
        )
        poses = poses_raw[keep_idx].astype(np.uint64)
        refs = np.array(
            [_decode(refs_raw[i]) for i in keep_idx], dtype=object
        )
        alts = np.array(
            [_decode(alts_raw[i]) for i in keep_idx], dtype=object
        )
        gene_syms = np.array(
            [_decode(gene_symbols_raw[i]) for i in keep_idx], dtype=object
        )
        labels = np.where(is_pathogenic[keep_idx], 1, 0).astype(np.int8)
        strands = np.array(
            [_resolve_strand(s, gene_strand) for s in gene_syms],
            dtype=object,
        )

    n_no_strand = int((strands == "").sum())
    logger.info(
        "Strand resolution: matched=%d, missing=%d (passed through as empty)",
        n_kept - n_no_strand,
        n_no_strand,
    )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    chunks = (1, seq_length) if seq_length > 0 else None

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset(
            "X_ref",
            data=x_ref.astype(np.int8),
            dtype="int8",
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        h5.create_dataset(
            "X_alt",
            data=x_alt.astype(np.int8),
            dtype="int8",
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        h5.create_dataset("label", data=labels, dtype="int8")
        h5.create_dataset(
            "chrom", data=chroms.astype(object), dtype=string_dtype
        )
        h5.create_dataset("pos", data=poses, dtype="uint64")
        h5.create_dataset(
            "ref", data=refs.astype(object), dtype=string_dtype
        )
        h5.create_dataset(
            "alt", data=alts.astype(object), dtype=string_dtype
        )
        h5.create_dataset(
            "gene_symbol",
            data=gene_syms.astype(object),
            dtype=string_dtype,
        )
        # gene_location / gene_region are not in ClinVar — pass empty.
        empty_str = np.array([""] * n_kept, dtype=object)
        h5.create_dataset(
            "gene_location", data=empty_str, dtype=string_dtype
        )
        h5.create_dataset(
            "gene_region", data=empty_str, dtype=string_dtype
        )
        h5.create_dataset(
            "strand", data=strands.astype(object), dtype=string_dtype
        )

        for key, value in in_attrs.items():
            h5.attrs[key] = value
        h5.attrs["source"] = "ClinVar (Pathogenic vs Benign SNV)"
        h5.attrs["filter_pathogenic_tier"] = list(PATHOGENIC_TIER)
        h5.attrs["filter_benign_tier"] = list(BENIGN_TIER)
        h5.attrs["target_length"] = int(target_length)
        h5.attrs["flank_length"] = int(flank_length)
        h5.attrs["total_length"] = int(seq_length)

    logger.info("Wrote %s (rows=%d, cols=%d)", out_path, n_kept, seq_length)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv (list[str] | None): Argument list.

    Returns:
        int: ``0`` on success, ``1`` on failure.
    """
    args = parse_args(argv)
    setup_logging(args.logging_level)
    try:
        reformat_clinvar(
            input_path=args.input,
            output_path=args.output,
            gene_annotation_gtf=args.gene_annotation_gtf,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Reformat failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
