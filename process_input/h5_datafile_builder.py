"""Build intermediate H5 datafile from canonical dataset and genome.

Replaces the original grab_sequence.sh + create_datafile.py pipeline
by extracting genomic sequences directly via pyfaidx (no bedtools
dependency) and writing an intermediate H5 file containing gene
metadata and DNA sequences.

Usage:
    uv run -m process_input.h5_datafile_builder \\
        --config-path config/build_data/h5_data.yaml \\
        train all

    uv run -m process_input.h5_datafile_builder \\
        --config-path config/build_data/h5_data.yaml \\
        test 0 -o /path/to/output.h5
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyfaidx import Fasta

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create intermediate H5 datafile "
            "from canonical dataset and genome"
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "split",
        choices=["train", "test", "all"],
        help="Data split to process",
    )
    parser.add_argument(
        "paralog",
        choices=["0", "1", "all"],
        help=(
            "Paralog filter "
            "(0=no paralogs, 1=has paralogs, all=both)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (overrides config)",
    )

    return parser.parse_args()


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging from config.

    Args:
        cfg (DictConfig): Configuration object.
    """
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.date_format,
    )

    return


def extract_sequence(
    genome: Fasta,
    chrom: str,
    start: int,
    end: int,
) -> str:
    """Extract DNA sequence from genome with boundary handling.

    Pads with N if the requested region extends beyond
    chromosome boundaries.

    Args:
        genome (Fasta): pyfaidx Fasta genome object.
        chrom (str): Chromosome name (e.g. 'chr1').
        start (int): 0-based start position (inclusive).
        end (int): 0-based end position (exclusive).

    Returns:
        str: DNA sequence string of length (end - start).
    """
    chrom_len = len(genome[chrom])

    left_pad = 0
    right_pad = 0

    if start < 0:
        left_pad = -start
        start = 0
    if end > chrom_len:
        right_pad = end - chrom_len
        end = chrom_len

    seq = str(genome[chrom][start:end])

    return "N" * left_pad + seq + "N" * right_pad


def build_datafile(
    cfg: DictConfig,
    split: str,
    paralog: str,
    output_path: str | None = None,
) -> str:
    """Build intermediate H5 datafile.

    Reads canonical dataset text file, extracts sequences from
    the reference genome using pyfaidx, and writes an H5 file
    containing gene metadata and sequences.

    Args:
        cfg (DictConfig): Configuration object.
        split (str): Data split ('train', 'test', or 'all').
        paralog (str): Paralog filter ('0', '1', or 'all').
        output_path (str | None): Output file path override.

    Returns:
        str: Path to the created H5 file.
    """
    start_time = time.time()

    cl_max = cfg.cl_max
    assert cl_max % 2 == 0, (
        f"cl_max must be even, got {cl_max}"
    )
    # With real flanks, read `sl` more bases on each side: the last block of
    # a transcript reaches up to stride - 1 < sl bases past the flank, and
    # create_datapoints fills that tail from the downstream extension.
    flank_fill = str(cfg.get("flank_fill", "N"))
    if flank_fill not in ("N", "real"):
        raise ValueError(f"flank_fill must be 'N' or 'real', got {flank_fill!r}")
    extra_flank = int(cfg.sl) if flank_fill == "real" else 0

    if split == "train":
        chrom_group = list(cfg.train_chromosomes)
    elif split == "test":
        chrom_group = list(cfg.test_chromosomes)
    else:
        chrom_group = (
            list(cfg.train_chromosomes)
            + list(cfg.test_chromosomes)
        )

    chrom_set = set(chrom_group)

    logger.info(
        f"Split: {split}, Paralog: {paralog}"
    )
    logger.info(
        f"CL_max: {cl_max}, SL: {cfg.sl}"
    )
    logger.info(
        f"Chromosomes ({len(chrom_group)}): "
        f"{', '.join(sorted(chrom_group))}"
    )

    logger.info(
        f"Loading genome: {cfg.paths.ref_genome}"
    )
    genome = Fasta(str(cfg.paths.ref_genome))

    NAME: list[str] = []
    PARALOG: list[int] = []
    CHROM: list[str] = []
    STRAND: list[str] = []
    TX_START: list[str] = []
    TX_END: list[str] = []
    JN_START: list[list[str]] = []
    JN_END: list[list[str]] = []
    EXON_START: list[str] = []
    EXON_END: list[str] = []
    CDS_START: list[str] = []
    CDS_END: list[str] = []
    CDS_FRAME: list[str] = []
    SEQ: list[str] = []

    splice_table = str(cfg.paths.splice_table)
    logger.info(f"Reading splice table: {splice_table}")

    with open(splice_table, "r") as f:
        for line in f:
            data = re.split(r"\n|\t", line)[:-1]

            chrom = data[2]
            if chrom not in chrom_set:
                continue

            if paralog != "all" and data[1] != paralog:
                continue

            tx_start = int(data[4])
            tx_end = int(data[5])

            # Match original grab_sequence.sh:
            #   BED start = TX_START - CL_max/2 - 1
            #   BED end   = TX_END + CL_max/2
            seq_start = tx_start - cl_max // 2 - 1 - extra_flank
            seq_end = tx_end + cl_max // 2 + extra_flank

            seq = extract_sequence(
                genome, chrom, seq_start, seq_end
            )

            # Legacy SpliceAI splice tables only have 8 columns
            # (cols 0-7).  Region annotation columns 8-12 were
            # added later; fall back to empty strings when the
            # table predates the schema extension.
            exon_start_col = data[8] if len(data) > 8 else ","
            exon_end_col = data[9] if len(data) > 9 else ","
            cds_start_col = data[10] if len(data) > 10 else ","
            cds_end_col = data[11] if len(data) > 11 else ","
            cds_frame_col = data[12] if len(data) > 12 else ","

            NAME.append(data[0])
            PARALOG.append(int(data[1]))
            CHROM.append(chrom)
            STRAND.append(data[3])
            TX_START.append(data[4])
            TX_END.append(data[5])
            JN_START.append([data[6]])
            JN_END.append([data[7]])
            EXON_START.append(exon_start_col)
            EXON_END.append(exon_end_col)
            CDS_START.append(cds_start_col)
            CDS_END.append(cds_end_col)
            CDS_FRAME.append(cds_frame_col)
            SEQ.append(seq)

    logger.info(f"Total genes loaded: {len(NAME)}")

    if output_path is None:
        out_dir = Path(str(cfg.paths.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            out_dir / f"datafile_{split}_{paralog}.h5"
        )

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    logger.info(f"Writing H5: {output_path}")

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["flank_fill"] = flank_fill
        h5f.attrs["extra_flank"] = extra_flank
        h5f.create_dataset(
            "NAME",
            data=np.asarray(NAME, dtype="S"),
        )
        h5f.create_dataset(
            "PARALOG",
            data=np.asarray(PARALOG),
        )
        h5f.create_dataset(
            "CHROM",
            data=np.asarray(CHROM, dtype="S"),
        )
        h5f.create_dataset(
            "STRAND",
            data=np.asarray(STRAND, dtype="S"),
        )
        h5f.create_dataset(
            "TX_START",
            data=np.asarray(TX_START, dtype="S"),
        )
        h5f.create_dataset(
            "TX_END",
            data=np.asarray(TX_END, dtype="S"),
        )
        h5f.create_dataset(
            "JN_START",
            data=np.asarray(JN_START, dtype="S"),
        )
        h5f.create_dataset(
            "JN_END",
            data=np.asarray(JN_END, dtype="S"),
        )
        h5f.create_dataset(
            "EXON_START",
            data=np.asarray(EXON_START, dtype="S"),
        )
        h5f.create_dataset(
            "EXON_END",
            data=np.asarray(EXON_END, dtype="S"),
        )
        h5f.create_dataset(
            "CDS_START",
            data=np.asarray(CDS_START, dtype="S"),
        )
        h5f.create_dataset(
            "CDS_END",
            data=np.asarray(CDS_END, dtype="S"),
        )
        h5f.create_dataset(
            "CDS_FRAME",
            data=np.asarray(CDS_FRAME, dtype="S"),
        )
        h5f.create_dataset(
            "SEQ",
            data=np.asarray(SEQ, dtype="S"),
        )

    elapsed = time.time() - start_time
    logger.info(f"Genes: {len(NAME)}")
    logger.info(f"Elapsed: {elapsed:.2f}s")
    logger.info(f"Output: {output_path}")

    return output_path


def main() -> None:
    """Main entry point for building intermediate H5."""
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(
            f"Config not found: {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    setup_logging(cfg)

    build_datafile(
        cfg, args.split, args.paralog, args.output
    )

    return


if __name__ == "__main__":
    main()
