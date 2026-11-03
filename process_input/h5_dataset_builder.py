"""Build final H5 dataset from intermediate H5 datafile.

Reads the intermediate H5 file produced by
h5_datafile_builder and creates a training-ready H5
dataset with integer-encoded (X, Y, Y_REGION, Y_EXON)
datapoints organized in chunks.

Storage format:
    X{i}:        (N, SL+CL_max) int8 -- sequence tokens
        0=N(pad), 1=A, 2=C, 3=G, 4=T
    Y{i}:        (N, SL) int8 -- splice labels
        0=neither, 1=acceptor, 2=donor, -1=padding
    Y_REGION{i}: (N, SL) int8 -- region annotation labels
        0=intron, 1=5'UTR, 2=CDS_phase0, 3=CDS_phase1,
        4=CDS_phase2, 5=3'UTR, 6=ncRNA_exon, -1=padding
    Y_EXON{i}:   (N, SL) int8 -- exon-position labels
        0=first_exon, 1=internal_exon, 2=last_exon,
        -1=intron/single-exon/padding

Usage:
    uv run -m process_input.h5_dataset_builder \\
        --config-path config/build_data/h5_data.yaml \\
        train all

    uv run -m process_input.h5_dataset_builder \\
        --config-path config/build_data/h5_data.yaml \\
        test 0 -i /path/to/datafile.h5 -o /path/to/dataset.h5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from process_input.h5_utils import create_datapoints

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create final H5 dataset "
            "from intermediate datafile"
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
        "-i",
        "--input",
        type=str,
        default=None,
        help=(
            "Input datafile path "
            "(default: {output_dir}/datafile_{split}_{paralog}.h5)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output dataset path "
            "(default: {output_dir}/dataset_{split}_{paralog}.h5)"
        ),
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


def build_dataset(
    cfg: DictConfig,
    split: str,
    paralog: str,
    input_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Build final training-ready H5 dataset.

    Reads gene data from intermediate H5 datafile, processes
    each gene into integer-encoded (X, Y) datapoints, and
    writes them in chunks to the output H5 file.

    Args:
        cfg (DictConfig): Configuration object.
        split (str): Data split ('train', 'test', or 'all').
        paralog (str): Paralog filter ('0', '1', or 'all').
        input_path (str | None): Input datafile path override.
        output_path (str | None): Output dataset path override.

    Returns:
        str: Path to the created dataset H5 file.
    """
    start_time = time.time()

    cl_max = cfg.cl_max
    sl = cfg.sl
    stride = int(cfg.get("stride", sl))
    chunk_size = cfg.chunk_size
    out_dir = Path(str(cfg.paths.output_dir))

    if input_path is None:
        input_path = str(
            out_dir / f"datafile_{split}_{paralog}.h5"
        )
    if output_path is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            out_dir / f"dataset_{split}_{paralog}.h5"
        )

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(
        f"CL_max: {cl_max}, SL: {sl}, Stride: {stride}, "
        f"Chunk size: {chunk_size}"
    )

    h5f = h5py.File(input_path, "r")

    # The datafile records how far past the flanks it read; datafiles written
    # before the attribute existed are N-flank with no extension.
    flank_fill = str(cfg.get("flank_fill", "N"))
    datafile_fill = str(h5f.attrs.get("flank_fill", "N"))
    extra_flank = int(h5f.attrs.get("extra_flank", 0))
    if datafile_fill != flank_fill:
        h5f.close()
        raise ValueError(
            f"config flank_fill={flank_fill!r} but {input_path} was built with "
            f"{datafile_fill!r}; rebuild the datafile with the same config"
        )
    logger.info(f"Flank fill: {flank_fill} (extra_flank={extra_flank})")

    SEQ = h5f["SEQ"][:].astype(str)
    STRAND = h5f["STRAND"][:].astype(str)
    TX_START = h5f["TX_START"][:].astype(str)
    TX_END = h5f["TX_END"][:].astype(str)
    JN_START = h5f["JN_START"][:].astype(str)
    JN_END = h5f["JN_END"][:].astype(str)

    has_region_meta = (
        "EXON_START" in h5f
        and "EXON_END" in h5f
        and "CDS_START" in h5f
        and "CDS_END" in h5f
        and "CDS_FRAME" in h5f
    )
    if has_region_meta:
        EXON_START = h5f["EXON_START"][:].astype(str)
        EXON_END = h5f["EXON_END"][:].astype(str)
        CDS_START = h5f["CDS_START"][:].astype(str)
        CDS_END = h5f["CDS_END"][:].astype(str)
        CDS_FRAME = h5f["CDS_FRAME"][:].astype(str)
    else:
        logger.warning(
            "Intermediate datafile lacks EXON_*/CDS_* datasets; "
            "Y_REGION and Y_EXON will be filled with the ignore "
            "index for every base."
        )
        EXON_START = np.full(SEQ.shape[0], ",", dtype=object)
        EXON_END = np.full(SEQ.shape[0], ",", dtype=object)
        CDS_START = np.full(SEQ.shape[0], ",", dtype=object)
        CDS_END = np.full(SEQ.shape[0], ",", dtype=object)
        CDS_FRAME = np.full(SEQ.shape[0], ",", dtype=object)

    h5f.close()

    n_genes = SEQ.shape[0]
    logger.info(f"Total genes: {n_genes}")

    if n_genes == 0:
        logger.warning("No genes found in datafile")
        return output_path

    h5f2 = h5py.File(output_path, "w")

    num_chunks = n_genes // chunk_size
    if num_chunks == 0:
        num_chunks = 1

    for i in range(num_chunks):
        if (i + 1) == num_chunks:
            cur_chunk_size = n_genes - i * chunk_size
        else:
            cur_chunk_size = chunk_size

        X_batch: list[np.ndarray] = []
        Y_batch: list[np.ndarray] = []
        Y_region_batch: list[np.ndarray] = []
        Y_exon_batch: list[np.ndarray] = []

        for j in range(cur_chunk_size):
            idx = i * chunk_size + j

            X, Y, Y_region, Y_exon = create_datapoints(
                SEQ[idx],
                STRAND[idx],
                TX_START[idx],
                TX_END[idx],
                JN_START[idx],
                JN_END[idx],
                cl_max,
                sl,
                exon_start=EXON_START[idx],
                exon_end=EXON_END[idx],
                cds_start=CDS_START[idx],
                cds_end=CDS_END[idx],
                cds_frame=CDS_FRAME[idx],
                stride=stride,
                flank_fill=flank_fill,
                extra_flank=extra_flank,
            )

            X_batch.extend(X)
            Y_batch.extend(Y)
            Y_region_batch.extend(Y_region)
            Y_exon_batch.extend(Y_exon)

        X_arr = np.asarray(X_batch, dtype="int8")
        Y_arr = np.asarray(Y_batch, dtype="int8")
        Y_region_arr = np.asarray(Y_region_batch, dtype="int8")
        Y_exon_arr = np.asarray(Y_exon_batch, dtype="int8")

        h5f2.create_dataset(f"X{i}", data=X_arr)
        h5f2.create_dataset(f"Y{i}", data=Y_arr)
        h5f2.create_dataset(f"Y_REGION{i}", data=Y_region_arr)
        h5f2.create_dataset(f"Y_EXON{i}", data=Y_exon_arr)

        if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            logger.info(
                f"  Chunk {i + 1}/{num_chunks} done "
                f"({X_arr.shape[0]} datapoints)"
            )

    h5f2.close()

    elapsed = time.time() - start_time
    logger.info(f"Created {num_chunks} chunks")
    logger.info(f"Elapsed: {elapsed:.2f}s")
    logger.info(f"Output: {output_path}")

    return output_path


def main() -> None:
    """Main entry point for building final H5 dataset."""
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

    build_dataset(
        cfg,
        args.split,
        args.paralog,
        args.input,
        args.output,
    )

    return


if __name__ == "__main__":
    main()
