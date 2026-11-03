"""Concatenate train (chr2,4,6,...,X,Y) + test (chr1,3,5,7,9) H5 datasets.

Produces ``dataset_all_chroms.h5`` for the region-strided variants
used by the Phase 3 ensemble training:

    h5_cl10000_regions/             (legacy 2-stage-DC, stride=sl=5000)
    h5_cl10000_sl5000_s2500_regions/
    h5_cl10000_sl5000_s1250_regions/
    h5_cl16384_sl16384_s1250_regions/  (32,768 nt long-context arm)

The output is schema-identical to ``dataset_train_all.h5`` (X{i}, Y{i},
Y_REGION{i}, Y_EXON{i}); test chunks are appended after the train chunks
with their indices remapped so the trainer's existing SplicingH5Dataset
loads them transparently.

Idempotent: if ``dataset_all_chroms.h5`` already exists at the target
path it is skipped unless ``--force`` is set.

Usage::

    python -m scripts.build_all_chroms_h5 --variant all
    python -m scripts.build_all_chroms_h5 --variant s1250 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py

VARIANTS: dict[str, Path] = {
    "v50mane": Path("./data/h5_cl10000_sl5000_s1250_regions_realflank_v50mane"),
    "regions": Path(
        "./data/h5_cl10000_regions"
    ),
    "s2500": Path(
        "./data/h5_cl10000_sl5000_s2500_regions"
    ),
    "s1250": Path(
        "./data/h5_cl10000_sl5000_s1250_regions"
    ),
    "ctx32768": Path(
        "./data/h5_cl16384_sl16384_s1250_regions"
    ),
    # Isoform-union labels: same geometry as s1250, junction columns widened to
    # every isoform clearing TSL<=2-or-NA. Carries no Y_REGION/Y_EXON -- exon and
    # intron are undefined per position once a gene contributes several isoforms.
    "union_tsl2na": Path(
        "./data/h5_cl10000_sl5000_s1250_union_tsl2na"
    ),
    "union_tsl2na_measured_ssu": Path(
        "./data/"
        "h5_cl10000_sl5000_s1250_union_tsl2na_measured_ssu"
    ),
    # GTEx-junction union: canonical sites plus both ends of every GTEx v8
    # junction seen in >= 5 donors. Same geometry, same missing region labels.
    "union_gtex_ind5": Path(
        "./data/h5_cl10000_sl5000_s1250_union_gtex_ind5"
    ),
    # The same rule on recount3's GTEx junctions, which include unannotated ones.
    "union_gtex_rc3_ind5": Path(
        "./data/h5_cl10000_sl5000_s1250_union_gtex_rc3_ind5"
    ),
    # s1250 with genome sequence outside the transcript instead of N
    # (flank_fill: real), the inference convention on the genome FASTA.
    "s1250_realflank": Path(
        "./data/h5_cl10000_sl5000_s1250_regions_realflank"
    ),
    # SpliceAI's GTEx-augmented table lifted to hg38: its own genes, bounds and
    # sites, real flanks, no region labels (ONBOARDING 4.95).
    "spliceai_gtex_realflank": Path(
        "./data/h5_cl10000_sl5000_s1250_spliceai_gtex_realflank"
    ),
}


def _sorted_chunk_ids(h5: h5py.File, prefix: str) -> list[int]:
    """Return chunk indices for keys like ``{prefix}{i}`` sorted ascending.

    Args:
        h5 (h5py.File): Open file handle.
        prefix (str): Key prefix to match (e.g., ``"X"``, ``"Y_REGION"``).

    Returns:
        list[int]: Sorted chunk indices.
    """
    ids: list[int] = []
    for k in h5.keys():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        if not suffix.isdigit():
            continue
        ids.append(int(suffix))
    return sorted(ids)


def merge_one_variant(
    src_dir: Path,
    *,
    force: bool,
    logger: logging.Logger,
    test_paralog: str = "0",
) -> Path:
    """Concat ``dataset_train_all.h5`` + ``dataset_test_{paralog}.h5`` for one variant.

    Args:
        src_dir (Path): Directory holding the source H5 files.
        force (bool): Overwrite an existing all-chroms file when True.
        logger (logging.Logger): Logger for progress messages.
        test_paralog (str): Paralog filter of the test-chromosome half.
            ``"0"`` (SpliceAI convention: genes without paralogs) writes
            ``dataset_all_chroms.h5``; ``"all"`` keeps every chr1/3/5/7/9 gene
            and writes ``dataset_all_chroms_allgenes.h5``.

    Returns:
        Path: Path to the created all-chroms H5.

    Raises:
        FileNotFoundError: If either source H5 is missing.
        ValueError: If the two source files have mismatched key prefixes.
    """
    train_path = src_dir / "dataset_train_all.h5"
    test_path = src_dir / f"dataset_test_{test_paralog}.h5"
    out_path = src_dir / (
        "dataset_all_chroms.h5"
        if test_paralog == "0"
        else "dataset_all_chroms_allgenes.h5"
    )

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing source train H5: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing source test H5: {test_path}")

    if out_path.is_file() and not force:
        logger.info(f"⏭  skip (already exists): {out_path}")
        return out_path

    tmp_path = out_path.with_suffix(".h5.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info(f"📦 merging:")
    logger.info(f"    train: {train_path}")
    logger.info(f"    test : {test_path}")
    logger.info(f"    out  : {out_path}")

    with (
        h5py.File(train_path, "r") as train_h5,
        h5py.File(test_path, "r") as test_h5,
        h5py.File(tmp_path, "w") as out_h5,
    ):
        train_prefixes = sorted({
            "".join(c for c in k if not c.isdigit()) for k in train_h5.keys()
        })
        test_prefixes = sorted({
            "".join(c for c in k if not c.isdigit()) for k in test_h5.keys()
        })
        if train_prefixes != test_prefixes:
            raise ValueError(
                f"Key prefixes differ between train and test H5: "
                f"train={train_prefixes} test={test_prefixes}"
            )
        prefixes = train_prefixes

        train_ids = _sorted_chunk_ids(train_h5, "X")
        test_ids = _sorted_chunk_ids(test_h5, "X")
        n_train = len(train_ids)
        n_test = len(test_ids)
        logger.info(
            f"   chunks: train={n_train} test={n_test} "
            f"-> total={n_train + n_test} (prefixes={prefixes})"
        )

        for new_id, old_id in enumerate(train_ids):
            for prefix in prefixes:
                src_key = f"{prefix}{old_id}"
                dst_key = f"{prefix}{new_id}"
                src_h5 = train_h5
                if src_key not in src_h5:
                    raise KeyError(
                        f"Missing key {src_key} in {train_path}"
                    )
                ds = src_h5[src_key]
                out_h5.create_dataset(
                    dst_key,
                    data=ds[:],
                    dtype=ds.dtype,
                    chunks=ds.chunks,
                    compression=ds.compression,
                    compression_opts=ds.compression_opts,
                    shuffle=ds.shuffle,
                )

        for offset, old_id in enumerate(test_ids):
            new_id = n_train + offset
            for prefix in prefixes:
                src_key = f"{prefix}{old_id}"
                dst_key = f"{prefix}{new_id}"
                if src_key not in test_h5:
                    raise KeyError(
                        f"Missing key {src_key} in {test_path}"
                    )
                ds = test_h5[src_key]
                out_h5.create_dataset(
                    dst_key,
                    data=ds[:],
                    dtype=ds.dtype,
                    chunks=ds.chunks,
                    compression=ds.compression,
                    compression_opts=ds.compression_opts,
                    shuffle=ds.shuffle,
                )

        out_h5.attrs["source_train"] = str(train_path)
        out_h5.attrs["source_test"] = str(test_path)
        out_h5.attrs["num_train_chunks"] = n_train
        out_h5.attrs["num_test_chunks"] = n_test
        out_h5.attrs["note"] = (
            f"Concatenation of dataset_train_all.h5 + {test_path.name} "
            "for all-chromosome training. Train chunks first "
            "[0..n_train-1], test chunks after [n_train..n_train+n_test-1]."
        )

    tmp_path.replace(out_path)
    logger.info(f"✅ wrote {out_path} ({out_path.stat().st_size / 2**30:.1f} GB)")
    return out_path


def main() -> int:
    """CLI entry point: merge selected variants and report timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS.keys(), "all"],
        default="all",
        help="Which stride variant to merge (default: all three).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset_all_chroms.h5 files.",
    )
    parser.add_argument(
        "--test-paralog",
        choices=["0", "all"],
        default="0",
        help=(
            "Test-chromosome half to append: 0 = dataset_test_0.h5 -> "
            "dataset_all_chroms.h5; all = dataset_test_all.h5 -> "
            "dataset_all_chroms_allgenes.h5."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("build_all_chroms_h5")

    targets = (
        list(VARIANTS.keys())
        if args.variant == "all"
        else [args.variant]
    )

    for name in targets:
        src_dir = VARIANTS[name]
        t0 = time.time()
        merge_one_variant(
            src_dir, force=args.force, logger=logger, test_paralog=args.test_paralog
        )
        logger.info(f"   ⏱  {name} took {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
