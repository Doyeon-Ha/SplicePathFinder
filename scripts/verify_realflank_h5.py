"""Verify a real-flank s1250 build against the genome and the N-flank build.

Per split (``dataset_{split}_all.h5``):

1. Labels: every ``Y{i}``, ``Y_REGION{i}`` and ``Y_EXON{i}`` equals the N-flank
   build's, so the genes, windows and targets are unchanged.
2. Body: ``X`` equals the N-flank ``X`` wherever the latter is not N.
3. Genome: for a sample of genes, every window of ``X`` equals the genome slice
   rebuilt here from the datafile coordinates -- strand, reverse complement and
   the downstream extension included -- independently of ``create_datapoints``.

The merged ``dataset_all_chroms_allgenes.h5`` must then hold the train chunks
followed by the test chunks.

    python -m scripts.verify_realflank_h5 \\
        --new-dir .../h5_cl10000_sl5000_s1250_regions_realflank \\
        --base-dir .../h5_cl10000_sl5000_s1250_regions --out-json verify.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from pyfaidx import Fasta

logger = logging.getLogger(__name__)

_ENCODE = np.zeros(256, dtype=np.int8)  # anything but ACGT stays 0 (N)
for _code, _base in enumerate("ACGT", start=1):
    _ENCODE[ord(_base)] = _code
    _ENCODE[ord(_base.lower())] = _code
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _n_points(tx_len: int, sl: int, stride: int) -> int:
    """Windows ``reformat_data`` emits for a transcript of ``tx_len`` bases."""
    return -(-max(tx_len - sl, 0) // stride) + 1


def _genome_slice(genome: Fasta, chrom: str, start: int, end: int) -> str:
    """Return 0-based ``[start, end)``, N-padded beyond the chromosome."""
    n = len(genome[chrom])
    left, right = max(0, -start), max(0, end - n)
    body = str(genome[chrom][max(start, 0) : min(end, n)])
    return "N" * left + body + "N" * right


def _expected_window(
    genome: Fasta,
    chrom: str,
    strand: str,
    tx_start: int,
    tx_end: int,
    window: int,
    half_cl: int,
    sl: int,
    stride: int,
) -> np.ndarray:
    """Encode window ``window`` of a transcript straight from the genome."""
    length = sl + 2 * half_cl
    if strand == "+":
        start = tx_start - half_cl - 1 + stride * window
        seq = _genome_slice(genome, chrom, start, start + length)
    else:
        end = tx_end + half_cl - stride * window
        seq = _genome_slice(genome, chrom, end - length, end)
        seq = seq.translate(_COMPLEMENT)[::-1]
    return _ENCODE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def verify_split(
    new_dir: Path,
    base_dir: Path,
    split: str,
    genome: Fasta,
    n_sample: int,
    seed: int,
    half_cl: int,
    sl: int,
    stride: int,
    chunk_size: int,
) -> dict:
    """Check one split; return counts and a list of failures."""
    failures: list[str] = []
    with (
        h5py.File(new_dir / f"datafile_{split}_all.h5", "r") as meta,
        h5py.File(new_dir / f"dataset_{split}_all.h5", "r") as new,
        h5py.File(base_dir / f"dataset_{split}_all.h5", "r") as base,
    ):
        chrom = meta["CHROM"][:].astype(str)
        strand = meta["STRAND"][:].astype(str)
        tx_start = meta["TX_START"][:].astype(int)
        tx_end = meta["TX_END"][:].astype(int)
        n_genes = len(chrom)
        n_chunks = max(n_genes // chunk_size, 1)
        n_x = sum(1 for k in new if k.startswith("X"))
        if n_x != n_chunks:
            failures.append(f"{split}: {n_x} X chunks, expected {n_chunks}")

        sample = set(
            np.random.default_rng(seed)
            .choice(n_genes, size=min(n_sample, n_genes), replace=False)
            .tolist()
        )
        filled = still_n = windows_checked = 0
        for i in range(n_chunks):
            for prefix in ("Y", "Y_REGION", "Y_EXON"):
                if not np.array_equal(new[f"{prefix}{i}"][:], base[f"{prefix}{i}"][:]):
                    failures.append(
                        f"{split}: {prefix}{i} differs from the N-flank build"
                    )
            x_new, x_base = new[f"X{i}"][:], base[f"X{i}"][:]
            body = x_base != 0
            if x_new.shape != x_base.shape or not np.array_equal(
                x_new[body], x_base[body]
            ):
                failures.append(f"{split}: X{i} transcript body differs")
                continue
            filled += int(np.count_nonzero((~body) & (x_new != 0)))
            still_n += int(np.count_nonzero((~body) & (x_new == 0)))

            first = i * chunk_size
            last = n_genes if i == n_chunks - 1 else first + chunk_size
            row = 0
            for g in range(first, last):
                n_win = _n_points(tx_end[g] - tx_start[g] + 1, sl, stride)
                if g in sample:
                    for w in range(n_win):
                        expected = _expected_window(
                            genome,
                            chrom[g],
                            strand[g],
                            tx_start[g],
                            tx_end[g],
                            w,
                            half_cl,
                            sl,
                            stride,
                        )
                        if not np.array_equal(x_new[row + w], expected):
                            failures.append(
                                f"{split}: gene {g} ({chrom[g]}{strand[g]}) window "
                                f"{w} differs from the genome"
                            )
                        windows_checked += 1
                row += n_win
            if row != x_new.shape[0]:
                failures.append(
                    f"{split}: X{i} has {x_new.shape[0]} rows, expected {row}"
                )
        logger.info(
            "%s: %d genes, %d chunks, %d windows checked against the genome, "
            "%d flank bases filled, %d still N, %d failures",
            split,
            n_genes,
            n_chunks,
            windows_checked,
            filled,
            still_n,
            len(failures),
        )
    return {
        "genes": n_genes,
        "chunks": n_chunks,
        "genes_sampled": len(sample),
        "windows_checked_against_genome": windows_checked,
        "flank_bases_filled": filled,
        "flank_bases_still_n": still_n,
        "failures": failures,
    }


def verify_merged(new_dir: Path, n_train: int, n_test: int) -> list[str]:
    """Check that the merged file is the train chunks, then the test chunks."""
    failures: list[str] = []
    with (
        h5py.File(new_dir / "dataset_all_chroms_allgenes.h5", "r") as merged,
        h5py.File(new_dir / "dataset_train_all.h5", "r") as train,
        h5py.File(new_dir / "dataset_test_all.h5", "r") as test,
    ):
        n_x = sum(1 for k in merged if k.startswith("X"))
        if n_x != n_train + n_test:
            failures.append(f"merged: {n_x} X chunks, expected {n_train + n_test}")
        pairs = [(0, train, 0), (n_train - 1, train, n_train - 1)]
        pairs += [(n_train, test, 0), (n_train + n_test - 1, test, n_test - 1)]
        for merged_i, src, src_i in pairs:
            if not np.array_equal(merged[f"X{merged_i}"][:], src[f"X{src_i}"][:]):
                failures.append(f"merged: X{merged_i} differs from its source chunk")
    return failures


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument(
        "--genome",
        type=Path,
        default=Path("./genome/GRCh38.fa"),
    )
    parser.add_argument("--genes-per-split", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cl-max", type=int, default=10000)
    parser.add_argument("--sl", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=1250)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )

    genome = Fasta(str(args.genome))
    report = {
        split: verify_split(
            args.new_dir,
            args.base_dir,
            split,
            genome,
            args.genes_per_split,
            args.seed,
            args.cl_max // 2,
            args.sl,
            args.stride,
            args.chunk_size,
        )
        for split in ("train", "test")
    }
    report["merged_failures"] = verify_merged(
        args.new_dir, report["train"]["chunks"], report["test"]["chunks"]
    )
    n_fail = (
        len(report["train"]["failures"])
        + len(report["test"]["failures"])
        + len(report["merged_failures"])
    )
    report["ok"] = n_fail == 0
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s (ok=%s, %d failures)", args.out_json, report["ok"], n_fail)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
