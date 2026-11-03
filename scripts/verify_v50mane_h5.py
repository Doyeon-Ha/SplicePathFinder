"""Verify the GENCODE v50 MANE-first real-flank build against the v49 build.

Per split (``dataset_{split}_all.h5``):

1. Unchanged genes: a gene whose datafile row (coordinates, junctions, exons,
   CDS) equals the v49 real-flank build's row must have bit-identical windows
   in ``X``, ``Y``, ``Y_REGION`` and ``Y_EXON``. A gene whose rows differ only
   in the CDS phase must match in everything but ``Y_REGION``. ``X`` equality
   also checks that the GENCODE v50 FASTA encodes like the NCBI FASTA.
2. Changed and new genes: every window of ``X`` equals the genome slice rebuilt
   from the datafile coordinates, and every window of ``Y`` equals the splice
   labels rebuilt from the table junctions (padding -1 past the transcript).

The merged ``dataset_all_chroms_allgenes.h5`` must hold the train chunks
followed by the test chunks.

    python -m scripts.verify_v50mane_h5 \\
        --new-dir .../h5_cl10000_sl5000_s1250_regions_realflank_v50mane \\
        --old-dir .../h5_cl10000_sl5000_s1250_regions_realflank \\
        --genome .../genome/gencode_v50/GRCh38.primary_assembly.genome.fa \\
        --out-json .../verify_v50mane.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from pyfaidx import Fasta

from scripts.verify_realflank_h5 import _expected_window, _n_points, verify_merged

logger = logging.getLogger(__name__)

ROW_KEYS = (
    "CHROM",
    "STRAND",
    "TX_START",
    "TX_END",
    "JN_START",
    "JN_END",
    "EXON_START",
    "EXON_END",
    "CDS_START",
    "CDS_END",
)


def _text(values: np.ndarray) -> list[str]:
    """Decode an h5py column to str (JN_* rows are stored as 1-element lists)."""
    out = []
    for v in values:
        v = v[0] if isinstance(v, np.ndarray) else v
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def _load_meta(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return gene names and the per-gene columns of a datafile."""
    with h5py.File(path, "r") as f:
        names = _text(f["NAME"][:])
        cols = {k: _text(f[k][:]) for k in (*ROW_KEYS, "CDS_FRAME")}
    return names, cols


def _layout(cols: dict[str, list[str]], sl: int, stride: int, chunk_size: int):
    """Map gene index -> (chunk, first row, n windows) for a dataset file."""
    n_genes = len(cols["CHROM"])
    n_chunks = max(n_genes // chunk_size, 1)
    out, row, chunk = [], 0, 0
    for g in range(n_genes):
        c = min(g // chunk_size, n_chunks - 1)
        if c != chunk:
            chunk, row = c, 0
        n = _n_points(int(cols["TX_END"][g]) - int(cols["TX_START"][g]) + 1, sl, stride)
        out.append((c, row, n))
        row += n
    return out, n_chunks


def _expected_labels(cols: dict[str, list[str]], g: int, sl: int, stride: int):
    """Splice labels of every window of gene ``g`` rebuilt from its junctions."""
    s, e = int(cols["TX_START"][g]), int(cols["TX_END"][g])
    tx_len = e - s + 1
    ends = [int(x) for x in cols["JN_START"][g].split(",") if x]
    starts = [int(x) for x in cols["JN_END"][g].split(",") if x]
    lab = np.zeros(tx_len, dtype=np.int8)
    if cols["STRAND"][g] == "+":
        lab[[c - s for c in ends]] = 2
        lab[[c - s for c in starts]] = 1
    else:
        lab[[e - c for c in starts]] = 2
        lab[[e - c for c in ends]] = 1
    n = _n_points(tx_len, sl, stride)
    padded = np.concatenate([lab, np.full(sl, -1, dtype=np.int8)])
    return np.stack([padded[w * stride : w * stride + sl] for w in range(n)])


class _ChunkCache:
    """Keep the most recently read chunk of each dataset in memory."""

    def __init__(self, h5: h5py.File):
        self.h5, self.key, self.arrays = h5, None, {}

    def get(self, chunk: int) -> dict[str, np.ndarray]:
        if self.key != chunk:
            self.arrays = {
                p: self.h5[f"{p}{chunk}"][:] for p in ("X", "Y", "Y_REGION", "Y_EXON")
            }
            self.key = chunk
        return self.arrays


def verify_split(
    new_dir: Path,
    old_dir: Path,
    split: str,
    genome: Fasta,
    half_cl: int,
    sl: int,
    stride: int,
    chunk_size: int,
) -> dict:
    """Check one split; return counts and a list of failures."""
    failures: list[str] = []
    new_names, new_cols = _load_meta(new_dir / f"datafile_{split}_all.h5")
    old_names, old_cols = _load_meta(old_dir / f"datafile_{split}_all.h5")
    new_lay, n_new_chunks = _layout(new_cols, sl, stride, chunk_size)
    old_lay, _ = _layout(old_cols, sl, stride, chunk_size)
    old_index = {n: i for i, n in enumerate(old_names)}
    counts = {"unchanged": 0, "phase_only": 0, "changed_or_new": 0}
    windows = {"compared_with_v49": 0, "rebuilt_from_genome": 0}
    with (
        h5py.File(new_dir / f"dataset_{split}_all.h5", "r") as new,
        h5py.File(old_dir / f"dataset_{split}_all.h5", "r") as old,
    ):
        n_x = sum(1 for k in new if k.startswith("X"))
        if n_x != n_new_chunks:
            failures.append(f"{split}: {n_x} X chunks, expected {n_new_chunks}")
        new_cache, old_cache = _ChunkCache(new), _ChunkCache(old)
        rows_per_chunk: dict[int, int] = {}
        for g, name in enumerate(new_names):
            c, r, n = new_lay[g]
            rows_per_chunk[c] = rows_per_chunk.get(c, 0) + n
            arrays = new_cache.get(c)
            o = old_index.get(name)
            same_row = o is not None and all(
                new_cols[k][g] == old_cols[k][o] for k in ROW_KEYS
            )
            if same_row:
                phase_only = new_cols["CDS_FRAME"][g] != old_cols["CDS_FRAME"][o]
                counts["phase_only" if phase_only else "unchanged"] += 1
                oc, orow, on = old_lay[o]
                old_arrays = old_cache.get(oc)
                prefixes = (
                    ("X", "Y", "Y_EXON")
                    if phase_only
                    else ("X", "Y", "Y_REGION", "Y_EXON")
                )
                for p in prefixes:
                    if not np.array_equal(
                        arrays[p][r : r + n], old_arrays[p][orow : orow + on]
                    ):
                        failures.append(f"{split}: {name} {p} differs from v49 build")
                windows["compared_with_v49"] += n
                continue
            counts["changed_or_new"] += 1
            labels = _expected_labels(new_cols, g, sl, stride)
            if not np.array_equal(arrays["Y"][r : r + n], labels):
                failures.append(f"{split}: {name} Y differs from its junctions")
            for w in range(n):
                expected = _expected_window(
                    genome,
                    new_cols["CHROM"][g],
                    new_cols["STRAND"][g],
                    int(new_cols["TX_START"][g]),
                    int(new_cols["TX_END"][g]),
                    w,
                    half_cl,
                    sl,
                    stride,
                )
                if not np.array_equal(arrays["X"][r + w], expected):
                    failures.append(f"{split}: {name} window {w} differs from genome")
            windows["rebuilt_from_genome"] += n
        for c, rows in rows_per_chunk.items():
            if new[f"X{c}"].shape[0] != rows:
                failures.append(
                    f"{split}: X{c} has {new[f'X{c}'].shape[0]} rows, expected {rows}"
                )
    total_windows = sum(n for _, _, n in new_lay)
    logger.info(
        "%s: %d genes %s, %d windows (%s), %d failures",
        split,
        len(new_names),
        counts,
        total_windows,
        windows,
        len(failures),
    )
    return {
        "genes": len(new_names),
        "chunks": n_new_chunks,
        "windows": total_windows,
        **counts,
        **{f"windows_{k}": v for k, v in windows.items()},
        "failures": failures,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--cl-max", type=int, default=10000)
    parser.add_argument("--sl", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=1250)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )

    genome = Fasta(args.genome)
    result = {}
    for split in ("train", "test"):
        result[split] = verify_split(
            args.new_dir,
            args.old_dir,
            split,
            genome,
            args.cl_max // 2,
            args.sl,
            args.stride,
            args.chunk_size,
        )
    merged = verify_merged(
        args.new_dir, result["train"]["chunks"], result["test"]["chunks"]
    )
    n_fail = sum(len(r["failures"]) for r in result.values()) + len(merged)
    result["merged_failures"] = merged
    result["ok"] = n_fail == 0
    args.out_json.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s (ok=%s, %d failures)", args.out_json, result["ok"], n_fail)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
