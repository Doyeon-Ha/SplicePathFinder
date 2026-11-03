"""Build a per-nt feature sidecar H5 aligned to the SpliceVarDB full H5.

For every variant row of the gencode-anchored SpliceVarDB H5 (X_ref/X_alt,
length 15000, in gene reading orientation), emit two 2-channel per-position
feature tracks — intron-structure (nearest-splice-site distance + segment
length) and conservation (phyloP + phastCons) — **row-aligned and
position-aligned** to X_ref, so the learned SAV head can concat them to its
per-nt probability channels (spec 017 feature-twin).

Window→genome mapping reuses the 004 gencode-anchored builder's convention
(``splicevardb_gencode_anchored_allchroms_indel_builder``): with the variant at
model-space index ``voff`` and 1-based genomic position ``pos``, the + strand
0-based window start is ``pos - voff - 1``; the minus strand + strand window
start is ``pos + voff - TOTAL_LENGTH``. ``intron_feature_for_chunk`` /
``ConservationReader.window`` then reverse the minus-strand array to align with
the reverse-complemented sequence the model receives.

Alignment is asserted (FR-002): the genomic position landing at model-index
``voff`` must equal the variant ``pos``, and ``source_row_key`` order must match
the source H5 row order — a mismatch aborts the build (silent misalignment
would corrupt every downstream label).
"""

from __future__ import annotations

import argparse
import glob
import logging

import h5py
import numpy as np

from process_input.conservation import CONSERVATION_CHANNELS, ConservationReader
from process_input.intron_feature import (
    INTRON_FEATURE_CHANNELS,
    intron_feature_for_chunk,
    load_gene_sites_lookup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TOTAL_LENGTH: int = 15000


def _decode(arr: np.ndarray) -> list[str]:
    """Decode an object/bytes H5 column to a list of str.

    Args:
        arr (np.ndarray): object or bytes array.

    Returns:
        list[str]: decoded strings.
    """
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _chunk_start_0b(pos_1b: int, voff: int, strand: str) -> int:
    """Compute the 0-based + strand genomic start of the model window.

    Args:
        pos_1b (int): variant 1-based genomic position.
        voff (int): variant model-space offset within the window.
        strand (str): ``'+'`` or ``'-'`` (canonical gene strand).

    Returns:
        int: 0-based + strand start of the 15000-bp window.
    """
    if strand == "+":
        return pos_1b - voff - 1
    return pos_1b + voff - TOTAL_LENGTH


def _gpos_at_offset(chunk_start_0b: int, voff: int, strand: str) -> int:
    """Genomic 1-based position that lands at model-index ``voff``.

    Used by the alignment assertion; must equal the variant ``pos``.

    Args:
        chunk_start_0b (int): 0-based + strand window start.
        voff (int): variant model-space offset.
        strand (str): ``'+'`` or ``'-'``.

    Returns:
        int: reconstructed 1-based genomic position at model-index ``voff``.
    """
    if strand == "+":
        return chunk_start_0b + 1 + voff
    return chunk_start_0b + 1 + (TOTAL_LENGTH - 1 - voff)


def _shard_range(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Contiguous [start, end) row range for a shard.

    Args:
        n (int): total rows.
        shard_index (int): 0-based shard index.
        num_shards (int): shard count.

    Returns:
        tuple[int, int]: (start, end) row indices.
    """
    per = -(-n // num_shards)
    start = shard_index * per
    end = min(n, start + per)
    return start, end


def _run_shard(args: argparse.Namespace) -> None:
    """Build one shard (or the whole file) of the feature sidecar.

    Args:
        args (argparse.Namespace): parsed CLI args.
    """
    with h5py.File(args.variant_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        pos = f["pos"][:].astype(np.int64)
        voff = f["variant_offset"][:].astype(np.int64)
        strand = _decode(f["strand"][:])
        chrom = _decode(f["chrom"][:])
        ref = _decode(f["ref"][:])
        alt = _decode(f["alt"][:])
        gene = _decode(f["source_gene_name"][:])

    start, end = _shard_range(n_total, args.shard_index, args.num_shards)
    n = end - start
    logger.info(
        "shard %d/%d: rows [%d, %d) (n=%d)",
        args.shard_index, args.num_shards, start, end, n,
    )

    lut = load_gene_sites_lookup(args.gene_sites)
    cons = ConservationReader(
        f"{args.conservation_dir}/hg38.phyloP100way.bw",
        f"{args.conservation_dir}/hg38.phastCons100way.bw",
        phylop_clip=args.phylop_clip,
    )

    if args.num_shards == 1:
        out_path = args.out_h5
    else:
        out_path = f"{args.out_h5}.shard{args.shard_index}_of{args.num_shards}.h5"
    empty = np.zeros(0, dtype=np.int64)
    n_missing_gene = 0
    with h5py.File(out_path, "w") as out:
        di = out.create_dataset(
            "feat_intron",
            (n, TOTAL_LENGTH, INTRON_FEATURE_CHANNELS),
            dtype="float16",
            chunks=(1, TOTAL_LENGTH, INTRON_FEATURE_CHANNELS),
            compression="gzip",
            compression_opts=1,
        )
        dc = out.create_dataset(
            "feat_conservation",
            (n, TOTAL_LENGTH, CONSERVATION_CHANNELS),
            dtype="float16",
            chunks=(1, TOTAL_LENGTH, CONSERVATION_CHANNELS),
            compression="gzip",
            compression_opts=1,
        )
        keys = np.empty(n, dtype=object)
        row_index = np.arange(start, end, dtype=np.int64)
        for j, i in enumerate(range(start, end)):
            cs = _chunk_start_0b(int(pos[i]), int(voff[i]), strand[i])
            # FR-002 alignment assertion (per row — cheap, catches any drift).
            assert _gpos_at_offset(cs, int(voff[i]), strand[i]) == int(pos[i]), (
                f"row {i}: gpos@voff != pos ({strand[i]})"
            )
            sites = lut.get(gene[i], empty)
            if sites.size == 0:
                n_missing_gene += 1
            intron = intron_feature_for_chunk(cs, TOTAL_LENGTH, strand[i], sites)
            di[j] = intron.astype(np.float16)
            consv = cons.window(chrom[i], cs, TOTAL_LENGTH, strand[i])
            dc[j] = consv.astype(np.float16)
            keys[j] = f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}"
            if (j + 1) % 2000 == 0:
                logger.info("  %d/%d", j + 1, n)
        out.create_dataset("source_row_key", data=keys.astype("S"))
        out.create_dataset("row_index", data=row_index)
    cons.close()
    logger.info(
        "shard done -> %s (gene-sites missing: %d/%d)", out_path, n_missing_gene, n
    )


def _merge(args: argparse.Namespace) -> None:
    """Merge shard files into the final sidecar H5 (row-index ordered).

    Args:
        args (argparse.Namespace): parsed CLI args (uses ``shard_glob``,
            ``out_h5``, ``variant_h5``).
    """
    shards = sorted(glob.glob(args.shard_glob))
    if not shards:
        raise FileNotFoundError(f"no shards match {args.shard_glob}")
    with h5py.File(args.variant_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        chrom = _decode(f["chrom"][:])
        pos = f["pos"][:].astype(np.int64)
        ref = _decode(f["ref"][:])
        alt = _decode(f["alt"][:])
    with h5py.File(args.out_h5, "w") as out:
        di = out.create_dataset(
            "feat_intron", (n_total, TOTAL_LENGTH, INTRON_FEATURE_CHANNELS),
            dtype="float16", chunks=(1, TOTAL_LENGTH, INTRON_FEATURE_CHANNELS),
            compression="gzip", compression_opts=1,
        )
        dc = out.create_dataset(
            "feat_conservation", (n_total, TOTAL_LENGTH, CONSERVATION_CHANNELS),
            dtype="float16", chunks=(1, TOTAL_LENGTH, CONSERVATION_CHANNELS),
            compression="gzip", compression_opts=1,
        )
        keys = np.empty(n_total, dtype=object)
        for sp in shards:
            with h5py.File(sp, "r") as s:
                ri = s["row_index"][:]
                di[ri[0] : ri[-1] + 1] = s["feat_intron"][:]
                dc[ri[0] : ri[-1] + 1] = s["feat_conservation"][:]
                for k, gi in enumerate(ri):
                    keys[gi] = s["source_row_key"][k].decode()
        # FR-002: row-order must match the source H5 exactly.
        for i in range(n_total):
            exp = f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}"
            assert keys[i] == exp, f"row {i}: key {keys[i]} != {exp}"
        out.create_dataset("source_row_key", data=np.array(keys, dtype="S"))
    logger.info("merged %d shards -> %s (row-order verified)", len(shards), args.out_h5)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed args.
    """
    p = argparse.ArgumentParser(
        description="Build feature sidecar (intron + conservation) for SpliceVarDB H5."
    )
    p.add_argument("--variant-h5", required=True, dest="variant_h5")
    p.add_argument("--conservation-dir", required=True, dest="conservation_dir")
    p.add_argument("--gene-sites", required=True, dest="gene_sites")
    p.add_argument("--out-h5", required=True, dest="out_h5")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--shard-glob", default="", dest="shard_glob")
    p.add_argument("--phylop-clip", type=float, default=10.0, dest="phylop_clip")
    return p.parse_args()


def main() -> None:
    """Entry point: build a shard or merge shards."""
    args = parse_args()
    if args.merge:
        _merge(args)
    else:
        _run_shard(args)


if __name__ == "__main__":
    main()
