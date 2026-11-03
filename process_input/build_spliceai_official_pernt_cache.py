"""Build an OFFICIAL Illumina SpliceAI per-nt probability cache for the learned
SAV head (ClinVar base-swap; extends spec 018 to the headline spliceai_illumina).

Unlike ``build_spliceai_pernt_cache.py`` (self-trained SpliceAI-10k, pytorch),
this runs the **official Illumina 5-model TF/Keras ensemble** (Jaganathan 2019
weights, `spliceai` package) per-nt. Reuses the pre-built variant-centered
one-hot sequences ``X_ref``/``X_alt`` from the gencode-anchored H5 (the SAME
sequences the med_ac cache used → **row-aligned** to `splits_seed42`, so the
identical gene-grouped split trains/evaluates the head).

Key decisions:
- **Input = anchored ``X_ref``/``X_alt``** (int8, A=1/C=2/G=3/T=4/N=0) → this is
  exactly SpliceAI's ``IN_MAP`` convention (verified) → one-hot via IN_MAP.
- **15,000 input → central 5,000 output** (official model context 10,000;
  variant at input idx 7,500 → output idx 2,500 = center), voff=2,500. Matches
  the SpliceAI-10k / med_ac ±2,500 window (DD-001).
- **Channel reorder [neither,acc,don] → [acc,don,neither]** ([1,2,0]) to match
  the head convention (ch0=acc, ch1=don) used by all other base caches.
- 5-model ensemble mean. Runs in the official env
  (`./external/spliceai/env`, TF/Keras + `spliceai`).

Run (official env):
    ENVPY=./external/spliceai/env/bin/python
    $ENVPY -m process_input.build_spliceai_official_pernt_cache \\
        --anchored-h5 <clinvar_..._gencode_anchored_indel.h5> \\
        --out-h5 <.../clinvar_20260628_spliceai_official/cache.h5> \\
        --shard-index K --num-shards N --batch-size 32
"""

from __future__ import annotations

import argparse
import glob
import logging

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TOTAL_LENGTH: int = 15000
TARGET_LENGTH: int = 5000
VOFF: int = TARGET_LENGTH // 2  # 2500
# SpliceAI one-hot: N=0, A=1, C=2, G=3, T=4 (matches anchored X_ref int8 encoding).
IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _decode(arr: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _shard_range(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    per = -(-n // num_shards)
    start = shard_index * per
    return start, min(n, start + per)


def _load_official_models():
    """Load the 5 official Illumina SpliceAI Keras models."""
    from pkg_resources import resource_filename

    try:
        from tensorflow.keras.models import load_model
    except Exception:  # noqa: BLE001
        from keras.models import load_model
    paths = [resource_filename("spliceai", f"models/spliceai{i}.h5") for i in range(1, 6)]
    models = [load_model(p) for p in paths]
    logger.info("loaded %d official SpliceAI models", len(models))
    return models


def _forward(models, x_int: np.ndarray) -> np.ndarray:
    """5-model ensemble mean per-nt probs, reordered to [acc,don,neither].

    Args:
        models: list of 5 Keras models.
        x_int (np.ndarray): (B, 15000) int8 seq (A=1..T=4, N=0).

    Returns:
        np.ndarray: (B, 5000, 3) float32, channels [acc, don, neither].
    """
    oh = IN_MAP[x_int]  # (B, 15000, 4)
    ys = [m.predict(oh, verbose=0) for m in models]  # each (B, 5000, 3) [nei,acc,don]
    y = np.mean(ys, axis=0)
    return y[..., [1, 2, 0]]  # -> [acc, don, neither]


def _run_shard(args: argparse.Namespace) -> None:
    with h5py.File(args.anchored_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        pos = f["pos"][:].astype(np.int64)
        chrom = _decode(f["chrom"][:])
        ref = _decode(f["ref"][:])
        alt = _decode(f["alt"][:])
        label = f["label"][:].astype(np.float32)
        gene = _decode(f["gene_symbol"][:])

    start, end = _shard_range(n_total, args.shard_index, args.num_shards)
    n = end - start
    logger.info("shard %d/%d rows [%d,%d) n=%d", args.shard_index, args.num_shards, start, end, n)
    kind = ["snv" if (len(ref[i]) == 1 and len(alt[i]) == 1) else "indel" for i in range(start, end)]

    models = _load_official_models()
    out_path = (
        args.out_h5 if args.num_shards == 1
        else f"{args.out_h5}.shard{args.shard_index}_of{args.num_shards}.h5"
    )
    with h5py.File(args.anchored_h5, "r") as f, h5py.File(out_path, "w") as out:
        Xr = f["X_ref"]; Xa = f["X_alt"]
        dref = out.create_dataset(
            "ref_probs", (n, TARGET_LENGTH, 3), dtype="float16",
            chunks=(1, TARGET_LENGTH, 3), compression="gzip", compression_opts=1,
        )
        dalt = out.create_dataset(
            "alt_probs", (n, TARGET_LENGTH, 3), dtype="float16",
            chunks=(1, TARGET_LENGTH, 3), compression="gzip", compression_opts=1,
        )
        keys = np.array(
            [f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}" for i in range(start, end)],
            dtype=object,
        )
        bs = args.batch_size
        for b0 in range(0, n, bs):
            b1 = min(n, b0 + bs)
            xr = Xr[start + b0 : start + b1].astype(np.int64)
            xa = Xa[start + b0 : start + b1].astype(np.int64)
            dref[b0:b1] = _forward(models, xr).astype(np.float16)
            dalt[b0:b1] = _forward(models, xa).astype(np.float16)
            if (b1) % 2000 < bs:
                logger.info("  %d/%d", b1, n)
        out.create_dataset("variant_offset", data=np.full(n, VOFF, dtype=np.int64))
        out.create_dataset("labels", data=label[start:end])
        out.create_dataset("gene_symbol", data=np.array(gene[start:end], dtype="S"))
        out.create_dataset("variant_kind", data=np.array(kind, dtype="S"))
        out.create_dataset("source_row_key", data=keys.astype("S"))
        out.create_dataset("row_index", data=np.arange(start, end, dtype=np.int64))
    logger.info("shard done -> %s", out_path)


def _merge(args: argparse.Namespace) -> None:
    shards = sorted(glob.glob(args.shard_glob))
    if not shards:
        raise FileNotFoundError(f"no shards match {args.shard_glob}")
    with h5py.File(args.anchored_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        chrom = _decode(f["chrom"][:]); pos = f["pos"][:].astype(np.int64)
        ref = _decode(f["ref"][:]); alt = _decode(f["alt"][:])
        gene = _decode(f["gene_symbol"][:])
    kind = ["snv" if (len(ref[i]) == 1 and len(alt[i]) == 1) else "indel" for i in range(n_total)]
    with h5py.File(args.out_h5, "w") as out:
        dref = out.create_dataset(
            "ref_probs", (n_total, TARGET_LENGTH, 3), dtype="float16",
            chunks=(1, TARGET_LENGTH, 3), compression="gzip", compression_opts=1,
        )
        dalt = out.create_dataset(
            "alt_probs", (n_total, TARGET_LENGTH, 3), dtype="float16",
            chunks=(1, TARGET_LENGTH, 3), compression="gzip", compression_opts=1,
        )
        labels = np.zeros(n_total, dtype=np.float32)
        keys = np.empty(n_total, dtype=object)
        for sp in shards:
            with h5py.File(sp, "r") as s:
                ri = s["row_index"][:]
                dref[ri[0] : ri[-1] + 1] = s["ref_probs"][:]
                dalt[ri[0] : ri[-1] + 1] = s["alt_probs"][:]
                labels[ri[0] : ri[-1] + 1] = s["labels"][:]
                for k, gi in enumerate(ri):
                    keys[gi] = s["source_row_key"][k].decode()
        for i in range(n_total):  # row-order assert
            exp = f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}"
            assert keys[i] == exp, f"row {i}: {keys[i]} != {exp}"
        out.create_dataset("variant_offset", data=np.full(n_total, VOFF, dtype=np.int64))
        out.create_dataset("labels", data=labels)
        out.create_dataset("gene_symbol", data=np.array(gene, dtype="S"))
        out.create_dataset("variant_kind", data=np.array(kind, dtype="S"))
        out.create_dataset("source_row_key", data=np.array(keys, dtype="S"))
    logger.info("merged %d shards -> %s (row-order verified)", len(shards), args.out_h5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OFFICIAL SpliceAI per-nt cache.")
    p.add_argument("--anchored-h5", required=True, dest="anchored_h5",
                   help="gencode-anchored H5 with X_ref/X_alt (med_ac cache source)")
    p.add_argument("--out-h5", required=True, dest="out_h5")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--shard-glob", default="", dest="shard_glob")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        _merge(args)
    else:
        _run_shard(args)


if __name__ == "__main__":
    main()
