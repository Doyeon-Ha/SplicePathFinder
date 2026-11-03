"""Per-nucleotide splice-site ANNOTATION eval for OFFICIAL Illumina SpliceAI
on a chunked X{i}/Y{i} test H5 (e.g. chr1,3,5,7,9 held-out test set).

This is the annotation (per-position acceptor/donor) task — NOT variant delta
scoring. It mirrors evaluate/evaluate_spliceai_v1_chunked_h5.py (same metric:
per-class AUPRC + top-k = precision @ k=pos_count) but runs the official
Illumina 5-model TF/Keras ensemble (Jaganathan 2019 weights, `spliceai`
package) instead of our self-trained SpliceAI-v1. chr1,3,5,7,9 is SpliceAI's
own held-out test split, so this is the classic apples-to-apples benchmark.

Model I/O (reused from process_input/build_spliceai_official_pernt_cache.py):
- input X{i}: int8 (N=0,A=1,C=2,G=3,T=4) → one-hot via IN_MAP.
- official model: 15,000 input context 10,000 → central 5,000 output; native
  channels [neither,acc,don] reordered to [acc,don,neither] ([1,2,0]).
- labels Y{i}: 0=neither,1=acceptor,2=donor,-1=pad (pad excluded).

Because top-k is a GLOBAL ranking over all positions, sharded runs dump their
valid (acc_score, don_score, label) arrays to per-shard .npz; a final --merge
pass concatenates all shards and computes the global metrics.

Run (official env ./external/spliceai/env), one shard per MIG job:
    ENVPY=./external/spliceai/env/bin/python
    $ENVPY scripts/eval_spliceai_official_annotation.py \\
        --test-h5-path .../dataset_test_0.h5 \\
        --out-prefix output/spliceai_official_annotation/test0 \\
        --shard-index K --num-shards N --batch-size 32
Then merge:
    $ENVPY scripts/eval_spliceai_official_annotation.py \\
        --test-h5-path .../dataset_test_0.h5 \\
        --out-prefix output/spliceai_official_annotation/test0 \\
        --merge --num-shards N
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TOTAL_LENGTH: int = 15000
TARGET_LENGTH: int = 5000
IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
_LABEL_ACCEPTOR: int = 1
_LABEL_DONOR: int = 2


def _load_official_models():
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
    """5-model ensemble mean per-nt probs, central 5000, [acc,don,neither]."""
    oh = IN_MAP[x_int]  # (B, 15000, 4)
    ys = [m.predict(oh, verbose=0) for m in models]  # each (B, 5000, 3) [nei,acc,don]
    y = np.mean(ys, axis=0)
    return y[..., [1, 2, 0]]  # -> [acc, don, neither]


def _chunk_suffixes(h5_path: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        xs = [k[1:] for k in f.keys() if k.startswith("X") and k[1:].isdigit()]
        xs = [s for s in xs if f"Y{s}" in f]
    return sorted(xs, key=int)


def _shard_bounds(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    per = -(-n // num_shards)
    start = shard_index * per
    return start, min(n, start + per)


def _run_shard(args: argparse.Namespace) -> None:
    suffixes = _chunk_suffixes(args.test_h5_path)
    with h5py.File(args.test_h5_path, "r") as f:
        sizes = [int(f[f"X{s}"].shape[0]) for s in suffixes]
    n_total = sum(sizes)
    g_start, g_end = _shard_bounds(n_total, args.shard_index, args.num_shards)
    logger.info(
        "shard %d/%d global rows [%d,%d) of %d",
        args.shard_index, args.num_shards, g_start, g_end, n_total,
    )

    models = _load_official_models()
    acc_sc: list[np.ndarray] = []
    don_sc: list[np.ndarray] = []
    lab: list[np.ndarray] = []

    bs = args.batch_size
    with h5py.File(args.test_h5_path, "r") as f:
        base = 0
        for s, sz in zip(suffixes, sizes):
            c0, c1 = max(g_start, base), min(g_end, base + sz)
            if c0 < c1:
                x_ds, y_ds = f[f"X{s}"], f[f"Y{s}"]
                for b0 in range(c0 - base, c1 - base, bs):
                    b1 = min(c1 - base, b0 + bs)
                    x = x_ds[b0:b1].astype(np.int64)
                    y = y_ds[b0:b1]
                    probs = _forward(models, x)  # (B,5000,3) [acc,don,nei]
                    valid = y >= 0
                    yv = y[valid]
                    acc_sc.append(probs[..., 0][valid].astype(np.float16))
                    don_sc.append(probs[..., 1][valid].astype(np.float16))
                    lab.append(yv.astype(np.int8))
                logger.info("  chunk %s rows [%d,%d) done", s, c0, c1)
            base += sz

    out = f"{args.out_prefix}.shard{args.shard_index}_of{args.num_shards}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        acc_scores=np.concatenate(acc_sc),
        don_scores=np.concatenate(don_sc),
        labels=np.concatenate(lab),
    )
    logger.info("shard dump -> %s", out)


def _metrics(acc_sc, acc_lb, don_sc, don_lb) -> dict:
    from sklearn.metrics import average_precision_score

    m: dict[str, float | int] = {"n_valid_positions": int(acc_lb.shape[0])}
    for name, sc, lb in [("acceptor", acc_sc, acc_lb), ("donor", don_sc, don_lb)]:
        pos = int(lb.sum())
        m[f"pos_count_{name}"] = pos
        if pos == 0:
            m[f"auprc_{name}"] = float("nan")
            m[f"topk_accuracy_{name}"] = float("nan")
            continue
        m[f"auprc_{name}"] = float(average_precision_score(lb, sc.astype(np.float32)))
        topk_idx = np.argpartition(sc, -pos)[-pos:]
        m[f"topk_accuracy_{name}"] = float(lb[topk_idx].mean())
    return m


def _merge(args: argparse.Namespace) -> None:
    shards = sorted(glob.glob(f"{args.out_prefix}.shard*_of{args.num_shards}.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards for {args.out_prefix} (num_shards={args.num_shards})")
    logger.info("merging %d shards", len(shards))
    acc_sc, don_sc, labs = [], [], []
    for sp in shards:
        d = np.load(sp)
        acc_sc.append(d["acc_scores"]); don_sc.append(d["don_scores"]); labs.append(d["labels"])
    acc_sc = np.concatenate(acc_sc); don_sc = np.concatenate(don_sc); labs = np.concatenate(labs)
    acc_lb = (labs == _LABEL_ACCEPTOR).astype(np.float32)
    don_lb = (labs == _LABEL_DONOR).astype(np.float32)
    metrics = _metrics(acc_sc, acc_lb, don_sc, don_lb)

    out_json = f"{args.out_prefix}.metrics.json"
    payload = {
        "model": "spliceai_illumina_official_5model",
        "task": "per_nt_splice_site_annotation",
        "test_h5_path": args.test_h5_path,
        "n_shards": len(shards),
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("=" * 60)
    logger.info("OFFICIAL SpliceAI per-nt annotation on %s", args.test_h5_path)
    logger.info("  valid positions=%d (acc_pos=%d, don_pos=%d)",
                metrics["n_valid_positions"], metrics["pos_count_acceptor"], metrics["pos_count_donor"])
    logger.info("  AUPRC:            acceptor=%.4f  donor=%.4f",
                metrics["auprc_acceptor"], metrics["auprc_donor"])
    logger.info("  top-k(k=pos) acc: acceptor=%.4f  donor=%.4f",
                metrics["topk_accuracy_acceptor"], metrics["topk_accuracy_donor"])
    logger.info("  saved -> %s", out_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official SpliceAI per-nt annotation eval.")
    p.add_argument("--test-h5-path", required=True, dest="test_h5_path")
    p.add_argument("--out-prefix", required=True, dest="out_prefix")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        _merge(args)
    else:
        _run_shard(args)


if __name__ == "__main__":
    main()
