"""Per-nucleotide splice-site ANNOTATION eval for SpliceSelectNet (SSNet,
Miyachi & Nakai 2025/2026) 5-model ensemble on the chr1,3,5,7,9 held-out test H5.

Mirrors scripts/eval_spliceai_official_annotation.py (same metric: per-class AUPRC +
top-k = precision @ k=pos_count, same shard/merge protocol) so top-k / AUPRC are
directly comparable to the SpliceAI / SplicePath / ST rows of the manuscript
annotation table (Table~ref{tab:annotation}).

WINDOW-MATCHED mode (user decision 2026-07-24): SSNet's native input is a fixed
100,000-nt frame (it left-aligns a transcript at [0:len] and PAD-pads the tail to
100k; the tail is masked out of attention). We reproduce exactly that native short-
sequence handling: each test-H5 window (15,000 nt = 5,000 central + 5,000 flank/side)
is placed left-aligned at frame positions [0:15000] and PAD-padded to 100,000. The
5,000-long labelled central region = X[5000:10000] -> model output_da[:,5000:10000].
NOTE this gives SSNet only +-5,000 nt of context vs its native <=100 kb receptive
field, so this is a context-matched (lower-bound) read of SSNet's annotation skill;
the flag is documented in the manuscript caption.

Encoding: our X int8 is N=0,A=1,C=2,G=3,T=4. SSNet input is (B,100000,4) float:
A=[1,0,0,0] C=[0,1,0,0] G=[0,0,1,0] T=[0,0,0,1] N=[.25,.25,.25,.25] PAD=[0,0,0,0].
SSNet output_da (B,100000,4) channels = [donor, acceptor, none, pad] (softmax) ->
acceptor = ch1, donor = ch0. Ensemble = mean over the 5 model_pangolin_{1..5}.h5.

Run (SSNet conda env, cwd or --ssnet-repo = ssnet repo), one shard per GPU, then merge:
    ENV=./external/ssnet/env
    $ENV/bin/python scripts/eval_ssnet_annotation.py --test-h5-path .../dataset_test_0.h5 \
        --ssnet-repo ./external/ssnet/repo \
        --weights-dir ./external/ssnet/models \
        --out-prefix /NAS/.../ssnet_annotation/test0 --shard-index K --num-shards N --batch-size 8
    $ENV/bin/python scripts/eval_ssnet_annotation.py --test-h5-path .../dataset_test_0.h5 \
        --out-prefix /NAS/.../ssnet_annotation/test0 --merge --num-shards N
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FRAME_LEN: int = 100000
WINDOW_LEN: int = 15000
CENTRAL_START: int = 5000
CENTRAL_END: int = 10000
_LABEL_ACCEPTOR: int = 1
_LABEL_DONOR: int = 2
_SSNET_ACC_CH: int = 1  # output_da channel for acceptor
_SSNET_DON_CH: int = 0  # output_da channel for donor

# our-int8 -> SSNet 4-channel float row
_ENC = np.zeros((5, 4), dtype=np.float32)
_ENC[0] = [0.25, 0.25, 0.25, 0.25]  # N
_ENC[1] = [1, 0, 0, 0]              # A
_ENC[2] = [0, 1, 0, 0]              # C
_ENC[3] = [0, 0, 1, 0]              # G
_ENC[4] = [0, 0, 0, 1]              # T


def _load_models(ssnet_repo: str, weights_dir: str):
    sys.path.insert(0, str(Path(ssnet_repo).resolve()))
    import tensorflow as tf
    import tensorflow.keras.mixed_precision as mixed_precision
    from prediction import build_model

    mixed_precision.set_global_policy("mixed_float16")
    dummy = tf.constant(0.0, shape=[1, FRAME_LEN, 4])
    models = []
    for i in range(1, 6):
        wp = str(Path(weights_dir) / f"model_pangolin_{i}.h5")
        m = build_model()
        m(dummy)
        m.load_weights(wp)
        models.append(m)
        logger.info("loaded SSNet model %d: %s", i, wp)
    return models


def _encode_batch(x_int: np.ndarray) -> np.ndarray:
    """x_int (B,15000) int8 -> (B,100000,4) float32, window left-aligned + PAD tail."""
    b = x_int.shape[0]
    frame = np.zeros((b, FRAME_LEN, 4), dtype=np.float32)
    frame[:, :WINDOW_LEN, :] = _ENC[x_int]
    return frame


def _forward(models, frame: np.ndarray):
    """-> (acc, don) each (B,5000) float32, 5-model ensemble mean over central region."""
    import tensorflow as tf

    x = tf.constant(frame)
    da_sum = None
    for m in models:
        out = m(x, training=False)[0]  # (B,100000,4) output_da
        central = out[:, CENTRAL_START:CENTRAL_END, :]  # (B,5000,4)
        central = tf.cast(central, tf.float32)
        da_sum = central if da_sum is None else da_sum + central
    da = (da_sum / float(len(models))).numpy()
    return da[:, :, _SSNET_ACC_CH], da[:, :, _SSNET_DON_CH]


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
    if args.limit:
        n_total = min(n_total, args.limit)
    g_start, g_end = _shard_bounds(n_total, args.shard_index, args.num_shards)
    logger.info("shard %d/%d global rows [%d,%d) of %d",
                args.shard_index, args.num_shards, g_start, g_end, n_total)

    models = _load_models(args.ssnet_repo, args.weights_dir)
    acc_sc, don_sc, lab = [], [], []
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
                    frame = _encode_batch(x)
                    acc, don = _forward(models, frame)
                    valid = y >= 0
                    acc_sc.append(acc[valid].astype(np.float16))
                    don_sc.append(don[valid].astype(np.float16))
                    lab.append(y[valid].astype(np.int8))
                logger.info("  chunk %s rows [%d,%d) done", s, c0, c1)
            base += sz

    out = f"{args.out_prefix}.shard{args.shard_index}_of{args.num_shards}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, acc_scores=np.concatenate(acc_sc),
                        don_scores=np.concatenate(don_sc), labels=np.concatenate(lab))
    logger.info("shard dump -> %s", out)


def _topk_auprc(sc, lb):
    from sklearn.metrics import average_precision_score

    pos = int(lb.sum())
    if pos == 0:
        return float("nan"), float("nan"), 0
    auprc = float(average_precision_score(lb, sc.astype(np.float32)))
    topk_idx = np.argpartition(sc, -pos)[-pos:]
    return float(lb[topk_idx].mean()), auprc, pos


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

    acc_topk, acc_auprc, acc_pos = _topk_auprc(acc_sc, acc_lb)
    don_topk, don_auprc, don_pos = _topk_auprc(don_sc, don_lb)
    alt_acc_topk, _, _ = _topk_auprc(don_sc, acc_lb)
    alt_don_topk, _, _ = _topk_auprc(acc_sc, don_lb)

    metrics = {
        "n_valid_positions": int(labs.shape[0]),
        "pos_count_acceptor": acc_pos, "pos_count_donor": don_pos,
        "topk_accuracy_acceptor": acc_topk, "auprc_acceptor": acc_auprc,
        "topk_accuracy_donor": don_topk, "auprc_donor": don_auprc,
        "ALT_topk_accuracy_acceptor_using_ch0": alt_acc_topk,
        "ALT_topk_accuracy_donor_using_ch1": alt_don_topk,
    }
    out_json = f"{args.out_prefix}.metrics.json"
    payload = {"model": "ssnet_spliceselectnet_5model", "mode": "window_matched_15kb",
               "task": "per_nt_splice_site_annotation",
               "test_h5_path": args.test_h5_path, "n_shards": len(shards),
               "metrics": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("=" * 60)
    logger.info("SSNet 5-model per-nt annotation (window-matched 15kb) on %s", args.test_h5_path)
    logger.info("  valid positions=%d (acc_pos=%d, don_pos=%d)",
                metrics["n_valid_positions"], acc_pos, don_pos)
    logger.info("  top-k(k=pos): acceptor=%.4f  donor=%.4f", acc_topk, don_topk)
    logger.info("  AUPRC:        acceptor=%.4f  donor=%.4f", acc_auprc, don_auprc)
    logger.info("  SANITY alt (swapped ch): acc=%.4f don=%.4f (should be MUCH lower)",
                alt_acc_topk, alt_don_topk)
    logger.info("  saved -> %s", out_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSNet per-nt annotation eval (window-matched).")
    p.add_argument("--test-h5-path", required=True, dest="test_h5_path")
    p.add_argument("--out-prefix", required=True, dest="out_prefix")
    p.add_argument("--ssnet-repo", default="./external/ssnet/repo", dest="ssnet_repo")
    p.add_argument("--weights-dir", default="./external/ssnet/models/SSNet_pangolin", dest="weights_dir")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--limit", type=int, default=0, help="cap total windows (bench)")
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
