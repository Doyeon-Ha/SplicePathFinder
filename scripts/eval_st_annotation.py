"""Per-nucleotide splice-site ANNOTATION eval for SpliceTransformer (SpTransformer,
You et al. 2024) on the chr1,3,5,7,9 held-out test H5 (the SpliceAI split).

Mirrors scripts/eval_spliceai_official_annotation.py exactly (same metric: per-class
AUPRC + top-k = precision @ k=pos_count, same shard/merge protocol) so the resulting
top-k / AUPRC are directly comparable to the SpliceAI / SplicePath rows of the
manuscript annotation table. ST is a native windowed per-position annotator: its
receptive field is context_len=4000 flank each side + an internal 8192-token
attention clamp, so the test H5's +-5000 flank (>= its 4000 context) gives ST its
FULL native context (window-matched == native for ST).

Test H5 layout (h5_cl10000_sl5000_s1250_regions/dataset_test_0.h5):
- X{i}: int8 (N=0,A=1,C=2,G=3,T=4), shape (n, 15000) = 5000 central + 5000 flank/side.
- Y{i}: int8 0=neither,1=acceptor,2=donor,-1=pad, shape (n, 5000) = central labels.

ST forward: input length L -> output length L - 2*context_len. We feed the central
13000 of each 15000 window (X[:, 1000:14000], context_len=4000) so the 5000-long
output aligns exactly to X[5000:10000] == the labelled central region == Y. Channels
0:3 are softmaxed (ST post_decorate); SpliceAI convention -> ch1=acceptor, ch2=donor
(auto-verified in --merge, which reports the alternative assignment too).

Run (ST env), one shard per GPU job, then merge:
    STPY=./external/splicetransformer_eval/.venv/bin/python
    $STPY scripts/eval_st_annotation.py --test-h5-path .../dataset_test_0.h5 \
        --st-repo ./external/splicetransformer_eval \
        --out-prefix output/st_annotation/test0 --shard-index K --num-shards N --batch-size 24
    $STPY scripts/eval_st_annotation.py --test-h5-path .../dataset_test_0.h5 \
        --out-prefix output/st_annotation/test0 --merge --num-shards N
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CONTEXT_LEN: int = 4000  # ST native flank each side
WINDOW_CROP: int = 1000  # 15000 -> central 13000 (= 5000 out + 2*4000 context)
IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
_LABEL_ACCEPTOR: int = 1
_LABEL_DONOR: int = 2


def _load_st_model(st_repo: str):
    import importlib.util
    import torch

    # The external checkout can be a namespace package named `model`, which
    # otherwise loses to this repository's regular package with the same name.
    spec = importlib.util.spec_from_file_location(
        "_annotation_sptransformer", Path(st_repo).resolve() / "model/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SpTransformer = module.SpTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpTransformer(128, context_len=CONTEXT_LEN, tissue_num=15,
                          max_seq_len=8192, attn_depth=8, training=False)
    ckpt = str(Path(st_repo) / "model" / "weights" / "SpTransformer_pytorch.ckpt")
    save_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(save_dict["state_dict"])
    model = model.to(device).eval()
    logger.info("loaded SpTransformer from %s (device=%s)", ckpt, device)
    return model, device


def _forward(model, device, x_int: np.ndarray):
    """x_int: (B,15000) int8. -> (acc, don) each (B,5000) float32 softmax-probs."""
    import torch
    import torch.nn.functional as F

    x_crop = x_int[:, WINDOW_CROP:-WINDOW_CROP]  # (B,13000)
    oh = IN_MAP[x_crop]  # (B,13000,4)
    xt = torch.from_numpy(oh).to(device).transpose(1, 2).float()  # (B,4,13000)
    with torch.no_grad():
        out = model(xt)  # (B, 18, 5000)
        probs = F.softmax(out[:, :3, :], dim=1).cpu().numpy()  # (B,3,5000)
    acc = probs[:, _LABEL_ACCEPTOR, :]
    don = probs[:, _LABEL_DONOR, :]
    return acc, don


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

    model, device = _load_st_model(args.st_repo)
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
                    acc, don = _forward(model, device, x)
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

    # primary (SpliceAI convention): ch1=acceptor, ch2=donor
    acc_topk, acc_auprc, acc_pos = _topk_auprc(acc_sc, acc_lb)
    don_topk, don_auprc, don_pos = _topk_auprc(don_sc, don_lb)
    # sanity: alternative (swapped) assignment
    alt_acc_topk, _, _ = _topk_auprc(don_sc, acc_lb)
    alt_don_topk, _, _ = _topk_auprc(acc_sc, don_lb)

    metrics = {
        "n_valid_positions": int(labs.shape[0]),
        "pos_count_acceptor": acc_pos, "pos_count_donor": don_pos,
        "topk_accuracy_acceptor": acc_topk, "auprc_acceptor": acc_auprc,
        "topk_accuracy_donor": don_topk, "auprc_donor": don_auprc,
        "ALT_topk_accuracy_acceptor_using_ch2": alt_acc_topk,
        "ALT_topk_accuracy_donor_using_ch1": alt_don_topk,
    }
    out_json = f"{args.out_prefix}.metrics.json"
    payload = {"model": "splicetransformer_sptransformer",
               "task": "per_nt_splice_site_annotation",
               "test_h5_path": args.test_h5_path, "n_shards": len(shards),
               "metrics": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("=" * 60)
    logger.info("SpliceTransformer per-nt annotation on %s", args.test_h5_path)
    logger.info("  valid positions=%d (acc_pos=%d, don_pos=%d)",
                metrics["n_valid_positions"], acc_pos, don_pos)
    logger.info("  top-k(k=pos): acceptor=%.4f  donor=%.4f", acc_topk, don_topk)
    logger.info("  AUPRC:        acceptor=%.4f  donor=%.4f", acc_auprc, don_auprc)
    logger.info("  SANITY alt (swapped ch): acc=%.4f don=%.4f "
                "(should be MUCH lower if ch1=acc,ch2=don is correct)",
                alt_acc_topk, alt_don_topk)
    logger.info("  saved -> %s", out_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpliceTransformer per-nt annotation eval.")
    p.add_argument("--test-h5-path", required=True, dest="test_h5_path")
    p.add_argument("--out-prefix", required=True, dest="out_prefix")
    p.add_argument("--st-repo", default="./external/splicetransformer_eval", dest="st_repo")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=24, dest="batch_size")
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
