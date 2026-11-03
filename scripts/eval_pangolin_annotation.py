"""Per-nucleotide splice-site ANNOTATION eval for native Pangolin on a chunked
X{i}/Y{i} test H5 (chr1,3,5,7,9 held-out), with a reusable per-nt cache.

Pangolin (tkzeng/Pangolin) does NOT separate acceptor vs donor: each tissue
group k emits [not-splice(3k), splice-prob(3k+1), usage(3k+2)]; the native
splice-site probability is channel 3k+1 (groups → [1,4,7,10]). So the honest
native annotation metric here is COMBINED splice-site detection:
  - label  = is-splice = (Y==1 acceptor) OR (Y==2 donor)
  - score  = per-position splice-prob aggregated over the 4 tissue groups (max)
  - top-k  = precision @ k=(#splice sites); plus AUPRC.
(SpliceAI/our-model can also separate acc vs donor; Pangolin cannot — for a fair
3-way number compute the SAME combined metric on those two as well.)

Model I/O mirrors process_input/build_pangolin_pernt_cache.py:
- 12 models: for i in [0,2,4,6]: for j in [1,2,3] → models[3k:3k+3] = group k.
- one-hot via IN_MAP (N=0,A=1..T=4), strand "+" (X is already gene-strand).
- CL=10,000 → input 15,000 crops to central 5,000 output (aligned to Y).

Caching: per-nt splice-prob for the 4 tissue groups (channels [1,4,7,10]) is
written to a cache H5 (n, 5000, 4) float16, row-aligned to the test H5, so
future analyses reuse it without re-running Pangolin.

Run (.venv + pglibs on PYTHONPATH), one shard per MIG job:
    python scripts/eval_pangolin_annotation.py \
      --test-h5-path .../dataset_test_0.h5 \
      --pangolin-models-dir ./output/pangolin_run/pglibs/pangolin/models \
      --cache-prefix /NAS/.../pangolin_test0/cache \
      --out-prefix output/pangolin_annotation/test0 \
      --shard-index K --num-shards N --batch-size 16
Merge (after array): add --merge (assembles cache.h5 + computes metrics.json).
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
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
SPLICE_CH = [1, 4, 7, 10]  # per-group splice-prob channels (3k+1)


def _load_pangolin_models(models_dir: str, device: torch.device):
    """Load the 12 Pangolin models (group k = models[3k:3k+3]); return (models, CL)."""
    from pangolin.model import AR, L, Pangolin, W

    models = []
    for i in [0, 2, 4, 6]:
        for j in [1, 2, 3]:
            m = Pangolin(L, W, AR)
            w = torch.load(f"{models_dir}/final.{j}.{i}.3.v2", map_location=device)
            m.load_state_dict(w)
            models.append(m.to(device).eval())
    cl = 2 * int(np.sum(np.array(AR) * (np.array(W) - 1)))
    logger.info("loaded %d Pangolin models, CL=%d", len(models), cl)
    return models, cl


def _forward_splice4(models, x_int: np.ndarray, target_length: int) -> np.ndarray:
    """4-group splice-prob per position. Returns (B, target_length, 4) float32."""
    oh = _IN_MAP[x_int]  # (B, L, 4)
    xt = torch.from_numpy(oh).float().transpose(1, 2).to(next(models[0].parameters()).device)
    out = np.zeros((xt.shape[0], target_length, 4), dtype=np.float32)
    with torch.no_grad():
        for k in range(4):
            ch = 3 * k + 1  # splice-prob channel for group k
            acc = np.zeros((xt.shape[0], target_length), dtype=np.float32)
            for m in models[3 * k : 3 * k + 3]:
                o = m(xt)[:, ch, :]  # (B, target_length)
                if o.shape[1] != target_length:
                    raise ValueError(f"model out len {o.shape[1]} != target {target_length}")
                acc += o.cpu().numpy()
            out[:, :, k] = acc / 3.0
    return out


def _chunk_suffixes(h5_path: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        xs = [k[1:] for k in f.keys() if k.startswith("X") and k[1:].isdigit()]
        xs = [s for s in xs if f"Y{s}" in f]
    return sorted(xs, key=int)


def _shard_bounds(n: int, i: int, k: int) -> tuple[int, int]:
    per = -(-n // k)
    return i * per, min(n, i * per + per)


def _run_shard(a: argparse.Namespace) -> None:
    sufs = _chunk_suffixes(a.test_h5_path)
    with h5py.File(a.test_h5_path, "r") as f:
        sizes = [int(f[f"X{s}"].shape[0]) for s in sufs]
        tlen = int(f["Y0"].shape[1])
    n_total = sum(sizes)
    g0, g1 = _shard_bounds(n_total, a.shard_index, a.num_shards)
    logger.info("shard %d/%d rows [%d,%d) of %d, target_len=%d",
                a.shard_index, a.num_shards, g0, g1, n_total, tlen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cl = _load_pangolin_models(a.pangolin_models_dir, device)

    cache_path = f"{a.cache_prefix}.shard{a.shard_index}_of{a.num_shards}.h5"
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    n = g1 - g0
    with h5py.File(a.test_h5_path, "r") as f, h5py.File(cache_path, "w") as cout:
        dsp = cout.create_dataset("splice4", (n, tlen, 4), dtype="float16",
                                  chunks=(1, tlen, 4), compression="gzip", compression_opts=1)
        cout.create_dataset("row_index", data=np.arange(g0, g1, dtype=np.int64))
        base = 0
        w = 0
        for s, sz in zip(sufs, sizes):
            c0, c1 = max(g0, base), min(g1, base + sz)
            if c0 < c1:
                xds = f[f"X{s}"]
                for b0 in range(c0 - base, c1 - base, a.batch_size):
                    b1 = min(c1 - base, b0 + a.batch_size)
                    x = xds[b0:b1].astype(np.int64)
                    if x.shape[1] - tlen != cl:
                        raise ValueError(f"X width {x.shape[1]} - target {tlen} != CL {cl}")
                    sp = _forward_splice4(models, x, tlen)  # (b,tlen,4)
                    dsp[w : w + (b1 - b0)] = sp.astype(np.float16)
                    w += b1 - b0
                logger.info("  chunk %s rows [%d,%d) done", s, c0, c1)
            base += sz
    logger.info("shard cache -> %s (%d rows)", cache_path, n)


def _metrics_from_scores(score: np.ndarray, is_splice: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score

    m: dict[str, float | int] = {
        "n_valid_positions": int(is_splice.shape[0]),
        "pos_count_splice": int(is_splice.sum()),
        "pos_count_acceptor": int((labels == 1).sum()),
        "pos_count_donor": int((labels == 2).sum()),
    }
    pos = m["pos_count_splice"]
    m["auprc_splice"] = float(average_precision_score(is_splice, score.astype(np.float32)))
    topk_idx = np.argpartition(score, -pos)[-pos:]
    m["topk_accuracy_splice"] = float(is_splice[topk_idx].mean())
    # informational: of the top-k splice predictions, acc/donor breakdown
    top_lab = labels[topk_idx]
    m["topk_frac_acceptor"] = float((top_lab == 1).mean())
    m["topk_frac_donor"] = float((top_lab == 2).mean())
    return m


def _merge(a: argparse.Namespace) -> None:
    shards = sorted(glob.glob(f"{a.cache_prefix}.shard*_of{a.num_shards}.h5"))
    if not shards:
        raise FileNotFoundError(f"no cache shards for {a.cache_prefix} (N={a.num_shards})")
    sufs = _chunk_suffixes(a.test_h5_path)
    with h5py.File(a.test_h5_path, "r") as f:
        sizes = [int(f[f"X{s}"].shape[0]) for s in sufs]
        tlen = int(f["Y0"].shape[1])
        ys = [f[f"Y{s}"][:] for s in sufs]
    n_total = sum(sizes)
    y_all = np.concatenate([y.reshape(y.shape[0], -1) for y in ys], axis=0)  # (n_total, tlen)

    # assemble unified cache + gather per-position splice score (max over 4 groups)
    merged_cache = f"{a.cache_prefix}.h5"
    scores = np.empty((n_total, tlen), dtype=np.float32)
    with h5py.File(merged_cache, "w") as cout:
        dsp = cout.create_dataset("splice4", (n_total, tlen, 4), dtype="float16",
                                  chunks=(1, tlen, 4), compression="gzip", compression_opts=1)
        for sp in shards:
            with h5py.File(sp, "r") as s:
                ri = s["row_index"][:]
                blk = s["splice4"][:]
                dsp[ri[0]: ri[-1] + 1] = blk
                scores[ri[0]: ri[-1] + 1] = blk.astype(np.float32).max(axis=2)  # max over groups
        cout.attrs["agg"] = "max_over_4_tissue_groups"
    logger.info("assembled cache -> %s", merged_cache)

    valid = y_all >= 0
    sc = scores[valid]
    lb = y_all[valid]
    is_splice = ((lb == 1) | (lb == 2)).astype(np.float32)
    metrics = _metrics_from_scores(sc, is_splice, lb)

    out_json = f"{a.out_prefix}.metrics.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "pangolin_native_12model",
        "task": "per_nt_splice_site_annotation_combined",
        "note": "Pangolin has no acceptor/donor separation; metric = combined splice-site detection (label=acc|donor, score=max splice-prob over 4 tissue groups).",
        "test_h5_path": a.test_h5_path,
        "cache_h5": merged_cache,
        "n_shards": len(shards),
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("=" * 60)
    logger.info("PANGOLIN native per-nt annotation (combined splice-site) on %s", a.test_h5_path)
    logger.info("  valid=%d splice_pos=%d (acc=%d don=%d)",
                metrics["n_valid_positions"], metrics["pos_count_splice"],
                metrics["pos_count_acceptor"], metrics["pos_count_donor"])
    logger.info("  AUPRC(splice)=%.4f  top-k(k=pos)=%.4f",
                metrics["auprc_splice"], metrics["topk_accuracy_splice"])
    logger.info("  cache -> %s ; metrics -> %s", merged_cache, out_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native Pangolin per-nt annotation eval + cache.")
    p.add_argument("--test-h5-path", required=True, dest="test_h5_path")
    p.add_argument("--pangolin-models-dir", required=True, dest="pangolin_models_dir")
    p.add_argument("--cache-prefix", required=True, dest="cache_prefix",
                   help="prefix for per-nt splice-prob cache H5 (shards + merged)")
    p.add_argument("--out-prefix", required=True, dest="out_prefix",
                   help="prefix for metrics.json")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.merge:
        _merge(a)
    else:
        _run_shard(a)


if __name__ == "__main__":
    main()
