"""SpliceVarDB early-stopping metric: the default label-free readout, per epoch.

After each epoch checkpoint the master rank scores the validation table with the
project's default readout -- the K=4 target-only multichunk cache over the
[-4,375, +4,375) window, conservative merge, delta max without canonical
masking -- and returns the AUROC of delta max against the table's labels. It
runs the offline scripts themselves, one cache shard per GPU of the job
(``build_multichunk_pernt_cache`` -> ``link_multichunk_cache_parts`` ->
``deltamax_from_multichunk_cache``), so the stopping signal is the number an
offline evaluation of that checkpoint reports.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]

# torchrun's process-group variables would make the scoring scripts look like
# DDP ranks of the training job.
_DIST_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@dataclass(frozen=True)
class SvdbValidationConfig:
    """Where and how to score a checkpoint.

    Attributes:
        variant_h5 (str): Gencode-anchored validation table carrying ``label``.
        work_dir (str): Each epoch writes to ``work_dir/epoch_NNN``.
        gpus (tuple[str, ...]): Device ids; one cache shard runs on each.
        threads_per_shard (int): CPU threads given to each shard process.
        batch_size (int): Forward batch of the cache builder.
    """

    variant_h5: str
    work_dir: str
    gpus: tuple[str, ...]
    threads_per_shard: int
    batch_size: int = 16


def visible_gpus() -> tuple[str, ...]:
    """Return the device ids this process may use, in CUDA_VISIBLE_DEVICES order."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return tuple(x.strip() for x in env.split(",") if x.strip())
    import torch

    return tuple(str(i) for i in range(torch.cuda.device_count()))


def shard_bounds(n_rows: int, n_shards: int) -> list[tuple[int, int]]:
    """Split ``[0, n_rows)`` into contiguous, near-equal, non-empty ranges."""
    edges = np.linspace(0, n_rows, n_shards + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def replay_early_stopping(
    values: list[float], mode: str, min_delta: float
) -> tuple[float, int]:
    """Return the early-stopping ``(best, wait)`` state after ``values``.

    Mirrors the trainer's update rule, so a resumed run continues the patience
    count of the run it resumes.
    """
    best = float("inf") if mode == "min" else float("-inf")
    wait = 0
    for value in values:
        improved = (
            value < best - min_delta if mode == "min" else value > best + min_delta
        )
        if improved:
            best, wait = value, 0
        else:
            wait += 1
    return best, wait


def _env(gpu: str, threads: int) -> dict[str, str]:
    """Environment for one scoring process pinned to ``gpu``."""
    env = {k: v for k, v in os.environ.items() if k not in _DIST_ENV}
    env = {k: v for k, v in env.items() if not k.startswith("TORCHELASTIC_")}
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(threads)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), os.environ.get("PYTHONPATH", "")) if p
    )
    return env


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    """Start ``cmd`` from the repo root, with output to ``log_path``."""
    with open(log_path, "w") as log:
        return subprocess.Popen(
            cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT
        )


def _wait(procs: list[tuple[subprocess.Popen, Path]]) -> None:
    """Wait for every process; raise with the log tail of the first failure."""
    failed = [(p.wait(), log) for p, log in procs]
    failed = [(rc, log) for rc, log in failed if rc != 0]
    if failed:
        rc, log = failed[0]
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise RuntimeError(
            f"{len(failed)} SpliceVarDB scoring process(es) failed; first exit {rc}, "
            f"log {log}:\n{tail}"
        )


def score_checkpoint(
    cfg: SvdbValidationConfig, checkpoint: str, epoch: int
) -> dict[str, float]:
    """Score ``checkpoint`` on the validation table; return the AUROC and counts.

    Args:
        cfg (SvdbValidationConfig): Table, output location and devices.
        checkpoint (str): Saved epoch checkpoint to score.
        epoch (int): 1-based epoch, naming the output directory.

    Returns:
        dict[str, float]: ``svdb_auroc``, ``svdb_n``, ``svdb_positives`` and
            ``svdb_seconds``.
    """
    start = time.time()
    out = Path(cfg.work_dir) / f"epoch_{epoch:03d}"
    parts = out / "parts"
    if parts.exists():  # a resumed run re-scores this epoch from scratch
        shutil.rmtree(parts)
    parts.mkdir(parents=True)
    with h5py.File(cfg.variant_h5, "r") as f:
        labels = f["label"][:].astype(int)
    py = sys.executable

    procs = []
    for i, ((lo, hi), gpu) in enumerate(
        zip(shard_bounds(len(labels), len(cfg.gpus)), cfg.gpus)
    ):
        log = out / f"cache_{i:03d}.log"
        cmd = [
            py,
            "process_input/build_multichunk_pernt_cache.py",
            "--variant-h5",
            cfg.variant_h5,
            "--checkpoint",
            checkpoint,
            "--out-h5",
            str(parts / f"part_{i:03d}.h5"),
            "--start",
            str(lo),
            "--end",
            str(hi),
            "--batch-size",
            str(cfg.batch_size),
        ]
        procs.append((_run(cmd, _env(gpu, cfg.threads_per_shard), log), log))
    _wait(procs)

    cache = out / "cache.h5"
    log = out / "link.log"
    cmd = [
        py,
        "scripts/link_multichunk_cache_parts.py",
        "--parts-dir",
        str(parts),
        "--out-h5",
        str(cache),
        "--variant-h5",
        cfg.variant_h5,
    ]
    _wait([(_run(cmd, _env("", cfg.threads_per_shard), log), log)])

    npz = out / "deltamax.npz"
    log = out / "deltamax.log"
    cmd = [
        py,
        "scripts/deltamax_from_multichunk_cache.py",
        "--cache",
        str(cache),
        "--source-h5",
        cfg.variant_h5,
        "--out",
        str(npz),
        "--no-mask-canonical",
    ]
    _wait([(_run(cmd, _env("", cfg.threads_per_shard), log), log)])

    delta_max = np.load(npz)["delta_max"]
    if len(delta_max) != len(labels):
        raise RuntimeError(
            f"{npz} has {len(delta_max)} rows, {cfg.variant_h5} {len(labels)}"
        )
    result = {
        "svdb_auroc": float(roc_auc_score(labels, delta_max)),
        "svdb_n": float(len(labels)),
        "svdb_positives": float(labels.sum()),
        "svdb_seconds": time.time() - start,
    }
    # The cache (~2.5 GB for the full table) is fully summarised by deltamax.npz.
    shutil.rmtree(parts)
    cache.unlink()
    (out / "metrics.json").write_text(
        json.dumps({**result, "epoch": epoch, "checkpoint": checkpoint}, indent=2)
    )
    return result


def main() -> None:
    """Score one checkpoint outside training (smoke tests, re-scoring)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant-h5", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    gpus = visible_gpus()
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    cfg = SvdbValidationConfig(
        variant_h5=args.variant_h5,
        work_dir=args.work_dir,
        gpus=gpus,
        threads_per_shard=max(1, cpus // max(len(gpus), 1)),
        batch_size=args.batch_size,
    )
    logger.info("%s", score_checkpoint(cfg, args.checkpoint, args.epoch))


if __name__ == "__main__":
    main()
