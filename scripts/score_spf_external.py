"""Score a variant set with explicitly selected SplicePathFinder heads.

``eval_drcons_cv5_oof.py`` scores the ClinVar release out-of-fold: every row is
the held-out test set of exactly one fold. A set that was never part of the CV
has no such assignment, so the default averages five folds. For a benchmark
whose genes were held out by one model, pass only that model (e.g. --folds 3).

Writes ``{p_mean, p_folds, row_index}`` plus the cache's identity columns so the
scores can be joined back to the source table on (chrom, pos, ref, alt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

SNAP = os.environ.get(
    "DRCONS_CODE_ROOT",
    "./code_snapshots/drcons_cv5_20260722",
)
sys.path.insert(0, SNAP)

from model.sav_classifier import RbfMamba2SavClassifier  # noqa: E402
from train.dataset_sav import SavFeatureDataset  # noqa: E402


def dec(a: np.ndarray) -> np.ndarray:
    """bytes/str 배열을 str 배열로 정규화한다.

    Args:
        a (np.ndarray): 입력 배열.

    Returns:
        np.ndarray: str dtype 배열.
    """
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a])


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """한 fold 체크포인트로 loader 전체를 채점한다.

    Args:
        model: 로드된 분류기.
        loader: DataLoader.
        device: torch device.

    Returns:
        np.ndarray: sigmoid 확률 (N,).
    """
    model.eval()
    out = []
    for b in loader:
        ch = b["channels"].to(device)
        sc = b["scalars"].to(device) if "scalars" in b else None
        out.append(torch.sigmoid(model(ch, None, sc)).float().cpu().numpy())
    return np.concatenate(out)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 파싱된 인자.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="per-nt cache.h5 for the external set")
    p.add_argument("--scalars", required=True, help="cache-row-aligned conservation scalars npz")
    p.add_argument("--cvdir", required=True, help="dir holding fold{k}/best.pt")
    p.add_argument("--out", required=True, help="output npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    p.add_argument("--require-complete", action="store_true",
                   help="require successful training completion markers before scoring")
    p.add_argument("--source-h5", default=None,
                   help="gencode-anchored table the cache was built from; supplies "
                        "chrom/pos/ref/alt when the cache does not carry them")
    args = p.parse_args()
    if len(set(args.folds)) != len(args.folds):
        p.error("--folds must not contain duplicates")
    return args


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.cache, "r") as f:
        n = int(f["ref_probs"].shape[0])
    idx = np.arange(n)
    print(f"cache rows: {n}", flush=True)

    p_folds = np.empty((len(args.folds), n), dtype=np.float32)
    checkpoints = []
    for slot, k in enumerate(args.folds):
        path = Path(args.cvdir) / f"fold{k}/best.pt"
        if args.require_complete:
            done = json.loads(path.with_name("complete.json").read_text())
            assert done["training_completed"]
        ck = torch.load(path, map_location=device,
                        weights_only=False)
        if args.require_complete:
            assert done["best_epoch"] == ck["epoch"]
        mk = dict(ck["model_kwargs"])
        model = RbfMamba2SavClassifier(**mk).to(device)
        model.load_state_dict(ck["model_state"])
        half = int(ck["config"]["window"]["half_bp"])
        nsf = int(mk.get("num_scalar_features", 0))
        ds = SavFeatureDataset(
            args.cache, idx, window_half_bp=half,
            scalar_features_npz=args.scalars if nsf > 0 else None,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        p_folds[slot] = predict(model, loader, device)
        assert np.isfinite(p_folds[slot]).all()
        checkpoints.append(dict(fold=k, path=str(path), epoch=ck.get("epoch"),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f"fold{k}: ep{ck.get('epoch')} val_auroc {ck.get('val_auroc'):.4f} "
              f"mean p {p_folds[slot].mean():.4f}", flush=True)

    p_mean = p_folds.mean(axis=0)
    ident: dict[str, np.ndarray] = {}
    src = args.source_h5
    with h5py.File(args.cache, "r") as f:
        have = [c for c in ("chrom", "pos", "ref", "alt", "gene_symbol") if c in f]
        for c in have:
            ident[c] = dec(f[c][:]) if f[c].dtype == object else f[c][:]
    if src and len(ident) < 4:
        with h5py.File(src, "r") as f:
            if len(f["pos"]) != n:
                raise RuntimeError(
                    f"{src} has {len(f['pos'])} rows but the cache covers {n}")
            for c in ("chrom", "pos", "ref", "alt", "gene_symbol"):
                ident[c] = dec(f[c][:]) if f[c].dtype == object else f[c][:]

    np.savez(args.out, p_mean=p_mean, p_folds=p_folds,
             fold_ids=np.asarray(args.folds), row_index=idx, **ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(dict(
        checkpoints=checkpoints, cache=args.cache, scalars=args.scalars,
        source_h5=args.source_h5, batch_size=args.batch_size, n=n,
    ), indent=2) + "\n")
    q = np.quantile(p_mean, [0.5, 0.9, 0.99])
    print(f"saved {args.out}", flush=True)
    print(f"p_mean median {q[0]:.4f} p90 {q[1]:.4f} p99 {q[2]:.4f} "
          f">=0.5 {(p_mean >= 0.5).sum()} >=0.75 {(p_mean >= 0.75).sum()} "
          f">=0.9 {(p_mean >= 0.9).sum()}", flush=True)


if __name__ == "__main__":
    main()
