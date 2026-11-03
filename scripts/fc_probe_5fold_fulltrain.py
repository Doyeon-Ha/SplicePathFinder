"""Two-layer FC probes on per-position delta profiles, gene-grouped 5-fold OOF.

Training uses the full consequence spectrum of three folds; validation and test
are the splice-region variants of the next fold and of the held-out fold, and the
pooled out-of-fold predictions are scored on the splice-region universe and its
Table-9 distance strata (the protocol of the manuscript's FC_X columns).

Heads (--head):
  unsorted          position-fixed profile
  sorted            profile sorted descending (FC_X)
  sorted_cons       sorted profile ++ six conservation scalars at the input layer
  sorted_cons_late  sorted profile; the six scalars join after the hidden layer,
                    where the SplicePathFinder head reads them

The profile is the per-position maximum directional change of a base's per-nt
cache: 5,001 positions for a single-window cache, 8,750 for a K=4 multichunk
cache. --extract-only builds full_perpos_feat_<base>.npz once per base so that
head jobs do not race on it; --register imports the pooled rows into the results
registry (a registry writer: run it serialized).

    python -m scripts.fc_probe_5fold_fulltrain --base pangolin --head sorted --split cc
    python -m scripts.fc_probe_5fold_fulltrain --base <name> --cache <h5> --extract-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from scripts.linear_probe_clinvar_perpos import perpos_split, train_probe
from scripts.table9_strata import TABLE9_STRATA, table9_masks

NAS = "."
OUT_DIR = Path(f"{NAS}/data/eval/clinvar_linear_probe")
COORD = f"{NAS}/data/eval/clinvar_pb_2026_split/coord_splice_fields_seed42.npz"
SPLITS = {
    "raw": f"{NAS}/data/eval/clinvar_pb_2026_split/gene_grouped_5fold_seed42.npz",
    "cc": f"{NAS}/data/eval/clinvar_pb_2026_split_cc/gene_grouped_5fold_cc_seed42.npz",
}
RELEASE = (f"{NAS}/data/eval/clinvar_20260628_gencode_anchored_indel/"
           "clinvar_20260628_gencode_anchored_indel.h5")
SCALAR_NPZ = (f"{NAS}/data/eval/clinvar2026_phastcons_probe/"
              "clinvar_20260628_conservation_scalars.npz")
SA = f"{NAS}/data/eval/sav_pernt_cache"
# base -> (cache, channel indices, kind, registry model family). The SSNet profile
# is prebuilt by lucy_ssnet_fullset_merge.py (X / y / covered), without a cache.
BASES = {
    "med_ei_ac": (f"{SA}/clinvar_20260628_med_ei_ac_ep14/cache.h5", None, "accdon3",
                  "ours"),
    "pangolin": (f"{SA}/clinvar_20260628_pangolin/cache.h5", (1, 4, 7, 10),
                 "pangolin", "pangolin"),
    "spliceai": (f"{SA}/clinvar_20260628_spliceai_official/cache.h5", None, "accdon3",
                 "spliceai"),
    "splicetransformer": (f"{SA}/clinvar_20260628_splicetransformer/cache.h5", None,
                          "accdon3", "splicetransformer"),
    "ssnet": (None, None, "prebuilt", "ssnet"),
}
# head -> (registry model prefix, scoring tag)
HEADS = {
    "unsorted": ("unsort_fc", "unsort_2layer_fc"),
    "sorted": ("sorted_fc", "sorted_2layer_fc"),
    "sorted_cons": ("sorted_fc_cons6", "sorted_2layer_fc"),
    "sorted_cons_late": ("sorted_fc_cons6late", "sorted_2layer_fc"),
}
N_FOLDS = 5
N_SCALAR = 6


class FcProbeLate(nn.Module):
    """Sorted-profile trunk with the last ``n_scalar`` inputs fused after it."""

    def __init__(self, in_features: int, n_scalar: int = N_SCALAR, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        """Build ``Linear(profile -> hidden) -> ReLU -> Dropout``, then the output."""
        super().__init__()
        self.n_scalar = int(n_scalar)
        self.trunk = nn.Sequential(nn.Linear(in_features - self.n_scalar, hidden),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden + self.n_scalar, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per row of ``[profile | scalars]``."""
        hidden = self.trunk(x[:, : -self.n_scalar])
        return self.out(torch.cat([hidden, x[:, -self.n_scalar:]], dim=-1)).squeeze(-1)


def build_features(base: str, cache: str | None, channels: tuple | None, kind: str,
                   workers: int) -> Path:
    """Return the base's cached full-release profile, extracting it if missing."""
    feat = OUT_DIR / f"full_perpos_feat_{base}.npz"
    if feat.exists():
        print(f"[extract] {base}: cached {feat}", flush=True)
        return feat
    if kind == "prebuilt" or cache is None:
        raise SystemExit(f"{feat} missing and {base} has no per-nt cache to extract")
    with h5py.File(cache, "r") as f:
        n = f["ref_probs"].shape[0]
    print(f"[extract] {base}: profiles for all {n} rows of {cache}", flush=True)
    X, y = perpos_split(cache, np.arange(n), channels, kind, workers)
    tmp = feat.with_name(feat.stem + ".tmp.npz")
    np.savez(tmp, X=X, y=y)
    tmp.rename(feat)
    print(f"[extract] {base}: wrote {feat} X={X.shape}", flush=True)
    return feat


def metric(y: np.ndarray, s: np.ndarray) -> dict:
    """Return n, positives, AUROC and AUPRC (None for a tiny or one-class set)."""
    if y.size < 20 or y.min() == y.max():
        return {"n": int(y.size), "pos": int(y.sum()), "auroc": None, "auprc": None}
    return {"n": int(y.size), "pos": int(y.sum()),
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s))}


def partitions(splice: np.ndarray, snv: np.ndarray, t9: dict) -> list:
    """Return (registry partition, mask) for the pooled rows the tables read."""
    rows = [("cv5_coord_restricted", splice), ("cv5_snv", splice & snv),
            ("cv5_indel", splice & ~snv)]
    return rows + [(f"cv5_t9_{name}", t9[name] & splice) for name in TABLE9_STRATA]


def main() -> None:
    """Train one head per fold, pool the OOF predictions, and write the results."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="known base, or a new name used with --cache")
    ap.add_argument("--cache", default=None, help="per-nt cache of a new base")
    ap.add_argument("--head", default="sorted", choices=list(HEADS))
    ap.add_argument("--split", default="cc", choices=list(SPLITS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--train-mask", default=None,
                    help="npz:key boolean mask restricting the training rows; "
                         "validation and test stay on the evaluation universe")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--model-family", default=None)
    ap.add_argument("--checkpoint", default="none", help="backbone checkpoint, or none")
    ap.add_argument("--git-commit", default="uncommitted")
    ap.add_argument("--coord", default=COORD,
                    help="coord-fields npz: splice-region universe and Table 9 strata")
    ap.add_argument("--fold-file", default=None,
                    help="explicit frozen fold_id npz; defaults to --split")
    ap.add_argument("--fit-mask", default=None,
                    help="npz with a boolean fit mask, applied to train and validation")
    ap.add_argument("--tag", default="",
                    help="universe tag appended to the output stem and dataset key")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="save the fitted model and recipe for every fold")
    args = ap.parse_args()

    new_base = (None, None, "accdon3", "ours")
    cache, channels, kind, family = BASES.get(args.base, new_base)
    cache = args.cache or cache
    family = args.model_family or family
    feat = build_features(args.base, cache, channels, kind, args.workers)
    if args.extract_only:
        return
    with np.load(feat) as z:
        P = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        covered = z["covered"].astype(bool) if "covered" in z.files else None
    n = len(y)
    covered = np.ones(n, bool) if covered is None else covered
    with h5py.File(RELEASE, "r") as f:
        release_y = f["label"][:].astype(np.int64)
        snv = np.array([str(v.decode() if isinstance(v, bytes) else v).lower() == "snv"
                        for v in f["variant_kind"][:]])
    if not np.array_equal(release_y, y):
        raise SystemExit("profile rows are not the ClinVar release rows")
    split_path = args.fold_file or SPLITS[args.split]
    fold_id = np.load(split_path)["fold_id"]
    cf = np.load(args.coord)
    if len(fold_id) != n or not np.array_equal(cf["labels"].astype(np.int64), y):
        raise SystemExit("split or coordinate fields are not row-aligned")
    splice = cf["splice"].astype(bool)
    fit = (np.ones(n, bool) if args.fit_mask is None
           else np.load(args.fit_mask)["fit"].astype(bool))
    if fit.shape != (n,):
        raise SystemExit("fit mask rows are not the release rows")
    train_mask = np.ones(n, bool)
    if args.train_mask:
        mask_path, _, mask_key = args.train_mask.rpartition(":")
        train_mask = np.load(mask_path)[mask_key].astype(bool)
        if len(train_mask) != n:
            raise SystemExit("train mask rows are not the release rows")
        print(f"train mask {mask_path}:{mask_key} keeps "
              f"{int(train_mask.sum())}/{n} rows", flush=True)
    t9 = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])
    if set(fold_id) != set(range(N_FOLDS)):
        raise SystemExit("invalid fold assignment")
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=False)

    X = P if args.head == "unsorted" else np.sort(P, axis=1)[:, ::-1].copy()
    if args.head in ("sorted_cons", "sorted_cons_late"):
        cons = np.load(SCALAR_NPZ)["scalars"].astype(np.float32)
        if len(cons) != n:
            raise SystemExit(f"scalar rows {len(cons)} != profile rows {n}")
        X = np.concatenate([X, cons], axis=1)
    model_cls = FcProbeLate if args.head == "sorted_cons_late" else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"base={args.base} head={args.head} split={args.split} width={P.shape[1]} "
          f"in_features={X.shape[1]} n={n} splice={int(splice.sum())} device={device}",
          flush=True)

    oof = np.full(n, np.nan)
    for k in range(N_FOLDS):
        kv = (k + 1) % N_FOLDS
        tr = ~np.isin(fold_id, [k, kv]) & covered & train_mask & fit
        va = (fold_id == kv) & splice & covered & fit
        te = (fold_id == k) & splice & covered
        extra = {"model_cls": model_cls} if model_cls else {}
        model = train_probe(X[tr], y[tr], X[va], y[va], X.shape[1], device,
                            args.epochs, args.patience, **extra)
        if args.checkpoint_dir:
            torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            base=args.base, head=args.head, in_features=X.shape[1],
                            best_epoch=model.best_epoch, best_val_auroc=model.best_val_auroc,
                            batch_size=4096, gradient_accumulation=1, seed=42,
                            shuffle=True, drop_last=False, fold=k,
                            train_n=int(tr.sum()), validation_n=int(va.sum()),
                            test_n=int(te.sum()),
                            split_path=split_path, fit_mask=args.fit_mask,
                            coord=args.coord, feature_path=str(feat),
                            git_commit=args.git_commit), args.checkpoint_dir / f"fold{k}.pt")
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X[te]).to(device))
        oof[te] = torch.sigmoid(logits).cpu().numpy()
        m = metric(y[te], oof[te])
        print(f"fold{k}: train={int(tr.sum())} val={int(va.sum())} "
              f"test={int(te.sum())} AUROC={m['auroc']} AUPRC={m['auprc']}", flush=True)

    suffix = "_gsplitcc" if args.split == "cc" else ""
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"cv5_oof_fulltrain_{args.head}_{args.base}{suffix}{tag}"
    np.savez(OUT_DIR / f"{stem}.npz", score=oof, label=y, splice=splice, snv=snv)
    prefix, scoring = HEADS[args.head]
    model_name = f"{prefix}_{args.base}{suffix}"
    rows = []
    for part, sel in partitions(splice, snv, t9):
        mask = sel & ~np.isnan(oof)
        m = metric(y[mask], oof[mask])
        print(f"  {part:36s} n={m['n']:7d} pos={m['pos']:6d} AUROC={m['auroc']} "
              f"AUPRC={m['auprc']}", flush=True)
        rows.append({
            "model": model_name, "model_family": family,
            "dataset": f"clinvar_20260628_cv5{tag}", "dataset_n": n, "partition": part,
            "partition_n": m["n"], "scoring": scoring, "window_bp": P.shape[1] // 2,
            "auroc": m["auroc"], "auprc": m["auprc"],
            "source": str(OUT_DIR / f"{stem}.npz"),
            "data_path": [str(feat), split_path, args.coord, RELEASE]
            + ([args.fit_mask] if args.fit_mask else [])
            + ([SCALAR_NPZ] if "cons" in args.head else []),
            "script": "scripts/fc_probe_5fold_fulltrain.py",
            "checkpoint": args.checkpoint, "git_commit": args.git_commit,
            "notes": (f"{args.head} two-layer FC on the {P.shape[1]}-position "
                      f"profile of {cache or feat}; {args.split} gene-grouped 5-fold, "
                      "train = full consequence (3 folds), val/test = splice-region, "
                      f"pooled OOF; split={split_path}; fit mask={args.fit_mask}."),
        })
    jsonl = OUT_DIR / f"{stem}.registry.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {OUT_DIR / f'{stem}.npz'} and {jsonl}", flush=True)
    if args.register:
        subprocess.run([sys.executable, "scripts/results_registry.py", "import",
                        str(jsonl)], check=True)


if __name__ == "__main__":
    main()
