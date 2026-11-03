"""Per-position delta-profile variant of the DeltaReader-clinvar head ablation.

Instead of collapsing the window to a few scalars (mean+max: linear_probe_
clinvar_splicemech.py; 4 gain/loss peaks: linear_probe_clinvar_gainloss.py), this
probe keeps the positional dimension. For every position in the +/-2500 window it
takes the single largest directional delta score --

    profile[pos] = max( acc_gain, acc_loss, don_gain, don_loss )[pos]
                 = max( |alt_acc - ref_acc|, |alt_don - ref_don| )[pos]

(Pangolin has no acc/don split -- per position it maxes over the 4 tissue splice
channels x gain/loss = max_tissue |alt - ref|[pos]) -- yielding a 5001-length
per-variant vector that is handed *whole* to the 2-layer FC. delta_max is the
global max of exactly this profile; this probe instead lets the FC learn a
position-weighted combination of it (variant is centered at index 2500, so
positions are comparable across variants).

No feature standardization: all 5001 features are the same [0,1] delta quantity,
so z-scoring would only amplify background-position noise. Frozen features, split
(splits_seed42), and window are identical to DeltaReader-clinvar; only the
aggregation head differs. CPU-only. Read-only w.r.t. the caches; writes
scores/probe/results under clinvar_linear_probe.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from train.dataset_sav import SavFeatureDataset  # noqa: E402

logger = logging.getLogger(__name__)

NAS = "."
SPLIT = f"{NAS}/data/eval/clinvar_pb_2026_split/splits_seed42.npz"
VTAB = f"{NAS}/data/eval/clinvar_20260628_variant_table.h5"
MED_CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
OUT_DIR = f"{NAS}/data/eval/clinvar_linear_probe"

BASES: dict[str, dict] = {
    "med_ei_ac": {"cache": MED_CACHE, "channel_indices": None,
                  "kind": "accdon3", "model_tag": "med_ei_ac_ep14"},
    "pangolin": {"cache": f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_pangolin/cache.h5",
                 "channel_indices": (1, 4, 7, 10), "kind": "pangolin", "model_tag": "pangolin"},
    "spliceai": {"cache": f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_spliceai_official/cache.h5",
                 "channel_indices": None, "kind": "accdon3", "model_tag": "spliceai_official"},
}
WINDOW_HALF_BP = 2500
SEED = 42
PRIO = [
    ("splice_donor", "splice_donor"), ("splice_acceptor", "splice_acceptor"),
    ("nonsense", "nonsense"), ("stop_gained", "nonsense"), ("frameshift", "frameshift"),
    ("missense", "missense"), ("initiator_codon", "missense"), ("stop_lost", "nonsense"),
    ("synonymous", "synonymous"), ("5_prime_utr", "UTR"), ("3_prime_utr", "UTR"),
    ("intron", "intron"),
]
SPLICE_MECH = {"splice_donor", "splice_acceptor", "intron"}


def dec(a):
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in a]


class FcProbe(nn.Module):
    """2-layer FC over the 5001-length per-position delta profile."""

    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def perpos_split(cache_h5, idx, ci, kind, workers):
    """Per-position max-directional-delta profile + labels for cache rows.

    Args:
        cache_h5 (str): per-nt probability cache path.
        idx (np.ndarray): cache row indices.
        ci (tuple | None): channel_indices (base-specific).
        kind (str): "accdon3" (acc=0/don=1) or "pangolin" (tissue channels).
        workers (int): DataLoader workers.

    Returns:
        tuple[np.ndarray, np.ndarray]: profiles ``(n, W)`` float16 and labels ``(n,)``.
    """
    ds = SavFeatureDataset(cache_h5, idx, window_half_bp=WINDOW_HALF_BP, channel_indices=ci)
    n_sel = len(ds.channel_indices)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=workers)
    feats, labels, ref_means = [], [], []
    for batch in loader:
        ch = batch["channels"]                # (B, W, 2*n_sel [+ coverage])
        # A multichunk cache appends its coverage channel after ref and alt.
        ref, alt = ch[..., :n_sel], ch[..., n_sel:2 * n_sel]
        diff = alt - ref                      # (B, W, n_sel)
        if kind == "accdon3":
            # per position: max over (acc_gain, acc_loss, don_gain, don_loss)
            comps = torch.stack([
                diff[..., 0].clamp(min=0), (-diff[..., 0]).clamp(min=0),
                diff[..., 1].clamp(min=0), (-diff[..., 1]).clamp(min=0),
            ], dim=-1)                        # (B, W, 4)
            profile = comps.amax(dim=-1)      # (B, W)
            ref_means.append(ref.mean(dim=1).numpy())
        else:  # pangolin: per position, max over tissues x {gain, loss}
            comps = torch.stack([diff.clamp(min=0), (-diff).clamp(min=0)], dim=-1)  # (B,W,n_sel,2)
            profile = comps.amax(dim=(-1, -2))   # (B, W)
        feats.append(profile.half().numpy())
        labels.append(batch["label"].numpy())
    X = np.concatenate(feats).astype(np.float16)
    y = np.concatenate(labels).astype(np.int64)
    if kind == "accdon3":
        chmean = np.concatenate(ref_means).mean(axis=0)
        assert int(np.argmax(chmean)) == 2, f"expected 'neither' at channel 2; means={chmean}"
    return X, y


def train_probe(xtr, ytr, xva, yva, in_features, device, epochs, patience,
                model_cls=FcProbe):
    torch.manual_seed(SEED)
    model = model_cls(in_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float((ytr == 1).sum()); neg = float((ytr == 0).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0), device=device))
    xtr_t = torch.from_numpy(xtr).to(device); ytr_t = torch.from_numpy(ytr).float().to(device)
    xva_t = torch.from_numpy(xva).to(device)
    n = len(xtr_t); bs = 4096
    best_auroc, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(xva_t)).cpu().numpy()
        auroc = roc_auc_score(yva, va)
        if auroc > best_auroc:
            best_auroc, best_ep = auroc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
        if ep % 10 == 0:
            logger.info("ep%d val AUROC=%.4f (best %.4f @%d)", ep, auroc, best_auroc, best_ep)
    logger.info("best val AUROC=%.4f @ep%d", best_auroc, best_ep)
    model.load_state_dict(best_state)
    model.best_epoch = best_ep
    model.best_val_auroc = best_auroc
    return model


def _consequence_arr(idx) -> np.ndarray:
    """Collapsed consequence category for the given cache rows (idx order)."""
    with h5py.File(MED_CACHE, "r") as f:
        c = dec(f["chrom"][:]); p = f["pos"][:].astype(int); r = dec(f["ref"][:]); a = dec(f["alt"][:])
    ckey = [f"{c[i]}:{int(p[i])}:{r[i]}:{a[i]}" for i in idx]
    with h5py.File(VTAB, "r") as f:
        vc = dec(f["chrom"][:]); vp = f["pos"][:].astype(int)
        vr = dec(f["ref"][:]); va = dec(f["alt"][:]); mc = dec(f["molecular_consequence"][:])
    cons = {f"{vc[i]}:{int(vp[i])}:{vr[i]}:{va[i]}": mc[i] for i in range(len(vp))}

    def region(k):
        toks = cons.get(k, "").lower().replace("&", ",").split(",")
        for pat, cat in PRIO:
            if any(pat in t for t in toks):
                return cat
        return "other"

    return np.array([region(k) for k in ckey])


def consequence_for_test(test_idx) -> np.ndarray:
    return _consequence_arr(test_idx)


def region_mask(idx) -> np.ndarray:
    """Boolean mask (idx order): consequence in {donor, acceptor, intron}."""
    return np.array([c in SPLICE_MECH for c in _consequence_arr(idx)])


def metric(y, s):
    if y.size < 20 or y.min() == y.max():
        return {"n": int(y.size), "pos": int(y.sum()), "auroc": None, "auprc": None}
    return {"n": int(y.size), "pos": int(y.sum()),
            "auroc": round(float(roc_auc_score(y, s)), 4),
            "auprc": round(float(average_precision_score(y, s)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--train-region-only", action="store_true",
                    help="train/early-stop on splice-region (donor/acceptor/intron) "
                         "train+val variants only; eval unchanged (gg-test strata)")
    args = ap.parse_args()
    sfx = "_regiontrain" if args.train_region_only else ""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = BASES[args.base]
    cache, ci, kind, tag = cfg["cache"], cfg["channel_indices"], cfg["kind"], cfg["model_tag"]
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sp = np.load(SPLIT)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    feat = f"{OUT_DIR}/perpos_feat_{args.base}.npz"
    if Path(feat).exists():
        logger.info("load cached features: %s", feat)
        z = np.load(feat)
        xtr, ytr, xva, yva, xte, yte = z["xtr"], z["ytr"], z["xva"], z["yva"], z["xte"], z["yte"]
    else:
        logger.info("extracting per-position profiles (%s, kind=%s)...", cache, kind)
        xtr, ytr = perpos_split(cache, train_idx, ci, kind, args.workers)
        xva, yva = perpos_split(cache, val_idx, ci, kind, args.workers)
        xte, yte = perpos_split(cache, test_idx, ci, kind, args.workers)
        np.savez(feat, xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte)
        logger.info("wrote %s", feat)

    in_features = xtr.shape[1]
    # no standardization: features are all the same [0,1] delta scale
    xtr = xtr.astype(np.float32); xva = xva.astype(np.float32); xte = xte.astype(np.float32)

    if args.train_region_only:
        # restrict train/val to splice-region (donor/acceptor/intron) variants;
        # features are cached in split order so masks align row-for-row.
        tr_m = region_mask(train_idx); va_m = region_mask(val_idx)
        xtr, ytr = xtr[tr_m], ytr[tr_m]
        xva, yva = xva[va_m], yva[va_m]
        logger.info("region-train: train %d->%d, val %d->%d",
                    len(tr_m), int(tr_m.sum()), len(va_m), int(va_m.sum()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("base=%s kind=%s in_features=%d device=%s train=%d val=%d test=%d region_train=%s",
                args.base, kind, in_features, device, len(ytr), len(yva), len(yte),
                args.train_region_only)
    model = train_probe(xtr, ytr, xva, yva, in_features, device, args.epochs, args.patience)

    torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "in_features": in_features, "head": "perpos_maxdelta_2layer_fc",
                "seed": SEED, "base": args.base, "channel_indices": ci,
                "window_half_bp": WINDOW_HALF_BP, "standardized": False,
                "train_region_only": args.train_region_only},
               f"{OUT_DIR}/probe_perpos{sfx}_{args.base}.pt")

    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(torch.from_numpy(xte).to(device))).cpu().numpy()
    with h5py.File(MED_CACHE, "r") as f:
        med_lab_test = f["labels"][:].astype(int)[test_idx]
    assert np.array_equal(yte, med_lab_test), f"{args.base} label misalignment vs med cache"
    np.savez(f"{OUT_DIR}/scores_perpos{sfx}_{args.base}.npz", row_index=test_idx, score=score, label=yte)

    cat = consequence_for_test(test_idx)
    with h5py.File(cache, "r") as f:
        kind_arr = np.array([k.lower() for k in dec(f["variant_kind"][:])])[test_idx]
    is_snv = kind_arr == "snv"
    splice_sel = np.array([c in SPLICE_MECH for c in cat])
    strata = [
        ("overall", np.ones(len(yte), bool)), ("SNV", is_snv), ("indel", ~is_snv),
        ("splice_region", splice_sel),
        ("splice_region SNV", splice_sel & is_snv), ("splice_region indel", splice_sel & ~is_snv),
        ("splice_donor", cat == "splice_donor"), ("splice_acceptor", cat == "splice_acceptor"),
        ("intron", cat == "intron"),
    ]
    out = {"base": args.base, "model_tag": tag, "channel_indices": ci,
           "head": "perpos_maxdelta_2layer_fc", "in_features": in_features,
           "train_region_only": args.train_region_only,
           "window_half_bp": WINDOW_HALF_BP, "definition": sorted(SPLICE_MECH), "strata": {}}
    print(f"\n=== per-position max-delta FC probe | base={args.base} ({tag}) | in={in_features} | region_train={args.train_region_only} ===")
    print(f"{'stratum':20s} {'n':>7s} {'pos':>6s} {'AUROC':>7s} {'AUPRC':>7s}")
    for name, sel in strata:
        m = metric(yte[sel], score[sel])
        out["strata"][name] = m
        au = f"{m['auroc']:.4f}" if m["auroc"] is not None else "   -"
        pr = f"{m['auprc']:.4f}" if m["auprc"] is not None else "   -"
        print(f"{name:20s} {m['n']:7d} {m['pos']:6d} {au:>7s} {pr:>7s}")
    json.dump(out, open(f"{OUT_DIR}/results_perpos{sfx}_{args.base}.json", "w"), indent=1)
    print(f"\nwrote {OUT_DIR}/results_perpos{sfx}_{args.base}.json")


if __name__ == "__main__":
    main()
