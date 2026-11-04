"""Conservation-ONLY ClinVar benchmark on the connected-component gene split.

Null hypothesis under test: the pathogenicity head (SplicePathFinder, SPF) only
looks good because it is handed conservation as an extra input. If the six
conservation scalars alone — fitted and read out under the SAME protocol as SPF —
already reach SPF's numbers, the head's advantage is conservation; if they fall
far short, the advantage is structural.

Protocol is matched to SPF (ONBOARDING 4.117) in every respect that is not the
feature set: connected-component gene 5-fold (ONBOARDING 4.67), test = fold k,
validation = fold (k+1)%5, training = the remaining three folds over the FULL
ClinVar consequence universe, scoring = pooled out-of-fold predictions on the
splice-region universe selected with ``--universe`` and the Table-9 strata.

Two feature arms, selected with ``--scalars``:
    anchor        the six scalars the head actually receives (both tracks read at
                  pos_1b, plus their mean/max over pos +/- 2500)
    intervalmax   the two @pos columns replaced by the maximum over the affected
                  reference interval (scripts/build_cons_interval_scalars.py),
                  which is the strongest conservation readout we measured
                  (ONBOARDING 4.27) and so the harder baseline to beat

Two learners per arm, so a shortfall cannot be blamed on probe capacity: the
2-layer head used by the appendix FC probes, and a logistic regression.

Derived from scripts/lucy_cons_only_probe.py (the seed42 gene-grouped run behind
registry ``cons6_only*``); that script and its outputs are left untouched.

CPU only, ~2 min. Usage: python scripts/lucy_cons_only_probe_cc.py --scalars anchor
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from scripts.table9_strata import TABLE9_STRATA, table9_masks

NAS = "."
SD = f"{NAS}/data/eval/clinvar_pb_2026_split"
FOLD_CC = f"{NAS}/data/eval/clinvar_pb_2026_split_cc/gene_grouped_5fold_cc_seed42.npz"
COORD = {
    "legacy": f"{SD}/coord_splice_fields_seed42.npz",
    "v50mane": f"{SD}/coord_splice_fields_v50mane_seed42.npz",
}
OUT_DIR = f"{NAS}/data/eval/clinvar_linear_probe"
PROBE_DIR = f"{NAS}/data/eval/clinvar2026_phastcons_probe"
SCALAR_NPZ = {
    "anchor": f"{PROBE_DIR}/clinvar_20260628_conservation_scalars.npz",
    "intervalmax": f"{PROBE_DIR}/clinvar_20260628_conservation_scalars_intervalmax.npz",
}
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
N_FOLDS = 5
SEED = 42


class ConsProbe(nn.Module):
    """2-layer head over the conservation scalars only.

    Args:
        in_features (int): 입력 스칼라 수(기본 6).
        hidden (int): hidden 폭(FC/FC^c와 동일한 64).
        dropout (float): dropout 확률(FC/FC^c와 동일한 0.1).
    """

    def __init__(self, in_features: int = 6, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): ``(batch, in_features)``.

        Returns:
            torch.Tensor: ``(batch,)`` logit.
        """
        return self.net(x).squeeze(-1)


def train_probe(xtr, ytr, xva, yva, device, epochs, patience):
    """FC/FC^c와 동일한 루프·하이퍼로 conservation-only head를 학습한다.

    Args:
        xtr (np.ndarray): 학습 피처.
        ytr (np.ndarray): 학습 라벨.
        xva (np.ndarray): 검증 피처.
        yva (np.ndarray): 검증 라벨.
        device (torch.device): 연산 디바이스.
        epochs (int): 최대 epoch.
        patience (int): early-stopping patience.

    Returns:
        ConsProbe: best val AUROC 가중치를 복원한 모델.
    """
    torch.manual_seed(SEED)
    model = ConsProbe(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos, neg = float((ytr == 1).sum()), float((ytr == 0).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0), device=device))
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    xva_t = torch.from_numpy(xva).to(device)
    n, bs = len(xtr_t), 4096
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            crit(model(xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auroc = roc_auc_score(yva, torch.sigmoid(model(xva_t)).cpu().numpy())
        if auroc > best:
            best, best_ep = auroc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    print(f"  best val AUROC={best:.4f} @ep{best_ep}", flush=True)
    model.load_state_dict(best_state)
    return model


def metric(y, s):
    """AUROC/AUPRC(+n,pos)를 계산한다(작거나 단일 클래스면 None)."""
    if y.size < 20 or y.min() == y.max():
        return {"n": int(y.size), "pos": int(y.sum()), "auroc": None, "auprc": None}
    return {"n": int(y.size), "pos": int(y.sum()),
            "auroc": round(float(roc_auc_score(y, s)), 4),
            "auprc": round(float(average_precision_score(y, s)), 4)}


def main() -> None:
    """conservation-only 5-fold OOF를 산출하고 strata 지표를 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalars", choices=sorted(SCALAR_NPZ), default="anchor",
                    help="feature arm: head-matched anchor readout, or interval max")
    ap.add_argument("--universe", choices=sorted(COORD), default="legacy",
                    help="splice-region universe: legacy coord fields, or the v50 MANE ones")
    ap.add_argument("--fold-npz", default=FOLD_CC)
    ap.add_argument("--split-tag", default="cc_seed42")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    sc = np.load(SCALAR_NPZ[args.scalars])
    X = sc["scalars"].astype(np.float32)
    with h5py.File(CACHE, "r") as f:
        y = f["labels"][:].astype(np.int64)
        vk = np.array([(x.decode() if isinstance(x, bytes) else str(x)).lower()
                       for x in f["variant_kind"][:]])
    is_snv = vk == "snv"
    n = len(y)
    assert len(X) == n, f"scalars {len(X)} != labels {n}"
    fold_id = np.load(args.fold_npz)["fold_id"]
    assert len(fold_id) == n, f"fold_id {len(fold_id)} != labels {n}"
    cf = np.load(COORD[args.universe])
    splice = cf["splice"]
    t9 = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"conservation-only probe: arm={args.scalars} split={args.split_tag} "
          f"universe={args.universe} "
          f"N={n} in_features={X.shape[1]} splice={int(splice.sum())} "
          f"columns={[str(c) for c in sc['columns']]} device={device}", flush=True)

    oof = np.full(n, np.nan)
    oof_lr = np.full(n, np.nan)
    per_fold = []
    for k in range(N_FOLDS):
        kv = (k + 1) % N_FOLDS
        tr = ~np.isin(fold_id, [k, kv])
        va = (fold_id == kv) & splice
        te = (fold_id == k) & splice
        model = train_probe(X[tr], y[tr], X[va], y[va], device, args.epochs, args.patience)
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(torch.from_numpy(X[te]).to(device))).cpu().numpy()
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        oof_lr[te] = lr.predict_proba(X[te])[:, 1]
        m = metric(y[te], oof[te])
        per_fold.append(m)
        print(f"fold{k}: train={int(tr.sum())} test={int(te.sum())} "
              f"AUROC={m['auroc']} AUPRC={m['auprc']}", flush=True)

    assert not np.isnan(oof[splice]).any(), "every splice-region row must be scored once"
    tag = f"{args.split_tag}_{args.scalars}"
    if args.universe != "legacy":
        tag = f"{tag}_{args.universe}"
    np.savez(f"{OUT_DIR}/cv5_oof_consonly_{tag}.npz",
             score=oof, score_logreg=oof_lr, label=y, splice=splice, snv=is_snv)
    out = {"model": f"cons6_only_{args.scalars}", "head": "2layer_fc_on_6_scalars",
           "cv": args.split_tag, "train_scope": "FULL_consequence",
           "feature_arm": args.scalars, "universe": args.universe,
           "coord": COORD[args.universe],
           "feature_columns": [str(c) for c in sc["columns"]],
           "regime": "train FULL (3 folds) / val+test splice-region / OOF",
           "note": ("conservation-only benchmark matched to the SPF protocol "
                    "(ONBOARDING 4.117): same connected-component folds, same "
                    "training universe, same strata, conservation features only."),
           "per_fold_test": {
               "auroc": {"mean": round(float(np.mean([m["auroc"] for m in per_fold])), 4),
                         "std": round(float(np.std([m["auroc"] for m in per_fold])), 4)},
               "auprc": {"mean": round(float(np.mean([m["auprc"] for m in per_fold])), 4),
                         "std": round(float(np.std([m["auprc"] for m in per_fold])), 4)}},
           "pooled_oof": {}, "pooled_oof_logreg": {}}
    rows = [("overall_splice", splice), ("SNV", splice & is_snv), ("indel", splice & ~is_snv)]
    rows += [(f"t9_{nm}", t9[nm] & splice) for nm in TABLE9_STRATA]
    print(f"\n=== conservation-only, arm={args.scalars}, split={args.split_tag} ===")
    for nm, sel in rows:
        mask = sel & ~np.isnan(oof)
        out["pooled_oof"][nm] = metric(y[mask], oof[mask])
        out["pooled_oof_logreg"][nm] = metric(y[mask], oof_lr[mask])
        a, b = out["pooled_oof"][nm], out["pooled_oof_logreg"][nm]
        print(f"  {nm:26s} n={a['n']:7d} pos={a['pos']:6d} "
              f"FC6 {a['auroc']}/{a['auprc']}   LogReg {b['auroc']}/{b['auprc']}")
    jpath = f"{OUT_DIR}/results_cv5_consonly_{tag}.json"
    json.dump(out, open(jpath, "w"), indent=1)
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
