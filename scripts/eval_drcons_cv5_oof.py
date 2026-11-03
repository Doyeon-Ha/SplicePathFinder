"""Out-of-fold evaluation of the 6-feature SpliceBoa-region DR_conservation
DeltaReader over the ClinVar 2026 gene-grouped 5-fold CV.

For each fold k: load cv5_drcons/fold{k}/best.pt, predict that fold's held-out
test_idx (fold{k}_split_seed42.npz) with the 6 conservation scalars, collect
(row, y, p). Every cache row is the test set of exactly one fold, so pooling the
5 folds gives a gene-disjoint OOF prediction over all 378,543 variants.

Reports overall + splice-region restricted (MC donor/acceptor/intron union, as in
Lucy's single-split eval) + coordinate-restricted (Table-8 tiling union) + the
Table-7 molecular-consequence strata. Writes pooled preds npz + json.

Runs the frozen snapshot's scalar-at-pool code. GPU (mamba_ssm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

# The scalar-at-pool code the canonical cv5_drcons run was trained against. It is
# frozen so those checkpoints keep loading against the code that produced them.
# DRCONS_CODE_ROOT points this at the repo instead, which is required for an arm
# whose cache the frozen snapshot cannot read -- the multichunk cache declares its
# window through attributes the snapshot's SavFeatureDataset does not know about.
# model/sav_classifier.py is byte-identical between the two, so only the dataset and
# training driver differ. It has to be an environment variable rather than a flag:
# sys.path is set at import time, before argparse runs.
SNAP = os.environ.get(
    "DRCONS_CODE_ROOT",
    "./code_snapshots/drcons_cv5_20260722",
)
sys.path.insert(0, SNAP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo scripts/ for table9_strata
from table9_strata import TABLE9_STRATA, table9_masks  # noqa: E402

from model.sav_classifier import RbfMamba2SavClassifier  # noqa: E402
from train.dataset_sav import SavFeatureDataset  # noqa: E402

NAS = "."
SD = f"{NAS}/data/eval/clinvar_pb_2026_split"
CVDIR = f"{NAS}/data/eval/clinvar2026_phastcons_probe/cv5_drcons"
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
VTAB = f"{NAS}/data/eval/clinvar_20260628_variant_table.h5"
# The 378,543-row table the per-nt caches are built from and indexed by. VTAB is a
# different, longer row space (384,073) -- do not use it to look up a cache row.
GTAB = (f"{NAS}/data/eval/clinvar_20260628_gencode_anchored_indel/"
        f"clinvar_20260628_gencode_anchored_indel.h5")
COORD = f"{SD}/coord_splice_fields_seed42.npz"

CATS = ["splice_donor", "splice_acceptor", "intron", "missense", "synonymous",
        "nonsense", "frameshift", "UTR", "other"]
PRIO = [("splice_donor", "splice_donor"), ("splice_acceptor", "splice_acceptor"),
        ("nonsense", "nonsense"), ("stop_gained", "nonsense"), ("frameshift", "frameshift"),
        ("missense", "missense"), ("initiator_codon", "missense"), ("stop_lost", "nonsense"),
        ("synonymous", "synonymous"), ("5_prime_utr", "UTR"), ("3_prime_utr", "UTR"),
        ("intron", "intron")]


def dec(a):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def region_of(mc):
    toks = mc.lower().replace("&", ",").split(",")
    for pat, cat in PRIO:
        if any(pat in t for t in toks):
            return cat
    return "other"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        ch = b["channels"].to(device)
        sc = b["scalars"].to(device) if "scalars" in b else None
        out.append(torch.sigmoid(model(ch, None, sc)).float().cpu().numpy())
    return np.concatenate(out)


def au(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def main() -> None:
    global CACHE, CVDIR, SD, COORD
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache", default=CACHE,
                    help="per-nt cache.h5 (frozen backbone tracks); default = med_ei_ac region")
    ap.add_argument("--cvdir", default=CVDIR,
                    help="dir with fold{k}/best.pt; results written here; default = region cv5_drcons")
    ap.add_argument("--split-dir", default=SD, dest="split_dir",
                    help="dir holding fold{k}_split_<tag>.npz")
    ap.add_argument("--split-tag", default="seed42", dest="split_tag",
                    help="split filename tag: seed42 (raw gene_symbol groups) "
                         "or cc_seed42 (token connected components)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--coord", default=COORD,
                    help="coord-fields npz: splice-region universe and Table 9 strata")
    args = ap.parse_args()
    CACHE = args.cache
    CVDIR = args.cvdir
    SD = args.split_dir
    COORD = args.coord
    out = args.out or f"{CVDIR}/drcons_cv5_oof_result.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_full = 378543
    p_oof = np.full(n_full, np.nan)
    covered = np.zeros(n_full, bool)
    for k in range(5):
        ckp = f"{CVDIR}/fold{k}/best.pt"
        ck = torch.load(ckp, map_location=device, weights_only=False)
        mk = dict(ck["model_kwargs"])
        model = RbfMamba2SavClassifier(**mk).to(device)
        model.load_state_dict(ck["model_state"])
        half = int(ck["config"]["window"]["half_bp"])
        nsf = int(mk.get("num_scalar_features", 0))
        scalar_npz = ck["config"]["feature"]["scalar_npz"] if nsf > 0 else None
        # An arm trained with canonical masking has to be scored with it too, or
        # the head sees deltas at train time that it never sees at test time. The
        # arm's own config travels in the checkpoint, so read it from there.
        mask_cfg = ck["config"].get("mask", None)
        canonical_mask = (
            mask_cfg.get("canonical_h5", None) if mask_cfg else None
        )
        if k == 0:
            print(f"canonical masking: {canonical_mask or 'off'}", flush=True)
        test_idx = np.load(f"{SD}/fold{k}_split_{args.split_tag}.npz")["test_idx"]
        # A swapped base reads the probability channels its head was trained on.
        ci = ck["config"].get("channel_indices", None)
        ci = tuple(ci) if ci is not None else None
        ds = SavFeatureDataset(CACHE, test_idx, window_half_bp=half,
                               scalar_features_npz=scalar_npz,
                               canonical_mask_h5=canonical_mask, channel_indices=ci)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        p = predict(model, loader, device)
        p_oof[test_idx] = p
        covered[test_idx] = True
        print(f"fold{k}: predicted {test_idx.size} (ep{ck.get('epoch')}, val {ck.get('val_auroc'):.4f})",
              flush=True)
    # An arm trained on the splice-region universe has its test folds restricted the
    # same way, so it covers that universe rather than the whole release. Accept
    # exactly that set and nothing in between: a partial cover with no explanation
    # would otherwise be reported as if it were the full thing.
    if not covered.all():
        restr = np.load(COORD)["splice"].astype(bool)
        if not np.array_equal(covered, restr):
            raise RuntimeError(
                f"OOF covers {covered.sum()} rows, which is neither the full "
                f"{n_full} nor the coord-restricted universe ({int(restr.sum())})"
            )
        print(f"restricted arm: OOF covers the coord universe ({covered.sum()})",
              flush=True)

    # labels + snv (small); Table 9 strata need only coord fields, not VTAB
    with h5py.File(CACHE, "r") as f:
        y = f["labels"][:].astype(np.int64)
        has_alleles = "ref" in f and "alt" in f
        if has_alleles:
            ref = dec(f["ref"][:]); alt = dec(f["alt"][:])
    # The multichunk cache stores only the probability tracks, so the alleles come
    # from the gencode-anchored table it was built from -- NOT from VTAB, which has
    # 384,073 rows against this row space's 378,543 and is not aligned to it. The
    # cache's row_index was checked to be the contiguous 0..N-1 when the shards were
    # linked, so row i there is row i here.
    if not has_alleles:
        with h5py.File(GTAB, "r") as f:
            ref = dec(f["ref"][:]); alt = dec(f["alt"][:])
        if len(ref) != n_full:
            raise RuntimeError(
                f"{GTAB} has {len(ref)} rows but the cache covers {n_full}"
            )
    snv = np.array([len(ref[i]) == 1 and len(alt[i]) == 1 for i in range(n_full)])
    cf = np.load(COORD)
    coord_restr = cf["splice"]
    t9 = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])

    # checkpoint predictions immediately (before the heavier VTAB join) so a re-run
    # only re-does aggregation, never GPU inference
    np.savez(out.replace(".json", "_preds.npz"),
             y=y, p=p_oof, snv=snv, coord_restr=coord_restr, **t9)
    print("saved preds npz", flush=True)

    res = {"n": n_full, "strata": {}}
    rows = [("overall", np.ones(n_full, bool)), ("SNV", snv), ("indel", ~snv),
            ("coord_restricted_overall", coord_restr),
            ("coord_restricted_SNV", coord_restr & snv),
            ("coord_restricted_indel", coord_restr & ~snv)]
    rows += [(f"t9_{name}", t9[name]) for name in TABLE9_STRATA]   # Table 9 strata (priority)

    # MC-consequence strata (secondary): memory-frugal — store category, not the full
    # molecular_consequence strings (some are ~18 KB; a 384k-entry dict of them OOMs)
    try:
        with h5py.File(CACHE, "r") as f:
            chrom = dec(f["chrom"][:]); pos = f["pos"][:]
        with h5py.File(VTAB, "r") as f:
            vc = dec(f["chrom"][:]); vp = f["pos"][:].astype(np.int64)
            vr = dec(f["ref"][:]); va = dec(f["alt"][:]); vm = dec(f["molecular_consequence"][:])
        cons_cat = {f"{vc[i]}:{vp[i]}:{vr[i]}:{va[i]}": region_of(vm[i]) for i in range(len(vp))}
        del vm, vc, vp, vr, va
        cat = np.array([cons_cat.get(f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}", "other")
                        for i in range(n_full)])
        del cons_cat
        mc_restr = np.isin(cat, ["splice_donor", "splice_acceptor", "intron"])
        rows += [("mc_restricted_overall", mc_restr), ("mc_restricted_SNV", mc_restr & snv),
                 ("mc_restricted_indel", mc_restr & ~snv)]
        rows += [(c, cat == c) for c in CATS]
    except Exception as e:  # noqa: BLE001
        print(f"WARN MC stratification skipped ({e}); Table 9 strata unaffected", flush=True)
    # A restricted arm has no prediction outside its universe, and those rows are
    # NaN. Intersect every stratum with what was actually predicted so the reported
    # n is the number the metric was computed on, not the number the mask names.
    rows = [(name, sel & covered) for name, sel in rows]
    res["restricted_to"] = None if covered.all() else int(covered.sum())
    for name, sel in rows:
        n = int(sel.sum())
        if n < 20 or y[sel].min() == y[sel].max():
            res["strata"][name] = {"n": n, "note": "small/single-class"}
            print(f"  {name:26s} n={n:7d}  --"); continue
        a, ap_ = au(y[sel], p_oof[sel])
        res["strata"][name] = {"n": n, "n_pos": int(y[sel].sum()), "auroc": a, "auprc": ap_}
        print(f"  {name:26s} n={n:7d} pos={int(y[sel].sum()):7d}  AUROC {a:.4f} AUPRC {ap_:.4f}")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
