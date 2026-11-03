"""Build a fresh gene-grouped 5-fold CV split over the ENTIRE ClinVar 2026 set.

Partitions all 378,543 ClinVar 2026 >=2-star P/B variants (med_ei_ac cache row
order = canonical, shared by all base caches) into 5 folds by gene_symbol so no
gene appears in more than one fold. Uses StratifiedGroupKFold (label-balanced +
group-disjoint) when available, else GroupKFold. Variants with no gene get a
unique per-variant pseudo-gene (never grouped with others).

For each CV iteration k: test = fold k, val = fold (k+1) % 5, train = other 3
folds -- all gene-disjoint, val for early-stopping. Writes:
  gene_grouped_5fold_seed42.npz    fold_id (N,) int8 + fold{k}_{train,val,test}_idx
  fold{k}_split_seed42.npz         train_idx/val_idx/test_idx  (drop-in freeze_npz)
Read-only inputs. Aligned to the per-nt caches (row index == cache row).
"""
from __future__ import annotations
import numpy as np, h5py

NAS = "."
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
OUT_DIR = f"{NAS}/data/eval/clinvar_pb_2026_split"
N_FOLDS = 5
SEED = 42


def dec(a):
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in a]


def main() -> None:
    with h5py.File(CACHE, "r") as f:
        gene = np.array(dec(f["gene_symbol"][:]))
        y = f["labels"][:].astype(int)
    n = len(gene)

    # group key: gene_symbol, or a unique token per ungrouped variant
    groups = gene.astype(object).copy()
    missing = np.array([g in ("", ".", "NA", "nan", "None") for g in gene])
    for i in np.where(missing)[0]:
        groups[i] = f"__nogene_{i}"
    print(f"N={n}  pos_frac={y.mean():.4f}  unique_groups={len(set(groups))}  ungrouped={int(missing.sum())}")

    fold_id = np.full(n, -1, dtype=np.int8)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        it = splitter.split(np.zeros(n), y, groups)
        method = "StratifiedGroupKFold"
    except Exception as e:
        from sklearn.model_selection import GroupKFold
        # GroupKFold has no shuffle; permute group->fold determinism via seeded relabel
        rng = np.random.default_rng(SEED)
        perm = {g: i for i, g in enumerate(rng.permutation(sorted(set(groups.tolist()))))}
        gperm = np.array([perm[g] for g in groups])
        it = GroupKFold(n_splits=N_FOLDS).split(np.zeros(n), y, gperm)
        method = f"GroupKFold (StratifiedGroupKFold unavailable: {e})"
    for k, (_, test_idx) in enumerate(it):
        fold_id[test_idx] = k
    assert (fold_id >= 0).all(), "some variants unassigned"
    print(f"split method: {method}")

    # gene-disjointness across folds (each group in exactly one fold)
    for g in {}.fromkeys(groups):
        pass
    gf = {}
    ok = True
    for i in range(n):
        gf.setdefault(groups[i], fold_id[i])
        if gf[groups[i]] != fold_id[i]:
            ok = False
    print(f"gene-disjoint across folds: {ok}")

    out = {"fold_id": fold_id}
    print(f"\n{'fold':>4} {'n':>8} {'pos':>7} {'pos_frac':>9} {'#genes':>7}")
    for k in range(N_FOLDS):
        sel = fold_id == k
        gk = len(set(groups[sel].tolist()))
        print(f"{k:>4} {int(sel.sum()):>8} {int(y[sel].sum()):>7} {y[sel].mean():>9.4f} {gk:>7}")

    all_idx = np.arange(n)
    for k in range(N_FOLDS):
        test = all_idx[fold_id == k]
        val = all_idx[fold_id == (k + 1) % N_FOLDS]
        train = all_idx[~np.isin(fold_id, [k, (k + 1) % N_FOLDS])]
        out[f"fold{k}_train_idx"] = train.astype(np.int64)
        out[f"fold{k}_val_idx"] = val.astype(np.int64)
        out[f"fold{k}_test_idx"] = test.astype(np.int64)
        np.savez(f"{OUT_DIR}/fold{k}_split_seed42.npz",
                 train_idx=train.astype(np.int64), val_idx=val.astype(np.int64),
                 test_idx=test.astype(np.int64))
        # verify gene-disjoint train/val/test for this fold
        gt = set(groups[train].tolist()); gv = set(groups[val].tolist()); gs = set(groups[test].tolist())
        assert not (gt & gv) and not (gt & gs) and not (gv & gs), f"fold {k} gene leakage"
        print(f"fold{k}: train={len(train)} val={len(val)} test={len(test)} (gene-disjoint OK)")
    np.savez(f"{OUT_DIR}/gene_grouped_5fold_seed42.npz", **out)
    print(f"\nwrote {OUT_DIR}/gene_grouped_5fold_seed42.npz + 5 per-fold split npz")


if __name__ == "__main__":
    main()
