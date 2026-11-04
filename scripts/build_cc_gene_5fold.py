"""Build five gene-component folds from a row-aligned ClinVar identity table.

Tokens that co-occur in a gene_symbol field remain together in one component.
StratifiedGroupKFold uses seed 42; test=k, validation=(k+1)%5, train=the rest.
"""
from __future__ import annotations

import json
import os
import argparse
from collections import Counter

import h5py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from process_input.variant_identity import variant_keys, row_digest

N_FOLDS, SEED = 5, 42
MISSING = {"", ".", "NA", "nan", "None"}


def dec(a: np.ndarray) -> np.ndarray:
    """Decode an h5 string array to a numpy array of str."""
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in a])


def main() -> None:
    """Build and verify the connected-component folds."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant-h5', required=True, help='Cohort table with gene_symbol and label(s)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tag', default='cc_seed42')
    args = p.parse_args()
    CACHE, OUT_DIR, TAG = args.variant_h5, args.out_dir, args.tag
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(CACHE, "r") as f:
        gene = dec(f["gene_symbol"][:])
        y = f["labels" if "labels" in f else "label"][:].astype(int)
        digest = row_digest(variant_keys(f))
    n = len(gene)
    row_tokens = [[t for t in g.split("|") if t and t not in MISSING] for g in gene]

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for toks in row_tokens:
        for t in toks[1:]:
            a, b = find(toks[0]), find(t)
            if a != b:
                parent[a] = b
    groups = np.array([find(t[0]) if t else f"__nogene_{i}"
                       for i, t in enumerate(row_tokens)], dtype=object)
    sizes = Counter(groups.tolist())
    big = sizes.most_common(1)[0][1]
    print(f"N={n}  groups={len(sizes)}  largest={big} ({100 * big / n:.3f}%)")

    fold_id = np.full(n, -1, dtype=np.int8)
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(sgk.split(np.zeros(n), y, groups)):
        fold_id[test_idx] = k
    assert (fold_id >= 0).all(), "some variants unassigned"

    all_idx = np.arange(n)
    out = {"fold_id": fold_id, "row_key_sha256": digest}
    summary = {"n": n, "n_groups": len(sizes), "tag": TAG, "folds": []}
    for k in range(N_FOLDS):
        test = all_idx[fold_id == k]
        val = all_idx[fold_id == (k + 1) % N_FOLDS]
        train = all_idx[~np.isin(fold_id, [k, (k + 1) % N_FOLDS])]
        # group-disjointness on the component key
        gt, gv, gs = (set(groups[train]), set(groups[val]), set(groups[test]))
        assert not (gt & gv) and not (gt & gs) and not (gv & gs), f"fold {k} group leak"
        # and on raw tokens, which is the property the raw-string split loses
        tt: set[str] = set()
        for i in train:
            tt.update(row_tokens[i])
        leaked = sum(1 for i in test if set(row_tokens[i]) & tt)
        assert leaked == 0, (
            f"fold {k}: {leaked} test rows share a gene token with train")
        np.savez(f"{OUT_DIR}/fold{k}_split_{TAG}.npz",
                 row_key_sha256=digest,
                 train_idx=train.astype(np.int64), val_idx=val.astype(np.int64),
                 test_idx=test.astype(np.int64))
        out[f"fold{k}_train_idx"] = train.astype(np.int64)
        out[f"fold{k}_val_idx"] = val.astype(np.int64)
        out[f"fold{k}_test_idx"] = test.astype(np.int64)
        rec = {"fold": k, "train": len(train), "val": len(val),
               "test": len(test), "test_token_leak": 0,
               "test_pos_frac": round(float(y[test].mean()), 4)}
        summary["folds"].append(rec)
        print(f"fold{k}: train={len(train)} val={len(val)} test={len(test)} "
              f"pos_frac={rec['test_pos_frac']:.4f} token-leak=0")
    np.savez(f"{OUT_DIR}/gene_grouped_5fold_{TAG}.npz", **out)
    with open(f"{OUT_DIR}/manifest_{TAG}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {OUT_DIR}/fold{{0..4}}_split_{TAG}.npz "
          f"+ gene_grouped_5fold_{TAG}.npz")


if __name__ == "__main__":
    main()
