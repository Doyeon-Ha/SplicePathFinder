"""Gene-grouped 5-fold CV split whose groups are token connected components.

``make_gene_grouped_5fold.py`` groups on the raw ``gene_symbol`` string, so
``ABCA4`` and ``ABCA4|LOC126805793`` become separate groups and the same gene
lands in several folds. Here the group key is instead the connected component of
the gene-token co-occurrence graph: two tokens are joined whenever they appear in
one key, and a component is one group. No naming heuristic is involved, so
overlapping loci (readthrough transcripts, antisense partners, LOC tiles) stay
together by construction.

Everything else matches the canonical builder: StratifiedGroupKFold(seed 42), the
test=k / val=(k+1)%5 / train=rest rotation, and the same per-fold npz layout, so
the split is a drop-in for ``split.freeze_npz``.

Read-only w.r.t. the canonical split. Inputs/outputs on /NAS.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import h5py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

NAS = "."
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
OUT_DIR = f"{NAS}/data/eval/clinvar_pb_2026_split_cc"
TAG = "cc_seed42"
N_FOLDS, SEED = 5, 42
MISSING = {"", ".", "NA", "nan", "None"}


def dec(a: np.ndarray) -> np.ndarray:
    """Decode an h5 string array to a numpy array of str."""
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in a])


def main() -> None:
    """Build and verify the connected-component folds."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(CACHE, "r") as f:
        gene = dec(f["gene_symbol"][:])
        y = f["labels"][:].astype(int)
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
    out = {"fold_id": fold_id}
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
