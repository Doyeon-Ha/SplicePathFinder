"""Check cohort identity/order, labels, masks, scalars and frozen fold coverage."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from process_input.variant_identity import variant_keys, decode, require_equal, row_digest


def validate(cache, variants, coord, scalars, split_dir, split_tag='cc_seed42', anchors=None):
    with h5py.File(variants) as f:
        key = variant_keys(f)
        y = f['label'][:]
    n = len(y)
    digest = row_digest(key)
    with h5py.File(cache) as f:
        if 'key' not in f:
            raise ValueError('Cache has no row identities; rebuild it with assemble_v50mane_reportedgene_cache')
        require_equal(key, decode(f['key'][:]), 'cache identities')
        require_equal(y, f['labels'][:], 'cache labels')
        require_equal(np.arange(n), f['row_index'][:], 'cache row_index')
        if f['ref_probs'].shape != f['alt_probs'].shape or f['ref_probs'].shape[0] != n:
            raise ValueError('Cache probability arrays have incompatible shapes')
    with np.load(coord, allow_pickle=True) as z:
        require_equal(y, z['labels'], 'coordinate labels')
        if str(z['row_key_sha256']) != digest:
            raise ValueError('Coordinate row identity/order differs from the cohort')
    if scalars:
        with np.load(scalars) as z:
            if z['scalars'].shape != (n, 6) or not np.isfinite(z['scalars']).all():
                raise ValueError('Conservation scalars must be a finite (N,6) array')
            if str(z['row_key_sha256']) != digest:
                raise ValueError('Conservation row identity/order differs from the cohort')
    counts = np.zeros(n, dtype=np.int8)
    for k in range(5):
        path = Path(split_dir) / f'fold{k}_split_{split_tag}.npz'
        with np.load(path) as z:
            if str(z['row_key_sha256']) != digest:
                raise ValueError(f'Fold {k} identity/order differs from the cohort')
            parts = [z[name] for name in ['train_idx', 'val_idx', 'test_idx']]
            joined = np.concatenate(parts)
            if not np.array_equal(np.sort(joined), np.arange(n)):
                raise ValueError(f'Fold {k} is not a complete disjoint partition')
            counts[z['test_idx']] += 1
    if not np.all(counts == 1):
        raise ValueError('Every cohort row must occur in exactly one test fold')
    if anchors:
        with np.load(anchors, allow_pickle=True) as z:
            require_equal(key, z['key'], 'anchor identities')
            require_equal(y, z['label'], 'anchor labels')
    return dict(rows=n, row_key_sha256=digest, folds=5, every_row_tested_once=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--variants-h5', required=True)
    p.add_argument('--coord', required=True)
    p.add_argument('--scalars', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--split-tag', default='cc_seed42')
    p.add_argument('--anchor-selection')
    a = p.parse_args()
    print(json.dumps(validate(a.cache, a.variants_h5, a.coord, a.scalars,
                              a.split_dir, a.split_tag, a.anchor_selection), indent=2))


if __name__ == '__main__':
    main()
