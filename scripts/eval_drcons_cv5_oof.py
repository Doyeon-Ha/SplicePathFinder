"""Evaluate the current SplicePathFinder head on five held-out gene folds.

All data/model paths are explicit. Inputs are checked for identical cohort row
order before any prediction. Batch 64, full precision, and partial batches are
preserved. Outputs contain pooled probabilities and MANE distance-stratum metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model.sav_classifier import RbfMamba2SavClassifier
from train.dataset_sav import SavFeatureDataset
from scripts.table9_strata import TABLE9_STRATA, table9_masks
from scripts.verify_clinvar_inputs import validate
from process_input.variant_identity import decode


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        channels = b['channels'].to(device)
        scalars = b['scalars'].to(device) if 'scalars' in b else None
        marker = b['marker'].to(device) if model.use_marker else None
        out.append(torch.sigmoid(model(channels, marker, scalars)).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--cvdir', required=True, help='Directory containing fold0..fold4/best.pt')
    p.add_argument('--split-dir', required=True)
    p.add_argument('--split-tag', default='cc_seed42')
    p.add_argument('--coord', required=True)
    p.add_argument('--variants-h5', required=True)
    p.add_argument('--scalars', help='Required for the six-conservation-feature head')
    p.add_argument('--out', required=True, help='Metrics JSON; predictions use the same stem plus _preds.npz')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    a = p.parse_args()
    if a.batch_size != 64:
        raise ValueError('The reported head inference protocol requires batch size 64')
    validation = validate(a.cache, a.variants_h5, a.coord, a.scalars, a.split_dir, a.split_tag)
    n = validation['rows']
    with h5py.File(a.variants_h5) as f:
        y = f['label'][:].astype(np.int64)
        ref, alt = decode(f['ref'][:]), decode(f['alt'][:])
    snv = np.asarray([len(r) == len(v) == 1 for r, v in zip(ref, alt)])
    with np.load(a.coord, allow_pickle=True) as cf:
        coord = cf['splice'].astype(bool)
        t9 = table9_masks(cf['intron_nt'], cf['exon_nt'], cf['akind'])
    probabilities = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k in range(5):
        checkpoint = Path(a.cvdir) / f'fold{k}/best.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        kwargs = dict(ck['model_kwargs'])
        if 'num_input_channels' in kwargs:
            kwargs['num_prob_channels'] = kwargs.pop('num_input_channels')
        n_scalar = int(kwargs.get('num_scalar_features', 0))
        if n_scalar and not a.scalars:
            raise ValueError('This head requires --scalars; checkpoint-local filesystem paths are not used')
        if n_scalar not in (0, 6):
            raise ValueError(f'Unsupported scalar feature count {n_scalar}')
        cfg = ck['config']
        if int(cfg['train']['batch_size']) != 64:
            raise ValueError('Checkpoint training batch size differs from the paper recipe')
        window = cfg['window']
        mask = cfg.get('mask') or {}
        if mask.get('canonical_h5'):
            raise ValueError('Use the unmasked current head recipe for this evaluator')
        with np.load(Path(a.split_dir) / f'fold{k}_split_{a.split_tag}.npz') as z:
            test_idx = z['test_idx']
        if covered[test_idx].any():
            raise ValueError('A row is assigned to more than one held-out fold')
        channels = cfg.get('channel_indices')
        ds = SavFeatureDataset(
            a.cache, test_idx, window_half_bp=int(window['half_bp']),
            window_left=window.get('left'), window_right=window.get('right'),
            use_coverage=window.get('coverage'), use_marker=bool(kwargs.get('use_marker', False)),
            scalar_features_npz=a.scalars if n_scalar else None,
            scalar_features_key=(cfg.get('feature') or {}).get('scalar_key', 'scalars'),
            channel_indices=tuple(channels) if channels is not None else None,
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False,
                            num_workers=a.num_workers)
        model = RbfMamba2SavClassifier(**kwargs).to(device)
        model.load_state_dict(ck['model_state'])
        probabilities[test_idx] = predict(model, loader, device)
        covered[test_idx] = True
        print(f'fold{k}: predicted {len(test_idx)} rows', flush=True)
        del model, loader, ds
    if not covered.all() or not np.isfinite(probabilities).all():
        raise ValueError('Every cohort row must have exactly one finite held-out prediction')
    path = Path(a.out)
    if path.suffix != '.json':
        raise ValueError('--out must have a .json suffix')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_name(path.stem + '_preds.npz'), y=y, p=probabilities,
             snv=snv, coord_restr=coord, row_key_sha256=validation['row_key_sha256'], **t9)
    masks = [('overall', np.ones(n, bool)), ('SNV', snv), ('indel', ~snv),
             ('coord_restricted_overall', coord), ('coord_restricted_SNV', coord & snv),
             ('coord_restricted_indel', coord & ~snv)]
    masks += [(f't9_{name}', t9[name]) for name in TABLE9_STRATA]
    results = dict(n=n, restricted_to=None, validation=validation, strata={})
    for name, sel in masks:
        count = int(sel.sum())
        if count < 20 or len(np.unique(y[sel])) < 2:
            results['strata'][name] = dict(n=count, note='small/single-class')
        else:
            results['strata'][name] = dict(n=count, n_pos=int(y[sel].sum()),
                auroc=float(roc_auc_score(y[sel], probabilities[sel])),
                auprc=float(average_precision_score(y[sel], probabilities[sel])))
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {path} and {path.stem}_preds.npz', flush=True)


if __name__ == '__main__':
    main()
