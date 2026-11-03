"""Verify native cache controls and compute matched v50 annotation metrics."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_v50_annotation import NAS, REFERENCE, suffixes

REFERENCE_SCORES = NAS/'data/eval/recomb2027/ep12_refresh_20261023/annotation'


def labels(path):
    with h5py.File(path) as f:
        return np.concatenate([f[f'Y{i}'][:] for i in suffixes(f)])


def metrics(score, truth):
    """Exact average precision/ROC area; stable row order breaks top-k ties."""
    positive = truth.astype(bool)
    n, p = len(score), int(positive.sum())
    assert 0 < p < n
    assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1
    if score.dtype == np.float16:
        bits = score.view(np.uint16)
        counts = np.bincount(bits, minlength=65536)
        pos = np.bincount(bits[positive], minlength=65536)
        occupied = counts > 0
        counts, pos = counts[occupied], pos[occupied]
        threshold = np.partition(score, n-p)[n-p]
        higher = score > threshold
        remaining = p - int(higher.sum())
        tied = np.flatnonzero(score == threshold)
        hit = int(positive[higher].sum())
        if remaining:
            hit += int(positive[tied[-remaining:]].sum())
    else:
        order = np.argsort(score, kind='stable')
        hit = int(positive[order[-p:]].sum())
        ordered = score[order]
        ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), n-1]
        cum_pos = np.cumsum(positive[order], dtype=np.int64)
        pos = np.diff(np.r_[0, cum_pos[ends]])
        counts = np.diff(np.r_[0, ends+1])
        del order, ordered, ends, cum_pos
    neg = counts - pos
    tp = np.cumsum(pos[::-1], dtype=np.int64)
    called = np.cumsum(counts[::-1], dtype=np.int64)
    ap = float(np.sum(pos[::-1] * tp / called) / p)
    neg_below = np.cumsum(neg, dtype=np.int64) - neg
    roc = float(np.sum(pos * (neg_below + .5*neg)) / (p*(n-p)))
    return {'auprc':ap, 'auroc':roc, 'topk_accuracy':hit/p, 'n_positive':p, 'n_valid_positions':n}


def verify_metrics():
    from sklearn.metrics import average_precision_score, roc_auc_score
    truth = np.asarray([0, 1, 0, 1, 1, 0, 0, 1])
    for dtype in (np.float16, np.float32):
        score = np.asarray([0, 0, .2, .2, .8, .8, 1, 1], dtype=dtype)
        result = metrics(score, truth)
        np.testing.assert_allclose(result['auprc'], average_precision_score(truth, score), atol=1e-12)
        np.testing.assert_allclose(result['auroc'], roc_auc_score(truth, score), atol=1e-12)
        assert result['topk_accuracy'] == truth[np.argsort(score, kind='stable')[-truth.sum():]].mean()


def reference_scores(tool, ref_y):
    expected = ref_y[ref_y >= 0]
    channels = 1 if tool == 'pangolin' else 2
    scores = np.empty((len(expected), channels), np.float16)
    offset = 0
    if tool == 'pangolin':
        with h5py.File(REFERENCE_SCORES/'pangolin/cache.h5') as f:
            assert f['splice4'].shape == (len(ref_y), 5000, 4)
            if 'row_index' in f:
                np.testing.assert_array_equal(f['row_index'][:], np.arange(len(ref_y)))
            for start in range(0, len(ref_y), 256):
                y = ref_y[start:start+256]
                p = f['splice4'][start:start+256].max(axis=-1)[y >= 0]
                scores[offset:offset+len(p), 0] = p
                offset += len(p)
    else:
        paths = sorted((REFERENCE_SCORES/tool).glob('test0.shard*_of*.npz'),
                       key=lambda p:int(re.search(r'shard(\d+)_',p.name)[1]))
        assert len(paths) == (4 if tool == 'st' else 8)
        for i, path in enumerate(paths):
            assert f'shard{i}_of{len(paths)}' in path.name
            with np.load(path) as z:
                y = z['labels']
                np.testing.assert_array_equal(y, expected[offset:offset+len(y)])
                scores[offset:offset+len(y), 0] = z['acc_scores']
                scores[offset:offset+len(y), 1] = z['don_scores']
                offset += len(y)
    assert offset == len(expected)
    return scores


def assemble_native(run, tool, y, mapping):
    ref_y = labels(REFERENCE)
    old = reference_scores(tool, ref_y)
    counts = (ref_y >= 0).sum(axis=1)
    np.testing.assert_array_equal(counts, mapping['reference_valid_counts'])
    old_offset = np.r_[0, np.cumsum(counts)]
    del ref_y
    target_offset = np.r_[0, np.cumsum(mapping['valid_counts'])]
    reference_rows = mapping['reference_rows']
    target = np.empty((int(target_offset[-1]), old.shape[1]), np.float16)
    assigned = np.zeros(len(y), bool)
    for row, old_row in enumerate(reference_rows):
        if old_row >= 0:
            target[target_offset[row]:target_offset[row+1]] = old[old_offset[old_row]:old_offset[old_row+1]]
            assigned[row] = True
    del old
    with h5py.File(run/tool/'inference.h5') as f:
        assert f.attrs['complete']
        new_scores = f['scores'][:]
        new_labels = f['labels'][:]
        runtime = json.loads(f.attrs['metadata'])
    control_max = 0.0
    control_positions = 0
    offset = 0
    for row in mapping['inference_rows']:
        expected = y[row][y[row] >= 0]
        prob = new_scores[offset:offset+len(expected)]
        np.testing.assert_array_equal(new_labels[offset:offset+len(expected)], expected)
        dst = slice(target_offset[row], target_offset[row+1])
        if reference_rows[row] >= 0:
            a, b = target[dst].astype(np.float32), prob.astype(np.float32)
            error = float(np.max(np.abs(a-b)))
            # Allow native float16 rounding plus small hardware/runtime drift.
            assert error <= 1e-3, (tool, int(row), error)
            np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-5)
            control_max = max(control_max, error)
            control_positions += len(expected)
        else:
            target[dst] = prob
            assigned[row] = True
        offset += len(expected)
    assert offset == len(new_labels) == len(new_scores)
    assert assigned.all()
    report = {'reference_scores':str(REFERENCE_SCORES/tool), 'inference':runtime,
              'reused_windows':int((reference_rows >= 0).sum()),
              'new_windows':int((reference_rows < 0).sum()), 'control_windows':len(mapping['control_rows']),
              'control_positions':control_positions, 'control_max_absolute_error':control_max,
              'control_passed':True, 'control_atol':2e-5, 'control_rtol':2e-3, 'control_max_error_limit':1e-3}
    expected = y[y >= 0]
    with h5py.File(run/tool/'scores.h5', 'w') as f:
        f.create_dataset('scores', data=target)
        f.create_dataset('labels', data=expected)
        f.attrs['complete'] = True
        f.attrs['metadata'] = json.dumps(report)
    return target, expected, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tool', choices=['splicepath','spliceai','pangolin','st','ssnet'], required=True)
    p.add_argument('--fresh-native', action='store_true', help='Use complete native inference instead of cache reuse.')
    args = p.parse_args()
    verify_metrics()
    run, tool = args.run, args.tool
    dataset = json.loads((run/'dataset.json').read_text())
    y = labels(run/'dataset_test_0.h5')
    expected = y[y >= 0]
    assert len(expected) == dataset['valid_positions']
    assert int((expected == 1).sum()) == dataset['acceptors']
    assert int((expected == 2).sum()) == dataset['donors']
    if tool == 'splicepath' or args.fresh_native:
        parts = sorted((run/tool).glob('scores.shard*_of*.h5'),
                       key=lambda p:int(re.search(r'shard(\d+)_',p.name)[1]))
        if parts:
            target = run/tool/'scores.h5'
            assert not target.exists(), target
            offset, row_end, metadata = 0, 0, []
            with h5py.File(target, 'w') as dst:
                dtype = 'float32' if tool == 'splicepath' else 'float16'
                batch_size = 8 if tool == 'ssnet' else 16
                output = dst.create_dataset('scores', (len(expected), 2), dtype=dtype)
                dst.create_dataset('labels', data=expected)
                for i, path in enumerate(parts):
                    with h5py.File(path) as src:
                        info = json.loads(src.attrs['metadata'])
                        assert src.attrs['complete'] and info['num_shards'] == len(parts)
                        assert info['shard_index'] == i and info['row_start'] == row_end
                        assert info['row_start'] % batch_size == 0 and info['batch_size'] == batch_size
                        p, lab = src['scores'][:], src['labels'][:]
                        np.testing.assert_array_equal(lab, expected[offset:offset+len(lab)])
                        output[offset:offset+len(lab)] = p
                        offset += len(lab); row_end = info['row_end']; metadata.append(info)
                assert offset == len(expected) and row_end == dataset['windows']
                provenance = {'batch_size':batch_size, 'windows':row_end, 'inference_parts':metadata,
                              'batch_grouping':'contiguous original batches; only final batch may be incomplete'}
                dst.attrs['complete'] = True
                dst.attrs['metadata'] = json.dumps(provenance)
        with h5py.File(run/tool/'scores.h5') as f:
            assert f.attrs['complete']
            score = f['scores'][:]
            np.testing.assert_array_equal(f['labels'][:], expected)
            provenance = json.loads(f.attrs['metadata'])
        assert provenance['batch_size'] == (8 if tool == 'ssnet' else 16)
        assert provenance['windows'] == dataset['windows']
    else:
        with np.load(run/'mapping.npz') as mapping:
            score, expected, provenance = assemble_native(run, tool, y, mapping)
    del y
    gc.collect()
    result = {'tool':tool, 'dataset':dataset, 'provenance':provenance, 'metrics':{},
              'score_file':str(run/tool/'scores.h5'), 'topk_tie_policy':'stable input order (last rows at boundary)'}
    for channel, name in enumerate(['splice'] if tool == 'pangolin' else ['acceptor','donor']):
        truth = expected > 0 if name == 'splice' else expected == channel+1
        result['metrics'][name] = metrics(np.ascontiguousarray(score[:, channel]), truth)
        print(tool, name, result['metrics'][name], flush=True)
        gc.collect()
    (run/tool/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
