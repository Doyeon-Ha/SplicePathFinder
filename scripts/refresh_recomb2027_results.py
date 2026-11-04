"""Shared identity, gene-component and weighted-bootstrap helpers."""
import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def dec(a):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)

def keys(c, p, r, a):
    return np.asarray([f'{str(ch).removeprefix("chr")}:{int(po)}:{re}:{al}'
                       for ch, po, re, al in zip(dec(c), p, dec(r), dec(a))], dtype=object)

def table(path):
    with h5py.File(path) as f:
        k = keys(f['chrom'][:], f['pos'][:], f['ref'][:], f['alt'][:])
        y = f['label'][:].astype(np.int8)
        snv = dec(f['variant_kind'][:]) == 'snv'
    assert len(set(k)) == len(k), f'duplicate variant in {path}'
    return k, y, snv

def metric(y, s):
    assert len(y) == len(s) and np.isfinite(s).all()
    r = dict(n=len(y), n_pos=int(y.sum()))
    if len(np.unique(y)) < 2:
        return r | dict(auroc=None, auprc=None)
    return r | dict(auroc=float(roc_auc_score(y, s)),
                    auprc=float(average_precision_score(y, s)))

def components(gene):
    missing = {'', '.', 'NA', 'nan', 'None'}
    tokens = [[t for t in g.split('|') if t not in missing] for g in dec(gene)]
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ts in tokens:
        for t in ts[1:]:
            parent[root(ts[0])] = root(t)
    return np.asarray([root(ts[0]) if ts else f'__nogene_{i}'
                       for i, ts in enumerate(tokens)])

class Ranker:
    """Fixed score order with weighted ties; checked against sklearn before use."""
    def __init__(self, y, s):
        self.order = np.argsort(s, kind='stable')
        self.y = y[self.order]
        self.group = np.r_[0, np.cumsum(np.diff(s[self.order]) != 0)]

    def score(self, weights):
        w = weights[self.order]
        p = np.bincount(self.group, weights=w * self.y)
        n = np.bincount(self.group, weights=w * (1 - self.y))
        if p.sum() == 0 or n.sum() == 0:
            return np.array([np.nan, np.nan])
        auc = np.sum(p * (np.cumsum(n) - .5 * n)) / (p.sum() * n.sum())
        tp, fp = np.cumsum(p[::-1]), np.cumsum(n[::-1])
        ap = np.sum(p[::-1] * tp / np.maximum(tp + fp, 1e-12)) / p.sum()
        return np.array([auc, ap])
