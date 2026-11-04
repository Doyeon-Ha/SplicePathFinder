"""Build a cache-row-aligned per-variant conservation SCALAR sidecar for the
DeltaReader (spec 019 scalar-at-pool). For every row of a per-nt prob cache,
reads phastCons/phyloP at the variant genomic position and emits 6 scalars:
    [phastCons@pos, phastCons_mean_defined, phastCons_max,
     phyloP@pos,    phyloP_mean_defined,    phyloP_max]
Window = pos ± HALF (default 2500). NaN(assembly gap) excluded from the mean;
phyloP clipped to +/-C then /C. Output npz {scalars:(N,6)} is indexed by cache
row directly (SavFeatureDataset scalar_features_npz).

Multiprocessing over --workers to keep wall-time low on the ~378k ClinVar cache.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import pyBigWig
from pathlib import Path
from process_input.variant_identity import variant_keys, row_digest, require_equal

HALF = 2500
PHYLOP_CLIP = 10.0
EPS = 1e-8
_PH = _PC = None
_PPA = _PCA = None


def _init(cons_dir):
    global _PH, _PC, _PPA, _PCA
    _PH = pyBigWig.open(f"{cons_dir}/hg38.phyloP100way.bw")
    _PC = pyBigWig.open(f"{cons_dir}/hg38.phastCons100way.bw")
    _PPA = set(_PH.chroms()); _PCA = set(_PC.chroms())


def _resolve(chrom, avail):
    if chrom in avail:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in avail else None


def _one(handle, avail, chrom, pos_1b, clip=None):
    bw = _resolve(chrom, avail)
    p0 = pos_1b - 1
    if bw is None:
        return 0.0, 0.0, 0.0
    clen = int(handle.chroms()[bw])
    lo, hi = max(0, p0 - HALF), min(clen, p0 + HALF + 1)
    v = handle.values(bw, lo, hi, numpy=True)
    m = ~np.isnan(v)
    v = np.nan_to_num(v, nan=0.0)
    if clip is not None:
        v = np.clip(v, -clip, clip) / clip
    at = v[p0 - lo] if 0 <= p0 - lo < len(v) else 0.0
    nd = int(m.sum())
    mean_def = float((v * m).sum() / (nd + EPS)) if nd else 0.0
    return float(at), mean_def, float(v.max())


def _work(chunk):
    chroms, poss = chunk
    out = np.zeros((len(poss), 6), dtype=np.float32)
    for i in range(len(poss)):
        a0, a1, a2 = _one(_PC, _PCA, chroms[i], int(poss[i]))
        b0, b1, b2 = _one(_PH, _PPA, chroms[i], int(poss[i]), clip=PHYLOP_CLIP)
        out[i] = [a0, a1, a2, b0, b1, b2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cons-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument('--variant-h5', help='Row-aligned cohort table for a cache without identity columns')
    args = ap.parse_args()

    with h5py.File(args.cache, "r") as cache:
        source = args.variant_h5 or args.cache
        with h5py.File(source, 'r') as f:
            if not all(k in f for k in ['chrom', 'pos', 'ref', 'alt']):
                raise ValueError('Cache has no identities; provide --variant-h5')
            key = variant_keys(f)
            chrom = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                for x in f['chrom'][:]])
            pos = f['pos'][:].astype(np.int64)
        if len(pos) != len(cache['labels']):
            raise ValueError('Variant table and cache row counts differ')
        require_equal(np.arange(len(pos)), cache['row_index'][:], 'cache row_index')
        if 'key' in cache:
            require_equal(key, [x.decode() if isinstance(x, bytes) else str(x)
                                for x in cache['key'][:]], 'cache identities')
    n = len(pos)
    print(f"cache rows={n}, workers={args.workers}", flush=True)

    nchunks = args.workers * 8
    idx = np.array_split(np.arange(n), nchunks)
    chunks = [(chrom[ix], pos[ix]) for ix in idx]
    from multiprocessing import Pool
    with Pool(args.workers, initializer=_init, initargs=(args.cons_dir,)) as pool:
        parts = []
        for j, r in enumerate(pool.imap(_work, chunks)):
            parts.append(r)
            if j % 8 == 0:
                print(f"  chunk {j}/{nchunks}", flush=True)
    scalars = np.concatenate(parts, axis=0)
    assert len(scalars) == n
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, scalars=scalars, row_key_sha256=row_digest(key),
             columns=np.array(["phastCons_pos", "phastCons_mean", "phastCons_max",
                               "phyloP_pos", "phyloP_mean", "phyloP_max"]))
    print(f"saved {scalars.shape} -> {args.out}", flush=True)
    print("col means:", np.round(scalars.mean(0), 4))


if __name__ == "__main__":
    main()
