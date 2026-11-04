"""Build the interval-max variant of the 6 per-variant conservation scalars.

The canonical sidecar (``process_input/build_conservation_scalar_sidecar.py``)
reads both tracks at ``pos_1b`` only, so for an indel the anchor base is the one
reference base the variant does NOT disturb (ONBOARDING 4.27: anchor -> affected
interval max is worth +0.10 AUROC on coord indels). Here the two ``@pos`` columns
are replaced by the maximum over the *affected reference interval*, reusing the
interval definition and reducer of
``scripts/eval_conservation_raw_clinvar_coord.py``. The four window columns
(mean/max over pos +/- 2500) are copied from the anchor sidecar unchanged, so the
two feature sets differ in exactly one readout and the probe comparison is a
clean one-factor ablation.

Normalisation follows the SIDECAR, not the raw readout, so the two feature sets
stay on one scale: phyloP is clipped to +/-CLIP and divided by CLIP, phastCons is
used as is. NaN (assembly gap) is excluded from the maximum; a variant whose whole
interval is undefined keeps the sidecar's 0.0 fallback and is flagged in
``interval_defined``.

Only the small identity columns of the prob cache are read (chrom/pos/ref/alt) —
never ref_probs/alt_probs (ONBOARDING 6: opening the 40GB arrays costs minutes).

CPU only. Usage: python scripts/build_cons_interval_scalars.py --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import h5py
import numpy as np

from scripts.eval_conservation_raw_clinvar_coord import affected_interval

NAS = "."
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
ANCHOR_NPZ = (
    f"{NAS}/data/eval/clinvar2026_phastcons_probe/clinvar_20260628_conservation_scalars.npz"
)
CONS_DIR = f"{NAS}/data/conservation"
OUT_NPZ = (
    f"{NAS}/data/eval/clinvar2026_phastcons_probe/"
    f"clinvar_20260628_conservation_scalars_intervalmax.npz"
)
CLIP = 10.0  # phyloP clip, matching build_conservation_scalar_sidecar.py
_PH = _PC = None
_PHC = _PCC = None


def _init(cons_dir: str) -> None:
    """Open one bigWig pair per worker process."""
    global _PH, _PC, _PHC, _PCC
    import pyBigWig

    _PH = pyBigWig.open(f"{cons_dir}/hg38.phyloP100way.bw")
    _PC = pyBigWig.open(f"{cons_dir}/hg38.phastCons100way.bw")
    _PHC = _PH.chroms()
    _PCC = _PC.chroms()


def _resolve(chrom: str, avail: dict) -> str | None:
    """Map a chrom label onto the bigWig's own naming (chr-prefixed or not)."""
    if chrom in avail:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in avail else None


def _interval_max(handle, avail: dict, chrom: str, s: int, e: int,
                  clip: float | None) -> tuple[float, int]:
    """Maximum of one track over a 1-based inclusive interval.

    Args:
        handle: open pyBigWig handle.
        avail (dict): the bigWig's chrom -> length map.
        chrom (str): chromosome label.
        s (int): 1-based inclusive interval start.
        e (int): 1-based inclusive interval end.
        clip (float | None): clip to +/-clip then divide by clip, as the sidecar
            does for phyloP. None leaves the values untouched (phastCons).

    Returns:
        tuple[float, int]: ``(interval max, number of defined bases)``. The max is
            0.0 when no base of the interval is defined.
    """
    bw = _resolve(chrom, avail)
    if bw is None:
        return 0.0, 0
    clen = int(avail[bw])
    lo, hi = max(0, s - 1), min(clen, e)
    if hi <= lo:
        return 0.0, 0
    v = np.asarray(handle.values(bw, lo, hi, numpy=True), dtype=np.float64)
    m = ~np.isnan(v)
    if not m.any():
        return 0.0, 0
    d = v[m]
    if clip is not None:
        d = np.clip(d, -clip, clip) / clip
    return float(d.max()), int(m.sum())


def _work(chunk):
    """Reduce one chunk of ``(chrom, s, e)`` triples over both tracks."""
    out = np.zeros((len(chunk), 4), dtype=np.float64)
    for i, (chrom, s, e) in enumerate(chunk):
        out[i, 0], out[i, 1] = _interval_max(_PC, _PCC, chrom, s, e, None)
        out[i, 2], out[i, 3] = _interval_max(_PH, _PHC, chrom, s, e, CLIP)
    return out


def main() -> None:
    """Swap the two anchor columns for interval maxima and write the npz."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--anchor-npz", default=ANCHOR_NPZ)
    ap.add_argument("--cons-dir", default=CONS_DIR)
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    anchor = np.load(args.anchor_npz)
    X = anchor["scalars"].astype(np.float32).copy()
    columns = [str(c) for c in anchor["columns"]]
    assert columns[0] == "phastCons_pos" and columns[3] == "phyloP_pos", columns

    with h5py.File(args.cache, "r") as f:
        chrom = np.array([x.decode() if isinstance(x, bytes) else str(x)
                          for x in f["chrom"][:]])
        pos = f["pos"][:].astype(np.int64)
        ref = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in f["ref"][:]])
        alt = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in f["alt"][:]])
    n = len(pos)
    assert len(X) == n, f"scalars {len(X)} != cache rows {n}"

    spans = np.empty((n, 2), dtype=np.int64)
    kinds = np.empty(n, dtype="<U6")
    for i in range(n):
        s, e, k = affected_interval(int(pos[i]), ref[i], alt[i])
        spans[i] = (s, e)
        kinds[i] = k
    print(f"intervals built: n={n} "
          f"{ {k: int((kinds == k).sum()) for k in np.unique(kinds)} }", flush=True)

    items = [(chrom[i], int(spans[i, 0]), int(spans[i, 1])) for i in range(n)]
    chunks = [items[i:i + args.chunk] for i in range(0, n, args.chunk)]
    with mp.Pool(args.workers, initializer=_init, initargs=(args.cons_dir,)) as pool:
        red = np.vstack(pool.map(_work, chunks))

    n_def = red[:, 1].astype(np.int64) + red[:, 3].astype(np.int64)
    X[:, 0] = red[:, 0].astype(np.float32)   # phastCons interval max
    X[:, 3] = red[:, 2].astype(np.float32)   # phyloP interval max (clipped/CLIP)
    out_columns = list(columns)
    out_columns[0] = "phastCons_intervalmax"
    out_columns[3] = "phyloP_intervalmax"

    snv = kinds == "snv"
    moved = np.abs(X[:, 0] - anchor["scalars"][:, 0]) + np.abs(X[:, 3] - anchor["scalars"][:, 3])
    assert np.allclose(X[snv, 0], anchor["scalars"][snv, 0], atol=1e-6), \
        "SNV interval max must equal the anchor readout"
    assert np.allclose(X[snv, 3], anchor["scalars"][snv, 3], atol=1e-6), \
        "SNV interval max must equal the anchor readout"
    assert np.array_equal(X[:, [1, 2, 4, 5]], anchor["scalars"][:, [1, 2, 4, 5]]), \
        "window columns must be copied unchanged"

    np.savez(args.out, scalars=X, columns=np.array(out_columns),
             interval_start=spans[:, 0], interval_end=spans[:, 1],
             interval_kind=kinds, interval_defined=(n_def > 0))
    print(f"indel rows changed: {int((moved > 1e-6).sum())} / {int((~snv).sum())} "
          f"non-SNV; undefined intervals: {int((n_def == 0).sum())}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
