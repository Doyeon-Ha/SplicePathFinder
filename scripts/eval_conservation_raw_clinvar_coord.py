"""Training-free raw-conservation pathogenicity readout on the ClinVar-2026
splice-region coordinate universe (n=106,504 = `cv5_coord_restricted`).

Answers "how far does conservation alone get you?" with NO learned probe: the
score IS the conservation value, so AUROC needs no fold split (contrast with the
registry's `cons6_only` / `cons6_only_logreg`, which fit a 6-scalar probe under
gene-grouped 5-fold CV).

Indel policy — one rule for every variant class. Each variant is reduced to its
minimal representation (right-trim then left-trim of shared bases), giving the
*affected reference interval* = the reference bases the variant actually
disturbs; the same reducer is then applied to all classes so SNV / deletion /
insertion sit on one ROC curve:

    SNV        POS                      e.g. 1048791 C>T          -> {1048791}
    deletion   deleted bases only       e.g. 964529 AGGTCCGCAGTG>A
                                             (anchor base unchanged)
                                                                  -> 964530..964541
    insertion  the two bases flanking   e.g. 1040717 G>GGGGCC
               the insertion point      (inserted bases have no
               (no reference base is     conservation value)       -> {1040717, 1040718}
               removed)
    delins/MNV replaced REF interval    e.g. 5927667 GTT>CCACG    -> 5927667..5927669

Primary reducer = max over the interval (pathogenicity is driven by the single
most constrained base the variant removes; mean dilutes with interval length).
`anchor` (the SNV-style POS-only readout used by the existing conservation
sidecar) and `interval_mean` are emitted as controls, together with an
indel-length-only control, because max over an interval grows with interval
length and deletion length correlates with pathogenicity.

Score arms: phyloP alone, phastCons alone, and their average. Single-track AUROC
is invariant to monotone rescaling, so only the average arm depends on
normalisation: it is reported both on the project's convention (phyloP clipped
to +/-C then /C, phastCons as-is) and on within-universe percentile ranks.

NaN (assembly gap) is never filled with 0 — it is excluded from the reduction,
and a variant whose whole interval is undefined is a no-call excluded from that
arm's metric with the coverage recorded.

CPU only (bigWig reads + sklearn). Writes an npz of per-variant readouts and a
JSON of AUROC/AUPRC per stratum.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table9_strata import TABLE9_STRATA, table9_masks  # noqa: E402

NAS = "."
FOLDSETS = f"{NAS}/data/eval/clinvar_pb_2026_split/coord_splice_foldsets"
COORD = f"{NAS}/data/eval/clinvar_pb_2026_split/coord_splice_fields_seed42.npz"
CONS_DIR = f"{NAS}/data/conservation"
OUTDIR = f"{NAS}/data/eval/clinvar_conservation_raw"

PHYLOP_CLIP = 10.0
EXPECT_N = 106_504

_T0 = time.time()


def _tick(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# variant -> affected reference interval
# --------------------------------------------------------------------------- #
def affected_interval(pos: int, ref: str, alt: str) -> tuple[int, int, str]:
    """Minimal-representation affected reference interval, 1-based inclusive.

    Args:
        pos (int): 1-based VCF POS.
        ref (str): REF allele.
        alt (str): ALT allele.

    Returns:
        tuple[int, int, str]: ``(start, end, kind)`` where kind is one of
            ``snv`` / ``del`` / ``ins`` / ``delins``. For an insertion the
            interval is the flanking base pair around the insertion point.
    """
    r, a, p = ref, alt, int(pos)
    while r and a and r[-1] == a[-1]:          # right-trim shared suffix
        r, a = r[:-1], a[:-1]
    while r and a and r[0] == a[0]:            # left-trim shared prefix
        r, a, p = r[1:], a[1:], p + 1
    if not r and not a:
        raise ValueError(f"REF==ALT at {pos} {ref}>{alt}")
    if not r:                                  # pure insertion before p
        return max(p - 1, 1), p, "ins"
    if not a:                                  # pure deletion of p..p+len(r)-1
        return p, p + len(r) - 1, "del"
    kind = "snv" if len(r) == 1 and len(a) == 1 else "delins"
    return p, p + len(r) - 1, kind


# --------------------------------------------------------------------------- #
# bigWig workers
# --------------------------------------------------------------------------- #
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
    if chrom in avail:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in avail else None


def _reduce(handle, avail, chrom: str, pos: int, s: int, e: int) -> tuple:
    """Read one variant's readouts from a single bigWig.

    Args:
        handle: open pyBigWig handle.
        avail (dict): the bigWig's chrom -> length map.
        chrom (str): chromosome label.
        pos (int): 1-based VCF POS (anchor readout).
        s (int): 1-based inclusive interval start.
        e (int): 1-based inclusive interval end.

    Returns:
        tuple: ``(anchor, imax, imean, n_def, n_tot)``; NaN where undefined.
    """
    bw = _resolve(chrom, avail)
    if bw is None:
        return np.nan, np.nan, np.nan, 0, e - s + 1
    clen = int(avail[bw])
    lo = max(0, min(pos - 1, s - 1))
    hi = min(clen, max(pos, e))
    if hi <= lo:
        return np.nan, np.nan, np.nan, 0, e - s + 1
    v = np.asarray(handle.values(bw, lo, hi, numpy=True), dtype=np.float64)

    ai = pos - 1 - lo
    anchor = v[ai] if 0 <= ai < v.size else np.nan

    i0, i1 = s - 1 - lo, e - lo                      # interval slice in v
    iv = v[max(i0, 0):max(i1, 0)]
    n_tot = e - s + 1
    if iv.size == 0:
        return anchor, np.nan, np.nan, 0, n_tot
    m = ~np.isnan(iv)
    if not m.any():
        return anchor, np.nan, np.nan, 0, n_tot
    d = iv[m]
    return anchor, float(d.max()), float(d.mean()), int(m.sum()), n_tot


def _work(chunk):
    out = np.full((len(chunk), 10), np.nan)
    for i, (chrom, pos, s, e) in enumerate(chunk):
        out[i, 0:5] = _reduce(_PH, _PHC, chrom, pos, s, e)
        out[i, 5:10] = _reduce(_PC, _PCC, chrom, pos, s, e)
    return out


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #
def load_universe(foldsets: str) -> dict:
    """Load the coord-restricted universe from the 5 fold test TSVs.

    The five ``fold*_test.tsv`` files partition the coordinate-restricted
    splice-region universe and carry ``global_idx`` (row index into the cache /
    coord arrays), the variant identity, the label and the nine Table-9 stratum
    flags -- so the 40 GB per-nt cache never has to be opened.

    Only the columns this script needs are kept, and REF/ALT stay Python strings:
    the widest allele in this universe is 9,558 bp, so a fixed-width numpy array
    of them would cost ~7.5 GB.

    Args:
        foldsets (str): directory holding ``fold{0..4}_test.tsv``.

    Returns:
        dict: column name -> ndarray (REF/ALT as lists of str), sorted by
            ``global_idx``.
    """
    paths = sorted(glob.glob(f"{foldsets}/fold*_test.tsv"))
    assert len(paths) == 5, paths
    want = ["global_idx", "chrom", "pos", "ref", "alt", "label", "variant_kind"] + TABLE9_STRATA
    hdr = None
    cols: dict[str, list] = {k: [] for k in want}
    for p in paths:
        with open(p) as fh:
            h = fh.readline().rstrip("\n").split("\t")
            hdr = h if hdr is None else hdr
            assert h == hdr, f"header mismatch in {p}"
            take = [(hdr.index(k), k) for k in want]
            for line in fh:
                f = line.rstrip("\n").split("\t")
                for i, k in take:
                    cols[k].append(f[i])
    order = np.argsort(np.asarray(cols["global_idx"], dtype=np.int64), kind="stable")
    out: dict[str, object] = {}
    for k, v in cols.items():
        if k in ("ref", "alt"):
            out[k] = [v[i] for i in order]          # keep as str; alleles are long
        elif k in ("global_idx", "pos", "label") or k in TABLE9_STRATA:
            out[k] = np.asarray(v, dtype=np.int64)[order]
        else:
            out[k] = np.asarray(v)[order]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldsets", default=FOLDSETS)
    ap.add_argument("--coord", default=COORD)
    ap.add_argument("--expected-n", type=int, default=EXPECT_N)
    ap.add_argument("--cons-dir", default=CONS_DIR)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test row cap")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    u = load_universe(args.foldsets)
    gidx = u["global_idx"].astype(np.int64)
    chrom, ref, alt = u["chrom"], u["ref"], u["alt"]
    pos = u["pos"].astype(np.int64)
    y = u["label"].astype(np.int64)
    vkind = u["variant_kind"]
    n = len(y)
    _tick(f"universe n={n} pos={int(y.sum())}")

    # ---- cross-check against the coord npz (row-aligned to the cache) ----
    cf = np.load(args.coord)
    sp = cf["splice"].astype(bool)
    assert n == args.expected_n == int(sp.sum()), (n, args.expected_n, int(sp.sum()))
    assert np.array_equal(gidx, np.where(sp)[0]), "global_idx != coord splice rows"
    assert np.array_equal(y, cf["labels"][sp].astype(np.int64)), "label mismatch"
    t9_ref = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])
    t9 = {}
    for name in TABLE9_STRATA:
        t9[name] = u[name].astype(np.int64).astype(bool)
        assert np.array_equal(t9[name], t9_ref[name][sp]), f"t9 mismatch {name}"
    _tick("universe cross-checks passed (identity, labels, 9 Table-9 masks)")

    # ---- affected intervals ----
    iv_s = np.zeros(n, np.int64)
    iv_e = np.zeros(n, np.int64)
    kind = np.empty(n, object)
    for i in range(n):
        iv_s[i], iv_e[i], kind[i] = affected_interval(pos[i], ref[i], alt[i])
    kind = kind.astype("U6")
    iv_len = iv_e - iv_s + 1
    rlen = np.array([len(s) for s in ref], np.int64)
    alen = np.array([len(s) for s in alt], np.int64)
    dlen = np.abs(rlen - alen)
    _tick("intervals: " + " ".join(f"{k}={int((kind == k).sum())}" for k in
                                   ("snv", "del", "ins", "delins"))
          + f"  interval_len max={int(iv_len.max())}")

    if args.limit:
        n = min(n, args.limit)
        _tick(f"LIMIT active: first {n} rows only (smoke test)")

    # ---- bigWig reduction ----
    items = list(zip(chrom[:n], pos[:n], iv_s[:n], iv_e[:n]))
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    res = np.empty((n, 10))
    filled = 0
    with Pool(args.workers, initializer=_init, initargs=(args.cons_dir,)) as pool:
        for ci, out in enumerate(pool.imap(_work, chunks, chunksize=1)):
            res[filled:filled + len(out)] = out
            filled += len(out)
            if ci % 20 == 0:
                _tick(f"bigWig {filled}/{n}")
    assert filled == n
    _tick("bigWig reduction done")

    read = {
        "phylop": {"anchor": res[:, 0], "interval_max": res[:, 1], "interval_mean": res[:, 2]},
        "phastcons": {"anchor": res[:, 5], "interval_max": res[:, 6], "interval_mean": res[:, 7]},
    }
    n_def = {"phylop": res[:, 3], "phastcons": res[:, 8]}

    np.savez(
        f"{args.outdir}/cons_raw_readouts.npz",
        global_idx=gidx[:n], chrom=chrom[:n], pos=pos[:n],
        ref_len=rlen[:n], alt_len=alen[:n],   # alleles themselves rejoin via global_idx
        label=y[:n], variant_kind=vkind[:n], kind=kind[:n],
        iv_start=iv_s[:n], iv_end=iv_e[:n], iv_len=iv_len[:n], len_delta=dlen[:n],
        phylop_anchor=res[:, 0], phylop_imax=res[:, 1], phylop_imean=res[:, 2],
        phylop_ndef=res[:, 3], phylop_ntot=res[:, 4],
        phastcons_anchor=res[:, 5], phastcons_imax=res[:, 6], phastcons_imean=res[:, 7],
        phastcons_ndef=res[:, 8], phastcons_ntot=res[:, 9],
        **{f"t9_{k}": v[:n] for k, v in t9.items()},
    )
    _tick(f"wrote {args.outdir}/cons_raw_readouts.npz")

    # ---- score arms ----
    def rankpct(v):
        """Within-universe percentile rank; NaN preserved."""
        out = np.full(v.shape, np.nan)
        m = ~np.isnan(v)
        if m.sum() > 1:
            o = np.argsort(v[m], kind="stable")
            r = np.empty(int(m.sum()))
            r[o] = np.arange(m.sum()) / (m.sum() - 1)
            out[m] = r
        return out

    arms = {}
    for ro in ("anchor", "interval_max", "interval_mean"):
        ph, pc = read["phylop"][ro], read["phastcons"][ro]
        arms[("phylop", ro)] = ph
        arms[("phastcons", ro)] = pc
        arms[("mean_norm", ro)] = (np.clip(ph, -PHYLOP_CLIP, PHYLOP_CLIP) / PHYLOP_CLIP + pc) / 2.0
        arms[("mean_rank", ro)] = (rankpct(ph) + rankpct(pc)) / 2.0
    arms[("indel_length", "control")] = dlen[:n].astype(float)

    # ---- strata ----
    snv = vkind[:n] == "snv"
    k = kind[:n]
    dl = iv_len[:n]
    strata = [
        ("coord_restricted", np.ones(n, bool)),
        ("coord_SNV", snv),
        ("coord_indel", ~snv),
        ("coord_deletion", k == "del"),
        ("coord_insertion", k == "ins"),
        ("coord_delins", k == "delins"),
        ("coord_del_len1", (k == "del") & (dl == 1)),
        ("coord_del_len2_5", (k == "del") & (dl >= 2) & (dl <= 5)),
        ("coord_del_len6plus", (k == "del") & (dl >= 6)),
    ]
    strata += [(f"t9_{name}", t9[name][:n]) for name in TABLE9_STRATA]

    def met(sel, s):
        m = sel & ~np.isnan(s)
        yy = y[:n][m]
        if yy.size < 20 or yy.min() == yy.max() or np.unique(s[m]).size < 2:
            return None
        return {"auroc": round(float(roc_auc_score(yy, s[m])), 4),
                "auprc": round(float(average_precision_score(yy, s[m])), 4),
                "n_covered": int(yy.size)}

    order = [("phylop", "anchor"), ("phylop", "interval_max"), ("phylop", "interval_mean"),
             ("phastcons", "anchor"), ("phastcons", "interval_max"), ("phastcons", "interval_mean"),
             ("mean_norm", "anchor"), ("mean_norm", "interval_max"), ("mean_norm", "interval_mean"),
             ("mean_rank", "anchor"), ("mean_rank", "interval_max"), ("mean_rank", "interval_mean"),
             ("indel_length", "control")]

    res_json = {
        "universe": "clinvar_20260628 coord-restricted splice-region union",
        "n": n, "n_pos": int(y[:n].sum()),
        "phylop_clip": PHYLOP_CLIP,
        "nocall": {t: {"all_undefined_interval": int((nd == 0).sum())} for t, nd in n_def.items()},
        "arms": ["|".join(a) for a in order],
        "strata": {},
    }
    hdr = f"{'stratum':22s} {'n':>7s} {'pos':>6s} | " + " | ".join(f"{a}/{b}"[:15].rjust(15) for a, b in order)
    print("\n" + hdr, flush=True)
    for name, sel in strata:
        row = {"n": int(sel.sum()), "n_pos": int(y[:n][sel].sum()), "by_arm": {}}
        cells = []
        for a in order:
            r = met(sel, arms[a])
            row["by_arm"]["|".join(a)] = r
            cells.append(f"{r['auroc']:.3f}/{r['auprc']:.3f}".rjust(15) if r else "--".rjust(15))
        res_json["strata"][name] = row
        print(f"{name:22s} {row['n']:7d} {row['n_pos']:6d} | " + " | ".join(cells), flush=True)

    out = f"{args.outdir}/cons_raw_strata.json"
    json.dump(res_json, open(out, "w"), indent=1)
    _tick(f"wrote {out}")


if __name__ == "__main__":
    main()
