"""delta max (label-free) from the head's K=4 target-only per-nt cache.

The cache keeps, at every position, the chunk maximising
``max(|d_acceptor|, |d_donor|)`` and carries that chunk's ref AND alt over
together. delta_max's headline is the maximum over the four signed directions,
which at any position equals exactly that quantity -- so the cache's selection
rule and delta_max's aggregation rule rank on the same number, and the headline
is recoverable here without running the backbone again.

The default reads all 8,750 cached positions, offsets [-4,375, +4,375).
Masked scoring requires a cache built with per-chunk masking before position-
wise selection; a raw cache cannot recover the exact masked score afterward.

delta_mass (L1 over the window) comes along because it costs one more line and
breaks ties among the many canonical variants that saturate delta_max at ~1.0.
It is an alternative aggregation, never the headline.
"""
from __future__ import annotations

import argparse
import csv
import logging

import h5py
import numpy as np

from scripts.canonical_site_track import (
    apply_canonical_mask,
    load_canonical_sites,
    load_sites_by_accession,
)

logger = logging.getLogger(__name__)

DIRECTIONS = ("acceptor_gain", "acceptor_loss", "donor_gain", "donor_loss")
_EMPTY = np.zeros(0, dtype=np.int64)


def row_to_ref_genomic(pos: int, ref: str, alt: str, minus: bool,
                       n_rows: int, var_row: int) -> np.ndarray:
    """정렬 축의 각 행이 가리키는 **ref 유전체 위치**를 만든다.

    v4 정렬은 유전체 축이 아니다 -- 변이 구간에 결실 행(ref 실측/alt 0)과 삽입
    행(ref 0/alt 실측)을 끼워 넣고, `-` strand에서는 그 행들을 anchor **앞**에
    놓는다. 그래서 상수 offset으로 옮기면 마스크가 밀린다. 결실 행은 참조에
    존재하는 염기이므로 유전체 위치를 갖고(주석 site일 수 있다), 삽입 행은
    참조에 없으므로 위치가 없다.

    SNV(`len(ref)==len(alt)==1`)에서는 단순 상수 offset 맵으로 환원된다.

    Args:
        pos (int): 변이의 1-based VCF 위치(왼쪽 anchor).
        ref (str): 참조 대립유전자.
        alt (str): 대체 대립유전자.
        minus (bool): 앵커 유전자가 `-` strand인가.
        n_rows (int): 축의 행 수.
        var_row (int): 변이 구간이 시작하는 행.

    Returns:
        np.ndarray: 행별 ref 유전체 위치. 삽입 행은 -1.
    """
    lr, la = len(ref), len(alt)
    ov = min(lr, la)
    n_ins = la - ov
    out = np.full(n_rows, -1, dtype=np.int64)
    if minus:
        # 결실 행(pos+lr-1 -> pos+ov) → 삽입 행 → overlap 행(pos+ov-1 -> pos)
        region = ([pos + lr - 1 - j for j in range(lr - ov)] + [-1] * n_ins
                  + [pos + ov - 1 - j for j in range(ov)])
        up = lambda k: pos + lr - 1 + k          # noqa: E731 - 행 k칸 위
        down = lambda k: pos - 1 - k             # noqa: E731 - 행 k칸 아래
    else:
        region = [pos + j for j in range(lr)] + [-1] * n_ins
        up = lambda k: pos - k                   # noqa: E731
        down = lambda k: pos + lr + k            # noqa: E731
    for j, g in enumerate(region):
        r = var_row + j
        if 0 <= r < n_rows:
            out[r] = g
    for k in range(1, var_row + 1):
        if var_row - k >= 0:
            out[var_row - k] = up(k)
    tail = var_row + len(region)
    for k in range(n_rows - tail):
        out[tail + k] = down(k)
    return out


def dec(a: np.ndarray) -> np.ndarray:
    """bytes/str 배열을 str 배열로 정규화한다.

    Args:
        a (np.ndarray): 입력 배열.

    Returns:
        np.ndarray: str dtype 배열.
    """
    # Object strings avoid expanding one long ClinVar allele to every row's
    # fixed-width Unicode storage (tens of GB for the full release).
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                      dtype=object)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 파싱된 인자.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--source-h5", required=True,
                   help="gencode-anchored table the cache rows align to")
    p.add_argument("--out", required=True)
    p.add_argument("--window-bp", type=int, default=None,
                   help="Optional symmetric inclusive legacy crop (+/- bp). "
                        "Omit to score the full cached 8,750-position window.")
    p.add_argument("--block-size", type=int, default=2000)
    p.add_argument("--mask-canonical", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Require a cache masked before merging both SNV and "
                        "indel chunks; omitted means unmasked scoring")
    p.add_argument("--legacy-postmerge-mask", action="store_true",
                   help="Reproduce historical lower-bound masked scores from "
                        "an unmasked cache; never use for the new method")
    p.add_argument("--annotation", default=None,
                   help="--mask-canonical에 필요한 7컬럼 annotation TSV(유전자 키)")
    p.add_argument("--transcript-annotation", default=None,
                   help="NM 접근번호 키 annotation. `--variant-transcript-csv`와 "
                        "함께 주면 변이가 **실제로 해석된 전사체**의 junction으로 "
                        "마스크를 만든다. 유전자당 대표 전사체만 쓰면 임상 보고가 "
                        "쓰는 전사체와 달라 canonical loss가 조용히 사라진다.")
    p.add_argument("--variant-transcript-csv", default=None,
                   help="cpra_grch38,transcript 컬럼을 가진 CSV(사내 코호트 원본 가능)")
    return p.parse_args()


def main() -> None:
    """CLI 엔트리포인트."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    with h5py.File(args.cache, "r") as f:
        cache_masked = bool(f.attrs.get("mask_canonical", False))
        mask_application = f.attrs.get("mask_application", "none")
    if cache_masked != args.mask_canonical and not args.legacy_postmerge_mask:
        raise SystemExit(
            "cache masking differs from --mask-canonical; build a separate "
            "cache with per-chunk masking before merging")
    if cache_masked and mask_application != "premerge":
        raise SystemExit("masked cache lacks premerge masking provenance")
    if args.legacy_postmerge_mask and (cache_masked or not args.mask_canonical):
        raise SystemExit("legacy postmerge masking needs an unmasked cache and --mask-canonical")
    if args.legacy_postmerge_mask and not args.annotation:
        raise SystemExit("legacy postmerge masking needs --annotation")

    mask_rows = None
    if args.legacy_postmerge_mask:
        genes = load_canonical_sites(args.annotation)
        by_tx: dict[str, dict] = {}
        var_tx: dict[str, str] = {}
        if args.transcript_annotation and args.variant_transcript_csv:
            by_tx = load_sites_by_accession(args.transcript_annotation)
            with open(args.variant_transcript_csv, newline="") as fh:
                for row in csv.DictReader(fh):
                    var_tx[row["cpra_grch38"]] = row["transcript"].split(".")[0]
        with h5py.File(args.source_h5, "r") as sf:
            g = dec(sf["source_gene_name"][:])
            st = dec(sf["source_strand"][:]) if "source_strand" in sf \
                else dec(sf["strand"][:])
            vp = sf["pos"][:].astype(np.int64)
            v_ref, v_alt = dec(sf["ref"][:]), dec(sf["alt"][:])
            v_chrom = dec(sf["chrom"][:])
        missing = sorted({x for x in g if x not in genes})
        if missing:
            # 일부만 마스킹된 결과가 masked로 기록되는 것을 막는다.
            raise SystemExit(
                f"{len(missing)} anchoring genes are absent from the annotation "
                f"(e.g. {missing[:5]}); masking would cover only part of the "
                f"table. Fix source_gene_name or run --no-mask-canonical.")

    with h5py.File(args.cache, "r") as f:
        n, w_cache, n_ch = f["ref_probs"].shape
        left = int(f.attrs["window_left"])
        if n_ch < 3:
            raise RuntimeError(f"cache has {n_ch} channels; expected [acc, don, nei]")
        lo, hi = ((0, w_cache) if args.window_bp is None else
                  (left - args.window_bp, left + args.window_bp + 1))
        if lo < 0 or hi > w_cache:
            raise RuntimeError(
                f"requested window does not fit the cache's {w_cache} "
                f"positions with the variant at {left}")
        logger.info("rows=%d cache_w=%d variant_at=%d -> slice[%d:%d]",
                    n, w_cache, left, lo, hi)

        if args.legacy_postmerge_mask:
            mask_rows = []
            n_tx_sites = n_gene_fallback = 0
            for i in range(n):
                minus = st[i] == "-"
                # 창(=[lo, hi))의 각 행이 가리키는 ref 유전체 위치. 변이 행은
                # 캐시 index `left`이므로 창 안에서는 `left - lo`다.
                gmap = row_to_ref_genomic(int(vp[i]), v_ref[i], v_alt[i], minus,
                                          hi - lo, left - lo)
                # 변이 자신의 임상 전사체가 있으면 그 junction을 쓰고, 없으면
                # 유전자 대표 전사체로 되돌린다.
                c = v_chrom[i][3:] if v_chrom[i].startswith("chr") else v_chrom[i]
                key = f"{c}-{vp[i]}-{v_ref[i]}-{v_alt[i]}"
                rec = by_tx.get(var_tx.get(key, ""))
                if rec is None:
                    rec = genes[g[i]]
                    n_gene_fallback += 1
                else:
                    n_tx_sites += 1
                where = {int(p): r for r, p in enumerate(gmap) if p >= 0}
                acc = np.array(sorted(where[int(p)] for p in rec["acc_exonic"]
                                      if int(p) in where), dtype=np.int64)
                don = np.array(sorted(where[int(p)] for p in rec["don_exonic"]
                                      if int(p) in where), dtype=np.int64)
                mask_rows.append((acc, don))

        per_dir = np.zeros((n, 4), dtype=np.float32)
        per_dir_off = np.zeros((n, 4), dtype=np.int32)
        dmass = np.zeros(n, dtype=np.float32)
        offset = np.zeros(n, dtype=np.int32)
        for s in range(0, n, args.block_size):
            e = min(s + args.block_size, n)
            r = f["ref_probs"][s:e, lo:hi, :2].astype(np.float32)
            a = f["alt_probs"][s:e, lo:hi, :2].astype(np.float32)
            d = a - r                                     # (B, W, 2)
            stacked = np.stack([
                np.clip(d[..., 0], 0, None),              # acceptor gain
                np.clip(-d[..., 0], 0, None),             # acceptor loss
                np.clip(d[..., 1], 0, None),              # donor gain
                np.clip(-d[..., 1], 0, None),             # donor loss
            ], axis=-1)                                   # (B, W, 4)
            if mask_rows is not None:
                for j in range(e - s):
                    acc, don = mask_rows[s + j]
                    stacked[j] = apply_canonical_mask(stacked[j], acc, don)
            per_dir[s:e] = stacked.max(axis=1)
            flat = stacked.reshape(e - s, -1).argmax(axis=1)
            offset[s:e] = flat // 4 - (left - lo)
            # each direction's own arg-max position. The headline offset above is
            # the winning direction's; consequence classification needs all four,
            # because a loss at a junction and the gain that compensates it are
            # different directions at different places.
            per_dir_off[s:e] = stacked.argmax(axis=1) - (left - lo)
            dmass[s:e] = np.abs(d).sum(axis=(1, 2))
            if (s // args.block_size) % 5 == 0:
                logger.info("  %d/%d", e, n)

    if args.legacy_postmerge_mask:
        logger.info("mask site source: transcript %d / gene fallback %d",
                    n_tx_sites, n_gene_fallback)

    delta_max = per_dir.max(axis=1)
    direction = np.array(DIRECTIONS, dtype=object)[per_dir.argmax(axis=1)]

    with h5py.File(args.source_h5, "r") as f:
        if len(f["pos"]) != n:
            raise RuntimeError(f"{args.source_h5} has {len(f['pos'])} rows, cache has {n}")
        ident = {c: (dec(f[c][:]) if f[c].dtype == object else f[c][:])
                 for c in ("chrom", "pos", "ref", "alt", "gene_symbol")}

    np.savez(args.out, delta_max=delta_max, delta_mass=dmass,
             masked=np.array(bool(args.mask_canonical)),
             mask_application=("postmerge_legacy" if args.legacy_postmerge_mask
                               else mask_application),
             per_direction=per_dir, per_direction_offset=per_dir_off,
             direction=direction.astype(str),
             offset=offset,
             window_bp=(left if args.window_bp is None else args.window_bp),
             window_left=left - lo, window_right=hi - left, **ident)
    q = np.quantile(delta_max, [0.5, 0.9, 0.99])
    logger.info("saved %s", args.out)
    logger.info("delta_max median %.4f p90 %.4f p99 %.4f | >=0.2 %d >=0.5 %d >=0.8 %d",
                q[0], q[1], q[2], int((delta_max >= 0.2).sum()),
                int((delta_max >= 0.5).sum()), int((delta_max >= 0.8).sum()))


if __name__ == "__main__":
    main()
