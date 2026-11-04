#!/usr/bin/env python3
"""shard에서 바로 보고 유전자 anchor 캐시를 만든다.

세 가지를 한 번에 한다:

1. **행 공간 복원** — v50 MANE 변이표는 anchor를 못 잡은 541행이 빠져 378,002행이다.
   출력은 릴리스와 같은 **378,543행**으로 되돌린다. 빠진 행은 0 트랙·coverage 0으로
   둔다. 그래야 분할·학습 마스크·보존도 sidecar·좌표 필드를 **하나도 다시 매핑하지
   않고** 그대로 쓸 수 있다.
2. **보고 유전자 anchor 반영** — 저장 anchor와 다른 행은 ALTANCHOR 캐시에서 가져온다.
3. **materialize** — ``link_multichunk_cache_parts``는 virtual dataset을 만드는데,
   virtual은 행 단위 임의 읽기가 느려(실측 735 ms/row) head 학습에 부적합하다.
   여기서는 shard를 직접 읽어 실제 배열로 쓴다. 중간 40GB 파일도 만들지 않는다.

shard는 v50 변이표의 연속 행 구간이므로 v50 행 j는 part ``j // per``의 ``j % per``
번째 행이다.

    python -m scripts.assemble_v50mane_reportedgene_cache --out-h5 <cache.h5>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from process_input.variant_identity import variant_keys, require_equal, row_digest

TRACKS = ("ref_probs", "alt_probs", "coverage")
GEOMETRY = dict(window_left=4375, window_right=4375, stride=1250,
                total_length=15000, target_length=5000)
POLICY = "reported_gene_v50mane; submitted or MANE Select; alias resolved"
STR = h5py.string_dtype(encoding="utf-8")


def dec(a) -> np.ndarray:
    """바이트 또는 문자열 배열을 str 객체 배열로 바꾼다."""
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                      dtype=object)


def main() -> None:
    """릴리스 행 공간의 materialized 캐시를 쓴다."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-h5", type=Path, required=True)
    ap.add_argument('--release-h5', type=Path, required=True)
    ap.add_argument('--v50-h5', type=Path, required=True)
    ap.add_argument('--anchor-selection', type=Path, required=True)
    ap.add_argument('--alt-h5', type=Path, required=True)
    ap.add_argument('--coord', type=Path, required=True)
    ap.add_argument("--parts-dir", type=Path, required=True)
    ap.add_argument("--alt-dir", type=Path, required=True)
    ap.add_argument("--block", type=int, default=512)
    args = ap.parse_args()
    RELEASE, V50, SEL = args.release_h5, args.v50_h5, args.anchor_selection
    if args.out_h5.exists():
        raise FileExistsError(args.out_h5)

    z = np.load(SEL, allow_pickle=True)
    key = dec(z["key"])
    needs_alt = z["needs_alt"].astype(bool)
    train_mask = z["train_mask"].astype(bool)
    anchored = z["anchored"].astype(bool)
    universe = z["universe"].astype(bool)
    gene_sel = dec(z["gene"])
    n = len(key)

    with h5py.File(RELEASE, "r") as f:
        require_equal(variant_keys(f), key, 'anchor selection')
        labels = f["label"][:].astype(np.int8)
        gene_symbol = dec(f["gene_symbol"][:])
        identity = {name: f[name][:] for name in ['chrom', 'pos', 'ref', 'alt', 'variant_kind']}
    cf = np.load(args.coord, allow_pickle=True)
    require_equal(labels, cf['labels'], 'coordinate labels')
    if str(cf['row_key_sha256']) != row_digest(key):
        raise ValueError('Coordinate identities do not match the cohort')
    universe = cf['splice'].astype(bool)
    with h5py.File(V50, "r") as f:
        v_key = np.array([f"{c.removeprefix('chr')}:{int(p)}:{r}:{a}"
                          for c, p, r, a in zip(dec(f["chrom"][:]), f["pos"][:],
                                                dec(f["ref"][:]), dec(f["alt"][:]))],
                         dtype=object)
    v_row = {k: i for i, k in enumerate(v_key.tolist())}
    to_v50 = np.array([v_row.get(k, -1) for k in key.tolist()], dtype=np.int64)
    print(f"release rows={n} mapped to v50={int((to_v50 >= 0).sum())} "
          f"missing={int((to_v50 < 0).sum())}", flush=True)

    parts = sorted(args.parts_dir.glob("part_*.h5"))
    if not parts:
        raise SystemExit(f"no shards in {args.parts_dir}")
    handles = [h5py.File(p, "r") for p in parts]
    counts = [h["ref_probs"].shape[0] for h in handles]
    print(f"shards={len(parts)} rows={sum(counts)}", flush=True)
    if sum(counts) != len(v_key):
        raise SystemExit(f"shard rows {sum(counts)} != v50 rows {len(v_key)}")
    starts = np.cumsum([0] + counts[:-1])
    for h, start, count in zip(handles, starts, counts):
        require_equal(np.arange(start, start + count), h['row_index'][:], 'base shard order')
        for name, expected in GEOMETRY.items():
            if int(h.attrs[name]) != expected:
                raise ValueError(f'Base shard geometry differs: {name}')

    alt_parts = sorted(args.alt_dir.glob("part_*.h5"))
    if not alt_parts and np.any(needs_alt & (to_v50 >= 0)):
        raise SystemExit(f"no shards in {args.alt_dir}")
    alt_h = [h5py.File(p, "r") for p in alt_parts]
    with h5py.File(args.alt_h5, "r") as f:
        alt_parent = f["parent_row"][:].astype(np.int64)
        require_equal(variant_keys(f), v_key[alt_parent], 'alternative input identities')
    alt_counts = [h["ref_probs"].shape[0] for h in alt_h]
    if sum(alt_counts) != len(alt_parent):
        raise SystemExit(f"alt shard rows {sum(alt_counts)} != {len(alt_parent)}")
    alt_starts = np.cumsum([0] + alt_counts[:-1])
    for h, start, count in zip(alt_h, alt_starts, alt_counts):
        require_equal(np.arange(start, start + count), h['row_index'][:], 'alternative shard order')
        for name, expected in GEOMETRY.items():
            if int(h.attrs[name]) != expected:
                raise ValueError(f'Alternative shard geometry differs: {name}')
    alt_of_v50 = {int(pr): i for i, pr in enumerate(alt_parent)}
    expected_alt = to_v50[needs_alt & (to_v50 >= 0)]
    if len(alt_of_v50) != len(alt_parent) or set(alt_parent) != set(expected_alt):
        raise ValueError('Alternative inputs do not cover exactly the changed anchors')

    def read_row(src: str, which: str, idx: int) -> np.ndarray:
        """part 묶음에서 한 행을 읽는다."""
        hs, st = (handles, starts) if src == "base" else (alt_h, alt_starts)
        p = int(np.searchsorted(st, idx, side="right") - 1)
        return hs[p][which][idx - int(st[p])]

    shape = handles[0]["ref_probs"].shape[1:]
    cov_shape = handles[0]["coverage"].shape[1:]
    tmp = args.out_h5.with_suffix(".partial.h5")
    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp, "w") as out:
        for name in TRACKS:
            s = shape if name != "coverage" else cov_shape
            dt = handles[0][name].dtype
            out.create_dataset(name, shape=(n, *s), dtype=dt, chunks=(1, *s))
        out.create_dataset("labels", data=labels)
        out.create_dataset('key', data=key.astype(object), dtype=STR)
        for name, values in identity.items():
            out.create_dataset(name, data=values, dtype=STR if values.dtype.kind in 'OSU' else values.dtype)
        out.create_dataset("row_index", data=np.arange(n, dtype=np.int64))
        out.create_dataset("gene_symbol", data=gene_symbol, dtype=STR)
        out.create_dataset("anchor_gene", data=gene_sel, dtype=STR)
        out.create_dataset("reported_gene_anchored", data=anchored)
        out.create_dataset("train_mask", data=train_mask)
        out.create_dataset("evaluation_mask", data=universe)
        out.create_dataset("in_v50mane", data=to_v50 >= 0)
        out.create_dataset("selected_alt_row",
                           data=np.where(needs_alt, 1, -1).astype(np.int64))
        n_missing = n_alt = 0
        for start in range(0, n, args.block):
            end = min(start + args.block, n)
            buf = {name: np.zeros((end - start, *(shape if name != "coverage"
                                                  else cov_shape)),
                                  dtype=handles[0][name].dtype)
                   for name in TRACKS}
            for r in range(start, end):
                j = int(to_v50[r])
                if j < 0:
                    n_missing += 1
                    continue
                if needs_alt[r] and j in alt_of_v50:
                    src, idx = "alt", alt_of_v50[j]
                    n_alt += 1
                else:
                    src, idx = "base", j
                for name in TRACKS:
                    buf[name][r - start] = read_row(src, name, idx)
            for name in TRACKS:
                out[name][start:end] = buf[name]
            if start % (args.block * 100) == 0:
                print(f"  wrote {end}/{n}", flush=True)
        for k, v in GEOMETRY.items():
            out.attrs[k] = str(v)
        out.attrs["anchor_selection"] = POLICY
        out.attrs["mask_canonical"] = False
        out.attrs["mask_application"] = "none"
        out.attrs["pairing"] = "per-position argmax|alt-ref|, ref+alt together"
        out.attrs["annotation_path"] = ""
        out.attrs["rows_without_anchor_zero_filled"] = str(n_missing)
        out.attrs["rows_from_alternative_anchor"] = str(n_alt)
        out.attrs["selection"] = str(SEL)
        out.attrs['row_key_sha256'] = row_digest(key)
        out.attrs["complete"] = True
    tmp.replace(args.out_h5)
    for h in handles + alt_h:
        h.close()

    summary = {"n": int(n), "zero_filled": int(n_missing),
               "from_alternative_anchor": int(n_alt),
               "train_mask": int(train_mask.sum()),
               "evaluation_mask": int(universe.sum()),
               "out_h5": str(args.out_h5)}
    args.out_h5.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
