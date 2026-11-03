#!/usr/bin/env python3
"""v50 MANE 표에서 ClinVar 보고 유전자 anchor를 정하고 학습 마스크를 만든다.

anchor 규칙(사용자 확정 2026-10-30):

1. 제출 유전자(``SubmittedGeneSymbol``)가 하나면 그것
2. 여러 개면 NCBI 대표(MANE Select) 유전자 — v50 MANE 표에서 변이를 덮는 후보
   중 ClinVar preferred name의 유전자를 우선한다
3. 고른 유전자가 표에 없거나 변이를 덮지 못하면, NCBI 대표 이름의 유전자가 덮는지
   보고 덮으면 그것을 쓴다. ClinVar가 옛 유전자 기호를 들고 있는 경우를 살린다
   (``AARS``→``AARS1``, ``ADCK3``→``COQ8A`` 등). 같은 유전자의 현재 이름이므로
   보고 유전자 원칙을 벗어나지 않는다
4. 그래도 안 되면 학습에서 제외한다. 평가에는 남겨 no-call로 둔다

좌표는 ``find_anchor``에 그 유전자 하나만 힌트로 넘겨 구한다. canonical 표의 유전자
이름이 유일하므로 힌트가 맞으면 후보가 하나뿐이고 최장 전사체 tie-break는 작동하지
않는다. 즉 ``|`` 구분자 파싱 문제의 영향을 받지 않는다.

학습 마스크는 여기서 함께 만든다. 연결 성분 분할은 주석 유전자 토큰으로 묶이므로
anchor 유전자가 그 토큰 집합 밖이면 같은 유전자가 두 fold에 걸칠 수 있다. 그런
유전자의 행은 **이 anchor 배열에서 직접 찾아** 학습·검증에서 뺀다. 학습기의
``assert_no_gene_leakage``는 유전자 문자열을 통째로 비교하므로 이 누수를 잡지 못한다.

이미 만들어 둔 v50 MANE 캐시는 기존 ``find_anchor``가 고른 anchor로 추론한다. 저장
anchor와 다른 행만 추려 그 행만 추가 추론하면 되도록 worklist를 쓴다.

    python -m scripts.build_v50mane_reportedgene_anchors --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np

from process_input.gencode_anchor import (
    build_genome_index,
    find_anchor,
    load_canonical_table,
)

NAS = Path(".")
EVAL = NAS / "data/eval"
RELEASE = (EVAL / "clinvar_20260628_gencode_anchored_indel"
           / "clinvar_20260628_gencode_anchored_indel.h5")
V50 = EVAL / "v50mane/clinvar_20260628_gencode_anchored_indel.anchored_on_v50mane.h5"
MANE = NAS / "data/canonical_dataset_v50_mane_regions.txt"
SUBG = (EVAL / "recomb2027/single_anchor_recordgene_hgvs_20261029"
        / "submitted_gene/per_variant.tsv.gz")
HGVS = EVAL / "clinvar_pb_2026_split/coord_splice_fields_recordgene_hgvs_seed42.npz"
SPLIT = EVAL / "clinvar_pb_2026_split_cc/gene_grouped_5fold_cc_seed42.npz"
SL, CL_MAX = 5000, 10000


def dec(a) -> np.ndarray:
    """바이트 또는 문자열 배열을 str 객체 배열로 바꾼다."""
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                      dtype=object)


def keys_of(chrom, pos, ref, alt) -> np.ndarray:
    """변이 키 배열을 만든다."""
    return np.array([f"{c.removeprefix('chr')}:{int(p)}:{r}:{a}"
                     for c, p, r, a in zip(chrom, pos, ref, alt)], dtype=object)


def main() -> None:
    """anchor를 정하고 학습 마스크와 재추론 worklist를 쓴다."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(RELEASE, "r") as f:
        chrom, pos = dec(f["chrom"][:]), f["pos"][:].astype(np.int64)
        ref, alt = dec(f["ref"][:]), dec(f["alt"][:])
        label = f["label"][:].astype(np.int64)
        geneinfo = dec(f["gene_symbol"][:])
    n = len(label)
    key = keys_of(chrom, pos, ref, alt)

    with h5py.File(V50, "r") as f:
        v_key = keys_of(dec(f["chrom"][:]), f["pos"][:], dec(f["ref"][:]),
                        dec(f["alt"][:]))
        v_gene = dec(f["source_gene_name"][:])
        v_strand = dec(f["source_strand"][:])
        v_chunk = f["source_chunk_idx"][:].astype(np.int64)
        v_local = f["source_local_idx"][:].astype(np.int64)
        v_off = f["variant_offset"][:].astype(np.int64)
    v_row = {k: i for i, k in enumerate(v_key.tolist())}
    in_v50 = np.array([k in v_row for k in key.tolist()])
    print(f"release rows={n} v50mane rows={len(v_key)} "
          f"mapped={int(in_v50.sum())}", flush=True)

    with gzip.open(SUBG, "rt", newline="") as fh:
        sub = list(csv.DictReader(fh, delimiter="\t"))
    assert len(sub) == n, "submitted-gene table is not row aligned"
    submitted = [[t for t in (r["submitted_genes"] or "").split("|") if t]
                 for r in sub]
    preferred = np.asarray([r["preferred_gene"] for r in sub], dtype=object)

    genome = build_genome_index(
        load_canonical_table(str(MANE), chrom_filter=None, sl=SL))
    spans: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for c, ents in genome.items():
        d: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for e in ents:
            d[e.gene_name].append((e.tx_start, e.tx_end))
        spans[c] = d

    def covers(g: str, c: str, p: int) -> bool:
        """v50 MANE 표에서 그 유전자가 이 위치를 덮는지."""
        return bool(g) and any(s <= p <= t for s, t in spans.get(c, {}).get(g, ()))

    gene = np.empty(n, dtype=object)
    gene[:] = ""
    why = np.empty(n, dtype=object)
    why[:] = "no_submitted_gene"
    for i in range(n):
        c, p, toks, pg = chrom[i], int(pos[i]), submitted[i], preferred[i]
        pick, tag = "", "no_submitted_gene"
        if len(toks) == 1:
            pick, tag = toks[0], "single_submitted"
        elif len(toks) > 1:
            mane = [t for t in toks if covers(t, c, p)]
            if pg in mane:
                pick, tag = pg, "multi_preferred_is_mane"
            elif len(mane) == 1:
                pick, tag = mane[0], "multi_single_mane"
            elif mane:
                pick, tag = sorted(mane)[0], "multi_several_mane_alphabetical"
            elif pg in toks:
                pick, tag = pg, "multi_preferred_not_in_mane_table"
            else:
                tag = "multi_unresolved"
        # rule 3: the reported gene under its current symbol
        if pick and not covers(pick, c, p) and pg and pg != pick and covers(pg, c, p):
            pick, tag = pg, "alias_rescued_by_preferred"
        gene[i], why[i] = pick, tag

    a_chunk = np.full(n, -1, np.int64)
    a_local = np.full(n, -1, np.int64)
    a_off = np.full(n, -1, np.int64)
    a_strand = np.empty(n, dtype=object)
    a_strand[:] = ""
    status = np.empty(n, dtype=object)
    status[:] = "no_reported_gene"
    for i in range(n):
        if not gene[i]:
            continue
        m = find_anchor(chrom=chrom[i], pos=int(pos[i]), sl=SL, cl_max=CL_MAX,
                        genome_index=genome, gene_symbol_hint=gene[i])
        if m is None or m.gene.gene_name != gene[i]:
            status[i] = "reported_gene_not_anchorable"
            continue
        a_chunk[i], a_local[i] = m.chunk_idx, m.local_idx
        a_off[i], a_strand[i] = m.variant_offset, m.gene.strand
        status[i] = "anchored"
        if i % 100000 == 0:
            print(f"  anchored {i}/{n}", flush=True)
    anchored = status == "anchored"

    # fold 횡단 anchor 유전자: 이 anchor 배열에서 직접 찾는다
    fold = np.load(SPLIT, allow_pickle=True)["fold_id"].astype(int)
    assert len(fold) == n, "split is not row aligned"
    gene_folds: dict[str, set[int]] = collections.defaultdict(set)
    for i in np.flatnonzero(anchored):
        if fold[i] >= 0:
            gene_folds[gene[i]].add(int(fold[i]))
    spanning = {g for g, f in gene_folds.items() if len(f) > 1}
    spans_fold = np.array([bool(anchored[i] and gene[i] in spanning)
                           for i in range(n)])
    train_mask = anchored & ~spans_fold
    # 제외 후 누수가 남지 않는지 확인
    check: dict[str, set[int]] = collections.defaultdict(set)
    for i in np.flatnonzero(train_mask):
        if fold[i] >= 0:
            check[gene[i]].add(int(fold[i]))
    still = [g for g, f in check.items() if len(f) > 1]
    assert not still, f"fold leakage remains for {still[:5]}"

    universe = np.load(HGVS, allow_pickle=True)["splice"].astype(bool)
    eval_nocall = universe & ~anchored
    print(f"anchored={int(anchored.sum())} "
          f"fold_spanning_excluded={int(spans_fold.sum())} "
          f"train={int(train_mask.sum())}", flush=True)

    same = np.zeros(n, bool)
    for i in np.flatnonzero(anchored):
        j = v_row.get(key[i])
        if j is None:
            continue
        same[i] = (v_gene[j] == gene[i] and int(v_chunk[j]) == int(a_chunk[i])
                   and int(v_local[j]) == int(a_local[i])
                   and int(v_off[j]) == int(a_off[i])
                   and v_strand[j] == a_strand[i])
    needs_alt = anchored & ~same
    print(f"identical_to_stored={int(same.sum())} "
          f"needs_alt_inference={int(needs_alt.sum())}", flush=True)

    np.savez(args.out / "anchor_selection_v50mane.npz",
             gene=gene, why=why, status=status, anchored=anchored,
             fold_spanning=spans_fold, train_mask=train_mask,
             eval_nocall=eval_nocall, same_as_stored=same, needs_alt=needs_alt,
             in_v50mane=in_v50, chunk_idx=a_chunk, local_idx=a_local,
             variant_offset=a_off, strand=a_strand, key=key, label=label,
             fold_id=fold, universe=universe)

    with open(args.out / "alt_worklist.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["release_row", "key", "chrom", "pos", "ref", "alt", "label",
                    "reported_gene", "stored_gene", "chunk_idx", "local_idx",
                    "variant_offset", "strand", "why"])
        for i in np.flatnonzero(needs_alt):
            j = v_row.get(key[i])
            w.writerow([i, key[i], chrom[i], int(pos[i]), ref[i], alt[i],
                        int(label[i]), gene[i],
                        v_gene[j] if j is not None else "",
                        int(a_chunk[i]), int(a_local[i]), int(a_off[i]),
                        a_strand[i], why[i]])

    with open(args.out / "fold_spanning_genes.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["anchor_gene", "folds", "n_rows", "n_pos"])
        for g in sorted(spanning):
            m = spans_fold & (gene == g)
            w.writerow([g, "|".join(map(str, sorted(gene_folds[g]))),
                        int(m.sum()), int(label[m].sum())])

    summary = {
        "n": int(n),
        "rule_branches": {k: int(v) for k, v in
                          collections.Counter(why.tolist()).items()},
        "status": {k: int(v) for k, v in
                   collections.Counter(status.tolist()).items()},
        "anchored": int(anchored.sum()),
        "anchored_pos": int(label[anchored].sum()),
        "fold_spanning_genes": len(spanning),
        "fold_spanning_rows": int(spans_fold.sum()),
        "fold_spanning_pos": int(label[spans_fold].sum()),
        "train_mask": int(train_mask.sum()),
        "train_mask_pos": int(label[train_mask].sum()),
        "excluded_total": int(n - train_mask.sum()),
        "evaluation_universe": int(universe.sum()),
        "evaluation_nocall": int(eval_nocall.sum()),
        "evaluation_nocall_pos": int(label[eval_nocall].sum()),
        "identical_to_stored": int(same.sum()),
        "needs_alt_inference": int(needs_alt.sum()),
        "needs_alt_pos": int(label[needs_alt].sum()),
        "inputs": [str(RELEASE), str(V50), str(MANE), str(SUBG), str(SPLIT),
                   str(HGVS)],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
